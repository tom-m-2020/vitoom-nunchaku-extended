"""
Adapters for efficient caching in SDXL diffusion pipelines.
"""

import functools

from diffusers import DiffusionPipeline
import torch

from ..fbcache import advance_cache_context_step, cache_context, configure_cache_context, create_cache_context, get_current_cache_context
from ..utils_sdxl import cached_forward_sdxl


def _run_cached_forward_with_hook(unet, *args, **kwargs):
    hook = getattr(unet, "_hf_hook", None)
    if hook is None:
        return cached_forward_sdxl(unet, *args, **kwargs)

    args, kwargs = hook.pre_forward(unet, *args, **kwargs)
    if getattr(hook, "no_grad", False):
        with torch.no_grad():
            output = cached_forward_sdxl(unet, *args, **kwargs)
    else:
        output = cached_forward_sdxl(unet, *args, **kwargs)
    return hook.post_forward(unet, output)


def apply_cache_on_unet(unet, *, residual_diff_threshold: float = 0.12, verbose: bool = False):
    """
    Enable first-block caching for an SDXL UNet.
    """
    if getattr(unet, "_is_cached", False):
        unet.residual_diff_threshold = residual_diff_threshold
        unet.verbose = verbose
        return unet

    original_forward = unet.forward
    unet._original_forward = original_forward
    unet.residual_diff_threshold = residual_diff_threshold
    unet.verbose = verbose

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        if get_current_cache_context() is not None:
            advance_cache_context_step()
            return _run_cached_forward_with_hook(self, *args, **kwargs)
        return self._original_forward(*args, **kwargs)

    unet.forward = new_forward.__get__(unet, unet.__class__)
    unet._is_cached = True
    return unet


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    residual_diff_threshold: float = 0.12,
    verbose: bool = False,
):
    """
    Enable first-block caching for a complete SDXL pipeline.
    """
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            cache_ctx = configure_cache_context(
                create_cache_context(),
                num_inference_steps=kwargs.get("num_inference_steps"),
            )
            with cache_context(cache_ctx):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_unet(pipe.unet, residual_diff_threshold=residual_diff_threshold, verbose=verbose)
    return pipe
