"""
Adapters for efficient caching in Qwen-Image diffusion pipelines.
"""

import functools

from diffusers import DiffusionPipeline
import torch

from ..fbcache import cache_context, configure_cache_context, create_cache_context
from ..utils_qwen import cached_forward_qwen


def _run_cached_forward_with_hook(transformer, *args, **kwargs):
    hook = getattr(transformer, "_hf_hook", None)
    if hook is None:
        return cached_forward_qwen(transformer, *args, **kwargs)

    args, kwargs = hook.pre_forward(transformer, *args, **kwargs)
    if getattr(hook, "no_grad", False):
        with torch.no_grad():
            output = cached_forward_qwen(transformer, *args, **kwargs)
    else:
        output = cached_forward_qwen(transformer, *args, **kwargs)
    return hook.post_forward(transformer, output)


def apply_cache_on_transformer(
    transformer,
    *,
    residual_diff_threshold: float = 0.12,
):
    """
    Enable first-block caching for a Qwen-Image transformer.
    """
    if getattr(transformer, "_is_cached", False):
        transformer.residual_diff_threshold_multi = residual_diff_threshold
        return transformer

    original_forward = transformer.forward
    transformer._original_forward = original_forward
    transformer.residual_diff_threshold_multi = residual_diff_threshold
    transformer.verbose = False

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        return _run_cached_forward_with_hook(self, *args, **kwargs)

    transformer.forward = new_forward.__get__(transformer, transformer.__class__)
    transformer._is_cached = True
    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
    """
    Apply first-block caching to a Qwen pipeline.
    """
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            true_cfg_scale = kwargs.get("true_cfg_scale", 1.0)
            has_negative_prompt = kwargs.get("negative_prompt") is not None or kwargs.get("negative_prompt_embeds") is not None
            cache_ctx = configure_cache_context(
                create_cache_context(),
                num_inference_steps=kwargs.get("num_inference_steps"),
                num_branches=2 if true_cfg_scale > 1.0 and has_negative_prompt else 1,
            )
            with cache_context(cache_ctx):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_transformer(pipe.transformer, **kwargs)
    return pipe
