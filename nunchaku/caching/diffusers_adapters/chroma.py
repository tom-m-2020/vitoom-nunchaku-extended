"""
Adapters for efficient caching in Chroma diffusion pipelines.
"""

import functools

from diffusers import DiffusionPipeline

from ..fbcache import cache_context, configure_cache_context, create_cache_context
from ..utils_chroma import cached_forward_chroma

DEFAULT_CHROMA_RESIDUAL_DIFF_THRESHOLD_MULTI = 0.06
DEFAULT_CHROMA_RESIDUAL_DIFF_THRESHOLD_SINGLE = 0.085
DEFAULT_CHROMA_TEAGATE_THRESHOLD_MULTI = 0.6


def apply_cache_on_transformer(
    transformer,
    *,
    use_double_fb_cache: bool = True,
    residual_diff_threshold: float = DEFAULT_CHROMA_RESIDUAL_DIFF_THRESHOLD_MULTI,
    residual_diff_threshold_multi: float | None = None,
    residual_diff_threshold_single: float | None = None,
    use_teacache_gate_multi: bool = True,
    teacache_gate_threshold_multi: float = DEFAULT_CHROMA_TEAGATE_THRESHOLD_MULTI,
    verbose: bool = False,
):
    """
    Enable first-block caching for a Chroma transformer.
    """
    if residual_diff_threshold_multi is None:
        residual_diff_threshold_multi = residual_diff_threshold
    if residual_diff_threshold_single is None:
        residual_diff_threshold_single = DEFAULT_CHROMA_RESIDUAL_DIFF_THRESHOLD_SINGLE

    if getattr(transformer, "_is_cached", False):
        transformer.use_double_fb_cache = use_double_fb_cache
        transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
        transformer.residual_diff_threshold_single = residual_diff_threshold_single
        transformer.use_teacache_gate_multi = use_teacache_gate_multi
        transformer.teacache_gate_threshold_multi = teacache_gate_threshold_multi
        transformer.verbose = verbose
        return transformer

    transformer._original_forward = transformer.forward
    transformer.use_double_fb_cache = use_double_fb_cache
    transformer.residual_diff_threshold_multi = residual_diff_threshold_multi
    transformer.residual_diff_threshold_single = residual_diff_threshold_single
    transformer.use_teacache_gate_multi = use_teacache_gate_multi
    transformer.teacache_gate_threshold_multi = teacache_gate_threshold_multi
    transformer.verbose = verbose
    transformer.forward = cached_forward_chroma.__get__(transformer, transformer.__class__)
    transformer._is_cached = True
    return transformer


def apply_cache_on_pipe(pipe: DiffusionPipeline, **kwargs):
    """
    Apply first-block caching to a Chroma pipeline.
    """
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            guidance_scale = kwargs.get("guidance_scale", 1.0)
            has_negative_prompt = kwargs.get("negative_prompt") is not None or kwargs.get("negative_prompt_embeds") is not None
            cache_ctx = configure_cache_context(
                create_cache_context(),
                num_inference_steps=kwargs.get("num_inference_steps"),
                num_branches=2 if guidance_scale > 1.0 and has_negative_prompt else 1,
            )
            with cache_context(cache_ctx):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    apply_cache_on_transformer(pipe.transformer, **kwargs)
    return pipe
