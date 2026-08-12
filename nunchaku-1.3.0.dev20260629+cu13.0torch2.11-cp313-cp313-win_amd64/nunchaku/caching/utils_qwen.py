"""
Caching utilities for Qwen-Image transformer models.

Implements first-block caching for Qwen-Image style dual-stream transformers by
reusing the residual produced by the remaining transformer blocks when the first
block residual changes only slightly.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import deprecate

from ..models.transformers.transformer_qwenimage import (
    _call_time_text_embed,
    _compute_text_seq_len_compat,
    compute_text_seq_len_from_mask,
)
from .fbcache import advance_cache_context_step, check_and_apply_cache, get_current_cache_context


def _maybe_add_controlnet_residual(
    hidden_states: torch.Tensor,
    *,
    controlnet_block_samples,
    block_idx: int,
    num_blocks: int,
) -> torch.Tensor:
    if controlnet_block_samples is None:
        return hidden_states

    interval_control = int(np.ceil(num_blocks / len(controlnet_block_samples)))
    return hidden_states + controlnet_block_samples[block_idx // interval_control]


def _get_qwen_cache_key_prefix() -> str:
    _, _, branch_idx = advance_cache_context_step()
    cache_ctx = get_current_cache_context()
    if cache_ctx is None:
        return "qwen"
    num_branches = max(int(getattr(cache_ctx, "fbcache_num_branches", 1)), 1)
    if num_branches == 1:
        return "qwen"
    return f"qwen_branch_{branch_idx}"


def run_remaining_blocks_qwen(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    encoder_hidden_states_mask: Optional[torch.Tensor],
    use_modern_qwenimage: bool,
    temb: torch.Tensor,
    image_rotary_emb,
    block_attention_kwargs: Dict[str, Any],
    modulate_index: Optional[torch.Tensor],
    controlnet_block_samples,
    compute_stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process QwenImage transformer blocks after the first block.
    """
    original_h = hidden_states
    original_enc = encoder_hidden_states

    if self.offload:
        self.offload_manager.step(compute_stream)

    for block_idx in range(1, len(self.transformer_blocks)):
        with torch.cuda.stream(compute_stream):
            block = self.offload_manager.get_block(block_idx) if self.offload else self.transformer_blocks[block_idx]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None if use_modern_qwenimage else encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None if use_modern_qwenimage else encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

            hidden_states = _maybe_add_controlnet_residual(
                hidden_states,
                controlnet_block_samples=controlnet_block_samples,
                block_idx=block_idx,
                num_blocks=len(self.transformer_blocks),
            )

        if self.offload:
            self.offload_manager.step(compute_stream)

    return hidden_states, encoder_hidden_states, hidden_states - original_h, encoder_hidden_states - original_enc


def cached_forward_qwen(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    Cached forward function for Qwen-Image transformers.
    """
    if self.residual_diff_threshold_multi < 0.0:
        return self._original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
            attention_kwargs=attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            additional_t_cond=additional_t_cond,
            return_dict=return_dict,
        )

    cache_key_prefix = _get_qwen_cache_key_prefix()
    device = hidden_states.device
    if self.offload:
        self.offload_manager.set_device(device)

    use_modern_qwenimage = compute_text_seq_len_from_mask is not None
    if use_modern_qwenimage and txt_seq_lens is not None:
        deprecate(
            "txt_seq_lens",
            "0.39.0",
            "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
            "Please use `encoder_hidden_states_mask` instead. "
            "The mask-based approach is more flexible and supports variable-length sequences.",
            standard_warn=False,
        )

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)

    if getattr(self, "zero_cond_t", False):
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * np.prod(sample[0]) + [1] * sum(np.prod(s) for s in sample[1:]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    text_seq_len, encoder_hidden_states_mask = _compute_text_seq_len_compat(
        encoder_hidden_states, encoder_hidden_states_mask, txt_seq_lens
    )

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = _call_time_text_embed(self, timestep, hidden_states, guidance, additional_t_cond)

    if use_modern_qwenimage:
        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
    else:
        if txt_seq_lens is None:
            txt_seq_lens = [text_seq_len] * encoder_hidden_states.shape[0]
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    if hasattr(self, "_get_packed_qwen_rotary_emb"):
        image_rotary_emb = self._get_packed_qwen_rotary_emb(
            image_rotary_emb,
            batch_size=hidden_states.shape[0],
            img_shapes=img_shapes,
            text_seq_len=text_seq_len,
            device=hidden_states.device,
        )

    block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    if use_modern_qwenimage and encoder_hidden_states_mask is not None:
        batch_size, image_seq_len = hidden_states.shape[:2]
        if hasattr(self, "_get_joint_attention_mask"):
            block_attention_kwargs["attention_mask"] = self._get_joint_attention_mask(
                encoder_hidden_states_mask, batch_size, image_seq_len, hidden_states.device
            )
        else:
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            block_attention_kwargs["attention_mask"] = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

    compute_stream = torch.cuda.current_stream()
    if self.offload:
        self.offload_manager.initialize(compute_stream)

    original_hidden_states = hidden_states
    first_block = self.offload_manager.get_block(0) if self.offload else self.transformer_blocks[0]

    with torch.cuda.stream(compute_stream):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                first_block,
                hidden_states,
                encoder_hidden_states,
                None if use_modern_qwenimage else encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
                block_attention_kwargs,
                modulate_index,
            )
        else:
            encoder_hidden_states, hidden_states = first_block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None if use_modern_qwenimage else encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

        hidden_states = _maybe_add_controlnet_residual(
            hidden_states,
            controlnet_block_samples=controlnet_block_samples,
            block_idx=0,
            num_blocks=len(self.transformer_blocks),
        )

    first_hidden_states_residual = hidden_states - original_hidden_states

    hidden_states, encoder_hidden_states, _ = check_and_apply_cache(
        first_residual=first_hidden_states_residual,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        threshold=self.residual_diff_threshold_multi,
        parallelized=False,
        mode="multi",
        verbose=self.verbose if hasattr(self, "verbose") else False,
        call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: run_remaining_blocks_qwen(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            use_modern_qwenimage=use_modern_qwenimage,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            block_attention_kwargs=block_attention_kwargs,
            modulate_index=modulate_index,
            controlnet_block_samples=controlnet_block_samples,
            compute_stream=compute_stream,
        ),
        remaining_kwargs={},
        cache_key_prefix=cache_key_prefix,
    )

    if getattr(self, "zero_cond_t", False):
        temb = temb.chunk(2, dim=0)[0]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if self.offload:
        torch.cuda.empty_cache()

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
