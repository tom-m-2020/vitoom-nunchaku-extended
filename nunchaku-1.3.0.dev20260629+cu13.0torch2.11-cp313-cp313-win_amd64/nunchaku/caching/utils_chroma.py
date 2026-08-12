"""
Caching utilities for Chroma transformer models.

Implements first-block caching for Chroma dual-stream transformers by comparing
the first dual-stream block residual and reusing cached residuals for the
remaining dual/single blocks when possible.
"""

from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ..models.embeddings import pack_rotemb
from ..models.transformers.transformer_chroma import (
    _expand_batch_dim,
    _prepare_cpp_context,
    _should_use_cpp_additive_attn,
)
from ..utils import pad_tensor
from .fbcache import advance_cache_context_step, check_and_apply_cache, get_current_cache_context

CHROMA_TEAGATE_COEFFICIENTS = (
    4.98651651e02,
    -2.83781631e02,
    5.58554382e01,
    -3.82021401e00,
    2.64230861e-01,
)


def _get_chroma_cache_key_prefix() -> str:
    _, _, branch_idx = advance_cache_context_step()
    cache_ctx = get_current_cache_context()
    if cache_ctx is None:
        return "chroma"

    num_branches = max(int(getattr(cache_ctx, "fbcache_num_branches", 1)), 1)
    if num_branches == 1:
        return "chroma"
    return f"chroma_branch_{branch_idx}"


def _poly_eval(coefficients: tuple[float, ...], x: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def _get_chroma_teagate_state(cache_key_prefix: str) -> dict[str, Any]:
    cache_ctx = get_current_cache_context()
    if cache_ctx is None:
        return {"acc": 0.0, "prev_mod": None}

    states = getattr(cache_ctx, "chroma_fbcache_teagate_states", None)
    if states is None:
        states = {}
        setattr(cache_ctx, "chroma_fbcache_teagate_states", states)
    state = states.get(cache_key_prefix)
    if state is None:
        state = {"acc": 0.0, "prev_mod": None}
        states[cache_key_prefix] = state
    return state


def _should_allow_chroma_multi_cache(self, hidden_states: torch.Tensor, first_temb: torch.Tensor, cache_key_prefix: str) -> bool:
    if not getattr(self, "use_teacache_gate_multi", True):
        return True

    cache_ctx = get_current_cache_context()
    if cache_ctx is None:
        return True

    state = _get_chroma_teagate_state(cache_key_prefix)
    step_idx = int(getattr(cache_ctx, "fbcache_current_step", 0))
    total_steps = getattr(cache_ctx, "fbcache_num_inference_steps", None)

    temb_img = first_temb[:, :6].clone()
    modulated_inp, *_ = self.transformer_blocks[0].norm1(hidden_states.clone(), emb=temb_img)

    if state["prev_mod"] is None:
        state["acc"] = 0.0
        state["prev_mod"] = modulated_inp
        return False

    if step_idx <= 0:
        state["acc"] = 0.0
        state["prev_mod"] = modulated_inp
        return False

    if total_steps is not None and step_idx >= max(int(total_steps) - 1, 0):
        state["acc"] = 0.0
        state["prev_mod"] = modulated_inp
        return False

    prev_mod = state["prev_mod"]
    denom = max(float(prev_mod.abs().mean().detach().cpu().item()), 1e-8)
    rel = float(((modulated_inp - prev_mod).abs().mean() / denom).detach().cpu().item())
    scaled = abs(_poly_eval(CHROMA_TEAGATE_COEFFICIENTS, rel))
    state["acc"] = float(state["acc"]) + scaled
    state["prev_mod"] = modulated_inp

    if float(state["acc"]) < float(getattr(self, "teacache_gate_threshold_multi", 0.6)):
        return True

    state["acc"] = 0.0
    return False


def run_remaining_blocks_chroma(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    pooled_temb: torch.Tensor,
    rotary_emb_img: torch.Tensor,
    rotary_emb_txt: torch.Tensor,
    rotary_emb_single: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    ws_dual: dict | None,
    ws_single: dict | None,
    mask_dual: torch.Tensor | None,
    mask_single: torch.Tensor | None,
    txt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    original_h = hidden_states
    original_enc = encoder_hidden_states

    num_layers = len(self.transformer_blocks)
    num_single = len(self.single_transformer_blocks)
    img_offset = 3 * num_single
    txt_offset = img_offset + 6 * num_layers

    for i, block in enumerate(self.transformer_blocks[1:], start=1):
        img_mod = img_offset + 6 * i
        txt_mod = txt_offset + 6 * i
        temb = torch.cat(
            (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, txt_mod : txt_mod + 6]),
            dim=1,
        )
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            attention_mask_1d=attention_mask_1d,
            cpp_workspace=ws_dual,
            cpp_mask=mask_dual,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    original_cat = hidden_states

    for i, block in enumerate(self.single_transformer_blocks):
        start = 3 * i
        temb = pooled_temb[:, start : start + 3]
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            attention_mask_1d=attention_mask_1d,
            cpp_workspace=ws_single,
            cpp_mask=mask_single,
        )

    final_enc = hidden_states[:, :txt_tokens, ...]
    final_h = hidden_states[:, txt_tokens:, ...]
    return final_h, final_enc, final_h - original_h, final_enc - original_enc


def run_remaining_multi_blocks_chroma(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    pooled_temb: torch.Tensor,
    rotary_emb_img: torch.Tensor,
    rotary_emb_txt: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    ws_dual: dict | None,
    mask_dual: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    original_h = hidden_states
    original_enc = encoder_hidden_states

    num_layers = len(self.transformer_blocks)
    num_single = len(self.single_transformer_blocks)
    img_offset = 3 * num_single
    txt_offset = img_offset + 6 * num_layers

    for i, block in enumerate(self.transformer_blocks[1:], start=1):
        img_mod = img_offset + 6 * i
        txt_mod = txt_offset + 6 * i
        temb = torch.cat(
            (pooled_temb[:, img_mod : img_mod + 6], pooled_temb[:, txt_mod : txt_mod + 6]),
            dim=1,
        )
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
            attention_mask_1d=attention_mask_1d,
            cpp_workspace=ws_dual,
            cpp_mask=mask_dual,
        )

    return hidden_states, encoder_hidden_states, hidden_states - original_h, encoder_hidden_states - original_enc


def run_remaining_single_blocks_chroma(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    pooled_temb: torch.Tensor,
    rotary_emb_single: torch.Tensor,
    attention_mask_1d: Optional[torch.Tensor],
    ws_single: dict | None,
    mask_single: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_hidden_states = hidden_states

    for i, block in enumerate(self.single_transformer_blocks[1:], start=1):
        start = 3 * i
        temb = pooled_temb[:, start : start + 3]
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=rotary_emb_single,
            attention_mask_1d=attention_mask_1d,
            cpp_workspace=ws_single,
            cpp_mask=mask_single,
        )

    return hidden_states, hidden_states - original_hidden_states


def cached_forward_chroma(
    self,
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    img_ids=None,
    txt_ids=None,
    attention_mask=None,
    joint_attention_kwargs: Optional[dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if self.residual_diff_threshold_multi < 0.0:
        return self._original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

    cache_key_prefix = _get_chroma_cache_key_prefix()
    del controlnet_blocks_repeat

    if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
        raise NotImplementedError("ControlNet is not supported in NunchakuChromaTransformer2dModel")
    if joint_attention_kwargs:
        raise NotImplementedError("joint_attention_kwargs is not supported in NunchakuChromaTransformer2dModel")

    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        img_ids = img_ids[0]

    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    batch_size = int(hidden_states.shape[0])

    input_vec = self.time_text_embed(timestep)
    pooled_temb = self.distilled_guidance_layer(input_vec)

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    txt_tokens = int(encoder_hidden_states.shape[1])
    attn_mask_1d = attention_mask
    image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + hidden_states.shape[1], *image_rotary_emb.shape[3:]])
    rotary_emb_txt = pack_rotemb(pad_tensor(image_rotary_emb[:, :txt_tokens, ...], 256, 1))
    rotary_emb_img = pack_rotemb(pad_tensor(image_rotary_emb[:, txt_tokens:, ...], 256, 1))
    rotary_emb_single = pack_rotemb(pad_tensor(image_rotary_emb, 256, 1))

    rotary_emb_txt = _expand_batch_dim(rotary_emb_txt, batch_size)
    rotary_emb_img = _expand_batch_dim(rotary_emb_img, batch_size)
    rotary_emb_single = _expand_batch_dim(rotary_emb_single, batch_size)

    use_cpp_ws = _should_use_cpp_additive_attn(
        attention_mask_1d=attn_mask_1d,
        hidden_states=hidden_states,
        head_dim=int(self.config.attention_head_dim),
    )
    ws_dual: dict | None = None
    ws_single: dict | None = None
    mask_dual: torch.Tensor | None = None
    mask_single: torch.Tensor | None = None
    if use_cpp_ws:
        ws_dual, ws_single, mask_dual, mask_single = _prepare_cpp_context(
            self, hidden_states, attention_mask, txt_tokens=txt_tokens, img_tokens=int(hidden_states.shape[1])
        )

    num_layers = len(self.transformer_blocks)
    num_single = len(self.single_transformer_blocks)
    img_offset = 3 * num_single
    txt_offset = img_offset + 6 * num_layers

    original_hidden_states = hidden_states
    original_encoder_hidden_states = encoder_hidden_states
    first_temb = torch.cat(
        (pooled_temb[:, img_offset : img_offset + 6], pooled_temb[:, txt_offset : txt_offset + 6]),
        dim=1,
    )
    first_block = self.transformer_blocks[0]
    encoder_hidden_states, hidden_states = first_block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=first_temb,
        image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
        attention_mask_1d=attn_mask_1d,
        cpp_workspace=ws_dual,
        cpp_mask=mask_dual,
    )
    # Chroma is more sensitive to drift in the text stream than Flux/Qwen.
    # Gate cache reuse on both streams to avoid visually obvious false-positive hits.
    first_hidden_states_residual_multi = torch.cat(
        [
            encoder_hidden_states - original_encoder_hidden_states,
            hidden_states - original_hidden_states,
        ],
        dim=1,
    )
    allow_multi_cache = _should_allow_chroma_multi_cache(self, original_hidden_states, first_temb, cache_key_prefix)

    call_remaining_fn = run_remaining_multi_blocks_chroma if self.use_double_fb_cache else run_remaining_blocks_chroma
    hidden_states, encoder_hidden_states, _ = check_and_apply_cache(
        first_residual=first_hidden_states_residual_multi,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        threshold=self.residual_diff_threshold_multi,
        parallelized=False,
        mode="multi",
        verbose=self.verbose if hasattr(self, "verbose") else False,
        call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: call_remaining_fn(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_temb=pooled_temb,
            rotary_emb_img=rotary_emb_img,
            rotary_emb_txt=rotary_emb_txt,
            rotary_emb_single=rotary_emb_single,
            attention_mask_1d=attn_mask_1d,
            ws_dual=ws_dual,
            ws_single=ws_single,
            mask_dual=mask_dual,
            mask_single=mask_single,
            txt_tokens=txt_tokens,
        ),
        remaining_kwargs={},
        cache_key_prefix=cache_key_prefix,
        allow_cache=allow_multi_cache,
    )

    if self.use_double_fb_cache:
        cat_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        original_cat = cat_hidden_states
        first_single_temb = pooled_temb[:, :3]
        first_single_block = self.single_transformer_blocks[0]
        cat_hidden_states = first_single_block(
            hidden_states=cat_hidden_states,
            temb=first_single_temb,
            image_rotary_emb=rotary_emb_single,
            attention_mask_1d=attn_mask_1d,
            cpp_workspace=ws_single,
            cpp_mask=mask_single,
        )
        first_hidden_states_residual_single = cat_hidden_states - original_cat

        cat_hidden_states, _, _ = check_and_apply_cache(
            first_residual=first_hidden_states_residual_single,
            hidden_states=cat_hidden_states,
            encoder_hidden_states=None,
            threshold=self.residual_diff_threshold_single,
            parallelized=False,
            mode="single",
            verbose=self.verbose if hasattr(self, "verbose") else False,
            call_remaining_fn=lambda hidden_states, encoder_hidden_states, **kw: run_remaining_single_blocks_chroma(
                self,
                hidden_states,
                encoder_hidden_states,
                pooled_temb=pooled_temb,
                rotary_emb_single=rotary_emb_single,
                attention_mask_1d=attn_mask_1d,
                ws_single=ws_single,
                mask_single=mask_single,
            ),
            remaining_kwargs={},
            cache_key_prefix=cache_key_prefix,
        )
        hidden_states = cat_hidden_states[:, txt_tokens:, ...]

    temb = pooled_temb[:, -2:]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
