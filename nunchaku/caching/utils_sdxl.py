"""
Caching utilities for SDXL UNet models.

This module implements a first-block cache for SDXL UNet inference by running the
first down block, comparing its output with the previous step, and reusing the
cached final UNet output when the difference is sufficiently small.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers

from .fbcache import get_buffer, get_can_use_cache, set_buffer


def _run_down_block(
    downsample_block,
    *,
    sample: torch.Tensor,
    emb: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cross_attention_kwargs: Optional[Dict[str, Any]],
    encoder_attention_mask: Optional[torch.Tensor],
    adapter_residuals: Optional[list[torch.Tensor]],
) -> tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        additional_residuals: Dict[str, torch.Tensor] = {}
        if adapter_residuals:
            additional_residuals["additional_residuals"] = adapter_residuals.pop(0)

        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
            **additional_residuals,
        )
    else:
        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        if adapter_residuals:
            sample = sample + adapter_residuals.pop(0)

    return sample, res_samples


def cached_forward_sdxl(
    self,
    sample: torch.Tensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor, ...]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor, ...]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple[torch.Tensor]]:
    """
    Forward pass for SDXL UNet with first-block caching.
    """
    default_overall_up_factor = 2**self.num_upsamplers
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            forward_upsample_size = True
            break

    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    sample = self.conv_in(sample)

    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    if cross_attention_kwargs is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    adapter_residuals: Optional[list[torch.Tensor]] = None

    if down_intrablock_additional_residuals is not None:
        adapter_residuals = list(down_intrablock_additional_residuals)
    elif mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated and will "
            "be removed in diffusers 1.3.0. `down_block_additional_residuals` should only be used for ControlNet. "
            "Please make sure to use `down_intrablock_additional_residuals` instead.",
            standard_warn=False,
        )
        adapter_residuals = list(down_block_additional_residuals)

    down_block_res_samples = (sample,)

    first_block = self.down_blocks[0]
    sample, res_samples = _run_down_block(
        first_block,
        sample=sample,
        emb=emb,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        encoder_attention_mask=encoder_attention_mask,
        adapter_residuals=adapter_residuals,
    )
    down_block_res_samples += res_samples

    can_use_cache, diff = get_can_use_cache(
        sample,
        threshold=self.residual_diff_threshold,
        parallelized=False,
        mode="single",
    )
    cached_final_output = get_buffer("final_output") if can_use_cache else None

    torch._dynamo.graph_break()
    if can_use_cache and cached_final_output is not None:
        if self.verbose:
            diff_val = diff.item() if isinstance(diff, torch.Tensor) else diff
            print(f"[SDXL] Cache hit! diff={diff_val:.6f}")
        sample = cached_final_output.to(device=sample.device, dtype=sample.dtype)
    else:
        if self.verbose:
            diff_val = diff.item() if isinstance(diff, torch.Tensor) else diff
            status = "cache buffer missing, fallback to miss" if can_use_cache else "Cache miss"
            print(f"[SDXL] {status}. diff={diff_val:.6f}")

        set_buffer("first_single_hidden_states_residual", sample)

        for downsample_block in self.down_blocks[1:]:
            sample, res_samples = _run_down_block(
                downsample_block,
                sample=sample,
                emb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                adapter_residuals=adapter_residuals,
            )
            down_block_res_samples += res_samples

        if is_controlnet:
            down_block_res_samples = tuple(
                down_block_res_sample + down_block_additional_residual
                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                )
            )

        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            if adapter_residuals and sample.shape == adapter_residuals[0].shape:
                sample = sample + adapter_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        set_buffer("final_output", sample)

    torch._dynamo.graph_break()

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)
