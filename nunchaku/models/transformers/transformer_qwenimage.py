"""
This module provides implementations of NunchakuQwenImageTransformer2DModel and its building blocks.
"""

import gc
import json
import os
import types
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from diffusers.utils import deprecate, logging as diffusers_logging
from huggingface_hub import utils

from ...torch_transfer_utils import pin_state_dict, resolve_pin_memory
from ...utils import get_precision, pad_tensor
from ..attention import NunchakuBaseAttention, NunchakuFeedForward
from ..attention_processors.qwenimage import NunchakuQwenImageNaiveFA2Processor
from ..embeddings import pack_rotemb
from ..linear import SVDQW4A4Linear
from ..linear_w8a16 import SVDQW8A16Linear
from ..utils import CPUOffloadManager, fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key

logger = diffusers_logging.get_logger(__name__)

try:
    from diffusers.models.transformers.transformer_qwenimage import QwenEmbedLayer3DRope
except ImportError:
    QwenEmbedLayer3DRope = None

try:
    from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
except ImportError:
    compute_text_seq_len_from_mask = None

try:
    from diffusers.utils import apply_lora_scale
except ImportError:

    def apply_lora_scale(_kwargs_name: str = "joint_attention_kwargs"):
        def decorator(func):
            return func

        return decorator

try:
    from ...lora.common.mixin import SVDQLoRAMixin
except ImportError:
    SVDQLoRAMixin = None


def _compute_text_seq_len_compat(
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: Optional[torch.Tensor],
    txt_seq_lens: Optional[List[int]],
) -> tuple[int, Optional[torch.Tensor]]:
    if compute_text_seq_len_from_mask is not None:
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )
        return int(text_seq_len), encoder_hidden_states_mask

    if txt_seq_lens is not None:
        return int(max(txt_seq_lens)), encoder_hidden_states_mask

    if encoder_hidden_states_mask is not None:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(dtype=torch.bool)
        text_seq_len = int(encoder_hidden_states_mask.sum(dim=1).max().item())
        return text_seq_len, encoder_hidden_states_mask

    return int(encoder_hidden_states.shape[1]), encoder_hidden_states_mask


def _call_time_text_embed(
    module: "NunchakuQwenImageTransformer2DModel",
    timestep: torch.Tensor,
    hidden_states: torch.Tensor,
    guidance: Optional[torch.Tensor],
    additional_t_cond: Optional[torch.Tensor],
) -> torch.Tensor:
    if guidance is None:
        if additional_t_cond is not None:
            try:
                return module.time_text_embed(timestep, hidden_states, additional_t_cond)
            except TypeError:
                pass
        return module.time_text_embed(timestep, hidden_states)

    if additional_t_cond is not None:
        try:
            return module.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        except TypeError:
            pass
    return module.time_text_embed(timestep, guidance, hidden_states)


def _pack_qwen_rotary_freqs(freqs: torch.Tensor, batch_size: int) -> torch.Tensor:
    if torch.is_complex(freqs):
        packed = torch.view_as_real(freqs).unsqueeze(0).unsqueeze(3)
        # Qwen RoPE stores complex values as [cos, sin]; the fused CUDA path expects [sin, cos].
        packed = torch.flip(packed, dims=[-1])
        packed = pack_rotemb(pad_tensor(packed, 256, 1))
    else:
        packed = freqs

    if packed.ndim != 3:
        raise ValueError(f"Unsupported packed Qwen rotary shape: {tuple(packed.shape)}")
    if packed.shape[0] != batch_size:
        packed = packed.expand(batch_size, -1, -1).contiguous()
    return packed


def _get_nested_attr_qwen(obj: Any, path: str) -> Any | None:
    for part in path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj


def _patch_mod_lora_forward(module: SVDQW8A16Linear) -> SVDQW8A16Linear:
    if getattr(module, "_qwen_lora_patched", False):
        return module

    module.register_buffer("_qwen_lora_A", None, persistent=False)
    module.register_buffer("_qwen_lora_B", None, persistent=False)
    module._qwen_lora_strength = 1.0
    module._qwen_orig_forward = module.forward

    def _forward_with_qwen_lora(self: SVDQW8A16Linear, x: torch.Tensor) -> torch.Tensor:
        output = self._qwen_orig_forward(x)
        if self._qwen_lora_A is None or self._qwen_lora_B is None:
            return output
        lora_input = self._get_smoothed_input(x).to(dtype=self._qwen_lora_A.dtype)
        lora_hidden = F.linear(lora_input, self._qwen_lora_A)
        lora_output = F.linear(lora_hidden, self._qwen_lora_B).to(dtype=output.dtype)
        return output + self._qwen_lora_strength * lora_output

    module.forward = types.MethodType(_forward_with_qwen_lora, module)
    module._qwen_lora_patched = True
    return module


def _set_mod_lora(
    module: SVDQW8A16Linear,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    strength: float,
) -> None:
    module = _patch_mod_lora_forward(module)
    dtype = module.bias.dtype if module.bias is not None else module.wscales.dtype
    device = module.qweight.device
    lora_a = lora_a.to(device=device, dtype=dtype)
    smooth = module.smooth_factor
    if smooth.dtype != lora_a.dtype or smooth.device != lora_a.device:
        smooth = smooth.to(device=lora_a.device, dtype=lora_a.dtype)
    module._qwen_lora_A = lora_a.mul(smooth.view(1, -1)).contiguous()
    module._qwen_lora_B = lora_b.to(device=device, dtype=dtype).contiguous()
    module._qwen_lora_strength = float(strength)


def _reset_mod_lora(module: SVDQW8A16Linear) -> None:
    if not getattr(module, "_qwen_lora_patched", False):
        return
    module._qwen_lora_A = None
    module._qwen_lora_B = None
    module._qwen_lora_strength = 1.0


def _qwen_linear_kwargs(kwargs: dict[str, Any], linear_cls) -> dict[str, Any]:
    out = dict(kwargs)
    out.pop("qwen_main_linear_cls", None)
    out.pop("qwen_main_weight_layout", None)
    out.pop("qwen_mod_weight_layout", None)
    if linear_cls is SVDQW4A4Linear:
        out.pop("weight_layout", None)
        out.pop("pad_k", None)
        out.pop("pad_n", None)
    return out


class _QwenImageSelectiveOffloadManager:
    """Wrap ``CPUOffloadManager`` for the homogeneous subset of QwenImage blocks.

    Blocks touched by ``fp_overrides`` keep their original mixed-precision
    structure and stay resident on GPU. The remaining homogeneous quantized
    blocks continue to use the existing ping-pong offload path unchanged.
    """

    def __init__(
        self,
        blocks: list[QwenImageTransformerBlock],
        special_block_indices: set[int] | frozenset[int],
        *,
        use_pin_memory: bool = True,
        on_gpu_modules: Optional[list[torch.nn.Module]] = None,
        num_blocks_on_gpu: int = 1,
    ):
        self.blocks = blocks
        self.special_block_indices = frozenset(int(i) for i in special_block_indices)
        self.managed_block_indices = [i for i in range(len(blocks)) if i not in self.special_block_indices]
        self.global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(self.managed_block_indices)}
        self._active_local_block_idx: Optional[int] = None
        self.device: Optional[torch.device] = None

        keep_on_gpu = list(on_gpu_modules or [])
        keep_on_gpu.extend(self.blocks[i] for i in sorted(self.special_block_indices))

        deduped_keep_on_gpu: list[torch.nn.Module] = []
        seen_module_ids: set[int] = set()
        for module in keep_on_gpu:
            module_id = id(module)
            if module_id in seen_module_ids:
                continue
            deduped_keep_on_gpu.append(module)
            seen_module_ids.add(module_id)
        self._keep_on_gpu_modules = deduped_keep_on_gpu

        self._offload_manager: Optional[CPUOffloadManager] = None
        if self.managed_block_indices:
            managed_blocks = [self.blocks[i] for i in self.managed_block_indices]
            self._offload_manager = CPUOffloadManager(
                managed_blocks,
                use_pin_memory=use_pin_memory,
                on_gpu_modules=self._keep_on_gpu_modules,
                num_blocks_on_gpu=min(num_blocks_on_gpu, len(managed_blocks)),
            )
            self.device = self._offload_manager.device

    def set_device(self, device: torch.device | str):
        if self._offload_manager is not None:
            self._offload_manager.set_device(device)
            self.device = self._offload_manager.device
            return

        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"
        if self.device == device:
            return
        self.device = device
        for module in self._keep_on_gpu_modules:
            module.to(device)

    def initialize(self, stream: torch.cuda.Stream | None = None):
        self._active_local_block_idx = None
        if self._offload_manager is not None:
            self._offload_manager.initialize(stream)

    def get_block(self, block_idx: int | None = None):
        if block_idx is None:
            raise ValueError("QwenImage selective offload requires an explicit block index.")

        local_block_idx = self.global_to_local.get(block_idx)
        self._active_local_block_idx = local_block_idx
        if local_block_idx is None or self._offload_manager is None:
            return self.blocks[block_idx]
        return self._offload_manager.get_block(local_block_idx)

    def step(self, compute_stream: torch.cuda.Stream | None = None):
        if self._active_local_block_idx is None or self._offload_manager is None:
            return
        self._offload_manager.step(compute_stream)
        self._active_local_block_idx = None


class NunchakuQwenAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized quantized attention module for QwenImage.

    Parameters
    ----------
    other : Attention
        The original QwenImage Attention module to wrap and quantize.
    processor : str, default="flashattn2"
        The attention processor to use.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: Attention, processor: str = "flashattn2", linear_cls=SVDQW4A4Linear, **kwargs):
        super(NunchakuQwenAttention, self).__init__(processor)
        self._qwen_main_uses_w8 = linear_cls is SVDQW8A16Linear
        self.inner_dim = getattr(other, "inner_dim", None)
        self.inner_kv_dim = getattr(other, "inner_kv_dim", None)
        self.query_dim = getattr(other, "query_dim", None)
        self.use_bias = getattr(other, "use_bias", False)
        self.is_cross_attention = getattr(other, "is_cross_attention", False)
        self.cross_attention_dim = getattr(other, "cross_attention_dim", None)
        self.upcast_attention = getattr(other, "upcast_attention", False)
        self.upcast_softmax = getattr(other, "upcast_softmax", False)
        self.rescale_output_factor = getattr(other, "rescale_output_factor", 1.0)
        self.residual_connection = getattr(other, "residual_connection", False)
        self.dropout = getattr(other, "dropout", 0.0)
        self.fused_projections = getattr(other, "fused_projections", False)
        self.out_dim = getattr(other, "out_dim", None)
        self.out_context_dim = getattr(other, "out_context_dim", None)
        self.context_pre_only = getattr(other, "context_pre_only", False)
        self.pre_only = getattr(other, "pre_only", False)
        self.is_causal = getattr(other, "is_causal", False)
        self.scale_qk = getattr(other, "scale_qk", True)
        self.scale = getattr(other, "scale", None)
        self.heads = getattr(other, "heads", None)
        self.head_dim = (self.inner_dim // self.heads) if self.inner_dim is not None and self.heads else None
        self.sliceable_head_dim = getattr(other, "sliceable_head_dim", None)
        self.added_kv_proj_dim = getattr(other, "added_kv_proj_dim", None)
        self.only_cross_attention = getattr(other, "only_cross_attention", False)
        self.group_norm = getattr(other, "group_norm", None)
        self.spatial_norm = getattr(other, "spatial_norm", None)

        self.norm_cross = getattr(other, "norm_cross", None)

        self.norm_q = getattr(other, "norm_q", None)
        self.norm_k = getattr(other, "norm_k", None)
        self.norm_added_q = getattr(other, "norm_added_q", None)
        self.norm_added_k = getattr(other, "norm_added_k", None)

        # Fuse the QKV projections for quantization
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = linear_cls.from_linear(to_qkv, **kwargs)
        self.to_out = other.to_out
        self.to_out[0] = linear_cls.from_linear(self.to_out[0], **kwargs)

        assert self.added_kv_proj_dim is not None
        # Fuse the additional QKV projections
        with torch.device("meta"):
            add_qkv_proj = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
        self.add_qkv_proj = linear_cls.from_linear(add_qkv_proj, **kwargs)
        self.to_add_out = linear_cls.from_linear(other.to_add_out, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for NunchakuQwenAttention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            Image stream input.
        encoder_hidden_states : torch.FloatTensor, optional
            Text stream input.
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Mask for encoder hidden states.
        attention_mask : torch.FloatTensor, optional
            Attention mask.
        image_rotary_emb : torch.Tensor, optional
            Rotary embedding for images.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            Attention outputs for image and text streams.
        """
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor to use. Only "flashattn2" is supported for now. See :class:`~nunchaku.models.attention_processors.qwenimage.NunchakuQwenImageNaiveFA2Processor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuQwenImageNaiveFA2Processor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


class NunchakuQwenImageTransformerBlock(QwenImageTransformerBlock):
    """
    Quantized QwenImage Transformer Block.

    This block supports quantized linear layers and joint attention for image and text streams.

    Parameters
    ----------
    other : QwenImageTransformerBlock
        The original transformer block to wrap and quantize.
    scale_shift : float, default=1.0
        Value to add to scale parameters. Default is 1.0.
        Nunchaku may have already fused the scale_shift into the linear weights, so you may want to set it to 0.
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, other: QwenImageTransformerBlock, scale_shift: float = 1.0, **kwargs):
        super(QwenImageTransformerBlock, self).__init__()
        main_linear_cls = kwargs.get("qwen_main_linear_cls", SVDQW4A4Linear)
        mod_kwargs = _qwen_linear_kwargs(kwargs, SVDQW8A16Linear)
        main_kwargs = _qwen_linear_kwargs(kwargs, main_linear_cls)
        mod_weight_layout = kwargs.get("qwen_mod_weight_layout", None)
        main_weight_layout = kwargs.get("qwen_main_weight_layout", None)
        if mod_weight_layout:
            mod_kwargs["weight_layout"] = mod_weight_layout
        if main_weight_layout:
            main_kwargs["weight_layout"] = main_weight_layout

        self.dim = other.dim
        self.img_mod = other.img_mod
        self.img_mod[1] = _patch_mod_lora_forward(SVDQW8A16Linear.from_linear(other.img_mod[1], **mod_kwargs))
        self.img_norm1 = other.img_norm1
        self.attn = NunchakuQwenAttention(other.attn, linear_cls=main_linear_cls, **main_kwargs)
        self.img_norm2 = other.img_norm2
        self.img_mlp = NunchakuFeedForward(other.img_mlp, linear_cls=main_linear_cls, **main_kwargs)

        # Text processing modules
        self.txt_mod = other.txt_mod
        self.txt_mod[1] = _patch_mod_lora_forward(SVDQW8A16Linear.from_linear(other.txt_mod[1], **mod_kwargs))
        self.txt_norm1 = other.txt_norm1
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = other.txt_norm2
        self.txt_mlp = NunchakuFeedForward(other.txt_mlp, linear_cls=main_linear_cls, **main_kwargs)

        self.scale_shift = scale_shift
        self.zero_cond_t = getattr(other, "zero_cond_t", False)

    def _modulate(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        *,
        scale_shift: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        shift : torch.Tensor
            Shift parameters.
        scale : torch.Tensor
            Scale parameters.
        gate : torch.Tensor
            Gate parameters.

        Returns
        -------
        tuple
            Modulated tensor and gate tensor.
        """
        if scale_shift is None:
            scale_shift = self.scale_shift
        if scale_shift != 0:
            scale = scale + scale_shift

        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)
            shift_result = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale_result = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate_result = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        if torch.is_grad_enabled():
            return x * scale_result + shift_result, gate_result
        x = x.mul_(scale_result)
        x.add_(shift_result)
        return x, gate_result

    def _split_stream_mod_params(
        self,
        mod_params: torch.Tensor,
        *,
        stream_name: str,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self._split_mod_params(mod_params)

    def _get_stream_scale_shift(self, stream_name: str) -> float:
        return self.scale_shift

    def _split_mod_params(
        self,
        mod_params: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Split raw modulation output into ``((shift1, scale1, gate1), (shift2, scale2, gate2))``.

        Default implementation assumes nunchaku's interleaved layout (quantization side
        has fused ``(1+scale)`` into bias, so ``scale_shift=0``).  Subclasses that keep
        mod layers in diffusers-native format should override this to use contiguous
        chunking.
        """
        return (
            (mod_params[:, 0::6], mod_params[:, 1::6], mod_params[:, 2::6]),
            (mod_params[:, 3::6], mod_params[:, 4::6], mod_params[:, 5::6]),
        )

    @staticmethod
    def _apply_gated_residual(
        residual: torch.Tensor,
        gate: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            return residual + gate * update
        residual.addcmul_(gate, update)
        return residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for NunchakuQwenImageTransformerBlock.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input.
        encoder_hidden_states : torch.Tensor
            Text stream input.
        encoder_hidden_states_mask : torch.Tensor
            Mask for encoder hidden states.
        temb : torch.Tensor
            Temporal embedding.
        image_rotary_emb : tuple of torch.Tensor, optional
            Rotary embedding for images.
        joint_attention_kwargs : dict, optional
            Additional arguments for joint attention.

        Returns
        -------
        tuple
            Updated encoder_hidden_states and hidden_states.
        """
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = self._split_stream_mod_params(
            img_mod_params, stream_name="img"
        )
        (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = self._split_stream_mod_params(
            txt_mod_params, stream_name="txt"
        )
        img_scale_shift = self._get_stream_scale_shift("img")
        txt_scale_shift = self._get_stream_scale_shift("txt")

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(
            img_normed, img_shift1, img_scale1, img_gate1, modulate_index, scale_shift=img_scale_shift
        )

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(
            txt_normed, txt_shift1, txt_scale1, txt_gate1, scale_shift=txt_scale_shift
        )

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_kwargs = {k: v for k, v in joint_attention_kwargs.items() if not k.startswith("_")}
        rotary_emb_for_attn = image_rotary_emb
        if getattr(self.attn, "_qwen_main_uses_w8", False):
            rotary_emb_for_attn = joint_attention_kwargs.get("_raw_image_rotary_emb", image_rotary_emb)
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=rotary_emb_for_attn,
            **attn_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = self._apply_gated_residual(hidden_states, img_gate1, img_attn_output)
        encoder_hidden_states = self._apply_gated_residual(encoder_hidden_states, txt_gate1, txt_attn_output)

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(
            img_normed2, img_shift2, img_scale2, img_gate2, modulate_index, scale_shift=img_scale_shift
        )
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = self._apply_gated_residual(hidden_states, img_gate2, img_mlp_output)

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(
            txt_normed2, txt_shift2, txt_scale2, txt_gate2, scale_shift=txt_scale_shift
        )
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = self._apply_gated_residual(encoder_hidden_states, txt_gate2, txt_mlp_output)

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformerBlockA16(QwenImageTransformerBlock):
    """Full-precision (FP16/BF16) transformer block that inherits directly from
    the diffusers ``QwenImageTransformerBlock``.

    The module tree is identical to diffusers, so ``load_state_dict`` works
    with diffusers-native key names stored in the nunchaku checkpoint.
    Only ``forward`` is overridden to translate nunchaku's packed rotary
    embeddings and ``_``-prefixed internal kwargs.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        jkw = joint_attention_kwargs or {}
        raw_rotary = jkw.get("_raw_image_rotary_emb", None)
        rotary = raw_rotary if raw_rotary is not None else image_rotary_emb
        clean_kwargs = {k: v for k, v in jkw.items() if not k.startswith("_")}

        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=rotary,
            joint_attention_kwargs=clean_kwargs,
            modulate_index=modulate_index,
        )


class NunchakuQwenImageTransformerBlockModOnly(NunchakuQwenImageTransformerBlock):
    """W4A4 block where selected sub-modules stay in original precision
    (FP16/BF16) instead of being quantized.

    Which sub-modules are kept in FP16 is controlled by ``fp16_suffixes``.
    The checkpoint stores these weights in diffusers-native format
    (``weight`` + ``bias``), so ``load_state_dict`` works without any
    extra conversion step.

    When mod layers (``img_mod`` / ``txt_mod``) are kept in FP16, modulation
    uses diffusers contiguous layout with ``(1+scale)`` convention
    (``scale_shift=1.0``).
    """

    _KNOWN_SUFFIXES = {"img_mod", "txt_mod", "attn", "img_mlp", "txt_mlp"}

    def __init__(
        self,
        other: QwenImageTransformerBlock,
        scale_shift: float = 1.0,
        fp16_suffixes: set[str] | None = None,
        **kwargs,
    ):
        super(QwenImageTransformerBlock, self).__init__()
        fp16 = fp16_suffixes or set()
        main_linear_cls = kwargs.get("qwen_main_linear_cls", SVDQW4A4Linear)
        mod_kwargs = _qwen_linear_kwargs(kwargs, SVDQW8A16Linear)
        main_kwargs = _qwen_linear_kwargs(kwargs, main_linear_cls)
        mod_weight_layout = kwargs.get("qwen_mod_weight_layout", None)
        main_weight_layout = kwargs.get("qwen_main_weight_layout", None)
        if mod_weight_layout:
            mod_kwargs["weight_layout"] = mod_weight_layout
        if main_weight_layout:
            main_kwargs["weight_layout"] = main_weight_layout

        unknown = fp16 - self._KNOWN_SUFFIXES
        if unknown:
            logger.warning(
                f"Unrecognised fp_overrides sub-modules {unknown}; "
                f"they will be ignored. Known: {self._KNOWN_SUFFIXES}"
            )

        self.dim = other.dim
        self._img_mod_fp16 = "img_mod" in fp16
        self._txt_mod_fp16 = "txt_mod" in fp16

        self.img_mod = other.img_mod
        if not self._img_mod_fp16:
            self.img_mod[1] = _patch_mod_lora_forward(SVDQW8A16Linear.from_linear(other.img_mod[1], **mod_kwargs))

        self.img_norm1 = other.img_norm1

        if "attn" in fp16:
            self.attn = other.attn
        else:
            self.attn = NunchakuQwenAttention(other.attn, linear_cls=main_linear_cls, **main_kwargs)

        self.img_norm2 = other.img_norm2

        if "img_mlp" in fp16:
            self.img_mlp = other.img_mlp
        else:
            self.img_mlp = NunchakuFeedForward(other.img_mlp, linear_cls=main_linear_cls, **main_kwargs)

        self.txt_mod = other.txt_mod
        if not self._txt_mod_fp16:
            self.txt_mod[1] = _patch_mod_lora_forward(SVDQW8A16Linear.from_linear(other.txt_mod[1], **mod_kwargs))

        self.txt_norm1 = other.txt_norm1
        self.txt_norm2 = other.txt_norm2

        if "txt_mlp" in fp16:
            self.txt_mlp = other.txt_mlp
        else:
            self.txt_mlp = NunchakuFeedForward(other.txt_mlp, linear_cls=main_linear_cls, **main_kwargs)

        mods_are_fp16 = self._img_mod_fp16 and self._txt_mod_fp16
        self.scale_shift = 1.0 if mods_are_fp16 else scale_shift
        self._quantized_scale_shift = scale_shift
        self.zero_cond_t = getattr(other, "zero_cond_t", False)

    @staticmethod
    def _split_mod_params_contiguous(mod_params):
        mod1, mod2 = mod_params.chunk(2, dim=-1)
        return mod1.chunk(3, dim=-1), mod2.chunk(3, dim=-1)

    def _split_stream_mod_params(self, mod_params, *, stream_name: str):
        if stream_name == "img":
            fp16_layout = self._img_mod_fp16
        elif stream_name == "txt":
            fp16_layout = self._txt_mod_fp16
        else:
            raise ValueError(f"Unsupported stream name: {stream_name}")
        if fp16_layout:
            return self._split_mod_params_contiguous(mod_params)
        return super()._split_mod_params(mod_params)

    def _get_stream_scale_shift(self, stream_name: str) -> float:
        if stream_name == "img":
            return 1.0 if self._img_mod_fp16 else self._quantized_scale_shift
        if stream_name == "txt":
            return 1.0 if self._txt_mod_fp16 else self._quantized_scale_shift
        raise ValueError(f"Unsupported stream name: {stream_name}")


_qwenimage_bases = (QwenImageTransformer2DModel, NunchakuModelLoaderMixin)
if SVDQLoRAMixin is not None:
    _qwenimage_bases = _qwenimage_bases + (SVDQLoRAMixin,)


class NunchakuQwenImageTransformer2DModel(*_qwenimage_bases):
    """
    Quantized QwenImage Transformer2DModel.

    This model supports quantized transformer blocks and optional CPU offloading for memory efficiency.

    Parameters
    ----------
    *args
        Positional arguments for the base model.
    **kwargs
        Keyword arguments for the base model and quantization.

    Attributes
    ----------
    offload : bool
        Whether CPU offloading is enabled.
    offload_manager : CPUOffloadManager or _QwenImageSelectiveOffloadManager or None
        Manager for offloading transformer blocks.
    _is_initialized : bool
        Whether the model has been patched for quantization.
    """

    def _pre_convert_lora_sd(self, lora_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize QwenImage-specific LoRA aliases to diffusers attention keys."""
        normalized_sd: dict[str, torch.Tensor] = {}
        adapter_names: set[str] = set()
        adapter_key_count = 0

        for key, value in lora_sd.items():
            new_key = key
            for marker in (".lora_A.", ".lora_B."):
                if marker not in key or not key.endswith(".weight"):
                    continue
                prefix, suffix = key.split(marker, 1)
                if suffix == "weight":
                    break
                adapter_name = suffix[: -len(".weight")]
                if not adapter_name:
                    break
                adapter_names.add(adapter_name)
                new_key = f"{prefix}{marker[:-1]}.weight"
                adapter_key_count += 1
                break

            if new_key in normalized_sd and new_key != key:
                logger.warning(
                    "QwenImage LoRA adapter-key collision: %s -> %s already exists; overriding with latest tensor.",
                    key,
                    new_key,
                )
            normalized_sd[new_key] = value

        if adapter_key_count:
            logger.info(
                "QwenImage LoRA adapter suffix normalization: %d keys normalized from adapters=%s",
                adapter_key_count,
                sorted(adapter_names),
            )
        lora_sd = normalized_sd

        rules = [
            {
                "type": "fused_qkv",
                "src": "attn.to_qkv",
                "dst": ["attn.to_q", "attn.to_k", "attn.to_v"],
            },
            {
                "type": "fused_qkv",
                "src": "attn.qkv",
                "dst": ["attn.to_q", "attn.to_k", "attn.to_v"],
            },
            {
                "type": "fused_qkv",
                "src": "attn.add_qkv_proj",
                "dst": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            },
            {
                "type": "fused_qkv",
                "src": "attn.to_added_qkv",
                "dst": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            },
            {
                "type": "linear",
                "src": "attn.to_out",
                "dst": "attn.to_out.0",
            },
            {
                "type": "linear",
                "src": "attn.out_proj",
                "dst": "attn.to_out.0",
            },
            {
                "type": "linear",
                "src": "attn.add_out_proj",
                "dst": "attn.to_add_out",
            },
            {
                "type": "linear",
                "src": "attn.context_out",
                "dst": "attn.to_add_out",
            },
        ]

        out: dict[str, torch.Tensor] = {}
        converted = 0

        for key, value in lora_sd.items():
            converted_this_key = False
            for rule in rules:
                marker = f".{rule['src']}."
                if marker not in key:
                    continue

                prefix, suffix = key.split(marker, 1)
                if suffix not in ("lora_A.weight", "lora_B.weight"):
                    continue

                if rule["type"] == "linear":
                    out[f"{prefix}.{rule['dst']}.{suffix}"] = value
                    converted += 1
                    converted_this_key = True
                    break

                targets = rule["dst"]
                if suffix == "lora_A.weight":
                    for target in targets:
                        out[f"{prefix}.{target}.{suffix}"] = value.clone()
                    converted += len(targets)
                    converted_this_key = True
                    break

                if value.shape[0] % len(targets) != 0:
                    logger.warning(
                        "QwenImage LoRA key conversion skipped uneven fused output: %s shape=%s",
                        key,
                        tuple(value.shape),
                    )
                    continue

                for target, part in zip(targets, value.chunk(len(targets), dim=0)):
                    out[f"{prefix}.{target}.{suffix}"] = part.contiguous()
                converted += len(targets)
                converted_this_key = True
                break

            if not converted_this_key:
                out[key] = value

        if converted:
            logger.info("QwenImage LoRA key conversion: %d keys converted", converted)

        return out

    def _init_lora_state(self) -> None:
        super()._init_lora_state()
        w8a16_ready = 0
        for block_prefix, idx, target_suffix, _cfg in self._iter_unquantized_targets():
            module = _get_nested_attr_qwen(self, f"{block_prefix}.{idx}.{target_suffix}")
            if isinstance(module, SVDQW8A16Linear):
                _patch_mod_lora_forward(module)
                w8a16_ready += 1
        if w8a16_ready:
            logger.info("QwenImage modulation LoRA ready on %d W8A16 targets", w8a16_ready)

    def _apply_unquantized_loras(self, loras: dict[str, torch.Tensor], strength: float) -> None:
        super()._apply_unquantized_loras(loras, strength)
        applied_w8a16 = 0
        for block_prefix, idx, target_suffix, _cfg in self._iter_unquantized_targets():
            out_prefix = f"{block_prefix}.{idx}.{target_suffix}"
            a_key = f"{out_prefix}.lora_A.weight"
            b_key = f"{out_prefix}.lora_B.weight"
            if a_key not in loras or b_key not in loras:
                continue
            module = _get_nested_attr_qwen(self, out_prefix)
            if not isinstance(module, SVDQW8A16Linear):
                continue
            _set_mod_lora(module, loras[a_key], loras[b_key], strength)
            applied_w8a16 += 1
        if applied_w8a16:
            logger.info("QwenImage modulation LoRA applied on %d W8A16 targets", applied_w8a16)

    def _reset_unquantized_loras(self) -> None:
        super()._reset_unquantized_loras()
        for block_prefix, idx, target_suffix, _cfg in self._iter_unquantized_targets():
            module = _get_nested_attr_qwen(self, f"{block_prefix}.{idx}.{target_suffix}")
            if isinstance(module, SVDQW8A16Linear):
                _reset_mod_lora(module)

    def _lora_key_map(self) -> dict[str, Any]:
        return {
            "block_prefixes": ["transformer_blocks"],
            "quantized_targets": {
                "attn.to_qkv": {
                    "lora_keys": ["attn.to_q", "attn.to_k", "attn.to_v"],
                    "type": "fused_qkv",
                },
                "attn.add_qkv_proj": {
                    "lora_keys": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                    "type": "fused_qkv",
                },
                "attn.to_out.0": {
                    "lora_keys": ["attn.to_out.0"],
                    "type": "linear",
                },
                "attn.to_add_out": {
                    "lora_keys": ["attn.to_add_out"],
                    "type": "linear",
                },
                "img_mlp.net.0.proj": {
                    "lora_keys": ["img_mlp.net.0.proj"],
                    "type": "linear",
                },
                "img_mlp.net.2": {
                    "lora_keys": ["img_mlp.net.2"],
                    "type": "linear",
                },
                "txt_mlp.net.0.proj": {
                    "lora_keys": ["txt_mlp.net.0.proj"],
                    "type": "linear",
                },
                "txt_mlp.net.2": {
                    "lora_keys": ["txt_mlp.net.2"],
                    "type": "linear",
                },
            },
            "unquantized_targets": {
                "img_mod.1": {
                    "lora_keys": ["img_mod.1"],
                    "type": "linear",
                },
                "txt_mod.1": {
                    "lora_keys": ["txt_mod.1"],
                    "type": "linear",
                },
            },
        }

    def __init__(self, *args, **kwargs):
        self.offload = kwargs.pop("offload", False)
        self.offload_manager = None
        self._offload_special_block_indices: frozenset[int] = frozenset()
        self._is_initialized = False
        self._cached_image_attention_mask = None
        self._cached_image_attention_mask_shape = None
        self._cached_image_attention_mask_device = None
        self._cached_joint_attention_mask = None
        self._cached_joint_attention_mask_source_ptr = None
        self._cached_joint_attention_mask_image_shape = None
        self._cached_joint_attention_mask_device = None
        self._cached_packed_qwen_rotary_emb = None
        self._cached_packed_qwen_rotary_emb_key = None
        super().__init__(*args, **kwargs)

    def _patch_model(self, **kwargs):
        """
        Patch the transformer blocks for quantization.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self
        """
        from ..fp_overrides import classify_block_overrides

        fp_overrides = kwargs.pop("fp_overrides", [])
        full_fp16, submodule_fp16 = classify_block_overrides(
            len(self.transformer_blocks), "transformer_blocks", fp_overrides,
        )
        self._offload_special_block_indices = frozenset(full_fp16 | set(submodule_fp16.keys()))

        for i, block in enumerate(self.transformer_blocks):
            if i in full_fp16:
                block.__class__ = NunchakuQwenImageTransformerBlockA16
            elif i in submodule_fp16:
                self.transformer_blocks[i] = NunchakuQwenImageTransformerBlockModOnly(
                    block, scale_shift=0, fp16_suffixes=set(submodule_fp16[i]), **kwargs
                )
            else:
                self.transformer_blocks[i] = NunchakuQwenImageTransformerBlock(
                    block, scale_shift=0, **kwargs
                )
        self._is_initialized = True
        return self

    def _get_image_attention_mask(self, batch_size: int, image_seq_len: int, device: torch.device) -> torch.Tensor:
        image_mask_shape = (batch_size, image_seq_len)
        if (
            self._cached_image_attention_mask is None
            or self._cached_image_attention_mask_shape != image_mask_shape
            or self._cached_image_attention_mask_device != device
        ):
            self._cached_image_attention_mask = torch.ones(image_mask_shape, dtype=torch.bool, device=device)
            self._cached_image_attention_mask_shape = image_mask_shape
            self._cached_image_attention_mask_device = device
        return self._cached_image_attention_mask

    def _get_joint_attention_mask(
        self,
        encoder_hidden_states_mask: torch.Tensor,
        batch_size: int,
        image_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        image_mask_shape = (batch_size, image_seq_len)
        source_ptr = encoder_hidden_states_mask.data_ptr()
        if (
            self._cached_joint_attention_mask is None
            or self._cached_joint_attention_mask_source_ptr != source_ptr
            or self._cached_joint_attention_mask_image_shape != image_mask_shape
            or self._cached_joint_attention_mask_device != device
        ):
            image_mask = self._get_image_attention_mask(batch_size, image_seq_len, device)
            self._cached_joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            self._cached_joint_attention_mask_source_ptr = source_ptr
            self._cached_joint_attention_mask_image_shape = image_mask_shape
            self._cached_joint_attention_mask_device = device
        return self._cached_joint_attention_mask

    def _get_packed_qwen_rotary_emb(
        self,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        *,
        batch_size: int,
        img_shapes: Optional[List[Tuple[int, int, int]]],
        text_seq_len: int,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if image_rotary_emb is None:
            return None

        cache_key = (repr(img_shapes), int(text_seq_len), device, int(batch_size))
        if self._cached_packed_qwen_rotary_emb is not None and self._cached_packed_qwen_rotary_emb_key == cache_key:
            return self._cached_packed_qwen_rotary_emb

        img_freqs, txt_freqs = image_rotary_emb
        packed_rotary_emb = (
            _pack_qwen_rotary_freqs(img_freqs, batch_size),
            _pack_qwen_rotary_freqs(txt_freqs, batch_size),
        )
        self._cached_packed_qwen_rotary_emb = packed_rotary_emb
        self._cached_packed_qwen_rotary_emb_key = cache_key
        return packed_rotary_emb

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a quantized model from a pretrained checkpoint.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the pretrained model checkpoint. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments for loading and quantization.

        Returns
        -------
        NunchakuQwenImageTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        AssertionError
            If the checkpoint is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)
        pin_memory = kwargs.get("pin_memory", "auto")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        config = json.loads(metadata.get("config", "{}"))
        rank = quantization_config.get("rank", 32)
        transformer = transformer.to(torch_dtype)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        from ..fp_overrides import parse_fp_overrides
        fp_overrides = parse_fp_overrides(quantization_config)

        weight_config = quantization_config.get("weight", {})
        weight_dtype = weight_config.get("dtype", "")
        weight_layout = str(quantization_config.get("weight_layout", "")).strip().lower()
        linear_backend = str(quantization_config.get("linear_backend", "")).strip().lower()
        use_int8_main = weight_dtype == "int8"
        use_plain_w8 = weight_layout == "plain" or linear_backend == "w8a16" or use_int8_main

        patch_kwargs = {
            "precision": precision,
            "rank": rank,
            "fp_overrides": fp_overrides,
        }
        if use_plain_w8:
            patch_kwargs["qwen_mod_weight_layout"] = "plain"
        if use_int8_main:
            patch_kwargs["qwen_main_linear_cls"] = SVDQW8A16Linear
            patch_kwargs["qwen_main_weight_layout"] = "plain"

        transformer._patch_model(**patch_kwargs)

        transformer = transformer.to_empty(device=device)
        # need to re-init the pos_embed as to_empty does not work on it
        axes_dims_rope = list(config.get("axes_dims_rope", [16, 56, 56]))
        if config.get("use_layer3d_rope", False) and QwenEmbedLayer3DRope is not None:
            transformer.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=axes_dims_rope, scale_rope=True)
        else:
            if config.get("use_layer3d_rope", False) and QwenEmbedLayer3DRope is None:
                logger.warning("QwenEmbedLayer3DRope is unavailable in current diffusers; falling back to QwenEmbedRope.")
            transformer.pos_embed = QwenEmbedRope(theta=10000, axes_dim=axes_dims_rope, scale_rope=True)

        patch_scale_key(transformer, model_state_dict)

        if resolve_pin_memory(pin_memory, device):
            model_state_dict = pin_state_dict(model_state_dict)
            print("pin_memory is enabled for the QwenImage transformer state dict.")

        transformer.load_state_dict(model_state_dict)
        if SVDQLoRAMixin is not None and hasattr(transformer, "_init_lora_state"):
            transformer._init_lora_state()
        else:
            logger.warning("SVDQLoRAMixin not available, LoRA support is disabled.")

        transformer.set_offload(offload)

        if kwargs.get("return_metadata", False):
            return transformer, metadata
        return transformer

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable asynchronous CPU offloading for transformer blocks.

        Parameters
        ----------
        offload : bool
            Whether to enable offloading.
        **kwargs
            Additional arguments for offload manager.

        See Also
        --------
        :class:`~nunchaku.models.utils.CPUOffloadManager`
        """
        if offload == self.offload:
            # nothing changed, just return
            return
        self.offload = offload
        if offload:
            if self._offload_special_block_indices:
                logger.info(
                    "QwenImage offload: keeping fp_overrides blocks on GPU: %s",
                    sorted(self._offload_special_block_indices),
                )
            self.offload_manager = _QwenImageSelectiveOffloadManager(
                self.transformer_blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                special_block_indices=self._offload_special_block_indices,
                on_gpu_modules=[
                    self.img_in,
                    self.txt_in,
                    self.txt_norm,
                    self.time_text_embed,
                    self.norm_out,
                    self.proj_out,
                ],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()

    @apply_lora_scale("attention_kwargs")
    def forward(
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
        Forward pass for the Nunchaku QwenImage transformer model with ControlNet support.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input of shape `(batch_size, image_sequence_length, in_channels)`.
        encoder_hidden_states : torch.Tensor, optional
            Text stream input of shape `(batch_size, text_sequence_length, joint_attention_dim)`.
        encoder_hidden_states_mask : torch.Tensor, optional
            Mask for encoder hidden states of shape `(batch_size, text_sequence_length)`.
        timestep : torch.LongTensor, optional
            Timestep for temporal embedding.
        img_shapes : list of tuple, optional
            Image shapes for rotary embedding.
        txt_seq_lens : list of int, optional
            Text sequence lengths.
        guidance : torch.Tensor, optional
            Guidance tensor (for classifier-free guidance).
        attention_kwargs : dict, optional
            Additional attention arguments. A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
        controlnet_block_samples : optional
            ControlNet block samples for residual connections.
        return_dict : bool, default=True
            Whether to return a dict or tuple.

        Returns
        -------
        torch.Tensor or Transformer2DModelOutput
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
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
                [[0] * prod(sample[0]) + [1] * sum(prod(s) for s in sample[1:]) for sample in img_shapes],
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

        raw_image_rotary_emb = image_rotary_emb

        image_rotary_emb = self._get_packed_qwen_rotary_emb(
            image_rotary_emb,
            batch_size=hidden_states.shape[0],
            img_shapes=img_shapes,
            text_seq_len=text_seq_len,
            device=hidden_states.device,
        )

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        block_attention_kwargs["_raw_image_rotary_emb"] = raw_image_rotary_emb
        interval_control = None
        if controlnet_block_samples is not None:
            interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
        if use_modern_qwenimage and encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            block_attention_kwargs["attention_mask"] = self._get_joint_attention_mask(
                encoder_hidden_states_mask, batch_size, image_seq_len, hidden_states.device
            )

        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)
        for block_idx, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)

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

                # controlnet residual - same logic as in diffusers QwenImageTransformer2DModel
                if controlnet_block_samples is not None:
                    hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

            if self.offload:
                self.offload_manager.step(compute_stream)

        if getattr(self, "zero_cond_t", False):
            temb = temb.chunk(2, dim=0)[0]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        """
        Override the default ``.to()`` method.

        If offload is enabled, prevents moving the model to GPU.
        Prevents changing dtype after quantization.

        Parameters
        ----------
        *args
            Positional arguments for ``.to()``.
        **kwargs
            Keyword arguments for ``.to()``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If attempting to change dtype after quantization.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        # Try converting arguments to torch.device in case they are passed as strings
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`."
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
                return self
        return super(type(self), self).to(*args, **kwargs)

