"""
This module provides Nunchaku ZImageTransformer2DModel and its building blocks in Python.
"""

import inspect
import json
import logging
from os import PathLike
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import FeedForward as ZImageFeedForward
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock
from huggingface_hub import utils

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLFeedForward

from ...ops.gemm import svdq_gemm_w4a4_cuda
from ...ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
from ...torch_transfer_utils import pin_state_dict, resolve_pin_memory
from ...utils import get_precision, pad_tensor
from ..attention import NunchakuBaseAttention
from ..attention_processors.zimage import NunchakuZSingleStreamAttnProcessor
from ..embeddings import pack_rotemb
try:
    from ...lora.common.mixin import SVDQLoRAMixin
except ImportError:
    SVDQLoRAMixin = None

from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key

logger = logging.getLogger(__name__)

# Compatibility fix for the released z-image-turbo quantized checkpoints.
# The down projection at ``layers.0.feed_forward.net.2`` can see a single
# activation channel spike under real inputs, which disproportionately harms
# W4A4 quantization quality for this model family.
ZIMAGE_TURBO_FFN_GUARD_LAYER_INDEX = 0
ZIMAGE_TURBO_FFN_GUARD_NET_INDEX = 2
ZIMAGE_TURBO_FFN_GUARD_CHANNEL_IDX = 10078
ZIMAGE_TURBO_FFN_GUARD_MAX_ABS = 4096.0
ZIMAGE_TURBO_FFN_GUARD_MODE = "clamp"
ZIMAGE_TURBO_KEPT_FP_SUPPORTED_SUFFIXES = (
    "feed_forward.net.0.proj",
    "feed_forward.net.2",
)


class NunchakuZImageRopeHook:
    """
    Hook class for caching and substition of packed `freqs_cis` tensor.
    """

    def __init__(self):
        self.packed_cache = {}

    def __call__(self, module: nn.Module, input_args: tuple, input_kwargs: dict):
        freqs_cis: torch.Tensor = input_kwargs.get("freqs_cis", None)
        if freqs_cis is None:
            return None
        cache_key = freqs_cis.data_ptr()
        packed_freqs_cis = self.packed_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            packed_freqs_cis = torch.view_as_real(freqs_cis).unsqueeze(3)
            packed_freqs_cis = torch.flip(packed_freqs_cis, dims=[-1])
            packed_freqs_cis = pack_rotemb(pad_tensor(packed_freqs_cis, 256, 1))
            self.packed_cache[cache_key] = packed_freqs_cis
        new_input_kwargs = input_kwargs.copy()
        new_input_kwargs["freqs_cis"] = packed_freqs_cis
        return input_args, new_input_kwargs


class NunchakuZImageFusedModule(nn.Module):
    """
    Fused module for quantized QKV projection, RMS normalization, and rotary embedding for ZImage attention.

    Parameters
    ----------
    qkv : SVDQW4A4Linear
        Quantized QKV projection layer.
    norm_q : RMSNorm
        RMSNorm for query.
    norm_k : RMSNorm
        RMSNorm for key.
    """

    def __init__(self, qkv: SVDQW4A4Linear, norm_q: RMSNorm, norm_k: RMSNorm):
        super().__init__()
        for name, param in qkv.named_parameters(prefix="qkv_"):
            setattr(self, name.replace(".", ""), param)
        self.qkv_precision = qkv.precision
        self.qkv_out_features = qkv.out_features
        for name, param in norm_q.named_parameters(prefix="norm_q_"):
            setattr(self, name.replace(".", ""), param)
        for name, param in norm_k.named_parameters(prefix="norm_k_"):
            setattr(self, name.replace(".", ""), param)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None):
        """
        Fuse QKV projection, RMS normalizaion and rotary embedding.

        Parameters
        ----------
        x : torch.Tensor
            The hidden states tensor
        freqs_cis : torch.Tensor, optional
            The rotary embedding tensor

        Returns
        -------
        The projection results of q, k, v. q result and k result are RMS-normalized and applied RoPE.
        """
        batch_size, seq_len, channels = x.shape
        x = x.view(batch_size * seq_len, channels)
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.qkv_proj_down,
            smooth=self.qkv_smooth_factor,
            fp4=self.qkv_precision == "nvfp4",
            pad_size=256,
        )

        # The kernel asserts rotary_emb.shape[0] * rotary_emb.shape[1] == M,
        # where M is the (padded) row count of quantized_x.  The rope hook
        # pads the seq dimension of freqs_cis to a multiple of 256
        # *independently per batch element*, producing shape [B, padded_S, D].
        # After flattening batch*seq the total B*padded_S can exceed
        # M = pad(B*S, 256) when B > 1 and S is not a multiple of 256.
        # Fix: take only the actual seq_len tokens from each batch element
        # (which is safe because seq_len is always a multiple of 32 >= 16, the
        # pack_rotemb tile size), flatten to [1, B*S, D], then pad to M.
        if freqs_cis is not None and batch_size > 1:
            M = quantized_x.shape[0]
            D = freqs_cis.shape[2]
            freqs_cis = freqs_cis[:, :seq_len, :].reshape(1, batch_size * seq_len, D)
            if freqs_cis.shape[1] < M:
                pad_len = M - freqs_cis.shape[1]
                freqs_cis = torch.nn.functional.pad(freqs_cis, (0, 0, 0, pad_len))
            freqs_cis = freqs_cis.contiguous()

        output = torch.empty(batch_size * seq_len, self.qkv_out_features, dtype=x.dtype, device=x.device)
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qkv_qweight,
            out=output,
            ascales=ascales,
            wscales=self.qkv_wscales,
            lora_act_in=lora_act_out,
            lora_up=self.qkv_proj_up,
            bias=getattr(self, "qkv_bias", None),
            fp4=self.qkv_precision == "nvfp4",
            alpha=1.0 if self.qkv_precision == "nvfp4" else None,
            wcscales=self.qkv_wcscales if self.qkv_precision == "nvfp4" else None,
            norm_q=self.norm_q_weight,
            norm_k=self.norm_k_weight,
            rotary_emb=freqs_cis,
        )

        output = output.view(batch_size, seq_len, -1)
        return output


class NunchakuZImageAttention(NunchakuBaseAttention):
    """
    Nunchaku-optimized Attention module for ZImage with quantized and fused QKV projections.

    Parameters
    ----------
    other : Attention
        The original Attention module in ZImage model.
    processor : str, optional
        The attention processor to use ("flashattn2" or "nunchaku-fp16").
    **kwargs
        Additional arguments for quantization.
    """

    def __init__(self, orig_attn: Attention, processor: str = "flashattn2", **kwargs):
        super(NunchakuZImageAttention, self).__init__(processor)
        self.inner_dim = orig_attn.inner_dim
        self.query_dim = orig_attn.query_dim
        self.use_bias = orig_attn.use_bias
        self.dropout = orig_attn.dropout
        self.out_dim = orig_attn.out_dim
        self.context_pre_only = orig_attn.context_pre_only
        self.pre_only = orig_attn.pre_only
        self.heads = orig_attn.heads
        self.rescale_output_factor = orig_attn.rescale_output_factor
        self.is_cross_attention = orig_attn.is_cross_attention

        # region sub-modules
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k
        with torch.device("meta"):
            to_qkv = fuse_linears([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v])
        self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
        self.to_out = orig_attn.to_out
        self.to_out[0] = SVDQW4A4Linear.from_linear(self.to_out[0], **kwargs)
        # end of region

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for NunchakuZImageAttention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        encoder_hidden_states : torch.Tensor, optional
            Encoder hidden states for cross-attention.
        attention_mask : torch.Tensor, optional
            Attention mask.
        **cross_attention_kwargs
            Additional arguments for cross attention.

        Returns
        -------
        Output of the attention processor.
        """
        return self.processor(
            attn=self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def set_processor(self, processor: str):
        """
        Set the attention processor.

        Parameters
        ----------
        processor : str
            Name of the processor ("flashattn2").

            - ``"flashattn2"``: Standard FlashAttention-2. See :class:`~nunchaku.models.attention_processors.zimage.NunchakuZSingleStreamAttnProcessor`.

        Raises
        ------
        ValueError
            If the processor is not supported.
        """
        if processor == "flashattn2":
            self.processor = NunchakuZSingleStreamAttnProcessor()
        else:
            raise ValueError(f"Processor {processor} is not supported")


def _convert_z_image_ff(z_ff: ZImageFeedForward) -> FeedForward:
    """
    Replace custom FeedForward module in `ZImageTransformerBlock`s with standard FeedForward in diffusers lib.

    Parameters
    ----------
    z_ff : ZImageFeedForward
        The feed forward sub-module in the ZImageTransformerBlock module

    Returns
    -------
    FeedForward
        A diffusers FeedForward module which is equivalent to the input `z_ff`

    """
    assert isinstance(z_ff, ZImageFeedForward)
    assert z_ff.w1.in_features == z_ff.w3.in_features
    assert z_ff.w1.out_features == z_ff.w3.out_features
    assert z_ff.w1.out_features == z_ff.w2.in_features
    # Construct directly on the target device to avoid large temporary CPU
    # allocations when this path runs during meta-device model patching.
    target_dtype = z_ff.w1.weight.dtype
    target_device = z_ff.w1.weight.device
    with torch.device(target_device):
        converted_ff = FeedForward(
            dim=z_ff.w1.in_features,
            dim_out=z_ff.w2.out_features,
            dropout=0.0,
            activation_fn="swiglu",
            inner_dim=z_ff.w2.in_features,
            bias=False,
        )
    converted_ff = converted_ff.to(dtype=target_dtype)
    return converted_ff


def replace_fused_module(module, incompatible_keys):
    assert isinstance(module, NunchakuZImageAttention)
    module.fused_module = NunchakuZImageFusedModule(module.to_qkv, module.norm_q, module.norm_k)
    del module.to_qkv
    del module.norm_q
    del module.norm_k


def _normalize_fp_overrides(quantization_config: dict) -> set[str]:
    raw_overrides = quantization_config.get("fp_overrides", None)
    if raw_overrides is None:
        return set()
    if isinstance(raw_overrides, str):
        return {raw_overrides}
    if isinstance(raw_overrides, dict):
        return {str(name) for name in raw_overrides.keys()}
    if isinstance(raw_overrides, Iterable):
        return {str(name) for name in raw_overrides}
    raise TypeError(f"Unsupported fp_overrides format: {type(raw_overrides).__name__}")


def _validate_ffn_fp_overrides(fp_overrides: set[str], model_state_dict: dict[str, torch.Tensor]) -> set[str]:
    unsupported = sorted(
        override for override in fp_overrides if not override.endswith(ZIMAGE_TURBO_KEPT_FP_SUPPORTED_SUFFIXES)
    )
    if unsupported:
        supported = ", ".join(ZIMAGE_TURBO_KEPT_FP_SUPPORTED_SUFFIXES)
        raise NotImplementedError(
            "Only FFN fp_overrides are supported for z-image-turbo. "
            f"Unsupported fp_overrides: {unsupported}. Supported suffixes: {supported}"
        )

    missing_weights = sorted(f"{override}.weight" for override in fp_overrides if f"{override}.weight" not in model_state_dict)
    if missing_weights:
        raise KeyError(
            "z-image-turbo FFN fp_overrides were declared in metadata, "
            f"but the checkpoint is missing float weights: {missing_weights}"
        )

    return {override for override in fp_overrides if f"{override}.bias" in model_state_dict}


def _rebuild_linear(linear: nn.Linear, bias: bool) -> nn.Linear:
    if (linear.bias is not None) == bias:
        return linear
    return nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=bias,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )


def _prepare_zimage_ff_keep_fp_linears(
    ff: FeedForward, module_prefix: str, kept_fp_bias_modules: set[str] | None = None
) -> tuple[str, str]:
    kept_fp_bias_modules = kept_fp_bias_modules or set()
    up_proj_name = f"{module_prefix}.feed_forward.net.0.proj"
    down_proj_name = f"{module_prefix}.feed_forward.net.2"

    assert hasattr(ff.net[0], "proj"), "Unexpected ZImage FFN structure: net[0] must have a proj linear"
    assert isinstance(ff.net[0].proj, nn.Linear), "Unexpected ZImage FFN structure: net[0].proj must be nn.Linear"
    assert isinstance(ff.net[2], nn.Linear), "Unexpected ZImage FFN structure: net[2] must be nn.Linear"

    ff.net[0].proj = _rebuild_linear(ff.net[0].proj, bias=up_proj_name in kept_fp_bias_modules)
    ff.net[2] = _rebuild_linear(ff.net[2], bias=down_proj_name in kept_fp_bias_modules)
    return up_proj_name, down_proj_name


def _log_zimage_ff_kept_fp(module_name: str, has_bias: bool):
    logger.info("z-image-turbo FFN fp_override enabled: %s (bias=%s)", module_name, has_bias)


class ZImageTurboFFNDownGuard(nn.Module):
    """
    Compatibility wrapper for the z-image-turbo FFN down projection.

    This wrapper is applied only to ``layers.0.feed_forward.net.2`` and clamps
    the known problematic activation channel before it reaches W4A4 quantization.
    """

    def __init__(self, inner: nn.Module, channel_idx: int, max_abs: float, mode: str = "clamp"):
        super().__init__()
        self.inner = inner
        self.channel_idx = int(channel_idx)
        self.max_abs = float(max_abs)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return self.inner(x)

        if self.channel_idx < x.shape[-1]:
            channel = x[..., self.channel_idx]
            if self.mode == "clamp":
                if not torch.any(channel.abs() > self.max_abs):
                    return self.inner(x)
                guarded_channel = channel.clamp(-self.max_abs, self.max_abs)
            elif self.mode == "zero":
                if not torch.any(channel != 0):
                    return self.inner(x)
                guarded_channel = torch.zeros_like(channel)
            x = x.clone()
            x[..., self.channel_idx] = guarded_channel
        return self.inner(x)

    def extra_repr(self) -> str:
        return f"channel_idx={self.channel_idx}, max_abs={self.max_abs}, mode={self.mode}"


class NunchakuZImageFeedForward(NunchakuSDXLFeedForward):
    """
    Quantized feed-forward block for :class:`NunchakuZImageTransformerBlock`.

    Replaces linear layers in a FeedForward block with :class:`~nunchaku.models.linear.SVDQW4A4Linear` for quantized inference.

    Parameters
    ----------
    ff : FeedForward
        Source ZImage FeedForward module to quantize.
    **kwargs :
        Additional arguments for SVDQW4A4Linear.
    """

    def __init__(
        self,
        ff: ZImageFeedForward,
        module_prefix: str,
        fp_overrides: set[str] | None = None,
        kept_fp_bias_modules: set[str] | None = None,
        **kwargs,
    ):
        converted_ff = _convert_z_image_ff(ff)
        super(FeedForward, self).__init__()

        fp_overrides = fp_overrides or set()
        kept_fp_bias_modules = kept_fp_bias_modules or set()

        self.net = converted_ff.net
        up_proj_name, down_proj_name = _prepare_zimage_ff_keep_fp_linears(
            converted_ff,
            module_prefix=module_prefix,
            kept_fp_bias_modules=kept_fp_bias_modules,
        )

        if up_proj_name not in fp_overrides:
            self.net[0].proj = SVDQW4A4Linear.from_linear(self.net[0].proj, **kwargs)
        else:
            _log_zimage_ff_kept_fp(up_proj_name, self.net[0].proj.bias is not None)
        if down_proj_name not in fp_overrides:
            self.net[2] = SVDQW4A4Linear.from_linear(self.net[2], **kwargs)
        else:
            _log_zimage_ff_kept_fp(down_proj_name, self.net[2].bias is not None)


_zimage_bases = (ZImageTransformer2DModel, NunchakuModelLoaderMixin)
if SVDQLoRAMixin is not None:
    _zimage_bases = _zimage_bases + (SVDQLoRAMixin,)


class NunchakuZImageTransformer2DModel(*_zimage_bases):
    """
    Nunchaku-optimized ZImageTransformer2DModel.
    """

    def _unwrap_quantized_module(self, module):
        if isinstance(module, ZImageTurboFFNDownGuard):
            return module.inner
        return module

    def _pre_convert_lora_sd(self, lora_sd):
        """Convert ZImage-native ``w1``/``w3``/``w2`` FFN keys to diffusers convention.

        ZImage's custom FeedForward uses ``w1`` (gate/silu), ``w3`` (up),
        ``w2`` (down).  After conversion to diffusers :class:`FeedForward`
        with ``activation_fn="swiglu"`` the mapping is:

        - ``w3`` (up) + ``w1`` (gate) → ``net.0.proj``
          (SwiGLU proj output = ``[up; gate]``, i.e. w3 first, w1 second)
        - ``w2`` → ``net.2``  (down projection)
        """
        has_w_keys = any(".feed_forward.w1." in k or ".feed_forward.w2." in k for k in lora_sd)
        if not has_w_keys:
            return lora_sd

        out: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()

        for k in list(lora_sd.keys()):
            if k in consumed:
                continue

            if ".feed_forward.w1." in k:
                prefix, suffix = k.split(".feed_forward.w1.", 1)
                w3_key = f"{prefix}.feed_forward.w3.{suffix}"

                w1_val = lora_sd[k]
                w3_val = lora_sd.get(w3_key)
                if w3_val is None:
                    out[k] = w1_val
                    continue

                new_key = f"{prefix}.feed_forward.net.0.proj.{suffix}"
                if "lora_A" in suffix:
                    if torch.equal(w1_val, w3_val):
                        out[new_key] = w1_val
                    else:
                        # lora_A: cat along rank dim; w3 first (up), w1 second (gate)
                        out[new_key] = torch.cat([w3_val, w1_val], dim=0)
                elif "lora_B" in suffix:
                    w1_a_key = f"{prefix}.feed_forward.w1.lora_A.weight"
                    w3_a_key = f"{prefix}.feed_forward.w3.lora_A.weight"
                    shared_a = (
                        w1_a_key in lora_sd
                        and w3_a_key in lora_sd
                        and torch.equal(lora_sd[w1_a_key], lora_sd[w3_a_key])
                    )
                    if shared_a:
                        # Shared lora_A → simple cat of lora_B; w3 first (up), w1 second (gate)
                        out[new_key] = torch.cat([w3_val, w1_val], dim=0)
                    else:
                        # Different lora_A → block-diagonal lora_B; w3 first, w1 second
                        rank = w1_val.shape[1]
                        fused_b = torch.zeros(
                            w3_val.shape[0] + w1_val.shape[0],
                            rank * 2,
                            dtype=w1_val.dtype,
                            device=w1_val.device,
                        )
                        fused_b[: w3_val.shape[0], :rank] = w3_val
                        fused_b[w3_val.shape[0] :, rank:] = w1_val
                        out[new_key] = fused_b
                else:
                    out[k] = w1_val
                    continue

                consumed.update([k, w3_key])
                continue

            if ".feed_forward.w3." in k:
                if k not in consumed:
                    out[k] = lora_sd[k]
                continue

            if ".feed_forward.w2." in k:
                new_key = k.replace(".feed_forward.w2.", ".feed_forward.net.2.")
                out[new_key] = lora_sd[k]
                consumed.add(k)
                continue

            out[k] = lora_sd[k]

        n_converted = len(consumed)
        if n_converted:
            logger.info("ZImage FFN key conversion: %d w1/w2/w3 keys converted to net.0.proj/net.2", n_converted)

        return out

    def _lora_key_map(self):
        block_prefixes = ["layers"]
        if not getattr(self, "skip_refiners", True):
            block_prefixes += ["noise_refiner", "context_refiner"]
        return {
            "block_prefixes": block_prefixes,
            "quantized_targets": {
                "attention.to_qkv": {
                    "lora_keys": ["attention.to_q", "attention.to_k", "attention.to_v"],
                    "type": "fused_qkv",
                    "param_prefix": "attention.fused_module.qkv_",
                },
                "attention.to_out.0": {
                    "lora_keys": ["attention.to_out.0"],
                    "type": "linear",
                },
                "feed_forward.net.0.proj": {
                    "lora_keys": ["feed_forward.net.0.proj"],
                    "type": "linear",
                },
                "feed_forward.net.2": {
                    "lora_keys": ["feed_forward.net.2"],
                    "type": "linear",
                },
            },
            "unquantized_targets": {
                "adaLN_modulation.0": {
                    "lora_keys": ["adaLN_modulation.0"],
                    "type": "linear",
                },
            },
        }

    def _patch_model(
        self,
        skip_refiners: bool = False,
        fp_overrides: set[str] | None = None,
        kept_fp_bias_modules: set[str] | None = None,
        **kwargs,
    ):
        """
        Patch the model by replacing attention and feed_forward modules in the orginal ZImageTransformerBlock.

        Parameters
        ----------
        skip_refiners: bool
            Default to `False`
            if `True`, transformer blocks of `noise_refiner` and `context_refiner` will NOT be replaced.
        **kwargs
            Additional arguments for quantization.

        Returns
        -------
        self : NunchakuZImageTransformer2DModel
            The patched model.
        """

        fp_overrides = fp_overrides or set()
        kept_fp_bias_modules = kept_fp_bias_modules or set()

        def _patch_transformer_block(block_list: List[ZImageTransformerBlock], block_prefix: str):
            for idx, block in enumerate(block_list):
                module_prefix = f"{block_prefix}.{idx}"
                block.attention = NunchakuZImageAttention(block.attention, **kwargs)
                block.attention.register_load_state_dict_post_hook(replace_fused_module)
                block.feed_forward = NunchakuZImageFeedForward(
                    block.feed_forward,
                    module_prefix=module_prefix,
                    fp_overrides=fp_overrides,
                    kept_fp_bias_modules=kept_fp_bias_modules,
                    **kwargs,
                )

        def _convert_feed_forward(block_list: List[ZImageTransformerBlock], block_prefix: str):
            for idx, block in enumerate(block_list):
                module_prefix = f"{block_prefix}.{idx}"
                block.feed_forward = _convert_z_image_ff(block.feed_forward)
                up_proj_name, down_proj_name = _prepare_zimage_ff_keep_fp_linears(
                    block.feed_forward,
                    module_prefix=module_prefix,
                    kept_fp_bias_modules=kept_fp_bias_modules,
                )
                if up_proj_name in fp_overrides:
                    _log_zimage_ff_kept_fp(up_proj_name, block.feed_forward.net[0].proj.bias is not None)
                if down_proj_name in fp_overrides:
                    _log_zimage_ff_kept_fp(down_proj_name, block.feed_forward.net[2].bias is not None)

        self.skip_refiners = skip_refiners
        _patch_transformer_block(self.layers, "layers")
        if skip_refiners:
            _convert_feed_forward(self.noise_refiner, "noise_refiner")
            _convert_feed_forward(self.context_refiner, "context_refiner")
        else:
            _patch_transformer_block(self.noise_refiner, "noise_refiner")
            _patch_transformer_block(self.context_refiner, "context_refiner")
        return self

    def _apply_runtime_guards(self):
        if getattr(self, "_zimage_has_fp_overrides", False):
            return
        if len(self.layers) <= ZIMAGE_TURBO_FFN_GUARD_LAYER_INDEX:
            return

        block = self.layers[ZIMAGE_TURBO_FFN_GUARD_LAYER_INDEX]
        if not hasattr(block.feed_forward, "net") or len(block.feed_forward.net) <= ZIMAGE_TURBO_FFN_GUARD_NET_INDEX:
            return

        target = block.feed_forward.net[ZIMAGE_TURBO_FFN_GUARD_NET_INDEX]
        if isinstance(target, ZImageTurboFFNDownGuard) or not isinstance(target, SVDQW4A4Linear):
            return

        # Keep the fix local to the released z-image-turbo checkpoint layout.
        logger.info("z-image-turbo legacy FFN guard enabled: layers.0.feed_forward.net.2")
        block.feed_forward.net[ZIMAGE_TURBO_FFN_GUARD_NET_INDEX] = ZImageTurboFFNDownGuard(
            target,
            channel_idx=ZIMAGE_TURBO_FFN_GUARD_CHANNEL_IDX,
            max_abs=ZIMAGE_TURBO_FFN_GUARD_MAX_ABS,
            mode=ZIMAGE_TURBO_FFN_GUARD_MODE,
        )

    def register_rope_hook(self, rope_hook: NunchakuZImageRopeHook):
        self.rope_hook_handles = []
        block_lists = [self.layers]
        if not self.skip_refiners:
            block_lists += [self.noise_refiner, self.context_refiner]
        for block_list in block_lists:
            for block in block_list:
                self.rope_hook_handles.append(
                    block.attention.register_forward_pre_hook(rope_hook, with_kwargs=True)
                )

    def unregister_rope_hook(self):
        for h in self.rope_hook_handles:
            h.remove()
        self.rope_hook_handles.clear()

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
        controlnet_block_samples=None,
        siglip_feats=None,
        image_noise_mask=None,
    ):
        """
        Adapted from diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel#forward

        Register pre-forward hooks for caching and substitution of packed `freqs_cis` tensor for all attention submodules and unregister after forwarding is done.
        """
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            forward_kwargs = {
                "x": x,
                "t": t,
                "cap_feats": cap_feats,
                "return_dict": return_dict,
                "controlnet_block_samples": controlnet_block_samples,
                "siglip_feats": siglip_feats,
                "image_noise_mask": image_noise_mask,
                "patch_size": patch_size,
                "f_patch_size": f_patch_size,
            }
            supported_params = inspect.signature(super().forward).parameters
            forward_kwargs = {key: value for key, value in forward_kwargs.items() if key in supported_params}
            return super().forward(**forward_kwargs)
        finally:
            self.unregister_rope_hook()
            del rope_hook

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuZImageTransformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuZImageTransformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.pop("offload", False)
        pin_memory = kwargs.pop("pin_memory", "auto")

        if offload:
            raise NotImplementedError("Offload is not supported for ZImageTransformer2DModel")

        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))

        rank = quantization_config.get("rank", 32)
        skip_refiners = quantization_config.get("skip_refiners", False)
        fp_overrides = _normalize_fp_overrides(quantization_config)
        kept_fp_bias_modules = _validate_ffn_fp_overrides(fp_overrides, model_state_dict) if fp_overrides else set()
        transformer = transformer.to(torch_dtype)
        transformer._zimage_has_fp_overrides = bool(fp_overrides)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"

        logger.info("quantization_config: %s, rank=%s, skip_refiners=%s", quantization_config, rank, skip_refiners)

        transformer._patch_model(
            skip_refiners=skip_refiners,
            fp_overrides=fp_overrides,
            kept_fp_bias_modules=kept_fp_bias_modules,
            precision=precision,
            rank=rank,
            **kwargs,
        )
        transformer = transformer.to_empty(device=device)

        patch_scale_key(transformer, model_state_dict)
        if torch_dtype == torch.float16:
            convert_fp16(transformer, model_state_dict)

        if resolve_pin_memory(pin_memory, device):
            model_state_dict = pin_state_dict(model_state_dict)

        transformer.load_state_dict(model_state_dict)
        transformer._apply_runtime_guards()

        if SVDQLoRAMixin is not None and hasattr(transformer, "_init_lora_state"):
            transformer._init_lora_state()
        else:
            logger.warning("SVDQLoRAMixin not available, LoRA support is disabled.")

        return transformer
