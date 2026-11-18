"""Verified upstream wheel profiles and source transformations."""

from __future__ import annotations

import hashlib
from pathlib import Path


TRANSFORMER_PATH = Path("nunchaku/models/transformers/transformer_flux2.py")
CALLBACKS_PATH = Path("nunchaku/models/transformers/flux2_attention_callbacks.py")
PATCHED_TRANSFORMER_1_3_PATH = Path("patches/flux2/v1_3/transformer_flux2.py")
CALLBACK_SOURCE_PATH = Path("patches/flux2/common/flux2_attention_callbacks.py")

SOURCE_1_3_SHA256 = "A7B186D6EEB58F61AA926578DC87469A60FB523777F1D5737529DD721696559E"
SOURCE_1_4_SHA256 = "45C1B01D513A5E2983A2893CE598723CA32B51D114268EC53397DD885AD9BC81"

WHEELS = {
    "4A04F47405ABCA7968D919847DF853F08FE0FBCFC4F19C21B29184A9D68A1233": {
        "profile": "flux2-1.3",
        "version": "1.3.0.dev20260624+cu13.0torch2.12",
        "extension": "02A5706C296284FDF0C9237F611A073571656E1AB3D0785556CAF01B01AC6F5A",
    },
    "F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025": {
        "profile": "flux2-1.3",
        "version": "1.3.0.dev20260629+cu13.0torch2.11",
        "extension": "7B1ACA262EC9AA65E1226E3768109FB20866E8E94F5A4BF0D2D832DCCAAD29A9",
    },
    "C5A47BA1622DEF6C92F412365F8CBC9C959838B705F6AFC33439E5C60CB1DFB6": {
        "profile": "flux2-1.4",
        "version": "1.4.0+cu13.0torch2.11",
        "extension": "B1FC8BA716BF54321CA225EF312C3DF4014C5C7FA6F1C4EFD2D126114EEC3A0E",
    },
    "BE645687F99A4C8CC126080331810EDD244EB7D36B8F85042F4846132CAD2CE3": {
        "profile": "flux2-1.4",
        "version": "1.4.1+cu13.0torch2.14",
        "extension": "32A2F35EBB9D2BF8498AD51F820808C6538290C9A386B1DEA03AA9F8CB0F81B8",
    },
    "4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F": {
        "profile": "flux2-1.4",
        "version": "1.4.1+cu13.0torch2.14",
        "extension": "997178E9442170A4F68F2D6ED1736AE735CFE8A371885038558C377A368D7927",
    },
}


def normalized_sha256(data: bytes) -> str:
    normalized = data.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest().upper()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"FLUX.2 source does not contain exactly one {label} insertion point.")
    return source.replace(old, new, 1)


def patch_1_4(source: bytes) -> bytes:
    """Add API v3 while preserving the 1.4 reference and FP16 paths."""
    if normalized_sha256(source) != SOURCE_1_4_SHA256:
        raise ValueError("FLUX.2 1.4 source hash is not verified.")
    text = source.decode("utf-8").replace("\r\n", "\n")
    imports = """from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key
from .flux2_attention_callbacks import (
    FLUX2_ATTENTION_CALLBACK_API_VERSION,
    Flux2AttentionInvocation,
    GENERATED_SPATIAL_SHAPE_KEY,
    GENERATED_TOKEN_COUNT_KEY,
    REFERENCE_SPATIAL_SHAPES_KEY,
    REFERENCE_TOKEN_COUNTS_KEY,
    get_attention_callbacks,
    run_post_attention_callbacks,
    run_pre_attention_callbacks,
)
"""
    text = _replace_once(
        text,
        "from .utils import NunchakuModelLoaderMixin, convert_fp16, patch_scale_key\n",
        imports,
        "callback import",
    )
    text = _replace_once(
        text,
        """    def __init__(self, other: Flux2Attention, **kwargs):
        super(Flux2Attention, self).__init__()
        _copy_attn_attrs(self, other)
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias
        self.fused_projections = True
""",
        """    def __init__(self, other: Flux2Attention, *, block_index: int, **kwargs):
        super(Flux2Attention, self).__init__()
        _copy_attn_attrs(self, other)
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias
        self.fused_projections = True
        self.block_type = "double"
        self.block_index = int(block_index)
""",
        "double-block identity",
    )
    text = _replace_once(
        text,
        """    def __init__(self, other: Flux2ParallelSelfAttention, **kwargs):
        super(Flux2ParallelSelfAttention, self).__init__()
        _copy_attn_attrs(self, other)
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor
""",
        """    def __init__(self, other: Flux2ParallelSelfAttention, *, block_index: int, **kwargs):
        super(Flux2ParallelSelfAttention, self).__init__()
        _copy_attn_attrs(self, other)
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor
        self.block_type = "single"
        self.block_index = int(block_index)
""",
        "single-block identity",
    )
    text = _replace_once(
        text,
        "class NunchakuFlux2TransformerBlock(Flux2TransformerBlock):\n    def __init__(self, block: Flux2TransformerBlock, **kwargs):",
        "class NunchakuFlux2TransformerBlock(Flux2TransformerBlock):\n    def __init__(self, block: Flux2TransformerBlock, *, block_index: int, **kwargs):",
        "double-block constructor",
    )
    text = _replace_once(
        text,
        "self.attn = NunchakuFlux2Attention(block.attn, **kwargs)",
        "self.attn = NunchakuFlux2Attention(block.attn, block_index=block_index, **kwargs)",
        "double-attention constructor",
    )
    text = _replace_once(
        text,
        "class NunchakuFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):\n    def __init__(self, block: Flux2SingleTransformerBlock, **kwargs):",
        "class NunchakuFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):\n    def __init__(self, block: Flux2SingleTransformerBlock, *, block_index: int, **kwargs):",
        "single-block constructor",
    )
    text = _replace_once(
        text,
        "self.attn = NunchakuFlux2ParallelSelfAttention(block.attn, **kwargs)",
        "self.attn = NunchakuFlux2ParallelSelfAttention(block.attn, block_index=block_index, **kwargs)",
        "single-attention constructor",
    )
    text = _replace_once(
        text,
        """        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
""",
        """        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            num_txt_tokens=0 if text_seq_len is None else text_seq_len,
            **joint_attention_kwargs,
        )
""",
        "generic single-stream text count",
    )
    text = _replace_once(
        text,
        "self.transformer_blocks[i] = NunchakuFlux2TransformerBlock(block, **kwargs)",
        "self.transformer_blocks[i] = NunchakuFlux2TransformerBlock(block, block_index=i, **kwargs)",
        "double-block patch",
    )
    text = _replace_once(
        text,
        "self.single_transformer_blocks[i] = NunchakuFlux2SingleTransformerBlock(block, **kwargs)",
        "self.single_transformer_blocks[i] = NunchakuFlux2SingleTransformerBlock(block, block_index=i, **kwargs)",
        "single-block patch",
    )
    text = _replace_once(
        text,
        """        single_args = (
            hidden_states, None,
            single_stream_mod, rotary_emb_single, joint_attention_kwargs,
        )
""",
        """        single_args = (
            hidden_states, None,
            single_stream_mod, rotary_emb_single, joint_attention_kwargs,
            False, num_txt_tokens,
        )
""",
        "single-stream text count",
    )
    text = _add_attention_callbacks(text, parallel=False)
    text = _add_attention_callbacks(text, parallel=True)
    return text.encode("utf-8")


def _metadata(indent: str, *, parallel: bool, padded: bool) -> str:
    text_count = "num_txt_tokens" if parallel else "num_txt_tokens"
    image_count = (
        "generated_token_count + sum(reference_token_counts)"
        if parallel
        else "num_img_tokens"
    )
    padded_text = "num_txt_tokens" if parallel or not padded else "num_txt_tokens_pad"
    padded_image = (
        "num_tokens_pad - num_txt_tokens"
        if parallel and padded
        else "num_img_tokens_pad"
        if padded
        else image_count
    )
    packed = "num_tokens_pad" if padded else f"{text_count} + {image_count}"
    return f"""{indent}pre_callbacks, post_callbacks = get_attention_callbacks(kwargs)
{indent}metadata = None
{indent}if pre_callbacks or post_callbacks:
{indent}    if kv_cache_mode is not None:
{indent}        raise RuntimeError("FLUX.2 attention callbacks do not support KV-cache execution.")
{indent}    reference_token_counts = kwargs.get(REFERENCE_TOKEN_COUNTS_KEY)
{indent}    reference_spatial_shapes = kwargs.get(REFERENCE_SPATIAL_SHAPES_KEY)
{indent}    generated_token_count = kwargs.get(GENERATED_TOKEN_COUNT_KEY)
{indent}    generated_spatial_shape = kwargs.get(GENERATED_SPATIAL_SHAPE_KEY)
{indent}    if not isinstance(reference_token_counts, tuple):
{indent}        raise TypeError("reference_token_counts must be an explicit tuple when callbacks are active.")
{indent}    if not isinstance(reference_spatial_shapes, tuple):
{indent}        raise TypeError("reference_spatial_shapes must be an explicit tuple when callbacks are active.")
{indent}    if isinstance(generated_token_count, bool) or not isinstance(generated_token_count, int):
{indent}        raise TypeError("generated_token_count must be an explicit integer when callbacks are active.")
{indent}    if not isinstance(generated_spatial_shape, tuple):
{indent}        raise TypeError("generated_spatial_shape must be an explicit tuple when callbacks are active.")
{indent}    metadata = Flux2AttentionInvocation(
{indent}        block_type=self.block_type,
{indent}        block_index=self.block_index,
{indent}        text_token_count={text_count},
{indent}        generated_token_count=generated_token_count,
{indent}        generated_spatial_shape=generated_spatial_shape,
{indent}        reference_token_counts=reference_token_counts,
{indent}        reference_spatial_shapes=reference_spatial_shapes,
{indent}        logical_image_token_count={image_count},
{indent}        padded_text_token_count={padded_text},
{indent}        padded_image_token_count={padded_image},
{indent}        packed_sequence_length={packed},
{indent}        batch_size=batch_size,
{indent}        head_count=self.heads,
{indent}        head_dimension=self.head_dim,
{indent}    )
"""


def _add_attention_callbacks(text: str, *, parallel: bool) -> str:
    if parallel:
        packed_start = """        if use_packed_fp16:
            batch_size = hidden_states.shape[0]
            num_tokens = hidden_states.shape[1]
            query, key, value, num_tokens_pad = _alloc_packed_qkv(
                batch_size, self.heads, num_tokens, self.head_dim, hidden_states.device
            )
"""
        packed_new = packed_start + _metadata("            ", parallel=True, padded=True)
        packed_pre = """            attention_fp16(query, key, value, attn_output, self.head_dim ** (-0.5))
            attn_output = attn_output[:, :num_tokens]
"""
        packed_pre_new = """            if pre_callbacks:
                query, key, value = run_pre_attention_callbacks(pre_callbacks, query, key, value, metadata)
            attention_fp16(query, key, value, attn_output, self.head_dim ** (-0.5))
            if post_callbacks:
                attn_output = run_post_attention_callbacks(post_callbacks, attn_output, metadata)
            attn_output = attn_output[:, :num_tokens]
"""
        generic_anchor = """        attn_output = _dispatch_with_kv_cache(
            query, key, value, num_txt_tokens, num_ref_tokens,
"""
        generic_prefix = "        batch_size = hidden_states.shape[0]\n" + _metadata("        ", parallel=True, padded=False) + """        if pre_callbacks:
            callback_query, callback_key, callback_value = run_pre_attention_callbacks(
                pre_callbacks, query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), metadata
            )
            query = callback_query.transpose(1, 2)
            key = callback_key.transpose(1, 2)
            value = callback_value.transpose(1, 2)
""" + generic_anchor
        flatten = "        attn_output = attn_output.flatten(2, 3).to(query.dtype)\n"
        flatten_new = flatten + """        if post_callbacks:
            attn_output = run_post_attention_callbacks(post_callbacks, attn_output, metadata)
"""
    else:
        packed_start = """        if use_packed_fp16:
            batch_size = hidden_states.shape[0]
            num_txt_tokens = encoder_hidden_states.shape[1]
            num_img_tokens = hidden_states.shape[1]
            num_txt_tokens_pad = _pad256(num_txt_tokens)
            num_img_tokens_pad = _pad256(num_img_tokens)
            num_tokens_pad = num_txt_tokens_pad + num_img_tokens_pad
"""
        packed_new = packed_start + _metadata("            ", parallel=False, padded=True)
        packed_pre = """            attention_fp16(query, key, value, attention_output, self.head_dim ** (-0.5))
            encoder_hidden_states = attention_output[:, :num_txt_tokens]
"""
        packed_pre_new = """            if pre_callbacks:
                query, key, value = run_pre_attention_callbacks(pre_callbacks, query, key, value, metadata)
            attention_fp16(query, key, value, attention_output, self.head_dim ** (-0.5))
            if post_callbacks:
                attention_output = run_post_attention_callbacks(post_callbacks, attention_output, metadata)
            encoder_hidden_states = attention_output[:, :num_txt_tokens]
"""
        generic_anchor = """        hidden_states = _dispatch_with_kv_cache(
            query, key, value, encoder_seq_len, num_ref_tokens,
"""
        generic_prefix = """        batch_size = hidden_states.shape[0]
        num_txt_tokens = encoder_seq_len
        num_img_tokens = hidden_states.shape[1]
""" + _metadata("        ", parallel=False, padded=False) + """        if pre_callbacks:
            callback_query, callback_key, callback_value = run_pre_attention_callbacks(
                pre_callbacks, query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), metadata
            )
            query = callback_query.transpose(1, 2)
            key = callback_key.transpose(1, 2)
            value = callback_value.transpose(1, 2)
""" + generic_anchor
        flatten = "        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)\n"
        flatten_new = flatten + """        if post_callbacks:
            hidden_states = run_post_attention_callbacks(post_callbacks, hidden_states, metadata)
"""
    text = _replace_once(text, packed_start, packed_new, "packed callback metadata")
    text = _replace_once(text, packed_pre, packed_pre_new, "packed callback invocation")
    text = _replace_once(text, generic_anchor, generic_prefix, "generic callback metadata")
    text = _replace_once(text, flatten, flatten_new, "generic post callback")
    return text


__all__ = [
    "CALLBACKS_PATH",
    "CALLBACK_SOURCE_PATH",
    "PATCHED_TRANSFORMER_1_3_PATH",
    "SOURCE_1_3_SHA256",
    "SOURCE_1_4_SHA256",
    "TRANSFORMER_PATH",
    "WHEELS",
    "normalized_sha256",
    "patch_1_4",
]
