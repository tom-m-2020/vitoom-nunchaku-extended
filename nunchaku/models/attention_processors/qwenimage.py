"""
Attention processors for :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenAttention`.
"""

from typing import Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from ...ops.fused import fused_qkv_norm_rottary
from ..linear import SVDQW4A4Linear

_JOINT_QKV_WORKSPACES: dict[tuple, dict[str, torch.Tensor | tuple]] = {}


def _fused_qkv_heads(hidden_states, proj, norm_q, norm_k, rotary_emb, heads: int):
    qkv = fused_qkv_norm_rottary(hidden_states, proj, norm_q, norm_k, rotary_emb)
    query, key, value = qkv.chunk(3, dim=-1)
    return tuple(x.unflatten(-1, (heads, -1)) for x in (query, key, value))


def _get_joint_qkv_workspace(attn, *, batch_size: int, num_tokens: int, dtype: torch.dtype, device: torch.device):
    stream_key = None
    if device.type == "cuda":
        stream_key = int(torch.cuda.current_stream(device=device).cuda_stream)
    key = (str(device), stream_key, batch_size, num_tokens, int(attn.heads), int(attn.head_dim), dtype)
    workspace = _JOINT_QKV_WORKSPACES.get(key)
    if workspace is None:
        workspace = {
            "key": key,
            "query": torch.empty((batch_size, num_tokens, attn.heads, attn.head_dim), dtype=dtype, device=device),
            "key_tensor": torch.empty((batch_size, num_tokens, attn.heads, attn.head_dim), dtype=dtype, device=device),
            "value": torch.empty((batch_size, num_tokens, attn.heads, attn.head_dim), dtype=dtype, device=device),
        }
        _JOINT_QKV_WORKSPACES[key] = workspace
    return workspace


def _build_joint_qkv(
    attn,
    txt_query: torch.Tensor,
    txt_key: torch.Tensor,
    txt_value: torch.Tensor,
    img_query: torch.Tensor,
    img_key: torch.Tensor,
    img_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_grad_enabled():
        return (
            torch.cat([txt_query, img_query], dim=1),
            torch.cat([txt_key, img_key], dim=1),
            torch.cat([txt_value, img_value], dim=1),
        )

    batch_size = int(img_query.shape[0])
    seq_txt = int(txt_query.shape[1])
    seq_img = int(img_query.shape[1])
    num_tokens = seq_txt + seq_img
    workspace = _get_joint_qkv_workspace(
        attn, batch_size=batch_size, num_tokens=num_tokens, dtype=img_query.dtype, device=img_query.device
    )

    joint_query = workspace["query"]
    joint_key = workspace["key_tensor"]
    joint_value = workspace["value"]

    joint_query[:, :seq_txt].copy_(txt_query)
    joint_query[:, seq_txt:].copy_(img_query)
    joint_key[:, :seq_txt].copy_(txt_key)
    joint_key[:, seq_txt:].copy_(img_key)
    joint_value[:, :seq_txt].copy_(txt_value)
    joint_value[:, seq_txt:].copy_(img_value)
    return joint_query, joint_key, joint_value


class NunchakuQwenImageNaiveFA2Processor:
    """
    Naive attention processor for Qwen-Image joint text-image attention.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint text-image attention.

        Parameters
        ----------
        attn : :class:`~nunchaku.models.transformers.transformer_qwenimage.NunchakuQwenAttention`
            Attention module.
        hidden_states : torch.FloatTensor, shape (B, L, H*D)
            Image stream hidden states.
        encoder_hidden_states : torch.FloatTensor, shape (B, L_txt, H*D)
            Text stream hidden states.
        encoder_hidden_states_mask : torch.FloatTensor, optional
            Not used.
        attention_mask : Optional[torch.FloatTensor], shape (B, 1, L_total, L_total), optional
            Attention mask for joint attention.
        image_rotary_emb : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Tuple of rotary embeddings for image and text streams.

        Returns
        -------
        img_attn_output : torch.Tensor, shape (B, L, H*D)
            Output for image stream after attention and projection.
        txt_attn_output : torch.Tensor, shape (B, L_txt, H*D)
            Output for text stream after attention and projection.

        Raises
        ------
        ValueError
            If ``encoder_hidden_states`` (text stream) is not provided.

        Notes
        -----
        - B: batch size
        - L: sequence length (image)
        - L_txt: sequence length (text)
        - H: number of attention heads
        - D: head dimension
        """
        if encoder_hidden_states is None:
            raise ValueError("NunchakuQwenImageFA2Processor requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]
        use_fused_rotary = (
            isinstance(attn.to_qkv, SVDQW4A4Linear)
            and isinstance(attn.add_qkv_proj, SVDQW4A4Linear)
            and (
                image_rotary_emb is None
                or (
                    isinstance(image_rotary_emb, tuple)
                    and len(image_rotary_emb) == 2
                    and all(
                        torch.is_tensor(freqs) and freqs.ndim == 3 and not torch.is_complex(freqs)
                        for freqs in image_rotary_emb
                    )
                )
            )
        )

        if use_fused_rotary:
            rotary_img = image_rotary_emb[0] if image_rotary_emb is not None else None
            rotary_txt = image_rotary_emb[1] if image_rotary_emb is not None else None
            img_query, img_key, img_value = _fused_qkv_heads(
                hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, rotary_img, attn.heads
            )
            txt_query, txt_key, txt_value = _fused_qkv_heads(
                encoder_hidden_states,
                attn.add_qkv_proj,
                attn.norm_added_q,
                attn.norm_added_k,
                rotary_txt,
                attn.heads,
            )
        else:
            # Fallback to the original complex-RoPE path for older callers that still pass unpacked Qwen freqs.
            img_qkv = attn.to_qkv(hidden_states)
            img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

            txt_qkv = attn.add_qkv_proj(encoder_hidden_states)
            txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

            img_query = img_query.unflatten(-1, (attn.heads, -1))  # [B, L, H, D]
            img_key = img_key.unflatten(-1, (attn.heads, -1))
            img_value = img_value.unflatten(-1, (attn.heads, -1))

            txt_query = txt_query.unflatten(-1, (attn.heads, -1))
            txt_key = txt_key.unflatten(-1, (attn.heads, -1))
            txt_value = txt_value.unflatten(-1, (attn.heads, -1))

            assert attn.norm_q is not None
            img_query = attn.norm_q(img_query)
            assert attn.norm_k is not None
            img_key = attn.norm_k(img_key)
            assert attn.norm_added_q is not None
            txt_query = attn.norm_added_q(txt_query)
            assert attn.norm_added_k is not None
            txt_key = attn.norm_added_k(txt_key)

            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_use_real = not torch.is_complex(img_freqs)
                txt_use_real = not torch.is_complex(txt_freqs)
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=img_use_real)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=img_use_real)
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=txt_use_real)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=txt_use_real)

        # Build a reusable joint QKV buffer in inference to avoid per-call cat allocations.
        joint_query, joint_key, joint_value = _build_joint_qkv(
            attn, txt_query, txt_key, txt_value, img_query, img_key, img_value
        )

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
