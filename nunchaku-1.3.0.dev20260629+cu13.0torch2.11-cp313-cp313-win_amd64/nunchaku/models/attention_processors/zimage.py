"""
Attention processor implementations for :class:`~nunchaku.models.transformers.transformer_zimage.NunchakuZImageAttention`.
"""

from typing import Optional

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor


class NunchakuZSingleStreamAttnProcessor(ZSingleStreamAttnProcessor):
    """
    Nunchaku attention processor for Z-Image-Turbo.
    Adapted from diffusers.models.transformers.transformer_z_image.ZSingleStreamAttnProcessor.

    """

    def __init__(self):
        super().__init__()

    # Adapted from diffusers.models.transformers.transformer_z_image.ZSingleStreamAttnProcessor#__call__
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module. Adapted from diffusers.models.transformers.transformer_z_image.ZSingleStreamAttnProcessor#__call__.
        """
        qkv = attn.fused_module(hidden_states, freqs_cis)

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output
