from __future__ import annotations

from dataclasses import dataclass

import torch

from .._C import ops


def pad_to(x: int, multiple: int) -> int:
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


@dataclass
class W8A16WeightPackResult:
    qweight: torch.Tensor
    wscales: torch.Tensor
    n_pad: int
    k_pad: int


def quantize_w8a16_wgt_cuda(
    weight_f16: torch.Tensor,
    *,
    qweight: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    pad_n: int = 128,
    pad_k: int = 32,
) -> W8A16WeightPackResult:
    """
    Quantize FP16/BF16 weights to INT8 row-major buffers for the W8A16 kernel.
    """
    assert weight_f16.is_cuda, "W8A16 weight quantization requires CUDA tensors"
    assert weight_f16.dtype in (torch.float16, torch.bfloat16), f"expected fp16/bf16 weight, got {weight_f16.dtype}"
    assert weight_f16.ndim == 2, f"expected [N,K], got {tuple(weight_f16.shape)}"

    n, k = (int(weight_f16.shape[0]), int(weight_f16.shape[1]))
    n_pad = pad_to(n, pad_n)
    k_pad = pad_to(k, pad_k)

    if n_pad != n or k_pad != k:
        padded = torch.zeros((n_pad, k_pad), dtype=weight_f16.dtype, device=weight_f16.device)
        padded[:n, :k].copy_(weight_f16)
        weight_f16 = padded

    if qweight is None:
        qweight = torch.empty((n_pad, k_pad), dtype=torch.int8, device=weight_f16.device)
    if wscales is None:
        wscales = torch.empty((n_pad,), dtype=weight_f16.dtype, device=weight_f16.device)

    assert qweight.shape == (n_pad, k_pad) and qweight.dtype == torch.int8 and qweight.is_cuda
    assert wscales.shape == (n_pad,) and wscales.dtype == weight_f16.dtype and wscales.is_cuda

    ops.quantize_w8a16_wgt(weight_f16, qweight, wscales)
    return W8A16WeightPackResult(qweight=qweight, wscales=wscales, n_pad=n_pad, k_pad=k_pad)


def gemm_w8a16_cuda(
    *,
    input_f16: torch.Tensor,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weight-only W8A16 GEMM for padded row-major INT8 weights.
    """
    assert input_f16.is_cuda and input_f16.dtype in (torch.float16, torch.bfloat16)
    assert qweight.is_cuda and qweight.dtype == torch.int8 and qweight.ndim == 2
    assert wscales.is_cuda and wscales.dtype == input_f16.dtype and wscales.ndim == 1

    m = int(input_f16.numel() // input_f16.shape[-1])
    n = int(qweight.shape[0])
    if out is None:
        out_shape = list(input_f16.shape)
        out_shape[-1] = n
        out = torch.empty(out_shape, dtype=input_f16.dtype, device=input_f16.device)

    if bias is not None:
        assert bias.is_cuda and bias.dtype == input_f16.dtype and bias.ndim == 1

    ops.gemm_w8a16(input_f16, qweight, out, wscales, bias)
    return out
