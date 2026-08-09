from __future__ import annotations

import torch


def pad_to(x: int, multiple: int) -> int:
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def infer_plain_weight_orientation(
    qweight: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    out_features_pad: int,
    in_features_pad: int,
) -> torch.Tensor:
    """
    Normalize a plain INT8 checkpoint weight to row-major ``[out, in]`` layout.

    Exporters are expected to save ``qweight`` in row-major ``[out, in]`` form,
    optionally padded to the kernel-aligned sizes.
    """
    if qweight.ndim != 2:
        raise ValueError(f"plain qweight must be 2D, got {qweight.ndim}D")

    shape = tuple(int(x) for x in qweight.shape)

    if shape == (out_features_pad, in_features_pad):
        return qweight
    if shape == (out_features, in_features):
        return qweight
    if shape[0] in {out_features, out_features_pad} and shape[1] in {in_features, in_features_pad}:
        return qweight

    if shape[0] in {in_features, in_features_pad} and shape[1] in {out_features, out_features_pad}:
        raise ValueError(
            "plain qweight appears transposed ([in,out]) which is forbidden by contract. "
            f"got={shape}, expected [out,in]=({out_features},{in_features}) "
            f"or padded=({out_features_pad},{in_features_pad})."
        )

    raise ValueError(
        "plain qweight shape is not compatible with this Linear. "
        f"got={shape}, expected one of [out,in]=({out_features},{in_features}) "
        f"or padded=({out_features_pad},{in_features_pad}) with optional padding."
    )


def pad_plain_qweight(
    qweight_oi: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    out_features_pad: int,
    in_features_pad: int,
    device: torch.device,
) -> torch.Tensor:
    if qweight_oi.dtype != torch.int8:
        raise ValueError(f"plain qweight must be int8, got {qweight_oi.dtype}")

    padded = torch.zeros((out_features_pad, in_features_pad), dtype=torch.int8, device=device)
    rows = min(int(qweight_oi.shape[0]), out_features)
    cols = min(int(qweight_oi.shape[1]), in_features)
    padded[:rows, :cols].copy_(qweight_oi[:rows, :cols].to(device=device, dtype=torch.int8, non_blocking=True))
    return padded


def pad_output_vector(
    vec: torch.Tensor,
    *,
    name: str,
    out_features: int,
    out_features_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {vec.ndim}D")
    if int(vec.numel()) not in {out_features, out_features_pad}:
        raise ValueError(
            f"{name} length mismatch: got {int(vec.numel())}, "
            f"expected {out_features} or padded {out_features_pad}"
        )

    padded = torch.zeros((out_features_pad,), dtype=dtype, device=device)
    length = min(int(vec.numel()), out_features)
    padded[:length].copy_(vec[:length].to(device=device, dtype=dtype, non_blocking=True))
    return padded


def pad_fp_weight(
    weight: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    out_features_pad: int,
    in_features_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected weight 2D [out,in], got {tuple(weight.shape)}")

    shape = tuple(int(x) for x in weight.shape)
    if shape[0] not in {out_features, out_features_pad} or shape[1] not in {in_features, in_features_pad}:
        raise ValueError(
            f"weight shape mismatch: got {shape}, expected ({out_features},{in_features}) "
            f"or padded ({out_features_pad},{in_features_pad})"
        )

    padded = torch.zeros((out_features_pad, in_features_pad), dtype=dtype, device=device)
    rows = min(int(weight.shape[0]), out_features)
    cols = min(int(weight.shape[1]), in_features)
    padded[:rows, :cols].copy_(weight[:rows, :cols].to(device=device, dtype=dtype, non_blocking=True))
    return padded
