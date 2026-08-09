from __future__ import annotations

import torch
from torch import nn

from ..ops.w8a16 import gemm_w8a16_cuda, quantize_w8a16_wgt_cuda
from ._w8_common import infer_plain_weight_orientation, pad_fp_weight, pad_output_vector, pad_plain_qweight, pad_to


class SVDQW8A16Linear(nn.Module):
    """
    Weight-only W8A16 linear layer for quality-sensitive modulation paths.

    Internal contract
    -----------------
    - `qweight`: row-major INT8 buffer with padded shape `[N_pad, K_pad]`
    - `wscales`: per-output-channel scale with padded shape `[N_pad]`
    - `bias`: plain padded output bias `[N_pad]`

    Checkpoint compatibility
    ------------------------
    - If a checkpoint provides `.weight` / `.bias`, loading on CUDA will quantize
      them into the internal W8A16 buffers on the fly.
    - If a checkpoint provides `.qweight` / `.wscales`, `weight_layout="plain"`
      accepts unpadded or padded row-major INT8 weights and pads them into the
      internal layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        pad_k: int = 32,
        pad_n: int = 128,
        weight_layout: str = "packed",
        **_unused_kwargs,
    ):
        super().__init__()
        if torch_dtype not in (torch.float16, torch.bfloat16):
            torch_dtype = torch.bfloat16
        if device is None:
            device = torch.device("cpu")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.torch_dtype = torch_dtype

        self.pad_k = int(pad_k)
        self.pad_n = int(pad_n)
        self.in_features_pad = pad_to(self.in_features, self.pad_k)
        self.out_features_pad = pad_to(self.out_features, self.pad_n)

        layout = str(weight_layout or "packed").strip().lower()
        self.weight_layout = layout if layout in {"packed", "plain"} else "packed"

        self.qweight = nn.Parameter(
            torch.empty((self.out_features_pad, self.in_features_pad), dtype=torch.int8, device=device), requires_grad=False
        )
        self.wscales = nn.Parameter(
            torch.empty((self.out_features_pad,), dtype=torch_dtype, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty((self.out_features_pad,), dtype=torch_dtype, device=device), requires_grad=False)
            if bias
            else None
        )
        self.smooth_factor = nn.Parameter(torch.ones((self.in_features,), dtype=torch_dtype, device=device), requires_grad=False)
        self.smooth_factor_orig = nn.Parameter(
            torch.ones((self.in_features,), dtype=torch_dtype, device=device), requires_grad=False
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "SVDQW8A16Linear":
        kwargs.pop("precision", None)
        kwargs.pop("rank", None)
        kwargs.pop("act_unsigned", None)
        kwargs.pop("group_size", None)

        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=linear.weight.device,
            **kwargs,
        )

    def _quantize_from_fp_weight_bias(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        if not self.qweight.is_cuda:
            raise RuntimeError(
                "SVDQW8A16Linear quantization from fp weights requires the module to be on CUDA. "
                "Fix: load the model with `device='cuda'` before calling `load_state_dict`."
            )

        device = self.qweight.device
        weight_padded = pad_fp_weight(
            weight,
            out_features=self.out_features,
            in_features=self.in_features,
            out_features_pad=self.out_features_pad,
            in_features_pad=self.in_features_pad,
            dtype=self.torch_dtype,
            device=device,
        )
        quantize_w8a16_wgt_cuda(
            weight_padded, qweight=self.qweight, wscales=self.wscales, pad_n=self.pad_n, pad_k=self.pad_k
        )

        if self.bias is not None:
            if bias is None:
                self.bias.zero_()
            else:
                self.bias.copy_(
                    pad_output_vector(
                        bias,
                        name="bias",
                        out_features=self.out_features,
                        out_features_pad=self.out_features_pad,
                        dtype=self.torch_dtype,
                        device=device,
                    )
                )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        qweight_key = prefix + "qweight"
        wscales_key = prefix + "wscales"
        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        smooth_key = prefix + "smooth_factor"
        smooth_orig_key = prefix + "smooth_factor_orig"
        # W8A16 is weight-only. Some SVDQuant exporters may still include the
        # W4 low-rank residual keys; they are not parameters of this module.
        for aux_key in (prefix + "proj_down", prefix + "proj_up", prefix + "lora_down", prefix + "lora_up"):
            state_dict.pop(aux_key, None)

        if qweight_key in state_dict and wscales_key in state_dict and self.weight_layout == "plain":
            try:
                q_ckpt = state_dict.pop(qweight_key)
                ws_ckpt = state_dict.pop(wscales_key)
                b_ckpt = state_dict.pop(bias_key, None) if (self.bias is not None and bias_key in state_dict) else None

                q_oi = infer_plain_weight_orientation(
                    q_ckpt,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    out_features_pad=self.out_features_pad,
                    in_features_pad=self.in_features_pad,
                )
                self.qweight.copy_(
                    pad_plain_qweight(
                        q_oi,
                        out_features=self.out_features,
                        in_features=self.in_features,
                        out_features_pad=self.out_features_pad,
                        in_features_pad=self.in_features_pad,
                        device=self.qweight.device,
                    )
                )
                self.wscales.copy_(
                    pad_output_vector(
                        ws_ckpt,
                        name="wscales",
                        out_features=self.out_features,
                        out_features_pad=self.out_features_pad,
                        dtype=self.torch_dtype,
                        device=self.wscales.device,
                    )
                )
                if self.bias is not None:
                    if b_ckpt is None:
                        self.bias.zero_()
                    else:
                        self.bias.copy_(
                            pad_output_vector(
                                b_ckpt,
                                name="bias",
                                out_features=self.out_features,
                                out_features_pad=self.out_features_pad,
                                dtype=self.torch_dtype,
                                device=self.bias.device,
                            )
                        )

                state_dict[qweight_key] = self.qweight
                state_dict[wscales_key] = self.wscales
                if self.bias is not None:
                    state_dict[bias_key] = self.bias
            except Exception as exc:
                error_msgs.append(f"{prefix}: failed to load plain W8A16 qweight/wscales: {exc}")

        if qweight_key not in state_dict and weight_key in state_dict:
            weight = state_dict.pop(weight_key)
            bias = state_dict.pop(bias_key, None) if (self.bias is not None and bias_key in state_dict) else None
            try:
                self._quantize_from_fp_weight_bias(weight, bias)
            except Exception as exc:
                error_msgs.append(f"{prefix}: failed to quantize W8A16 from fp weight/bias: {exc}")

            state_dict[qweight_key] = self.qweight
            state_dict[wscales_key] = self.wscales
            if self.bias is not None:
                state_dict[bias_key] = self.bias

        if smooth_key not in state_dict and smooth_orig_key in state_dict:
            state_dict[smooth_key] = state_dict[smooth_orig_key]
        elif smooth_orig_key not in state_dict and smooth_key in state_dict:
            state_dict[smooth_orig_key] = state_dict[smooth_key]
        else:
            state_dict.setdefault(smooth_key, self.smooth_factor)
            state_dict.setdefault(smooth_orig_key, self.smooth_factor_orig)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _get_smoothed_input(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(f"Input last dim mismatch: got {int(x.shape[-1])}, expected {self.in_features}")
        if x.dtype != self.torch_dtype:
            x = x.to(self.torch_dtype)
        smooth = self.smooth_factor
        if smooth.dtype != x.dtype or smooth.device != x.device:
            smooth = smooth.to(device=x.device, dtype=x.dtype)
        view_shape = [1] * x.ndim
        view_shape[-1] = self.in_features
        return x.div(smooth.view(view_shape))

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"expected fp16/bf16 input, got {x.dtype}")
        if x.device != self.qweight.device:
            raise ValueError(f"input device mismatch: got {x.device}, expected {self.qweight.device}")

        if x.ndim == 2:
            x2d = x
            batch_shape = None
        elif x.ndim == 3:
            b, s, c = x.shape
            x2d = x.reshape(b * s, c)
            batch_shape = (b, s)
        else:
            raise ValueError(f"Expected x rank 2 or 3, got {x.ndim}")

        x2d = self._get_smoothed_input(x2d)

        if self.in_features_pad != self.in_features:
            x_pad = torch.zeros((int(x2d.shape[0]), self.in_features_pad), dtype=self.torch_dtype, device=x2d.device)
            x_pad[:, : self.in_features].copy_(x2d)
        else:
            x_pad = x2d.contiguous()

        out_pad = torch.empty((int(x2d.shape[0]), self.out_features_pad), dtype=self.torch_dtype, device=x2d.device)
        gemm_w8a16_cuda(input_f16=x_pad, qweight=self.qweight, wscales=self.wscales, bias=self.bias, out=out_pad)

        out2d = out_pad[:, : self.out_features]
        if output is not None:
            output_2d = output if batch_shape is None else output.reshape(-1, self.out_features)
            if tuple(output_2d.shape) != (int(x2d.shape[0]), self.out_features):
                raise ValueError(
                    f"output shape mismatch: got {tuple(output_2d.shape)}, "
                    f"expected {(int(x2d.shape[0]), self.out_features)}"
                )
            output_2d.copy_(out2d)
            out2d = output_2d

        if batch_shape is not None:
            out2d = out2d.reshape(batch_shape[0], batch_shape[1], self.out_features)
        return out2d

    def __repr__(self) -> str:
        return (
            f"SVDQW8A16Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"in_features_pad={self.in_features_pad}, out_features_pad={self.out_features_pad})"
        )
