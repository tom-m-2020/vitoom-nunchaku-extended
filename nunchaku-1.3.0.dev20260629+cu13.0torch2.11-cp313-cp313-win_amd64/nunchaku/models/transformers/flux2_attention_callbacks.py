"""Generic per-forward attention callbacks for the FLUX.2 Python runtime.

Copyright 2026 Nunchaku contributors.
Licensed under the Apache License, Version 2.0.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


FLUX2_ATTENTION_CALLBACK_API_VERSION = 3
PRE_ATTENTION_CALLBACKS_KEY = "pre_attention_callbacks"
POST_ATTENTION_CALLBACKS_KEY = "post_attention_callbacks"
GENERATED_TOKEN_COUNT_KEY = "generated_token_count"
GENERATED_SPATIAL_SHAPE_KEY = "generated_spatial_shape"
REFERENCE_TOKEN_COUNTS_KEY = "reference_token_counts"
REFERENCE_SPATIAL_SHAPES_KEY = "reference_spatial_shapes"


@dataclass(frozen=True, slots=True)
class Flux2AttentionInvocation:
    block_type: str
    block_index: int
    text_token_count: int
    generated_token_count: int
    generated_spatial_shape: tuple[int, int]
    reference_token_counts: tuple[int, ...]
    reference_spatial_shapes: tuple[tuple[int, int], ...]
    logical_image_token_count: int
    padded_text_token_count: int
    padded_image_token_count: int
    packed_sequence_length: int
    batch_size: int
    head_count: int
    head_dimension: int

    def __post_init__(self) -> None:
        if self.block_type not in ("double", "single"):
            raise ValueError(f"block_type must be 'double' or 'single', got {self.block_type!r}.")
        integer_fields = (
            "block_index",
            "text_token_count",
            "generated_token_count",
            "logical_image_token_count",
            "padded_text_token_count",
            "padded_image_token_count",
            "packed_sequence_length",
            "batch_size",
            "head_count",
            "head_dimension",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
        if not isinstance(self.reference_token_counts, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.reference_token_counts
        ):
            raise ValueError("reference_token_counts must be a tuple of positive integers.")
        if (
            not isinstance(self.generated_spatial_shape, tuple)
            or len(self.generated_spatial_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.generated_spatial_shape
            )
        ):
            raise ValueError("generated_spatial_shape must be a pair of positive integers.")
        if self.generated_spatial_shape[0] * self.generated_spatial_shape[1] != self.generated_token_count:
            raise ValueError(
                "generated_spatial_shape product does not match generated_token_count: "
                f"{self.generated_spatial_shape[0]} * {self.generated_spatial_shape[1]} "
                f"!= {self.generated_token_count}."
            )
        if (
            not isinstance(self.reference_spatial_shapes, tuple)
            or len(self.reference_spatial_shapes) != len(self.reference_token_counts)
        ):
            raise ValueError(
                "reference_spatial_shapes must contain one (height, width) tuple "
                "for each reference."
            )
        for index, (shape, count) in enumerate(
            zip(self.reference_spatial_shapes, self.reference_token_counts, strict=True)
        ):
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            ):
                raise ValueError(
                    f"reference_spatial_shapes[{index}] must be a pair of positive integers."
                )
            if shape[0] * shape[1] != count:
                raise ValueError(
                    f"reference_spatial_shapes[{index}] product does not match "
                    f"reference_token_counts[{index}]: {shape[0]} * {shape[1]} != {count}."
                )
        expected_image = self.generated_token_count + sum(self.reference_token_counts)
        if self.logical_image_token_count != expected_image:
            raise ValueError(
                "logical_image_token_count does not match generated and reference counts: "
                f"{self.logical_image_token_count} != {expected_image}."
            )


def get_attention_callbacks(runtime_options: dict | None):
    """Return validated immutable callback tuples without allocating tensors."""
    if runtime_options is None:
        return (), ()
    if not isinstance(runtime_options, dict):
        raise TypeError("joint_attention_kwargs must be a dict or None.")

    pre = runtime_options.get(PRE_ATTENTION_CALLBACKS_KEY, ())
    post = runtime_options.get(POST_ATTENTION_CALLBACKS_KEY, ())
    for name, callbacks in ((PRE_ATTENTION_CALLBACKS_KEY, pre), (POST_ATTENTION_CALLBACKS_KEY, post)):
        if not isinstance(callbacks, tuple):
            raise TypeError(f"{name} must be an immutable tuple of callables.")
        if not all(isinstance(callback, Callable) for callback in callbacks):
            raise TypeError(f"{name} must contain only callables.")
    return pre, post


def _validate_replacement(replacement, original, *, name):
    if not torch.is_tensor(replacement):
        raise TypeError(f"{name} replacement must be a torch.Tensor.")
    if replacement.shape != original.shape:
        raise ValueError(
            f"{name} replacement shape must be {list(original.shape)}, got {list(replacement.shape)}."
        )
    if replacement.dtype != original.dtype:
        raise TypeError(
            f"{name} replacement dtype must be {original.dtype}, got {replacement.dtype}."
        )
    if replacement.device != original.device:
        raise ValueError(
            f"{name} replacement device must be {original.device}, got {replacement.device}."
        )
    if not replacement.is_contiguous():
        raise ValueError(f"{name} replacement must be contiguous.")
    return replacement


def run_pre_attention_callbacks(callbacks, query, key, value, metadata):
    """Run callbacks in order; ``None`` means optional in-place modification."""
    for index, callback in enumerate(callbacks):
        result = callback(query, key, value, metadata)
        if result is None:
            continue
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                f"pre-attention callback {index} must return None or a (query, key, value) tuple."
            )
        query = _validate_replacement(result[0], query, name="query")
        key = _validate_replacement(result[1], key, name="key")
        value = _validate_replacement(result[2], value, name="value")
    return query, key, value


def run_post_attention_callbacks(callbacks, attention_output, metadata):
    """Run callbacks in order; ``None`` means optional in-place modification."""
    for index, callback in enumerate(callbacks):
        result = callback(attention_output, metadata)
        if result is None:
            continue
        attention_output = _validate_replacement(
            result, attention_output, name=f"post-attention callback {index} output"
        )
    return attention_output


__all__ = [
    "FLUX2_ATTENTION_CALLBACK_API_VERSION",
    "Flux2AttentionInvocation",
    "GENERATED_TOKEN_COUNT_KEY",
    "GENERATED_SPATIAL_SHAPE_KEY",
    "POST_ATTENTION_CALLBACKS_KEY",
    "PRE_ATTENTION_CALLBACKS_KEY",
    "REFERENCE_TOKEN_COUNTS_KEY",
    "REFERENCE_SPATIAL_SHAPES_KEY",
    "get_attention_callbacks",
    "run_post_attention_callbacks",
    "run_pre_attention_callbacks",
]
