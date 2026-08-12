"""
FP overrides: block/sub-block level mixed-precision support.

Reads ``fp_overrides`` from checkpoint metadata and classifies blocks
(or sub-modules within blocks) that should remain in FP16/BF16.

The quantization side stores FP16 weights directly in the same nunchaku
checkpoint file, using diffusers-native key names for full-FP16 blocks
and standard ``nn.Linear`` keys for mod-only sub-modules.  This allows
``load_state_dict`` to handle everything in a single pass—no external
diffusers checkpoint path is needed at inference time.

This module is model-agnostic.  Model-specific wrappers live in their
respective transformer files.
"""

from __future__ import annotations

import fnmatch
from typing import Any


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def parse_fp_overrides(quantization_config: dict[str, Any]) -> list[str]:
    """Extract ``fp_overrides`` from checkpoint metadata.

    Returns a list of module-path patterns whose parameters should be kept
    in FP16/BF16 rather than quantized.
    """
    raw = quantization_config.get("fp_overrides", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(s) for s in raw]
    raise TypeError(f"fp_overrides must be str | list[str], got {type(raw)}")


def matches_fp_override(name: str, fp_overrides: list[str]) -> bool:
    """Check whether *name* (a fully-qualified module path) falls under any
    of the *fp_overrides* patterns.

    Supports exact match, prefix match (``name`` starts with ``pattern + '.'``),
    and glob-style wildcards via :func:`fnmatch.fnmatch`.
    """
    for pattern in fp_overrides:
        if pattern == "*":
            return True
        if name == pattern or name.startswith(pattern + "."):
            return True
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Block-level helpers
# ---------------------------------------------------------------------------

def classify_block_overrides(
    num_blocks: int,
    block_prefix: str,
    fp_overrides: list[str],
) -> tuple[set[int], dict[int, list[str]]]:
    """Classify which blocks need full FP16 vs sub-module FP16.

    Parameters
    ----------
    num_blocks : int
        Total number of transformer blocks.
    block_prefix : str
        The prefix in the state_dict (e.g. ``"transformer_blocks"``).
    fp_overrides : list[str]
        The parsed override patterns.

    Returns
    -------
    full_fp16_indices : set[int]
        Block indices that should be entirely replaced with diffusers FP16.
    submodule_fp16_map : dict[int, list[str]]
        Block index → list of sub-module suffixes to keep in FP16 (e.g.
        ``["img_mod", "txt_mod"]``).
    """
    full_fp16: set[int] = set()
    submodule_map: dict[int, list[str]] = {}

    for i in range(num_blocks):
        block_fqn = f"{block_prefix}.{i}"
        if matches_fp_override(block_fqn, fp_overrides):
            full_fp16.add(i)
            continue
        subs = []
        for pattern in fp_overrides:
            if pattern.startswith(block_fqn + "."):
                suffix = pattern[len(block_fqn) + 1:]
                subs.append(suffix)
        if subs:
            submodule_map[i] = subs

    return full_fp16, submodule_map
