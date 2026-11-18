# FLUX.2 callback patch notes

## Current maintained behavior

- Generic ordered FLUX.2 pre-attention and post-attention callbacks.
- Immutable callback invocation metadata and strict replacement validation.
- Ordered reference token-grid shapes, validated against reference token counts.
- Generated token-grid shape, validated against the generated token count.
- Immutable double/single block identity assigned during backend patching.
- No compiled-extension changes.
- No consumer policy such as Enhancer weighting, masks, schedules, Direct K/V,
  Identity Feature Transfer, or LoRA composition.

Maintained Python surfaces:

- `patches/flux2/common/flux2_attention_callbacks.py`
- `patches/flux2/v1_3/transformer_flux2.py`
- The FLUX.2 1.4 transformation in `scripts/wheel_profiles.py`

Output module added to every derivative wheel:
`nunchaku.models.transformers.flux2_attention_callbacks`.

## Historical releases

- `v1.0.0`: fixed the single-stream callback metadata bug without weakening
  validation.
- `v1.1.0`: API v2 added ordered reference token-grid shapes.
- `v1.2.0`: API v3 added the generated token-grid shape.

The proposed `v1.3.0` repository release does not change callback API behavior.
It promotes the existing five-profile repacker to the repository root, removes
the redundant extracted upstream wheel tree, and documents the qualified
Nunchaku 1.4.1 Windows artifact.
