# Project-maintained changes

Baseline wheel: `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64.whl`

- Added generic ordered FLUX.2 pre-attention and post-attention callbacks.
- Added immutable invocation metadata and strict replacement validation.
- Added immutable double/single block identity at backend construction.
- No Enhancer policy, weighting, masks, schedules, or identity logic.
- No compiled-extension changes.

Modified upstream Python symbol surface:

- `NunchakuFlux2Attention`
- `NunchakuFlux2ParallelSelfAttention`
- `NunchakuFlux2TransformerBlock`
- `NunchakuFlux2SingleTransformerBlock`
- `NunchakuFlux2Transformer2DModel._patch_model`

Added module: `nunchaku.models.transformers.flux2_attention_callbacks`.
