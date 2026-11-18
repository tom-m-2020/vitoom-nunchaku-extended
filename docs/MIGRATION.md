# Repository layout migration

The repository originally tracked a complete extracted 1.3 wheel under a
version-specific directory. The multi-version repacker was added inside that
snapshot. The extracted upstream tree was redundant and made the maintained
source boundary unclear.

| Old location | New location |
| --- | --- |
| `<1.3-wheel>/scripts/` | `scripts/` |
| `<1.3-wheel>/tests/` | `tests/` |
| `<1.3-wheel>/nunchaku/models/transformers/flux2_attention_callbacks.py` | `patches/flux2/common/flux2_attention_callbacks.py` |
| `<1.3-wheel>/nunchaku/models/transformers/transformer_flux2.py` | `patches/flux2/v1_3/transformer_flux2.py` |
| `<1.3-wheel>/SUPPORTED_WHEELS.md` | `docs/COMPATIBILITY.md` |

All other extracted wheel members were removed from the current tree after the
new layout reproduced both qualified Windows artifacts twice and passed the
complete test suite. They remain available in Git history. No compiled binary
was tracked or modified.

The authoritative implementation is now `scripts/build_wheel.py` plus
`scripts/wheel_profiles.py` and the narrowly scoped sources under `patches/`.
