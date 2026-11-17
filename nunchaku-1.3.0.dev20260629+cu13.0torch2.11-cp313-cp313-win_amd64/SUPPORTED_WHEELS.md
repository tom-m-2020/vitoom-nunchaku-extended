# Verified wheel profiles

`scripts/build_wheel.py` accepts only the exact upstream SHA-256 values registered in
`scripts/wheel_profiles.py`. It reads and writes wheel archives without importing or
installing Nunchaku. Unknown wheels and unexpected FLUX.2 source or extension hashes
are rejected.

| Upstream wheel | Profile | Archive support | Runtime status |
| --- | --- | --- | --- |
| 1.3.0.dev20260624, CP313, Linux x86_64, CUDA 13.0 / Torch 2.12 | `flux2-1.3` | verified | not run on Linux |
| 1.3.0.dev20260629, CP313, Windows x86_64, CUDA 13.0 / Torch 2.11 | `flux2-1.3` | verified | existing qualified baseline |
| 1.4.0, CP311, Linux aarch64, CUDA 13.0 / Torch 2.11 | `flux2-1.4` | verified | not run on Linux/aarch64 |
| 1.4.1, CP313, Linux x86_64, CUDA 13.0 / Torch 2.14 | `flux2-1.4` | verified | not run on Linux |
| 1.4.1, CP313, Windows x86_64, CUDA 13.0 / Torch 2.14 | `flux2-1.4` | verified | isolated GPU validation only; not a qualified release |

The `flux2-1.4` adapter preserves upstream FP16 conversion and the reference-forward
fallback. Compiled `.pyd` and `.so` files are copied byte-for-byte. The output keeps
the original Python/ABI/platform tag, adds `.kleincallback2` to the distribution
version, rebuilds `RECORD`, and uses deterministic ZIP metadata.

Archive compatibility does not imply binary or runtime compatibility. Linux and
aarch64 outputs require testing on matching systems before release.
