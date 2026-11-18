# vitoom-nunchaku-extended

`vitoom-nunchaku-extended` creates verified Python-only derivative wheels from
exact upstream [tonera/vitoom-nunchaku](https://huggingface.co/tonera/vitoom-nunchaku)
wheels. The derivative adds FLUX.2 Klein attention callback API v3 for
[ComfyUI-Nunchaku-Klein](https://github.com/tom-m-2020/ComfyUI-Nunchaku-Klein).

This repository is a wheel repacker, not a Nunchaku source-build checkout. It
does not maintain a full extracted wheel, compile CUDA/C++, or modify the
upstream compiled extension.

## What the patch adds

- Ordered pre- and post-attention callbacks.
- Immutable double/single block identity.
- Generated and ordered-reference token counts and spatial grids.
- Strict callback tensor replacement validation.
- Callback API version `3`.

It does not implement Enhancer policy, masks, schedules, Direct K/V weighting,
Identity Feature Transfer, or LoRA policy. Those consumers live in
ComfyUI-Nunchaku-Klein.

## Repository layout

```text
patches/flux2/common/   Maintained callback API module
patches/flux2/v1_3/    Verified FLUX.2 1.3 replacement source
scripts/                Hash-verified deterministic wheel repacker
tests/                  Callback and archive/reproducibility tests
docs/                   Compatibility, build, install, and provenance records
dist/                   Ignored local output directory
```

The FLUX.2 1.4 patch is an exact, validated transformation in
`scripts/wheel_profiles.py`; it preserves the upstream 1.4 reference-forward
fallback and FP16 conversion behavior. Unknown wheel bytes, source revisions,
and compiled-extension hashes fail closed.

## Build

Obtain an authoritative upstream wheel and verify that it is listed in
[the compatibility matrix](docs/COMPATIBILITY.md). From the repository root:

```powershell
python scripts/build_wheel.py C:\path\to\verified-upstream.whl dist
```

The input is never modified. The output keeps the original Python, ABI, and
platform tags, adds `.kleincallback2` to the package version, reconstructs
`RECORD`, and uses deterministic ZIP metadata. See [BUILDING.md](docs/BUILDING.md).

## Test

Callback tests do not import Nunchaku. Archive integration tests require a
directory containing all five authoritative upstream wheels:

```powershell
$env:NUNCHAKU_WHEEL_FIXTURES = 'C:\path\to\upstream-wheels'
python -m unittest discover -s tests -v
```

The suite verifies all source profiles, unknown-input rejection, missing patch
sources, compiled-extension hash rejection, archive integrity, deterministic
packaging, and exact reproduction of both qualified Windows artifacts.

## Qualified Windows artifacts

- Production baseline: Nunchaku 1.3.0.dev20260629, Python 3.13,
  PyTorch 2.11.0+cu130, CUDA 13.0, Windows x86-64.
- Release candidate: Nunchaku 1.4.1, Python 3.13,
  PyTorch 2.14.0+cu130, CUDA 13.0, Windows x86-64.

The 1.4.1 runtime qualification used an RTX 3060 12 GB. Linux x86-64 and
Linux aarch64 profiles are archive-verified only and are not runtime-qualified.
See [INSTALLATION.md](docs/INSTALLATION.md) before changing environments.

## Licensing and provenance

The repository is licensed under Apache-2.0. The repacker retains the upstream
wheel's license and attribution and copies its compiled extension byte-for-byte.
The derivative is reproducible from a pinned upstream wheel; the upstream
compiled extension is **not** independently reproducible from source here.
See [PROVENANCE.md](docs/PROVENANCE.md) and [NOTICE](NOTICE).

## Release history

- `v1.0.0`: fixed single-stream callback metadata validation.
- `v1.1.0`: added ordered reference token-grid shapes (API v2).
- `v1.2.0`: added generated token-grid shape (API v3).
- Proposed `v1.3.0`: root-level multi-version repacker and qualified 1.4.1
  Windows artifact. No tag has been created.
