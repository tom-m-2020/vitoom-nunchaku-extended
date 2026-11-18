# Provenance and licensing

## Component boundaries

- **Upstream wheel:** an exact archive published by `tonera/vitoom-nunchaku`.
- **Maintained patch:** the Python callback module, the verified FLUX.2 1.3
  replacement, and the FLUX.2 1.4 source transformation in this repository.
- **Compiled extension:** copied byte-for-byte from the upstream wheel.
- **Derivative wheel:** the verified upstream archive plus maintained Python
  changes, derivative version metadata, and a reconstructed RECORD.

The repository root `LICENSE` contains Apache License 2.0, matching the license
retained in verified upstream wheels. `NOTICE` describes the maintained changes
and upstream relationship. Every derivative wheel retains the upstream license
member in its renamed `.dist-info` directory.

## Reproducibility claim

The derivative wheel is deterministically reproducible from its exact pinned
upstream wheel. This includes byte-identical qualified output hashes, unchanged
compiled-extension hashes, valid metadata and RECORD, and deterministic ZIP
construction.

This repository does not contain or claim a complete corresponding source/build
recipe for the upstream compiled extension. It is not an independent source
reproduction of the CUDA/C++ binary. Public release notes must preserve this
limitation.

## Authoritative inputs

The accepted upstream filenames and hashes are recorded in
[COMPATIBILITY.md](COMPATIBILITY.md) and enforced in
`scripts/wheel_profiles.py`. Download provenance is the upstream Hugging Face
repository; filenames without matching hashes are not authoritative inputs.
