# Building derivative wheels

## Requirements

- Python 3.10 or newer; the repacker itself uses only the standard library.
- An authoritative upstream wheel downloaded from
  `https://huggingface.co/tonera/vitoom-nunchaku`.
- Enough local disk space for the input, temporary extraction, and output.

The build does not import or install Nunchaku and does not require CUDA.

## Build command

Run from the repository root:

```powershell
$input = 'C:\path\to\nunchaku-verified-upstream.whl'
(Get-FileHash -Algorithm SHA256 -LiteralPath $input).Hash
python scripts/build_wheel.py $input dist
```

Compare the input hash with [COMPATIBILITY.md](COMPATIBILITY.md). The builder
also performs this check and rejects unknown bytes.

To demonstrate deterministic output, build into two empty directories:

```powershell
python scripts/build_wheel.py $input dist\first
python scripts/build_wheel.py $input dist\second
Get-FileHash -Algorithm SHA256 dist\first\*.whl
Get-FileHash -Algorithm SHA256 dist\second\*.whl
```

## Validation performed by the builder

1. Full input-wheel SHA-256 allowlist lookup.
2. Expected FLUX.2 Python source hash.
3. Profile-specific 1.3 replacement or 1.4 transformation.
4. Presence of the maintained callback module.
5. Exact compiled-extension SHA-256.
6. Exact metadata version replacement.
7. Complete RECORD reconstruction.
8. Deterministic member order, timestamps, permissions, and compression.
9. Post-write compiled-extension byte verification.

The builder never overwrites the input wheel. `scripts/patch_wheel.py` remains a
compatibility entry point and delegates to the same authoritative builder.

## Tests

```powershell
$env:NUNCHAKU_WHEEL_FIXTURES = 'C:\path\to\all-five-upstream-wheels'
python -m unittest discover -s tests -v
```

Do not add a profile using only a filename or version. A new profile requires
the exact wheel hash, relevant source hash, extension hash, a reviewed patch
contract, archive tests, and appropriate runtime qualification.
