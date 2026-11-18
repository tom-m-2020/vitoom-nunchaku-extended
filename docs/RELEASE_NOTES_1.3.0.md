# Draft release notes: repository v1.3.0

This proposed repository release reorganizes `vitoom-nunchaku-extended` into a
root-level multi-version repacker and adds the qualified Nunchaku 1.4.1 Windows
derivative. It does not change callback API v3 or add backend functionality.

## Qualified 1.4.1 artifact

- `nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256 `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`
- Upstream input SHA-256 `4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F`
- Windows x86-64, Python 3.13, PyTorch 2.14.0+cu130, CUDA 13.0
- Runtime qualification hardware: RTX 3060 12 GB
- Callback API: v3

The qualified Nunchaku 1.3/PyTorch 2.11 Windows environment remains the rollback
baseline. Upgrade only with a separately provisioned environment; see
[INSTALLATION.md](INSTALLATION.md).

## Repacker changes

- Five exact upstream distributions remain SHA-256 allowlisted.
- FLUX.2 1.3 and 1.4 patch behavior is unchanged.
- Deterministic packaging, metadata/RECORD reconstruction, extension integrity,
  and fail-closed validation are preserved.
- The full extracted upstream wheel was removed from the maintained tree.
- Build, compatibility, provenance, installation, and rollback documentation is
  now version-independent.

## Limitations

- Linux x86-64 and Linux aarch64 wheels are archive-verified, not runtime-qualified.
- Other GPUs and environment combinations are unqualified.
- Fresh empty-environment dependency resolution was not qualified.
- Controlled prompt-level perceptual parity was not established.
- The upstream compiled extension is unchanged but is not independently
  reproducible from source by this repository.

No tag or release has been created by this preparation work.
