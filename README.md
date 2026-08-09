# Vitoom Nunchaku callback derivative

This directory is an exact Python/package snapshot extracted from the qualified
Vitoom wheel, plus project-maintained Apache-2.0 Python changes. It is not a
source-build checkout and has no known public Git commit corresponding exactly
to the complete released wheel.

Build a derivative from the authoritative wheel rather than packaging this
directory directly:

```powershell
python scripts/build_wheel.py path\to\qualified.whl dist
```

The script verifies the wheel and compiled-extension hashes, extracts to a
temporary directory, overlays only the maintained Python files, updates the
derivative version and RECORD, and writes a new wheel. It never changes the
input wheel. The compiled `_C*.pyd` remains byte-for-byte upstream and is
ignored by Git.
