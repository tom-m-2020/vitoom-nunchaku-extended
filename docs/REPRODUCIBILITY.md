# Qualified artifact reproducibility

Reproducibility was verified on 2026-11-18 after relocating the maintained
repacker and patch sources to their root-level layout. Tests used all five
authoritative upstream wheel fixtures and Python 3.13.18.

Command:

```powershell
$env:NUNCHAKU_WHEEL_FIXTURES = 'C:\path\to\authoritative-upstream-wheels'
python -m unittest discover -s tests -v
```

Result: **20 tests passed** in 411.804 seconds.

## Qualified Windows baseline

- Input:
  `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64.whl`
- Input SHA-256:
  `F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025`
- Output:
  `nunchaku-1.3.0.dev20260629+cu13.0torch2.11.kleincallback2-cp313-cp313-win_amd64.whl`
- First output SHA-256:
  `759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C`
- Second output SHA-256:
  `759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C`
- Compiled extension SHA-256 before and after:
  `7B1ACA262EC9AA65E1226E3768109FB20866E8E94F5A4BF0D2D832DCCAAD29A9`
- Result: **PASS**.

## Qualified Windows 1.4.1 candidate

- Input:
  `nunchaku-1.4.1+cu13.0torch2.14-cp313-cp313-win_amd64.whl`
- Input SHA-256:
  `4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F`
- Output:
  `nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl`
- First output SHA-256:
  `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`
- Second output SHA-256:
  `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`
- Compiled extension SHA-256 before and after:
  `997178E9442170A4F68F2D6ED1736AE735CFE8A371885038558C377A368D7927`
- Result: **PASS**.

For both artifacts, `ZipFile.testzip()` returned no failing member, RECORD
digests and sizes validated, wheel metadata and tags were retained with the
derivative version, and compiled-extension bytes were identical to upstream.

This proves deterministic derivative reconstruction from pinned wheel inputs.
It does not prove source-build reproducibility of the upstream compiled binary.
