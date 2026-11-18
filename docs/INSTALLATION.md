# Installation and rollback

Install only a wheel matching the intended Python, PyTorch, CUDA, operating
system, and architecture. Verify its SHA-256 before installation.

## Nunchaku 1.4.1 Windows candidate

- File: `nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256: `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF`
- Python 3.13, PyTorch 2.14.0+cu130, CUDA 13.0, Windows x86-64

Create and retain a separate environment. Never install this wheel over the
qualified PyTorch 2.11 environment.

```powershell
$wheel = 'C:\path\to\nunchaku-1.4.1+cu13.0torch2.14.kleincallback2-cp313-cp313-win_amd64.whl'
$expected = 'AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $wheel).Hash -ne $expected) {
    throw 'Wheel SHA-256 mismatch'
}
python -c "import sys, torch; assert sys.version_info[:2] == (3, 13); assert torch.__version__ == '2.14.0+cu130'; assert torch.version.cuda == '13.0'"
python -m pip install $wheel
python -m pip check
python -c "import importlib.metadata as m; import nunchaku._C; from nunchaku.models.transformers.flux2_attention_callbacks import FLUX2_ATTENTION_CALLBACK_API_VERSION as api; print(m.version('nunchaku'), api); assert api == 3"
```

Review dependency resolver changes before accepting them. Start ComfyUI from
this separate environment and verify node registration plus one sequential 4B
and 9B workflow before operational promotion.

## Qualified 1.3 baseline and rollback

- File: `nunchaku-1.3.0.dev20260629+cu13.0torch2.11.kleincallback2-cp313-cp313-win_amd64.whl`
- SHA-256: `759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C`
- Python 3.13, PyTorch 2.11.0+cu130, CUDA 13.0, Windows x86-64

Rollback means stopping the candidate ComfyUI process and starting the retained
qualified PyTorch 2.11 environment. It must not depend on reconstructing an old
environment after overwriting it. Keep the baseline environment and exact wheel
archive intact before testing 1.4.1.

Linux derivatives are not runtime-qualified by this project.
