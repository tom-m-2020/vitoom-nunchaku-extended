# Compatibility matrix

The repacker accepts only the exact upstream SHA-256 values registered in
`scripts/wheel_profiles.py`. Archive support means deterministic repacking,
source-profile validation, unchanged compiled-extension bytes, valid metadata
and RECORD, and archive integrity. It does not imply runtime qualification.

| Upstream wheel | Input SHA-256 | Profile | Derivative SHA-256 | Runtime status |
| --- | --- | --- | --- | --- |
| `nunchaku-1.3.0.dev20260624+cu13.0torch2.12-cp313-cp313-linux_x86_64.whl` | `4A04F47405ABCA7968D919847DF853F08FE0FBCFC4F19C21B29184A9D68A1233` | `flux2-1.3` | `B8ED10A1AF487BBCBF352F962AEDCCDA750009C5F1F894D9F156EC4444256279` | Archive verified; Linux runtime not run |
| `nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp313-cp313-win_amd64.whl` | `F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025` | `flux2-1.3` | `759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C` | Production-qualified baseline |
| `nunchaku-1.4.0+cu13.0torch2.11-cp311-cp311-linux_aarch64.whl` | `C5A47BA1622DEF6C92F412365F8CBC9C959838B705F6AFC33439E5C60CB1DFB6` | `flux2-1.4` | `07B4922C9B0D786AF1FDF10CEDE2AE9D37AFC3AED05467596B987FB3BC0696F2` | Archive verified; Linux/aarch64 runtime not run |
| `nunchaku-1.4.1+cu13.0torch2.14-cp313-cp313-linux_x86_64.whl` | `BE645687F99A4C8CC126080331810EDD244EB7D36B8F85042F4846132CAD2CE3` | `flux2-1.4` | `CEE8CD01EC58953DF2D212D97D7EEDE9BA43914AB6566799DC9FFB8FA3DE2E78` | Archive verified; Linux runtime not run |
| `nunchaku-1.4.1+cu13.0torch2.14-cp313-cp313-win_amd64.whl` | `4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F` | `flux2-1.4` | `AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF` | Runtime-qualified release candidate |

The 1.4.1 Windows qualification is limited to Python 3.13,
PyTorch 2.14.0+cu130, CUDA 13.0, Windows x86-64, and the tested RTX 3060
12 GB environment. The 1.3 Windows baseline remains qualified in its separate
PyTorch 2.11.0+cu130 environment.

Other operating systems, architectures, GPUs, Python versions, PyTorch builds,
CUDA builds, and wheel bytes are unsupported until separately verified.
