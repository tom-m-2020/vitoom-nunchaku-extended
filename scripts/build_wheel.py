"""Build the callback-capable derivative from an authoritative Vitoom wheel."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

from rebuild_record import rebuild_record


EXPECTED_VERSION = "1.3.0.dev20260629+cu13.0torch2.11"
DERIVATIVE_VERSION = EXPECTED_VERSION + ".kleincallback1"
EXPECTED_WHEEL_SHA256 = "F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025"
EXPECTED_EXTENSION_SHA256 = "7B1ACA262EC9AA65E1226E3768109FB20866E8E94F5A4BF0D2D832DCCAAD29A9"
OVERLAYS = (
    Path("nunchaku/models/transformers/transformer_flux2.py"),
    Path("nunchaku/models/transformers/flux2_attention_callbacks.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build(input_wheel: Path, output_directory: Path) -> Path:
    if sha256(input_wheel) != EXPECTED_WHEEL_SHA256:
        raise ValueError("Input wheel SHA-256 does not match the qualified Vitoom wheel.")
    source_root = Path(__file__).resolve().parents[1]
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nunchaku-wheel-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(input_wheel) as archive:
            archive.extractall(root)

        old_dist = root / f"nunchaku-{EXPECTED_VERSION}.dist-info"
        new_dist = root / f"nunchaku-{DERIVATIVE_VERSION}.dist-info"
        if not old_dist.is_dir():
            raise ValueError(f"Input wheel lacks expected metadata directory {old_dist.name}.")
        old_dist.rename(new_dist)
        metadata = new_dist / "METADATA"
        text = metadata.read_text(encoding="utf-8")
        expected_line = f"Version: {EXPECTED_VERSION}"
        if expected_line not in text:
            raise ValueError("Input METADATA version does not match the qualified wheel.")
        metadata.write_text(text.replace(expected_line, f"Version: {DERIVATIVE_VERSION}", 1), encoding="utf-8")

        for relative in OVERLAYS:
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Missing maintained overlay {source}.")
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        extensions = list((root / "nunchaku").glob("_C*.pyd"))
        if len(extensions) != 1 or sha256(extensions[0]) != EXPECTED_EXTENSION_SHA256:
            raise ValueError("Compiled extension is absent or differs from the qualified wheel.")

        rebuild_record(root, new_dist)
        tags = input_wheel.name[:-4].split("-")[-3:]
        output = output_directory / f"nunchaku-{DERIVATIVE_VERSION}-{'-'.join(tags)}.whl"
        if output.resolve() == input_wheel.resolve():
            raise ValueError("Refusing to overwrite the authoritative input wheel.")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                archive.write(path, path.relative_to(root).as_posix())
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(build(args.input_wheel.resolve(), args.output_directory.resolve()))


if __name__ == "__main__":
    main()
