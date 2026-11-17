"""Build a callback-capable derivative from a verified Vitoom wheel."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

from rebuild_record import rebuild_record
from wheel_profiles import (
    CALLBACKS_PATH,
    SOURCE_1_3_SHA256,
    TRANSFORMER_PATH,
    WHEELS,
    normalized_sha256,
    patch_1_4,
)


DERIVATIVE_SUFFIX = ".kleincallback2"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metadata_directory(root: Path, version: str) -> Path:
    expected = root / f"nunchaku-{version}.dist-info"
    if not expected.is_dir():
        raise ValueError(f"Input wheel lacks expected metadata directory {expected.name}.")
    return expected


def _verify_extension(root: Path, expected_hash: str) -> Path:
    extensions = [
        path
        for path in (root / "nunchaku").iterdir()
        if path.is_file() and path.name.startswith("_C") and path.suffix in {".pyd", ".so"}
    ]
    if len(extensions) != 1:
        raise ValueError("Input wheel must contain exactly one compiled Nunchaku extension.")
    if sha256(extensions[0]) != expected_hash:
        raise ValueError("Compiled extension differs from the verified upstream wheel.")
    return extensions[0]


def _write_deterministic_wheel(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build(input_wheel: Path, output_directory: Path) -> Path:
    wheel_hash = sha256(input_wheel)
    profile = WHEELS.get(wheel_hash)
    if profile is None:
        raise ValueError(
            f"Unsupported input wheel SHA-256 {wheel_hash}; refusing an unknown distribution."
        )

    source_root = Path(__file__).resolve().parents[1]
    output_directory.mkdir(parents=True, exist_ok=True)
    version = profile["version"]
    derivative_version = version + DERIVATIVE_SUFFIX

    with tempfile.TemporaryDirectory(prefix="nunchaku-wheel-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(input_wheel) as archive:
            archive.extractall(root)

        old_dist = _metadata_directory(root, version)
        transformer = root / TRANSFORMER_PATH
        if not transformer.is_file():
            raise ValueError(f"Input wheel lacks {TRANSFORMER_PATH.as_posix()}.")

        if profile["profile"] == "flux2-1.3":
            if normalized_sha256(transformer.read_bytes()) != SOURCE_1_3_SHA256:
                raise ValueError("FLUX.2 1.3 source hash is not verified.")
            transformer.write_bytes((source_root / TRANSFORMER_PATH).read_bytes())
        elif profile["profile"] == "flux2-1.4":
            transformer.write_bytes(patch_1_4(transformer.read_bytes()))
        else:
            raise ValueError(f"Unknown internal patch profile {profile['profile']!r}.")

        callback_source = source_root / CALLBACKS_PATH
        if not callback_source.is_file():
            raise FileNotFoundError(f"Missing maintained callback module {callback_source}.")
        callback_destination = root / CALLBACKS_PATH
        callback_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(callback_source, callback_destination)

        extension = _verify_extension(root, profile["extension"])
        extension_hash = sha256(extension)

        new_dist = root / f"nunchaku-{derivative_version}.dist-info"
        old_dist.rename(new_dist)
        metadata = new_dist / "METADATA"
        text = metadata.read_text(encoding="utf-8")
        expected_line = f"Version: {version}"
        if text.count(expected_line) != 1:
            raise ValueError("Input METADATA version does not match the verified wheel.")
        metadata.write_text(
            text.replace(expected_line, f"Version: {derivative_version}", 1),
            encoding="utf-8",
            newline="\n",
        )

        rebuild_record(root, new_dist)
        tags = input_wheel.name[:-4].split("-")[-3:]
        output = output_directory / f"nunchaku-{derivative_version}-{'-'.join(tags)}.whl"
        if output.resolve() == input_wheel.resolve():
            raise ValueError("Refusing to overwrite the authoritative input wheel.")
        _write_deterministic_wheel(root, output)

        with zipfile.ZipFile(output) as archive:
            extension_name = extension.relative_to(root).as_posix()
            if hashlib.sha256(archive.read(extension_name)).hexdigest().upper() != extension_hash:
                raise RuntimeError("Compiled extension changed while writing the derivative wheel.")
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(build(args.input_wheel.resolve(), args.output_directory.resolve()))


if __name__ == "__main__":
    main()
