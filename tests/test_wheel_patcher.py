import base64
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_wheel import _verify_extension, build, sha256  # noqa: E402
from wheel_profiles import (  # noqa: E402
    CALLBACKS_PATH,
    CALLBACK_SOURCE_PATH,
    PATCHED_TRANSFORMER_1_3_PATH,
    SOURCE_1_3_SHA256,
    SOURCE_1_4_SHA256,
    TRANSFORMER_PATH,
    WHEELS,
    normalized_sha256,
    patch_1_4,
)


FIXTURES = os.environ.get("NUNCHAKU_WHEEL_FIXTURES")
QUALIFIED_OUTPUTS = {
    "F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025":
        "759566A8FDA3559E8BA516065EEBDC41F2A39EBD31A5C1357C75DB0C0FF9FB9C",
    "4EA96A0E649ADEA7439D38C0A743DEAF5725B74F367C9A5CFD01012BF45B124F":
        "AD2CDAA40E48B138D4B1F9B6AA6F9CD99364DCE5169B836BBDEC036D83053FEF",
}


def _record_is_valid(archive: zipfile.ZipFile) -> bool:
    record = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
    rows = csv.reader(archive.read(record).decode("utf-8").splitlines())
    for name, digest, size in rows:
        if name == record:
            if digest or size:
                return False
            continue
        data = archive.read(name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        if digest != f"sha256={encoded}" or size != str(len(data)):
            return False
    return True


class WheelPatcherTests(unittest.TestCase):
    def test_unknown_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = root / "unknown.whl"
            unknown.write_bytes(b"not a verified wheel")
            with self.assertRaisesRegex(ValueError, "Unsupported input wheel SHA-256"):
                build(unknown, root / "output")

    def test_maintained_patch_sources_are_present(self):
        self.assertTrue((ROOT / CALLBACK_SOURCE_PATH).is_file())
        self.assertTrue((ROOT / PATCHED_TRANSFORMER_1_3_PATH).is_file())

    @unittest.skipUnless(FIXTURES, "set NUNCHAKU_WHEEL_FIXTURES for archive integration tests")
    def test_missing_callback_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = next(
                path for path in Path(FIXTURES).glob("*.whl")
                if sha256(path) in WHEELS
            )
            with mock.patch("build_wheel.CALLBACK_SOURCE_PATH", Path("missing.py")):
                with self.assertRaisesRegex(FileNotFoundError, "Missing maintained callback module"):
                    build(source, root / "output")

    def test_compiled_extension_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "nunchaku"
            package.mkdir()
            (package / "_C.cp313-win_amd64.pyd").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Compiled extension differs"):
                _verify_extension(Path(temporary), "0" * 64)

    @unittest.skipUnless(FIXTURES, "set NUNCHAKU_WHEEL_FIXTURES for archive integration tests")
    def test_every_verified_source_profile_transforms(self):
        found = set()
        for wheel in Path(FIXTURES).glob("*.whl"):
            wheel_hash = sha256(wheel)
            profile = WHEELS.get(wheel_hash)
            if profile is None:
                continue
            found.add(wheel_hash)
            with zipfile.ZipFile(wheel) as archive:
                source = archive.read(TRANSFORMER_PATH.as_posix())
            if profile["profile"] == "flux2-1.3":
                self.assertEqual(normalized_sha256(source), SOURCE_1_3_SHA256)
            else:
                self.assertEqual(normalized_sha256(source), SOURCE_1_4_SHA256)
                patched = patch_1_4(source)
                compile(patched, TRANSFORMER_PATH.as_posix(), "exec")
                text = patched.decode("utf-8")
                self.assertIn("run_pre_attention_callbacks", text)
                self.assertIn("run_post_attention_callbacks", text)
                self.assertIn("convert_fp16(transformer, model_state_dict)", text)
                self.assertIn("if kv_cache_mode is not None or num_ref_tokens > 0:", text)
                self.assertLess(
                    text.index("num_tokens_pad = num_txt_tokens_pad + num_img_tokens_pad"),
                    text.index("padded_text_token_count=num_txt_tokens_pad"),
                )
        self.assertEqual(found, set(WHEELS))

    @unittest.skipUnless(FIXTURES, "set NUNCHAKU_WHEEL_FIXTURES for archive integration tests")
    def test_build_is_reproducible_and_preserves_extension(self):
        wheel_hash = "F8F9E7D3DE46603F4772F96FA68A8F02C57E0DA511CD274FC9EF6FADAEC62025"
        source = next(path for path in Path(FIXTURES).glob("*.whl") if sha256(path) == wheel_hash)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build(source, root / "first")
            second = build(source, root / "second")
            self.assertEqual(sha256(first), sha256(second))
            with zipfile.ZipFile(source) as original, zipfile.ZipFile(first) as patched:
                extension = next(
                    name for name in original.namelist()
                    if name.startswith("nunchaku/_C") and name.endswith((".pyd", ".so"))
                )
                self.assertEqual(original.read(extension), patched.read(extension))
                self.assertIn(CALLBACKS_PATH.as_posix(), patched.namelist())
                self.assertTrue(_record_is_valid(patched))
                self.assertIsNone(patched.testzip())

    @unittest.skipUnless(FIXTURES, "set NUNCHAKU_WHEEL_FIXTURES for archive integration tests")
    def test_qualified_windows_outputs_are_exact(self):
        fixtures = {sha256(path): path for path in Path(FIXTURES).glob("*.whl")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for input_hash, output_hash in QUALIFIED_OUTPUTS.items():
                with self.subTest(input_hash=input_hash):
                    source = fixtures[input_hash]
                    first = build(source, root / input_hash / "first")
                    second = build(source, root / input_hash / "second")
                    self.assertEqual(sha256(first), output_hash)
                    self.assertEqual(sha256(second), output_hash)


if __name__ == "__main__":
    unittest.main()
