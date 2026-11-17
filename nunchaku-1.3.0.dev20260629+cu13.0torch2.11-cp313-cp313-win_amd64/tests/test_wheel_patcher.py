import base64
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_wheel import build, sha256  # noqa: E402
from wheel_profiles import (  # noqa: E402
    CALLBACKS_PATH,
    SOURCE_1_3_SHA256,
    SOURCE_1_4_SHA256,
    TRANSFORMER_PATH,
    WHEELS,
    normalized_sha256,
    patch_1_4,
)


FIXTURES = os.environ.get("NUNCHAKU_WHEEL_FIXTURES")


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


if __name__ == "__main__":
    unittest.main()
