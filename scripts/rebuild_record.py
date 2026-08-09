"""Regenerate a wheel RECORD using only the Python standard library."""

import base64
import csv
import hashlib
from pathlib import Path


def rebuild_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path == record:
            rows.append((relative, "", ""))
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        rows.append((relative, f"sha256={digest}", str(len(data))))
    with record.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


__all__ = ["rebuild_record"]
