"""Preserve invalid ARCH_C receipts before the origin-alignment successor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tools.i6a_campaign import atomic_json


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    root = args.results_dir
    archive = root / "archive" / "ARCH_C_pre_origin_alignment_v1"
    if archive.exists():
        raise ValueError("ARCHIVE_ALREADY_EXISTS")
    files = sorted(set(root.glob("ARCH_C_val2024_week*.json")) |
                   set(root.glob("ARCH_C_val2024_week*.log")) |
                   set(root.glob("ARCH_C_week*.json")) |
                   set(root.glob("ARCH_C_week*.log")))
    if not files:
        raise ValueError("NO_PREDECESSOR_FILES")
    manifest = {"schema": "i6a.arch_c_superseded_artifacts.v1",
                "reason": "standard stride-2 Conv1D discarded the origin row; successor aligns each halving to the last sample",
                "files": [{"name": f.name, "sha256": sha256(f), "bytes": f.stat().st_size} for f in files]}
    archive.mkdir(parents=True)
    for f in files:
        f.rename(archive / f.name)
    atomic_json(archive / "MANIFEST.json", manifest)
    print({"archived": len(files), "destination": str(archive)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
