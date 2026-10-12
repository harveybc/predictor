"""Physical snapshot of the archive state with a digest manifest, and verified restore.

Sources are named (``cas``, ``outbox``, ``warehouse`` ...); the manifest records names,
relative paths, sizes and SHA-256, never absolute source paths or host names. SQLite files
are copied through SQLite's online backup API (a consistent copy even with WAL). A DuckDB file
is refused while its ``.wal`` exists: stop the owner process or CHECKPOINT first, as the
deploy steps say. Snapshots hold data and stay out of Git; only the manifest digest and a
summary are committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "i6n.archive.snapshot.v1"
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".duckdb" and Path(str(src) + ".wal").exists():
        raise ValueError("DUCKDB_WAL_PRESENT: stop the owner process or CHECKPOINT before a physical copy")
    with src.open("rb") as handle:
        magic = handle.read(16)
    if magic == _SQLITE_MAGIC:
        source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        target = sqlite3.connect(str(dst))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    else:
        shutil.copy2(src, dst)


def take_snapshot(sources: dict, dest: str | Path) -> dict:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("SNAPSHOT_DESTINATION_NOT_EMPTY")
    files = []
    kinds = {}
    for name, src in sorted(sources.items()):
        src = Path(src)
        if src.is_dir():
            kinds[name] = "dir"
            members = sorted(p for p in src.rglob("*") if p.is_file() and ".tmp" not in p.name)
            for member in members:
                rel = Path(name) / member.relative_to(src)
                _copy_file(member, dest / rel)
        elif src.is_file():
            kinds[name] = "file"
            rel = Path(name) / src.name
            _copy_file(src, dest / rel)
        else:
            raise ValueError(f"SNAPSHOT_SOURCE_MISSING: {name}")
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        files.append({"path": str(path.relative_to(dest)), "size": path.stat().st_size, "sha256": _sha(path)})
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "sources": kinds,
                "files": files}
    manifest["files_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify_snapshot(src: str | Path, restore_to: str | Path | None = None) -> dict:
    src = Path(src)
    manifest = json.loads((src / "MANIFEST.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("SNAPSHOT_SCHEMA")
    if hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest() != manifest["files_sha256"]:
        raise ValueError("SNAPSHOT_MANIFEST_DIGEST_MISMATCH")
    listed = {f["path"] for f in manifest["files"]}
    present = {str(p.relative_to(src)) for p in src.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    if listed != present:
        raise ValueError(f"SNAPSHOT_INVENTORY_MISMATCH: {sorted(listed ^ present)[:5]}")
    for entry in manifest["files"]:
        if _sha(src / entry["path"]) != entry["sha256"]:
            raise ValueError("SNAPSHOT_DIGEST_MISMATCH: " + entry["path"])
    if restore_to is not None:
        restore_to = Path(restore_to)
        if restore_to.exists() and any(restore_to.iterdir()):
            raise ValueError("RESTORE_DESTINATION_NOT_EMPTY")
        for entry in manifest["files"]:
            target = restore_to / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / entry["path"], target)
            if _sha(target) != entry["sha256"]:
                raise ValueError("RESTORE_DIGEST_MISMATCH: " + entry["path"])
    return manifest


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Take or verify a lane D physical snapshot.")
    sub = parser.add_subparsers(dest="command", required=True)
    take = sub.add_parser("take")
    take.add_argument("--source", action="append", required=True, help="name=path (dir or file)")
    take.add_argument("--dest", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--dest", type=Path, required=True)
    verify.add_argument("--restore-to", type=Path)
    args = parser.parse_args(argv)
    if args.command == "take":
        sources = dict(item.split("=", 1) for item in args.source)
        manifest = take_snapshot(sources, args.dest)
    else:
        manifest = verify_snapshot(args.dest, restore_to=args.restore_to)
    print(json.dumps({"schema": manifest["schema"], "files": len(manifest["files"]),
                      "bytes": sum(f["size"] for f in manifest["files"]),
                      "files_sha256": manifest["files_sha256"], "verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
