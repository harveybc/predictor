"""Verified physical analytical copies of an I6-DMO root on the local mirror and a second host.

Mirrored: sealed configs, plan, status, gates, stored terminals (with their artifacts: checkpoint,
predictions, receipts), REPORTS.json/PUBLICATION.json and consistent SQLite backups of the queue and
outbox. Not mirrored: the private deployment copy and live attempt scratch. Each target is
verified by re-hashing every file there; MIRRORS.json records the manifest digest per target.
"""
import hashlib
import json
import shlex
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

INCLUDE = ("EFFECTIVE_CONFIG.json", "LONG_EFFECTIVE_CONFIG.json", "PLAN.json", "STATUS.json", "REPORTS.json",
           "PUBLICATION.json", "CLOSE.json")
INCLUDE_DIRS = ("gates", "terminals", "backup")


def _backup_sqlite(root):
    out = Path(root) / "backup"
    out.mkdir(exist_ok=True)
    for name in ("queue.sqlite3", "outbox.sqlite3"):
        src = Path(root) / "state" / name
        if not src.exists():
            continue
        tmp = out / f".{name}.tmp"
        with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as a, sqlite3.connect(tmp) as b:
            a.backup(b)
        tmp.replace(out / name)


def manifest(root):
    root = Path(root)
    _backup_sqlite(root)
    files = [root / n for n in INCLUDE if (root / n).exists()]
    for folder in INCLUDE_DIRS:
        if (root / folder).exists():
            files.extend(p for p in sorted((root / folder).rglob("*")) if p.is_file() and not p.name.startswith("."))
    return {str(p.relative_to(root)): file_sha256(p) for p in files}


def _sync_local(root, dest, entries):
    dest = Path(dest)
    for rel in entries:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_sha256(target) != entries[rel]:
            shutil.copy2(Path(root) / rel, target)
    bad = [rel for rel, sha in entries.items() if not (dest / rel).exists() or file_sha256(dest / rel) != sha]
    return bad


def _sync_remote(root, alias, dest, entries, run):
    listing = Path(root) / "backup" / ".mirror-files.txt"
    listing.write_text("\n".join(sorted(entries)) + "\n")
    out = run(["rsync", "-a", "--checksum", f"--files-from={listing}", f"{root}/", f"{alias}:{dest}/"],
              capture_output=True, timeout=3600)
    if out.returncode != 0:
        return sorted(entries)
    script = "cd " + shlex.quote(str(dest)) + " && sha256sum -- " + " ".join(shlex.quote(r) for r in sorted(entries))
    out = run(["ssh", "-o", "BatchMode=yes", alias, "--", script], capture_output=True, text=True, timeout=600)
    seen = {}
    for line in out.stdout.splitlines():
        sha, _, rel = line.partition("  ")
        seen[rel] = sha
    return [rel for rel, sha in entries.items() if seen.get(rel) != sha]


def sync_all(root, spec, run=None):
    run = run or subprocess.run
    root = Path(root)
    entries = manifest(root)
    digest = sha256_json(entries)
    targets = {}
    mirrors = spec["mirrors"]
    local_bad = _sync_local(root, mirrors["local"], entries)
    targets["local"] = dict(state="VERIFIED" if not local_bad else "MISMATCH", files=len(entries), bad=local_bad[:20])
    role = spec["roles"].get(mirrors.get("second_host_role") or "", {})
    alias = role.get("ssh_alias")
    if alias:
        bad = _sync_remote(root, alias, mirrors["second_host_path"], entries, run)
    else:
        bad = _sync_local(root, mirrors["second_host_path"], entries)
    targets["second_host"] = dict(state="VERIFIED" if not bad else "MISMATCH", files=len(entries), bad=bad[:20],
                                  role=mirrors.get("second_host_role"))
    doc = dict(schema="i6dmo.mirrors.v1", state="VERIFIED" if all(t["state"] == "VERIFIED" for t in targets.values())
               else "MISMATCH", manifest_sha256=digest, files=len(entries), targets=targets, verified_at=time.time())
    atomic_json(root / "MIRRORS.json", doc)
    return doc


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load(root):
    return json.loads((Path(root) / "MIRRORS.json").read_text())
