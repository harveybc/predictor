#!/usr/bin/env python3
"""Copy assigned E3-M artifacts to a second disk and verify sealed outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_package_worker import atomic, digest, identity, read


def require_storage(destination, mount):
    mount, destination = Path(mount).resolve(), Path(destination).resolve()
    if not os.path.ismount(mount):
        raise ValueError("STORAGE_MOUNT_REQUIRED")
    if not destination.is_relative_to(mount) or destination == mount:
        raise ValueError("STORAGE_DESTINATION_OUTSIDE_MOUNT")
    state = os.statvfs(mount)
    if state.f_flag & os.ST_RDONLY:
        raise ValueError("STORAGE_READ_ONLY")
    if state.f_bavail * state.f_frsize < 1024**3:
        raise ValueError("STORAGE_FREE_SPACE")


def validate_mirror(root):
    root = Path(root)
    cfg = read(root / "ops/CONFIG.private.json")
    if not isinstance(cfg, dict):
        raise ValueError("CONFIG_NOT_COLLECTED")
    config_id = identity(cfg)
    verified = []
    for path in sorted((root / "ops").glob("*.RESULT.json")):
        receipt = read(path)
        job = receipt["job"]["name"]
        if Path(job).name != job or receipt["config_sha256"] != config_id:
            raise ValueError("MIRROR_RECEIPT_IDENTITY")
        for name, expected in receipt["hashes"].items():
            if Path(name).name != name or digest(root / "out" / job / name) != expected:
                raise ValueError("MIRROR_OUTPUT_CHANGED:" + job + ":" + name)
        verified.append(job)
    closed = read(root / "ops/CLOSE.json")
    complete = False
    if closed:
        if (closed["config_sha256"] != config_id or closed["done"] != len(verified)
                or digest(root / "CONSUMERS.json") != closed["consumer_sha256"]):
            raise ValueError("MIRROR_CLOSURE_MISMATCH")
        complete = closed["pending"] == 0
    return {"verified_jobs": verified, "count": len(verified), "closed": complete,
            "config_sha256": config_id, "closure": closed,
            "scope": "physical mirror, not warehouse delivery or scientific promotion"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--storage-mount", required=True)
    p.add_argument("--timer-unit", default="")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.host):
        raise ValueError("INVALID_SSH_ALIAS")
    target = Path(a.destination)
    require_storage(target, a.storage_mount)
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--checksum", "--timeout=60", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"]
    for exclude in ("/code", "/data", "/hs_code_final.tar.gz", "/expert_package_worker.py", "/ops/LOCK"):
        cmd.extend(["--exclude", exclude])
    cmd.extend([a.host + ":" + a.remote_root.rstrip("/") + "/", str(target) + "/"])
    subprocess.run(cmd, check=True, timeout=180)
    state = validate_mirror(target)
    atomic(target / "MIRROR_STATUS.json", state)
    if state["closed"]:
        atomic(target / "MIRROR_CLOSE.json", state)
        if a.timer_unit:
            subprocess.run(["systemctl", "--user", "disable", "--now", a.timer_unit], check=True)
    print(json.dumps({"verified_jobs": state["count"], "closed": state["closed"]}))


if __name__ == "__main__":
    main()
