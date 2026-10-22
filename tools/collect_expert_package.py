#!/usr/bin/env python3
"""Copy assigned E3-M artifacts to a second disk and verify sealed outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
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


def source_location(host, root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("INVALID_SSH_ALIAS")
    if host == "local":
        if not Path(root).is_absolute():
            raise ValueError("LOCAL_SOURCE_MUST_BE_ABSOLUTE")
        return str(Path(root)) + "/"
    return host + ":" + root.rstrip("/") + "/"


def replicate(target, host, root, verifier, expected):
    destination = source_location(host, root)
    if host == "local" or not Path(root).is_absolute():
        raise ValueError("REPLICA_REQUIRES_REMOTE_ABSOLUTE_ROOT")
    subprocess.run(["rsync", "-a", "--checksum", "--timeout=60", "-e",
                    "ssh -o BatchMode=yes -o ConnectTimeout=10", str(target) + "/", destination],
                   check=True, timeout=180)
    code = ("import json,sys;sys.path.insert(0,sys.argv[1]);"
            "from collect_expert_package import validate_mirror;"
            "print(json.dumps(validate_mirror(sys.argv[2])))")
    remote = shlex.join(["python3", "-c", code, verifier, root])
    actual = json.loads(subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o",
                                               "ConnectTimeout=10", host, remote], text=True, timeout=90))
    if actual != expected:
        raise ValueError("REMOTE_REPLICA_MISMATCH")
    return {"verified": True, "count": actual["count"], "closed": actual["closed"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--storage-mount", required=True)
    p.add_argument("--timer-unit", default="")
    p.add_argument("--replica-host")
    p.add_argument("--replica-root")
    p.add_argument("--replica-verifier")
    a = p.parse_args()
    source = source_location(a.host, a.remote_root)
    target = Path(a.destination)
    require_storage(target, a.storage_mount)
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--checksum", "--timeout=60", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"]
    for exclude in ("/code", "/data", "/hs_code_final.tar.gz", "/expert_package_worker.py", "/ops/LOCK"):
        cmd.extend(["--exclude", exclude])
    if a.host == "local" and (target.resolve() == Path(a.remote_root).resolve()
                             or target.resolve().is_relative_to(Path(a.remote_root).resolve())):
        raise ValueError("LOCAL_MIRROR_MUST_BE_SEPARATE")
    cmd.extend([source, str(target) + "/"])
    subprocess.run(cmd, check=True, timeout=180)
    state = validate_mirror(target)
    if a.replica_host:
        if not a.replica_root or not a.replica_verifier:
            raise ValueError("REPLICA_CONFIG_INCOMPLETE")
        replica = replicate(target, a.replica_host, a.replica_root, a.replica_verifier, state)
        atomic(target / "REMOTE_REPLICA_STATUS.json", replica)
    atomic(target / "MIRROR_STATUS.json", state)
    if state["closed"]:
        atomic(target / "MIRROR_CLOSE.json", state)
        if a.timer_unit:
            subprocess.run(["systemctl", "--user", "disable", "--now", a.timer_unit], check=True)
    print(json.dumps({"verified_jobs": state["count"], "closed": state["closed"]}))


if __name__ == "__main__":
    main()
