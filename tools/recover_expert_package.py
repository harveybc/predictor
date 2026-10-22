#!/usr/bin/env python3
"""Recover an interrupted native final JSON write, never an accepted result."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from expert_package_worker import atomic, identity, jobs, live_children, read, verify_pins


def recover(cfg):
    root = Path(cfg["root"])
    ops = root / "ops"
    with (ops / "LOCK").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        verify_pins(cfg)
        owner = read(ops / "IDENTITY.json")
        if owner and owner["config_sha256"] != identity(cfg):
            raise ValueError("CONFIG_IDENTITY_CHANGED")
        if live_children(root):
            return []
        states, restored = read(ops / "JOBS.json", {}), []
        for job in jobs():
            name = job["name"]
            rec = states.get(name, {})
            out = root / "out" / name
            if rec.get("state") != "RUNNING" or not (out / "E3M.json").exists():
                continue
            if (ops / (name + ".RESULT.json")).exists():
                continue
            try:
                read(out / "E3M.json")
            except json.JSONDecodeError:
                destination = root / "attempts" / name / ("interrupted-json-" + str(time.time_ns()))
                destination.parent.mkdir(parents=True, exist_ok=True)
                out.rename(destination)
                rec.update(state="RETRY_PENDING", cause="INTERRUPTED_FINAL_JSON_WRITE",
                           preserved_attempt=str(destination))
                states[name] = rec
                atomic(ops / "JOBS.json", states)
                atomic(ops / (name + ".BOOT_RECOVERY.json"), rec)
                restored.append(name)
        return restored


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    cfg = read(Path(a.root) / "ops/CONFIG.private.json")
    print(json.dumps({"recovered": recover(cfg)}))
