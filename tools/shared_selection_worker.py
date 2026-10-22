#!/usr/bin/env python3
"""Operations-only worker for the assigned half of Satoshi's selection queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def verify_pins(cfg):
    for path, expected in cfg["pins"].items():
        if sha256(path) != expected:
            raise ValueError(f"IDENTITY_CHANGED:{path}")


def base_marker_ready(log, shard):
    events = [line.split()[-1] for line in log.splitlines() if f" base shard {shard} " in line]
    return bool(events and events[-1] == "DONE" and "start" in events)


def is_science_process(argv, root):
    return (len(argv) > 2 and Path(argv[0]).name.startswith("python")
            and argv[1].endswith("pool_selection_v2.py") and argv[2] == "run"
            and "--out" in argv and argv.index("--out") + 1 < len(argv)
            and argv[argv.index("--out") + 1] == str(Path(root) / "out_full"))


def science_processes(root):
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if is_science_process(argv, root):
                found.append(int(proc.name))
        except (OSError, UnicodeError, IndexError):
            continue
    return found


def read_ledger(path):
    rows, unique = {}, True
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                unique &= row["key"] not in rows
                rows[row["key"]] = row
    return rows, unique


def successful(rows, keys):
    return all(k in rows and "error" not in rows[k]
               and isinstance(rows[k].get("val_loss"), (float, int))
               and math.isfinite(rows[k]["val_loss"]) for k in keys)


def stage_keys(units, stage, shard=None):
    a, b = map(int, shard.split("/")) if shard else (0, 1)
    return [u[0] for i, u in enumerate(units) if u[1] == stage and i % b == a]


def ops_dir(cfg):
    return Path(cfg["root"]) / "musashi_ops"


def barrier(cfg, stage):
    root = Path(cfg["root"])
    if not base_marker_ready((root / "LOG").read_text(), "0/2"):
        return "WAIT_BASE", {}
    rows, unique = read_ledger(root / "out_full/LEDGER.jsonl")
    if not successful(rows, cfg["base_keys"]):
        return "WAIT_BASE_MERGE", rows
    if not unique:
        return "WAIT_UNIQUE_MERGE", rows
    if stage == "rescue":
        manifest = json.loads((ops_dir(cfg) / "contrast_KEYS.json").read_text())
        if not successful(rows, manifest["all_keys"]):
            return "WAIT_CONTRAST_MERGE", rows
    return "READY", rows


def science_command(cfg, stage):
    root = Path(cfg["root"])
    return [cfg["python"], "tools/pool_selection_v2.py", "run", "--stage", stage,
            "--shard", "0/2", "--admitted", str(root / "data/admitted"),
            "--csv-dir", str(root / "data/sel15"),
            "--store", str(root / "data/eurusd_h1_layer1_pool_r4"),
            "--store", str(root / "data/eurusd_h1_layer2_pool"),
            "--catalog", str(root / "data/eurusd_h1_catalog/FEATURE_CATALOG.csv"),
            "--out", str(root / "out_full")]


def make_keys(cfg, stage):
    verify_pins(cfg)
    state, _ = barrier(cfg, stage)
    if state != "READY":
        raise ValueError(state)
    root = Path(cfg["root"])
    sys.path.insert(0, str(root / "code"))
    from tools.pool_selection_v2 import Engine
    args = SimpleNamespace(admitted=str(root / "data/admitted"), csv_dir=str(root / "data/sel15"),
                           store=[str(root / "data/eurusd_h1_layer1_pool_r4"), str(root / "data/eurusd_h1_layer2_pool")],
                           catalog=str(root / "data/eurusd_h1_catalog/FEATURE_CATALOG.csv"),
                           out=str(root / "out_full"), fit_end="2024-01-01", stage=stage, shard="0/2")
    engine = Engine(args)
    units = engine.arms()
    manifest = {"stage": stage, "all_keys": stage_keys(units, stage),
                "own_keys": stage_keys(units, stage, "0/2"), "columns": len(engine.cols),
                "families": len(engine.families), "informative_tasks": engine.informative_tasks(),
                "effective_store_sha256": engine.pins,
                "source_ledger_sha256": sha256(root / "out_full/LEDGER.jsonl")}
    atomic_json(ops_dir(cfg) / f"{stage}_KEYS.json", manifest)


def mark(cfg, stage, value):
    with (Path(cfg["root"]) / "LOG_full").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {stage} shard 0/2 {value}\n")
        f.flush()
        os.fsync(f.fileno())


def run_stage(cfg, stage):
    verify_pins(cfg)
    manifest = json.loads((ops_dir(cfg) / f"{stage}_KEYS.json").read_text())
    path = ops_dir(cfg) / f"{stage}_RUN.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    attempt = old.get("attempt", 0) + 1
    attempt_path = ops_dir(cfg) / f"{stage}_ATTEMPT_{attempt}.json"
    record = {"attempt": attempt, "state": "RUNNING", "started": time.time(),
              "command": science_command(cfg, stage), "keys_sha256": sha256(ops_dir(cfg) / f"{stage}_KEYS.json")}
    atomic_json(attempt_path, record)
    atomic_json(path, record)
    mark(cfg, stage, "start")
    with (Path(cfg["root"]) / "psel_full.log").open("a") as log:
        result = subprocess.run(science_command(cfg, stage), cwd=Path(cfg["root"]) / "code",
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    verify_pins(cfg)
    rows, _ = read_ledger(Path(cfg["root"]) / "out_full/LEDGER.jsonl")
    ok = result.returncode == 0 and successful(rows, manifest["own_keys"])
    record.update({"state": "DONE" if ok else "RETRY_PENDING",
                   "returncode": result.returncode, "finished": time.time(),
                   "missing_or_failed": [k for k in manifest["own_keys"] if not successful(rows, [k])]})
    atomic_json(attempt_path, record)
    atomic_json(path, record)
    # Only a complete stage emits DONE: the original shell wrapper emitted it on partial failure.
    if ok:
        mark(cfg, stage, "DONE")
    return 0 if ok else 1


def unit_state(unit):
    return subprocess.run(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
                          capture_output=True, text=True, check=True).stdout.strip()


def set_status(cfg, state, **extra):
    atomic_json(ops_dir(cfg) / "STATUS.json", {"state": state, "updated": time.time(), **extra})


def tick(cfg, config_path):
    # The timer runs a light process; only the admitted probe/worker imports scientific libraries.
    active = science_processes(cfg["root"])
    if active:
        set_status(cfg, "WAIT_ACTIVE_SELECTION", pids=active)
        return
    for stage in ("contrast", "rescue"):
        manifest_path = ops_dir(cfg) / f"{stage}_KEYS.json"
        run_path = ops_dir(cfg) / f"{stage}_RUN.json"
        unit = f"musashi-psel-{stage}-20261022.service"
        if unit_state(unit) in ("active", "activating", "deactivating"):
            set_status(cfg, "RUNNING_OR_WAIT_RESOURCES", stage=stage)
            return
        if not manifest_path.exists():
            state, _ = barrier(cfg, stage)
            if state != "READY":
                set_status(cfg, state, stage=stage)
                return
            verify_pins(cfg)
            result = subprocess.run([cfg["admission"], "-q", "-m", "3G", "-t", "180s", "-n", "psel-keys",
                                     "--", cfg["python"], str(Path(__file__).resolve()), "keys",
                                     "--config", config_path, "--stage", stage],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            if result.returncode or not manifest_path.exists():
                set_status(cfg, "WAIT_KEY_PROBE", stage=stage, returncode=result.returncode)
                return
        manifest = json.loads(manifest_path.read_text())
        rows, _ = read_ledger(Path(cfg["root"]) / "out_full/LEDGER.jsonl")
        record = json.loads(run_path.read_text()) if run_path.exists() else {}
        if successful(rows, manifest["own_keys"]):
            if record.get("state") != "DONE":
                atomic_json(run_path, {**record, "state": "DONE", "adopted_verified_ledger": True})
                mark(cfg, stage, "DONE")
            continue
        if record.get("attempt", 0) >= 3:
            set_status(cfg, "NEEDS_CODE_REPAIR", stage=stage, failures=record.get("missing_or_failed", []))
            return
        if time.time() - record.get("finished", 0) < 120:
            set_status(cfg, "RETRY_COOLDOWN", stage=stage)
            return
        verify_pins(cfg)
        subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
        subprocess.run(["systemd-run", "--user", "--collect", f"--unit={unit.removesuffix('.service')}",
                        "--property=MemoryAccounting=yes", "--property=TimeoutStartSec=87000",
                        "--setenv=CUDA_VISIBLE_DEVICES=", cfg["admission"], "-q", "-m", "3G", "-t", "86400s",
                        "-n", f"psel-{stage}", "--", cfg["python"], str(Path(__file__).resolve()),
                        "run-stage", "--config", config_path, "--stage", stage], check=True)
        set_status(cfg, "DISPATCHED", stage=stage, assigned=len(manifest["own_keys"]))
        return
    set_status(cfg, "OWN_SHARD_COMPLETE_AGGREGATION_OWNED_BY_PEER")
    subprocess.run(["systemctl", "--user", "disable", "--now", "musashi-psel-20261022.timer"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("tick", "keys", "run-stage"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("contrast", "rescue"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    ops_dir(cfg).mkdir(parents=True, exist_ok=True)
    if args.cmd == "keys":
        make_keys(cfg, args.stage)
    elif args.cmd == "run-stage":
        return run_stage(cfg, args.stage)
    else:
        with (ops_dir(cfg) / "tick.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            try:
                tick(cfg, args.config)
            except Exception as exc:
                set_status(cfg, "OPERATIONS_ERROR", error=f"{type(exc).__name__}:{exc}")
                raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
