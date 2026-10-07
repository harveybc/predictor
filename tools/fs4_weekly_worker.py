#!/usr/bin/env python3
"""Durable worker for the phase-4 WEEKLY queue: extends tools/fs4_worker.py, it does not replace it.

One invocation = at most ``--max-tasks`` tasks (timer slots use 1): host health check, one claim from
``tools/fs4_weekly_campaign.py`` on the coordinator over SSH, one ``fs4_weekly_wrapper.py run-task`` under
``crispdm-run`` with the measured cap, the terminal persisted locally FIRST, then delivery. A rejected delivery
is reported as FAILED and the terminal is set aside; it is never marked COMPLETE by exit code alone. The slot
serves one population and one input mode (RAW on CPU for stage 1; encoder modes for stage 2). A GPU is used only
when ``--gpu-uuid`` is given, in which case the physical UUID is checked by the host health step; otherwise the
child sees no GPU at all (no CPU/GPU fallback decision is left to TensorFlow).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs4_worker as BASE  # noqa: E402  (health, controller, deliver, atomic_json are reused as they are)

MODES = ("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER")


def wrapper_command(args, task_file: Path) -> list[str]:
    cmd = [str(args.python_local), str(args.wrapper), "run-task", "--population", args.population,
           "--train-features", *args.train_features, "--train-targets", args.train_targets, "--bar-hours", str(args.bar_hours),
           "--task-file", str(task_file)]
    if args.val_features:
        cmd += ["--val-features", *args.val_features, "--val-targets", args.val_targets]
    if args.input_mode != "RAW":
        cmd += ["--runner-results", args.runner_results, "--extractor-code", args.extractor_code]
    if args.test_freeze:
        cmd += ["--test-freeze", args.test_freeze]
    return cmd


def child_env(args) -> dict:
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu_uuid:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def run_one(args, task: dict) -> dict:
    task_id = task["task_id"]
    terminal = Path(args.output_root) / f"{task_id}.json"
    if terminal.exists():                       # restart: adopt the retained terminal, never retrain (BW16)
        existing = json.loads(terminal.read_text())
        if existing.get("task_id") != task_id:
            raise RuntimeError("LOCAL_TERMINAL_IDENTITY_MISMATCH")
        return BASE.deliver(args, task_id, terminal, existing)
    claim_dir = Path(args.output_root) / "claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    claim_file = claim_dir / f"{task_id}.json"
    BASE.atomic_json(claim_file, task) if not claim_file.exists() else None
    stop = threading.Event()
    errors: list[str] = []

    def pulse():
        while not stop.wait(args.heartbeat):
            try:
                BASE.controller(args, "heartbeat", ["--task-id", task_id])
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                stop.set()

    heart = threading.Thread(target=pulse, daemon=True)
    heart.start()
    command = [args.crispdm_run, "-q", "-m", args.cap, "-t", args.wall, "-n", f"fs4w-{task_id[:12]}", "--", *wrapper_command(args, claim_file)]
    try:
        with tempfile.TemporaryFile(mode="w+t") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, text=True, stdout=subprocess.PIPE, stderr=log,
                                       env=child_env(args), start_new_session=True)
            started = time.monotonic()
            while True:
                if errors or time.monotonic() - started >= args.timeout:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    raise RuntimeError("LEASE_HEARTBEAT_FAILED" if errors else "WORKER_TIMEOUT")
                try:
                    stdout, _ = process.communicate(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 1000))
            error_tail = log.read()[-300:]
    finally:
        stop.set()
        heart.join(timeout=2)
    if errors:
        raise RuntimeError(f"LEASE_HEARTBEAT_FAILED: {errors[0]}")
    if process.returncode:
        reason = f"wrapper rc={process.returncode}: {error_tail}"[:1000]
        BASE.controller(args, "fail", ["--task-id", task_id, "--reason", reason])
        return {"task_id": task_id, "failed": True, "reason": reason}
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        BASE.controller(args, "fail", ["--task-id", task_id, "--reason", "wrapper did not emit one JSON result"])
        return {"task_id": task_id, "failed": True, "reason": "INVALID_WRAPPER_RESULT"}
    if result.get("task_id") != task_id or result.get("status") != "COMPLETE":
        reason = f"WRAPPER_RESULT_MISMATCH: task {result.get('task_id')} status {result.get('status')}"[:1000]
        BASE.controller(args, "fail", ["--task-id", task_id, "--reason", reason])
        return {"task_id": task_id, "failed": True, "reason": reason}
    BASE.atomic_json(terminal, result)
    return BASE.deliver(args, task_id, terminal, result)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coordinator", required=True)
    p.add_argument("--controller", required=True, help="path of tools/fs4_weekly_campaign.py ON THE COORDINATOR")
    p.add_argument("--python", default="python3", help="python on the coordinator")
    p.add_argument("--python-local", default=sys.executable, help="python of this host (TensorFlow env)")
    p.add_argument("--wrapper", default=str(HERE / "fs4_weekly_wrapper.py"))
    p.add_argument("--db", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--crispdm-run", default=str(Path.home() / ".local/bin/crispdm-run"))
    p.add_argument("--cap", required=True)
    p.add_argument("--wall", default="3h")
    p.add_argument("--timeout", type=int, default=10500)
    p.add_argument("--heartbeat", type=int, default=120)
    p.add_argument("--output-root", required=True)
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--population", required=True)
    p.add_argument("--bar-hours", type=int, required=True)
    p.add_argument("--train-features", nargs="+", required=True)
    p.add_argument("--train-targets", required=True)
    p.add_argument("--val-features", nargs="+")
    p.add_argument("--val-targets")
    p.add_argument("--input-mode", choices=MODES, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--runner-results")
    p.add_argument("--extractor-code")
    p.add_argument("--test-freeze")
    p.add_argument("--gpu-uuid")
    p.add_argument("--max-gpu-temp", type=int, default=75)
    p.add_argument("--task-id")
    p.add_argument("--status", action="store_true")
    p.add_argument("--health", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.arm = args.input_mode                    # host_health reports it
    if args.status:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.coordinator, args.python, args.controller, "--db", args.db, "status"]
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=90, check=False)
        if r.returncode:
            raise SystemExit(r.stderr)
        print(r.stdout.strip())
        return
    if args.input_mode != "RAW" and not (args.runner_results and args.extractor_code):
        parser.error("encoder input modes need --runner-results and --extractor-code")
    if bool(args.val_features) != bool(args.val_targets):
        parser.error("--val-features and --val-targets go together")
    if args.split == "validation" and not args.val_features:
        parser.error("validation slots need --val-features/--val-targets")
    if args.gpu_uuid and not BASE.UUID_RE.match(args.gpu_uuid):
        parser.error("--gpu-uuid must be a physical GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx identifier")
    if args.health:
        try:
            print(json.dumps({"healthy": True, **BASE.host_health(args)}, sort_keys=True))
        except BASE.Unhealthy as exc:
            print(json.dumps({"healthy": False, "reason": str(exc)}, sort_keys=True))
            raise SystemExit(3)
        return
    for _ in range(args.max_tasks):
        try:
            BASE.host_health(args)
        except BASE.Unhealthy as exc:
            print(json.dumps({"skipped": True, "reason": str(exc)}, sort_keys=True), flush=True)
            break
        filters = ["--input-mode", args.input_mode, "--split", args.split, "--population", args.population] + (["--task-id", args.task_id] if args.task_id else [])
        task = BASE.controller(args, "claim", filters)
        if task is None:
            print(json.dumps({"idle": True, "input_mode": args.input_mode, "population": args.population}, sort_keys=True), flush=True)
            break
        print(json.dumps(run_one(args, task), sort_keys=True), flush=True)
        if args.task_id:
            break


if __name__ == "__main__":
    main()
