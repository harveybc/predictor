#!/usr/bin/env python3
"""Run bounded Phase-4 jobs and report small, durable results to the coordinator."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def controller(args, action: str, extra: list[str] | None = None, input_value=None):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               args.coordinator, args.python, args.controller, "--db", args.db,
               action, "--owner", args.owner]
    command.extend(extra or [])
    result = subprocess.run(command, input=json.dumps(input_value) if input_value is not None else None,
                            text=True, capture_output=True, timeout=90, check=False)
    if result.returncode:
        raise RuntimeError(f"controller {action} rc={result.returncode}: {result.stderr[:500]}")
    return json.loads(result.stdout)


def run_one(args, task):
    task_id = task["task_id"]
    terminal = Path(args.output_root) / f"{task_id}.json"
    if terminal.exists():
        existing = json.loads(terminal.read_text())
        if existing.get("task_id") != task_id:
            raise RuntimeError("LOCAL_TERMINAL_IDENTITY_MISMATCH")
        return controller(args, "complete", input_value=existing)
    stop = threading.Event()
    errors = []

    def pulse():
        while not stop.wait(120):
            try:
                controller(args, "heartbeat", ["--task-id", task_id])
            except Exception as exc:
                errors.append(str(exc))
                stop.set()

    heart = threading.Thread(target=pulse, daemon=True)
    heart.start()
    command = [args.crispdm_run, "-q", "-m", args.cap, "-t", args.wall,
               "-n", f"fs4-{task_id[:12]}", "--", args.runner]
    env = os.environ.copy()
    if args.gpu_uuid:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    elif task["arm"] == "TRAINED_ENCODER":
        raise RuntimeError("TRAINED_ENCODER_REQUIRES_PHYSICAL_GPU_UUID")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with tempfile.TemporaryFile(mode="w+t") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True,
                                       stdout=subprocess.PIPE, stderr=log, env=env,
                                       start_new_session=True)
            started = time.monotonic()
            payload = json.dumps(task)
            while True:
                if errors or time.monotonic() - started >= args.timeout:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                    raise RuntimeError("LEASE_HEARTBEAT_FAILED" if errors else "WORKER_TIMEOUT")
                try:
                    stdout, _ = process.communicate(input=payload, timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    payload = None
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 1000))
            error_tail = log.read()[-300:]
    finally:
        stop.set()
        heart.join(timeout=2)
    if errors:
        raise RuntimeError(f"LEASE_HEARTBEAT_FAILED: {errors[0]}")
    if process.returncode:
        controller(args, "fail", ["--task-id", task_id, "--reason", f"runner rc={process.returncode}: {error_tail}"])
        return {"task_id": task_id, "failed": True}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        controller(args, "fail", ["--task-id", task_id, "--reason", "runner did not emit one JSON result"])
        raise RuntimeError("INVALID_RUNNER_RESULT") from exc
    if result.get("task_id") != task_id:
        raise RuntimeError("RUNNER_RETURNED_WRONG_TASK")
    atomic_json(terminal, result)
    return controller(args, "complete", input_value=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--db", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--crispdm-run", default=str(Path.home() / ".local/bin/crispdm-run"))
    parser.add_argument("--cap", required=True)
    parser.add_argument("--wall", default="2h")
    parser.add_argument("--timeout", type=int, default=7500)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--task-id", help="Run one selected pilot task")
    parser.add_argument("--arm", choices=("RAW", "RANDOM_ENCODER", "TRAINED_ENCODER"))
    parser.add_argument("--gpu-uuid", help="Physical CUDA UUID for this worker")
    parser.add_argument("--status", action="store_true", help="Read coordinator progress without claiming work")
    args = parser.parse_args()
    if args.status:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   args.coordinator, args.python, args.controller, "--db", args.db, "status"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
        if result.returncode:
            raise SystemExit(result.stderr)
        print(result.stdout.strip())
        return
    if args.arm is None:
        parser.error("--arm is required when running work")
    if args.arm == "TRAINED_ENCODER" and not args.gpu_uuid:
        parser.error("TRAINED_ENCODER requires --gpu-uuid")
    if args.max_tasks < 1 or args.timeout < 1:
        parser.error("max-tasks and timeout must be positive")
    for _ in range(args.max_tasks):
        filters = (["--task-id", args.task_id] if args.task_id else []) + (["--arm", args.arm] if args.arm else [])
        task = controller(args, "claim", filters)
        if task is None:
            break
        print(json.dumps(run_one(args, task), sort_keys=True), flush=True)
        if args.task_id:
            break


if __name__ == "__main__":
    main()
