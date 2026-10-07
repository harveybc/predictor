#!/usr/bin/env python3
"""Run bounded Phase-4 jobs and report small, durable results to the coordinator.

One invocation claims at most ``--max-tasks`` tasks (default 1). Before every claim the
host is checked (available memory against the cap plus the desktop reserve, the physical
GPU UUID and its temperature for TRAINED_ENCODER slots, a durable output root). The
runner executes under ``crispdm-run`` with the task JSON on stdin and must print one
JSON terminal on stdout; that terminal is persisted atomically on this host FIRST and
delivered to the coordinator's controller over SSH afterwards. A delivery the controller
rejects is recorded as FAILED with its reason: exit code 0 never means COMPLETE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

UUID_RE = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DESKTOP_RESERVE_BYTES = 3 * 1024 ** 3
UNITS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


class Unhealthy(RuntimeError):
    """The host is not fit for one more job right now; nothing was claimed."""


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def cap_bytes(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)(?:i?B)?\s*", str(text), re.IGNORECASE)
    if not match:
        raise ValueError(f"INVALID_CAP: {text!r}")
    return int(match.group(1)) * UNITS[match.group(2).upper()]


def read_meminfo(path="/proc/meminfo") -> dict:
    values = {}
    with open(path) as stream:
        for line in stream:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    return values


def nvidia_query(uuid: str):
    """Return (visible, temperature_c, error) for one physical UUID, reading only."""
    listing = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, timeout=30, check=False)
    visible = f"(UUID: {uuid})" in listing.stdout
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,temperature.gpu", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, timeout=30, check=False)
    temperature = None
    for row in query.stdout.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 2 and parts[0] == uuid:
            try:
                temperature = int(parts[1])
            except ValueError:
                temperature = None
    error = "\n".join(x for x in (listing.stderr, query.stderr) if x).strip()[:300]
    return visible, temperature, error


def host_health(args, meminfo=None, gpu=None) -> dict:
    """Measure the host and raise Unhealthy when it must not take one more job."""
    meminfo = read_meminfo() if meminfo is None else meminfo
    cap = cap_bytes(args.cap)
    available = int(meminfo.get("MemAvailable", 0))
    needed = cap + DESKTOP_RESERVE_BYTES
    report = {"schema": "fs4.host_health.v1", "arm": args.arm, "cap_bytes": cap,
              "mem_available_bytes": available, "required_bytes": needed,
              "desktop_reserve_bytes": DESKTOP_RESERVE_BYTES, "gpu_uuid": args.gpu_uuid,
              "output_root": str(args.output_root), "measured_at": time.time()}
    root = Path(args.output_root).resolve()
    if str(root) == "/tmp" or str(root).startswith("/tmp/") or str(root).startswith("/dev/shm"):
        raise Unhealthy("OUTPUT_ROOT_NOT_DURABLE")
    if available < needed:
        raise Unhealthy(f"MEM_AVAILABLE_BELOW_CAP_PLUS_RESERVE: {available} < {needed}")
    if args.gpu_uuid:
        if not UUID_RE.match(args.gpu_uuid):
            raise Unhealthy("GPU_UUID_NOT_PHYSICAL_FORMAT")
        visible, temperature, error = nvidia_query(args.gpu_uuid) if gpu is None else gpu
        report.update({"gpu_visible": visible, "gpu_temperature_c": temperature, "nvidia_smi_stderr": error})
        if not visible:
            raise Unhealthy(f"GPU_UUID_NOT_VISIBLE: {args.gpu_uuid}")
        if temperature is None:
            raise Unhealthy("GPU_TEMPERATURE_UNREADABLE")
        if temperature > args.max_gpu_temp:
            raise Unhealthy(f"GPU_TOO_HOT: {temperature} > {args.max_gpu_temp}")
    return report


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


def deliver(args, task_id: str, terminal: Path, result: dict):
    """Hand the retained terminal to the coordinator; a rejection becomes FAILED, never COMPLETE."""
    try:
        return controller(args, "complete", input_value=result)
    except RuntimeError as exc:
        reason = f"DELIVERY_REJECTED: {exc}"[:1000]
        rejected = terminal.with_name(f"{terminal.name}.rejected.{int(time.time())}")
        os.replace(terminal, rejected)
        controller(args, "fail", ["--task-id", task_id, "--reason", reason])
        return {"task_id": task_id, "failed": True, "reason": reason, "rejected_terminal": str(rejected)}


def child_env(args, task) -> dict:
    env = os.environ.copy()
    env["FS4_TASK_ID"] = task["task_id"]
    env["FS4_ARM"] = task["arm"]
    if args.gpu_uuid:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        env["FS4_EXPECTED_GPU_UUID"] = args.gpu_uuid
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    elif task["arm"] == "TRAINED_ENCODER":
        raise RuntimeError("TRAINED_ENCODER_REQUIRES_PHYSICAL_GPU_UUID")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["FS4_EXPECTED_GPU_UUID"] = ""
    return env


def run_one(args, task):
    task_id = task["task_id"]
    terminal = Path(args.output_root) / f"{task_id}.json"
    if terminal.exists():
        existing = json.loads(terminal.read_text())
        if existing.get("task_id") != task_id:
            raise RuntimeError("LOCAL_TERMINAL_IDENTITY_MISMATCH")
        return deliver(args, task_id, terminal, existing)
    env = child_env(args, task)
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
        reason = f"runner rc={process.returncode}: {error_tail}"[:1000]
        controller(args, "fail", ["--task-id", task_id, "--reason", reason])
        return {"task_id": task_id, "failed": True, "reason": reason}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        controller(args, "fail", ["--task-id", task_id, "--reason", "runner did not emit one JSON result"])
        return {"task_id": task_id, "failed": True, "reason": "INVALID_RUNNER_RESULT"}
    if result.get("task_id") != task_id:
        controller(args, "fail", ["--task-id", task_id, "--reason", "RUNNER_RETURNED_WRONG_TASK"])
        return {"task_id": task_id, "failed": True, "reason": "RUNNER_RETURNED_WRONG_TASK"}
    if result.get("status") != "COMPLETE":
        reason = f"RUNNER_STATUS_{result.get('status')}: {result.get('reason', '')}"[:1000]
        atomic_json(terminal.with_name(f"{terminal.name}.refusal"), result)
        controller(args, "fail", ["--task-id", task_id, "--reason", reason])
        return {"task_id": task_id, "failed": True, "reason": reason}
    atomic_json(terminal, result)
    return deliver(args, task_id, terminal, result)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--max-gpu-temp", type=int, default=75, help="Do not start a job above this GPU temperature")
    parser.add_argument("--status", action="store_true", help="Read coordinator progress without claiming work")
    parser.add_argument("--health", action="store_true", help="Print the host health report and exit (no claim)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
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
    if args.gpu_uuid and not UUID_RE.match(args.gpu_uuid):
        parser.error("--gpu-uuid must be a physical GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx identifier")
    if args.max_tasks < 1 or args.timeout < 1:
        parser.error("max-tasks and timeout must be positive")
    if args.health:
        try:
            print(json.dumps({"healthy": True, **host_health(args)}, sort_keys=True))
        except Unhealthy as exc:
            print(json.dumps({"healthy": False, "reason": str(exc)}, sort_keys=True))
            raise SystemExit(3)
        return
    for _ in range(args.max_tasks):
        try:
            host_health(args)
        except Unhealthy as exc:
            print(json.dumps({"skipped": True, "reason": str(exc)}, sort_keys=True), flush=True)
            break
        filters = (["--task-id", args.task_id] if args.task_id else []) + (["--arm", args.arm] if args.arm else [])
        task = controller(args, "claim", filters)
        if task is None:
            print(json.dumps({"idle": True, "arm": args.arm}, sort_keys=True), flush=True)
            break
        print(json.dumps(run_one(args, task), sort_keys=True), flush=True)
        if args.task_id:
            break


if __name__ == "__main__":
    main()
