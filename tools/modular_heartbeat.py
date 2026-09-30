"""Unbuffered heartbeat for live modular experimental children.

A daemon thread writes ``heartbeat.json`` atomically (write + fsync + rename) and
appends the same record to ``heartbeat.jsonl`` at most ``interval`` seconds
apart (default 30 s, hard maximum 60 s). Each record carries: stage, epoch and
update progress, last completed checkpoint (selected epoch/updates), resources
(whole-cgroup memory current/peak when readable, process RSS/HWM, CPU seconds)
and an ETA with its stated basis. Progress is fed by the evaluator's
``progress`` callback; the heartbeat never alters training.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

MAX_INTERVAL = 60.0


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_memory():
    """Whole-cgroup (v2) memory of this process's cgroup: current and peak bytes."""
    line = _read("/proc/self/cgroup") or ""
    rel = line.split("::", 1)[1] if "::" in line else None
    if not rel:
        return {}
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    out = {"cgroup": rel}
    for key, name in (("memory.current", "current_bytes"), ("memory.peak", "peak_bytes"),
                      ("memory.max", "max")):
        value = _read(base / key)
        if value is not None:
            out[name] = int(value) if value.isdigit() else value
    return out


def process_memory():
    out = {}
    for line in (_read("/proc/self/status") or "").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) * 1024
    times = os.times()
    out["cpu_seconds"] = times.user + times.system
    return out


class Heartbeat:
    def __init__(self, path, *, interval=30.0, identity=None):
        if not 0 < interval <= MAX_INTERVAL:
            raise ValueError("heartbeat interval must be in (0, 60] seconds")
        self.path = Path(path)
        self.log = self.path.with_suffix(".jsonl")
        self.interval = interval
        self.identity = identity or {}
        self.state = {"stage": "start"}
        self.checkpoint = None
        self.started = time.time()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="heartbeat")

    def update(self, **fields):
        with self._lock:
            self.state.update(fields)
            if fields.get("stage") == "validated":
                self.checkpoint = {"epoch": fields.get("epoch"), "updates": fields.get("updates"),
                                   "validation_loss": fields.get("validation_loss")}
            if fields.get("stage") in ("load", "build", "score", "save", "done", "failed"):
                self.write()

    def _eta(self, state):
        updates, bpe = state.get("updates"), state.get("batches_per_epoch")
        elapsed, max_epochs = state.get("elapsed_seconds"), state.get("max_epochs")
        if not updates or not bpe or not elapsed or not max_epochs:
            return {"seconds": None, "basis": "no completed update yet"}
        per_update = elapsed / updates
        remaining = max(0, max_epochs * bpe - updates)
        return {"seconds": remaining * per_update, "per_update_seconds": per_update,
                "basis": "upper bound: remaining updates to max_epochs at the observed mean "
                         "seconds/update; patience may stop earlier; excludes validation passes"}

    def record(self):
        with self._lock:
            state = dict(self.state)
            checkpoint = self.checkpoint
        return {"time": time.time(), "wall_seconds": time.time() - self.started, "pid": os.getpid(),
                **self.identity, "stage": state.get("stage"), "progress": state,
                "last_checkpoint": checkpoint,
                "resources": {"cgroup": cgroup_memory(), "process": process_memory(),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "eta": self._eta(state)}

    def write(self):
        record = self.record()
        text = json.dumps(record, sort_keys=True, default=str)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)
        with open(self.log, "a", buffering=1) as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.write()
            except OSError:
                pass

    def __enter__(self):
        self.write()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=5)
        self.update(stage="failed" if exc_type else "done",
                    error=None if exc is None else f"{exc_type.__name__}: {exc}")
        return False


def run_request(request_path, response_path, heartbeat_path, interval=30.0):
    """Evaluate one bridge request under a heartbeat; used by the DOIN worker."""
    from tools.modular_candidate_evaluator import evaluate_candidate

    request = json.loads(Path(request_path).read_text())
    identity = {"candidate_output": request["output_dir"]}
    with Heartbeat(heartbeat_path, interval=interval, identity=identity) as beat:
        result = evaluate_candidate(request["config"], request["train_path"], request["validation_path"],
                                    request["output_dir"], progress=beat.update)
    Path(response_path).write_text(json.dumps(result, allow_nan=False) + "\n")
    return result
