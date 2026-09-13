#!/usr/bin/env python3
"""C150 (order 2026-09-13): one OS process per dataset under hard limits, and a durable terminal per attempt.

Isolation. Each task is started with

    systemd-run --user --scope --quiet --unit=<unique> --slice=<slice> -p MemoryMax=<limit> -p MemorySwapMax=0 ...

(default slice crispdm-batch.slice, so an out-of-memory kill stays inside the
batch slice and never reaches a desktop scope). When systemd user scopes are
unavailable the fallback is RLIMIT_AS via setrlimit in the child
(limit_mechanism PRLIMIT_AS), recorded as such. The child also gets RLIMIT_CPU,
single-thread BLAS variables, its own session (the whole group is killed on a
wall-time overrun), a heartbeat file and a stop file.

Safety ratios (declared constants):

    memory_limit_bytes = floor(assigned_bytes * LIMIT_RATIO)        LIMIT_RATIO  = 0.90
    budget_bytes       = floor(memory_limit_bytes * BUDGET_RATIO)   BUDGET_RATIO = 0.80

so the hard RSS limit is strictly below the memory assigned to the task and
the planner budget is strictly below the limit; the 20% between budget and
limit absorbs allocator fragmentation and page cache the planner does not
model.

Terminal. The PARENT writes one df_fact_dataset_terminal row per attempt,
atomically (temporary file, fsync, rename, directory fsync), write-once, with
status in exactly TERMINAL_STATUSES:

* COMPLETED          child exited 0 and its output file re-hashes to its declared digest;
* FAILED             the child recorded an exception;
* INCONCLUSIVE       the child stopped gracefully at a stop file;
* REFUSED            contract or bytes refusal, or the task cannot fit its budget;
* RESOURCE_EXCEEDED  cgroup OOM kill (scope Result=oom-kill or memory.events oom_kill),
                     RLIMIT_AS MemoryError, CPU-time limit (SIGXCPU) or wall-time limit;
* UNCERTAIN          anything the parent cannot verify (missing or mismatching result/output,
                     an unexplained signal).

A sibling's failure never touches another task's files. The scope of a failed
task is left loaded only long enough to read its Result, then reset.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import time
from pathlib import Path

TERMINAL_STATUSES = ("COMPLETED", "FAILED", "INCONCLUSIVE", "REFUSED", "RESOURCE_EXCEEDED", "UNCERTAIN")
TERMINAL_KEYS = ("run_id", "host_role", "bank", "dataset_id", "contract_sha256", "code_sha256", "status", "reason",
                 "rows_written", "variables_profiled", "metrics_completed", "metrics_missing", "planned_peak_bytes",
                 "observed_peak_rss_bytes", "memory_limit_bytes", "limit_mechanism", "wall_seconds", "cpu_seconds",
                 "output_file", "output_sha256", "started_at", "ended_at")
HOST_ROLES = ("COORDINATOR", "WORKER_A", "WORKER_B")
LIMIT_RATIO = 0.90
BUDGET_RATIO = 0.80
DEFAULT_SLICE = "crispdm-batch.slice"
POLL_SECONDS = 0.2
CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"}


class RunnerRefusal(ValueError):
    pass


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def limits_for(assigned_bytes: int) -> dict:
    limit = int(assigned_bytes * LIMIT_RATIO)
    budget = int(limit * BUDGET_RATIO)
    if not (budget < limit < assigned_bytes):
        raise RunnerRefusal("safety ratios must give budget < limit < assigned")
    return {"assigned_bytes": int(assigned_bytes), "memory_limit_bytes": limit, "budget_bytes": budget,
            "limit_ratio": LIMIT_RATIO, "budget_ratio": BUDGET_RATIO}


def atomic_write_once(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise RunnerRefusal(f"{path.name} exists; terminals are write-once")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)          # fails if a concurrent writer created it: never overwrite
    os.unlink(tmp)
    dfd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def validate_terminal(row: dict) -> list[str]:
    p = []
    if not isinstance(row, dict) or set(row) != set(TERMINAL_KEYS):
        return [f"keys differ: expected {sorted(TERMINAL_KEYS)}"]
    if row["status"] not in TERMINAL_STATUSES:
        p.append(f"status {row['status']!r} not in {list(TERMINAL_STATUSES)}")
    if row["status"] != "COMPLETED" and not (isinstance(row["reason"], str) and row["reason"].strip()):
        p.append("a terminal that did not complete needs a reason")
    if row["host_role"] not in HOST_ROLES:
        p.append("host_role must be a logical role, never a host name")
    for c in ("rows_written", "variables_profiled", "metrics_completed", "metrics_missing", "planned_peak_bytes",
              "memory_limit_bytes"):
        if type(row[c]) is not int or row[c] < 0:
            p.append(f"{c}: expected a non-negative integer")
    if not (row["observed_peak_rss_bytes"] is None or type(row["observed_peak_rss_bytes"]) is int):
        p.append("observed_peak_rss_bytes: integer or null")
    for c in ("wall_seconds", "cpu_seconds"):
        if type(row[c]) not in (int, float) or row[c] < 0:
            p.append(f"{c}: expected a non-negative number")
    for c in ("contract_sha256", "code_sha256", "output_sha256"):
        v = row[c]
        if c == "code_sha256" and v is None:
            p.append("code_sha256 is required")
        if v is not None and not (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)):
            p.append(f"{c}: expected a sha256 hex digest or null")
    if row["output_sha256"] is not None and row["status"] != "COMPLETED":
        p.append("output_sha256 is only recorded for a verified COMPLETED output")
    if "/home/" in json.dumps(row):
        p.append("no absolute home path in a terminal")
    return p


def write_terminal(path: Path, row: dict) -> None:
    problems = validate_terminal(row)
    if problems:
        raise RunnerRefusal("; ".join(problems))
    atomic_write_once(path, json.dumps(row, indent=1, sort_keys=True, allow_nan=False) + "\n")


# --------------------------------------------------------------- mechanism
def detect_mechanism(slice_: str = DEFAULT_SLICE) -> str:
    """SYSTEMD_USER_SCOPE when `systemd-run --user --scope` works here, else PRLIMIT_AS."""
    if shutil.which("systemd-run") and shutil.which("systemctl"):
        try:
            r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--slice={slice_}",
                                "-p", "MemoryMax=64M", "-p", "MemorySwapMax=0", "true"],
                               capture_output=True, timeout=30)
            if r.returncode == 0:
                return "SYSTEMD_USER_SCOPE"
        except (OSError, subprocess.SubprocessError):
            pass
    return "PRLIMIT_AS"


def mechanism_label(mechanism: str, slice_: str) -> str:
    return f"SYSTEMD_USER_SCOPE slice={slice_} MemoryMax MemorySwapMax=0 RLIMIT_CPU" \
        if mechanism == "SYSTEMD_USER_SCOPE" else "PRLIMIT_AS RLIMIT_AS RLIMIT_CPU"


def _systemctl_show(unit: str, prop: str) -> str:
    try:
        r = subprocess.run(["systemctl", "--user", "show", unit, "-p", prop, "--value"], capture_output=True,
                           text=True, timeout=15)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _cgroup_dir(pid: int):
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[0]
        return Path("/sys/fs/cgroup" + line.split("::", 1)[1])
    except (OSError, IndexError):
        return None


# ------------------------------------------------------------------- task
class Task:
    """One supervised child. `argv` is the child command (without any wrapper)."""

    def __init__(self, *, argv, name, attempt_dir: Path, assigned_bytes: int, wall_seconds: float,
                 cpu_seconds: float, slice_: str = DEFAULT_SLICE, mechanism: str = "SYSTEMD_USER_SCOPE",
                 extra_env=None):
        self.argv, self.name, self.dir = list(argv), name, Path(attempt_dir)
        self.lim = limits_for(assigned_bytes)
        self.wall, self.cpu, self.slice, self.mechanism = float(wall_seconds), int(cpu_seconds), slice_, mechanism
        self.unit = "crispdm-df-" + re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80] + f"-{os.getpid()}-{int(time.time() * 1000)}"
        self.extra_env = dict(extra_env or {})
        self.proc = None
        self.outcome = None
        self.cg = None
        self.polled_peak = None
        self.polled_oom_kill = 0

    def start(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(CHILD_ENV)
        env.update(self.extra_env)
        cpu = self.cpu

        if self.mechanism == "SYSTEMD_USER_SCOPE":
            cmd = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={self.unit}", f"--slice={self.slice}",
                   "-p", f"MemoryMax={self.lim['memory_limit_bytes']}", "-p", "MemorySwapMax=0", *self.argv]
            as_limit = None
        else:
            cmd = list(self.argv)
            as_limit = self.lim["memory_limit_bytes"]

        def pre():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 10))
            if as_limit is not None:
                resource.setrlimit(resource.RLIMIT_AS, (as_limit, as_limit))

        self.log = open(self.dir / "child.log", "ab")
        self.t0, self.started_at = time.time(), now_iso()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT, preexec_fn=pre,
                                     start_new_session=True, cwd=str(self.dir))
        return self

    def _poll_cgroup(self):
        if self.cg is None:
            cg = _cgroup_dir(self.proc.pid)
            if cg is not None and cg.name.endswith(".scope") and self.unit in cg.name:
                self.cg = cg
        if self.cg is not None:
            try:
                self.polled_peak = int((self.cg / "memory.peak").read_text().split()[0])
                ev = dict(ln.split() for ln in (self.cg / "memory.events").read_text().splitlines())
                self.polled_oom_kill = max(self.polled_oom_kill, int(ev.get("oom_kill", 0)))
            except (OSError, ValueError):
                pass

    def poll(self) -> bool:
        """Non-blocking; True once the task has an outcome."""
        if self.outcome is not None:
            return True
        self._poll_cgroup()
        timed_out = False
        pid, status, ru = os.wait4(self.proc.pid, os.WNOHANG)
        if pid == 0:
            if time.time() - self.t0 <= self.wall:
                return False
            timed_out = True
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid, status, ru = os.wait4(self.proc.pid, 0)
        self.proc.returncode = os.waitstatus_to_exitcode(status)
        self.log.close()
        result = None
        if self.mechanism == "SYSTEMD_USER_SCOPE":
            scope_result = _systemctl_show(f"{self.unit}.scope", "Result")
            if scope_result and scope_result != "success":
                subprocess.run(["systemctl", "--user", "reset-failed", f"{self.unit}.scope"], capture_output=True,
                               timeout=15)
        else:
            scope_result = None
        try:
            result = json.loads((self.dir / "result.json").read_text())
        except (OSError, ValueError):
            result = None
        self.outcome = {"returncode": self.proc.returncode, "timed_out": timed_out, "scope_result": scope_result,
                        "polled_oom_kill": self.polled_oom_kill, "cgroup_memory_peak": self.polled_peak,
                        "child_maxrss_bytes": int(ru.ru_maxrss) * 1024,
                        "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 3),
                        "wall_seconds": round(time.time() - self.t0, 3), "result": result,
                        "started_at": self.started_at, "ended_at": now_iso()}
        return True

    def wait(self):
        while not self.poll():
            time.sleep(POLL_SECONDS)
        return self.outcome


def heartbeat_tail(attempt_dir: Path) -> str:
    try:
        hb = json.loads((Path(attempt_dir) / "heartbeat.json").read_text())
        return f"last heartbeat {hb.get('module')}.{hb.get('metric')} variable={hb.get('variable')} " \
               f"rows={hb.get('rows_written')} rss={hb.get('rss_bytes')}"
    except (OSError, ValueError):
        return "no heartbeat"


def classify(outcome: dict, attempt_dir: Path, mutation: bool = False) -> tuple[str, str, dict]:
    """-> (status, reason, verified) from what the parent can observe and re-verify."""
    rc, res = outcome["returncode"], outcome["result"]
    tail = heartbeat_tail(attempt_dir)
    tag = "; MUTATION_BYPASS_PREFLIGHT" if mutation else ""
    verified = {"output_sha256": None, "rows_written": 0}
    if outcome["timed_out"]:
        return "RESOURCE_EXCEEDED", f"WALL_TIME_LIMIT; {tail}{tag}", verified
    if outcome["scope_result"] == "oom-kill" or outcome["polled_oom_kill"] > 0:
        return "RESOURCE_EXCEEDED", f"CGROUP_OOM_KILL scope_result={outcome['scope_result']}; {tail}{tag}", verified
    if rc == -signal.SIGXCPU or outcome["scope_result"] == "timeout":
        return "RESOURCE_EXCEEDED", f"CPU_TIME_LIMIT; {tail}{tag}", verified
    if res and res.get("exception_type") == "MemoryError":
        return "RESOURCE_EXCEEDED", f"RLIMIT_AS_MEMORY_ERROR; {tail}{tag}", verified
    if rc < 0:
        return "UNCERTAIN", f"KILLED_BY_SIGNAL {signal.Signals(-rc).name} without a recorded limit; {tail}{tag}", verified
    if res is None:
        return "UNCERTAIN", f"CHILD_RESULT_MISSING exit={rc}; {tail}{tag}", verified
    status = res.get("status")
    if status == "COMPLETED" and rc == 0:
        out = Path(attempt_dir) / str(res.get("output_file", ""))
        if not res.get("output_file") or not out.is_file():
            return "UNCERTAIN", "OUTPUT_FILE_MISSING" + tag, verified
        digest = sha_file(out)
        with open(out, "rb") as f:
            lines = sum(1 for _ in f)
        if digest != res.get("output_sha256") or lines != res.get("rows_written"):
            return "UNCERTAIN", "OUTPUT_DIGEST_OR_ROW_COUNT_MISMATCH" + tag, verified
        verified = {"output_sha256": digest, "rows_written": lines}
        return "COMPLETED", ("" if not mutation else "MUTATION_BYPASS_PREFLIGHT"), verified
    if status in ("FAILED", "INCONCLUSIVE", "REFUSED"):
        return status, (res.get("reason") or f"{status}_WITHOUT_REASON") + tag, verified
    return "UNCERTAIN", f"UNRECOGNIZED_CHILD_STATUS {status!r} exit={rc}{tag}", verified
