#!/usr/bin/env python3
"""C174 (order 2026-09-13): memory-aware job dispatcher across host roles; jobs survive a dispatcher crash.

Each job runs as a transient systemd USER SERVICE (not a scope tied to this process
or to an ssh session; lingering keeps the user manager alive), started on its role by

    <admission check> && exec systemd-run --user --quiet --unit=<unit> --slice=crispdm-batch.slice \
        -p MemoryMax=<request> -p MemoryHigh=<90%> -p MemorySwapMax=0 -p RuntimeMaxSec=<wall> \
        -p RemainAfterExit=yes -p Type=exec --working-directory="$HOME"/<checkout> \
        env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=<""|uuid> argv

through `bash -c` locally for the COORDINATOR and `ssh <alias> bash -c` for a worker.

* Unit name: crispdm-dispatch-<job id>-<first 16 hex of the job identity>; deterministic.
* No --collect (systemd 259 rejects `--collect=no`; omitting it is the no-collect default):
  a failed unit stays loaded with its Result until reset-failed. RemainAfterExit=yes keeps a
  successful unit loaded (active/exited) with ExecMainStatus and MemoryPeak until it is
  stopped. Verified on this host: an exited unit is not timed out by RuntimeMaxSec.
* Admission (same rule as crispdm-run, on the target host): refuse with exit 75 when the
  request exceeds MemAvailable - 3G or the slice MemoryMax -> REFUSED_AT_LAUNCH (requeued,
  at most max_launch_refusals times). A unit of that name already loaded -> exit 76 ->
  re-attached, never started twice.
* Polling (batched per role): systemctl --user show <units> -p Id -p LoadState -p ActiveState
  -p SubState -p Result -p ExecMainCode -p ExecMainStatus -p MemoryPeak -p MemoryCurrent.
  Running: ActiveState activating/active(not exited)/deactivating/reloading; MemoryCurrent
  feeds the reservation. Finished: active/exited, failed or inactive, classified by
  classify_unit:
      Result=oom-kill -> RESOURCE_EXCEEDED; Result=timeout -> RESOURCE_EXCEEDED;
      Result success/exit-code with ExecMainCode=1 (exited) -> COMPLETED (status 0) or FAILED;
      Result=signal or anything unrecognized/unreadable -> UNCERTAIN;
      LoadState=not-found -> LAUNCH_NOT_FOUND (never started, host restarted, or reset by
      someone else; requeued with the refusal counter).
  The peak is MemoryPeak. The terminal receipt is written first, then the unit is released:
  reset-failed when failed, stop when active/exited.
  A role that cannot be reached is simply polled again later (the unit keeps running).

Loop: poll; merge requeued jobs; unless the stop file exists, for each pending job re-read
the LIVE inventory when none is held (dropped after EVERY launch and when older than
max_inventory_age_seconds), place it (df_placement, workers first), and launch / split /
refuse / wait (WAIT_TIMEOUT -> UNPLACEABLE after max_wait_seconds with nothing running).

Receipts (root, write-once, no host names, no home paths):
    DISPATCH_MANIFEST.json, inventories/<digest>.json,
    receipts/<job>.attempt-N.launch.json   written BEFORE systemd-run: role, unit, GPU, decision,
                                           inventory digest, requested bytes, start script
    receipts/<job>.attempt-N.json          terminal: launch fields + status, reason, systemd state,
                                           exit code, MemoryPeak, wall seconds
    DISPATCH_PROGRESS.N.json | DISPATCH_RECEIPT.json

Resume by identity: a launch record without a terminal is looked up on its role: an active
unit is re-attached (never relaunched), a finished unit is classified from its stored Result
and released, a missing unit is LAUNCH_NOT_FOUND and requeued, an unreachable role is
re-attached and polled. Otherwise the latest terminal of the same identity decides:
COMPLETED, FAILED, RESOURCE_EXCEEDED, UNPLACEABLE and SPLIT are not run again (a SPLIT
parent's chunks are regenerated and resumed); UNCERTAIN only with --rerun-uncertain.

`--dry-run` prints the live inventory and the placement plan for a jobs file and starts nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import df_host_capacity as HC  # noqa: E402
import df_placement as PL  # noqa: E402

SCHEMA = "crispdm.data_foundation.dispatch_receipt.v2"
RESUME_SKIP = ("COMPLETED", "FAILED", "RESOURCE_EXCEEDED", "UNPLACEABLE", "SPLIT")
REQUEUE = ("REFUSED_AT_LAUNCH", "LAUNCH_NOT_FOUND")
SLICE = "crispdm-batch.slice"
UNIT_PREFIX = "crispdm-dispatch-"
ADMISSION_RESERVE_BYTES = 3 * PL.GIB          # crispdm-run's host reserve
EXIT_REFUSED = 75
EXIT_UNIT_LOADED = 76
POLL_PROPS = ("Id", "LoadState", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus",
              "MemoryPeak", "MemoryCurrent")
THREAD_ENV = ("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1")
DEFAULT_WORKER_CHECKOUT = "Documents/GitHub/.worktrees/predictor-c146"
WALL_RE = re.compile(r"^\d+(s|m|min|h|d)?$")
CUDA_RE = re.compile(r"^[A-Za-z0-9-]*$")
SIGNALS = {1: "SIGHUP", 2: "SIGINT", 6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(text: str) -> str:
    return text.replace(str(Path.home()), "~")


def identity(job: dict) -> str:
    keys = ("job_id", "argv", "cpu_bytes", "gpu_bytes", "cpus", "wall", "roles", "gpu_index", "split")
    doc = {k: job.get(k) for k in keys}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def safe(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)[:100]


ROLE_PLACEHOLDER = "{role}"


def bind_role(job: dict, role: str) -> dict:
    """A copy of the job whose argv has every {role} replaced by the placed role (e.g. an output root per role, or
    a --host-role label). The template job keeps its identity; only the launched command carries the role."""
    if role not in ("COORDINATOR", "WORKER_A", "WORKER_B"):
        raise ValueError(f"unknown role {role!r}")
    bound = dict(job)
    bound["argv"] = [str(a).replace(ROLE_PLACEHOLDER, role) for a in job["argv"]]
    return bound


def unit_name(job: dict) -> str:
    """Deterministic from the job identity (without the .service suffix)."""
    return UNIT_PREFIX + re.sub(r"[^A-Za-z0-9_.-]", "_", job["job_id"])[:48] + "-" + identity(job)[:16]


def write_once(path: Path, doc: dict) -> None:
    import df_isolated_runner as IR
    IR.atomic_write_once(path, redact(json.dumps(doc, indent=1, sort_keys=True, default=str)) + "\n")


# ------------------------------------------------------------------ systemd
def build_start_script(job: dict, decision: dict, checkout_rel: str) -> str:
    """The bash text run on the role: admission check, then exec systemd-run (no alias, no home path)."""
    req = int(decision["request_bytes"])
    unit = unit_name(job)
    wall = str(job.get("wall", "4h"))
    if not WALL_RE.match(wall):
        raise ValueError(f"wall {wall!r}")
    cuda = (decision.get("gpu") or {}).get("uuid") or ""
    if not CUDA_RE.match(cuda):
        raise ValueError("GPU uuid with unexpected characters")

    def q(a: str) -> str:
        return '"$HOME"/' + shlex.quote(a[2:]) if a.startswith("~/") else shlex.quote(a)

    run = ["exec systemd-run --user --quiet", f"--unit={unit}", f"--slice={SLICE}", f"-p MemoryMax={req}",
           f"-p MemoryHigh={req * 9 // 10}", "-p MemorySwapMax=0", f"-p RuntimeMaxSec={wall}",
           "-p RemainAfterExit=yes", "-p Type=exec", '--working-directory="$HOME"/' + shlex.quote(checkout_rel),
           "env -u PYTHONPATH", *THREAD_ENV, f"CUDA_VISIBLE_DEVICES={cuda}", *(q(str(a)) for a in job["argv"])]
    return "\n".join([
        "set -u",
        f"REQ={req}",
        "AVAIL=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))",
        f'if [ "$REQ" -gt $(( AVAIL - {ADMISSION_RESERVE_BYTES} )) ]; then '
        f'echo "REFUSED_AT_LAUNCH: request exceeds MemAvailable - 3G" >&2; exit {EXIT_REFUSED}; fi',
        f"SLICE_MAX=$(systemctl --user show {SLICE} -p MemoryMax --value)",
        f'if [ -n "$SLICE_MAX" ] && [ "$SLICE_MAX" != infinity ] && [ "$REQ" -gt "$SLICE_MAX" ]; then '
        f'echo "REFUSED_AT_LAUNCH: request exceeds the batch slice MemoryMax" >&2; exit {EXIT_REFUSED}; fi',
        f'if [ "$(systemctl --user show {unit}.service -p LoadState --value)" = loaded ]; then '
        f'echo "UNIT_ALREADY_LOADED" >&2; exit {EXIT_UNIT_LOADED}; fi',
        " ".join(run),
    ]) + "\n"


def parse_show(text: str) -> dict:
    """`systemctl show` of several units -> {Id: {prop: value}}."""
    out, cur = {}, {}
    for line in text.splitlines() + [""]:
        if not line.strip():
            if cur.get("Id"):
                out[cur["Id"]] = cur
            cur = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k] = v
    return out


def _int(v):
    return int(v) if isinstance(v, str) and v.isdigit() else None


def classify_unit(props) -> dict:
    """-> {"state": RUNNING|FINISHED|NOT_FOUND|UNREADABLE, "status", "reason", "exit_code",
    "memory_peak_bytes", "memory_current_bytes", "systemd"}."""
    out = {"state": "UNREADABLE", "status": "UNCERTAIN", "reason": "UNIT_PROPERTIES_UNREADABLE", "exit_code": None,
           "memory_peak_bytes": None, "memory_current_bytes": None, "systemd": None}
    if not isinstance(props, dict) or "LoadState" not in props or "ActiveState" not in props:
        return out
    out["systemd"] = {k: props.get(k) for k in POLL_PROPS if k != "Id"}
    out["memory_peak_bytes"] = _int(props.get("MemoryPeak"))
    out["memory_current_bytes"] = _int(props.get("MemoryCurrent"))
    load, active, sub, result = props["LoadState"], props["ActiveState"], props.get("SubState"), props.get("Result")
    if load == "not-found":
        return dict(out, state="NOT_FOUND", status="LAUNCH_NOT_FOUND",
                    reason="UNIT_NOT_FOUND: never started, host restarted, or reset by someone else")
    if load != "loaded":
        return dict(out, reason=f"UNIT_LOADSTATE_{load}")
    if active in ("activating", "deactivating", "reloading") or (active == "active" and sub != "exited"):
        return dict(out, state="RUNNING", status=None, reason="")
    if not (active in ("failed", "inactive") or (active == "active" and sub == "exited")):
        return dict(out, reason=f"UNRECOGNIZED_ACTIVESTATE_{active}")
    code, status = props.get("ExecMainCode"), _int(props.get("ExecMainStatus"))
    fin = dict(out, state="FINISHED")
    if result == "oom-kill":
        return dict(fin, status="RESOURCE_EXCEEDED", reason="CGROUP_OOM_KILL (Result=oom-kill)")
    if result == "timeout":
        return dict(fin, status="RESOURCE_EXCEEDED", reason="WALL_TIME_LIMIT (Result=timeout, RuntimeMaxSec)")
    if result in ("success", "exit-code") and code == "1" and status is not None:
        if status == 0 and result == "success":
            return dict(fin, status="COMPLETED", reason="", exit_code=0)
        if status != 0:
            return dict(fin, status="FAILED", reason=f"exit={status}" + (" (exec failed)" if status == 203 else ""),
                        exit_code=status)
    if result == "signal":
        return dict(fin, status="UNCERTAIN", reason=f"KILLED_BY_SIGNAL {SIGNALS.get(status, status)} without an "
                                                    "OOM or timeout result (memguard stops the batch slice by signal)")
    return dict(fin, status="UNCERTAIN", reason=f"UNRECOGNIZED_RESULT Result={result} ExecMainCode={code} "
                                                f"ExecMainStatus={props.get('ExecMainStatus')}")


class SystemdBackend:
    """Runs start/show/release on a role: `bash -c` here, `ssh <alias> bash -c` for a worker."""

    def __init__(self, roles_map: dict, runner=subprocess.run, timeout: float = 60.0):
        self.roles_map, self.runner, self.timeout = roles_map, runner, timeout

    def _alias(self, role):
        if role == "COORDINATOR":
            return None
        alias = (self.roles_map.get(role) or {}).get("ssh")
        if not alias:
            raise SystemExit(f"REFUSED: no ssh alias for {role} in the private role map")
        return alias

    def argv(self, role: str, script: str) -> list:
        alias = self._alias(role)
        if alias is None:
            return ["bash", "-c", script]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, "bash -c " + shlex.quote(script)]

    def _exec(self, role, script):
        try:
            r = self.runner(self.argv(role, script), capture_output=True, text=True, timeout=self.timeout)
            return r.returncode, r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired:
            return 124, "", "TIMEOUT"
        except OSError as exc:
            return 127, "", f"{type(exc).__name__}: {exc}"

    def start(self, role: str, script: str) -> dict:
        rc, out, err = self._exec(role, script)
        return {"rc": rc, "message": HC.redact((err or out).strip()[-300:], [self._alias(role)])}

    def show(self, role: str, units: list):
        """-> {unit: props} or None when the role cannot be read."""
        script = "systemctl --user show " + " ".join(shlex.quote(u + ".service") for u in units) + " " + \
            " ".join(f"-p {p}" for p in POLL_PROPS)
        rc, out, err = self._exec(role, script)
        blocks = parse_show(out)
        if not blocks:
            return None
        return {u: blocks.get(u + ".service") for u in units}

    def release(self, role: str, unit: str, props: dict) -> int:
        verb = "reset-failed" if props.get("ActiveState") == "failed" else "stop"
        rc, _, _ = self._exec(role, f"systemctl --user {verb} {shlex.quote(unit + '.service')}")
        return rc


# ------------------------------------------------------------- dispatcher
class Dispatcher:
    def __init__(self, root: Path, jobs: list, *, inventory_fn, backend, policy: dict | None = None,
                 checkouts: dict | None = None, stop_file: Path | None = None, poll_seconds: float = 5.0,
                 max_wait_seconds: float = 3600.0, max_inventory_age_seconds: float = 2.0,
                 max_launch_refusals: int = 5, rerun_uncertain: bool = False, resume: bool = False,
                 clock=time.time, sleep=time.sleep):
        self.root, self.jobs = Path(root), list(jobs)
        self.inventory_fn, self.backend = inventory_fn, backend
        self.policy = policy or PL.default_policy()
        self.checkouts = checkouts or {}
        self.stop_file = Path(stop_file) if stop_file else None
        self.poll_seconds, self.max_wait, self.max_inv_age = poll_seconds, max_wait_seconds, max_inventory_age_seconds
        self.max_refusals, self.rerun_uncertain, self.resume = max_launch_refusals, rerun_uncertain, resume
        self.clock, self.sleep = clock, sleep
        ids = [j.get("job_id") for j in self.jobs]
        if len(set(ids)) != len(ids):
            raise SystemExit("REFUSED: job_id repeats in the jobs file")
        for j in self.jobs:
            p = PL.validate_job(j)
            if not isinstance(j.get("argv"), list) or not j["argv"]:
                p.append("argv: non-empty list required")
            if not WALL_RE.match(str(j.get("wall", "4h"))):
                p.append("wall: like 30m, 4h")
            if p:
                raise SystemExit(f"REFUSED: job {j.get('job_id')!r}: {'; '.join(p)}")
        self.running = []
        self.deferred = []
        self.results = {}
        self.refusals = {}
        self.wait_since = {}
        self.release_failures = []
        self.inventory_reads = 0
        self.launches = []
        self.reattached = []
        self._inv = None
        self._inv_t = None

    # ---------------------------------------------------------------- files
    def _prepare_root(self):
        if self.root.exists():
            if not self.resume:
                raise SystemExit(f"REFUSED: {self.root.name} exists; a dispatch root is write-once (use --resume)")
            if (self.root / "DISPATCH_RECEIPT.json").exists():
                raise SystemExit(f"REFUSED: {self.root.name} is sealed by its final receipt")
            return
        for d in ("inventories", "receipts"):
            (self.root / d).mkdir(parents=True)
        code = {n: hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
                for n in ("df_dispatch", "df_placement", "df_host_capacity")}
        write_once(self.root / "DISPATCH_MANIFEST.json",
                   {"schema": SCHEMA, "created_at": now_iso(), "policy": self.policy, "code_sha256": code,
                    "jobs": [{"job_id": j["job_id"], "identity": identity(j), "unit": unit_name(j)}
                             for j in self.jobs]})

    def _attempts(self, job_id: str) -> list:
        out = []
        for p in (self.root / "receipts").glob(f"{safe(job_id)}.attempt-*.json"):
            m = re.match(r".*\.attempt-(\d+)(\.launch)?\.json$", p.name)
            if m:
                out.append((int(m.group(1)), bool(m.group(2)), p))
        return sorted(out)

    def _next_attempt(self, job_id: str) -> int:
        a = self._attempts(job_id)
        return (max(n for n, _, _ in a) + 1) if a else 1

    def _terminal(self, job: dict, attempt: int, status: str, reason: str, decision=None, launch=None, **extra):
        doc = {"job_id": job["job_id"], "identity": identity(job), "attempt": attempt, "status": status,
               "reason": reason, "parent_job_id": job.get("parent_job_id"), "cpu_bytes_estimate": job["cpu_bytes"],
               "gpu_bytes_estimate": job.get("gpu_bytes", 0), "cpus": job.get("cpus", 1),
               "decision": _decision_record(decision) if decision else None, "ended_at": now_iso()}
        if launch:
            doc.update({k: launch.get(k) for k in ("role", "unit", "gpu", "request_bytes", "inventory_digest",
                                                   "command", "launched_at")})
        doc.update(extra)
        write_once(self.root / "receipts" / f"{safe(job['job_id'])}.attempt-{attempt}.json", doc)
        if status not in REQUEUE:
            self.results[job["job_id"]] = {"status": status, "attempt": attempt, "reason": reason,
                                           "role": (launch or {}).get("role")}
        return doc

    def _save_inventory(self, inv: dict):
        p = self.root / "inventories" / f"{inv.get('digest_sha256') or HC.inventory_digest(inv)}.json"
        if not p.exists():
            write_once(p, inv)

    def _requeue(self, job: dict):
        n = self.refusals[job["job_id"]] = self.refusals.get(job["job_id"], 0) + 1
        if n >= self.max_refusals:
            self._terminal(job, self._next_attempt(job["job_id"]), "UNPLACEABLE", f"LAUNCH_REQUEUED_{n}_TIMES")
        else:
            self.deferred.append(job)

    # --------------------------------------------------------------- units
    def _attach(self, job: dict, launch: dict, how: str):
        gpu = launch.get("gpu") or {}
        self.running.append({"job": job, "launch": launch, "unit": launch["unit"], "role": launch["role"],
                             "attempt": launch["attempt"], "t0": self.clock(), "how": how,
                             "observed_peak_bytes": None,
                             "reservation": {"job_id": job["job_id"], "role": launch["role"],
                                             "request_bytes": launch["request_bytes"], "cpus": job.get("cpus", 1),
                                             "gpu_uuid": gpu.get("uuid"), "gpu_reserved_bytes": gpu.get("reserved_bytes", 0),
                                             "observed_current_bytes": 0}})

    def _finish(self, entry: dict, c: dict, props: dict | None) -> None:
        """Terminal receipt from a classified unit, then release it."""
        job = entry["job"]
        peak = c["memory_peak_bytes"] if c["memory_peak_bytes"] is not None else entry.get("observed_peak_bytes")
        self._terminal(job, entry["attempt"], c["status"], c["reason"], launch=entry["launch"],
                       systemd=c["systemd"], exit_code=c["exit_code"], memory_peak_bytes=peak,
                       wall_seconds=round(self.clock() - entry["t0"], 3), attached_by=entry.get("how"))
        if c["state"] == "FINISHED" and props:
            rc = self.backend.release(entry["role"], entry["unit"], props)
            if rc != 0:
                self.release_failures.append({"job_id": job["job_id"], "unit": entry["unit"], "rc": rc})
        if c["status"] in REQUEUE:
            self._requeue(job)

    def _poll(self):
        by_role = {}
        for e in self.running:
            by_role.setdefault(e["role"], []).append(e)
        still = []
        for role, entries in by_role.items():
            props = self.backend.show(role, [e["unit"] for e in entries])
            if props is None:                       # role unreachable: the unit keeps running; poll again
                still += entries
                continue
            for e in entries:
                p = props.get(e["unit"])
                c = classify_unit(p)
                if c["state"] == "RUNNING":
                    e["reservation"]["observed_current_bytes"] = c["memory_current_bytes"] or 0
                    if c["memory_peak_bytes"] is not None:
                        e["observed_peak_bytes"] = max(e["observed_peak_bytes"] or 0, c["memory_peak_bytes"])
                    still.append(e)
                else:
                    self._finish(e, c, p)
        self.running = still

    # --------------------------------------------------------------- resume
    def _resume_filter(self, jobs: list) -> list:
        out = []
        for job in jobs:
            ident = identity(job)
            att = self._attempts(job["job_id"])
            terminals = {n: json.loads(p.read_text()) for n, launch, p in att if not launch}
            launches = {n: json.loads(p.read_text()) for n, launch, p in att if launch}
            attached = False
            for n, rec in sorted(launches.items()):
                if n in terminals or rec.get("identity") != ident:
                    continue
                props = self.backend.show(rec["role"], [rec["unit"]])
                if props is None:
                    self._attach(job, rec, "RESUME_ROLE_UNREADABLE")
                    self.reattached.append(job["job_id"])
                    attached = True
                    break
                c = classify_unit(props.get(rec["unit"]))
                if c["state"] == "RUNNING":
                    self._attach(job, rec, "RESUME_REATTACHED_ACTIVE_UNIT")
                    self.reattached.append(job["job_id"])
                    attached = True
                    break
                entry = {"job": job, "launch": rec, "unit": rec["unit"], "role": rec["role"], "attempt": n,
                         "t0": self.clock(), "how": "RESUME_CLASSIFIED_STORED_RESULT", "observed_peak_bytes": None}
                self._finish(entry, c, props.get(rec["unit"]))
                terminals[n] = json.loads((self.root / "receipts" / f"{safe(job['job_id'])}.attempt-{n}.json")
                                          .read_text())
                if c["status"] in REQUEUE:
                    attached = True           # _finish requeued it (deferred) or refused it after too many tries
                    break
            if attached:
                continue
            mine = [terminals[n] for n in sorted(terminals) if terminals[n].get("identity") == ident]
            last = mine[-1] if mine else None
            if last and last["status"] in RESUME_SKIP:
                if last["status"] == "SPLIT":
                    self.results[job["job_id"]] = {"status": "SPLIT", "attempt": last["attempt"],
                                                   "reason": last["reason"], "role": None, "resumed_skip": True}
                    out += self._resume_filter(PL.split_job(job, int(last["n_chunks"])))
                else:
                    self.results[job["job_id"]] = {"status": last["status"], "attempt": last["attempt"],
                                                   "reason": last["reason"], "role": last.get("role"),
                                                   "resumed_skip": True}
                continue
            if last and last["status"] == "UNCERTAIN" and not self.rerun_uncertain:
                self.results[job["job_id"]] = {"status": "UNCERTAIN", "attempt": last["attempt"],
                                               "reason": last["reason"] + "; not re-run without --rerun-uncertain",
                                               "role": last.get("role"), "resumed_skip": True}
                continue
            out.append(job)
        return out

    # ------------------------------------------------------------ inventory
    def _inventory(self) -> dict:
        if self._inv is None or self.clock() - self._inv_t > self.max_inv_age:
            self._inv = self.inventory_fn()
            self._inv_t = self.clock()
            self.inventory_reads += 1
            self._save_inventory(self._inv)
        return self._inv

    def reservations(self) -> list:
        return [r["reservation"] for r in self.running]

    # --------------------------------------------------------------- launch
    def _launch(self, job: dict, decision: dict, inv: dict):
        role = decision["role"]
        default_ck = DEFAULT_WORKER_CHECKOUT
        if role == "COORDINATOR" and HERE.parent.is_relative_to(Path.home()):
            default_ck = str(HERE.parent.relative_to(Path.home()))
        # {role} in argv is bound to the placed role only in the command; identity and unit name stay those of the
        # template job, so resume matches the same job wherever it was placed
        script = build_start_script(bind_role(job, role), decision, self.checkouts.get(role, default_ck))
        attempt = self._next_attempt(job["job_id"])
        launch = {"job_id": job["job_id"], "identity": identity(job), "attempt": attempt, "role": role,
                  "unit": unit_name(job), "gpu": decision.get("gpu"), "request_bytes": decision["request_bytes"],
                  "inventory_digest": inv.get("digest_sha256"), "inventory_read_at": inv.get("read_at"),
                  "decision": _decision_record(decision), "command": script, "launched_at": now_iso()}
        write_once(self.root / "receipts" / f"{safe(job['job_id'])}.attempt-{attempt}.launch.json", launch)
        self.launches.append((job["job_id"], role, inv.get("digest_sha256")))
        self.wait_since.pop(job["job_id"], None)
        res = self.backend.start(role, script)
        rc = res["rc"]
        if rc == 0:
            self._attach(job, launch, "LAUNCHED")
        elif rc == EXIT_UNIT_LOADED:
            self._attach(job, launch, "ALREADY_LOADED_REATTACHED")
        elif rc == EXIT_REFUSED:
            self._terminal(job, attempt, "REFUSED_AT_LAUNCH", res["message"] or "admission refused", launch=launch,
                           exit_code=rc)
            self._requeue(job)
        else:
            props = self.backend.show(role, [launch["unit"]])
            c = classify_unit(props.get(launch["unit"])) if props else None
            if props is None or c["state"] != "NOT_FOUND":
                self._attach(job, launch, f"START_EXIT_{rc}_UNIT_PRESENT_OR_UNREADABLE")
            else:
                self._terminal(job, attempt, "LAUNCH_NOT_FOUND", f"systemd-run exit={rc}: {res['message']}",
                               launch=launch, exit_code=rc)
                self._requeue(job)

    # ----------------------------------------------------------------- loop
    def run(self) -> dict:
        self._prepare_root()
        pending = self._resume_filter(self.jobs)
        stopped = False
        while pending or self.running or self.deferred:
            self._poll()
            pending += self.deferred
            self.deferred = []
            if self.stop_file is not None and self.stop_file.exists():
                stopped = True
            if not stopped:
                i = 0
                while i < len(pending):
                    if self.stop_file is not None and self.stop_file.exists():
                        stopped = True           # checked before every placement, not once per pass
                        break
                    job = pending[i]
                    inv = self._inventory()
                    d = PL.place(job, inv, self.reservations(), self.policy)
                    if d["decision"] == "PLACED":
                        pending.pop(i)
                        self._launch(job, d, inv)
                        self._inv = None           # the next launch needs a fresh reading
                        continue
                    if d["decision"] == "SPLIT":
                        pending.pop(i)
                        self._terminal(job, self._next_attempt(job["job_id"]), "SPLIT", d["reason"], decision=d,
                                       n_chunks=d["n_chunks"], chunk_job_ids=[c["job_id"] for c in d["chunks"]])
                        pending[i:i] = self._resume_filter(d["chunks"])
                        continue
                    if d["decision"] == "UNPLACEABLE":
                        pending.pop(i)
                        self._terminal(job, self._next_attempt(job["job_id"]), "UNPLACEABLE", d["reason"], decision=d)
                        continue
                    first = self.wait_since.setdefault(job["job_id"], self.clock())
                    if not self.running and self.clock() - first > self.max_wait:
                        pending.pop(i)
                        self._terminal(job, self._next_attempt(job["job_id"]), "UNPLACEABLE",
                                       f"WAIT_TIMEOUT_{self.max_wait}s: {d['reason']}", decision=d)
                        continue
                    i += 1
            if stopped and not self.running:
                break
            if not pending and not self.running and not self.deferred:
                break
            self.sleep(self.poll_seconds)
        return self._receipt(pending + self.deferred, stopped)

    def _receipt(self, pending: list, stopped: bool) -> dict:
        rows = dict(self.results)
        for job in pending:
            rows.setdefault(job["job_id"], {"status": "NOT_STARTED",
                                            "reason": "STOP_FILE" if stopped else "PENDING", "role": None})
        counts = {}
        for v in rows.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        final = not pending and not stopped
        doc = {"schema": SCHEMA, "final": final, "stopped": stopped, "counts": counts, "jobs": rows,
               "inventory_reads": self.inventory_reads, "reattached_on_resume": self.reattached,
               "release_failures": self.release_failures,
               "launches": [{"job_id": j, "role": r, "inventory_digest": d} for j, r, d in self.launches],
               "ended_at": now_iso()}
        if final:
            write_once(self.root / "DISPATCH_RECEIPT.json", doc)
        else:
            n = len(list(self.root.glob("DISPATCH_PROGRESS.*.json"))) + 1
            write_once(self.root / f"DISPATCH_PROGRESS.{n}.json", doc)
        return doc


def _decision_record(d: dict) -> dict:
    return {k: d.get(k) for k in ("decision", "role", "gpu", "request_bytes", "reason", "per_role",
                                  "inventory_digest", "n_chunks")}


def load_jobs(path: Path) -> list:
    doc = json.loads(Path(path).read_text())
    jobs = doc["jobs"] if isinstance(doc, dict) else doc
    if not isinstance(jobs, list):
        raise SystemExit("REFUSED: jobs file must be a list or {\"jobs\": [...]}")
    return jobs


def summarize_inventory(inv: dict) -> dict:
    out = {}
    for role, r in inv.get("roles", {}).items():
        if not r.get("reachable"):
            out[role] = {"reachable": False, "error": r.get("error")}
            continue
        sl = r["batch_slice"]
        out[role] = {"cpus": r["cpus"], "load1": r["load1"], "mem_total_gib": round(r["mem_total_bytes"] / PL.GIB, 2),
                     "mem_available_gib": round(r["mem_available_bytes"] / PL.GIB, 2),
                     "slice_max_gib": round(sl["memory_max_bytes"] / PL.GIB, 2) if sl["memory_max_bytes"] else None,
                     "slice_current_gib": round((sl["memory_current_bytes"] or 0) / PL.GIB, 2),
                     "memguard_active": r["memguard_active"], "crispdm_run_present": r["crispdm_run_present"],
                     "gpus": [{"index": g["index"], "name": g["name"], "status": g["status"],
                               "vram_total_gib": round(g["vram_total_bytes"] / PL.GIB, 2)
                               if g["vram_total_bytes"] else None,
                               "vram_free_gib": round(g["vram_free_bytes"] / PL.GIB, 2)
                               if g["vram_free_bytes"] is not None else None,
                               "compute_processes": g["compute_processes"]} for g in r["gpus"]]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs-file", type=Path, required=True)
    ap.add_argument("--root", type=Path, help="write-once dispatch root (required unless --dry-run)")
    ap.add_argument("--roles", type=Path, default=HC.DEFAULT_ROLES)
    ap.add_argument("--quarantine-file", type=Path, default=HC.DEFAULT_QUARANTINE_FILE)
    ap.add_argument("--allow-gpu", action="store_true", help="admit GPU jobs (default: refused)")
    ap.add_argument("--role-cap", action="append", default=[], help="ROLE=N concurrency cap")
    ap.add_argument("--coordinator-max-fraction", type=float, default=PL.COORDINATOR_MAX_FRACTION)
    ap.add_argument("--checkout", action="append", default=[], help="ROLE=path relative to that host's home")
    ap.add_argument("--dry-run", action="store_true", help="print the live inventory and the plan; start nothing")
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rerun-uncertain", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--max-wait-seconds", type=float, default=3600.0)
    a = ap.parse_args(argv)
    caps = dict(PL.DEFAULT_ROLE_CAPS)
    for s in a.role_cap:
        k, _, v = s.partition("=")
        if k not in PL.ROLES:
            ap.error(f"--role-cap: unknown role {k}")
        caps[k] = int(v)
    checkouts = {}
    for s in a.checkout:
        k, _, v = s.partition("=")
        if k not in PL.ROLES or v.startswith("/"):
            ap.error("--checkout ROLE=path relative to home")
        checkouts[k] = v
    policy = PL.default_policy(allow_gpu=a.allow_gpu, role_caps=caps, coordinator_max_fraction=a.coordinator_max_fraction)
    jobs = load_jobs(a.jobs_file)
    roles_map = json.loads(a.roles.read_text())
    quarantine = HC.load_quarantine(a.quarantine_file)

    def inventory_fn():
        return HC.read_inventory(roles_map, quarantine)

    if a.dry_run:
        inv = inventory_fn()
        doc = {"dry_run": True, "inventory_read_at": inv["read_at"], "inventory_digest": inv["digest_sha256"],
               "inventory": summarize_inventory(inv), "policy": policy, "plan": PL.plan(jobs, inv, policy)}
        text = json.dumps(doc, indent=1, sort_keys=True, default=str)
        for cfg in roles_map.values():
            if cfg.get("ssh") and cfg["ssh"] in text:
                raise SystemExit("REFUSED: an alias leaked into the plan")
        print(redact(text))
        return 0
    if not a.root:
        ap.error("--root is required unless --dry-run")
    d = Dispatcher(a.root, jobs, inventory_fn=inventory_fn, backend=SystemdBackend(roles_map), policy=policy,
                   checkouts=checkouts, stop_file=a.stop_file, poll_seconds=a.poll_seconds,
                   max_wait_seconds=a.max_wait_seconds, rerun_uncertain=a.rerun_uncertain, resume=a.resume)
    r = d.run()
    print(json.dumps({"counts": r["counts"], "final": r["final"], "stopped": r["stopped"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
