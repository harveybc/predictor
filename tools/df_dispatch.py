#!/usr/bin/env python3
"""C174 (order 2026-09-13): memory-aware job dispatcher across host roles.

Loop (declared):

1. poll running jobs; a finished job gets its durable terminal receipt;
2. if the stop file exists, launch nothing more; running jobs finish normally;
3. for each pending job, in order (a job that must WAIT does not block later jobs):
   * re-read the LIVE inventory (df_host_capacity) when none is held; the held
     inventory is dropped after EVERY launch and whenever it is older than
     max_inventory_age_seconds, so no launch uses a reading taken before an earlier
     launch (another agent may have consumed COORDINATOR memory meanwhile);
   * df_placement.place against that inventory and this dispatcher's running
     reservations (request minus the scope's observed memory.current);
   * PLACED: launch through
         crispdm-run -m <request MiB>M -t <wall> -n <name> -- env -u PYTHONPATH CUDA_VISIBLE_DEVICES=<""|uuid> argv
     in the role's checkout (ssh for a worker), write the launch record;
   * SPLIT: write the parent's SPLIT receipt; its chunks take its place in the queue;
   * UNPLACEABLE: terminal receipt with the reason per role;
   * WAIT: stays pending; after max_wait_seconds of waiting with nothing of this
     dispatcher running, it becomes UNPLACEABLE (WAIT_TIMEOUT) with the last reasons;
4. sleep, repeat until nothing is pending or running.

Per-role concurrency caps are enforced by the placement (policy role_caps).

Receipts (root, write-once, no host names, no home paths):
    DISPATCH_MANIFEST.json                       policy and code digests at creation
    inventories/<digest>.json                    every inventory snapshot used, content-addressed
    receipts/<job>.attempt-N.launch.json         intent: role, GPU, decision, inventory digest,
                                                 requested bytes, command
    receipts/<job>.attempt-N.json                terminal: the launch fields plus status, exit code,
                                                 observed (sampled) scope memory peak, times
    logs/<job>.attempt-N.log                     the job's output
    DISPATCH_PROGRESS.N.json | DISPATCH_RECEIPT.json

Terminal statuses: COMPLETED (exit 0), FAILED (other exit), RESOURCE_EXCEEDED (124 wall
limit, 137/-9 SIGKILL: cgroup OOM, memguard or kill-after), REFUSED_AT_LAUNCH (75:
crispdm-run refused because free memory changed after the reading; requeued, at most
max_launch_refusals times), UNCERTAIN (ssh exit 255, another signal, or a launch record
without a terminal found on resume), UNPLACEABLE, SPLIT.

Resume by identity (sha256 of job_id, argv, cpu_bytes, gpu_bytes, cpus, wall, roles,
gpu_index, split): a job whose latest receipt with the same identity is COMPLETED,
FAILED, RESOURCE_EXCEEDED, UNPLACEABLE or SPLIT is not run again (a SPLIT parent's
chunks are regenerated with the recorded chunk count and resumed individually).
UNCERTAIN is re-run only with --rerun-uncertain (a lost job may still hold memory).
Known limit: jobs are attached to this process (and to their ssh session); if the
dispatcher dies they are lost and resume records them UNCERTAIN.

`--dry-run` prints the live inventory and the placement plan for a jobs file and
launches nothing.
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

SCHEMA = "crispdm.data_foundation.dispatch_receipt.v1"
RESUME_SKIP = ("COMPLETED", "FAILED", "RESOURCE_EXCEEDED", "UNPLACEABLE", "SPLIT")
REQUEUE = ("REFUSED_AT_LAUNCH",)
CRISPDM_RUN_REL = ".local/bin/crispdm-run"
DEFAULT_WORKER_CHECKOUT = "Documents/GitHub/.worktrees/predictor-c146"
WALL_RE = re.compile(r"^\d+(s|m|h|d)?$")


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


def unit_name(job: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job["job_id"])[:40] + "-" + identity(job)[:8]


def scope_prefix(job: dict) -> str:
    # crispdm-run names the scope crispdm-<name>-<epoch>-<pid>
    return f"crispdm-{unit_name(job)}-"


def write_once(path: Path, doc: dict) -> None:
    import df_isolated_runner as IR
    IR.atomic_write_once(path, redact(json.dumps(doc, indent=1, sort_keys=True, default=str)) + "\n")


def classify_exit(rc: int | None) -> tuple:
    if rc == 0:
        return "COMPLETED", ""
    if rc == 75:
        return "REFUSED_AT_LAUNCH", "crispdm-run refused: memory changed after the inventory reading"
    if rc == 124:
        return "RESOURCE_EXCEEDED", "WALL_TIME_LIMIT (timeout exit 124)"
    if rc in (137, -9):
        return "RESOURCE_EXCEEDED", "SIGKILL: cgroup OOM kill, memguard hard stop or wall kill-after"
    if rc == 255:
        return "UNCERTAIN", "ssh exit 255: connection lost or remote shell error"
    if rc is None or rc < 0 or rc > 128:
        return "UNCERTAIN", f"KILLED_BY_SIGNAL_OR_UNKNOWN exit={rc}"
    return "FAILED", f"exit={rc}"


# --------------------------------------------------------------- commands
def build_command(job: dict, decision: dict, alias, checkout_rel: str) -> dict:
    """-> {"local_argv": [...]} for this host or {"ssh_argv": [...]} for a worker, plus a redacted display."""
    mib = -(-int(decision["request_bytes"]) // PL.MIB)
    wall = str(job.get("wall", "4h"))
    cuda = (decision.get("gpu") or {}).get("uuid") or ""
    inner = ["-m", f"{mib}M", "-t", wall, "-n", unit_name(job), "--", "env", "-u", "PYTHONPATH",
             f"CUDA_VISIBLE_DEVICES={cuda}"]
    argv = [str(a) for a in job["argv"]]
    if alias:
        def q(a):
            return '"$HOME"/' + shlex.quote(a[2:]) if a.startswith("~/") else shlex.quote(a)
        remote = f'cd "$HOME"/{shlex.quote(checkout_rel)} && "$HOME"/{CRISPDM_RUN_REL} ' + \
            " ".join(shlex.quote(a) for a in inner) + " " + " ".join(q(a) for a in argv)
        return {"ssh_argv": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", alias, remote],
                "display": remote}
    home = Path.home()
    local = [str(home / CRISPDM_RUN_REL)] + inner + [str(home / a[2:]) if a.startswith("~/") else a for a in argv]
    return {"local_argv": local, "cwd": str(home / checkout_rel), "display": redact(shlex.join(local))}


class _PopenHandle:
    def __init__(self, proc, log):
        self.proc, self.log = proc, log

    def poll(self):
        rc = self.proc.poll()
        if rc is not None and not self.log.closed:
            self.log.close()
        return rc


def real_launcher(role: str, command: dict, log_path: Path):
    """Starts the command for real. Never used by the tests."""
    log = open(log_path, "ab")
    if "ssh_argv" in command:
        proc = subprocess.Popen(command["ssh_argv"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.Popen(command["local_argv"], cwd=command["cwd"], stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT)
    return _PopenHandle(proc, log)


# ------------------------------------------------------------- dispatcher
class Dispatcher:
    def __init__(self, root: Path, jobs: list, *, inventory_fn, launcher, roles_map: dict,
                 policy: dict | None = None, checkouts: dict | None = None, stop_file: Path | None = None,
                 poll_seconds: float = 5.0, max_wait_seconds: float = 3600.0, max_inventory_age_seconds: float = 2.0,
                 observe_seconds: float = 30.0, max_launch_refusals: int = 5, rerun_uncertain: bool = False,
                 resume: bool = False, clock=time.time, sleep=time.sleep):
        self.root, self.jobs = Path(root), list(jobs)
        self.inventory_fn, self.launcher, self.roles_map = inventory_fn, launcher, roles_map
        self.policy = policy or PL.default_policy()
        self.checkouts = checkouts or {}
        self.stop_file = Path(stop_file) if stop_file else None
        self.poll_seconds, self.max_wait = poll_seconds, max_wait_seconds
        self.max_inv_age, self.observe_seconds = max_inventory_age_seconds, observe_seconds
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
        self.running = []          # dicts: job, decision, handle, attempt, reservation, launched_at, peak
        self.results = {}          # job_id -> final status summary
        self.refusals = {}
        self.wait_since = {}
        self.inventory_reads = 0
        self.launches = []         # (job_id, role, inventory digest) in launch order
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
        for d in ("inventories", "receipts", "logs"):
            (self.root / d).mkdir(parents=True)
        code = {n: hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
                for n in ("df_dispatch", "df_placement", "df_host_capacity")}
        write_once(self.root / "DISPATCH_MANIFEST.json",
                   {"schema": SCHEMA, "created_at": now_iso(), "policy": self.policy, "code_sha256": code,
                    "jobs": [{"job_id": j["job_id"], "identity": identity(j)} for j in self.jobs]})

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
            doc.update({k: launch[k] for k in ("role", "gpu", "request_bytes", "inventory_digest", "command",
                                               "launched_at")})
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

    # --------------------------------------------------------------- resume
    def _resume_filter(self, jobs: list) -> list:
        out = []
        for job in jobs:
            ident = identity(job)
            att = [(n, launch, p) for n, launch, p in self._attempts(job["job_id"])]
            terminals = {n: json.loads(p.read_text()) for n, launch, p in att if not launch}
            launches = {n: json.loads(p.read_text()) for n, launch, p in att if launch}
            for n, rec in sorted(launches.items()):
                if n not in terminals and rec.get("identity") == ident:
                    terminals[n] = self._terminal(job, n, "UNCERTAIN", "DISPATCHER_RESTARTED_WITHOUT_TERMINAL",
                                                  launch=rec)
            mine = [terminals[n] for n in sorted(terminals) if terminals[n].get("identity") == ident]
            last = mine[-1] if mine else None
            if last and last["status"] in RESUME_SKIP:
                if last["status"] == "SPLIT":
                    chunks = PL.split_job(job, int(last["n_chunks"]))
                    self.results[job["job_id"]] = {"status": "SPLIT", "attempt": last["attempt"],
                                                   "reason": last["reason"], "role": None, "resumed_skip": True}
                    out += self._resume_filter(chunks)
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
            self._observe(self._inv)
        return self._inv

    def _observe(self, inv: dict):
        for r in self.running:
            role = inv.get("roles", {}).get(r["decision"]["role"]) or {}
            scopes = [s for s in role.get("batch_scopes") or [] if s["unit"].startswith(scope_prefix(r["job"]))]
            if scopes:
                r["reservation"]["observed_current_bytes"] = max(s["memory_current_bytes"] or 0 for s in scopes)
                peak = max(s["memory_peak_bytes"] or 0 for s in scopes)
                r["observed_peak_bytes"] = max(r.get("observed_peak_bytes") or 0, peak)

    def reservations(self) -> list:
        return [r["reservation"] for r in self.running]

    # --------------------------------------------------------------- launch
    def _launch(self, job: dict, decision: dict, inv: dict):
        role = decision["role"]
        alias = None if role == "COORDINATOR" else (self.roles_map.get(role) or {}).get("ssh")
        default_ck = str(HERE.parent.relative_to(Path.home())) if role == "COORDINATOR" and \
            HERE.parent.is_relative_to(Path.home()) else DEFAULT_WORKER_CHECKOUT
        cmd = build_command(job, decision, alias, self.checkouts.get(role, default_ck))
        attempt = self._next_attempt(job["job_id"])
        launch = {"job_id": job["job_id"], "identity": identity(job), "attempt": attempt, "role": role,
                  "gpu": decision.get("gpu"), "request_bytes": decision["request_bytes"],
                  "inventory_digest": inv.get("digest_sha256"), "inventory_read_at": inv.get("read_at"),
                  "decision": _decision_record(decision), "command": cmd["display"], "launched_at": now_iso()}
        write_once(self.root / "receipts" / f"{safe(job['job_id'])}.attempt-{attempt}.launch.json", launch)
        handle = self.launcher(role, cmd, self.root / "logs" / f"{safe(job['job_id'])}.attempt-{attempt}.log")
        gpu = decision.get("gpu") or {}
        self.running.append({"job": job, "decision": decision, "handle": handle, "attempt": attempt, "launch": launch,
                             "t0": self.clock(), "observed_peak_bytes": None,
                             "reservation": {"job_id": job["job_id"], "role": role,
                                             "request_bytes": decision["request_bytes"],
                                             "cpus": job.get("cpus", 1), "gpu_uuid": gpu.get("uuid"),
                                             "gpu_reserved_bytes": gpu.get("reserved_bytes", 0),
                                             "observed_current_bytes": 0}})
        self.launches.append((job["job_id"], role, inv.get("digest_sha256")))
        self.wait_since.pop(job["job_id"], None)

    def _poll(self, pending: list):
        still = []
        for r in self.running:
            rc = r["handle"].poll()
            if rc is None:
                still.append(r)
                continue
            status, reason = classify_exit(rc)
            job = r["job"]
            self._terminal(job, r["attempt"], status, reason, launch=r["launch"], exit_code=rc,
                           observed_peak_bytes_sampled=r["observed_peak_bytes"],
                           wall_seconds=round(self.clock() - r["t0"], 3))
            if status in REQUEUE:
                self.refusals[job["job_id"]] = self.refusals.get(job["job_id"], 0) + 1
                if self.refusals[job["job_id"]] >= self.max_refusals:
                    self._terminal(job, self._next_attempt(job["job_id"]), "UNPLACEABLE",
                                   f"REFUSED_AT_LAUNCH_{self.refusals[job['job_id']]}_TIMES")
                else:
                    pending.insert(0, job)
        self.running = still

    # ----------------------------------------------------------------- loop
    def run(self) -> dict:
        self._prepare_root()
        pending = self._resume_filter(self.jobs)
        stopped = False
        last_observe = self.clock()
        while pending or self.running:
            self._poll(pending)
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
            if not pending and not self.running:
                break
            if self.running and self.clock() - last_observe >= self.observe_seconds:
                self._inv = None
                self._inventory()
                last_observe = self.clock()
            self.sleep(self.poll_seconds)
        return self._receipt(pending, stopped)

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
               "inventory_reads": self.inventory_reads,
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
    ap.add_argument("--checkout", action="append", default=[], help="ROLE=path relative to that host's home")
    ap.add_argument("--dry-run", action="store_true", help="print the live inventory and the plan; launch nothing")
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
    policy = PL.default_policy(allow_gpu=a.allow_gpu, role_caps=caps)
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
    d = Dispatcher(a.root, jobs, inventory_fn=inventory_fn, launcher=real_launcher, roles_map=roles_map,
                   policy=policy, checkouts=checkouts, stop_file=a.stop_file, poll_seconds=a.poll_seconds,
                   max_wait_seconds=a.max_wait_seconds, rerun_uncertain=a.rerun_uncertain, resume=a.resume)
    r = d.run()
    print(json.dumps({"counts": r["counts"], "final": r["final"], "stopped": r["stopped"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
