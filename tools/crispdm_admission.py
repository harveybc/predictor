#!/usr/bin/env python3
"""DR01 (order 2026-09-26): the one atomic per-host memory admission every compute launch goes through.

The defect this replaces
-----------------------
Every launch path on this fleet read available memory and *then* launched, holding nothing in
between. Two requests of 8 GiB were both admitted against a single 12 GiB reading (Musashi,
MUSASHI_DAY_REVIEW_2026_09_26.md F1). Serialising the instant of the reading does not fix that:
the commitment has to outlive the read and cover the whole load.

What this module is
-------------------
An admission authority with a durable, per-host **reservation** (a lease):

  acquire  -> under an exclusive file lock: reclaim, read fresh capacity, subtract the
              reservations of every live load, check the slice's aggregate observed budget,
              the desktop reserve and the live memory pressure, then ADMIT (write the lease)
              or QUEUE (write nothing, say why).  A request that can never fit any ceiling is
              REFUSED, terminally: the caller must not poll it.
  arm      -> bind the lease to the witness that proves the load is alive: the launcher pid
              (with its /proc start time, so a recycled pid cannot impersonate it) and/or the
              scope cgroup.  Before it is armed a lease is protected by a grace window: the
              absence of a witness is not yet evidence of death.
  renew    -> heartbeat; the lease lives as long as the load does.
  release  -> only when the witness is dead.  A release requested while the child is still
              alive RENEWS instead: crash recovery never frees the memory of a live child.
  reclaim  -> the sweep every acquire performs first: an expired lease whose witness is still
              alive is *extended*; a lease whose witness is dead is dropped, whether it
              expired or not (the orphan case).

Invariants the order names
--------------------------
* One integer.  ``cap_bytes`` is the single source of the byte value: MemoryMax, MemoryHigh and
  the reservation are all derived from it.  A caller that also passes a size string must pass
  one that parses to exactly that integer, or the request is REFUSED as CONTRADICTORY_CAP.  No
  second, discordant integer is ever written.
* Tree, not main process.  A pilot footprint offered to size an admission must be a cgroup or
  whole-tree peak; a main-process RSS (or getrusage maxrss) peak is REFUSED.
* Evidence, not assertion.  ``peak_bytes`` is only accepted together with the retained record
  it was read from; the file is hashed and the number re-read from it.
* Observed aggregate budget.  The slice gate is
  ``slice.memory.current + sum(max(0, reserved - observed)) + cap <= slice MemoryMax``, not the
  child's own limit against the ceiling.
* Never a retry with a bigger cap.  This module has no retry.  QUEUED means *not started yet*;
  REFUSED means not started at all.  An out-of-memory kill is a terminal outcome here.

It reads and writes nothing under a running child, signals nothing, and changes no kernel or
systemd setting.  Resource readings and the clock are injected, so the whole thing is testable
without pressuring real memory: set CRISPDM_ADMISSION_RESOURCES_JSON to a readings file and
CRISPDM_ADMISSION_NOW to a simulated epoch.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

KIB = 1024
MIB = 1 << 20
GIB = 1 << 30

# ---- declared constants; each one is the only place its value exists ----------------------
DESKTOP_RESERVE_BYTES = 3 * GIB       # the reserve crispdm-run has always kept for the owner's session
PRESSURE_ADMIT_MAX = 25.0             # user-slice memory PSI some/avg10 above which nothing new is admitted.
                                      # systemd-oomd kills this slice at 50% sustained over 20s; admission
                                      # stops well below that instead of racing it.
LEASE_TTL_SECONDS = 900               # a lease heartbeats; expiry alone never frees a live child
ARM_GRACE_SECONDS = 120               # before a lease is armed, a missing witness is not proof of death
PILOT_MARGIN_BYTES = GIB              # a measured pilot peak must fit with this much room to spare
DEFAULT_SLICE = "crispdm-batch.slice"
TREE_PEAK_SCOPES = ("cgroup", "tree") # the only footprint scopes that may size an admission
REFUSED_EXIT = 75                     # EX_TEMPFAIL, the code crispdm-run has always used

ADMITTED = "ADMITTED"
QUEUED = "QUEUED"
REFUSED = "REFUSED"


def parse_size(text: str) -> int:
    """'9G' -> 9663676416.  IEC only, integral, no floats: a size is one exact integer or an error."""
    s = str(text).strip()
    m = re.fullmatch(r"(\d+)\s*([KMGTP]?)i?B?", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"not an IEC size: {text!r}")
    mult = {"": 1, "K": KIB, "M": MIB, "G": GIB, "T": GIB * KIB, "P": GIB * MIB}[m.group(2).upper()]
    return int(m.group(1)) * mult


def parse_duration(text: str) -> int:
    """'4h' -> 14400, '900' -> 900.  The wall limit recorded in the lease and the one the child
    is given are the same number."""
    s = str(text).strip()
    m = re.fullmatch(r"(\d+)\s*(s|sec|m|min|h|d)?", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a duration: {text!r}")
    unit = (m.group(2) or "s").lower()
    return int(m.group(1)) * {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "d": 86400}[unit]


def human(n) -> str:
    if n is None:
        return "infinity"
    n = int(n)
    for unit, size in (("G", GIB), ("M", MIB), ("K", KIB)):
        if n >= size:
            return f"{n / size:.2f}{unit}"
    return f"{n}B"


# ---- resources -----------------------------------------------------------------------------

class SystemResources:
    """The live host.  Read-only: /proc, the cgroup tree and `systemctl --user show`."""

    def __init__(self, slice_name: str = DEFAULT_SLICE, cgroup_root=None):
        self.slice_name = slice_name
        self.cgroup_root = Path(cgroup_root or os.environ.get("CRISPDM_CGROUP_ROOT") or "/sys/fs/cgroup")
        self._slice_path = None

    # -- host
    def _meminfo(self) -> dict:
        out = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                out[parts[0][:-1]] = int(parts[1]) * KIB
        return out

    def mem_available_bytes(self) -> int:
        return self._meminfo()["MemAvailable"]

    def mem_total_bytes(self) -> int:
        return self._meminfo()["MemTotal"]

    # -- pressure: the PSI the user slice's own oomd watcher reads, not the global one
    def pressure_some_avg10(self) -> float:
        for candidate in (self._user_slice_pressure(), Path("/proc/pressure/memory")):
            if candidate and candidate.exists():
                for line in candidate.read_text().splitlines():
                    if line.startswith("some "):
                        for field_ in line.split():
                            if field_.startswith("avg10="):
                                return float(field_.split("=", 1)[1])
        return 0.0

    def _user_slice_pressure(self):
        p = self.slice_cgroup_path()
        while p is not None and p != self.cgroup_root:
            if p.name.endswith("user@%d.service" % os.getuid()):
                return p / "memory.pressure"
            p = p.parent
        guess = self.cgroup_root / f"user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/memory.pressure"
        return guess if guess.exists() else None

    # -- the shared slice
    def _show(self, prop: str):
        import subprocess
        try:
            r = subprocess.run(["systemctl", "--user", "show", self.slice_name, "-p", prop, "--value"],
                               capture_output=True, text=True, timeout=20)
        except Exception:
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""

    def slice_memory_max(self):
        v = self._show("MemoryMax")
        if not v or v == "infinity":
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def slice_cgroup_path(self):
        if self._slice_path is None:
            v = self._show("ControlGroup")
            self._slice_path = (self.cgroup_root / v.lstrip("/")) if v else None
        return self._slice_path

    def slice_memory_current(self) -> int:
        p = self.slice_cgroup_path()
        f = (p / "memory.current") if p else None
        try:
            return int(f.read_text().strip()) if f and f.exists() else 0
        except (OSError, ValueError):
            return 0

    # -- per-lease observation.  The cgroup is the whole process tree; never one process's RSS.
    def cgroup_current_bytes(self, cgroup: str):
        return self._cgroup_int(cgroup, "memory.current")

    def cgroup_peak_bytes(self, cgroup: str):
        for name in ("memory.peak", "memory.max_usage_in_bytes"):
            v = self._cgroup_int(cgroup, name)
            if v is not None:
                return v
        return None

    def _cgroup_int(self, cgroup, name):
        if not cgroup:
            return None
        f = self.cgroup_root / str(cgroup).lstrip("/") / name
        try:
            return int(f.read_text().strip())
        except (OSError, ValueError):
            return None

    def cgroup_alive(self, cgroup) -> bool:
        if not cgroup:
            return False
        d = self.cgroup_root / str(cgroup).lstrip("/")
        procs = d / "cgroup.procs"
        try:
            return bool(procs.read_text().strip())
        except OSError:
            return False

    def pid_alive(self, pid, starttime=None) -> bool:
        if not pid:
            return False
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
        except OSError:
            return False
        after = stat.rsplit(")", 1)[-1].split()
        if after and after[0] == "Z":          # a zombie holds no memory
            return False
        if starttime is not None:
            try:
                if int(after[19]) != int(starttime):   # field 22 overall; pid was recycled
                    return False
            except (IndexError, ValueError):
                return False
        return True

    def pid_starttime(self, pid):
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text()
        except OSError:
            return None
        try:
            return int(stat.rsplit(")", 1)[-1].split()[19])
        except (IndexError, ValueError):
            return None


class FileResources:
    """Readings from a JSON file: the simulated host the tests use.  No real memory is touched.

    {"mem_available_bytes":..., "mem_total_bytes":..., "slice_memory_max":... | null,
     "slice_memory_current":..., "pressure_some_avg10":...,
     "alive": {"<pid or cgroup>": true/false},
     "cgroup_current": {"<cgroup>": bytes}, "cgroup_peak": {"<cgroup>": bytes}}
    """

    def __init__(self, path, slice_name: str = DEFAULT_SLICE):
        self.path = Path(path)
        self.slice_name = slice_name

    @property
    def d(self) -> dict:
        return json.loads(self.path.read_text())

    def mem_available_bytes(self):
        return int(self.d["mem_available_bytes"])

    def mem_total_bytes(self):
        return int(self.d.get("mem_total_bytes", 64 * GIB))

    def pressure_some_avg10(self):
        return float(self.d.get("pressure_some_avg10", 0.0))

    def slice_memory_max(self):
        v = self.d.get("slice_memory_max", None)
        return None if v in (None, "infinity") else int(v)

    def slice_memory_current(self):
        return int(self.d.get("slice_memory_current", 0))

    def cgroup_current_bytes(self, cgroup):
        v = self.d.get("cgroup_current", {}).get(str(cgroup))
        return None if v is None else int(v)

    def cgroup_peak_bytes(self, cgroup):
        v = self.d.get("cgroup_peak", {}).get(str(cgroup))
        return None if v is None else int(v)

    def cgroup_alive(self, cgroup):
        return bool(self.d.get("alive", {}).get(str(cgroup), False))

    def pid_alive(self, pid, starttime=None):
        """The readings file decides for any pid it names; a pid it does not name falls back to
        the real /proc check, so an end-to-end test can use a genuine child process as the
        witness without ever pressuring memory."""
        table = self.d.get("alive", {})
        if str(pid) in table:
            return bool(table[str(pid)])
        return SystemResources().pid_alive(pid, starttime)

    def pid_starttime(self, pid):
        table = self.d.get("starttime", {})
        if str(pid) in table:
            return table[str(pid)]
        return SystemResources().pid_starttime(pid)


def resources_from_env(slice_name: str = DEFAULT_SLICE):
    p = os.environ.get("CRISPDM_ADMISSION_RESOURCES_JSON")
    return FileResources(p, slice_name) if p else SystemResources(slice_name)


def now_from_env() -> float:
    v = os.environ.get("CRISPDM_ADMISSION_NOW")
    return float(v) if v else time.time()


# ---- leases --------------------------------------------------------------------------------

@dataclass
class Lease:
    lease_id: str
    name: str
    label: str
    cap_bytes: int                  # THE integer: the reservation and the cgroup limit are the same number
    wall_seconds: int
    slice_name: str
    created_at: float
    expires_at: float
    armed: bool = False
    pid: int | None = None
    pid_starttime: int | None = None
    cgroup: str | None = None
    unit: str | None = None
    peak_bytes: int | None = None
    peak_scope: str | None = None
    peak_evidence_path: str | None = None
    peak_evidence_sha256: str | None = None
    argv_sha256: str | None = None
    host_reserve_bytes: int = DESKTOP_RESERVE_BYTES
    recovered: int = 0
    armed_at: float | None = None
    notes: list = field(default_factory=list)

    def witness_alive(self, res, now: float) -> bool:
        """Is the load this reservation is for still running?

        A reservation is written BEFORE the load exists, so the absence of a witness is not by
        itself evidence of death.  The rule, in order of how much it proves:

        * a live scope cgroup, or a live pid whose /proc start time still matches: alive;
        * a pid was recorded and that pid is gone: DEAD, immediately -- the holder's own child is
          an authoritative witness, so a short run's reservation is freed the moment it ends;
        * no pid was recorded (a detached unit whose only witness is its cgroup), or the lease is
          not armed yet: alive while inside the arming grace window, because the unit may not have
          created its cgroup yet.  This errs toward HOLDING memory, never toward over-admitting.
        """
        if self.cgroup and res.cgroup_alive(self.cgroup):
            return True
        if self.armed and self.pid:
            return bool(res.pid_alive(self.pid, self.pid_starttime))
        return now < (self.armed_at or self.created_at) + ARM_GRACE_SECONDS

    def observed_bytes(self, res):
        return res.cgroup_current_bytes(self.cgroup) if self.cgroup else None

    def unrealised_bytes(self, res) -> int:
        """The part of the reservation the load has not taken yet.  MemAvailable already
        accounts for what it HAS taken, so only this part must be subtracted again."""
        seen = self.observed_bytes(res)
        return self.cap_bytes if seen is None else max(0, self.cap_bytes - int(seen))


class Store:
    """The per-host lease directory.  One exclusive flock serialises the whole transaction."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("CRISPDM_ADMISSION_DIR")
                         or (Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
                             / "crispdm/admission"))
        self.leases = self.root / "leases"
        self.lock_path = self.root / "admission.lock"
        self.ledger_path = self.root / "ledger.jsonl"
        self.queue_path = self.root / "queue.jsonl"
        self._fh = None

    def prepare(self):
        self.leases.mkdir(parents=True, exist_ok=True)

    # -- the lock.  Held across read, decide and write: that is what makes admission atomic.
    def __enter__(self):
        self.prepare()
        self._fh = open(self.lock_path, "a+")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None
        return False

    def all_leases(self) -> list:
        out = []
        if not self.leases.exists():
            return out
        for p in sorted(self.leases.glob("*.json")):
            try:
                out.append(Lease(**json.loads(p.read_text())))
            except Exception:
                self.log({"event": "LEASE_UNREADABLE", "path": p.name})
        return out

    def path_of(self, lease_id) -> Path:
        return self.leases / f"{lease_id}.json"

    def write(self, lease: Lease):
        self.prepare()
        p = self.path_of(lease.lease_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(lease), indent=1, sort_keys=True))
        os.replace(tmp, p)                      # atomic: a reader sees the old or the new lease, never half

    def drop(self, lease_id):
        try:
            self.path_of(lease_id).unlink()
        except FileNotFoundError:
            pass

    def log(self, record: dict):
        self.prepare()
        record = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.pop("_now", now_from_env()))),
                  **record}
        with open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


# ---- the transaction ------------------------------------------------------------------------

@dataclass
class Request:
    name: str
    cap_bytes: int
    wall_seconds: int = 4 * 3600
    label: str = ""
    slice_name: str = DEFAULT_SLICE
    argv_sha256: str | None = None
    peak_bytes: int | None = None
    peak_scope: str | None = None
    peak_evidence_path: str | None = None
    peak_evidence_sha256: str | None = None
    size_text: str | None = None
    detached: bool = False      # a fire-and-forget unit: its cgroup, not a holder pid, is the witness,
                                # so the lease must outlive the process that asked for it


def read_peak_evidence(path, declared_bytes):
    """Bind a pilot footprint to its own retained record.  Returns (bytes, scope, sha256).

    The record must be JSON carrying the peak and the SCOPE it was measured at.  A
    main-process RSS is refused: an admission sized on one process under-counts the tree.
    """
    p = Path(path)
    if not p.exists():
        raise Refusal("PEAK_EVIDENCE_MISSING", f"no retained record at {p.name}")
    raw = p.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError:
        raise Refusal("PEAK_EVIDENCE_UNREADABLE", f"{p.name} is not JSON")
    scope = str(doc.get("peak_scope", "")).strip().lower()
    if scope not in TREE_PEAK_SCOPES:
        raise Refusal("PEAK_NOT_A_TREE_PEAK",
                      f"{p.name} declares peak_scope={doc.get('peak_scope')!r}; admission is sized on the "
                      f"cgroup or whole-tree peak, never one process's RSS")
    for key in ("peak_bytes", "cgroup_peak_bytes", "tree_peak_bytes"):
        if key in doc:
            found = int(doc[key])
            break
    else:
        raise Refusal("PEAK_EVIDENCE_HAS_NO_PEAK", f"{p.name} carries no peak_bytes")
    if declared_bytes is not None and int(declared_bytes) != found:
        raise Refusal("PEAK_NOT_BOUND_TO_EVIDENCE",
                      f"--peak-bytes {declared_bytes} is not the {found} the record holds")
    return found, scope, hashlib.sha256(raw).hexdigest()


class Refusal(Exception):
    """Terminal: the request never becomes admissible by waiting.  The caller must not poll."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def reclaim(store: Store, res, now: float) -> dict:
    """The sweep.  A lease is freed only when its witness is DEAD; an expired lease whose child
    is still alive is extended.  This is how a lease survives a crash of whatever held it
    without ever releasing the memory of a load that is still running."""
    freed, extended, live = [], [], []
    for lease in store.all_leases():
        alive = lease.witness_alive(res, now)
        if alive:
            live.append(lease)
            if lease.expires_at <= now:
                lease.expires_at = now + LEASE_TTL_SECONDS
                lease.recovered += 1
                lease.notes.append("EXPIRED_BUT_CHILD_ALIVE_EXTENDED")
                store.write(lease)
                extended.append(lease.lease_id)
                store.log({"_now": now, "event": "LEASE_EXTENDED_CHILD_ALIVE", "lease_id": lease.lease_id,
                           "cap_bytes": lease.cap_bytes, "pid": lease.pid, "cgroup": lease.cgroup})
        else:
            freed.append(lease.lease_id)
            store.drop(lease.lease_id)
            store.log({"_now": now, "event": "LEASE_RECLAIMED_WITNESS_DEAD", "lease_id": lease.lease_id,
                       "cap_bytes": lease.cap_bytes, "expired": lease.expires_at <= now,
                       "armed": lease.armed, "pid": lease.pid, "cgroup": lease.cgroup,
                       "observed_peak_bytes": res.cgroup_peak_bytes(lease.cgroup) if lease.cgroup else None})
    return {"freed": freed, "extended": extended, "live": live}


def evaluate(store: Store, res, req: Request, now: float) -> dict:
    """The decision, with every reading it used.  Pure with respect to the store: writes nothing."""
    swept = reclaim(store, res, now)
    live = swept["live"]

    avail = int(res.mem_available_bytes())
    total = int(res.mem_total_bytes())
    slice_max = res.slice_memory_max()
    slice_current = int(res.slice_memory_current())
    pressure = float(res.pressure_some_avg10())

    held_unrealised = sum(l.unrealised_bytes(res) for l in live)
    held_reserved = sum(l.cap_bytes for l in live)
    host_free = avail - DESKTOP_RESERVE_BYTES - held_unrealised
    aggregate_committed = slice_current + held_unrealised

    readings = {
        "mem_available_bytes": avail, "mem_total_bytes": total,
        "desktop_reserve_bytes": DESKTOP_RESERVE_BYTES,
        "slice": req.slice_name, "slice_memory_max": slice_max,
        "slice_memory_current": slice_current,
        "pressure_some_avg10": pressure, "pressure_admit_max": PRESSURE_ADMIT_MAX,
        "live_leases": len(live),
        "live_lease_ids": [l.lease_id for l in live],
        "held_reserved_bytes": held_reserved,
        "held_unrealised_bytes": held_unrealised,
        "host_free_for_new_bytes": host_free,
        "aggregate_committed_bytes": aggregate_committed,
        "request_cap_bytes": req.cap_bytes,
        "reclaimed_lease_ids": swept["freed"], "extended_lease_ids": swept["extended"],
    }

    # ceilings first: a request that fits nothing is terminal, never queued
    ceiling = total - DESKTOP_RESERVE_BYTES
    if req.cap_bytes > ceiling:
        return {"verdict": REFUSED, "code": "ABOVE_HOST_CEILING",
                "reason": f"{human(req.cap_bytes)} exceeds MemTotal minus the {human(DESKTOP_RESERVE_BYTES)} "
                          f"desktop reserve ({human(ceiling)}); it will never fit this host",
                "readings": readings}
    if slice_max is not None and req.cap_bytes > slice_max:
        return {"verdict": REFUSED, "code": "ABOVE_SLICE_CEILING",
                "reason": f"{human(req.cap_bytes)} exceeds the {req.slice_name} ceiling {human(slice_max)}",
                "readings": readings}

    # then what is free now: these are QUEUE conditions, they can change without any retry policy
    if req.peak_bytes is not None:
        need = req.peak_bytes + PILOT_MARGIN_BYTES
        readings["pilot_peak_bytes"] = req.peak_bytes
        readings["pilot_required_bytes"] = need
        if need > host_free:
            return {"verdict": QUEUED, "code": "PILOT_PEAK_DOES_NOT_FIT",
                    "reason": f"the arm's own measured {req.peak_scope} peak {human(req.peak_bytes)} plus the "
                              f"{human(PILOT_MARGIN_BYTES)} margin needs {human(need)}; "
                              f"{human(host_free)} is free for new work",
                    "readings": readings}
    if req.cap_bytes > host_free:
        return {"verdict": QUEUED, "code": "HOST_HEADROOM",
                "reason": f"{human(req.cap_bytes)} requested; {human(host_free)} free "
                          f"(MemAvailable {human(avail)} - {human(DESKTOP_RESERVE_BYTES)} desktop reserve "
                          f"- {human(held_unrealised)} held by {len(live)} live reservation(s))",
                "readings": readings}
    if slice_max is not None and aggregate_committed + req.cap_bytes > slice_max:
        return {"verdict": QUEUED, "code": "SLICE_AGGREGATE_BUDGET",
                "reason": f"the observed aggregate budget would be "
                          f"{human(aggregate_committed + req.cap_bytes)} against the {req.slice_name} ceiling "
                          f"{human(slice_max)} (in use {human(slice_current)}, "
                          f"unrealised reservations {human(held_unrealised)})",
                "readings": readings}
    if pressure > PRESSURE_ADMIT_MAX:
        return {"verdict": QUEUED, "code": "MEMORY_PRESSURE",
                "reason": f"user-slice memory PSI some/avg10 {pressure:.2f} is above the {PRESSURE_ADMIT_MAX:.2f} "
                          f"admission limit; systemd-oomd kills this slice at sustained 50%",
                "readings": readings}
    return {"verdict": ADMITTED, "code": "ADMITTED", "reason": "reserved", "readings": readings}


def acquire(store: Store, res, req: Request, now: float) -> dict:
    """One atomic admission: under the lock, decide and (on ADMITTED) write the reservation."""
    with store:
        decision = evaluate(store, res, req, now)
        if decision["verdict"] == ADMITTED:
            # the id is one filesystem segment: a caller's name may carry slashes and dots
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", req.name)[:80] or "job"
            lease = Lease(
                lease_id=f"{stem}-{int(now)}-{os.getpid()}-{os.urandom(3).hex()}",
                name=req.name, label=req.label, cap_bytes=req.cap_bytes,
                wall_seconds=req.wall_seconds, slice_name=req.slice_name,
                created_at=now,
                expires_at=now + (req.wall_seconds + LEASE_TTL_SECONDS if req.detached
                                  else LEASE_TTL_SECONDS),
                peak_bytes=req.peak_bytes, peak_scope=req.peak_scope,
                peak_evidence_path=req.peak_evidence_path,
                peak_evidence_sha256=req.peak_evidence_sha256,
                argv_sha256=req.argv_sha256)
            store.write(lease)
            decision["lease_id"] = lease.lease_id
            decision["lease"] = asdict(lease)
        store.log({"_now": now, "event": "ADMISSION_" + decision["verdict"], "name": req.name,
                   "label": req.label, "code": decision["code"], "reason": decision["reason"],
                   "cap_bytes": req.cap_bytes, "lease_id": decision.get("lease_id"),
                   "readings": decision["readings"]})
        if decision["verdict"] == QUEUED:
            with open(store.queue_path, "a") as fh:
                fh.write(json.dumps({"at": now, "name": req.name, "label": req.label,
                                     "cap_bytes": req.cap_bytes, "code": decision["code"]},
                                    sort_keys=True) + "\n")
        return decision


def arm(store: Store, res, lease_id: str, now: float, pid=None, cgroup=None, unit=None) -> dict:
    with store:
        p = store.path_of(lease_id)
        if not p.exists():
            return {"ok": False, "code": "NO_SUCH_LEASE", "lease_id": lease_id}
        lease = Lease(**json.loads(p.read_text()))
        if pid:
            lease.pid = int(pid)
            lease.pid_starttime = res.pid_starttime(pid)
        if cgroup:
            lease.cgroup = str(cgroup)
        if unit:
            lease.unit = str(unit)
        lease.armed = True
        lease.armed_at = now
        lease.expires_at = now + LEASE_TTL_SECONDS
        store.write(lease)
        store.log({"_now": now, "event": "LEASE_ARMED", "lease_id": lease_id, "pid": lease.pid,
                   "cgroup": lease.cgroup, "unit": lease.unit, "cap_bytes": lease.cap_bytes})
        return {"ok": True, "lease": asdict(lease)}


def renew(store: Store, res, lease_id: str, now: float) -> dict:
    with store:
        p = store.path_of(lease_id)
        if not p.exists():
            return {"ok": False, "code": "NO_SUCH_LEASE", "lease_id": lease_id}
        lease = Lease(**json.loads(p.read_text()))
        lease.expires_at = now + LEASE_TTL_SECONDS
        store.write(lease)
        return {"ok": True, "expires_at": lease.expires_at}


def release(store: Store, res, lease_id: str, now: float, observed_peak_bytes=None) -> dict:
    """Release, but never under a live child.  A release asked for while the witness is still
    alive RENEWS the lease and says so: that is the crash-recovery guarantee."""
    with store:
        p = store.path_of(lease_id)
        if not p.exists():
            return {"ok": True, "code": "ALREADY_RELEASED", "lease_id": lease_id}
        lease = Lease(**json.loads(p.read_text()))
        if lease.witness_alive(res, now):
            lease.expires_at = now + LEASE_TTL_SECONDS
            lease.notes.append("RELEASE_REFUSED_CHILD_STILL_ALIVE")
            store.write(lease)
            store.log({"_now": now, "event": "RELEASE_REFUSED_CHILD_ALIVE", "lease_id": lease_id,
                       "pid": lease.pid, "cgroup": lease.cgroup, "cap_bytes": lease.cap_bytes})
            return {"ok": False, "code": "CHILD_STILL_ALIVE", "lease_id": lease_id,
                    "reason": "the reservation is kept: this load's process tree is still alive"}
        peak = observed_peak_bytes
        if peak is None and lease.cgroup:
            peak = res.cgroup_peak_bytes(lease.cgroup)
        store.drop(lease_id)
        store.log({"_now": now, "event": "LEASE_RELEASED", "lease_id": lease_id, "name": lease.name,
                   "label": lease.label, "cap_bytes": lease.cap_bytes, "unit": lease.unit,
                   "observed_tree_peak_bytes": peak,
                   "peak_scope": "cgroup" if lease.cgroup else "unobserved",
                   "note": "an out-of-memory kill is terminal here; this launcher never retries with a "
                           "bigger cap"})
        return {"ok": True, "code": "RELEASED", "lease_id": lease_id, "observed_tree_peak_bytes": peak}


def state(store: Store, res, now: float) -> dict:
    with store:
        swept = reclaim(store, res, now)
        live = swept["live"]
        slice_max = res.slice_memory_max()
        avail = int(res.mem_available_bytes())
        held = sum(l.unrealised_bytes(res) for l in live)
        return {
            "store": str(store.root),
            "now": now,
            "live": [{"lease_id": l.lease_id, "name": l.name, "label": l.label,
                      "cap_bytes": l.cap_bytes, "armed": l.armed, "pid": l.pid, "unit": l.unit,
                      "cgroup": l.cgroup, "expires_at": l.expires_at,
                      "observed_bytes": l.observed_bytes(res),
                      "unrealised_bytes": l.unrealised_bytes(res)} for l in live],
            "mem_available_bytes": avail,
            "desktop_reserve_bytes": DESKTOP_RESERVE_BYTES,
            "held_unrealised_bytes": held,
            "host_free_for_new_bytes": avail - DESKTOP_RESERVE_BYTES - held,
            "slice_memory_max": slice_max,
            "slice_memory_current": res.slice_memory_current(),
            "pressure_some_avg10": res.pressure_some_avg10(),
            "reclaimed_lease_ids": swept["freed"], "extended_lease_ids": swept["extended"],
        }


# ---- in-process use: the same authority, for launch paths written in Python -------------------

class AdmissionRefused(RuntimeError):
    """Raised instead of launching.  `decision` carries every reading the verdict used."""

    def __init__(self, decision: dict):
        super().__init__(f"{decision['verdict']} {decision['code']}: {decision['reason']}")
        self.decision = decision
        self.verdict = decision["verdict"]
        self.code = decision["code"]


class Reservation:
    """The reservation, for a Python launcher that supervises its own child.

    Used exactly like the shell launcher uses the CLI: ``open()`` before the child is created,
    ``arm()`` with the child's pid and scope cgroup once it exists, and ``close()`` after the
    whole tree has finished.  There is no admit-and-forget: the reservation is written before the
    launch and it outlives the reading.

    It never retries.  ``open()`` raises AdmissionRefused for both QUEUED and REFUSED; a caller
    that is allowed to wait passes ``queue=True`` with its own bounded wait, which is waiting
    before a first start and not a retry after a rejection.
    """

    def __init__(self, *, name, cap_bytes, wall_seconds, label="", slice_name=DEFAULT_SLICE,
                 store=None, res=None, peak_bytes=None, peak_scope=None, detached=False,
                 queue=False, poll_seconds=30, max_wait_seconds=0, clock=None):
        self.store = store if isinstance(store, Store) else Store(store)
        self.res = res or resources_from_env(slice_name)
        self.clock = clock or now_from_env
        self.queue = bool(queue)
        self.poll_seconds = max(1, int(poll_seconds))
        self.max_wait_seconds = int(max_wait_seconds)
        self.request = Request(name=name, cap_bytes=int(cap_bytes), wall_seconds=int(wall_seconds),
                               label=label or name, slice_name=slice_name, peak_bytes=peak_bytes,
                               peak_scope=peak_scope, detached=bool(detached))
        self.lease_id = None
        self.decision = None

    def open(self):
        waited = 0
        while True:
            self.decision = acquire(self.store, self.res, self.request, self.clock())
            if self.decision["verdict"] == ADMITTED:
                self.lease_id = self.decision["lease_id"]
                return self
            if self.decision["verdict"] == REFUSED or not self.queue or waited >= self.max_wait_seconds:
                raise AdmissionRefused(self.decision)
            time.sleep(self.poll_seconds)
            waited += self.poll_seconds

    def arm(self, pid=None, cgroup=None, unit=None):
        if self.lease_id:
            arm(self.store, self.res, self.lease_id, self.clock(), pid=pid, cgroup=cgroup, unit=unit)
        return self

    def renew(self):
        if self.lease_id:
            renew(self.store, self.res, self.lease_id, self.clock())
        return self

    def close(self, observed_peak_bytes=None) -> dict:
        if not self.lease_id:
            return {"ok": True, "code": "NO_LEASE"}
        out = release(self.store, self.res, self.lease_id, self.clock(), observed_peak_bytes)
        if out.get("ok"):
            self.lease_id = None
        return out

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False


# ---- CLI -------------------------------------------------------------------------------------

def _argv_sha256(path, null_delimited: bool):
    if not path:
        return None
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw).hexdigest()


def build_request(a) -> Request:
    """The ONE integer.  --cap-bytes and -m/--mem must agree exactly or the request is refused."""
    size_bytes = parse_size(a.mem) if a.mem else None
    cap_bytes = int(a.cap_bytes) if a.cap_bytes else None
    if cap_bytes is None and size_bytes is None:
        raise Refusal("NO_CAP", "a request needs --mem or --cap-bytes")
    if cap_bytes is not None and size_bytes is not None and cap_bytes != size_bytes:
        raise Refusal("CONTRADICTORY_CAP",
                      f"--mem {a.mem} is {size_bytes} bytes but --cap-bytes says {cap_bytes}; "
                      f"one request carries one integer")
    cap = cap_bytes if cap_bytes is not None else size_bytes
    if cap <= 0:
        raise Refusal("NO_CAP", "the cap must be a positive number of bytes")
    peak = peak_scope = ev_sha = None
    if a.peak_evidence:
        peak, peak_scope, ev_sha = read_peak_evidence(a.peak_evidence, a.peak_bytes)
    elif a.peak_bytes:
        raise Refusal("PEAK_NOT_BOUND_TO_EVIDENCE",
                      "--peak-bytes is only accepted with --peak-evidence, the retained record it was read from")
    return Request(name=a.name, cap_bytes=cap, wall_seconds=int(a.wall or 4 * 3600), label=a.label or "",
                   slice_name=a.slice, argv_sha256=_argv_sha256(a.argv_file, True),
                   peak_bytes=peak, peak_scope=peak_scope,
                   peak_evidence_path=str(a.peak_evidence) if a.peak_evidence else None,
                   peak_evidence_sha256=ev_sha, size_text=a.mem, detached=bool(getattr(a, "detached", False)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispdm-admission", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="the lease directory (default $CRISPDM_ADMISSION_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    acq = sub.add_parser("acquire", help="admit or queue; on ADMITTED a reservation is written")
    acq.add_argument("-n", "--name", required=True)
    acq.add_argument("-m", "--mem", default=None, help="IEC size, e.g. 9G")
    acq.add_argument("--cap-bytes", type=int, default=None)
    acq.add_argument("-t", "--wall", type=int, default=None)
    acq.add_argument("--label", default="")
    acq.add_argument("--slice", default=DEFAULT_SLICE)
    acq.add_argument("--argv-file", default=None, help="NUL-delimited argv, hashed into the lease")
    acq.add_argument("--peak-bytes", type=int, default=None)
    acq.add_argument("--peak-evidence", default=None, help="the retained pilot record the peak is read from")
    acq.add_argument("--detached", action="store_true",
                     help="a fire-and-forget unit: the lease outlives the asking process and its "
                          "cgroup is the witness; it expires only after the wall limit")
    acq.add_argument("--queue", action="store_true", help="wait for admission instead of returning QUEUED")
    acq.add_argument("--poll-seconds", type=int, default=30)
    acq.add_argument("--max-wait-seconds", type=int, default=3600)

    for name, helptext in (("arm", "bind the lease to the pid/cgroup that proves the load is alive"),
                           ("renew", "heartbeat"), ("release", "release, only if the witness is dead")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("lease_id")
        if name == "arm":
            p.add_argument("--pid", type=int, default=None)
            p.add_argument("--cgroup", default=None)
            p.add_argument("--unit", default=None)
        if name == "release":
            p.add_argument("--observed-peak-bytes", type=int, default=None)

    sub.add_parser("inside-scope",
                   help="is THIS process inside a cgroup a live reservation covers?  A runner that "
                        "spawns its own fit children shares its parent's cgroup and its parent's "
                        "MemoryMax, so it needs no reservation of its own -- but it does need its "
                        "parent to have one.  Exit 0 covered, 1 not covered.")
    sub.add_parser("state", help="the live reservations and the capacity they leave")
    sub.add_parser("reclaim", help="sweep: free only the leases whose witness is dead")

    sz = sub.add_parser("size", help="the one integer an IEC size denotes (the launcher's only cap source)")
    sz.add_argument("text")
    sc = sub.add_parser("seconds", help="a systemd-style duration (4h, 900, 30m) in seconds")
    sc.add_argument("text")
    ex = sub.add_parser("explain", help="read back a decision file: a human line, or the lease id")
    ex.add_argument("path")
    ex.add_argument("--lease-id", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "size":
        print(parse_size(a.text))
        return 0
    if a.cmd == "seconds":
        print(parse_duration(a.text))
        return 0
    if a.cmd == "explain":
        try:
            d = json.loads(Path(a.path).read_text().strip().splitlines()[-1])
        except Exception:
            return 0 if a.lease_id else 1
        if a.lease_id:
            print(d.get("lease_id", ""))
        else:
            print(f"crispdm-run: {d.get('verdict')} {d.get('code')} -- {d.get('reason')}")
        return 0
    store = Store(a.store)
    res = resources_from_env(getattr(a, "slice", DEFAULT_SLICE))
    now = now_from_env()

    try:
        if a.cmd == "acquire":
            req = build_request(a)
            waited = 0
            while True:
                d = acquire(store, res, req, now_from_env() if not os.environ.get("CRISPDM_ADMISSION_NOW") else now)
                print(json.dumps(d, sort_keys=True))
                if d["verdict"] == ADMITTED:
                    return 0
                if d["verdict"] == REFUSED:
                    return REFUSED_EXIT
                if not a.queue or waited >= a.max_wait_seconds:
                    return REFUSED_EXIT
                time.sleep(max(1, a.poll_seconds))
                waited += max(1, a.poll_seconds)
        if a.cmd == "arm":
            print(json.dumps(arm(store, res, a.lease_id, now, a.pid, a.cgroup, a.unit), sort_keys=True))
            return 0
        if a.cmd == "renew":
            print(json.dumps(renew(store, res, a.lease_id, now), sort_keys=True))
            return 0
        if a.cmd == "release":
            d = release(store, res, a.lease_id, now, a.observed_peak_bytes)
            print(json.dumps(d, sort_keys=True))
            return 0 if d["ok"] else 1
        if a.cmd == "state":
            print(json.dumps(state(store, res, now), indent=1, sort_keys=True))
            return 0
        if a.cmd == "inside-scope":
            try:
                mine = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::", 1)[1]
            except (OSError, IndexError):
                mine = ""
            with store:
                live = reclaim(store, res, now)["live"]
            covering = [l.lease_id for l in live
                        if l.cgroup and mine and ("/" + str(l.cgroup).lstrip("/")) in mine + "/"]
            print(json.dumps({"cgroup": mine, "covered": bool(covering),
                              "covering_lease_ids": covering,
                              "live_lease_ids": [l.lease_id for l in live]}, sort_keys=True))
            return 0 if covering else 1
        if a.cmd == "reclaim":
            with store:
                swept = reclaim(store, res, now)
            print(json.dumps({"freed": swept["freed"], "extended": swept["extended"],
                              "live": [l.lease_id for l in swept["live"]]}, sort_keys=True))
            return 0
    except Refusal as exc:
        print(json.dumps({"verdict": REFUSED, "code": exc.code, "reason": exc.message}, sort_keys=True))
        store.log({"_now": now, "event": "ADMISSION_REFUSED", "code": exc.code, "reason": exc.message})
        return REFUSED_EXIT
    except ValueError as exc:
        print(json.dumps({"verdict": REFUSED, "code": "BAD_REQUEST", "reason": str(exc)}, sort_keys=True))
        return REFUSED_EXIT
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
