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
  ``slice charged bytes + sum(max(0, reserved - observed)) + cap <= slice MemoryMax``, not the
  child's own limit against the ceiling.  "Charged" is memory.current minus the slice's CLEAN
  file cache (file - shmem - file_dirty - file_writeback - unevictable, from memory.stat): clean
  page cache left behind by a finished scope is reclaimed by the kernel the moment a live load
  under the same ceiling needs it, so counting it as live use queued requests forever
  (ADM-DEADCACHE-01, 2026-09-30).  Anon, shmem/tmpfs, dirty/writeback, unevictable, slab and all
  other kernel memory stay charged.
* Own-scope cache at scope end.  ``scope-exec`` runs INSIDE the job's own scope, after the command
  ends, and asks the kernel to reclaim that scope's own clean file cache through the scope's own
  memory.reclaim -- never the host's, never the slice's, never another scope's.
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
NEVER_ARMED_BOUND_SECONDS = 60        # ADM-STALE-LEASE-01: a non-detached lease still unarmed after this
                                      # long, whose holder (the launcher wrapper) is dead or unknown, can
                                      # never be armed: it is reclaimed instead of holding its cap for the
                                      # full grace
PILOT_MARGIN_BYTES = GIB              # a measured pilot peak must fit with this much room to spare
DEFAULT_SLICE = "crispdm-batch.slice"
TREE_PEAK_SCOPES = ("cgroup", "tree") # the only footprint scopes that may size an admission
REFUSED_EXIT = 75                     # EX_TEMPFAIL, the code crispdm-run has always used

# ---- RR02 (order 2026-09-26): what happens AFTER admission -----------------------------------
# Admission was a gate at entry.  On 2026-09-26 at 16:27 it refused new work correctly while
# pressure climbed (ledger refusals at PSI some/avg10 38.76, 60.48, 61.20 against 25.00) and the
# two scopes already admitted ran to the end, because nothing re-examined a reservation once it
# was granted.  These constants close that, and EVERY ONE OF THEM IS DERIVED -- from the host's
# own oomd policy as recorded in docs/audits/evidence/DR01_20260926/OOM_INCIDENT_REGISTER.json,
# from DR01's existing admission limit, or from the retained pressure series in the admission
# ledger of the previous boot.  Nothing here was tuned by putting a host under pressure.
PRESSURE_OOMD_LIMIT = 50.0            # the HOST's policy, not mine: systemd-oomd killed the user
                                      # slice "being 55.11% > 50.00% for > 20s with reclaim
                                      # activity" (incidents 3 and 4 of the register).  Recorded
                                      # so the monitor can act BEFORE it, never to change it.
PRESSURE_OOMD_DURATION_SECONDS = 20   # same source: "> 20s".
# PRESSURE_ADMIT_MAX (25.00) is DR01's existing limit, half the oomd limit: the level below which
# the host is calm enough to admit NEW work.  The response threshold is the midpoint of the only
# two thresholds policy already declares.  It must be strictly above the admission limit -- a load
# legitimately admitted at 24.5 must not stop itself the instant it starts -- and strictly below
# the oomd limit, so the response happens before the kill.
PRESSURE_RESPOND_AT = (PRESSURE_ADMIT_MAX + PRESSURE_OOMD_LIMIT) / 2.0            # 37.50
PRESSURE_RESPOND_WINDOW_SECONDS = PRESSURE_OOMD_DURATION_SECONDS                  # 20
PRESSURE_SAMPLE_SECONDS = 5           # the launcher's existing heartbeat period; unchanged
PRESSURE_RESPOND_MIN_SAMPLES = 4      # 20s / 5s: a window is never judged on fewer
# The oscillation path.  In the retained series the first sample above 37.50 is 47.85 at
# 21:18:21Z and the first owner application was killed at 21:27:10Z: 529 s of elevation ending in
# an owner-visible kill, during which pressure fell below 25.00 on single samples seven times.
# The budget is half that observed interval, so a response happens at least halfway along the
# path the host was actually observed to take.
PRESSURE_ELEVATED_BUDGET_SECONDS = 240
# Hysteresis.  The longest run of CONSECUTIVE samples at or below 25.00 inside that window is
# three samples spanning 60 s (21:15:51 -> 21:16:51Z), after which pressure returned to 47.85
# within 90 s: 60 s of calm was observed to be false recovery.  The recovery window is twice that
# observed false-recovery span.  The crossing the order names, 52.18 -> 24.5, is ONE sample and
# fails this by a factor of four.
PRESSURE_RECOVERY_WINDOW_SECONDS = 120
PRESSURE_RECOVERY_MIN_SAMPLES = PRESSURE_RECOVERY_WINDOW_SECONDS // PRESSURE_SAMPLE_SECONDS   # 24
PRESSURE_STOP_TERM_GRACE_SECONDS = 30 # the launcher's existing `timeout --kill-after=30s` grace:
                                      # a pressure stop gets the same courtesy the wall limit has

CALM = "CALM"
ELEVATED = "ELEVATED"
RULE_SUSTAINED = "SUSTAINED_ABOVE_RESPOND"
RULE_BUDGET = "ELEVATED_BUDGET_EXHAUSTED"

# A refusal is answered by waiting or by moving the work to an admitted host -- never by asking
# again for less than the work needs.  A name that was refused remembers that refusal for as long
# as the launcher's own default bounded wait (-W 3600), after which the refusal is stale.
LOWERED_CAP_MEMORY_SECONDS = 3600

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

def clean_file_bytes(stat: dict) -> int:
    """Clean, reclaimable page cache in a memory.stat reading: file pages that are neither
    shmem/tmpfs, dirty, under writeback nor unevictable.  Never negative."""
    g = lambda k: int(stat.get(k, 0) or 0)  # noqa: E731
    return max(0, g("file") - g("shmem") - g("file_dirty") - g("file_writeback") - g("unevictable"))


def parse_memory_stat(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


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

    def pressure_full_avg10(self) -> float:
        """The field systemd-oomd itself acts on.  Recorded beside `some` so a stop can be read
        against the host's own policy; the decision still uses `some`, which is never lower."""
        for candidate in (self._user_slice_pressure(), Path("/proc/pressure/memory")):
            if candidate and candidate.exists():
                for line in candidate.read_text().splitlines():
                    if line.startswith("full "):
                        for field_ in line.split():
                            if field_.startswith("avg10="):
                                return float(field_.split("=", 1)[1])
        return 0.0

    def cgroup_pressure_some_avg10(self, cgroup):
        """This load's OWN memory pressure.  A reservation must be answerable for the cgroup it
        covers, not only for the host, so both are sampled."""
        if not cgroup:
            return None
        f = self.cgroup_root / str(cgroup).lstrip("/") / "memory.pressure"
        try:
            for line in f.read_text().splitlines():
                if line.startswith("some "):
                    for field_ in line.split():
                        if field_.startswith("avg10="):
                            return float(field_.split("=", 1)[1])
        except OSError:
            return None
        return None

    # -- identity.  A pid start time is measured in clock ticks SINCE BOOT, so it is meaningless
    # across a reboot; and a lease written on one host says nothing about another.
    def boot_id(self):
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            return None

    def boot_time(self):
        """CLOCK_REALTIME seconds at boot (/proc/stat btime).  With it, a pid's start time in
        ticks becomes an absolute instant and can be compared at all."""
        try:
            for line in Path("/proc/stat").read_text().splitlines():
                if line.startswith("btime "):
                    return int(line.split()[1])
        except (OSError, IndexError, ValueError):
            return None
        return None

    def host_key(self):
        """An OPAQUE per-host key: sha256 of the machine id, truncated.  Never a host name, an
        address or an account identifier -- those may not be written anywhere in this repository."""
        for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                raw = Path(p).read_text().strip()
            except OSError:
                continue
            if raw:
                return hashlib.sha256(raw.encode()).hexdigest()[:16]
        return None

    def cgroup_procs(self, cgroup) -> list:
        """The pids this cgroup holds, for a response that may touch NOTHING else."""
        if not cgroup:
            return []
        f = self.cgroup_root / str(cgroup).lstrip("/") / "cgroup.procs"
        try:
            return [int(x) for x in f.read_text().split() if x.isdigit()]
        except OSError:
            return []

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

    def slice_memory_stat(self):
        p = self.slice_cgroup_path()
        try:
            return parse_memory_stat((p / "memory.stat").read_text()) if p else None
        except OSError:
            return None

    def cgroup_memory_stat(self, cgroup):
        if not cgroup:
            return None
        try:
            return parse_memory_stat((self.cgroup_root / str(cgroup).lstrip("/") / "memory.stat").read_text())
        except OSError:
            return None

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
        """Does this cgroup still hold a RUNNING task?

        `cgroup.procs` being non-empty is not the question: during a scope's teardown it can
        still list tasks that are already zombies, and a zombie holds no memory.  Reading it as
        "alive" made a launcher refuse to release its own reservation the instant its load
        finished, so the reservation leaked until a later sweep.  Each listed task is checked.
        """
        if not cgroup:
            return False
        procs = self.cgroup_root / str(cgroup).lstrip("/") / "cgroup.procs"
        try:
            pids = procs.read_text().split()
        except OSError:
            return False
        return any(self.pid_alive(pid) for pid in pids)

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

    def pressure_full_avg10(self):
        return float(self.d.get("pressure_full_avg10", 0.0))

    def cgroup_pressure_some_avg10(self, cgroup):
        v = self.d.get("cgroup_pressure", {}).get(str(cgroup))
        return None if v is None else float(v)

    def boot_id(self):
        return self.d.get("boot_id", "simulated-boot-0")

    def boot_time(self):
        return self.d.get("boot_time", 1790000000)

    def host_key(self):
        return self.d.get("host_key", "simulatedhost0000")

    def cgroup_procs(self, cgroup):
        return [int(x) for x in self.d.get("cgroup_procs", {}).get(str(cgroup), [])]

    def slice_memory_max(self):
        v = self.d.get("slice_memory_max", None)
        return None if v in (None, "infinity") else int(v)

    def slice_memory_current(self):
        return int(self.d.get("slice_memory_current", 0))

    def slice_memory_stat(self):
        return self.d.get("slice_memory_stat")

    def cgroup_memory_stat(self, cgroup):
        return self.d.get("cgroup_stat", {}).get(str(cgroup))

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
    # RR02: the identity a pid and a start time are only meaningful inside.  `pid_starttime` is in
    # clock ticks since boot, so without `boot_id` a post-reboot process can in principle match a
    # dead lease's pid and start time and make that lease read as live; and without `host_key` a
    # coordinator's sweep cannot tell a worker's lease from its own.
    boot_id: str | None = None
    boot_time: int | None = None
    host_key: str | None = None
    # ADM-STALE-LEASE-01: the launcher wrapper that asked for this lease and will arm it
    holder_pid: int | None = None
    holder_starttime: int | None = None
    detached: bool = False

    def never_armed_and_orphaned(self, res, now: float) -> bool:
        """Unarmed, not detached, older than NEVER_ARMED_BOUND_SECONDS, and its holder is dead (or
        was never recorded): nobody is left to arm it."""
        if self.armed or self.detached or self.cgroup:
            return False
        if now < self.created_at + NEVER_ARMED_BOUND_SECONDS:
            return False
        if self.holder_pid is None:
            return True
        return not res.pid_alive(self.holder_pid, self.holder_starttime)

    # -- identity questions, each answered before liveness is even asked
    def foreign_host(self, res) -> bool:
        """A lease written on ANOTHER host.  This sweep may not judge it and may never reclaim it:
        a coordinator reboot says nothing about a worker's load.  It is also not counted against
        this host's capacity, because the bytes it reserves are not this host's bytes."""
        mine = res.host_key() if hasattr(res, "host_key") else None
        return bool(self.host_key and mine and self.host_key != mine)

    def old_boot(self, res) -> bool:
        """A lease written before this host booted.  Its local load cannot have survived, and its
        pid/start-time witness is not comparable across the boot, so it is never read as live."""
        if self.foreign_host(res):
            return False
        mine = res.boot_id() if hasattr(res, "boot_id") else None
        return bool(self.boot_id and mine and self.boot_id != mine)

    def witness_alive(self, res, now: float) -> bool:
        """Is the load this reservation is for still running?

        A reservation is written BEFORE the load exists, so the absence of a witness is not by
        itself evidence of death.  The witnesses, in the order of how much each proves:

        1. **The holder's own child pid**, pinned to its /proc start time so a recycled pid cannot
           impersonate it.  When a holder supervises its child and has reaped it, the load is over
           by definition, and the reservation must be freed at that instant.  Consulting the scope
           cgroup as well was wrong here and was observed to be wrong: for a few milliseconds
           after the child is reaped the cgroup still lists tasks that are winding down, so the
           launcher refused to release its own reservation and it leaked until a later sweep.
        2. **This load's own scope cgroup**, for a DETACHED unit that no holder waits for (the
           ssh dispatcher's fire-and-forget services).  It must be the unit's own cgroup and not
           an ancestor the child merely inherited, and it counts only while it holds a task that
           is not already a zombie.
        3. **The arming grace window**, while a lease has no witness yet -- it is written before
           the unit exists, and a unit takes a moment to create its cgroup.  This errs toward
           HOLDING memory, never toward over-admitting.
        4. **Boot and host identity first.**  A lease from an earlier boot of THIS host holds no
           live load -- the reboot ended it -- and its pid/start-time witness is not comparable,
           so it is never read as live (and its body is kept as a historical record, not deleted).
           A lease from ANOTHER host is not judged here at all: its liveness is not this host's
           question, and a coordinator reboot must never reclaim a worker's lease.
        """
        if self.foreign_host(res):
            return False                    # not judged here; reclaim() also never removes it
        if self.old_boot(res):
            return False                    # the boot that held this load is gone
        if self.armed and self.pid:
            if res.pid_alive(self.pid, self.pid_starttime):
                return True
            # RR02, and a deliberate reversal of one DR01 decision, reported as such.  DR01 read
            # the holder's reaped child as the whole answer, because the scope cgroup still listed
            # tasks that were winding down and the launcher therefore refused to release its own
            # reservation.  That symptom is now handled where it belongs -- `cgroup_alive` checks
            # each listed task and a zombie holds no memory -- and reading the pid alone leaves the
            # case the order names: a DETACHED DESCENDANT still running in this load's own scope
            # after the direct child exited.  Releasing there would hand a second admission the
            # bytes a live descendant is using.  So a dead pid is not the end while this load's own
            # scope still holds a live task.  The launcher waits a bounded moment for the scope to
            # drain, so the ordinary case still frees the reservation at once.
            if self.cgroup and self._cgroup_is_my_own() and res.cgroup_alive(self.cgroup):
                return True
            return False
        if self.cgroup and self._cgroup_is_my_own() and res.cgroup_alive(self.cgroup):
            return True
        if self.never_armed_and_orphaned(res, now):
            return False
        return now < (self.armed_at or self.created_at) + ARM_GRACE_SECONDS

    def _cgroup_is_my_own(self) -> bool:
        """A witness must be THIS load's own cgroup, never an ancestor it merely inherited.

        A runner whose child shares its parent's scope (the PRLIMIT_AS fallback, or any tool that
        spawns a plain subprocess) would otherwise record the enclosing scope as its witness -- a
        cgroup that stays alive for as long as everything else in it does, so the reservation
        could never be released.
        """
        if not self.unit:
            return True                     # cgroup-only witness: nothing better to compare with
        return str(self.cgroup).rstrip("/").endswith("/" + str(self.unit).lstrip("/"))

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
        self.retained = self.root / "retained"          # RR02: lease BODIES, kept after the end
        self.requests = self.root / "requests"          # RR02: what was asked for, and refused
        self.incidents = self.root / "incidents"        # RR02: why a load was stopped
        self.lock_path = self.root / "admission.lock"
        self.ledger_path = self.root / "ledger.jsonl"
        self.queue_path = self.root / "queue.jsonl"
        self._fh = None

    def prepare(self):
        self.leases.mkdir(parents=True, exist_ok=True)
        self.retained.mkdir(parents=True, exist_ok=True)

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

    # -- RR02: a lease that ends keeps its BODY ------------------------------------------------
    def retire(self, lease: Lease, cause: str, now: float, **extra) -> Path:
        """End a lease WITHOUT destroying the record of what it was.

        Before RR02 the store deleted the lease file and kept only a ledger line carrying the
        lease id, so cap, cgroup, argv digest, wall and expiry -- exactly the fields an incident
        review needs -- were gone.  Two bodies survived the 2026-09-26 reboot only because they
        were transcribed by hand into
        docs/audits/evidence/RR01_RESTART_20260926/INTERRUPTED_ATTEMPTS.json.  Now every end
        writes the whole body here first, and only then removes it from the live set.
        """
        self.prepare()
        body = {"schema": "crispdm.retained_lease.v1", "reclaim_cause": cause,
                "retired_at_epoch": now, "lease": asdict(lease), **extra}
        p = self.retained / f"{lease.lease_id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
        os.replace(tmp, p)
        self.drop(lease.lease_id)
        return p

    def retained_body(self, lease_id):
        p = self.retained / f"{lease_id}.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    # -- RR02: the register that makes a lowered re-ask visible ---------------------------------
    def request_path(self, name) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))[:80] or "job"
        return self.requests / f"{stem}.json"

    def read_request(self, name):
        try:
            return json.loads(self.request_path(name).read_text())
        except (OSError, ValueError):
            return None

    def write_request(self, name, record: dict):
        self.requests.mkdir(parents=True, exist_ok=True)
        p = self.request_path(name)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
        os.replace(tmp, p)

    def write_incident(self, lease_id, record: dict) -> Path:
        self.incidents.mkdir(parents=True, exist_ok=True)
        p = self.incidents / f"{lease_id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True))
        os.replace(tmp, p)
        return p

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
    holder_pid: int | None = None   # ADM-STALE-LEASE-01: the launcher wrapper that will arm the lease
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
    without ever releasing the memory of a load that is still running.

    RR02 adds the two identity questions that come before liveness, and keeps every body:

    * a lease from ANOTHER host is **never reclaimed and never judged** -- a coordinator reboot
      must not touch a worker's lease -- and it does not count against this host's capacity,
      because its bytes are not this host's bytes;
    * a lease from an earlier boot of THIS host is retired as a **historical record**: the reboot
      ended its local load, so its reservation is released, but its body is preserved and the
      cause says which of the two things happened;
    * every end -- old boot, dead witness, release -- writes the body to ``retained/`` first.
    """
    freed, extended, live, foreign, old_boot = [], [], [], [], []
    for lease in store.all_leases():
        if lease.foreign_host(res):
            foreign.append(lease.lease_id)
            if "FOREIGN_HOST_NOT_JUDGED" not in lease.notes:
                lease.notes.append("FOREIGN_HOST_NOT_JUDGED")
                store.write(lease)
                store.log({"_now": now, "event": "LEASE_FOREIGN_HOST_NOT_JUDGED",
                           "lease_id": lease.lease_id, "cap_bytes": lease.cap_bytes,
                           "note": "written on another host; this sweep neither reclaims it nor "
                                   "counts it against this host's capacity"})
            continue
        if lease.old_boot(res):
            old_boot.append(lease.lease_id)
            store.retire(lease, "LEASE_RETIRED_OLD_BOOT", now,
                         boot_id_at_retirement=res.boot_id() if hasattr(res, "boot_id") else None,
                         observed_peak_bytes=res.cgroup_peak_bytes(lease.cgroup) if lease.cgroup else None,
                         reading="its boot ended its local load; pid_starttime is ticks since boot "
                                 "and is not comparable across a reboot, so no post-reboot process "
                                 "was allowed to witness it")
            store.log({"_now": now, "event": "LEASE_RETIRED_OLD_BOOT", "lease_id": lease.lease_id,
                       "cap_bytes": lease.cap_bytes, "lease_boot_id": lease.boot_id,
                       "armed": lease.armed, "pid": lease.pid, "cgroup": lease.cgroup,
                       "body_retained": str(store.retained / f"{lease.lease_id}.json")})
            continue
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
            cause = ("LEASE_RECLAIMED_NEVER_ARMED" if lease.never_armed_and_orphaned(res, now)
                     else "LEASE_RECLAIMED_UNARMED_GRACE_EXPIRED" if not lease.armed
                     else "LEASE_RECLAIMED_WITNESS_DEAD")
            peak = res.cgroup_peak_bytes(lease.cgroup) if lease.cgroup else None
            store.retire(lease, cause, now, observed_peak_bytes=peak,
                         expired=lease.expires_at <= now)
            store.log({"_now": now, "event": "LEASE_RECLAIMED_WITNESS_DEAD", "lease_id": lease.lease_id,
                       "cap_bytes": lease.cap_bytes, "expired": lease.expires_at <= now,
                       "armed": lease.armed, "pid": lease.pid, "cgroup": lease.cgroup,
                       "reclaim_cause": cause,
                       "observed_peak_bytes": peak,
                       "body_retained": str(store.retained / f"{lease.lease_id}.json")})
    return {"freed": freed, "extended": extended, "live": live,
            "foreign": foreign, "old_boot": old_boot}


def slice_charged_bytes(res, live, slice_current=None):
    """(charged, clean bytes inside live scopes).  charged = memory.current minus the slice's DEAD
    clean file cache, i.e. clean cache outside every live leased scope.

    Review 2026-10-01 (owner cases "live loads" and "reservation"): a live scope's own cache is
    already inside its observed bytes, and its unrealised reservation (cap - observed) must not
    shrink because that cache was uncharged, or cap-sized headroom would be handed out twice.  An
    unreadable live-scope stat counts all of the slice's clean cache as live (nothing uncharged).
    An unreadable slice stat charges everything.  This is bookkeeping against the slice ceiling,
    never RAM: the host gate still uses MemAvailable minus the desktop reserve and reservations."""
    cur = int(res.slice_memory_current()) if slice_current is None else int(slice_current)
    st = res.slice_memory_stat() if hasattr(res, "slice_memory_stat") else None
    if st is None:
        return cur, 0
    total_clean = clean_file_bytes(st)
    live_clean = 0
    for l in live:
        if not l.cgroup:
            continue          # not armed to a scope yet: it holds no pages; its whole cap is unrealised
        cs = res.cgroup_memory_stat(l.cgroup) if hasattr(res, "cgroup_memory_stat") else None
        if cs is None:
            live_clean = total_clean
            break
        live_clean += clean_file_bytes(cs)
    dead = max(0, total_clean - live_clean)
    return max(0, cur - dead), min(live_clean, total_clean)


def evaluate(store: Store, res, req: Request, now: float) -> dict:
    """The decision, with every reading it used.  Pure with respect to the store: writes nothing."""
    swept = reclaim(store, res, now)
    live = swept["live"]

    avail = int(res.mem_available_bytes())
    total = int(res.mem_total_bytes())
    slice_max = res.slice_memory_max()
    slice_current = int(res.slice_memory_current())
    slice_charged, live_clean = slice_charged_bytes(res, live, slice_current)
    pressure = float(res.pressure_some_avg10())

    held_unrealised = sum(l.unrealised_bytes(res) for l in live)
    held_reserved = sum(l.cap_bytes for l in live)
    host_free = avail - DESKTOP_RESERVE_BYTES - held_unrealised
    aggregate_committed = slice_charged + held_unrealised

    readings = {
        "mem_available_bytes": avail, "mem_total_bytes": total,
        "desktop_reserve_bytes": DESKTOP_RESERVE_BYTES,
        "slice": req.slice_name, "slice_memory_max": slice_max,
        "slice_memory_current": slice_current,
        "slice_charged_bytes": slice_charged,
        "slice_clean_file_bytes_not_charged": slice_current - slice_charged,
        "slice_clean_file_bytes_in_live_scopes": live_clean,
        "pressure_some_avg10": pressure, "pressure_admit_max": PRESSURE_ADMIT_MAX,
        "live_leases": len(live),
        "live_lease_ids": [l.lease_id for l in live],
        "held_reserved_bytes": held_reserved,
        "held_unrealised_bytes": held_unrealised,
        "host_free_for_new_bytes": host_free,
        "aggregate_committed_bytes": aggregate_committed,
        "request_cap_bytes": req.cap_bytes,
        "reclaimed_lease_ids": swept["freed"], "extended_lease_ids": swept["extended"],
        "old_boot_retired_lease_ids": swept.get("old_boot", []),
        "foreign_host_lease_ids": swept.get("foreign", []),
        "pressure_full_avg10": (float(res.pressure_full_avg10())
                                if hasattr(res, "pressure_full_avg10") else None),
        "boot_id": res.boot_id() if hasattr(res, "boot_id") else None,
        "host_key": res.host_key() if hasattr(res, "host_key") else None,
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
                          f"{human(slice_max)} (charged {human(slice_charged)} = in use {human(slice_current)} "
                          f"minus {human(slice_current - slice_charged)} dead clean file cache outside live scopes, "
                          f"unrealised reservations {human(held_unrealised)})",
                "readings": readings}
    if pressure > PRESSURE_ADMIT_MAX:
        return {"verdict": QUEUED, "code": "MEMORY_PRESSURE",
                "reason": f"user-slice memory PSI some/avg10 {pressure:.2f} is above the {PRESSURE_ADMIT_MAX:.2f} "
                          f"admission limit; systemd-oomd kills this slice at sustained 50%",
                "readings": readings}
    return {"verdict": ADMITTED, "code": "ADMITTED", "reason": "reserved", "readings": readings}


def lowered_after_refusal(store: Store, req: Request, now: float):
    """RR02 defect 4.  A refused request answered by asking again for LESS.

    The retained queue log of 2026-09-26 alternates cap 3 GiB and 1 GiB under one name while the
    host was under pressure.  Lowering a request below its measured need in order to pass a gate
    is forbidden: it converts a refusal into an under-capped launch, which is the same defect the
    cap exists to prevent.  The only two correct answers to a refusal are to WAIT or to move the
    work to a host that admits it.

    Returns a Refusal-shaped dict when this request is a lowered re-ask of a refused one, else
    None.  Genuinely different, smaller work is not blocked: it is asked to carry its own name,
    which costs nothing and keeps the record readable.  A name whose earlier request was ADMITTED
    never poisons a later smaller one -- nothing was dodged there.
    """
    prior = store.read_request(req.name)
    if not prior:
        return None
    refused = prior.get("last_refused") or {}
    cap = refused.get("cap_bytes")
    at = refused.get("at_epoch")
    if not cap or at is None:
        return None
    if now - float(at) > LOWERED_CAP_MEMORY_SECONDS:
        return None                                  # the refusal is stale; waiting has expired
    if int(req.cap_bytes) >= int(cap):
        return None
    same_argv = bool(req.argv_sha256 and req.argv_sha256 == refused.get("argv_sha256"))
    return {
        "code": "CAP_LOWERED_AFTER_REFUSAL",
        "reason": (f"{human(req.cap_bytes)} is LESS than the {human(cap)} this name was refused "
                   f"{int(now - float(at))}s ago ({refused.get('code')}); "
                   + ("the command is byte-identical to the refused one. " if same_argv else
                      "the command differs from the refused one. ")
                   + "A refusal is answered by waiting for capacity or by moving the work to a "
                     "host that admits it, never by asking for less than the work needs. If this "
                     "is genuinely different, smaller work, give it its own -n name."),
        "prior_refusal": refused, "same_argv": same_argv,
    }


def _note_request(store: Store, req: Request, decision: dict, now: float):
    """The register the check above reads.  One record per name; refusals are what it remembers."""
    rec = store.read_request(req.name) or {"schema": "crispdm.request_register.v1", "name": req.name}
    entry = {"at_epoch": now, "cap_bytes": req.cap_bytes, "verdict": decision["verdict"],
             "code": decision["code"], "label": req.label, "argv_sha256": req.argv_sha256}
    rec["last"] = entry
    rec["seen"] = int(rec.get("seen", 0)) + 1
    rec["max_cap_bytes_ever_asked"] = max(int(rec.get("max_cap_bytes_ever_asked", 0)), int(req.cap_bytes))
    if decision["verdict"] in (QUEUED, REFUSED):
        # A CAP_LOWERED refusal must NOT become the remembered refusal: if it did, the remembered
        # cap would drift downwards with every lowered re-ask and the rule would erode itself.
        if decision["code"] != "CAP_LOWERED_AFTER_REFUSAL":
            rec["last_refused"] = entry
    elif decision["verdict"] == ADMITTED:
        rec["last_admitted"] = entry
        rec.pop("last_refused", None)     # capacity was found: the refusal no longer stands
    store.write_request(req.name, rec)


def acquire(store: Store, res, req: Request, now: float) -> dict:
    """One atomic admission: under the lock, decide and (on ADMITTED) write the reservation."""
    with store:
        lowered = lowered_after_refusal(store, req, now)
        if lowered:
            decision = {"verdict": REFUSED, "code": lowered["code"], "reason": lowered["reason"],
                        "readings": {"prior_refusal": lowered["prior_refusal"],
                                     "same_argv_as_refused": lowered["same_argv"],
                                     "request_cap_bytes": req.cap_bytes}}
            _note_request(store, req, decision, now)
            store.log({"_now": now, "event": "ADMISSION_REFUSED", "name": req.name,
                       "label": req.label, "code": decision["code"], "reason": decision["reason"],
                       "cap_bytes": req.cap_bytes, "readings": decision["readings"]})
            return decision
        decision = evaluate(store, res, req, now)
        holder_start = None
        if req.holder_pid is not None:
            # ADM-STALE-LEASE-01: checked under the lock, immediately before the commit.  A holder
            # that died while this acquirer waited (e.g. TERM to a queued wrapper) can never arm
            # the lease, so none is written.
            holder_start = res.pid_starttime(req.holder_pid)
            if not res.pid_alive(req.holder_pid, holder_start):
                decision = {"verdict": REFUSED, "code": "HOLDER_GONE",
                            "reason": f"the launcher that asked for this reservation (pid {req.holder_pid}) "
                                      f"is gone; no lease is written",
                            "readings": decision["readings"]}
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
                argv_sha256=req.argv_sha256,
                # RR02: the identity that makes pid + pid_starttime mean anything at all
                boot_id=res.boot_id() if hasattr(res, "boot_id") else None,
                boot_time=res.boot_time() if hasattr(res, "boot_time") else None,
                host_key=res.host_key() if hasattr(res, "host_key") else None,
                holder_pid=req.holder_pid, holder_starttime=holder_start, detached=bool(req.detached))
            store.write(lease)
            decision["lease_id"] = lease.lease_id
            decision["lease"] = asdict(lease)
        _note_request(store, req, decision, now)
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


def release(store: Store, res, lease_id: str, now: float, observed_peak_bytes=None,
            witness_pid=None) -> dict:
    """Release, but never under a live child.  A release asked for while the witness is still
    alive RENEWS the lease and says so: that is the crash-recovery guarantee.

    ``witness_pid`` is for the one case a holder cannot otherwise answer: its lease could not be
    armed, so the lease carries no witness at all and is protected by the arming grace, yet the
    holder has stopped the child it started and knows the load is over.  The pid is not taken on
    trust -- it is CHECKED.  If that process is still alive the release is refused exactly as any
    other; only a dead pid, on a lease that was never armed, releases early.
    """
    with store:
        p = store.path_of(lease_id)
        if not p.exists():
            return {"ok": True, "code": "ALREADY_RELEASED", "lease_id": lease_id}
        lease = Lease(**json.loads(p.read_text()))
        if witness_pid and not lease.armed and not lease.cgroup:
            if res.pid_alive(int(witness_pid)):
                lease.expires_at = now + LEASE_TTL_SECONDS
                lease.notes.append("RELEASE_REFUSED_UNARMED_WITNESS_PID_ALIVE")
                store.write(lease)
                store.log({"_now": now, "event": "RELEASE_REFUSED_CHILD_ALIVE", "lease_id": lease_id,
                           "witness_pid": int(witness_pid), "cap_bytes": lease.cap_bytes})
                return {"ok": False, "code": "CHILD_STILL_ALIVE", "lease_id": lease_id,
                        "reason": "the reservation is kept: the pid offered as the witness is alive"}
            lease.notes.append("RELEASED_UNARMED_WITNESS_PID_DEAD")
            peak = observed_peak_bytes
            store.retire(lease, "LEASE_RELEASED_NEVER_ARMED", now, observed_peak_bytes=peak,
                         witness_pid=int(witness_pid))
            store.log({"_now": now, "event": "LEASE_RELEASED", "lease_id": lease_id,
                       "name": lease.name, "label": lease.label, "cap_bytes": lease.cap_bytes,
                       "unit": lease.unit, "observed_tree_peak_bytes": peak,
                       "peak_scope": "unobserved",
                       "code": "LEASE_RELEASED_NEVER_ARMED", "witness_pid": int(witness_pid),
                       "note": "the lease could never be armed; the holder's own child pid was "
                               "checked dead before the bytes were given back"})
            return {"ok": True, "code": "RELEASED_NEVER_ARMED", "lease_id": lease_id,
                    "observed_tree_peak_bytes": peak}
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
        # RR02: the body is kept.  A released lease is as much an incident-review record as a
        # reclaimed one -- the cap, the cgroup, the argv digest and the wall it ran under are the
        # fields a later review needs, and the ledger line alone does not carry them.
        store.retire(lease, "LEASE_RELEASED", now, observed_peak_bytes=peak)
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
            "slice_charged_bytes": slice_charged_bytes(res, live)[0],
            "pressure_some_avg10": res.pressure_some_avg10(),
            "pressure_full_avg10": (res.pressure_full_avg10()
                                    if hasattr(res, "pressure_full_avg10") else None),
            "pressure_admit_max": PRESSURE_ADMIT_MAX,
            "pressure_respond_at": PRESSURE_RESPOND_AT,
            "boot_id": res.boot_id() if hasattr(res, "boot_id") else None,
            "host_key": res.host_key() if hasattr(res, "host_key") else None,
            "reclaimed_lease_ids": swept["freed"], "extended_lease_ids": swept["extended"],
            "old_boot_retired_lease_ids": swept.get("old_boot", []),
            "foreign_host_lease_ids": swept.get("foreign", []),
            "retained_bodies": len(list(store.retained.glob("*.json"))) if store.retained.exists() else 0,
        }


# ---- RR02: the monitor that watches a reservation AFTER it was granted ------------------------

class PressureMonitor:
    """Pure decision logic: a series of pressure samples in, a verdict out.

    This is the defect RR01 named RR-C: "admission is a gate at entry with no monitor after it".
    On the previous boot two scopes, each inside its own 3 GiB cap, kept running while the user
    slice's PSI rose past 60 and systemd-oomd killed the owner's browser and then the owner's
    editor.  Admission refused everything NEW correctly.  Nothing re-examined what was already
    admitted.

    Two rules stop a load, both derived (see the constants above), and one rule -- hysteresis --
    decides when the host has actually recovered:

    * ``SUSTAINED_ABOVE_RESPOND``: every sample in the trailing 20 s (at least 4 of them) is above
      the 37.50 midpoint of the two thresholds policy declares.  The fast path: a genuine,
      unambiguous crossing.
    * ``ELEVATED_BUDGET_EXHAUSTED``: cumulative time spent above the 25.00 admission limit since
      elevation began, without a CONFIRMED recovery, exceeds 240 s.  This is the path the host was
      actually observed to take: pressure oscillated between 4.9 and 74.96 for 529 s and then the
      owner lost two applications.  A rule that demanded every sample be high would have watched
      that happen.
    * recovery is confirmed only by 120 s (>= 24 samples) with NO sample above 25.00.  The
      crossing the order names -- 52.18 down to 24.5 -- is one sample, and the sample after it was
      7.04 and the one after that 54.03.  It does not clear anything.

    What it may do about it: stop ITS OWN identified experiment scope, keeping partial evidence
    and recording the exit cause.  It never targets another lease, another cgroup, a process that
    is not in its own scope's ``cgroup.procs``, an unrelated user application or any service.  It
    never changes a limit, a ceiling, a swap setting or oomd.
    """

    def __init__(self, *, respond_at=PRESSURE_RESPOND_AT, admit_max=PRESSURE_ADMIT_MAX,
                 respond_window=PRESSURE_RESPOND_WINDOW_SECONDS,
                 respond_min_samples=PRESSURE_RESPOND_MIN_SAMPLES,
                 budget_seconds=PRESSURE_ELEVATED_BUDGET_SECONDS,
                 recovery_window=PRESSURE_RECOVERY_WINDOW_SECONDS,
                 recovery_min_samples=PRESSURE_RECOVERY_MIN_SAMPLES):
        self.respond_at = float(respond_at)
        self.admit_max = float(admit_max)
        self.respond_window = float(respond_window)
        self.respond_min_samples = int(respond_min_samples)
        self.budget_seconds = float(budget_seconds)
        self.recovery_window = float(recovery_window)
        self.recovery_min_samples = int(recovery_min_samples)
        self.state = CALM
        self.samples = []                 # (at, host_pressure, cgroup_pressure)
        self.elevated_since = None
        self.elevated_seconds = 0.0
        self.recovery_since = None
        self.recovery_samples = 0
        self.recoveries_confirmed = 0
        self.last_at = None
        self.last_pressure = None
        self.verdict = None

    # -- the series
    def feed(self, at: float, host_pressure: float, cgroup_pressure=None) -> dict:
        """One sample.  Returns {"stop": bool, "rule": str|None, ...}; never signals anything."""
        at = float(at)
        host_pressure = float(host_pressure)
        delta = 0.0 if self.last_at is None else max(0.0, at - float(self.last_at))
        prev = self.last_pressure
        self.last_at = at
        self.last_pressure = host_pressure
        self.samples.append((at, host_pressure, cgroup_pressure))
        # The interval between two samples counts as elevated only when BOTH of its ends are above
        # the admission limit.  Crediting an interval whose earlier end was calm would charge a
        # load for time the host was not under pressure -- in the retained series the sample before
        # the first crossing was 16.07 -- and a monitor that over-counts stops good runs.
        elevated_interval = (delta if (host_pressure > self.admit_max
                                       and prev is not None and prev > self.admit_max) else 0.0)

        if self.state == CALM and host_pressure > self.respond_at:
            self.state = ELEVATED
            self.elevated_since = at
            self.elevated_seconds = 0.0
            self._cancel_recovery()

        if self.state == ELEVATED:
            if host_pressure > self.admit_max:
                # Time spent above the level at which the host is calm enough to admit new work
                # is time this load is riding pressure.  A single dip does not give it back.
                self.elevated_seconds += elevated_interval
                self._cancel_recovery()
            else:
                if self.recovery_since is None:
                    self.recovery_since = at
                    self.recovery_samples = 0
                self.recovery_samples += 1
                if (at - self.recovery_since >= self.recovery_window
                        and self.recovery_samples >= self.recovery_min_samples):
                    self.state = CALM
                    self.recoveries_confirmed += 1
                    self.elevated_since = None
                    self.elevated_seconds = 0.0
                    self._cancel_recovery()

        return self.decide(at)

    def _cancel_recovery(self):
        self.recovery_since = None
        self.recovery_samples = 0

    # -- the verdict
    def decide(self, at: float) -> dict:
        out = {"stop": False, "rule": None, "state": self.state,
               "elevated_seconds": round(self.elevated_seconds, 3),
               "recovery_seconds": (0.0 if self.recovery_since is None
                                    else round(float(at) - self.recovery_since, 3)),
               "recovery_samples": self.recovery_samples,
               "recoveries_confirmed": self.recoveries_confirmed}
        if self.state != ELEVATED:
            return out
        # Coverage is a property of the SERIES: we must HAVE observations reaching back a full
        # window.  Requiring the samples INSIDE the window to span it instead was wrong and a real
        # run found it: the sample period is exactly a quarter of the window, so a few
        # milliseconds of drift per tick left the window permanently one sample short and this rule
        # could never fire at all.
        covered = (float(at) - self.samples[0][0]) >= self.respond_window - 1e-9
        window = [s for s in self.samples if s[0] >= float(at) - self.respond_window - 1e-9]
        if (covered and len(window) >= self.respond_min_samples
                and all(s[1] > self.respond_at for s in window)):
            return dict(out, stop=True, rule=RULE_SUSTAINED,
                        detail=(f"every one of {len(window)} samples in the trailing "
                                f"{self.respond_window:.0f}s is above {self.respond_at:.2f}"))
        if self.elevated_seconds >= self.budget_seconds:
            return dict(out, stop=True, rule=RULE_BUDGET,
                        detail=(f"{self.elevated_seconds:.0f}s above {self.admit_max:.2f} since "
                                f"elevation began, with no confirmed recovery; the budget is "
                                f"{self.budget_seconds:.0f}s, half the 529s the host was observed "
                                f"to take from elevation to an owner-visible kill"))
        return out

    def series(self) -> list:
        return [{"at": a, "host_some_avg10": h, "cgroup_some_avg10": c} for a, h, c in self.samples]


def _monitor_clock():
    """Real elapsed time, or a file the tests advance by hand.  The monitor's WINDOWS are policy
    constants and are never injectable; only the reading of the clock is, exactly as the readings
    of memory already are, so reboot/pressure logic can be proved without pressuring a host."""
    p = os.environ.get("CRISPDM_ADMISSION_MONITOR_CLOCK")
    if p:
        try:
            return float(Path(p).read_text().strip())
        except (OSError, ValueError):
            return 0.0
    return time.monotonic()


def _scope_stop(res, lease: Lease, self_pid: int, signals, log, on_step=None) -> dict:
    """Stop THIS load's own scope and nothing else.

    Order of escalation, each step bounded to this lease's own witnesses:
      1. SIGTERM the scope leader this launcher started (the pid recorded in the lease), so the
         child's own `timeout`/handlers run and its partial evidence is written;
      2. after the grace the launcher's wall limit already grants, SIGTERM anything still in THIS
         unit's own cgroup -- that is how a detached grandchild is reached;
      3. SIGKILL the same, still only this unit's own cgroup.
    A pid that is not the recorded leader and not listed in this lease's own ``cgroup.procs`` is
    never signalled.  pid 1 and this monitor itself are excluded unconditionally.
    """
    acted = {"term_leader": None, "term_cgroup": [], "kill_cgroup": [], "refused": []}

    def allowed(pid):
        if not pid or int(pid) <= 1 or int(pid) == int(self_pid):
            return False
        return True

    def step():
        if on_step:
            on_step(dict(acted))

    if lease.pid and allowed(lease.pid):
        acted["term_leader"] = int(lease.pid)
        signals(int(lease.pid), "TERM")
        step()                      # durable before the grace, not after it
    elif lease.pid:
        acted["refused"].append({"pid": lease.pid, "why": "not a signalable own-scope pid"})

    if lease.cgroup and lease._cgroup_is_my_own():
        signals(None, "GRACE")
        for pid in res.cgroup_procs(lease.cgroup):
            if allowed(pid):
                acted["term_cgroup"].append(int(pid))
                signals(int(pid), "TERM")
        step()
        signals(None, "GRACE")
        for pid in res.cgroup_procs(lease.cgroup):
            if allowed(pid):
                acted["kill_cgroup"].append(int(pid))
                signals(int(pid), "KILL")
    elif lease.cgroup:
        acted["refused"].append({"cgroup": lease.cgroup,
                                 "why": "not this unit's own cgroup; an inherited ancestor is "
                                        "never signalled"})
    log(acted)
    return acted


def monitor(store: Store, res, lease_id: str, *, self_pid=None, peak_file=None, cause_file=None,
            sample_seconds=PRESSURE_SAMPLE_SECONDS, max_samples=0, clock=None, signals=None,
            child_alive=None) -> dict:
    """The loop crispdm-run runs beside its child: heartbeat, tree peak, pressure, response.

    It replaces the launcher's shell sampler, which renewed the lease and read the tree peak but
    never looked at pressure.  Reads only, except for the lease heartbeat, the peak file, the
    ledger and -- when a rule fires -- signals to its own scope.
    """
    clock = clock or _monitor_clock
    self_pid = int(self_pid or os.getpid())
    mon = PressureMonitor()
    p = store.path_of(lease_id)
    if not p.exists():
        return {"ok": False, "code": "NO_SUCH_LEASE", "lease_id": lease_id}
    lease = Lease(**json.loads(p.read_text()))

    def default_alive():
        return res.pid_alive(lease.pid, lease.pid_starttime) if lease.pid else \
            res.cgroup_alive(lease.cgroup)
    alive = child_alive or default_alive

    def default_signals(pid, what):
        if what == "GRACE":
            time.sleep(PRESSURE_STOP_TERM_GRACE_SECONDS if not os.environ.get(
                "CRISPDM_ADMISSION_MONITOR_CLOCK") else 0.05)
            return
        try:
            os.kill(int(pid), {"TERM": 15, "KILL": 9}[what])
        except (OSError, ProcessLookupError):
            pass
    signals = signals or default_signals

    best_peak, n, stopped = 0, 0, None
    while alive():
        at = clock()
        host_p = float(res.pressure_some_avg10())
        cg_p = (res.cgroup_pressure_some_avg10(lease.cgroup)
                if hasattr(res, "cgroup_pressure_some_avg10") else None)
        verdict = mon.feed(at, host_p, cg_p)

        cur = res.cgroup_peak_bytes(lease.cgroup) or res.cgroup_current_bytes(lease.cgroup) or 0
        if int(cur) > best_peak:
            best_peak = int(cur)
            if peak_file:
                Path(peak_file).write_text(str(best_peak))
        renew(store, res, lease_id, now_from_env())

        if verdict["stop"]:
            # A response may act only on a load that actually holds memory here: a scope holding
            # nothing cannot be part of the problem, and stopping it would buy the host nothing.
            holds = res.cgroup_current_bytes(lease.cgroup) if lease.cgroup else None
            if holds is not None and int(holds) <= 0:
                verdict = dict(verdict, stop=False, rule=None,
                               withheld="this scope holds no memory; stopping it would free none")
            else:
                stopped = verdict
                break
        n += 1
        if max_samples and n >= max_samples:
            break
        if not os.environ.get("CRISPDM_ADMISSION_MONITOR_CLOCK"):
            time.sleep(max(0.05, float(sample_seconds)))
        else:
            time.sleep(0.01)

    record = {"schema": "crispdm.pressure_incident.v1", "lease_id": lease_id,
              "name": lease.name, "label": lease.label, "unit": lease.unit,
              "cgroup": lease.cgroup, "cap_bytes": lease.cap_bytes,
              "argv_sha256": lease.argv_sha256, "boot_id": lease.boot_id,
              "host_key": lease.host_key,
              "observed_tree_peak_bytes": best_peak,
              "samples": mon.series(),
              "policy": {"pressure_admit_max": PRESSURE_ADMIT_MAX,
                         "pressure_respond_at": PRESSURE_RESPOND_AT,
                         "respond_window_seconds": PRESSURE_RESPOND_WINDOW_SECONDS,
                         "elevated_budget_seconds": PRESSURE_ELEVATED_BUDGET_SECONDS,
                         "recovery_window_seconds": PRESSURE_RECOVERY_WINDOW_SECONDS,
                         "oomd_limit_read_from_host_policy": PRESSURE_OOMD_LIMIT},
              "state": mon.state, "elevated_seconds": round(mon.elevated_seconds, 3),
              "recoveries_confirmed": mon.recoveries_confirmed}
    if stopped:
        record["exit_cause"] = "PRESSURE_STOP_" + stopped["rule"]
        record["rule"] = stopped["rule"]
        record["detail"] = stopped.get("detail")
        record["partial_evidence_retained"] = True
        record["scope_only"] = True
        # The cause is made DURABLE BEFORE anything is signalled.  RR01's lesson was that a record
        # which exists only in a process's pipes does not survive the event it describes.
        store.write_incident(lease_id, record)
        if cause_file:
            Path(cause_file).write_text(json.dumps(record, indent=1, sort_keys=True))
        def _durable(a):
            record["acted"] = a
            store.write_incident(lease_id, record)
            if cause_file:
                Path(cause_file).write_text(json.dumps(record, indent=1, sort_keys=True))

        acted = _scope_stop(res, lease, self_pid, signals, on_step=_durable,
                            log=lambda a: store.log({"event": "PRESSURE_STOP_OWN_SCOPE",
                                                 "lease_id": lease_id, "rule": stopped["rule"],
                                                 "unit": lease.unit, "cgroup": lease.cgroup,
                                                 "cap_bytes": lease.cap_bytes,
                                                 "acted": a,
                                                 "note": "only this load's own experiment scope "
                                                         "was stopped; no unrelated process and no "
                                                         "service was signalled, and no limit, "
                                                         "ceiling, cache or swap was touched"}))
        record["acted"] = acted
        store.write_incident(lease_id, record)          # again, now carrying what it acted on
    return {"ok": True, "stopped": bool(stopped), "record": record}


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
                   peak_evidence_sha256=ev_sha, size_text=a.mem, detached=bool(getattr(a, "detached", False)),
                   holder_pid=getattr(a, "holder_pid", None))


# ---- own-scope clean-cache reclaim at scope end (ADM-DEADCACHE-01) ---------------------------

def own_scope_cgroup(slice_name: str):
    """This process's own cgroup, relative to the cgroup root, ONLY if it is a crispdm job scope
    directly under ``slice_name``; otherwise None.  Never the host, never the slice itself, never
    another slice.  In simulation (a readings file is set) the path comes from the readings'
    "self_cgroup" instead of /proc/self/cgroup."""
    rel = None
    sim = os.environ.get("CRISPDM_ADMISSION_RESOURCES_JSON")
    if sim:
        try:
            rel = json.loads(Path(sim).read_text()).get("self_cgroup")
        except (OSError, ValueError):
            rel = None
    else:
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                if line.startswith("0::"):
                    rel = line[3:]
        except OSError:
            rel = None
    if not rel:
        return None
    parts = [x for x in str(rel).strip().split("/") if x]
    if len(parts) < 2:
        return None
    leaf, parent = parts[-1], parts[-2]
    if not (leaf.startswith("crispdm-") and leaf.endswith(".scope")) or parent != slice_name:
        return None
    if any(x in ("..", ".") for x in parts):
        return None
    return "/".join(parts)


def reclaim_own_scope(store: "Store", slice_name: str, lease_id: str | None, now: float) -> dict:
    """Ask the kernel to reclaim this scope's OWN clean file cache through its OWN memory.reclaim.

    The amount written is the clean file bytes read from the scope's own memory.stat; anon,
    shmem, dirty, writeback and unevictable pages are never asked for.  A partial reclaim (EAGAIN)
    is recorded, not retried.  Nothing is written when this process is not inside a crispdm job
    scope under ``slice_name``."""
    rel = own_scope_cgroup(slice_name)
    out = {"event": "SCOPE_CLEAN_CACHE_RECLAIM", "lease_id": lease_id, "cgroup": rel, "slice": slice_name}
    if rel is None:
        out.update(result="SKIPPED_NOT_OWN_CRISPDM_SCOPE")
    else:
        root = Path(os.environ.get("CRISPDM_CGROUP_ROOT") or "/sys/fs/cgroup")
        cg = root / rel
        try:
            before = parse_memory_stat((cg / "memory.stat").read_text())
            want = clean_file_bytes(before)
            out.update(file_before=before.get("file"), shmem_before=before.get("shmem"),
                       clean_file_before=want)
            if want <= 0:
                out.update(result="NOTHING_CLEAN_TO_RECLAIM", requested=0)
            else:
                out["requested"] = want
                try:
                    with open(cg / "memory.reclaim", "w") as fh:
                        fh.write(str(want))
                    out["result"] = "RECLAIMED"
                except OSError as e:
                    out["result"] = "PARTIAL" if e.errno == errno.EAGAIN else f"ERROR_{errno.errorcode.get(e.errno, e.errno)}"
                try:
                    after = parse_memory_stat((cg / "memory.stat").read_text())
                    out.update(file_after=after.get("file"), clean_file_after=clean_file_bytes(after))
                except OSError:
                    pass
        except OSError as e:
            out.update(result=f"UNREADABLE_{errno.errorcode.get(e.errno, e.errno)}")
    try:
        store.log({**out, "_now": now})
    except OSError:
        pass
    return out


def scope_exec(store: "Store", slice_name: str, lease_id: str | None, argv: list) -> int:
    """Run ``argv`` as this process's child inside the job's own scope, forward the operator's
    TERM/INT/HUP to it, and when it has ended reclaim the scope's own clean file cache before the
    scope empties.  The command's exit status is returned unchanged (a death by signal N is
    re-raised on this process, as the shell would see it)."""
    import signal
    import subprocess
    if not argv:
        return 2
    child = subprocess.Popen(argv)

    def fwd(sig, _frame):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, fwd)
    while True:
        try:
            rc = child.wait()
            break
        except InterruptedError:
            continue
    reclaim_own_scope(store, slice_name, lease_id, now_from_env())
    if rc < 0:
        sig = -rc
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        return 128 + sig
    return rc


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
    acq.add_argument("--holder-pid", type=int, default=None,
                     help="the launcher wrapper's pid; the lease is not written if it has died (ADM-STALE-LEASE-01)")
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
            p.add_argument("--witness-pid", type=int, default=None,
                           help="for a lease that could never be armed: the holder's own child "
                                "pid, CHECKED dead before the bytes are given back")

    mon = sub.add_parser("monitor",
                         help="watch a reservation AFTER it was granted: heartbeat, tree peak, and "
                              "host plus cgroup memory pressure with hysteresis.  On a sustained "
                              "crossing it stops THIS load's own scope only, keeping its partial "
                              "evidence and writing the exit cause.")
    mon.add_argument("lease_id")
    mon.add_argument("--peak-file", default=None, help="where to keep the running tree peak")
    mon.add_argument("--cause-file", default=None,
                     help="where to write the exit cause, BEFORE anything is signalled")
    mon.add_argument("--sample-seconds", type=int, default=PRESSURE_SAMPLE_SECONDS)
    mon.add_argument("--max-samples", type=int, default=0, help="0 = until the load ends")

    pol = sub.add_parser("policy", help="the derived post-admission thresholds and where each comes from")

    ret = sub.add_parser("retained", help="a retained lease body, or all of their ids")
    ret.add_argument("lease_id", nargs="?", default=None)

    sub.add_parser("inside-scope",
                   help="is THIS process inside a cgroup a live reservation covers?  A runner that "
                        "spawns its own fit children shares its parent's cgroup and its parent's "
                        "MemoryMax, so it needs no reservation of its own -- but it does need its "
                        "parent to have one.  Exit 0 covered, 1 not covered.")
    se = sub.add_parser("scope-exec",
                        help="INSIDE a job's own scope: run the command, forward TERM/INT/HUP, and when it "
                             "ends reclaim that scope's own clean file cache through its own "
                             "memory.reclaim before the scope empties (ADM-DEADCACHE-01)")
    se.add_argument("--lease", default=None)
    se.add_argument("--slice", default=DEFAULT_SLICE)
    se.add_argument("argv", nargs=argparse.REMAINDER)
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
    if a.cmd == "scope-exec":
        argv = list(a.argv)
        if argv and argv[0] == "--":
            argv = argv[1:]
        return scope_exec(store, a.slice, a.lease, argv)
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
            d = release(store, res, a.lease_id, now, a.observed_peak_bytes,
                        witness_pid=a.witness_pid)
            print(json.dumps(d, sort_keys=True))
            return 0 if d["ok"] else 1
        if a.cmd == "state":
            print(json.dumps(state(store, res, now), indent=1, sort_keys=True))
            return 0
        if a.cmd == "monitor":
            out = monitor(store, res, a.lease_id, peak_file=a.peak_file, cause_file=a.cause_file,
                          sample_seconds=a.sample_seconds, max_samples=a.max_samples)
            print(json.dumps(out, sort_keys=True))
            return 0 if out.get("ok") else 1
        if a.cmd == "policy":
            print(json.dumps({
                "pressure_admit_max": PRESSURE_ADMIT_MAX,
                "pressure_respond_at": PRESSURE_RESPOND_AT,
                "respond_window_seconds": PRESSURE_RESPOND_WINDOW_SECONDS,
                "respond_min_samples": PRESSURE_RESPOND_MIN_SAMPLES,
                "sample_seconds": PRESSURE_SAMPLE_SECONDS,
                "elevated_budget_seconds": PRESSURE_ELEVATED_BUDGET_SECONDS,
                "recovery_window_seconds": PRESSURE_RECOVERY_WINDOW_SECONDS,
                "recovery_min_samples": PRESSURE_RECOVERY_MIN_SAMPLES,
                "term_grace_seconds": PRESSURE_STOP_TERM_GRACE_SECONDS,
                "host_oomd_limit_observed": PRESSURE_OOMD_LIMIT,
                "host_oomd_duration_seconds_observed": PRESSURE_OOMD_DURATION_SECONDS,
                "derivations": {
                    "pressure_respond_at": "midpoint of the admission limit (25.00, DR01) and the "
                                           "host's own oomd limit (50.00, read from the journal "
                                           "lines retained in OOM_INCIDENT_REGISTER.json)",
                    "respond_window_seconds": "the oomd duration the host's own policy declares",
                    "elevated_budget_seconds": "half the 529 s the retained ledger series shows "
                                               "between the first sample above 37.50 and the first "
                                               "owner-application kill",
                    "recovery_window_seconds": "twice the longest run of consecutive samples at or "
                                               "below 25.00 observed inside that window (3 samples, "
                                               "60 s), which was followed by 47.85 within 90 s",
                    "sample_seconds": "the launcher's existing heartbeat period; unchanged",
                    "term_grace_seconds": "the launcher's existing timeout --kill-after grace",
                }}, indent=1, sort_keys=True))
            return 0
        if a.cmd == "retained":
            if a.lease_id:
                body = store.retained_body(a.lease_id)
                print(json.dumps(body, indent=1, sort_keys=True) if body else "{}")
                return 0 if body else 1
            ids = sorted(p.stem for p in store.retained.glob("*.json")) if store.retained.exists() else []
            print(json.dumps({"retained": ids}, indent=1, sort_keys=True))
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
