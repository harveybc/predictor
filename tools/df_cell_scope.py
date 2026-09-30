#!/usr/bin/env python3
"""QRM01: the cell-scope instrument.  One cell, one fresh exclusive scope, one reservation.

WHY THIS EXISTS.  Two numbers were circulated as "the measured Q2 pilot peak" and neither is a
cell's footprint:

  * 7.4 G is systemd's end-of-unit CGROUP peak for a memory-gated WRAPPER scope that hosted a
    driver and its parallel children, taken AT AN oomd KILL.  It is a kill-time watermark of a
    multi-child tree.  It is not a resident set and it is not one cell.
  * 8,458,399,744 B is getrusage(RUSAGE_SELF).ru_maxrss inside ONE cell child: one process's
    resident set.  It is not a tree peak.

They measure different quantities.  In the one bounded pilot where both were read of the same
load the resident set came out 9,158,656 B ABOVE the whole-scope charged peak, so RSS is not a
conservative bound on a cgroup peak in either direction, and the single observed ordering
establishes no invariant.  This module therefore records them as two labelled fields, never
substitutes one for the other, and refuses to offer an RSS to admission as a footprint.

  * 1,463,877,632 B is the whole-cgroup peak of ONE 20,000-window W1440 data materialization,
    with no model, no gradients, no optimizer slots and no Keras graph.  It is a FLOOR for that
    data stage and nothing else.  Every record here therefore carries its STAGE, so a data-stage
    floor can never be read as a training cap.

WHAT IT ENFORCES.  The defect the Q2 round actually had was not the instrument but the launch:
`run_units` started every cell with a bare `subprocess.run`, so a cell child took no scope of its
own and inherited whatever cgroup the driver sat in, three at a time.  A `memory.peak` read there
is a per-BATCH figure of a shared driver scope, not a per-cell one.  So:

  * every cell child is launched through the EXISTING launcher (`crispdm-run`) and its atomic
    admission module -- never a second scheduler -- which gives it a fresh transient scope inside
    `crispdm-batch.slice`, its own `MemoryMax`, and its own reservation held for the whole tree;
  * a child that finds itself in its driver's cgroup, in a cgroup that is not a scope, in a scope
    with no kernel limit, or in a scope another cell has already claimed, REFUSES BY NAME and
    measures nothing;
  * the peak is read from INSIDE the scope, before the scope is removed, because `memory.peak` is
    a kernel high-watermark and survives a release that never happens;
  * an EXTERNAL supervisor retains the termination status of a child that fails, and independently
    identifies the scope instead of trusting the child's word for it;
  * a peak that could not be read is UNKNOWN.  Never 0, never a success, never usable for costing.

  identity      python tools/df_cell_scope.py identity
  peak          python tools/df_cell_scope.py peak
  verify-runner python tools/df_cell_scope.py verify-runner [--json OUT]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

RECORD_SCHEMA = "df_cell_scope_record.v2"
SCHEMA = RECORD_SCHEMA
# v1 carried no attempt identity, no reservation evidence and no checked domains, so a v1 record
# cannot satisfy the fresh-attempt contract and is refused BY ITS VERSION rather than by guessing
# which of its fields might still be trustworthy.
SUPERSEDED_RECORD_SCHEMAS = ("df_cell_scope_record.v1",)
ENVELOPE_SCHEMA = "df_cell_scope_envelope.v1"
CONTRACT_SCHEMA = "df_cell_scope_fresh_attempt_contract.v1"
RUNNER_SCHEMA = "df_runner_identity.v1"
MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"

ATTEMPT_ENV = "CRISPDM_CELL_SCOPE_ATTEMPT"
STAGE_ENV = "CRISPDM_CELL_SCOPE_STAGE"
CLOCK_TOLERANCE_SECONDS = 2.0
LIMIT_PAGE_TOLERANCE_BYTES = 4096          # the kernel stores memory.max at page granularity

ABSENCE_IS_NOT_COINCIDENCE = (
    "ABSENCE_IS_NOT_COINCIDENCE: a missing identity, clock, scope inode, kernel limit, peak or "
    "confirmed reservation REFUSES.  It is never read as agreement, never inferred from another "
    "record, never filled in from a sampled floor, and never passes by default.  The supervisor "
    "not having seen something is a refusal of the thing it did not see, not a licence.")

RSS_BASIS = ("MAIN_PROCESS_RSS_ONLY: resource.getrusage(RUSAGE_SELF).ru_maxrss of this process "
             "alone.  NOT a process-tree or cgroup peak: it excludes every other task in the "
             "cgroup and all cgroup-charged page and kernel memory.  Never size a cap with it.")
CGROUP_BASIS = ("CGROUP_TREE_PEAK: memory.peak of this cell's OWN exclusive scope -- the complete "
                "process tree charged to it, plus the cgroup's page and kernel memory.  This is "
                "the only host-RAM basis a MemoryMax or a placement decision may be judged on.")
FLOOR = "FLOOR_NOT_A_PEAK"
FLOOR_BASIS = (
    "SAMPLED_FLOOR: the last memory.peak this cgroup was OBSERVED to hold before its scope was "
    "removed.  It is a lower bound on the scope's LIFETIME watermark and on nothing else, and it "
    "is never a footprint: a child that ends between two observations -- or before the first -- "
    "takes its true high-watermark away with its scope.  Measured on a short child by the "
    "classification lane: 27,262,976 B retained against 258,584,576 B read from inside the same "
    "scope, an undercount of 9.5x; a cap sized on that number would be nine times too small.  It "
    "does NOT bound the child's own in-scope read in the other direction either: the QRM01 "
    "producer-to-supervisor run of 2026-09-29 sampled 559,525,888 B against 557,654,016 B read in "
    "scope, because a child reads its own peak before it has finished.  Two readings of one scope "
    "at two instants, unsigned, so this never shares a name or a field with a peak.")
RETAINED_LEASE_PEAK_IS_NOT_READ = (
    "The admission store's retained observed_peak_bytes is NOT read as a footprint by this "
    "instrument, in any field.  It is the launcher monitor's SAMPLED series and it was measured to "
    "undercount a short child by 9.5x (27,262,976 B against 258,584,576 B on the same child).  The "
    "only host-RAM measurement here is memory.peak read from INSIDE the cell's own scope before the "
    "scope was removed; anything a sampler saw is carried separately, as a floor.")
COMPARABILITY = ("NOT_INTERCHANGEABLE: a resident set and a cgroup peak measure different "
                 "quantities.  In the one bounded pilot where both were read of the same load the "
                 "resident set exceeded the whole-scope charged peak by 9,158,656 B, so neither "
                 "bounds the other and one observed ordering is not an invariant.")


class ScopeRefusal(RuntimeError):
    """Terminal.  The cell is NOT measured and NOT reported as run."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"REFUSED {code}: {detail}")
        self.code = code
        self.detail = detail


# ---- the existing admission and launcher API -------------------------------------------------

_ADMISSION_LOCK = threading.Lock()


def admission_module():
    """The DEPLOYED admission module, found exactly the way the launcher finds it: an explicit
    override, the installed copy, then this checkout.  There is no second implementation here.

    Serialised, and deliberately so.  `run_units` supervises several cells on several threads, and
    two threads loading this module at once is a genuine race: one publishes the module and starts
    executing its body while the other finds the half-built module in sys.modules and reads
    attributes that do not exist yet.  Observed as `lease_id: null` on every cell of a parallel
    batch whose reservations the launcher had in fact taken.
    """
    with _ADMISSION_LOCK:
        return _admission_module_locked()


def _admission_module_locked():
    for c in (os.environ.get("CRISPDM_ADMISSION_MODULE"),
              Path.home() / ".local/libexec/crispdm/crispdm_admission.py",
              HERE / "crispdm_admission.py"):
        if c and Path(c).is_file():
            if "crispdm_admission" in sys.modules:
                return sys.modules["crispdm_admission"]
            spec = importlib.util.spec_from_file_location("crispdm_admission", str(c))
            mod = importlib.util.module_from_spec(spec)
            # registered BEFORE the body runs: @dataclass resolves its annotations through
            # sys.modules[cls.__module__], so a module executed without being published raises
            # AttributeError on the first dataclass and the caller silently loses the reservation
            # it was trying to read.  Observed, not theoretical.
            sys.modules["crispdm_admission"] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop("crispdm_admission", None)
                raise
            mod.__module_path__ = str(c)
            return mod
    raise ScopeRefusal("ADMISSION_MODULE_NOT_INSTALLED",
                       "the admission module is not installed; run tools/install_crispdm_launcher.sh")


def launcher_path() -> Path | None:
    """`crispdm-run`: an explicit override, then PATH, then the deployed copy.  None when absent --
    and absent is a refusal, never a silent fallback to a bare subprocess."""
    env = os.environ.get("CRISPDM_RUN")
    if env:
        p = Path(env)
        return p if p.is_file() and os.access(p, os.X_OK) else None
    w = shutil.which("crispdm-run")
    if w:
        return Path(w)
    p = Path.home() / ".local/bin/crispdm-run"
    return p if p.is_file() and os.access(p, os.X_OK) else None


def launcher_available() -> bool:
    """Both halves must be there: the launcher, and a user systemd that can make a transient
    scope.  Without the second the launcher cannot give a cell a scope of its own."""
    if launcher_path() is None:
        return False
    if not shutil.which("systemd-run") or not shutil.which("systemctl"):
        return False
    try:
        r = subprocess.run(["systemctl", "--user", "show", "-p", "Version", "--value"],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:                                                       # noqa: BLE001
        return False


# ---- cgroup readings ---------------------------------------------------------------------------

def cgroup_root() -> Path:
    return Path(os.environ.get("CRISPDM_CGROUP_ROOT", "/sys/fs/cgroup"))


def own_cgroup() -> str | None:
    """This process's cgroup v2 path, relative to the root and without its leading slash."""
    override = os.environ.get("CRISPDM_CELL_SCOPE_SELF_CGROUP")
    if override:
        return override.strip().lstrip("/")
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":")
            if parts[0] == "0" or len(parts) >= 3 and parts[1] == "":
                return parts[-1].strip().lstrip("/")
    except Exception:                                                       # noqa: BLE001
        return None
    return None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:                                                       # noqa: BLE001
        return None


def scope_identity(rel: str | None = None) -> dict:
    """WHICH cgroup this is, not merely what it is called.

    A scope NAME can repeat -- systemd builds it from a job name, an epoch second and a pid.  The
    directory INODE cannot: a fresh scope is a fresh directory.  Identity, ancestry and lifetime
    are recorded, so reuse is detected by comparing inodes rather than by inferring a rule from a
    name or from one memory reading.
    """
    rel = (rel if rel is not None else own_cgroup())
    if rel is None:
        return {"cgroup": None, "unit": None, "is_scope": False, "inode": None,
                "path_exists": False, "parent": None, "boot_id": _boot_id(),
                "read_at": time.time(), "why": "this process's cgroup could not be read"}
    rel = str(rel).lstrip("/")
    p = cgroup_root() / rel
    unit = rel.rsplit("/", 1)[-1] if rel else None
    try:
        ino = p.stat().st_ino
        exists = True
    except OSError:
        ino, exists = None, False
    return {"cgroup": rel, "unit": unit, "is_scope": bool(unit and unit.endswith(".scope")),
            "inode": ino, "path_exists": exists,
            "parent": rel.rsplit("/", 1)[0] if "/" in rel else None,
            "boot_id": _boot_id(), "read_at": time.time()}


def _read_int(path: Path):
    try:
        raw = path.read_text().split()[0]
    except Exception:                                                       # noqa: BLE001
        return None
    if raw == "max":
        return "max"
    try:
        return int(raw)
    except ValueError:
        return None


def kernel_limit(rel: str | None = None) -> dict:
    """`memory.max` of this scope: the limit the KERNEL will enforce, which is what a declared cap
    has to be checked against.  'max' is no limit at all and is reported as such."""
    rel = rel if rel is not None else own_cgroup()
    if rel is None:
        return {"bytes": None, "status": UNKNOWN, "source": "memory.max",
                "why": "the cgroup of this process could not be read"}
    v = _read_int(cgroup_root() / str(rel).lstrip("/") / "memory.max")
    if v == "max":
        return {"bytes": None, "status": UNKNOWN, "source": "memory.max", "unlimited": True,
                "why": "memory.max is 'max': this cgroup carries NO kernel limit"}
    if v is None:
        return {"bytes": None, "status": UNKNOWN, "source": "memory.max",
                "why": "memory.max is not readable for this cgroup"}
    return {"bytes": int(v), "status": MEASURED, "source": "memory.max", "unlimited": False}


def cgroup_peak(rel: str | None = None) -> dict:
    """`memory.peak` of a cgroup: a KERNEL HIGH-WATERMARK, not a sample.

    Read from inside the scope it belongs to it is a measurement even for a child shorter than any
    sampler's interval, and it is not lost when a release never succeeds.  When it cannot be read
    the answer is UNKNOWN -- never 0, which would read as a cell that used no memory.
    """
    rel = rel if rel is not None else own_cgroup()
    if rel is None:
        return {"bytes": None, "status": UNKNOWN, "basis": CGROUP_BASIS,
                "why": "the cgroup of this process could not be read"}
    v = _read_int(cgroup_root() / str(rel).lstrip("/") / "memory.peak")
    if not isinstance(v, int):
        return {"bytes": None, "status": UNKNOWN, "basis": CGROUP_BASIS, "cgroup": rel,
                "why": "memory.peak is not readable for this cgroup: this cell has NO tree peak"}
    return {"bytes": int(v), "status": MEASURED, "basis": CGROUP_BASIS, "cgroup": rel}


def lifetime_peak(rel: str | None = None) -> dict:
    """The LIFETIME high-watermark of the scope, read and never reset.

    Kept separately from any per-stage figure on purpose.  `memory.peak` can be reset, and a reset
    destroys the only number that answers `what did this cell cost in all`.  This function opens a
    fresh descriptor, reads, and writes nothing.
    """
    out = dict(cgroup_peak(rel))
    out["reset"] = False
    out["watermark"] = "LIFETIME_OF_THE_SCOPE"
    out["why_separate"] = (
        "a lifetime watermark is retained under its own name because a per-stage figure obtained "
        "by resetting memory.peak destroys it; the two are never the same field")
    return out


def stage_peak_one_descriptor(rel: str | None = None, *, reset_after: bool = False) -> dict:
    """A per-STAGE peak read through ONE open file descriptor, with the reset on that same fd.

    The kernel documents the reset as descriptor-scoped: "a write to this file resets it to the
    current memory usage for subsequent reads through the same file descriptor"
    (https://docs.kernel.org/admin-guide/cgroup-v2.html, memory.peak).  A `write_text` followed by
    a `read_text` opens TWO descriptors and therefore establishes nothing about the reset -- an
    earlier report of ours asserted that experiment and it was not the experiment performed.  So
    the read and the reset happen on one `os.open` here, the descriptor discipline is recorded in
    the result, and the LIFETIME watermark is kept by `lifetime_peak()` in a field of its own.
    """
    rel = rel if rel is not None else own_cgroup()
    base = {"basis": CGROUP_BASIS, "cgroup": rel, "watermark": "SINCE_THE_LAST_RESET_ON_THIS_FD",
            "descriptor": "SAME_OPEN_FILE_DESCRIPTOR",
            "kernel_doc": "https://docs.kernel.org/admin-guide/cgroup-v2.html (memory.peak)",
            "why_one_descriptor": ("memory.peak's reset applies to subsequent reads THROUGH THE "
                                   "SAME open file descriptor; a write_text followed by a "
                                   "read_text opens a different descriptor and does not "
                                   "establish it")}
    if rel is None:
        return {**base, "bytes": None, "status": UNKNOWN, "reset": False,
                "why": "the cgroup of this process could not be read"}
    path = cgroup_root() / str(rel).lstrip("/") / "memory.peak"
    fd = None
    try:
        fd = os.open(path, os.O_RDWR if reset_after else os.O_RDONLY)
        raw = os.pread(fd, 64, 0).decode().split()
        value = int(raw[0]) if raw and raw[0] != "max" else None
        out = {**base, "bytes": value,
               "status": MEASURED if isinstance(value, int) else UNKNOWN, "reset": False}
        if value is None:
            out["why"] = "memory.peak held no integer for this cgroup"
        if reset_after and isinstance(value, int):
            try:
                os.pwrite(fd, b"0", 0)
                out["reset"] = True
                out["reset_on_the_same_fd"] = True
            except OSError as e:
                out["reset"] = False
                out["why_not_reset"] = f"the reset was refused on the same fd: {e.__class__.__name__}"
        return out
    except OSError as e:
        return {**base, "bytes": None, "status": UNKNOWN, "reset": False,
                "why": f"memory.peak could not be opened: {e.__class__.__name__}"}
    finally:
        if fd is not None:
            os.close(fd)


# ---- the reservation, confirmed rather than assumed ------------------------------------------

def confirm_reservation(*, cgroup: str | None, unit: str | None = None, cap_bytes=None,
                        now: float | None = None) -> dict:
    """Is there a reservation bound to THIS cgroup, and does it hold the declared cap?

    Two sources, both the launcher's own store and neither the child's word:

      LIVE_ADMISSION_STORE    a live lease whose `cgroup` or `unit` is this scope.  This is the
                              path a long child confirms itself on, from inside its own scope.
      RETAINED_LEASE_BODY     the lease BODY the store keeps after the load ends (RR02).  This is
                              the path that works for a SHORT child: a child whose scope is gone
                              before any external observer could look still left a lease body
                              naming its cgroup and its cap, so its reservation is provable after
                              the fact without the scope being alive.

    A store that cannot be read is UNKNOWN, and UNKNOWN refuses.  Absence is not coincidence.
    """
    out = {"schema": "df_cell_scope_reservation.v1", "confirmed": False, "lease_id": None,
           "source": None, "cgroup": cgroup, "unit": unit, "cap_bytes": None,
           "declared_cap_bytes": None if cap_bytes is None else int(cap_bytes),
           "checked_at": time.time(), "why": None}
    if not cgroup and not unit:
        out["why"] = "no scope was named, so no reservation could be looked for"
        return out
    want = str(cgroup).lstrip("/") if cgroup else None
    try:
        adm = admission_module()
        store = adm.Store()
        res = adm.resources_from_env()
    except Exception as e:                                                  # noqa: BLE001
        out["why"] = f"the admission store could not be read: {type(e).__name__}: {e}"
        return out
    try:
        for lease in adm.state(store, res, now if now is not None else time.time())["live"]:
            if (want and lease.get("cgroup") and str(lease["cgroup"]).lstrip("/") == want) or \
               (unit and lease.get("unit") == unit):
                out.update(confirmed=True, lease_id=lease["lease_id"],
                           source="LIVE_ADMISSION_STORE", cap_bytes=lease.get("cap_bytes"),
                           cgroup=lease.get("cgroup") or cgroup, unit=lease.get("unit") or unit)
                return _cap_checked(out)
    except Exception as e:                                                  # noqa: BLE001
        out["why"] = f"the live reservations could not be read: {type(e).__name__}: {e}"
        return out
    retained = store.retained if hasattr(store, "retained") else None
    if retained and Path(retained).is_dir():
        for p in sorted(Path(retained).glob("*.json")):
            try:
                body = json.loads(p.read_text()).get("lease") or {}
            except Exception:                                               # noqa: BLE001
                continue
            if (want and body.get("cgroup") and str(body["cgroup"]).lstrip("/") == want) or \
               (unit and body.get("unit") == unit):
                out.update(confirmed=True, lease_id=body.get("lease_id"),
                           source="RETAINED_LEASE_BODY", cap_bytes=body.get("cap_bytes"),
                           cgroup=body.get("cgroup") or cgroup, unit=body.get("unit") or unit)
                return _cap_checked(out)
    out["why"] = (f"no live reservation and no retained lease body names {cgroup or unit}; this "
                  f"cell holds NO confirmed reservation, which refuses rather than passing")
    return out


def _cap_checked(out: dict) -> dict:
    """A reservation confirmed for the wrong number is not this cell's reservation."""
    declared, held = out.get("declared_cap_bytes"), out.get("cap_bytes")
    if declared is None or held is None:
        return out
    if int(held) != int(declared):
        out["confirmed"] = False
        out["cap_agrees_with_the_declaration"] = False
        out["why"] = (f"the reservation bound to this scope holds {int(held)} B, not the declared "
                      f"{int(declared)} B; a cap is declared once and a reservation is never "
                      f"reduced to make a record pass")
    else:
        out["cap_agrees_with_the_declaration"] = True
    return out


def verify_reservation_independently(*, lease_id, cgroup, cap_bytes=None) -> dict:
    """Re-verify a lease the CHILD named, against the launcher's store rather than the child.

    The child can confirm its own reservation while it is alive -- the only path a short child
    has -- so the supervisor does not take that on trust: it looks the named lease up in the live
    set and then in the retained bodies, and requires the store's own copy to bind that lease to
    the cgroup and the cap the record claims.
    """
    out = {"schema": "df_cell_scope_reservation_verification.v1", "verified": False,
           "lease_id": lease_id, "source": None, "why": None}
    if not lease_id:
        out["why"] = "the record named no lease id, so there was nothing to verify"
        return out
    try:
        adm = admission_module()
        store = adm.Store()
        res = adm.resources_from_env()
    except Exception as e:                                                  # noqa: BLE001
        out["why"] = f"the admission store could not be read: {type(e).__name__}: {e}"
        return out
    body = None
    try:
        for lease in adm.state(store, res, time.time())["live"]:
            if lease.get("lease_id") == lease_id:
                body, out["source"] = lease, "LIVE_ADMISSION_STORE"
                break
    except Exception as e:                                                  # noqa: BLE001
        out["why"] = f"the live reservations could not be read: {type(e).__name__}: {e}"
        return out
    if body is None:
        retained = store.retained_body(lease_id) if hasattr(store, "retained_body") else None
        if retained:
            body, out["source"] = retained.get("lease") or {}, "RETAINED_LEASE_BODY"
    if body is None:
        out["why"] = (f"the store holds no lease {lease_id}, live or retained: the reservation the "
                      f"record claims cannot be verified and is therefore refused")
        return out
    held_cgroup = str(body.get("cgroup") or "").lstrip("/")
    out["store_cgroup"], out["store_cap_bytes"] = body.get("cgroup"), body.get("cap_bytes")
    if cgroup and held_cgroup and held_cgroup != str(cgroup).lstrip("/"):
        out["why"] = (f"the store binds {lease_id} to {body.get('cgroup')}, not to the scope this "
                      f"record claims ({cgroup}): the charge cannot be attributed to this cell")
        return out
    if cgroup and not held_cgroup:
        out["why"] = (f"the store's copy of {lease_id} names no cgroup, so it cannot be shown to "
                      f"be the reservation of {cgroup}")
        return out
    if cap_bytes is not None and body.get("cap_bytes") is not None and \
            int(body["cap_bytes"]) != int(cap_bytes):
        out["why"] = (f"the store holds {int(body['cap_bytes'])} B for {lease_id}, not the "
                      f"declared {int(cap_bytes)} B")
        return out
    out["verified"] = True
    return out


# ---- the attempt -------------------------------------------------------------------------------

def new_attempt_id() -> str:
    """A token minted by the SUPERVISOR before this child starts.

    This is what makes freshness checkable at all.  A record that already existed on disk cannot
    carry a token that was created after it was written, so `attempt_id` turns "is this record
    about the run I just performed" from an assumption into a comparison.  No host name, no
    account identifier, no pid: an opaque nonce and a clock.
    """
    return f"{int(time.time() * 1000):013d}-{secrets.token_hex(12)}"


def attempt_id_from_env() -> str | None:
    v = (os.environ.get(ATTEMPT_ENV) or "").strip()
    return v or None


def process_rss_peak() -> dict:
    import resource as _r
    return {"bytes": int(_r.getrusage(_r.RUSAGE_SELF).ru_maxrss) * 1024, "status": MEASURED,
            "basis": RSS_BASIS}


def gpu_memory() -> dict:
    """Device memory, recorded SEPARATELY from host RAM and never added to it.

    The probe is deliberately not automatic: importing a framework to ask costs hundreds of MiB of
    the very host RAM this record is measuring.  A cell that selected no device reports UNKNOWN
    with the reason -- absence, not zero.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if os.environ.get("CRISPDM_CELL_SCOPE_GPU", "auto") == "off":
        return {"allocated_bytes": None, "reserved_bytes": None, "status": UNKNOWN,
                "source": "none", "why": "the GPU probe is switched off for this cell"}
    if vis is None or vis.strip() == "":
        return {"allocated_bytes": None, "reserved_bytes": None, "status": UNKNOWN,
                "source": "none", "visible_devices": vis,
                "why": ("no device was selected for this cell (CUDA_VISIBLE_DEVICES is empty or "
                        "unset), so a GPU figure is ABSENT -- it is not zero")}
    try:
        import torch                                                        # noqa: PLC0415
        if not torch.cuda.is_available():
            return {"allocated_bytes": None, "reserved_bytes": None, "status": UNKNOWN,
                    "source": "torch", "visible_devices": vis,
                    "why": "a device was selected but torch reports no usable CUDA device"}
        return {"allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                "current_reserved_bytes": int(torch.cuda.memory_reserved()),
                "status": MEASURED, "source": "torch.cuda", "visible_devices": vis,
                "basis": ("DEVICE peaks since the process started: max_memory_allocated is live "
                          "tensors, max_memory_reserved is the caching allocator's arena.  Neither "
                          "is host RAM and neither may be added to the cgroup peak")}
    except Exception as e:                                                  # noqa: BLE001
        return {"allocated_bytes": None, "reserved_bytes": None, "status": UNKNOWN,
                "source": "torch", "visible_devices": vis,
                "why": f"the device could not be interrogated: {type(e).__name__}"}


def host_identity() -> dict:
    """The executing host, recorded WITHOUT its name (AGENTS.md forbids one in this repository).
    Carried from tools/df_e1_block.py at 84bcd605 so two records still answer `same host or not`.
    """
    node = os.uname().nodename.encode()
    return {"host_id": hashlib.sha256(b"df_e1_block.host_id.v1|" + node).hexdigest()[:16],
            "host_name_recorded": False,
            "why": "AGENTS.md forbids a machine host name in this repository; an opaque per-host id is recorded instead",
            "scope": os.environ.get("CRISPDM_SCOPE")}


# ---- the exclusive claim -------------------------------------------------------------------------

def require_fresh_exclusive_scope(claims_dir, cell_id: str, *, identity: dict | None = None,
                                  parent_cgroup: str | None = None) -> dict:
    """Refuse unless THIS cell has a fresh exclusive scope of its own.

    The four ways a cell can fail to have one, each refused by its own name:

      NOT_A_SCOPE                          the cgroup is not a transient scope at all
      REUSED_DRIVER_SCOPE                  it is the cgroup the driver itself sits in -- one
                                           sequential child inside a reused driver scope
      NO_KERNEL_LIMIT                      the scope carries no memory.max, so nothing is enclosed
      SCOPE_ALREADY_CLAIMED_BY_ANOTHER_CELL  a second cell is being charged to one scope

    The claim is created with O_EXCL and keyed by the scope's INODE, so two concurrent children in
    one scope collide in the filesystem rather than in a check-then-act race.  A cell re-entering
    its own claim is not a reuse and says so.
    """
    ident = identity if identity is not None else scope_identity()
    claims = Path(claims_dir)
    claims.mkdir(parents=True, exist_ok=True)

    if not ident.get("cgroup") or not ident.get("path_exists"):
        raise ScopeRefusal("SCOPE_CGROUP_UNREADABLE",
                           f"the cgroup of {cell_id} could not be read: {ident.get('why') or ident.get('cgroup')}")
    if parent_cgroup and str(ident["cgroup"]).strip("/") == str(parent_cgroup).strip("/"):
        raise ScopeRefusal("REUSED_DRIVER_SCOPE",
                           f"{cell_id} is running in the driver's own cgroup {ident['cgroup']}: a cell in a "
                           f"reused driver scope shares its charge with the driver and every sibling, so its "
                           f"memory.peak is a per-batch figure and not this cell's footprint")
    if not ident.get("is_scope"):
        raise ScopeRefusal("NOT_A_SCOPE",
                           f"{cell_id} is in {ident['cgroup']}, which is not a transient scope; a cell's "
                           f"reservation and its cgroup limit must belong to one unit created for it")
    limit = kernel_limit(ident["cgroup"])
    if limit["status"] != MEASURED:
        raise ScopeRefusal("NO_KERNEL_LIMIT",
                           f"{cell_id}'s scope {ident['cgroup']} carries no enforceable memory.max "
                           f"({limit.get('why')}); an unlimited scope encloses nothing")

    claim = {"schema": "df_cell_scope_claim.v1", "cell_id": cell_id, "scope": ident,
             "kernel_limit": limit, "claimed_at": time.time(), "pid": os.getpid()}
    path = claims / f"scope-{ident['inode']}.json"
    body = json.dumps(claim, sort_keys=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        claim["reentrant"] = False
        return claim
    except FileExistsError:
        held = json.loads(path.read_text())
        if held.get("cell_id") == cell_id:
            held["reentrant"] = True
            return held
        raise ScopeRefusal("SCOPE_ALREADY_CLAIMED_BY_ANOTHER_CELL",
                           f"scope {ident['cgroup']} (inode {ident['inode']}) is already claimed by "
                           f"{held.get('cell_id')}; {cell_id} would be charged to a scope that is not its own")


# ---- the record ------------------------------------------------------------------------------------

def cell_scope_record(cell_id: str, *, stage: str, claim: dict | None = None, updates=None,
                      cpu_seconds=None, wall_seconds=None, rel: str | None = None,
                      extra: dict | None = None, attempt_id: str | None = None,
                      declared_cap_bytes=None) -> dict:
    """Everything the order names, each quantity in its own labelled field -- v2.

    `stage` is mandatory and is carried into the peak evidence: a data-materialization floor and a
    training peak are different measurements of different things, and a record that does not say
    which it is can be misread as the other.

    v2 adds the four things v1 let a reader assume, each of which the gate now checks:
    the ATTEMPT this record belongs to, the BOOT it was written on, the KERNEL LIMIT in force, and
    a CONFIRMED RESERVATION read from the launcher's own store while this child is alive -- the
    one path a child shorter than any external observation still has.
    """
    ident = (claim or {}).get("scope") or scope_identity(rel)
    rel = ident.get("cgroup")
    peak = cgroup_peak(rel)
    limit = (claim or {}).get("kernel_limit") or kernel_limit(rel)
    attempt = attempt_id or attempt_id_from_env()
    declared = declared_cap_bytes
    if declared is None:
        raw = (os.environ.get("CRISPDM_CELL_SCOPE_DECLARED_CAP") or "").strip()
        declared = int(raw) if raw.isdigit() else None
    reservation = confirm_reservation(cgroup=rel, unit=ident.get("unit"), cap_bytes=declared)
    refusals = []
    if attempt is None:
        refusals.append("NO_ATTEMPT_IDENTITY: this record was not produced under a supervised "
                        "attempt, so it cannot be shown to describe any particular run")
    if peak["status"] != MEASURED:
        refusals.append("NO_IN_SCOPE_PEAK: the cell's own cgroup peak is UNKNOWN, so this cell "
                        "costs nothing and bounds nothing")
    elif not isinstance(peak.get("bytes"), int) or isinstance(peak.get("bytes"), bool) \
            or peak["bytes"] <= 0:
        refusals.append("PEAK_OUT_OF_DOMAIN: a cgroup peak is a positive count of bytes")
    if limit.get("status") != MEASURED:
        refusals.append("NO_KERNEL_LIMIT: the scope carries no enforceable memory.max, so nothing "
                        "was enclosed and no peak of it bounds a placement")
    if not reservation.get("confirmed"):
        refusals.append(f"NO_CONFIRMED_RESERVATION: {reservation.get('why')}")
    rec = {
        "schema": RECORD_SCHEMA,
        "cell_id": cell_id,
        "attempt_id": attempt,
        "stage": stage,
        "recorded_at": time.time(),
        "boot_id": _boot_id(),
        "scope": ident,
        "kernel_limit": limit,
        "declared_cap_bytes": None if declared is None else int(declared),
        "reservation": reservation,
        "host_ram": {"cgroup_peak": peak, "process_rss_peak": process_rss_peak(),
                     "cgroup_lifetime_peak": lifetime_peak(rel),
                     "comparability": COMPARABILITY},
        "gpu": gpu_memory(),
        "optimizer_updates": None if updates is None else int(updates),
        "cpu_seconds": None if cpu_seconds is None else float(cpu_seconds),
        "wall_seconds": None if wall_seconds is None else float(wall_seconds),
        "host_identity": host_identity(),
        "absence_is_not_coincidence": ABSENCE_IS_NOT_COINCIDENCE,
        "usable_for_costing": not refusals,
        "why_not_usable": None if not refusals else "; ".join(refusals),
    }
    if extra:
        rec.update(extra)
    return rec


# ---- the producer/consumer boundary, declared by schema version --------------------------------

def embed_record(container: dict, record: dict, *, at: str = "cell_scope") -> dict:
    """Put a scope record inside a PRODUCER's own document and DECLARE where it is.

    F2 of the 23b2efa3 dictamen: `df_e1_block` wrote this record nested under `cell_scope` while
    the supervisor read `host_ram` at the ROOT of the same file, so the producer's real document
    yielded UNKNOWN and `usable_for_costing=false` while physically containing the peak.  The two
    sides did not disagree about a value, they disagreed about a shape, silently.

    The repair is not to try both places -- that replaces one silent mismatch with another, and a
    field that merely shares a name is exactly what must never be accepted.  It is an explicit
    envelope carrying the ENVELOPE version, the RECORD version and the key the record sits under.
    A consumer that does not understand either version refuses by name.
    """
    container[at] = record
    container["cell_scope_envelope"] = {
        "schema": ENVELOPE_SCHEMA,
        "record_schema": record.get("schema"),
        "record_at": at,
        "why": ("the consumer reads the scope record by this declaration alone; it never searches "
                "for a field that merely shares a name, at the root or anywhere else"),
    }
    return container


def extract_record(document) -> tuple[dict | None, str | None]:
    """Read a scope record out of a producer document by its DECLARED envelope.

    Returns `(record, None)` or `(None, "CODE: why")`.  The two accepted shapes, both declared:

      * a document carrying `cell_scope_envelope` of a known envelope version, whose declared
        record version is supported and whose declared key holds a record that agrees about its
        own version;
      * a BARE record, i.e. a document whose own root `schema` is the supported record version --
        the shape the instrument writes when it is the whole file.

    Everything else refuses.  A v1 record refuses by its version, not by a field-by-field guess.
    """
    if not isinstance(document, dict):
        return None, "RECORD_NOT_A_MAPPING: the retained document is not a JSON object"
    env = document.get("cell_scope_envelope")
    if env is None:
        root = document.get("schema")
        if root == RECORD_SCHEMA:
            return document, None
        if root in SUPERSEDED_RECORD_SCHEMAS:
            return None, (f"RECORD_SCHEMA_SUPERSEDED: {root} carries no attempt identity, no "
                          f"reservation evidence and no checked domains; {RECORD_SCHEMA} is "
                          f"required and a superseded record is refused by its version")
        return None, ("ENVELOPE_MISSING: the document declares no cell_scope_envelope and its own "
                      f"schema is {root!r}, not {RECORD_SCHEMA}; no field is read by name alone")
    if not isinstance(env, dict):
        return None, "ENVELOPE_NOT_A_MAPPING: cell_scope_envelope is not a JSON object"
    if env.get("schema") != ENVELOPE_SCHEMA:
        return None, (f"ENVELOPE_SCHEMA_UNKNOWN: {env.get('schema')!r} is not {ENVELOPE_SCHEMA}; "
                      f"this consumer does not know where that producer put its record")
    declared = env.get("record_schema")
    if declared in SUPERSEDED_RECORD_SCHEMAS:
        return None, (f"RECORD_SCHEMA_SUPERSEDED: the envelope declares {declared}, which carries "
                      f"no attempt identity and no reservation evidence; {RECORD_SCHEMA} is required")
    if declared != RECORD_SCHEMA:
        return None, (f"RECORD_SCHEMA_NOT_SUPPORTED: the envelope declares {declared!r}, not "
                      f"{RECORD_SCHEMA}")
    at = env.get("record_at")
    if not isinstance(at, str) or not at:
        return None, "ENVELOPE_POINTER_MISSING: the envelope declares no record_at key"
    if at not in document:
        return None, (f"ENVELOPE_POINTER_DANGLING: the envelope points at {at!r}, which the "
                      f"document does not contain")
    record = document[at]
    if not isinstance(record, dict):
        return None, f"RECORD_NOT_A_MAPPING: {at!r} does not hold a JSON object"
    if record.get("schema") != declared:
        return None, (f"ENVELOPE_RECORD_SCHEMA_DISAGREES: the envelope declares {declared}, the "
                      f"record says {record.get('schema')!r}; a disagreement is not resolved by "
                      f"preferring one of them")
    return record, None


# ---- the fresh-attempt contract ----------------------------------------------------------------

def _pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def verify_fresh_attempt(document, *, cell_id: str, attempt_id: str, stage: str,
                         declared_cap_bytes, started_at: float, finished_at: float,
                         boot_id: str | None = None, observed_scope: dict | None = None,
                         verify_reservation=True,
                         clock_tolerance: float = CLOCK_TOLERANCE_SECONDS) -> dict:
    """Does this document describe THIS attempt of THIS cell, completely and in domain?

    F1 of the 23b2efa3 dictamen: the old gate read ANY pre-existing JSON at `record_path` and
    accepted `MEASURED` plus exit 0 as a cost, demanding no cell identity, no attempt identity, no
    stage, no clock, no limit and no reservation -- and comparing scopes only IF it happened to
    have seen one.  Two of the auditor's counterexamples walked straight through it: a record of
    ANOTHER cell with clock 1 and no lease, and a peak of MINUS ONE.

    Every clause below refuses BY ITS OWN NAME, and every clause refuses on ABSENCE.  Nothing here
    depends on the supervisor having observed the scope, because a short-lived child is precisely
    where that observation does not exist -- and the old gate's conditional comparison is the hole
    the order names.  What replaces it is checkable without an external observer: a token minted
    before this child started, a boot, a clock inside this attempt, a kernel limit that is the
    declared cap, and a reservation re-verified in the launcher's own store.  When an external
    observation DOES exist it must agree; it is a further check, never the only one.
    """
    refusals = []
    checks = {}

    def refuse(code: str, detail: str):
        refusals.append({"code": code, "detail": detail})

    record, why = extract_record(document) if document is not None else (
        None, "RECORD_ABSENT: no record was retained at the path this attempt was told to write")
    if record is None:
        code, _, detail = (why or "RECORD_ABSENT: none").partition(": ")
        refuse(code, detail or why)
        return {"schema": CONTRACT_SCHEMA, "accepted": False, "refusals": refusals,
                "refused_by": [r["code"] for r in refusals], "checks": checks, "record": None,
                "cell_id": cell_id, "attempt_id": attempt_id, "stage": stage,
                "declared_cap_bytes": None if declared_cap_bytes is None
                                      else int(declared_cap_bytes),
                "absence_is_not_coincidence": ABSENCE_IS_NOT_COINCIDENCE}

    # -- identity: cell, attempt, stage, boot ---------------------------------------------------
    if record.get("cell_id") is None:
        refuse("CELL_IDENTITY_MISSING", "the record names no cell")
    elif record["cell_id"] != cell_id:
        refuse("CELL_IDENTITY_MISMATCH",
               f"the record belongs to {record['cell_id']!r}, not to {cell_id!r}; another cell's "
               f"evidence is never this cell's cost")
    checks["cell_id"] = record.get("cell_id")

    if not attempt_id:
        refuse("ATTEMPT_IDENTITY_NOT_MINTED",
               "the supervisor minted no attempt token, so freshness could not be checked at all")
    elif record.get("attempt_id") is None:
        refuse("ATTEMPT_IDENTITY_MISSING",
               "the record carries no attempt_id, so it cannot be shown to describe the run that "
               "just happened rather than a file that was already on disk")
    elif record["attempt_id"] != attempt_id:
        refuse("ATTEMPT_IDENTITY_MISMATCH",
               f"the record belongs to attempt {record['attempt_id']!r}, not to {attempt_id!r}")
    checks["attempt_id"] = record.get("attempt_id")

    if record.get("stage") is None:
        refuse("STAGE_MISSING", "the record names no stage; a data-stage floor and a training "
                                "peak are different measurements and must never be interchanged")
    elif stage is not None and record["stage"] != stage:
        refuse("STAGE_MISMATCH", f"the record is stage {record['stage']!r}, the attempt was "
                                 f"{stage!r}")
    checks["stage"] = record.get("stage")

    want_boot = boot_id if boot_id is not None else _boot_id()
    if record.get("boot_id") is None:
        refuse("BOOT_IDENTITY_MISSING", "the record names no boot; a watermark from an earlier "
                                        "boot describes a load this host no longer has")
    elif want_boot is not None and record["boot_id"] != want_boot:
        refuse("BOOT_IDENTITY_MISMATCH",
               "the record was written on a different boot than this attempt")
    checks["boot_id_agrees"] = (record.get("boot_id") == want_boot) if want_boot else None

    # -- the scope, with its inode -------------------------------------------------------------
    scope = record.get("scope") if isinstance(record.get("scope"), dict) else None
    if scope is None:
        refuse("SCOPE_IDENTITY_MISSING", "the record names no scope")
    else:
        if not scope.get("cgroup"):
            refuse("SCOPE_CGROUP_MISSING", "the record's scope names no cgroup")
        if scope.get("inode") is None:
            refuse("SCOPE_INODE_MISSING",
                   "the record's scope carries no directory inode, so the scope it was charged to "
                   "cannot be identified; a missing inode REFUSES instead of being skipped")
        elif not _pos_int(scope.get("inode")):
            refuse("SCOPE_INODE_OUT_OF_DOMAIN",
                   f"the scope inode {scope.get('inode')!r} is not a positive integer")
        if scope.get("is_scope") is not True:
            refuse("NOT_A_SCOPE",
                   "the record's cgroup is not a transient scope, so its charge is shared with "
                   "whatever else lives in that cgroup")
    checks["scope_cgroup"] = (scope or {}).get("cgroup")
    checks["scope_inode"] = (scope or {}).get("inode")

    # -- the clock: it must belong to THIS attempt ----------------------------------------------
    clock = record.get("recorded_at")
    if clock is None:
        refuse("CLOCK_MISSING", "the record carries no recorded_at")
    elif isinstance(clock, bool) or not isinstance(clock, (int, float)):
        refuse("CLOCK_OUT_OF_DOMAIN", f"recorded_at {clock!r} is not a number of seconds")
    elif not (started_at - clock_tolerance <= float(clock) <= finished_at + clock_tolerance):
        refuse("CLOCK_OUTSIDE_THIS_ATTEMPT",
               f"recorded_at {float(clock)!r} lies outside this attempt's window "
               f"[{started_at}, {finished_at}] (+/-{clock_tolerance}s); the record was not written "
               f"while this child ran")
    checks["recorded_at"] = clock

    # -- the kernel limit in force --------------------------------------------------------------
    limit = record.get("kernel_limit") if isinstance(record.get("kernel_limit"), dict) else None
    limit_bytes = None
    if limit is None:
        refuse("KERNEL_LIMIT_MISSING", "the record names no kernel limit")
    elif limit.get("status") != MEASURED:
        refuse("KERNEL_LIMIT_NOT_MEASURED",
               f"the scope's memory.max was not measured ({limit.get('why')}); an unlimited scope "
               f"encloses nothing and its peak bounds nothing")
    elif not _pos_int(limit.get("bytes")):
        refuse("KERNEL_LIMIT_OUT_OF_DOMAIN",
               f"memory.max {limit.get('bytes')!r} is not a positive integer")
    else:
        limit_bytes = int(limit["bytes"])
        if declared_cap_bytes is not None:
            delta = limit_bytes - int(declared_cap_bytes)
            if delta < -LIMIT_PAGE_TOLERANCE_BYTES:
                refuse("KERNEL_LIMIT_BELOW_THE_DECLARED_CAP",
                       f"the kernel is enforcing {limit_bytes} B where {int(declared_cap_bytes)} B "
                       f"was declared; a cap is never shrunk to make a refusal go away, and a "
                       f"record measured under a smaller limit is not the declared attempt")
            elif delta > LIMIT_PAGE_TOLERANCE_BYTES:
                refuse("KERNEL_LIMIT_ABOVE_THE_DECLARED_CAP",
                       f"the kernel is enforcing {limit_bytes} B where {int(declared_cap_bytes)} B "
                       f"was declared; the scope was not limited at the declared size")
    checks["kernel_limit_bytes"] = limit_bytes

    # -- the peak, typed, in domain, and of the right basis --------------------------------------
    host = record.get("host_ram") if isinstance(record.get("host_ram"), dict) else {}
    peak = host.get("cgroup_peak") if isinstance(host.get("cgroup_peak"), dict) else None
    if peak is None:
        refuse("PEAK_MISSING", "the record carries no cgroup peak field")
    else:
        if peak.get("status") != MEASURED:
            refuse("PEAK_NOT_MEASURED",
                   f"the cgroup peak is {peak.get('status')!r}: {peak.get('why')}")
        value = peak.get("bytes")
        if value is None:
            refuse("PEAK_BYTES_MISSING", "the cgroup peak holds no byte count")
        elif isinstance(value, bool) or not isinstance(value, int):
            refuse("PEAK_BYTES_OUT_OF_DOMAIN",
                   f"the cgroup peak {value!r} is not an integer count of bytes")
        elif value < 0:
            refuse("PEAK_BYTES_NEGATIVE",
                   f"the cgroup peak is {value}: a high-watermark cannot be negative, and a typed "
                   f"field with a checked domain is what makes that impossible to accept")
        elif value == 0:
            refuse("PEAK_BYTES_ZERO",
                   "the cgroup peak is 0: a cell that ran charged something, and a zero here is "
                   "the absence of a measurement rather than a cell that used no memory")
        elif limit_bytes is not None and value > limit_bytes:
            refuse("PEAK_ABOVE_THE_KERNEL_LIMIT",
                   f"the cgroup peak {value} exceeds the {limit_bytes} B the kernel was enforcing, "
                   f"so it was not charged to this scope")
        if peak.get("basis") != CGROUP_BASIS:
            refuse("PEAK_BASIS_IS_NOT_THE_TREE_PEAK",
                   "the peak field does not carry the cgroup-tree basis; a resident set and a "
                   "sampled floor are different quantities and never occupy this field")
        checks["peak_bytes"] = peak.get("bytes")

    # -- a reservation, confirmed and independently re-verified ---------------------------------
    reservation = record.get("reservation") if isinstance(record.get("reservation"), dict) else None
    if reservation is None:
        refuse("RESERVATION_MISSING",
               "the record carries no reservation evidence; a cell with no reservation was never "
               "admitted, and its charge is not a governed cost")
    elif reservation.get("confirmed") is not True:
        refuse("RESERVATION_NOT_CONFIRMED",
               f"the record's reservation is unconfirmed: {reservation.get('why')}")
    elif not reservation.get("lease_id"):
        refuse("RESERVATION_LEASE_ID_MISSING",
               "the reservation claims to be confirmed but names no lease")
    else:
        rc = str(reservation.get("cgroup") or "").lstrip("/")
        sc = str((scope or {}).get("cgroup") or "").lstrip("/")
        if rc and sc and rc != sc:
            refuse("RESERVATION_BOUND_TO_ANOTHER_SCOPE",
                   f"the reservation is bound to {reservation.get('cgroup')}, the record was "
                   f"charged to {(scope or {}).get('cgroup')}")
        if declared_cap_bytes is not None and reservation.get("cap_bytes") is not None and \
                int(reservation["cap_bytes"]) != int(declared_cap_bytes):
            refuse("RESERVATION_CAP_IS_NOT_THE_DECLARED_CAP",
                   f"the reservation holds {int(reservation['cap_bytes'])} B, not the declared "
                   f"{int(declared_cap_bytes)} B; a reservation is never reduced to pass")
        if verify_reservation:
            v = verify_reservation_independently(lease_id=reservation["lease_id"],
                                                 cgroup=(scope or {}).get("cgroup"),
                                                 cap_bytes=declared_cap_bytes)
            checks["reservation_verification"] = v
            if not v["verified"]:
                refuse("RESERVATION_NOT_VERIFIABLE_IN_THE_STORE",
                       f"the lease the record names could not be verified against the launcher's "
                       f"own store: {v.get('why')}")
    checks["lease_id"] = (reservation or {}).get("lease_id")

    # -- the external observation: a FURTHER check, never the only one ---------------------------
    if observed_scope and observed_scope.get("inode") is not None:
        agrees = observed_scope.get("inode") == (scope or {}).get("inode")
        checks["scope_identity_agrees_with_the_supervisor"] = agrees
        if not agrees:
            refuse("SCOPE_IS_NOT_THE_SCOPE_THE_SUPERVISOR_OBSERVED",
                   f"the supervisor observed inode {observed_scope.get('inode')} for this cell and "
                   f"the record claims {(scope or {}).get('inode')}; the charge cannot be attributed")
    else:
        checks["scope_identity_agrees_with_the_supervisor"] = None
        checks["why_no_external_observation"] = (
            "the supervisor observed no scope for this cell -- the ordinary case for a child "
            "shorter than one observation interval.  This is NOT read as agreement: acceptance "
            "rests on the minted attempt token, the boot, the clock window, the enforced kernel "
            "limit and the reservation re-verified in the launcher's store, each of which refuses "
            "on absence.")

    return {"schema": CONTRACT_SCHEMA, "accepted": not refusals, "refusals": refusals,
            "refused_by": [r["code"] for r in refusals], "checks": checks,
            "record": record, "cell_id": cell_id, "attempt_id": attempt_id, "stage": stage,
            "declared_cap_bytes": None if declared_cap_bytes is None else int(declared_cap_bytes),
            "absence_is_not_coincidence": ABSENCE_IS_NOT_COINCIDENCE}


def peak_evidence(record: dict) -> dict:
    """The retained record the launcher's `-E` reads, in the shape its admission module demands.

    A MEASURED cgroup peak is offered as `peak_scope: "cgroup"`.  An UNKNOWN one is offered as
    nothing at all: the admission module then refuses the request rather than sizing a cap on a
    silence, and a resident set is never offered here in its place.
    """
    peak = record["host_ram"]["cgroup_peak"]
    out = {"schema": "df_cell_scope_peak_evidence.v1", "cell_id": record["cell_id"],
           "stage": record["stage"], "scope": record["scope"], "recorded_at": record["recorded_at"],
           "attempt_id": record.get("attempt_id"),
           "reservation_lease_id": (record.get("reservation") or {}).get("lease_id"),
           "kernel_limit_bytes": record["kernel_limit"].get("bytes"),
           "basis": peak.get("basis"), "host_identity": record["host_identity"]}
    if record.get("usable_for_costing") is False:
        # a record its own producer marked unusable is not offered to admission as a footprint
        out["peak_scope"] = UNKNOWN
        out["why_no_peak"] = (f"this record is not usable for costing "
                              f"({record.get('why_not_usable')}), so it offers no footprint")
        return out
    if peak["status"] == MEASURED:
        out["peak_scope"] = "cgroup"
        out["peak_bytes"] = int(peak["bytes"])
        out["cgroup_peak_bytes"] = int(peak["bytes"])
    else:
        out["peak_scope"] = UNKNOWN
        out["why_no_peak"] = peak.get("why")
    return out


# ---- the external supervisor ------------------------------------------------------------------------

def _sanitised(name: str) -> str:
    """The unit stem crispdm-run builds: ${NAME//[^A-Za-z0-9_.-]/_}."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _slice_cgroup(slice_name: str) -> str | None:
    override = os.environ.get("CRISPDM_CELL_SCOPE_SLICE_CGROUP")
    if override:
        return override.strip().lstrip("/")
    try:
        r = subprocess.run(["systemctl", "--user", "show", slice_name, "-p", "ControlGroup", "--value"],
                           capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        return v.lstrip("/") or None
    except Exception:                                                       # noqa: BLE001
        return None


class _ScopeWatcher(threading.Thread):
    """The supervisor's own eyes.  It finds the scope the launcher created for THIS cell, records
    its identity independently of anything the child says, and keeps reading `memory.peak` WHILE
    THE SCOPE STILL EXISTS -- so a child that dies before it can write a record still leaves a
    tree peak behind, and the peak is never read after the scope is removed."""

    def __init__(self, slice_rel: str | None, cell_id: str, started_at: float, poll_seconds: float = 0.2):
        super().__init__(daemon=True)
        self.slice_rel, self.cell_id, self.started_at = slice_rel, cell_id, started_at
        self.poll_seconds = poll_seconds
        self.stop = threading.Event()
        self.identity = None
        self.last_peak = None
        self.samples = 0

    def _find(self):
        if not self.slice_rel:
            return None
        base = cgroup_root() / self.slice_rel
        pattern = f"crispdm-{_sanitised(self.cell_id)}-*.scope"
        best = None
        for d in base.glob(pattern):
            try:
                st = d.stat()
            except OSError:
                continue
            if st.st_ctime + 2.0 < self.started_at:
                continue                     # an older scope of the same name is NOT this launch
            if best is None or st.st_ctime > best[0]:
                best = (st.st_ctime, d)
        return best[1] if best else None

    def run(self):
        while not self.stop.is_set():
            d = self._find()
            if d is not None:
                rel = str(d.relative_to(cgroup_root()))
                if self.identity is None:
                    self.identity = scope_identity(rel)
                p = cgroup_peak(rel)
                if p["status"] == MEASURED:
                    self.last_peak = {"bytes": p["bytes"], "status": FLOOR, "basis": FLOOR_BASIS,
                                      "cgroup": rel, "observed_at": time.time(),
                                      "observation_index": self.samples + 1,
                                      "poll_seconds": self.poll_seconds}
                    self.samples += 1
            self.stop.wait(self.poll_seconds)


def supervise(*, cell_id: str, argv: list, cap_bytes: int, wall_seconds: int, supervisor_dir,
              log_path, slice_name: str = "crispdm-batch.slice", launcher=None, env: dict | None = None,
              claims_dir=None, record_path=None, stage: str = "CELL", queue: bool = False,
              peak_evidence_path=None, peak_bytes=None) -> dict:
    """Launch ONE cell through the deployed launcher and retain what happened to it.

    The supervisor is EXTERNAL: it is not in the child's scope, it does not depend on the child
    writing anything, and it keeps the termination status of a child that fails, is signalled,
    exhausts its wall limit or is refused admission before it ever started.

    The cap is declared ONCE, at the size the work needs.  A refusal is terminal here: nothing is
    retried, and above all nothing is re-asked smaller -- the launcher itself makes a lowered cap
    after a refusal terminal, and this supervisor never attempts it.
    """
    run = Path(launcher) if launcher else launcher_path()
    if run is None or not Path(run).is_file():
        raise ScopeRefusal("LAUNCHER_NOT_INSTALLED",
                           "crispdm-run is not installed; a cell is never started with a bare "
                           "subprocess, because a bare subprocess takes no scope and no reservation")
    sup = Path(supervisor_dir)
    sup.mkdir(parents=True, exist_ok=True)
    cap_bytes = int(cap_bytes)

    cmd = [str(run), "-m", str(cap_bytes), "-t", str(int(wall_seconds)), "-n", str(cell_id),
           "-L", f"cell:{cell_id}", "-S", slice_name]
    if queue:
        cmd += ["-q"]
    if peak_evidence_path:
        cmd += ["-E", str(peak_evidence_path)]
        if peak_bytes is not None:
            cmd += ["-P", str(int(peak_bytes))]
    cmd += ["--"] + [str(x) for x in argv]

    # The token that makes freshness checkable.  It is minted HERE, before the child exists, so a
    # record already lying at `record_path` cannot carry it: that is what turns "this record is
    # about the run I just did" from an assumption into a comparison the gate performs.
    attempt = new_attempt_id()

    child_env = dict(env if env is not None else os.environ)
    child_env["CRISPDM_CELL_SCOPE_PARENT_CGROUP"] = own_cgroup() or ""
    child_env["CRISPDM_CELL_SCOPE_CELL_ID"] = str(cell_id)
    child_env[ATTEMPT_ENV] = attempt
    child_env[STAGE_ENV] = str(stage)
    child_env["CRISPDM_CELL_SCOPE_DECLARED_CAP"] = str(cap_bytes)
    if claims_dir:
        child_env["CRISPDM_CELL_SCOPE_CLAIMS"] = str(claims_dir)

    slice_rel = _slice_cgroup(slice_name)
    started = time.time()
    watcher = _ScopeWatcher(slice_rel, cell_id, started)
    watcher.start()
    wall0 = time.monotonic()
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        lease, lease_why = _find_lease(watcher, proc, deadline=started + min(30, wall_seconds))
        code = proc.wait()
    wall = time.monotonic() - wall0
    watcher.stop.set()
    watcher.join(timeout=5)

    term = _termination(code)
    term["launched"] = term["status"] != "REFUSED_BY_ADMISSION"
    term["wall_seconds"] = wall
    term["cap_bytes_declared"] = cap_bytes
    term["cap_asked_once_at_the_declared_size"] = True
    term["command"] = cmd

    finished = time.time()
    document, document_why = None, None
    if not record_path:
        document_why = ("no record_path was given, so this attempt was never told where to write "
                        "its in-scope reading")
    elif not Path(record_path).is_file():
        document_why = (f"this attempt retained no record at {record_path}: the absence of a "
                        f"reading is a refusal, not a coincidence")
    else:
        try:
            document = json.loads(Path(record_path).read_text())
        except Exception as e:                                              # noqa: BLE001
            document_why = f"the retained document is not readable JSON: {type(e).__name__}: {e}"

    # F1.  The gate.  Not "is there a MEASURED field somewhere in a file that happens to be here",
    # but "does this document describe THIS attempt of THIS cell, completely and in domain".
    contract = verify_fresh_attempt(
        document, cell_id=cell_id, attempt_id=attempt, stage=stage,
        declared_cap_bytes=cap_bytes, started_at=started, finished_at=finished,
        observed_scope=watcher.identity)
    if document_why and not contract["accepted"]:
        contract["why_no_document"] = document_why
    child_record = contract["record"] if contract["accepted"] else None

    rec = {"schema": "df_cell_scope_supervision.v2", "cell_id": cell_id, "stage": stage,
           "attempt_id": attempt, "declared_cap_bytes": cap_bytes,
           "fresh_attempt_contract": contract,
           "absence_is_not_coincidence": ABSENCE_IS_NOT_COINCIDENCE,
           "started_at": started, "finished_at": finished,
           "termination": term, "lease_id": lease,
           "lease_confirmed": lease is not None,
           "why_lease_unconfirmed": None if lease is not None else lease_why,
           "scope": watcher.identity or {"cgroup": None, "inode": None, "path_exists": False,
                                         "why": "no scope for this cell was observed before it ended"},
           "supervisor_sampled_peak_floor": watcher.last_peak or {
               "bytes": None, "status": UNKNOWN, "basis": FLOOR_BASIS,
               "why": ("the supervisor observed no memory.peak before this cell's scope was "
                       "removed -- a child shorter than one observation interval leaves none.  "
                       "This is UNKNOWN, not zero, and it would not be a peak even if it existed")},
           "supervisor_peak_observations": watcher.samples,
           "retained_lease_peak_is_not_read": RETAINED_LEASE_PEAK_IS_NOT_READ,
           "child_record": child_record,
           "host_identity": host_identity()}

    # the authoritative host-RAM peak: the child's own in-scope read, and ONLY when the whole
    # fresh-attempt contract accepted it.  A MEASURED field in a document that failed the contract
    # is not a cost of this attempt and never enters this field.
    if child_record:
        peak = dict(child_record["host_ram"]["cgroup_peak"])
        peak["read_by"] = "THE_CHILD_INSIDE_ITS_OWN_SCOPE_BEFORE_THE_SCOPE_WAS_REMOVED"
        peak["accepted_by"] = CONTRACT_SCHEMA
    elif document is not None:
        peak = {"bytes": None, "status": UNKNOWN, "basis": CGROUP_BASIS, "read_by": None,
                "why": ("a record was retained for this attempt but it did NOT satisfy the "
                        "fresh-attempt contract, so nothing in it is this attempt's cost: "
                        + ", ".join(contract["refused_by"]))}
    else:
        # A sampled observation is NOT promoted here, whatever the supervisor saw.  Only a read
        # taken from inside the scope is a peak; the floor stays in its own field, under its own
        # name, so no later reader can mistake the two.
        peak = {"bytes": None, "status": UNKNOWN, "basis": CGROUP_BASIS, "read_by": None,
                "why": ("this cell wrote no in-scope memory.peak read "
                        f"(termination: {term['status']}); a missing peak is UNKNOWN, never zero, "
                        "never a success, and never filled in from a sampled floor or from the "
                        "launcher's retained lease record")}
    rec["host_ram"] = {"cgroup_peak": peak,
                       "cgroup_peak_floor": rec["supervisor_sampled_peak_floor"],
                       "comparability": COMPARABILITY,
                       "retained_lease_peak_is_not_read": RETAINED_LEASE_PEAK_IS_NOT_READ}
    if (peak["status"] == MEASURED and rec["supervisor_sampled_peak_floor"].get("bytes")):
        f, k = int(rec["supervisor_sampled_peak_floor"]["bytes"]), int(peak["bytes"])
        rec["host_ram"]["floor_against_in_scope_peak"] = {
            "floor_bytes": f, "in_scope_peak_bytes": k, "ratio": (k / f) if f else None,
            "direction": ("SAMPLED_BELOW_THE_IN_SCOPE_READ" if f < k else
                          "SAMPLED_ABOVE_THE_IN_SCOPE_READ" if f > k else "EQUAL"),
            "reading": (
                "one observation of two readings of ONE scope at two instants, and the comparison "
                "is UNSIGNED.  Both directions have now been observed on real loads: the "
                "classification lane saw 27,262,976 B sampled against 258,584,576 B read in "
                "scope (sampled far below), and the QRM01 producer-to-supervisor run of "
                "2026-09-29 saw 559,525,888 B sampled against 557,654,016 B read in scope "
                "(sampled ABOVE), because the child reads its own memory.peak before it has "
                "finished and the kernel watermark keeps rising afterwards.  So a sampled "
                "observation is a lower bound on the scope's LIFETIME watermark and on nothing "
                "else: it does not bound the child's in-scope read in either direction, it is "
                "not a correction factor, and neither number may be substituted for the other."),
            "what_bounds_both": ("host_ram.cgroup_lifetime_peak of the record: memory.peak is "
                                 "monotone while it is never reset, so the lifetime watermark is "
                                 "at least as large as any reading taken during the scope's life")}
    rec["usable_for_costing"] = bool(contract["accepted"] and term["status"] == "COMPLETED")
    rec["why_not_usable"] = None if rec["usable_for_costing"] else "; ".join(
        [f"{r['code']}: {r['detail']}" for r in contract["refusals"]]
        or ([] if term["status"] == "COMPLETED" else
            [f"TERMINATION_IS_NOT_COMPLETED: {term['status']}"]))
    rec["refused_by"] = list(contract["refused_by"])
    rec["scope_identity_agrees_with_the_child"] = \
        contract["checks"].get("scope_identity_agrees_with_the_supervisor")

    tmp = sup / f".{_sanitised(cell_id)}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    tmp.replace(sup / f"{_sanitised(cell_id)}.json")
    return rec


def _termination(code: int) -> dict:
    """What actually ended the child, kept apart from what the science says.

    75 is the launcher's refusal: nothing was started, no limit and no kernel setting was changed,
    and this must never be recorded as a cell that ran and failed.  124 is `timeout`'s wall limit.
    128+N and a negative return code are both a signal.
    """
    sig = None
    if code is not None and code < 0:
        sig = -code
    elif code is not None and 128 < code < 160:
        sig = code - 128
    if code == 0:
        status = "COMPLETED"
    elif code == 75:
        status = "REFUSED_BY_ADMISSION"
    elif code == 124:
        status = "WALL_TIMEOUT"
    else:
        status = "FAILED"
    return {"status": status, "exit_code": code, "signal": sig,
            "meaning": {"COMPLETED": "the child exited 0",
                        "REFUSED_BY_ADMISSION": "the launcher refused: NOTHING was started and no limit or kernel setting was changed",
                        "WALL_TIMEOUT": "the wall limit expired and the child was stopped",
                        "FAILED": "the child ended non-zero" + (f" on signal {sig}" if sig else "")}[status]}


def _find_lease(watcher: "_ScopeWatcher", proc, deadline: float):
    """The reservation this cell's scope is bound to, read from the admission store while the load
    is live.  Returns (lease_id, why_not).

    The supervisor CONFIRMS a reservation exists instead of assuming the launcher took one.  An
    unconfirmed lease says why: `could not confirm` and `there is none` are different statements,
    and recording the first as a bare null is how a reservation the launcher did take came to be
    reported as absent.
    """
    try:
        adm = admission_module()
        store = adm.Store()
        res = adm.resources_from_env()
    except Exception as e:                                                  # noqa: BLE001
        return None, f"the admission store could not be read: {type(e).__name__}: {e}"
    why = "the cell's scope was never observed while the load was live"
    while time.time() < deadline and proc.poll() is None:
        ident = watcher.identity
        if ident and ident.get("cgroup"):
            why = f"no live reservation was bound to {ident['cgroup']} while the load ran"
            try:
                for l in adm.state(store, res, time.time())["live"]:
                    if l.get("cgroup") and str(l["cgroup"]).lstrip("/") == ident["cgroup"]:
                        return l["lease_id"], None
                    if l.get("unit") and ident.get("unit") == l["unit"]:
                        return l["lease_id"], None
            except Exception as e:                                          # noqa: BLE001
                return None, f"the admission store could not be read: {type(e).__name__}: {e}"
        time.sleep(0.2)
    return None, why


# ---- the deployed runner's identity ----------------------------------------------------------------

def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(HERE.parent), *args], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                                       # noqa: BLE001
        return None


def verify_runner_identity() -> dict:
    """Verify the DEPLOYED runner, not only the tracked bytes.

    A checkout can carry a repaired launcher while the copy that actually starts cells is an older
    one; the two are different questions.  This reports, for each half of the runner: the digest of
    the file that WILL run, the digest tracked in this checkout, whether they agree, and the commit
    the tracked bytes are reachable from -- so a measured cell can name the runner that produced it.
    """
    pairs = {"launcher": (launcher_path(), HERE / "crispdm-run", "tools/crispdm-run"),
             "admission_module": (Path(os.environ.get("CRISPDM_ADMISSION_MODULE")
                                       or Path.home() / ".local/libexec/crispdm/crispdm_admission.py"),
                                  HERE / "crispdm_admission.py", "tools/crispdm_admission.py")}
    out = {"schema": RUNNER_SCHEMA, "checked_at": time.time(),
           "integration_commit": _git("rev-parse", "HEAD"),
           "integration_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
           "host_identity": host_identity()}
    agree = True
    for key, (deployed, tracked, rel) in pairs.items():
        d_sha = _sha(deployed) if deployed and Path(deployed).is_file() else None
        t_sha = _sha(tracked) if tracked.is_file() else None
        match = (d_sha is not None and d_sha == t_sha)
        agree = agree and match
        out[key] = {"deployed_path_exists": bool(d_sha), "deployed_sha256": d_sha,
                    "tracked_path": rel, "tracked_sha256": t_sha,
                    "matches_tracked_blob": match,
                    "tracked_blob": _git("rev-parse", f"HEAD:{rel}"),
                    "reachable_from": _git("rev-parse", "HEAD") if t_sha else None,
                    "why": None if match else
                           "the bytes that will start a cell are NOT the bytes this checkout tracks"}
    out["systemd_user_available"] = launcher_available()
    out["verdict"] = "DEPLOYED_MATCHES_TRACKED" if agree else "DEPLOYED_DIFFERS_FROM_TRACKED"
    return out


def _sha(p) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:                                                       # noqa: BLE001
        return None


# ---- CLI --------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="df_cell_scope", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity", help="this process's scope identity, ancestry and kernel limit")
    sub.add_parser("peak", help="this scope's cgroup peak and this process's resident set, labelled apart")
    vr = sub.add_parser("verify-runner", help="the DEPLOYED runner's identity against the tracked bytes")
    vr.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "identity":
        print(json.dumps({"scope": scope_identity(), "kernel_limit": kernel_limit(),
                          "parent_cgroup_of_the_driver": os.environ.get("CRISPDM_CELL_SCOPE_PARENT_CGROUP"),
                          "host_identity": host_identity()}, indent=1, sort_keys=True))
        return 0
    if a.cmd == "peak":
        print(json.dumps({"cgroup_peak": cgroup_peak(), "process_rss_peak": process_rss_peak(),
                          "comparability": COMPARABILITY, "gpu": gpu_memory()}, indent=1, sort_keys=True))
        return 0
    v = verify_runner_identity()
    print(json.dumps(v, indent=1, sort_keys=True))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1, sort_keys=True))
    return 0 if v["verdict"] == "DEPLOYED_MATCHES_TRACKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
