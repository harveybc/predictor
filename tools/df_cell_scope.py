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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCHEMA = "df_cell_scope_record.v1"
RUNNER_SCHEMA = "df_runner_identity.v1"
MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"

RSS_BASIS = ("MAIN_PROCESS_RSS_ONLY: resource.getrusage(RUSAGE_SELF).ru_maxrss of this process "
             "alone.  NOT a process-tree or cgroup peak: it excludes every other task in the "
             "cgroup and all cgroup-charged page and kernel memory.  Never size a cap with it.")
CGROUP_BASIS = ("CGROUP_TREE_PEAK: memory.peak of this cell's OWN exclusive scope -- the complete "
                "process tree charged to it, plus the cgroup's page and kernel memory.  This is "
                "the only host-RAM basis a MemoryMax or a placement decision may be judged on.")
FLOOR = "FLOOR_NOT_A_PEAK"
FLOOR_BASIS = (
    "SAMPLED_FLOOR: the last memory.peak this cgroup was OBSERVED to hold before its scope was "
    "removed.  It is a LOWER BOUND, never a footprint: a child that ends between two observations "
    "-- or before the first -- takes its true high-watermark away with its scope.  Measured on a "
    "short child by the classification lane: 27,262,976 B retained against 258,584,576 B read from "
    "inside the same scope, an undercount of 9.5x.  A cap sized on this number would be nine times "
    "too small, so it never shares a name or a field with a peak.")
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
                      extra: dict | None = None) -> dict:
    """Everything the order names, each quantity in its own labelled field.

    `stage` is mandatory and is carried into the peak evidence: a data-materialization floor and a
    training peak are different measurements of different things, and a record that does not say
    which it is can be misread as the other.
    """
    ident = (claim or {}).get("scope") or scope_identity(rel)
    rel = ident.get("cgroup")
    peak = cgroup_peak(rel)
    rec = {
        "schema": SCHEMA,
        "cell_id": cell_id,
        "stage": stage,
        "recorded_at": time.time(),
        "scope": ident,
        "kernel_limit": (claim or {}).get("kernel_limit") or kernel_limit(rel),
        "host_ram": {"cgroup_peak": peak, "process_rss_peak": process_rss_peak(),
                     "comparability": COMPARABILITY},
        "gpu": gpu_memory(),
        "optimizer_updates": None if updates is None else int(updates),
        "cpu_seconds": None if cpu_seconds is None else float(cpu_seconds),
        "wall_seconds": None if wall_seconds is None else float(wall_seconds),
        "host_identity": host_identity(),
        "usable_for_costing": peak["status"] == MEASURED,
        "why_not_usable": None if peak["status"] == MEASURED else
                          "the cell's own cgroup peak is UNKNOWN, so this cell costs nothing and bounds nothing",
    }
    if extra:
        rec.update(extra)
    return rec


def peak_evidence(record: dict) -> dict:
    """The retained record the launcher's `-E` reads, in the shape its admission module demands.

    A MEASURED cgroup peak is offered as `peak_scope: "cgroup"`.  An UNKNOWN one is offered as
    nothing at all: the admission module then refuses the request rather than sizing a cap on a
    silence, and a resident set is never offered here in its place.
    """
    peak = record["host_ram"]["cgroup_peak"]
    out = {"schema": "df_cell_scope_peak_evidence.v1", "cell_id": record["cell_id"],
           "stage": record["stage"], "scope": record["scope"], "recorded_at": record["recorded_at"],
           "kernel_limit_bytes": record["kernel_limit"].get("bytes"),
           "basis": peak.get("basis"), "host_identity": record["host_identity"]}
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

    child_env = dict(env if env is not None else os.environ)
    child_env["CRISPDM_CELL_SCOPE_PARENT_CGROUP"] = own_cgroup() or ""
    child_env["CRISPDM_CELL_SCOPE_CELL_ID"] = str(cell_id)
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

    child_record = None
    if record_path and Path(record_path).is_file():
        try:
            child_record = json.loads(Path(record_path).read_text())
        except Exception:                                                   # noqa: BLE001
            child_record = None

    rec = {"schema": "df_cell_scope_supervision.v1", "cell_id": cell_id, "stage": stage,
           "started_at": started, "termination": term, "lease_id": lease,
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

    # the authoritative host-RAM peak: the child's own in-scope read if it exists, else the
    # supervisor's last pre-removal read, else UNKNOWN.  Never zero and never inferred from RSS.
    if child_record and child_record.get("host_ram", {}).get("cgroup_peak", {}).get("status") == MEASURED:
        peak = dict(child_record["host_ram"]["cgroup_peak"])
        peak["read_by"] = "THE_CHILD_INSIDE_ITS_OWN_SCOPE_BEFORE_THE_SCOPE_WAS_REMOVED"
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
            "reading": ("one observation of how far a sampled floor fell short of the in-scope "
                        "read on this cell.  It is an observation, not a correction factor and "
                        "not an invariant: the shortfall depends on when the child ended")}
    rec["usable_for_costing"] = bool(peak["status"] == MEASURED and term["status"] == "COMPLETED")

    if watcher.identity and child_record and child_record.get("scope", {}).get("inode") is not None:
        same = watcher.identity.get("inode") == child_record["scope"]["inode"]
        rec["scope_identity_agrees_with_the_child"] = same
        if not same:
            rec["usable_for_costing"] = False
            rec["why_not_usable"] = ("the scope the supervisor observed is not the scope the child "
                                     "reported; the cell's charge cannot be attributed")

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
