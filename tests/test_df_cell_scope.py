#!/usr/bin/env python3
"""QRM01: the cell-scope instrument, tested BEFORE it exists.

Every cell of a block must run in a FRESH EXCLUSIVE scope that encloses its complete process
tree and carries its OWN reservation.  One sequential child inside a reused driver scope is the
defect this file exists to reject: it is what produced a 7.4 G kill-time watermark of a
multi-child wrapper tree and an 8,458,399,744 B main-process resident set, two different
quantities that were then circulated as one cell's footprint.

The six contracts the order names, one test each, plus the runner-identity check:

  1. two concurrent children               -> two distinct scopes, two distinct reservations
  2. a reused scope                        -> REFUSED by name, never measured
  3. a missing peak                        -> UNKNOWN; never 0, never a success
  4. a failed child                        -> an EXTERNAL supervisor retains the termination status
  5. aggregate admission                   -> a fresh per-child admission, refused on the aggregate
  6. reservation release                   -> the capacity comes back after the tree ends

Bounded fixtures only.  No test pressures host memory, and no test provokes an OOM kill: the
admission readings come from a JSON file (FileResources, the module's own simulated host) or the
caps are small enough (256 MiB) that a real scope is trivial.  A test that needs a real transient
scope skips when systemd --user is not available rather than faking one and claiming it ran.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
GIB = 1024 ** 3
MIB = 1024 ** 2


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load("df_cell_scope")


# ---- bounded fixtures -----------------------------------------------------------------------

def make_cgroup(root: Path, rel: str, *, peak=None, limit=None, procs=()) -> Path:
    """A simulated cgroup directory.  Nothing here charges a byte of real memory."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    if peak is not None:
        (d / "memory.peak").write_text(f"{int(peak)}\n")
    if limit is not None:
        (d / "memory.max").write_text("max\n" if limit == "max" else f"{int(limit)}\n")
    (d / "cgroup.procs").write_text("".join(f"{p}\n" for p in procs))
    return d


@pytest.fixture()
def cg(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    root.mkdir()
    monkeypatch.setenv("CRISPDM_CGROUP_ROOT", str(root))
    return root


@pytest.fixture()
def claims(tmp_path):
    d = tmp_path / "SCOPE_CLAIMS"
    d.mkdir()
    return d


SLICE_REL = "user.slice/crispdm-batch.slice"


def readings(path: Path, **over) -> Path:
    """The simulated host the admission module reads.  `mem_available_bytes` is a number in a
    file: no test ever has to make a real host short of memory to see a refusal."""
    doc = {"mem_available_bytes": 8 * GIB, "mem_total_bytes": 32 * GIB,
           "slice_memory_max": 2 * GIB, "slice_memory_current": 0,
           "pressure_some_avg10": 0.0, "boot_id": "bounded-fixture-boot",
           "host_key": "boundedfixture01", "alive": {}, "cgroup_current": {}, "cgroup_peak": {}}
    doc.update(over)
    path.write_text(json.dumps(doc))
    return path


# ---- 1. two concurrent children --------------------------------------------------------------

def test_two_concurrent_children_take_two_distinct_scopes_and_two_distinct_reservations(cg, claims, tmp_path, monkeypatch):
    """The failure this rejects: both cells charged to ONE cgroup, so each reads the other's
    allocations in its own peak and neither figure is that cell's footprint."""
    a = make_cgroup(cg, f"{SLICE_REL}/crispdm-cell_a-1-11.scope", peak=101 * MIB, limit=256 * MIB)
    b = make_cgroup(cg, f"{SLICE_REL}/crispdm-cell_b-1-12.scope", peak=202 * MIB, limit=256 * MIB)

    barrier = threading.Barrier(2)
    out = {}

    def claim(cell, rel):
        barrier.wait(timeout=10)                       # the two claims race on purpose
        ident = CS.scope_identity(rel)
        out[cell] = CS.require_fresh_exclusive_scope(claims, cell, identity=ident,
                                                     parent_cgroup="user.slice/driver.scope")

    ta = threading.Thread(target=claim, args=("cell_a", f"{SLICE_REL}/crispdm-cell_a-1-11.scope"))
    tb = threading.Thread(target=claim, args=("cell_b", f"{SLICE_REL}/crispdm-cell_b-1-12.scope"))
    ta.start(); tb.start(); ta.join(10); tb.join(10)

    assert set(out) == {"cell_a", "cell_b"}
    ia, ib = out["cell_a"]["scope"], out["cell_b"]["scope"]
    assert ia["cgroup"] != ib["cgroup"]
    assert ia["inode"] != ib["inode"], "two scopes that share an inode are one scope"
    assert a.name in ia["unit"] and b.name in ib["unit"]

    ra = CS.cell_scope_record("cell_a", stage="TRAIN_UPDATES", claim=out["cell_a"], updates=600,
                              cpu_seconds=1.0, wall_seconds=2.0)
    rb = CS.cell_scope_record("cell_b", stage="TRAIN_UPDATES", claim=out["cell_b"], updates=600,
                              cpu_seconds=1.0, wall_seconds=2.0)
    # each peak is its OWN scope's, and the two are not the same number by construction
    assert ra["host_ram"]["cgroup_peak"]["bytes"] == 101 * MIB
    assert rb["host_ram"]["cgroup_peak"]["bytes"] == 202 * MIB
    assert ra["host_ram"]["cgroup_peak"]["status"] == CS.MEASURED
    assert ra["kernel_limit"]["bytes"] == 256 * MIB
    # host RAM and GPU are recorded SEPARATELY and are never added or compared
    assert "gpu" in ra and "cgroup_peak" not in ra["gpu"]
    assert ra["usable_for_costing"] is True


@pytest.mark.skipif(not CS.launcher_available(), reason="the deployed launcher or systemd --user is not available here")
def test_two_concurrent_children_under_the_real_launcher_get_two_real_scopes(tmp_path, monkeypatch):
    """The same contract against the DEPLOYED launcher and the real kernel, at 256 MiB a child:
    bounded, and it displaces nothing."""
    store = tmp_path / "admission"
    monkeypatch.setenv("CRISPDM_ADMISSION_DIR", str(store))
    sup = tmp_path / "SUPERVISOR"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, os, sys, time\n"
        "sys.path.insert(0, %r)\n" % str(TOOLS) +
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('df_cell_scope', %r)\n" % str(TOOLS / "df_cell_scope.py") +
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "claim = m.require_fresh_exclusive_scope(os.environ['CLAIMS'], sys.argv[1],\n"
        "                                        parent_cgroup=os.environ.get('CRISPDM_CELL_SCOPE_PARENT_CGROUP'))\n"
        "time.sleep(1.0)\n"
        "rec = m.cell_scope_record(sys.argv[1], stage='FIXTURE', claim=claim, updates=0)\n"
        "open(os.environ['OUT'], 'w').write(json.dumps(rec))\n")
    claims = tmp_path / "claims"; claims.mkdir()

    res = {}

    def run(cell):
        env = {**os.environ, "CLAIMS": str(claims), "OUT": str(tmp_path / f"{cell}.json")}
        res[cell] = CS.supervise(cell_id=cell, argv=[sys.executable, str(child), cell],
                                 cap_bytes=256 * MIB, wall_seconds=120, supervisor_dir=sup,
                                 log_path=tmp_path / f"{cell}.log", env=env)

    ts = [threading.Thread(target=run, args=(c,)) for c in ("real_a", "real_b")]
    for t in ts: t.start()
    for t in ts: t.join(300)

    assert set(res) == {"real_a", "real_b"}, res
    for cell in ("real_a", "real_b"):
        assert res[cell]["termination"]["status"] == "COMPLETED", res[cell]
    sa = json.loads((tmp_path / "real_a.json").read_text())["scope"]
    sb = json.loads((tmp_path / "real_b.json").read_text())["scope"]
    assert sa["cgroup"] != sb["cgroup"] and sa["inode"] != sb["inode"]
    assert sa["cgroup"].endswith(".scope") and "crispdm-real_a-" in sa["cgroup"]
    # the supervisor identified the scope itself instead of trusting the child's word for it
    assert res["real_a"]["scope"]["cgroup"] == sa["cgroup"]
    assert res["real_a"]["lease_id"] and res["real_a"]["lease_id"] != res["real_b"]["lease_id"]


# ---- 2. a reused scope is rejected ------------------------------------------------------------

def test_a_child_in_the_drivers_own_scope_is_refused_by_name(cg, claims):
    """`One sequential child inside a reused driver scope is NOT sufficient` -- so it is refused,
    not silently measured."""
    driver = f"{SLICE_REL}/crispdm-driver-1-9.scope"
    make_cgroup(cg, driver, peak=7 * GIB, limit=9 * GIB)
    with pytest.raises(CS.ScopeRefusal) as e:
        CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(driver),
                                         parent_cgroup=driver)
    assert e.value.code == "REUSED_DRIVER_SCOPE"
    assert not list(Path(claims).glob("*.json")), "a refused cell must leave no claim behind"


def test_a_second_cell_claiming_a_claimed_scope_is_refused_by_name(cg, claims):
    rel = f"{SLICE_REL}/crispdm-shared-1-9.scope"
    make_cgroup(cg, rel, peak=3 * GIB, limit=9 * GIB)
    CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                     parent_cgroup="user.slice/driver.scope")
    with pytest.raises(CS.ScopeRefusal) as e:
        CS.require_fresh_exclusive_scope(claims, "cell_b", identity=CS.scope_identity(rel),
                                         parent_cgroup="user.slice/driver.scope")
    assert e.value.code == "SCOPE_ALREADY_CLAIMED_BY_ANOTHER_CELL"
    assert e.value.detail and "cell_a" in e.value.detail


def test_a_cgroup_that_is_not_a_scope_is_refused(cg, claims):
    rel = "user.slice/user-1000.slice/session.slice"
    make_cgroup(cg, rel, peak=1 * GIB, limit="max")
    with pytest.raises(CS.ScopeRefusal) as e:
        CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                         parent_cgroup="user.slice/driver.scope")
    assert e.value.code in ("NOT_A_SCOPE", "NO_KERNEL_LIMIT")


def test_the_same_cell_re_entering_its_own_claim_is_not_a_reuse(cg, claims):
    rel = f"{SLICE_REL}/crispdm-cell_a-1-11.scope"
    make_cgroup(cg, rel, peak=1 * GIB, limit=2 * GIB)
    first = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                             parent_cgroup="user.slice/driver.scope")
    again = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                             parent_cgroup="user.slice/driver.scope")
    assert again["scope"]["inode"] == first["scope"]["inode"]
    assert again["reentrant"] is True


# ---- 3. a missing peak is UNKNOWN --------------------------------------------------------------

def test_a_missing_peak_is_unknown_and_is_never_zero_and_never_a_success(cg, claims):
    rel = f"{SLICE_REL}/crispdm-cell_a-1-11.scope"
    make_cgroup(cg, rel, peak=None, limit=512 * MIB)          # memory.peak absent: unreadable cgroup
    claim = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                            parent_cgroup="user.slice/driver.scope")
    rec = CS.cell_scope_record("cell_a", stage="TRAIN_UPDATES", claim=claim, updates=600,
                               cpu_seconds=1.0, wall_seconds=2.0)
    peak = rec["host_ram"]["cgroup_peak"]
    assert peak["status"] == CS.UNKNOWN
    assert peak["bytes"] is None and peak["bytes"] != 0
    assert rec["usable_for_costing"] is False
    assert "UNKNOWN" in json.dumps(rec)
    # and the number must not be substituted from the process RSS, which is a different quantity
    assert rec["host_ram"]["process_rss_peak"]["bytes"] != peak["bytes"]
    assert "NOT_INTERCHANGEABLE" in rec["host_ram"]["comparability"]


def test_an_unknown_peak_cannot_size_an_admission(tmp_path, cg, claims):
    """A cap is sized from a retained CGROUP peak.  An UNKNOWN one is not evidence, and the
    admission module must refuse it rather than read a zero."""
    rel = f"{SLICE_REL}/crispdm-cell_a-1-11.scope"
    make_cgroup(cg, rel, peak=None, limit=512 * MIB)
    claim = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                            parent_cgroup="user.slice/driver.scope")
    rec = CS.cell_scope_record("cell_a", stage="TRAIN_UPDATES", claim=claim)
    ev = tmp_path / "peak_evidence.json"
    ev.write_text(json.dumps(CS.peak_evidence(rec)))
    adm = CS.admission_module()
    with pytest.raises(adm.Refusal):
        adm.read_peak_evidence(ev, None)


def test_a_measured_peak_is_offered_to_admission_as_a_cgroup_peak(tmp_path, cg, claims):
    rel = f"{SLICE_REL}/crispdm-cell_a-1-11.scope"
    make_cgroup(cg, rel, peak=1_463_877_632, limit=6 * GIB)
    claim = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                            parent_cgroup="user.slice/driver.scope")
    rec = CS.cell_scope_record("cell_a", stage="DATA_MATERIALIZATION", claim=claim)
    ev = tmp_path / "peak_evidence.json"
    ev.write_text(json.dumps(CS.peak_evidence(rec)))
    adm = CS.admission_module()
    found, scope, sha = adm.read_peak_evidence(ev, 1_463_877_632)
    assert found == 1_463_877_632 and scope in adm.TREE_PEAK_SCOPES and len(sha) == 64
    # the stage is carried: a data-stage floor is not a training cap
    assert rec["stage"] == "DATA_MATERIALIZATION"
    assert json.loads(ev.read_text())["stage"] == "DATA_MATERIALIZATION"


# ---- 4. a failed child ------------------------------------------------------------------------

def test_the_supervisor_retains_the_termination_status_of_a_failed_child(tmp_path, monkeypatch):
    sup = tmp_path / "SUPERVISOR"
    launcher = _stub_launcher(tmp_path, exit_code=3)
    r = CS.supervise(cell_id="cell_fail", argv=[sys.executable, "-c", "raise SystemExit(3)"],
                     cap_bytes=256 * MIB, wall_seconds=60, supervisor_dir=sup,
                     log_path=tmp_path / "cell_fail.log", launcher=launcher)
    assert r["termination"]["status"] == "FAILED"
    assert r["termination"]["exit_code"] == 3
    held = json.loads((sup / "cell_fail.json").read_text())
    assert held["termination"]["exit_code"] == 3
    assert held["termination"]["status"] == "FAILED"
    # a failed child wrote no record, so its peak is UNKNOWN -- not zero and not a success
    assert held["host_ram"]["cgroup_peak"]["status"] == CS.UNKNOWN
    assert held["host_ram"]["cgroup_peak"]["bytes"] is None
    assert held["usable_for_costing"] is False


def test_the_supervisor_retains_a_signalled_child_and_says_which_signal(tmp_path):
    sup = tmp_path / "SUPERVISOR"
    launcher = _stub_launcher(tmp_path, exit_code=137)       # 128 + SIGKILL, as a shell reports it
    r = CS.supervise(cell_id="cell_killed", argv=[sys.executable, "-c", "pass"],
                     cap_bytes=256 * MIB, wall_seconds=60, supervisor_dir=sup,
                     log_path=tmp_path / "cell_killed.log", launcher=launcher)
    assert r["termination"]["status"] == "FAILED"
    assert r["termination"]["signal"] == 9
    assert r["termination"]["exit_code"] == 137
    assert json.loads((sup / "cell_killed.json").read_text())["termination"]["signal"] == 9


def test_a_launcher_refusal_is_retained_and_nothing_is_reported_as_run(tmp_path):
    sup = tmp_path / "SUPERVISOR"
    launcher = _stub_launcher(tmp_path, exit_code=75, stderr="crispdm-run: REFUSED")
    r = CS.supervise(cell_id="cell_refused", argv=[sys.executable, "-c", "pass"],
                     cap_bytes=256 * MIB, wall_seconds=60, supervisor_dir=sup,
                     log_path=tmp_path / "cell_refused.log", launcher=launcher)
    assert r["termination"]["status"] == "REFUSED_BY_ADMISSION"
    assert r["termination"]["launched"] is False
    assert r["usable_for_costing"] is False


def _stub_launcher(tmp_path: Path, *, exit_code: int, stderr: str = "") -> Path:
    """A BOUNDED stand-in for the deployed launcher: it accepts the same flags and returns the
    exit code under test.  It is a fixture for the supervisor's own accounting, never a second
    scheduler: the supervisor's production launcher is the deployed crispdm-run and the real
    launcher is exercised by its own test above."""
    p = tmp_path / "stub-crispdm-run"
    p.write_text("#!/usr/bin/env bash\n"
                 "while getopts \"m:t:n:W:L:E:P:S:qh\" o; do :; done\n"
                 "shift $((OPTIND - 1)); [ \"${1:-}\" = \"--\" ] && shift\n"
                 f"[ -n \"{stderr}\" ] && echo \"{stderr}\" >&2\n"
                 f"exit {exit_code}\n")
    p.chmod(0o755)
    return p


# ---- 5. aggregate admission -------------------------------------------------------------------

def test_each_child_takes_a_fresh_aggregate_admission_and_the_second_is_refused(tmp_path, monkeypatch):
    """Fresh aggregate admission per child: the second child is judged against the FIRST child's
    live reservation, not against a stale reading taken before either started."""
    adm = CS.admission_module()
    store = adm.Store(tmp_path / "store")
    res = adm.FileResources(readings(tmp_path / "res.json", slice_memory_max=1 * GIB), adm.DEFAULT_SLICE)
    now = time.time()
    first = adm.acquire(store, res, adm.Request(name="cell_a", cap_bytes=768 * MIB, wall_seconds=60), now)
    assert first["verdict"] == adm.ADMITTED
    adm.arm(store, res, first["lease_id"], now, pid=os.getpid())
    second = adm.acquire(store, res, adm.Request(name="cell_b", cap_bytes=768 * MIB, wall_seconds=60), now)
    assert second["verdict"] == adm.REFUSED
    assert "aggregate" in second["reason"].lower()
    # and the refusal may NOT be evaded by asking again for less
    third = adm.acquire(store, res, adm.Request(name="cell_b", cap_bytes=128 * MIB, wall_seconds=60), now)
    assert third["verdict"] == adm.REFUSED, "a lowered cap after a refusal must stay terminal"


def test_the_supervisor_never_lowers_a_declared_cap_to_pass_admission(tmp_path):
    sup = tmp_path / "SUPERVISOR"
    seen = tmp_path / "caps.txt"
    p = tmp_path / "recording-launcher"
    p.write_text("#!/usr/bin/env bash\n"
                 "while getopts \"m:t:n:W:L:E:P:S:qh\" o; do case $o in m) M=$OPTARG;; esac; done\n"
                 f"echo \"$M\" >> {seen}\n"
                 "exit 75\n")
    p.chmod(0o755)
    r = CS.supervise(cell_id="cell_a", argv=[sys.executable, "-c", "pass"], cap_bytes=1536 * MIB,
                     wall_seconds=60, supervisor_dir=sup, log_path=tmp_path / "a.log", launcher=p)
    assert r["termination"]["status"] == "REFUSED_BY_ADMISSION"
    caps = [c for c in seen.read_text().split() if c]
    assert caps == ["1610612736"], "the cap was asked for once, at the declared size, and not retried smaller"


# ---- 6. reservation release --------------------------------------------------------------------

def test_the_reservation_is_released_after_the_tree_finishes(tmp_path):
    adm = CS.admission_module()
    store = adm.Store(tmp_path / "store")
    res = adm.FileResources(readings(tmp_path / "res.json", slice_memory_max=1 * GIB), adm.DEFAULT_SLICE)
    now = time.time()
    d = adm.acquire(store, res, adm.Request(name="cell_a", cap_bytes=768 * MIB, wall_seconds=60), now)
    adm.arm(store, res, d["lease_id"], now, pid=os.getpid())
    assert adm.state(store, res, now)["live"], "a live reservation is visible while the tree runs"
    out = adm.release(store, res, d["lease_id"], now, observed_peak_bytes=101 * MIB)
    assert out["ok"], out
    assert not adm.state(store, res, now)["live"]
    again = adm.acquire(store, res, adm.Request(name="cell_b", cap_bytes=768 * MIB, wall_seconds=60), now)
    assert again["verdict"] == adm.ADMITTED, "the released capacity is available to the next cell"


@pytest.mark.skipif(not CS.launcher_available(), reason="the deployed launcher or systemd --user is not available here")
def test_the_real_launcher_releases_its_reservation_when_the_child_ends(tmp_path, monkeypatch):
    store = tmp_path / "admission"
    monkeypatch.setenv("CRISPDM_ADMISSION_DIR", str(store))
    r = CS.supervise(cell_id="rel_a", argv=[sys.executable, "-c", "pass"], cap_bytes=256 * MIB,
                     wall_seconds=60, supervisor_dir=tmp_path / "SUP", log_path=tmp_path / "rel.log")
    assert r["termination"]["status"] == "COMPLETED", r
    adm = CS.admission_module()
    st = adm.state(adm.Store(store), adm.resources_from_env(), time.time())
    assert not st["live"], f"a finished tree left a reservation held: {st['live']}"


# ---- the deployed runner's identity, not only the tracked bytes --------------------------------

def test_the_deployed_runner_identity_is_verified_against_a_tracked_commit():
    v = CS.verify_runner_identity()
    assert v["schema"] == "df_runner_identity.v1"
    for part in ("launcher", "admission_module"):
        assert v[part]["deployed_sha256"], v[part]
        assert v[part]["deployed_path_exists"] is True
        assert v[part]["matches_tracked_blob"] is not None
    assert v["verdict"] in ("DEPLOYED_MATCHES_TRACKED", "DEPLOYED_DIFFERS_FROM_TRACKED")


def test_an_absent_launcher_is_a_refusal_and_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CRISPDM_RUN", str(tmp_path / "does-not-exist"))
    assert CS.launcher_available() is False
    with pytest.raises(CS.ScopeRefusal) as e:
        CS.supervise(cell_id="x", argv=[sys.executable, "-c", "pass"], cap_bytes=MIB,
                     wall_seconds=10, supervisor_dir=tmp_path / "S", log_path=tmp_path / "x.log")
    assert e.value.code == "LAUNCHER_NOT_INSTALLED"


# ---- the record carries every field the order names ---------------------------------------------

def test_the_record_carries_every_field_the_order_names(cg, claims):
    rel = f"{SLICE_REL}/crispdm-cell_a-1-11.scope"
    make_cgroup(cg, rel, peak=900 * MIB, limit=2 * GIB)
    claim = CS.require_fresh_exclusive_scope(claims, "cell_a", identity=CS.scope_identity(rel),
                                            parent_cgroup="user.slice/driver.scope")
    rec = CS.cell_scope_record("cell_a", stage="TRAIN_UPDATES", claim=claim, updates=600,
                               cpu_seconds=12.5, wall_seconds=30.25)
    assert rec["host_ram"]["cgroup_peak"]["bytes"] == 900 * MIB
    assert rec["gpu"]["status"] in (CS.MEASURED, CS.UNKNOWN)
    assert set(rec["gpu"]) >= {"allocated_bytes", "reserved_bytes", "status", "source"}
    assert rec["scope"]["cgroup"] == rel and rec["scope"]["inode"] > 0
    assert rec["kernel_limit"]["bytes"] == 2 * GIB and rec["kernel_limit"]["source"] == "memory.max"
    assert rec["optimizer_updates"] == 600
    assert rec["cpu_seconds"] == 12.5 and rec["wall_seconds"] == 30.25
    assert rec["stage"] == "TRAIN_UPDATES"
    # no machine name anywhere in a published record
    blob = json.dumps(rec)
    assert os.uname().nodename not in blob
    assert rec["host_identity"]["host_name_recorded"] is False
