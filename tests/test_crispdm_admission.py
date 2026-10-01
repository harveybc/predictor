"""DR01 (order 2026-09-26): the atomic per-host admission, proved on a SIMULATED host.

Every test here injects its readings through a JSON file (CRISPDM_ADMISSION_RESOURCES_JSON) and
its clock through CRISPDM_ADMISSION_NOW.  Nothing allocates memory, nothing is signalled, no
kernel or systemd setting is read for a decision and no real RAM is ever pressured: the order
forbids validating this by putting the host under pressure.

The headline is the first test: two requests that each fit alone but not together.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
MODULE = TOOLS / "crispdm_admission.py"
LAUNCHER = TOOLS / "crispdm-run"
INSTALLER = TOOLS / "install_crispdm_launcher.sh"
GIB = 1 << 30

sys.path.insert(0, str(TOOLS))
import crispdm_admission as A  # noqa: E402


# ---- the simulated host ----------------------------------------------------------------------

DEFAULT_READINGS = {
    "mem_available_bytes": 12 * GIB,
    "mem_total_bytes": 32 * GIB,
    "slice_memory_max": 14 * GIB,
    "slice_memory_current": 0,
    "pressure_some_avg10": 0.0,
    "alive": {},
    "cgroup_current": {},
    "cgroup_peak": {},
}


class Host:
    """A readings file plus a lease store, with a clock the test advances by hand."""

    def __init__(self, tmp_path, **readings):
        self.dir = Path(tmp_path)
        self.readings_path = self.dir / "readings.json"
        self.store_dir = self.dir / "admission"
        self.now = 1790400000.0                      # a fixed simulated epoch
        self.set(**readings)

    def set(self, **readings):
        d = dict(DEFAULT_READINGS)
        d.update(readings)
        self.readings_path.write_text(json.dumps(d))
        self._d = d

    def patch(self, **readings):
        d = dict(self._d)
        d.update(readings)
        self.set(**d)

    def tick(self, seconds):
        self.now += float(seconds)

    @property
    def env(self):
        return {
            **os.environ,
            "CRISPDM_ADMISSION_RESOURCES_JSON": str(self.readings_path),
            "CRISPDM_ADMISSION_DIR": str(self.store_dir),
            "CRISPDM_ADMISSION_NOW": repr(self.now),
            "CRISPDM_ADMISSION_MODULE": str(MODULE),
            "CRISPDM_CGROUP_ROOT": str(self.dir / "cgroup"),
        }

    # in-process handles, for the unit-level tests
    @property
    def store(self):
        return A.Store(self.store_dir)

    @property
    def res(self):
        return A.FileResources(self.readings_path)

    def acquire(self, name, cap_bytes, **kw):
        req = A.Request(name=name, cap_bytes=cap_bytes, label=kw.pop("label", name), **kw)
        return A.acquire(self.store, self.res, req, self.now)

    # CLI handle, for the launcher tests
    def cli(self, *argv, expect=None):
        r = subprocess.run([sys.executable, str(MODULE), *argv], env=self.env,
                           capture_output=True, text=True, timeout=120)
        if expect is not None:
            assert r.returncode == expect, r.stdout + r.stderr
        return r

    def ledger(self):
        p = self.store_dir / "ledger.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def live_lease_ids(self):
        d = self.store_dir / "leases"
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


# ---- 1. the defect itself --------------------------------------------------------------------

def test_2026_09_26_two_requests_that_each_fit_alone_are_not_both_admitted(host):
    """Musashi F1, reproduced and closed.  12 GiB available; 8 GiB fits alone, twice it does not.

    The old launcher read MemAvailable and then launched, so both were accepted.  Here the first
    admission WRITES a reservation and the second is measured against it.
    """
    first = host.acquire("shard1", 8 * GIB)
    assert first["verdict"] == A.ADMITTED

    second = host.acquire("shard2", 8 * GIB)
    assert second["verdict"] == A.QUEUED, "a second 8 GiB must not be admitted against a held 8 GiB"
    assert second["code"] == "HOST_HEADROOM"
    assert second["readings"]["held_unrealised_bytes"] == 8 * GIB
    assert second["readings"]["live_leases"] == 1
    assert host.live_lease_ids() == [first["lease_id"]], "the queued request reserved nothing"

    # and the same is true across processes, which is where the real defect lived
    host.cli("acquire", "-n", "shard3", "-m", "8G", expect=75)
    assert len(host.live_lease_ids()) == 1


def test_2026_09_26_the_pair_is_admitted_one_after_the_other_not_at_once(host):
    """The fix queues; it does not forbid.  Once the first load is gone the second is admitted."""
    first = host.acquire("shard1", 8 * GIB)
    A.arm(host.store, host.res, first["lease_id"], host.now, pid=424242, cgroup="cg/shard1")
    host.patch(alive={"424242": True, "cg/shard1": True})
    assert host.acquire("shard2", 8 * GIB)["verdict"] == A.QUEUED

    host.patch(alive={"424242": False, "cg/shard1": False}, cgroup_peak={"cg/shard1": 7 * GIB})
    host.tick(60)
    assert A.release(host.store, host.res, first["lease_id"], host.now)["ok"] is True
    assert host.acquire("shard2", 8 * GIB)["verdict"] == A.ADMITTED


# ---- 2. an orphaned child --------------------------------------------------------------------

def test_2026_09_26_an_orphan_lease_whose_witness_is_dead_is_reclaimed_not_leaked(host):
    """A load that died without releasing (the launcher was killed, the scope vanished) must not
    hold the host's memory hostage.  The witness, not the bookkeeping, decides."""
    lease = host.acquire("orphan", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease, host.now, pid=999001, cgroup="cg/orphan")
    host.patch(alive={"999001": False, "cg/orphan": False})       # the tree is gone
    host.tick(30)                                                  # still well inside the TTL

    freed = A.reclaim(host.store, host.res, host.now)
    assert freed["freed"] == [lease], "a dead witness frees the reservation without waiting for expiry"
    assert host.live_lease_ids() == []
    assert host.acquire("next", 8 * GIB)["verdict"] == A.ADMITTED
    assert any(r["event"] == "LEASE_RECLAIMED_WITNESS_DEAD" for r in host.ledger())


def test_2026_09_26_a_recycled_pid_does_not_keep_an_orphan_lease_alive(host):
    """Crash recovery must not be fooled by pid reuse: the lease pins the /proc start time."""
    lease_id = host.acquire("orphan", 4 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=os.getpid())
    lease = A.Lease(**json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text()))
    assert lease.pid_starttime is not None
    lease.pid_starttime = int(lease.pid_starttime) + 1           # a DIFFERENT process now holds this pid
    host.store.write(lease)
    host.tick(300)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]


# ---- 3. a lease expired while its process is still alive -------------------------------------

def test_2026_09_26_an_expired_lease_over_a_live_child_is_extended_never_freed(host):
    """Recovering leases after a crash must not release the memory of a child that is still
    alive.  Expiry alone is not evidence of death."""
    lease_id = host.acquire("long", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=777001, cgroup="cg/long")
    host.patch(alive={"777001": True, "cg/long": True}, cgroup_current={"cg/long": 5 * GIB})

    host.tick(A.LEASE_TTL_SECONDS + 600)                          # the heartbeat stopped: the holder crashed
    swept = A.reclaim(host.store, host.res, host.now)
    assert swept["freed"] == []
    assert swept["extended"] == [lease_id]
    assert host.live_lease_ids() == [lease_id]

    # and the memory is still accounted for, less what the child has already taken
    queued = host.acquire("newcomer", 8 * GIB)
    assert queued["verdict"] == A.QUEUED
    assert queued["readings"]["held_unrealised_bytes"] == 3 * GIB   # 8 reserved - 5 observed
    assert any(r["event"] == "LEASE_EXTENDED_CHILD_ALIVE" for r in host.ledger())


def test_2026_09_26_a_release_asked_for_under_a_live_child_keeps_the_reservation(host):
    lease_id = host.acquire("long", 6 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=777002, cgroup="cg/long")
    host.patch(alive={"777002": True, "cg/long": True})
    out = A.release(host.store, host.res, lease_id, host.now)
    assert out["ok"] is False and out["code"] == "CHILD_STILL_ALIVE"
    assert host.live_lease_ids() == [lease_id]
    assert any(r["event"] == "RELEASE_REFUSED_CHILD_ALIVE" for r in host.ledger())


# ---- 4. a parent-imposed limit ---------------------------------------------------------------

def test_2026_09_26_the_parent_slice_ceiling_bounds_the_observed_aggregate_not_just_the_child(host):
    """The old rule compared ONE request with the slice ceiling.  A child's own limit is not a
    budget: the gate is the observed aggregate, slice usage plus unrealised reservations."""
    host.patch(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB,
               slice_memory_max=14 * GIB, slice_memory_current=2 * GIB)
    first = host.acquire("a", 8 * GIB)
    assert first["verdict"] == A.ADMITTED                  # 2 in use + 8 = 10 <= 14

    second = host.acquire("b", 6 * GIB)                    # each fits the ceiling on its own
    assert second["verdict"] == A.QUEUED
    assert second["code"] == "SLICE_AGGREGATE_BUDGET"
    assert second["readings"]["aggregate_committed_bytes"] == 10 * GIB


def test_2026_09_26_a_request_above_a_parent_ceiling_is_refused_terminally_not_queued(host):
    """A request that cannot fit any ceiling must never be polled: waiting cannot help it, and
    the order forbids insisting after a rejection."""
    host.patch(mem_available_bytes=30 * GIB, slice_memory_max=14 * GIB)
    d = host.acquire("huge", 20 * GIB)
    assert d["verdict"] == A.REFUSED and d["code"] == "ABOVE_SLICE_CEILING"

    host.patch(slice_memory_max=None, mem_total_bytes=32 * GIB)
    d = host.acquire("huge", 31 * GIB)
    assert d["verdict"] == A.REFUSED and d["code"] == "ABOVE_HOST_CEILING"

    # the CLI turns a terminal refusal into exit 75 with no waiting, even when asked to queue
    r = host.cli("acquire", "-n", "huge", "-m", "31G", "--queue", "--max-wait-seconds", "600",
                 expect=75)
    assert json.loads(r.stdout.strip().splitlines()[-1])["verdict"] == A.REFUSED
    assert host.live_lease_ids() == []


# ---- 5. high pressure ------------------------------------------------------------------------

def test_2026_09_26_high_memory_pressure_queues_instead_of_racing_oomd(host):
    """Two of the four terminations on 2026-09-26 were systemd-oomd acting on user-slice memory
    pressure above 50% for more than 20s, with the child's own cgroup limit never reached.  A
    gate that reads MemAvailable alone cannot see that.  Admission reads PSI and stops first."""
    host.patch(pressure_some_avg10=55.11)                  # the reading from the huntdgst kill
    d = host.acquire("under_pressure", 4 * GIB)
    assert d["verdict"] == A.QUEUED and d["code"] == "MEMORY_PRESSURE"
    assert d["readings"]["pressure_admit_max"] < 50.0, "the limit must sit below where oomd acts"
    assert host.live_lease_ids() == []

    host.patch(pressure_some_avg10=1.0)
    assert host.acquire("under_pressure", 4 * GIB)["verdict"] == A.ADMITTED


# ---- 6/7. an out-of-memory kill, and memory released -----------------------------------------

def test_2026_09_26_an_out_of_memory_kill_is_terminal_and_frees_the_reservation_once(host):
    """An OOM never becomes a retry with a bigger cap.  The reservation is released, the observed
    TREE peak is recorded, and the module offers no path that re-asks for more."""
    lease_id = host.acquire("victim", 9 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=555001, cgroup="cg/victim",
          unit="crispdm-victim.scope")
    host.patch(alive={"555001": False, "cg/victim": False},
               cgroup_peak={"cg/victim": 7 * GIB})         # killed below its own 9 GiB cap
    host.tick(120)

    out = A.release(host.store, host.res, lease_id, host.now)
    assert out["ok"] and out["observed_tree_peak_bytes"] == 7 * GIB
    released = [r for r in host.ledger() if r["event"] == "LEASE_RELEASED"]
    assert len(released) == 1
    assert released[0]["peak_scope"] == "cgroup", "the peak is the cgroup's, never one process's RSS"
    assert "never retries with a bigger cap" in released[0]["note"]

    admitted = [r for r in host.ledger() if r["event"] == "ADMISSION_ADMITTED"]
    assert len(admitted) == 1, "one admission, one outcome: nothing re-asked automatically"


def test_2026_09_26_memory_released_by_a_finished_load_is_available_to_the_next(host):
    a = host.acquire("a", 5 * GIB)["lease_id"]
    b = host.acquire("b", 4 * GIB)["lease_id"]
    assert host.acquire("c", 4 * GIB)["verdict"] == A.QUEUED
    for lease in (a, b):
        A.arm(host.store, host.res, lease, host.now, pid=1, cgroup=f"cg/{lease}")
    host.patch(alive={"1": False})
    host.tick(10)
    assert host.acquire("c", 4 * GIB)["verdict"] == A.ADMITTED
    assert len(host.live_lease_ids()) == 1


# ---- 8. a contradictory cap -------------------------------------------------------------------

def test_2026_09_26_a_contradictory_cap_is_refused_rather_than_silently_resolved(host):
    """The byte value has ONE source.  A second, discordant integer is a refusal, not a choice."""
    r = host.cli("acquire", "-n", "x", "-m", "9G", "--cap-bytes", str(8 * GIB), expect=75)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["verdict"] == A.REFUSED and d["code"] == "CONTRADICTORY_CAP"
    assert host.live_lease_ids() == []

    # the agreeing pair is accepted, and the lease carries exactly that one integer
    r = host.cli("acquire", "-n", "x", "-m", "9G", "--cap-bytes", str(9 * GIB), expect=0)
    lease = json.loads(r.stdout.strip().splitlines()[-1])["lease"]
    assert lease["cap_bytes"] == 9 * GIB == A.parse_size("9G")


def test_2026_09_26_the_launcher_and_the_module_derive_the_same_single_integer(host):
    for text, want in (("4G", 4 * GIB), ("9G", 9 * GIB), ("512M", 512 << 20), ("2048", 2048)):
        r = subprocess.run([sys.executable, str(MODULE), "size", text], env=host.env,
                           capture_output=True, text=True, check=True)
        assert int(r.stdout.strip()) == want == A.parse_size(text)


# ---- the pilot footprint must be bound to its own evidence, and be a TREE peak ----------------

def test_2026_09_26_a_pilot_peak_is_only_accepted_bound_to_its_own_retained_record(host, tmp_path):
    ev = tmp_path / "PILOT.json"
    ev.write_text(json.dumps({"peak_scope": "cgroup", "peak_bytes": 8458399744,
                              "unit": "pilot_long_window_own_depth"}))

    # a typed peak with no record is refused
    r = host.cli("acquire", "-n", "p", "-m", "9G", "--peak-bytes", "8458399744", expect=75)
    assert json.loads(r.stdout.strip().splitlines()[-1])["code"] == "PEAK_NOT_BOUND_TO_EVIDENCE"

    # a peak that disagrees with the record is refused
    r = host.cli("acquire", "-n", "p", "-m", "9G", "--peak-evidence", str(ev),
                 "--peak-bytes", "1000", expect=75)
    assert json.loads(r.stdout.strip().splitlines()[-1])["code"] == "PEAK_NOT_BOUND_TO_EVIDENCE"

    # bound: admitted, and the lease records the record's digest
    r = host.cli("acquire", "-n", "p", "-m", "9G", "--peak-evidence", str(ev),
                 "--peak-bytes", "8458399744", expect=0)
    lease = json.loads(r.stdout.strip().splitlines()[-1])["lease"]
    assert lease["peak_bytes"] == 8458399744 and lease["peak_scope"] == "cgroup"
    assert len(lease["peak_evidence_sha256"]) == 64


def test_2026_09_26_a_main_process_rss_peak_may_not_size_an_admission(host, tmp_path):
    """Sizing on one process's RSS under-counts the tree: the q2deep cell peaked at 7.4 GiB in
    its cgroup while its main process's anon RSS was a fraction of that."""
    ev = tmp_path / "RSS.json"
    ev.write_text(json.dumps({"peak_scope": "main_process_rss", "peak_bytes": 2 * GIB}))
    r = host.cli("acquire", "-n", "p", "-m", "4G", "--peak-evidence", str(ev), expect=75)
    assert json.loads(r.stdout.strip().splitlines()[-1])["code"] == "PEAK_NOT_A_TREE_PEAK"


def test_2026_09_26_a_pilot_peak_that_does_not_fit_queues_even_when_the_cap_would(host, tmp_path):
    ev = tmp_path / "PILOT.json"
    ev.write_text(json.dumps({"peak_scope": "tree", "peak_bytes": 8 * GIB}))
    host.patch(mem_available_bytes=11 * GIB)          # 11 - 3 reserve = 8 free; 8 + 1 margin does not fit
    r = host.cli("acquire", "-n", "p", "-m", "8G", "--peak-evidence", str(ev), expect=75)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["verdict"] == A.QUEUED and d["code"] == "PILOT_PEAK_DOES_NOT_FIT"


# ---- 9. the launcher, end to end on the simulated host ----------------------------------------

@pytest.fixture
def fake_systemd(tmp_path):
    """A recording stand-in for systemd-run and systemctl.  No scope, slice or unit is created."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "systemd-run.argv"
    (bin_dir / "systemd-run").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\0' \"$@\" > {record}\n"
        "# drop everything up to and including `timeout --kill-after=30s <wall>`\n"
        "while [ $# -gt 0 ] && [ \"$1\" != timeout ]; do shift; done\n"
        "shift 3\n"
        'exec "$@"\n')
    (bin_dir / "systemctl").write_text(
        "#!/usr/bin/env bash\n"
        "echo /fake.slice/crispdm-batch.slice\n")
    for f in ("systemd-run", "systemctl"):
        (bin_dir / f).chmod(0o755)
    return {"bin": bin_dir, "record": record}


def _run_launcher(host, fake_systemd, *argv, extra_env=None, timeout=120):
    env = {**host.env, "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"}
    env.update(extra_env or {})
    return subprocess.run(["bash", str(LAUNCHER), *argv], env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_2026_09_26_an_interior_double_dash_reaches_the_command_verbatim(host, fake_systemd):
    """`--` separates the launcher's flags from the command exactly once.  Every later `--` is
    part of the command and must arrive unchanged: a pytest invocation, a `python -m x -- -v`,
    or an ssh command line stops working the moment the launcher eats one."""
    out = tmp = host.dir / "argv.out"
    printer = host.dir / "print_argv.sh"
    printer.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > ' + str(out) + "\n")
    printer.chmod(0o755)

    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "dashes", "--",
                      str(printer), "alpha", "--", "-v", "--beta=1", "--", "zulu")
    assert r.returncode == 0, r.stdout + r.stderr
    assert tmp.read_text().splitlines() == ["alpha", "--", "-v", "--beta=1", "--", "zulu"]

    # and the same arguments reached systemd-run in that order
    passed = fake_systemd["record"].read_bytes().split(b"\0")
    passed = [p.decode() for p in passed if p]
    tail = passed[passed.index(str(printer)):]
    assert tail == [str(printer), "alpha", "--", "-v", "--beta=1", "--", "zulu"]


def test_2026_09_26_the_launcher_derives_the_cgroup_limit_from_the_same_one_integer(host, fake_systemd):
    r = _run_launcher(host, fake_systemd, "-m", "2G", "-n", "one", "--", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    passed = [p.decode() for p in fake_systemd["record"].read_bytes().split(b"\0") if p]
    assert f"MemoryMax={2 * GIB}" in passed
    assert f"MemoryHigh={2 * GIB * 9 // 10}" in passed
    assert "MemorySwapMax=0" in passed


def test_2026_09_26_the_launcher_holds_the_reservation_for_the_whole_load_not_the_read(host, fake_systemd):
    """Serialising the instant of the reading is not enough.  While the child runs, a second
    request for memory the first is holding must be queued -- from a different process."""
    gate = host.dir / "gate"
    waiter = host.dir / "wait.sh"
    waiter.write_text('#!/usr/bin/env bash\nfor i in $(seq 1 600); do [ -f "$1" ] && exit 0; sleep 0.1; done\nexit 1\n')
    waiter.chmod(0o755)

    proc = subprocess.Popen(["bash", str(LAUNCHER), "-m", "8G", "-n", "held", "--",
                             str(waiter), str(gate)],
                            env={**host.env, "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not host.live_lease_ids():
            time.sleep(0.1)
        assert host.live_lease_ids(), "the launcher must reserve before it launches"

        second = host.cli("acquire", "-n", "second", "-m", "8G")
        assert second.returncode == 75
        d = json.loads(second.stdout.strip().splitlines()[-1])
        assert d["verdict"] == A.QUEUED and d["code"] == "HOST_HEADROOM"
        assert d["readings"]["held_unrealised_bytes"] == 8 * GIB
    finally:
        gate.write_text("go")
        proc.wait(timeout=60)
    assert proc.returncode == 0, proc.stdout
    # the whole tree has finished: the reservation is gone, and only now
    assert host.live_lease_ids() == []
    assert any(r["event"] == "LEASE_RELEASED" for r in host.ledger())


def test_2026_09_26_the_launcher_propagates_an_out_of_memory_exit_and_does_not_relaunch(host, fake_systemd):
    killed = host.dir / "killed.sh"
    killed.write_text("#!/usr/bin/env bash\nexit 137\n")           # what a SIGKILL looks like downstream
    killed.chmod(0o755)
    r = _run_launcher(host, fake_systemd, "-m", "2G", "-n", "oom", "--", str(killed))
    assert r.returncode == 137, r.stdout + r.stderr
    assert host.live_lease_ids() == []
    started = [x for x in host.ledger() if x["event"] == "ADMISSION_ADMITTED"]
    assert len(started) == 1, "one admission; the launcher has no retry and no cap escalation"


def test_2026_09_26_a_refused_launch_starts_nothing_and_changes_nothing(host, fake_systemd):
    host.acquire("first", 8 * GIB)                      # 8 of 9 GiB of free headroom is held
    before = sorted(host.live_lease_ids())
    r = _run_launcher(host, fake_systemd, "-m", "8G", "-n", "refused", "--", "true")
    assert r.returncode == 75
    assert "REFUSED" in r.stderr
    assert not fake_systemd["record"].exists(), "systemd-run must not be reached at all"
    assert sorted(host.live_lease_ids()) == before


def test_2026_09_26_the_launcher_refuses_without_the_admission_module(host, fake_systemd, tmp_path):
    r = _run_launcher(host, fake_systemd, "-m", "1G", "--", "true",
                      extra_env={"CRISPDM_ADMISSION_MODULE": str(tmp_path / "absent.py")})
    # the in-tree sibling is found, so point the launcher at a copy that has no sibling either
    solo = tmp_path / "solo"
    solo.mkdir()
    (solo / "crispdm-run").write_text(LAUNCHER.read_text())
    (solo / "crispdm-run").chmod(0o755)
    r = subprocess.run(["bash", str(solo / "crispdm-run"), "-m", "1G", "--", "true"],
                       env={**host.env, "CRISPDM_ADMISSION_MODULE": str(tmp_path / "absent.py"),
                            "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 75 and "admission module is not installed" in r.stderr


def test_2026_09_26_the_travel_hold_still_refuses_before_any_admission(host, fake_systemd, tmp_path):
    fake_home = tmp_path / "home"
    (fake_home / ".config" / "crispdm").mkdir(parents=True)
    (fake_home / ".config" / "crispdm" / "coordinator-travel-hold").write_text("")
    r = _run_launcher(host, fake_systemd, "-m", "1G", "--", "true",
                      extra_env={"HOME": str(fake_home)})
    assert r.returncode == 75 and "travel thermal hold" in r.stderr
    assert host.live_lease_ids() == []


# ---- deployment: future launches only ---------------------------------------------------------

def test_2026_09_26_the_installer_replaces_by_rename_and_reports_both_digests(tmp_path):
    """The deployed copy is refreshed by an atomic rename, so a crispdm-run that is already
    running keeps its own text and no running child's limits are touched."""
    bin_dir, lib_dir = tmp_path / "bin", tmp_path / "lib"
    env = {**os.environ, "CRISPDM_INSTALL_BIN": str(bin_dir), "CRISPDM_INSTALL_LIBEXEC": str(lib_dir)}

    check = subprocess.run(["bash", str(INSTALLER), "--check"], env=env, capture_output=True, text=True)
    assert check.returncode == 1 and "ABSENT" in check.stdout

    done = subprocess.run(["bash", str(INSTALLER)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (bin_dir / "crispdm-run").read_text() == LAUNCHER.read_text()
    assert (lib_dir / "crispdm_admission.py").read_text() == MODULE.read_text()
    assert os.access(bin_dir / "crispdm-run", os.X_OK)

    again = subprocess.run(["bash", str(INSTALLER), "--check"], env=env, capture_output=True, text=True)
    assert again.returncode == 0

    idem = subprocess.run(["bash", str(INSTALLER)], env=env, capture_output=True, text=True)
    assert "unchanged" in idem.stdout


# ---- what the module must never do ------------------------------------------------------------

def _code_only(path: Path) -> str:
    """The file with its comments and docstrings removed, so a prohibition named in prose is not
    mistaken for a call that makes it."""
    import re
    text = re.sub(r'""".*?"""', "", path.read_text(), flags=re.S)
    out = []
    for line in text.splitlines():
        cut = line.find("#")
        out.append(line if cut < 0 else line[:cut])
    return "\n".join(out)


def test_2026_09_26_nothing_in_the_admission_path_kills_relaxes_or_disables_anything():
    """The order forbids killing live processes, disabling oomd, raising swap or ceilings and
    dropping caches.  No executable line may do any of it."""
    code = "\n".join(_code_only(p) for p in (MODULE, LAUNCHER, INSTALLER))
    for forbidden in ("drop_caches", "swapoff", "swapon", "oomd.conf", "systemd-oomd.service",
                      "set-property", "sysctl", "SIGKILL", "kill -9", "pkill", "killall",
                      "MemoryMax=infinity", "MemorySwapMax=infinity", "systemctl --user stop",
                      "systemctl --user kill", "systemctl --user disable", "systemctl stop"):
        assert forbidden not in code, f"{forbidden} must not be executed by the admission path"

    # the only signals the launcher sends are the operator's own TERM/INT forwarded to its own
    # child, plus the liveness probe (signal 0) and the end of its own sampler subshell.
    import re
    assert sorted(set(re.findall(r"kill -(\S+)", _code_only(LAUNCHER)))) == ["0", "INT", "TERM"]
    # systemctl is only ever read from
    assert re.findall(r"systemctl[^\n]*", _code_only(LAUNCHER)) == [
        'systemctl --user show "$SLICE" -p ControlGroup --value 2>/dev/null || true)']


def test_2026_09_26_inside_scope_reports_whether_a_live_reservation_covers_this_process(host):
    """A runner that spawns its own fit children shares its parent's cgroup and therefore its
    parent's MemoryMax: it must NOT take a second reservation (that would double-count the same
    bytes), but it must be able to prove its parent has one.  That is what this reports."""
    r = host.cli("inside-scope")
    assert r.returncode == 1                        # this test process is in no reserved scope
    assert json.loads(r.stdout)["covered"] is False

    mine = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::", 1)[1]
    lease_id = host.acquire("covering", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=os.getpid(), cgroup=mine.lstrip("/"))
    r = host.cli("inside-scope")
    assert r.returncode == 0
    assert json.loads(r.stdout)["covering_lease_ids"] == [lease_id]


def test_2026_09_26_a_lease_id_is_one_filesystem_segment_whatever_the_caller_calls_the_job(host):
    """A caller's job name carries slashes and dots (`df-utility-fab/v0/case/transformed`).  The
    lease id must still be one path segment, or the reservation cannot be written at all -- and a
    reservation that cannot be written is an admission that holds nothing."""
    d = host.acquire("df-utility-fab/v0/mad_extremes_trailing/transformed", 2 * GIB)
    assert d["verdict"] == A.ADMITTED
    assert "/" not in d["lease_id"] and d["lease_id"].endswith(tuple("0123456789abcdef"))
    assert (host.store_dir / "leases" / f"{d['lease_id']}.json").exists()
    assert d["lease"]["name"] == "df-utility-fab/v0/mad_extremes_trailing/transformed"


def test_2026_09_26_an_inherited_ancestor_cgroup_may_not_witness_a_reservation(host):
    """A DETACHED unit is witnessed by its cgroup.  It must be that unit's OWN cgroup: a runner
    whose child shares an enclosing scope would otherwise record the ancestor, which outlives the
    task, so the reservation could never be released."""
    outer = "user.slice/crispdm-batch.slice/crispdm-outer.scope"
    host.tick(A.ARM_GRACE_SECONDS + 10)          # past the grace window, so only a witness speaks

    lease_id = host.acquire("inner", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, cgroup=outer, unit="crispdm-inner.scope")
    host.patch(alive={outer: True})
    host.tick(A.ARM_GRACE_SECONDS + 10)
    assert A.release(host.store, host.res, lease_id, host.now)["ok"] is True, \
        "an ancestor's liveness must not hold this reservation open"

    # the same cgroup, correctly named as this unit's own, does witness it
    lease_id = host.acquire("inner2", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, cgroup=outer, unit="crispdm-outer.scope")
    host.tick(A.ARM_GRACE_SECONDS + 10)
    assert A.release(host.store, host.res, lease_id, host.now)["code"] == "CHILD_STILL_ALIVE"


def test_2026_09_26_a_supervised_child_that_has_been_reaped_frees_its_reservation_at_once(host):
    """When a holder waited for its own child and this load's own scope is EMPTY, the child's death
    is the whole answer and the reservation is freed at that instant.

    RR02 amends this test, and says so.  As written on 2026-09-26 it asserted that a reaped child
    frees the reservation *even while its own scope cgroup still holds a task*, because a scope
    tearing down still lists tasks winding down and the launcher was observed refusing to release
    its own reservation.  That symptom now has its own fix -- `cgroup_alive` checks each listed task
    and a zombie holds no memory -- and the old assertion also covered the case the RR02 order
    names: a detached descendant still running in this scope after the direct child exited.
    Releasing there hands a second admission the bytes a live descendant is using.  So the empty
    scope is asserted here and the live descendant in
    tests/test_crispdm_rr02_monitor.py; the reversal is reported in the RR02 return.
    """
    unit = "crispdm-x.scope"
    cg = f"user.slice/crispdm-batch.slice/{unit}"
    lease_id = host.acquire("supervised", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=888003, cgroup=cg, unit=unit)
    host.patch(alive={"888003": False, cg: False})         # reaped, and its scope holds no task
    out = A.release(host.store, host.res, lease_id, host.now)
    assert out["ok"] is True and out["code"] == "RELEASED"
    assert host.live_lease_ids() == []


def test_2026_09_26_a_teardown_zombie_in_the_cgroup_is_not_a_live_child(tmp_path):
    """`cgroup.procs` can still list a task that is already a zombie while a scope tears down, and
    a zombie holds no memory.  Reading that as 'alive' made the launcher refuse to release its own
    reservation the moment its load finished."""
    root = tmp_path / "cgroup"
    cg = root / "user.slice/crispdm-batch.slice/crispdm-x.scope"
    cg.mkdir(parents=True)
    res = A.SystemResources(cgroup_root=root)

    (cg / "cgroup.procs").write_text("")
    assert res.cgroup_alive("user.slice/crispdm-batch.slice/crispdm-x.scope") is False

    (cg / "cgroup.procs").write_text("999999999\n")        # a pid that does not exist any more
    assert res.cgroup_alive("user.slice/crispdm-batch.slice/crispdm-x.scope") is False

    (cg / "cgroup.procs").write_text(f"{os.getpid()}\n")   # a task that really is running
    assert res.cgroup_alive("user.slice/crispdm-batch.slice/crispdm-x.scope") is True


# ---- ADM-DEADCACHE-01 (2026-09-30): dead clean file cache is not live use ----------------------

# the preferred worker's slice at 18:40 local: no scope, no process, no lease, only the page cache a
# finished NPZ build left recharged to the slice
DEAD_CACHE_STAT = {"anon": 0, "file": 2_327_924_736, "file_dirty": 0, "file_writeback": 0,
                   "shmem": 16_564_224, "slab": 57_008_080, "unevictable": 0}


def test_2026_09_30_dead_clean_cache_left_by_a_finished_scope_does_not_queue_the_next_load(host):
    host.set(mem_available_bytes=12 * GIB, slice_memory_max=8 * GIB,
             slice_memory_current=2_385_801_216, slice_memory_stat=DEAD_CACHE_STAT)
    d = host.acquire("m04-pilot", 6_442_450_944)
    assert d["verdict"] == A.ADMITTED, d
    r = d["readings"]
    clean = 2_327_924_736 - 16_564_224
    assert r["slice_clean_file_bytes_not_charged"] == clean
    assert r["slice_charged_bytes"] == 2_385_801_216 - clean
    assert r["aggregate_committed_bytes"] == r["slice_charged_bytes"]


def test_2026_09_30_without_a_memory_stat_reading_everything_stays_charged_as_before(host):
    """The defect, reproduced: the same bytes read only as memory.current queue forever."""
    host.set(mem_available_bytes=12 * GIB, slice_memory_max=8 * GIB, slice_memory_current=2_385_801_216)
    d = host.acquire("m04-pilot", 6_442_450_944)
    assert d["verdict"] == A.QUEUED and d["code"] == "SLICE_AGGREGATE_BUDGET"


@pytest.mark.parametrize("kind", ["shmem", "file_dirty", "file_writeback", "unevictable"])
def test_2026_09_30_shmem_dirty_writeback_and_unevictable_pages_stay_charged(host, kind):
    stat = {"anon": 0, "file": 3 * GIB, "shmem": 0, "file_dirty": 0, "file_writeback": 0,
            "unevictable": 0, "slab": 0}
    stat[kind] = 3 * GIB                      # the whole file charge is NOT clean
    host.set(mem_available_bytes=12 * GIB, slice_memory_max=8 * GIB,
             slice_memory_current=3 * GIB, slice_memory_stat=stat)
    d = host.acquire("x", 6 * GIB)
    assert d["verdict"] == A.QUEUED and d["code"] == "SLICE_AGGREGATE_BUDGET"
    assert d["readings"]["slice_charged_bytes"] == 3 * GIB


def test_2026_09_30_anon_and_slab_are_charged_and_clean_cache_is_never_negative(host):
    assert A.clean_file_bytes({"file": 1, "shmem": 5}) == 0
    host.set(mem_available_bytes=12 * GIB, slice_memory_max=8 * GIB, slice_memory_current=5 * GIB,
             slice_memory_stat={"anon": 4 * GIB, "slab": GIB, "file": 0})
    d = host.acquire("x", 4 * GIB)
    assert d["verdict"] == A.QUEUED and d["readings"]["slice_charged_bytes"] == 5 * GIB


def _fake_scope(host, rel, stat):
    cg = host.dir / "cgroup" / rel
    cg.mkdir(parents=True)
    (cg / "memory.stat").write_text("".join(f"{k} {v}\n" for k, v in stat.items()))
    (cg / "memory.reclaim").write_text("")
    return cg


SCOPE_STAT = {"anon": 3 * GIB, "file": 2 * GIB, "shmem": 100, "file_dirty": 200,
              "file_writeback": 0, "unevictable": 0}


def test_2026_09_30_the_launcher_reclaims_its_own_scope_clean_cache_at_scope_end(host, fake_systemd):
    rel = "user.slice/u.service/crispdm.slice/crispdm-batch.slice/crispdm-job-1-2.scope"
    cg = _fake_scope(host, rel, SCOPE_STAT)
    host.patch(self_cgroup=rel)
    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "job", "--", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cg / "memory.reclaim").read_text() == str(2 * GIB - 100 - 200)
    ev = [x for x in host.ledger() if x["event"] == "SCOPE_CLEAN_CACHE_RECLAIM"]
    assert len(ev) == 1 and ev[0]["result"] == "RECLAIMED" and ev[0]["cgroup"] == rel
    assert ev[0]["lease_id"] and ev[0]["requested"] == 2 * GIB - 300
    assert host.live_lease_ids() == []                       # released after, as before


@pytest.mark.parametrize("rel", [
    "user.slice/u.service/crispdm.slice/crispdm-batch.slice",                 # the slice itself
    "user.slice/u.service/crispdm.slice/other.slice/crispdm-job-1-2.scope",   # another slice
    "user.slice/u.service/app.slice/crispdm-job-1-2.scope",                   # not under crispdm
    "user.slice/u.service/crispdm.slice/crispdm-batch.slice/session-3.scope",  # not a job scope
    "",                                                                        # the host root
])
def test_2026_09_30_scope_exec_never_reclaims_anything_but_its_own_job_scope(host, fake_systemd, rel):
    cg = _fake_scope(host, rel or "rootcg", SCOPE_STAT)
    host.patch(self_cgroup=rel)
    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "job", "--", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cg / "memory.reclaim").read_text() == ""
    ev = [x for x in host.ledger() if x["event"] == "SCOPE_CLEAN_CACHE_RECLAIM"]
    assert ev and ev[0]["result"] == "SKIPPED_NOT_OWN_CRISPDM_SCOPE"


def test_2026_09_30_scope_exec_keeps_the_command_exit_status_and_a_death_by_signal(host, fake_systemd):
    rel = "s/crispdm.slice/crispdm-batch.slice/crispdm-job-1-2.scope"
    _fake_scope(host, rel, SCOPE_STAT)
    host.patch(self_cgroup=rel)
    three = host.dir / "three.sh"
    three.write_text("#!/usr/bin/env bash\nexit 3\n")
    three.chmod(0o755)
    assert _run_launcher(host, fake_systemd, "-m", "1G", "-n", "a", "--", str(three)).returncode == 3
    termed = host.dir / "termed.sh"
    termed.write_text("#!/usr/bin/env bash\nkill -TERM $$\nsleep 5\n")
    termed.chmod(0o755)
    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "b", "--", str(termed))
    assert r.returncode == 128 + 15, r.stdout + r.stderr
    # the reclaim ran after both, and both reservations were released
    assert len([x for x in host.ledger() if x["event"] == "SCOPE_CLEAN_CACHE_RECLAIM"]) == 2
    assert host.live_lease_ids() == []


def test_2026_09_30_the_launcher_routes_the_command_through_scope_exec_inside_the_scope(host, fake_systemd):
    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "one", "--", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    passed = [p.decode() for p in fake_systemd["record"].read_bytes().split(b"\0") if p]
    i = passed.index("timeout")
    assert passed[i + 4:i + 6] == [str(MODULE), "scope-exec"] or passed[i + 4].endswith("crispdm_admission.py")
    assert passed[-2:] == ["--", "true"]


# ---- 2026-10-01 owner review of ADM-DEADCACHE-01: five required cases ---------------------------
# A subtracted clean-cache estimate is NOT permission to over-allocate RAM.  Each case below pins
# one way the subtraction could over-admit and shows it does not.

def _live(host, name, cap, cg, current, stat):
    lid = host.acquire(name, cap)["lease_id"]
    A.arm(host.store, host.res, lid, host.now, pid=880000 + len(name), cgroup=cg)
    d = dict(host._d)
    d.setdefault("alive", {}).update({str(880000 + len(name)): True, cg: True})
    d.setdefault("cgroup_current", {})[cg] = current
    d.setdefault("cgroup_stat", {})[cg] = stat
    host.set(**d)
    return lid


def test_2026_10_01_case_live_loads_their_own_clean_cache_is_never_uncharged(host):
    """A LIVE scope's clean cache is inside its observed bytes; uncharging it would also shrink its
    unrealised reservation and hand out its headroom twice."""
    host.set(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=14 * GIB,
             slice_memory_current=4 * GIB, slice_memory_stat={"anon": GIB, "file": 3 * GIB})
    _live(host, "fit", 8 * GIB, "cg/fit", 4 * GIB, {"anon": GIB, "file": 3 * GIB})   # all cache is live
    d = host.acquire("next", 6 * GIB)
    # committed = 4 in use + (8 - 4) unrealised = 8; + 6 = 14 <= 14: admitted only at the exact ceiling
    assert d["readings"]["slice_charged_bytes"] == 4 * GIB
    assert d["readings"]["slice_clean_file_bytes_in_live_scopes"] == 3 * GIB
    assert d["readings"]["aggregate_committed_bytes"] == 8 * GIB
    d2 = host.acquire("one-more", 1 * GIB)
    assert d2["verdict"] == A.QUEUED and d2["code"] == "SLICE_AGGREGATE_BUDGET"


def test_2026_10_01_case_shared_cache_only_the_part_outside_live_scopes_is_dead(host):
    """Slice cache = 1 GiB inside a live scope + 2 GiB left by a finished scope.  Only the 2 GiB
    stop counting; the live job's 1 GiB stays charged."""
    host.set(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=14 * GIB,
             slice_memory_current=5 * GIB, slice_memory_stat={"anon": 2 * GIB, "file": 3 * GIB})
    _live(host, "fit", 6 * GIB, "cg/fit", 3 * GIB, {"anon": 2 * GIB, "file": GIB})
    d = host.acquire("x", GIB)
    assert d["readings"]["slice_clean_file_bytes_not_charged"] == 2 * GIB
    assert d["readings"]["slice_charged_bytes"] == 3 * GIB


def test_2026_10_01_case_unreadable_live_scope_stat_uncharges_nothing(host):
    host.set(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=14 * GIB,
             slice_memory_current=5 * GIB, slice_memory_stat={"anon": 2 * GIB, "file": 3 * GIB})
    lid = host.acquire("fit", 6 * GIB)["lease_id"]
    A.arm(host.store, host.res, lid, host.now, pid=881111, cgroup="cg/nostat")
    host.patch(alive={"881111": True, "cg/nostat": True}, cgroup_current={"cg/nostat": 3 * GIB})
    d = host.acquire("x", GIB)
    assert d["readings"]["slice_charged_bytes"] == 5 * GIB


def test_2026_10_01_case_partial_reclaim_is_recorded_and_the_gate_reads_what_remains(host, monkeypatch):
    """memory.reclaim may return EAGAIN after reclaiming part of the request.  The result is
    PARTIAL, nothing is retried, and the gate keeps reading the real memory.stat afterwards."""
    import builtins
    import errno
    rel = "s/crispdm.slice/crispdm-batch.slice/crispdm-job-1-2.scope"
    cg = host.dir / "cgroup" / rel
    cg.mkdir(parents=True)
    (cg / "memory.stat").write_text("anon 1\nfile 1073741824\nshmem 0\n")
    (cg / "memory.reclaim").write_text("")
    host.patch(self_cgroup=rel)
    monkeypatch.setenv("CRISPDM_ADMISSION_RESOURCES_JSON", str(host.readings_path))
    monkeypatch.setenv("CRISPDM_CGROUP_ROOT", str(host.dir / "cgroup"))
    real_open = builtins.open

    def fake_open(path, mode="r", *a, **k):
        if str(path).endswith("memory.reclaim") and "w" in mode:
            raise OSError(errno.EAGAIN, "partial")
        return real_open(path, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", fake_open)
    out = A.reclaim_own_scope(host.store, "crispdm-batch.slice", "lease-x", host.now)
    assert out["result"] == "PARTIAL" and out["requested"] == GIB
    assert out["clean_file_after"] == GIB          # nothing assumed reclaimed: the stat is re-read
    # and the slice gate with the same, unreclaimed cache still charges what memory.stat says
    host.set(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=8 * GIB,
             slice_memory_current=2 * GIB, slice_memory_stat={"anon": GIB, "file": GIB})
    assert host.acquire("y", GIB)["readings"]["slice_charged_bytes"] == GIB


def test_2026_10_01_case_reservations_are_still_held_in_full_beside_dead_cache(host):
    """Dead cache never reduces another load's unrealised reservation."""
    host.set(mem_available_bytes=30 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=14 * GIB,
             slice_memory_current=3 * GIB, slice_memory_stat={"anon": 0, "file": 3 * GIB})
    first = host.acquire("held", 8 * GIB)                       # admitted, not yet armed or used
    assert first["verdict"] == A.ADMITTED
    d = host.acquire("next", 6 * GIB)
    assert d["readings"]["held_unrealised_bytes"] == 8 * GIB
    assert d["readings"]["aggregate_committed_bytes"] == 8 * GIB      # dead 3 GiB uncharged, 8 held
    assert d["verdict"] == A.ADMITTED                                   # 8 + 6 = 14 <= 14
    e = host.acquire("over", GIB)                                       # 8 + 6 + 1 > 14
    assert e["verdict"] == A.QUEUED and e["code"] == "SLICE_AGGREGATE_BUDGET"


def test_2026_10_01_case_memory_max_request_above_ceiling_refused_and_host_gate_unchanged(host):
    """The slice memory.max still refuses a request above it terminally, and the HOST gate
    (MemAvailable - desktop reserve - reservations) is untouched by the cache subtraction, so no
    RAM beyond what the kernel reports available is ever promised."""
    host.set(mem_available_bytes=6 * GIB, mem_total_bytes=64 * GIB, slice_memory_max=8 * GIB,
             slice_memory_current=7 * GIB, slice_memory_stat={"anon": 0, "file": 7 * GIB})
    assert host.acquire("big", 9 * GIB)["code"] == "ABOVE_SLICE_CEILING"
    d = host.acquire("mid", 4 * GIB)          # slice-wise fine (7 GiB dead), host-wise 6 - 3 = 3 GiB free
    assert d["verdict"] == A.QUEUED and d["code"] == "HOST_HEADROOM"


# ---- ADM-STALE-LEASE-01 (2026-10-01): a dead holder must not leave a lease holding bytes ------------
# 03:53:27Z a queued wrapper was TERMed while its acquirer was mid-acquire; at 03:53:42Z the acquirer
# wrote an ADMITTED 4G lease and exited without arming; the gate held the 4G for the grace and queued
# another load on HOST_HEADROOM until a manual release at 03:55:16Z.

def test_2026_10_01_no_lease_is_written_when_the_holder_wrapper_is_dead(host):
    host.patch(alive={"990001": False})
    req = A.Request(name="orphan", cap_bytes=4 * GIB, label="orphan", holder_pid=990001)
    d = A.acquire(host.store, host.res, req, host.now)
    assert d["verdict"] == A.REFUSED and d["code"] == "HOLDER_GONE"
    assert host.live_lease_ids() == []


def test_2026_10_01_a_live_holder_still_gets_its_lease_and_it_is_recorded(host):
    host.patch(alive={"990002": True})
    req = A.Request(name="ok", cap_bytes=4 * GIB, label="ok", holder_pid=990002)
    d = A.acquire(host.store, host.res, req, host.now)
    assert d["verdict"] == A.ADMITTED and d["lease"]["holder_pid"] == 990002


def test_2026_10_01_an_unarmed_lease_whose_holder_died_is_reclaimed_after_the_short_bound(host):
    host.patch(alive={"990003": True})
    req = A.Request(name="late", cap_bytes=4 * GIB, label="late", holder_pid=990003)
    lid = A.acquire(host.store, host.res, req, host.now)["lease_id"]
    host.patch(alive={"990003": False})               # the wrapper is TERMed before arming
    host.tick(A.NEVER_ARMED_BOUND_SECONDS - 5)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == []        # still inside the bound
    host.tick(10)
    sw = A.reclaim(host.store, host.res, host.now)
    assert sw["freed"] == [lid]
    assert any(r.get("reclaim_cause") == "LEASE_RECLAIMED_NEVER_ARMED" for r in host.ledger())
    d = host.acquire("next", 4 * GIB)
    assert d["readings"]["held_unrealised_bytes"] == 0     # the 4G is no longer held


def test_2026_10_01_an_unarmed_lease_with_a_live_holder_keeps_the_full_grace(host):
    host.patch(alive={"990004": True})
    req = A.Request(name="slow", cap_bytes=4 * GIB, label="slow", holder_pid=990004)
    lid = A.acquire(host.store, host.res, req, host.now)["lease_id"]
    host.tick(A.NEVER_ARMED_BOUND_SECONDS + 30)       # past the bound but inside ARM_GRACE_SECONDS
    assert A.reclaim(host.store, host.res, host.now)["freed"] == []
    assert host.live_lease_ids() == [lid]


def test_2026_10_01_a_detached_lease_is_not_reclaimed_by_the_never_armed_bound(host):
    req = A.Request(name="det", cap_bytes=2 * GIB, label="det", detached=True)
    lid = A.acquire(host.store, host.res, req, host.now)["lease_id"]
    host.tick(A.NEVER_ARMED_BOUND_SECONDS + 30)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == []
    assert host.live_lease_ids() == [lid]


def test_2026_10_01_the_launcher_passes_its_own_pid_as_the_holder(host, fake_systemd):
    r = _run_launcher(host, fake_systemd, "-m", "1G", "-n", "holder", "--", "true")
    assert r.returncode == 0, r.stdout + r.stderr
    adm = [x for x in host.ledger() if x["event"] == "ADMISSION_ADMITTED"]
    assert adm
    ret = list((host.store_dir / "retained").glob("*.json"))
    assert ret and json.loads(ret[-1].read_text())["lease"].get("holder_pid")
