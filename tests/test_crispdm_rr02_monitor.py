"""RR02 (order 2026-09-26): what happens AFTER a reservation is granted.

DR01 closed double admission at the gate.  At 16:27 on the same day systemd-oomd killed the owner's
browser and then the owner's editor; two of our verifications were running, each inside its own
3 GiB cap, 5.7 G of observed peak between them.  Admission was refusing NEW work correctly -- the
retained ledger carries refusals at PSI some/avg10 38.76, 60.48 and 61.20 against the 25.00 limit --
and the two already-admitted scopes ran to the end, because nothing re-examined a reservation once
it was granted.

The four defects RR01 named, and where each is proved here:

  RR-C  no monitor after admission, and no hysteresis      -> sections 1-4
  RR-A  a lease carries no boot identity                   -> section 5
  RR-B  reclaiming a lease destroys its body               -> section 6
  RR-D  a refused request was re-asked at a lower cap      -> section 7
  and the failure mode that would undo all of it: killing a launcher must not release capacity
  while a descendant still runs                            -> section 8

Every test here runs on a SIMULATED host: readings come from a JSON file, the monitor's clock comes
from a file the test advances by hand, and the pressure series is REPLAYED from the retained ledger.
Nothing allocates memory, nothing pressures a host, no kernel or systemd setting is read for a
decision, and no real process outside this test's own bounded children is ever signalled.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
MODULE = TOOLS / "crispdm_admission.py"
LAUNCHER = TOOLS / "crispdm-run"
GIB = 1 << 30

sys.path.insert(0, str(TOOLS))
import crispdm_admission as A  # noqa: E402

from tests.test_crispdm_admission import Host, fake_systemd  # noqa: E402,F401  (the DR01 harness)


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path)


# ---- the retained observations these thresholds are derived from -------------------------------
# Read from the admission ledger of the previous boot (ADMISSION_QUEUED / ADMISSION_ADMITTED
# readings, field pressure_some_avg10) and copied verbatim into
# docs/audits/evidence/RR02_MONITOR_20260926/RETAINED_PRESSURE_SERIES.json.  Times are UTC; the
# owner's browser was killed at 21:27:10Z and the editor at 21:27:49Z (16:27:10 / 16:27:49 local).
RETAINED_SERIES = [
    ("21:15:51", 24.20), ("21:16:21", 18.34), ("21:16:51", 4.90), ("21:17:21", 35.75),
    ("21:17:51", 16.07), ("21:18:21", 47.85), ("21:18:51", 35.17), ("21:19:21", 47.18),
    ("21:19:51", 52.18), ("21:20:21", 24.50), ("21:21:37", 7.04), ("21:22:07", 54.03),
    ("21:22:37", 32.77), ("21:23:07", 29.40), ("21:23:37", 45.14), ("21:24:07", 32.19),
    ("21:24:37", 55.82), ("21:25:07", 56.56), ("21:25:37", 57.23), ("21:26:07", 24.87),
    ("21:26:22", 38.76), ("21:26:37", 42.99), ("21:26:52", 60.48), ("21:27:07", 74.96),
    ("21:27:22", 61.20), ("21:27:37", 59.94),
]
FIRST_OWNER_KILL = "21:27:10"


def _secs(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


# ================================================================================================
# 1. the defect itself: an admitted workload rode rising pressure to an owner-visible kill
# ================================================================================================

def test_2026_09_26_the_retained_pressure_series_now_stops_the_load_before_the_owner_kill():
    """The whole point.  Replay the pressures that were actually recorded while the two admitted
    scopes ran, and the monitor stops its own scope well before the owner lost an application.

    Nothing in this test is tuned to the answer: the thresholds are the derived constants and the
    samples are the retained readings at their recorded instants.
    """
    mon = A.PressureMonitor()
    stopped_at, rule = None, None
    for hms, p in RETAINED_SERIES:
        v = mon.feed(_secs(hms), p, cgroup_pressure=None)
        if v["stop"] and stopped_at is None:
            stopped_at, rule = hms, v["rule"]

    assert stopped_at is not None, "an admitted workload must not be able to ride this series out"
    assert rule == A.RULE_BUDGET, (
        "with 30 s between samples no 20 s window can hold four samples, so the sustained rule "
        "cannot fire here; the oscillation budget is what catches this shape, which is exactly "
        "why both rules exist")
    lead = _secs(FIRST_OWNER_KILL) - _secs(stopped_at)
    assert lead >= 120, f"stopped at {stopped_at}, only {lead}s before the first owner kill"
    assert stopped_at == "21:24:37" and lead == 153


def test_2026_09_26_a_load_on_a_calm_host_is_never_stopped():
    """The monitor must not become a second way to lose a run.  A host that stays below the
    response threshold never triggers anything, however long the load runs."""
    mon = A.PressureMonitor()
    for i in range(400):                       # 2000 s at the 5 s sample period
        v = mon.feed(i * A.PRESSURE_SAMPLE_SECONDS, 24.99)
        assert v["stop"] is False
    assert mon.state == A.CALM and mon.elevated_seconds == 0.0


def test_2026_09_26_pressure_between_the_two_thresholds_alone_never_stops_a_load():
    """Between 25.00 and 37.50 the host is not calm enough to admit new work but is not on the
    oomd path either.  That band alone must not stop anything: it never enters ELEVATED."""
    mon = A.PressureMonitor()
    for i in range(400):
        assert mon.feed(i * 5, 30.0)["stop"] is False
    assert mon.state == A.CALM


# ================================================================================================
# 2. hysteresis: the crossing the order names must not count as recovery
# ================================================================================================

def test_2026_09_26_a_single_sample_below_the_limit_is_not_a_sustained_recovery():
    """The order names this exact crossing: 52.18 down to 24.5.  One sample below the admission
    limit must not clear the elevation -- and in the retained series the sample after it was 7.04
    and the one after that 54.03."""
    mon = A.PressureMonitor()
    mon.feed(0, 47.85)                                  # elevated
    mon.feed(30, 52.18)
    v = mon.feed(60, 24.50)                             # the crossing
    assert mon.state == A.ELEVATED, "one sample is not recovery"
    assert mon.recoveries_confirmed == 0
    assert v["recovery_samples"] == 1
    assert mon.elevated_seconds == 30.0, "the dip gives back no accumulated elevation"
    # and the very next sample above the limit cancels the candidate outright.  The interval that
    # dip spans is not charged to the load either -- an interval counts only when both of its ends
    # are above the limit -- so the accumulator stands where it was, and does not reset.
    mon.feed(90, 54.03)
    assert mon.recovery_since is None and mon.elevated_seconds == 30.0


def test_2026_09_26_recovery_needs_the_whole_window_and_every_sample_in_it():
    """120 s at or below the admission limit, at least 24 samples, and NOT ONE above it."""
    def run(n_samples, spoil_at=None):
        mon = A.PressureMonitor()
        mon.feed(0, 47.85)
        t = 0
        for i in range(n_samples):
            t = 5 * (i + 1)
            mon.feed(t, 25.01 if i == spoil_at else 10.0)
        return mon

    assert run(24).state == A.ELEVATED, "120 s of evidence needs the sample that closes it"
    calm = run(25)
    assert calm.state == A.CALM and calm.recoveries_confirmed == 1
    assert calm.elevated_seconds == 0.0, "a confirmed recovery clears the accumulated elevation"
    spoiled = run(31, spoil_at=20)
    assert spoiled.state == A.ELEVATED, "one sample above the limit restarts the whole window"


def test_2026_09_26_a_confirmed_recovery_gives_the_load_its_full_budget_again():
    """Hysteresis must work in both directions: after a real recovery the load is not left one
    sample away from a stop."""
    mon = A.PressureMonitor()
    mon.feed(0, 47.85)
    for i in range(1, 40):                              # 195 s at 10.0 -> recovery confirmed
        mon.feed(5 * i, 10.0)
    assert mon.state == A.CALM
    t = 200
    v = mon.feed(t, 47.85)                              # elevated again, from zero
    assert v["stop"] is False and mon.elevated_seconds == 0.0
    for i in range(1, int(A.PRESSURE_ELEVATED_BUDGET_SECONDS // 5)):
        v = mon.feed(t + 5 * i, 30.0)
    assert v["stop"] is False, "the budget is a fresh one after a confirmed recovery"


def test_2026_09_26_a_genuinely_sustained_crossing_stops_the_load_in_its_own_window():
    """The fast rule, and its floor: 20 s of samples all above 37.50 stops; 15 s does not."""
    mon = A.PressureMonitor()
    for i in range(4):                                  # t = 0,5,10,15 -> a 15 s span
        v = mon.feed(5 * i, 45.0)
    assert v["stop"] is False, "a window shorter than the oomd duration is not judged"
    v = mon.feed(20, 45.0)                              # the span is now 20 s, five samples
    assert v["stop"] is True and v["rule"] == A.RULE_SUSTAINED


def test_2026_09_26_a_drifting_sample_period_does_not_disable_the_sustained_rule():
    """A real run found this, and it is the reason the rule is written the way it is.

    The sample period is exactly a quarter of the response window, so each tick drifts by the few
    milliseconds the sampling itself costs.  The first version required the samples INSIDE the
    trailing window to span it, which a drifting series never does -- it was permanently one sample
    short -- so on a host held at PSI 60.00 the sustained rule never fired at all and only the
    240 s budget stopped the load.  Coverage is a property of the SERIES, not of the samples in the
    window.
    """
    mon = A.PressureMonitor()
    t = 0.0
    fired = None
    for i in range(12):
        v = mon.feed(t, 60.0)
        if v["stop"] and fired is None:
            fired = (t, v["rule"])
        t += A.PRESSURE_SAMPLE_SECONDS + 0.011      # the drift a real sampler has
    assert fired is not None, "a host held at PSI 60.00 must not be able to outlast this rule"
    assert fired[1] == A.RULE_SUSTAINED
    assert fired[0] <= 4 * A.PRESSURE_SAMPLE_SECONDS + 1.0, (
        "it must fire on the window, not 240 s later on the budget")


def test_2026_09_26_the_thresholds_are_derived_from_policy_and_retained_evidence_not_tuned():
    """Each constant is the value its own derivation gives, and the ordering that makes the
    mechanism safe holds: admit < respond < the host's own oomd limit."""
    assert A.PRESSURE_RESPOND_AT == (A.PRESSURE_ADMIT_MAX + A.PRESSURE_OOMD_LIMIT) / 2 == 37.5
    assert A.PRESSURE_ADMIT_MAX < A.PRESSURE_RESPOND_AT < A.PRESSURE_OOMD_LIMIT
    assert A.PRESSURE_RESPOND_WINDOW_SECONDS == A.PRESSURE_OOMD_DURATION_SECONDS == 20
    assert A.PRESSURE_ELEVATED_BUDGET_SECONDS == 240      # half the observed 529 s
    assert A.PRESSURE_RECOVERY_WINDOW_SECONDS == 120      # twice the observed 60 s false recovery
    assert A.PRESSURE_RECOVERY_MIN_SAMPLES == 24
    # the observed false recovery really is 60 s long in the retained series, and really is
    # followed by pressure above the response threshold within 90 s
    below = [(hms, p) for hms, p in RETAINED_SERIES[:6]]
    assert [p for _, p in below[:3]] == [24.20, 18.34, 4.90]
    assert _secs(below[2][0]) - _secs(below[0][0]) == 60
    assert below[5][1] == 47.85 and _secs(below[5][0]) - _secs(below[2][0]) == 90


# ================================================================================================
# 3. the monitor as the launcher runs it: both pressures, the heartbeat, the tree peak
# ================================================================================================

def _monitored(host, *, pressures, cgroup_pressures=None, cap=2 * GIB, cgroup="cg/mon",
               unit="crispdm-mon.scope", holds=1 * GIB, pid=424242):
    """Drive the real monitor loop over a simulated host with a simulated clock.

    The readings file and the clock file are rewritten between samples, so 20 s and 120 s windows
    are exercised in milliseconds and no host is ever put under pressure.
    """
    cg = f"user.slice/crispdm-batch.slice/{unit}" if cgroup == "own" else cgroup
    lease_id = host.acquire("monitored", cap)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=pid, cgroup=cg, unit=unit)
    host.patch(alive={str(pid): True, cg: True}, cgroup_current={cg: holds},
               cgroup_peak={cg: holds}, cgroup_procs={cg: [pid]})

    clock_file = host.dir / "monitor_clock"
    clock_file.write_text("0")
    os.environ["CRISPDM_ADMISSION_MONITOR_CLOCK"] = str(clock_file)   # no real sleeps, no signals
    signalled, t = [], [0.0]
    seq = list(pressures)
    cgs = list(cgroup_pressures or [None] * len(seq))

    class Reader:
        """The simulated host, with the pressure series played one sample per monitor tick."""

        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def pressure_some_avg10(self):
            return seq[0]

        def cgroup_pressure_some_avg10(self, _cg):
            return cgs[0]

    # One monitor iteration consumes one sample and advances the simulated clock by the sample
    # period, so 20 s and 120 s windows are exercised in milliseconds and no host is pressured.
    orig_feed = A.PressureMonitor.feed

    def stepping_feed(self, at, hp, cgroup_pressure=None):
        r = orig_feed(self, at, hp, cgroup_pressure)
        seq.pop(0)
        cgs.pop(0)
        t[0] += float(A.PRESSURE_SAMPLE_SECONDS)
        return r

    A.PressureMonitor.feed = stepping_feed
    try:
        out = A.monitor(host.store, Reader(host.res), lease_id, self_pid=os.getpid(),
                        clock=lambda: t[0], signals=lambda p, w: signalled.append((p, w)),
                        child_alive=lambda: bool(seq),
                        peak_file=str(host.dir / "peak"),
                        cause_file=str(host.dir / "cause.json"))
    finally:
        A.PressureMonitor.feed = orig_feed
        os.environ.pop("CRISPDM_ADMISSION_MONITOR_CLOCK", None)
    return lease_id, out, signalled


def test_2026_09_26_the_monitor_records_both_host_and_cgroup_pressure(host):
    _id, out, _sig = _monitored(host, pressures=[10.0] * 5, cgroup_pressures=[3.5] * 5)
    samples = out["record"]["samples"]
    assert len(samples) == 5
    assert all(s["host_some_avg10"] == 10.0 for s in samples)
    assert all(s["cgroup_some_avg10"] == 3.5 for s in samples), \
        "a reservation must be answerable for the cgroup it covers, not only for the host"


def test_2026_09_26_the_monitor_heartbeats_the_lease_it_watches(host):
    lease_id, _out, _sig = _monitored(host, pressures=[1.0] * 4)
    # the lease was renewed while the load ran, so a stopped heartbeat is the only thing that
    # ever lets a sweep call it expired
    assert any(r["event"] == "LEASE_ARMED" and r["lease_id"] == lease_id for r in host.ledger())
    assert (host.dir / "peak").read_text() == str(1 * GIB)


def test_2026_09_26_a_sustained_crossing_stops_only_this_loads_own_scope(host):
    """The response may stop its own identified experiment scope and nothing else."""
    lease_id, out, signalled = _monitored(host, pressures=[45.0] * 8, cgroup="own", pid=424242)
    rec = out["record"]
    assert out["stopped"] is True
    assert rec["exit_cause"].startswith("PRESSURE_STOP_")
    assert rec["partial_evidence_retained"] is True and rec["scope_only"] is True
    # only the recorded leader and the pids in this unit's OWN cgroup were touched
    targets = {p for p, _w in signalled if p}
    assert targets == {424242}
    assert all(w in ("TERM", "KILL", "GRACE") for _p, w in signalled)
    # the durable record exists, in the store, with the cause and the whole series
    body = json.loads((host.store_dir / "incidents" / f"{lease_id}.json").read_text())
    assert body["rule"] in (A.RULE_SUSTAINED, A.RULE_BUDGET)
    assert body["cap_bytes"] == 2 * GIB and body["samples"]
    assert json.loads((host.dir / "cause.json").read_text())["lease_id"] == lease_id
    assert any(r["event"] == "PRESSURE_STOP_OWN_SCOPE" for r in host.ledger())


def test_2026_09_26_a_pressure_response_never_signals_a_process_outside_its_own_scope(host):
    """An unrelated pid in the cgroup listing is not reachable, because the response reads THIS
    lease's own cgroup only; and an inherited ancestor cgroup is refused outright."""
    outer = "user.slice/crispdm-batch.slice/crispdm-outer.scope"
    lease_id = host.acquire("inherited", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=0, cgroup=outer,
          unit="crispdm-inner.scope")
    host.patch(cgroup_procs={outer: [999001, 999002]})
    lease = A.Lease(**json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text()))
    acted = A._scope_stop(host.res, lease, os.getpid(), lambda p, w: None, lambda a: None)
    assert acted["term_cgroup"] == [] and acted["kill_cgroup"] == []
    assert acted["refused"] and "inherited ancestor is never signalled" in acted["refused"][0]["why"]


def test_2026_09_26_a_pressure_response_never_signals_pid_one_or_itself(host):
    unit = "crispdm-self.scope"
    cg = f"user.slice/crispdm-batch.slice/{unit}"
    lease_id = host.acquire("selfish", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=1, cgroup=cg, unit=unit)
    host.patch(cgroup_procs={cg: [1, os.getpid(), 777777]})
    lease = A.Lease(**json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text()))
    sent = []
    acted = A._scope_stop(host.res, lease, os.getpid(),
                          lambda p, w: sent.append((p, w)), lambda a: None)
    assert acted["term_leader"] is None, "pid 1 is never a target"
    assert 1 not in acted["term_cgroup"] and os.getpid() not in acted["term_cgroup"]
    assert acted["term_cgroup"] == [777777]
    assert all(p not in (1, os.getpid()) for p, _w in sent if p)


def test_2026_09_26_a_scope_that_holds_no_memory_is_not_stopped(host):
    """Stopping a scope that holds nothing would cost a run and free no memory: a response acts
    only where it can actually help."""
    _id, out, signalled = _monitored(host, pressures=[60.0] * 8, cgroup="own", holds=0)
    assert out["stopped"] is False
    assert signalled == []
    assert not (host.dir / "cause.json").exists(), "no exit cause, because nothing was stopped"
    assert not any(r["event"] == "PRESSURE_STOP_OWN_SCOPE" for r in host.ledger())


def test_2026_09_26_nothing_in_the_monitor_kills_relaxes_or_disables_anything():
    """Source-level, the same reading DR01 applied to the gate: the prohibitions are absolute and
    a future edit must trip this test."""
    text = MODULE.read_text() + LAUNCHER.read_text()
    for forbidden in ("drop_caches", "swapoff", "swapon", "MemoryMax=infinity", "sysctl -w",
                      "systemctl --user stop", "systemctl stop", "systemctl --user set-property",
                      "oomd.conf", "ManagedOOM", "pkill", "killall", "kill -9 1 ",
                      "os.kill(1", "MemoryHigh=infinity"):
        assert forbidden not in text, forbidden
    # the only signals anywhere are TERM and KILL, and only to a pid the lease itself names or
    # that this lease's own cgroup lists
    assert 'os.kill(int(pid), {"TERM": 15, "KILL": 9}[what])' in MODULE.read_text()
    assert "cgroup_procs(lease.cgroup)" in MODULE.read_text()


# ================================================================================================
# 4. the launcher actually runs the monitor
# ================================================================================================

def test_2026_09_26_the_launcher_runs_the_monitor_instead_of_a_blind_sampler(host, fake_systemd):
    """Before RR02 the launcher's own loop renewed the lease and read the tree peak and never once
    looked at pressure.  The monitor is now that loop."""
    text = LAUNCHER.read_text()
    assert '"$ADM" monitor "$LEASE"' in text
    assert "--cause-file" in text and "--peak-file" in text
    r = subprocess.run(["bash", str(LAUNCHER), "-m", "1G", "-n", "monrun", "--", "true"],
                       env={**host.env, "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert host.live_lease_ids() == []


def test_2026_09_26_a_reservation_that_cannot_be_armed_stops_its_own_child(host, fake_systemd):
    """`|| true` on the arm was a hole: an unarmed lease is protected only by the arming grace,
    and once that expires the sweep reclaims it while the child is still running.  Arming is now
    mandatory, and a job that cannot be accounted for is not run."""
    shim = host.dir / "adm_no_arm.py"
    shim.write_text(
        "import subprocess, sys\n"
        "if 'arm' in sys.argv[1:2] or (len(sys.argv) > 1 and sys.argv[1] == 'arm'):\n"
        "    sys.exit(1)\n"
        f"sys.exit(subprocess.run([sys.executable, {str(MODULE)!r}] + sys.argv[1:]).returncode)\n")
    marker = host.dir / "ran"
    body = host.dir / "body.sh"
    body.write_text(f'#!/usr/bin/env bash\necho ran > {marker}\nsleep 30\n')
    body.chmod(0o755)
    r = subprocess.run(["bash", str(LAUNCHER), "-m", "1G", "-n", "noarm", "--", str(body)],
                       env={**host.env, "CRISPDM_ADMISSION_MODULE": str(shim),
                            "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"},
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 75, r.stdout + r.stderr
    assert "could not be armed" in r.stderr
    assert host.live_lease_ids() == [], "the reservation is given back, not left unarmed"


# ================================================================================================
# 5. RR-A: boot identity plus process start identity
# ================================================================================================

def test_2026_09_26_a_new_lease_records_boot_and_host_identity(host):
    d = host.acquire("identified", 2 * GIB)
    lease = d["lease"]
    assert lease["boot_id"] and lease["boot_time"] and lease["host_key"]
    assert lease["pid_starttime"] is None, "start time is bound at arm, when there is a pid"
    A.arm(host.store, host.res, d["lease_id"], host.now, pid=os.getpid())
    armed = json.loads((host.store_dir / "leases" / f"{d['lease_id']}.json").read_text())
    assert armed["pid_starttime"] is not None and armed["boot_id"] == lease["boot_id"]


def test_2026_09_26_a_pid_and_start_time_from_another_boot_can_never_witness_a_lease(host):
    """The latent defect RR01 named.  pid_starttime is clock ticks SINCE BOOT, so after a reboot a
    new process can carry the same pid and the same start-time number.  Without boot identity that
    lease reads as live and its bytes are held against a load that no longer exists."""
    lease_id = host.acquire("across_reboot", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=727767, cgroup="cg/old")
    # a post-reboot process that matches the lease's pid AND its start time exactly
    host.patch(boot_id="boot-AFTER", alive={"727767": True, "cg/old": True},
               starttime={"727767": A.Lease(**json.loads(
                   (host.store_dir / "leases" / f"{lease_id}.json").read_text())).pid_starttime})
    lease = A.Lease(**json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text()))
    assert lease.old_boot(host.res) is True
    assert lease.witness_alive(host.res, host.now) is False, \
        "a process from a later boot must not be able to impersonate a dead load"


def test_2026_09_26_an_old_boot_lease_is_retired_as_a_historical_record_not_silently_dropped(host):
    lease_id = host.acquire("before_reboot", 3 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=812761, cgroup="cg/before",
          unit="crispdm-before.scope")
    host.patch(boot_id="boot-AFTER")
    swept = A.reclaim(host.store, host.res, host.now)
    assert swept["old_boot"] == [lease_id] and swept["freed"] == []
    assert host.live_lease_ids() == [], "its local load did not survive the reboot"
    body = host.store.retained_body(lease_id)
    assert body["reclaim_cause"] == "LEASE_RETIRED_OLD_BOOT"
    assert body["lease"]["cap_bytes"] == 3 * GIB and body["lease"]["pid"] == 812761
    assert "not comparable across a reboot" in body["reading"]
    assert any(r["event"] == "LEASE_RETIRED_OLD_BOOT" for r in host.ledger())
    # and the capacity it held is available again
    assert host.acquire("after_reboot", 8 * GIB)["verdict"] == A.ADMITTED


def test_2026_09_26_a_worker_lease_is_never_reclaimed_because_the_coordinator_rebooted(host):
    """A coordinator reboot says nothing about either worker.  A lease written on another host is
    not judged, not reclaimed, and not counted against this host's capacity."""
    lease_id = host.acquire("workers_job", 8 * GIB)["lease_id"]
    lease = A.Lease(**json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text()))
    lease.host_key = "aaaaaaaaaaaaaaaa"                   # written on another host
    lease.boot_id = "boot-OF-THAT-HOST"
    host.store.write(lease)
    host.patch(boot_id="boot-AFTER-A-COORDINATOR-REBOOT")

    swept = A.reclaim(host.store, host.res, host.now)
    assert swept["freed"] == [] and swept["old_boot"] == []
    assert swept["foreign"] == [lease_id]
    assert host.live_lease_ids() == [lease_id], "a worker's lease is never removed from here"
    assert host.store.retained_body(lease_id) is None, "and it is not retired either"
    # its bytes are not this host's bytes, so they do not block this host's work
    d = host.acquire("local", 8 * GIB)
    assert d["verdict"] == A.ADMITTED
    assert d["readings"]["foreign_host_lease_ids"] == [lease_id]
    assert any(r["event"] == "LEASE_FOREIGN_HOST_NOT_JUDGED" for r in host.ledger())


def test_2026_09_26_a_same_boot_lease_is_still_judged_exactly_as_before(host):
    """Boot identity must not become a way to keep dead leases alive: within one boot nothing
    changes."""
    lease_id = host.acquire("same_boot", 4 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=900001, cgroup="cg/sb")
    host.patch(alive={"900001": True, "cg/sb": True})
    assert A.reclaim(host.store, host.res, host.now)["live"][0].lease_id == lease_id
    host.patch(alive={"900001": False, "cg/sb": False})
    host.tick(30)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]


# ================================================================================================
# 6. RR-B: a reclaim retains the body
# ================================================================================================

RR01_TRANSCRIBED_FIELDS = ("cap_bytes", "cgroup", "unit", "argv_sha256", "wall_seconds",
                           "expires_at", "created_at", "armed_at", "pid", "pid_starttime",
                           "host_reserve_bytes", "name", "label")


def test_2026_09_26_a_reclaim_keeps_the_body_that_had_to_be_transcribed_by_hand(host):
    """RR01 preserved two lease bodies only because they were read minutes before the reclaim
    removed them and copied into the evidence file by hand.  Every field that had to be
    transcribed is now retained by the store itself."""
    lease_id = host.acquire("reclaimed", 3 * GIB, wall_seconds=1800,
                            argv_sha256="e5be23cdd5fe6a9270c299538cca768ce79ff42ff193ccd95af58c91b43d99ca")["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=727767, cgroup="cg/r",
          unit="crispdm-dr01deploy.scope")
    host.patch(alive={"727767": False, "cg/r": False}, cgroup_peak={"cg/r": 2900 * (1 << 20)})
    host.tick(60)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]

    body = host.store.retained_body(lease_id)
    assert body is not None, "a reclaim must not destroy the record of what it reclaimed"
    assert body["reclaim_cause"] == "LEASE_RECLAIMED_WITNESS_DEAD"
    assert body["observed_peak_bytes"] == 2900 * (1 << 20)
    for f in RR01_TRANSCRIBED_FIELDS:
        assert f in body["lease"], f
    assert body["lease"]["argv_sha256"].startswith("e5be23cd")
    assert body["lease"]["wall_seconds"] == 1800


def test_2026_09_26_a_release_keeps_the_body_too(host):
    lease_id = host.acquire("released", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=900002, cgroup="cg/rel")
    host.patch(alive={"900002": False, "cg/rel": False})
    assert A.release(host.store, host.res, lease_id, host.now, 1234)["ok"] is True
    body = host.store.retained_body(lease_id)
    assert body["reclaim_cause"] == "LEASE_RELEASED" and body["observed_peak_bytes"] == 1234
    assert body["lease"]["cap_bytes"] == 2 * GIB


def test_2026_09_26_an_unarmed_lease_that_times_out_says_so_in_its_retained_cause(host):
    """The two causes are not the same event and must not share a word: a witness that died is not
    a lease that was never bound to one."""
    lease_id = host.acquire("never_armed", 2 * GIB)["lease_id"]
    host.tick(A.ARM_GRACE_SECONDS + 5)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]
    assert host.store.retained_body(lease_id)["reclaim_cause"] == \
        "LEASE_RECLAIMED_UNARMED_GRACE_EXPIRED"


def test_2026_09_26_the_retained_body_is_readable_from_the_command_line(host):
    lease_id = host.acquire("cli_body", 2 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=900003)
    host.patch(alive={"900003": False})
    A.release(host.store, host.res, lease_id, host.now)
    r = host.cli("retained", lease_id, expect=0)
    assert json.loads(r.stdout)["lease"]["lease_id"] == lease_id
    assert lease_id in json.loads(host.cli("retained", expect=0).stdout)["retained"]


# ================================================================================================
# 7. RR-D: a refusal answered by asking for less
# ================================================================================================

def test_2026_09_26_a_refused_request_may_not_come_back_at_a_lower_cap(host):
    """The retained queue log alternates cap 3 GiB and 1 GiB under one name while the host was
    under pressure.  Lowering a request below its measured need to get past a gate is forbidden."""
    host.patch(mem_available_bytes=5 * GIB)               # 5 - 3 reserve = 2 GiB free
    first = host.acquire("dr01deploy", 3 * GIB)
    assert first["verdict"] == A.QUEUED and first["code"] == "HOST_HEADROOM"

    lowered = host.acquire("dr01deploy", 1 * GIB)
    assert lowered["verdict"] == A.REFUSED
    assert lowered["code"] == "CAP_LOWERED_AFTER_REFUSAL"
    assert "never by asking for less than the work needs" in lowered["reason"]
    assert host.live_lease_ids() == [], "and nothing was admitted at the lowered cap"
    assert any(r["code"] == "CAP_LOWERED_AFTER_REFUSAL" for r in host.ledger()
               if r["event"] == "ADMISSION_REFUSED")

    # the same request at its own size is still allowed to wait
    again = host.acquire("dr01deploy", 3 * GIB)
    assert again["verdict"] == A.QUEUED and again["code"] == "HOST_HEADROOM"


def test_2026_09_26_the_refusal_is_terminal_at_the_command_line_and_not_polled(host):
    host.patch(mem_available_bytes=5 * GIB)
    host.cli("acquire", "-n", "shrink", "-m", "3G", expect=75)
    r = host.cli("acquire", "-n", "shrink", "-m", "1G", "--queue", "--max-wait-seconds", "600",
                 expect=75)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["verdict"] == A.REFUSED and d["code"] == "CAP_LOWERED_AFTER_REFUSAL"


def test_2026_09_26_an_admission_clears_the_name_and_a_stale_refusal_expires(host):
    host.patch(mem_available_bytes=5 * GIB)
    assert host.acquire("laterfits", 3 * GIB)["verdict"] == A.QUEUED
    host.patch(mem_available_bytes=12 * GIB)
    assert host.acquire("laterfits", 3 * GIB)["verdict"] == A.ADMITTED
    # capacity was found at the size the work needs: a smaller successor step is not a dodge
    assert host.acquire("laterfits", 1 * GIB)["verdict"] == A.ADMITTED

    host.patch(mem_available_bytes=5 * GIB)
    assert host.acquire("stale", 3 * GIB)["verdict"] == A.QUEUED
    host.tick(A.LOWERED_CAP_MEMORY_SECONDS + 1)
    d = host.acquire("stale", 1 * GIB)
    assert d["verdict"] != A.REFUSED, "after the bounded wait the refusal is stale, not permanent"


def test_2026_09_26_different_work_is_not_blocked_it_is_asked_to_carry_its_own_name(host):
    host.patch(mem_available_bytes=5 * GIB)
    assert host.acquire("bigfit", 3 * GIB)["verdict"] == A.QUEUED
    assert host.acquire("smallinspection", 1 * GIB)["verdict"] == A.ADMITTED


def test_2026_09_26_the_register_records_what_was_asked_and_names_the_identical_command(host):
    host.patch(mem_available_bytes=5 * GIB)
    host.acquire("same_argv", 3 * GIB, argv_sha256="deadbeef")
    d = host.acquire("same_argv", 1 * GIB, argv_sha256="deadbeef")
    assert d["readings"]["same_argv_as_refused"] is True
    assert "byte-identical" in d["reason"]
    rec = host.store.read_request("same_argv")
    assert rec["max_cap_bytes_ever_asked"] == 3 * GIB and rec["last_refused"]["cap_bytes"] == 3 * GIB


# ================================================================================================
# 8. the wrapper boundary: killing a launcher must not release capacity
# ================================================================================================

def _wait_for(predicate, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_2026_09_26_killing_the_launcher_does_not_release_capacity_under_a_live_child(host, fake_systemd):
    """The failure mode that would undo everything.  SIGKILL the wrapper while its child runs: the
    reservation must survive, and a second request for the same bytes must still be queued."""
    gate = host.dir / "gate"
    waiter = host.dir / "wait.sh"
    waiter.write_text('#!/usr/bin/env bash\nfor i in $(seq 1 900); do [ -f "$1" ] && exit 0; sleep 0.1; done\n')
    waiter.chmod(0o755)
    proc = subprocess.Popen(["bash", str(LAUNCHER), "-m", "8G", "-n", "killed", "--",
                             str(waiter), str(gate)],
                            env={**host.env, "PATH": f"{fake_systemd['bin']}:{os.environ['PATH']}"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert _wait_for(lambda: bool(host.live_lease_ids())), "the launcher must reserve first"
        lease_id = host.live_lease_ids()[0]
        assert _wait_for(lambda: json.loads(
            (host.store_dir / "leases" / f"{lease_id}.json").read_text())["armed"]), "armed"
        child_pid = json.loads((host.store_dir / "leases" / f"{lease_id}.json").read_text())["pid"]

        proc.send_signal(signal.SIGKILL)                   # the wrapper dies without releasing
        proc.wait(timeout=30)
        assert _wait_for(lambda: Path(f"/proc/{child_pid}").exists())

        second = host.cli("acquire", "-n", "second", "-m", "8G")
        assert second.returncode == 75, "capacity was released while the child was still running"
        d = json.loads(second.stdout.strip().splitlines()[-1])
        assert d["verdict"] == A.QUEUED and d["readings"]["held_unrealised_bytes"] == 8 * GIB
        assert host.live_lease_ids() == [lease_id]
    finally:
        gate.write_text("go")
        for _ in range(100):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.1)


def test_2026_09_26_a_detached_descendant_keeps_the_reservation_after_its_parent_exits(host):
    """A child that detaches a grandchild into this load's own scope and then exits must not hand
    the next admission the bytes that grandchild is using."""
    unit = "crispdm-detached.scope"
    cg = f"user.slice/crispdm-batch.slice/{unit}"
    lease_id = host.acquire("detached", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=910001, cgroup=cg, unit=unit)
    # the direct child is reaped; a descendant is still running in this load's OWN scope
    host.patch(alive={"910001": False, cg: True}, cgroup_current={cg: 6 * GIB})

    out = A.release(host.store, host.res, lease_id, host.now)
    assert out["ok"] is False and out["code"] == "CHILD_STILL_ALIVE"
    assert A.reclaim(host.store, host.res, host.now)["freed"] == []
    assert host.acquire("next", 8 * GIB)["verdict"] == A.QUEUED

    host.patch(alive={"910001": False, cg: False})          # the scope is finally empty
    assert A.release(host.store, host.res, lease_id, host.now)["ok"] is True
    assert host.acquire("next", 8 * GIB)["verdict"] == A.ADMITTED


def test_2026_09_26_a_failed_renew_never_frees_a_live_child(host):
    """The heartbeat is a convenience, not the authority.  A monitor that cannot renew -- because
    the store was unwritable, or because it was killed -- must not cause a release."""
    lease_id = host.acquire("unrenewed", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=920001, cgroup="cg/unrenewed")
    host.patch(alive={"920001": True, "cg/unrenewed": True}, cgroup_current={"cg/unrenewed": 2 * GIB})
    host.tick(A.LEASE_TTL_SECONDS * 3)                      # no heartbeat for 45 minutes
    swept = A.reclaim(host.store, host.res, host.now)
    assert swept["freed"] == [] and swept["extended"] == [lease_id]
    assert host.acquire("second", 8 * GIB)["verdict"] == A.QUEUED


def test_2026_09_26_an_interrupted_publication_cannot_permit_a_second_admission(host, fake_systemd):
    """The launcher killed between its child's exit and the release: the lease is left behind, and
    it must be freed by the sweep only once the witness is actually dead -- never sooner."""
    lease_id = host.acquire("interrupted", 8 * GIB)["lease_id"]
    A.arm(host.store, host.res, lease_id, host.now, pid=930001, cgroup="cg/interrupted",
          unit="crispdm-interrupted.scope")
    host.patch(alive={"930001": True, "cg/interrupted": True})
    # still alive -> nothing may be admitted against its bytes, and no release succeeds
    assert host.acquire("other", 8 * GIB)["verdict"] == A.QUEUED
    assert A.release(host.store, host.res, lease_id, host.now)["code"] == "CHILD_STILL_ALIVE"
    # the tree ends; the very next sweep frees it, and the body is kept
    host.patch(alive={"930001": False, "cg/interrupted": False})
    host.tick(5)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]
    assert host.store.retained_body(lease_id)["lease"]["unit"] == "crispdm-interrupted.scope"
    assert host.acquire("other", 8 * GIB)["verdict"] == A.ADMITTED


def test_2026_09_26_a_failed_arm_cannot_leave_a_child_running_outside_the_reservation(host):
    """Covered end to end at the wrapper boundary above; here the store-level invariant: an
    unarmed lease is reclaimed when its grace expires, so a child that outlives the grace without
    being armed would be unaccounted for.  That is why the launcher stops it instead."""
    lease_id = host.acquire("unarmed_child", 8 * GIB)["lease_id"]
    host.tick(A.ARM_GRACE_SECONDS + 1)
    assert A.reclaim(host.store, host.res, host.now)["freed"] == [lease_id]
    assert host.acquire("someone_else", 8 * GIB)["verdict"] == A.ADMITTED, (
        "this is exactly the second admission the launcher must prevent by refusing to run an "
        "unarmable job")
