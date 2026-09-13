"""C174: the dispatcher re-reads the live inventory before every launch, starts each job as a
transient systemd user service through an injected backend (a fake here: nothing is started),
writes write-once receipts, re-attaches active units after a crash, classifies finished ones
from their stored Result, and never starts a unit twice."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import df_capacity_fakes as F  # noqa: E402
import df_dispatch as D  # noqa: E402
import df_placement as PL  # noqa: E402

GIB = F.GIB
ROLES_MAP = {"COORDINATOR": {"ssh": None}, "WORKER_A": {"ssh": "secret-alias-a"}, "WORKER_B": {"ssh": "secret-alias-b"}}

# Captured on this host (systemd 259) from `systemctl --user show <units> -p ...` of three tiny probe units
# (true, false, sleep 30 with RuntimeMaxSec=1, RemainAfterExit=yes) and a unit that does not exist.
CAPTURED_SHOW = """Id=c174-probe-ok.service
LoadState=loaded
ActiveState=active
SubState=exited
Result=success
ExecMainCode=1
ExecMainStatus=0
MemoryCurrent=[not set]
MemoryPeak=2097152

Id=c174-probe-bad.service
LoadState=loaded
ActiveState=failed
SubState=failed
Result=exit-code
ExecMainCode=1
ExecMainStatus=1
MemoryCurrent=[not set]
MemoryPeak=2097152

Id=c174-probe-slow.service
LoadState=loaded
ActiveState=failed
SubState=failed
Result=timeout
ExecMainCode=2
ExecMainStatus=15
MemoryCurrent=[not set]
MemoryPeak=2097152

Id=c174-probe-none.service
LoadState=not-found
ActiveState=inactive
SubState=dead
Result=success
ExecMainCode=0
ExecMainStatus=0
MemoryCurrent=[not set]
MemoryPeak=[not set]
"""


class Crash(Exception):
    pass


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _props(unit, load="loaded", active="active", sub="running", result="success", code="0", status="0",
           peak="[not set]", current="[not set]"):
    return {"Id": unit + ".service", "LoadState": load, "ActiveState": active, "SubState": sub, "Result": result,
            "ExecMainCode": code, "ExecMainStatus": status, "MemoryPeak": peak, "MemoryCurrent": current}


class FakeHosts:
    """In-memory transient user services on each role, driven by the fake clock."""

    def __init__(self, clock, duration=10.0, durations=None, outcomes=None, start_rcs=None, crash_on_start=None,
                 unreachable=()):
        self.clock, self.duration = clock, duration
        self.durations, self.outcomes = dict(durations or {}), dict(outcomes or {})
        self.start_rcs, self.crash_on_start = dict(start_rcs or {}), dict(crash_on_start or {})
        self.unreachable = set(unreachable)
        self.units = {}
        self.starts, self.releases, self.shows = [], [], []
        self.max_running = {}

    @staticmethod
    def job_of(unit):
        return unit[len(D.UNIT_PREFIX):].rsplit("-", 1)[0]

    def _running(self, rec):
        return self.clock() < rec["started"] + rec["duration"]

    def start(self, role, script):
        unit = re.search(r"--unit=(\S+)", script).group(1)
        jid = self.job_of(unit)
        if self.crash_on_start.get(jid):
            self.crash_on_start[jid] -= 1
            raise Crash(f"dispatcher died while starting {jid}")
        if self.start_rcs.get(jid):
            return {"rc": self.start_rcs[jid].pop(0), "message": "REFUSED_AT_LAUNCH: request exceeds MemAvailable - 3G"}
        if (role, unit) in self.units:
            return {"rc": D.EXIT_UNIT_LOADED, "message": "UNIT_ALREADY_LOADED"}
        self.starts.append((role, unit, script))
        self.units[(role, unit)] = {"started": self.clock(), "duration": self.durations.get(jid, self.duration),
                                    "outcome": self.outcomes.get(jid, ("exit", 0))}
        live = sum(1 for (r, _), rec in self.units.items() if r == role and self._running(rec))
        self.max_running[role] = max(self.max_running.get(role, 0), live)
        return {"rc": 0, "message": ""}

    def show(self, role, units):
        self.shows.append((role, tuple(units)))
        if role in self.unreachable:
            return None
        out = {}
        for u in units:
            rec = self.units.get((role, u))
            if rec is None:
                out[u] = _props(u, load="not-found", active="inactive", sub="dead")
            elif self._running(rec):
                out[u] = _props(u, current=str(100 << 20), peak=str(200 << 20))
            else:
                o, peak = rec["outcome"], str(700 << 20)
                if o == ("exit", 0):
                    out[u] = _props(u, sub="exited", code="1", status="0", peak=peak)
                elif o[0] == "exit":
                    out[u] = _props(u, active="failed", sub="failed", result="exit-code", code="1", status=str(o[1]),
                                    peak=peak)
                else:
                    res = {"oom": "oom-kill", "timeout": "timeout", "signal": "signal"}[o[0]]
                    out[u] = _props(u, active="failed", sub="failed", result=res, code="2",
                                    status="9" if o[0] == "oom" else "15", peak=peak)
        return out

    def release(self, role, unit, props):
        self.releases.append((role, unit, props["ActiveState"]))
        self.units.pop((role, unit), None)
        return 0

    def start_count(self, jid):
        return sum(1 for _, u, _ in self.starts if self.job_of(u) == jid)


def job(jid, cpu_gib=1.0, gpu_gib=0.0, **kw):
    return {"job_id": jid, "argv": ["~/anaconda3/envs/trading-stack/bin/python", "tools/x.py", jid],
            "cpu_bytes": int(cpu_gib * GIB), "gpu_bytes": int(gpu_gib * GIB), "cpus": 1, "wall": "30m", **kw}


def make(tmp_path, jobs, inventory_fn, hosts, clock, **kw):
    kw.setdefault("sleep", clock.sleep)
    return D.Dispatcher(tmp_path / "dispatch", jobs, inventory_fn=inventory_fn, backend=hosts, clock=clock,
                        poll_seconds=5, max_inventory_age_seconds=2, **kw)


def receipts(root):
    return {p.name: json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))}


# ------------------------------------------------------------ unit semantics
def test_unit_names_are_deterministic_from_the_job_identity():
    j = job("det", 1)
    n = D.unit_name(j)
    assert n == D.unit_name(json.loads(json.dumps(j))) and n.startswith("crispdm-dispatch-det-")
    assert re.fullmatch(r"[0-9a-f]{16}", n.rsplit("-", 1)[1])
    assert D.unit_name(job("det", 2)) != n


def test_start_script_is_a_user_service_in_the_batch_slice_behind_the_admission_check():
    j = job("svc", 2)
    s = D.build_start_script(j, {"request_bytes": PL.request_bytes(j["cpu_bytes"], PL.default_policy())},
                             "Documents/GitHub/.worktrees/predictor-c146")
    req = 2560 << 20
    assert f"REQ={req}" in s and "MemAvailable" in s and f"AVAIL - {3 << 30}" in s and "exit 75" in s
    assert "systemctl --user show crispdm-batch.slice -p MemoryMax --value" in s
    assert f"systemctl --user show {D.unit_name(j)}.service -p LoadState --value" in s and "exit 76" in s
    run = s.strip().splitlines()[-1]
    assert run.startswith(f"exec systemd-run --user --quiet --unit={D.unit_name(j)} --slice=crispdm-batch.slice ")
    for part in (f"-p MemoryMax={req}", "-p MemorySwapMax=0", "-p RuntimeMaxSec=30m", "-p RemainAfterExit=yes",
                 '--working-directory="$HOME"/Documents/GitHub/.worktrees/predictor-c146',
                 "env -u PYTHONPATH OMP_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES= ",
                 '"$HOME"/anaconda3/envs/trading-stack/bin/python tools/x.py svc'):
        assert part in run
    assert "--collect" not in s and "crispdm-run" not in s and str(Path.home()) not in s
    g = D.build_start_script(j, {"request_bytes": req, "gpu": {"uuid": "GPU-good-b"}}, "ck")
    assert "CUDA_VISIBLE_DEVICES=GPU-good-b " in g


def test_captured_systemd_states_classify_as_declared():
    blocks = D.parse_show(CAPTURED_SHOW)
    c = {k.split(".")[0].rsplit("-", 1)[1]: D.classify_unit(v) for k, v in blocks.items()}
    assert (c["ok"]["state"], c["ok"]["status"], c["ok"]["exit_code"], c["ok"]["memory_peak_bytes"]) == \
        ("FINISHED", "COMPLETED", 0, 2097152)
    assert (c["bad"]["status"], c["bad"]["exit_code"]) == ("FAILED", 1)
    assert c["slow"]["status"] == "RESOURCE_EXCEEDED" and "timeout" in c["slow"]["reason"]
    assert c["none"]["state"] == "NOT_FOUND" and c["none"]["status"] == "LAUNCH_NOT_FOUND"
    oom = D.classify_unit(_props("u", active="failed", sub="failed", result="oom-kill", code="2", status="9"))
    assert oom["status"] == "RESOURCE_EXCEEDED" and "OOM" in oom["reason"]
    assert D.classify_unit(_props("u"))["state"] == "RUNNING"
    assert D.classify_unit(_props("u", active="failed", sub="failed", result="signal", code="2",
                                  status="15"))["status"] == "UNCERTAIN"
    assert D.classify_unit(None)["status"] == "UNCERTAIN"
    assert D.classify_unit({"Id": "x"})["status"] == "UNCERTAIN"
    assert D.classify_unit(_props("u", active="weird"))["status"] == "UNCERTAIN"


def test_systemd_backend_runs_locally_or_over_ssh_and_never_leaks_the_alias():
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        if argv[0] == "ssh" and "secret-alias-b" in argv:
            return subprocess.CompletedProcess(argv, 255, "", "ssh: connect to host secret-alias-b (10.9.8.7): refused")
        if "systemctl --user show" in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, CAPTURED_SHOW, "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    b = D.SystemdBackend(ROLES_MAP, runner=runner)
    j = job("be", 1)
    script = D.build_start_script(j, {"request_bytes": 1280 << 20}, "ck")
    assert b.start("COORDINATOR", script)["rc"] == 0 and calls[-1] == ["bash", "-c", script]
    b.start("WORKER_A", script)
    assert calls[-1][0] == "ssh" and calls[-1][5] == "secret-alias-a"
    assert calls[-1][6].startswith("bash -c ") and "MemAvailable" in calls[-1][6] and "systemd-run --user" in calls[-1][6]
    refused = b.start("WORKER_B", script)
    assert refused["rc"] == 255 and "secret-alias" not in refused["message"] and "10.9.8.7" not in refused["message"]
    shown = b.show("COORDINATOR", ["c174-probe-ok", "c174-probe-slow", "missing-unit"])
    assert shown["c174-probe-ok"]["Result"] == "success" and shown["missing-unit"] is None
    assert "-p MemoryPeak" in calls[-1][-1] and "-p ExecMainStatus" in calls[-1][-1]
    assert b.show("WORKER_B", ["x"]) is None
    b.release("WORKER_A", "u1", {"ActiveState": "failed"})
    assert "systemctl --user reset-failed u1.service" in calls[-1][-1]
    b.release("COORDINATOR", "u2", {"ActiveState": "active"})
    assert calls[-1] == ["bash", "-c", "systemctl --user stop u2.service"]


# ------------------------------------------------------------------- loop
def test_coordinator_memory_drop_between_launches_is_seen_before_the_second_launch(tmp_path):
    reads = []

    def inv():
        n = len(reads) + 1
        avail = 22 if n == 1 or n >= 6 else 6
        doc = F.inventory(coord=F.role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=avail, slice_gib=14),
                          stamp=f"read-{n}")
        reads.append(doc)
        return doc

    clock = Clock()
    hosts = FakeHosts(clock, duration=500)
    d = make(tmp_path, [job("a", 4, roles=["COORDINATOR"]), job("b", 4, roles=["COORDINATOR"])], inv, hosts, clock)
    r = d.run()
    assert r["final"] and r["counts"] == {"COMPLETED": 2}
    (ja, _, dig_a), (jb, _, dig_b) = d.launches
    assert (ja, jb) == ("a", "b") and dig_a != dig_b
    launch_b = receipts(tmp_path / "dispatch")["b.attempt-1.launch.json"]
    used = next(x for x in reads if x["digest_sha256"] == launch_b["inventory_digest"])
    assert used["roles"]["COORDINATOR"]["mem_available_bytes"] == 22 * GIB
    assert d.inventory_reads >= 6


def test_every_launch_uses_a_reading_taken_after_the_previous_launch_and_workers_come_first(tmp_path):
    count = {"n": 0}

    def inv():
        count["n"] += 1
        return F.inventory(stamp=f"r{count['n']}")

    clock = Clock()
    hosts = FakeHosts(clock, duration=200)
    d = make(tmp_path, [job(f"j{i}", 1) for i in range(5)], inv, hosts, clock)
    d.run()
    digests = [x[2] for x in d.launches]
    assert len(digests) == 5 and len(set(digests)) == 5
    assert [x[1] for x in d.launches] == ["WORKER_A"] * 3 + ["WORKER_B"] * 2       # never the COORDINATOR
    assert hosts.max_running["WORKER_A"] <= 3 and hosts.max_running["WORKER_B"] <= 2


def test_a_job_too_big_for_every_role_is_split_and_its_chunks_run(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5)
    d = make(tmp_path, [job("big", 30, split={"max_chunks": 6, "fixed_cpu_bytes": GIB})], lambda: F.inventory(),
             hosts, clock)
    r = d.run()
    rec = receipts(tmp_path / "dispatch")
    assert rec["big.attempt-1.json"]["status"] == "SPLIT" and rec["big.attempt-1.json"]["n_chunks"] == 3
    assert r["counts"] == {"SPLIT": 1, "COMPLETED": 3} and len(hosts.starts) == 3
    req = PL.request_bytes(GIB + -(-29 * GIB // 3), PL.default_policy())
    assert all(f"-p MemoryMax={req} " in s for _, _, s in hosts.starts)
    assert {role for role, _, _ in hosts.starts} == {"WORKER_A"}


def test_gpu_job_refused_without_allow_gpu_and_quarantined_gpu_never_chosen(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5)
    r = make(tmp_path, [job("g", 1, 2)], lambda: F.inventory(), hosts, clock).run()
    assert r["counts"] == {"UNPLACEABLE": 1} and not hosts.starts
    assert receipts(tmp_path / "dispatch")["g.attempt-1.json"]["reason"] == "GPU_JOB_REFUSED_ALLOW_GPU_IS_FALSE"

    bad = F.gpu(1, 80, 80, status="QUARANTINED_NOT_SCHEDULABLE", uuid="GPU-quarantined")
    inv = F.inventory(b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8,
                               gpus=[F.gpu(0, 12, 11.4, uuid="GPU-good-b"), bad]))
    hosts2 = FakeHosts(clock, duration=5)
    r2 = D.Dispatcher(tmp_path / "gpu", [job("g1", 1, 6, roles=["WORKER_B"]), job("g2", 1, 40, roles=["WORKER_B"])],
                      inventory_fn=lambda: inv, backend=hosts2, policy=PL.default_policy(allow_gpu=True),
                      clock=clock, sleep=clock.sleep, max_wait_seconds=0).run()
    assert all("GPU-quarantined" not in s for _, _, s in hosts2.starts)
    assert "CUDA_VISIBLE_DEVICES=GPU-good-b " in hosts2.starts[0][2]
    assert r2["jobs"]["g2"]["status"] == "UNPLACEABLE"


def test_receipts_hold_no_alias_or_home_path(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5)
    make(tmp_path, [job("w", 2)], lambda: F.inventory(), hosts, clock).run()
    t = receipts(tmp_path / "dispatch")["w.attempt-1.json"]
    assert t["role"] == "WORKER_A" and t["unit"] == D.unit_name(job("w", 2)) and t["memory_peak_bytes"] == 700 << 20
    assert t["systemd"]["Result"] == "success" and t["exit_code"] == 0
    for p in (tmp_path / "dispatch").rglob("*.json"):
        text = p.read_text()
        assert "secret-alias" not in text and str(Path.home()) not in text


def test_worker_unreachable_work_goes_elsewhere_and_pinned_work_times_out(tmp_path):
    inv = F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14, reachable=False))
    clock = Clock()
    hosts = FakeHosts(clock, duration=5)
    r = make(tmp_path, [job("free", 2), job("pinned", 2, roles=["WORKER_A"])], lambda: inv, hosts, clock,
             max_wait_seconds=60).run()
    assert [s[0] for s in hosts.starts] == ["WORKER_B"]
    assert r["jobs"]["pinned"]["status"] == "UNPLACEABLE" and "WAIT_TIMEOUT" in r["jobs"]["pinned"]["reason"]


def test_a_running_unit_on_a_role_that_stops_answering_is_polled_again_not_classified(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=30)
    state = {"polls": 0}
    real_show = hosts.show

    def flaky(role, units):
        state["polls"] += 1
        return None if 2 <= state["polls"] <= 4 else real_show(role, units)

    hosts.show = flaky
    r = make(tmp_path, [job("flaky", 1)], lambda: F.inventory(), hosts, clock).run()
    assert r["counts"] == {"COMPLETED": 1} and hosts.start_count("flaky") == 1


def test_admission_refusal_is_requeued_and_the_unit_is_started_once(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=15, start_rcs={"r0": [75]})
    r = make(tmp_path, [job(f"r{i}", 1, roles=["WORKER_B"]) for i in range(4)], lambda: F.inventory(), hosts,
             clock).run()
    assert r["counts"] == {"COMPLETED": 4}
    assert hosts.max_running["WORKER_B"] <= PL.DEFAULT_ROLE_CAPS["WORKER_B"]
    rec = receipts(tmp_path / "dispatch")
    assert rec["r0.attempt-1.json"]["status"] == "REFUSED_AT_LAUNCH" and "MemAvailable" in rec["r0.attempt-1.json"]["reason"]
    assert rec["r0.attempt-2.json"]["status"] == "COMPLETED" and hosts.start_count("r0") == 1


def test_oom_kill_maps_to_resource_exceeded_with_memory_peak_and_the_failed_unit_is_reset(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5, outcomes={"oom": ("oom",), "bad": ("exit", 3)})
    r = make(tmp_path, [job("oom", 1), job("bad", 1)], lambda: F.inventory(), hosts, clock).run()
    assert r["jobs"]["oom"]["status"] == "RESOURCE_EXCEEDED" and r["jobs"]["bad"]["status"] == "FAILED"
    t = receipts(tmp_path / "dispatch")["oom.attempt-1.json"]
    assert "OOM" in t["reason"] and t["memory_peak_bytes"] == 700 << 20 and t["systemd"]["Result"] == "oom-kill"
    assert sorted((FakeHosts.job_of(u), s) for _, u, s in hosts.releases) == [("bad", "failed"), ("oom", "failed")]


def test_crash_and_resume_reattaches_an_active_unit_and_classifies_a_finished_one(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, durations={"long": 120, "short": 3})
    jobs = [job("long", 1), job("short", 1)]

    def dying_sleep(s):
        raise Crash("dispatcher killed")

    with pytest.raises(Crash):
        make(tmp_path, jobs, lambda: F.inventory(), hosts, clock, sleep=dying_sleep).run()
    assert hosts.start_count("long") == 1 and hosts.start_count("short") == 1
    assert not list((tmp_path / "dispatch" / "receipts").glob("*.attempt-1.json"))    # no terminal yet
    clock.t += 10                                   # short has finished while no dispatcher was running

    d2 = make(tmp_path, jobs, lambda: F.inventory(), hosts, clock, resume=True)
    r = d2.run()
    assert r["final"] and r["counts"] == {"COMPLETED": 2}
    assert d2.launches == [] and r["reattached_on_resume"] == ["long"]
    assert hosts.start_count("long") == 1 and hosts.start_count("short") == 1          # nothing started twice
    rec = receipts(tmp_path / "dispatch")
    assert rec["short.attempt-1.json"]["attached_by"] == "RESUME_CLASSIFIED_STORED_RESULT"
    assert rec["short.attempt-1.json"]["memory_peak_bytes"] == 700 << 20
    assert rec["long.attempt-1.json"]["attached_by"] == "RESUME_REATTACHED_ACTIVE_UNIT"
    assert [(FakeHosts.job_of(u), s) for _, u, s in hosts.releases] == [("short", "active"), ("long", "active")]
    with pytest.raises(SystemExit):
        make(tmp_path, jobs, lambda: F.inventory(), hosts, clock, resume=True).run()     # sealed


def test_resume_after_a_crash_before_the_unit_existed_starts_it_exactly_once(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5, crash_on_start={"ghost": 1})
    with pytest.raises(Crash):
        make(tmp_path, [job("ghost", 1)], lambda: F.inventory(), hosts, clock).run()
    assert (tmp_path / "dispatch" / "receipts" / "ghost.attempt-1.launch.json").is_file()
    r = make(tmp_path, [job("ghost", 1)], lambda: F.inventory(), hosts, clock, resume=True).run()
    assert r["counts"] == {"COMPLETED": 1} and hosts.start_count("ghost") == 1
    rec = receipts(tmp_path / "dispatch")
    assert rec["ghost.attempt-1.json"]["status"] == "LAUNCH_NOT_FOUND"
    assert rec["ghost.attempt-2.json"]["status"] == "COMPLETED"


def test_a_unit_already_loaded_is_reattached_instead_of_started(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=20)
    j = job("dup", 1)
    hosts.units[("WORKER_A", D.unit_name(j))] = {"started": clock(), "duration": 20, "outcome": ("exit", 0)}
    r = make(tmp_path, [j], lambda: F.inventory(), hosts, clock).run()
    assert r["counts"] == {"COMPLETED": 1} and hosts.start_count("dup") == 0
    assert receipts(tmp_path / "dispatch")["dup.attempt-1.json"]["attached_by"] == "ALREADY_LOADED_REATTACHED"


def test_stop_file_and_resume_by_identity(tmp_path):
    stop = tmp_path / "STOP"
    clock = Clock()

    class StopAfterFirst(FakeHosts):
        def start(self, role, script):
            stop.write_text("stop")
            return super().start(role, script)

    hosts = StopAfterFirst(clock, duration=10)
    jobs = [job("s1", 1), job("s2", 1), job("s3", 1)]
    r = make(tmp_path, jobs, lambda: F.inventory(), hosts, clock, stop_file=stop).run()
    assert r["stopped"] and not r["final"] and len(hosts.starts) == 1 and r["jobs"]["s2"]["status"] == "NOT_STARTED"
    assert (tmp_path / "dispatch" / "DISPATCH_PROGRESS.1.json").is_file()
    with pytest.raises(SystemExit):
        make(tmp_path, jobs, lambda: F.inventory(), FakeHosts(clock), clock).run()     # write-once without resume
    stop.unlink()
    hosts2 = FakeHosts(clock, duration=5)
    r2 = make(tmp_path, jobs, lambda: F.inventory(), hosts2, clock, resume=True, stop_file=stop).run()
    assert r2["final"] and r2["counts"] == {"COMPLETED": 3}
    assert sorted(FakeHosts.job_of(u) for _, u, _ in hosts2.starts) == ["s2", "s3"]
    assert r2["jobs"]["s1"]["resumed_skip"] is True


def test_a_changed_estimate_is_a_new_identity_and_runs_again(tmp_path):
    clock = Clock()
    hosts = FakeHosts(clock, duration=5)
    make(tmp_path, [job("s1", 1), job("s2", 1)], lambda: F.inventory(), hosts, clock).run()
    (tmp_path / "dispatch" / "DISPATCH_RECEIPT.json").unlink()     # test only: reopen the sealed root
    hosts2 = FakeHosts(clock, duration=5)
    r = make(tmp_path, [job("s1", 2), job("s2", 1)], lambda: F.inventory(), hosts2, clock, resume=True).run()
    assert [FakeHosts.job_of(u) for _, u, _ in hosts2.starts] == ["s1"]
    assert r["jobs"]["s1"]["attempt"] == 2 and r["jobs"]["s2"]["resumed_skip"] is True
    assert hosts2.starts[0][1] != hosts.starts[0][1]                                 # new identity, new unit


def test_dry_run_plan_starts_nothing():
    rows = PL.plan([job("p1", 8), job("p2", 8), job("p3", 1, 2)], F.inventory())
    assert [r["decision"] for r in rows] == ["PLACED", "WAIT", "UNPLACEABLE"] and rows[0]["role"] == "WORKER_A"
