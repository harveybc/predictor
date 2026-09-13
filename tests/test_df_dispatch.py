"""C174: the dispatcher re-reads the live inventory before every launch, launches only through
an injected launcher (a fake here: nothing is started), writes write-once receipts, splits,
refuses, caps concurrency, stops on a stop file and resumes by identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import df_capacity_fakes as F  # noqa: E402
import df_dispatch as D  # noqa: E402
import df_placement as PL  # noqa: E402

GIB = F.GIB
ROLES_MAP = {"COORDINATOR": {"ssh": None}, "WORKER_A": {"ssh": "secret-alias-a"}, "WORKER_B": {"ssh": "secret-alias-b"}}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Handle:
    def __init__(self, polls, rc):
        self.left, self.rc = polls, rc

    def poll(self):
        if self.left <= 0:
            return self.rc
        self.left -= 1
        return None


class Launcher:
    def __init__(self, polls=2, rc=0, rcs=None):
        self.calls, self.polls, self.rc, self.rcs = [], polls, rc, dict(rcs or {})
        self.max_running = {}
        self.handles = []

    def __call__(self, role, command, log_path):
        self.calls.append((role, command))
        name = command["display"]
        rc = self.rc
        for k, v in list(self.rcs.items()):
            if k in name and v:
                rc = v.pop(0)
        h = Handle(self.polls, rc)
        h.role = role
        self.handles.append(h)
        live = sum(1 for x in self.handles if x.role == role and x.left > 0)
        self.max_running[role] = max(self.max_running.get(role, 0), live)
        return h


def job(jid, cpu_gib=1.0, gpu_gib=0.0, **kw):
    return {"job_id": jid, "argv": ["~/anaconda3/envs/trading-stack/bin/python", "tools/x.py", jid],
            "cpu_bytes": int(cpu_gib * GIB), "gpu_bytes": int(gpu_gib * GIB), "cpus": 1, "wall": "30m", **kw}


def make(tmp_path, jobs, inventory_fn, launcher, clock=None, **kw):
    clock = clock or Clock()
    return D.Dispatcher(tmp_path / "dispatch", jobs, inventory_fn=inventory_fn, launcher=launcher,
                        roles_map=ROLES_MAP, clock=clock, sleep=clock.sleep, poll_seconds=5,
                        max_inventory_age_seconds=2, **kw)


def receipts(root):
    return {p.name: json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))}


def test_coordinator_memory_drop_between_launches_is_seen_before_the_second_launch(tmp_path):
    reads = []

    def inv():
        # read 1: plenty; from read 2 on another agent holds COORDINATOR memory; read 6+: released
        n = len(reads) + 1
        avail = 22 if n == 1 or n >= 6 else 6
        doc = F.inventory(coord=F.role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=avail, slice_gib=14),
                          stamp=f"read-{n}")
        reads.append(doc)
        return doc

    L = Launcher(polls=100)
    d = make(tmp_path, [job("a", 4, roles=["COORDINATOR"]), job("b", 4, roles=["COORDINATOR"])], inv, L)
    r = d.run()
    assert r["final"] and r["counts"] == {"COMPLETED": 2}
    (ja, _, dig_a), (jb, _, dig_b) = d.launches
    assert (ja, jb) == ("a", "b") and dig_a != dig_b
    rec = receipts(tmp_path / "dispatch")
    launch_b = rec["b.attempt-1.launch.json"]
    used = next(x for x in reads if x["digest_sha256"] == launch_b["inventory_digest"])
    assert used["roles"]["COORDINATOR"]["mem_available_bytes"] == 22 * GIB   # launched only after memory returned
    assert d.inventory_reads >= 6
    assert (tmp_path / "dispatch" / "inventories" / f"{dig_b}.json").is_file()


def test_every_launch_uses_a_reading_taken_after_the_previous_launch(tmp_path):
    count = {"n": 0}

    def inv():
        count["n"] += 1
        return F.inventory(stamp=f"r{count['n']}")

    L = Launcher(polls=50)
    d = make(tmp_path, [job(f"j{i}", 1) for i in range(4)], inv, L)
    d.run()
    digests = [x[2] for x in d.launches]
    assert len(digests) == 4 and len(set(digests)) == 4


def test_a_job_too_big_for_every_role_is_split_and_its_chunks_run(tmp_path):
    big = job("big", 30, split={"max_chunks": 6, "fixed_cpu_bytes": GIB})
    L = Launcher(polls=1)
    d = make(tmp_path, [big], lambda: F.inventory(), L)
    r = d.run()
    rec = receipts(tmp_path / "dispatch")
    assert rec["big.attempt-1.json"]["status"] == "SPLIT" and rec["big.attempt-1.json"]["n_chunks"] == 3
    assert r["counts"] == {"SPLIT": 1, "COMPLETED": 3} and len(L.calls) == 3
    # chunk peak = 1G + ceil(29G / 3); request = that * 1.25 rounded up to a MiB, under the 14G slice
    req = PL.request_bytes(GIB + -(-29 * GIB // 3), PL.default_policy())
    assert req <= 14 * GIB
    assert all(f"-m {req // PL.MIB}M" in cmd["display"] for _, cmd in L.calls)
    assert {role for role, _ in L.calls} <= {"COORDINATOR", "WORKER_A"}


def test_gpu_job_refused_without_allow_gpu_and_quarantined_gpu_never_chosen(tmp_path):
    L = Launcher(polls=1)
    d = make(tmp_path, [job("g", 1, 2)], lambda: F.inventory(), L)
    r = d.run()
    assert r["counts"] == {"UNPLACEABLE": 1} and not L.calls
    assert receipts(tmp_path / "dispatch")["g.attempt-1.json"]["reason"] == "GPU_JOB_REFUSED_ALLOW_GPU_IS_FALSE"

    bad = F.gpu(1, 80, 80, status="QUARANTINED_NOT_SCHEDULABLE", uuid="GPU-quarantined")
    inv = F.inventory(b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8,
                               gpus=[F.gpu(0, 12, 11.4, uuid="GPU-good-b"), bad]))
    L2 = Launcher(polls=1)
    d2 = D.Dispatcher(tmp_path / "gpu", [job("g1", 1, 6, roles=["WORKER_B"]), job("g2", 1, 40, roles=["WORKER_B"])],
                      inventory_fn=lambda: inv, launcher=L2, roles_map=ROLES_MAP,
                      policy=PL.default_policy(allow_gpu=True), clock=Clock(), sleep=lambda s: None,
                      max_wait_seconds=0)
    r2 = d2.run()
    assert all("GPU-quarantined" not in c["display"] for _, c in L2.calls)
    assert "CUDA_VISIBLE_DEVICES=GPU-good-b" in L2.calls[0][1]["display"]
    assert r2["jobs"]["g2"]["status"] == "UNPLACEABLE"


def test_cpu_job_command_hides_gpus_and_remote_command_has_no_alias_or_home(tmp_path):
    L = Launcher(polls=1)
    d = make(tmp_path, [job("w", 2)], lambda: F.inventory(), L)
    d.run()
    role, cmd = L.calls[0]
    assert role == "WORKER_A" and cmd["ssh_argv"][5] == "secret-alias-a"
    assert "CUDA_VISIBLE_DEVICES=" in cmd["display"] and "CUDA_VISIBLE_DEVICES=GPU" not in cmd["display"]
    assert '"$HOME"/.local/bin/crispdm-run -m 2560M -t 30m' in cmd["display"]
    assert '"$HOME"/anaconda3/envs/trading-stack/bin/python' in cmd["display"]
    for p in (tmp_path / "dispatch").rglob("*.json"):
        t = p.read_text()
        assert "secret-alias" not in t and str(Path.home()) not in t


def test_worker_unreachable_work_goes_elsewhere_and_pinned_work_times_out(tmp_path):
    inv = F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14, reachable=False))
    L = Launcher(polls=1)
    d = make(tmp_path, [job("free", 2), job("pinned", 2, roles=["WORKER_A"])], lambda: inv, L,
             max_wait_seconds=60)
    r = d.run()
    assert [c[0] for c in L.calls] == ["COORDINATOR"]
    assert r["jobs"]["pinned"]["status"] == "UNPLACEABLE" and "WAIT_TIMEOUT" in r["jobs"]["pinned"]["reason"]


def test_launch_refusal_is_requeued_and_concurrency_cap_is_never_exceeded(tmp_path):
    L = Launcher(polls=3, rcs={"tools/x.py r0": [75]})
    jobs = [job(f"r{i}", 1, roles=["WORKER_B"]) for i in range(5)]
    d = make(tmp_path, jobs, lambda: F.inventory(), L)
    r = d.run()
    assert r["counts"] == {"COMPLETED": 5}
    assert L.max_running["WORKER_B"] <= PL.DEFAULT_ROLE_CAPS["WORKER_B"]
    rec = receipts(tmp_path / "dispatch")
    assert rec["r0.attempt-1.json"]["status"] == "REFUSED_AT_LAUNCH" and rec["r0.attempt-2.json"]["status"] == "COMPLETED"


def test_exit_codes_map_to_terminal_statuses(tmp_path):
    assert D.classify_exit(137)[0] == "RESOURCE_EXCEEDED" and D.classify_exit(124)[0] == "RESOURCE_EXCEEDED"
    assert D.classify_exit(255)[0] == "UNCERTAIN" and D.classify_exit(1)[0] == "FAILED"
    L = Launcher(polls=1, rcs={"x.py oom": [137]})
    d = make(tmp_path, [job("oom", 1)], lambda: F.inventory(), L)
    assert d.run()["jobs"]["oom"]["status"] == "RESOURCE_EXCEEDED"


def test_observed_peak_is_sampled_from_the_job_scope(tmp_path):
    j = job("obs", 1, roles=["WORKER_A"])
    prefix = D.scope_prefix(j)
    state = {"n": 0}

    def inv():
        state["n"] += 1
        scopes = [{"unit": prefix + "1-2.scope", "memory_current_bytes": GIB // 2, "memory_peak_bytes": 700 << 20}]
        return F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14,
                                    scopes=scopes if state["n"] > 1 else ()), stamp=str(state["n"]))

    L = Launcher(polls=20)
    d = make(tmp_path, [j], inv, L, observe_seconds=10)
    d.run()
    t = receipts(tmp_path / "dispatch")["obs.attempt-1.json"]
    assert t["observed_peak_bytes_sampled"] == 700 << 20 and t["request_bytes"] == 1280 << 20


def test_stop_file_and_resume_by_identity(tmp_path):
    stop = tmp_path / "STOP"
    L = Launcher(polls=2)

    class StopAfterFirst(Launcher):
        def __call__(self, role, command, log_path):
            stop.write_text("stop")
            return super().__call__(role, command, log_path)

    L = StopAfterFirst(polls=2)
    jobs = [job("s1", 1), job("s2", 1), job("s3", 1)]
    d = make(tmp_path, jobs, lambda: F.inventory(), L, stop_file=stop)
    r = d.run()
    assert r["stopped"] and not r["final"] and len(L.calls) == 1
    assert r["jobs"]["s2"]["status"] == "NOT_STARTED"
    assert (tmp_path / "dispatch" / "DISPATCH_PROGRESS.1.json").is_file()

    with pytest.raises(SystemExit):
        make(tmp_path, jobs, lambda: F.inventory(), Launcher()).run()      # write-once without --resume
    stop.unlink()
    L2 = Launcher(polls=1)
    r2 = make(tmp_path, jobs, lambda: F.inventory(), L2, resume=True, stop_file=stop).run()
    assert r2["final"] and r2["counts"] == {"COMPLETED": 3}
    assert sorted(c[1]["display"].split()[-1] for c in L2.calls) == ["s2", "s3"]
    assert r2["jobs"]["s1"]["resumed_skip"] is True

    with pytest.raises(SystemExit):     # the root is sealed by its final receipt
        make(tmp_path, jobs, lambda: F.inventory(), Launcher(), resume=True).run()


def test_a_changed_estimate_is_a_new_identity_and_runs_again(tmp_path):
    stop = tmp_path / "STOP"
    L = Launcher(polls=1)
    make(tmp_path, [job("s1", 1), job("s2", 1)], lambda: F.inventory(), L, stop_file=stop,
         clock=Clock()).run()
    # simulate an unfinished root: remove the final seal only for this test
    (tmp_path / "dispatch" / "DISPATCH_RECEIPT.json").unlink()
    L2 = Launcher(polls=1)
    r = make(tmp_path, [job("s1", 2), job("s2", 1)], lambda: F.inventory(), L2, resume=True).run()
    assert [c[1]["display"].split()[-1] for c in L2.calls] == ["s1"]
    assert r["jobs"]["s1"]["attempt"] == 2 and r["jobs"]["s2"]["resumed_skip"] is True


def test_launch_without_terminal_is_uncertain_on_resume_and_not_rerun_by_default(tmp_path):
    jobs = [job("lost", 1)]

    class Crash(Exception):
        pass

    def crashing_inv():
        crashing_inv.n += 1
        if crashing_inv.n > 1:
            raise Crash()
        return F.inventory()
    crashing_inv.n = 0

    L = Launcher(polls=100)
    d = make(tmp_path, jobs + [job("next", 1)], crashing_inv, L)
    with pytest.raises(Crash):
        d.run()
    assert (tmp_path / "dispatch" / "receipts" / "lost.attempt-1.launch.json").is_file()
    L2 = Launcher(polls=1)
    r = make(tmp_path, jobs + [job("next", 1)], lambda: F.inventory(), L2, resume=True).run()
    assert r["jobs"]["lost"]["status"] == "UNCERTAIN" and [c[1]["display"].split()[-1] for c in L2.calls] == ["next"]
    assert receipts(tmp_path / "dispatch")["lost.attempt-1.json"]["reason"] == "DISPATCHER_RESTARTED_WITHOUT_TERMINAL"


def test_dry_run_plan_launches_nothing():
    rows = PL.plan([job("p1", 8), job("p2", 8), job("p3", 1, 2)], F.inventory())
    assert [r["decision"] for r in rows] == ["PLACED", "PLACED", "UNPLACEABLE"]
