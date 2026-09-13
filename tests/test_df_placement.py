"""C174: placement fits RAM under min(slice headroom, MemAvailable - reserve) with a margin,
VRAM on schedulable GPUs only, prefers workers over the COORDINATOR, is deterministic,
splits what fits nowhere and refuses the rest with a reason per role."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import df_capacity_fakes as F  # noqa: E402
import df_placement as PL  # noqa: E402

GIB = F.GIB
POL = PL.default_policy()


def job(jid="j", cpu_gib=1.0, gpu_gib=0.0, cpus=1, **kw):
    return {"job_id": jid, "argv": ["true"], "cpu_bytes": int(cpu_gib * GIB), "gpu_bytes": int(gpu_gib * GIB),
            "cpus": cpus, **kw}


def worker(name, avail_gib, slice_gib=14, **kw):
    total = 14.3 if name == "WORKER_B" else 30.6
    return F.role(name, cpus=32, total_gib=total, avail_gib=avail_gib, slice_gib=slice_gib, **kw)


def test_request_carries_the_declared_margin_below_memoryhigh():
    r = PL.request_bytes(4 * GIB, POL)
    assert r == 5 * GIB and r % PL.MIB == 0
    assert 4 * GIB < 0.9 * r          # the estimate sits below MemoryHigh (90% of MemoryMax)


def test_equal_headroom_picks_a_worker():
    # WORKER_A identical to the COORDINATOR in every number: the worker wins the tie
    same = F.inventory(a=F.role("WORKER_A", cpus=16, total_gib=30.5, avail_gib=22, slice_gib=14))
    assert [PL.place(job(cpu_gib=2), same, [], POL)["role"] for _ in range(3)] == ["WORKER_A"] * 3
    # even when the COORDINATOR has more headroom than any worker
    rich = F.inventory(a=worker("WORKER_A", 9), b=worker("WORKER_B", 9, slice_gib=8))
    d = PL.place(job(cpu_gib=2), rich, [], POL)
    assert d["decision"] == "PLACED" and d["role"] in PL.WORKERS
    assert PL.place(job(cpu_gib=2), rich, [], POL)["role"] == d["role"]          # deterministic


def test_coordinator_is_chosen_only_when_no_worker_fits():
    busy = F.inventory(a=worker("WORKER_A", 6), b=worker("WORKER_B", 6, slice_gib=8))   # 1G headroom each
    d = PL.place(job(cpu_gib=2), busy, [], POL)
    assert d["decision"] == "PLACED" and d["role"] == "COORDINATOR"
    assert "OVER_HEADROOM" in d["per_role"]["WORKER_A"] and "OVER_HEADROOM" in d["per_role"]["WORKER_B"]
    down = F.inventory(a=worker("WORKER_A", 20, reachable=False), b=worker("WORKER_B", 11, reachable=False))
    assert PL.place(job(cpu_gib=2), down, [], POL)["role"] == "COORDINATOR"


def test_coordinator_not_used_while_a_worker_waits_for_this_dispatchers_own_jobs():
    inv = F.inventory()
    run = [{"role": "WORKER_A", "request_bytes": 13 * GIB, "cpus": 1, "observed_current_bytes": 0},
           {"role": "WORKER_B", "request_bytes": 5 * GIB, "cpus": 1, "observed_current_bytes": 0}]
    d = PL.place(job(cpu_gib=2), inv, run, POL)
    assert d["decision"] == "WAIT" and d["reason"].startswith("WORKERS_FIRST") and "WORKER_A" in d["reason"]
    capped = [{"role": "WORKER_A", "request_bytes": 1, "cpus": 1, "observed_current_bytes": 1}] * 3 + \
             [{"role": "WORKER_B", "request_bytes": 1, "cpus": 1, "observed_current_bytes": 1}] * 2
    assert PL.place(job(cpu_gib=1), inv, capped, POL)["decision"] == "WAIT"


def test_coordinator_reserve_fraction_of_free_memory():
    down = F.inventory(a=worker("WORKER_A", 20, reachable=False), b=worker("WORKER_B", 11, reachable=False))
    # COORDINATOR: 22G available -> at most 0.25 * 22 = 5.5G per request
    ok = PL.place(job(cpu_gib=4), down, [], POL)                     # request 5G
    assert ok["role"] == "COORDINATOR"
    big = PL.place(job(cpu_gib=6), down, [], POL)                    # request 7.5G
    assert big["decision"] == "WAIT" and "RAM_REQUEST" in big["per_role"]["COORDINATOR"]
    loose = PL.default_policy(coordinator_max_fraction=0.9)
    assert PL.place(job(cpu_gib=6), down, [], loose)["role"] == "COORDINATOR"
    # own reservations on the COORDINATOR shrink its free memory for the fraction too
    run = [{"role": "COORDINATOR", "request_bytes": 4 * GIB, "cpus": 1, "observed_current_bytes": 0}]
    d = PL.place(job(cpu_gib=4), down, run, POL)
    assert d["decision"] == "WAIT" and "COORDINATOR_RESERVE" in d["per_role"]["COORDINATOR"]


def test_ram_must_fit_both_slice_headroom_and_host_available_minus_reserve():
    # host: 9G available - 5G reserve = 4G; slice has 14G. A 3.5G estimate requests 4.375G: no.
    inv = F.inventory(coord=F.role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=9, slice_gib=14),
                      a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14, slice_cur_gib=11),
                      b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8, slice_cur_gib=6))
    d = PL.place(job(cpu_gib=3.5), inv, [], POL)
    assert d["decision"] == "WAIT"
    assert "host_4294967296" in d["per_role"]["COORDINATOR"] and "slice_3221225472" in d["per_role"]["WORKER_A"]
    assert "slice_2147483648" in d["per_role"]["WORKER_B"]
    assert PL.place(job(cpu_gib=2), inv, [], POL)["role"] == "WORKER_A"      # slice headroom 3G >= 2.5G


def test_running_reservations_count_until_observed_and_caps_are_enforced():
    inv = F.inventory()
    run = [{"role": "WORKER_A", "request_bytes": 12 * GIB, "cpus": 1, "observed_current_bytes": 0}]
    assert PL.place(job(cpu_gib=2), inv, run, POL)["role"] == "WORKER_B"
    run[0]["observed_current_bytes"] = 12 * GIB        # already visible in the live reading: not counted twice
    assert PL.place(job(cpu_gib=2), inv, run, POL)["role"] == "WORKER_A"
    capped = [{"role": "WORKER_A", "request_bytes": 1, "cpus": 1, "observed_current_bytes": 1}] * 3
    d = PL.place(job(cpu_gib=1, roles=["WORKER_A"]), inv, capped, POL)
    assert d["decision"] == "WAIT" and "CONCURRENCY_CAP_3" in d["per_role"]["WORKER_A"]


def test_gpu_job_is_refused_without_allow_gpu():
    d = PL.place(job(cpu_gib=1, gpu_gib=2), F.inventory(), [], POL)
    assert d["decision"] == "UNPLACEABLE" and d["reason"] == "GPU_JOB_REFUSED_ALLOW_GPU_IS_FALSE"


def test_gpu_placement_with_allow_gpu_never_uses_a_quarantined_gpu():
    pol = PL.default_policy(allow_gpu=True)
    d = PL.place(job(cpu_gib=1, gpu_gib=10), F.inventory(), [], pol)
    assert d["decision"] == "PLACED" and d["role"] == "WORKER_A" and d["gpu"]["index"] == 0   # most free VRAM
    bad = dict(F.gpu(1, 80, 80, status="QUARANTINED_NOT_SCHEDULABLE"))
    inv2 = F.inventory(b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8,
                                gpus=[F.gpu(0, 12, 11.4), bad]))
    for gi in (None, 1):
        extra = {} if gi is None else {"gpu_index": gi}
        d2 = PL.place(job(cpu_gib=1, gpu_gib=40, roles=["WORKER_B"], **extra), inv2, [], pol)
        assert d2["decision"] != "PLACED" and d2["gpu"] is None
    assert PL.place(job(cpu_gib=1, gpu_gib=6, roles=["WORKER_B"]), inv2, [], pol)["gpu"]["index"] == 0


def test_gpu_with_a_compute_process_or_reservation_is_not_shared_by_default():
    pol = PL.default_policy(allow_gpu=True)
    inv = F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14,
                               gpus=[F.gpu(0, 16, 15, procs=1)]))
    assert PL.place(job(cpu_gib=1, gpu_gib=5, roles=["WORKER_A"]), inv, [], pol)["decision"] == "WAIT"
    assert PL.place(job(cpu_gib=1, gpu_gib=5, roles=["WORKER_A"]), inv, [],
                    PL.default_policy(allow_gpu=True, exclusive_gpu=False))["decision"] == "PLACED"


def test_too_big_for_every_role_is_split_into_the_fewest_chunks_else_refused():
    big = job("big", cpu_gib=30, split={"max_chunks": 8, "fixed_cpu_bytes": GIB},
              argv=["python", "x.py", "--part", "{chunk_index}", "--of", "{n_chunks}"])
    d = PL.place(big, F.inventory(), [], POL)
    # largest ceiling is WORKER_A's 14G slice -> 1.25 * (1 + 29/n) G <= 14 -> n = 3
    assert d["decision"] == "SPLIT" and d["n_chunks"] == 3
    assert [c["argv"][3] for c in d["chunks"]] == ["0", "1", "2"] and d["chunks"][0]["argv"][5] == "3"
    assert all(PL.request_bytes(c["cpu_bytes"], POL) <= 14 * GIB for c in d["chunks"])
    r = PL.place(job("solid", cpu_gib=30), F.inventory(), [], POL)
    assert r["decision"] == "UNPLACEABLE" and "NOT_SPLITTABLE" in r["reason"]
    assert all("OVER_CEILING" in r["per_role"][x] for x in PL.ROLES)


def test_unreachable_and_unguarded_roles_are_never_chosen():
    inv = F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14, reachable=False),
                      coord=F.role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=22, slice_gib=14, memguard=False))
    d = PL.place(job(cpu_gib=2), inv, [], POL)
    assert d["role"] == "WORKER_B"
    assert d["per_role"]["WORKER_A"].startswith("UNREACHABLE") and d["per_role"]["COORDINATOR"] == "MEMGUARD_NOT_ACTIVE"
    only_a = PL.place(job(cpu_gib=2, roles=["WORKER_A"]), inv, [], POL)
    assert only_a["decision"] == "WAIT" and "UNREACHABLE_ROLES_MAY_FIT: WORKER_A" in only_a["reason"]


def test_plan_reserves_each_placed_job_in_order_and_keeps_work_off_the_coordinator():
    rows = PL.plan([job(f"j{i}", cpu_gib=8) for i in range(4)], F.inventory(), POL)
    assert rows[0]["decision"] == "PLACED" and rows[0]["role"] == "WORKER_A"
    assert [r["decision"] for r in rows[1:]] == ["WAIT", "WAIT", "WAIT"]
    small = PL.plan([job(f"s{i}", cpu_gib=1) for i in range(6)], F.inventory(), POL)
    assert [r["role"] for r in small] == ["WORKER_A", "WORKER_A", "WORKER_A", "WORKER_B", "WORKER_B", None]
