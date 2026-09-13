"""C174: placement fits RAM under min(slice headroom, MemAvailable - reserve) with a margin,
VRAM on schedulable GPUs only, is deterministic, splits what fits nowhere and refuses
the rest with a reason per role."""
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


def test_request_carries_the_declared_margin_below_memoryhigh():
    r = PL.request_bytes(4 * GIB, POL)
    assert r == 5 * GIB and r % PL.MIB == 0
    assert 4 * GIB < 0.9 * r          # the estimate sits below crispdm-run's MemoryHigh (90% of MemoryMax)


def test_cpu_job_goes_to_most_headroom_then_cpus_then_role_order_deterministically():
    inv = F.inventory()
    d = PL.place(job(cpu_gib=2), inv, [], POL)
    # COORDINATOR and WORKER_A both have slice headroom 14G (the binding term); WORKER_A has more free CPUs
    assert d["decision"] == "PLACED" and d["role"] == "WORKER_A" and d["gpu"] is None
    assert [PL.place(job(cpu_gib=2), inv, [], POL)["role"] for _ in range(5)] == ["WORKER_A"] * 5
    same = F.inventory(a=F.role("WORKER_A", cpus=16, total_gib=30.5, avail_gib=22, slice_gib=14))
    assert PL.place(job(cpu_gib=2), same, [], POL)["role"] == "COORDINATOR"   # full tie -> role order


def test_ram_must_fit_both_slice_headroom_and_host_available_minus_reserve():
    # host: 9G available - 5G reserve = 4G; slice has 14G. A 3.5G estimate requests 4.375G: no.
    inv = F.inventory(coord=F.role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=9, slice_gib=14),
                      a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14, slice_cur_gib=11),
                      b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8, slice_cur_gib=6))
    d = PL.place(job(cpu_gib=3.5, roles=["COORDINATOR", "WORKER_A", "WORKER_B"]), inv, [], POL)
    assert d["decision"] == "WAIT"
    assert "host_" in d["per_role"]["COORDINATOR"] and "slice_" in d["per_role"]["WORKER_A"]
    assert PL.place(job(cpu_gib=2), inv, [], POL)["role"] == "COORDINATOR"


def test_running_reservations_count_until_observed_and_caps_are_enforced():
    inv = F.inventory()
    run = [{"role": "WORKER_A", "request_bytes": 12 * GIB, "cpus": 1, "observed_current_bytes": 0}]
    assert PL.place(job(cpu_gib=2), inv, run, POL)["role"] == "COORDINATOR"
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
    inv = F.inventory()
    d = PL.place(job(cpu_gib=1, gpu_gib=10), inv, [], pol)
    assert d["decision"] == "PLACED" and d["role"] == "WORKER_A" and d["gpu"]["index"] == 0   # most free VRAM
    # a quarantined GPU with a huge free VRAM reading is still never chosen
    bad = dict(F.gpu(1, 80, 80, status="QUARANTINED_NOT_SCHEDULABLE"))
    inv2 = F.inventory(b=F.role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8,
                                gpus=[F.gpu(0, 12, 11.4), bad]))
    for gi in (None, 1):
        extra = {} if gi is None else {"gpu_index": gi}
        d2 = PL.place(job(cpu_gib=1, gpu_gib=40, roles=["WORKER_B"], **extra), inv2, [], pol)
        assert d2["decision"] != "PLACED" and (d2["gpu"] is None)
    d3 = PL.place(job(cpu_gib=1, gpu_gib=6, roles=["WORKER_B"]), inv2, [], pol)
    assert d3["gpu"]["index"] == 0


def test_gpu_with_a_compute_process_or_reservation_is_not_shared_by_default():
    pol = PL.default_policy(allow_gpu=True)
    inv = F.inventory(a=F.role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14,
                               gpus=[F.gpu(0, 16, 15, procs=1)]))
    d = PL.place(job(cpu_gib=1, gpu_gib=5, roles=["WORKER_A"]), inv, [], pol)
    assert d["decision"] == "WAIT"
    assert PL.place(job(cpu_gib=1, gpu_gib=5, roles=["WORKER_A"]), inv, [],
                    PL.default_policy(allow_gpu=True, exclusive_gpu=False))["decision"] == "PLACED"


def test_too_big_for_every_role_is_split_into_the_fewest_chunks_else_refused():
    big = job("big", cpu_gib=30, split={"max_chunks": 8, "fixed_cpu_bytes": GIB, "argv_note": 1},
              argv=["python", "x.py", "--part", "{chunk_index}", "--of", "{n_chunks}"])
    d = PL.place(big, F.inventory(), [], POL)
    # ceiling is 14G slice -> request = 1.25 * (1 + 29/n) G <= 14 -> n = 3 (1+29/3 = 10.67 -> 13.34G)
    assert d["decision"] == "SPLIT" and d["n_chunks"] == 3
    assert [c["argv"][3] for c in d["chunks"]] == ["0", "1", "2"] and d["chunks"][0]["argv"][5] == "3"
    assert all(PL.request_bytes(c["cpu_bytes"], POL) <= 14 * GIB for c in d["chunks"])
    solid = job("solid", cpu_gib=30)
    r = PL.place(solid, F.inventory(), [], POL)
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


def test_plan_reserves_each_placed_job_in_order():
    rows = PL.plan([job(f"j{i}", cpu_gib=8) for i in range(5)], F.inventory(), POL)
    assert [r["role"] for r in rows[:2]] == ["WORKER_A", "COORDINATOR"]
    assert rows[-1]["decision"] == "WAIT"
