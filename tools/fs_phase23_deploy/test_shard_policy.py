import json
import math

import pytest

import shard_policy as sp


def health(role, cores=32, load=0.0, headroom=12 * sp.GIB):
    return {"role": role, "cpu": {"cores": cores, "loadavg_1_5_15": [load, 0, 0]},
            "admission": {"host_free_for_new_bytes": headroom}}


def plan(costs):
    return {"plan_digest": "abc", "shards": [{"shard_id": f"s{i:03d}", "est_cost": c} for i, c in enumerate(costs)]}


HEALTH = {"coordinator": health("coordinator", 16, 1.0, 17 * sp.GIB),
          "worker_a": health("worker_a", 32, 0.3, 6 * sp.GIB),
          "worker_b": health("worker_b", 32, 0.2, 12 * sp.GIB)}


def test_coordinator_gets_cheapest_third_by_count_and_small_cost_share():
    p = plan([100, 1, 50, 2, 75, 3, 60, 4, 90])
    a = sp.assign(p, HEALTH)
    coord = a["roles"]["coordinator"]
    assert coord["shard_count"] == math.ceil(9 / 3)
    assert coord["shards"] == ["s001", "s003", "s005"]  # costs 1, 2, 3
    assert coord["cost_share"] < 1 / 3
    assert coord["slots"] == 1 and coord["cap_bytes"] == sp.GIB and coord["nice"] == 19


def test_exclusive_and_complete():
    p = plan([float(i) for i in range(1, 41)])
    a = sp.assign(p, HEALTH)
    assert sp.check_exclusive(a, p) == []
    total = sum(len(a["roles"][r]["shards"]) for r in sp.ROLES)
    assert total == 40


def test_workers_weighted_by_slots():
    p = plan([10.0] * 30)
    a = sp.assign(p, HEALTH)
    wa, wb = a["roles"]["worker_a"], a["roles"]["worker_b"]
    # worker_a: 6 GiB headroom * 0.5 / 2 GiB = 1 slot; worker_b: 12*0.5/2 = 3 slots
    assert wa["slots"] == 1 and wb["slots"] == 3
    assert wb["cost"] > wa["cost"]
    assert wa["cost"] + wb["cost"] + a["roles"]["coordinator"]["cost"] == pytest.approx(300.0)


def test_slots_bounded_by_cpu_and_headroom():
    h = health("worker_a", cores=4, load=3.5, headroom=64 * sp.GIB)
    s = sp.worker_slots(h, 2 * sp.GIB, 2, 0.5, 8)
    assert s["slots"] == 0  # 0.5 free cores // 2 threads
    h = health("worker_a", cores=32, load=0.0, headroom=3 * sp.GIB)
    s = sp.worker_slots(h, 2 * sp.GIB, 2, 0.5, 8)
    assert s["slots"] == 0  # 1.5 GiB safe headroom < one 2 GiB cap
    h = health("worker_a", cores=32, load=0.0, headroom=100 * sp.GIB)
    assert sp.worker_slots(h, 2 * sp.GIB, 2, 0.5, 8)["slots"] == 8


def test_no_admission_monitor_means_no_slots():
    h = {"role": "worker_b", "cpu": {"cores": 32, "loadavg_1_5_15": [0, 0, 0]}, "admission": None}
    assert sp.worker_slots(h, sp.GIB, 1, 0.5, 8)["slots"] == 0


def test_no_worker_headroom_refuses():
    hh = dict(HEALTH, worker_a=health("worker_a", 32, 0, 0), worker_b=health("worker_b", 32, 0, 0))
    with pytest.raises(ValueError):
        sp.assign(plan([1, 2, 3, 4, 5, 6]), hh)


def test_deterministic_regardless_of_input_order():
    costs = [5, 3, 9, 1, 7, 2, 8, 4, 6]
    p1 = plan(costs)
    p2 = {"shards": list(reversed(p1["shards"]))}
    assert sp.assign(p1, HEALTH)["roles"] == sp.assign(p2, HEALTH)["roles"]


def test_pair_count_is_cost_when_no_estimate():
    p = {"shards": [{"shard_id": "a", "pair_count": 10}, {"shard_id": "b", "pair_count": 1}, {"shard_id": "c", "pair_count": 5}]}
    a = sp.assign(p, HEALTH)
    assert a["roles"]["coordinator"]["shards"] == ["b"]


def test_duplicate_shard_rejected():
    with pytest.raises(ValueError):
        sp.load_shards({"shards": [{"shard_id": "x", "est_cost": 1}, {"shard_id": "x", "est_cost": 2}]})


def test_cli_writes_assignment(tmp_path):
    p = plan([1, 2, 3, 4, 5, 6])
    (tmp_path / "plan.json").write_text(json.dumps(p))
    for r, h in HEALTH.items():
        (tmp_path / f"{r}.json").write_text(json.dumps(h))
    out = tmp_path / "assignment.json"
    rc = sp.main(["--plan", str(tmp_path / "plan.json"), "--out", str(out)] + [f"--health={r}={tmp_path / (r + '.json')}" for r in HEALTH])
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == sp.SCHEMA and doc["exclusive"] is True
