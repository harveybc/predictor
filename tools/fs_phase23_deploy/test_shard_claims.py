import datetime as dt
import hashlib
import json

import pytest

import shard_claims as sc


PLAN = {"shards": [{"shard_id": f"s{i}", "est_cost": i + 1} for i in range(6)]}
ASSIGN = {"roles": {"coordinator": {"shards": ["s0", "s1"]}, "worker_a": {"shards": ["s2", "s3"]}, "worker_b": {"shards": ["s4", "s5"]}}}


def host(tmp_path, role):
    d = tmp_path / role
    d.mkdir(exist_ok=True)
    return sc.State(d, role)


def write_terminal(root, shard, body=b"row\n", valid=True):
    d = root / shard
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest() if valid else "0" * 64
    (d / "run_manifest.json").write_text(json.dumps({"status": "COMPLETED", "results_sha256": digest}))


def heartbeat(state, cycle):
    sc.atomic_write_json(state.heartbeat, {"cycle": cycle, "relayed_at_utc": sc.utc_now()})


def relay(src, dst, shard):
    """Simulate the coordinator loop: copy src's own claim for shard into dst's peer_claims."""
    c = sc.read_json(src.own_claim_path(shard))
    sc.atomic_write_json(dst.peer_claims / shard / f"claim.{src.role}.json", c)


def test_claims_own_shards_in_plan_order_and_slots_are_exclusive(tmp_path):
    a = host(tmp_path, "worker_a")
    r1 = sc.next_shard(a, PLAN, ASSIGN, slot="1")
    r2 = sc.next_shard(a, PLAN, ASSIGN, slot="2")
    assert (r1["shard_id"], r2["shard_id"]) == ("s2", "s3")
    assert sc.next_shard(a, PLAN, ASSIGN, slot="3")["action"] == "NONE"


def test_own_shard_settles_immediately_without_heartbeat(tmp_path):
    a = host(tmp_path, "worker_a")
    sc.next_shard(a, PLAN, ASSIGN, slot="1")
    assert sc.settle(a, "s2")["action"] == "WIN"


def test_valid_terminal_is_never_reclaimed_and_invalid_one_is(tmp_path):
    a = host(tmp_path, "worker_a")
    write_terminal(a.terminals, "s2")
    write_terminal(a.peer_terminals, "s3", valid=False)
    r = sc.next_shard(a, PLAN, ASSIGN, slot="1")
    assert r["shard_id"] == "s3"  # s2 valid -> skipped; s3 corrupt -> recomputed


def test_mark_completed_requires_valid_terminal(tmp_path):
    a = host(tmp_path, "worker_a")
    sc.next_shard(a, PLAN, ASSIGN, slot="1")
    with pytest.raises(ValueError):
        sc.mark(a, "s2", "COMPLETED")
    write_terminal(a.terminals, "s2")
    c = sc.mark(a, "s2", "COMPLETED")
    assert c["state"] == "COMPLETED" and len(c["results_sha256"]) == 64
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1")["shard_id"] == "s3"


def test_orphan_own_claim_is_reclaimed_when_its_pid_is_gone(tmp_path):
    a = host(tmp_path, "worker_a")
    sc.next_shard(a, PLAN, ASSIGN, slot="1")
    c = sc.read_json(a.own_claim_path("s2"))
    c["pid"] = 2 ** 22 - 1  # above pid_max on every host here: never alive
    sc.atomic_write_json(a.own_claim_path("s2"), c)
    r = sc.next_shard(a, PLAN, ASSIGN, slot="1")
    assert r["shard_id"] == "s2" and r["claim"]["reclaimed_orphan"]["slot"] == "1"
    assert r["claim"]["history"][-1]["state"] == "CLAIMED"


def test_live_own_claim_blocks_other_slot(tmp_path):
    a = host(tmp_path, "worker_a")
    sc.next_shard(a, PLAN, ASSIGN, slot="1")
    sc.next_shard(a, PLAN, ASSIGN, slot="2")
    assert sc.next_shard(a, PLAN, ASSIGN, slot="3")["action"] == "NONE"


def test_steal_waits_for_relay_and_home_role_wins(tmp_path):
    a, b = host(tmp_path, "worker_a"), host(tmp_path, "worker_b")
    for s in ("s2", "s3"):
        write_terminal(a.terminals, s)
    heartbeat(a, 5)
    r = sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=True)
    assert r["shard_id"] == "s5" and r["stolen"] is True  # last shard of the plan: backwards walk
    assert sc.settle(a, "s5")["action"] == "WAIT"
    heartbeat(a, 7)
    assert sc.settle(a, "s5")["action"] == "WAIT"
    # home role reaches s5 meanwhile (s4 done); relay delivers its claim
    write_terminal(b.terminals, "s4")
    sc.next_shard(b, PLAN, ASSIGN, slot="1")
    relay(b, a, "s5")
    heartbeat(a, 8)
    r = sc.settle(a, "s5")
    assert r["action"] == "LOSE" and r["winner"] == "worker_b"
    assert sc.read_json(a.own_claim_path("s5"))["state"] == "ABANDONED"
    # the home role never waits
    assert sc.settle(b, "s5")["action"] == "WIN"
    # the loser does not take an abandoned shard again; it moves on (s4 not yet relayed as terminal)
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=True)["shard_id"] == "s4"


def test_steal_wins_when_home_is_absent(tmp_path):
    a = host(tmp_path, "worker_a")
    for s in ("s2", "s3"):
        write_terminal(a.terminals, s)
    heartbeat(a, 1)
    sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=True)
    heartbeat(a, 4)
    assert sc.settle(a, "s5")["action"] == "WIN"


def test_peer_live_claim_blocks_unless_stale(tmp_path):
    a = host(tmp_path, "worker_a")
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sc.atomic_write_json(a.peer_claims / "s2" / "claim.worker_b.json", {"host_role": "worker_b", "state": "RUNNING", "claimed_at_utc": old, "pid": 1})
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1")["shard_id"] == "s3"
    assert sc.next_shard(a, PLAN, ASSIGN, slot="2")["action"] == "NONE"
    assert sc.next_shard(a, PLAN, ASSIGN, slot="2", stale_after_seconds=3600)["shard_id"] == "s2"


def test_all_done_when_every_shard_terminal(tmp_path):
    a = host(tmp_path, "worker_a")
    for s in ("s2", "s3"):
        write_terminal(a.terminals, s)
    for s in ("s0", "s1", "s4", "s5"):
        write_terminal(a.peer_terminals, s)
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=True)["action"] == "ALL_DONE"
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=False)["action"] == "ALL_DONE"


def test_failed_marker_is_terminal_and_never_retried(tmp_path):
    a = host(tmp_path, "worker_a")
    (a.terminals / "s2").mkdir(parents=True)
    (a.terminals / "s2" / "FAILED.json").write_text("{}")
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1")["shard_id"] == "s3"


def test_ledger_counts_and_contest(tmp_path):
    a, b = host(tmp_path, "worker_a"), host(tmp_path, "worker_b")
    sc.next_shard(a, PLAN, ASSIGN, slot="1")
    write_terminal(b.terminals, "s4")
    led = sc.ledger(PLAN, ASSIGN, {"a": a.claims, "b": b.claims}, {"a": a.terminals, "b": b.terminals})
    assert led["counts"] == {"live": 1, "free": 4, "terminal:COMPLETED": 1}
    assert led["contested"] == []


def test_cli_roundtrip(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    (tmp_path / "assign.json").write_text(json.dumps(ASSIGN))
    sd = tmp_path / "state"
    args = ["--state-dir", str(sd), "--role", "coordinator"]
    rc = sc.main(["next", *args, "--plan", str(tmp_path / "plan.json"), "--assignment", str(tmp_path / "assign.json"), "--slot", "1", "--extra", '{"code_commit": "abc"}'])
    assert rc == 0
    assert sc.read_json(sd / "claims" / "s0" / "claim.coordinator.json")["code_commit"] == "abc"
    assert sc.main(["settle", *args, "--shard", "s0"]) == 0
    assert sc.main(["mark", *args, "--shard", "s0", "--state", "FAILED", "--extra", '{"exit_code": 1, "reason": "boom"}']) == 0
    assert sc.read_json(sd / "claims" / "s0" / "claim.coordinator.json")["reason"] == "boom"


def test_all_done_counts_relayed_terminal_claims(tmp_path):
    a = host(tmp_path, "worker_a")
    for s in ("s2", "s3"):
        write_terminal(a.terminals, s)
    for s, role in (("s0", "coordinator"), ("s1", "coordinator"), ("s4", "worker_b"), ("s5", "worker_b")):
        sc.atomic_write_json(a.peer_claims / s / f"claim.{role}.json", {"host_role": role, "state": "COMPLETED", "claimed_at_utc": sc.utc_now()})
    assert sc.next_shard(a, PLAN, ASSIGN, slot="1", steal=True)["action"] == "ALL_DONE"
