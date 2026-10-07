import json

from tools import fs4_observed as observed

GOOD_STATUS = {"schema": "fs4.closure_status.v1", "generated_epoch": 1000.0, "complete": 12, "eta_seconds": 3600,
               "eta_reason": "observed median", "warehouse": {"last_pass": {"errors": [], "verified": 2},
                                                             "verified_receipts": 12, "pending_submit": 0, "quarantined": 0}}
QUEUE = json.dumps({"schema": "fs4.status.v1", "total": 6495, "complete": 12})


def runner_ok(alias, script, timeout=90):
    return 0, QUEUE + "\n---TIMERS---\nfs4-worker@cpu-raw\nfs4-worker@gpu-4090\n", ""


def runner_denied(alias, script, timeout=90):
    return 255, "", "harveybc@host: Permission denied (publickey,password).\n"


def test_no_credential_item_when_every_observation_passes():
    workers = [observed.observe_worker("worker_a", "x", runner_ok), observed.observe_worker("worker_b", "y", runner_ok)]
    doc = observed.build(GOOD_STATUS, workers, now=1100.0)
    assert doc["credential_items"] == []
    assert workers[0]["timers"] == ["fs4-worker@cpu-raw", "fs4-worker@gpu-4090"]
    assert doc["extractibility_eta"] == {"seconds": 3600, "basis": "observed median", "completed_tasks": 12}
    assert not any(key in doc for key in ("blockers", "notes", "blocker"))


def test_a_failed_gate_check_is_the_only_way_the_ssh_item_appears():
    workers = [observed.observe_worker("worker_a", "x", runner_denied), observed.observe_worker("worker_b", "y", runner_ok)]
    items = observed.build(GOOD_STATUS, workers, now=1100.0)["credential_items"]
    assert [i["item"] for i in items] == ["worker_ssh_gate"] and items[0]["role"] == "worker_a"
    assert "Permission denied" in items[0]["observed_failure"]


def test_warehouse_item_appears_on_observed_errors_or_a_stale_follower():
    ok_workers = [observed.observe_worker("worker_a", "x", runner_ok)]
    bad = {**GOOD_STATUS, "warehouse": {"error": "refused: 401", "last_pass": None}}
    assert observed.build(bad, ok_workers, now=1100.0)["credential_items"][0]["item"] == "warehouse_readback"
    stale = observed.build(GOOD_STATUS, ok_workers, now=1000.0 + observed.STALE_SECONDS + 5)
    assert "follower not running" in stale["credential_items"][0]["observed_failure"]
    assert observed.build(None, ok_workers, now=1100.0)["credential_items"][0]["item"] == "warehouse_readback"


def test_eta_is_null_until_five_tasks_are_complete():
    young = {**GOOD_STATUS, "complete": 4}
    eta = observed.build(young, [], now=1100.0)["extractibility_eta"]
    assert eta["seconds"] is None and "fewer than 5" in eta["basis"]


def test_hosts_env_aliases_are_read_without_being_stored(tmp_path):
    path = tmp_path / "hosts.env"
    path.write_text("WORKER_A_SSH=alias-a\nWORKER_B_SSH='alias-b'\nCOORD_PYTHON=python3\n")
    assert observed.read_hosts(path) == {"worker_a": "alias-a", "worker_b": "alias-b"}
