"""Tests for the declared-vs-observed state logic (tiny inputs, pure functions, no I/O)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m06_observed_state as o  # noqa: E402

NOW = o.parse_iso("2026-10-03T05:00:00Z")
AT = "2026-10-03T03:56:00Z"


def lane(**spec):
    return {"lane": "X", "state": "running", "declared_state": "RUNNING per registry", "observe": spec}


def test_declaration_alone_is_stale_never_running():
    r = o.observe(lane(jobs=r"x-.*"), [], {}, {}, {}, NOW)
    assert r["observed_state"] == "STALE_DECLARATION"
    assert r["declared_state"] == "RUNNING per registry"
    assert r["last_observed_at"] is None and r["evidence_path"] is None
    rows = o.lane_rows([lane(jobs=r"x-.*")], [], {}, {}, {}, NOW)
    assert "state" not in rows[0]


def test_live_lease_is_running_with_evidence():
    jobs = [{"id": "x-cell-1", "state": "running", "heartbeat_path": "~/hb.json", "host_alias": "worker_b"}]
    r = o.observe(lane(jobs=r"x-.*"), jobs, {}, {}, {}, NOW)
    assert r["observed_state"] == "RUNNING" and r["evidence_path"] == "~/hb.json"
    assert r["last_observed_at"] == "2026-10-03T05:00:00Z"


def test_queued_launcher_is_not_running():
    jobs = [{"id": "x-cell-1", "state": "queued"}]
    assert o.observe(lane(jobs=r"x-.*"), jobs, {}, {}, {}, NOW)["observed_state"] == "QUEUED_NOT_RUNNING"


def test_completed_job_record_is_not_an_observation():
    jobs = [{"id": "x-cell-1", "state": "completed"}]
    assert o.observe(lane(jobs=r"x-.*"), jobs, {}, {}, {}, NOW)["observed_state"] == "STALE_DECLARATION"


def test_commit_after_assignment_counts_old_branch_does_not():
    spec = dict(branches=[{"repo": "r", "branch": "b", "assigned_at": AT}])
    fresh = {"r:b": (o.parse_iso("2026-10-03T04:30:00Z"), "abcdef1234567")}
    old = {"r:b": (o.parse_iso("2026-10-01T04:30:00Z"), "abcdef1234567")}
    assert o.observe(lane(**spec), [], {}, fresh, {}, NOW)["observed_state"] == "ACTIVE_COMMIT_AFTER_ASSIGNMENT"
    assert o.observe(lane(**spec), [], {}, old, {}, NOW)["observed_state"] == "STALE_DECLARATION"


def test_result_present_is_idle_not_running():
    r = o.observe(lane(results=["res"]), [], {}, {}, {"res": "lts@77739a9"}, NOW)
    assert r["observed_state"] == "IDLE_RESULT_PRESENT" and r["evidence_path"] == "lts@77739a9"


RAW = {"status_counts": {"verified": 2, "running": 1, "queued": 1},
       "candidates": [["c1", "neat_g0000_i0001", 2021, "verified"], ["c2", "neat_g0000_i0002", 2021, "verified"],
                      ["c3", "neat_g0000_i0003", 2021, "running"], ["c4", "neat_g0001_i0001", 2021, "queued"]],
       "attempts": [["c1", 1, "train", "completed", 100.0, 1], ["c1", 1, "verify", "completed", 50.0, 1],
                    ["c2", 1, "train", "interrupted", None, 1], ["c2", 2, "train", "completed", 120.0, 1],
                    ["c2", 2, "verify", "completed", 70.0, 1], ["c3", 1, "train", "running", None, 1]]}


def test_queue_eta_needs_same_family_completion():
    q = o.queue_summary("n", RAW, NOW, planned_total=6)
    assert q["eta"]["basis"] == "not_estimable"
    assert "neat_g0001" in q["eta"]["first_estimable_event"]
    assert q["active_rows"] == 1 and q["unenqueued"] == 2
    assert [x["label"] for x in q["retry_lineage"]] == ["neat_g0000_i0002"]


def test_queue_eta_from_same_family():
    raw = {**RAW, "candidates": RAW["candidates"][:3], "status_counts": {"verified": 2, "running": 1}}
    q = o.queue_summary("n", raw, NOW)
    assert q["eta"]["basis"] == "same_family_completed_cells"
    assert q["eta"]["families"] == {"neat_g0000": 170}   # mean train 110 + mean verify 60


def test_queue_active_rows_without_process_is_named():
    q = o.queue_summary("n", RAW, NOW)
    r = o.observe(lane(jobs=r"x-.*", queues=["n"]), [], {"n": q}, {}, {}, NOW)
    assert r["observed_state"] == "QUEUE_CLAIMS_ACTIVE_NO_PROCESS"


def test_drained_queue():
    raw = {"status_counts": {"verified": 1}, "candidates": [["c1", "a_b", 1, "verified"]], "attempts": []}
    q = o.queue_summary("n", raw, NOW)
    assert o.observe(lane(queues=["n"]), [], {"n": q}, {}, {}, NOW)["observed_state"] == "IDLE_QUEUE_DRAINED"


def test_agents_inherit_lane_observation_and_drop_state():
    lanes = [{"lane": "X", "observed_state": "RUNNING", "last_observed_at": "t", "evidence_path": "p", "next_transition": "n"}]
    a = o.observe_agents([{"id": "a1", "lane": "X", "state": "running"}, {"id": "a2", "lane": "Q", "state": "assigned"}], lanes)
    assert a[0]["observed_state"] == "RUNNING" and a[0]["declared_state"] == "running" and "state" not in a[0]
    assert a[1]["observed_state"] == "STALE_DECLARATION"
