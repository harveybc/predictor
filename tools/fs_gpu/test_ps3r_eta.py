"""Tests for the ETA math, log parsing and queue accounting of ps3r_eta.py."""

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps3r_eta as E  # noqa: E402

NOW = dt.datetime(2026, 10, 5, 4, 0, 0, tzinfo=dt.timezone.utc)


def test_nearest_rank_p90():
    assert E.nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9.0
    assert E.nearest_rank([5.0], 0.9) == 5.0


def test_cell_id_from_log_handles_both_driver_formats():
    assert E.cell_id_from_log("cap=6670M exploration batch_002 past_to_current_siamese yh.xlu.logret_1d") == "batch_002::yh.xlu.logret_1d::past_to_current_siamese"
    assert E.cell_id_from_log("rc=0 exploration batch_002 past_to_current_siamese yh.xlu.logret_1d") == "batch_002::yh.xlu.logret_1d::past_to_current_siamese"
    assert E.cell_id_from_log("cap=8000M tier_1 batch_002 yh.usdzar_x.logret_1d") == "batch_002::yh.usdzar_x.logret_1d::baseline"
    assert E.cell_id_from_log("tier_1 batch_001 px.close_loc") == "batch_001::px.close_loc::baseline"
    assert E.cell_id_from_log("START gamma external UUID=x") is None


def test_parse_log_running_is_begin_without_end():
    lines = [
        "2026-10-05T03:36:06Z BEGIN cap=8000M tier_1 batch_002 a.b",
        "2026-10-05T03:57:13Z END rc=0 tier_1 batch_002 a.b",
        "2026-10-05T03:57:13Z BEGIN cap=8000M tier_1 batch_002 c.d",
        "2026-10-05T03:00:00Z WAIT for something (waiter, not GPU work)",
    ]
    cells = E.parse_log(lines)
    running = E.running_from_logs(cells, NOW)
    assert [r["cell_id"] for r in running] == ["batch_002::c.d::baseline"]
    assert running[0]["elapsed_seconds"] == 167 and running[0]["stale"] is False
    assert cells["batch_002::a.b::baseline"]["rc"] == 0


def test_stale_running_cell_is_flagged():
    cells = E.parse_log(["2026-10-04T10:00:00Z BEGIN tier_1 batch_002 x.y"])
    assert E.running_from_logs(cells, NOW)[0]["stale"] is True


def test_estimate_single_worker_counts_remaining_fraction_of_running_cell():
    running = [{"role": "worker_b", "elapsed_seconds": 600}]
    result = E.estimate(pending=9, running=running, active_roles={"worker_b": 1200.0}, now=NOW)
    # 9 pending + 0.5 of the running cell = 9.5 cells at 1200 s each
    assert result["seconds"] == 11400
    assert result["finish_utc"] == "2026-10-05T07:10:00Z"


def test_estimate_two_workers_sum_rates():
    result = E.estimate(pending=10, running=[], active_roles={"worker_a": 600.0, "worker_b": 1200.0}, now=NOW)
    # rates 1/600 + 1/1200 = 1/400 cells per second -> 4000 s
    assert result["seconds"] == 4000


def test_estimate_without_active_workers_or_durations():
    assert E.estimate(3, [], {}, NOW)["reason"] == "NO_ACTIVE_WORKER"
    assert E.estimate(3, [], {"worker_a": None}, NOW)["reason"] == "NO_ACTIVE_WORKER_WITH_DURATION"


def _terminal(directory: Path, wall: float, valid: bool = True, mtime: float = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = f'{{"cell":"{directory.name}"}}\n'.encode()
    (directory / "results.jsonl").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest() if valid else "0" * 64
    manifest = directory / "run_manifest.json"
    manifest.write_text(json.dumps({"status": "COMPLETED", "results_sha256": digest, "wall_seconds": wall, "cgroup_peak_bytes": 1}))
    if mtime:
        os.utime(manifest, (mtime, mtime))


def _plan(path: Path, cells) -> None:
    path.write_text("cell_id\tresult_dir\n" + "".join(f"{c}\t{d}\n" for c, d in cells))


def test_summarize_queue_counts_failed_markers_not_pending(tmp_path):
    root_b = tmp_path / "mirror_b"
    root_a = tmp_path / "mirror_a"
    plan = tmp_path / "plan.tsv"
    cells = [(f"batch_002::f{i}::baseline", f"batch_002/f{i}") for i in range(6)]
    _plan(plan, cells)
    _terminal(root_b / "batch_002/f0", 1000)
    _terminal(root_b / "batch_002/f1", 1400)
    _terminal(root_a / "batch_002/f5", 700)              # stolen cell, completed by worker_a
    (root_b / "batch_002/f2").mkdir(parents=True)
    (root_b / "batch_002/f2/FAILED.codex.json").write_text('{"status":"FAILED","rc":1}')
    _terminal(root_b / "batch_002/f3", 999, valid=False)  # digest mismatch is a failure, not a terminal
    log = tmp_path / "driver.log"
    log.write_text("2026-10-05T03:50:00Z BEGIN cap=8000M tier_1 batch_002 f4\n")
    queue = {
        "name": "q", "denominator": 6, "plan": str(plan),
        "roots": [{"role": "worker_b", "path": str(root_b)}, {"role": "worker_a", "path": str(root_a)}],
        "logs": [{"role": "worker_b", "path": str(log)}],
    }
    summary = E.summarize_queue(queue, NOW, tmp_path)
    assert summary["counts"] == {"done": 3, "failed": 2, "running": 1, "pending": 0}
    assert {f["cell_id"] for f in summary["failures"]} == {"batch_002::f2::baseline", "batch_002::f3::baseline"}
    assert summary["by_role"]["worker_b"]["done"] == 2 and summary["by_role"]["worker_a"]["done"] == 1
    assert summary["by_role"]["worker_b"]["median_seconds"] == 1200.0
    assert summary["active_roles"] == ["worker_b"] and summary["active_workers"] == 1
    # remaining = 1 - 600/1200 of the running cell; only worker_b is active
    assert summary["eta"]["seconds_median"] == 600
    assert summary["eta"]["finish_utc_median"] == "2026-10-05T04:10:00Z"
    assert summary["duplicates"] == []


def test_summarize_queue_flags_duplicate_terminals_and_waiter_is_not_a_worker(tmp_path):
    root_b, root_a = tmp_path / "b", tmp_path / "a"
    plan = tmp_path / "plan.tsv"
    _plan(plan, [("batch_002::x::baseline", "batch_002/x"), ("batch_002::y::baseline", "batch_002/y")])
    _terminal(root_b / "batch_002/x", 1000)
    _terminal(root_a / "batch_002/x", 900)
    log = tmp_path / "steal.log"
    log.write_text("2026-10-05T03:00:00Z WAIT for the successor (waiter, not GPU work)\n")
    queue = {"name": "q", "plan": str(plan), "roots": [{"role": "worker_b", "path": str(root_b)}, {"role": "worker_a", "path": str(root_a)}], "logs": [{"role": "worker_a", "path": str(log)}]}
    summary = E.summarize_queue(queue, NOW, tmp_path)
    assert summary["duplicates"][0]["roles"] == ["worker_b", "worker_a"]
    assert summary["active_workers"] == 0
    assert summary["counts"]["pending"] == 1
    assert summary["eta"]["reason"] == "NO_ACTIVE_WORKER"


def test_waiting_queue_chains_after_upstream_with_fallback_median(tmp_path):
    plan_up, plan_down = tmp_path / "up.tsv", tmp_path / "down.tsv"
    _plan(plan_up, [("batch_001::a::masked_temporal_ae", "batch_001/a")])
    _plan(plan_down, [("batch_001::a::baseline", "batch_001/a"), ("batch_001::b::baseline", "batch_001/b")])
    root_up, root_down, root_peer = tmp_path / "up", tmp_path / "down", tmp_path / "peer"
    log_up = tmp_path / "up.log"
    log_up.write_text("2026-10-05T03:59:00Z BEGIN cap=1M exploration batch_001 masked_temporal_ae a\n")
    # upstream has a duration history via a completed cell outside the plan? no: give it a completed cell in plan
    _plan(plan_up, [("batch_001::a::masked_temporal_ae", "batch_001/a"), ("batch_001::z::masked_temporal_ae", "batch_001/z")])
    _terminal(root_up / "batch_001/z", 120)
    plan_peer = tmp_path / "peer.tsv"
    _plan(plan_peer, [("batch_002::p::baseline", "batch_002/p")])
    _terminal(root_peer / "batch_002/p", 1000)
    config = {
        "queues": [
            {"name": "alts", "plan": str(plan_up), "roots": [{"role": "worker_a", "path": str(root_up)}], "logs": [{"role": "worker_a", "path": str(log_up)}]},
            {"name": "peer", "plan": str(plan_peer), "roots": [{"role": "worker_b", "path": str(root_peer)}], "logs": []},
            {"name": "succ", "plan": str(plan_down), "roots": [{"role": "worker_a", "path": str(root_down)}], "logs": [], "starts_after": "alts", "fallback_median_from": "peer"},
        ],
        "totals": {"heavy": {"queues": ["peer", "succ"], "denominator": 3}},
    }
    report = E.build_report(config, tmp_path, NOW)
    alts = report["queues"]["alts"]["eta"]
    assert alts["finish_utc_median"] == "2026-10-05T04:01:00Z"   # 120 s cell started 60 s ago
    succ = report["queues"]["succ"]["eta"]
    assert succ["waiting_on"] == "alts" and succ["provisional"] is True
    assert succ["finish_utc_median"] == "2026-10-05T04:34:20Z"  # 04:01:00 + 2 x 1000 s
    assert report["totals"]["heavy"] == {"done": 1, "failed": 0, "denominator": 3}
    assert report["hosts"] == {}


def test_progress_csv_appends_rows(tmp_path):
    plan = tmp_path / "plan.tsv"
    _plan(plan, [("batch_002::x::baseline", "batch_002/x")])
    config = {"queues": [{"name": "q", "denominator": 86, "plan": str(plan), "roots": [], "logs": []}]}
    report = E.build_report(config, tmp_path, NOW)
    csv_path = tmp_path / "progress.csv"
    E.append_progress(csv_path, report["generated_at_utc"], report["queues"])
    E.append_progress(csv_path, report["generated_at_utc"], report["queues"])
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("generated_at_utc,queue,done,denominator")
    assert len(lines) == 3 and lines[1].startswith("2026-10-05T04:00:00Z,q,0,86,0,0,1,0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
