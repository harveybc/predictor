"""DMO19: real task progress/age, measured ETA or UNMEASURED; stale/failed is never COMPLETE."""
import json
import time

from tests.i6dmo.ops_fakes import FakeWarehouse, make_campaign, run_worker


def _running(root, now):
    from tools.i6dmo import queue, worker
    q = queue.Queue(root / "state" / "queue.sqlite3", clock=lambda: now - 100)
    claim = q.claim("short.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
                    device_class="rtx5090", heavy=True, ttl=600)
    q.close()
    folder = worker.attempt_dir(root, "short.fit", claim["attempt"]["index"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "progress.json").write_text(json.dumps(dict(done=5, total=200, unit="epoch", phase="fit",
                                                          updated_at=now - 10)))
    return folder


def test_dmo19_running_unit_reports_progress_age_and_no_invented_eta(tmp_path):
    from tools.i6dmo import status
    root = make_campaign(tmp_path)
    now = time.time()
    _running(root, now)
    unit = status.compute(root, level="unit", unit="short.fit", now=now)
    assert unit["state"] == "RUNNING"
    assert unit["progress"]["done"] == 5 and unit["progress"]["total"] == 200
    assert 9 <= unit["heartbeat_age_seconds"] <= 11
    assert unit["eta"]["kind"] == "UPPER_BOUND_AT_DECLARED_CEILING"  # early stopping may end sooner
    camp = status.compute(root, level="campaign", now=now)
    assert camp["eta"]["fit"]["kind"] == "UNMEASURED" and camp["eta"]["publication"]["kind"] == "UNMEASURED"
    assert camp["state"] == "RUNNING" and camp["expected"]["validation_units"] == 5


def test_dmo19_stale_heartbeat_and_failure_are_never_complete(tmp_path):
    from tools.i6dmo import status
    root = make_campaign(tmp_path)
    now = time.time()
    _running(root, now)
    stale = status.compute(root, level="unit", unit="short.fit", now=now + 3600)
    assert stale["state"] == "STALE"
    assert status.compute(root, level="campaign", now=now + 3600)["state"] not in ("COMPLETE", "CLOSED")

    failed_root = make_campaign(tmp_path, name="failed")
    out = run_worker(failed_root, "preferred", tmp_path / "c.txt", mode="fail")
    assert out["result"] == "FAILED"
    camp = status.compute(failed_root, level="campaign")
    assert camp["state"] == "FAILED" and camp["counts"]["FAILED"] == 1
    assert status.compute(failed_root, level="unit", unit="short.fit")["state"] == "FAILED"


def test_dmo19_eta_measured_only_from_completed_production_timings(tmp_path):
    from tools.i6dmo import campaign, status
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    campaign.publish(root, warehouse=FakeWarehouse())
    camp = status.compute(root, level="campaign")
    assert camp["eta"]["fit"]["kind"] == "MEASURED" and camp["eta"]["fit"]["basis_n"] == 1
    assert camp["eta"]["score"]["kind"] in ("MEASURED", "UNMEASURED")
    assert camp["counts"]["COMPLETE"] == 1 and camp["state"] != "COMPLETE"
    closed = campaign.close_campaign(root)
    assert closed["state"] == "REFUSED" and closed["missing"]
    assert status.compute(root, level="campaign")["state"] != "CLOSED"
    written = json.loads((root / "STATUS.json").read_text()) if (root / "STATUS.json").exists() else None
    campaign.tick(root)
    written = json.loads((root / "STATUS.json").read_text())
    assert written["campaign"]["state"] != "COMPLETE"
