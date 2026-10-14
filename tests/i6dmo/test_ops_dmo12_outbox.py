"""DMO12: outbox retry/restart duplicates no OLAP facts, never retrains, closes only after exact readback."""
import pytest

from tests.i6dmo.ops_fakes import FakeWarehouse, fit_count, make_campaign, run_worker


def _report(i):
    return dict(experiment_key=f"i6dmo:test:{i}", experiment_set_key="i6dmo:test", actor="predictor",
                lake="LOCAL_RETAINED", lineage="UNVERIFIED", config_sha256="a" * 64, code_commit="c" * 40,
                project="predictor", phase="I6DMO", tags={"i": str(i)},
                metrics=[dict(metric="Price_MAE", value=0.001 * (i + 1), split="validation", horizon=h,
                              unit="USD_per_EUR") for h in (2, 3)])


def test_dmo12_retry_and_restart_deliver_exactly_once(tmp_path):
    from tools.i6dmo import outbox
    path = tmp_path / "outbox.sqlite3"
    box = outbox.Outbox(path)
    ids = [box.enqueue(_report(i), source_unit="short.report") for i in range(2)]
    assert box.enqueue(_report(0), source_unit="short.report") == ids[0]  # idempotent
    warehouse = FakeWarehouse(fail_post=1, fail_query=1)
    first = box.pump(warehouse)
    assert first["delivered"] == 0 and first["pending"] == 2
    box.close()
    box = outbox.Outbox(path)  # restart
    second = box.pump(warehouse)
    assert second["delivered"] == 2 and second["pending"] == 0
    assert len(warehouse.rows) == 4 and warehouse.posts >= 3  # retried, yet no duplicate facts
    posts = warehouse.posts
    third = box.pump(warehouse)
    assert third["delivered"] == 2 and warehouse.posts == posts  # delivered items are not re-sent
    assert all(item["state"] == "DELIVERED" and item["readback_sha256"] for item in box.items())


def test_dmo12_readback_mismatch_never_closes_delivery(tmp_path):
    from tools.i6dmo import outbox
    box = outbox.Outbox(tmp_path / "outbox.sqlite3")
    box.enqueue(_report(0), source_unit="short.report")
    result = box.pump(FakeWarehouse(corrupt=True))
    assert result["delivered"] == 0 and result["mismatch"] == 1
    assert box.items()[0]["state"] == "MISMATCH"


def test_dmo12_unsealed_or_altered_report_is_refused(tmp_path):
    from tools.i6dmo import outbox
    box = outbox.Outbox(tmp_path / "outbox.sqlite3")
    report = _report(0)
    report["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="REPORT_DIGEST_MISMATCH"):
        box.enqueue(report, source_unit="short.report")


def test_dmo12_publication_retries_never_retrain(tmp_path):
    from tools.i6dmo import campaign, status
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", counter)
    warehouse = FakeWarehouse(fail_query=4)  # one readback per pass
    for _ in range(3):
        campaign.publish(root, warehouse=warehouse)
    pub = status.compute(root, level="publication")
    assert pub["state"] != "COMPLETE" and pub["delivered"] == 0
    campaign.publish(root, warehouse=warehouse)
    campaign.publish(root, warehouse=warehouse)
    pub = status.compute(root, level="publication")
    assert pub["delivered"] == pub["expected"] == 1 and pub["state"] == "DELIVERED"
    assert fit_count(counter) == 1
    unit = status.compute(root, level="unit", unit="short.fit")
    assert len(unit["attempts"]) == 1
