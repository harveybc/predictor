"""The OLAP report must bind the real cell and avoid claiming governed lineage."""

from tools.i6a_publish_olap import make_report, report_sha256


def test_report_is_canonical_and_carries_same_row_naive():
    cell = {"sha256": "a" * 64, "result": {
        "metrics": {"mae": 0.2, "mse": 0.3, "naive_mae": 0.25, "naive_mse": 0.35},
        "cost": {"fit_seconds": 2.0, "n_params": 42}, "skill_mae": 0.2,
        "n_scored": 100, "plan_sha256": "b" * 64, "rows_sha256": "c" * 64}}
    report = make_report("ARCH_A", 3, cell, "deadbeef")
    assert report["report_sha256"] == report_sha256(report)
    assert report["lineage"] == "UNVERIFIED"
    assert report["tags"]["cell_sha256"] == cell["sha256"]
    assert {m["metric"]: m["value"] for m in report["metrics"]}["Naive_MAE"] == 0.25
