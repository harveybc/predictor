"""write_metrics on SQLite: DDL once, store, already_stored, hash mismatch,
null split/horizon, gov_metric_current, gov_dataset lineage columns."""

import sqlite3

import pytest

from query_plugins.sql_query import Plugin, canonical_body, normalise_report


def _plugin(tmp_path):
    q = Plugin()
    q.set_params(sqlite_path=str(tmp_path / "lab" / "cube.sqlite"), holdout_start=None)
    return q


def _names(tmp_path, kind):
    conn = sqlite3.connect(tmp_path / "lab" / "cube.sqlite")
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _rows(tmp_path, sql, params=()):
    conn = sqlite3.connect(tmp_path / "lab" / "cube.sqlite")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def test_tables_and_view_created_once(tmp_path, monkeypatch, make_report):
    q = _plugin(tmp_path)
    calls = []
    original = q._ensure_schema
    monkeypatch.setattr(q, "_ensure_schema", lambda engine: (calls.append(1), original(engine)))
    q.engine()
    q.write_metrics(make_report())
    q.write_metrics(make_report(experiment_key="other"))
    assert calls == [1]
    assert {"gov_report", "gov_metric", "gov_dataset"} <= _names(tmp_path, "table")
    assert "gov_metric_current" in _names(tmp_path, "view")
    assert {
        "gov_metric_report_idx", "gov_dataset_report_idx",
        "gov_dataset_sha256_idx", "gov_report_experiment_idx",
    } <= _names(tmp_path, "index")
    assert _rows(tmp_path, "PRAGMA journal_mode")[0]["journal_mode"] == "wal"
    # SELECT-only query still works and the file was created by the plugin
    assert q.query("SELECT COUNT(*) AS n FROM gov_report LIMIT 1")["rows"][0]["n"] == 2


def test_report_stored_then_already_stored_with_stored_lineage(tmp_path, make_report):
    q = _plugin(tmp_path)
    report = make_report()
    first = q.write_metrics(report)
    assert first == {"stored": True, "already_stored": False, "lineage": "VERIFIED"}
    stored = _rows(tmp_path, "SELECT * FROM gov_report")
    assert len(stored) == 1
    assert stored[0]["experiment_key"] == "toy-ann-1d"
    assert stored[0]["actor"] == "predictor"
    assert stored[0]["lake_id"] == "olap_cube"
    assert stored[0]["tags_json"] == '{"plugin":"ann"}'
    assert stored[0]["n_metrics"] == 1 and stored[0]["n_datasets"] == 1
    assert stored[0]["received_at"]

    # Lineage is stored but not hashed: a re-post claiming UNVERIFIED is the
    # same report and the answer carries the lineage stored the first time.
    again = dict(report)
    again["lineage"] = "UNVERIFIED"
    again["datasets"] = [dict(report["datasets"][0], lineage="UNVERIFIED", event_id=None)]
    second = q.write_metrics(again)
    assert second == {"stored": False, "already_stored": True, "lineage": "VERIFIED"}
    assert len(_rows(tmp_path, "SELECT * FROM gov_metric")) == 1
    assert len(_rows(tmp_path, "SELECT * FROM gov_dataset")) == 1


def test_hash_mismatch_refused(tmp_path, make_report):
    q = _plugin(tmp_path)
    q.engine()
    report = make_report()
    report["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        q.write_metrics(report)
    tampered = make_report()
    tampered["metrics"][0]["value"] = 0.5
    with pytest.raises(ValueError, match="mismatch"):
        q.write_metrics(tampered)
    missing = make_report()
    del missing["report_sha256"]
    with pytest.raises(ValueError, match="report_sha256"):
        q.write_metrics(missing)
    assert _rows(tmp_path, "SELECT * FROM gov_report") == []


def test_invalid_reports_refused(tmp_path, make_report):
    q = _plugin(tmp_path)
    nan = make_report()
    nan["metrics"][0]["value"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        q.write_metrics(nan)
    with pytest.raises(ValueError, match="metrics"):
        normalise_report(dict(make_report(), metrics=[]))
    with pytest.raises(ValueError, match="experiment_key"):
        normalise_report(dict(make_report(), experiment_key="bad key"))
    with pytest.raises(ValueError, match="sha256"):
        normalise_report(dict(
            make_report(),
            datasets=[{"lake": "l", "resource": "r", "sha256": "xyz"}],
        ))
    with pytest.raises(ValueError):
        q.write_metrics("not an object")


def test_metrics_with_null_split_and_horizon_stored(tmp_path, make_report):
    q = _plugin(tmp_path)
    report = make_report(metrics=[
        {"metric": "AUC_ROC", "value": 0.7},
        {"metric": "MAE", "value": 1, "split": "test", "horizon": 1.0},
    ])
    assert q.write_metrics(report)["stored"]
    rows = _rows(tmp_path, "SELECT * FROM gov_metric ORDER BY metric")
    assert rows[0]["metric"] == "AUC_ROC"
    assert rows[0]["split"] is None and rows[0]["horizon"] is None
    assert rows[0]["std_dev"] is None and rows[0]["unit"] is None
    assert rows[1]["horizon"] == 1 and rows[1]["value"] == 1.0


def test_gov_metric_current_returns_latest_report_per_experiment(tmp_path, make_report):
    q = _plugin(tmp_path)
    older = make_report(received_at="2026-09-13T00:00:00.000000+00:00")
    newer = make_report(
        received_at="2026-09-13T00:00:01.000000+00:00",
        metrics=[{"metric": "MAE", "value": 0.001, "split": "train", "horizon": 24}],
    )
    other = make_report(experiment_key="another", received_at="2026-09-13T00:00:02.000000+00:00")
    assert older["report_sha256"] != newer["report_sha256"]
    q.write_metrics(newer)
    q.write_metrics(older)
    q.write_metrics(other)
    current = _rows(tmp_path, "SELECT * FROM gov_metric_current ORDER BY experiment_key")
    assert [(r["experiment_key"], r["value"]) for r in current] == [
        ("another", 0.0065), ("toy-ann-1d", 0.001),
    ]
    assert current[1]["report_sha256"] == newer["report_sha256"]
    assert current[1]["lake_id"] == "olap_cube"
    assert len(_rows(tmp_path, "SELECT * FROM gov_metric")) == 3


def test_gov_dataset_carries_lineage_columns(tmp_path, make_report):
    q = _plugin(tmp_path)
    report = make_report(datasets=[
        {
            "lake": "predictor_examples", "resource": "phase_1/normalized_d5.csv",
            "sha256": "5" * 64, "role": "x_validation_file", "lineage": "VERIFIED",
            "event_id": 41, "source_sha256": "6" * 64, "from": "2020-01-01",
            "to": "2024-12-31", "delivery": "CUT", "time_column": "DATE_TIME",
        },
        {
            "lake": "predictor_examples", "resource": "phase_1/normalized_d5.csv",
            "sha256": "5" * 64, "role": "y_validation_file", "lineage": "UNVERIFIED",
            "reason": "never served",
        },
        # duplicate by (lake, resource, sha256, role): collapsed, not an error
        {
            "lake": "predictor_examples", "resource": "phase_1/normalized_d5.csv",
            "sha256": "5" * 64, "role": "y_validation_file",
        },
    ], lineage=None)
    assert q.write_metrics(report) == {
        "stored": True, "already_stored": False, "lineage": "UNVERIFIED",
    }
    rows = _rows(tmp_path, "SELECT * FROM gov_dataset ORDER BY role")
    assert len(rows) == 2
    verified = rows[0]
    assert verified["role"] == "x_validation_file"
    assert verified["lineage"] == "VERIFIED" and verified["reason"] is None
    assert verified["event_id"] == 41
    assert verified["source_sha256"] == "6" * 64
    assert (verified["range_from"], verified["range_to"]) == ("2020-01-01", "2024-12-31")
    assert verified["delivery"] == "CUT" and verified["time_column"] == "DATE_TIME"
    unverified = rows[1]
    assert unverified["lineage"] == "UNVERIFIED"
    assert unverified["reason"] == "never served"
    assert unverified["event_id"] is None and unverified["source_sha256"] is None
    # the canonical body knows nothing of lineage and collapses the duplicate
    body = canonical_body(normalise_report(report))
    assert [set(d) for d in body["datasets"]] == [{"lake", "resource", "sha256", "role"}] * 2
