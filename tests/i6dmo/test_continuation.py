import copy
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import continuation as N, resolution_batch as B


def schedule():
    return {k: [dict(id=k+"_one", cadence=k)] for k in N.CADENCES}


def test_complete_finite_graph_joint_search_and_declared_order():
    cells = N.cells(schedule())
    assert len(cells) == 275
    assert len({c["id"] for c in cells}) == len(cells)
    assert [sum(c["stage"] == stage for c in cells) for stage in
            ("heads", "single72", "joint", "transfer", "cadence")] == [12, 15, 240, 2, 6]
    for group in ("short", "long"):
        grid = [c for c in cells if c["stage"] == "joint" and c["group"] == group]
        assert {(c["spacing"], c["slots"], tuple(c["core"]), c["head"]) for c in grid} == {
            (s, n, tuple(core), head) for s in (2, 4) for n in B.LENGTHS for core in N.CORES for head in N.HEADS}
    assert all(1 not in c["horizons"] for c in cells)
    assert N.next_cell(dict(cells=cells), {}) == cells[0]


def result(c, value, params=100):
    return dict(cell=c, state="COMPLETE", parameters=params,
                resolved=dict(kind="conv", head=c.get("head", "conv"), core=[8, 4], slots=13, spacing=2),
                metrics={"VALIDATION": [dict(horizon=h, n=2, mae=.01*(1+value), naive_mae=.01,
                    relative_error=value, pairs_sha256=B.digest([1, 2]), sum_abs_error=.02*(1+value),
                    sum_abs_naive_error=.02) for h in c["horizons"]]})


def test_head_selection_waits_for_complete_paired_comparison_and_no_naive_gate():
    spec = dict(cells=N.cells(schedule()))
    heads = [c for c in spec["cells"] if c["stage"] == "heads" and c["group"] == "long"]
    results = {c["id"]: result(c, .5) for c in heads}
    results[heads[1]["id"]] = result(heads[1], .2)
    task = next(c for c in spec["cells"] if c["stage"] == "single72")
    assert N.resolve(spec, task, results)["kind"] == "linear"
    assert N.best(spec, results, lambda c: c in heads)["cell"] == heads[1]
    results.pop(heads[-1]["id"])
    assert N.resolve(spec, task, results)["disposition"] == "BLOCKED_DEPENDENCY"


def test_joint_grid_is_independent_of_a_failed_head_pilot():
    spec = dict(cells=N.cells(schedule()))
    task = next(c for c in spec["cells"] if c["stage"] == "joint")
    actual = N.resolve(spec, task, {})
    assert actual == {k: task[k] for k in ("kind", "head", "core", "slots", "spacing")}


def test_ranking_refuses_one_mismatched_pair_or_naive():
    spec = dict(cells=N.cells(schedule()))
    heads = [c for c in spec["cells"] if c["stage"] == "heads" and c["group"] == "short"]
    results = {c["id"]: result(c, .2) for c in heads}
    results[heads[-1]["id"]]["metrics"]["VALIDATION"][0]["pairs_sha256"] = "different"
    with pytest.raises(ValueError, match="UNPAIRED"):
        N.best(spec, results, lambda c: c in heads)


def test_zero_support_is_disposition_not_fake_zero_error():
    a = dict(mask=np.array([[True, False], [True, False]]), origin_price=np.array([1., 2.]),
             target_price=np.array([[2., np.nan], [4., np.nan]]), origin_ts=np.array([1, 2]))
    rows = N.measured_metrics(a, np.array([[2., np.nan], [4., np.nan]]), [4, 72])
    assert rows[0]["relative_error"] == -1
    assert rows[1]["n"] == 0 and rows[1]["mae"] is None and rows[1]["relative_error"] is None
    assert rows[1]["disposition"] == "NO_VALID_TARGET_IN_PERIOD"


def test_partial_period_report_never_inserts_fake_mae_or_unsortable_null(tmp_path):
    from tools.i6dmo.outbox import Outbox, expected_rows
    a = dict(mask=np.array([[True, False], [True, False]]), origin_price=np.array([1., 2.]),
             target_price=np.array([[2., np.nan], [4., np.nan]]), origin_ts=np.array([1, 2]))
    rows = N.measured_metrics(a, np.array([[2., np.nan], [4., np.nan]]), [4, 72])
    cell = dict(id="partial", kind="conv", slots=13)
    r = dict(state="COMPLETE", cell=cell, metrics={"VALIDATION": rows}, parameters=10,
             training_seconds=1., epochs=2, best_epoch=1, actual_spacing=2, population={}, sha256="a"*64)
    report = B.report(dict(batch_id="test", code_commit="b"*40), r)
    assert not any(m["metric"] == "MAE" and m["horizon"] == 72 for m in report["metrics"])
    box = Outbox(tmp_path/"outbox.sqlite3")
    try:
        box.enqueue(report, source_unit="partial")
        assert expected_rows(report)
        assert box.summary()["expected"] == 1
    finally:
        box.close()


def annual_fixture(tmp_path, score=.2):
    spec = dict(cells=N.cells(schedule()), gpu_root=str(tmp_path))
    transfer = next(c for c in spec["cells"] if c["id"] == "transfer_short")
    quarter = next(c for c in spec["cells"] if c.get("cadence") == "quarterly" and c["group"] == "short")
    results = {transfer["id"]: result(transfer, .2), quarter["id"]: result(quarter, score)}
    p = tmp_path/"results"/quarter["id"]
    p.mkdir(parents=True)
    np.savez(p/"VALIDATION.npz", origin_ts=np.array([1, 2]), mask=np.ones((2, 6), bool))
    return spec, results


def test_cadence_transition_uses_full_annual_pooled_comparison(tmp_path):
    spec, results = annual_fixture(tmp_path, .1)
    monthly = next(c for c in spec["cells"] if c.get("cadence") == "monthly" and c["group"] == "short")
    assert N.resolve(spec, monthly, results)["slots"] == 13
    quarter = next(c for c in spec["cells"] if c.get("cadence") == "quarterly" and c["group"] == "short")
    results[quarter["id"]] = result(quarter, .2)
    assert N.resolve(spec, monthly, results)["disposition"] == "SKIPPED_NOT_BENEFICIAL"


def test_cadence_cannot_score_other_annual_rows(tmp_path):
    spec, results = annual_fixture(tmp_path)
    results["transfer_short"]["metrics"]["VALIDATION"][0]["pairs_sha256"] = "other"
    with pytest.raises(ValueError, match="ANNUAL_SUPPORT"):
        N.aggregate(spec, results, "short", "quarterly")


def test_new_protocol_rejects_old_plan_and_old_inner_artifact_set(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "REPO", tmp_path)
    spec = dict(protocol=N.PROTOCOL, periods=schedule(), cells=N.cells(schedule()), seed=B.SEED,
                training=B.TRAINING, code_files={})
    B.verify_code(spec)
    spec["cells"] = B.cells()
    with pytest.raises(ValueError, match="PLAN_CHANGED"):
        B.verify_code(spec)


def test_decision_report_is_development_only_and_normalises(tmp_path):
    from tools.i6dmo.outbox import Outbox
    spec = dict(batch_id="continuation-test", code_commit="a"*40)
    select = dict(short=dict(state="COMPLETE", metrics=[dict(horizon=4, n=2, mae=.01,
                  naive_mae=.02, relative_error=-.5)]), long=dict(state="INCOMPLETE"))
    reports = N.decision_reports(spec, select, dict(disposition="PAIR_NOT_ELIGIBLE"))
    box = Outbox(tmp_path/"outbox.sqlite3")
    try:
        for report in reports:
            box.enqueue(report, source_unit="final")
        assert box.summary()["expected"] == 1
        assert reports[0]["tags"]["test_read"] == "false"
    finally:
        box.close()
