"""RR04: the governed-unit closure generator must fail where a badge-reader would pass.

The fixture is the RR04 evidence itself: two real governed units and the live-warehouse rows that
were queried back through data-gov. Every negative below is a mutation of that fixture.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/audits/evidence/RR04_GOVERNED_DELIVERY_20260926"


def _module():
    spec = importlib.util.spec_from_file_location(
        "df_governed_unit_closure", REPO / "tools/df_governed_unit_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _module()

pytestmark = pytest.mark.skipif(
    not (EVIDENCE / "WAREHOUSE_VERIFY.json").is_file(),
    reason="the RR04 evidence fixture is not present in this checkout",
)


@pytest.fixture()
def fixture(tmp_path):
    root = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, root)
    return root


def _build(root, **kwargs):
    doc = json.loads((root / "WAREHOUSE_VERIFY.json").read_text())
    units = [("rr04-gov-smoke-u1", root / "unit1"), ("rr04-gov-smoke-u2", root / "unit2")]
    return M.build(units, doc, disposition=kwargs.get("disposition", "MECHANICAL_TRANSPORT_ONLY"),
                   reference_source=kwargs.get("reference_source", "NOT_CARRIED"))


def _test_rows(table):
    return [r for r in table["rows"] if r["task_horizon_split"].endswith(".test")]


def test_the_real_units_close_with_transport_verified_and_no_problems(fixture):
    table = _build(fixture)
    rows = _test_rows(table)
    assert rows, "the completed unit must produce forecast rows"
    assert all(r["verified"] for r in rows)
    assert all(r["binding"] == "TERMINAL_ARTIFACT" for r in rows)
    assert all(r["warehouse"] == "WAREHOUSE_ACCEPTED" for r in rows)
    assert table["summary"]["rows_with_problems"] == 0
    # the model and the naive are on the same rows, by construction
    assert all(r["model_population"] == r["naive_population"] > 0 for r in rows)
    # and nothing is claimed comparable
    assert all(r["comparability"] == "NOT_COMPARABLE" for r in rows)


def test_a_forged_prediction_breaks_the_binding_and_moves_the_error(fixture):
    before = _test_rows(_build(fixture))[0]
    path = next((fixture / "unit2").glob("*_prediction.csv"))
    lines = path.read_text().splitlines()
    head, first = lines[0].split(","), lines[1].split(",")
    first[head.index("Prediction_H9")] = first[head.index("Target_H9")]   # a perfect forecast, forged
    lines[1] = ",".join(first)
    path.write_text("\n".join(lines) + "\n")

    after = _test_rows(_build(fixture))[0]
    assert after["model_error"] != before["model_error"], "the generator must read the rows, not a badge"
    assert after["binding"] == "DIGEST_MISMATCH"
    assert after["verified"] is False
    assert after["problems"], "a recomputation that disagrees with the warehouse must be a problem"


def test_a_terminal_absent_from_the_warehouse_is_never_verified(fixture):
    doc = json.loads((fixture / "WAREHOUSE_VERIFY.json").read_text())
    doc["gov_terminal"]["rows"] = []
    doc["artifacts"]["rows"] = []
    doc["metric_mae"]["rows"] = []
    (fixture / "WAREHOUSE_VERIFY.json").write_text(json.dumps(doc))
    rows = _test_rows(_build(fixture))
    assert rows and all(r["warehouse"] == "NOT_IN_WAREHOUSE" for r in rows)
    assert not any(r["verified"] for r in rows)


def test_a_completed_unit_with_no_retained_rows_is_refused_not_emitted_empty(fixture):
    for path in (fixture / "unit2").glob("*_prediction.csv"):
        path.unlink()
    with pytest.raises(M.TableRefusal):
        _build(fixture)


def test_a_failed_unit_reports_absent_and_not_zero(fixture):
    rows = [r for r in _build(fixture)["rows"] if r["unit"] == "rr04-gov-smoke-u1"]
    assert len(rows) == 1
    assert rows[0]["model_error"] is None and rows[0]["naive_error"] is None
    assert rows[0]["skill_vs_naive"] is None
    assert rows[0]["verified"] is False


def test_splits_without_retained_rows_carry_a_null_recomputation(fixture):
    table = _build(fixture)
    held = [r for r in table["rows"] if r["task_horizon_split"].endswith((".train", ".validation"))]
    assert held, "the warehouse carries train and validation values for this unit"
    for row in held:
        assert row["model_error"] is None and row["naive_error"] is None
        assert row["warehouse_reported_model_error"] is not None
        assert row["verified"] is False
        assert "NOT_RECOMPUTABLE_NO_RETAINED_ROWS" in row["verified_reason"]


def test_a_zero_naive_error_leaves_skill_null_rather_than_infinite():
    pred = {"fields": ["base_CLOSE", "Target_H1", "Prediction_H1"],
            "base_column": "base_CLOSE", "horizons": [1],
            "rows": [{"base_CLOSE": "1.0", "Target_H1": "1.0", "Prediction_H1": "1.5"},
                     {"base_CLOSE": "2.0", "Target_H1": "2.0", "Prediction_H1": "2.5"}]}
    out = M.recompute(pred, 1)
    assert out["naive_mae"] == 0.0
    assert out["model_mae"] == pytest.approx(0.5)
    assert out["skill_vs_naive"] is None


def test_rows_missing_any_of_the_three_values_are_dropped_from_both_errors():
    pred = {"fields": ["b_CLOSE", "Target_H1", "Prediction_H1"], "base_column": "b_CLOSE",
            "horizons": [1],
            "rows": [{"b_CLOSE": "1.0", "Target_H1": "2.0", "Prediction_H1": "3.0"},
                     {"b_CLOSE": "", "Target_H1": "2.0", "Prediction_H1": "9.0"},
                     {"b_CLOSE": "1.0", "Target_H1": "nan", "Prediction_H1": "9.0"}]}
    out = M.recompute(pred, 1)
    assert out["n"] == 1
    assert out["model_mae"] == pytest.approx(1.0)
    assert out["naive_mae"] == pytest.approx(1.0)
    assert out["skill_vs_naive"] == pytest.approx(0.0)


def test_an_empty_unit_list_is_refused():
    with pytest.raises(M.TableRefusal):
        M.build([], {}, disposition="X", reference_source="NOT_CARRIED")


def test_markdown_prints_absent_for_null_and_never_a_zero(fixture):
    text = M.markdown(_build(fixture))
    assert "null (absent)" in text
    assert "| 0 |" not in text
