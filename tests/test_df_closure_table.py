"""Point 14: the owner-facing closure table refuses what would mislead, and is built from artifacts.

Report acceptance tests the order names: missing naive, reference or status; mixed scales in one row;
mismatched populations or horizons; an unsupported percentage; a zero naive error read as a ratio;
a difference of 1e-6 erased by rounding; and NO_NEW_MEASUREMENT labelling. The positive path builds
rows from a SYNTHETIC run root (declared as such) with the same artifact layout the runners save.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("df_closure_table")
B = _load("df_benchmark_contract")


def _row(**over):
    base = {"run": "r", "unit": "core_mae_s1", "arm": "core_mae", "seed": "1",
            "task_horizon_split": "t | h=60", "metric_and_scale": "MAE, kW",
            "model_error": 0.5, "naive_error": 0.6, "n_evaluated": 10, "horizon_steps": 60,
            "skill_vs_naive": T.skill(0.5, 0.6),
            "literature_value_and_source": {"status": "NOT_COMPARABLE", "source": "x", "published_value": "y",
                                            "placed_in_comparison_column": False, "why_not": "fields differ",
                                            "planned_matched_comparison": "re-execute under our contract"},
            "comparability_status": "NOT_COMPARABLE"}
    base.update(over)
    return base


def test_skill_is_one_minus_ratio_and_undefined_on_a_zero_naive():
    assert T.skill(0.5, 0.6)["value"] == pytest.approx(1-0.5/0.6)
    assert T.skill(0.5, 0.0)["status"] == "UNDEFINED" and T.skill(0.5, 0.0)["value"] is None
    assert T.skill(float("nan"), 0.6)["status"] == "UNDEFINED"


@pytest.mark.parametrize("missing", ["naive_error", "literature_value_and_source", "comparability_status"])
def test_a_row_missing_naive_reference_or_status_is_refused(missing):
    row = _row()
    row.pop(missing)
    problems = T.validate({"rows": [row]})
    assert any(missing in p for p in problems)


def test_an_unmatched_published_value_in_the_comparison_column_is_refused():
    row = _row()
    row["literature_value_and_source"]["placed_in_comparison_column"] = True
    assert any("comparison column" in p for p in T.validate({"rows": [row]}))


def test_not_comparable_without_reason_or_plan_is_refused():
    row = _row()
    row["literature_value_and_source"]["planned_matched_comparison"] = ""
    assert any("planned matched comparison" in p for p in T.validate({"rows": [row]}))


def test_mixed_scales_populations_or_horizons_in_one_row_are_refused():
    assert any("different scales" in p for p in T.validate({"rows": [_row(model_scale="kW", naive_scale="z")]}))
    assert any("different populations" in p for p in T.validate({"rows": [_row(model_population=10020, naive_population=9000)]}))
    assert any("different horizons" in p for p in T.validate({"rows": [_row(model_horizon=60, naive_horizon=1)]}))


def test_a_percentage_without_its_denominator_is_refused():
    row = _row(percentage_claims=[{"text": "19% better", "numerator": 0.11}])
    assert any("percentage" in p for p in T.validate({"rows": [row]}))


def test_a_zero_naive_with_a_defined_skill_is_refused():
    row = _row(naive_error=0.0, skill_vs_naive={"value": 0.3, "status": "MEASURED"})
    assert any("zero naive" in p for p in T.validate({"rows": [row]}))


def test_skill_that_does_not_match_the_errors_is_refused():
    row = _row(skill_vs_naive={"value": 0.5, "status": "MEASURED"})
    assert any("does not equal" in p for p in T.validate({"rows": [row]}))


def test_full_precision_survives_the_json_and_a_1e_6_difference_is_not_erased(tmp_path):
    a, b = _row(model_error=0.552498366323, unit="a_s1"), _row(model_error=0.552497366323, unit="b_s1")
    table = {"rows": [a, b], "no_new_measurement": False}
    path = tmp_path/"t.json"
    path.write_text(json.dumps(table))
    back = json.loads(path.read_text())["rows"]
    assert back[0]["model_error"]-back[1]["model_error"] == pytest.approx(1e-6, abs=1e-12)


def test_no_new_measurement_is_a_valid_empty_table_and_is_said_in_the_markdown():
    table = {"rows": [], "no_new_measurement": True}
    assert T.validate(table) == []
    assert "NO_NEW_MEASUREMENT" in T.markdown({**table, "problems": []})
    assert T.validate({"rows": [], "no_new_measurement": False})


@pytest.fixture
def synthetic_root(tmp_path):
    """A SYNTHETIC run root with the runners' artifact layout — a software fixture, declared as such."""
    rng = np.random.default_rng(3)
    n, h = 400, 60
    Y = np.abs(rng.normal(1.0, 0.5, n+2*h))
    origins = np.arange(h, h+n)
    root = tmp_path/"root"
    (root/"attempts"/"core_mae_s1").mkdir(parents=True)
    np.savez(root/"DATA.npz", Y=Y, horizon=np.array([h]), target_channel=np.array([0]),
             scaler_sd=np.array([0.5]), scaler_mean=np.array([1.0]))
    pred = Y[origins+h] + rng.normal(0, 0.1, n)
    np.savez(root/"attempts"/"core_mae_s1"/"arrays.npz", validation_pred=pred[:, None],
             validation_y=Y[origins+h][:, None], eval_origins=origins)
    (root/"DESIGN.json").write_text(json.dumps({"purpose": "SYNTHETIC_FIXTURE", "design_sha256": "d"*64}))
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {
        "core_mae_s1": {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64}}}))
    return root, Y, origins, pred, h


def test_rows_are_recomputed_from_arrays_with_the_naive_on_identical_origins(synthetic_root):
    root, Y, origins, pred, h = synthetic_root
    reg = B.registry()
    calls = []

    def warehouse(campaign):
        calls.append(campaign)
        return {"current": {"core_mae_s1": {"terminal_sha256": "t"*64}}}
    rows = T.rows_from_run(root, label="synthetic", registry=reg, warehouse=warehouse)
    assert len(rows) == 1 and calls == ["c"*64]
    r = rows[0]
    assert r["model_error"] == pytest.approx(float(np.mean(np.abs(pred-Y[origins+h]))), abs=1e-15)
    assert r["naive_error"] == pytest.approx(float(np.mean(np.abs(Y[origins]-Y[origins+h]))), abs=1e-15)
    assert r["n_evaluated"] == 400 and r["horizon_steps"] == 60
    assert r["skill_vs_naive"]["value"] == pytest.approx(1-r["model_error"]/r["naive_error"])
    assert r["comparability_status"] == "NOT_COMPARABLE"
    assert r["literature_value_and_source"]["placed_in_comparison_column"] is False
    assert r["warehouse"]["digest_matches_receipt"] is True
    assert T.validate({"rows": rows}) == []


def test_labels_that_are_not_the_sources_labels_are_refused(synthetic_root):
    root, Y, origins, pred, h = synthetic_root
    path = root/"attempts"/"core_mae_s1"/"arrays.npz"
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["validation_y"] = d["validation_y"]+1e-3
    np.savez(path, **d)
    with pytest.raises(T.TableRefusal, match="not the source's labels"):
        T.rows_from_run(root, label="synthetic", registry=B.registry())
