"""RP66: the owner-facing closure table is verified along the whole chain, from a registered design.

Musashi's probes against 4ef9f71 are the red side: predictions changed under the same receipt scored
2.0 then 0.0 with no problem; a missing warehouse terminal, a NaN prediction and a missing arrays file
all passed silently; pilots were excluded by filename. Here every one of those is a PROBLEM on a row
that is not verified, roles come from the design, and model/naive population, horizon and scale are
produced fields. The positive path builds rows from a SYNTHETIC run root (declared as such) with the
artifact layout the runners save: attempts/<unit>/{arrays.npz,cell.json}, TERMINALS/<unit>.json,
TERMINAL_RECEIPTS.json, DESIGN.json with cells/pilots, DATA.npz + DATA.json.
"""
import hashlib
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
UNIT, N, H = "core_mae_s1", 400, 60


def _row(**over):
    base = {"run": "r", "unit": UNIT, "arm": "core_mae", "seed": "1",
            "task_horizon_split": "t | h=60", "metric_and_scale": "MAE, kW",
            "model_error": 0.5, "naive_error": 0.6, "n_evaluated": 10, "horizon_steps": 60,
            "model_population": 10, "naive_population": 10, "model_horizon": 60, "naive_horizon": 60,
            "model_scale": "kW", "naive_scale": "kW", "binding": {"level": "TERMINAL_ARTIFACT"},
            "skill_vs_naive": T.skill(0.5, 0.6),
            "literature_value_and_source": {"status": "NOT_COMPARABLE", "source": "x", "published_value": "y",
                                            "placed_in_comparison_column": False, "why_not": "fields differ",
                                            "planned_matched_comparison": "re-execute under our contract"},
            "comparability_status": "NOT_COMPARABLE", "problems": [], "verified": True}
    base.update(over)
    return base


# --- row rules ---------------------------------------------------------------------------------------------

def test_skill_is_one_minus_ratio_and_undefined_on_a_zero_naive():
    assert T.skill(0.5, 0.6)["value"] == pytest.approx(1-0.5/0.6)
    assert T.skill(0.5, 0.0)["status"] == "UNDEFINED" and T.skill(0.5, 0.0)["value"] is None
    assert T.skill(float("nan"), 0.6)["status"] == "UNDEFINED"


@pytest.mark.parametrize("missing", ["naive_error", "literature_value_and_source", "comparability_status",
                                     "model_population", "naive_horizon", "model_scale", "binding"])
def test_a_row_missing_a_mandatory_field_is_refused(missing):
    row = _row()
    row.pop(missing)
    assert any(missing in p for p in T.validate({"rows": [row]}))


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
    assert any("percentage" in p for p in T.validate({"rows": [_row(percentage_claims=[{"text": "19% better", "numerator": 0.11}])]}))


def test_a_zero_naive_with_a_defined_skill_is_refused():
    assert any("zero naive" in p for p in T.validate({"rows": [_row(naive_error=0.0, skill_vs_naive={"value": 0.3, "status": "MEASURED"})]}))


def test_skill_that_does_not_match_the_errors_is_refused():
    assert any("does not equal" in p for p in T.validate({"rows": [_row(skill_vs_naive={"value": 0.5, "status": "MEASURED"})]}))


def test_a_rows_own_problems_are_the_tables_problems():
    row = _row(problems=[f"{UNIT}: CHANGED ARRAYS"], verified=False)
    assert any("CHANGED ARRAYS" in p for p in T.validate({"rows": [row]}))


def test_full_precision_survives_the_json_and_a_1e_6_difference_is_not_erased(tmp_path):
    a, b = _row(model_error=0.552498366323, unit="a_s1"), _row(model_error=0.552497366323, unit="b_s1")
    path = tmp_path/"t.json"
    path.write_text(json.dumps({"rows": [a, b], "no_new_measurement": False}))
    back = json.loads(path.read_text())["rows"]
    assert back[0]["model_error"]-back[1]["model_error"] == pytest.approx(1e-6, abs=1e-12)


def test_no_new_measurement_is_a_valid_empty_table_and_is_said_in_the_markdown():
    table = {"rows": [], "no_new_measurement": True}
    assert T.validate(table) == []
    assert "NO_NEW_MEASUREMENT" in T.markdown({**table, "problems": []})
    assert T.validate({"rows": [], "no_new_measurement": False})


# --- roles from the registered design ------------------------------------------------------------------------

def test_roles_come_from_the_design_not_from_filenames():
    design = {"pilots": [{"cell_id": "pilot_x"}],
              "cells": [{"cell_id": "ae_s1", "kind": "ae"}, {"cell_id": "controls", "kind": "controls"},
                        {"cell_id": "fit_s1", "kind": "fit"}, {"cell_id": "core_mae_s1", "arm": "core_mae"},
                        {"cell_id": "odd"}]}
    roles = T.unit_roles(design)
    assert roles == {"prepare": "preparation", "pilot_x": "cost_pilot", "ae_s1": "pretraining", "controls": "control_forecast",
                     "fit_s1": "forecast", "core_mae_s1": "forecast", "odd": "unknown"}


# --- the synthetic run root ------------------------------------------------------------------------------------

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_arrays(root, pred, y, origins, *, unit=UNIT, huber_layout=True, Y=None):
    path = root/"attempts"/unit/"arrays.npz"
    if huber_layout:
        np.savez(path, pred=pred, y=y, naive=Y[origins], origins=origins)
    else:
        np.savez(path, validation_pred=pred[:, None], validation_y=y[:, None], eval_origins=origins)
    return _sha(path)


def _bind(root, unit, arrays_sha, *, mae=None, terminal=True, record=True):
    if terminal:
        (root/"TERMINALS").mkdir(exist_ok=True)
        (root/"TERMINALS"/f"{unit}.json").write_text(json.dumps({"unit": unit, "status": "COMPLETED", "artifacts": [
            {"role": "predictions", "sha256": arrays_sha}, {"role": "record", "sha256": "r"*64}]}))
    if record:
        (root/"attempts"/unit/"cell.json").write_text(json.dumps({"arrays_sha256": arrays_sha, "scores": {"validation": {
            "model": {"mae_kw": mae}}}}))


@pytest.fixture
def synthetic_root(tmp_path):
    """A SYNTHETIC run root with the runners' artifact layout — a software fixture, declared as such."""
    rng = np.random.default_rng(3)
    Y = np.abs(rng.normal(1.0, 0.5, N+2*H))
    origins = np.arange(H, H+N)
    root = tmp_path/"root"
    (root/"attempts"/UNIT).mkdir(parents=True)
    np.savez(root/"DATA.npz", Y=Y, horizon=np.array([H]), target_channel=np.array([0]), scaler_sd=np.array([0.5]),
             scaler_mean=np.array([1.0]), eval_origins=origins)
    (root/"DATA.json").write_text(json.dumps({"input_columns": ["Global_active_power"], "target_channel": 0}))
    pred = Y[origins+H] + rng.normal(0, 0.1, N)
    sha = _write_arrays(root, pred, Y[origins+H], origins, Y=Y)
    mae = float(np.mean(np.abs(pred-Y[origins+H])))
    _bind(root, UNIT, sha, mae=mae)
    (root/"DESIGN.json").write_text(json.dumps({"purpose": "SYNTHETIC_FIXTURE", "design_sha256": "d"*64,
                                                "pilots": [{"cell_id": "pilot_cost"}],
                                                "cells": [{"cell_id": UNIT, "arm": "core_mae", "seed": 1}]}))
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {
        UNIT: {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64},
        "pilot_cost": {"campaign_sha256": "c"*64, "terminal_sha256": "p"*64}}}))
    return {"root": root, "Y": Y, "origins": origins, "pred": pred, "sha": sha, "mae": mae}


def _wh(sha, *, status="COMPLETED", present=True, digest="t"*64):
    def warehouse(campaign):
        if not present:
            return {"current": {}}
        return {"current": {UNIT: {"terminal_sha256": digest, "status": status,
                                   "artifacts": [{"role": "predictions", "sha256": sha}]}}}
    return warehouse


def _table(fx, warehouse=None, sha=None):
    return T.build([f"{fx['root']}:synthetic"], registry=B.registry(),
                   warehouse=warehouse or _wh(sha or fx["sha"]), no_new_measurement=True)


def test_rows_are_recomputed_from_arrays_bound_to_the_terminal_and_the_warehouse(synthetic_root):
    fx = synthetic_root
    Y, origins, pred = fx["Y"], fx["origins"], fx["pred"]
    table = _table(fx)
    assert table["problems"] == [] and table["verified_rows"] == 1 and len(table["rows"]) == 1
    r = table["rows"][0]
    assert r["model_error"] == pytest.approx(float(np.mean(np.abs(pred-Y[origins+H]))), abs=1e-15)
    assert r["naive_error"] == pytest.approx(float(np.mean(np.abs(Y[origins]-Y[origins+H]))), abs=1e-15)
    assert r["model_error_z"] == pytest.approx(r["model_error"]/0.5)
    assert (r["model_population"], r["naive_population"], r["model_horizon"], r["naive_horizon"]) == (N, N, H, H)
    assert r["model_scale"] == r["naive_scale"] == "kW" and r["target"] == "Global_active_power"
    assert r["binding"]["level"] == "TERMINAL_ARTIFACT" and r["binding"]["arrays_sha256"] == fx["sha"]
    assert r["warehouse"] == {"checked": True, "terminal_in_warehouse": True, "digest_matches_receipt": True,
                              "status": "COMPLETED", "artifact_rows": 1}
    assert r["record_score_checked"] and r["verified"] and r["role"] == "forecast"
    assert r["comparability_status"] == "NOT_COMPARABLE" and r["literature_value_and_source"]["placed_in_comparison_column"] is False
    assert r["units_not_scored_by_role"] == [{"unit": "pilot_cost", "role": "cost_pilot", "why": "not a forecast unit by its registered role"},
                                             {"unit": "prepare", "role": "preparation", "why": "not a forecast unit by its registered role"}]
    md = T.markdown(table)
    assert "TERMINAL_ARTIFACT" in md and "| yes |" in md


def test_PROBE_changed_predictions_under_the_same_receipt_are_a_problem(synthetic_root):
    fx = synthetic_root
    before = _table(fx)
    assert before["rows"][0]["model_error"] > 0 and before["problems"] == []
    _write_arrays(fx["root"], fx["Y"][fx["origins"]+H], fx["Y"][fx["origins"]+H], fx["origins"], Y=fx["Y"])   # perfect, unbound
    after = _table(fx)
    r = after["rows"][0]
    assert r["model_error"] is None and not r["verified"]
    assert any("CHANGED ARRAYS" in p for p in after["problems"]) and after["verified_rows"] == 0
    assert any("warehouse's predictions artifact digest" in p for p in after["problems"])
    assert "| NO:" in T.markdown(after)


def test_PROBE_a_missing_warehouse_terminal_is_a_problem(synthetic_root):
    fx = synthetic_root
    t = _table(fx, warehouse=_wh(fx["sha"], present=False))
    assert any("holds NO terminal" in p for p in t["problems"]) and t["verified_rows"] == 0
    t = _table(fx, warehouse=_wh(fx["sha"], digest="x"*64))
    assert any("digest differs from the client's receipt" in p for p in t["problems"])
    t = _table(fx, warehouse=_wh(fx["sha"], status="FAILED"))
    assert any("not COMPLETED" in p for p in t["problems"])


def test_PROBE_a_nonfinite_prediction_is_a_problem(synthetic_root):
    fx = synthetic_root
    pred = fx["pred"].copy(); pred[7] = np.nan
    sha = _write_arrays(fx["root"], pred, fx["Y"][fx["origins"]+H], fx["origins"], Y=fx["Y"])
    _bind(fx["root"], UNIT, sha, mae=0.0)
    t = _table(fx, sha=sha)
    assert any("1 non-finite predictions" in p for p in t["problems"]) and t["rows"][0]["model_error"] is None


def test_PROBE_a_missing_arrays_file_of_a_registered_forecast_is_a_problem_not_an_absence(synthetic_root):
    fx = synthetic_root
    (fx["root"]/"attempts"/UNIT/"arrays.npz").unlink()
    t = _table(fx)
    assert len(t["rows"]) == 1 and any("no arrays.npz" in p and "missing, not absent" in p for p in t["problems"])
    assert t["no_new_measurement"] is True and t["verified_rows"] == 0


def test_a_registered_forecast_without_a_receipt_and_a_stranger_terminal_are_problems(synthetic_root):
    fx = synthetic_root
    (fx["root"]/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {
        "ghost_s9": {"campaign_sha256": "c"*64, "terminal_sha256": "g"*64}}}))
    t = _table(fx)
    units = {r["unit"]: r for r in t["rows"]}
    assert not units[UNIT]["verified"] and any("NO accepted terminal receipt" in p for p in units[UNIT]["problems"])
    assert units["ghost_s9"]["role"] == "unregistered" and any("does not register" in p for p in units["ghost_s9"]["problems"])


def test_labels_origins_and_record_score_identities_are_checked(synthetic_root):
    fx = synthetic_root
    Y, o = fx["Y"], fx["origins"]
    truth = Y[o+H]
    # labels that are not Y[origin + h]
    sha = _write_arrays(fx["root"], fx["pred"], truth+1e-3, o, Y=Y); _bind(fx["root"], UNIT, sha, mae=fx["mae"])
    assert any("label/horizon identity" in p for p in _table(fx, sha=sha)["problems"])
    # DATA's origins in another order
    perm = np.random.default_rng(1).permutation(N)
    sha = _write_arrays(fx["root"], fx["pred"][perm], truth[perm], o[perm], Y=Y); _bind(fx["root"], UNIT, sha, mae=fx["mae"])
    assert any("another order" in p for p in _table(fx, sha=sha)["problems"])
    # a subset of the population
    sha = _write_arrays(fx["root"], fx["pred"][:300], truth[:300], o[:300], Y=Y); _bind(fx["root"], UNIT, sha, mae=fx["mae"])
    assert any("not DATA's (300 vs 400" in p for p in _table(fx, sha=sha)["problems"])
    # the record claims another score than the arrays give
    sha = _write_arrays(fx["root"], fx["pred"], truth, o, Y=Y); _bind(fx["root"], UNIT, sha, mae=fx["mae"]-1e-6)
    assert any("differs from the record's" in p for p in _table(fx, sha=sha)["problems"])


def test_binding_levels_record_digest_and_not_bound(synthetic_root):
    fx = synthetic_root
    (fx["root"]/"TERMINALS"/f"{UNIT}.json").unlink()
    t = _table(fx)
    assert t["rows"][0]["binding"]["level"] == "RECORD_DIGEST" and t["problems"] == []
    (fx["root"]/"attempts"/UNIT/"cell.json").unlink()
    t = _table(fx)
    assert t["rows"][0]["binding"]["level"] == "NOT_BOUND" and any("NOT BOUND" in p for p in t["problems"])


def test_the_pilot_phase1_layout_is_read_and_the_naive_is_persistence_on_identical_origins(synthetic_root):
    fx = synthetic_root
    sha = _write_arrays(fx["root"], fx["pred"], fx["Y"][fx["origins"]+H], fx["origins"], huber_layout=False)
    _bind(fx["root"], UNIT, sha, mae=fx["mae"])
    t = _table(fx, sha=sha)
    assert t["problems"] == [] and t["rows"][0]["naive_error"] == pytest.approx(
        float(np.mean(np.abs(fx["Y"][fx["origins"]]-fx["Y"][fx["origins"]+H]))), abs=1e-15)


def test_the_cli_exits_nonzero_on_problems_and_writes_both_files(synthetic_root, tmp_path):
    fx = synthetic_root
    reg = tmp_path/"reg.json"; reg.write_text(json.dumps(B.registry(), default=str))
    out, md = tmp_path/"t.json", tmp_path/"t.md"
    # no warehouse token -> warehouse unchecked, still verified against the terminal artifact
    assert T.main(["--run", f"{fx['root']}:synthetic", "--registry", str(reg), "--out", str(out), "--markdown", str(md),
                   "--no-new-measurement"]) == 0
    table = json.loads(out.read_text())
    assert table["schema"] == T.SCHEMA and table["rows"][0]["warehouse"] == {"checked": False}
    (fx["root"]/"attempts"/UNIT/"arrays.npz").unlink()
    assert T.main(["--run", f"{fx['root']}:synthetic", "--registry", str(reg), "--out", str(out), "--no-new-measurement"]) == 1
