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
    # the preparation's OWN evidence (RP82): BLOCK_DATA with its record, and an accepted prepare terminal in the stub warehouse
    np.savez(root/"BLOCK_DATA.npz", Y=Y, common_eval=origins, horizon=np.array([H]), target_channel=np.array([0]), scaler_sd=np.array([0.5]),
             scaler_mean=np.array([1.0]), row_offset=np.array([0]))
    (root/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": "d"*64, "data_sha256": _sha(root/"BLOCK_DATA.npz")}))
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {
        UNIT: {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64},
        "pilot_cost": {"campaign_sha256": "c"*64, "terminal_sha256": "p"*64},
        "prepare": {"campaign_sha256": "c"*64, "terminal_sha256": "q"*64}}}))
    return {"root": root, "Y": Y, "origins": origins, "pred": pred, "sha": sha, "mae": mae}


def _wh(sha, *, status="COMPLETED", present=True, digest="t"*64, root=None, record=True, metrics=None):
    """The ACCEPTED canonical payload the warehouse would return: predictions + record artifact digests (the record's
    digest is read from the fixture at call time, as the accepted payload would have recorded it)."""
    def warehouse(campaign):
        if not present:
            return {"current": {}}
        arts = [{"role": "predictions", "sha256": sha}]
        if record and root is not None and (root/"attempts"/UNIT/"cell.json").is_file():
            arts.append({"role": "record", "sha256": _sha(root/"attempts"/UNIT/"cell.json")})
        current = {UNIT: {"terminal_sha256": digest, "status": status, "artifacts": arts, "metrics": metrics or []}}
        if root is not None and (root/"BLOCK_DATA.npz").is_file():                              # the accepted prepare terminal
            current["prepare"] = {"terminal_sha256": "q"*64, "status": "COMPLETED", "artifacts": [{"role": "data", "sha256": _sha(root/"BLOCK_DATA.npz")}]}
        return {"current": current}
    return warehouse


def _table(fx, warehouse=None, sha=None):
    return T.build([f"{fx['root']}:synthetic"], registry=B.registry(),
                   warehouse=warehouse or _wh(sha or fx["sha"], root=fx["root"]), no_new_measurement=True)


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
    assert {k: r["warehouse"][k] for k in ("checked", "terminal_in_warehouse", "digest_matches_receipt", "status", "artifact_rows")} == \
        {"checked": True, "terminal_in_warehouse": True, "digest_matches_receipt": True, "status": "COMPLETED", "artifact_rows": 2}
    assert r["custody"]["class"] == "ACCEPTED_ARTIFACT_CHAIN" and table["custody_classes"] == {"ACCEPTED_ARTIFACT_CHAIN": 1}
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
    assert "| NO (" in T.markdown(after)


def test_PROBE_a_missing_warehouse_terminal_is_a_problem(synthetic_root):
    fx = synthetic_root
    t = _table(fx, warehouse=_wh(fx["sha"], present=False, root=fx["root"]))
    assert any("holds NO terminal" in p for p in t["problems"]) and t["verified_rows"] == 0 and t["rows"][0]["custody"]["class"] == "NO_ACCEPTED_TERMINAL"
    t = _table(fx, warehouse=_wh(fx["sha"], digest="x"*64, root=fx["root"]))
    assert any("digest differs from the client's receipt" in p for p in t["problems"])
    t = _table(fx, warehouse=_wh(fx["sha"], status="FAILED", root=fx["root"]))
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
    (fx["root"]/"TERMINALS"/f"{UNIT}.json").unlink()                                  # the local terminal file is gone: custody is the warehouse's
    t = _table(fx)
    assert t["rows"][0]["binding"]["level"] == "RECORD_DIGEST" and t["problems"] == [] and t["rows"][0]["verified"]
    (fx["root"]/"attempts"/UNIT/"cell.json").unlink()
    t = _table(fx)
    assert t["rows"][0]["binding"]["level"] == "NOT_BOUND" and any("NOT BOUND" in p for p in t["problems"]) and not t["rows"][0]["verified"]


def test_PROBE_a_rewritten_record_and_arrays_never_certify_a_score_without_an_accepted_anchor(synthetic_root):
    """Musashi's RP73 counterexample: predictions AND their local record rewritten, the local terminal removed, the
    warehouse terminal without artifact rows and the receipt unchanged. Neither table may say verified."""
    fx = synthetic_root
    (fx["root"]/"TERMINALS"/f"{UNIT}.json").unlink()
    prep = lambda: {"terminal_sha256": "q"*64, "status": "COMPLETED", "artifacts": [{"role": "data", "sha256": _sha(fx["root"]/"BLOCK_DATA.npz")}]}
    no_artifacts = lambda campaign: {"current": {UNIT: {"terminal_sha256": "t"*64, "status": "COMPLETED", "artifacts": [], "metrics": []}, "prepare": prep()}}
    before = _table(fx, warehouse=no_artifacts)
    assert before["rows"][0]["model_error"] > 0 and not before["rows"][0]["verified"] and before["rows"][0]["custody"]["class"] == "UNANCHORED"
    assert any("a local record is not custody" in p for p in before["problems"])
    perfect = fx["Y"][fx["origins"]+H]
    sha = _write_arrays(fx["root"], perfect, perfect, fx["origins"], Y=fx["Y"]); _bind(fx["root"], UNIT, sha, mae=0.0, terminal=False)
    after = _table(fx, warehouse=no_artifacts)
    assert after["rows"][0]["model_error"] == 0.0 and not after["rows"][0]["verified"] and after["verified_rows"] == 0 and after["problems"]


def test_a_historical_score_is_preserved_when_the_accepted_metric_anchors_it_and_refused_when_it_disagrees(synthetic_root):
    """History without artifact rows: the metric the accepted terminal carries is the only independent anchor. Equal ->
    PRESERVED with a qualified scope (never 'verified'); different -> CHANGED ARRAYS."""
    fx = synthetic_root
    prep = lambda: {"terminal_sha256": "q"*64, "status": "COMPLETED", "artifacts": [{"role": "data", "sha256": _sha(fx["root"]/"BLOCK_DATA.npz")}]}
    anchored = lambda campaign: {"current": {UNIT: {"terminal_sha256": "t"*64, "status": "COMPLETED", "artifacts": [],
                                                    "metrics": [{"metric": "e1.phase1.mae_validation", "split": "validation", "value": fx["mae"]}]}, "prepare": prep()}}
    t = _table(fx, warehouse=anchored)
    r = t["rows"][0]
    assert t["problems"] == [] and not r["verified"] and r["preserved_with_qualified_scope"] and r["custody"]["class"] == "METRIC_ANCHORED"
    assert t["preserved_qualified_rows"] == 1 and "PRESERVED (METRIC_ANCHORED)" in T.markdown(t)
    perfect = fx["Y"][fx["origins"]+H]
    sha = _write_arrays(fx["root"], perfect, perfect, fx["origins"], Y=fx["Y"]); _bind(fx["root"], UNIT, sha, mae=0.0)
    t = _table(fx, warehouse=anchored)
    assert any("not the MAE recomputed from the arrays" in p for p in t["problems"]) and t["rows"][0]["custody"]["class"] == "UNANCHORED"


def test_a_new_result_needs_the_full_accepted_chain_predictions_and_record(synthetic_root):
    fx = synthetic_root
    E = _load("df_mod_e0")
    d = json.loads((fx["root"]/"DESIGN.json").read_text()); d["schema"] = "df_e1_block_design.v1"; d.pop("design_sha256"); d["design_sha256"] = E.sha_obj(d)
    (fx["root"]/"DESIGN.json").write_text(json.dumps(d))                                            # a sealed design: its digest recomputes
    (fx["root"]/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": d["design_sha256"], "data_sha256": _sha(fx["root"]/"BLOCK_DATA.npz")}))
    t = _table(fx)
    assert t["problems"] == [] and t["rows"][0]["verified"]
    t = _table(fx, warehouse=_wh(fx["sha"], root=fx["root"], record=False))            # accepted predictions, no record anchor
    assert any("must be anchored by the accepted artifact chain" in p for p in t["problems"]) and not t["rows"][0]["verified"]
    assert t["rows"][0]["custody"]["class"] == "ACCEPTED_PREDICTIONS_NO_RECORD_ANCHOR"


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
    assert table["rows"][0]["custody"]["class"] == "UNCHECKED" and table["verified_rows"] == 0     # nothing local verifies itself
    (fx["root"]/"attempts"/UNIT/"arrays.npz").unlink()
    assert T.main(["--run", f"{fx['root']}:synthetic", "--registry", str(reg), "--out", str(out), "--no-new-measurement"]) == 1


# --- RP82/RP83: preparation custody, the denominator, design identity, accepted tags, one authority --------------------------

def _block_root(tmp_path, *, seeds=(1,), with_receipts=True):
    """A block-shaped root: BLOCK_DATA (the preparation's own evidence), a sealed design whose digest recomputes, cells with arms,
    terminals with predictions+record artifacts, receipts incl. the prepare unit."""
    E = _load("df_mod_e0")
    rng = np.random.default_rng(5)
    Y = np.abs(rng.normal(1.0, 0.5, N+2*H)); o = np.arange(H, H+N)
    root = tmp_path/"block"; root.mkdir()
    sd = 0.9125164391265214
    np.savez(root/"BLOCK_DATA.npz", Y=Y, common_eval=o, horizon=np.array([H]), target_channel=np.array([0]), scaler_sd=np.array([sd]), scaler_mean=np.array([1.0]), row_offset=np.array([0]))
    design = {"schema": "df_e1_block_design.v1", "block": "T", "cells": [{"cell_id": f"gru_s{s}", "arm": "gru", "seed": s} for s in seeds] + [{"cell_id": "mod_s1", "arm": "mod", "seed": 1}],
              "pilots": [], "benchmark_contract": B.household_ours().to_design_block(comparability={**B.decide(B.household_ours(), B.gasparin_2019()), "comparator_state": "NONE"}),
              "source_run": {"input_columns": ["Global_active_power"], "target_channel": 0}}
    design["design_sha256"] = E.sha_obj(design)
    (root/"DESIGN.json").write_text(json.dumps(design))
    (root/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": design["design_sha256"], "data_sha256": _sha(root/"BLOCK_DATA.npz")}))
    held = {"prepare": {"terminal_sha256": "p"*64, "status": "COMPLETED", "config_sha256": design["design_sha256"],
                        "artifacts": [{"role": "data", "sha256": _sha(root/"BLOCK_DATA.npz")}, {"role": "record", "sha256": _sha(root/"BLOCK_DATA.json")}]}}
    receipts = {"prepare": {"campaign_sha256": "c"*64, "terminal_sha256": "p"*64}}
    (root/"TERMINALS").mkdir()
    for c in design["cells"]:
        u = c["cell_id"]; (root/"attempts"/u).mkdir(parents=True)
        pred = Y[o+H] + rng.normal(0, 0.1, N)
        np.savez(root/"attempts"/u/"arrays.npz", pred=pred, y=Y[o+H], naive=Y[o], origins=o, reload_pred=pred)
        (root/"attempts"/u/"cell.json").write_text(json.dumps({"arrays_sha256": _sha(root/"attempts"/u/"arrays.npz"), "scores": {"mae_kw": float(np.mean(np.abs(pred-Y[o+H])))}}))
        arts = [{"role": "predictions", "sha256": _sha(root/"attempts"/u/"arrays.npz")}, {"role": "record", "sha256": _sha(root/"attempts"/u/"cell.json")}]
        (root/"TERMINALS"/f"{u}.json").write_text(json.dumps({"status": "COMPLETED", "artifacts": arts}))
        held[u] = {"terminal_sha256": "t"*64, "status": "COMPLETED", "artifacts": arts, "config_sha256": design["design_sha256"], "tags": {"arm": c["arm"], "seed": str(c["seed"])}}
        receipts[u] = {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64}
    if with_receipts:
        (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": receipts}))
    return root, design, (lambda campaign: {"current": json.loads(json.dumps(held))}), Y, o


def test_RP82_a_tenfold_scaler_on_disk_is_a_scale_problem_not_a_verified_row(tmp_path):
    """Musashi's RP81 probe: only DATA/BLOCK_DATA scaler_sd changed; predictions, record, payloads and receipts untouched."""
    root, design, wh, Y, o = _block_root(tmp_path)
    ok = T.verify_run(root, label="r", registry=B.registry(), warehouse=wh)
    assert ok["problems"] == [] and ok["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT" and ok["denominator"]["equal_to_contract"]
    assert set(ok["verified_units"]) == {"gru_s1", "mod_s1"}
    with np.load(root/"BLOCK_DATA.npz") as z:
        d = {k: z[k] for k in z.files}
    d["scaler_sd"] = d["scaler_sd"]*10
    np.savez(root/"BLOCK_DATA.npz", **d)
    bad = T.verify_run(root, label="r", registry=B.registry(), warehouse=wh)
    assert bad["verified_units"] == [] and any("SCALE" in p for p in bad["problems"]) and any("PREPARATION_CHANGED" in p for p in bad["problems"])
    assert all(not r["verified"] for r in bad["rows"])
    # a closure-time DATA.npz with another scaler is never the denominator: the block's own preparation evidence is read
    np.savez(root/"BLOCK_DATA.npz", **{**d, "scaler_sd": d["scaler_sd"]/10})
    np.savez(root/"DATA.npz", **{**d, "scaler_sd": d["scaler_sd"]})                      # the tenfold copy, unread
    again = T.verify_run(root, label="r", registry=B.registry(), warehouse=wh)
    assert again["problems"] == [] and again["denominator"]["sd_used"] == pytest.approx(0.9125164391265214)


def test_RP82_the_preparation_must_be_the_accepted_one_and_history_gets_a_weaker_scope(tmp_path):
    root, design, wh, Y, o = _block_root(tmp_path)
    # the accepted prepare terminal anchors OTHER bytes
    def other_prepare(campaign):
        h = wh(campaign); h["current"]["prepare"]["artifacts"][0]["sha256"] = "0"*64; return h
    v = T.verify_run(root, label="r", registry=B.registry(), warehouse=other_prepare)
    assert v["preparation_custody"]["class"] == "PREPARATION_NOT_ACCEPTED" and v["verified_units"] == []
    # no warehouse read: arrays may check locally, nothing is verified and the preparation is LOCAL_ONLY
    v = T.verify_run(root, label="r", registry=B.registry(), warehouse=None)
    assert v["preparation_custody"]["class"] == "PREPARATION_LOCAL_ONLY" and v["verified_units"] == []
    # an old-layout root (DATA.npz, no accepted preparation artifact): preserved with the weaker scope, never verified
    old = tmp_path/"old"; old.mkdir()
    for f in ("DESIGN.json", "TERMINAL_RECEIPTS.json"):
        (old/f).write_text((root/f).read_text())
    import shutil; shutil.copytree(root/"attempts", old/"attempts"); shutil.copytree(root/"TERMINALS", old/"TERMINALS")
    with np.load(root/"BLOCK_DATA.npz") as z:
        np.savez(old/"DATA.npz", Y=z["Y"], eval_origins=z["common_eval"], horizon=z["horizon"], target_channel=z["target_channel"], scaler_sd=z["scaler_sd"], scaler_mean=z["scaler_mean"])
    (old/"DATA.json").write_text(json.dumps({"input_columns": ["Global_active_power"], "target_channel": 0, "data_sha256": _sha(old/"DATA.npz")}))
    d = json.loads((old/"DESIGN.json").read_text()); d["schema"] = "historical"; (old/"DESIGN.json").write_text(json.dumps(d))
    v = T.verify_run(old, label="old", registry=B.registry(), warehouse=wh)
    assert v["preparation_custody"]["class"] == "PREPARATION_LOCAL_ONLY" and v["verified_units"] == []
    assert all(r["preserved_with_qualified_scope"] and r["custody"]["scope"] == "ARRAYS_ANCHORED_PREPARATION_NOT_ACCEPTED" for r in v["rows"])


def test_RP83_a_relabeled_design_or_accepted_tags_that_disagree_are_identity_problems(tmp_path):
    root, design, wh, Y, o = _block_root(tmp_path)
    forged = json.loads(json.dumps(design))
    for c in forged["cells"]:
        if c["arm"] == "gru":
            c["arm"] = "UNTRAINED_REFERENCE"
    (root/"DESIGN.json").write_text(json.dumps(forged))                                         # old digest kept: does not recompute
    v = T.verify_run(root, label="r", registry=B.registry(), warehouse=wh)
    assert v["design_identity"]["recomputes"] is False and any("does not recompute" in p for p in v["problems"]) and v["verified_units"] == []
    E = _load("df_mod_e0")
    forged.pop("design_sha256"); forged["design_sha256"] = E.sha_obj(forged)                     # consistently rehashed relabel
    (root/"DESIGN.json").write_text(json.dumps(forged))
    v = T.verify_run(root, label="r", registry=B.registry(), warehouse=wh)
    assert any("IDENTITY" in p and "UNTRAINED_REFERENCE" in p for p in v["problems"])              # the accepted tags say 'gru'
    assert any("another" in p or "not this design" in p for p in v["problems"])                    # the accepted configuration is the old digest
    assert v["verified_units"] == []


def test_RP83_the_reference_resolver_binds_identity_and_population_through_the_same_authority(tmp_path):
    root, design, wh, Y, o = _block_root(tmp_path, seeds=(1, 2))
    ours = B.replace(B.household_ours(), source={**B.household_ours().source, "evaluation_origins": N})
    block = ours.to_design_block(comparability={**B.decide(ours, B.gasparin_2019()), "comparator_state": "NONE"})
    E = _load("df_mod_e0")
    d = json.loads((root/"DESIGN.json").read_text()); d["benchmark_contract"] = block; d.pop("design_sha256"); d["design_sha256"] = E.sha_obj(d)
    (root/"DESIGN.json").write_text(json.dumps(d))
    (root/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": d["design_sha256"], "data_sha256": _sha(root/"BLOCK_DATA.npz")}))   # prepared under the resealed design
    def wh2(campaign):                                                                             # the accepted configuration is the resealed design
        h = wh(campaign)
        for u in h["current"]:
            h["current"][u]["config_sha256"] = d["design_sha256"]
        return h
    assert B.reference_evidence(ours, root, reference_arm="gru", warehouse=wh2)["state"] == "VERIFIED_COMPARATOR"
    forged = json.loads(json.dumps(d))
    for c in forged["cells"]:
        if c["arm"] == "gru":
            c["arm"] = "UNTRAINED_REFERENCE"
    (root/"DESIGN.json").write_text(json.dumps(forged))
    assert B.reference_evidence(ours, root, reference_arm="UNTRAINED_REFERENCE", warehouse=wh2)["state"] == "PLANNED_REFERENCE"
    forged.pop("design_sha256"); forged["design_sha256"] = E.sha_obj(forged); (root/"DESIGN.json").write_text(json.dumps(forged))
    assert B.reference_evidence(ours, root, reference_arm="UNTRAINED_REFERENCE", warehouse=wh2)["state"] == "PLANNED_REFERENCE"
    (root/"DESIGN.json").write_text(json.dumps(d))
    assert B.reference_evidence(ours, root, reference_arm="gru", seeds=(1, 2, 3), warehouse=wh2)["state"] == "PLANNED_REFERENCE"     # missing seed
    def no_tags(campaign):
        h = wh2(campaign)
        for u in h["current"]:
            h["current"][u].pop("tags", None)
        return h
    assert B.reference_evidence(ours, root, reference_arm="gru", warehouse=no_tags)["state"] == "PLANNED_REFERENCE"                 # untagged payload
