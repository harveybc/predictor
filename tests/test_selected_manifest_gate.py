"""Fail-closed selected-manifest gate, same-row-naive strategy guard and the NEAT sequence guard.

Pure python: no TensorFlow import (the NEAT module imports only stdlib, numpy and the
plugin loader at module level).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/selection_manifest"


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _mod("selected_manifest_gate", "tools/selected_manifest_gate.py")
N = _mod("strategy_naive_guard", "tools/strategy_naive_guard.py")

ECL = FIX / "tsl_electricity.admissible_inputs.v1.json"
ETH_ADM = FIX / "eth_4h.admissible_inputs.v1.json"
ETH_FROZEN = FIX / "SELECTED_FEATURE_MANIFEST.eth_4h.v1.FROZEN_DEVELOPMENT.json"
BINDING = FIX / "m04_INPUT_BINDING.ecl.json"

# Exact bytes of the copied declarations (copied as-is from feature-eng m03 / laneB).
FIXTURE_SHA256 = {
    ECL: "9f344687aa0703d76cc49910ec94dec15f51203d808397617989389e57ab701b",
    ETH_ADM: "5129f274683561e2971e5ec9c3cea7841876247a90a34a215f87638ac6665c1e",
    ETH_FROZEN: "fdff0c85fc376cd6930cede4981a64b076022bd701c6a045339689e3ab892d4c",
    BINDING: "8ad5c6e9a7b5a42227603bdcf557a961c3e14a8909cc876bc47e2cb9e4bdb4af",
}


def _j(p):
    return json.loads(Path(p).read_text())


def _sealed_manifest(features, *, producer="feature_selection_producer", decider="independent_selection_reviewer"):
    m = {"schema": G.MANIFEST_SCHEMA, "manifest_version": 1,
         "dataset_id": "synthetic.selection_fixture.train",
         "targets": [{"name": "Y_s", "horizons": [1, 2, 3]}, {"name": "Y_l", "horizons": [24, 48]}],
         "features": features,
         "producer": {"role": producer},
         "independent_decision_record": {"decision_id": "D-0001", "decider_role": decider},
         "source_digests": {"profile": "a" * 64, "causal_ladder": "b" * 64}}
    m["manifest_sha256"] = G.canonical_sha256(m, "manifest_sha256")
    return m


def _decision(m, *, decider="independent_selection_reviewer", verdict=G.ACCEPT_VERDICT):
    d = {"schema": G.DECISION_SCHEMA, "decision_id": "D-0001", "decider_role": decider,
         "decided_on": "2026-10-03", "manifest_sha256": m["manifest_sha256"], "verdict": verdict}
    d["record_sha256"] = G.canonical_sha256(d, "record_sha256")
    return d


def _feat(name, state, reasons=("FIXTURE",)):
    return {"name": name, "state": state, "reason_codes": list(reasons)}


GOOD = [_feat("x1", "selected", ["OOF_GAIN", "NONREDUNDANT"]), _feat("x2", "selected"),
        _feat("x3", "rejected", ["REDUNDANT_WITH_x1"]), _feat("x4", "pending", ["CAUSAL_NOT_IDENTIFIED"])]


# ── fixtures are the exact declarations ────────────────────────────────────
@pytest.mark.parametrize("path", list(FIXTURE_SHA256))
def test_negative_fixtures_are_byte_exact_copies(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE_SHA256[path]


def test_ecl_declaration_digest_rederives_so_the_fixture_is_the_declaration_the_campaign_bound():
    d = _j(ECL)
    assert G.canonical_sha256(d, "declaration_sha256") == d["declaration_sha256"]
    assert d["declaration_sha256"] == _j(BINDING)["admissible_input_declaration"]["declaration_sha256"]


# ── negative: ECL 321 ──────────────────────────────────────────────────────
def test_ecl_all_321_admissible_columns_are_refused_as_a_selected_set():
    d = _j(ECL)
    cols = G.features_declared_by(d)
    assert len(cols) == 321 == d["admissible_count"]
    rep = G.evaluate(ECL, cols)
    assert rep["admitted"] is False
    assert rep["refused_count"] == 321 and rep["refused_features"] == cols
    assert any(r.startswith("ADMISSIBLE_DECLARATION_IS_NOT_SELECTION") for r in rep["reasons"])
    with pytest.raises(G.SelectionManifestRefused):
        G.require_selected_manifest(ECL, cols)


def test_ecl_columns_rewrapped_in_the_v1_schema_as_admissible_are_still_refused():
    cols = G.features_declared_by(_j(ECL))
    m = _sealed_manifest([_feat(c, "admissible", ["M03_ADMISSIBLE"]) for c in cols])
    rep = G.evaluate(m, cols, decision_record=_decision(m))
    assert not rep["admitted"] and rep["refused_count"] == 321
    assert any("NON_SELECTION_STATE: 321 feature(s) in state 'admissible'" in r for r in rep["reasons"])
    assert any(r.startswith("NO_SELECTED_FEATURE") for r in rep["reasons"])


def test_ecl_columns_marked_selected_by_their_producer_are_refused_as_self_certified():
    cols = G.features_declared_by(_j(ECL))
    m = _sealed_manifest([_feat(c, "selected") for c in cols], decider="feature_selection_producer")
    rep = G.evaluate(m, cols, decision_record=_decision(m, decider="feature_selection_producer"))
    assert not rep["admitted"] and rep["refused_count"] == 321
    assert any(r.startswith("PRODUCER_SELF_CERTIFIED") for r in rep["reasons"])


def test_the_m04_binding_final_admissible_declaration_bound_is_not_feature_selection_complete():
    b = _j(BINDING)
    assert b["input_set_status"] == G.ADMISSIBLE_ONLY_STATUS
    cols = G.features_declared_by(_j(ECL))
    with pytest.raises(G.SelectionManifestRefused) as e:
        G.require_campaign_inputs({"input_binding": b}, cols)
    rep = e.value.report
    assert rep["refused_count"] == 321
    assert rep["reasons"][0].startswith("ADMISSIBLE_BINDING_IS_NOT_SELECTION")
    assert "MISSING_MANIFEST" in rep["reasons"]


# ── negative: ETH 83 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("path,tag", [(ETH_ADM, "ADMISSIBLE_DECLARATION_IS_NOT_SELECTION"),
                                      (ETH_FROZEN, "LEGACY_PRODUCER_FROZEN_LIST_IS_NOT_SELECTION")])
def test_eth_83_feature_tables_are_refused(path, tag):
    cols = G.features_declared_by(_j(path))
    assert len(cols) == 83
    rep = G.evaluate(path, cols)
    assert not rep["admitted"] and rep["refused_count"] == 83
    assert any(r.startswith(tag) for r in rep["reasons"])


# ── negative: the generic refusals ─────────────────────────────────────────
def test_missing_manifest_refuses():
    rep = G.evaluate(None, ["x1"])
    assert not rep["admitted"] and rep["reasons"] == ["MISSING_MANIFEST"]
    assert not G.evaluate(FIX / "does_not_exist.json", ["x1"])["admitted"]


@pytest.mark.parametrize("state", ["pending", "profiled", "admissible"])
def test_state_only_manifests_refuse(state):
    m = _sealed_manifest([_feat(f"x{i}", state) for i in range(4)])
    rep = G.evaluate(m, ["x0"], decision_record=_decision(m))
    assert not rep["admitted"]
    assert any(r.startswith("NO_SELECTED_FEATURE") for r in rep["reasons"])


def test_missing_decision_record_refuses():
    m = _sealed_manifest(GOOD)
    rep = G.evaluate(m, ["x1"])
    assert not rep["admitted"] and "MISSING_DECISION_RECORD" in rep["reasons"]


@pytest.mark.parametrize("decider", ["feature_selection_producer", "producer", "Successor Technical Lead"])
def test_producer_self_certified_refuses(decider):
    m = _sealed_manifest(GOOD, decider=decider)
    rep = G.evaluate(m, ["x1"], decision_record=_decision(m, decider=decider))
    assert not rep["admitted"] and any(r.startswith("PRODUCER_SELF_CERTIFIED") for r in rep["reasons"])


def test_self_certified_flag_refuses():
    m = _sealed_manifest(GOOD)
    m["self_certified"] = True
    m["manifest_sha256"] = G.canonical_sha256(m, "manifest_sha256")
    rep = G.evaluate(m, ["x1"], decision_record=_decision(m))
    assert any("self_certified" in r for r in rep["reasons"]) and not rep["admitted"]


def test_tampered_manifest_digest_mismatch_refuses():
    m = _sealed_manifest(GOOD)
    d = _decision(m)
    m2 = copy.deepcopy(m)
    m2["features"][2]["state"] = "selected"          # promote a rejected feature after sealing
    rep = G.evaluate(m2, ["x1", "x3"], decision_record=d)
    assert not rep["admitted"]
    assert "DIGEST_MISMATCH: manifest_sha256 does not re-derive" in rep["reasons"]


def test_decision_bound_to_another_manifest_refuses():
    m = _sealed_manifest(GOOD)
    other = _sealed_manifest(GOOD[:2])
    rep = G.evaluate(m, ["x1"], decision_record=_decision(other))
    assert "DIGEST_MISMATCH: decision does not bind this manifest_sha256" in rep["reasons"]


def test_tampered_decision_record_refuses():
    m = _sealed_manifest(GOOD)
    d = _decision(m)
    d["verdict"] = "ACCEPT_SELECTED_SET_EDITED"
    rep = G.evaluate(m, ["x1"], decision_record=d)
    assert "DIGEST_MISMATCH: decision record_sha256 does not re-derive" in rep["reasons"]


def test_rejected_verdict_refuses():
    m = _sealed_manifest(GOOD)
    rep = G.evaluate(m, ["x1"], decision_record=_decision(m, verdict="RETURN_FOR_REWORK"))
    assert any(r.startswith("DECISION_NOT_ACCEPTED") for r in rep["reasons"])


def test_source_file_digest_mismatch_refuses(tmp_path):
    src = tmp_path / "profile.json"
    src.write_text("{}")
    m = _sealed_manifest(GOOD)
    rep = G.evaluate(m, ["x1"], decision_record=_decision(m), source_files={"profile": src})
    assert "DIGEST_MISMATCH: source profile" in rep["reasons"]


def test_consuming_a_rejected_pending_or_absent_feature_refuses_the_whole_set():
    m = _sealed_manifest(GOOD)
    for extra in ("x3", "x4", "x9"):
        rep = G.evaluate(m, ["x1", extra], decision_record=_decision(m))
        assert not rep["admitted"] and rep["refused_count"] == 2


def test_dataset_mismatch_and_bad_targets_refuse():
    m = _sealed_manifest(GOOD)
    assert not G.evaluate(m, ["x1"], decision_record=_decision(m), expected_dataset_id="other")["admitted"]
    m2 = _sealed_manifest(GOOD)
    m2["targets"] = [{"name": "Y_s", "horizons": [0]}]
    m2["manifest_sha256"] = G.canonical_sha256(m2, "manifest_sha256")
    assert any(r.startswith("BAD_TARGET") for r in G.evaluate(m2, ["x1"], decision_record=_decision(m2))["reasons"])


def test_missing_reason_codes_refuse():
    m = _sealed_manifest([_feat("x1", "selected", []), _feat("x2", "rejected")])
    assert any(r.startswith("MISSING_REASON_CODES") for r in G.evaluate(m, ["x1"], decision_record=_decision(m))["reasons"])


# ── positive control: the gate is not a constant refusal ───────────────────
def test_a_sealed_independently_decided_manifest_admits_only_its_selected_features(tmp_path):
    m = _sealed_manifest(GOOD)
    d = _decision(m)
    (tmp_path / "m.json").write_text(json.dumps(m))
    (tmp_path / "d.json").write_text(json.dumps(d))
    rep = G.require_selected_manifest(tmp_path / "m.json", ["x1", "x2"], decision_record=tmp_path / "d.json",
                                      expected_dataset_id="synthetic.selection_fixture.train")
    assert rep["admitted"] and rep["refused_count"] == 0 and rep["manifest_sha256"] == m["manifest_sha256"]
    camp = {"selection_manifest": "m.json", "selection_decision": "d.json"}
    assert G.require_campaign_inputs(camp, ["x1"], base_dir=tmp_path)["admitted"]


def test_cli_exit_codes(tmp_path, capsys):
    assert G.main(["--manifest", str(ECL)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["refused_count"] == 321 and out["admitted"] is False


# ── tripwire: every modular/DOIN builder entry point must call the gate ────
# Builders live on unmerged lines (satoshi/neat-gen1-20261003, satoshi/parallel-dispatch-20261002).
# When any of them reaches this tree it must reference the gate, or this test fails.
GATED_BUILDERS = ("modular_doin_campaign.py", "modular_doin_declare_corrected.py", "modular_doin_ecl_npz.py",
                  "modular_pretrain.py", "modular_supervised_donor_pretrain.py", "modular_doin_cost_pilot.py",
                  "modular_neat_matched_arms.py", "modular_neat_policy.py")
MARKERS = re.compile(r"input_binding|FINAL_ADMISSIBLE_DECLARATION_BOUND|def init_campaign|all_admissible_control")
EXEMPT = {"selected_manifest_gate.py", "strategy_naive_guard.py"}


def _builders_in_tree():
    out = []
    for p in sorted((ROOT / "tools").glob("*.py")) + sorted((ROOT / "predictor_plugins").rglob("*.py")):
        if p.name in EXEMPT:
            continue
        if p.name in GATED_BUILDERS or MARKERS.search(p.read_text(errors="replace")):
            out.append(p)
    return out


def test_every_builder_entry_point_in_this_tree_calls_the_gate():
    ungated = [str(p.relative_to(ROOT)) for p in _builders_in_tree()
               if not re.search(r"selected_manifest_gate|refuse_keras_config_fields", p.read_text(errors="replace"))]
    assert ungated == [], f"builder entry points without the selected-manifest gate: {ungated}"


# ── same-row naive guard: zero strategy invocations when skill <= 0 ────────
def _row(h, model, naive, **kw):
    r = {"family": "short", "horizon": h, "metric": "MAE", "model_error": model, "naive_error": naive,
         "rows_sha256": "r" * 64, "naive_rows_sha256": "r" * 64, "scale": "price", "naive_scale": "price"}
    r.update(kw)
    return r


@pytest.mark.parametrize("rows", [
    [_row(1, 1.0, 1.0)],                                   # skill == 0
    [_row(1, 1.2, 1.0), _row(2, 2.0, 1.5)],                 # skill < 0
    [_row(1, 0.5, 0.0)],                                   # undefined on zero naive
    [_row(1, float("nan"), 1.0)],
    [_row(1, 0.5, 1.0, naive_rows_sha256="s" * 64)],       # naive on other rows
    [_row(1, 0.5, 1.0, naive_scale="z")],                  # other scale
    [],
])
def test_zero_strategy_invocations_when_no_horizon_beats_its_same_row_naive(rows):
    calls = []
    res, rep = N.guarded_strategy_call(rows, lambda admitted: calls.append(admitted) or "ran")
    assert calls == [] and res is None and rep["strategy_invocations"] == 0
    assert rep["strategy_invocations_allowed"] is False


def test_only_horizons_that_strictly_beat_their_naive_reach_the_strategy():
    calls = []
    rows = [_row(1, 0.9, 1.0), _row(2, 1.0, 1.0), _row(3, 1.1, 1.0)]
    res, rep = N.guarded_strategy_call(rows, lambda admitted: calls.append([r["horizon"] for r in admitted]) or "ran")
    assert res == "ran" and calls == [[1]] and rep["strategy_invocations"] == 1
    assert set(rep["refused"]) == {"short:2", "short:3"}
    assert N.skill(0.9, 1.0) == pytest.approx(0.1)


# ── NEAT is not a configuration optimizer ──────────────────────────────────
@pytest.fixture(scope="module")
def neat():
    return _mod("neat_optimizer_under_test", "optimizer_plugins/neat_optimizer.py")


KERAS_CFG = {"hyperparameter_bounds": {"learning_rate": [1e-5, 1e-2], "dropout": [0.0, 0.5],
                                       "num_layers": [1, 5], "loss_type": [0, 4]}}


def test_neat_optimize_refuses_keras_config_fields(neat):
    with pytest.raises(neat.NeatKerasConfigRefused, match="NEAT_AS_CONFIG_OPTIMIZER_REFUSED at optimize.*learning_rate"):
        neat.Plugin().optimize(None, None, dict(KERAS_CFG))


def test_neat_refuses_with_plugin_default_bounds_too(neat):
    with pytest.raises(neat.NeatKerasConfigRefused):
        neat.Plugin().optimize(None, None, {})


@pytest.mark.parametrize("call", [
    lambda P: P.create_shared_population(4, dict(KERAS_CFG), seed=1),
    lambda P: P.reproduce_shared([], 0, 1, dict(KERAS_CFG), {}, [], {}),
    lambda P: P.evaluate_single_genome({}, 0, dict(KERAS_CFG)),
])
def test_neat_shared_population_entry_points_refuse(neat, call):
    with pytest.raises(neat.NeatKerasConfigRefused):
        call(neat.Plugin)


def test_no_tensorflow_was_imported_by_this_module():
    import sys
    assert "tensorflow" not in sys.modules
