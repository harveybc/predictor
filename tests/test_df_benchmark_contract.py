"""BENCHMARK-CONTRACTS (RP67): typed contracts, comparability from fields, binding to the runtime.

Musashi's probes against 4ef9f71 are the red side of these rules: a scale change and a physical-time
change that stayed REPRODUCTION, a boolean that promoted a mismatched protocol, a null contract that
was accepted, and a foreign target/horizon with a re-digested outer design that the real factorial
validator let through. Each is refused here, and the green side proves the legitimate paths still
work: an identical protocol, a declared contrast, a closed reference run, an affine conversion.
"""
import copy
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
SOURCE = Path("~/.local/state/crispdm-data-foundation/e1_household_successor_v3").expanduser()


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("df_benchmark_contract")


def _twin():
    ours = B.household_ours()
    theirs = replace(ours, task_id="their-copy", source={"kind": "PRIMARY", "citation": "x"})
    return ours, theirs


# --- Musashi's probes, now refused -------------------------------------------------------------------

def test_PROBE_a_scale_change_is_not_a_reproduction():
    ours, _ = _twin()
    d = B.decide(ours, replace(ours, metric_scale="native"))
    assert d["mode"] == "NOT_COMPARABLE" and "metric_scale" in d["fields_that_differ"]


def test_PROBE_a_physical_time_change_is_refused_as_invalid_or_non_comparable():
    ours, _ = _twin()
    inconsistent = replace(ours, horizon_seconds=72*3600)               # 60 steps of 60 s cannot be 72 h
    assert "physical time is inconsistent" in " ".join(inconsistent.validate())
    assert B.decide(ours, inconsistent)["mode"] == "NOT_COMPARABLE"
    consistent_72h = replace(ours, horizon_steps=72*60, horizon_seconds=72*3600)
    d = B.decide(ours, consistent_72h)
    assert d["mode"] == "NOT_COMPARABLE" and {"horizon_steps", "horizon_seconds"} <= set(d["fields_that_differ"])


def test_PROBE_the_boolean_authority_is_gone():
    ours, _ = _twin()
    with pytest.raises(TypeError):
        B.decide(ours, B.gasparin_2019(), reexecuted_on_our_rows=True)


def test_PROBE_a_null_contract_is_refused():
    empty = {k: None for k in B.REQUIRED_NONNULL}
    empty.update(schema=B.SCHEMA, comparability={"mode": "REPRODUCTION", "comparator_state": "NONE"})
    with pytest.raises(B.ContractRefusal):
        B.require({"benchmark_contract": empty})


def test_PROBE_unknown_placeholders_never_match_as_proof():
    k = B.kim_cho_2019()
    d = B.decide(k, replace(k, task_id="another-unknown-paper"))
    assert d["mode"] == "NOT_COMPARABLE" and "unknown identity fields" in d["why"] and d["reference_evidence"]["state"] == "NONE"


# --- typed validity ----------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,fragment", [
    ("resolution_seconds", 0, "positive integer"),
    ("horizon_steps", -1, "positive integer"),
    ("horizon_seconds", 3599, "physical time is inconsistent"),
    ("target_transform", "cube_root", "not one of"),
    ("metric_scale", "USD", "not one of"),
    ("target", "", "null or empty"),
    ("naive_baseline", None, "null or empty"),
])
def test_invalid_fields_are_named(field, value, fragment):
    ours = B.household_ours()
    bad = replace(ours, **{field: value})
    assert any(fragment in p for p in bad.validate()), bad.validate()


def test_a_zscore_contract_cannot_report_in_a_foreign_scale():
    ours = B.household_ours()
    assert any("z_train or in native" in p for p in replace(ours, metric_scale="log1p").validate())


def test_the_registry_contracts_of_ours_are_valid_and_every_source_is_not_comparable():
    reg = B.registry()
    assert reg["validity"]["household_W60_h60"] == [] and reg["validity"]["fx_eurusd_1h"] == []
    for name, d in reg["decisions_against_household_W60_h60"].items():
        assert d["mode"] == "NOT_COMPARABLE" and d["comparator_state"] == "NONE", name


# --- comparability from fields --------------------------------------------------------------------------

def test_identical_protocol_is_a_reproduction_without_a_comparator_until_a_run_exists():
    ours, theirs = _twin()
    d = B.decide(ours, theirs)
    assert d["mode"] == "REPRODUCTION" and d["fields_that_differ"] == [] and d["comparator_state"] == "NONE"


@pytest.mark.parametrize("field,value", [
    ("target", "Global_reactive_power"), ("horizon_steps", 96), ("split_rule", "last year test"),
    ("scaler_fit_population", "whole series"), ("metric_formula", "RMSE over steps"),
    ("metric_aggregation", "mean per series then concatenated"), ("target_transform", "log1p"),
    ("resolution_seconds", 900), ("missing_policy", "imputed by time-slot mean"), ("permitted_inputs", "target only"),
])
def test_a_changed_identity_field_rejects_comparability(field, value):
    ours, theirs = _twin()
    kw = {field: value}
    if field == "horizon_steps":
        kw["horizon_seconds"] = value*ours.resolution_seconds
    if field == "resolution_seconds":
        kw["horizon_seconds"] = value*ours.horizon_steps
    if field == "target_transform":
        kw["metric_scale"] = "log1p"
    theirs = replace(theirs, **kw)
    d = B.decide(ours, theirs)
    assert d["mode"] == "NOT_COMPARABLE" and field in d["fields_that_differ"]


def test_a_declared_contrast_under_one_estimand_stays_comparable_within_it():
    ours, theirs = _twin()
    ours = replace(ours, varying_factors=("permitted_inputs",), estimand="effect of calendar inputs at h60")
    theirs = replace(theirs, permitted_inputs="the 7 declared channels + 4 calendar channels",
                     varying_factors=("permitted_inputs",), estimand="effect of calendar inputs at h60")
    d = B.decide(ours, theirs)
    assert d["mode"] == "REPRODUCTION" and d["declared_contrast"] == ["permitted_inputs"]
    other = replace(theirs, estimand="something else")
    assert B.decide(ours, other)["mode"] == "NOT_COMPARABLE"


def _reference_root(tmp_path, ours, *, seeds=(1,), arm="gru", complete=True):
    """A closed reference run in the runners' layout: sealed design with our contract, cells of a named arm, receipts,
    arrays + record per cell, the DATA the table reads, and a warehouse stub with the accepted artifact chain."""
    import hashlib
    import numpy as np
    root = tmp_path/"ref"; root.mkdir()
    rng = np.random.default_rng(0)
    n, h = ours.source["evaluation_origins"], 60
    Y = np.abs(rng.normal(1.0, 0.5, n+2*h)); o = np.arange(h, h+n)
    np.savez(root/"BLOCK_DATA.npz", Y=Y, common_eval=o, horizon=np.array([h]), target_channel=np.array([0]), scaler_sd=np.array([ours.source["sd_train"]]),
             scaler_mean=np.array([1.0]), row_offset=np.array([0]))
    (root/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": "d"*64, "data_sha256": hashlib.sha256((root/"BLOCK_DATA.npz").read_bytes()).hexdigest()}))
    cells = [{"cell_id": f"{arm}_s{s}", "arm": arm, "seed": s} for s in seeds] + [{"cell_id": "modular_s1", "arm": "modular", "seed": 1}]
    block = ours.to_design_block(comparability={**B.decide(ours, B.gasparin_2019()), "comparator_state": "NONE"})
    design = {"schema": "df_e1_block_design.v1", "benchmark_contract": block, "cells": cells, "source_run": {"input_columns": ["Global_active_power"], "target_channel": 0}}
    design["design_sha256"] = _load("df_mod_e0").sha_obj(design)                                   # a SEALED reference: its digest recomputes
    (root/"DESIGN.json").write_text(json.dumps(design))
    (root/"BLOCK_DATA.json").write_text(json.dumps({"design_sha256": design["design_sha256"], "data_sha256": hashlib.sha256((root/"BLOCK_DATA.npz").read_bytes()).hexdigest()}))
    receipts = {"prepare": {"campaign_sha256": "c"*64, "terminal_sha256": "q"*64}}
    held = {"prepare": {"terminal_sha256": "q"*64, "status": "COMPLETED", "config_sha256": design["design_sha256"],
                        "artifacts": [{"role": "data", "sha256": hashlib.sha256((root/"BLOCK_DATA.npz").read_bytes()).hexdigest()}]}}
    for c in cells:
        if not complete and c["seed"] == seeds[-1] and c["arm"] == arm:
            continue
        u = c["cell_id"]; (root/"attempts"/u).mkdir(parents=True)
        pred = Y[o+h] + rng.normal(0, 0.1, n)
        np.savez(root/"attempts"/u/"arrays.npz", pred=pred, y=Y[o+h], naive=Y[o], origins=o)
        sha = lambda f: hashlib.sha256((root/"attempts"/u/f).read_bytes()).hexdigest()
        (root/"attempts"/u/"cell.json").write_text(json.dumps({"arrays_sha256": sha("arrays.npz"), "scores": {"mae_kw": float(np.mean(np.abs(pred-Y[o+h])))}}))
        receipts[u] = {"campaign_sha256": "c"*64, "terminal_sha256": "t"*64}
        held[u] = {"terminal_sha256": "t"*64, "status": "COMPLETED", "config_sha256": design["design_sha256"], "tags": {"arm": c["arm"], "seed": str(c["seed"])},
                   "artifacts": [{"role": "predictions", "sha256": sha("arrays.npz")}, {"role": "record", "sha256": sha("cell.json")}]}
    (root/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": receipts}))
    return root, (lambda campaign: {"current": held})


def test_PROBE_a_contract_hash_plus_a_prepare_receipt_is_preparation_not_a_comparator(tmp_path):
    """Musashi's RP73 probe: DESIGN with only the matching contract hash and one prepare: COMPLETED receipt."""
    ours = B.household_ours()
    ref = tmp_path/"reference"; ref.mkdir()
    (ref/"DESIGN.json").write_text(json.dumps({"benchmark_contract": {"contract_sha256": ours.sha256()}}))
    (ref/"TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {"prepare": {"status": "COMPLETED"}}}))
    out = B.reference_evidence(ours, ref, reference_arm="gru")
    # RP90: the household task is HISTORICAL_DEV_ONLY; its lane verification is reported UNDER that disposition
    assert out["state"] == "HISTORICAL_DEV_ONLY"
    assert out["underlying"]["state"] == "PLANNED_REFERENCE" and "design digest" in out["underlying"]["why"]


def test_the_matched_lane_needs_a_named_arm_a_complete_population_and_the_accepted_chain(tmp_path):
    ours, theirs = _twin()
    # the lane rules are exercised under an ACTIVE task id (RP90): a household task would be HISTORICAL_DEV_ONLY
    ours = replace(ours, task_id=B.ACTIVE_BENCHMARK_TASK_IDS[0], source={**ours.source, "evaluation_origins": 400})
    theirs = replace(theirs, horizon_steps=96, horizon_seconds=96*60)
    assert B.decide(ours, theirs, reference_run=tmp_path)["comparator_state"] == "PLANNED_REFERENCE"
    root, wh = _reference_root(tmp_path, ours, seeds=(1, 2))
    assert B.reference_evidence(ours, root)["state"] == "PLANNED_REFERENCE"                                   # no arm named
    assert B.reference_evidence(ours, root, reference_arm="gru")["state"] == "LOCALLY_CHECKED_REFERENCE"      # no warehouse read
    d = B.decide(ours, theirs, reference_run=root, reference_arm="gru", warehouse=wh)
    assert d["mode"] == "MATCHED_DOMAIN_COMPARISON" and d["comparator_state"] == "VERIFIED_COMPARATOR"
    assert set(d["reference_evidence"]["derived_mae_z"]) == {"gru_s1", "gru_s2"} and d["reference_evidence"]["n_evaluated"] == 400
    # a partial population (one seed without an accepted forecast) is not a comparator
    root2, wh2 = _reference_root(tmp_path/"p", ours, seeds=(1, 2), complete=False) if (tmp_path/"p").mkdir() is None else (None, None)
    e = B.reference_evidence(ours, root2, reference_arm="gru", warehouse=wh2)
    assert e["state"] == "PLANNED_REFERENCE" and "incomplete" in e["why"]
    # the accepted chain missing its record anchor is not a comparator either
    def no_record(campaign):
        held = wh(campaign)["current"]
        return {"current": {u: {**r, "artifacts": [a for a in r["artifacts"] if a["role"] != "record"]} for u, r in held.items()}}
    e = B.reference_evidence(ours, root, reference_arm="gru", warehouse=no_record)
    assert e["state"] == "PLANNED_REFERENCE" and "do not verify" in e["why"]                        # a new-result chain without its record anchor


# --- affine re-expression ------------------------------------------------------------------------------

def test_affine_reexpression_reproduces_and_refuses():
    ok = B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "zscore", "sd": 2.0},
                               same_target=True, same_population=True, same_horizon=True)
    assert ok["ok"] and ok["value"] == pytest.approx(0.25)
    for bad in ({"kind": "zscore", "sd": 0.0}, {"kind": "zscore", "sd": float("nan")}, {"kind": "log1p"}, {"kind": "zscore"}):
        assert not B.affine_reexpression(0.5, from_transform=bad, to_transform={"kind": "identity"},
                                         same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(float("inf"), from_transform={"kind": "identity"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=True, same_horizon=True)["ok"]
    assert not B.affine_reexpression(0.5, from_transform={"kind": "identity"}, to_transform={"kind": "identity"},
                                     same_target=True, same_population=False, same_horizon=True)["ok"]


# --- require() and bind() ------------------------------------------------------------------------------

def _good_block():
    ours = B.household_ours()
    return ours.to_design_block(comparability={**B.decide(ours, B.gasparin_2019())})


def test_require_checks_digest_mode_and_decision_identity():
    block = _good_block()
    assert B.require({"benchmark_contract": block}).task_id == block["task_id"]
    stale = copy.deepcopy(block); stale["target"] = "Voltage"                       # edited, digest not recomputed
    with pytest.raises(B.ContractRefusal, match="does not recompute"):
        B.require({"benchmark_contract": stale})
    foreign = copy.deepcopy(block); foreign["target"] = "Voltage"
    foreign["contract_sha256"] = B.BenchmarkContract.from_block(foreign).sha256()   # consistently re-digested
    with pytest.raises(B.ContractRefusal, match="another contract"):               # but the decision was for ours
        B.require({"benchmark_contract": foreign})
    mode = copy.deepcopy(block); mode["comparability"]["mode"] = "BEST_IN_CLASS"
    with pytest.raises(B.ContractRefusal, match="decided from fields"):
        B.require({"benchmark_contract": mode})


def test_bind_refuses_a_contract_that_does_not_match_the_prepared_data():
    ours = B.household_ours()
    data = {"input_columns": ["a", "b", "Global_active_power"], "target_channel": 2, "horizon": 60, "window": 60,
            "panel_sha256": ours.source["panel_sha256"], "scaler": {"sd": [1.0, 1.0, ours.source["sd_train"]]},
            "enumerator": {"validation": {"admissible": 10020}}}
    assert B.bind(ours, data)["bound"]
    for change, key, value in (("target", "target_channel", 0), ("horizon", "horizon", 72),
                               ("window", "window", 1440), ("panel", "panel_sha256", "f"*64),
                               ("population", "enumerator", {"validation": {"admissible": 9000}})):
        bad = dict(data); bad[key] = value
        with pytest.raises(B.ContractRefusal, match="does not bind"):
            B.bind(ours, bad)
    degenerate = dict(data); degenerate["scaler"] = {"sd": [1.0, 1.0, 0.0]}
    with pytest.raises(B.ContractRefusal, match="finite positive sd"):
        B.bind(ours, degenerate)


# --- the real entry points, on the real design ------------------------------------------------------------

@pytest.mark.skipif(not (SOURCE/"DESIGN.json").is_file(), reason="the successor run root is not on this host")
def test_PROBE_the_real_factorial_validator_refuses_a_foreign_task_even_when_re_digested():
    H = _load("df_e1_huber")
    d = H.seal(SOURCE)
    H.validate(d)                                                            # the genuine design passes
    foreign = copy.deepcopy(d)
    foreign["benchmark_contract"]["target"] = "unrelated_target"
    foreign["benchmark_contract"]["horizon_steps"] = 72
    foreign["benchmark_contract"]["horizon_seconds"] = 72*60
    foreign["benchmark_contract"]["contract_sha256"] = B.BenchmarkContract.from_block(foreign["benchmark_contract"]).sha256()
    foreign["benchmark_contract"]["comparability"]["ours_sha256"] = foreign["benchmark_contract"]["contract_sha256"]
    foreign["design_sha256"] = H.P._module("df_mod_e0").sha_obj({k: v for k, v in foreign.items() if k != "design_sha256"})
    with pytest.raises(B.ContractRefusal, match="does not bind"):            # consistent everywhere, foreign to the data
        H.validate(foreign)
    stale = copy.deepcopy(d)
    stale["benchmark_contract"]["target"] = "unrelated_target"
    stale["design_sha256"] = H.P._module("df_mod_e0").sha_obj({k: v for k, v in stale.items() if k != "design_sha256"})
    with pytest.raises(B.ContractRefusal, match="does not recompute"):        # inner digest stale
        H.validate(stale)


@pytest.mark.skipif(not (SOURCE/"DESIGN.json").is_file(), reason="the successor run root is not on this host")
def test_the_phase_runner_binds_before_it_acquires(tmp_path):
    P = _load("df_e1_phase1")
    design = P.seal(SOURCE)
    assert design["benchmark_contract"]["schema"] == B.SCHEMA
    stripped = {k: v for k, v in design.items() if k != "benchmark_contract"}
    with pytest.raises(B.ContractRefusal, match="without a benchmark contract"):
        P.run(stripped, root=tmp_path/"r", run_id="x", source_run=SOURCE, gov_url="http://127.0.0.1:1",
              api_key_file=tmp_path/"none", lake="public_panels", resource="r", cost_pilot_only=True)


# --- RP90: HISTORICAL_DEV_ONLY in the real selection paths ------------------------------------------------------

def test_RP90_the_disposition_is_recomputed_from_the_task_id_and_only_the_official_ecl_is_active():
    assert B.disposition("household_W60_h60")["disposition"] == "HISTORICAL_DEV_ONLY"
    assert B.disposition({"benchmark_contract": {"task_id": "household_W60_h60"}})["disposition"] == "HISTORICAL_DEV_ONLY"
    assert B.disposition("fx_eurusd_1h")["disposition"] == "DEFERRED_MANDATORY_STAGE"
    assert B.disposition(None)["disposition"] == "HISTORICAL_DEV_ONLY" and B.disposition({})["disposition"] == "HISTORICAL_DEV_ONLY"
    assert B.disposition("ecl321_official_tsl")["disposition"] == "ACTIVE"
    assert "HISTORICAL_DEV_ONLY" in B.COMPARATOR_STATES


def test_RP90_an_old_low_error_household_reference_never_becomes_a_comparator_however_well_it_verifies(tmp_path):
    """Musashi/owner rule: an old low-error result cannot enter the active reference ranking. The genuine household reference
    (accepted chain, complete seeds, tags) verifies underneath — and is reported under HISTORICAL_DEV_ONLY, opening no lane."""
    ours, theirs = _twin()
    ours = replace(ours, source={**ours.source, "evaluation_origins": 400})
    theirs = replace(theirs, horizon_steps=96, horizon_seconds=96*60)
    root, wh = _reference_root(tmp_path, ours, seeds=(1, 2))
    e = B.reference_evidence(ours, root, reference_arm="gru", warehouse=wh)
    assert e["state"] == "HISTORICAL_DEV_ONLY" and e["disposition"]["task_id"] == B.household_ours().task_id
    assert e["underlying"]["state"] == "VERIFIED_COMPARATOR"                       # history stays queryable, verified underneath
    d = B.decide(ours, theirs, reference_run=root, reference_arm="gru", warehouse=wh)
    assert d["mode"] == "NOT_COMPARABLE" and d["comparator_state"] == "HISTORICAL_DEV_ONLY" and "opens no lane" in d["why"]
    # the same protocol (a reproduction lane) under a historical task: no comparator state either
    d2 = B.decide(ours, ours, reference_run=root, reference_arm="gru", warehouse=wh)
    assert d2["mode"] == "REPRODUCTION" and d2["comparator_state"] == "HISTORICAL_DEV_ONLY"
    # the active ranking recomputes the disposition from the task id: a flipped label on a cached row does not enter
    rows = [{"verified": True, "task_id": B.household_ours().task_id, "model_error": 0.0, "run": "old", "unit": "gru_s1", "disposition": "ACTIVE"},
            {"verified": True, "task_horizon_split": B.household_ours().task_id + " | h=60", "model_error": 0.0, "run": "older", "unit": "gru_s2"},
            {"verified": True, "task_id": "ecl321_official_tsl", "model_error": 0.9, "run": "ecl", "unit": "h96_s1"},
            {"verified": False, "task_id": "ecl321_official_tsl", "model_error": 0.1, "run": "ecl", "unit": "h96_s2"}]
    ranking = B.active_ranking(rows)
    assert [r["unit"] for r in ranking] == ["h96_s1"]                                  # the 0.0-error household rows never rank; unverified never ranks
