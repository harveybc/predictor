"""R0-versus-R2 design contract (design layer only): typed refusals, R0-reuse predicate,
decision rule and the shape-proof branch-donor import. No pretraining, no fit, CPU only.

Acceptance rows exercised here: R2-01, R2-02, R2-03, R2-06, R2-08, R2-09, R2-12 (design layer)
and the contract halves of R2-04/R2-05/R2-07. Rows needing the future runner stay open.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import design as D, r2_design as R2

MATCHED = R2.MATCHED_DIR
CONTRACT = MATCHED / "R2_CONTRAST_DESIGN.json"
PROPOSAL = MATCHED / "R2_DESIGN_PROPOSAL.md"


def _contract():
    return json.loads(CONTRACT.read_text())


RULED = MATCHED / "R2_CONTRAST_DESIGN.ruled.json"


def _ruled_o1():
    """The technical lead's ruling (LEAD_RULING, review 93bff41d) as a complete objective block."""
    return R2.lead_objective()


def _ruled_o1_v0():
    """The ruling RECOMMENDED at 7488e899 (200 epochs, no budgets): refused since D1."""
    return {"option": "O1", "name": "autoencoder_reconstruction", "version": "1.0.0",
            "params": {"decoder_channels": 32, "loss": "mse"},
            "fit_settings": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
                             "batch_size": 256, "max_epochs": 200, "patience": 10, "min_delta": 1e-8,
                             "seed_rule": "6102026 + feature_index", "restore": "best_inner_epoch"},
            "ruling": {"reviewer": "test-reviewer", "document_sha256": "a" * 64}}


# ============================================================================ contract (R2-01/06/12)

def test_committed_contract_is_the_deterministic_derivation_and_is_preparing(tmp_path):
    built = R2.build_contract()
    assert CONTRACT.read_text() == json.dumps(built, indent=1, sort_keys=True) + "\n"
    assert R2.seal(MATCHED, CONTRACT)["idempotent"] is True
    with pytest.raises(R2.R2DesignError, match="R2_DESIGN_EXISTS"):
        other = tmp_path / "x.json"
        other.write_text("{}")
        R2.seal(MATCHED, other)
    doc = _contract()
    assert doc["pretraining"]["objective"] == R2.UNDECIDED == "UNDECIDED_REQUIRES_REVIEW"
    assert R2.validate_r2_design(doc) == {"status": "PREPARING", "objective": R2.UNDECIDED}
    with pytest.raises(R2.R2DesignError, match="R2_OBJECTIVE_UNDECIDED"):
        R2.validate_r2_design(doc, for_execution=True)
    report = D.validate_design_report(CONTRACT)
    assert report["status"] == "FAILED" and report["code"] == "R2_OBJECTIVE_UNDECIDED"
    # The matched arm designs themselves are untouched.
    assert D.validate_design_report(MATCHED / "EXPERIMENT.mg2m.json")["status"] == "PASS"
    shared = doc["shared"]
    assert shared["arm"] == "mg2m" and shared["parameter_total"] == 4882
    assert shared["population_sha256"].startswith("ba01f87a")
    assert shared["experiment_design_sha256"].startswith("d71d2b3c")
    assert shared["effective_config_sha256"].startswith("ad949872")
    assert shared["seed"] == 6102026 and shared["targets"] == "RAW_EXACT_T_PLUS_H_PRICE"
    assert doc["selection"]["comparison_sha256"].startswith("2ee85a12")
    assert sorted(doc["acceptance_tests"]) == [f"R2-{i:02d}" for i in range(1, 19)]
    assert doc["shared"]["fit_pinning"]["runtime"] == {"tensorflow": "2.21.0", "keras": "3.13.2"}
    assert PROPOSAL.exists()


def _set(doc, path, value):
    node = doc
    for key in path[:-1]:
        node = node[key]
    if value is _DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value


_DELETE = object()


@pytest.mark.parametrize(("path", "value", "code"), [
    (("shared", "population_sha256"), "0" * 64, "R2_POPULATION_MISMATCH"),
    (("shared", "scaler_fit_row_digest"), "0" * 64, "R2_POPULATION_MISMATCH"),
    (("shared", "masks"), "R2-only masks", "R2_POPULATION_MISMATCH"),
    (("shared", "targets"), "RECONSTRUCTED_DENOISED_TARGET", "R2_TARGET_NOT_RAW"),
    (("shared", "seed"), 1, "R2_TRAINING_CHANGE"),
    (("shared", "training", "learning_rate"), 3e-4, "R2_TRAINING_CHANGE"),
    (("shared", "parameter_total"), 4883, "R2_ARCHITECTURE_CHANGE"),
    (("shared", "experiment_design_sha256"), "0" * 64, "R2_ARCHITECTURE_CHANGE"),
    (("shared", "splits", "DEVELOPMENT", "end"), "2025-06-01T00:00:00Z", "R2_SPLIT_MISMATCH"),
    (("arms", "R2", "branches_trainable"), False, "R2_FROZEN_BRANCH"),
    (("arms", "R2", "non_branch_init"), "PRETRAINED_CORE", "R2_ARM_CHANGE"),
    (("selection", "decision"), "MG1_SELECTED", "R2_SELECTION_MISMATCH"),
    (("pretraining", "data", "development"), "SELECTION", "R2_PRETRAINING_SPLIT"),
    (("pretraining", "data", "fit"), "FIT_AND_DEVELOPMENT", "R2_PRETRAINING_SPLIT"),
    (("pretraining", "early_stopping", "monitor"), "DEVELOPMENT_LOSS", "R2_PRETRAINING_SPLIT"),
    (("pretraining", "donor_policy"), "SHAPE_MATCH_ACCEPTED", "R2_DONOR_POLICY"),
    (("pretraining", "donor_architecture", "params", "filters"), 8, "R2_DONOR_POLICY"),
    (("pretraining", "frozen_evaluations", "predictive", "use"), "GATE", "R2_EVALUATION_CHANGE"),
    (("pretraining", "objective_options"), ["O1", "O9"], "R2_OBJECTIVE_UNKNOWN"),
    (("r0_reference", "policy"), "ALWAYS_REUSE", "R2_R0_POLICY_CHANGE"),
    (("decision", "tie_tolerance"), 1e-6, "R2_DECISION_CHANGE"),
    (("out_of_scope",), ["TEST"], "R2_SCOPE"),
    (("core_pretraining",), {"objective": "masked"}, "R2_SCOPE"),
    (("cadence",), "quarterly", "R2_SCOPE"),
    (("status",), "READY", "R2_CONTRACT_MISMATCH"),
])
def test_every_frozen_field_change_is_a_typed_refusal(path, value, code):
    doc = _contract()
    _set(doc, path, value)
    doc["seal_sha256"] = R2._sha({k: v for k, v in doc.items() if k != "seal_sha256"})  # resealed
    with pytest.raises(R2.R2DesignError, match=code):
        R2.validate_r2_design(doc)


def test_seal_and_schema_refusals():
    doc = _contract()
    doc["seal_sha256"] = "0" * 64
    with pytest.raises(R2.R2DesignError, match="R2_SEAL"):
        R2.validate_r2_design(doc)
    with pytest.raises(R2.R2DesignError, match="R2_SCHEMA_REQUIRED"):
        R2.validate_r2_design(json.loads((MATCHED / "EXPERIMENT.mg2m.json").read_text()))


def test_the_contract_binds_the_sealed_matched_artifacts(tmp_path):
    copy_dir = tmp_path / "matched"
    shutil.copytree(MATCHED, copy_dir, ignore=shutil.ignore_patterns("review"))
    seal = json.loads((copy_dir / "POPULATION_SEAL.json").read_text())
    seal["population_digest"] = "1" * 64
    (copy_dir / "POPULATION_SEAL.json").write_text(json.dumps(seal))
    with pytest.raises(R2.R2DesignError, match="R2_POPULATION_MISMATCH"):
        R2.validate_r2_design(_contract(), matched_dir=copy_dir)
    cap = json.loads((copy_dir / "CAPACITY_MATCHED.json").read_text())
    cap["arms"]["mg2m"]["sealed_design_sha256"] = "2" * 64
    (copy_dir / "CAPACITY_MATCHED.json").write_text(json.dumps(cap))
    with pytest.raises(R2.R2DesignError, match="R2_RECIPE_MISMATCH"):
        R2.build_contract(copy_dir)


def test_the_committed_ruled_contract_carries_the_lead_ruling_and_is_executable(tmp_path):
    ruled = json.loads(RULED.read_text())
    assert ruled == R2.rule_objective(_contract(), _ruled_o1())
    assert R2.validate_r2_design(ruled, for_execution=True)["status"] == "READY_FOR_EXECUTION"
    assert D.validate_design_report(RULED)["status"] == "PASS"
    o = ruled["pretraining"]["objective"]
    assert (o["option"], o["name"], o["version"], o["params"]) == \
        ("O1", "autoencoder_reconstruction", "1.0.0", {"decoder_channels": 32, "loss": "mse"})
    fs = o["fit_settings"]
    assert (fs["optimizer"], fs["learning_rate"], fs["weight_decay"], fs["batch_size"], fs["max_epochs"],
            fs["patience"], fs["min_delta"], fs["max_updates"], fs["max_seconds"]) == \
        ("AdamW", 1e-3, 1e-4, 256, 100, 10, 1e-8, 100000, 86400.0)
    assert fs["restore"] == "best_monitored_epoch" and "6102026 + feature_index" in fs["seed_rule"]
    assert fs["monitor"] == "validation_loss_on_purged_INNER_inputs"
    assert fs["budget_stop"].startswith("TYPED_FAILURE")
    assert o["ruling"]["review_commit"] == "93bff41d"
    assert any("200" in d for d in o["disclosures"]) and any("twice" in d for d in o["disclosures"])
    assert any("DEVELOPMENT" in d for d in o["disclosures"])
    out = tmp_path / "ruled.json"
    assert R2.write_ruled(CONTRACT, out)["seal_sha256"] == ruled["seal_sha256"]
    assert out.read_text() == RULED.read_text()


def _edit(mutate):
    objective = _ruled_o1()
    mutate(objective)
    return R2.rule_objective(_contract(), objective)


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda o: o["fit_settings"].update(max_epochs=200), "R2_OBJECTIVE_NOT_EXECUTABLE"),   # D1
    (lambda o: o["fit_settings"].update(max_seconds=90000.0), "R2_OBJECTIVE_NOT_EXECUTABLE"),
    (lambda o: o["fit_settings"].update(patience=0), "R2_OBJECTIVE_NOT_EXECUTABLE"),
    (lambda o: o["fit_settings"].update(optimizer="Adam"), "R2_OBJECTIVE_NOT_EXECUTABLE"),
    (lambda o: o["fit_settings"].update(monitor="train_validation_mean"), "R2_OBJECTIVE_NOT_EXECUTABLE"),
    (lambda o: o["fit_settings"].pop("max_updates"), "R2_OBJECTIVE_UNKNOWN"),               # budgets fixed
    (lambda o: o["fit_settings"].pop("max_seconds"), "R2_OBJECTIVE_UNKNOWN"),
    (lambda o: o["fit_settings"].update(max_updates=R2.UNDECIDED), "R2_OBJECTIVE_UNDECIDED"),
    (lambda o: o["params"].update(loss="huber"), "R2_OBJECTIVE_UNKNOWN"),                   # identity sha
    (lambda o: o["params"].pop("loss"), "R2_OBJECTIVE_UNKNOWN"),
    (lambda o: o.update(option="O3", name="masked_point_reconstruction", version=None), "R2_OBJECTIVE_UNKNOWN"),
    (lambda o: o.update(name="ts2vec_contrastive"), "R2_OBJECTIVE_UNKNOWN"),
    (lambda o: o.update(identity_sha256="0" * 64), "R2_OBJECTIVE_UNKNOWN"),
    (lambda o: o["ruling"].update(review_document_sha256="0" * 64), "R2_OBJECTIVE_UNREVIEWED"),
    (lambda o: o["ruling"].pop("review_commit"), "R2_OBJECTIVE_UNREVIEWED"),
    (lambda o: o.update(disclosures=[]), "R2_OBJECTIVE_UNREVIEWED"),
])
def test_a_ruling_must_be_executable_complete_and_reviewed(mutate, code):
    with pytest.raises(R2.R2DesignError, match=code):
        R2.validate_r2_design(_edit(mutate), for_execution=True)


def test_the_7488e899_recommendation_is_refused_and_status_follows_the_objective():
    with pytest.raises(R2.R2DesignError, match="R2_OBJECTIVE_UNKNOWN"):
        R2.validate_r2_design(R2.rule_objective(_contract(), _ruled_o1_v0()), for_execution=True)
    stale = R2.rule_objective(_contract(), _ruled_o1())
    stale["status"] = "FROZEN_BEFORE_FIT_DESIGN_ONLY"
    with pytest.raises(R2.R2DesignError, match="R2_CONTRACT_MISMATCH"):
        R2.validate_r2_design(stale)


def test_cli_validate_and_seal(tmp_path, capsys):
    assert R2.main(["validate", "--design", str(CONTRACT), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREPARING"
    assert R2.main(["validate", "--design", str(CONTRACT), "--for-execution", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "R2_OBJECTIVE_UNDECIDED"
    assert R2.main(["seal", "--out", str(tmp_path / "c.json"), "--json"]) == 0
    assert (tmp_path / "c.json").read_text() == CONTRACT.read_text()


# ============================================================================ R0 reuse (R2-08)

CODE = {"tools/i6dmo/train.py": "c" * 64, "tools/i6dmo/model.py": "d" * 64}
SCALES = "e" * 64
MONITOR = {"procedure": "i6dmo.monitor_eval.v2", "partition": "sequential_row_id_fixed",
           "input_materialization": "i6dmo.monitor_input.v1", "monitor_forward": {"id": "f"}}


def _agreeing_receipt(shared):
    return {"status": "COMPLETED", "scientific": True, "overrides": {}, "device": "gpu",
            "identity": {"design_sha256": shared["experiment_design_sha256"],
                         "effective_config_sha256": shared["effective_config_sha256"],
                         "common_population_sha256": shared["population_sha256"],
                         "matched_arm": "mg2m", "capacity_sha256": shared["capacity_sha256"],
                         "scales_sha256": SCALES, "code_files_sha256": dict(CODE)},
            "matched": {"scaler": {"fit_row_digest": shared["scaler_fit_row_digest"]}},
            "resources": {"tensorflow_version": shared["runtime"]["tensorflow"]}}


def _decide(receipt, monitor=MONITOR, verdict="ACCEPT"):
    return R2.r0_reuse_decision(receipt, dict(monitor), _contract(), expected_code_files=dict(CODE),
                                expected_scales_sha256=SCALES, current_monitor_identity=MONITOR,
                                r1_verdict=verdict)


def test_r0_is_reused_only_when_every_component_agrees():
    shared = _contract()["shared"]
    ok = _decide(_agreeing_receipt(shared))
    assert ok["decision"] == "REUSE_R0" and ok["failed"] == []
    assert tuple(ok["components"]) == R2.R0_REUSE_COMPONENTS


_MUTATIONS = {
    "status_completed": lambda r: r.update(status="FAILED_TYPED"),
    "scientific": lambda r: r.update(scientific=False),
    "no_overrides": lambda r: r.update(overrides={"max_epochs": 2}),
    "device_gpu": lambda r: r.update(device="cpu"),
    "design_sha256": lambda r: r["identity"].update(design_sha256="0" * 64),
    "effective_config_sha256": lambda r: r["identity"].update(effective_config_sha256="0" * 64),
    "common_population_sha256": lambda r: r["identity"].update(common_population_sha256="0" * 64),
    "matched_arm": lambda r: r["identity"].update(matched_arm="mg1"),
    "capacity_sha256": lambda r: r["identity"].update(capacity_sha256="0" * 64),
    "scales_sha256": lambda r: r["identity"].update(scales_sha256="0" * 64),
    "scaler_fit_row_digest": lambda r: r["matched"]["scaler"].update(fit_row_digest="0" * 64),
    "runtime_tensorflow": lambda r: r["resources"].update(tensorflow_version="2.18.0"),
    "code_files_sha256": lambda r: r["identity"]["code_files_sha256"].update(
        {"tools/i6dmo/model.py": "0" * 64}),
}


@pytest.mark.parametrize("component", R2.R0_REUSE_COMPONENTS)
def test_any_disagreeing_component_requires_a_fresh_r0(component):
    receipt = _agreeing_receipt(_contract()["shared"])
    monitor, verdict = MONITOR, "ACCEPT"
    if component == "monitor_identity":
        monitor = dict(MONITOR, partition="permuted")
    elif component == "r1_donor_verifier":
        verdict = "REJECT:FAKE_RECEIPT"
    else:
        _MUTATIONS[component](receipt)
    out = _decide(receipt, monitor, verdict)
    assert out["decision"] == "FRESH_R0_REQUIRED" and out["failed"] == [component]


def test_a_full_population_or_unverified_receipt_is_never_r0():
    shared = _contract()["shared"]
    closed = _agreeing_receipt(shared)
    for key in ("common_population_sha256", "matched_arm", "capacity_sha256"):
        closed["identity"].pop(key)
    out = _decide(closed)
    assert out["decision"] == "FRESH_R0_REQUIRED"
    assert {"common_population_sha256", "matched_arm", "capacity_sha256"} <= set(out["failed"])
    assert _decide(None, {}, None)["decision"] == "FRESH_R0_REQUIRED"


# ============================================================================ decision (R2-09)

def _metrics(score, *, failed=(), arm="mg2m", rows="r" * 64, capacity="5b62", import_sha=None):
    heads = [2, 3, 4, 5, 6, 12]
    selection = {"status": "INCOMPLETE" if failed else "COMPLETE", "score": None if failed else score,
                 "heads": heads, "failed_heads": [{"horizon": h} for h in failed]}
    identity = {"common_population_sha256": "p" * 64}
    if import_sha is not None:
        identity["r2_import_sha256"] = import_sha
    return {"per_horizon": {str(h): {"relative_error": None if h in failed else score} for h in heads},
            "matched": {"arm": arm, "common_population_sha256": "p" * 64,
                        "development_row_ids_sha256": rows, "development_head_mask_sha256": "m" * 64,
                        "development_counts": {"2": 10}, "capacity": {"capacity_sha256": capacity},
                        "scientific_fit": True, "fit_device": "gpu", "fit_runtime": "2.21.0",
                        "fit_gpu_class": "GPU-CLASS-A", "fit_identity": identity,
                        "selection": selection}}


def _record():
    record = {"schema": "i6dmo.r2_import.v1", "arm": "mg2m", "trainable": True,
              "donor_terminals": {f"f{i:02d}": f"{i:02d}" * 32 for i in range(19)},
              "non_branch_weights_sha256": "n" * 64, "branches": {}}
    record["import_sha256"] = R2._import_record_sha(record)
    return record


def _pair(r0_score, r2_score, **kw):
    rec = _record()
    return _metrics(r0_score), _metrics(r2_score, import_sha=rec["import_sha256"], **kw), rec


def test_decision_rule_lower_score_ties_retain_r0_and_failed_heads_are_incomplete():
    r0, r2, rec = _pair(0.001, -0.002)
    assert R2.decide_r2(r0, r2, r2_import_record=rec)["selected"] == "R2"
    r0, r2, rec = _pair(0.001, 0.001 - 5e-9)
    tie = R2.decide_r2(r0, r2, r2_import_record=rec)
    assert tie["selected"] == "R0" and tie["decision"] == "R0_RETAINED"
    r0, r2, rec = _pair(-0.001, 0.0)
    assert R2.decide_r2(r0, r2, r2_import_record=rec)["selected"] == "R0"
    r0, r2, rec = _pair(0.001, -0.5, failed=(12,))
    out = R2.decide_r2(r0, r2, r2_import_record=rec)
    assert out["decision"] == "INCOMPLETE_COMPARISON" and out["selected"] is None
    assert out["missing"] == [{"horizon": 12, "arm": "R2"}]
    for kw, code in (({"rows": "x" * 64}, "R2_POPULATION_MISMATCH"), ({"arm": "mg1"}, "R2_ARM_CHANGE"),
                     ({"capacity": "other"}, "R2_ARCHITECTURE_CHANGE")):
        r0, r2, rec = _pair(0.0, 0.0, **kw)
        with pytest.raises(R2.R2DesignError, match=code):
            R2.decide_r2(r0, r2, r2_import_record=rec)


@pytest.mark.parametrize(("edit", "code"), [
    (lambda r0, r2, rec: r2["matched"].update(scientific_fit=False), "R2_NON_SCIENTIFIC_FIT"),
    (lambda r0, r2, rec: r0["matched"].update(fit_device="cpu"), "R2_NON_SCIENTIFIC_FIT"),
    (lambda r0, r2, rec: r2["matched"].update(fit_runtime="2.18.0"), "R2_FIT_PINNING"),
    (lambda r0, r2, rec: r2["matched"].update(fit_gpu_class="GPU-CLASS-B"), "R2_FIT_PINNING"),
    (lambda r0, r2, rec: [m["matched"].update(fit_gpu_class=None) for m in (r0, r2)], "R2_FIT_PINNING"),
    (lambda r0, r2, rec: rec.clear(), "R2_IMPORT_RECORD_MISSING"),
    (lambda r0, r2, rec: rec["donor_terminals"].pop("f03"), "R2_IMPORT_RECORD_INVALID"),
    (lambda r0, r2, rec: rec.update(trainable=False), "R2_IMPORT_RECORD_INVALID"),
    (lambda r0, r2, rec: r2["matched"]["fit_identity"].pop("r2_import_sha256"), "R2_IMPORT_NOT_BOUND"),
    (lambda r0, r2, rec: r0["matched"]["fit_identity"].update(r2_import_sha256=rec["import_sha256"]),
     "R2_R0_HAS_IMPORT"),
])
def test_decision_integrity_is_strict_by_default(edit, code):
    r0, r2, rec = _pair(0.001, -0.002)
    edit(r0, r2, rec)
    with pytest.raises(R2.R2DesignError, match=code):
        R2.decide_r2(r0, r2, r2_import_record=rec or None)


def test_two_plain_r0_fits_are_never_r0_vs_r2_even_for_engineering():
    with pytest.raises(R2.R2DesignError, match="R2_IMPORT_RECORD_MISSING"):
        R2.decide_r2(_metrics(-0.01), _metrics(-0.02))
    r0, r2, rec = _pair(0.001, -0.002)
    r0["matched"].update(scientific_fit=False, fit_device="cpu", fit_gpu_class=None)
    assert R2.decide_r2(r0, r2, r2_import_record=rec, require_scientific=False)["selected"] == "R2"
    with pytest.raises(R2.R2DesignError, match="R2_NON_SCIENTIFIC_FIT"):
        R2.decide_r2(r0, r2, r2_import_record=rec, require_scientific=0)     # only literal False relaxes


# ============================================================================ donor import (R2-02/03)

tf = pytest.importorskip("tensorflow")
from tools.i6dmo import model as M, train as T  # noqa: E402

POP = "ba01f87a14e1ceb39d1a5770d4d00f3803cd29748c64edf7254c1e6c57b249d1"



def _scaler_sha(seed=0):
    rng = np.random.default_rng(seed)
    return M.scaler_identity_sha256([f"f{i}" for i in range(19)], rng.normal(size=19),
                                    rng.uniform(0.5, 2.0, size=19), "8" * 64, 1234)


def _bundle(arm="mg2m"):
    return M.build_dmo(T.effective_from_design(MATCHED / f"EXPERIMENT.{arm}.json")["model"])


def _context(bundle, *, objective=None, population=POP, scaler=None, provenance=None):
    return M.branch_donor_context(bundle, population_sha256=population,
                                  scaler_sha256=scaler or _scaler_sha(),
                                  objective=objective if objective is not None else _ruled_o1(),
                                  provenance=provenance)


def _pretend_pretrained(bundle):
    """Stand-in for future pretrained weights: deterministic edits of each branch (no fitting)."""
    for i, model in enumerate(bundle.branch_models.values()):
        model.set_weights([w + 0.01 * (i + 1) for w in model.get_weights()])
    return bundle


@pytest.fixture(scope="module")
def donors():
    source = _pretend_pretrained(_bundle())
    ctx = _context(source)
    return {f: {"manifest": M.branch_donor_manifest(source, f, ctx),
                "weights": [np.array(w) for w in source.branch_models[f].get_weights()]}
            for f in source.config["features"]}


@pytest.fixture(scope="module")
def target():
    return _bundle()


# ---------------------------------------------------------------- authenticated terminals (D3)

INNER_START = 1_700_000_000


def _tiny_corpus(feature_index, n_fit=24, n_inner=12, seed=0):
    rng = np.random.default_rng(seed)
    fit_x = rng.normal(size=(n_fit, 13, 19))
    inner_x = rng.normal(size=(n_inner, 13, 19))
    fit_ids = np.arange(n_fit, dtype=np.int64) * 3600 + 1_600_000_000
    inner_ids = np.arange(n_inner, dtype=np.int64) * 3600 + INNER_START + 25 * 3600
    slots = inner_ids[:, None] + np.arange(-24, 1, 2)[None, :] * 3600
    slots[:2, 0] = INNER_START - 3600                      # two windows overlap FIT bars: purged
    return M.branch_pretraining_corpus(fit_x, fit_ids, inner_x, inner_ids, slots,
                                       feature_index=feature_index, scaler_mean=np.zeros(19),
                                       scaler_std=np.ones(19), inner_start_epoch=INNER_START)


def _provenance(features, corpus, campaign="i6dmo-r2-test"):
    return {"campaign_id": campaign, "ruling_seal_sha256": json.loads(RULED.read_text())["seal_sha256"],
            "code_pin": {"commit": "test-pin"},
            "units": {f: f"r2.pretrain.{i:02d}" for i, f in enumerate(features)},
            "corpus": {f: corpus[f]["sha256"] for f in features}}


def _write_terminals(source, root, provenance, *, stop_reason="patience"):
    ctx = _context(source, provenance=provenance)
    refs = {}
    for i, f in enumerate(source.config["features"]):
        pretrained = {"applied_seed": M.r2_seed(i),
                      "weights": [np.array(w) for w in source.branch_models[f].get_weights()],
                      "fit": {"stop_reason": stop_reason, "stop_class": "no_improvement",
                              "selected_epoch": 1}, "reconstruction": {"state": "MEASURED"}}
        refs[f] = {"terminal": M.write_branch_donor_terminal(
            root / f"unit_{i:02d}", bundle=source, feature=f, context=ctx,
            unit_id=provenance["units"][f], corpus=_CORPUS[f], pretrained=pretrained)}
    return refs


_CORPUS = {}


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    source = _pretend_pretrained(_bundle())
    features = source.config["features"]
    for i, f in enumerate(features):
        _CORPUS[f] = _tiny_corpus(i)[2]
    prov = _provenance(features, _CORPUS)
    root = tmp_path_factory.mktemp("terminals")
    return {"source": source, "prov": prov, "refs": _write_terminals(source, root, prov), "root": root}


def test_genuine_terminal_donors_import_and_bind_all_19_terminal_digests(genuine):
    r0, r2 = _bundle(), _bundle()
    record = M.import_branch_donors(r2, copy.deepcopy(genuine["refs"]),
                                    _context(r2, provenance=genuine["prov"]))
    assert record["trainable"] is True and record["trainable_parameters"] == 4882
    assert record["non_branch_weights_sha256"] == M._non_branch_digest(r0)
    assert sorted(record["donor_terminals"]) == sorted(r2.config["features"])
    assert R2.verify_import_record(record) == record["import_sha256"]
    for feature in r2.config["features"]:
        for a, b in zip(r2.branch_models[feature].get_weights(),
                        genuine["source"].branch_models[feature].get_weights()):
            assert np.array_equal(a, b)
        assert r2.branch_models[feature].trainable
    assert M.capacity_record(r2)["capacity_sha256"] == M.capacity_record(r0)["capacity_sha256"]
    for a, b in zip(r2.core_model.get_weights(), r0.core_model.get_weights()):
        assert np.array_equal(a, b)
    x = [np.zeros((2, 13, 19), np.float32), np.tile(np.arange(-24.0, 1.0, 2.0, dtype=np.float32), (2, 1)),
         np.zeros((2, 13), np.float32)]
    assert not np.array_equal(M.predict_matrix(r0.forecast_model, x), M.predict_matrix(r2.forecast_model, x))


def test_a_self_declared_manifest_never_passes_even_with_a_perfect_identity(genuine, target):
    forged = {f: {"manifest": M.branch_donor_manifest(target, f, _context(target)),
                  "weights": [np.array(w) for w in target.branch_models[f].get_weights()]}
              for f in target.config["features"]}
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    with pytest.raises(M.DonorImportError, match="R2_DONOR_PROVENANCE_MISSING"):
        M.import_branch_donors(target, forged, _context(target, provenance=genuine["prov"]))
    with pytest.raises(M.DonorImportError, match="R2_DONOR_PROVENANCE_MISSING"):
        M.import_branch_donors(target, genuine["refs"], _context(target))          # no provenance
    assert all(np.array_equal(a, b) for a, b in zip(before, target.forecast_model.get_weights()))


def _rewrite_terminal(ref, edit):
    path = Path(ref["path"])
    doc = json.loads(path.read_text())
    edit(doc, path)
    path.write_text(json.dumps(doc, sort_keys=True, indent=1))
    return {"path": str(path), "sha256": M._file_hash(path)}


@pytest.mark.parametrize(("case", "code"), [
    ("stale_reference", "R2_DONOR_TERMINAL_DIGEST"),
    ("other_campaign", "R2_DONOR_PROVENANCE_MISMATCH"),
    ("other_seed", "R2_DONOR_PROVENANCE_MISMATCH"),
    ("other_code_pin", "R2_DONOR_PROVENANCE_MISMATCH"),
    ("other_ruling", "R2_DONOR_PROVENANCE_MISMATCH"),
    ("other_corpus", "R2_DONOR_CORPUS_MISMATCH"),
    ("budget_stop", "R2_PRETRAINING_BUDGET_STOP"),
    ("foreign_artifact", "R2_DONOR_WEIGHTS_DIGEST"),
])
def test_terminal_provenance_refusals(genuine, target, tmp_path, case, code):
    features = target.config["features"]
    f0 = features[0]
    root = tmp_path / "copy"
    shutil.copytree(genuine["root"], root)
    refs = {f: {"terminal": {"path": str(root / Path(r["terminal"]["path"]).relative_to(genuine["root"])),
                             "sha256": r["terminal"]["sha256"]}} for f, r in genuine["refs"].items()}
    prov = copy.deepcopy(genuine["prov"])
    ref = refs[f0]["terminal"]
    if case == "stale_reference":
        Path(ref["path"]).write_text(Path(ref["path"]).read_text() + " ")
    elif case in ("other_campaign", "other_seed", "other_code_pin", "other_ruling", "budget_stop"):
        key, value = {"other_campaign": ("campaign_id", "another-campaign"),
                      "other_seed": ("applied_seed", 42), "other_code_pin": ("code_pin", {"commit": "x"}),
                      "other_ruling": ("ruling_seal_sha256", "0" * 64),
                      "budget_stop": ("fit", {"stop_reason": "max_seconds", "stop_class": "budget"})}[case]
        refs[f0]["terminal"] = _rewrite_terminal(ref, lambda d, p: d.update({key: value}))
    elif case == "other_corpus":
        prov["corpus"][f0] = _tiny_corpus(0, seed=1)[2]["sha256"]
    elif case == "foreign_artifact":
        foreign = _pretend_pretrained(_bundle("mg1")).branch_models[f0].get_weights()
        np.savez(Path(ref["path"]).parent / "donor.npz", kernel=foreign[0] + 1.0, bias=foreign[1])
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    with pytest.raises(M.DonorImportError, match=code):
        M.import_branch_donors(target, refs, _context(target, provenance=prov))
    assert all(np.array_equal(a, b) for a, b in zip(before, target.forecast_model.get_weights()))


def test_import_rolls_back_every_branch_when_setting_weights_fails(genuine, monkeypatch):
    r0, r2 = _bundle(), _bundle()
    features = r2.config["features"]
    victim = r2.branch_models[features[5]]

    def boom(*a, **k):
        raise RuntimeError("simulated failure while setting weights")
    monkeypatch.setattr(victim, "set_weights", boom)
    with pytest.raises(M.DonorImportError, match="R2_DONOR_IMPORT_ROLLED_BACK"):
        M.import_branch_donors(r2, copy.deepcopy(genuine["refs"]), _context(r2, provenance=genuine["prov"]))
    for f in features:
        assert all(np.array_equal(a, b) for a, b in zip(r2.branch_models[f].get_weights(),
                                                         r0.branch_models[f].get_weights()))


def test_learned_corpus_digest_binds_fit_and_kept_inner_windows_only():
    fit, sel, corpus = _tiny_corpus(3)
    assert fit.shape == (24, 13, 1) and sel.shape == (10, 13, 1)
    assert corpus["counts"] == {"fit": 24, "selection": 10, "purged": 2}
    assert corpus["feature_index"] == 3 and corpus["purge_rule"] == M.R2_PURGE_RULE
    assert _tiny_corpus(3)[2]["sha256"] == corpus["sha256"]
    assert _tiny_corpus(4)[2]["sha256"] != corpus["sha256"]                  # feature index bound


# ---------------------------------------------------------------- seeded pretraining unit (D2)

def _test_objective(**fit):
    objective = _ruled_o1()
    objective["fit_settings"] = dict(objective["fit_settings"], max_epochs=3, **fit)   # TEST ONLY
    return objective


def test_pretraining_is_seeded_before_the_autoencoder_and_replays_bitwise_on_cpu():
    fit, sel, _ = _tiny_corpus(3)
    runs = []
    for noise in (0, 1):
        bundle = _bundle()
        tf.random.normal((7,))                                   # perturb global RNG state first
        np.random.default_rng(noise).normal(size=5)
        feature = bundle.config["features"][3]
        before = [np.array(w) for w in bundle.branch_models[feature].get_weights()]
        out = M.pretrain_branch(bundle, feature, fit, sel, _test_objective())
        assert out["applied_seed"] == 6102029 and out["fit"]["stop_reason"] not in M.R2_BUDGET_STOPS
        assert out["reconstruction"]["state"] == "MEASURED"
        assert not all(np.array_equal(a, b) for a, b in zip(before, out["weights"]))
        runs.append(out)
    for a, b in zip(runs[0]["weights"], runs[1]["weights"]):
        assert np.array_equal(a, b)                                # bitwise CPU replay
    assert runs[0]["fit"]["history"] == runs[1]["fit"]["history"]


def test_a_budget_stop_is_a_typed_failure_and_undecided_objective_never_pretrains():
    fit, sel, _ = _tiny_corpus(2)
    bundle = _bundle()
    feature = bundle.config["features"][2]
    with pytest.raises(M.DonorImportError, match="R2_PRETRAINING_BUDGET_STOP"):
        M.pretrain_branch(bundle, feature, fit, sel, _test_objective(max_updates=1))
    with pytest.raises(M.DonorImportError, match="R2_OBJECTIVE_UNDECIDED"):
        M.pretrain_branch(bundle, feature, fit, sel, M.R2_OBJECTIVE_UNDECIDED)


def _one(donors, feature, mutate):
    out = copy.deepcopy(donors)
    mutate(out, feature)
    return out


def _first(target):
    return target.config["features"][0]


def test_shape_is_never_compatibility_mg1_and_closed_population_branches(target, donors):
    """MG1 and MG2m branch kernels have IDENTICAL shapes; identity, not shape, decides."""
    mg1 = _pretend_pretrained(_bundle("mg1"))
    feature = _first(target)
    assert [w.shape for w in mg1.branch_models[feature].get_weights()] == \
        [w.shape for w in target.branch_models[feature].get_weights()]
    mg1_ctx = _context(mg1)
    offered = copy.deepcopy(donors)
    offered[feature] = {"manifest": M.branch_donor_manifest(mg1, feature, mg1_ctx),
                        "weights": [np.array(w) for w in mg1.branch_models[feature].get_weights()]}
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    with pytest.raises(M.DonorImportError, match="R2_DONOR_ARM_MISMATCH"):
        M.import_branch_donors(target, offered, _context(target))
    # A same-recipe branch from ANOTHER population (e.g., the closed full-population MG2 fit).
    with pytest.raises(M.DonorImportError, match="R2_DONOR_POPULATION_MISMATCH"):
        M.import_branch_donors(target, donors, _context(target, population="7" * 64))
    with pytest.raises(M.DonorImportError, match="R2_DONOR_SCALER_MISMATCH"):
        M.import_branch_donors(target, donors, _context(target, scaler=_scaler_sha(seed=1)))
    after = target.forecast_model.get_weights()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))      # atomic: nothing imported


def _set_manifest(key, value):
    def mutate(d, f):
        d[f]["manifest"][key] = value
    return mutate


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d, f: d[f].update(manifest={"schema": M.BRANCH_DONOR_SCHEMA, "input_shape": [13, 1],
                                        "output_shape": [13, 4], "weights": d[f]["manifest"]["weights"]}),
     "R2_DONOR_IDENTITY_INCOMPLETE"),                              # shape-only manifest
    (lambda d, f: d[f].update(manifest={"input_shape": [13, 1]}), "R2_DONOR_SCHEMA"),
    (_set_manifest("rule", "DS2_EXACT"), "R2_DONOR_SAMPLING_MISMATCH"),         # elapsed DS2
    (_set_manifest("spacing_open_market_hours", 1), "R2_DONOR_SAMPLING_MISMATCH"),  # hourly/lagged
    (_set_manifest("slots", 24), "R2_DONOR_SAMPLING_MISMATCH"),                 # I6-B 24-step grid
    (_set_manifest("feature_names_sha256", "0" * 64), "R2_DONOR_FEATURE_ORDER"),
    (_set_manifest("conv_layer_name", "branch_00_dense"), "R2_DONOR_LAYER_MISMATCH"),
    (_set_manifest("branch_params", {"filters": 16, "kernel_size": 3, "strides": 1,
                                     "activation": "relu"}), "R2_DONOR_ARCHITECTURE_MISMATCH"),
    (_set_manifest("objective", {"option": "O2"}), "R2_DONOR_OBJECTIVE_MISMATCH"),
    (lambda d, f: d[f]["weights"][0].__setitem__((0, 0, 0), 9.0), "R2_DONOR_WEIGHTS_DIGEST"),
    (lambda d, f: d[f].update(weights=[np.zeros((3, 1, 16), np.float32), np.zeros(16, np.float32)]),
     "R2_DONOR_ARCHITECTURE_MISMATCH"),                            # FS4/I6-B-like widths
    (lambda d, f: d.pop(f), "R2_DONOR_SET"),
])
def test_donor_refusals_are_typed_and_atomic(target, donors, mutate, code):
    feature = _first(target)
    offered = _one(donors, feature, mutate)
    before = [np.array(w) for w in target.forecast_model.get_weights()]
    with pytest.raises(M.DonorImportError, match=code):
        M.import_branch_donors(target, offered, _context(target))
    assert all(np.array_equal(a, b) for a, b in zip(before, target.forecast_model.get_weights()))


def test_feature_order_swap_and_undecided_objective_are_refused(target, donors):
    features = target.config["features"]
    swapped = copy.deepcopy(donors)
    swapped[features[0]], swapped[features[1]] = swapped[features[1]], swapped[features[0]]
    with pytest.raises(M.DonorImportError, match="R2_DONOR_FEATURE_ORDER"):
        M.import_branch_donors(target, swapped, _context(target))
    with pytest.raises(M.DonorImportError, match="R2_OBJECTIVE_UNDECIDED"):
        M.import_branch_donors(target, donors, _context(target, objective=M.R2_OBJECTIVE_UNDECIDED))
    assert M.R2_OBJECTIVE_UNDECIDED == R2.UNDECIDED
    with pytest.raises(M.DonorImportError, match="R2_DONOR_ARM_MISMATCH"):
        elapsed = M.build_dmo(T.effective_from_design(MATCHED.parent / "EXPERIMENT.short.json")["model"])
        M.branch_donor_context(elapsed, population_sha256=POP, scaler_sha256=_scaler_sha(),
                               objective=_ruled_o1())
