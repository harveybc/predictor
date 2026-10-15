"""I6-DMO R0-versus-R2 design contract, frozen BEFORE any pretraining or fit.

NEXT_CONTRASTS_PLAN_20261015 section 3 and MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP section 4:
on the input recipe selected by the closed matched comparison (MG2m, 13 slots every 2 open-market
hours), compare R0 (seed initialization) with R2 (the SAME network whose per-feature branch
encoders start from compatible pretrained branch weights, imported, kept TRAINABLE and
fine-tuned end to end) on exactly the same common population, arm tensors, shared FIT-only
scaler, masks, seed, architecture and training recipe. Targets stay RAW exact t+h prices.

This module is the DESIGN layer only: it derives the contract deterministically from the sealed
matched artifacts, validates it with typed refusals, evaluates the R0-reuse identity predicate
and applies the frozen decision rule. It never pretrains, fits or reads data. The pretraining
objective is deliberately ``UNDECIDED_REQUIRES_REVIEW`` (see R2_DESIGN_PROPOSAL.md); a contract
in that state validates as PREPARING but is refused for execution (R2_OBJECTIVE_UNDECIDED).

    python -m tools.i6dmo.r2_design seal --matched-dir <matched_short> --out <R2_CONTRAST_DESIGN.json> [--json]
    python -m tools.i6dmo.r2_design validate --design <R2_CONTRAST_DESIGN.json> [--for-execution] [--json]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

from tools.i6dmo import design as D

R2_SCHEMA = "i6dmo.r2_contrast_design.v1"
R2_CONTRAST_ID = "I6-DMO-SHORT-MG2M-R0-VS-R2-BRANCH"
#: Same literal as tools.i6dmo.model.R2_OBJECTIVE_UNDECIDED (kept import-free of TensorFlow here).
UNDECIDED = "UNDECIDED_REQUIRES_REVIEW"
SELECTED_ARM = "mg2m"
MATCHED_DIR = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static/matched_short"
#: The closed matched comparison that selected the input recipe (campaign CLOSED, automatic).
SELECTION = {"campaign_id": "i6dmo-6004b711281c", "decision": "MG2_SELECTED",
             "comparison_sha256": "2ee85a125c3603e4069dbde42f1ccbf7ad32ac0b6c75ca798b7a8300cece13ee",
             "selected_arm": SELECTED_ARM, "rule": "lower equal-head mean annual pooled relative MAE; "
                                                   "ties within 1e-8 retain cheaper MG2m"}
SELECTION_TIE = 1e-8

#: Branch-pretraining objective OPTIONS (R2_DESIGN_PROPOSAL.md section 3). Only options whose
#: implementation already exists in this repository are executable after a ruling.
OBJECTIVE_OPTIONS = {
    "O1": {"name": "autoencoder_reconstruction", "version": "1.0.0",
           "implementation": "predictor_plugins.modular_temporal.objectives:AutoencoderReconstruction",
           "declared_params": ["decoder_channels", "loss"], "reconstruction": "MEASURED",
           "exists_in_repo": True},
    "O2": {"name": "ts2vec_contrastive", "version": "1.0.1",
           "implementation": "predictor_plugins.modular_temporal.objectives:TS2VecContrastive",
           "declared_params": ["alpha", "mask_probability", "min_overlap", "temporal_unit"],
           "reconstruction": "NOT_APPLICABLE", "exists_in_repo": True},
    "O3": {"name": "masked_point_reconstruction", "version": None, "implementation": None,
           "declared_params": [], "reconstruction": "MEASURED", "exists_in_repo": False},
    "O4": {"name": "timesiam_siamese_masked", "version": None, "implementation": None,
           "declared_params": [], "reconstruction": "MEASURED", "exists_in_repo": False},
}
RECOMMENDED_OPTION = "O1"
#: Fit-setting keys a ruling must fix (all of them; nothing defaults silently).
OBJECTIVE_FIT_SETTINGS = ("optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
                          "patience", "min_delta", "seed_rule", "restore")

R0_REUSE_COMPONENTS = (
    "status_completed", "scientific", "no_overrides", "device_gpu", "design_sha256",
    "effective_config_sha256", "common_population_sha256", "matched_arm", "capacity_sha256",
    "scales_sha256", "scaler_fit_row_digest", "runtime_tensorflow", "code_files_sha256",
    "monitor_identity", "r1_donor_verifier",
)

ACCEPTANCE_TESTS = {
    "R2-01": "R0 and R2 consume the identical common population, MG2m arm tensors, shared FIT-only "
             "scaler, masks, splits and seed; a changed population/scaler/design digest is refused",
    "R2-02": "R2 differs from R0 ONLY in the 19 branch initial weights: core and head initial weights "
             "bitwise equal to the R0 seed build, identical architecture/capacity (4882), branches "
             "trainable (never R1-frozen), end-to-end fine-tune with the R0 training recipe",
    "R2-03": "Donor import refuses shape-only matches: MG1, DS2/hourly/lagged, closed full-population, "
             "FS4 and I6-B branches (identical Conv1D kernel shapes) are refused by sampling rule, slot "
             "count, layer names, feature order, scaler digest, population digest, objective identity "
             "or weight digest",
    "R2-04": "Pretraining reads FIT windows only and selects on purged INNER only; DEVELOPMENT and TEST "
             "are never opened (perturbing DEVELOPMENT leaves donors bitwise unchanged)",
    "R2-05": "Targets stay RAW exact t+h prices; no reconstructed/denoised target or replacement input "
             "series reaches the fine-tune (input witness equal to R0's)",
    "R2-06": "The objective is ruled before execution: UNDECIDED_REQUIRES_REVIEW refuses execution "
             "(R2_OBJECTIVE_UNDECIDED); a ruled objective is a registered option with every parameter "
             "and fit setting fixed and a review reference; each donor records and must match it",
    "R2-07": "Reconstruction (or NOT_APPLICABLE) and the predictive probe are computed exactly as "
             "frozen, on purged INNER only, published as separate facts, never used to select R0/R2 "
             "and never claimed as utility or noise removal",
    "R2-08": "R0 reference is reused only when every R0_REUSE_COMPONENTS entry agrees AND the common "
             "R1 donor verifier accepts it; otherwise a fresh R0 fit is part of the graph; a "
             "full-population or MG1 fit is never R0; the R0 report is always recomputed",
    "R2-09": "Decision: equal-head mean of annual pooled relative MAE on the common DEVELOPMENT rows "
             "with each head's own naive; R2 selected iff score_R2 < score_R0 - 1e-8, ties retain R0; "
             "any failed head or arm closes INCOMPLETE_COMPARISON",
    "R2-10": "Pretraining loss/curves, pretraining compute, fine-tune FIT/INNER diagnostics and R0/R2 "
             "fit costs are separate OLAP facts with readback; one seed, no superiority claim",
    "R2-11": "Donor and R2-model serialization reload bitwise on CPU; bounded GPU replay under the "
             "declared numerical policy (no broadened tolerance)",
    "R2-12": "Scope: no core pretraining, cadence change, long family or TEST unit; any such field is "
             "refused (R2_SCOPE / TEST_FORBIDDEN)",
}


class R2DesignError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- contract

def build_contract(matched_dir=MATCHED_DIR):
    """The R0-vs-R2 contract derived deterministically from the sealed matched artifacts."""
    matched_dir = Path(matched_dir)
    design = D.load_design(matched_dir / "EXPERIMENT.mg2m.json")
    if D.matched_arm(design) != SELECTED_ARM:
        raise R2DesignError("R2_RECIPE_MISMATCH", "EXPERIMENT.mg2m.json is not the mg2m arm")
    effective = D.effective_config(design)
    seal = _read(matched_dir / "POPULATION_SEAL.json")
    capacity = _read(matched_dir / "CAPACITY_MATCHED.json")["arms"][SELECTED_ARM]
    sealed_arm = seal["identities"]["arms"][SELECTED_ARM]
    if sealed_arm["experiment_design_sha256"] != effective["design_sha256"] or \
            capacity["sealed_design_sha256"] != effective["design_sha256"]:
        raise R2DesignError("R2_RECIPE_MISMATCH", "population/capacity seals bind another mg2m design")
    mt = effective["design"]["market_time"]
    splits = {k: effective["splits"][k] for k in ("fit", "inner", "validation", "test")}
    shared = {
        "arm": SELECTED_ARM, "experiment_id": effective["experiment_id"],
        "experiment_design_sha256": effective["design_sha256"],
        "experiment_seal_sha256": design["seal_sha256"],
        "effective_config_sha256": effective["effective_config_sha256"],
        "model_config_sha256": _sha(effective["model"]),
        "capacity_sha256": capacity["capacity_sha256"],
        "parameter_total": capacity["counts"]["total"],
        "market_time": {k: mt[k] for k in ("rule", "observations", "spacing_open_market_hours",
                                           "span_open_market_hours")},
        "horizons_hours": list(effective["horizons_hours"]),
        "population_sha256": seal["population_digest"],
        "population_rules_sha256": seal["identities"]["matched_population_rules_sha256"],
        "scaler_fit_row_digest": seal["scaler"]["fit_row_digest"],
        "scaler_values_sha256": seal["scaler"]["values_sha256"],
        "naive_loss_scale": seal["naive_loss_scale"],
        "seed": effective["seed"],
        "splits": {"FIT": splits["fit"], "INNER": splits["inner"], "DEVELOPMENT": splits["validation"],
                   "TEST": "CLOSED_NEVER_READ"},
        "masks": "population SplitPopulation.head_mask per split (common, never re-derived)",
        "targets": "RAW_EXACT_T_PLUS_H_PRICE",
        "loss": "equal-head masked MAE in delta/naive_loss_scale[h] coordinates (FIT-only naive)",
        "training": copy.deepcopy(effective["training"]),
        "runtime": dict(D.MATCHED_RUNTIME),
        "accessor": "tools.i6dmo.common_population.arm_tensors(pop, EXPERIMENT.mg2m.json, split)",
    }
    contract = {
        "schema": R2_SCHEMA, "contrast_id": R2_CONTRAST_ID,
        "status": "FROZEN_BEFORE_FIT_DESIGN_ONLY",
        "authority": ["NEXT_CONTRASTS_PLAN_20261015.md section 3",
                      "MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP_20261015.md section 4"],
        "question": "On the selected MG2m input recipe, do compatible pretrained per-feature branch "
                    "encoders, imported and fine-tuned (R2), improve on the matched R0 reference?",
        "selection": dict(SELECTION),
        "shared": shared,
        "arms": {
            "R0": {"regime": "R0", "branch_init": "SEED_BUILD", "non_branch_init": "SEED_BUILD",
                   "branches_trainable": True},
            "R2": {"regime": "R2", "branch_init": "IMPORTED_PRETRAINED_BRANCH_DONORS",
                   "non_branch_init": "IDENTICAL_TO_R0_SEED_BUILD", "branches_trainable": True,
                   "fine_tune": "END_TO_END_WITH_THE_R0_TRAINING_RECIPE",
                   "import_check": "tools.i6dmo.model.import_branch_donors"},
        },
        "r0_reference": {
            "policy": "REUSE_IF_ALL_COMPONENTS_AGREE_AND_R1_ACCEPTS_ELSE_FRESH_R0",
            "candidate": {"campaign_id": SELECTION["campaign_id"], "unit": "mg2m.fit"},
            "components": list(R0_REUSE_COMPONENTS),
            "predicate": "tools.i6dmo.r2_design.r0_reuse_decision",
            "r1_verifier": "the ONE common OPS donor verifier governing prepare and adoption "
                           "(campaign.donor_fit_receipt_problem / donor_problem, "
                           "matched_runner._check_donor, worker._adoptable) must return ACCEPT",
            "report": "the R0 report unit is always recomputed inside the R2 graph",
        },
        "pretraining": {
            "unit": "ONE_DONOR_PER_FEATURE",
            "donor_architecture": {"component": "tools.i6dmo.model:dmo_branch", "version": "1.0.0",
                                   "params": {"filters": 4, "kernel_size": 3, "strides": 1,
                                              "padding": "causal", "activation": "relu"},
                                   "input_shape": [mt["observations"], 1],
                                   "output_shape": [mt["observations"], 4]},
            "start_weights": "R0_SEED_BUILD_BRANCH_WEIGHTS",
            "inputs": "arm_tensors(pop, EXPERIMENT.mg2m.json, split).x[:, :, f] scaled by pop.scaler "
                      "(the identical MG2m slot windows of the common origins)",
            "time_inputs": "not branch inputs; the windows are the identical MG2m market slots, so "
                           "support, calendar and time identity are the fine-tune's own",
            "data": {"fit": "FIT_COMMON_ORIGINS", "selection": "INNER_COMMON_ORIGINS_PURGED",
                     "purge": "INNER windows whose earliest slot instant precedes the INNER start "
                              "are excluded from selection (counted)",
                     "development": "NEVER", "test": "NEVER", "labels": "NONE_INPUTS_ONLY"},
            "early_stopping": {"monitor": "INNER_ONLY_OBJECTIVE_LOSS", "restore": "best_inner_epoch"},
            "objective": UNDECIDED,
            "objective_options": sorted(OBJECTIVE_OPTIONS),
            "recommended_option": RECOMMENDED_OPTION,
            "donor_policy": "FULL_IDENTITY_REQUIRED_NEVER_SHAPE_ALONE",
            "excluded_donors": ["MG1 matched branches (25 slots, 1 h)", "closed full-population MG2/MG12 fits",
                                "DS2/DS12 elapsed and hourly/lagged branches",
                                "FS4 extractibility autoencoders (24->12->6, 6x8 latent)",
                                "I6-B causal_conv1d 2.0.0 donors (24x2 -> 24x16)",
                                "any donor from another population, scaler, objective or arm"],
            "frozen_evaluations": {
                "reconstruction": {
                    "population": "INNER_COMMON_ORIGINS_PURGED (also FIT, reported separately)",
                    "metrics": ["mae_z", "mse_z"], "per": "feature",
                    "reference": "zero (scaler-mean) reconstruction on the same windows",
                    "not_applicable": "decoder-free objectives report NOT_APPLICABLE",
                    "disclosure": "reconstruction alone proves neither predictive utility nor "
                                  "noise removal"},
                "predictive": {
                    "probe": "frozen branches (pretrained vs R0 seed init) -> flatten 19x13x4 -> "
                             "closed-form ridge per head, alpha 1.0, fitted on FIT common rows "
                             "with the common masks in delta/naive_loss_scale coordinates",
                    "population": "INNER_COMMON_ORIGINS (all, common masks)",
                    "metric": "per-head masked MAE ratio to naive and equal-head mean; "
                              "probe_gain = R0-init probe minus pretrained probe",
                    "use": "REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS"}},
        },
        "decision": {
            "score": "equal-head mean of annual pooled (MAE_h-naive_h)/naive_h on the common "
                     "DEVELOPMENT rows; each head keeps its own naive; negative is better",
            "rule": "R2 selected iff score_R2 < score_R0 - 1e-8; otherwise R0 retained (no "
                    "pretraining cost); any failed head or arm -> INCOMPLETE_COMPARISON",
            "tie_tolerance": SELECTION_TIE, "function": "tools.i6dmo.r2_design.decide_r2",
            "per_head_primary": True,
            "limits": "one seed; DEVELOPMENT 2024 informed the feature set; no statistical "
                      "superiority, holdout or trading eligibility claim",
        },
        "reporting": {"olap_facts": ["r2.pretraining.loss_curve", "r2.pretraining.reconstruction",
                                     "r2.pretraining.predictive_probe", "r2.pretraining.compute_cost",
                                     "r2.fit.fit_inner_diagnostics", "r0.fit.fit_inner_diagnostics",
                                     "r0_r2.fit.compute_cost", "r0_r2.development.per_head_metrics",
                                     "r0_r2.decision"],
                      "separate_facts": True, "readback": True},
        "out_of_scope": ["core pretraining", "cadence changes", "long family", "TEST", "R1 frozen arm",
                         "objective or architecture search"],
        "acceptance_tests": dict(ACCEPTANCE_TESTS),
    }
    contract["seal_sha256"] = _sha(contract)
    return contract


_SECTION_CODES = (
    (("selection",), "R2_SELECTION_MISMATCH"),
    (("shared", "population_sha256"), "R2_POPULATION_MISMATCH"),
    (("shared", "population_rules_sha256"), "R2_POPULATION_MISMATCH"),
    (("shared", "scaler_fit_row_digest"), "R2_POPULATION_MISMATCH"),
    (("shared", "scaler_values_sha256"), "R2_POPULATION_MISMATCH"),
    (("shared", "naive_loss_scale"), "R2_POPULATION_MISMATCH"),
    (("shared", "masks"), "R2_POPULATION_MISMATCH"),
    (("shared", "targets"), "R2_TARGET_NOT_RAW"),
    (("shared", "splits"), "R2_SPLIT_MISMATCH"),
    (("shared", "training"), "R2_TRAINING_CHANGE"),
    (("shared", "seed"), "R2_TRAINING_CHANGE"),
    (("shared", "loss"), "R2_TRAINING_CHANGE"),
    (("shared", "runtime"), "R2_TRAINING_CHANGE"),
    (("shared",), "R2_ARCHITECTURE_CHANGE"),
    (("arms", "R2", "branches_trainable"), "R2_FROZEN_BRANCH"),
    (("arms",), "R2_ARM_CHANGE"),
    (("r0_reference",), "R2_R0_POLICY_CHANGE"),
    (("pretraining", "data"), "R2_PRETRAINING_SPLIT"),
    (("pretraining", "early_stopping"), "R2_PRETRAINING_SPLIT"),
    (("pretraining", "frozen_evaluations"), "R2_EVALUATION_CHANGE"),
    (("pretraining", "objective_options"), "R2_OBJECTIVE_UNKNOWN"),
    (("pretraining", "recommended_option"), "R2_OBJECTIVE_UNKNOWN"),
    (("pretraining",), "R2_DONOR_POLICY"),
    (("decision",), "R2_DECISION_CHANGE"),
    (("reporting",), "R2_REPORTING_CHANGE"),
    (("out_of_scope",), "R2_SCOPE"),
)


def _get(doc, path):
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return ("<absent>",)
        doc = doc[key]
    return doc


def _diff(expected, actual, path=()):
    """Leaf paths where ``actual`` differs from ``expected`` (missing/extra keys included)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                out.append(path + (key,))
            else:
                out.extend(_diff(expected[key], actual[key], path + (key,)))
        return out
    return [] if _canonical(expected) == _canonical(actual) else [path]


def _code_for(path):
    if path and path[0] in ("core_pretraining", "cadence", "long", "test", "TEST"):
        return "R2_SCOPE"
    for prefix, code in _SECTION_CODES:
        if tuple(path[:len(prefix)]) == prefix:
            return code
    return "R2_CONTRACT_MISMATCH"


def _validate_objective(objective):
    """A ruled objective: a registered option with every parameter and fit setting fixed."""
    if not isinstance(objective, dict) or set(objective) != {"option", "name", "version", "params",
                                                             "fit_settings", "ruling"}:
        raise R2DesignError("R2_OBJECTIVE_UNKNOWN",
                            "a ruled objective needs exactly option/name/version/params/fit_settings/ruling")
    option = OBJECTIVE_OPTIONS.get(objective["option"])
    if option is None or not option["exists_in_repo"]:
        raise R2DesignError("R2_OBJECTIVE_UNKNOWN",
                            f"option {objective['option']!r} is not an implemented registered objective")
    if (objective["name"], objective["version"]) != (option["name"], option["version"]):
        raise R2DesignError("R2_OBJECTIVE_UNKNOWN", "objective name/version differ from the option")
    if sorted(objective["params"]) != sorted(option["declared_params"]):
        raise R2DesignError("R2_OBJECTIVE_UNKNOWN",
                            f"every declared parameter must be fixed: {option['declared_params']}")
    if sorted(objective["fit_settings"]) != sorted(OBJECTIVE_FIT_SETTINGS):
        raise R2DesignError("R2_OBJECTIVE_UNKNOWN",
                            f"fit settings must be exactly {list(OBJECTIVE_FIT_SETTINGS)}")
    for value in list(objective["params"].values()) + list(objective["fit_settings"].values()):
        if value is None or value == UNDECIDED or (isinstance(value, float) and not math.isfinite(value)):
            raise R2DesignError("R2_OBJECTIVE_UNDECIDED", "an objective parameter is still open")
    ruling = objective["ruling"]
    if not isinstance(ruling, dict) or not ruling.get("reviewer") or \
            not isinstance(ruling.get("document_sha256"), str) or len(ruling["document_sha256"]) != 64:
        raise R2DesignError("R2_OBJECTIVE_UNREVIEWED",
                            "a ruling needs a reviewer and the sha256 of the ruling document")


def validate_r2_design(doc, *, for_execution=False, matched_dir=MATCHED_DIR):
    """Typed validation. PREPARING while the objective is undecided; READY only once ruled."""
    if not isinstance(doc, dict) or doc.get("schema") != R2_SCHEMA:
        raise R2DesignError("R2_SCHEMA_REQUIRED", f"schema must be {R2_SCHEMA}")
    expected = build_contract(matched_dir)
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    reference = {k: v for k, v in expected.items() if k != "seal_sha256"}
    objective = _get(body, ("pretraining", "objective"))
    status = body.get("status")
    allowed = "FROZEN_BEFORE_FIT_DESIGN_ONLY" if objective == UNDECIDED else "FROZEN_BEFORE_FIT_OBJECTIVE_RULED"
    if status != allowed:
        raise R2DesignError("R2_CONTRACT_MISMATCH", f"status {status!r} != {allowed!r} for this objective state")
    differences = [p for p in _diff(reference, body)
                   if p not in (("pretraining", "objective"), ("status",))]
    if differences:
        first = differences[0]
        raise R2DesignError(_code_for(first),
                            f"{'.'.join(first)} differs from the contract derived from the sealed "
                            f"matched artifacts ({len(differences)} difference(s))")
    if doc.get("seal_sha256") != _sha(body):
        raise R2DesignError("R2_SEAL", "the R2 contract was modified after sealing")
    if objective == UNDECIDED:
        if for_execution:
            raise R2DesignError("R2_OBJECTIVE_UNDECIDED",
                                "pretraining.objective is UNDECIDED_REQUIRES_REVIEW; R2 cannot be "
                                "executed until the objective is ruled (R2_DESIGN_PROPOSAL.md)")
        return {"status": "PREPARING", "objective": UNDECIDED}
    _validate_objective(objective)
    return {"status": "READY_FOR_EXECUTION", "objective": objective}


def rule_objective(doc, objective):
    """A NEW sealed contract carrying a reviewed objective ruling (the reviewer's act)."""
    out = copy.deepcopy({k: v for k, v in doc.items() if k != "seal_sha256"})
    out["pretraining"]["objective"] = copy.deepcopy(objective)
    out["status"] = "FROZEN_BEFORE_FIT_OBJECTIVE_RULED"
    out["seal_sha256"] = _sha(out)
    return out


# --------------------------------------------------------------------------- R0 reuse predicate

def r0_reuse_decision(receipt, monitor_record, contract, *, expected_code_files,
                      expected_scales_sha256, current_monitor_identity, r1_verdict):
    """Explicit identity-agreement predicate for reusing the closed campaign's mg2m fit as R0.

    REUSE_R0 only when EVERY component agrees; otherwise FRESH_R0_REQUIRED with the failed
    components named. ``expected_code_files`` / ``expected_scales_sha256`` /
    ``current_monitor_identity`` come from the R2 graph's own pinned code and population;
    ``r1_verdict`` is the common OPS donor verifier's verdict on the candidate root.
    """
    shared = contract["shared"]
    receipt = receipt or {}
    identity = receipt.get("identity") or {}
    matched = receipt.get("matched") or {}
    monitor_record = monitor_record or {}
    observed_monitor = {k: monitor_record.get(k) for k in sorted(current_monitor_identity)}
    checks = {
        "status_completed": (receipt.get("status"), "COMPLETED"),
        "scientific": (receipt.get("scientific"), True),
        "no_overrides": (receipt.get("overrides"), {}),
        "device_gpu": (receipt.get("device"), "gpu"),
        "design_sha256": (identity.get("design_sha256"), shared["experiment_design_sha256"]),
        "effective_config_sha256": (identity.get("effective_config_sha256"),
                                    shared["effective_config_sha256"]),
        "common_population_sha256": (identity.get("common_population_sha256"),
                                     shared["population_sha256"]),
        "matched_arm": (identity.get("matched_arm"), shared["arm"]),
        "capacity_sha256": (identity.get("capacity_sha256"), shared["capacity_sha256"]),
        "scales_sha256": (identity.get("scales_sha256"), expected_scales_sha256),
        "scaler_fit_row_digest": ((matched.get("scaler") or {}).get("fit_row_digest"),
                                  shared["scaler_fit_row_digest"]),
        "runtime_tensorflow": ((receipt.get("resources") or {}).get("tensorflow_version"),
                               shared["runtime"]["tensorflow"]),
        "code_files_sha256": (identity.get("code_files_sha256"), expected_code_files),
        "monitor_identity": (observed_monitor, dict(current_monitor_identity)),
        "r1_donor_verifier": (r1_verdict, "ACCEPT"),
    }
    assert tuple(checks) == R0_REUSE_COMPONENTS
    components = {name: {"agree": _canonical(obs) == _canonical(exp), "observed": obs, "expected": exp}
                  for name, (obs, exp) in checks.items()}
    failed = [name for name, row in components.items() if not row["agree"]]
    return {"schema": "i6dmo.r0_reuse_decision.v1",
            "decision": "FRESH_R0_REQUIRED" if failed else "REUSE_R0",
            "failed": failed, "components": components,
            "note": "the R0 report is recomputed in the R2 graph either way"}


# --------------------------------------------------------------------------- decision rule

def _selection_ok(selection):
    return isinstance(selection, dict) and selection.get("status") == "COMPLETE" and \
        isinstance(selection.get("score"), float) and math.isfinite(selection["score"])


def decide_r2(r0_metrics, r2_metrics):
    """Frozen R0-vs-R2 decision from the two matched METRICS.json documents (both mg2m)."""
    a, b = (r0_metrics or {}).get("matched") or {}, (r2_metrics or {}).get("matched") or {}
    if a.get("arm") != SELECTED_ARM or b.get("arm") != SELECTED_ARM:
        raise R2DesignError("R2_ARM_CHANGE", "both R0 and R2 are scored on the mg2m arm")
    for key in ("common_population_sha256", "development_row_ids_sha256",
                "development_head_mask_sha256", "development_counts"):
        if a.get(key) != b.get(key):
            raise R2DesignError("R2_POPULATION_MISMATCH", key)
    if (a.get("capacity") or {}).get("capacity_sha256") != (b.get("capacity") or {}).get("capacity_sha256"):
        raise R2DesignError("R2_ARCHITECTURE_CHANGE", "R0 and R2 must have identical capacity")
    sa, sb = a.get("selection") or {}, b.get("selection") or {}
    heads = sa.get("heads") or sb.get("heads") or []
    per_head = {str(h): {"R0": ((r0_metrics.get("per_horizon") or {}).get(str(h)) or {}).get("relative_error"),
                         "R2": ((r2_metrics.get("per_horizon") or {}).get(str(h)) or {}).get("relative_error")}
                for h in heads}
    doc = {"schema": "i6dmo.r2_decision.v1", "per_head": per_head,
           "scores": {"R0": sa.get("score"), "R2": sb.get("score")}, "tie_tolerance": SELECTION_TIE,
           "note": "one seed; a recipe decision, not statistical superiority; an aggregate never "
                   "makes a failed head trading-eligible"}
    missing = [dict(f, arm=arm) for arm, sel in (("R0", sa), ("R2", sb))
               for f in (sel.get("failed_heads") or [])]
    if missing or not _selection_ok(sa) or not _selection_ok(sb):
        return dict(doc, decision="INCOMPLETE_COMPARISON", selected=None, missing=missing)
    diff = sb["score"] - sa["score"]
    if diff < -SELECTION_TIE:
        return dict(doc, decision="R2_LOWER_SCORE", selected="R2", difference=diff)
    return dict(doc, decision="R0_RETAINED", selected="R0", difference=diff)


# --------------------------------------------------------------------------- CLI

def seal(matched_dir, out):
    contract = build_contract(matched_dir)
    text = json.dumps(contract, indent=1, sort_keys=True) + "\n"
    out = Path(out)
    if out.exists():
        if out.read_text(encoding="utf-8") != text:
            raise R2DesignError("R2_DESIGN_EXISTS", "a different R2 contract already exists at this path")
        return {"status": "SEALED", "idempotent": True, "seal_sha256": contract["seal_sha256"]}
    out.write_text(text, encoding="utf-8")
    return {"status": "SEALED", "idempotent": False, "seal_sha256": contract["seal_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.r2_design")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--matched-dir", default=str(MATCHED_DIR))
    s.add_argument("--out", required=True)
    s.add_argument("--json", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--design", required=True)
    v.add_argument("--matched-dir", default=str(MATCHED_DIR))
    v.add_argument("--for-execution", action="store_true")
    v.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = seal(args.matched_dir, args.out)
        else:
            result = validate_r2_design(_read(args.design), for_execution=args.for_execution,
                                        matched_dir=args.matched_dir)
    except (R2DesignError, D.DesignError) as exc:
        print(json.dumps({"status": "FAILED", "code": exc.code, "detail": str(exc)[:400]}))
        return 2
    print(json.dumps(result, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
