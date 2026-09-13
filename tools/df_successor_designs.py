#!/usr/bin/env python3
"""C141, C142, C143 (order 2026-09-12): successor designs D3, D4 and D5.

Three designs with no scores, for the steps that follow D2 in the binding
sequence (06_ESTADO_REAL_PREPROCESAMIENTO_Y_SECUENCIA_2026_09_12.md):

* D3 (C141), STEP 04-07: quantization/companding with train-frozen cells,
  source coding/MDL, time-frequency representations and detectors, each
  with a declared causal/online feasibility;
* D4 (C142), STEP 08-10: equalization, common/private and redundancy
  cancellation, causal alignment; temporal availability is declared and
  verified before any lead/lag correction and no correction uses a future
  sample;
* D5 (C143), STEP 11-13: controlled redundancy, adaptive routing and
  multi-branch allocation, with a cost cap per branch and abstention to
  the raw branch as part of the contract.

Each design binds its protocols by file and sha256 computed from the bytes
on disk, consumes only externally reviewed D2 lab decisions in
{LAB_CALIBRATED, REGIME_LIMITED} plus D0/D1 contracts and profiles, keeps
the raw branch as a control arm, and names arms, controls and metrics with
their estimators. Nothing is run, no value is computed, nothing is granted.

The validator is strict: exact keys and types per block, numbers only in
budget caps, no result-like keys, PUBLICLY_ELIGIBLE only inside a "never
grants" statement, protocol digests re-derived from disk and design_sha256
re-derived from the content.

It imports no numeric library.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROTOCOL_DIR = "docs/tres_temas_entrevista"
EVIDENCE_DIR = "docs/audits/evidence"
MASTER_PLAN = f"{PROTOCOL_DIR}/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md"

STATUS = "DESIGN_NO_SCORES_COMPUTED"
RAW_ARM = "A0_RAW"
LAB_DECISIONS = ("LAB_CALIBRATED", "REGIME_LIMITED", "NOT_IDENTIFIABLE", "LAB_REJECTED")
CONSUMABLE = ("LAB_CALIBRATED", "REGIME_LIMITED")
NEVER_CONSUMED = ("NOT_IDENTIFIABLE", "LAB_REJECTED")
FORBIDDEN_STATE = "PUBLICLY_ELIGIBLE"
NEVER_GRANTS = re.compile(r"never\s+grant", re.IGNORECASE)
HEX64 = re.compile(r"[0-9a-f]{64}")
CAUSAL_CONTRACTS = ("TIMESTAMP_CAUSAL", "ENDPOINT_CAUSAL", "ANALYSIS_ONLY_NON_CAUSAL")
ARM_ROLES = ("RAW_CONTROL", "CANDIDATE")
FITS = ("NONE", "TRAIN_ONLY_FROZEN", "CAUSAL_ONLINE_UPDATE_FROM_TRAIN_INIT")
CONTROL_KINDS = ("RAW", "NULL_SURROGATE", "NEGATIVE_CONTROL_OPERATOR")
DIRECTIONS = ("HIGHER_IS_BETTER", "LOWER_IS_BETTER", "TWO_SIDED", "MUST_BE_ZERO")
METRIC_PARTITIONS = ("SYNTHETIC_KNOWN_TRUTH", "TRAIN", "CALIBRATION", "CONFIRMATION")
RECORD_STATES = ("COMPLETED", "FAILED", "INCONCLUSIVE", "REFUSED", "REJECTED",
                 "NOT_RUN", "UNAVAILABLE")
VERIFIED_AVAILABILITY = "VERIFIED_TEMPORAL_AVAILABILITY"
LICENSE_REQUIRED = "EXTERNAL_REVIEW_OF_D2_OUTPUTS_AND_OF_THIS_DESIGN"
# Keys whose presence would declare a result inside a design.
RESULT_KEYS = frozenset({
    "value", "values", "score", "scores", "metric_value", "metric_values", "result",
    "results", "p_value", "p_values", "estimate", "estimates", "observed", "outcome",
    "outcomes", "auc", "rmse", "mae", "mse", "r2", "accuracy", "snr_db", "passed",
    "lopo_result", "measured", "measurement"})
# Keys whose string content is a decision or a state.
DECISION_KEYS = frozenset({"decision", "decisions", "state", "states", "status",
                           "outcome", "lab_decision", "states_recorded",
                           "accepted_d2_decisions", "never_consumed", "action", "on_cap"})

NON_CAUSAL_RULE = ("Any non-causal transform is analysis-only and never an input: an "
                   "arm declared ANALYSIS_ONLY_NON_CAUSAL is not online feasible, its "
                   "output is never delivered to a model, router or allocator, and an "
                   "arm that fails the prefix-invariance audit is reclassified "
                   "ANALYSIS_ONLY_NON_CAUSAL.")
AVAILABILITY_RULE = ("Temporal availability is declared and verified before any lead/lag "
                     "correction. No correction may use future samples: every aligned, "
                     "equalized or cancelled output at decision time t uses only "
                     "observations whose verified availability time is <= t, and "
                     "alignment metadata may shift interpretation forward, never data "
                     "backward.")
AVAILABILITY_ORDER = ("DECLARE_AVAILABILITY_PER_SOURCE", "VERIFY_AVAILABILITY_AGAINST_D0_CONTRACT",
                      "ESTIMATE_LEAD_LAG_ON_TRAIN_ONLY", "APPLY_CAUSAL_CORRECTION")
ABSTENTION_RULE = ("Routing may always abstain to the raw branch: when the information "
                   "quality state is unavailable, in warm-up or outside its reviewed "
                   "regime, when a branch cost cap is reached, or when the router cannot "
                   "decide, the output is the raw branch and the abstention is recorded.")


class StrictJsonRefusal(ValueError):
    pass


class WriteRefusal(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def sha_obj(o) -> str:
    """Same canonical digest as the per-variable designs."""
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")
                                     ).encode()).hexdigest()


def strict_json_loads(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise StrictJsonRefusal(f"DUPLICATE_KEY: {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise StrictJsonRefusal(f"NON_FINITE_CONSTANT: {name}")

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_entry(rel: str, repo_root=REPO) -> dict:
    return {"file": rel, "sha256": file_sha256(Path(repo_root) / rel)}


def protocol_files(prefixes, repo_root=REPO) -> list[str]:
    d = Path(repo_root) / PROTOCOL_DIR
    out = []
    for pre in prefixes:
        hits = sorted(p.name for p in d.glob(f"{pre}_*_FINAL.md"))
        if len(hits) != 1:
            raise FileNotFoundError(f"protocol for {pre}: expected one file, found {hits}")
        out.append(f"{PROTOCOL_DIR}/{hits[0]}")
    return out


def arm(arm_id, step, family, description, parameters, fit, causal, usable, requires):
    return {"arm_id": arm_id, "step": step, "role": "CANDIDATE", "family": family,
            "description": description, "parameters": list(parameters), "fit": fit,
            "causal_contract": causal, "usable_as_input": usable, "requires": list(requires)}


def raw_arm(description):
    return {"arm_id": RAW_ARM, "step": "ALL", "role": "RAW_CONTROL", "family": "identity",
            "description": description, "parameters": [], "fit": "NONE",
            "causal_contract": "TIMESTAMP_CAUSAL", "usable_as_input": True,
            "requires": ["D0_COMMON_CONTRACT"]}


def control(cid, kind, description, steps, synthetic_only):
    return {"control_id": cid, "kind": kind, "description": description,
            "steps": list(steps), "synthetic_only": synthetic_only, "consumed_as_input": False}


def metric(mid, step, name, estimator, direction, partitions, compares):
    return {"metric_id": mid, "step": step, "name": name, "estimator": estimator,
            "direction": direction, "partitions": list(partitions), "compares": list(compares)}


def question(qid, step, text):
    return {"question_id": qid, "step": step, "question": text}


def hypothesis(hid, step, statement, falsified_if, section):
    return {"hypothesis_id": hid, "step": step, "statement": statement,
            "falsified_if": falsified_if, "protocol_section": section}


def rule(rid, condition, action):
    return {"rule_id": rid, "condition": condition, "action": action}


def grain(name, keys, records):
    return {"grain": name, "keys": list(keys), "records": list(records)}


# --------------------------------------------------------- common sections
def _consumes(extra_d2_use: str) -> dict:
    return {
        "accepted_d2_decisions": list(CONSUMABLE),
        "never_consumed": list(NEVER_CONSUMED),
        "external_review_record_required": True,
        "unreviewed_d2_output": "NEVER_CONSUMED",
        "d2_grain": "operator x regime lab decision (C138)",
        "review_record_fields": ["operator_id", "regime_id", "lab_decision",
                                 "lab_decision_record_sha256", "external_reviewer",
                                 "review_record_sha256", "reviewed_design_sha256"],
        "d0_d1_inputs": [
            "C129 common dataset/variable contract (identity, source, license, semantics, "
            "unit, frequency, availability, missingness, sentinels, partitions, digest)",
            "C130 univariate profile per partition",
            "C131 information and compression profile",
            "C132 multivariate profile and grouping (training only)",
            "C133 sampling and aliasing diagnostics"],
        "d2_use": extra_d2_use,
        "successor_outputs": "Not consumed. Outputs of other successor designs may be bound "
                             "only by a later version, after their own lab decisions and "
                             "external review records exist.",
        "rule": "Only D2 lab decisions LAB_CALIBRATED or REGIME_LIMITED that carry an "
                "external review record are consumed, each within its reviewed regime. "
                "NOT_IDENTIFIABLE and LAB_REJECTED are never consumed. Nothing is consumed "
                "without an external review record. If no such record exists, real-data "
                "stages are NOT_RUN and only synthetic calibration may proceed."}


def _raw_control(steps_text: str) -> dict:
    return {"kept": True, "arm_id": RAW_ARM, "replaced_by_transform": False,
            "rule": "The raw (untransformed) branch is always kept as a control arm in every "
                    f"comparison of {steps_text}, under the same partition, sample, regime "
                    "and budget. No transform replaces the original data before a separate, "
                    "reviewed utility test."}


def _synthetic_first(families) -> dict:
    return {"required": True,
            "order": ["SYNTHETIC_KNOWN_TRUTH", "PUBLIC", "FINANCIAL"],
            "generator_bank": "C128 synthetic bank: clean component, perturbation, observed "
                              "series, seeds and parameters stored separately and rebuildable",
            "families": list(families),
            "parameters_frozen_before_real_data": True,
            "real_data_before_calibration": "REFUSED",
            "rule": "Every arm parameter is calibrated on known-truth synthetic data and "
                    "frozen before any public or financial data is read. An arm that does "
                    "not identify its known truth stops at the synthetic stage."}


def _partitions(target_note: str) -> dict:
    return {"fit": "TRAIN_ONLY", "decide": "CALIBRATION", "confirm": "CONFIRMATION",
            "mixing_partitions": "FORBIDDEN", "sealed_periods": "NEVER_READ",
            "test_set_tuning": "FORBIDDEN", "target_use": "NONE",
            "target_exception": target_note,
            "rule": "All fitting (cells, codebooks, templates, scales, projections, lags, "
                    "thresholds, allocations) uses training data only. Calibration decides "
                    "the lab decision; confirmation confirms it without refitting. No sealed "
                    "period is read. No target participates in this design."}


TARGET_NOTE = ("A target may appear only in a later, separately licensed utility test and "
               "then only inside training folds; it is not part of this design.")


def _failure_regions(grain_text: str, extra) -> dict:
    return {"required": True, "grain": grain_text,
            "publish": ["every region whose lab decision is NOT_IDENTIFIABLE or LAB_REJECTED",
                        "every region where the raw control is not dominated",
                        "every region where the null surrogate is indistinguishable",
                        "every region where the negative-control operator is not flagged",
                        "every causality or availability audit failure",
                        "every budget failure"] + list(extra),
            "best_case_only": "REFUSED",
            "states_recorded": list(RECORD_STATES),
            "rule": "Failure regions are published with the same grain and prominence as "
                    "regions that calibrate; a summary of the best case alone is refused."}


def _decision_vocabulary() -> dict:
    return {"decisions": list(LAB_DECISIONS), "grain": "arm x regime",
            "none_equals_public_eligibility": True,
            "statement": "LAB_CALIBRATED, REGIME_LIMITED, NOT_IDENTIFIABLE and LAB_REJECTED "
                         "are laboratory decisions only; none of them equals public "
                         "eligibility, and this design never grants PUBLICLY_ELIGIBLE."}


def _budget(core_hours, wall, mem, workers) -> dict:
    return {"device": "CPU_ONLY", "accelerator": "NONE",
            "cpu_core_hours_cap": core_hours, "wall_seconds_cap_per_arm_regime": wall,
            "peak_memory_mib_cap": mem, "parallel_workers_cap": workers,
            "same_cap_for_every_arm": True,
            "on_cap": "RECORD_FAILED_BUDGET_EXCEEDED_NO_EXTENSION"}


def _license() -> dict:
    return {"scoring": "NOT_GRANTED", "execution": "NOT_GRANTED",
            "consumption": "NOT_GRANTED", "selection": "NOT_GRANTED",
            "training": "NOT_GRANTED", "required": LICENSE_REQUIRED}


GRANTS_NOTHING = ("This design grants nothing: no execution, no scoring, no consumption, no "
                  "feature selection, no training, no GPU, no RL, no DOIN, no live or venue "
                  "action, and it never grants PUBLICLY_ELIGIBLE to any variable, operator, "
                  "representation, router or allocation.")

COMMON_STOPS = [
    rule("SR_NO_REVIEWED_D2", "no D2 lab decision in {LAB_CALIBRATED, REGIME_LIMITED} with an "
         "external review record exists for the operator and regime",
         "real-data stages recorded NOT_RUN; only synthetic calibration may proceed"),
    rule("SR_SYNTHETIC_FAIL", "an arm does not identify its known truth on the synthetic bank",
         "arm stops before public and financial data; decision NOT_IDENTIFIABLE or LAB_REJECTED"),
    rule("SR_FROZEN_GRID", "a parameter or arm not predeclared in this design is proposed "
         "after any calibration output is seen", "refused; a new design version is required"),
    rule("SR_BUDGET", "an arm reaches a declared budget cap",
         "run recorded FAILED with reason BUDGET_EXCEEDED; no extension"),
    rule("SR_RAW_NOT_DOMINATED", "a candidate does not improve on the raw control on its "
         "declared preservation metric within a regime",
         "decision for that regime is REGIME_LIMITED or LAB_REJECTED, never LAB_CALIBRATED"),
    rule("SR_SEALED_OR_TARGET", "any read of a sealed period, a confirmation refit or a target",
         "whole run REFUSED and recorded; no decision is emitted"),
]


def _cost_metric(step):
    return metric(f"M_COST_{step}", step, "computational cost and delay",
                  "CPU seconds (process time), peak RSS in MiB, algorithmic delay in samples "
                  "from the impulse response, and warm-up length, each measured separately "
                  "per arm x regime", "LOWER_IS_BETTER", ["SYNTHETIC_KNOWN_TRUTH", "CALIBRATION"],
                  ["ALL_ARMS"])


# ----------------------------------------------------------------------- D3
D3_SCHEMA = "crispdm.data_foundation.successor_design.d3.v1"
D4_SCHEMA = "crispdm.data_foundation.successor_design.d4.v1"
D5_SCHEMA = "crispdm.data_foundation.successor_design.d5.v1"

SPECS = {
    "D3": {"schema": D3_SCHEMA, "order_item": "C141",
           "steps": ["STEP_04", "STEP_05", "STEP_06", "STEP_07"],
           "file": "D3_QUANTIZATION_COMPRESSION_TIMEFREQ_DETECTORS_DESIGN.v1.json",
           "specific": "causal_feasibility"},
    "D4": {"schema": D4_SCHEMA, "order_item": "C142",
           "steps": ["STEP_08", "STEP_09", "STEP_10"],
           "file": "D4_EQUALIZATION_REDUNDANCY_ALIGNMENT_DESIGN.v1.json",
           "specific": "temporal_availability"},
    "D5": {"schema": D5_SCHEMA, "order_item": "C143",
           "steps": ["STEP_11", "STEP_12", "STEP_13"],
           "file": "D5_ROBUSTNESS_ROUTING_ALLOCATION_DESIGN.v1.json",
           "specific": "budget_and_abstention"},
}
COMMON_KEYS = ("schema", "design_id", "status", "order_item", "steps", "protocols", "consumes",
               "raw_control", "questions", "hypotheses", "arms", "controls", "metrics",
               "synthetic_calibration_first", "partitions", "stopping_rules",
               "failure_region_reporting", "decision_vocabulary", "budget", "olap_grains",
               "grants_nothing", "license", "design_sha256")


def _protocols(steps, repo_root):
    files = protocol_files(steps, repo_root) + [MASTER_PLAN]
    return [protocol_entry(f, repo_root) for f in files]


def _finish(d: dict) -> dict:
    d["design_sha256"] = sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


def build_d3(repo_root=REPO) -> dict:
    TF, TC, EC, NC = "TRAIN_ONLY_FROZEN", "TIMESTAMP_CAUSAL", "ENDPOINT_CAUSAL", "ANALYSIS_ONLY_NON_CAUSAL"
    D2 = "D2_REVIEWED_LAB_DECISION"
    arms = [
        raw_arm("untransformed variable as delivered by the D0 contract, float resolution"),
        # STEP 04
        arm("Q04_UNIFORM", "STEP_04", "uniform scalar quantizer",
            "uniform cells over a train-frozen robust range with saturation cells at both ends",
            ["cell_count", "range_lower_quantile", "range_upper_quantile"], TF, TC, True, []),
        arm("Q04_QUANTILE", "STEP_04", "equiprobable quantile quantizer",
            "cell edges at train-frozen empirical quantiles", ["cell_count"], TF, TC, True, []),
        arm("Q04_LLOYD_MAX", "STEP_04", "Lloyd-Max quantizer",
            "MSE-optimal cells iterated on training samples only, then frozen",
            ["cell_count", "iterations_cap"], TF, TC, True, []),
        arm("Q04_MU_LAW", "STEP_04", "mu-law companding + uniform quantizer",
            "train-frozen scale, mu-law compression, uniform cells in the compressed domain",
            ["mu", "cell_count", "scale_quantile"], TF, TC, True, []),
        arm("Q04_NOISE_AWARE", "STEP_04", "noise-aware resolution",
            "cell width tied to the reviewed D2 noise-scale estimate of the variable, frozen on train",
            ["cells_per_noise_scale"], TF, TC, True, [D2]),
        # STEP 05
        arm("S05_PREDICTIVE_RESIDUAL", "STEP_05", "causal predictive residual",
            "E_t = X_t - Xhat_t from a fixed-order linear predictor fit on train and frozen",
            ["predictor_order"], TF, TC, True, []),
        arm("S05_CONTEXT_SURPRISAL", "STEP_05", "variable-order context model surprisal",
            "S_t = -log2 p(q_t | context) from a PPM/CTW-style context model over train-frozen "
            "quantized symbols, counts frozen on train", ["max_context_order", "symbol_quantizer_arm"],
            TF, TC, True, []),
        arm("S05_MDL_ORDER", "STEP_05", "MDL context-order diagnostic",
            "two-part code length per context order on train; selected order is a memory-depth "
            "diagnostic, not an input", ["max_context_order"], TF, TC, False, []),
        arm("S05_COMPRESSED_LENGTH", "STEP_05", "fixed-compressor code length diagnostic",
            "compressed bytes per sample of trailing windows with compressor and level fixed "
            "in the run manifest", ["compressor", "level", "window_length"], "NONE", TC, False, []),
        # STEP 06
        arm("T06_TRAILING_STFT", "STEP_06", "trailing-window STFT magnitude",
            "one-sided window whose right edge is t; magnitude per train-declared bin",
            ["window_length", "window_family", "hop"], "NONE", TC, True, []),
        arm("T06_TRAILING_PHASE", "STEP_06", "trailing-window circular phase",
            "phase of declared bins encoded as (sin, cos) from the same trailing window",
            ["window_length", "bins"], "NONE", TC, True, []),
        arm("T06_CAUSAL_SWT", "STEP_06", "causal stationary wavelet bands",
            "undecimated wavelet with one-sided filters and no right boundary extension",
            ["wavelet_family", "levels"], "NONE", TC, True, []),
        arm("T06_TRAILING_MULTITAPER", "STEP_06", "trailing multitaper spectrum",
            "DPSS tapers over a trailing window", ["window_length", "time_bandwidth", "tapers"],
            "NONE", EC, True, []),
        arm("T06_CENTERED_CWT", "STEP_06", "centered continuous wavelet transform",
            "two-sided CWT over the full series; boundary uses future samples", ["wavelet_family", "scales"],
            "NONE", NC, False, []),
        arm("T06_FULL_HILBERT_HHT", "STEP_06", "full-series Hilbert / Hilbert-Huang",
            "analytic signal and EMD over the whole series; non-causal by construction",
            ["emd_stop_rule"], "NONE", NC, False, []),
        # STEP 07
        arm("D07_MATCHED_FILTER", "STEP_07", "normalized cross-correlation matched filter",
            "templates frozen from the synthetic motif bank or train-only motif discovery, "
            "correlated over trailing windows", ["template_bank", "template_length"], TF, TC, True, []),
        arm("D07_GENERALIZED_MF", "STEP_07", "generalized matched filter",
            "template weighted by the inverse of a train-estimated noise covariance (whitened)",
            ["template_bank", "covariance_shrinkage"], TF, TC, True, []),
        arm("D07_MINIROCKET", "STEP_07", "MiniRocket random convolution transform",
            "fixed-seed kernels, biases from training quantiles, applied to the window ending at t; "
            "transform only, no classifier", ["kernel_count", "seed", "window_length"], TF, EC, True, []),
        arm("D07_LEFT_MATRIX_PROFILE", "STEP_07", "left (trailing) matrix profile",
            "nearest-neighbour distance of the subsequence ending at t against past subsequences only",
            ["subsequence_length", "exclusion_zone"], "NONE", TC, True, []),
    ]
    feas = []
    warm = {"TIMESTAMP_CAUSAL": "declared warm-up samples are typed UNAVAILABLE",
            "ENDPOINT_CAUSAL": "outputs exist only at window endpoints; internal positions are not emitted",
            "ANALYSIS_ONLY_NON_CAUSAL": "not applicable; never emitted as input"}
    for a in arms:
        if a["role"] != "CANDIDATE":
            continue
        feas.append({"arm_id": a["arm_id"], "representation": a["family"],
                     "causal_contract": a["causal_contract"],
                     "online_feasible": a["causal_contract"] != NC,
                     "usable_as_input": a["usable_as_input"],
                     "warm_up": warm[a["causal_contract"]],
                     "boundary_rule": "output at t uses samples <= t only"
                     if a["causal_contract"] != NC else
                     "uses samples after t; analysis-only on synthetic and training data"})
    controls = [
        control("C_RAW", "RAW", "A0_RAW evaluated on the same partition, sample, regime and budget",
                ["STEP_04", "STEP_05", "STEP_06", "STEP_07"], False),
        control("C_NULL_IAAFT", "NULL_SURROGATE", "IAAFT phase-randomized surrogate preserving "
                "amplitude distribution and spectrum, generated from training data only",
                ["STEP_05", "STEP_06", "STEP_07"], False),
        control("C_NULL_BLOCK_PERMUTATION", "NULL_SURROGATE", "within-train block permutation "
                "destroying temporal order beyond the block length", ["STEP_04", "STEP_05"], False),
        control("C_NEG_RANDOM_CODEBOOK", "NEGATIVE_CONTROL_OPERATOR", "quantizer with the same cell "
                "count whose edges are drawn independently of the data", ["STEP_04", "STEP_05"], True),
        control("C_NEG_CENTERED_LEAK", "NEGATIVE_CONTROL_OPERATOR", "centered two-sided filter that "
                "the prefix-invariance audit must flag as look-ahead", ["STEP_06", "STEP_07"], True),
        control("C_NEG_SIGNAL_NULL", "NEGATIVE_CONTROL_OPERATOR", "detector bank applied to "
                "signal-null synthetic series (noise only); any detection is a false alarm",
                ["STEP_07"], True),
    ]
    S, CAL = "SYNTHETIC_KNOWN_TRUTH", "CALIBRATION"
    metrics = [
        metric("M04_RECON_DISTORTION", "STEP_04", "reconstruction distortion against clean",
               "mean squared error between the dequantized output and the clean component, per "
               "family x noise x SNR band x length", "LOWER_IS_BETTER", [S], ["STEP_04_ARMS", RAW_ARM]),
        metric("M04_TAIL_PRESERVATION", "STEP_04", "tail event preservation",
               "recall of clean extreme events (beyond train-frozen tail quantiles) mapped to a "
               "distinct extreme cell", "HIGHER_IS_BETTER", [S, CAL], ["STEP_04_ARMS", RAW_ARM]),
        metric("M04_SATURATION_RATE", "STEP_04", "saturation rate",
               "fraction of samples outside the train-frozen range, per partition",
               "LOWER_IS_BETTER", [CAL], ["STEP_04_ARMS"]),
        metric("M04_DISCRETE_ENTROPY", "STEP_04", "discrete entropy under train-frozen cells",
               "plug-in entropy with Miller-Madow bias correction, bits per sample",
               "TWO_SIDED", [S, CAL], ["STEP_04_ARMS", "C_NEG_RANDOM_CODEBOOK"]),
        metric("M05_ENTROPY_RATE_GAIN", "STEP_05", "entropy-rate gain over permutation",
               "Lempel-Ziv (LZ78) entropy-rate estimate on quantized symbols minus the same "
               "estimate on block-permutation surrogates", "TWO_SIDED", [S, CAL],
               ["STEP_05_ARMS", "C_NULL_BLOCK_PERMUTATION"]),
        metric("M05_CODE_LENGTH_GAIN", "STEP_05", "compression gain",
               "compressed bytes per sample with the manifest compressor and level, relative to raw "
               "bytes and to permutation surrogates", "HIGHER_IS_BETTER", [S, CAL],
               ["S05_COMPRESSED_LENGTH", "C_NULL_BLOCK_PERMUTATION"]),
        metric("M05_ORDER_RECOVERY", "STEP_05", "context-order recovery",
               "agreement of the MDL-selected order with the generating order on known-order "
               "families, and its stability over train block bootstraps", "HIGHER_IS_BETTER", [S],
               ["S05_MDL_ORDER"]),
        metric("M05_RESIDUAL_WHITENESS", "STEP_05", "residual whiteness",
               "Ljung-Box statistic at lags predeclared in the D1 profile, input versus residual",
               "LOWER_IS_BETTER", [S, CAL], ["S05_PREDICTIVE_RESIDUAL", RAW_ARM]),
        metric("M06_PREFIX_INVARIANCE", "STEP_06", "causality audit",
               "count of time points whose output changes when samples after t are appended "
               "(batch versus truncated recomputation)", "MUST_BE_ZERO", [S],
               ["STEP_06_ARMS", "STEP_07_ARMS", "C_NEG_CENTERED_LEAK"]),
        metric("M06_AMP_FREQ_PHASE_ERROR", "STEP_06", "amplitude/frequency/phase recovery",
               "absolute amplitude and frequency error and circular mean absolute phase error "
               "against known synthetic components", "LOWER_IS_BETTER", [S], ["STEP_06_ARMS"]),
        metric("M06_SURROGATE_CONTRAST", "STEP_06", "structure beyond surrogates",
               "rank of the observed descriptor among IAAFT surrogates", "HIGHER_IS_BETTER", [S, CAL],
               ["STEP_06_ARMS", "C_NULL_IAAFT"]),
        metric("M07_DETECTION_POWER", "STEP_07", "detection probability at fixed false-alarm rate",
               "Neyman-Pearson detection probability at the false-alarm rate declared in the "
               "synthetic manifest, event onsets known by construction", "HIGHER_IS_BETTER", [S],
               ["STEP_07_ARMS", RAW_ARM]),
        metric("M07_FALSE_ALARM_NULL", "STEP_07", "false-alarm rate on signal-null series",
               "detections per sample on noise-only synthetic series against the nominal rate",
               "LOWER_IS_BETTER", [S], ["STEP_07_ARMS", "C_NEG_SIGNAL_NULL"]),
        metric("M07_DETECTION_DELAY", "STEP_07", "detection delay",
               "samples between known event onset and first detection", "LOWER_IS_BETTER", [S],
               ["STEP_07_ARMS"]),
        metric("M07_EVENT_PRESERVATION", "STEP_07", "event preservation (C137)",
               "recall of impulses, steps, bumps and motifs of the clean component after the "
               "representation", "HIGHER_IS_BETTER", [S], ["STEP_07_ARMS", RAW_ARM]),
    ] + [_cost_metric(s) for s in ("STEP_04", "STEP_05", "STEP_06", "STEP_07")]
    d = {
        "schema": D3_SCHEMA, "design_id": "D3", "status": STATUS, "order_item": "C141",
        "steps": SPECS["D3"]["steps"],
        "protocols": _protocols(SPECS["D3"]["steps"], repo_root),
        "consumes": _consumes("reviewed D2 noise-scale and SNR estimates (MODEL_CONDITIONAL_SNR_ESTIMATE) "
                              "per variable and regime, and reviewed denoised views, each only in "
                              "its reviewed regime"),
        "raw_control": _raw_control("STEP 04-07"),
        "questions": [
            question("Q04", "STEP_04", "What amplitude resolution, with cells frozen on training data, "
                     "preserves known clean structure and tail events given the reviewed D2 noise "
                     "scale, relative to raw floating point?"),
            question("Q05", "STEP_05", "What causal source structure (memory depth, contexts, "
                     "innovations) remains after quantization, beyond what permutation surrogates show?"),
            question("Q06", "STEP_06", "Which amplitude, frequency, phase and time-frequency "
                     "representations recover known components under a causal, online-feasible "
                     "contract?"),
            question("Q07", "STEP_07", "Given a representation and a noise model, which detector "
                     "family detects known patterns at a controlled false-alarm rate without look-ahead?"),
        ],
        "hypotheses": [
            hypothesis("H04.1", "STEP_04", "A noise-aware resolution reduces discrete entropy relative "
                       "to fine uniform cells without increasing reconstruction distortion against clean "
                       "beyond the raw control", "on every synthetic family the noise-aware arm has higher "
                       "distortion against clean than A0_RAW, or no entropy reduction",
                       "STEP_04 §5, §50"),
            hypothesis("H04.2", "STEP_04", "On heavy-tailed families, mu-law companding preserves tail "
                       "events better than uniform quantization at equal cell count",
                       "tail preservation of Q04_MU_LAW is not above Q04_UNIFORM on the heavy-tailed families",
                       "STEP_04 §16, §17, §50"),
            hypothesis("H05.1", "STEP_05", "The MDL-selected context order recovers the generating order "
                       "of known finite-memory families", "selected order differs from the generating order "
                       "or is unstable across train block bootstraps", "STEP_05 §34, §51"),
            hypothesis("H05.2", "STEP_05", "On independent noise the compression and entropy-rate gains "
                       "over permutation surrogates are absent", "the estimator reports a gain over "
                       "surrogates on iid synthetic noise (then the estimator is NOT_IDENTIFIABLE)",
                       "STEP_05 §54, §59"),
            hypothesis("H06.1", "STEP_06", "Every arm declared TIMESTAMP_CAUSAL or ENDPOINT_CAUSAL is "
                       "prefix-invariant", "any output at t changes when samples after t are appended",
                       "STEP_06 §12, §15"),
            hypothesis("H06.2", "STEP_06", "Trailing-window representations recover known amplitude, "
                       "frequency and phase within the analytic resolution of their window",
                       "recovery error exceeds the analytic bound declared for the window on "
                       "the clean-plus-white-noise family", "STEP_06 §58, §59, §62"),
            hypothesis("H07.1", "STEP_07", "Under colored noise the generalized matched filter has higher "
                       "detection power than the white matched filter at equal false-alarm rate",
                       "D07_GENERALIZED_MF power is not above D07_MATCHED_FILTER on colored-noise families",
                       "STEP_07 §45-§47, §64"),
            hypothesis("H07.2", "STEP_07", "On signal-null series every detector keeps its nominal "
                       "false-alarm rate", "false-alarm rate on noise-only series exceeds nominal",
                       "STEP_07 §76, §83"),
        ],
        "arms": arms,
        "controls": controls,
        "metrics": metrics,
        "synthetic_calibration_first": _synthetic_first([
            "sinusoids and multiband sums with known amplitude and phase", "trend, seasonality, chirps",
            "impulses, steps, bumps, motifs, regime changes", "white, colored, impulsive, cross-correlated "
            "and heteroscedastic noise", "heavy-tailed amplitude families", "known-order Markov symbol "
            "sources", "signal-null and noise-null controls", "MCAR and block missingness"]),
        "partitions": _partitions(TARGET_NOTE),
        "stopping_rules": COMMON_STOPS + [
            rule("SR_D3_CAUSALITY", "an arm fails the prefix-invariance audit",
                 "arm reclassified ANALYSIS_ONLY_NON_CAUSAL and never used as input"),
            rule("SR_D3_NO_NEURAL", "a detector family requires neural training (InceptionTime, "
                 "pretrained encoders, learned shapelets)", "not in this design; deferred to a separately "
                 "licensed phase"),
        ],
        "failure_region_reporting": _failure_regions(
            "arm x synthetic family x noise family x SNR band x length x regime",
            ["every cell count where tails collapse", "every representation that is analysis-only"]),
        "decision_vocabulary": _decision_vocabulary(),
        "budget": _budget(64, 3600, 8192, 4),
        "olap_grains": [
            grain("quantizer_run", ["run_id", "design_id", "dataset_sha256", "variable_id", "partition",
                                    "arm_id", "parameters_sha256", "code_sha256"],
                  ["cell_edges_sha256", "fit_partition", "saturation_rate", "record_state"]),
            grain("information_compression_metric", ["run_id", "variable_id", "partition", "arm_id",
                                                     "metric_id", "estimator_id"],
                  ["compressor", "level", "surrogate_id", "record_state"]),
            grain("representation_causality_audit", ["run_id", "arm_id", "causal_contract"],
                  ["prefix_invariance_count", "warm_up_samples", "reclassified", "record_state"]),
            grain("detector_run", ["run_id", "arm_id", "template_bank_sha256", "noise_family", "snr_band"],
                  ["detection_power", "false_alarm_rate", "detection_delay", "record_state"]),
            grain("raw_transformed_residual_metric", ["run_id", "variable_id", "arm_id", "regime_id",
                                                      "metric_id"], ["control_id", "record_state"]),
            grain("delay_and_cost", ["run_id", "arm_id", "regime_id"],
                  ["cpu_seconds", "peak_rss_mib", "algorithmic_delay", "warm_up", "record_state"]),
            grain("lab_decision", ["run_id", "arm_id", "regime_id"],
                  ["lab_decision", "evidence_sha256", "record_state"]),
            grain("failure_region", ["run_id", "arm_id", "region_id"],
                  ["region_definition", "lab_decision", "record_state"]),
        ],
        "grants_nothing": GRANTS_NOTHING,
        "license": _license(),
        "causal_feasibility": {"rule": NON_CAUSAL_RULE, "entries": feas},
    }
    return _finish(d)


# ----------------------------------------------------------------------- D4
def build_d4(repo_root=REPO) -> dict:
    TF, CU, TC, NC = ("TRAIN_ONLY_FROZEN", "CAUSAL_ONLINE_UPDATE_FROM_TRAIN_INIT",
                      "TIMESTAMP_CAUSAL", "ANALYSIS_ONLY_NON_CAUSAL")
    V = VERIFIED_AVAILABILITY
    arms = [
        raw_arm("untransformed, as-of joined on each source's verified availability time"),
        # STEP 08
        arm("E08_TRAIN_AFFINE", "STEP_08", "train-frozen affine canonicalization",
            "robust location and scale (median, MAD) fit on train and frozen", ["scale_estimator"],
            TF, TC, True, [V]),
        arm("E08_TRAILING_ROBUST", "STEP_08", "trailing robust normalization",
            "median and MAD over a trailing window ending at t", ["window_length"], "NONE", TC, True, [V]),
        arm("E08_PREWHITEN", "STEP_08", "train-frozen inverse (prewhitening) filter",
            "causal AR inverse filter fit on train, frozen", ["ar_order"], TF, TC, True, [V]),
        arm("E08_RLS_EQUALIZER", "STEP_08", "RLS adaptive affine equalizer",
            "recursive least squares update from past samples only, initialized on train",
            ["forgetting_factor", "regularization"], CU, TC, True, [V]),
        # STEP 09
        arm("X09_TRAIN_PCA_COMMON_PRIVATE", "STEP_09", "train-fit PCA common/private split",
            "projection fit on common training samples; delivers [X, C, U], never destructive subtraction",
            ["components"], TF, TC, True, [V, "C132_GROUP_IDENTIFIED"]),
        arm("X09_LAGGED_RESIDUALIZATION", "STEP_09", "cross-feature residualization on lagged references",
            "ridge projection on strictly past reference samples, fit on train", ["reference_lags", "ridge"],
            TF, TC, True, [V, "C132_GROUP_IDENTIFIED"]),
        arm("X09_ANC_LMS", "STEP_09", "adaptive noise cancellation (LMS)",
            "normalized LMS with reference r_{<=t}; delivers [d, nhat, e]", ["taps", "step_size"],
            CU, TC, True, [V]),
        arm("X09_FULLSAMPLE_ICA", "STEP_09", "full-sample ICA",
            "ICA over the whole series; diagnostic of mixing only", ["components"], "NONE", NC, False, []),
        # STEP 10
        arm("L10_ASOF_AVAILABILITY", "STEP_10", "availability-only as-of join",
            "each source enters at its verified availability time; no shift", [], "NONE", TC, True, [V]),
        arm("L10_XCORR_TRAIN_LAG", "STEP_10", "train-estimated cross-correlation lag",
            "lag estimated on train; the follower at t is paired with the leader at t - lag (past only)",
            ["max_lag"], TF, TC, True, [V]),
        arm("L10_GCC_PHAT_TRAIN_LAG", "STEP_10", "GCC-PHAT lag",
            "phase-transform weighted cross-correlation delay on train, frozen", ["max_lag", "window_length"],
            TF, TC, True, [V]),
        arm("L10_ROLLING_LAG_METADATA", "STEP_10", "trailing lag and confidence as metadata",
            "lag and coherence estimated on a trailing window, delivered as metadata, no data shift",
            ["window_length", "max_lag"], "NONE", TC, True, [V]),
        arm("L10_DTW_ANALYSIS", "STEP_10", "unconstrained DTW",
            "two-sided warping path; diagnostic of elastic timing only", ["band"], "NONE", NC, False, []),
    ]
    controls = [
        control("C_RAW", "RAW", "A0_RAW on the same partition, sample, regime and budget",
                ["STEP_08", "STEP_09", "STEP_10"], False),
        control("C_NULL_CIRCULAR_SHIFT", "NULL_SURROGATE", "circular time shift of one source against "
                "the others within train, destroying cross-dependence and keeping autocorrelation",
                ["STEP_09", "STEP_10"], False),
        control("C_NULL_MULTIVARIATE_IAAFT", "NULL_SURROGATE", "multivariate phase-randomized surrogate "
                "for the lag significance test", ["STEP_10"], False),
        control("C_NULL_DOMAIN_LABEL_PERMUTATION", "NULL_SURROGATE", "domain labels permuted across "
                "training windows before fitting equalizers", ["STEP_08"], False),
        control("C_NEG_CONTAMINATED_REFERENCE", "NEGATIVE_CONTROL_OPERATOR", "cancellation with a "
                "reference that contains the desired signal (STEP 09 S1); must be flagged destructive",
                ["STEP_09"], True),
        control("C_NEG_FUTURE_ALIGNMENT", "NEGATIVE_CONTROL_OPERATOR", "alignment that pairs t with the "
                "follower at t + lag; the availability audit must refuse it", ["STEP_10"], True),
        control("C_NEG_COMMON_PERIODICITY", "NEGATIVE_CONTROL_OPERATOR", "two sources sharing only a "
                "periodicity with no propagation (STEP 10 T7); any lag found is false", ["STEP_10"], True),
        control("C_NEG_MARGINAL_ONLY_EQUALIZER", "NEGATIVE_CONTROL_OPERATOR", "quantile-mapping to a "
                "reference marginal under a conditional mechanism shift (STEP 08 C8)", ["STEP_08"], True),
    ]
    S, CAL = "SYNTHETIC_KNOWN_TRUTH", "CALIBRATION"
    metrics = [
        metric("M08_CANONICAL_RECON", "STEP_08", "canonical reconstruction error",
               "mean squared error between equalized output and the known canonical process on "
               "synthetic channels C0-C9", "LOWER_IS_BETTER", [S], ["STEP_08_ARMS", RAW_ARM]),
        metric("M08_DOMAIN_ALIGNMENT", "STEP_08", "cross-domain distribution alignment",
               "Wasserstein-1 distance between domains of train-frozen outputs, per variable",
               "LOWER_IS_BETTER", [S, CAL], ["STEP_08_ARMS", RAW_ARM, "C_NULL_DOMAIN_LABEL_PERMUTATION"]),
        metric("M08_INFORMATION_PRESERVATION", "STEP_08", "information preservation",
               "KSG k-nearest-neighbour mutual information between output and known canonical "
               "process, k declared in the manifest", "HIGHER_IS_BETTER", [S],
               ["STEP_08_ARMS", RAW_ARM, "C_NEG_MARGINAL_ONLY_EQUALIZER"]),
        metric("M08_REVERSIBILITY", "STEP_08", "reversibility error",
               "reconstruction error of the input from output and frozen parameters", "LOWER_IS_BETTER",
               [S, CAL], ["E08_TRAIN_AFFINE", "E08_PREWHITEN"]),
        metric("M09_COMMON_RECOVERY", "STEP_09", "common-factor recovery",
               "canonical correlation between recovered C and the known common factors (S2, S3) and "
               "its stability over train block bootstraps", "HIGHER_IS_BETTER", [S],
               ["X09_TRAIN_PCA_COMMON_PRIVATE", "X09_LAGGED_RESIDUALIZATION"]),
        metric("M09_CANCELLATION_SAFETY", "STEP_09", "desired-signal distortion",
               "scale-invariant signal-to-distortion ratio of e against the known desired signal, clean "
               "versus contaminated reference", "HIGHER_IS_BETTER", [S],
               ["X09_ANC_LMS", "C_NEG_CONTAMINATED_REFERENCE"]),
        metric("M09_RESIDUAL_CROSS_DEPENDENCE", "STEP_09", "residual cross-dependence",
               "maximum absolute cross-correlation of U across sources at predeclared lags against "
               "circular-shift surrogates", "LOWER_IS_BETTER", [S, CAL],
               ["STEP_09_ARMS", "C_NULL_CIRCULAR_SHIFT"]),
        metric("M10_LAG_ERROR", "STEP_10", "lag recovery error",
               "absolute error in samples between estimated and known delay (T0, T2)", "LOWER_IS_BETTER",
               [S], ["STEP_10_ARMS"]),
        metric("M10_FALSE_LAG_RATE", "STEP_10", "false lag rate",
               "fraction of surrogate-significant lags on T6/T7 families and surrogate pairs",
               "LOWER_IS_BETTER", [S, CAL], ["STEP_10_ARMS", "C_NULL_MULTIVARIATE_IAAFT",
                                            "C_NEG_COMMON_PERIODICITY"]),
        metric("M10_LAG_STABILITY", "STEP_10", "lag stability",
               "dispersion of the train lag estimate over block bootstraps", "LOWER_IS_BETTER",
               [CAL], ["L10_XCORR_TRAIN_LAG", "L10_GCC_PHAT_TRAIN_LAG"]),
        metric("M10_AVAILABILITY_VIOLATIONS", "STEP_10", "availability audit",
               "count of emitted values using any observation whose verified availability time is "
               "after the decision time", "MUST_BE_ZERO", [S, CAL],
               ["ALL_ARMS", "C_NEG_FUTURE_ALIGNMENT"]),
    ] + [_cost_metric(s) for s in ("STEP_08", "STEP_09", "STEP_10")]
    d = {
        "schema": D4_SCHEMA, "design_id": "D4", "status": STATUS, "order_item": "C142",
        "steps": SPECS["D4"]["steps"],
        "protocols": _protocols(SPECS["D4"]["steps"], repo_root),
        "consumes": _consumes("reviewed D2 denoised views and noise estimates per variable and regime; "
                              "C132 groups only when not GROUP_NOT_IDENTIFIED"),
        "raw_control": _raw_control("STEP 08-10"),
        "questions": [
            question("Q08", "STEP_08", "Can known systematic channel distortions be compensated causally "
                     "so that sources and regimes become canonical without erasing known structure?"),
            question("Q09", "STEP_09", "Can shared and private components be separated on training data "
                     "without destroying the desired signal when the reference is contaminated?"),
            question("Q10", "STEP_10", "After availability is verified, are information-bearing events "
                     "aligned, and can a train-estimated lag be applied without any future sample?"),
        ],
        "hypotheses": [
            hypothesis("H08.1", "STEP_08", "On gain, offset, drift and FIR channels (C1-C3) causal or "
                       "train-frozen equalization reduces canonical reconstruction error relative to raw",
                       "no STEP_08 arm improves on A0_RAW on C1-C3", "STEP_08 §36-§38, §69"),
            hypothesis("H08.2", "STEP_08", "Under a conditional mechanism shift (C8), marginal alignment "
                       "does not imply information preservation", "the marginal-only negative control "
                       "preserves information as well as the raw control on C8", "STEP_08 §43, §47, §76"),
            hypothesis("H09.1", "STEP_09", "The cancellation-safety metric separates a clean reference "
                       "(S0) from a contaminated reference (S1)", "safety on S1 is not below safety on S0",
                       "STEP_09 §39, §40, §72, §87"),
            hypothesis("H09.2", "STEP_09", "Train-fit common/private decomposition recovers known common "
                       "factors stably on S2/S3", "recovery unstable across train bootstraps (group then "
                       "GROUP_NOT_IDENTIFIED)", "STEP_09 §41, §42, §74"),
            hypothesis("H10.1", "STEP_10", "Train-estimated cross-correlation and GCC-PHAT recover a known "
                       "fixed delay (T0)", "lag error exceeds one sample in the SNR bands declared "
                       "identifiable by D2", "STEP_10 §65, §89"),
            hypothesis("H10.2", "STEP_10", "Lag without causality (T6) and common periodicity (T7) are "
                       "flagged by the surrogate test", "the surrogate test accepts the false lag",
                       "STEP_10 §71, §72, §105"),
            hypothesis("H10.3", "STEP_10", "No alignment, equalization or cancellation arm emits a value "
                       "using an observation not yet available", "any availability violation; the arm is "
                       "then LAB_REJECTED", "STEP_10 §33, §34, §95"),
        ],
        "arms": arms,
        "controls": controls,
        "metrics": metrics,
        "synthetic_calibration_first": _synthetic_first([
            "equalization channels C0-C9 (STEP 08 §35-§44)",
            "cancellation benchmarks S0-S7 (STEP 09 §39-§46)",
            "synchronization benchmarks T0-T8 (STEP 10 §65-§73)",
            "asynchronous sampling with declared availability delays",
            "signal-null and noise-null controls"]),
        "partitions": _partitions(TARGET_NOTE),
        "stopping_rules": COMMON_STOPS + [
            rule("SR_D4_AVAILABILITY_UNVERIFIED", "a source has no verified availability declaration",
                 "every lead/lag, cancellation and equalization arm using that source is REFUSED"),
            rule("SR_D4_FUTURE_SAMPLE", "the availability audit counts any violation",
                 "arm LAB_REJECTED in every regime; no correction that uses a future sample is kept"),
            rule("SR_D4_GROUP_NOT_IDENTIFIED", "C132 reports GROUP_NOT_IDENTIFIED",
                 "common/private arms not run for that group; recorded NOT_RUN"),
        ],
        "failure_region_reporting": _failure_regions(
            "arm x synthetic channel/benchmark x noise family x SNR band x lag x regime",
            ["every contaminated-reference region where cancellation destroys signal",
             "every false-lag region"]),
        "decision_vocabulary": _decision_vocabulary(),
        "budget": _budget(48, 3600, 8192, 4),
        "olap_grains": [
            grain("availability_contract", ["dataset_sha256", "source_id", "contract_sha256"],
                  ["declared_availability_rule", "verification_state", "record_state"]),
            grain("equalizer_run", ["run_id", "design_id", "dataset_sha256", "variable_id", "partition",
                                    "arm_id", "parameters_sha256", "code_sha256"],
                  ["fit_partition", "record_state"]),
            grain("common_private_decomposition", ["run_id", "group_id", "arm_id", "partition"],
                  ["projection_sha256", "bootstrap_stability", "record_state"]),
            grain("lag_estimate", ["run_id", "source_pair", "arm_id", "partition"],
                  ["availability_contract_sha256", "lag_samples", "surrogate_id", "record_state"]),
            grain("availability_audit", ["run_id", "arm_id"], ["violation_count", "record_state"]),
            grain("raw_transformed_residual_metric", ["run_id", "variable_id", "arm_id", "regime_id",
                                                      "metric_id"], ["control_id", "record_state"]),
            grain("delay_and_cost", ["run_id", "arm_id", "regime_id"],
                  ["cpu_seconds", "peak_rss_mib", "algorithmic_delay", "warm_up", "record_state"]),
            grain("lab_decision", ["run_id", "arm_id", "regime_id"],
                  ["lab_decision", "evidence_sha256", "record_state"]),
            grain("failure_region", ["run_id", "arm_id", "region_id"],
                  ["region_definition", "lab_decision", "record_state"]),
        ],
        "grants_nothing": GRANTS_NOTHING,
        "license": _license(),
        "temporal_availability": {
            "availability_precedes_alignment": True,
            "rule": AVAILABILITY_RULE,
            "order": list(AVAILABILITY_ORDER),
            "availability_function": "A_i(t): the latest observation of source i actually available "
                                     "to the system by decision time t (STEP 10 §33)",
            "verification": ["availability declared per source in the C129 contract",
                             "availability checked against publication/arrival timestamps in D0",
                             "C133 gaps, jitter and truncated bars accounted before any join",
                             "prefix-invariance and availability audits on synthetic data with "
                             "known delays"],
            "future_samples_in_correction": "FORBIDDEN",
            "lead_lag_correction_without_verified_availability": "REFUSED",
            "delay_compensation_by_future_shift": "FORBIDDEN",
        },
    }
    return _finish(d)


# ----------------------------------------------------------------------- D5
def build_d5(repo_root=REPO) -> dict:
    TF, TC, NC = "TRAIN_ONLY_FROZEN", "TIMESTAMP_CAUSAL", "ANALYSIS_ONLY_NON_CAUSAL"
    D2 = "D2_REVIEWED_LAB_DECISION"
    arms = [
        raw_arm("untransformed branch; always available and the abstention target"),
        # STEP 11
        arm("R11_MULTIVIEW", "STEP_11", "redundant multi-view encoding",
            "raw plus reviewed D2 views delivered together", ["views"], "NONE", TC, True, [D2]),
        arm("R11_MEDIAN_OF_VIEWS", "STEP_11", "robust combination of views",
            "sample-wise median of reviewed causal views", ["views"], "NONE", TC, True, [D2]),
        arm("R11_MASK_CAUSAL_FILL", "STEP_11", "train-only masking with causal fill",
            "masks applied on train/synthetic only; filled by last observation or the reviewed D2 "
            "state-space view", ["mask_rate", "mask_block_length", "fill_method"], TF, TC, True, [D2]),
        # STEP 12
        arm("P12_FIXED_MODE", "STEP_12", "fixed mode baseline",
            "each reviewed mode used constantly; one arm instance per mode", ["mode"], "NONE", TC, True, [D2]),
        arm("P12_IQS_THRESHOLD", "STEP_12", "causal information-quality threshold router",
            "mode chosen from a causal quality state (reviewed D2 SNR estimate, missingness, OOD "
            "distance) with thresholds fit on train", ["quality_components", "thresholds_fit"],
            TF, TC, True, [D2]),
        arm("P12_IQS_HYSTERESIS", "STEP_12", "threshold router with hysteresis",
            "as P12_IQS_THRESHOLD with a minimum dwell time against chatter",
            ["quality_components", "dwell_samples"], TF, TC, True, [D2]),
        arm("P12_ORACLE_ANALYSIS", "STEP_12", "oracle router",
            "best mode per block known by construction on synthetic data; regret reference only",
            [], "NONE", NC, False, []),
        # STEP 13
        arm("B13_UNIFORM", "STEP_13", "uniform allocation",
            "equal share of the total cost cap to every active branch", [], "NONE", TC, True, []),
        arm("B13_WATERFILL", "STEP_13", "water-filling allocation",
            "allocation increasing with train-estimated branch quality, zero allowed", ["quality_source"],
            TF, TC, True, [D2]),
        arm("B13_REDUNDANCY_AWARE", "STEP_13", "redundancy-discounted allocation",
            "water-filling with a discount for train-estimated redundancy between branches",
            ["redundancy_estimator"], TF, TC, True, [D2]),
        arm("B13_RESERVE", "STEP_13", "allocation with reserve",
            "unallocated budget kept when marginal utility on train is not above surrogates",
            ["quanta"], TF, TC, True, []),
    ]
    all_ids = [a["arm_id"] for a in arms]
    cand = [a["arm_id"] for a in arms if a["role"] == "CANDIDATE" and a["usable_as_input"]]
    budget_block = {
        "cost_unit": "CPU_MILLISECONDS_PER_1000_SAMPLES",
        "total_cost_cap": 400,
        "branches": [
            {"branch_id": "B0_RAW", "description": "raw branch; abstention target",
             "arm_ids": all_ids, "cost_cap": 20, "memory_mib_cap": 256,
             "on_cap_exceeded": "RECORD_FAILED_BUDGET_EXCEEDED_NO_EXTENSION"},
            {"branch_id": "B1_D2_REVIEWED_VIEWS", "description": "reviewed D2 operator views, each in "
             "its reviewed regime", "arm_ids": cand,
             "cost_cap": 160, "memory_mib_cap": 1024, "on_cap_exceeded": "ROUTE_TO_RAW_BRANCH"},
            {"branch_id": "B2_REDUNDANT_MULTIVIEW", "description": "STEP 11 redundant combinations of "
             "B0 and B1", "arm_ids": ["R11_MULTIVIEW", "R11_MEDIAN_OF_VIEWS", "R11_MASK_CAUSAL_FILL"],
             "cost_cap": 160, "memory_mib_cap": 1024, "on_cap_exceeded": "ROUTE_TO_RAW_BRANCH"},
            {"branch_id": "B3_ROUTER_ALLOCATOR_OVERHEAD", "description": "quality state, router and "
             "allocator computation", "arm_ids": ["P12_IQS_THRESHOLD", "P12_IQS_HYSTERESIS",
                                                  "B13_WATERFILL", "B13_REDUNDANCY_AWARE", "B13_RESERVE"],
             "cost_cap": 60, "memory_mib_cap": 256, "on_cap_exceeded": "ROUTE_TO_RAW_BRANCH"},
        ],
        "sum_of_active_costs_within_total": True,
        "reserve_allowed": True,
        "abstention": {
            "allowed": True, "target_branch": "B0_RAW", "target_arm": RAW_ARM,
            "when": ["information quality state unavailable or in warm-up",
                     "quality state outside the reviewed regime of the D2 decision",
                     "a D2 view whose decision is not in {LAB_CALIBRATED, REGIME_LIMITED} with review",
                     "a branch reaches its cost cap", "an input is not available at decision time",
                     "the router cannot choose a mode"],
            "recorded": True,
            "rule": ABSTENTION_RULE},
    }
    controls = [
        control("C_RAW", "RAW", "A0_RAW on the same partition, sample, regime and budget",
                ["STEP_11", "STEP_12", "STEP_13"], False),
        control("C_NULL_PERMUTED_IQS", "NULL_SURROGATE", "quality state block-permuted within train "
                "before thresholds are fit", ["STEP_12"], False),
        control("C_NULL_RANDOM_ROUTING", "NULL_SURROGATE", "random mode sequence with the same switch "
                "rate and mode frequencies as the router", ["STEP_12"], False),
        control("C_NULL_RANDOM_MASK_VIEWS", "NULL_SURROGATE", "redundant views replaced by independent "
                "noise with matched marginals", ["STEP_11"], True),
        control("C_NEG_USELESS_STATE", "NEGATIVE_CONTROL_OPERATOR", "router driven by a quality state "
                "independent of mode superiority (STEP 12 benchmark E)", ["STEP_12"], True),
        control("C_NEG_NOISE_BRANCH_ALLOCATION", "NEGATIVE_CONTROL_OPERATOR", "allocation that gives "
                "the budget to a pure-noise synthetic branch", ["STEP_13"], True),
    ]
    S, CAL = "SYNTHETIC_KNOWN_TRUTH", "CALIBRATION"
    metrics = [
        metric("M11_CORRUPTION_DEGRADATION", "STEP_11", "degradation under corruption",
               "increase of clean-recovery mean squared error from uncorrupted to corrupted synthetic "
               "input, per mask rate and block length", "LOWER_IS_BETTER", [S],
               ["STEP_11_ARMS", RAW_ARM, "C_NULL_RANDOM_MASK_VIEWS"]),
        metric("M11_REPRESENTATION_STABILITY", "STEP_11", "representation stability",
               "median absolute difference between outputs on clean and corrupted input, scaled by "
               "the train-frozen MAD", "LOWER_IS_BETTER", [S, CAL], ["STEP_11_ARMS", RAW_ARM]),
        metric("M12_ROUTING_REGRET", "STEP_12", "regret against the oracle mode",
               "mean clean-recovery loss of routed output minus oracle-mode loss on synthetic "
               "benchmarks A-F", "LOWER_IS_BETTER", [S],
               ["STEP_12_ARMS", "C_NULL_RANDOM_ROUTING", "C_NEG_USELESS_STATE"]),
        metric("M12_CHATTER", "STEP_12", "route chatter", "mode switches per sample",
               "LOWER_IS_BETTER", [S, CAL], ["P12_IQS_THRESHOLD", "P12_IQS_HYSTERESIS"]),
        metric("M12_ABSTENTION", "STEP_12", "abstention rate and abstention-conditioned loss",
               "fraction of samples routed to B0_RAW by abstention, and loss on those samples",
               "TWO_SIDED", [S, CAL], ["STEP_12_ARMS", RAW_ARM]),
        metric("M12_IQS_CAUSALITY", "STEP_12", "quality-state causality audit",
               "count of quality values changing when samples after t are appended", "MUST_BE_ZERO",
               [S], ["P12_IQS_THRESHOLD", "P12_IQS_HYSTERESIS"]),
        metric("M13_ALLOCATION_EFFICIENCY", "STEP_13", "allocation efficiency",
               "clean-recovery loss reduction relative to A0_RAW per unit of spent cost",
               "HIGHER_IS_BETTER", [S], ["STEP_13_ARMS", "C_NEG_NOISE_BRANCH_ALLOCATION"]),
        metric("M13_PARETO_FRONTIER", "STEP_13", "loss-cost-memory frontier",
               "non-dominated set over (clean-recovery loss, CPU cost, peak memory) per budget quantum",
               "TWO_SIDED", [S, CAL], ["STEP_13_ARMS", RAW_ARM]),
        metric("M13_BUDGET_VIOLATIONS", "STEP_13", "budget audit",
               "count of blocks where a branch exceeds its cost cap or the total exceeds the total cap",
               "MUST_BE_ZERO", [S, CAL], ["ALL_ARMS"]),
    ] + [_cost_metric(s) for s in ("STEP_11", "STEP_12", "STEP_13")]
    d = {
        "schema": D5_SCHEMA, "design_id": "D5", "status": STATUS, "order_item": "C143",
        "steps": SPECS["D5"]["steps"],
        "protocols": _protocols(SPECS["D5"]["steps"], repo_root),
        "consumes": _consumes("reviewed D2 operator views as branches, and reviewed D2 SNR/noise "
                              "estimates as quality-state components, each only in its reviewed regime"),
        "raw_control": _raw_control("STEP 11-13"),
        "questions": [
            question("Q11", "STEP_11", "Does deliberate redundancy across reviewed causal views reduce "
                     "degradation under train-only masking and corruption relative to raw?"),
            question("Q12", "STEP_12", "Does a causal information-quality router choose among reviewed "
                     "modes better than the best fixed mode, and does it abstain to raw when it should?"),
            question("Q13", "STEP_13", "Under a declared cost cap per branch and in total, does "
                     "non-uniform allocation beat uniform allocation, and is leaving budget unused "
                     "sometimes optimal?"),
        ],
        "hypotheses": [
            hypothesis("H11.1", "STEP_11", "Redundant multi-view encodings degrade less than raw under "
                       "train-only corruption", "no STEP_11 arm shows smaller degradation than A0_RAW",
                       "STEP_11 §59 H11.1"),
            hypothesis("H11.2", "STEP_11", "The benefit is not universal: some corruption level erases it",
                       "every tested corruption level keeps the benefit (tested range insufficient; no "
                       "universal claim)", "STEP_11 §59 H11.2"),
            hypothesis("H12.1", "STEP_12", "On benchmarks with SNR- or missingness-dependent mode "
                       "superiority (A, B) a causal router has lower regret than the best fixed mode",
                       "router regret is not below the best fixed mode on A and B", "STEP_12 §48, §49, §74"),
            hypothesis("H12.2", "STEP_12", "With a useless quality state (E) the router does not beat the "
                       "best fixed mode", "a gain is reported on E (router claims then NOT_IDENTIFIABLE)",
                       "STEP_12 §52, §74"),
            hypothesis("H12.3", "STEP_12", "Hysteresis reduces chatter (F) without raising regret above "
                       "the threshold router", "chatter not reduced, or regret higher", "STEP_12 §53, §74"),
            hypothesis("H13.1", "STEP_13", "With branches of unequal known SNR, water-filling allocation "
                       "beats uniform allocation at the same total cap", "B13_WATERFILL efficiency is not "
                       "above B13_UNIFORM", "STEP_13 §25, §26, §100"),
            hypothesis("H13.2", "STEP_13", "With equal branch quality, uniform allocation is not dominated",
                       "a non-uniform allocation dominates on equal-quality families (estimator overfit)",
                       "STEP_13 §69, §100"),
            hypothesis("H13.3", "STEP_13", "A pure-noise branch receives zero budget and the reserve is "
                       "used when capacity adds no utility", "allocating to the pure-noise branch improves "
                       "efficiency", "STEP_13 §66, §100"),
        ],
        "arms": arms,
        "controls": controls,
        "metrics": metrics,
        "synthetic_calibration_first": _synthetic_first([
            "routing benchmarks A-F (STEP 12 §48-§53)",
            "parallel branches of unequal and equal known SNR",
            "pure-noise branch", "MCAR and block missingness masks",
            "signal-null and noise-null controls"]),
        "partitions": _partitions(TARGET_NOTE),
        "stopping_rules": COMMON_STOPS + [
            rule("SR_D5_NO_NEURAL_TRAINING", "STEP 11 corruption-hardening of the existing feature "
                 "extractor (§59) requires neural training", "recorded DEFERRED with no arm in this design"),
            rule("SR_D5_NO_REVIEWED_MODES", "fewer than two reviewed modes exist for a regime",
                 "routing and allocation for that regime NOT_RUN; output is the raw branch"),
            rule("SR_D5_BUDGET_VIOLATION", "the budget audit counts any violation",
                 "arm LAB_REJECTED; route to raw branch"),
        ],
        "failure_region_reporting": _failure_regions(
            "arm x synthetic benchmark x quality-state regime x corruption level x budget quantum",
            ["every region where abstention to raw beats routing", "every region where uniform allocation "
             "is not dominated"]),
        "decision_vocabulary": _decision_vocabulary(),
        "budget": _budget(24, 1800, 4096, 4),
        "olap_grains": [
            grain("redundancy_view_run", ["run_id", "design_id", "dataset_sha256", "variable_id", "partition",
                                          "arm_id", "parameters_sha256", "code_sha256"],
                  ["views_sha256", "corruption_level", "record_state"]),
            grain("router_decision", ["run_id", "arm_id", "block_id"],
                  ["quality_state_sha256", "mode", "abstained", "record_state"]),
            grain("abstention_event", ["run_id", "arm_id", "block_id"], ["reason", "record_state"]),
            grain("budget_allocation", ["run_id", "arm_id", "branch_id", "budget_quantum"],
                  ["allocated_cost", "branch_cost_cap", "total_cost_cap", "record_state"]),
            grain("delay_and_cost", ["run_id", "arm_id", "regime_id"],
                  ["cpu_seconds", "peak_rss_mib", "algorithmic_delay", "warm_up", "record_state"]),
            grain("lab_decision", ["run_id", "arm_id", "regime_id"],
                  ["lab_decision", "evidence_sha256", "record_state"]),
            grain("failure_region", ["run_id", "arm_id", "region_id"],
                  ["region_definition", "lab_decision", "record_state"]),
        ],
        "grants_nothing": GRANTS_NOTHING,
        "license": _license(),
        "budget_and_abstention": budget_block,
    }
    return _finish(d)


BUILDERS = {"D3": build_d3, "D4": build_d4, "D5": build_d5}


def build_all(repo_root=REPO) -> dict:
    return {k: f(repo_root) for k, f in BUILDERS.items()}


# ---------------------------------------------------------------- validator
def _str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _int_pos(x) -> bool:
    return type(x) is int and x > 0


def _str_list(x, *, nonempty=True, unique=True) -> bool:
    return (isinstance(x, list) and (bool(x) or not nonempty) and all(_str(s) for s in x)
            and (not unique or len(set(x)) == len(x)))


def _walk(node, path=""):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _exact(obj, keys, where, p) -> bool:
    if not isinstance(obj, dict):
        p.append(f"TYPE: {where} must be an object")
        return False
    if set(obj) != set(keys):
        p.append(f"KEYS: {where} expected {sorted(keys)}, got {sorted(obj)}")
        return False
    return True


def _records(lst, keys, where, p) -> list:
    if not isinstance(lst, list) or not lst:
        p.append(f"TYPE: {where} must be a non-empty list")
        return []
    ok = []
    for i, r in enumerate(lst):
        if _exact(r, keys, f"{where}[{i}]", p):
            ok.append(r)
    return ok


NUMERIC_ALLOWED = re.compile(
    r"^(budget\.(cpu_core_hours_cap|wall_seconds_cap_per_arm_regime|peak_memory_mib_cap|"
    r"parallel_workers_cap)|budget_and_abstention\.total_cost_cap|"
    r"budget_and_abstention\.branches\[\d+\]\.(cost_cap|memory_mib_cap))$")


def _global_scans(d, p):
    home = str(Path.home())
    for path, node in _walk(d):
        leaf = path.rsplit(".", 1)[-1].split("[")[0] if path else ""
        if isinstance(node, dict):
            for k in node:
                if k.lower() in RESULT_KEYS:
                    p.append(f"RESULT_FIELD_IN_DESIGN: {path + '.' if path else ''}{k}")
                if FORBIDDEN_STATE in k:
                    p.append(f"PUBLICLY_ELIGIBLE_AS_KEY: {path}.{k}")
        elif type(node) in (int, float):
            if type(node) is float and not math.isfinite(node):
                p.append(f"NON_FINITE: {path}")
            if not NUMERIC_ALLOWED.match(path):
                p.append(f"NUMERIC_VALUE_OUTSIDE_BUDGET_CAPS: {path}")
        elif isinstance(node, str):
            if home in node or node.startswith("/home/") or "/home/" in node:
                p.append(f"ABSOLUTE_HOME_PATH: {path}")
            if FORBIDDEN_STATE in node:
                if node.strip() == FORBIDDEN_STATE or leaf in DECISION_KEYS \
                        or not NEVER_GRANTS.search(node):
                    p.append(f"PUBLICLY_ELIGIBLE_AS_DECISION_OR_STATE: {path}")


def _check_protocols(d, spec, repo_root, p):
    prots = _records(d.get("protocols"), ("file", "sha256"), "protocols", p)
    root = Path(repo_root).resolve()
    names = []
    for i, e in enumerate(prots):
        f, s = e["file"], e["sha256"]
        if not (_str(f) and isinstance(s, str) and HEX64.fullmatch(s)):
            p.append(f"TYPE: protocols[{i}] file/sha256")
            continue
        if Path(f).is_absolute() or ".." in Path(f).parts or not f.startswith(PROTOCOL_DIR + "/"):
            p.append(f"PROTOCOL_PATH_NOT_RELATIVE_UNDER_{PROTOCOL_DIR}: {f}")
            continue
        fp = root / f
        if not fp.is_file():
            p.append(f"PROTOCOL_MISSING_ON_DISK: {f}")
            continue
        if file_sha256(fp) != s:
            p.append(f"PROTOCOL_SHA_MISMATCH: {f}")
        names.append(Path(f).name)
    for step in spec["steps"]:
        if not any(n.startswith(step + "_") for n in names):
            p.append(f"PROTOCOL_NOT_CITED_FOR: {step}")
    if len(names) != len(set(names)):
        p.append("DUPLICATE_PROTOCOL")


def _check_consumes(c, p):
    keys = ("accepted_d2_decisions", "never_consumed", "external_review_record_required",
            "unreviewed_d2_output", "d2_grain", "review_record_fields", "d0_d1_inputs", "d2_use",
            "successor_outputs", "rule")
    if not _exact(c, keys, "consumes", p):
        return
    acc = c["accepted_d2_decisions"]
    if not _str_list(acc) or not set(acc) <= set(CONSUMABLE):
        p.append(f"CONSUMES_UNACCEPTABLE_D2_STATE: accepted_d2_decisions must be a non-empty subset "
                 f"of {list(CONSUMABLE)}, got {acc!r}")
    if not isinstance(c["never_consumed"], list) or set(c["never_consumed"]) != set(NEVER_CONSUMED) \
            or len(c["never_consumed"]) != len(NEVER_CONSUMED):
        p.append("CONSUMES_RULE: never_consumed must list exactly NOT_IDENTIFIABLE and LAB_REJECTED")
    if c["external_review_record_required"] is not True:
        p.append("CONSUMES_UNREVIEWED: external_review_record_required must be true")
    if c["unreviewed_d2_output"] != "NEVER_CONSUMED":
        p.append("CONSUMES_UNREVIEWED: unreviewed_d2_output must be NEVER_CONSUMED")
    for k in ("d2_grain", "d2_use", "successor_outputs", "rule"):
        if not _str(c[k]):
            p.append(f"TYPE: consumes.{k}")
    if not _str_list(c["review_record_fields"]) or "review_record_sha256" not in c["review_record_fields"]:
        p.append("CONSUMES_UNREVIEWED: review_record_fields must include review_record_sha256")
    if not _str_list(c["d0_d1_inputs"]):
        p.append("TYPE: consumes.d0_d1_inputs")
    rule_text = c["rule"] if isinstance(c["rule"], str) else ""
    for word in ("NOT_IDENTIFIABLE", "LAB_REJECTED", "never consumed", "external review record"):
        if word not in rule_text:
            p.append(f"CONSUMES_RULE: rule must state {word!r}")


def _check_arms(d, spec, p) -> dict:
    keys = ("arm_id", "step", "role", "family", "description", "parameters", "fit",
            "causal_contract", "usable_as_input", "requires")
    arms = _records(d.get("arms"), keys, "arms", p)
    by_id = {}
    for i, a in enumerate(arms):
        w = f"arms[{i}]"
        for k in ("arm_id", "family", "description"):
            if not _str(a[k]):
                p.append(f"TYPE: {w}.{k}")
        if a["arm_id"] in by_id:
            p.append(f"DUPLICATE_ARM: {a['arm_id']}")
        by_id[a["arm_id"]] = a
        if a["role"] not in ARM_ROLES:
            p.append(f"DOMAIN: {w}.role")
        if a["fit"] not in FITS:
            p.append(f"DOMAIN: {w}.fit")
        if a["causal_contract"] not in CAUSAL_CONTRACTS:
            p.append(f"DOMAIN: {w}.causal_contract")
        if type(a["usable_as_input"]) is not bool:
            p.append(f"TYPE: {w}.usable_as_input must be a boolean")
        if not _str_list(a["parameters"], nonempty=False) or not _str_list(a["requires"], nonempty=False):
            p.append(f"TYPE: {w}.parameters/requires must be distinct names")
        else:
            for r in a["requires"]:
                if any(s in r for s in NEVER_CONSUMED):
                    p.append(f"CONSUMES_UNACCEPTABLE_D2_STATE: {w}.requires {r}")
        if a["role"] == "CANDIDATE" and a["step"] not in spec["steps"]:
            p.append(f"DOMAIN: {w}.step {a['step']!r} not in {spec['steps']}")
        if a["role"] == "RAW_CONTROL" and a["step"] != "ALL":
            p.append(f"DOMAIN: {w} raw control must cover step ALL")
        if a["causal_contract"] == "ANALYSIS_ONLY_NON_CAUSAL" and a["usable_as_input"] is not False:
            p.append(f"NON_CAUSAL_USED_AS_INPUT: {a['arm_id']}")
    for step in spec["steps"]:
        if not any(a["role"] == "CANDIDATE" and a["step"] == step for a in arms):
            p.append(f"NO_CANDIDATE_ARM_FOR: {step}")
    return by_id


def _check_raw(d, by_id, p):
    rc = d.get("raw_control")
    if _exact(rc, ("kept", "arm_id", "replaced_by_transform", "rule"), "raw_control", p):
        if rc["kept"] is not True or rc["replaced_by_transform"] is not False or rc["arm_id"] != RAW_ARM \
                or not _str(rc["rule"]):
            p.append("MISSING_RAW_CONTROL: raw_control must keep A0_RAW and never replace it")
    raw = by_id.get(RAW_ARM)
    if raw is None or raw.get("role") != "RAW_CONTROL" or raw.get("fit") != "NONE" \
            or raw.get("parameters") != [] or raw.get("usable_as_input") is not True:
        p.append("MISSING_RAW_CONTROL: arms must contain A0_RAW as an untransformed RAW_CONTROL arm")
    if sum(1 for a in by_id.values() if a.get("role") == "RAW_CONTROL") != 1:
        p.append("MISSING_RAW_CONTROL: exactly one RAW_CONTROL arm")


def _check_controls(d, spec, p):
    keys = ("control_id", "kind", "description", "steps", "synthetic_only", "consumed_as_input")
    cs = _records(d.get("controls"), keys, "controls", p)
    ids = set()
    for i, c in enumerate(cs):
        w = f"controls[{i}]"
        if not _str(c["control_id"]) or c["control_id"] in ids or not _str(c["description"]):
            p.append(f"TYPE: {w} id/description")
        ids.add(c["control_id"])
        if c["kind"] not in CONTROL_KINDS:
            p.append(f"DOMAIN: {w}.kind")
        if not _str_list(c["steps"]) or not set(c["steps"]) <= set(spec["steps"]):
            p.append(f"DOMAIN: {w}.steps")
        if type(c["synthetic_only"]) is not bool:
            p.append(f"TYPE: {w}.synthetic_only")
        if c["consumed_as_input"] is not False:
            p.append(f"CONTROL_CONSUMED_AS_INPUT: {w}")
    kinds = {c["kind"] for c in cs}
    if "RAW" not in kinds:
        p.append("MISSING_RAW_CONTROL: controls must include kind RAW")
    for k in ("NULL_SURROGATE", "NEGATIVE_CONTROL_OPERATOR"):
        if k not in kinds:
            p.append(f"MISSING_CONTROL: {k}")
    return ids


def _check_metrics(d, spec, arm_ids, control_ids, p):
    keys = ("metric_id", "step", "name", "estimator", "direction", "partitions", "compares")
    ms = _records(d.get("metrics"), keys, "metrics", p)
    ids = set()
    refs = set(arm_ids) | set(control_ids) | {"ALL_ARMS"} | {f"{s}_ARMS" for s in spec["steps"]}
    for i, m in enumerate(ms):
        w = f"metrics[{i}]"
        for k in ("metric_id", "name", "estimator"):
            if not _str(m[k]):
                p.append(f"METRIC_WITHOUT_{k.upper()}: {w}")
        if m["metric_id"] in ids:
            p.append(f"DUPLICATE_METRIC: {m['metric_id']}")
        ids.add(m["metric_id"])
        if m["step"] not in spec["steps"]:
            p.append(f"DOMAIN: {w}.step")
        if m["direction"] not in DIRECTIONS:
            p.append(f"DOMAIN: {w}.direction")
        if not _str_list(m["partitions"]) or not set(m["partitions"]) <= set(METRIC_PARTITIONS):
            p.append(f"DOMAIN: {w}.partitions")
        if not _str_list(m["compares"]) or not set(m["compares"]) <= refs:
            p.append(f"DOMAIN: {w}.compares references unknown arms or controls")
    for step in spec["steps"]:
        if not any(m["step"] == step for m in ms):
            p.append(f"NO_METRIC_FOR: {step}")


def _check_q_h(d, spec, p):
    qs = _records(d.get("questions"), ("question_id", "step", "question"), "questions", p)
    hs = _records(d.get("hypotheses"), ("hypothesis_id", "step", "statement", "falsified_if",
                                        "protocol_section"), "hypotheses", p)
    for name, rows, idk in (("questions", qs, "question_id"), ("hypotheses", hs, "hypothesis_id")):
        ids = [r[idk] for r in rows]
        if len(ids) != len(set(ids)):
            p.append(f"DUPLICATE_ID: {name}")
        for i, r in enumerate(rows):
            if r["step"] not in spec["steps"] or not all(_str(v) for v in r.values()):
                p.append(f"TYPE_OR_DOMAIN: {name}[{i}]")
        for step in spec["steps"]:
            if not any(r["step"] == step for r in rows):
                p.append(f"NO_{name.upper()}_FOR: {step}")
    for i, h in enumerate(hs):
        if not _str(h["falsified_if"]):
            p.append(f"HYPOTHESIS_NOT_FALSIFIABLE: hypotheses[{i}]")
        elif isinstance(h["protocol_section"], str) and not h["protocol_section"].startswith(h["step"]):
            p.append(f"HYPOTHESIS_SECTION_NOT_IN_ITS_STEP: hypotheses[{i}]")


def _check_blocks(d, p):
    s = d.get("synthetic_calibration_first")
    if _exact(s, ("required", "order", "generator_bank", "families", "parameters_frozen_before_real_data",
                  "real_data_before_calibration", "rule"), "synthetic_calibration_first", p):
        if s["required"] is not True or s["order"] != ["SYNTHETIC_KNOWN_TRUTH", "PUBLIC", "FINANCIAL"] \
                or s["parameters_frozen_before_real_data"] is not True \
                or s["real_data_before_calibration"] != "REFUSED" or not _str_list(s["families"]) \
                or not _str(s["generator_bank"]) or not _str(s["rule"]):
            p.append("SYNTHETIC_CALIBRATION_FIRST: synthetic known truth must precede public and financial data")
    pa = d.get("partitions")
    if _exact(pa, ("fit", "decide", "confirm", "mixing_partitions", "sealed_periods", "test_set_tuning",
                   "target_use", "target_exception", "rule"), "partitions", p):
        want = {"fit": "TRAIN_ONLY", "decide": "CALIBRATION", "confirm": "CONFIRMATION",
                "mixing_partitions": "FORBIDDEN", "sealed_periods": "NEVER_READ",
                "test_set_tuning": "FORBIDDEN", "target_use": "NONE"}
        for k, v in want.items():
            if pa[k] != v:
                p.append(f"PARTITIONS: {k} must be {v}")
        if not _str(pa["rule"]) or not (_str(pa["target_exception"]) and "training folds" in pa["target_exception"]):
            p.append("PARTITIONS: target_exception must confine any target to training folds of a later test")
    rules = _records(d.get("stopping_rules"), ("rule_id", "condition", "action"), "stopping_rules", p)
    if any(not all(_str(v) for v in r.values()) for r in rules) \
            or len({r["rule_id"] for r in rules}) != len(rules):
        p.append("STOPPING_RULES: typed, distinct rules required")
    f = d.get("failure_region_reporting")
    if _exact(f, ("required", "grain", "publish", "best_case_only", "states_recorded", "rule"),
              "failure_region_reporting", p):
        if f["required"] is not True or f["best_case_only"] != "REFUSED" or not _str_list(f["publish"]) \
                or set(f["states_recorded"]) != set(RECORD_STATES) or not _str(f["grain"]):
            p.append("FAILURE_REGION_REPORTING: failure regions must be published, best case alone refused")
    dv = d.get("decision_vocabulary")
    if _exact(dv, ("decisions", "grain", "none_equals_public_eligibility", "statement"),
              "decision_vocabulary", p):
        if dv["decisions"] != list(LAB_DECISIONS):
            p.append(f"DECISION_VOCABULARY: decisions must be exactly {list(LAB_DECISIONS)}")
        if dv["none_equals_public_eligibility"] is not True or not isinstance(dv["statement"], str) \
                or not NEVER_GRANTS.search(dv["statement"]) or FORBIDDEN_STATE not in dv["statement"]:
            p.append("DECISION_VOCABULARY: must state that none equals public eligibility and that "
                     "PUBLICLY_ELIGIBLE is never granted")
    b = d.get("budget")
    if _exact(b, ("device", "accelerator", "cpu_core_hours_cap", "wall_seconds_cap_per_arm_regime",
                  "peak_memory_mib_cap", "parallel_workers_cap", "same_cap_for_every_arm", "on_cap"),
              "budget", p):
        if b["device"] != "CPU_ONLY" or b["accelerator"] != "NONE" or b["same_cap_for_every_arm"] is not True \
                or not all(_int_pos(b[k]) for k in ("cpu_core_hours_cap", "wall_seconds_cap_per_arm_regime",
                                                   "peak_memory_mib_cap", "parallel_workers_cap")) \
                or b["on_cap"] != "RECORD_FAILED_BUDGET_EXCEEDED_NO_EXTENSION":
            p.append("BUDGET: CPU only, no accelerator, positive integer caps, no extension")
    gs = _records(d.get("olap_grains"), ("grain", "keys", "records"), "olap_grains", p)
    names = [g["grain"] for g in gs]
    if len(names) != len(set(names)) or not all(_str(n) and _str_list(g["keys"]) and _str_list(g["records"])
                                                for n, g in zip(names, gs)):
        p.append("OLAP_GRAINS: distinct named grains with keys and records")
    for need in ("lab_decision", "failure_region", "delay_and_cost"):
        if need not in names:
            p.append(f"OLAP_GRAINS: missing {need}")
    if not (isinstance(d.get("grants_nothing"), str) and "grants nothing" in d["grants_nothing"]):
        p.append("GRANTS_NOTHING: must state that the design grants nothing")
    lic = d.get("license")
    if _exact(lic, ("scoring", "execution", "consumption", "selection", "training", "required"), "license", p):
        if any(lic[k] != "NOT_GRANTED" for k in ("scoring", "execution", "consumption", "selection", "training")) \
                or lic["required"] != LICENSE_REQUIRED:
            p.append("LICENSE: scoring, execution, consumption, selection and training must be NOT_GRANTED")


def _check_d3(d, by_id, p):
    cf = d.get("causal_feasibility")
    if not _exact(cf, ("rule", "entries"), "causal_feasibility", p):
        return
    if cf["rule"] != NON_CAUSAL_RULE:
        p.append("D3_NON_CAUSAL_RULE_MISSING: causal_feasibility.rule must state that any non-causal "
                 "transform is analysis-only and never an input")
    keys = ("arm_id", "representation", "causal_contract", "online_feasible", "usable_as_input",
            "warm_up", "boundary_rule")
    es = _records(cf["entries"], keys, "causal_feasibility.entries", p)
    seen = set()
    for i, e in enumerate(es):
        w = f"causal_feasibility.entries[{i}]"
        a = by_id.get(e["arm_id"])
        if a is None or a["role"] != "CANDIDATE" or e["arm_id"] in seen:
            p.append(f"D3_FEASIBILITY_UNKNOWN_OR_DUPLICATE_ARM: {w}")
            continue
        seen.add(e["arm_id"])
        if e["causal_contract"] != a["causal_contract"] or e["usable_as_input"] != a["usable_as_input"]:
            p.append(f"D3_FEASIBILITY_DISAGREES_WITH_ARM: {e['arm_id']}")
        if type(e["online_feasible"]) is not bool or not _str(e["warm_up"]) or not _str(e["boundary_rule"]):
            p.append(f"TYPE: {w}")
        nc = e["causal_contract"] == "ANALYSIS_ONLY_NON_CAUSAL"
        if nc and (e["online_feasible"] is not False or e["usable_as_input"] is not False):
            p.append(f"NON_CAUSAL_USED_AS_INPUT: {e['arm_id']}")
        if not nc and e["online_feasible"] is not True:
            p.append(f"D3_CAUSAL_ARM_NOT_ONLINE: {e['arm_id']}")
    missing = {k for k, a in by_id.items() if a["role"] == "CANDIDATE"} - seen
    if missing:
        p.append(f"D3_FEASIBILITY_NOT_DECLARED_FOR: {sorted(missing)}")
    for a in by_id.values():
        if a["step"] == "STEP_04" and a["role"] == "CANDIDATE" and a["fit"] != "TRAIN_ONLY_FROZEN":
            p.append(f"D3_QUANTIZER_NOT_TRAIN_FROZEN: {a['arm_id']}")


def _check_d4(d, by_id, p):
    ta = d.get("temporal_availability")
    keys = ("availability_precedes_alignment", "rule", "order", "availability_function", "verification",
            "future_samples_in_correction", "lead_lag_correction_without_verified_availability",
            "delay_compensation_by_future_shift")
    if not _exact(ta, keys, "temporal_availability", p):
        p.append("D4_AVAILABILITY_RULE_MISSING")
        return
    if ta["availability_precedes_alignment"] is not True or ta["rule"] != AVAILABILITY_RULE \
            or ta["order"] != list(AVAILABILITY_ORDER) \
            or ta["lead_lag_correction_without_verified_availability"] != "REFUSED" \
            or not _str(ta["availability_function"]) or not _str_list(ta["verification"]):
        p.append("D4_AVAILABILITY_RULE_MISSING: availability must be declared and verified before any "
                 "lead/lag correction")
    if ta["future_samples_in_correction"] != "FORBIDDEN" or ta["delay_compensation_by_future_shift"] != "FORBIDDEN":
        p.append("D4_FUTURE_SAMPLES_ALLOWED: no correction may use future samples")
    for a in by_id.values():
        if a["role"] != "CANDIDATE" or a["usable_as_input"] is not True:
            continue
        if a["causal_contract"] != "TIMESTAMP_CAUSAL":
            p.append(f"D4_FUTURE_SAMPLES_ALLOWED: {a['arm_id']} is used as input but not TIMESTAMP_CAUSAL")
        if VERIFIED_AVAILABILITY not in a["requires"]:
            p.append(f"D4_AVAILABILITY_RULE_MISSING: {a['arm_id']} does not require {VERIFIED_AVAILABILITY}")


def _check_d5(d, by_id, p):
    ba = d.get("budget_and_abstention")
    keys = ("cost_unit", "total_cost_cap", "branches", "sum_of_active_costs_within_total",
            "reserve_allowed", "abstention")
    if not _exact(ba, keys, "budget_and_abstention", p):
        p.append("D5_BUDGET_OR_ABSTENTION_MISSING")
        return
    if not _str(ba["cost_unit"]) or not _int_pos(ba["total_cost_cap"]) \
            or ba["sum_of_active_costs_within_total"] is not True or type(ba["reserve_allowed"]) is not bool:
        p.append("D5_BUDGET_OR_ABSTENTION_MISSING: cost unit, positive total cap and the sum rule")
    bkeys = ("branch_id", "description", "arm_ids", "cost_cap", "memory_mib_cap", "on_cap_exceeded")
    brs = _records(ba["branches"], bkeys, "budget_and_abstention.branches", p)
    covered, raw_branch = set(), None
    for i, b in enumerate(brs):
        w = f"budget_and_abstention.branches[{i}]"
        if not _int_pos(b["cost_cap"]) or not _int_pos(b["memory_mib_cap"]):
            p.append(f"D5_BRANCH_WITHOUT_COST_CAP: {w}")
        elif _int_pos(ba["total_cost_cap"]) and b["cost_cap"] > ba["total_cost_cap"]:
            p.append(f"D5_BRANCH_CAP_ABOVE_TOTAL: {w}")
        if not _str_list(b["arm_ids"]) or not set(b["arm_ids"]) <= set(by_id):
            p.append(f"DOMAIN: {w}.arm_ids")
        else:
            covered |= set(b["arm_ids"])
            if RAW_ARM in b["arm_ids"]:
                raw_branch = b
        if b["on_cap_exceeded"] not in ("ROUTE_TO_RAW_BRANCH", "RECORD_FAILED_BUDGET_EXCEEDED_NO_EXTENSION"):
            p.append(f"DOMAIN: {w}.on_cap_exceeded")
    if len({b["branch_id"] for b in brs}) != len(brs):
        p.append("DUPLICATE_BRANCH")
    runnable = {k for k, a in by_id.items() if a["role"] == "CANDIDATE" and a["usable_as_input"]}
    if runnable - covered:
        p.append(f"D5_BRANCH_WITHOUT_COST_CAP: arms outside any capped branch {sorted(runnable - covered)}")
    if raw_branch is None or not set(runnable) <= set(raw_branch["arm_ids"]):
        p.append("D5_BUDGET_OR_ABSTENTION_MISSING: the raw branch must be reachable from every arm")
    ab = ba["abstention"]
    if not _exact(ab, ("allowed", "target_branch", "target_arm", "when", "recorded", "rule"),
                  "budget_and_abstention.abstention", p):
        p.append("D5_BUDGET_OR_ABSTENTION_MISSING")
        return
    if ab["allowed"] is not True or ab["recorded"] is not True or ab["rule"] != ABSTENTION_RULE \
            or not _str_list(ab["when"]):
        p.append("D5_BUDGET_OR_ABSTENTION_MISSING: routing must be allowed to abstain to raw, recorded")
    if ab["target_arm"] != RAW_ARM or raw_branch is None or ab["target_branch"] != raw_branch["branch_id"]:
        p.append("D5_ABSTENTION_NOT_TO_RAW: abstention target must be the raw branch and A0_RAW")


def validate(d, *, repo_root=REPO) -> list[str]:
    if not isinstance(d, dict):
        return ["TYPE: design must be an object"]
    did = d.get("design_id")
    spec = SPECS.get(did)
    if spec is None:
        return [f"DESIGN_ID: unknown {did!r}"]
    p: list[str] = []
    want = set(COMMON_KEYS) | {spec["specific"]}
    if set(d) != want:
        p.append(f"KEYS: missing {sorted(want - set(d))}, extra {sorted(set(d) - want)}")
        if spec["specific"] not in d:
            p.append({"D3": "D3_NON_CAUSAL_RULE_MISSING", "D4": "D4_AVAILABILITY_RULE_MISSING",
                      "D5": "D5_BUDGET_OR_ABSTENTION_MISSING"}[did])
        if "raw_control" not in d:
            p.append("MISSING_RAW_CONTROL")
    if d.get("schema") != spec["schema"] or d.get("status") != STATUS \
            or d.get("order_item") != spec["order_item"] or d.get("steps") != spec["steps"]:
        p.append("IDENTITY: schema, status, order_item or steps")
    ds = d.get("design_sha256")
    if not (isinstance(ds, str) and ds == sha_obj({k: v for k, v in d.items() if k != "design_sha256"})):
        p.append("DESIGN_SHA256_DOES_NOT_REDERIVE")
    _global_scans(d, p)
    try:
        _check_protocols(d, spec, repo_root, p)
        _check_consumes(d.get("consumes"), p)
        by_id = _check_arms(d, spec, p)
        _check_raw(d, by_id, p)
        control_ids = _check_controls(d, spec, p)
        _check_metrics(d, spec, set(by_id), control_ids, p)
        _check_q_h(d, spec, p)
        _check_blocks(d, p)
        if did == "D3" and "causal_feasibility" in d:
            _check_d3(d, by_id, p)
        if did == "D4" and "temporal_availability" in d:
            _check_d4(d, by_id, p)
        if did == "D5" and "budget_and_abstention" in d:
            _check_d5(d, by_id, p)
    except (KeyError, TypeError, AttributeError) as exc:
        p.append(f"STRUCTURE: {exc!r}")
    return p


def validate_file(path, *, repo_root=REPO) -> list[str]:
    try:
        return validate(strict_json_loads(Path(path).read_bytes()), repo_root=repo_root)
    except (StrictJsonRefusal, ValueError) as exc:
        return [f"STRICT_JSON: {exc}"]


# ---------------------------------------------------------------------- CLI
def write_designs(out_dir, repo_root=REPO) -> dict:
    """Write-once: refuse before writing anything if any target exists."""
    out = Path(out_dir)
    targets = {k: out / s["file"] for k, s in SPECS.items()}
    existing = sorted(t.name for t in targets.values() if t.exists())
    if existing:
        raise WriteRefusal(f"REFUSED: write-once, already present: {existing}")
    docs = build_all(repo_root)
    texts = {}
    for k, doc in docs.items():
        problems = validate(doc, repo_root=repo_root)
        if problems:
            raise WriteRefusal(f"REFUSED: {k} does not validate: {problems}")
        text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if str(Path.home()) in text:
            raise WriteRefusal("REFUSED: absolute home path in a published document")
        texts[k] = text
    out.mkdir(parents=True, exist_ok=True)
    for k, t in targets.items():
        with open(t, "x", encoding="utf-8") as fh:
            fh.write(texts[k])
    return {k: docs[k]["design_sha256"] for k in docs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-dir", type=Path, help="write the three designs (write-once)")
    ap.add_argument("--validate", type=Path, nargs="*", default=[], help="design files to validate")
    a = ap.parse_args(argv)
    rc = 0
    if a.write_dir:
        try:
            print(json.dumps({"design_sha256": write_designs(a.write_dir)}, indent=1, sort_keys=True))
        except WriteRefusal as exc:
            print(str(exc), file=sys.stderr)
            return 2
    for f in a.validate:
        problems = validate_file(f)
        print(json.dumps({"file": Path(f).name, "problems": problems}, indent=1, sort_keys=True))
        rc = 1 if problems else rc
    if not a.write_dir and not a.validate:
        ap.print_help()
    return rc


if __name__ == "__main__":
    sys.exit(main())
