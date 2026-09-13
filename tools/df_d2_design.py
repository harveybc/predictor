#!/usr/bin/env python3
"""C171 (order 2026-09-13): the superseding D2 v2 design, sealed before any result.

A design is a JSON document built only from explicit inputs and bound by the
sha256 of its canonical body (``design_sha256``). ``validate_design`` re-derives
the seal and checks every field C171 requires:

* bank, contract module and temporal roles (TRAIN fits, CALIBRATION is
  diagnostic, CONFIRMATION alone governs a fresh decision);
* every operator spec with its current parameters, its fit mode (declared for
  the kind in ``df_operators.KIND_FIT_MODES``), its arm role and the operator
  code digest it was designed against;
* the three branches RAW ``X``, TRANSFORMED ``D(X)`` and RESIDUAL ``X - D(X)``;
* metrics of improvement, distortion, delay, events, extremes and cost with
  their estimators;
* abstention, support and missing-data rules;
* unit of analysis = one independent generator seed (never a time row);
* parameter selection only in historical development (C137), primary
  confirmation only on the fresh reserve, without retuning;
* multiplicity by operator family and a non-inferiority rule;
* budget, timeout, memory and role assignment (CPU only);
* seeds per regime fixed BEFORE the reserve exists: minimum 10, maximum 30,
  target power 0.80, ``UNDERPOWERED`` when 30 does not reach it.

Seed count (``required_seeds``). For a paired one-sided test of H0: mu <= null
against the seed-level effect d_i (one value per seed), with historical mean m
and standard deviation s (floored at ``sd_floor``), effect delta = m - null and
the operator-family adjusted level alpha_adj = alpha / m_family (Bonferroni,
m_family = number of candidate specs of the same operator kind), the power at n
seeds is the exact non-central t power

    power(n) = 1 - F_nct( t_{1 - alpha_adj, n - 1} ; df = n - 1, nc = delta * sqrt(n) / s )

and n is the smallest integer in [10, 30] with power(n) >= 0.80. When delta <= 0
(the historical effect does not exceed the null) or power(30) < 0.80 the answer
is ``UNDERPOWERED`` with n = 30. A regime with no historical dispersion for a
candidate is ``UNDERPOWERED`` for that candidate (unknown dispersion never
claims power). Noise-free regimes (declared SNR inf) power the distortion test
instead: d_i = distortion_ratio_max - distortion_ratio_i, null 0.

C137 dispersion (``extract_c137_dispersion``), read-only. Metric
``snr_improvement_db`` and ``distortion_ratio`` of
``df_fact_operator_signal_metric`` rows with status COMPLETED on partition
``confirmation`` (the partition that governs the fresh test), joined to their
``df_fact_operator_run`` row by ``row_sha256``; aggregation: mean over the
unit's variables gives ONE value per unit (= seed); across the seeds of an
operator x regime the table reports n, mean and sample sd (ddof=1). The retired
name ``wavelet_haar_atrous`` is mapped to ``trailing_haar_threshold`` and the
mapping is recorded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN_SCHEMA = "crispdm.data_foundation.d2_design.v2"
DISPERSION_SCHEMA = "crispdm.data_foundation.c137_dispersion.v1"
SEEDS_MIN, SEEDS_MAX, POWER_TARGET = 10, 30, 0.80
HISTORICAL_MODE = "HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY"
FRESH_MODE = "FRESH_CONFIRMATION"
MODES = (HISTORICAL_MODE, FRESH_MODE)
ARM_ROLES = ("CANDIDATE", "IDENTITY_RAW_CONTROL", "PREVIOUSLY_REJECTED_CONTROL", "NON_CAUSAL_ORACLE_CONTROL")
RETIRED_NAMES = {"wavelet_haar_atrous": "trailing_haar_threshold"}
REGIME_FIELDS = ("family", "perturbation", "declared_snr_db", "length", "missingness")
D2_CODE_FILES = ("df_d2_design", "df_seed_tape", "df_d2_unit_worker", "df_d2_adjudicate", "df_lab_evaluation",
                 "df_synthetic_bank", "df_synthetic_contract", "df_snr", "df_operators", "df_snapshot", "df_contract",
                 "df_isolated_runner", "df_profile_run")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DesignRefusal(ValueError):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lab_code_sha256s() -> dict:
    return {m: file_sha(HERE / f"{m}.py") for m in D2_CODE_FILES if (HERE / f"{m}.py").is_file()}


def lab_code_sha256() -> str:
    return sha_obj(lab_code_sha256s())


def regime_key(regime: dict) -> str:
    missing = [f for f in REGIME_FIELDS if f not in regime]
    if missing or set(regime) != set(REGIME_FIELDS):
        raise DesignRefusal(f"a regime is exactly {list(REGIME_FIELDS)}; never pooled over a field (got {sorted(regime)})")
    return json.dumps({k: regime[k] for k in REGIME_FIELDS}, sort_keys=True)


def regime_of_cell(cell: dict) -> dict:
    return {"family": cell["family"], "perturbation": cell["perturbation"], "declared_snr_db": str(cell["snr_db"]),
            "length": int(cell["length"]), "missingness": cell["missingness"]["kind"]}


def spec_sha(spec: dict) -> str:
    return sha_obj({"kind": spec["kind"], "params": spec["params"]})


# ------------------------------------------------------------------ rules
DENOISING_RULES = {
    "version": "d2v2.c176.v1",
    "partition": "confirmation",
    "unit_of_analysis": "SEED",
    "seed_aggregation": {"improvement": "mean over variables", "retention": "min over variables (worst)",
                         "delay_leakage_cost": "max over variables (worst)"},
    "alpha": 0.05,
    "multiplicity": {"family": "OPERATOR_KIND_WITHIN_REGIME", "method": "BONFERRONI_ONE_SIDED",
                     "m": "number of CANDIDATE specs of the kind"},
    "improvement": {"metric": "snr_improvement_db", "margin_db": 0.0, "min_mean_effect_db": 1.0,
                    "seed_floor_db": 0.0},
    "non_inferiority": {"pairs": {"impulse_retention": "impulse_retention_raw", "motif_corr": "motif_corr_raw",
                                  "extreme_retention": "extreme_retention_raw"}, "margin": 0.1},
    "event_floors": {"impulse_retention": ["min", 0.5], "bump_retention": ["min", 0.5],
                     "motif_corr": ["min", 0.5], "extreme_retention": ["min", 0.7],
                     "step_delay_samples": ["max", 8.0], "regime_mean_latency_samples": ["max", 48.0]},
    "noise_free": {"distortion_ratio_max": 0.1},
    "delay": {"metric": "delay_samples", "max_samples": 8.0},
    "residual_leakage": {"metric": "residual_signal_share", "max": 0.1},
    "cost": {"metric": "cpu_seconds_per_1000_samples", "max": 1.0},
    "abstention": {"max_rate": 0.10},
    "precedence": [
        "CONTROL_NOT_AN_ARM for the non-causal oracle: never decided, only detected",
        "NOT_IDENTIFIABLE if arms are not paired on the same seeds, fewer valid seeds than the design, or abstention above max",
        "UNDERPOWERED if the sealed design marked the spec x regime UNDERPOWERED",
        "LAB_REJECTED if ANY seed destroys an event or extreme (floors) or leaks signal into the residual",
        "noise-free: LAB_REJECTED unless upper bound of mean distortion and every seed are within the maximum",
        "LAB_REJECTED if the upper bound of the mean improvement is below the minimum effect; NOT_IDENTIFIABLE if inconclusive",
        "LAB_REJECTED if non-inferiority is refuted (upper bound below -margin); NOT_IDENTIFIABLE if inconclusive",
        "LAB_REJECTED if any seed exceeds the delay limit or the cost limit",
        "REGIME_LIMITED if some seed does not improve (below seed floor) although the mean does",
        "LAB_CALIBRATED otherwise: improvement, non-inferiority, events/extremes, delay, leakage and cost all pass"],
    "never": "no decision equals or implies public eligibility; the raw branch is always kept",
}

SNR_RULES = {
    "version": "d2v2.c175.v1",
    "partition": "confirmation",
    "unit_of_analysis": "SEED",
    "abs_error": "|snr_hat - snr_true| per variable, mean over the seed's variables, THEN averaged across seeds",
    "ci_level": 0.95,
    "mean_abs_error_ci_upper_max_db": 1.0,
    "coverage_min": 0.90,
    "not_identifiable_rate_max": 0.10,
    "precedence": [
        "SNR_NOT_IDENTIFIABLE if the estimator has no confirmation estimate by contract, fewer identifiable seeds than the design, or NOT_IDENTIFIABLE rate above max",
        "SNR_CALIBRATED_FOR_REGIME if the 95% upper bound of the mean absolute error <= 1 dB and coverage >= 0.90",
        "SNR_REJECTED if the 95% lower bound of the mean absolute error > 1 dB",
        "SNR_REGIME_LIMITED otherwise"],
    "never": "no general estimator from an average across perturbations or regimes; real data keep "
             "MODEL_CONDITIONAL_SNR_ESTIMATE",
}

METRICS = {
    "rmse_raw": {"branch": "RAW", "estimator": "sqrt(mean((x - clean)^2)) on available, non-missing samples"},
    "snr_raw_db": {"branch": "RAW", "estimator": "10 log10(var(clean) / mse(x))"},
    "impulse_retention_raw": {"branch": "RAW", "estimator": "(x_i - median pre-10) / height"},
    "motif_corr_raw": {"branch": "RAW", "estimator": "corr(x, clean) over the motif"},
    "extreme_retention_raw": {"branch": "RAW", "estimator": "mean|x - med| / mean|clean - med| at the top 1% |clean - med|"},
    "rmse_denoised": {"branch": "TRANSFORMED", "estimator": "sqrt(mean((D - clean)^2))"},
    "snr_denoised_db": {"branch": "TRANSFORMED", "estimator": "10 log10(var(clean) / mse(D))"},
    "amplitude_ratio": {"branch": "TRANSFORMED", "estimator": "cov(D, clean) / var(clean)"},
    "distortion_ratio": {"branch": "TRANSFORMED", "estimator": "rmse(D) / sd(clean)"},
    "delay_samples": {"branch": "TRANSFORMED", "estimator": "lag k in [0, 32] maximizing corr(D[k:], clean[:-k])"},
    "impulse_retention": {"branch": "TRANSFORMED", "estimator": "c137 event_metrics"},
    "bump_retention": {"branch": "TRANSFORMED", "estimator": "c137 event_metrics"},
    "step_delay_samples": {"branch": "TRANSFORMED", "estimator": "c137 event_metrics"},
    "motif_corr": {"branch": "TRANSFORMED", "estimator": "c137 event_metrics"},
    "regime_mean_latency_samples": {"branch": "TRANSFORMED", "estimator": "c137 event_metrics"},
    "extreme_retention": {"branch": "TRANSFORMED", "estimator": "c137 partition_metrics"},
    "residual_signal_share": {"branch": "RESIDUAL", "estimator": "corr(X - D, clean)^2"},
    "residual_excess_acf1": {"branch": "RESIDUAL", "estimator": "acf1(X - D) - acf1(noise)"},
    "snr_improvement_db": {"branch": "COMPARISON", "estimator": "snr_denoised_db - snr_raw_db (paired in the seed)"},
    "rmse_ratio": {"branch": "COMPARISON", "estimator": "rmse_denoised / rmse_raw"},
    "support": {"branch": "COMPARISON", "estimator": "count of available, non-missing samples"},
    "cpu_seconds_fit": {"branch": "COST", "estimator": "process_time of fit"},
    "cpu_seconds_transform": {"branch": "COST", "estimator": "process_time of transform"},
    "cpu_seconds_per_1000_samples": {"branch": "COST", "estimator": "(fit + transform) / T * 1000"},
}


# ------------------------------------------------------------------ power
def power_at(n: int, delta: float, sd: float, alpha: float) -> float:
    from scipy import stats
    if n < 2 or sd <= 0:
        return 0.0
    tcrit = stats.t.ppf(1.0 - alpha, n - 1)
    return float(1.0 - stats.nct.cdf(tcrit, n - 1, delta * math.sqrt(n) / sd))


def required_seeds(mean_effect, sd, *, null: float = 0.0, alpha: float = 0.05, family_m: int = 1,
                   power: float = POWER_TARGET, n_min: int = SEEDS_MIN, n_max: int = SEEDS_MAX,
                   sd_floor: float = 0.25) -> dict:
    """-> {"status": POWERED|UNDERPOWERED, "n": int in [n_min, n_max], "power_at_n", ...}."""
    if type(family_m) is not int or family_m < 1:
        raise DesignRefusal("family_m must be a positive integer")
    a = alpha / family_m
    out = {"formula": "paired one-sided t; power(n)=1-F_nct(t_{1-a,n-1}; n-1, delta*sqrt(n)/s)",
           "alpha": alpha, "family_m": family_m, "alpha_adjusted": a, "null": null, "target_power": power,
           "n_min": n_min, "n_max": n_max, "sd_floor": sd_floor, "mean_effect": mean_effect, "sd_input": sd}
    if mean_effect is None or not math.isfinite(float(mean_effect)):
        return dict(out, status="UNDERPOWERED", n=n_max, power_at_n=None, reason="NO_HISTORICAL_DISPERSION")
    s = float(sd) if sd is not None and math.isfinite(float(sd)) and float(sd) > sd_floor else sd_floor
    delta = float(mean_effect) - null
    out.update(sd_used=s, delta=delta)
    if delta <= 0:
        return dict(out, status="UNDERPOWERED", n=n_max, power_at_n=power_at(n_max, delta, s, a),
                    reason="HISTORICAL_EFFECT_DOES_NOT_EXCEED_NULL")
    for n in range(n_min, n_max + 1):
        pw = power_at(n, delta, s, a)
        if pw >= power:
            return dict(out, status="POWERED", n=n, power_at_n=pw, reason="")
    return dict(out, status="UNDERPOWERED", n=n_max, power_at_n=power_at(n_max, delta, s, a),
                reason="MAXIMUM_SEEDS_DO_NOT_REACH_TARGET_POWER")


def seed_plan(dispersion: list[dict], operators: list[dict], regimes: list[dict], *,
              rules: dict = DENOISING_RULES, sd_floor: float = 0.25) -> dict:
    """Seeds per regime from the dispersion table; decided before any reserve exists."""
    cands = [o for o in operators if o["arm_role"] == "CANDIDATE"]
    fam = {}
    for o in cands:
        fam[o["spec"]["kind"]] = fam.get(o["spec"]["kind"], 0) + 1
    table = {(d["spec_sha256"], d["regime_key"], d["metric"]): d for d in dispersion}
    plan = {}
    for regime in regimes:
        rk = regime_key(regime)
        noise_free = str(regime["declared_snr_db"]) == "inf"
        per_spec = {}
        for o in cands:
            ss = spec_sha(o["spec"])
            if noise_free:
                d = table.get((ss, rk, "distortion_ratio"))
                mx = rules["noise_free"]["distortion_ratio_max"]
                mean = None if d is None else mx - d["mean"]
                sd = None if d is None else d["sd"]
                metric = "distortion_ratio_max - distortion_ratio"
            else:
                d = table.get((ss, rk, rules["improvement"]["metric"]))
                mean = None if d is None else d["mean"]
                sd = None if d is None else d["sd"]
                metric = rules["improvement"]["metric"]
            r = required_seeds(mean, sd, null=rules["improvement"]["margin_db"] if not noise_free else 0.0,
                               alpha=rules["alpha"], family_m=fam[o["spec"]["kind"]], sd_floor=sd_floor)
            per_spec[ss] = dict(r, metric=metric, historical_n=None if d is None else d["n"])
        powered = [r["n"] for r in per_spec.values() if r["status"] == "POWERED"]
        n = max(powered) if powered and len(powered) == len(per_spec) else SEEDS_MAX
        plan[rk] = {"regime": regime, "n_seeds": n,
                    "status": "POWERED" if powered else "UNDERPOWERED",
                    "per_spec": per_spec}
    return plan


# ------------------------------------------------------------- dispersion
def extract_c137_dispersion(root: Path, metrics=("snr_improvement_db", "distortion_ratio"),
                            partition: str = "confirmation") -> dict:
    """Seed-level effect statistics per operator x regime from a C137 lab root (read-only)."""
    import numpy as np
    L = _load("load_data_foundation")
    root = Path(root)
    runs = {}
    with open(root / "df_fact_operator_run.jsonl") as f:
        for line in f:
            r = json.loads(line)
            runs[L.row_sha256("df_fact_operator_run", r)] = r
    per = {}
    with open(root / "df_fact_operator_signal_metric.jsonl") as f:
        for line in f:
            if '"metric": "' not in line:
                continue
            m = json.loads(line)
            if m["metric"] not in metrics or m["partition"] != partition or m["status"] != "COMPLETED":
                continue
            r = runs.get(m["operator_run_sha256"])
            if r is None or r["status"] != "COMPLETED":
                continue
            kind = RETIRED_NAMES.get(r["operator_kind"], r["operator_kind"])
            spec = {"kind": kind, "params": r["operator_params"]}
            key = (spec_sha(spec), regime_key(r["regime"]), m["metric"])
            ent = per.setdefault(key, {"spec": spec, "historical_kind": r["operator_kind"], "regime": r["regime"],
                                       "units": {}})
            ent["units"].setdefault(r["subject_id"], []).append(float(m["value"]))
    rows = []
    for (ss, rk, metric), ent in sorted(per.items()):
        seed_values = [float(np.mean(v)) for _, v in sorted(ent["units"].items())]
        n = len(seed_values)
        rows.append({"spec_sha256": ss, "spec": ent["spec"], "historical_kind": ent["historical_kind"],
                     "regime_key": rk, "regime": ent["regime"], "metric": metric, "n": n,
                     "mean": float(np.mean(seed_values)),
                     "sd": float(np.std(seed_values, ddof=1)) if n >= 2 else None,
                     "seed_values": seed_values, "seeds": sorted(ent["units"])})
    sources = {p: file_sha(root / p) for p in ("df_fact_operator_run.jsonl", "df_fact_operator_signal_metric.jsonl")}
    return {"schema": DISPERSION_SCHEMA, "root_name": root.name, "files_sha256": sources, "partition": partition,
            "metrics": list(metrics), "aggregation": "mean over a unit's variables -> one value per seed; "
            "n, mean, sd(ddof=1) across seeds", "name_mapping": RETIRED_NAMES, "rows": rows,
            "use": "dispersion and candidate choice only; never a decision"}


# ------------------------------------------------------------------ design
def _default_fit_modes() -> dict:
    OPS = _load("df_operators")
    return {k: (OPS.EXPANDING_PREFIX if OPS.EXPANDING_PREFIX in modes else OPS.FROZEN_PREVIOUS_PARTITION)
            for k, modes in OPS.KIND_FIT_MODES.items()}


def build_design(inputs: dict) -> dict:
    """Build and seal a D2 v2 design from explicit inputs:
    design_id, cells (generator cells), operators [{spec, arm_role}], dispersion (rows) with
    dispersion_source, snr {estimators, bootstrap}, budget, roles; optional fit_modes, rules overrides."""
    OPS = _load("df_operators")
    BANK = _load("df_synthetic_bank")
    SYNC = _load("df_synthetic_contract")
    inputs = copy.deepcopy(inputs)
    modes = dict(_default_fit_modes(), **inputs.get("fit_modes", {}))
    ops = []
    for o in inputs["operators"]:
        spec = OPS.validate_spec(o["spec"])
        ops.append({"spec": spec, "spec_sha256": spec_sha(spec), "kind": spec["kind"], "arm_role": o["arm_role"],
                    "fit_mode": modes[spec["kind"]], "control_label": OPS.control_label(spec),
                    "competes": o["arm_role"] != "NON_CAUSAL_ORACLE_CONTROL"})
    cells = []
    for c in inputs["cells"]:
        c = {k: c[k] for k in ("family", "perturbation", "perturbation_params", "snr_db", "length", "n_variables",
                               "missingness", "noise_seed_tag")}
        BANK._validate_cell(c)
        cells.append(c)
    regimes = [regime_of_cell(c) for c in cells]
    if len({regime_key(r) for r in regimes}) != len(regimes):
        raise DesignRefusal("two cells map to the same regime")
    den_rules = copy.deepcopy(inputs.get("denoising_rules", DENOISING_RULES))
    plan = seed_plan(inputs.get("dispersion", []), ops, regimes, rules=den_rules,
                     sd_floor=inputs.get("sd_floor", 0.25))
    doc = {
        "schema": DESIGN_SCHEMA, "design_id": inputs["design_id"], "order_items": ["C171", "C173", "C174", "C175",
                                                                                   "C176", "C177"],
        "status": "SEALED_BEFORE_RESERVE",
        "bank": {"generator_version": BANK.GENERATOR_VERSION, "generator_code_sha256": BANK.code_sha256(),
                 "contract_code_sha256": file_sha(Path(SYNC.__file__)), "cells": cells,
                 "partitions": "chronological 60/20/20 from the length alone (df_synthetic_bank.partitions)"},
        "temporal_roles": {"TRAIN": "FIT_ONLY", "CALIBRATION": "DIAGNOSTIC_NEVER_GOVERNS",
                           "CONFIRMATION": "GOVERNS_FRESH_DECISIONS"},
        "operators": ops, "operators_code_sha256": OPS.code_sha256(), "current_kinds": list(OPS.KINDS),
        "branches": {"RAW": "X (observed)", "TRANSFORMED": "D(X) (operator output through the public API)",
                     "RESIDUAL": "X - D(X)"},
        "metrics": METRICS,
        "rules": {"abstention": "an ABSTAIN artifact or a refused fit/transform is a typed outcome (REFUSED) of that "
                                "arm on that seed; it counts toward the abstention rate, never as a pass",
                  "support": "a partition metric needs >= 32 available, non-missing samples, else UNAVAILABLE",
                  "missing_data": "no imputation: fit on the longest complete contiguous TRAIN stretch (>= 50 rows); "
                                  "outputs whose window holds a NaN are MISSING_INPUT; metrics only where available "
                                  "and not missing"},
        "unit_of_analysis": {"unit": "INDEPENDENT_GENERATOR_SEED", "time_rows_are_units": False,
                             "pairing": "the same seed and unit feed every arm"},
        "parameter_selection": {"allowed_in": "HISTORICAL_DEVELOPMENT_ONLY", "source": "lab_evaluation_c137_v1",
                                "used_for": ["dispersion", "candidate choice"]},
        "confirmation": {"primary_root": FRESH_MODE, "partition": "confirmation", "retuning": "FORBIDDEN",
                         "historical_stratum": HISTORICAL_MODE + " never grants consumption"},
        "multiplicity": den_rules["multiplicity"], "non_inferiority": den_rules["non_inferiority"],
        "denoising_rules": den_rules, "snr_rules": copy.deepcopy(inputs.get("snr_rules", SNR_RULES)),
        "snr": {"estimators": list(inputs["snr"]["estimators"]), "bootstrap": inputs["snr"].get("bootstrap"),
                "partitions": ["confirmation", "calibration"], "governing_partition": "confirmation"},
        "power": {"target": POWER_TARGET, "seeds_min": SEEDS_MIN, "seeds_max": SEEDS_MAX,
                  "dispersion_source": inputs.get("dispersion_source", {}),
                  "formula": required_seeds.__doc__, "module_doc": "tools/df_d2_design.py docstring"},
        "seeds_per_regime": plan,
        "budget": inputs["budget"], "roles": inputs["roles"],
        "compute": {"gpu": "NONE", "excluded": ["WORKER_B/GPU1"], "one_dataset_per_process": True},
        "lab_code_sha256s": lab_code_sha256s(),
        "design_sha256": "",
    }
    return seal_design(doc)


def seal_design(doc: dict) -> dict:
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    doc["design_sha256"] = sha_obj(body)
    return doc


REQUIRED_KEYS = {"schema", "design_id", "order_items", "status", "bank", "temporal_roles", "operators",
                 "operators_code_sha256", "current_kinds", "branches", "metrics", "rules", "unit_of_analysis",
                 "parameter_selection", "confirmation", "multiplicity", "non_inferiority", "denoising_rules",
                 "snr_rules", "snr", "power", "seeds_per_regime", "budget", "roles", "compute", "lab_code_sha256s",
                 "design_sha256"}


def validate_design(doc, *, require_current_code: bool = False) -> list[str]:
    OPS = _load("df_operators")
    if not isinstance(doc, dict) or set(doc) != REQUIRED_KEYS:
        return [f"design keys differ from {sorted(REQUIRED_KEYS)}"]
    p = []
    if doc["schema"] != DESIGN_SCHEMA:
        p.append("foreign schema")
    try:
        if sha_obj({k: v for k, v in doc.items() if k != "design_sha256"}) != doc["design_sha256"]:
            p.append("design digest does not re-derive")
    except (TypeError, ValueError):
        p.append("design is not canonical JSON")
    if set(doc["branches"]) != {"RAW", "TRANSFORMED", "RESIDUAL"}:
        p.append("branches must be RAW, TRANSFORMED and RESIDUAL")
    if doc["temporal_roles"].get("CONFIRMATION") != "GOVERNS_FRESH_DECISIONS" or \
            doc["temporal_roles"].get("CALIBRATION") != "DIAGNOSTIC_NEVER_GOVERNS":
        p.append("only CONFIRMATION governs; CALIBRATION is diagnostic")
    if doc["unit_of_analysis"].get("unit") != "INDEPENDENT_GENERATOR_SEED" or doc["unit_of_analysis"].get(
            "time_rows_are_units") is not False:
        p.append("the unit of analysis is an independent generator seed, never a time row")
    if doc["confirmation"].get("retuning") != "FORBIDDEN" or doc["confirmation"].get("partition") != "confirmation":
        p.append("confirmation is on the fresh reserve's confirmation partition without retuning")
    if doc["parameter_selection"].get("allowed_in") != "HISTORICAL_DEVELOPMENT_ONLY":
        p.append("parameters are selected only in historical development")
    for group in ("improvement", "distortion", "delay", "events", "extremes", "cost"):
        need = {"improvement": "snr_improvement_db", "distortion": "distortion_ratio", "delay": "delay_samples",
                "events": "impulse_retention", "extremes": "extreme_retention",
                "cost": "cpu_seconds_per_1000_samples"}[group]
        if need not in doc["metrics"]:
            p.append(f"metric group {group} lacks {need}")
    roles = [o.get("arm_role") for o in doc["operators"]]
    for o in doc["operators"]:
        spec = o.get("spec", {})
        try:
            OPS.validate_spec(spec)
        except OPS.OperatorRefusal as exc:
            p.append(f"operator spec refused: {exc}")
            continue
        if spec["kind"] not in OPS.KINDS:
            p.append(f"{spec['kind']} is not a current kind")
        if o["fit_mode"] not in OPS.KIND_FIT_MODES[spec["kind"]] or o["fit_mode"] == OPS.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL:
            p.append(f"fit mode {o['fit_mode']} is not a per-timestamp mode declared for {spec['kind']}")
        if o["arm_role"] not in ARM_ROLES:
            p.append(f"unknown arm role {o['arm_role']}")
        if (spec["kind"] in OPS.NON_CAUSAL_KINDS) != (o["arm_role"] == "NON_CAUSAL_ORACLE_CONTROL"):
            p.append(f"{spec['kind']}: the non-causal oracle is only a control and never competes")
        if o["arm_role"] == "NON_CAUSAL_ORACLE_CONTROL" and o.get("competes") is not False:
            p.append("the oracle never competes as an arm")
        if o.get("spec_sha256") != spec_sha(spec):
            p.append("spec digest does not re-derive")
    for need in ("CANDIDATE", "IDENTITY_RAW_CONTROL", "PREVIOUSLY_REJECTED_CONTROL", "NON_CAUSAL_ORACLE_CONTROL"):
        if need not in roles:
            p.append(f"design lacks a {need} arm")
    if require_current_code and doc["operators_code_sha256"] != OPS.code_sha256():
        p.append("design was built against different operator code")
    rks = set()
    for c in doc["bank"]["cells"]:
        rks.add(regime_key(regime_of_cell(c)))
    if set(doc["seeds_per_regime"]) != rks:
        p.append("seeds_per_regime must cover exactly the regimes of the bank cells")
    for rk, ent in doc["seeds_per_regime"].items():
        n = ent.get("n_seeds")
        if type(n) is not int or not SEEDS_MIN <= n <= SEEDS_MAX:
            p.append(f"{rk}: seeds must be an integer in [{SEEDS_MIN}, {SEEDS_MAX}]")
        if ent.get("status") not in ("POWERED", "UNDERPOWERED"):
            p.append(f"{rk}: status must be POWERED or UNDERPOWERED")
        if ent.get("status") == "UNDERPOWERED" and n != SEEDS_MAX:
            p.append(f"{rk}: an UNDERPOWERED regime uses the maximum seeds")
    b = doc["budget"]
    for k in ("task_memory_bytes", "wall_seconds", "cpu_seconds", "host_budget_bytes"):
        if type(b.get(k)) not in (int, float) or b[k] <= 0:
            p.append(f"budget.{k} must be positive")
    if doc["compute"].get("gpu") != "NONE":
        p.append("CPU only")
    for role in doc["roles"]:
        if role not in ("COORDINATOR", "WORKER_A", "WORKER_B"):
            p.append(f"unknown role {role}")
    if "/home/" in json.dumps(doc):
        p.append("no absolute home path in a design")
    return p


def require_valid(doc, **kw) -> dict:
    problems = validate_design(doc, **kw)
    if problems:
        raise DesignRefusal("; ".join(problems[:5]))
    return doc


def write_design(doc: dict, path: Path) -> str:
    require_valid(doc)
    IR = _load("df_isolated_runner")
    IR.atomic_write_once(Path(path), json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return doc["design_sha256"]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, help="JSON inputs for build_design")
    ap.add_argument("--c137-root", type=Path, help="extract dispersion from a C137 lab root (read-only)")
    ap.add_argument("--out", type=Path, required=True, help="write-once output")
    ap.add_argument("--validate", type=Path, help="validate a sealed design and print problems")
    a = ap.parse_args(argv)
    if a.validate:
        print(json.dumps(validate_design(json.loads(a.validate.read_text()), require_current_code=True), indent=1))
        return 0
    if a.c137_root and not a.inputs:
        doc = extract_c137_dispersion(a.c137_root)
        _load("df_isolated_runner").atomic_write_once(a.out, json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return 0
    inputs = json.loads(a.inputs.read_text())
    print(write_design(build_design(inputs), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
