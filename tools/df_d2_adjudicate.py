#!/usr/bin/env python3
"""C172, C175, C176 (order 2026-09-13): D2 decisions from unit-level rows of a FRESH root only.

Refusals before any statistic (``check_rows``):

* a row whose ``mode`` is not ``FRESH_CONFIRMATION`` (the historical
  reanalysis stratum never decides);
* a row from any partition but ``confirmation`` (calibration is diagnostic;
  cost rows of partition ``all`` are the only exception);
* a row carrying a time grain (``t``, ``timestamp``, ``row_index``...) or a
  duplicated unit-level grain (which is what an aggregation over time rows
  looks like);
* a fresh row without its tape binding, or rows of several designs.

``load_fresh_root`` additionally refuses a root with any
``ROOT_INVALIDATED__*`` marker (C177) and re-hashes every COMPLETED unit output
against its durable terminal.

SNR (C175), per estimator x regime (a regime is always family, perturbation,
declared SNR, length and missingness; there is no pooled path):
absolute error |snr_hat - snr_true| per variable, averaged over the seed's
variables, THEN across seeds, so opposite biases cannot cancel. With n
identifiable seeds, mean m and sd s, the 95% interval is m -+ t_{0.975,n-1} s/sqrt(n).
``SNR_CALIBRATED_FOR_REGIME`` needs upper bound <= 1 dB, coverage of the nominal
bootstrap interval >= 0.90 and a NOT_IDENTIFIABLE rate <= 0.10;
``SNR_NOT_IDENTIFIABLE`` when the estimator has no confirmation estimate by
contract, fewer identifiable seeds than the design or a NOT_IDENTIFIABLE rate
above 0.10; ``SNR_REJECTED`` when the lower bound > 1 dB; ``SNR_REGIME_LIMITED``
otherwise.

Denoising (C176), per operator x regime, with seed-level aggregates (mean over
variables for the improvement; worst variable for retention, delay, leakage and
cost) and one-sided bounds at alpha / m_family (Bonferroni over the CANDIDATE
specs of the same kind): the precedence is ``df_d2_design.DENOISING_RULES``.
``LAB_CALIBRATED`` only when improvement, non-inferiority, event/extreme floors
on EVERY seed, delay limit, residual leakage and cost pass simultaneously.
The oracle is never decided (``CONTROL_NOT_AN_ARM``).

Historical comparison (C172): ``reanalysis_decisions_c138`` applies the frozen
C137 rule to current HISTORICAL rows; ``compare_with_historical`` reports per
operator x regime the output and decision differences against C137 with a flip
cause (ABSTENTION, TRANSFORMED_RANGE, NAME, ARITHMETIC, TEMPORAL_MODE,
REAL_CORRECTION) and REFUSED / FAILED / INCONCLUSIVE / RESULT counts kept apart.

``PROPOSED_TABLES`` are the OLAP grains proposed for the lead to merge into
``load_data_foundation.TABLES``; this module does not edit the loader.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("df_d2_design")
L = _load("load_data_foundation")

DENOISING_DECISIONS = ("LAB_CALIBRATED", "REGIME_LIMITED", "NOT_IDENTIFIABLE", "LAB_REJECTED", "UNDERPOWERED")
CONTROL_STATES = ("CONTROL_NOT_AN_ARM",)
SNR_DECISIONS = ("SNR_CALIBRATED_FOR_REGIME", "SNR_REGIME_LIMITED", "SNR_NOT_IDENTIFIABLE", "SNR_REJECTED")
NON_DECISIONS = ("NOT_IDENTIFIABLE", "UNDERPOWERED", "SNR_NOT_IDENTIFIABLE", "CONTROL_NOT_AN_ARM")
FLIP_CAUSES = ("ABSTENTION", "TRANSFORMED_RANGE", "NAME", "ARITHMETIC", "TEMPORAL_MODE", "REAL_CORRECTION")
TIME_KEYS = {"t", "timestamp", "row_index", "time_index", "sample_index"}
ROW_STATUSES = ("COMPLETED", "INCONCLUSIVE", "UNAVAILABLE", "NOT_APPLICABLE", "REFUSED", "FAILED")
BRANCHES = ("RAW", "TRANSFORMED", "RESIDUAL", "COMPARISON", "COST")


class AdjudicationRefusal(ValueError):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


# --------------------------------------------------------------- tables
PROPOSED_TABLES = {
    "df_fact_d2_unit_denoising": {
        "run_id": L.TEXT, "mode": L.enum(D.MODES), "design_sha256": L.TEXT, "tape_sha256": L.TEXT_OR_NULL,
        "unit_id": L.TEXT, "seed": L.INT, "content_sha256": L.TEXT, "regime": L.JSON, "variable_id": L.TEXT,
        "variable_index": L.INT, "arm_role": L.enum(D.ARM_ROLES), "operator_kind": L.TEXT, "operator_params": L.JSON,
        "spec_sha256": L.TEXT, "fit_mode": L.TEXT, "fitted_sha256": L.TEXT_OR_NULL, "partition": L.TEXT,
        "branch": L.enum(BRANCHES), "metric": L.TEXT, "estimator": L.TEXT, "value": L.NUM_OR_NULL,
        "status": L.enum(ROW_STATUSES), "reason": L.TEXT, "code_sha256": L.TEXT, "operator_code_sha256": L.TEXT},
    "df_fact_d2_unit_snr": {
        "run_id": L.TEXT, "mode": L.enum(D.MODES), "design_sha256": L.TEXT, "tape_sha256": L.TEXT_OR_NULL,
        "unit_id": L.TEXT, "seed": L.INT, "content_sha256": L.TEXT, "regime": L.JSON, "variable_index": L.INT,
        "estimator": L.TEXT, "contract_state": L.TEXT, "partition": L.TEXT, "segment_start": L.INT_OR_NULL,
        "segment_end": L.INT_OR_NULL, "snr_db_hat": L.NUM_OR_NULL, "ci_low_db": L.NUM_OR_NULL,
        "ci_high_db": L.NUM_OR_NULL, "ci_lower_unbounded": L.BOOL, "ci_upper_unbounded": L.BOOL,
        "true_snr_db": L.NUM_OR_NULL, "error_db": L.NUM_OR_NULL, "abs_error_db": L.NUM_OR_NULL,
        "ci_covers_true": L.NUM_OR_NULL, "identifiability": L.enum(("ESTIMATED", "NOT_IDENTIFIABLE", "NOT_APPLICABLE")),
        "status": L.enum(ROW_STATUSES), "reason": L.TEXT, "code_sha256": L.TEXT},
    "df_fact_d2_decision": {
        "run_id": L.TEXT, "design_sha256": L.TEXT, "stratum": L.enum((D.FRESH_MODE,)),
        "subject_kind": L.enum(("OPERATOR", "SNR_ESTIMATOR")), "subject": L.TEXT, "operator_params": L.JSON,
        "spec_sha256": L.TEXT_OR_NULL, "arm_role": L.TEXT_OR_NULL, "regime": L.JSON,
        "decision": L.enum(DENOISING_DECISIONS + CONTROL_STATES + SNR_DECISIONS), "is_decision": L.BOOL,
        "reasons": L.JSON, "evidence": L.JSON, "n_seeds_design": L.INT, "n_seeds_valid": L.INT,
        "rule_sha256": L.TEXT, "externally_reviewed": L.BOOL, "code_sha256": L.TEXT},
    "df_fact_d2_historical_reanalysis": {
        "run_id": L.TEXT, "stratum": L.enum((D.HISTORICAL_MODE,)), "historical_run_id": L.TEXT,
        "historical_operator_kind": L.TEXT, "operator_kind": L.TEXT, "operator_params": L.JSON, "spec_sha256": L.TEXT,
        "regime": L.JSON, "historical_decision": L.TEXT_OR_NULL, "reanalysis_decision": L.TEXT_OR_NULL,
        "flipped": L.BOOL, "flip_cause": L.TEXT_OR_NULL, "flip_causes": L.JSON, "output_differences": L.JSON,
        "historical_counts": L.JSON, "reanalysis_counts": L.JSON, "code_sha256": L.TEXT},
}


def validate_proposed_row(table: str, row: dict) -> list[str]:
    """The loader's type conventions applied to a proposed table (no loader edit)."""
    spec = PROPOSED_TABLES.get(table)
    if spec is None:
        return [f"unknown proposed table {table!r}"]
    if not isinstance(row, dict) or set(row) != set(spec):
        return [f"keys differ: expected {sorted(spec)}, got {sorted(row) if isinstance(row, dict) else row}"]
    p = []
    for c, t in spec.items():
        v = row[c]
        if isinstance(t, tuple):
            if v not in t[1]:
                p.append(f"{c}: {v!r} not in {list(t[1])}")
        elif t == L.TEXT and not isinstance(v, str):
            p.append(f"{c}: expected text")
        elif t == L.TEXT_OR_NULL and not (v is None or isinstance(v, str)):
            p.append(f"{c}: expected text or null")
        elif t == L.INT and type(v) is not int:
            p.append(f"{c}: expected an integer")
        elif t == L.INT_OR_NULL and not (v is None or type(v) is int):
            p.append(f"{c}: expected an integer or null")
        elif t == L.BOOL and type(v) is not bool:
            p.append(f"{c}: expected a boolean")
        elif t == L.NUM_OR_NULL and not (v is None or (type(v) in (int, float) and math.isfinite(v))):
            p.append(f"{c}: expected a finite number or null")
        elif t == L.JSON and not isinstance(v, (dict, list)):
            p.append(f"{c}: expected an object or list")
    if "status" in spec:
        if row.get("status") != "COMPLETED" and not str(row.get("reason", "")).strip():
            p.append("a row that did not complete needs a reason")
        if table == "df_fact_d2_unit_denoising":
            if row["status"] == "COMPLETED" and row["value"] is None:
                p.append("a COMPLETED row needs a value")
            if row["status"] != "COMPLETED" and row["value"] is not None:
                p.append("only a COMPLETED row carries a value")
        if table == "df_fact_d2_unit_snr" and row["status"] == "COMPLETED" and row["snr_db_hat"] is None:
            p.append("a COMPLETED SNR row needs an estimate")
    if table == "df_fact_d2_decision" and row.get("externally_reviewed") is not False:
        p.append("decisions are loaded unreviewed")
    if any(s in L.FORBIDDEN for s in L._walk_strings(row)):
        p.append("never PUBLICLY_ELIGIBLE or LIVE_ELIGIBLE")
    for c in ("content_sha256", "code_sha256", "spec_sha256", "design_sha256", "tape_sha256", "rule_sha256",
              "operator_code_sha256"):
        if c in spec and isinstance(row.get(c), str) and not (len(row[c]) == 64 and all(ch in "0123456789abcdef"
                                                                                         for ch in row[c])):
            p.append(f"{c}: expected a sha256 hex digest")
    return p


# --------------------------------------------------------------- guards
def check_rows(rows: list, kind: str) -> dict:
    """Refuse historical, calibration, time-grained or duplicated rows before any statistic."""
    if kind not in ("denoising", "snr"):
        raise AdjudicationRefusal(f"unknown row kind {kind}")
    seen, designs, tapes, seeds = set(), set(), set(), {}
    for r in rows:
        bad = TIME_KEYS & set(r)
        if bad:
            raise AdjudicationRefusal(f"a row carries a time grain {sorted(bad)}: decisions never aggregate time rows")
        if r.get("mode") != D.FRESH_MODE:
            raise AdjudicationRefusal(f"row mode {r.get('mode')!r}: only FRESH_CONFIRMATION rows decide; the "
                                      "historical reanalysis never grants a decision")
        is_cost = kind == "denoising" and r.get("branch") == "COST" and r.get("partition") == "all"
        if r.get("partition") != "confirmation" and not is_cost:
            raise AdjudicationRefusal(f"row from partition {r.get('partition')!r}: calibration is diagnostic, only "
                                      "confirmation governs")
        if not r.get("tape_sha256"):
            raise AdjudicationRefusal("a fresh row must bind its seed tape")
        grain = ((r["unit_id"], r["variable_index"], r["spec_sha256"], r["partition"], r["branch"], r["metric"])
                 if kind == "denoising" else (r["unit_id"], r["variable_index"], r["estimator"], r["partition"]))
        if grain in seen:
            raise AdjudicationRefusal(f"duplicated unit-level grain {grain}: rows below the seed/variable grain "
                                      "(e.g. time rows) are refused")
        seen.add(grain)
        designs.add(r["design_sha256"])
        tapes.add(r["tape_sha256"])
        if seeds.setdefault(r["unit_id"], r["seed"]) != r["seed"]:
            raise AdjudicationRefusal(f"unit {r['unit_id']} carries two seeds")
    if len(designs) > 1 or len(tapes) > 1:
        raise AdjudicationRefusal("rows of several designs or tapes")
    return {"rows": len(rows), "units": len(seeds), "design_sha256": next(iter(designs), None)}


def _t(p: float, df: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(p, df))


def _mean_bounds(values: list, one_sided_alpha: float) -> dict:
    n = len(values)
    a = np.asarray(values, dtype=float)
    m = float(a.mean()) if n else None
    if n < 2:
        return {"n": n, "mean": m, "sd": None, "lower": None, "upper": None}
    sd = float(a.std(ddof=1))
    h = _t(1.0 - one_sided_alpha, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lower": m - h, "upper": m + h}


def _regime_entry(design: dict, regime: dict) -> tuple:
    rk = D.regime_key(regime)
    ent = design["seeds_per_regime"].get(rk)
    if ent is None:
        raise AdjudicationRefusal(f"regime {rk} is not a regime of the sealed design")
    return rk, ent


# ------------------------------------------------------------------ SNR
def decide_snr(rows: list, design: dict) -> list:
    D.require_valid(design)
    check_rows(rows, "snr")
    if any(r["design_sha256"] != design["design_sha256"] for r in rows):
        raise AdjudicationRefusal("rows belong to another design")
    R = design["snr_rules"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["estimator"], D.regime_key(r["regime"])), []).append(r)
    out = []
    for (est, rk), rs in sorted(groups.items()):
        _, ent = _regime_entry(design, rs[0]["regime"])
        n_design = ent["n_seeds"]
        by_seed: dict = {}
        for r in rs:
            by_seed.setdefault(r["unit_id"], []).append(r)
        applicable = [r for r in rs if r["identifiability"] != "NOT_APPLICABLE"]
        reasons, decision = [], None
        ni = sum(r["identifiability"] == "NOT_IDENTIFIABLE" or r["status"] == "FAILED" for r in rs
                 if r["status"] != "NOT_APPLICABLE")
        ni_rate = ni / len(applicable) if applicable else None
        per_seed = {}
        incomplete = []
        for u, vs in by_seed.items():
            # a seed's error averages ALL its required variables; a variable that is not
            # ESTIMATED makes the seed not identifiable instead of leaving the average (D2-R1 rule 8)
            required = [v for v in vs if v["status"] != "NOT_APPLICABLE" and v["identifiability"] != "NOT_APPLICABLE"]
            errs = [abs(v["error_db"]) for v in required if v["identifiability"] == "ESTIMATED" and v["error_db"] is not None]
            if required and len(errs) == len(required):
                per_seed[u] = float(np.mean(errs))
            else:
                incomplete.append(u)
        cov = [v["ci_covers_true"] for v in rs if v["ci_covers_true"] is not None]
        coverage = float(np.mean(cov)) if cov else None
        signed = [v["error_db"] for v in rs if v["error_db"] is not None]
        b = _mean_bounds(list(per_seed.values()), (1.0 - R["ci_level"]) / 2.0)
        ev = {"n_seeds_design": n_design, "n_seeds_observed": len(by_seed), "n_seeds_identifiable": len(per_seed),
              "not_identifiable_rate": ni_rate, "per_seed_mean_abs_error_db": per_seed,
              "mean_abs_error_db": b["mean"], "sd_seed_abs_error_db": b["sd"], "ci95_abs_error_db": [b["lower"], b["upper"]],
              "mean_signed_error_db_for_information_only": float(np.mean(signed)) if signed else None,
              "coverage": coverage, "n_coverage": len(cov), "thresholds": {k: R[k] for k in (
                  "mean_abs_error_ci_upper_max_db", "coverage_min", "not_identifiable_rate_max", "ci_level")},
              "support": {"seeds_planned": n_design, "seeds_observed": len(by_seed),
                          "seeds_identifiable": len(per_seed), "seeds_incomplete": sorted(incomplete)}}
        if not applicable:
            decision, reasons = "SNR_NOT_IDENTIFIABLE", ["NO_CONFIRMATION_ESTIMATE_BY_CONTRACT"]
        elif all(r["true_snr_db"] is None for r in applicable):
            decision, reasons = "SNR_NOT_IDENTIFIABLE", ["TRUE_SNR_NOT_FINITE_IN_REGIME"]
        elif ni_rate > R["not_identifiable_rate_max"]:
            decision, reasons = "SNR_NOT_IDENTIFIABLE", [f"NOT_IDENTIFIABLE_RATE {ni_rate:.3f} > {R['not_identifiable_rate_max']}"]
        elif len(per_seed) < n_design or b["upper"] is None:
            decision, reasons = "SNR_NOT_IDENTIFIABLE", [f"IDENTIFIABLE_SEEDS {len(per_seed)} < DESIGN {n_design}"]
        elif b["upper"] <= R["mean_abs_error_ci_upper_max_db"] and coverage is not None and coverage >= R["coverage_min"]:
            decision, reasons = "SNR_CALIBRATED_FOR_REGIME", ["UPPER_BOUND_AND_COVERAGE_AND_IDENTIFIABILITY_PASS"]
        elif b["lower"] > R["mean_abs_error_ci_upper_max_db"]:
            decision, reasons = "SNR_REJECTED", [f"LOWER_BOUND_MEAN_ABS_ERROR {b['lower']:.3f} dB > "
                                                 f"{R['mean_abs_error_ci_upper_max_db']}"]
        else:
            decision = "SNR_REGIME_LIMITED"
            if b["upper"] > R["mean_abs_error_ci_upper_max_db"]:
                reasons.append(f"UPPER_BOUND_MEAN_ABS_ERROR {b['upper']:.3f} dB > {R['mean_abs_error_ci_upper_max_db']}")
            if coverage is None or coverage < R["coverage_min"]:
                reasons.append(f"COVERAGE {coverage} < {R['coverage_min']}")
        out.append({"subject_kind": "SNR_ESTIMATOR", "subject": est, "operator_params": {}, "spec_sha256": None,
                    "arm_role": None, "regime": rs[0]["regime"], "decision": decision, "reasons": reasons,
                    "evidence": ev, "n_seeds_design": n_design, "n_seeds_valid": len(per_seed),
                    "rule_sha256": D.sha_obj(R), "design_sha256": design["design_sha256"],
                    "externally_reviewed": False})
    return out


# ------------------------------------------------------------ denoising
def _seed_table(rs: list) -> dict:
    """unit -> {"seed", "abstained", "vars": {v: {metric: value}}, "states": {v: {metric: status}},
    "events": {v: {event_metric: count}}, "variables": set, "cost": {metric: max}}.
    Every row leaves a trace: a metric that is not COMPLETED is still known by its status
    (D2-R1), so absence and inconclusiveness can never be read as a pass."""
    out: dict = {}
    for r in rs:
        e = out.setdefault(r["unit_id"], {"seed": r["seed"], "abstained": False, "vars": {}, "states": {},
                                          "events": {}, "variables": set(), "cost": {}, "unavailable": 0})
        if r["metric"] == "arm_status":
            e["abstained"] = True
            e["abstain_reason"] = r["reason"]
            continue
        if r["branch"] == "COST":
            if r["value"] is not None:
                e["cost"][r["metric"]] = max(e["cost"].get(r["metric"], -math.inf), r["value"])
            continue
        v = r["variable_index"]
        e["variables"].add(v)
        if r["status"] == "UNAVAILABLE":
            e["unavailable"] += 1
            e["states"].setdefault(v, {})["partition_support"] = "UNAVAILABLE"
            continue
        if r["metric"].endswith("__events"):
            if r["status"] == "COMPLETED" and r["value"]:
                e["events"].setdefault(v, {})[r["metric"][:-len("__events")]] = float(r["value"])
            continue
        e["states"].setdefault(v, {})[r["metric"]] = r["status"]
        if r["status"] == "COMPLETED" and r["value"] is not None:
            e["vars"].setdefault(v, {})[r["metric"]] = r["value"]
    return out


PRIMARY_METRICS = ("distortion_ratio", "delay_samples", "residual_signal_share")


def _applicable(seed: dict, v: int, metric: str, noise_free: bool, R: dict) -> bool:
    """Applicability derived from the contract and the evaluator's own event/geometry
    rows, never from whether a metric row happened to arrive (D2-R1 rules 3 and 4)."""
    if metric == R["improvement"]["metric"]:
        return not noise_free
    if metric in PRIMARY_METRICS:
        return True
    states = seed["states"].get(v, {})
    events = seed["events"].get(v, {})
    if metric in ("extreme_retention", "extreme_retention_raw"):
        # extremes exist when the raw counterpart could be measured; an undefined raw
        # (flat clean signal) means no extreme geometry, an absent raw means unknown -> required
        raw = states.get("extreme_retention_raw")
        return raw != "INCONCLUSIVE"
    base = metric[:-len("_raw")] if metric.endswith("_raw") else metric
    if base in R["event_floors"]:
        return events.get(base, 0) > 0
    return True


def _support(seeds: dict, R: dict, noise_free: bool, n_design: int) -> dict:
    """Per seed: which metrics apply, which applicable ones are unsupported, and whether the
    seed is complete (every applicable metric of every variable observed, cost observed).
    Seeds are never removed from the denominator: planned, observed, complete and
    inapplicable counts are all published (D2-R1 rules 5 and 6)."""
    required = [R["improvement"]["metric"], *PRIMARY_METRICS, "extreme_retention", "extreme_retention_raw",
                *R["event_floors"], *sum(([m, raw] for m, raw in R["non_inferiority"]["pairs"].items()), [])]
    required = list(dict.fromkeys(required))
    unsupported: dict = {}
    inapplicable: dict = {m: 0 for m in required}
    per_seed = {}
    for u, s in seeds.items():
        missing = {}
        applicable_any = set()
        if s["abstained"]:
            per_seed[u] = {"complete": False, "abstained": True, "unsupported": {}}
            continue
        for v in sorted(s["variables"]):
            if s["states"].get(v, {}).get("partition_support") == "UNAVAILABLE":
                missing.setdefault("partition_support", []).append(v)
                continue
            for m in required:
                if not _applicable(s, v, m, noise_free, R):
                    continue
                applicable_any.add(m)
                if s["vars"].get(v, {}).get(m) is None:
                    missing.setdefault(m, []).append(v)
        if not s["variables"]:
            missing["primary_contrast"] = []
        if s["cost"].get(R["cost"]["metric"]) is None:
            missing.setdefault(R["cost"]["metric"], [])
        for m in required:
            if m not in applicable_any:
                inapplicable[m] += 1
        per_seed[u] = {"complete": not missing, "abstained": False, "unsupported": missing}
        for m in missing:
            unsupported.setdefault(m, []).append(u)
    complete = sorted(u for u, p in per_seed.items() if p["complete"])
    return {"seeds_planned": n_design, "seeds_observed": len(seeds), "seeds_complete": len(complete),
            "seeds_abstained": sum(p["abstained"] for p in per_seed.values()),
            "unsupported_by_metric": {m: sorted(us) for m, us in sorted(unsupported.items())},
            "inapplicable_by_metric": {m: n for m, n in inapplicable.items() if n},
            "per_seed": per_seed, "complete_units": complete}


def _agg(seed: dict, metric: str, how: str):
    vals = [m[metric] for m in seed["vars"].values() if m.get(metric) is not None]
    if not vals:
        return None
    return float({"mean": np.mean, "min": np.min, "max": np.max}[how](vals))


def decide_denoising(rows: list, design: dict) -> list:
    D.require_valid(design)
    check_rows(rows, "denoising")
    if any(r["design_sha256"] != design["design_sha256"] for r in rows):
        raise AdjudicationRefusal("rows belong to another design")
    R = design["denoising_rules"]
    ops = {o["spec_sha256"]: o for o in design["operators"]}
    fam = {}
    for o in design["operators"]:
        if o["arm_role"] == "CANDIDATE":
            fam[o["kind"]] = fam.get(o["kind"], 0) + 1
    by_regime: dict = {}
    for r in rows:
        if r["spec_sha256"] not in ops:
            raise AdjudicationRefusal(f"spec {r['spec_sha256'][:12]} is not an arm of the sealed design")
        by_regime.setdefault(D.regime_key(r["regime"]), {}).setdefault(r["spec_sha256"], []).append(r)
    out = []
    for rk, arms in sorted(by_regime.items()):
        regime = next(iter(arms.values()))[0]["regime"]
        _, ent = _regime_entry(design, regime)
        n_design = ent["n_seeds"]
        seed_sets = {ss: frozenset(r["unit_id"] for r in rs) for ss, rs in arms.items()}
        paired = len(set(seed_sets.values())) == 1 and set(arms) == set(ops)
        noise_free = str(regime["declared_snr_db"]) == "inf"
        for ss, rs in sorted(arms.items()):
            op = ops[ss]
            out.append(_decide_arm(op, regime, rs, R, ent, n_design, paired, noise_free, fam, design))
    return out


def _decide_arm(op, regime, rs, R, ent, n_design, paired, noise_free, fam, design) -> dict:
    seeds = _seed_table(rs)
    m_family = fam.get(op["kind"], 1) if op["arm_role"] == "CANDIDATE" else 1
    alpha = R["alpha"] / m_family
    ev = {"n_seeds_design": n_design, "n_seeds_observed": len(seeds), "alpha_one_sided_adjusted": alpha,
          "family_m": m_family, "paired_on_same_seeds": paired, "noise_free_regime": noise_free, "checks": {},
          "per_seed": {}}
    base = {"subject_kind": "OPERATOR", "subject": op["kind"], "operator_params": op["spec"]["params"],
            "spec_sha256": op["spec_sha256"], "arm_role": op["arm_role"], "regime": regime,
            "rule_sha256": D.sha_obj(R), "design_sha256": design["design_sha256"], "externally_reviewed": False,
            "n_seeds_design": n_design}

    def result(decision, reasons, n_valid):
        return dict(base, decision=decision, reasons=reasons, evidence=ev, n_seeds_valid=n_valid)

    support = _support(seeds, R, noise_free, n_design)
    ev["support"] = {k: v for k, v in support.items() if k not in ("per_seed", "complete_units")}
    # a valid seed is a COMPLETE seed: every applicable metric observed (D2-R2)
    valid = {u: seeds[u] for u in support["complete_units"]}
    abst = sum(s["abstained"] for s in seeds.values())
    ev["abstention_rate"] = abst / len(seeds) if seeds else None
    for u, s in seeds.items():
        ev["per_seed"][u] = {"seed": s["seed"], "abstained": s["abstained"],
                             "snr_improvement_db": _agg(s, "snr_improvement_db", "mean"),
                             "distortion_ratio": _agg(s, "distortion_ratio", "max"),
                             "residual_signal_share": _agg(s, "residual_signal_share", "max"),
                             "delay_samples": _agg(s, "delay_samples", "max"),
                             "cost": s["cost"].get(R["cost"]["metric"])}
    if op["arm_role"] == "NON_CAUSAL_ORACLE_CONTROL":
        return result("CONTROL_NOT_AN_ARM", ["NON_CAUSAL_ORACLE: detected as a negative control, never an arm"],
                      len(valid))
    if not paired:
        return result("NOT_IDENTIFIABLE", ["UNPAIRED: arms of the regime do not share the same seeds"], len(valid))
    if ev["abstention_rate"] > R["abstention"]["max_rate"]:
        return result("NOT_IDENTIFIABLE", [f"ABSTENTION_RATE {ev['abstention_rate']:.3f} > {R['abstention']['max_rate']}"],
                      len(valid))
    per_spec = ent["per_spec"].get(op["spec_sha256"])
    underpowered = per_spec is not None and per_spec["status"] == "UNDERPOWERED"
    reasons_rej = []
    # Event and extreme floors, and leakage, on EVERY observed seed. Damage measured on
    # one seed is proven by that seed alone: another seed's incompleteness never hides it
    # (D2-R1 rule 1, post-result clarification of the sealed precedence). A seed whose
    # applicable floor could not be measured makes the arm NOT_IDENTIFIABLE below.
    observed = {u: s for u, s in seeds.items() if not s["abstained"] and s["variables"]}
    unmeasured_floors = {}
    for metric, (how, bound) in R["event_floors"].items():
        applicable = {u: s for u, s in observed.items()
                      if any(_applicable(s, v, metric, noise_free, R) for v in s["variables"])}
        worst = {u: _agg(s, metric, "min" if how == "min" else "max") for u, s in applicable.items()}
        unmeasured = sorted(u for u, v in worst.items() if v is None)
        bad = {u: v for u, v in worst.items() if v is not None and ((how == "min" and v < bound) or
                                                                 (how == "max" and v > bound))}
        ev["checks"][f"floor:{metric}"] = {"passed": not bad and not unmeasured, "bound": bound,
                                          "failing_seeds": bad, "unmeasured_seeds": unmeasured,
                                          "applicable": bool(applicable), "applicable_seeds": len(applicable)}
        if bad:
            reasons_rej.append(f"EVENT_OR_EXTREME_DESTROYED {metric} on seeds {sorted(bad)}")
        if unmeasured:
            unmeasured_floors[metric] = unmeasured
    leak = {u: _agg(s, R["residual_leakage"]["metric"], "max") for u, s in observed.items()}
    bad = {u: v for u, v in leak.items() if v is not None and v > R["residual_leakage"]["max"]}
    ev["checks"]["residual_leakage"] = {"passed": not bad and all(v is not None for v in leak.values()),
                                        "failing_seeds": bad,
                                        "unmeasured_seeds": sorted(u for u, v in leak.items() if v is None)}
    if bad:
        reasons_rej.append(f"RESIDUAL_LEAKAGE on seeds {sorted(bad)}")
    if underpowered:
        return result("UNDERPOWERED", [f"DESIGN_UNDERPOWERED: {per_spec.get('reason')}"] +
                      [f"(not governing) {x}" for x in reasons_rej], len(valid))
    if reasons_rej:
        return result("LAB_REJECTED", reasons_rej, len(valid))
    if len(valid) < n_design:
        gaps = {m: len(us) for m, us in support["unsupported_by_metric"].items()}
        return result("NOT_IDENTIFIABLE", [f"COMPLETE_SEEDS {len(valid)} < DESIGN {n_design}"
                                           + (f"; unsupported seeds by metric {gaps}" if gaps else "")], len(valid))
    if unmeasured_floors:
        metric, units = next(iter(unmeasured_floors.items()))
        return result("NOT_IDENTIFIABLE", [f"FLOOR_UNSUPPORTED {metric} on seeds {units}"], len(valid))
    if noise_free:
        dist = {u: _agg(s, "distortion_ratio", "max") for u, s in valid.items()}
        vals = [v for v in dist.values() if v is not None]
        b = _mean_bounds(vals, alpha)
        mx = R["noise_free"]["distortion_ratio_max"]
        ok = bool(vals) and b["upper"] is not None and b["upper"] <= mx and max(vals) <= mx
        ev["checks"]["noise_free_distortion"] = dict(b, passed=ok, max=mx)
        if not ok:
            return result("LAB_REJECTED", [f"FALSE_POSITIVE_DISTORTION upper {b['upper']} or a seed above {mx}"],
                          len(valid))
    else:
        imp = {u: _agg(s, R["improvement"]["metric"], "mean") for u, s in valid.items()}
        vals = [v for v in imp.values() if v is not None]
        b = _mean_bounds(vals, alpha)
        I = R["improvement"]
        ev["checks"]["improvement"] = dict(b, margin_db=I["margin_db"], min_mean_effect_db=I["min_mean_effect_db"])
        if b["upper"] is None:
            return result("NOT_IDENTIFIABLE", ["IMPROVEMENT_NOT_ESTIMABLE"], len(valid))
        if b["upper"] < I["min_mean_effect_db"]:
            ev["checks"]["improvement"]["passed"] = False
            return result("LAB_REJECTED", [f"NO_IMPROVEMENT: upper bound {b['upper']:.3f} dB < minimum effect "
                                           f"{I['min_mean_effect_db']}"], len(valid))
        if not (b["lower"] > I["margin_db"] and b["mean"] >= I["min_mean_effect_db"]):
            ev["checks"]["improvement"]["passed"] = False
            return result("NOT_IDENTIFIABLE", [f"IMPROVEMENT_INCONCLUSIVE: lower {b['lower']:.3f}, mean {b['mean']:.3f}"],
                          len(valid))
        ev["checks"]["improvement"]["passed"] = True
    NI = R["non_inferiority"]
    for metric, raw in NI["pairs"].items():
        deltas = []
        applicable_any = False
        for s in valid.values():
            ds = []
            for v, m in s["vars"].items():
                if not _applicable(s, v, metric, noise_free, R):
                    continue
                applicable_any = True
                if m.get(metric) is not None and m.get(raw) is not None:
                    ds.append(m[metric] - m[raw])
            if ds:
                deltas.append(float(np.min(ds)))
        if not applicable_any:
            # the event does not exist in this regime's windows: inapplicable, never a pass
            ev["checks"][f"non_inferiority:{metric}"] = {"passed": None, "applicable": False, "n": 0}
            continue
        if len(deltas) < 2:
            ev["checks"][f"non_inferiority:{metric}"] = {"passed": False, "applicable": True, "n": len(deltas)}
            return result("NOT_IDENTIFIABLE", [f"NON_INFERIORITY_UNSUPPORTED {metric}: {len(deltas)} complete pair(s) < 2"],
                          len(valid))
        b = _mean_bounds(deltas, alpha)
        passed = b["lower"] > -NI["margin"]
        ev["checks"][f"non_inferiority:{metric}"] = dict(b, passed=passed, margin=NI["margin"], applicable=True)
        if b["upper"] < -NI["margin"]:
            return result("LAB_REJECTED", [f"INFERIOR_TO_RAW {metric}: upper {b['upper']:.3f} < -{NI['margin']}"],
                          len(valid))
        if not passed:
            return result("NOT_IDENTIFIABLE", [f"NON_INFERIORITY_INCONCLUSIVE {metric}: lower {b['lower']:.3f}"],
                          len(valid))
    delay = {u: _agg(s, R["delay"]["metric"], "max") for u, s in valid.items()}
    bad = {u: v for u, v in delay.items() if v is not None and v > R["delay"]["max_samples"]}
    ev["checks"]["delay"] = {"passed": not bad, "failing_seeds": bad, "max_samples": R["delay"]["max_samples"]}
    cost = {u: s["cost"].get(R["cost"]["metric"]) for u, s in valid.items()}
    badc = {u: v for u, v in cost.items() if v is not None and v > R["cost"]["max"]}
    ev["checks"]["cost"] = {"passed": not badc and all(v is not None for v in cost.values()), "failing_seeds": badc}
    if bad or badc or not ev["checks"]["cost"]["passed"]:
        return result("LAB_REJECTED", ([f"DELAY_LIMIT on seeds {sorted(bad)}"] if bad else []) +
                      ([f"COST_LIMIT on seeds {sorted(badc)}"] if badc else []) +
                      ([] if all(v is not None for v in cost.values()) else ["COST_NOT_MEASURED"]), len(valid))
    if not noise_free:
        floor = R["improvement"]["seed_floor_db"]
        low = {u: v for u, v in imp.items() if v is not None and v < floor}
        ev["checks"]["seed_floor"] = {"passed": not low, "failing_seeds": low, "floor_db": floor}
        if low:
            return result("REGIME_LIMITED", [f"SEED_DEPENDENT: seeds {sorted(low)} below {floor} dB although the mean "
                                             "improves"], len(valid))
    return result("LAB_CALIBRATED", ["IMPROVEMENT_NONINFERIORITY_EVENTS_EXTREMES_DELAY_LEAKAGE_COST_ALL_PASS"],
                  len(valid))


def decision_rows(decisions: list, run_id: str) -> list:
    code = D.lab_code_sha256()
    rows = []
    for d in decisions:
        row = {"run_id": run_id, "design_sha256": d["design_sha256"], "stratum": D.FRESH_MODE,
               "subject_kind": d["subject_kind"], "subject": d["subject"], "operator_params": d["operator_params"],
               "spec_sha256": d["spec_sha256"], "arm_role": d["arm_role"], "regime": d["regime"],
               "decision": d["decision"], "is_decision": d["decision"] not in NON_DECISIONS,
               "reasons": d["reasons"], "evidence": json.loads(json.dumps(d["evidence"], default=float)),
               "n_seeds_design": d["n_seeds_design"], "n_seeds_valid": d["n_seeds_valid"],
               "rule_sha256": d["rule_sha256"], "externally_reviewed": False, "code_sha256": code}
        problems = validate_proposed_row("df_fact_d2_decision", row)
        if problems:
            raise AdjudicationRefusal(f"decision row does not validate: {problems[:3]}")
        rows.append(row)
    return rows


# ------------------------------------------------------------ fresh root
def load_fresh_root(root: Path, design: dict) -> dict:
    """Rows of a FRESH run root, re-hashed against their durable terminals; confirmation rows only."""
    IR = _load("df_isolated_runner")
    root = Path(root)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest.get("mode") != D.FRESH_MODE:
        raise AdjudicationRefusal(f"root mode {manifest.get('mode')!r} is not FRESH_CONFIRMATION")
    if manifest.get("design_sha256") != design["design_sha256"]:
        raise AdjudicationRefusal("root was run under another design")
    markers = sorted(p.name for p in root.glob("ROOT_INVALIDATED__*.json"))
    if markers:
        raise AdjudicationRefusal(f"root is invalidated as a whole by {markers}")
    latest: dict = {}
    for p in sorted((root / "terminals").glob("*.attempt-*.json")):
        name, attempt = p.name.rsplit(".attempt-", 1)
        n = int(attempt.split(".")[0])
        if name not in latest or n > latest[name][0]:
            latest[name] = (n, json.loads(p.read_text()))
    counts: dict = {}
    den, snr = [], []
    excluded = 0
    for name, (_, term) in sorted(latest.items()):
        problems = IR.validate_terminal(term)
        if problems:
            raise AdjudicationRefusal(f"terminal {name} does not validate: {problems[:2]}")
        counts[term["status"]] = counts.get(term["status"], 0) + 1
        if term["status"] != "COMPLETED":
            continue
        out = root / term["output_file"]
        if IR.sha_file(out) != term["output_sha256"]:
            raise AdjudicationRefusal(f"{name}: output does not re-hash to its terminal")
        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                r = obj["row"]
                keep = r["partition"] == "confirmation" or (obj["table"] == "df_fact_d2_unit_denoising"
                                                            and r["branch"] == "COST")
                if not keep:
                    excluded += 1
                    continue
                (den if obj["table"] == "df_fact_d2_unit_denoising" else snr).append(r)
    return {"denoising": den, "snr": snr, "terminal_counts": counts, "non_governing_rows_excluded": excluded,
            "run_id": manifest["run_id"]}


# ------------------------------------------------------ historical (C172)
def reanalysis_decisions_c138(rows: list) -> list:
    """The frozen C137 rule (c138.v1) applied to current HISTORICAL rows. Never a D2 decision."""
    LAB = _load("df_lab_evaluation")
    recs: dict = {}
    for r in rows:
        if r["mode"] != D.HISTORICAL_MODE:
            raise AdjudicationRefusal("the reanalysis comparison takes HISTORICAL rows only")
        if r["branch"] == "COST":
            continue
        key = (r["spec_sha256"], r["unit_id"], r["variable_index"])
        e = recs.setdefault(key, {"status": "COMPLETED", "partitions": {}, "spec": {"kind": r["operator_kind"],
                                                                                   "params": r["operator_params"]},
                                  "regime": r["regime"], "unit_id": r["unit_id"]})
        if r["metric"] == "arm_status":
            e["status"] = r["status"]
            continue
        p = e["partitions"].setdefault(r["partition"], {"status": "COMPLETED"})
        if r["metric"] == "partition_support":
            p["status"] = "UNAVAILABLE"
        elif r["status"] == "COMPLETED":
            p[r["metric"]] = 1.0 if r["metric"] == "noise_free" and r["value"] == 1.0 else r["value"]
            if r["metric"] == "noise_free":
                p["noise_free"] = bool(r["value"])
    grouped: dict = {}
    for (ss, _, _), e in recs.items():
        grouped.setdefault((ss, D.regime_key(e["regime"])), []).append(e)
    out = []
    for (ss, rk), es in sorted(grouped.items()):
        decision, reasons, ev = LAB.decide(es[0]["spec"], es[0]["regime"], es)
        out.append({"operator_kind": es[0]["spec"]["kind"], "operator_params": es[0]["spec"]["params"],
                    "spec_sha256": ss, "regime": es[0]["regime"], "decision": decision, "reasons": reasons,
                    "evidence": ev, "rule": "c138.v1", "stratum": D.HISTORICAL_MODE})
    return out


_COMPARED = ("support", "snr_improvement_db", "rmse_ratio")


def compare_with_historical(hist_decisions: list, hist_runs: list, hist_metrics, re_rows: list, design: dict,
                            *, arithmetic_rel_tol: float = 1e-9) -> dict:
    """Per operator x regime: output and decision differences against C137 and the cause of each flip."""
    OPS = _load("df_operators")
    fit_modes = {o["kind"]: o["fit_mode"] for o in design["operators"]}
    run_by_sha = {}
    h_counts: dict = {}
    h_units: dict = {}
    for r in hist_runs:
        kind = D.RETIRED_NAMES.get(r["operator_kind"], r["operator_kind"])
        key = (D.spec_sha({"kind": kind, "params": r["operator_params"]}), D.regime_key(r["regime"]))
        run_by_sha[L.row_sha256("df_fact_operator_run", r)] = (key, r)
        c = h_counts.setdefault(key, {"RESULT": 0, "REFUSED": 0, "FAILED": 0, "INCONCLUSIVE": 0})
        c["RESULT" if r["status"] == "COMPLETED" else r["status"] if r["status"] in c else "FAILED"] += 1
        h_units[r["subject_id"]] = r["content_sha256"]
    h_vals: dict = {}
    for m in hist_metrics:
        if m["metric"] not in _COMPARED or m["partition"] not in ("calibration", "confirmation"):
            continue
        hit = run_by_sha.get(m["operator_run_sha256"])
        if hit is None:
            continue
        key, r = hit
        if m["status"] == "COMPLETED":
            h_vals.setdefault(key, {})[(r["subject_id"], r["variable_id"], m["partition"], m["metric"])] = m["value"]
        elif m["status"] == "INCONCLUSIVE":
            h_counts[key]["INCONCLUSIVE"] += 1
    r_vals: dict = {}
    r_counts: dict = {}
    r_units: dict = {}
    seen_uv: dict = {}
    for r in re_rows:
        if r["mode"] != D.HISTORICAL_MODE:
            raise AdjudicationRefusal("the comparison takes HISTORICAL rows only")
        key = (r["spec_sha256"], D.regime_key(r["regime"]))
        c = r_counts.setdefault(key, {"RESULT": 0, "REFUSED": 0, "FAILED": 0, "INCONCLUSIVE": 0})
        r_units[r["unit_id"]] = r["content_sha256"]
        uv = (r["unit_id"], r["variable_id"])
        if r["metric"] == "arm_status":
            if seen_uv.setdefault((key, uv), r["status"]) == r["status"] and r["partition"] == "calibration":
                c[r["status"] if r["status"] in c else "FAILED"] += 1
            continue
        if r["branch"] == "COST":
            continue
        if r["status"] == "INCONCLUSIVE":
            c["INCONCLUSIVE"] += 1
        if (key, uv) not in seen_uv:
            seen_uv[(key, uv)] = "COMPLETED"
            c["RESULT"] += 1
        if r["metric"] in _COMPARED and r["status"] == "COMPLETED":
            r_vals.setdefault(key, {})[(r["unit_id"], r["variable_id"], r["partition"], r["metric"])] = r["value"]
    re_dec = {(d["spec_sha256"], D.regime_key(d["regime"])): d for d in reanalysis_decisions_c138(re_rows)}
    h_dec = {}
    for d in hist_decisions:
        kind = D.RETIRED_NAMES.get(d["operator_kind"], d["operator_kind"])
        h_dec[(D.spec_sha({"kind": kind, "params": d["operator_params"]}), D.regime_key(d["regime"]))] = d
    rows = []
    flips: dict = {c: 0 for c in FLIP_CAUSES}
    for key in sorted(set(h_dec) | set(re_dec)):
        hd, rd = h_dec.get(key), re_dec.get(key)
        hv, rv = h_vals.get(key, {}), r_vals.get(key, {})
        common = set(hv) & set(rv)
        diffs: dict = {}
        max_rel = 0.0
        for k in common:
            a, b = hv[k], rv[k]
            d = abs(a - b)
            rel = d / max(1.0, abs(a), abs(b))
            max_rel = max(max_rel, rel)
            m = k[3]
            diffs.setdefault(m, {"max_abs_diff": 0.0, "n": 0})
            diffs[m]["max_abs_diff"] = max(diffs[m]["max_abs_diff"], d)
            diffs[m]["n"] += 1
        only_h = sorted({(k[0], k[1]) for k in set(hv) - set(rv)})
        only_r = sorted({(k[0], k[1]) for k in set(rv) - set(hv)})
        support_changed = any(hv[k] != rv[k] for k in common if k[3] == "support")
        hist_kind = (hd or {}).get("operator_kind")
        kind = (rd or {}).get("operator_kind") or D.RETIRED_NAMES.get(hist_kind, hist_kind)
        flipped = hd is not None and rd is not None and hd["decision"] != rd["decision"]
        causes = []
        if hd is None or rd is None:
            causes = ["ABSTENTION"]
        elif flipped:
            if only_h or only_r:
                causes.append("ABSTENTION")
            if support_changed:
                causes.append("TRANSFORMED_RANGE")
            if hist_kind != kind and max_rel == 0.0:
                causes.append("NAME")
            if not causes and 0.0 < max_rel <= arithmetic_rel_tol:
                causes.append("ARITHMETIC")
            if not causes and fit_modes.get(kind) == OPS.FROZEN_PREVIOUS_PARTITION and kind not in OPS.DATA_INDEPENDENT_KINDS:
                causes.append("TEMPORAL_MODE")
            if not causes:
                causes.append("REAL_CORRECTION")
            flips[causes[0]] += 1
        rows.append({"operator_kind": kind, "historical_operator_kind": hist_kind,
                     "operator_params": (rd or hd)["operator_params"], "spec_sha256": key[0],
                     "regime": (rd or hd)["regime"], "historical_decision": (hd or {}).get("decision"),
                     "reanalysis_decision": (rd or {}).get("decision"), "flipped": flipped,
                     "flip_cause": causes[0] if causes else None, "flip_causes": causes,
                     "output_differences": {"per_metric": diffs, "max_relative_diff": max_rel,
                                            "unit_variables_only_historical": len(only_h),
                                            "unit_variables_only_reanalysis": len(only_r),
                                            "support_changed": support_changed},
                     "historical_counts": h_counts.get(key, {}), "reanalysis_counts": r_counts.get(key, {})})
    tot = lambda cs: {s: sum(c.get(s, 0) for c in cs.values()) for s in ("RESULT", "REFUSED", "FAILED", "INCONCLUSIVE")}
    return {"stratum": D.HISTORICAL_MODE, "grants_consumption": False, "rows": rows,
            "flips_by_primary_cause": flips, "flips": sum(r["flipped"] for r in rows),
            "decision_counts": {"historical": _count(h_dec.values()), "reanalysis": _count(re_dec.values())},
            "status_counts": {"historical": tot(h_counts), "reanalysis": tot(r_counts)},
            "units_equal": set(h_units) == set(r_units),
            "truth_content_equal": all(h_units.get(u) == c for u, c in r_units.items()) and set(h_units) == set(r_units),
            "units": {"historical": len(h_units), "reanalysis": len(r_units)}}


def _count(ds) -> dict:
    out: dict = {}
    for d in ds:
        out[d["decision"]] = out.get(d["decision"], 0) + 1
    return out


def historical_rows(report: dict, run_id: str, historical_run_id: str) -> list:
    code = D.lab_code_sha256()
    rows = []
    for r in report["rows"]:
        row = {"run_id": run_id, "stratum": D.HISTORICAL_MODE, "historical_run_id": historical_run_id,
               "historical_operator_kind": r["historical_operator_kind"] or "", "operator_kind": r["operator_kind"],
               "operator_params": r["operator_params"], "spec_sha256": r["spec_sha256"], "regime": r["regime"],
               "historical_decision": r["historical_decision"], "reanalysis_decision": r["reanalysis_decision"],
               "flipped": r["flipped"], "flip_cause": r["flip_cause"], "flip_causes": r["flip_causes"],
               "output_differences": r["output_differences"], "historical_counts": r["historical_counts"],
               "reanalysis_counts": r["reanalysis_counts"], "code_sha256": code}
        problems = validate_proposed_row("df_fact_d2_historical_reanalysis", row)
        if problems:
            raise AdjudicationRefusal(f"historical row does not validate: {problems[:3]}")
        rows.append(row)
    return rows
