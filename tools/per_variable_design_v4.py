#!/usr/bin/env python3
"""C100-C104 (order 2026-09-12): the per-variable preprocessing design v4.

v3 mixed two contrasts (a dimension control A3 inside H1, whose arm A1
adds no dimension), promised a t interval with three panels, admitted
variables with no semantic type, unit, role, license or missing policy,
ignored truncated bars and gaps, and stated Holm as prose. v4 fixes each,
field by field, and stays a draft that scores nothing.

This module builds the design, validates it, runs its Holm and bound
algorithms on numbers supplied to it (never on scores it computed), and
derives the population from artifacts. It imports no numeric library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "crispdm.per_variable_preprocessing_design.v4"
STATUS = "DRAFT_CANDIDATE_NO_SCORES_COMPUTED"
ARMS = ("A0_BASELINE", "A1_PER_VARIABLE", "A2_AUGMENTED", "A3_CAPACITY_CONTROL")
MIN_PANELS = 6
MIN_VARIABLES_PER_PANEL = 5


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")
                                     ).encode()).hexdigest()


# ------------------------------------------------------- executable inference
def holm_adjust(pvalues: dict) -> dict:
    """Holm step-down on one-sided p-values. Returns {contrast: {p,
    rank, alpha_level_factor, adjusted_p, reject_at(alpha)}} where the
    adjusted p is the running maximum of (m - rank + 1) * p, capped at 1."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items, start=1):
        factor = m - i + 1
        running = max(running, min(1.0, factor * p))
        out[name] = {"p": p, "rank": i, "factor": factor, "adjusted_p": running}
    return out


def holm_rejections(pvalues: dict, alpha: float) -> dict:
    adj = holm_adjust(pvalues)
    return {k: v["adjusted_p"] <= alpha for k, v in adj.items()}


def holm_bound_levels(contrasts_ordered_by_p: list, alpha: float) -> dict:
    """The one-sided confidence level each contrast's lower bound is
    reported at: the i-th smallest p is bounded at 1 - alpha/(m - i + 1)."""
    m = len(contrasts_ordered_by_p)
    return {c: 1.0 - alpha / (m - i + 1)
            for i, c in enumerate(contrasts_ordered_by_p, start=1)}


def t_quantile_one_sided(level: float, df: int) -> float:
    """Student t quantile by bisection on the regularized incomplete beta
    CDF — pure Python so the validator needs no numeric library."""
    def betacf(a, b, x):
        c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
        d = 1.0 / d if abs(d) > 1e-300 else 1e300
        h = d
        for m_ in range(1, 300):
            m2 = 2 * m_
            aa = m_ * (b - m_) * x / ((a - 1 + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / d if abs(d) > 1e-300 else 1e300
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
            h *= d * c
            aa = -(a + m_) * (a + b + m_) * x / ((a + m2) * (a + 1 + m2))
            d = 1.0 + aa * d; d = 1.0 / d if abs(d) > 1e-300 else 1e300
            c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
            delta = d * c; h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return h

    def ibeta(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + \
            a * math.log(x) + b * math.log(1 - x)
        if x < (a + 1) / (a + b + 2):
            return math.exp(lbt) * betacf(a, b, x) / a
        return 1.0 - math.exp(lbt) * betacf(b, a, 1 - x) / b

    def cdf(t):
        x = df / (df + t * t)
        tail = 0.5 * ibeta(df / 2.0, 0.5, x)
        return 1.0 - tail if t >= 0 else tail

    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def panel_contrast(values: list, level: float) -> dict:
    """One value per panel for one contrast -> mean, one-sided lower
    bound and p for H0: mean <= 0; inconclusive below MIN_PANELS."""
    n = len(values)
    if n < MIN_PANELS:
        return {"panels": n, "state": "DESCRIPTIVE_INCONCLUSIVE",
                "mean": (sum(values) / n) if n else None}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    tq = t_quantile_one_sided(level, n - 1)
    return {"panels": n, "state": "INFERENTIAL", "mean": mean, "se": se,
            "lower_bound": mean - tq * se, "level": level, "df": n - 1}


# ------------------------------------------------------------------- design
def build_design() -> dict:
    d = {
        "schema": SCHEMA, "status": STATUS,
        "supersedes": {"document": "PER_VARIABLE_PREPROCESSING_DESIGN.v3.json",
                       "rewritten": False,
                       "why": "A3 sat in H1 although A1 adds no dimension; t "
                              "inference with 3 panels; no semantic, unit, "
                              "role, license or missing-policy requirement; "
                              "no temporal-quality exclusion; Holm as prose"},
        "hypotheses": {
            "H1": {"contrast": "A1_PER_VARIABLE minus A0_BASELINE",
                   "question": "does replacing each variable by its selected "
                               "causal operator beat identity at the SAME "
                               "dimension?", "direction": "GREATER",
                   "margin": 0.01},
            "H2": {"contrasts": ["A2_AUGMENTED minus A0_BASELINE",
                                 "A2_AUGMENTED minus A3_CAPACITY_CONTROL"],
                   "question": "does the augmented representation beat both "
                               "identity and the same number of columns "
                               "carrying no target information?",
                   "direction": "GREATER", "margins": [0.01, 0.0],
                   "requires_both": True},
            "H3": {"question": "abstention frequency, selection stability "
                               "across origins, and no harm under a change "
                               "of panel", "inference": "DESCRIPTIVE_PLUS_HARM_RULE",
                   "measures": ["abstention_rate_per_panel",
                                "selected_operator_jaccard_across_origins",
                                "per_panel_A1_minus_A0_against_harm_margin"]}},
        "arms": {
            "A0_BASELINE": "every population variable untransformed",
            "A1_PER_VARIABLE": "every population variable replaced by its "
                               "selected operator; dimension equals A0",
            "A2_AUGMENTED": "per variable: untransformed, selected operator, "
                            "and their difference where both are defined",
            "A3_CAPACITY_CONTROL": {
                "inputs": "A0 inputs plus, per variable, exactly as many "
                          "extra columns as A2 adds for it",
                "generation": "i.i.d. Normal(mu_train, sigma_train) where "
                              "mu_train and sigma_train are the mean and "
                              "sample standard deviation of the corresponding "
                              "source variable on the TRAINING fold only",
                "rng_seed": "sha256(seed, panel, variable, origin, column)",
                "never_uses": ["labels", "validation fold", "test fold"],
                "controls": "the extra dimension of A2 only; it is not a "
                            "comparator for A1"}},
        "population": {
            "artifacts": {"terminals": "crispdm.lake_characterization_terminal.v4",
                          "lineage": "financial_data.feature_dag.v4",
                          "temporal": "ETH_H4_TEMPORAL_CONTRACT.v2 and its "
                                      "sample eligibility mask, one per panel",
                          "manifests": "PRODUCER_BINDING_MANIFEST.v2"},
            "requires_all": [
                "terminal v4 layer INDEPENDENTLY_RECOMPUTED",
                "semantic state NUMERIC_MEASURABLE",
                "FEATURE_DAG.v4 class CAUSAL_ACTIVE with a complete binding",
                "declared semantic type (not UNKNOWN)",
                "declared role equal to input_feature",
                "declared unit, where the semantic type has one (not UNKNOWN)",
                "declared license (not UNKNOWN)",
                "declared missing and sentinel policy",
                "missingness_fraction <= 0.20 and n_observations >= 2000"],
            "excluded_by_rule": ["identifiers", "targets", "timestamps",
                                 "variables available only after the origin"],
            "empty": "reported as empty; no requirement is relaxed"},
        "panels": {"minimum_independent_panels": MIN_PANELS,
                   "minimum_eligible_variables_per_panel": MIN_VARIABLES_PER_PANEL,
                   "independence": "distinct dataset and distinct underlying "
                                   "instrument or source",
                   "inventory": "built from artifacts without reading any score",
                   "below_minimum": "BANK_INSUFFICIENT; the screen does not run"},
        "temporal_quality": {
            "rule": "a sample is excluded before any fit if its lookback, the "
                    "operator window or its target contains a truncated bar or "
                    "crosses a gap, per the panel's temporal quality mask",
            "next_bar": "the next NOMINAL interval; an absent bar is an absent "
                        "target, never the next observed row",
            "interpolation": "NONE"},
        "common_sample": {
            "rule": "the intersection, across all four arms, of samples that "
                    "are eligible after the maximum warm-up (512 bars) and the "
                    "temporal mask",
            "frozen": "its digest is recorded before any model is fitted and "
                      "every arm is evaluated on exactly it"},
        "operators": ["O0_IDENTITY", "O1_FIRST_DIFFERENCE", "O2_LOG_RETURN",
                      "O3_TRAILING_ZSCORE_64", "O4_TRAILING_RANK_PERCENTILE_256",
                      "O5_CAUSAL_EWMA_0.3", "O6_TRAILING_WINSORIZE_512"],
        "model": {"family": "ridge", "lags": 24, "alpha": 1.0},
        "evaluation": {"origins": 5, "test_block_bars": 1000, "embargo_bars": 24,
                       "inner_validation_fraction": 0.10, "seeds": [101, 202, 303]},
        "budget": {"same_for_every_arm": True, "cpu_wall_seconds_per_panel_arm_seed": 1800,
                   "accelerator": "NONE", "includes": ["operator selection",
                                                       "transformation", "fit", "predict"]},
        "inference": {
            "unit": "one value per panel per contrast: the mean over origins "
                    "and seeds of (MAE(comparator) - MAE(arm)) / MAE(comparator) "
                    "on the frozen common sample",
            "family": ["H1", "H2_vs_A0", "H2_vs_A3"],
            "test": "one-sided paired t on panel values against the margin",
            "confidence": 0.95,
            "minimum_panels_for_inference": MIN_PANELS,
            "below_minimum": "DESCRIPTIVE_INCONCLUSIVE",
            "multiplicity": {"method": "HOLM_STEP_DOWN",
                             "algorithm": ["sort the family's one-sided p-values ascending",
                                           "for rank i of m, adjusted_p = max over ranks <= i of min(1, (m - i + 1) * p_i)",
                                           "reject where adjusted_p <= alpha",
                                           "report the rank-i lower bound at level 1 - alpha/(m - i + 1)"],
                             "implementation": "per_variable_design_v4.holm_adjust / holm_bound_levels"},
            "sensitivity": "leave-one-panel-out: every contrast recomputed "
                           "without each panel; a claim requires every LOPO "
                           "lower bound to clear its margin"},
        "rules": {"harm": "a panel with A1 minus A0 below -0.02 is harmed; any "
                          "harm blocks H1",
                  "abstention": "a variable with no operator beating identity by "
                                "0.1% on inner validation keeps O0 and counts as "
                                "an abstention in H3",
                  "missingness": "rows with non-finite inputs are dropped "
                                 "identically for every arm before the common "
                                 "sample is frozen",
                  "incompleteness": "an arm over budget or failing makes its "
                                    "panel NOT_EVALUABLE for every arm",
                  "null_result": "NO_EVIDENCE_OF_BENEFIT, reported, never re-run "
                                 "with other settings"},
        "license": {"scoring": "NOT_GRANTED",
                    "required": "EXTERNAL_V4_DESIGN_REVIEW_AND_LICENSE_REQUIRED",
                    "allowed": ["validators", "fixtures",
                                "materialization without labels", "CPU dry-run"]},
        "grants_nothing": "a design opens no screen",
    }
    d["design_sha256"] = sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


REQUIRED = ("schema", "status", "supersedes", "hypotheses", "arms", "population",
            "panels", "temporal_quality", "common_sample", "operators", "model",
            "evaluation", "budget", "inference", "rules", "license",
            "grants_nothing", "design_sha256")


def validate(d: dict, *, reviewed_sha256: str | None = None) -> list[str]:
    p = []
    if set(d) != set(REQUIRED):
        return [f"keys differ: {sorted(set(REQUIRED) ^ set(d))}"]
    if d["schema"] != SCHEMA or d["status"] != STATUS:
        p.append("schema or status")
    if d["design_sha256"] != sha_obj({k: v for k, v in d.items() if k != "design_sha256"}):
        p.append("design_sha256 does not match the content")
    if reviewed_sha256 is not None and d["design_sha256"] != reviewed_sha256:
        p.append("DELTA_AFTER_REVIEW")
    h = d["hypotheses"]
    if "A3" in h["H1"]["contrast"]:
        p.append("H1 must not use the dimension control A3")
    if h["H1"]["contrast"] != "A1_PER_VARIABLE minus A0_BASELINE":
        p.append("H1 must be A1 against A0")
    if not (h["H2"]["requires_both"] and any("A3" in c for c in h["H2"]["contrasts"])):
        p.append("H2 must compare A2 with A0 and with A3")
    a3 = d["arms"]["A3_CAPACITY_CONTROL"]
    if set(a3.get("never_uses", [])) != {"labels", "validation fold", "test fold"} \
            or "TRAINING fold only" not in a3.get("generation", ""):
        p.append("A3 must be generated from the training fold only")
    if d["panels"]["minimum_independent_panels"] < 6 or \
            d["inference"]["minimum_panels_for_inference"] < 6:
        p.append("inference needs at least six panels")
    req = " ".join(d["population"]["requires_all"]).lower()
    for word in ("semantic type", "role", "unit", "license", "missing and sentinel policy"):
        if word not in req:
            p.append(f"population must require {word}")
    if "truncated bar" not in d["temporal_quality"]["rule"] or "gap" not in d["temporal_quality"]["rule"]:
        p.append("temporal quality exclusion missing")
    mult = d["inference"]["multiplicity"]
    if not isinstance(mult, dict) or mult.get("method") != "HOLM_STEP_DOWN" or \
            not mult.get("algorithm"):
        p.append("Holm must be executable, not prose")
    if d["license"]["scoring"] != "NOT_GRANTED":
        p.append("scoring may not be granted")
    if not d["budget"]["same_for_every_arm"] or d["budget"]["accelerator"] != "NONE":
        p.append("budget")
    return p


def derive_population(*, terminals_v4: Path | None, dag_v4: Path | None,
                      temporal_contracts: list, census: Path | None) -> dict:
    missing = [n for n, x in (("terminals_v4", terminals_v4), ("feature_dag_v4", dag_v4),
                              ("census", census)) if x is None or not Path(x).exists()]
    if missing:
        return {"state": "UNDETERMINED", "missing_artifacts": missing, "members": 0}
    cen = json.loads(Path(census).read_text())
    declared_ok = {v["variable_id"] for v in cen["variables"]
                   if v.get("semantics", "UNKNOWN") != "UNKNOWN"
                   and v.get("role") == "input_feature"
                   and v.get("license", "UNKNOWN") != "UNKNOWN"
                   and v.get("unit", "UNKNOWN") != "UNKNOWN"}
    recomputed = set()
    for f in Path(terminals_v4).glob("*.json"):
        b = json.loads(f.read_text())
        if b.get("layer") == "INDEPENDENTLY_RECOMPUTED":
            recomputed.add(b["variable_id"])
    dag = json.loads(Path(dag_v4).read_text())
    active = {}
    for n in dag.get("nodes", []):
        if n.get("class") == "CAUSAL_ACTIVE":
            active.setdefault(n.get("dataset_id"), set()).add(n.get("column"))
    temporal_ok = set()
    for c in temporal_contracts:
        doc = json.loads(Path(c).read_text())
        ds = (doc.get("contract") or {}).get("dataset_id") or doc.get("dataset_id")
        if ds:
            temporal_ok.add(ds)
    panels = {ds: len(cols) for ds, cols in active.items() if ds in temporal_ok
              and len(cols) >= MIN_VARIABLES_PER_PANEL}
    return {"state": "DERIVED",
            "terminals_independently_recomputed": len(recomputed),
            "census_variables_with_declared_semantics_role_unit_license": len(declared_ok),
            "causal_active_columns": sum(len(v) for v in active.values()),
            "datasets_with_temporal_quality_contract": sorted(temporal_ok),
            "panels": panels, "panel_count": len(panels),
            "verdict": "BANK_INSUFFICIENT" if len(panels) < MIN_PANELS else "BANK_SUFFICIENT_FOR_REVIEW",
            "members": 0 if len(panels) < MIN_PANELS else sum(panels.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", type=Path)
    ap.add_argument("--validate", type=Path)
    ap.add_argument("--terminals-v4", type=Path)
    ap.add_argument("--dag-v4", type=Path)
    ap.add_argument("--census", type=Path)
    ap.add_argument("--temporal-contract", type=Path, action="append", default=[])
    a = ap.parse_args(argv)
    if a.write:
        d = build_design()
        a.write.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"design_sha256": d["design_sha256"]}))
    if a.validate:
        d = json.loads(a.validate.read_text())
        print(json.dumps({"problems": validate(d), "population": derive_population(
            terminals_v4=a.terminals_v4, dag_v4=a.dag_v4, census=a.census,
            temporal_contracts=a.temporal_contract)}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
