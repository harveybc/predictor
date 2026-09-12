#!/usr/bin/env python3
"""C84 (order 2026-09-12): the per-variable preprocessing design v3, as a
document that can be validated field by field and refuses any change
after review.

This module builds the design, validates it, and derives its population
from the three artifacts the order names — v3 lake terminals, FEATURE_DAG
v3 and the E5a temporal gate — without scoring anything. It reads no
labels, fits no model and imports no learning library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "crispdm.per_variable_preprocessing_design.v3"
STATUS = "DRAFT_CANDIDATE_NO_SCORES_COMPUTED"
ARMS = ("A0_BASELINE", "A1_PER_VARIABLE", "A2_AUGMENTED",
        "A3_CAPACITY_CONTROL")


class DesignRefusal(SystemExit):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True,
                                     separators=(",", ":")).encode()
                          ).hexdigest()


def build_design() -> dict:
    d = {
        "schema": SCHEMA,
        "status": STATUS,
        "supersedes": {
            "document": "PER_VARIABLE_PREPROCESSING_DESIGN.v2.json",
            "why": "v2 fixed neither population, operators, parameters, "
                   "models, windows, seeds, budgets, numeric margins, "
                   "minimum panels nor the construction of the capacity "
                   "control",
            "rewritten": False},
        "question": "On causally available financial input variables, does "
                    "choosing one causal preprocessing operator PER "
                    "VARIABLE reduce one-step-ahead forecast error relative "
                    "to leaving every variable untransformed, by more than "
                    "adding the same number of uninformative columns does?",
        "primary_hypothesis": {
            "id": "H1",
            "statement": "the panel-mean relative MAE improvement of "
                         "A1_PER_VARIABLE over A0_BASELINE exceeds the "
                         "practical margin, and A1 also beats "
                         "A3_CAPACITY_CONTROL",
            "direction": "ONE_SIDED_IMPROVEMENT"},
        "population": {
            "rule": "a variable enters only if ALL hold, evaluated at "
                    "validation time from the artifacts, never from a list",
            "conditions": [
                {"id": "P1", "artifact": "crispdm.lake_characterization_"
                                         "terminal.v3",
                 "requires": "layer == INDEPENDENTLY_RECOMPUTED and "
                             "recomputation == AGREES"},
                {"id": "P2", "artifact": "financial_data.feature_dag.v3",
                 "requires": "class == CAUSAL_ACTIVE (bound producer; "
                             "prefix invariance passed if executable)"},
                {"id": "P3", "artifact": "financial_data.temporal_"
                                         "availability contract",
                 "requires": "gate E5a == OPEN for the variable's dataset"},
                {"id": "P4", "artifact": "crispdm.lake_characterization_"
                                         "terminal.v3",
                 "requires": "missingness_fraction <= 0.20 and "
                             "n_observations >= 2000"}],
            "empty_population": "reported as EMPTY; no condition is "
                                "relaxed to obtain members"},
        "panels": {
            "definition": "one panel per dataset_id holding at least one "
                          "population member",
            "minimum_panels": 3,
            "minimum_members_per_panel": 5,
            "below_minimum": "NOT_EVALUABLE; the screen does not run"},
        "statistical_unit": {
            "unit": "panel",
            "nested": "variables, origins and seeds are repeated measures "
                      "inside a panel and never add independent samples",
            "inference": "one value per panel per contrast"},
        "target": {
            "definition": "next-bar log return of the panel's CLOSE, "
                          "log(CLOSE[t+1]) - log(CLOSE[t])",
            "availability": "labels are read only after every input's E5a "
                            "bound precedes the forecast origin",
            "labels_read_by_this_document": False},
        "operators": [
            {"id": "O0_IDENTITY", "params": {}, "causal": True},
            {"id": "O1_FIRST_DIFFERENCE", "params": {"lag": 1},
             "causal": True},
            {"id": "O2_LOG_RETURN", "params": {"lag": 1,
                                               "requires": "strictly positive"},
             "causal": True},
            {"id": "O3_TRAILING_ZSCORE", "params": {"window": 64,
                                                    "min_periods": 64,
                                                    "ddof": 1},
             "causal": True},
            {"id": "O4_TRAILING_RANK_PERCENTILE", "params": {"window": 256,
                                                             "min_periods": 256},
             "causal": True},
            {"id": "O5_CAUSAL_EWMA", "params": {"alpha": 0.3,
                                                "adjust": False},
             "causal": True},
            {"id": "O6_TRAILING_WINSORIZE", "params": {"window": 512,
                                                       "min_periods": 512,
                                                       "lower_q": 0.01,
                                                       "upper_q": 0.99},
             "causal": True}],
        "operator_selection": {
            "where": "inner validation split of the development fold of "
                     "each origin only; never on a test block",
            "criterion": "lowest inner-validation MAE of the fixed model "
                         "using that single variable's transformed series "
                         "plus the untransformed remaining inputs",
            "tie_break": "operator id order",
            "abstention": [
                {"state": "ABSTAIN_NO_GAIN",
                 "rule": "no operator beats O0_IDENTITY on inner "
                         "validation MAE by more than 0.1%; the variable "
                         "keeps O0 in A1 and is counted as an abstention"},
                {"state": "ABSTAIN_DOMAIN",
                 "rule": "an operator whose domain condition fails (e.g. "
                         "O2 on a non-positive series) is not a candidate"},
                {"state": "ABSTAIN_INSUFFICIENT_SUPPORT",
                 "rule": "fewer than 500 finite development rows"}],
            "abstentions_count_in_multiplicity": True},
        "model": {
            "family": "ridge regression on lagged inputs",
            "lags": list(range(1, 25)),
            "alpha": 1.0,
            "standardization": "fit on the development fold only",
            "deterministic": True},
        "evaluation_windows": {
            "scheme": "rolling origin, expanding development window",
            "origins": 5,
            "test_block_bars": 1000,
            "inner_validation_fraction": 0.10,
            "embargo_bars": 24,
            "data_scope": "DEVELOPMENT_EXPOSED data only; nothing here "
                          "touches confirmatory data"},
        "seeds": [101, 202, 303],
        "arms": {
            "A0_BASELINE": {"inputs": "every population variable, O0"},
            "A1_PER_VARIABLE": {"inputs": "every population variable, its "
                                          "selected operator"},
            "A2_AUGMENTED": {"inputs": "for every variable: O0 series, "
                                       "selected-operator series, and their "
                                       "difference where both are defined"},
            "A3_CAPACITY_CONTROL": {
                "inputs": "A0 inputs plus, per variable, exactly as many "
                          "extra columns as A2 adds for it",
                "extra_columns": "i.i.d. standard normal values generated "
                                 "from seed only: sha256(seed, panel, "
                                 "variable, origin, column) as the RNG seed",
                "information": "the generator reads no data, so the columns "
                               "carry no information and cannot leak"}},
        "budget": {
            "same_for_every_arm": True,
            "cpu_wall_seconds_per_panel_per_arm_per_seed": 1800,
            "accelerator": "NONE",
            "exceeding_budget": "the arm is INCOMPLETE for that panel and "
                                "the panel is NOT_EVALUABLE for every arm"},
        "metrics": {
            "primary": {"id": "REL_MAE_IMPROVEMENT",
                        "definition": "(MAE(A0) - MAE(arm)) / MAE(A0), "
                                      "averaged over origins and seeds "
                                      "within a panel",
                        "polarity": "HIGHER_IS_BETTER"},
            "secondary": [
                {"id": "REL_MAE_A1_VS_A3", "polarity": "HIGHER_IS_BETTER"},
                {"id": "REL_MAE_A2_VS_A0", "polarity": "HIGHER_IS_BETTER"},
                {"id": "DIRECTION_ACCURACY", "polarity": "HIGHER_IS_BETTER"},
                {"id": "TOTAL_CPU_SECONDS", "polarity": "LOWER_IS_BETTER"}]},
        "decision_rule": {
            "practical_margin": 0.01,
            "per_panel_harm_margin": -0.02,
            "confidence_level": 0.95,
            "interval": "one-sided t lower bound over panel values",
            "family": ["A1_VS_A0", "A1_VS_A3", "A2_VS_A0"],
            "multiplicity": "HOLM over the family",
            "advance_requires": [
                "Holm-adjusted lower bound of A1_VS_A0 > practical_margin",
                "Holm-adjusted lower bound of A1_VS_A3 > 0",
                "no panel with A1_VS_A0 below per_panel_harm_margin",
                "complete cost accounting including operator fitting",
                "minimum_panels met"]},
        "missingness": {
            "rule": "rows with any non-finite input are dropped per origin "
                    "identically for every arm; no imputation; operator "
                    "warm-up rows are dropped identically for every arm"},
        "cost": {"recorded": ["operator fitting", "selection", "model fit",
                              "prediction"],
                 "unit": "CPU wall seconds per arm, panel, origin and seed"},
        "withdrawal": [
            "population below 10 variables or panels below minimum_panels",
            "budget cannot be met identically for every arm",
            "any operator fails a prefix-invariance probe",
            "any population batch stops being EXACT",
            "FEATURE_DAG.v3, the v3 terminals or the temporal contract are "
            "revised after review"],
        "null_result": {
            "label": "NO_EVIDENCE_OF_BENEFIT",
            "rule": "reported as found; the design is not re-run with other "
                    "operators, parameters, margins or panels"},
        "license": {
            "scoring": "NOT_GRANTED",
            "required": "EXTERNAL_DESIGN_REVIEW_AND_LICENSE_REQUIRED",
            "cpu_mechanics_allowed": ["parsers", "materialization without "
                                      "targets", "dry runs", "fixtures"]},
        "grants_nothing": "a design describes what would be done; it opens "
                          "no screen",
    }
    d["design_sha256"] = sha_obj({k: v for k, v in d.items()
                                  if k != "design_sha256"})
    return d


REQUIRED = ("schema", "status", "supersedes", "question",
            "primary_hypothesis", "population", "panels",
            "statistical_unit", "target", "operators",
            "operator_selection", "model", "evaluation_windows", "seeds",
            "arms", "budget", "metrics", "decision_rule", "missingness",
            "cost", "withdrawal", "null_result", "license",
            "grants_nothing", "design_sha256")


def validate(d: dict, *, reviewed_sha256: str | None = None) -> list[str]:
    problems = []
    if set(d) != set(REQUIRED):
        problems.append(f"keys differ: missing {sorted(set(REQUIRED) - set(d))}"
                        f" extra {sorted(set(d) - set(REQUIRED))}")
        return problems
    if d["schema"] != SCHEMA:
        problems.append("foreign schema")
    if d["status"] != STATUS:
        problems.append("status must stay DRAFT_CANDIDATE_NO_SCORES_COMPUTED")
    if d["design_sha256"] != sha_obj({k: v for k, v in d.items()
                                      if k != "design_sha256"}):
        problems.append("design_sha256 does not match the content")
    if reviewed_sha256 is not None and d["design_sha256"] != reviewed_sha256:
        problems.append("DELTA_AFTER_REVIEW: the design differs from the "
                        "reviewed digest")
    if set(d["arms"]) != set(ARMS):
        problems.append("arms are not exactly A0..A3")
        return problems
    if not d["budget"]["same_for_every_arm"]:
        problems.append("arms do not share one budget")
    if d["budget"]["accelerator"] != "NONE":
        problems.append("an accelerator is not allowed")
    ops = d["operators"]
    if len({o["id"] for o in ops}) != len(ops) or ops[0]["id"] != "O0_IDENTITY":
        problems.append("operator ids not unique or identity not first")
    if not all(o["causal"] is True for o in ops):
        problems.append("a non-causal operator is listed")
    for o in ops:
        w = o["params"].get("window")
        if w is not None and (type(w) is not int or w < 2 or
                              o["params"].get("min_periods") != w):
            problems.append(f"{o['id']}: trailing window must be an int with "
                            "min_periods equal to it")
    dr = d["decision_rule"]
    for k, lo, hi in (("practical_margin", 0.0, 1.0),
                      ("per_panel_harm_margin", -1.0, 0.0),
                      ("confidence_level", 0.5, 1.0)):
        v = dr[k]
        if type(v) is not float or not (lo <= v <= hi) or math.isnan(v):
            problems.append(f"decision_rule.{k} out of range")
    if dr["multiplicity"] != "HOLM over the family" or len(dr["family"]) != 3:
        problems.append("multiplicity family not fixed")
    if type(d["panels"]["minimum_panels"]) is not int or \
            d["panels"]["minimum_panels"] < 3:
        problems.append("minimum_panels must be an int >= 3")
    if d["seeds"] != sorted(set(d["seeds"])) or len(d["seeds"]) < 3:
        problems.append("seeds must be at least three distinct, sorted")
    ew = d["evaluation_windows"]
    if not (type(ew["origins"]) is int and ew["origins"] >= 3 and
            type(ew["embargo_bars"]) is int and ew["embargo_bars"] >=
            max(d["model"]["lags"])):
        problems.append("origins < 3 or embargo shorter than the longest lag")
    a3 = d["arms"]["A3_CAPACITY_CONTROL"]
    if "reads no data" not in a3["information"]:
        problems.append("A3 extra columns must be generated without data")
    if d["target"]["labels_read_by_this_document"] is not False:
        problems.append("the design document may not read labels")
    if d["license"]["scoring"] != "NOT_GRANTED":
        problems.append("the design may not grant scoring")
    return problems


def derive_population(*, terminals_v3: Path | None, dag_v3: Path | None,
                      temporal_contracts: list[Path]) -> dict:
    """Count members from the artifacts. A missing artifact makes the
    population UNDETERMINED, never assumed."""
    missing = [n for n, p in (("terminals_v3", terminals_v3),
                              ("feature_dag_v3", dag_v3))
               if p is None or not Path(p).exists()]
    if missing:
        return {"state": "UNDETERMINED", "missing_artifacts": missing,
                "members": 0}
    recomputed = set()
    for f in sorted(Path(terminals_v3).glob("*.json")):
        b = json.loads(f.read_text())
        if b.get("layer") == "INDEPENDENTLY_RECOMPUTED" and \
                b.get("recomputation") == "AGREES":
            recomputed.add(b["variable_id"])
    dag = json.loads(Path(dag_v3).read_text())
    causal_datasets = {}
    for n in dag.get("nodes", []):
        if n.get("class") == "CAUSAL_ACTIVE":
            causal_datasets.setdefault(n["dataset_id"], set()).add(n["column"])
    e5a_open = set()
    for c in temporal_contracts:
        doc = json.loads(Path(c).read_text())
        if doc.get("gates", {}).get("E5a", {}).get("status") == "OPEN":
            ds = doc.get("contract", {}).get("dataset_id") or \
                doc.get("artifact", {}).get("dataset_id")
            if ds:
                e5a_open.add(ds)
    panels = {ds: sorted(cols) for ds, cols in causal_datasets.items()
              if ds in e5a_open}
    return {
        "state": "DERIVED",
        "terminal_members_independently_recomputed": len(recomputed),
        "causal_active_columns": sum(len(v) for v in causal_datasets.values()),
        "datasets_with_e5a_open": sorted(e5a_open),
        "panels": {k: len(v) for k, v in panels.items()},
        "panel_count": len(panels),
        "note": "P1 binds lake variables and P2/P3 bind dataset columns; "
                "no mapping between the lake census and the dataset "
                "inventory exists yet, so a column is counted only where "
                "P2 and P3 agree and P1 is reported separately rather than "
                "joined by name",
        "members": 0 if len(panels) < 3 else sum(len(v) for v in panels.values()),
        "evaluable": len(panels) >= 3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--validate", type=Path, default=None)
    ap.add_argument("--reviewed-sha256", default=None)
    ap.add_argument("--terminals-v3", type=Path, default=None)
    ap.add_argument("--dag-v3", type=Path, default=None)
    ap.add_argument("--temporal-contract", type=Path, action="append",
                    default=[])
    a = ap.parse_args(argv)
    if a.write:
        d = build_design()
        a.write.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"written": str(a.write.name),
                          "design_sha256": d["design_sha256"]}))
    if a.validate:
        d = json.loads(a.validate.read_text())
        problems = validate(d, reviewed_sha256=a.reviewed_sha256)
        pop = derive_population(terminals_v3=a.terminals_v3,
                                dag_v3=a.dag_v3,
                                temporal_contracts=a.temporal_contract)
        print(json.dumps({"problems": problems, "population": pop},
                         indent=1, sort_keys=True))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
