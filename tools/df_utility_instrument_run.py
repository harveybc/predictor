#!/usr/bin/env python3
"""Governed validation of the contrast instrument before any wider selection (R3).

Frozen BEFORE any outcome (write-once `DESIGN.json`): generators and effect sizes, fresh seeds
disjoint from Q3, n, replicates, blocks, both branch pairs, loss, margin, the exact multiplicity
(one family of two contrast ids -> alpha / 2), the calibration plan at the campaign's confidence
with the simulations DERIVED from it, every criterion with its finite-sample justification, and
the budgets. Then, governed: cost pilot -> projection against the aggregate ceiling (refuse with
the sealed design and the cost result if it does not fit) -> calibration campaign (one unit per
pair; verified equivalent records reused through the cache, never a relabeled one) -> controls
campaign (one unit per control x replicate; the leak control runs the leaking operator through
the real child and must be refused before scoring) -> terminals with the children's instants and
costs -> reconciliation -> DEVELOPMENT envelope. The budget is enforced through the isolated
runner before every child, calibrations included. A failed calibration makes the pair's controls
INCONCLUSIVE, never negative utility. Detection fractions are reported with Clopper-Pearson
intervals and complete denominators; the instrument outcome is PASS / DOES_NOT_SEPARATE /
INCONCLUSIVE / BUDGET_LIMITED. Twelve replicates do not prove a universal power guarantee, and the
positive generator is aligned to this operator only.

    python tools/df_utility_instrument_run.py --root ROOT --run-id ID --api-key-file KEY \\
        [--cache-dir DIR] [--cpu-cap-seconds 7200] [--already-spent SECONDS]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
R = _load("df_utility_run")
CTL = _load("df_utility_controls")
DEV = _load("df_utility_dev_run")
campaign = _load("df_d3_campaign")
contract = _load("df_d3_contract")
ops = _load("df_d3_operators")

DESIGN_SCHEMA = "df_utility_instrument_design.v1"
UNIT, VARIABLE = "ctl", "v0"
FAMILY = ("ctl__v0__mad_extremes_trailing__transformed", "ctl__v0__mad_extremes_trailing__augmented")
PAIRS = {"H_T": ("raw", "transformed"), "H_A": ("raw_wide", "augmented")}
CONTROL_PAIR = {"positive_H_T": "H_T", "positive_H_A": "H_A", "null_contrast": "H_T", "info_loss": "H_T", "future_leak": "H_T"}
FINAL = (H.ADVANCES, H.DOES_NOT_ADVANCE, H.INCONCLUSIVE_UNCALIBRATED, H.INSUFFICIENT_ROWS, H.REFUSED)


def sealed_design(*, n: int = 2048, replicates: int = 12, seed0: int = 200, bound_confidence: float = 0.95,
                  operator: str = CTL.OPERATOR) -> dict:
    alpha = 0.05
    alpha_adjusted = alpha / len(FAMILY)
    n_sims = H.sims_required_for_zero(alpha_adjusted, bound_confidence)
    protocol = {"target": "return", "horizon": 1, "model": "ridge", "window": 4, "n_blocks": 4, "margin": 0.0, "seed": 7,
                "min_rows_per_block": 30, "alpha": alpha, "ridge_lambda": 1.0}
    import math
    pos_min = math.ceil(replicates * 10 / 12)                    # 10 of 12 at the sealed size
    criteria = {
        "positive_H_T": {"outcome": H.ADVANCES, "min_count": pos_min, "of": replicates,
                         "why": "10/12: with a true detection rate of 0.95 the chance of fewer than 10 is 2%; with 0.5 the "
                                "chance of 10 or more is 1.9% - separates a working from a weak instrument at 12 replicates"},
        "positive_H_A": {"outcome": H.ADVANCES, "min_count": pos_min, "of": replicates, "why": "as positive_H_T, with the capacity control"},
        "null_contrast": {"outcome": H.DOES_NOT_ADVANCE, "min_count": replicates - 1, "of": replicates, "max_advances": 1,
                          "why": f"under the null the advance probability is <= alpha_adjusted = {alpha_adjusted}: expected "
                                 f"{alpha_adjusted * replicates:.2f} advances in {replicates}; P(>= 2) = 3.7%"},
        "info_loss": {"outcome": H.DOES_NOT_ADVANCE, "delta_sign": -1, "min_count": replicates - 1, "of": replicates,
                      "why": "the unsigned score cannot carry signed momentum: the transformed branch must lose in 11/12 "
                             "(one replicate of slack for a paired-block accident)"},
        "future_leak": {"outcome": H.REFUSED, "why_contains": "not causal", "min_count": replicates, "of": replicates,
                        "why": "a leak must be refused before scoring every time; no slack"},
    }
    doc = {"schema": DESIGN_SCHEMA, "purpose": "UTILITY_INSTRUMENT_VALIDATION", "operator": operator,
           "generators": {"mad_extremeness_drift": {"a_drift": CTL.A_DRIFT, "noise": CTL.NOISE, "window": CTL.W},
                          "signed_momentum": {"b_momentum": CTL.B_MOMENTUM, "noise": CTL.NOISE},
                          "white_null": {"noise": CTL.NOISE}},
           "controls": {name: {**{k: v for k, v in spec.items() if k != "expected"}, "pair": list(PAIRS[CONTROL_PAIR[name]]),
                               "hypothesis": CONTROL_PAIR[name], "criterion": criteria[name]}
                        for name, spec in CTL.CONTROLS.items()},
           "seeds": {"seed0": seed0, "replicates": replicates, "list": [seed0 + r for r in range(replicates)],
                     "disjoint_from": "Q3 (100-105)"},
           "n": n, "replicates": replicates, "protocol": protocol, "family": list(FAMILY),
           "multiplicity": {"comparisons": len(FAMILY), "alpha_adjusted": alpha_adjusted},
           "calibration_plan": {"generator": "white_null", "n": n, "bound_confidence": bound_confidence, "n_sims": n_sims},
           "calibration_contracts": [{"hypothesis": h, "branch_a": a, "branch_b": b, "protocol_key": f"{operator}__{h}", "n_sims": n_sims}
                                     for h, (a, b) in PAIRS.items()],
           "loss": "MAE on the return target", "blocks": 4,
           "reuse_policy": "verified calibration records of the SAME computation may be reused (code-bound key); never a relabeled one",
           "failed_calibration_policy": "controls of a pair whose calibration gives no support are INCONCLUSIVE, not negative",
           "scope_note": "twelve replicates estimate detection at these effect sizes only; the positive generator is aligned to "
                         "the unsigned extremeness score of this operator and validates no other representation",
           "design_sha256": ""}
    doc["design_sha256"] = campaign.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple:
    from scipy.stats import beta
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf((1 - confidence) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - confidence) / 2, k + 1, n - k))
    return (lo, hi)


def _stop(report: dict, root: Path, stopped: str, outcome: str, spent: float) -> dict:
    report.update(stopped=stopped, instrument_outcome=outcome, spent_cpu_seconds=spent)
    campaign.write_once(root / "REPORT.json", report)
    return report


def run_instrument(design: dict, *, root: Path, run_id: str, gov, trace, GR, outbox, OB, CE, code_identity: dict,
                   budgets: dict, cap_seconds: float, already_spent: float, cost_pilot_sims: int, contrast_seconds: float,
                   cache_dir: str | None = None, isolated=None) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DESIGN.json").is_file():
        if json.loads((root / "DESIGN.json").read_text())["design_sha256"] != design["design_sha256"]:
            raise R.Refusal("REFUSED: this root was frozen under another instrument design")
    else:
        campaign.write_once(root / "DESIGN.json", design)
    trace("design-frozen", sha256=design["design_sha256"])
    op = ops.build(design["operator"])
    elig = {"schema": "d3_mechanics_cells.v1", "verified": True, "run_id": run_id,
            "freeze_sha256": "instrument-fixture", "design_sha256": design["design_sha256"],
            "cells": [{"unit": UNIT, "variable": VARIABLE, "operator": design["operator"], "verdict": "MECHANICALLY_ACCEPTED",
                       "spec_sha256": contract.spec_sha256(op.describe())},
                      {"unit": UNIT, "variable": VARIABLE, "operator": "future_leak_control", "verdict": "MECHANICALLY_ACCEPTED",
                       "spec_sha256": contract.spec_sha256(CTL.FutureLeakOperator().describe()),
                       "note": "FORGED on purpose: the leak control must be refused by the prefix check, not by eligibility"}]}
    elig_path = root / "eligibility.instrument.json"
    if not elig_path.is_file():
        elig_path.write_text(json.dumps(elig, indent=1))
    base = H.Protocol(**design["protocol"], family=tuple(design["family"]), calibration_plan=dict(design["calibration_plan"]),
                      branches=("raw", "transformed", "augmented", "raw_wide"))
    report = {"schema": "df_utility_instrument_report.v1", "run_id": run_id, "design_sha256": design["design_sha256"],
              "code_identity": code_identity, "cap_seconds": cap_seconds, "already_spent_seconds": already_spent,
              "cost_pilot": None, "projection": None, "calibration": {}, "contrasts": {}, "controls": {}, "reconciliation": {},
              "terminals": [], "stopped": None, "instrument_outcome": None, "envelope": None,
              "live_checks_note": "reconciliation is a live data-gov query at run time; warehouse content checks are separate"}
    spent = lambda: already_spent + DEV.spent_cpu(root)
    plan = design["calibration_plan"]
    pairs = {c["hypothesis"]: (c["branch_a"], c["branch_b"]) for c in design["calibration_contracts"]}
    pilot_cfg = {"root": str(root / "cost_pilot"), "run_id": f"{run_id}-cost-pilot", "code_identity": code_identity,
                 "units": [{"unit": UNIT, "variable": VARIABLE, "values": [0.0] * 8, "eligibility": str(elig_path)}],
                 "operators": [design["operator"]], "hypotheses": {h: {"branch_a": a, "branch_b": b} for h, (a, b) in pairs.items()},
                 "purpose": "UTILITY_INSTRUMENT_COST_PILOT", "eligibility_state": "FABRICATED_INSTRUMENT",
                 "exposure": "DEVELOPMENT_INSTRUMENT_NO_RESERVE", "slow_control": False, "contrasts": False,
                 "plan": {**plan, "n_sims": int(cost_pilot_sims)},
                 "protocol": {**design["protocol"], "branches": ["raw", "transformed", "augmented", "raw_wide"]}, "budgets": budgets}
    guard0 = DEV.capped_isolated(root, cap_seconds - already_spent, isolated, expected=lambda j: 0.0)
    try:
        pilot = R.run_calibrations_only(pilot_cfg, gov, trace, GR=GR, outbox=outbox, isolated=guard0)
    except DEV.CapExhausted as e:
        return _stop(report, root, str(e), "BUDGET_LIMITED", spent())
    per_sim = {}
    for key, entry in pilot["calibration"].items():
        if key == "campaign":
            continue
        cpu = float((entry.get("cost") or {}).get("cpu_seconds") or 0.0)
        if entry.get("outcome") != "COMPLETED" or cpu <= 0:
            report["cost_pilot"] = pilot["calibration"]
            return _stop(report, root, f"COST_PILOT_FAILED: {key}", "INCONCLUSIVE", spent())
        per_sim[key] = cpu / cost_pilot_sims
    n_controls = len(design["controls"]) * design["replicates"]
    pilot_seconds = DEV.spent_cpu(root / "cost_pilot")
    projection = {"per_sim_seconds": per_sim, "cost_pilot_seconds": pilot_seconds,
                  "calibrations": {k: v * plan["n_sims"] for k, v in per_sim.items()},
                  "controls": n_controls * contrast_seconds, "n_controls": n_controls, "already_spent": already_spent,
                  "projected_cpu_seconds": pilot_seconds + sum(v * plan["n_sims"] for v in per_sim.values()) + n_controls * contrast_seconds}
    report.update(cost_pilot=pilot["calibration"], projection=projection)
    trace("projection", cpu_seconds=projection["projected_cpu_seconds"], cap=cap_seconds, already=already_spent)
    if already_spent + projection["projected_cpu_seconds"] > cap_seconds:
        return _stop(report, root, "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED", "BUDGET_LIMITED", spent())

    def expected(job):
        if job.get("kind") == "calibrate":
            return per_sim.get(job.get("protocol_key"), max(per_sim.values())) * float((job.get("plan") or {}).get("n_sims") or 0)
        return float(contrast_seconds)
    guard = DEV.capped_isolated(root, cap_seconds - already_spent, isolated, expected=expected)
    cal_cfg = {**pilot_cfg, "root": str(root / "calibration"), "run_id": f"{run_id}-calibration", "plan": dict(plan),
               "purpose": "UTILITY_INSTRUMENT_CALIBRATION", "calibration_cache_dir": cache_dir}
    try:
        cal = R.run_calibrations_only(cal_cfg, gov, trace, GR=GR, outbox=outbox, isolated=guard)
    except DEV.CapExhausted as e:
        return _stop(report, root, str(e), "BUDGET_LIMITED", spent())
    report["calibration"] = cal["calibration"]
    report["reconciliation"]["calibration"] = cal["reconciliation"]["calibration"]
    records, protocols = {}, {}
    for c in design["calibration_contracts"]:
        key = c["protocol_key"]
        attempt = root / "calibration" / "attempts" / f"calibrate__{key}"
        rec = json.loads((attempt / "calibration.json").read_text()) if (attempt / "calibration.json").is_file() else None
        entry = dict(report["calibration"].get(key) or {})
        if rec is not None and not H.calibration_record_problems(rec):
            records[key] = rec
            protocols[key] = base.with_calibration(rec)
            ok, why = H.calibration_supports(base, op, design["n"], record=rec, branch_a=c["branch_a"], branch_b=c["branch_b"])
            entry["supports"] = {"decision": ok, "why": why}
            entry["derived"] = {k: v for k, v in H.derive_calibration(rec).items() if k != "problems"}
        else:
            protocols[key] = base
            entry["supports"] = {"decision": False, "why": "no verified record"}
        report["calibration"][key] = entry
    frozen = {"schema": "df_utility_freeze.v2", "run_id": run_id, "frozen_utc": R.now_iso(), "freeze_pre_sha256": design["design_sha256"],
              "protocols": {k: p.sealed() for k, p in protocols.items()}, "calibrated": sorted(records),
              "eligibility": {UNIT: str(elig_path)}, "eligibility_sha256": {UNIT: hashlib.sha256(elig_path.read_bytes()).hexdigest()},
              "code_identity": code_identity, "freeze_sha256": ""}
    frozen["freeze_sha256"] = campaign.sha_obj({k: v for k, v in frozen.items() if k != "freeze_sha256"})
    if (root / "FREEZE.json").is_file():
        frozen = json.loads((root / "FREEZE.json").read_text())
    else:
        campaign.write_once(root / "FREEZE.json", frozen)
    if not (root / "FREEZE.pre.json").is_file():
        campaign.write_once(root / "FREEZE.pre.json", {
            "schema": "df_utility_freeze_pre.v1", "run_id": run_id, "plan": plan, "operators": [design["operator"]],
            "protocol_base": base.sealed(), "protocol_base_sha256": base.base_sha256(),
            "units": [{"unit": UNIT, "variable": VARIABLE, "n": design["n"], "data_sha256": "generated-per-replicate"}],
            "code_identity": code_identity, "purpose": design["purpose"], "freeze_sha256": design["design_sha256"]})
    units = [f"{name}__r{r:02d}" for name in design["controls"] for r in range(design["replicates"])]
    key = f"{run_id}-utility-contrasts"
    registrations_path = root / "CAMPAIGNS.json"
    registrations = json.loads(registrations_path.read_text()) if registrations_path.is_file() else {}
    if key in registrations:
        sha = registrations[key]["campaign_sha256"]
        trace("register", key=key, http="resumed")
    else:
        status, reg = gov.submit_campaign({"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                                           "project": "predictor", "code_identity": code_identity, "config_sha256": frozen["freeze_sha256"],
                                           "input_mode": "SYNTHETIC", "synthetic_spec_sha256": design["design_sha256"], "units": units,
                                           "datasets": [], "terminal_lake": "olap_cube"})
        trace("register", key=key, http=status)
        if status not in (200, 201):
            raise R.Refusal(f"REFUSED: campaign {key} refused: http {status}; no child was started")
        sha = reg["campaign_sha256"]
        registrations[key] = {"campaign_sha256": sha, "http": status, "at": R.now_iso()}
        registrations_path.write_text(json.dumps(registrations, indent=1))
    report["contrasts"] = {"campaign": {"key": key, "campaign_sha256": sha}, "outcomes": {}}
    _, done = gov.reconcile_campaign(sha)
    missing = (done or {}).get("missing_units")
    reported = set(units) - set(units if missing is None else missing)
    outcomes = {}
    for unit_id in units:
        name, rep = unit_id.rsplit("__r", 1)
        r = int(rep)
        spec = design["controls"][name]
        hyp = spec["hypothesis"]
        pkey = f"{design['operator']}__{hyp}"
        seed = design["seeds"]["list"][r]
        values = CTL.generate(spec["generator"], design["n"], np.random.default_rng(seed)).tolist()
        job = {"contrast_id": design["family"][0] if hyp == "H_T" else design["family"][1], "unit": UNIT, "variable": VARIABLE,
               "operator": "future_leak_control" if name == "future_leak" else design["operator"],
               "protocol": protocols[pkey].sealed(), "series": {"values": values}, "eligibility": str(elig_path),
               "protocol_key": pkey, "hypothesis": hyp, "branch_a": spec["pair"][0], "branch_b": spec["pair"][1],
               "control": name, "replicate": r, "seed": seed}
        attempt = root / "attempts" / unit_id
        resumed = unit_id in reported and (attempt / "outcome.json").is_file()
        if not resumed:
            GR._require_reconciled(gov, sha, unit_id, before_run=True)
            trace("before_run", key=key, unit=unit_id)
        trace("child", kind="contrast", name=unit_id)
        try:
            out = guard({**job, "kind": "contrast"}, attempt_dir=attempt, assigned_bytes=budgets["task_memory_bytes"],
                        wall_seconds=budgets["wall_seconds"], cpu_seconds=budgets["cpu_seconds"])
        except DEV.CapExhausted as e:
            report["stopped"] = str(e)
            break
        trace("child-done", kind="contrast", name=unit_id, outcome=out.get("outcome"))
        out.update(control=name, replicate=r, hypothesis=hyp, operator=design["operator"])
        outcomes[unit_id] = out
        score = out.get("score") or {}
        cost = out["cost"]
        if resumed:
            report["terminals"].append({"unit_id": unit_id, "status": "RESUMED", "outcome": out["outcome"], "cost": cost,
                                        "delta_mean": score.get("delta_mean"), "resumed": True})
            continue
        if out["outcome"] in FINAL:
            status_t, reason = "COMPLETED", None
        elif out["outcome"] == H.RESOURCE_EXCEEDED:
            status_t, reason = "FAILED", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        else:
            status_t, reason = "INCONCLUSIVE", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        metrics = []
        if status_t == "COMPLETED" and isinstance(score.get("delta_mean"), (int, float)):
            metrics = [R._metric("utility.delta_mean", score["delta_mean"], score["loss_name"]),
                       R._metric("utility.delta_lower", score["delta_lower"], score["loss_name"]),
                       R._metric("utility.delta_se", score["delta_se"], score["loss_name"]),
                       R._metric("utility.blocks_used", score["blocks_used"], "count")]
        terminal = R._terminal(status=status_t, reason=reason, cost=cost, metrics=metrics,
                               started=cost.get("started_at") or R.now_iso(), finished=cost.get("ended_at") or R.now_iso(),
                               tags={"purpose": design["purpose"], "grants": "NONE", "classification": "NON_GOVERNING",
                                     "outcome": str(out["outcome"]), "control": name, "replicate": str(r), "seed": str(seed),
                                     "hypothesis": hyp, "branch_a": spec["pair"][0], "branch_b": spec["pair"][1],
                                     "protocol_sha256": protocols[pkey].sealed()["protocol_sha256"], "freeze_sha256": frozen["freeze_sha256"],
                                     "output_sha256": out.get("output_sha256") or "", "calibrated": str(pkey in records)})
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        report["terminals"].append({"unit_id": unit_id, "status": status_t, "outcome": out["outcome"], "cost": cost,
                                    "delta_mean": score.get("delta_mean"), "pending_after_flush": flushed["pending"]})
    rstatus, rbody = gov.reconcile_campaign(sha)
    report["reconciliation"]["contrasts"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                             "accounting_only": rbody.get("accounting_only"), "lake_only": rbody.get("lake_only")}
    report["contrasts"]["outcomes"] = {k: {kk: vv for kk, vv in v.items() if kk != "score"} for k, v in outcomes.items()}
    report["controls"] = judge(design, outcomes, report["calibration"])
    report["instrument_outcome"] = "BUDGET_LIMITED" if report["stopped"] else report["controls"]["instrument_outcome"]
    report["spent_cpu_seconds"] = spent()
    if outcomes:
        cfg_env = {"purpose": design["purpose"], "run_id": run_id, "code_identity": code_identity, "eligibility_state": "FABRICATED_INSTRUMENT",
                   "exposure": "DEVELOPMENT_INSTRUMENT_NO_RESERVE", "operators": [design["operator"]]}
        pre_env = {"protocol_base_sha256": base.base_sha256(), "units": [{"unit": UNIT, "variable": VARIABLE, "data_sha256": "generated-per-replicate"}]}
        report["envelope"] = R.emit_envelope(cfg_env, outcomes, frozen, pre_env, OB, CE)
    campaign.write_once(root / "REPORT.json", report)
    return report


def judge(design: dict, outcomes: dict, calibration: dict) -> dict:
    """Detection fractions with 95 % Clopper-Pearson intervals and complete denominators;
    calibration support apart; incomplete replicates listed; the outcome of the instrument."""
    out = {"per_control": {}, "incomplete": [], "instrument_outcome": None}
    all_met, any_incomplete, any_inconclusive = True, False, False
    for name, spec in design["controls"].items():
        crit = spec["criterion"]
        done = [x for x in (outcomes.get(f"{name}__r{r:02d}") for r in range(design["replicates"])) if x is not None]
        pkey = f"{design['operator']}__{spec['hypothesis']}"
        supported = ((calibration.get(pkey) or {}).get("supports") or {}).get("decision") is True
        hits = [x for x in done if x.get("outcome") == crit["outcome"]
                and (crit.get("delta_sign") is None or ((x.get("score") or {}).get("delta_mean") is not None
                                                        and np.sign((x.get("score") or {}).get("delta_mean")) == crit["delta_sign"]))
                and (crit.get("why_contains") is None or crit["why_contains"] in ((x.get("score") or {}).get("why") or ""))]
        advances = sum(1 for x in done if x.get("outcome") == H.ADVANCES)
        inconclusive = sum(1 for x in done if x.get("outcome") == H.INCONCLUSIVE_UNCALIBRATED)
        k, n = len(hits), len(done)
        lo, hi = clopper_pearson(k, n)
        complete = n == design["replicates"]
        needs_support = crit["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE)
        met = complete and k >= crit["min_count"] and advances <= crit.get("max_advances", design["replicates"])
        status = "MET" if met else "NOT_MET"
        if needs_support and not supported:
            status = "INCONCLUSIVE_CALIBRATION"
            any_inconclusive = True
        if not complete:
            status = "INCOMPLETE"
            any_incomplete = True
            out["incomplete"].append({"control": name, "completed": n, "of": design["replicates"]})
        all_met &= status == "MET"
        out["per_control"][name] = {"criterion": crit, "hits": k, "completed": n, "of": design["replicates"],
                                    "fraction": (k / n) if n else None, "ci95": [lo, hi], "advances": advances,
                                    "inconclusive_uncalibrated": inconclusive, "calibration_supports": supported, "status": status,
                                    "deltas": [(x.get("score") or {}).get("delta_mean") for x in done],
                                    "lower_bounds": [(x.get("score") or {}).get("delta_lower") for x in done]}
    if any_incomplete:
        out["instrument_outcome"] = "BUDGET_LIMITED"
    elif all_met:
        out["instrument_outcome"] = "PASS"
    elif any_inconclusive and all(v["status"] in ("MET", "INCONCLUSIVE_CALIBRATION") for v in out["per_control"].values()):
        out["instrument_outcome"] = "INCONCLUSIVE"
    else:
        out["instrument_outcome"] = "DOES_NOT_SEPARATE"
    return out


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    CE = _load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cpu-cap-seconds", type=float, default=7200.0)
    parser.add_argument("--already-spent", type=float, default=0.0, help="CPU already spent by this order's rehearsals")
    parser.add_argument("--cost-pilot-sims", type=int, default=6)
    parser.add_argument("--contrast-seconds", type=float, default=3.0)
    parser.add_argument("--task-memory", type=int, default=1 << 30)
    parser.add_argument("--wall-seconds", type=float, default=300.0)
    parser.add_argument("--cpu-seconds", type=int, default=300)
    parser.add_argument("--calibration-wall-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    design = sealed_design()
    code_identity = GR.strict_code_identity(REPO)
    budgets = {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds, "cpu_seconds": args.cpu_seconds,
               "calibration_wall_seconds": args.calibration_wall_seconds, "calibration_cpu_seconds": int(args.calibration_wall_seconds),
               "mechanics_wall_seconds": args.wall_seconds, "mechanics_cpu_seconds": args.cpu_seconds,
               "slow_control": {"slow_seconds": 0.0, "wall_seconds": 1.0}}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": R.now_iso(), **facts})
        print(json.dumps({"event": event, **{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:80])
                                              for k, v in facts.items()}}), flush=True)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    report = run_instrument(design, root=args.root, run_id=args.run_id, gov=gov, trace=trace, GR=GR, outbox=outbox, OB=OB, CE=CE,
                            code_identity=code_identity, budgets=budgets, cap_seconds=args.cpu_cap_seconds,
                            already_spent=args.already_spent, cost_pilot_sims=args.cost_pilot_sims,
                            contrast_seconds=args.contrast_seconds, cache_dir=args.cache_dir)
    (args.root / "TRACE.json").write_text(json.dumps(trace_log, indent=1, default=str))
    print(json.dumps({"instrument_outcome": report["instrument_outcome"], "stopped": report["stopped"],
                      "spent_cpu_seconds": report.get("spent_cpu_seconds"),
                      "projection": (report.get("projection") or {}).get("projected_cpu_seconds"),
                      "calibration": {k: (v.get("outcome"), (v.get("supports") or {}).get("decision"), (v.get("source") or {}).get("kind"))
                                      for k, v in report["calibration"].items() if k != "campaign"},
                      "controls": {k: (v["hits"], v["completed"], v["of"], v["status"])
                                   for k, v in (report.get("controls") or {}).get("per_control", {}).items()}}, indent=1, default=str))
    return 0 if report["instrument_outcome"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
