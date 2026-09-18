#!/usr/bin/env python3
"""Governed DEVELOPMENT selection and replication from a sealed design (P3), family by family.

    validate the sealed design against the pilot's inherited protocol and the bank
    cost pilot   one governed calibration child per calibration contract type (operator ×
                 hypothesis) with a few simulations; per-simulation CPU measured, the whole
                 campaign projected (every contract × its simulations + every contrast); if the
                 projection exceeds the aggregate CPU ceiling, PLAN.json is written and nothing
                 more is launched
    families     for each family of the design, in order: the governed rehearsal entry point
                 (freeze-pre → calibration campaign with one unit per contract → contrasts
                 campaign → terminals → reconcile → envelope), under the same child ceilings and
                 an aggregate CPU ceiling over calibrations, contrasts, failures and retries;
                 exhausted mid-way, the run stops and keeps every incomplete attempt
    close        REPORT.json with every family's receipt, spent CPU and the stop reason

Resume: completed attempts are re-verified, never re-run; a different job is never taken as
the same attempt (harness O2). Outcomes without a favourable calibration stay descriptive
(INCONCLUSIVE_UNCALIBRATED); nothing here promotes a candidate beyond PROPOSED_FOR_REVIEW.

    python tools/df_utility_dev_run.py --design DESIGN.json --pilot-root PILOT --bank BANK \\
        --root ROOT --run-id ID --gov-url URL --api-key-file KEY [--cpu-cap-seconds 14400]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
R = _load("df_utility_run")
D = _load("df_utility_next_design")
campaign = _load("df_d3_campaign")

CAP_EXHAUSTED = "CPU_CAP_EXHAUSTED"


class CapExhausted(SystemExit):
    pass


def spent_cpu(root: Path) -> float:
    """CPU seconds recorded by every attempt under the root (calibrations, contrasts, cost pilot,
    failures and retries alike) — read from the attempts, never from a counter alone."""
    total = 0.0
    for path in Path(root).rglob("outcome.json"):
        try:
            rec = json.loads(path.read_text())
        except ValueError:
            continue
        summary = rec.get("summary") if isinstance(rec.get("summary"), dict) else rec
        total += float(((summary or {}).get("cost") or {}).get("cpu_seconds") or 0.0)
    return total


def capped_isolated(root: Path, cap_seconds: float, isolated=None, expected=None):
    """The child runner under the aggregate ceiling: before a fresh child, the CPU already spent
    plus the child's own ceiling must fit; a resumed attempt costs nothing more."""
    run = isolated or H.run_isolated
    root = Path(root)

    def _run(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        if not (Path(attempt_dir) / "outcome.json").is_file():
            spent = spent_cpu(root)
            need = expected(job) if expected else float(cpu_seconds)
            if spent + need > cap_seconds:
                raise CapExhausted(f"{CAP_EXHAUSTED}: spent {spent:.0f} s + next child up to {need:.0f} s "
                                   f"exceeds the ceiling {cap_seconds:.0f} s; nothing more is launched")
        return run(job, attempt_dir=attempt_dir, assigned_bytes=assigned_bytes,
                   wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)
    return _run


def family_cfg(design: dict, fam: dict, *, root: Path, run_id: str, code_identity: dict, values: list,
               budgets: dict, resume_under_new_code: bool = False) -> dict:
    pdoc = fam["protocol"]
    protocol = {k: v for k, v in pdoc.items()
                if k not in ("protocol_sha256", "comparisons", "alpha_adjusted", "family", "calibration",
                             "calibration_plan", "schema")}
    protocol["branches"] = list(pdoc["branches"])
    return {"root": str(root), "run_id": run_id, "code_identity": code_identity,
            "units": [{"unit": fam["unit"], "variable": fam["variable"], "values": values,
                       "eligibility": design["cells_record"]}],
            "operators": list(design["operators"]),
            "hypotheses": {h: {"branch_a": s["branch_a"], "branch_b": s["branch_b"]}
                           for h, s in design["hypotheses"].items()},
            "expected_family": [m["contrast_id"] for m in fam["members"]],
            "purpose": "UTILITY_DEVELOPMENT_SELECTION" if fam["role"] == "selection" else "UTILITY_DEVELOPMENT_REPLICATION",
            "eligibility_state": design["eligibility_state"],
            "exposure": "DEVELOPMENT_SELECTION_NO_RESERVE", "slow_control": False,
            "resume_under_new_code": resume_under_new_code,
            "plan": dict(fam["calibration_plan"]), "protocol": protocol, "budgets": budgets,
            "design_sha256": design["design_sha256"], "family_role": fam["role"], "replica_of": fam.get("replica_of")}


def project(design: dict, per_sim: dict, contrast_seconds: float, pilot_seconds: float) -> dict:
    """The campaign's CPU, contract by contract, from the measured cost pilot."""
    contracts = []
    total = pilot_seconds
    for fam in design["families"]:
        for c in fam["calibration_contracts"]:
            key = f"{c['operator']}__{c['hypothesis']}"
            secs = per_sim[key] * c["n_sims"]
            contracts.append({"family": fam["unit"], "contract": key, "n_sims": c["n_sims"],
                              "per_sim_seconds": per_sim[key], "seconds": secs})
            total += secs
        total += contrast_seconds * fam["comparisons"]
    return {"cost_pilot_seconds": pilot_seconds, "per_sim_seconds": per_sim,
            "contrast_seconds_each": contrast_seconds,
            "contrasts": sum(f["comparisons"] for f in design["families"]),
            "calibration_contracts": len(contracts), "contracts": contracts,
            "projected_cpu_seconds": total}


def run_development(design: dict, *, root: Path, run_id: str, gov, trace, GR, outbox, OB, CE, code_identity: dict,
                    load_values, budgets: dict, cap_seconds: float, cost_pilot_sims: int, contrast_seconds: float,
                    isolated=None, resume_under_new_code: bool = False) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "REPORT.json"
    report = {"schema": "df_utility_development_report.v1", "run_id": run_id, "design_sha256": design["design_sha256"],
              "code_identity": code_identity, "cap_seconds": cap_seconds, "cost_pilot": None, "projection": None,
              "families": {}, "stopped": None, "spent_cpu_seconds": None}
    # --- cost pilot: one governed calibration child per contract type, few simulations ------------
    first = design["families"][0]
    types = {}
    for c in first["calibration_contracts"]:
        types.setdefault(f"{c['operator']}__{c['hypothesis']}", c)
    pilot_root = root / "cost_pilot"
    pilot_plan = {**first["calibration_plan"], "n_sims": int(cost_pilot_sims)}
    pilot_cfg = family_cfg(design, first, root=pilot_root, run_id=f"{run_id}-cost-pilot", code_identity=code_identity,
                           values=load_values(first["unit"]), budgets=budgets, resume_under_new_code=resume_under_new_code)
    pilot_cfg.update({"plan": pilot_plan, "purpose": "UTILITY_COST_PILOT", "contrasts": False})
    try:
        pilot_receipt = R.run_calibrations_only(pilot_cfg, gov, trace, GR=GR, outbox=outbox,
                                                isolated=capped_isolated(root, cap_seconds, isolated, expected=lambda j: 0.0))
    except CapExhausted as e:
        report["stopped"] = str(e)
        report["spent_cpu_seconds"] = spent_cpu(root)
        campaign.write_once(report_path, report)
        return report
    per_sim = {}
    for key, entry in pilot_receipt["calibration"].items():
        if key == "campaign":
            continue
        cpu = float((entry.get("cost") or {}).get("cpu_seconds") or 0.0)
        if entry.get("outcome") != "COMPLETED" or cpu <= 0:
            report["stopped"] = f"COST_PILOT_FAILED: {key} {entry.get('outcome')}"
            report["cost_pilot"] = pilot_receipt["calibration"]
            campaign.write_once(report_path, report)
            return report
        per_sim[key] = cpu / cost_pilot_sims
    pilot_seconds = spent_cpu(pilot_root)
    projection = project(design, per_sim, contrast_seconds, pilot_seconds)
    report["cost_pilot"] = pilot_receipt["calibration"]
    report["projection"] = projection

    def expected(job):
        if job.get("kind") == "calibrate":
            return per_sim[job["protocol_key"]] * float((job.get("plan") or {}).get("n_sims") or 0)
        return float(contrast_seconds)
    isolated_capped = capped_isolated(root, cap_seconds, isolated, expected=expected)
    trace("projection", cpu_seconds=projection["projected_cpu_seconds"], cap=cap_seconds)
    if projection["projected_cpu_seconds"] > cap_seconds:
        feasible = []
        acc = pilot_seconds
        for fam in design["families"]:
            fam_cost = sum(per_sim[f"{c['operator']}__{c['hypothesis']}"] * c["n_sims"] for c in fam["calibration_contracts"]) \
                + contrast_seconds * fam["comparisons"]
            if acc + fam_cost <= cap_seconds:
                feasible.append(fam["unit"])
                acc += fam_cost
        plan = {"schema": "df_utility_development_plan.v1", "projection": projection, "cap_seconds": cap_seconds,
                "feasible_families_in_order": feasible, "verdict": "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED"}
        campaign.write_once(root / "PLAN.json", plan)
        report["stopped"] = "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED"
        report["spent_cpu_seconds"] = spent_cpu(root)
        campaign.write_once(report_path, report)
        return report
    # --- families, in the design's order -------------------------------------------------------------
    for fam in design["families"]:
        froot = root / "families" / fam["unit"]
        cfg = family_cfg(design, fam, root=froot, run_id=f"{run_id}-{fam['role'][:3]}-{fam['resource']['seed']}-{fam['resource']['regime']['family']}",
                         code_identity=code_identity, values=load_values(fam["unit"]), budgets=budgets,
                         resume_under_new_code=resume_under_new_code)
        try:
            receipt, outcomes, frozen, pre, _ = R.run_rehearsal(cfg, gov, trace, GR=GR, outbox=outbox,
                                                                isolated=isolated_capped)
            receipt["envelope"] = R.emit_envelope(cfg, outcomes, frozen, pre, OB, CE)
            if not (froot / "REPORT.json").is_file():
                campaign.write_once(froot / "REPORT.json", receipt)
            report["families"][fam["unit"]] = {"role": fam["role"], "replica_of": fam.get("replica_of"),
                                               "run_id": cfg["run_id"], "root": str(froot),
                                               "outcomes": {k: {"outcome": v["outcome"], "hypothesis": v.get("hypothesis"),
                                                                "delta_mean": (v.get("score") or {}).get("delta_mean")}
                                                            for k, v in outcomes.items()},
                                               "reconciliation": receipt["reconciliation"],
                                               "envelope": receipt["envelope"]}
        except CapExhausted as e:
            report["families"][fam["unit"]] = {"role": fam["role"], "root": str(froot), "incomplete": str(e)}
            report["stopped"] = str(e)
            break
    report["spent_cpu_seconds"] = spent_cpu(root)
    campaign.write_once(report_path, report)
    return report


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = R._load("outbox", REPO / "olap")
    CE = R._load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--cpu-cap-seconds", type=float, default=4 * 3600.0)
    parser.add_argument("--cost-pilot-sims", type=int, default=6)
    parser.add_argument("--contrast-seconds", type=float, default=2.0, help="projected CPU per contrast (pilot v2: ~1 s)")
    parser.add_argument("--task-memory", type=int, default=1 << 30)
    parser.add_argument("--wall-seconds", type=float, default=240.0)
    parser.add_argument("--cpu-seconds", type=int, default=240)
    parser.add_argument("--calibration-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--resume-under-new-code", action="store_true")
    args = parser.parse_args(argv)
    design = json.loads(args.design.read_text())
    pre = json.loads((args.pilot_root / "FREEZE.pre.json").read_text())
    D.validate_design(design, inherited_protocol=pre["protocol_base"], pilot_units=[u["unit"] for u in pre["units"]],
                      bank_root=args.bank)
    code_identity = GR.strict_code_identity(REPO)
    worker = _load("df_d3_unit_worker")

    def load_values(unit_id):
        u = worker.load_unit(args.bank / unit_id)
        if u["bank"] != "SYNTHETIC":
            raise SystemExit("REFUSED: development takes synthetic bank units only")
        return u["inputs"][0]["values"]
    budgets = {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds, "cpu_seconds": args.cpu_seconds,
               "calibration_wall_seconds": args.calibration_wall_seconds,
               "calibration_cpu_seconds": int(args.calibration_wall_seconds),
               "mechanics_wall_seconds": args.wall_seconds, "mechanics_cpu_seconds": args.cpu_seconds,
               "slow_control": {"slow_seconds": 0.0, "wall_seconds": 1.0}}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": R.now_iso(), **facts})
        print(json.dumps({"event": event, **{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:80])
                                              for k, v in facts.items()}}), flush=True)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    report = run_development(design, root=args.root, run_id=args.run_id, gov=gov, trace=trace, GR=GR, outbox=outbox,
                             OB=OB, CE=CE, code_identity=code_identity, load_values=load_values, budgets=budgets,
                             cap_seconds=args.cpu_cap_seconds, cost_pilot_sims=args.cost_pilot_sims,
                             contrast_seconds=args.contrast_seconds, resume_under_new_code=args.resume_under_new_code)
    (args.root / "TRACE.json").write_text(json.dumps(trace_log, indent=1, default=str))
    print(json.dumps({"stopped": report["stopped"], "spent_cpu_seconds": report["spent_cpu_seconds"],
                      "projection": (report.get("projection") or {}).get("projected_cpu_seconds"),
                      "families": {k: (v.get("incomplete") or {c: o["outcome"] for c, o in v.get("outcomes", {}).items()})
                                   for k, v in report["families"].items()}}, indent=1, default=str)[:6000])
    return 0 if report["stopped"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
