#!/usr/bin/env python3
"""Governed execution of MOD-E0-DEV (RP4): cost pilot, projection with headroom, then every cell
of the sealed design as an isolated child under the aggregate CPU ceiling; H3 arms run only after
their replicate's extractor cell completed and verified.

    design     DESIGN.json frozen write-once in the run root (another one refuses)
    pilot      one governed child per hypothesis/arm type at the sealed size with a small update
               ceiling and NO test access; seconds per update and per child measured; overhead apart
    projection every cell at its full update allowance (early stopping only lowers it) + 25 %
               headroom must fit the remaining ceiling; otherwise PLAN.json and nothing launched
    cells      campaign registered before any cell (units = cell ids); before_run per cell; child
               under memory/wall/CPU ceilings; terminal per cell with the child's instants, cost and
               metrics (MASE/MAE validation, MASE test descriptive, updates, parameters) with
               MEDIDO / NO_APLICA states; reconcile; DEVELOPMENT envelope
    failure    RESOURCE_EXCEEDED kept with cost, no partial score; a missing extractor makes its
               arms INCONCLUSIVE, never re-run silently

    python tools/df_mod_e0_run.py --design DESIGN.json --root ROOT --run-id ID --api-key-file KEY \\
        [--cpu-cap-seconds 14400] [--pilot-only] [--already-spent S]
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


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
R = _load("df_utility_run")
DEV = _load("df_utility_dev_run")
H = _load("df_utility_harness")
campaign = _load("df_d3_campaign")

SCORE_UNVERIFIED = "SCORE_UNVERIFIED"


def verified_cell(attempt_dir: Path, result: dict, verified: dict) -> tuple:
    name = (result or {}).get("output_file")
    path = Path(attempt_dir) / str(name)
    if not name or not path.is_file():
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"{name!r} absent"}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (verified or {}).get("output_sha256"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the output's bytes are not the ones the child declared and the runner verified"}
    rec = json.loads(body)
    if rec.get("schema") != E.CELL_SCHEMA:
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"schema {rec.get('schema')!r}"}
    arrays = Path(attempt_dir) / "arrays.npz"
    if not arrays.is_file() or hashlib.sha256(arrays.read_bytes()).hexdigest() != rec.get("arrays_sha256"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "arrays absent or altered"}
    arr = np.load(arrays)
    denom = arr["denominator"].tolist()
    for part in [q for q in ("train", "validation", "test") if f"{q}_y" in arr.files]:
        rec_m = E.mase(arr[f"{part}_pred"], arr[f"{part}_y"], denom)["mase_mean"]
        a, b = rec_m, rec["scores"][part]["model"]["mase_mean"]
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
            return None, {"outcome": SCORE_UNVERIFIED, "why": f"{part} MASE in the record is not the one recomputed from the arrays"}
    return rec, None


def run_isolated(job: dict, *, attempt_dir: Path, assigned_bytes: int, wall_seconds: float, cpu_seconds: float) -> dict:
    IR = _load("df_isolated_runner")
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    prior = attempt_dir / "outcome.json"
    if prior.is_file():
        recorded = json.loads(prior.read_text())
        refusal = H._job_binding_refusal(attempt_dir, job)
        score = None
        if refusal is None and recorded.get("status") == "COMPLETED":
            result = json.loads((attempt_dir / "result.json").read_text()) if (attempt_dir / "result.json").is_file() else None
            score, refusal = verified_cell(attempt_dir, result, recorded.get("verified"))
        history = dict(recorded.get("summary") or {})
        if refusal is not None:
            return {"outcome": SCORE_UNVERIFIED, "reason": refusal["why"], "cost": history.get("cost", {}), "score": None,
                    "output_sha256": None, "resumed": True, "refusal": refusal, "history": history}
        return {**history, "score": score, "resumed": True}
    job_file = attempt_dir / "job.json"
    job_file.write_text(json.dumps({**job, "attempt_dir": str(attempt_dir)}, default=H._jsonable))
    task = IR.Task(argv=[sys.executable, "-B", str(HERE / "df_mod_e0.py"), "--worker", str(job_file)],
                   name=f"mod-e0-{job.get('cell_id', 'c')}", attempt_dir=attempt_dir, assigned_bytes=assigned_bytes,
                   wall_seconds=wall_seconds, cpu_seconds=cpu_seconds, mechanism=IR.detect_mechanism())
    task.start()
    task.wait()
    status, reason, verified = IR.classify(task.outcome, attempt_dir)
    result = json.loads((attempt_dir / "result.json").read_text()) if (attempt_dir / "result.json").is_file() else None
    cost = {"cpu_seconds": task.outcome.get("cpu_seconds"), "wall_seconds": task.outcome.get("wall_seconds"),
            "peak_rss_bytes": task.outcome.get("child_maxrss_bytes"), "cgroup_memory_peak": task.outcome.get("cgroup_memory_peak"),
            "started_at": task.outcome.get("started_at"), "ended_at": task.outcome.get("ended_at")}
    if status != "COMPLETED":
        summary = {"outcome": H.RESOURCE_EXCEEDED if status == "RESOURCE_EXCEEDED" else "UNCERTAIN", "reason": reason, "cost": cost,
                   "score": None, "output_sha256": None}
    else:
        score, refusal = verified_cell(attempt_dir, result, verified)
        summary = {"outcome": "COMPLETED" if score else SCORE_UNVERIFIED, "reason": reason, "cost": cost, "score": score,
                   "output_sha256": verified.get("output_sha256"), **({"refusal": refusal} if refusal else {})}
    prior.write_text(json.dumps({"status": status, "verified": verified, "summary": {k: v for k, v in summary.items() if k != "score"}},
                                default=H._jsonable))
    return summary


def _metric(name, value, unit, status="MEDIDO"):
    m = R._metric(name, value if value is not None else 0.0, unit)
    m["status"] = status if value is not None else "NO_APLICA"
    if value is None:
        m["value"] = None
    return m


def _metrics(rec: dict) -> list:
    v = rec["scores"]["validation"]
    m = [_metric("mod_e0.mase_validation", v["model"]["mase_mean"], "mase"), _metric("mod_e0.mae_validation", v["model"]["mae_mean"], "mae"),
         _metric("mod_e0.naive_mase_validation", v["naive"]["mase_mean"], "mase"), _metric("mod_e0.oracle_mase_validation", v["oracle"]["mase_mean"], "mase"),
         _metric("mod_e0.linear_mase_validation", v["linear_window"]["mase_mean"], "mase"),
         _metric("mod_e0.updates", rec["training"]["updates"], "count"), _metric("mod_e0.params_trainable", rec["parameters"]["trainable"], "count"),
         _metric("mod_e0.profiles_ari", rec["profiles"]["ari_vs_latent"], "ratio")]
    if "test" in rec["scores"]:
        t = rec["scores"]["test"]
        m += [_metric("mod_e0.mase_test", t["model"]["mase_mean"], "mase"), _metric("mod_e0.mae_test", t["model"]["mae_mean"], "mae")]
    return [x for x in m if x["value"] is not None]


def run_mod_e0(design: dict, *, root: Path, run_id: str, gov, trace, GR, outbox, OB, CE, code_identity: dict, budgets: dict,
               cap_seconds: float, already_spent: float, pilot_updates: int, isolated=None, pilot_only: bool = False) -> dict:
    isolated = isolated or run_isolated
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DESIGN.json").is_file():
        if json.loads((root / "DESIGN.json").read_text())["design_sha256"] != design["design_sha256"]:
            raise R.Refusal("REFUSED: this root was frozen under another design")
    else:
        campaign.write_once(root / "DESIGN.json", design)
    trace("design-frozen", sha256=design["design_sha256"])
    report = {"schema": "df_mod_e0_report.v1", "run_id": run_id, "design_sha256": design["design_sha256"], "code_identity": code_identity,
              "cap_seconds": cap_seconds, "already_spent_seconds": already_spent, "cost_pilot": {}, "projection": None, "campaign": None,
              "cells": {}, "terminals": [], "reconciliation": None, "stopped": None, "envelope": None,
              "live_checks_note": "reconciliation is a live data-gov query at run time; warehouse content checks are separate"}
    spent = lambda: already_spent + DEV.spent_cpu(root)
    registrations_path = root / "CAMPAIGNS.json"
    registrations = json.loads(registrations_path.read_text()) if registrations_path.is_file() else {}

    def register(key, units):
        if key in registrations:
            trace("register", key=key, http="resumed")
            return registrations[key]["campaign_sha256"]
        status, reg = gov.submit_campaign({"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                                           "project": "predictor", "code_identity": code_identity, "config_sha256": design["design_sha256"],
                                           "input_mode": "SYNTHETIC", "synthetic_spec_sha256": design["design_sha256"], "units": units,
                                           "datasets": [], "terminal_lake": "olap_cube"})
        trace("register", key=key, http=status)
        if status not in (200, 201):
            raise R.Refusal(f"REFUSED: campaign {key} refused: http {status}; no child was started")
        registrations[key] = {"campaign_sha256": reg["campaign_sha256"], "http": status, "at": R.now_iso()}
        registrations_path.write_text(json.dumps(registrations, indent=1))
        return reg["campaign_sha256"]

    def child(sha, key, unit_id, job, tags, need=0.0):
        attempt = root / "attempts" / unit_id
        resumed = (attempt / "outcome.json").is_file()
        if not resumed:
            if spent() + need > cap_seconds:
                raise DEV.CapExhausted(f"{DEV.CAP_EXHAUSTED}: spent {spent():.0f} s + next child up to {need:.0f} s exceeds {cap_seconds:.0f} s")
            GR._require_reconciled(gov, sha, unit_id, before_run=True)
            trace("before_run", key=key, unit=unit_id)
        trace("child", kind="mod_e0_cell", name=unit_id)
        out = isolated(job, attempt_dir=attempt, assigned_bytes=budgets["task_memory_bytes"], wall_seconds=budgets["wall_seconds"],
                       cpu_seconds=budgets["cpu_seconds"])
        trace("child-done", kind="mod_e0_cell", name=unit_id, outcome=out.get("outcome"))
        if resumed:
            report["terminals"].append({"unit_id": unit_id, "status": "RESUMED", "outcome": out["outcome"], "cost": out["cost"], "resumed": True})
            return out
        rec = out.get("score")
        cost = out["cost"]
        if out["outcome"] == H.RESOURCE_EXCEEDED:
            status_t, reason = "FAILED", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        elif rec is not None:
            status_t, reason = "COMPLETED", None
        else:
            status_t, reason = "INCONCLUSIVE", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        terminal = R._terminal(status=status_t, reason=reason, cost=cost, metrics=_metrics(rec) if rec else [],
                               started=cost.get("started_at") or R.now_iso(), finished=cost.get("ended_at") or R.now_iso(),
                               tags={"purpose": "MOD_E0_DEV", "proposal": "P-MOD", "grants": "NONE", "classification": "NON_GOVERNING",
                                     "outcome": str(out["outcome"]), "design_sha256": design["design_sha256"],
                                     "output_sha256": out.get("output_sha256") or "", "phase": "DEVELOPMENT", **tags})
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        report["terminals"].append({"unit_id": unit_id, "status": status_t, "outcome": out["outcome"], "cost": cost, "pending_after_flush": flushed["pending"]})
        return out

    def job_for(cell, **extra):
        return {"kind": "mod_e0_cell", "cell_id": cell["cell_id"], "hypothesis": cell["hypothesis"], "level": cell["level"], "r": cell["r"],
                "seed": cell["seed"], "arm": cell["arm"], "window": design["window"], "training": design["training"],
                "design_sha256": design["design_sha256"], "run_id": run_id, **extra}
    # --- cost pilots: one per arm type at the sealed size, no test access, small update ceiling ----------
    pilot_key = f"{run_id}-mod-e0-cost-pilot"
    pilots = [{"cell_id": "pilot__H2_profiles", "hypothesis": "H2", "level": 3, "r": 1, "seed": 1, "arm": "profiles"},
              {"cell_id": "pilot__H3_extractor", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "extractor"}]
    pilot_sha = register(pilot_key, [c["cell_id"] for c in pilots])
    measured = {}
    try:
        for c in pilots:
            out = child(pilot_sha, pilot_key, c["cell_id"], job_for(c, max_updates_override=int(pilot_updates), role="COST_PILOT"),
                        {"role": "COST_PILOT", "hypothesis": c["hypothesis"], "arm": c["arm"]})
            rec = out.get("score")
            if rec is None:
                report["cost_pilot"][c["cell_id"]] = {"outcome": out["outcome"], "cost": out["cost"]}
                report.update(stopped=f"COST_PILOT_FAILED: {c['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
                campaign.write_once(root / "REPORT.json", report)
                return report
            if "test" in rec["scores"] or rec.get("exposure") != "NO_TEST_ACCESS":
                raise R.Refusal("REFUSED: a cost pilot scored the test")
            cpu = float(out["cost"].get("cpu_seconds") or rec["cost"]["cpu_seconds"])
            fit_s = float(rec["cost"].get("fit_seconds") or 0.0)
            updates = max(1, int(rec["training"]["updates"]))
            measured[c["cell_id"]] = {"cpu_seconds": cpu, "fit_seconds": fit_s, "overhead_seconds": max(0.0, cpu - fit_s), "updates": updates,
                                      "seconds_per_update": fit_s / updates, "exposure": rec["exposure"]}
            report["cost_pilot"][c["cell_id"]] = measured[c["cell_id"]]
        # an H3 arm child (frozen extractor) at the pilot's size, chained to the pilot extractor
        arm = {"cell_id": "pilot__H3_sequence", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "sequence"}
        register(pilot_key + "-arm", [arm["cell_id"]])
        arm_sha = registrations[pilot_key + "-arm"]["campaign_sha256"]
        out = child(arm_sha, pilot_key + "-arm", arm["cell_id"], job_for(arm, max_updates_override=int(pilot_updates), role="COST_PILOT",
                    extractor_weights=str(root / "attempts" / "pilot__H3_extractor" / "weights.weights.h5")),
                    {"role": "COST_PILOT", "hypothesis": "H3", "arm": "sequence"})
        rec = out.get("score")
        if rec is None:
            report.update(stopped=f"COST_PILOT_FAILED: {arm['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
            campaign.write_once(root / "REPORT.json", report)
            return report
        cpu = float(out["cost"].get("cpu_seconds") or rec["cost"]["cpu_seconds"])
        fit_s = float(rec["cost"].get("fit_seconds") or 0.0)
        updates = max(1, int(rec["training"]["updates"]))
        measured[arm["cell_id"]] = {"cpu_seconds": cpu, "fit_seconds": fit_s, "overhead_seconds": max(0.0, cpu - fit_s), "updates": updates,
                                    "seconds_per_update": fit_s / updates, "exposure": rec["exposure"]}
        report["cost_pilot"][arm["cell_id"]] = measured[arm["cell_id"]]
    except DEV.CapExhausted as e:
        report.update(stopped=str(e), spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.json", report)
        return report
    # --- projection ----------------------------------------------------------------------------------------
    batch = int(design["training"]["batch"])
    max_updates = int(design["training"]["max_updates"])
    max_epochs = int(design["training"]["max_epochs"])
    rows_train = E.boundaries(design["n_total"], design["window"], design["horizon"])["train"]
    steps = -(-(rows_train[1] - rows_train[0]) // batch)
    updates_full = min(max_updates, steps * max_epochs)
    headroom = float(design["budget"]["headroom"])
    per_cell = {}
    for c in design["cells"]:
        key = "pilot__H3_sequence" if (c["hypothesis"] == "H3" and c["arm"] in ("sequence", "summary")) else \
              ("pilot__H3_extractor" if c["hypothesis"] == "H3" else "pilot__H2_profiles")
        m = measured[key]
        per_cell[c["cell_id"]] = m["seconds_per_update"] * updates_full + m["overhead_seconds"]
    remaining = [c for c in design["cells"] if not (root / "attempts" / c["cell_id"] / "outcome.json").is_file()]
    total_remaining = sum(per_cell[c["cell_id"]] for c in remaining)
    with_headroom = total_remaining * (1 + headroom)
    report["projection"] = {"per_cell": per_cell, "cells": len(design["cells"]), "remaining_cells": len(remaining),
                            "projected_remaining_cpu_seconds": total_remaining, "headroom": headroom, "projected_with_headroom": with_headroom,
                            "spent_so_far": spent(), "cap_seconds": cap_seconds, "fits": spent() + with_headroom <= cap_seconds,
                            "assumption": "every cell at its full update allowance (early stopping only lowers it); per-update cost from the "
                                          "pilot of its arm type; overhead per child from the pilots; a projection, not an exact need"}
    trace("projection", cpu_seconds=with_headroom, cap=cap_seconds, spent=spent())
    if spent() + with_headroom > cap_seconds:
        campaign.write_once(root / "PLAN.json", {"schema": "df_mod_e0_plan.v1", "verdict": "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED",
                                                 "projection": report["projection"], "cap_seconds": cap_seconds, "measured_costs": measured})
        report.update(stopped="PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.json", report)
        return report
    if pilot_only:
        report.update(stopped="PILOT_ONLY", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.pilot.json", report)
        return report
    # --- the cells, governed, in dependency order ------------------------------------------------------
    key = f"{run_id}-mod-e0-cells"
    sha = register(key, [c["cell_id"] for c in design["cells"]])
    report["campaign"] = {"key": key, "campaign_sha256": sha}
    done = {}
    try:
        for c in design["cells"]:
            extra = {"role": "CELL"}
            if c.get("depends_on"):
                dep = done.get(c["depends_on"])
                if dep is None or dep.get("score") is None:
                    report["cells"][c["cell_id"]] = {"outcome": "INCONCLUSIVE_DEPENDENCY", "depends_on": c["depends_on"],
                                                     "why": "its extractor cell did not complete and verify; not re-run silently"}
                    continue
                extra["extractor_weights"] = str(root / "attempts" / c["depends_on"] / "weights.weights.h5")
            out = child(sha, key, c["cell_id"], job_for(c, **extra),
                        {"role": "CELL", "hypothesis": c["hypothesis"], "arm": c["arm"], "level": str(c["level"]), "r": str(c["r"]),
                         "seed": str(c["seed"]), "condition": f"h{c['level']}_r{c['r']}", "replicate": str(c["seed"])},
                        need=per_cell[c["cell_id"]] * (1 + headroom))
            done[c["cell_id"]] = out
            rec = out.get("score")
            report["cells"][c["cell_id"]] = {"outcome": out["outcome"], "cost": out["cost"], "resumed": out.get("resumed", False),
                                             **({"mase_validation": rec["scores"]["validation"]["model"]["mase_mean"],
                                                 "mase_test": (rec["scores"].get("test") or {}).get("model", {}).get("mase_mean"),
                                                 "naive_validation": rec["scores"]["validation"]["naive"]["mase_mean"],
                                                 "linear_validation": rec["scores"]["validation"]["linear_window"]["mase_mean"],
                                                 "updates": rec["training"]["updates"], "stop_reason": rec["training"]["stop_reason"],
                                                 "profiles_ari": rec["profiles"]["ari_vs_latent"], "extractor_weight_change": rec.get("extractor_weight_change")}
                                                if rec else {})}
    except DEV.CapExhausted as e:
        report["stopped"] = str(e)
    rstatus, rbody = gov.reconcile_campaign(sha)
    report["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"), "accounting_only": rbody.get("accounting_only"),
                                "lake_only": rbody.get("lake_only")}
    report["spent_cpu_seconds"] = spent()
    if report["cells"]:
        outcomes = {k: {"outcome": v["outcome"], "cost": v.get("cost") or {}, "operator": "modular",
                        "score": {"delta_mean": v.get("mase_validation"), "delta_lower": v.get("mase_validation")} if v.get("mase_validation") is not None else None}
                    for k, v in report["cells"].items()}
        cfg = {"purpose": "MOD_E0_DEV", "run_id": run_id, "code_identity": code_identity, "eligibility_state": "SYNTHETIC_DEVELOPMENT",
               "exposure": "DEVELOPMENT_NO_RESERVE", "operators": []}
        pre = {"protocol_base_sha256": design["design_sha256"], "units": [{"unit": "mod-e0-generator", "variable": "x", "data_sha256": design["design_sha256"]}]}
        try:
            report["envelope"] = R.emit_envelope(cfg, outcomes, {"freeze_sha256": design["design_sha256"]}, pre, OB, CE)
        except Exception as e:  # noqa: BLE001
            report["envelope"] = {"error": str(e)[:300]}
    campaign.write_once(root / "REPORT.json", report)
    return report


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    CE = _load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--cpu-cap-seconds", type=float, default=14400.0)
    parser.add_argument("--already-spent", type=float, default=0.0)
    parser.add_argument("--pilot-updates", type=int, default=300)
    parser.add_argument("--task-memory", type=int, default=3 << 30)
    parser.add_argument("--wall-seconds", type=float, default=1200.0)
    parser.add_argument("--cpu-seconds", type=int, default=1200)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args(argv)
    design = json.loads(args.design.read_text())
    if design.get("schema") != D.DESIGN_SCHEMA or E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design is not a sealed MOD-E0-DEV design")
    code_identity = GR.strict_code_identity(REPO)
    budgets = {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds, "cpu_seconds": args.cpu_seconds}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": R.now_iso(), **facts})
        print(json.dumps({"event": event, **{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:80]) for k, v in facts.items()}}), flush=True)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    report = run_mod_e0(design, root=args.root, run_id=args.run_id, gov=gov, trace=trace, GR=GR, outbox=outbox, OB=OB, CE=CE,
                        code_identity=code_identity, budgets=budgets, cap_seconds=args.cpu_cap_seconds, already_spent=args.already_spent,
                        pilot_updates=args.pilot_updates, pilot_only=args.pilot_only)
    (args.root / "TRACE.json").write_text(json.dumps(trace_log, indent=1, default=str))
    print(json.dumps({"stopped": report["stopped"], "spent_cpu_seconds": report.get("spent_cpu_seconds"),
                      "projection": {k: v for k, v in (report.get("projection") or {}).items() if k != "per_cell"},
                      "cost_pilot": report["cost_pilot"], "cells_done": len(report["cells"]), "reconciliation": report["reconciliation"]},
                     indent=1, default=str))
    return 0 if report["stopped"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
