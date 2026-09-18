#!/usr/bin/env python3
"""Governed execution of the frozen adequacy design (S3): cost pilot, projection, bounded cells.

    design    the sealed DESIGN.json (write-once in the run root; a different one refuses)
    pilot     one governed child per model at the largest context and length with a small update
              ceiling: seconds per update and per cell measured, the whole factorial projected
    campaign  registered before any cell (units = cell ids); before_run per cell; each cell is an
              isolated child (memory/wall/CPU ceilings, RESOURCE_EXCEEDED kept, no partial score)
              under the aggregate CPU ceiling read from the attempts; terminal per cell with the
              child's instants, cost and the losses recomputed by the parent from the arrays;
              reconcile; DEVELOPMENT envelope
    stop      projection beyond the ceiling -> PLAN.json with the exact need, nothing launched;
              ceiling exhausted mid-way -> incompletes kept, the rest reported as such

Descriptive model/context adequacy only: no ADVANCES, no eligibility, no promotion.

    python tools/df_adequacy_run.py --design DESIGN.json --bank BANK --root ROOT --run-id ID \\
        --api-key-file KEY [--cpu-cap-seconds 7200] [--already-spent S]
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


D = _load("df_adequacy_design")
M = _load("df_adequacy_models")
R = _load("df_utility_run")
DEV = _load("df_utility_dev_run")
H = _load("df_utility_harness")
campaign = _load("df_d3_campaign")

SCORE_UNVERIFIED = "SCORE_UNVERIFIED"


def verified_cell(attempt_dir: Path, result: dict, verified: dict) -> tuple:
    """The cell record is the file the child named and the runner re-hashed; its losses are
    recomputed here from the arrays it saved (never trusted from the summary)."""
    name = (result or {}).get("output_file")
    path = Path(attempt_dir) / str(name)
    if not name or not path.is_file():
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"{name!r} absent"}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (verified or {}).get("output_sha256"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "the output's bytes are not the ones the child declared and the runner verified"}
    rec = json.loads(body)
    if rec.get("schema") != M.CELL_SCHEMA:
        return None, {"outcome": SCORE_UNVERIFIED, "why": f"schema {rec.get('schema')!r}"}
    arrays_path = Path(attempt_dir) / "arrays.npz"
    if not arrays_path.is_file() or hashlib.sha256(arrays_path.read_bytes()).hexdigest() != rec.get("arrays_sha256"):
        return None, {"outcome": SCORE_UNVERIFIED, "why": "arrays absent or altered"}
    arr = np.load(arrays_path)
    recomputed = {}
    for part in [x for x in ("train", "validation", "test") if x in rec["losses"]]:   # a pilot has no test (T2)
        y, p, b = arr[f"{part}_y"], arr[f"{part}_pred"], arr[f"{part}_baseline"]
        recomputed[part] = {"model": M.mae(p, y), "baseline": M.mae(b, y)}
        for k in ("model", "baseline"):
            if abs(recomputed[part][k] - rec["losses"][part][k]) > 1e-9:
                return None, {"outcome": SCORE_UNVERIFIED, "why": f"{part} {k} loss in the record is not the one recomputed from the arrays"}
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
    task = IR.Task(argv=[sys.executable, "-B", str(HERE / "df_adequacy_models.py"), "--worker", str(job_file)],
                   name=f"adequacy-{job.get('cell_id', 'c')}", attempt_dir=attempt_dir, assigned_bytes=assigned_bytes,
                   wall_seconds=wall_seconds, cpu_seconds=cpu_seconds, mechanism=IR.detect_mechanism())
    task.start()
    task.wait()
    status, reason, verified = IR.classify(task.outcome, attempt_dir)
    result = json.loads((attempt_dir / "result.json").read_text()) if (attempt_dir / "result.json").is_file() else None
    cost = {"cpu_seconds": task.outcome.get("cpu_seconds"), "wall_seconds": task.outcome.get("wall_seconds"),
            "peak_rss_bytes": task.outcome.get("child_maxrss_bytes"), "cgroup_memory_peak": task.outcome.get("cgroup_memory_peak"),
            "started_at": task.outcome.get("started_at"), "ended_at": task.outcome.get("ended_at")}
    if status != "COMPLETED":
        summary = {"outcome": H.RESOURCE_EXCEEDED if status == "RESOURCE_EXCEEDED" else "UNCERTAIN", "reason": reason,
                   "cost": cost, "score": None, "output_sha256": None}
    else:
        score, refusal = verified_cell(attempt_dir, result, verified)
        summary = {"outcome": (score["diagnosis"]["class"] if score else SCORE_UNVERIFIED), "reason": reason, "cost": cost,
                   "score": score, "output_sha256": verified.get("output_sha256"), **({"refusal": refusal} if refusal else {})}
    prior.write_text(json.dumps({"status": status, "verified": verified,
                                 "summary": {k: v for k, v in summary.items() if k != "score"}}, default=H._jsonable))
    return summary


def _job(design: dict, cell: dict, bank: Path, run_id: str, **extra) -> dict:
    return {"kind": "adequacy_cell", "cell_id": cell["cell_id"], "unit": cell["unit"], "task": cell["task"], "model": cell["model"],
            "window": cell["window"], "train_length": cell["train_length"], "seed": cell["seed"], "horizon": design["horizon"],
            "bank": str(bank), "training": design["training"], "design_sha256": design["design_sha256"], "run_id": run_id, **extra}


def _metrics(rec: dict) -> list:
    """Validation metrics always; test metrics only when the role scored the test (never for pilots)."""
    v = rec["losses"]["validation"]
    m = [R._metric("adequacy.mae_validation", v["model"], "mae"), R._metric("adequacy.mae_baseline_validation", v["baseline"], "mae"),
         R._metric("adequacy.rows_validation", v["rows"], "count"), R._metric("adequacy.updates", rec["training"]["updates"], "count")]
    if rec.get("skill_validation") is not None:
        m.append(R._metric("adequacy.skill_validation", rec["skill_validation"], "ratio"))
    t = rec["losses"].get("test")
    if t:
        m += [R._metric("adequacy.mae_test", t["model"], "mae"), R._metric("adequacy.mae_baseline_test", t["baseline"], "mae"),
              R._metric("adequacy.rows_test", t["rows"], "count")]
        if rec.get("skill_test") is not None:
            m.append(R._metric("adequacy.skill_test", rec["skill_test"], "ratio"))
        if t.get("oracle") is not None:
            m.append(R._metric("adequacy.mae_oracle_test", t["oracle"], "mae"))
    return m


def run_adequacy(design: dict, *, root: Path, run_id: str, bank: Path, gov, trace, GR, outbox, OB, CE, code_identity: dict,
                 budgets: dict, cap_seconds: float, already_spent: float, pilot_updates: int, isolated=None,
                 pilot_only: bool = False, units_filter: list | None = None) -> dict:
    isolated = isolated or run_isolated
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "DESIGN.json").is_file():
        if json.loads((root / "DESIGN.json").read_text())["design_sha256"] != design["design_sha256"]:
            raise R.Refusal("REFUSED: this root was frozen under another design")
    else:
        campaign.write_once(root / "DESIGN.json", design)
    trace("design-frozen", sha256=design["design_sha256"])
    cells = D.cells(design)
    report = {"schema": "df_adequacy_report.v1", "run_id": run_id, "design_sha256": design["design_sha256"], "code_identity": code_identity,
              "cap_seconds": cap_seconds, "already_spent_seconds": already_spent, "cost_pilot": {}, "projection": None,
              "campaign": None, "cells": {}, "terminals": [], "reconciliation": None, "stopped": None, "envelope": None,
              "live_checks_note": "reconciliation is a live data-gov query at run time; warehouse content checks are separate"}
    spent = lambda: already_spent + DEV.spent_cpu(root)

    def cap_ok(need):
        return spent() + need <= cap_seconds
    # --- cost pilot: one governed child per model at the largest cell with a small update ceiling ---------
    pilot_key = f"{run_id}-adequacy-cost-pilot"
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

    def child(sha, key, unit_id, job, wall, cpu, tags):
        attempt = root / "attempts" / unit_id
        resumed = (attempt / "outcome.json").is_file()
        if not resumed:
            if not cap_ok(0.0):
                raise DEV.CapExhausted(f"{DEV.CAP_EXHAUSTED}: spent {spent():.0f} s of {cap_seconds:.0f} s before {unit_id}")
            GR._require_reconciled(gov, sha, unit_id, before_run=True)
            trace("before_run", key=key, unit=unit_id)
        trace("child", kind="adequacy_cell", name=unit_id)
        out = isolated(job, attempt_dir=attempt, assigned_bytes=budgets["task_memory_bytes"], wall_seconds=wall, cpu_seconds=cpu)
        trace("child-done", kind="adequacy_cell", name=unit_id, outcome=out.get("outcome"))
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
                               tags={"purpose": design["purpose"], "grants": "NONE", "classification": "NON_GOVERNING",
                                     "outcome": str(out["outcome"]), "design_sha256": design["design_sha256"],
                                     "output_sha256": out.get("output_sha256") or "", **tags})
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        report["terminals"].append({"unit_id": unit_id, "status": status_t, "outcome": out["outcome"], "cost": cost,
                                    "pending_after_flush": flushed["pending"]})
        return out

    # cost pilots at a representative SHORT and LONG context, no test access (T2), overhead apart (T3)
    short_w, long_w = min(design["contexts"]), max(design["contexts"])
    pilot_cells = [{"cell_id": f"pilot__{model}__W{w}", "unit": design["units"][0], "task": "observed_increment", "model": model,
                    "window": w, "train_length": max(design["train_lengths"]), "seed": design["seeds"][0]}
                   for model in D.factorial_models(design) for w in (short_w, long_w)]
    pilot_sha = register(pilot_key, [c["cell_id"] for c in pilot_cells])
    measured = {}
    try:
        for c in pilot_cells:
            job = _job(design, c, bank, run_id, max_updates_override=int(pilot_updates), role="COST_PILOT")
            out = child(pilot_sha, pilot_key, c["cell_id"], job, budgets["wall_seconds"], budgets["cpu_seconds"],
                        {"role": "COST_PILOT", "model": c["model"], "window": str(c["window"])})
            rec = out.get("score")
            if rec is None:
                report["cost_pilot"][c["cell_id"]] = {"outcome": out["outcome"], "cost": out["cost"]}
                report.update(stopped=f"COST_PILOT_FAILED: {c['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
                campaign.write_once(root / "REPORT.json", report)
                return report
            if "test" in rec.get("losses", {}) or rec.get("exposure") != "NO_TEST_ACCESS":
                raise R.Refusal("REFUSED: a cost pilot scored the test")
            cpu = float(out["cost"].get("cpu_seconds") or rec["cost"]["cpu_seconds"])
            fit_s = float(rec["cost"].get("fit_seconds") or 0.0)
            updates = max(1, int(rec["training"]["updates"]))
            measured[(c["model"], c["window"])] = {"cpu_seconds": cpu, "fit_seconds": fit_s, "overhead_seconds": max(0.0, cpu - fit_s),
                                                    "updates": updates, "seconds_per_update": (fit_s / updates) if c["model"] != "ridge" else 0.0,
                                                    "diagnosis": rec["diagnosis"]["class"], "exposure": rec["exposure"]}
            report["cost_pilot"][c["cell_id"]] = measured[(c["model"], c["window"])]
    except DEV.CapExhausted as e:
        report.update(stopped=str(e), spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.json", report)
        return report
    _, rb = gov.reconcile_campaign(pilot_sha)
    report["cost_pilot"]["reconciliation"] = rb
    # --- projection: per-update cost interpolated between the short and long pilots, overhead per child,
    # every NN cell at its full update allowance (early stopping only lowers it), + 25 % headroom -------
    max_updates = int(design["training"]["max_updates"])
    batch = int(design["training"]["batch"])
    max_epochs = int(design["training"]["max_epochs"])
    headroom = float((design.get("budget") or {}).get("headroom", 0.25))
    projected = {}
    for c in cells:
        m = c["model"]
        ms, ml = measured[(m, short_w)], measured[(m, long_w)]
        overhead = max(ms["overhead_seconds"], ml["overhead_seconds"])
        if m == "ridge":
            projected[c["cell_id"]] = max(ms["cpu_seconds"], ml["cpu_seconds"])
        else:
            frac = (c["window"] - short_w) / max(1, (long_w - short_w))
            per_update = ms["seconds_per_update"] + frac * (ml["seconds_per_update"] - ms["seconds_per_update"])
            steps = -(-c["train_length"] // batch)
            updates = min(max_updates, steps * max_epochs)
            projected[c["cell_id"]] = per_update * updates + overhead
    remaining = [c for c in cells if not (root / "attempts" / c["cell_id"] / "outcome.json").is_file()]
    total_remaining = sum(projected[c["cell_id"]] for c in remaining)
    with_headroom = total_remaining * (1.0 + headroom)
    report["projection"] = {"per_cell": projected, "cells": len(cells), "remaining_cells": len(remaining),
                            "projected_remaining_cpu_seconds": total_remaining, "headroom": headroom,
                            "projected_with_headroom": with_headroom, "spent_so_far": spent(), "cap_seconds": cap_seconds,
                            "fits": spent() + with_headroom <= cap_seconds,
                            "assumption": "per-update cost interpolated linearly in W between the short and long pilots; updates per cell = "
                                          "min(max_updates, steps_per_epoch x max_epochs) (early stopping only lowers it); overhead per child "
                                          "from the pilots; a projection, not an exact need"}
    trace("projection", cpu_seconds=with_headroom, cap=cap_seconds, spent=spent())
    if spent() + with_headroom > cap_seconds:
        campaign.write_once(root / "PLAN.json", {"schema": "df_adequacy_plan.v2", "verdict": "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED",
                                                 "projection": report["projection"], "cap_seconds": cap_seconds,
                                                 "measured_costs": {f"{k[0]}__W{k[1]}": v for k, v in measured.items()}})
        report.update(stopped="PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.json", report)
        return report
    if pilot_only:
        report.update(stopped="PILOT_ONLY", spent_cpu_seconds=spent())
        campaign.write_once(root / "REPORT.pilot.json", report)
        return report
    if units_filter:
        cells = [c for c in cells if c["unit"] in units_filter]
    # --- the factorial, governed --------------------------------------------------------------------------
    key = f"{run_id}-adequacy-cells" + (f"-{units_filter[0].split('__')[-1]}" if units_filter and len(units_filter) == 1 else "")
    sha = register(key, [c["cell_id"] for c in cells])
    report["campaign"] = {"key": key, "campaign_sha256": sha}
    try:
        for c in cells:
            need = projected[c["cell_id"]] * (1.0 + headroom)
            if not (root / "attempts" / c["cell_id"] / "outcome.json").is_file() and not cap_ok(need):
                raise DEV.CapExhausted(f"{DEV.CAP_EXHAUSTED}: spent {spent():.0f} s + next cell up to {need:.0f} s exceeds {cap_seconds:.0f} s")
            job = _job(design, c, bank, run_id, role="CELL")
            out = child(sha, key, c["cell_id"], job, budgets["wall_seconds"], budgets["cpu_seconds"],
                        {"role": "CELL", "unit": c["unit"], "task": c["task"], "model": c["model"], "window": str(c["window"]),
                         "train_length": str(c["train_length"]), "seed": str(c["seed"])})
            rec = out.get("score")
            report["cells"][c["cell_id"]] = {"outcome": out["outcome"], "cost": out["cost"], "resumed": out.get("resumed", False),
                                             **({"skill_test": rec["skill_test"], "losses_test": rec["losses"]["test"], "diagnosis": rec["diagnosis"],
                                                 "block_skill_test": rec.get("block_skill_test"),
                                                 "updates": rec["training"]["updates"], "receptive_field": rec["graph"]["receptive_field"],
                                                 "consumed_span_over_P": rec["consumed_span_over_P"]} if rec else {})}
    except DEV.CapExhausted as e:
        report["stopped"] = str(e)
    rstatus, rbody = gov.reconcile_campaign(sha)
    report["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                "accounting_only": rbody.get("accounting_only"), "lake_only": rbody.get("lake_only")}
    report["spent_cpu_seconds"] = spent()
    done = {k: v for k, v in report["cells"].items()}
    if done:
        outcomes = {k: {"outcome": v["outcome"], "cost": v["cost"], "operator": v.get("diagnosis", {}).get("class", "n/a"),
                        "score": {"delta_mean": v.get("skill_test"),
                                  "delta_lower": min([b for b in (v.get("block_skill_test") or []) if b is not None] or [v.get("skill_test")])}
                        if v.get("skill_test") is not None else None}
                    for k, v in done.items()}
        frozen = {"freeze_sha256": design["design_sha256"]}
        pre = {"protocol_base_sha256": design["design_sha256"],
               "units": [{"unit": u, "variable": "v0", "data_sha256": design["unit_metadata"][u]["digests"]["observed"]} for u in design["units"]]}
        cfg = {"purpose": design["purpose"], "run_id": run_id, "code_identity": code_identity, "eligibility_state": "SYNTHETIC_DEVELOPMENT",
               "exposure": "DEVELOPMENT_ADEQUACY_NO_RESERVE", "operators": []}
        try:
            report["envelope"] = R.emit_envelope(cfg, outcomes, frozen, pre, OB, CE)
        except Exception as e:  # noqa: BLE001 — the envelope is secondary evidence; its failure is recorded, never hidden
            report["envelope"] = {"error": str(e)[:300]}
    campaign.write_once(root / ("REPORT.json" if not units_filter else f"REPORT.{units_filter[0].split('__')[-1]}.json"), report)
    return report


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    CE = _load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--cpu-cap-seconds", type=float, default=7200.0)
    parser.add_argument("--already-spent", type=float, default=0.0)
    parser.add_argument("--pilot-updates", type=int, default=200)
    parser.add_argument("--task-memory", type=int, default=2 << 30)
    parser.add_argument("--wall-seconds", type=float, default=900.0)
    parser.add_argument("--cpu-seconds", type=int, default=900)
    parser.add_argument("--pilot-only", action="store_true", help="measure and project, launch nothing")
    parser.add_argument("--units", nargs="+", default=None, help="run only these units' cells (parallel processes share the root ledger)")
    args = parser.parse_args(argv)
    design = json.loads(args.design.read_text())
    if design.get("schema") != D.DESIGN_SCHEMA or D.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design is not a sealed adequacy design")
    code_identity = GR.strict_code_identity(REPO)
    budgets = {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds, "cpu_seconds": args.cpu_seconds}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": R.now_iso(), **facts})
        print(json.dumps({"event": event, **{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:80]) for k, v in facts.items()}}), flush=True)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    report = run_adequacy(design, root=args.root, run_id=args.run_id, bank=args.bank, gov=gov, trace=trace, GR=GR, outbox=outbox,
                          OB=OB, CE=CE, code_identity=code_identity, budgets=budgets, cap_seconds=args.cpu_cap_seconds,
                          already_spent=args.already_spent, pilot_updates=args.pilot_updates, pilot_only=args.pilot_only,
                          units_filter=args.units)
    (args.root / "TRACE.json").write_text(json.dumps(trace_log, indent=1, default=str))
    print(json.dumps({"stopped": report["stopped"], "spent_cpu_seconds": report.get("spent_cpu_seconds"),
                      "projection": {k: v for k, v in (report.get("projection") or {}).items() if k != "per_cell"},
                      "cost_pilot": {k: v for k, v in report["cost_pilot"].items() if k != "reconciliation"},
                      "cells_done": len(report["cells"]), "reconciliation": report["reconciliation"]}, indent=1, default=str))
    return 0 if report["stopped"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
