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


_LOAD_LOCK = __import__("threading").RLock()


def _load(name: str, where: Path = HERE):
    """Thread-safe: concurrent children (RP14 --parallel) must never see a half-initialised module
    (a worker crashed with `df_isolated_runner has no attribute Task` under that race)."""
    with _LOAD_LOCK:                                      # the lock alone removes the race; the module must be registered
        if name in sys.modules:                           # BEFORE its body runs (dataclasses look themselves up in sys.modules)
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
AD = _load("df_mod_e0_arch_design")
R = _load("df_utility_run")
DEV = _load("df_utility_dev_run")
H = _load("df_utility_harness")
IR = _load("df_isolated_runner")
MET = _load("df_mod_e0_metrics")
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


METRIC_KEYS = {"metric", "split", "horizon", "unit", "value", "std_dev", "min_value", "max_value"}   # governed_terminal.v1 (data-gov)


def _metric(name, value, unit):
    """A governed metric carries exactly the schema's keys; a NO_APLICA value is not a metric row
    (the state travels in the terminal's tags), never a fabricated zero."""
    return R._metric(name, value, unit) if value is not None else None


def _metrics(rec: dict) -> tuple:
    """(metric rows, metric states) — states MEDIDO / NO_APLICA per contract, carried in tags."""
    v = rec["scores"]["validation"]
    wanted = [("mod_e0.mase_validation", v["model"]["mase_mean"], "mase"), ("mod_e0.mae_validation", v["model"]["mae_mean"], "mae"),
              ("mod_e0.naive_mase_validation", v["naive"]["mase_mean"], "mase"), ("mod_e0.oracle_mase_validation", v["oracle"]["mase_mean"], "mase"),
              ("mod_e0.linear_mase_validation", v["linear_window"]["mase_mean"], "mase"),
              ("mod_e0.updates", rec["training"]["updates"], "count"), ("mod_e0.params_trainable", rec["parameters"]["trainable"], "count"),
              ("mod_e0.profiles_ari", rec["profiles"]["ari_vs_latent"], "ratio")]
    if "test" in rec["scores"]:
        t = rec["scores"]["test"]
        wanted += [("mod_e0.mase_test", t["model"]["mase_mean"], "mase"), ("mod_e0.mae_test", t["model"]["mae_mean"], "mae")]
    rows, states = [], {}
    for name, value, unit in wanted:
        m = _metric(name, value, unit)
        if m is not None:
            states[name] = "MEDIDO"
            rows.append(m)
        else:
            # NO_MEDIDO when the score itself says so (non-finite / empty), NO_APLICA otherwise (zero denominators)
            source = {"mod_e0.mase_validation": v["model"], "mod_e0.mae_validation": v["model"], "mod_e0.naive_mase_validation": v["naive"],
                      "mod_e0.oracle_mase_validation": v["oracle"], "mod_e0.linear_mase_validation": v["linear_window"]}.get(name)
            if name.startswith("mod_e0.mase_test") or name.startswith("mod_e0.mae_test"):
                source = rec["scores"]["test"]["model"]
            states[name] = "NO_MEDIDO" if (source or {}).get("status") == E.NO_MEDIDO else "NO_APLICA"
    # RP13: the D/Y/M/G rows of the metrics contract (data per split, model per checkpoint)
    contract_rows, contract_states = MET.terminal_rows(rec)
    for name, value, unit, split in contract_rows:
        rows.append({**R._metric(name, value, unit), "split": split})
    states.update(contract_states)
    return rows, states


def corrected_terminal(terminal: dict) -> dict:
    """The successor of a terminal refused for its metric rows (RP5 repair): the same outcome,
    instants, costs, deliveries and tags; every metric row reduced to the schema's keys, rows
    without a finite value dropped and their state recorded in the tags."""
    rows, states = [], {}
    for m in terminal.get("metrics") or []:
        row = {k: m.get(k) for k in METRIC_KEYS}
        if isinstance(row.get("value"), (int, float)) and np.isfinite(row["value"]):
            rows.append(row)
            states[m["metric"]] = "MEDIDO"
        else:
            states[m["metric"]] = "NO_APLICA"
    fixed = {k: v for k, v in terminal.items() if k != "generation"}
    fixed["metrics"] = rows
    fixed["tags"] = {**(terminal.get("tags") or {}), "metric_states": json.dumps(states, sort_keys=True),
                     "repair": "RP5: metric rows reduced to the governed schema (a status key had been added)"}
    return fixed


def repair_refused_terminals(outbox, gov, GR, campaign_shas: set, reason: str) -> dict:
    """Supersede every pending terminal of the given campaigns that the server refused: the
    corrected successor is sent as generation 2 and the original disposed SUPERSEDED."""
    done, failed = [], []
    for item in list(outbox.status()["pending"]):
        if item.get("campaign_sha256") not in campaign_shas:
            continue
        original = json.loads((outbox.pending / item["file"]).read_text(encoding="ascii"))
        successor = corrected_terminal(original["terminal"])

        def sender(envelope):
            status, receipt = gov.report_terminal(envelope["campaign_sha256"], envelope["unit_id"], envelope["terminal"])
            if status not in (200, 201):
                raise GR.GovernedRunError(f"terminal refused: http {status} {receipt.get('error', '')}".strip())
            return receipt
        try:
            rec = outbox.supersede(item["file"], successor, sender, reason)
            done.append({"unit_id": item["unit_id"], "disposition": rec})
        except Exception as e:  # noqa: BLE001
            failed.append({"unit_id": item["unit_id"], "error": str(e)[:200]})
    return {"superseded": done, "failed": failed}


DELEGATED = "DELEGATED"


def _measured_cost(out: dict, rec: dict) -> dict:
    """Per-update cost NET of the descriptor checkpoints (RP13 instrumentation, a per-cell cost that
    does not scale with the allowance); the descriptor seconds join the per-child overhead."""
    cpu = float(out["cost"].get("cpu_seconds") or rec["cost"]["cpu_seconds"])
    fit_s = float(rec["cost"].get("fit_seconds") or 0.0)
    desc = float(((rec.get("metrics_cost_seconds") or {}).get("model_descriptors")) or 0.0)
    fit_net = max(0.0, fit_s - desc)
    updates = max(1, int(rec["training"]["updates"]))
    return {"cpu_seconds": cpu, "fit_seconds": fit_s, "descriptor_seconds": desc, "fit_seconds_net": fit_net,
            "overhead_seconds": max(0.0, cpu - fit_net), "updates": updates, "seconds_per_update": fit_net / updates, "exposure": rec["exposure"]}


def _cell_summary(out: dict) -> dict:
    rec = out.get("score")
    return {"outcome": out["outcome"], "cost": out.get("cost"), "resumed": out.get("resumed", False),
            **({"mase_validation": rec["scores"]["validation"]["model"]["mase_mean"],
                "mase_test": (rec["scores"].get("test") or {}).get("model", {}).get("mase_mean"),
                "naive_validation": rec["scores"]["validation"]["naive"]["mase_mean"],
                "linear_validation": rec["scores"]["validation"]["linear_window"]["mase_mean"],
                "oracle_validation": rec["scores"]["validation"]["oracle"]["mase_mean"],
                "updates": rec["training"]["updates"], "stop_reason": rec["training"]["stop_reason"],
                "profiles_ari": rec["profiles"]["ari_vs_latent"], "extractor_weight_change": rec.get("extractor_weight_change"),
                "arch": rec.get("arch"), "fusion": rec.get("fusion"), "parameters": rec.get("parameters")}
               if rec else {})}


def _run_waves(cells: list, done: dict, report: dict, run_one, parallel: int, lock, trace, dep_state=None) -> None:
    """Run `cells` respecting `depends_on`, up to `parallel` at a time; a cell whose dependency did
    not complete and verify is INCONCLUSIVE_DEPENDENCY (never re-run silently); the CPU ceiling
    raised by any child stops the dispatch after the running wave finishes."""
    import concurrent.futures
    pending = list(cells)
    ids = {c["cell_id"] for c in cells}
    while pending:
        ready, later = [], []
        for c in pending:
            dep = c.get("depends_on")
            if not dep:
                ready.append(c)
                continue
            state = done.get(dep)
            if state is None and dep_state is not None and dep not in ids:
                state = dep_state(dep)
            if state is None:
                later.append(c)
            elif state.get("score") is None and state.get("outcome") != "AVAILABLE":
                report["cells"][c["cell_id"]] = {"outcome": "INCONCLUSIVE_DEPENDENCY", "depends_on": dep,
                                                 "why": "its extractor cell did not complete and verify; not re-run silently"}
                done[c["cell_id"]] = {"score": None, "outcome": "INCONCLUSIVE_DEPENDENCY"}
            else:
                ready.append(c)
        if not ready:
            for c in later:
                report["cells"][c["cell_id"]] = {"outcome": "INCONCLUSIVE_DEPENDENCY", "depends_on": c.get("depends_on"),
                                                 "why": "its dependency is not available on this host"}
                done[c["cell_id"]] = {"score": None, "outcome": "INCONCLUSIVE_DEPENDENCY"}
            break
        stop = None
        if int(parallel) <= 1:
            for c in ready:
                try:
                    out = run_one(c)
                except DEV.CapExhausted as e:
                    stop = e
                    break
                with lock:
                    done[c["cell_id"]] = out
                    report["cells"][c["cell_id"]] = _cell_summary(out)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(parallel)) as pool:
                futures = {pool.submit(run_one, c): c for c in ready}
                for fut in concurrent.futures.as_completed(futures):
                    c = futures[fut]
                    try:
                        out = fut.result()
                    except DEV.CapExhausted as e:
                        stop = e
                        continue
                    with lock:
                        done[c["cell_id"]] = out
                        report["cells"][c["cell_id"]] = _cell_summary(out)
        if stop is not None:
            raise stop
        pending = list(later)                                  # every ready cell is now in `done`


def run_mod_e0(design: dict, *, root: Path, run_id: str, gov, trace, GR, outbox, OB, CE, code_identity: dict, budgets: dict,
               cap_seconds: float, already_spent: float, pilot_updates: int, isolated=None, pilot_only: bool = False,
               role: str | None = None, execute_only: bool = False, report_collected: bool = False, parallel: int = 1,
               pilot_costs_from: dict | None = None) -> dict:
    """`role`: this host's role (a v2 design deals cells to roles); `execute_only`: a worker runs its
    cells with no governance (the coordinator registered every unit before and reports the collected
    outcomes after); `report_collected`: the coordinator emits the terminals of delegated cells whose
    attempts were collected; `parallel`: concurrent children (dependencies respected)."""
    import threading
    isolated = isolated or run_isolated
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    v2 = design.get("schema") == AD.DESIGN_SCHEMA
    role = role or "COORDINATOR"
    lock = threading.RLock()
    if (root / "DESIGN.json").is_file():
        if json.loads((root / "DESIGN.json").read_text())["design_sha256"] != design["design_sha256"]:
            raise R.Refusal("REFUSED: this root was frozen under another design")
    else:
        campaign.write_once(root / "DESIGN.json", design)
    trace("design-frozen", sha256=design["design_sha256"])
    inherited_report = {}
    for inh in design.get("inherited") or []:
        src = Path(inh["inherited_from"]["root"]) / "attempts" / inh["inherited_from"]["cell_id"]
        dst = root / "attempts" / inh["cell_id"]
        oc = json.loads((src / "outcome.json").read_text()) if (src / "outcome.json").is_file() else {}
        if (oc.get("status") or oc.get("outcome")) != "COMPLETED":
            raise R.Refusal(f"REFUSED: inherited donor {inh['cell_id']} has no completed attempt at {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)                       # read-only inheritance: the parent's bytes, never copied or re-trained
        elif dst.resolve() != src.resolve():
            raise R.Refusal(f"REFUSED: {dst} is not the inherited donor's attempt")
        inherited_report[inh["cell_id"]] = {"outcome": "INHERITED", "from": inh["inherited_from"], "weights_sha256": hashlib.sha256((src / "weights.weights.h5").read_bytes()).hexdigest()}
        trace("inherited", cell=inh["cell_id"], run=inh["inherited_from"]["run_id"])
    report = {"schema": "df_mod_e0_report.v1", "run_id": run_id, "inherited": inherited_report, "design_sha256": design["design_sha256"], "code_identity": code_identity,
              "cap_seconds": cap_seconds, "already_spent_seconds": already_spent, "cost_pilot": {}, "projection": None, "campaign": None,
              "cells": {}, "terminals": [], "reconciliation": None, "stopped": None, "envelope": None, "role": role, "execute_only": execute_only,
              "report_collected": report_collected, "parallel": int(parallel),
              "live_checks_note": "reconciliation is a live data-gov query at run time; warehouse content checks are separate"}
    if execute_only:
        # a worker: its cells only, no governance; the coordinator registered every unit before and reports after collection
        prior = root / "REPORT.json"
        if prior.is_file() and json.loads(prior.read_text()).get("run_id") != run_id:
            raise R.Refusal("REFUSED: this root belongs to another run")
        mine = [c for c in design["cells"] if c.get("host_role", "COORDINATOR") == role]
        spent_w = lambda: already_spent + DEV.spent_cpu(root)
        done_w = {}

        def worker_child(c):
            extra = {"role": "CELL", **{k: c[k] for k in ("arch", "donor", "diagnostic", "depends_on") if k in c}}
            if c.get("depends_on"):
                extra["extractor_weights"] = str(root / "attempts" / c["depends_on"] / "weights.weights.h5")
            job = {"kind": "mod_e0_cell", "cell_id": c["cell_id"], "hypothesis": c["hypothesis"], "level": c["level"], "r": c["r"], "seed": c["seed"],
                   "arm": c["arm"], "window": design["window"], "training": design["training"], "design_sha256": design["design_sha256"], "run_id": run_id, **extra}
            attempt = root / "attempts" / c["cell_id"]
            with lock:
                if not (attempt / "outcome.json").is_file() and spent_w() > cap_seconds:
                    raise DEV.CapExhausted(f"{DEV.CAP_EXHAUSTED}: worker {role} spent {spent_w():.0f} s of {cap_seconds:.0f} s")
            trace("child", kind="mod_e0_cell", name=c["cell_id"], role=role)
            out = isolated(job, attempt_dir=attempt, assigned_bytes=budgets["task_memory_bytes"], wall_seconds=budgets["wall_seconds"], cpu_seconds=budgets["cpu_seconds"])
            trace("child-done", kind="mod_e0_cell", name=c["cell_id"], outcome=out.get("outcome"), role=role)
            return out
        try:
            _run_waves(mine, done_w, report, worker_child, parallel, lock, trace)
        except DEV.CapExhausted as e:
            report["stopped"] = str(e)
        report["spent_cpu_seconds"] = spent_w()
        target = root / f"REPORT.{role}.json"
        if target.exists():
            target = root / f"REPORT.{role}.{R.now_iso().replace(':', '')}.json"
        campaign.write_once(target, report)
        return report
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
        registrations[key] = {"campaign_sha256": reg["campaign_sha256"], "http": status, "at": R.now_iso(), "code_identity": code_identity}
        registrations_path.write_text(json.dumps(registrations, indent=1))
        return reg["campaign_sha256"]

    missing_units_ref = {"sha": None, "set": set()}

    def child(sha, key, unit_id, job, tags, need=0.0, emit_resumed=False):
        attempt = root / "attempts" / unit_id
        resumed = (attempt / "outcome.json").is_file()
        with lock:
            if not resumed:
                if spent() + need > cap_seconds:
                    raise DEV.CapExhausted(f"{DEV.CAP_EXHAUSTED}: spent {spent():.0f} s + next child up to {need:.0f} s exceeds {cap_seconds:.0f} s")
                GR._require_reconciled(gov, sha, unit_id, before_run=True)
                trace("before_run", key=key, unit=unit_id)
            trace("child", kind="mod_e0_cell", name=unit_id)
        out = isolated(job, attempt_dir=attempt, assigned_bytes=budgets["task_memory_bytes"], wall_seconds=budgets["wall_seconds"],
                       cpu_seconds=budgets["cpu_seconds"])
        with lock:
            trace("child-done", kind="mod_e0_cell", name=unit_id, outcome=out.get("outcome"))
            if resumed and not emit_resumed and not (unit_id in (missing_units_ref.get("set") or set()) and sha == missing_units_ref.get("sha")):
                report["terminals"].append({"unit_id": unit_id, "status": "RESUMED", "outcome": out["outcome"], "cost": out["cost"], "resumed": True})
                return out
            return _emit(sha, unit_id, job, tags, out)

    def _emit(sha, unit_id, job, tags, out):
        rec = out.get("score")
        cost = out["cost"]
        if out["outcome"] == H.RESOURCE_EXCEEDED:
            status_t, reason = "FAILED", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        elif rec is not None:
            status_t, reason = "COMPLETED", None
        else:
            status_t, reason = "INCONCLUSIVE", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        rows, states = _metrics(rec) if rec else ([], {})
        terminal = R._terminal(status=status_t, reason=reason, cost=cost, metrics=rows,
                               started=cost.get("started_at") or R.now_iso(), finished=cost.get("ended_at") or R.now_iso(),
                               tags={"purpose": "MOD_E0_DEV", "proposal": "P-MOD", "grants": "NONE", "classification": "NON_GOVERNING",
                                     "outcome": str(out["outcome"]), "design_sha256": design["design_sha256"],
                                     "output_sha256": out.get("output_sha256") or "", "phase": "DEVELOPMENT",
                                     "metric_states": json.dumps(states, sort_keys=True), **tags})
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        report["terminals"].append({"unit_id": unit_id, "status": status_t, "outcome": out["outcome"], "cost": cost, "pending_after_flush": flushed["pending"]})
        return out

    def job_for(cell, **extra):
        return {"kind": "mod_e0_cell", "cell_id": cell["cell_id"], "hypothesis": cell["hypothesis"], "level": cell["level"], "r": cell["r"],
                "seed": cell["seed"], "arm": cell["arm"], "window": design["window"], "training": design["training"],
                "design_sha256": design["design_sha256"], "run_id": run_id, **{k: cell[k] for k in ("arch", "donor", "diagnostic") if k in cell}, **extra}
    # --- cost pilots: one per arm type (per architecture in v2), no test access, small update ceiling ------
    pilot_key = f"{run_id}-mod-e0-cost-pilot"
    inherited = None
    if v2 and not design.get("pilots"):
        # a hypothesis-only successor (design.only_hypotheses): its parent's MEASURED pilot costs are inherited, named and checked
        if not pilot_costs_from or not pilot_costs_from.get("cost_pilot"):
            raise R.Refusal("REFUSED: this design has no pilots; pass the parent's report (--pilot-costs-from) with its measured costs")
        if pilot_costs_from.get("design_sha256") != design.get("successor_of"):
            raise R.Refusal("REFUSED: --pilot-costs-from is not the report of this design's parent")
        inherited = pilot_costs_from["cost_pilot"]
        report["cost_pilot"] = {"inherited_from_run": pilot_costs_from.get("run_id"), "parent_design_sha256": pilot_costs_from.get("design_sha256"),
                                "measured": inherited}
        trace("pilots-inherited", run=pilot_costs_from.get("run_id"), pilots=len(inherited))
    if v2:
        pilots = [c for c in design["pilots"] if c["campaign"] == "-mod-e0-cost-pilot"]
        arm_pilots = [c for c in design["pilots"] if c["campaign"] == "-mod-e0-cost-pilot-arm"]
    else:
        pilots = [{"cell_id": "pilot__H2_profiles", "hypothesis": "H2", "level": 3, "r": 1, "seed": 1, "arm": "profiles"},
                  {"cell_id": "pilot__H3_extractor", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "extractor"}]
        arm_pilots = [{"cell_id": "pilot__H3_sequence", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "sequence",
                       "depends_on": "pilot__H3_extractor"}]
    pilot_sha = register(pilot_key, [c["cell_id"] for c in pilots]) if pilots else None
    measured = dict(inherited or {})
    try:
        for c in pilots:
            out = child(pilot_sha, pilot_key, c["cell_id"], job_for(c, max_updates_override=int(pilot_updates), role="COST_PILOT"),
                        {"role": "COST_PILOT", "hypothesis": c["hypothesis"], "arm": c["arm"], **({"arch": c["arch"]} if "arch" in c else {})})
            rec = out.get("score")
            if rec is None:
                report["cost_pilot"][c["cell_id"]] = {"outcome": out["outcome"], "cost": out["cost"]}
                report.update(stopped=f"COST_PILOT_FAILED: {c['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
                campaign.write_once(root / "REPORT.json", report)
                return report
            if "test" in rec["scores"] or rec.get("exposure") != "NO_TEST_ACCESS":
                raise R.Refusal("REFUSED: a cost pilot scored the test")
            measured[c["cell_id"]] = _measured_cost(out, rec)
            report["cost_pilot"][c["cell_id"]] = measured[c["cell_id"]]
        # the H3 arm children (frozen extractor) at the pilot's size, chained to their pilot extractor
        if arm_pilots:
            register(pilot_key + "-arm", [c["cell_id"] for c in arm_pilots])
        arm_sha = registrations[pilot_key + "-arm"]["campaign_sha256"] if arm_pilots else None
        for arm in arm_pilots:
            out = child(arm_sha, pilot_key + "-arm", arm["cell_id"], job_for(arm, max_updates_override=int(pilot_updates), role="COST_PILOT",
                        extractor_weights=str(root / "attempts" / arm["depends_on"] / "weights.weights.h5"), depends_on=arm["depends_on"]),
                        {"role": "COST_PILOT", "hypothesis": "H3", "arm": arm["arm"], **({"arch": arm["arch"]} if "arch" in arm else {})})
            rec = out.get("score")
            if rec is None:
                report.update(stopped=f"COST_PILOT_FAILED: {arm['cell_id']} {out['outcome']}", spent_cpu_seconds=spent())
                campaign.write_once(root / "REPORT.json", report)
                return report
            measured[arm["cell_id"]] = _measured_cost(out, rec)
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
        if v2:
            key = AD.pilot_key_for(c)
        else:
            key = "pilot__H3_sequence" if (c["hypothesis"] == "H3" and c["arm"] in ("sequence", "summary")) else \
                  ("pilot__H3_extractor" if c["hypothesis"] == "H3" else "pilot__H2_profiles")
        m = measured[key]
        per_cell[c["cell_id"]] = m["seconds_per_update"] * updates_full + m["overhead_seconds"]
    remaining = [c for c in design["cells"] if not (root / "attempts" / c["cell_id"] / "outcome.json").is_file()]
    total_remaining = sum(per_cell[c["cell_id"]] for c in remaining)
    with_headroom = total_remaining * (1 + headroom)
    by_role = {}
    for c in remaining:
        by_role[c.get("host_role", "COORDINATOR")] = by_role.get(c.get("host_role", "COORDINATOR"), 0.0) + per_cell[c["cell_id"]] * (1 + headroom)
    report["projection"] = {"per_cell": per_cell, "cells": len(design["cells"]), "remaining_cells": len(remaining), "by_role_with_headroom": by_role,
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
    # --- the cells, governed, in dependency order (waves of `parallel` children) ---------------------------
    key = f"{run_id}-mod-e0-cells"
    sha = register(key, [c["cell_id"] for c in design["cells"]])
    report["campaign"] = {"key": key, "campaign_sha256": sha}
    done = {}
    # units registered but without a terminal (a crash between a child's end and its report, or delegated cells): reported on resume
    rs, rb = gov.reconcile_campaign(sha)
    missing_now = set(rb.get("missing_units") or []) if rs == 200 else set()
    missing_units_ref.update(sha=sha, set=set(missing_now))
    mine, delegated = [], []
    for c in design["cells"]:
        owner = c.get("host_role", "COORDINATOR")
        if owner == role:
            mine.append(c)
        elif report_collected and (root / "attempts" / c["cell_id"] / "outcome.json").is_file() and c["cell_id"] in (missing_now or set()):
            mine.append({**c, "_collected": True})
        else:
            delegated.append(c)
    for c in delegated:
        report["cells"][c["cell_id"]] = {"outcome": DELEGATED, "host_role": c.get("host_role"), "why": "dealt to another host; reported after collection"}

    def cell_child(c):
        extra = {"role": "CELL", **{k: c[k] for k in ("depends_on",) if k in c}}
        if c.get("depends_on"):
            extra["extractor_weights"] = str(root / "attempts" / c["depends_on"] / "weights.weights.h5")
        return child(sha, key, c["cell_id"], job_for(c, **extra),
                     {"role": "CELL", "hypothesis": c["hypothesis"], "arm": c["arm"], "level": str(c["level"]), "r": str(c["r"]),
                      "seed": str(c["seed"]), "condition": f"h{c['level']}_r{c['r']}", "replicate": str(c["seed"]),
                      **({"arch": c["arch"]} if "arch" in c else {}), **({"donor": c["donor"]} if c.get("donor") else {}),
                      **({"diagnostic": c["diagnostic"]} if c.get("diagnostic") else {}), "host_role": c.get("host_role", "COORDINATOR"),
                      **({"collected": "true"} if c.get("_collected") else {})},
                     need=per_cell[c["cell_id"]] * (1 + headroom), emit_resumed=bool(c.get("_collected")))
    def dep_state(dep):
        attempt = root / "attempts" / dep
        oc = json.loads((attempt / "outcome.json").read_text()) if (attempt / "outcome.json").is_file() else {}
        if (oc.get("status") or oc.get("outcome")) == "COMPLETED" and ((attempt / "cell.json").is_file() or oc.get("output_sha256") or (oc.get("summary") or {}).get("output_sha256")):
            return {"outcome": "AVAILABLE", "score": None}
        return {"outcome": "ABSENT", "score": None} if not (attempt / "outcome.json").is_file() else {"outcome": "FAILED", "score": None}
    try:
        _run_waves(mine, done, report, cell_child, parallel, lock, trace, dep_state=dep_state)
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
    campaign.write_once(root / ("REPORT.collected.json" if report_collected else "REPORT.json"), report)
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
    parser.add_argument("--api-key-file", default=None, help="required unless --execute-only")
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--cpu-cap-seconds", type=float, default=14400.0)
    parser.add_argument("--already-spent", type=float, default=0.0)
    parser.add_argument("--pilot-updates", type=int, default=300)
    parser.add_argument("--task-memory", type=int, default=3 << 30)
    parser.add_argument("--wall-seconds", type=float, default=1200.0)
    parser.add_argument("--cpu-seconds", type=int, default=1200)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--repair-terminals", action="store_true", help="RP5: supersede refused terminals of this run's campaigns; no child runs")
    parser.add_argument("--role", default="COORDINATOR", help="RP14: this host's role in a v2 design (COORDINATOR, WORKER_A, WORKER_B)")
    parser.add_argument("--execute-only", action="store_true", help="RP14 worker: run this role's cells without governance (registered before, reported after)")
    parser.add_argument("--report-collected", action="store_true", help="RP14 coordinator: emit the terminals of collected delegated attempts")
    parser.add_argument("--parallel", type=int, default=1, help="concurrent children on this host (memory-aware: task memory x parallel must fit)")
    parser.add_argument("--pilot-costs-from", type=Path, default=None, help="parent run's REPORT.json whose measured pilot costs a hypothesis-only successor inherits")
    args = parser.parse_args(argv)
    design = json.loads(args.design.read_text())
    if design.get("schema") not in (D.DESIGN_SCHEMA, AD.DESIGN_SCHEMA) or E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"}) != design["design_sha256"]:
        raise SystemExit("REFUSED: the design is not a sealed MOD-E0 design")
    code_identity = GR.strict_code_identity(REPO)
    budgets = {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds, "cpu_seconds": args.cpu_seconds}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": R.now_iso(), **facts})
        print(json.dumps({"event": event, **{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)[:80]) for k, v in facts.items()}}), flush=True)
    gov = None if args.execute_only else GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = None if args.execute_only else GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    if args.repair_terminals:
        regs = json.loads((args.root / "CAMPAIGNS.json").read_text())
        shas = {v["campaign_sha256"] for v in regs.values()}
        rep = repair_refused_terminals(outbox, gov, GR, shas, "RP5 repair: metric rows carried a status key the governed schema does not admit")
        recon = {k: gov.reconcile_campaign(v["campaign_sha256"])[1] for k, v in regs.items()}
        out = {"schema": "df_mod_e0_terminal_repair.v1", "run_id": args.run_id, "superseded": len(rep["superseded"]), "failed": rep["failed"],
               "reconciliation": {k: {"missing": len(v.get("missing_units") or []), "accounting_only": v.get("accounting_only"), "lake_only": v.get("lake_only")}
                                  for k, v in recon.items()}}
        (args.root / "TERMINAL_REPAIR.json").write_text(json.dumps({**out, "details": rep}, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str))
        return 0 if not rep["failed"] else 2
    if not args.execute_only and not args.api_key_file:
        raise SystemExit("REFUSED: --api-key-file is required unless --execute-only")
    report = run_mod_e0(design, root=args.root, run_id=args.run_id, gov=gov, trace=trace, GR=GR, outbox=outbox, OB=OB, CE=CE,
                        code_identity=code_identity, budgets=budgets, cap_seconds=args.cpu_cap_seconds, already_spent=args.already_spent,
                        pilot_updates=args.pilot_updates, pilot_only=args.pilot_only, role=args.role, execute_only=args.execute_only,
                        report_collected=args.report_collected, parallel=args.parallel,
                        pilot_costs_from=json.loads(args.pilot_costs_from.read_text()) if args.pilot_costs_from else None)
    trace_name = "TRACE.json" if not (args.root / "TRACE.json").exists() else f"TRACE.{args.role}.{R.now_iso().replace(':', '')}.json"
    (args.root / trace_name).write_text(json.dumps(trace_log, indent=1, default=str))
    print(json.dumps({"stopped": report["stopped"], "spent_cpu_seconds": report.get("spent_cpu_seconds"),
                      "projection": {k: v for k, v in (report.get("projection") or {}).items() if k != "per_cell"},
                      "cost_pilot": report["cost_pilot"], "cells_done": len(report["cells"]), "reconciliation": report["reconciliation"]},
                     indent=1, default=str))
    return 0 if report["stopped"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
