#!/usr/bin/env python3
"""The governed entry point of the utility harness — a REHEARSAL on fabricated data, and the
descriptive development pilot (M4, N3, N4).

Order of events, observable through `trace`:

    freeze-pre    protocol base, plan, family, operators, budgets, data digest — write-once
    register      the CALIBRATION campaign (one unit per operator) — before any child
    before_run    governance's before_run for that unit, then the child
    calibrate     one isolated child per operator under the same ceilings (a preparatory
                  campaign with its own design: the record is the child's verified output,
                  its cost and real instants its terminal's)
    mechanics     the real D3 battery on the series in an isolated child → the eligibility
                  record (rehearsal) — or the verified matrix's cells (pilot)
    seal          the protocol per operator with its own record — FREEZE, write-once
    register      the CONTRASTS campaign — before any contrast child
    contrast      before_run, then one isolated child per contrast; the score is the verified
                  output file; instants and cost are the child's
    report        one terminal per contrast (metrics from the verified score; RESOURCE_EXCEEDED
                  with cost), reconciled; one DEVELOPMENT envelope

A refusal at registration stops everything before the first child. A completed attempt is
never re-run (resume without duplicates: the same bytes rebuild the same terminal).

    python tools/df_utility_run.py --root RUN_ROOT --run-id ID --api-key-file KEY [--pilot-unit ...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
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
ops = _load("df_d3_operators")
contract = _load("df_d3_contract")
campaign = _load("df_d3_campaign")

UNIT, VARIABLE = "fab", "v0"
SAMPLE_CONTRACT = {"frequency": "1s", "availability": {
    "label": "WINDOW_START", "completion_lag_max": "0s",
    "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "LIVE_EQUIVALENT"}}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fabricated(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        drive = 0.0
        if t > 20:
            window = x[t - 16:t]
            med = np.median(window)
            mad = np.median(np.abs(window - med)) or 1.0
            drive = -0.6 * np.sign(x[t - 1] - med) * min(3.0, abs(x[t - 1] - med) / mad)
        x[t] = x[t - 1] + drive + rng.normal(0, 1.0)
    return x


class Refusal(SystemExit):
    pass


def _terminal(*, status, reason, cost, metrics, tags, started, finished) -> dict:
    return {"schema": "governed_terminal.v1", "generation": 1, "status": status,
            "reason": reason, "started_at": started, "finished_at": finished,
            "costs": {"wall_seconds": max(0.0, float(cost.get("wall_seconds") or 0)),
                      "cpu_seconds": max(0.0, float(cost.get("cpu_seconds") or 0))},
            "deliveries": [], "artifacts": [], "metrics": metrics, "tags": tags}


def _metric(name, value, unit, split="development", horizon=1):
    return {"metric": name, "split": split, "horizon": horizon, "unit": unit,
            "value": float(value), "std_dev": None, "min_value": None, "max_value": None}


def run_rehearsal(cfg: dict, gov, trace, *, GR, outbox, isolated=None) -> tuple:
    """The whole chain with injectable governance (`gov`), observation (`trace(event, **facts)`)
    and child runner (`isolated`, default H.run_isolated)."""
    isolated = isolated or H.run_isolated
    root = Path(cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    code_identity = cfg["code_identity"]
    run_id = cfg["run_id"]
    units = cfg["units"]
    operators = list(cfg["operators"])
    plan = cfg["plan"]
    budgets = cfg["budgets"]

    def child(kind, name, job, wall, cpu, memory):
        trace("child", kind=kind, name=name)
        out = isolated({**job, "kind": kind}, attempt_dir=root / "attempts" / name,
                       assigned_bytes=memory, wall_seconds=wall, cpu_seconds=cpu)
        trace("child-done", kind=kind, name=name, outcome=out.get("outcome"))
        return out

    family = tuple(f"{u['unit']}__{u['variable']}__{k}__transformed"
                   for u in units for k in operators)
    if cfg.get("slow_control"):
        family = family + (f"{units[0]['unit']}__{units[0]['variable']}__{operators[0]}"
                           f"__transformed__slow-control",)
    base = H.Protocol(**cfg["protocol"], family=family, calibration_plan=dict(plan))
    pre = {"schema": "df_utility_freeze_pre.v1", "run_id": run_id, "frozen_utc": now_iso(),
           "protocol_base": base.sealed(), "protocol_base_sha256": base.base_sha256(),
           "plan": plan, "operators": operators,
           "units": [{"unit": u["unit"], "variable": u["variable"],
                      "data_sha256": sha_bytes(np.asarray(u["values"], dtype=float).tobytes()),
                      "n": len(u["values"])} for u in units],
           "budgets": budgets, "code_identity": code_identity, "purpose": cfg["purpose"],
           "classification": "NON_GOVERNING", "freeze_sha256": ""}
    pre["freeze_sha256"] = campaign.sha_obj({k: v for k, v in pre.items() if k != "freeze_sha256"})
    if (root / "FREEZE.pre.json").is_file():
        pre = json.loads((root / "FREEZE.pre.json").read_text())      # resume: the sealed one
    else:
        campaign.write_once(root / "FREEZE.pre.json", pre)
    trace("freeze-pre", sha256=pre["freeze_sha256"])
    receipt = {"schema": "df_utility_rehearsal_report.v2", "run_id": run_id,
               "purpose": cfg["purpose"], "freeze_pre_sha256": pre["freeze_sha256"],
               "calibration": {}, "mechanics": {}, "contrasts": {}, "terminals": [],
               "reconciliation": {}, "envelope": None}

    # --- calibration campaign: registered before any child -------------------------------------
    cal_key = f"{run_id}-utility-calibration"
    status, reg = gov.submit_campaign({
        "schema": "governed_campaign.v1", "campaign_key": cal_key, "classification": "NON_GOVERNING",
        "project": cfg.get("project", "predictor"), "code_identity": code_identity,
        "config_sha256": pre["freeze_sha256"], "input_mode": "SYNTHETIC",
        "synthetic_spec_sha256": campaign.sha_obj(plan), "units": operators, "datasets": [],
        "terminal_lake": cfg.get("metrics_lake", "olap_cube")})
    trace("register", key=cal_key, http=status)
    if status not in (200, 201):
        raise Refusal(f"REFUSED: calibration campaign refused: http {status} "
                      f"{(reg or {}).get('error', '')}; no child was started")
    cal_sha = reg["campaign_sha256"]
    receipt["calibration"]["campaign"] = {"key": cal_key, "campaign_sha256": cal_sha}
    records = {}
    for kind in operators:
        GR._require_reconciled(gov, cal_sha, kind, before_run=True)
        trace("before_run", key=cal_key, unit=kind)
        out = child("calibrate", f"calibrate__{kind}",
                    {"contrast_id": family[0], "operator": kind, "protocol": base.sealed(),
                     "plan": plan, "seed": base.seed + 1000 + operators.index(kind)},
                    budgets["calibration_wall_seconds"], budgets["calibration_cpu_seconds"],
                    budgets["task_memory_bytes"])
        cost = out["cost"]
        rec = out.get("score")
        tags = {"purpose": "UTILITY_CALIBRATION", "grants": "NONE", "classification": "NON_GOVERNING",
                "operator": kind, "protocol_base_sha256": base.base_sha256(),
                "record_sha256": out.get("output_sha256") or ""}
        if out["outcome"] == H.RESOURCE_EXCEEDED or rec is None:
            terminal = _terminal(status="FAILED", reason=f"{out['outcome']}: {out.get('reason') or ''}"[:300],
                                 cost=cost, metrics=[], started=cost.get("started_at") or now_iso(),
                                 finished=cost.get("ended_at") or now_iso(), tags=tags)
        else:
            records[kind] = rec
            terminal = _terminal(status="COMPLETED", reason=None, cost=cost,
                                 metrics=[_metric("calibration.false_advance_rate", rec["false_advance_rate"], "rate"),
                                          _metric("calibration.upper_bound", rec["upper_bound"], "rate"),
                                          _metric("calibration.scored", rec["scored"], "count"),
                                          _metric("calibration.failed", rec["failed"], "count")],
                                 started=cost.get("started_at") or now_iso(),
                                 finished=cost.get("ended_at") or now_iso(),
                                 tags={**tags, "generator": rec["generator"], "n_sims": str(rec["n_sims"])})
        outbox.put({"campaign_sha256": cal_sha, "unit_id": kind, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        receipt["calibration"][kind] = {"outcome": out["outcome"], "cost": cost,
                                        "record_sha256": out.get("output_sha256"),
                                        "upper_bound": (rec or {}).get("upper_bound"),
                                        "pending_after_flush": flushed["pending"]}
    rstatus, rbody = gov.reconcile_campaign(cal_sha)
    receipt["reconciliation"]["calibration"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                                "accounting_only": rbody.get("accounting_only"),
                                                "lake_only": rbody.get("lake_only")}

    # --- eligibility: mechanics in an isolated child (rehearsal) or the verified cells (pilot) ---
    eligibility_paths = {}
    for u in units:
        if u.get("eligibility"):
            eligibility_paths[u["unit"]] = Path(u["eligibility"])
            trace("eligibility", unit=u["unit"], source="verified-cells")
            continue
        out = child("mechanics", f"mechanics__{u['unit']}",
                    {"contrast_id": family[0], "protocol": base.sealed(), "operators": operators,
                     "series": {"values": list(map(float, u["values"]))}, "unit": u["unit"],
                     "variable": u["variable"], "resource_contract": SAMPLE_CONTRACT,
                     "run_id": run_id, "freeze_sha256": pre["freeze_sha256"]},
                    budgets["mechanics_wall_seconds"], budgets["mechanics_cpu_seconds"],
                    budgets["task_memory_bytes"])
        if out["outcome"] == H.RESOURCE_EXCEEDED or out.get("score") is None:
            raise Refusal(f"REFUSED: mechanics for {u['unit']} did not complete: {out.get('reason')}")
        path = root / "attempts" / f"mechanics__{u['unit']}" / "cells.json"
        eligibility_paths[u["unit"]] = path
        receipt["mechanics"][u["unit"]] = {"cost": out["cost"], "cells_sha256": out.get("output_sha256"),
                                           "verdicts": {c["operator"]: c["verdict"] for c in out["score"]["cells"]}}

    # --- seal per operator ---------------------------------------------------------------------------
    protocols = {kind: base.with_calibration(records[kind]) if kind in records else base
                 for kind in operators}
    frozen = {"schema": "df_utility_freeze.v2", "run_id": run_id, "frozen_utc": now_iso(),
              "freeze_pre_sha256": pre["freeze_sha256"],
              "protocols": {k: p.sealed() for k, p in protocols.items()},
              "calibrated": sorted(records),
              "eligibility": {u: str(p) for u, p in eligibility_paths.items()},
              "eligibility_sha256": {u: sha_bytes(p.read_bytes()) for u, p in eligibility_paths.items()},
              "code_identity": code_identity, "freeze_sha256": ""}
    frozen["freeze_sha256"] = campaign.sha_obj({k: v for k, v in frozen.items() if k != "freeze_sha256"})
    if (root / "FREEZE.json").is_file():
        frozen = json.loads((root / "FREEZE.json").read_text())
    else:
        campaign.write_once(root / "FREEZE.json", frozen)
    trace("freeze", sha256=frozen["freeze_sha256"])

    # --- contrasts campaign: registered before any contrast child ---------------------------------
    key = f"{run_id}-utility-contrasts"
    status, reg = gov.submit_campaign({
        "schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
        "project": cfg.get("project", "predictor"), "code_identity": code_identity,
        "config_sha256": frozen["freeze_sha256"], "input_mode": "SYNTHETIC",
        "synthetic_spec_sha256": pre["freeze_sha256"], "units": list(family), "datasets": [],
        "terminal_lake": cfg.get("metrics_lake", "olap_cube")})
    trace("register", key=key, http=status)
    if status not in (200, 201):
        raise Refusal(f"REFUSED: contrasts campaign refused: http {status}; no contrast started")
    campaign_sha = reg["campaign_sha256"]
    receipt["contrasts"]["campaign"] = {"key": key, "campaign_sha256": campaign_sha}
    by_unit = {u["unit"]: u for u in units}
    outcomes = {}
    for contrast_id in family:
        unit_id, variable, kind = contrast_id.split("__")[:3]
        slow = contrast_id.endswith("__slow-control")
        u = by_unit[unit_id]
        GR._require_reconciled(gov, campaign_sha, contrast_id, before_run=True)
        trace("before_run", key=key, unit=contrast_id)
        job = {"contrast_id": contrast_id, "unit": unit_id, "variable": variable, "operator": kind,
               "protocol": protocols[kind].sealed(),
               "series": {"values": list(map(float, u["values"]))},
               "eligibility": str(eligibility_paths[unit_id])}
        if slow:
            job["slow_seconds"] = budgets["slow_control"]["slow_seconds"]
        out = child("contrast", contrast_id, job,
                    budgets["slow_control"]["wall_seconds"] if slow else budgets["wall_seconds"],
                    budgets["cpu_seconds"], budgets["task_memory_bytes"])
        outcomes[contrast_id] = out
        score = out.get("score") or {}
        cost = out["cost"]
        if out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE, H.INCONCLUSIVE_UNCALIBRATED,
                              H.INSUFFICIENT_ROWS, H.REFUSED):
            status_t, reason = "COMPLETED", None
        elif out["outcome"] == H.RESOURCE_EXCEEDED:
            status_t, reason = "FAILED", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        else:
            status_t, reason = "INCONCLUSIVE", f"{out['outcome']}: {out.get('reason') or ''}"[:300]
        metrics = []
        if status_t == "COMPLETED" and isinstance(score.get("delta_mean"), (int, float)):
            metrics = [_metric("utility.delta_mean", score["delta_mean"], score["loss_name"]),
                       _metric("utility.delta_lower", score["delta_lower"], score["loss_name"]),
                       _metric("utility.delta_se", score["delta_se"], score["loss_name"]),
                       _metric("utility.blocks_used", score["blocks_used"], "count")]
        terminal = _terminal(status=status_t, reason=reason, cost=cost, metrics=metrics,
                             started=cost.get("started_at") or now_iso(),
                             finished=cost.get("ended_at") or now_iso(),
                             tags={"purpose": cfg["purpose"], "grants": "NONE",
                                   "classification": "NON_GOVERNING", "outcome": str(out["outcome"]),
                                   "protocol_sha256": protocols[kind].sealed()["protocol_sha256"],
                                   "freeze_sha256": frozen["freeze_sha256"],
                                   "output_sha256": out.get("output_sha256") or "",
                                   "calibrated": str(kind in records)})
        outbox.put({"campaign_sha256": campaign_sha, "unit_id": contrast_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        receipt["terminals"].append({"unit_id": contrast_id, "status": status_t,
                                     "outcome": out["outcome"], "cost": cost,
                                     "delta_mean": score.get("delta_mean"),
                                     "pending_after_flush": flushed["pending"]})
    rstatus, rbody = gov.reconcile_campaign(campaign_sha)
    receipt["reconciliation"]["contrasts"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                              "accounting_only": rbody.get("accounting_only"),
                                              "lake_only": rbody.get("lake_only")}
    receipt["contrasts"]["outcomes"] = {k: {kk: vv for kk, vv in v.items() if kk != "score"}
                                        for k, v in outcomes.items()}
    receipt["finished_utc"] = now_iso()
    return receipt, outcomes, frozen, pre, campaign_sha


def emit_envelope(cfg, outcomes, frozen, pre, OB, CE) -> dict:
    units = []
    for contrast_id, out in outcomes.items():
        score = out.get("score") or {}
        has = isinstance(score.get("delta_mean"), (int, float))
        units.append({"candidate_key": contrast_id.split("__")[2], "cell_key": contrast_id,
                      "metric_name": "utility.delta_mean" if has else "outcome",
                      "metric_value": float(score["delta_mean"]) if has else "UNAVAILABLE",
                      "terminal_state": "COMPLETE" if out["outcome"] in (
                          H.ADVANCES, H.DOES_NOT_ADVANCE, H.INCONCLUSIVE_UNCALIBRATED)
                      else str(out["outcome"]), "uncertainty_kind": "BLOCK_T_LOWER",
                      "uncertainty_low": float(score["delta_lower"]) if has else "UNAVAILABLE",
                      "uncertainty_high": "UNAVAILABLE"})
    envelope = CE.build_envelope(
        campaign_key=f"utility-{cfg['purpose'].lower()}-{cfg['run_id']}", producer="predictor",
        result_class="DEVELOPMENT",
        identity={"run_id": cfg["run_id"], "code_identity": cfg["code_identity"]["value"],
                  "design_sha256": pre["protocol_base_sha256"], "record_sha256": frozen["freeze_sha256"]},
        data_consumed={"datasets": [{"id": u["unit"], "digest": u["data_sha256"],
                                     "eligibility_state": cfg["eligibility_state"]} for u in pre["units"]],
                       "variables": [{"id": u["variable"], "digest": "UNAVAILABLE",
                                      "eligibility_state": cfg["eligibility_state"]} for u in pre["units"]],
                       "operators": [{"id": k, "digest": contract.spec_sha256(ops.build(k).describe()),
                                      "eligibility_state": cfg["eligibility_state"]} for k in cfg["operators"]]},
        partitions={"exposure": cfg["exposure"], "splits": "UNAVAILABLE"},
        budget={"device": "cpu", "wall_seconds": float(sum((o["cost"].get("wall_seconds") or 0)
                                                           for o in outcomes.values())),
                "cost_units": len(outcomes)},
        terminal={"state": "COMPLETED", "adjudication": "DESCRIPTIVE_NO_ADJUDICATION"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL", "freeze": frozen["freeze_sha256"]},
        units=units)
    emitted = OB.emit(envelope, kind="envelope")
    return {"envelope_sha256": envelope["envelope_sha256"], **emitted}


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    CE = _load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--n", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--operators", nargs="+",
                        default=["mad_extremes_trailing", "delta_run_length", "cusum_causal"])
    parser.add_argument("--bound-confidence", type=float, default=0.95)
    parser.add_argument("--calibration-sims", type=int, default=None,
                        help="default: the simulations zero advances need for the bound")
    parser.add_argument("--task-memory", type=int, default=1 << 30)
    parser.add_argument("--wall-seconds", type=float, default=240.0)
    parser.add_argument("--cpu-seconds", type=int, default=240)
    parser.add_argument("--calibration-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--slow-seconds", type=float, default=30.0)
    parser.add_argument("--slow-wall-seconds", type=float, default=6.0)
    parser.add_argument("--pilot-unit", action="append", default=[],
                        help="descriptive pilot: a synthetic bank unit directory (up to three)")
    parser.add_argument("--pilot-cells", type=Path, help="the verified matrix's cells record")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--n-blocks", type=int, default=4)
    args = parser.parse_args(argv)
    code_identity = GR.strict_code_identity(REPO)
    pilot = bool(args.pilot_unit)
    if pilot:
        if len(args.pilot_unit) > 3 or not args.pilot_cells:
            raise SystemExit("REFUSED: the descriptive pilot takes up to three development units "
                             "and the verified cells record")
        worker = _load("df_d3_unit_worker")
        units = []
        for d in args.pilot_unit:
            u = worker.load_unit(Path(d))
            if u["bank"] != "SYNTHETIC":
                raise SystemExit("REFUSED: the pilot takes synthetic development units only")
            units.append({"unit": u["unit_id"], "variable": "v0", "values": u["inputs"][0]["values"],
                          "eligibility": str(args.pilot_cells)})
        purpose, eligibility_state, exposure = ("UTILITY_DESCRIPTIVE_PILOT", "DEVELOPMENT_SYNTHETIC",
                                                "DEVELOPMENT_PILOT_NO_RESERVE")
    else:
        units = [{"unit": UNIT, "variable": VARIABLE, "values": fabricated(args.n, args.seed).tolist()}]
        purpose, eligibility_state, exposure = ("UTILITY_HARNESS_REHEARSAL", "FABRICATED_REHEARSAL",
                                                "DEVELOPMENT_REHEARSAL_NO_RESERVE")
    lengths = {len(u["values"]) for u in units}
    if len(lengths) != 1:
        raise SystemExit("REFUSED: one calibration length per run; units differ in length")
    n_len = lengths.pop()
    family_size = len(units) * len(args.operators) + (0 if pilot else 1)
    alpha_adjusted = 0.05 / family_size
    n_sims = args.calibration_sims or H.sims_required_for_zero(alpha_adjusted, args.bound_confidence)
    cfg = {"root": str(args.root), "run_id": args.run_id, "code_identity": code_identity,
           "units": units, "operators": list(args.operators), "purpose": purpose,
           "eligibility_state": eligibility_state, "exposure": exposure,
           "slow_control": not pilot,
           "plan": {"generator": "white_null", "n_sims": int(n_sims), "n": int(n_len),
                    "bound_confidence": args.bound_confidence},
           "protocol": {"target": "return", "horizon": 1, "model": "ridge", "window": 4,
                        "n_blocks": args.n_blocks, "margin": args.margin, "seed": args.seed,
                        "min_rows_per_block": 30},
           "budgets": {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds,
                       "cpu_seconds": args.cpu_seconds,
                       "calibration_wall_seconds": args.calibration_wall_seconds,
                       "calibration_cpu_seconds": int(args.calibration_wall_seconds),
                       "mechanics_wall_seconds": args.wall_seconds,
                       "mechanics_cpu_seconds": args.cpu_seconds,
                       "slow_control": {"slow_seconds": args.slow_seconds,
                                        "wall_seconds": args.slow_wall_seconds}}}
    trace_log = []

    def trace(event, **facts):
        trace_log.append({"event": event, "at": now_iso(), **facts})

    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    receipt, outcomes, frozen, pre, _ = run_rehearsal(cfg, gov, trace, GR=GR, outbox=outbox)
    receipt["envelope"] = emit_envelope(cfg, outcomes, frozen, pre, OB, CE)
    receipt["trace"] = trace_log
    campaign.write_once(args.root / "REPORT.json", receipt)
    print(json.dumps({"calibration": receipt["calibration"],
                      "reconciliation": receipt["reconciliation"],
                      "terminals": [(t["unit_id"], t["outcome"], t["delta_mean"]) for t in receipt["terminals"]],
                      "envelope": receipt["envelope"]}, indent=1, default=str)[:4000])
    ok = all(not (r or {}).get("missing_units") for r in receipt["reconciliation"].values()) \
        and all(not t["pending_after_flush"] for t in receipt["terminals"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
