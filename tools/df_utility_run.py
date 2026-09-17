#!/usr/bin/env python3
"""The governed entry point of the utility harness — a REHEARSAL on fabricated data (M4).

    mechanics  the real D3 battery runs on the fabricated series for every operator in the
               protocol's family: its verdicts ARE the eligibility record (no flag, no forgery)
    calibrate  the false-advance rate of the protocol under a dependent null, sealed into it
    freeze     protocol, family, calibration, eligibility digest, data digest, budgets — write-once
    contrasts  one isolated child per contrast under df_isolated_runner (observed budgets);
               a deliberately slow control is part of the family and must end RESOURCE_EXCEEDED
    report     one data-gov SYNTHETIC campaign, one terminal per contrast through the durable
               outbox (COMPLETED with the loss delta as metrics, or RESOURCE_EXCEEDED with cost),
               reconciled; one DEVELOPMENT envelope tagged REHEARSAL to the OLAP outbox

This is not the utility experiment: no project data, no reserve, `NON_GOVERNING`, and the
envelope's result_class is DEVELOPMENT with purpose UTILITY_HARNESS_REHEARSAL.

    python tools/df_utility_run.py --root RUN_ROOT --run-id ID --api-key-file KEY
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
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
battery = _load("df_d3_acceptance")
contract = _load("df_d3_contract")
campaign = _load("df_d3_campaign")

OPERATORS = ("mad_extremes_trailing", "delta_run_length", "cusum_causal")
#: the fabricated series is SAMPLE_INDEX with immediate availability, as the bank units are
SAMPLE_CONTRACT = {"frequency": "1s", "availability": {
    "label": "WINDOW_START", "completion_lag_max": "0s",
    "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "LIVE_EQUIVALENT"}}
UNIT, VARIABLE = "fab", "v0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fabricated(n: int, seed: int) -> np.ndarray:
    """The rehearsal series: the next increment depends on how extreme the last value is
    relative to a trailing median — a truth a trailing detector can carry."""
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


def mechanics(x: np.ndarray) -> dict:
    """The real battery on the fabricated series: verdict per operator = eligibility."""
    s = H.series(x)
    xin = H._as_operator_input(s)
    train = battery.prefix(xin, max(2, x.size // 2))
    cells = []
    for kind in OPERATORS:
        op = ops.build(kind)
        report = battery.run_battery(op, xin, train=train, twin=ops.twin_of(op),
                                     resource_contract=SAMPLE_CONTRACT)
        cells.append({"unit": UNIT, "variable": VARIABLE, "operator": kind,
                      "verdict": report["verdict"],
                      "spec_sha256": contract.spec_sha256(op.describe()),
                      "failed": report["failed"], "undecided": report["undecided"]})
    return {"schema": "d3_mechanics_cells.v1", "run_id": "rehearsal", "verified": True,
            "freeze_sha256": "rehearsal-mechanics", "design_sha256":
            _load("df_d3_design").D3_DESIGN_CURRENT["design_sha256"], "cells": cells,
            "note": "verdicts from the real battery on the fabricated series; a rehearsal record"}


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
    parser.add_argument("--metrics-lake", default="olap_cube")
    parser.add_argument("--project", default="predictor")
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--n", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--calibration-sims", type=int, default=12)
    parser.add_argument("--task-memory", type=int, default=1 << 30)
    parser.add_argument("--wall-seconds", type=float, default=240.0)
    parser.add_argument("--cpu-seconds", type=int, default=240)
    parser.add_argument("--slow-seconds", type=float, default=30.0,
                        help="the deliberately slow control exceeds --wall-seconds on purpose")
    parser.add_argument("--slow-wall-seconds", type=float, default=6.0)
    args = parser.parse_args(argv)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    code_identity = GR.strict_code_identity(REPO)

    # 1. data + mechanics (eligibility from the real battery)
    x = fabricated(args.n, args.seed)
    data_sha = sha_bytes(x.tobytes())
    record = mechanics(x)
    campaign.write_once(root / "ELIGIBILITY.cells.json", record)
    eligibility = H.eligibility_record(root / "ELIGIBILITY.cells.json")

    # 2. protocol + calibration + family (the slow control is a member, declared as such)
    # contrast ids double as governed unit ids: no separator data-gov refuses
    family = tuple(f"{UNIT}__{VARIABLE}__{kind}__transformed" for kind in OPERATORS) \
        + (f"{UNIT}__{VARIABLE}__mad_extremes_trailing__transformed__slow-control",)
    base = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=4,
                      margin=0.0, seed=args.seed, family=family, min_rows_per_block=30)
    calibration = H.calibrate(base, ops.build("delta_run_length"), n_sims=args.calibration_sims,
                              seed=args.seed + 1000, n=700)
    protocol = base.with_calibration(calibration)
    frozen = {"schema": "df_utility_rehearsal_freeze.v1", "run_id": args.run_id,
              "frozen_utc": now_iso(), "protocol": protocol.sealed(),
              "calibration": calibration, "data": {"generator": "fabricated_extreme",
                                                   "n": args.n, "seed": args.seed,
                                                   "sha256": data_sha},
              "eligibility_sha256": sha_bytes((root / "ELIGIBILITY.cells.json").read_bytes()),
              "budgets": {"task_memory_bytes": args.task_memory, "wall_seconds": args.wall_seconds,
                          "cpu_seconds": args.cpu_seconds,
                          "slow_control": {"slow_seconds": args.slow_seconds,
                                           "wall_seconds": args.slow_wall_seconds}},
              "code_identity": code_identity, "classification": "NON_GOVERNING",
              "purpose": "UTILITY_HARNESS_REHEARSAL", "freeze_sha256": ""}
    body = {k: v for k, v in frozen.items() if k != "freeze_sha256"}
    frozen["freeze_sha256"] = campaign.sha_obj(body)
    campaign.write_once(root / "FREEZE.json", frozen)

    # 3. contrasts, each in an isolated child
    outcomes = {}
    for contrast_id in family:
        kind = contrast_id.split("__")[2]
        slow = contrast_id.endswith("__slow-control")
        job = {"contrast_id": contrast_id, "unit": UNIT, "variable": VARIABLE, "operator": kind,
               "protocol": protocol.sealed(), "series": {"values": x.tolist()},
               "eligibility": str(root / "ELIGIBILITY.cells.json")}
        if slow:
            job["slow_seconds"] = args.slow_seconds
        out = H.run_isolated(job, attempt_dir=root / "attempts" / contrast_id,
                             assigned_bytes=args.task_memory,
                             wall_seconds=args.slow_wall_seconds if slow else args.wall_seconds,
                             cpu_seconds=args.cpu_seconds)
        outcomes[contrast_id] = out

    # 4. governance: one SYNTHETIC campaign, one terminal per contrast, reconciled
    key = f"{args.run_id}-utility-rehearsal"
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), key)
    status, receipt = gov.submit_campaign({
        "schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
        "project": args.project, "code_identity": code_identity,
        "config_sha256": frozen["freeze_sha256"], "input_mode": "SYNTHETIC",
        "synthetic_spec_sha256": data_sha, "units": list(family), "datasets": [],
        "terminal_lake": args.metrics_lake})
    if status not in (200, 201):
        raise SystemExit(f"REFUSED: campaign refused: http {status} {receipt.get('error', '')}")
    campaign_sha = receipt["campaign_sha256"]
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    terminals = []
    for contrast_id, out in outcomes.items():
        score = out.get("score") or {}
        status_t = "COMPLETED" if out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE,
                                                     H.INCONCLUSIVE_UNCALIBRATED,
                                                     H.INSUFFICIENT_ROWS, H.REFUSED) \
            else "FAILED" if out["outcome"] == H.RESOURCE_EXCEEDED else "INCONCLUSIVE"
        metrics = []
        if status_t == "COMPLETED" and "delta_mean" in score:
            for name, val in (("utility.delta_mean", score["delta_mean"]),
                              ("utility.delta_lower", score["delta_lower"]),
                              ("utility.blocks_used", float(score["blocks_used"]))):
                metrics.append({"metric": name, "split": "development", "horizon": 1,
                                "unit": score["loss_name"], "value": float(val), "std_dev": None,
                                "min_value": None, "max_value": None})
        terminal = {"schema": "governed_terminal.v1", "generation": 1, "status": status_t,
                    "reason": None if status_t == "COMPLETED" else
                    f"{out['outcome']}: {out.get('reason') or ''}"[:300],
                    "started_at": now_iso(), "finished_at": now_iso(),
                    "costs": {"wall_seconds": max(0.0, float(out["cost"].get("wall_seconds") or 0)),
                              "cpu_seconds": max(0.0, float(out["cost"].get("cpu_seconds") or 0))},
                    "deliveries": [], "artifacts": [], "metrics": metrics,
                    "tags": {"purpose": "UTILITY_HARNESS_REHEARSAL", "grants": "NONE",
                             "classification": "NON_GOVERNING", "outcome": str(out["outcome"]),
                             "protocol_sha256": protocol.sealed()["protocol_sha256"],
                             "freeze_sha256": frozen["freeze_sha256"]}}
        GR._require_reconciled(gov, campaign_sha, contrast_id, before_run=True)
        outbox.put({"campaign_sha256": campaign_sha, "unit_id": contrast_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        terminals.append({"unit_id": contrast_id, "status": status_t, "outcome": out["outcome"],
                          "cost": out["cost"], "pending_after_flush": flushed["pending"]})
    rstatus, rbody = gov.reconcile_campaign(campaign_sha)

    # 5. the envelope: DEVELOPMENT, a rehearsal, never MECHANICAL evidence nor a claim
    units = []
    for contrast_id, out in outcomes.items():
        score = out.get("score") or {}
        has_delta = isinstance(score.get("delta_mean"), (int, float))
        units.append({"candidate_key": contrast_id.split("__")[2], "cell_key": contrast_id,
                      "metric_name": "utility.delta_mean" if has_delta else "outcome",
                      # an absent measurement is UNAVAILABLE (stored as null), never a zero
                      "metric_value": float(score["delta_mean"]) if has_delta else "UNAVAILABLE",
                      "terminal_state": "COMPLETE" if out["outcome"] in (
                          H.ADVANCES, H.DOES_NOT_ADVANCE, H.INCONCLUSIVE_UNCALIBRATED)
                      else str(out["outcome"]), "uncertainty_kind": "BLOCK_T_LOWER",
                      "uncertainty_low": float(score["delta_lower"]) if isinstance(score.get("delta_lower"), (int, float)) else "UNAVAILABLE",
                      "uncertainty_high": "UNAVAILABLE"})
    envelope = CE.build_envelope(
        campaign_key=f"utility-rehearsal-{args.run_id}", producer="predictor",
        result_class="DEVELOPMENT",
        identity={"run_id": args.run_id, "code_identity": code_identity["value"],
                  "design_sha256": protocol.sealed()["protocol_sha256"],
                  "record_sha256": frozen["freeze_sha256"]},
        data_consumed={"datasets": [{"id": f"fabricated_extreme/{args.seed}", "digest": data_sha,
                                     "eligibility_state": "FABRICATED_REHEARSAL"}],
                       "variables": [{"id": VARIABLE, "digest": "UNAVAILABLE",
                                      "eligibility_state": "FABRICATED_REHEARSAL"}],
                       "operators": [{"id": k, "digest": contract.spec_sha256(ops.build(k).describe()),
                                      "eligibility_state": "FABRICATED_REHEARSAL"} for k in OPERATORS]},
        partitions={"exposure": "DEVELOPMENT_REHEARSAL_NO_RESERVE", "splits": "UNAVAILABLE"},
        budget={"device": "cpu", "wall_seconds": float(sum(
            (o["cost"].get("wall_seconds") or 0) for o in outcomes.values())),
                "cost_units": len(family)},
        terminal={"state": "COMPLETED", "adjudication": "REHEARSAL_NO_ADJUDICATION"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL", "freeze": frozen["freeze_sha256"]},
        units=units)
    emitted = OB.emit(envelope, kind="envelope")
    receipt_doc = {"schema": "df_utility_rehearsal_report.v1", "run_id": args.run_id,
                   "campaign": {"key": key, "campaign_sha256": campaign_sha},
                   "terminals": terminals,
                   "reconciliation": {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                      "accounting_only": rbody.get("accounting_only"),
                                      "lake_only": rbody.get("lake_only")},
                   "outcomes": {k: {kk: vv for kk, vv in v.items() if kk != "score"}
                                for k, v in outcomes.items()},
                   "envelope": {"envelope_sha256": envelope["envelope_sha256"], **emitted},
                   "finished_utc": now_iso()}
    campaign.write_once(root / "REPORT.json", receipt_doc)
    print(json.dumps({k: v for k, v in receipt_doc.items() if k in ("campaign", "reconciliation",
                                                                    "outcomes", "envelope")},
                     indent=1, default=str)[:3000])
    ok = not rbody.get("missing_units") and all(not t["pending_after_flush"] for t in terminals)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
