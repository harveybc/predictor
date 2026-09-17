#!/usr/bin/env python3
"""Report the seven D3 toy units under one DATASETS campaign each (J3).

`df_d3_campaign.py toys` registered ONE campaign declaring all seven toy resources and one
unit per resource. data-gov's rule for a DATASETS campaign is that a COMPLETED terminal
covers every dataset the campaign declares, and a delivery belongs to (campaign, actor,
unit): with seven datasets declared campaign-wide, no per-resource unit can complete. The
server was right and the campaign shape was mine. The mechanics rows are unaffected — they
were computed on bytes delivered and confirmed under that campaign.

This tool registers, per toy resource, a campaign `<run_id>-toy-<stem>` declaring that ONE
resource and that ONE unit, delivers and confirms the resource again under it, refuses to go
on if the confirmed bytes differ from the `source_sha256` the unit was computed on, reports
the unit's terminal (the rows the collect receipt verified, the new delivery) through the
durable outbox, reconciles the campaign, and disposes the refused generation-1 envelope of
the original campaign as INVALID_ENVELOPE with this reason. Nothing is deleted; the original
campaign, its deliveries and its refusals stay as evidence. Write-once receipt.

    python tools/df_d3_toys_recampaign.py --root RUN_ROOT --run-id ID --collect COLLECT.x.json
        --api-key-file KEY --lake-config HOST.json --receipt TOYS.recampaign-1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

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


campaign = _load("df_d3_campaign")
report = _load("df_d3_report")
design = _load("df_d3_design")

INVALID_REASON = ("generation 1 was reported under a campaign declaring all seven toy resources "
                  "campaign-wide; a per-resource unit cannot complete it (422). Reported instead "
                  "under a one-resource, one-unit campaign; this envelope is closed as evidence.")


def main(argv=None) -> int:
    GR = _load("governed_run")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collect", default="COLLECT.json")
    parser.add_argument("--receipt", default="TOYS.recampaign-1.json")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--lake", default="governance_smoke")
    parser.add_argument("--lake-config", type=Path, required=True)
    parser.add_argument("--metrics-lake", default="olap_cube")
    parser.add_argument("--project", default="predictor")
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    args = parser.parse_args(argv)

    frozen = json.loads((args.root / "FREEZE.json").read_text(encoding="utf-8"))
    prior_toys = json.loads((args.root / "TOYS.json").read_text(encoding="utf-8"))
    code_identity = GR.strict_code_identity(REPO)
    config_sha = campaign.sha_obj({"schema": "d3_mechanics_execution.v1", "run_id": args.run_id,
                                   "freeze_sha256": frozen["freeze_sha256"],
                                   "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"]})
    units = {u["unit_id"]: u for u in report.collected_units(args.root, args.collect)
             if u["bank"] == "TOY"}
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    cache = Path(GR.DEFAULT_CACHE).expanduser() / f"{args.run_id}-recampaign"
    receipt = {"schema": "d3_toys_recampaign.v1", "run_id": args.run_id,
               "original_campaign_sha256": prior_toys["campaign_sha256"],
               "collect": args.collect, "campaigns": [], "disposed": [], "refused": []}

    for resource, role in campaign.TOY_RESOURCES:
        unit_id = campaign.toy_unit_id(resource)
        unit = units.get(unit_id)
        toy = json.loads((args.root / "toys" / unit_id / "TOY.json").read_text(encoding="utf-8"))
        if unit is None or not unit.get("output_verified"):
            receipt["refused"].append({"unit_id": unit_id, "why": "not verified by the collect"})
            continue
        key = f"{args.run_id}-toy-{Path(resource).stem}"
        gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), key)
        status, reg = gov.submit_campaign({
            "schema": "governed_campaign.v1", "campaign_key": key,
            "classification": "NON_GOVERNING", "project": args.project,
            "code_identity": code_identity, "config_sha256": config_sha,
            "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": [unit_id],
            "datasets": [{"lake": args.lake, "resource": resource, "role": role,
                          "from": None, "to": None}],
            "terminal_lake": args.metrics_lake})
        if status not in (200, 201):
            receipt["refused"].append({"unit_id": unit_id, "why": f"campaign refused: http "
                                       f"{status} {reg.get('error', '')}".strip()})
            continue
        sha = reg["campaign_sha256"]
        dstatus, info = gov.governed_download(sha, unit_id, args.lake, resource, role, str(cache))
        if dstatus != 200:
            receipt["refused"].append({"unit_id": unit_id, "campaign_sha256": sha,
                                       "why": f"download refused: http {dstatus}"})
            continue
        got = hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()
        if got != toy["source_sha256"] or info["sha256"] != toy["source_sha256"]:
            receipt["refused"].append({"unit_id": unit_id, "campaign_sha256": sha,
                                       "why": "delivered bytes differ from the bytes the unit "
                                              "was computed on", "expected": toy["source_sha256"],
                                       "got": got})
            continue
        terminal = campaign.unit_terminal(
            unit["rows"], status="COMPLETED", reason=None, wall=unit.get("wall_seconds") or 0.0,
            cpu=unit.get("cpu_seconds") or 0.0, deliveries=[info["delivery_id"]], bank="TOY",
            unit_id=unit_id, run_id=args.run_id)
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        rstatus, body = gov.reconcile_campaign(sha)
        entry = {"unit_id": unit_id, "campaign_key": key, "campaign_sha256": sha,
                 "resource": resource, "delivery_id": info["delivery_id"],
                 "verification_state": info.get("verification_state"),
                 "bytes_sha256": got, "pending_after_flush": flushed["pending"],
                 "reconciliation": {"http": rstatus, "missing_units": body.get("missing_units"),
                                    "accounting_only": body.get("accounting_only"),
                                    "lake_only": body.get("lake_only")}}
        receipt["campaigns"].append(entry)
        if entry["reconciliation"]["missing_units"]:
            continue
        # the generation-1 envelope of the original campaign, closed as evidence
        for path in outbox._pending_files():
            env = json.loads(path.read_text(encoding="ascii"))
            if env["campaign_sha256"] == prior_toys["campaign_sha256"] \
                    and env["unit_id"] == unit_id:
                disposition = outbox.dispose(path.name, "INVALID_ENVELOPE", INVALID_REASON)
                receipt["disposed"].append({"file": path.name, "unit_id": unit_id,
                                            "failure": disposition["failure"]})
    receipt["outbox_status"] = {k: v for k, v in outbox.status().items()
                                if k not in ("pending", "adjudicated")}
    receipt["finished_utc"] = campaign.now_iso()
    campaign.write_once(args.root / args.receipt, receipt)
    ok = not receipt["refused"] and all(not c["reconciliation"]["missing_units"]
                                        and not c["pending_after_flush"]
                                        for c in receipt["campaigns"])
    print(json.dumps({"campaigns": [(c["unit_id"], c["reconciliation"]["missing_units"],
                                     c["pending_after_flush"]) for c in receipt["campaigns"]],
                      "disposed": len(receipt["disposed"]), "refused": receipt["refused"],
                      "outbox_status": receipt["outbox_status"]}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
