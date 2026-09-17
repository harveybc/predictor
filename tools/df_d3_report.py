#!/usr/bin/env python3
"""Report the D3 mechanics through governance (J3): two campaigns, terminals, reconciliation,
and one MECHANICAL envelope for the warehouse.

data-gov keeps SYNTHETIC and DATASETS inputs apart, so the run is reported as two campaigns
under one run id:

    <run_id>-synthetic   input_mode SYNTHETIC, `synthetic_spec_sha256` binding the freeze (bank
                         manifest, generator, design amendment, code digests); one unit per
                         bank unit; `deliveries: []`;
    <run_id>-toys        input_mode DATASETS, declaring every (lake, resource, role) the toy
                         units were delivered as; one unit per toy resource; `deliveries` are
                         the confirmed delivery ids.

Every terminal goes through the durable `TerminalOutbox` and is reconciled with data-gov
before the next is sent, exactly as `governed_run` does. Then one MECHANICAL envelope is
emitted to the OLAP outbox, from which the running loader posts it to the warehouse; nothing
here opens the cube.

Both campaigns are NON_GOVERNING: the terminals grant nothing.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, directory: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


design = _load("df_d3_design")
campaign = _load("df_d3_campaign")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def collected_units(root: Path, receipt: str = "COLLECT.json") -> list:
    """Every unit the collect step verified, with the rows of the attempt it verified."""
    units = []
    collect = json.loads((root / receipt).read_text(encoding="utf-8"))
    for entry in collect["units"]:
        unit = dict(entry)
        rows_path = (root / "collected" / entry["role"] / entry["shard"] / "attempts"
                     / entry["unit"] / f"attempt-{entry.get('attempt', 1)}" / "rows.jsonl")
        unit["rows"] = ([json.loads(line) for line in rows_path.open(encoding="utf-8")]
                        if entry["status"] == "COMPLETED" and entry.get("output_verified")
                        else [])
        unit["verdict_rows"] = [r for r in unit["rows"] if r["test"] == "verdict"]
        unit["unit_id"] = entry["unit"]
        unit["bank"] = (unit["rows"][0]["bank"] if unit["rows"] else
                        ("TOY" if entry["unit"].startswith("toy-") else "SYNTHETIC"))
        units.append(unit)
    return units


def synthetic_spec(frozen: dict) -> dict:
    manifest = Path(frozen["bank"]["root"].replace("~", str(Path.home()))) / "BANK_MANIFEST.json"
    return {"schema": "d3_synthetic_evidence_spec.v1",
            "bank": "SYNTHETIC_D2_C128_V1",
            "bank_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.is_file() else "UNAVAILABLE",
            "design_sha256": design.D3_AMENDMENT_V1["design_sha256"],
            "supersedes_design_sha256": design.ORIGINAL_SHA256,
            "freeze_sha256": frozen["freeze_sha256"],
            "code_sha256s": frozen["code_sha256s"],
            "library_versions": frozen["library_versions"],
            "units": len(frozen["bank"]["units"])}


def register(gov, GR, *, key: str, classification: str, code_identity: dict, config_sha256: str,
             input_mode: str, synthetic_spec_sha256, units: list, datasets: list,
             metrics_lake: str, project: str) -> str:
    body = {"schema": "governed_campaign.v1", "campaign_key": key,
            "classification": classification, "project": project,
            "code_identity": code_identity, "config_sha256": config_sha256,
            "input_mode": input_mode, "synthetic_spec_sha256": synthetic_spec_sha256,
            "units": units, "datasets": datasets, "terminal_lake": metrics_lake}
    status, receipt = gov.submit_campaign(body)
    if status not in (200, 201):
        raise SystemExit(f"REFUSED: campaign {key} refused: http {status} "
                         f"{receipt.get('error', '')}".strip())
    return receipt["campaign_sha256"]


def report_terminals(gov, GR, outbox, *, campaign_sha256: str, units: list, run_id: str,
                     deliveries_of) -> list:
    receipts = []
    for unit in units:
        status = "COMPLETED" if unit["status"] == "COMPLETED" and unit.get("output_verified") \
            else "FAILED" if unit["status"] in ("FAILED", "RESOURCE_EXCEEDED") else "INCONCLUSIVE"
        reason = None if status == "COMPLETED" else \
            f"{unit['status']}: {unit.get('reason') or 'output not verified'}"[:300]
        terminal = campaign.unit_terminal(
            unit["rows"], status=status, reason=reason, wall=unit.get("wall_seconds") or 0.0,
            cpu=unit.get("cpu_seconds") or 0.0, deliveries=deliveries_of(unit),
            bank=unit["bank"], unit_id=unit["unit_id"], run_id=run_id)
        GR._require_reconciled(gov, campaign_sha256, unit["unit_id"], before_run=True)
        outbox.put({"campaign_sha256": campaign_sha256, "unit_id": unit["unit_id"],
                    "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        receipts.append({"unit_id": unit["unit_id"], "status": status,
                         "pending_after_flush": flushed["pending"]})
    return receipts


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--metrics-lake", default="olap_cube")
    parser.add_argument("--project", default="predictor")
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--collect", default="COLLECT.json",
                        help="the collect receipt (under --root) whose verified rows are reported")
    parser.add_argument("--toy-campaign-sha256",
                        help="the DATASETS campaign the toys were delivered under (already "
                             "registered by the toys step); its terminals are reported here")
    args = parser.parse_args(argv)

    root = args.root
    frozen = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
    units = collected_units(root, args.collect)
    synthetic_units = [u for u in units if u["bank"] == "SYNTHETIC"]
    toy_units = [u for u in units if u["bank"] == "TOY"]
    code_identity = GR.strict_code_identity(REPO)
    spec = synthetic_spec(frozen)
    spec_sha = sha_text(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    config_sha = sha_text(json.dumps({"schema": "d3_mechanics_execution.v1",
                                      "run_id": args.run_id,
                                      "freeze_sha256": frozen["freeze_sha256"],
                                      "design_sha256": design.D3_AMENDMENT_V1["design_sha256"]},
                                     sort_keys=True, separators=(",", ":")))
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    prior = GR._send_pending(gov, outbox)
    if prior["pending"]:
        raise SystemExit(f"REFUSED: a prior terminal remains pending: {prior}")

    started = time.monotonic()
    receipt = {"schema": "d3_mechanics_report.v1", "run_id": args.run_id,
               "code_identity": code_identity, "synthetic_spec": spec,
               "synthetic_spec_sha256": spec_sha, "config_sha256": config_sha,
               "campaigns": {}, "terminals": {}, "reconciliation": {}, "envelope": None}

    if synthetic_units:
        key = f"{args.run_id}-synthetic"
        sha = register(gov, GR, key=key, classification="NON_GOVERNING",
                       code_identity=code_identity, config_sha256=config_sha,
                       input_mode="SYNTHETIC", synthetic_spec_sha256=spec_sha,
                       units=[u["unit_id"] for u in synthetic_units], datasets=[],
                       metrics_lake=args.metrics_lake, project=args.project)
        receipt["campaigns"]["synthetic"] = {"campaign_key": key, "campaign_sha256": sha,
                                             "units": len(synthetic_units)}
        receipt["terminals"]["synthetic"] = report_terminals(
            gov, GR, outbox, campaign_sha256=sha, units=synthetic_units, run_id=args.run_id,
            deliveries_of=lambda u: [])
        status, body = gov.reconcile_campaign(sha)
        receipt["reconciliation"]["synthetic"] = {"http": status,
                                                  "missing_units": body.get("missing_units"),
                                                  "accounting_only": body.get("accounting_only"),
                                                  "lake_only": body.get("lake_only")}

    if toy_units and args.toy_campaign_sha256:
        toys_root = root / "toys"
        deliveries = {}
        for unit in toy_units:
            rec_path = toys_root / unit["unit_id"] / "TOY.json"
            if rec_path.is_file():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                deliveries[unit["unit_id"]] = [rec["delivery"]["delivery_id"]]
        receipt["campaigns"]["toys"] = {"campaign_sha256": args.toy_campaign_sha256,
                                        "units": len(toy_units)}
        receipt["terminals"]["toys"] = report_terminals(
            gov, GR, outbox, campaign_sha256=args.toy_campaign_sha256, units=toy_units,
            run_id=args.run_id, deliveries_of=lambda u: deliveries.get(u["unit_id"], []))
        status, body = gov.reconcile_campaign(args.toy_campaign_sha256)
        receipt["reconciliation"]["toys"] = {"http": status,
                                             "missing_units": body.get("missing_units"),
                                             "accounting_only": body.get("accounting_only"),
                                             "lake_only": body.get("lake_only")}

    envelope = campaign.build_mechanics_envelope(
        campaign_key="d3-mechanics-v1", run_id=args.run_id, code_identity=code_identity,
        frozen=frozen, units=units, wall_seconds=time.monotonic() - started)
    emitted = OB.emit(envelope, kind="envelope")
    receipt["envelope"] = {"envelope_sha256": envelope["envelope_sha256"], **emitted}
    receipt["finished_utc"] = now_iso()
    campaign.write_once(root / "REPORT.json", receipt)
    print(json.dumps({"campaigns": receipt["campaigns"], "reconciliation": receipt["reconciliation"],
                      "envelope": receipt["envelope"]}, indent=1))
    pending = any(t["pending_after_flush"] for group in receipt["terminals"].values()
                  for t in group)
    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
