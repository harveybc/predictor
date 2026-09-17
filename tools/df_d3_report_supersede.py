#!/usr/bin/env python3
"""Supersede the D3 terminals data-gov refused, generation by generation (J3).

`df_d3_report.py` enqueued one terminal per unit through the durable outbox. The server
refused 67 of them (`duplicate metric identity`: one `d3.verdict.<op>` per variable). Those
envelopes stay in `pending/` with their `.failure` sidecar. This tool rebuilds each refused
unit's terminal from the SAME collect receipt with the corrected metric identity and sends it
as the next generation through `TerminalOutbox.supersede`, which keeps the original outcome
and deliveries, moves the refused envelope unchanged to `adjudicated/` with a write-once
SUPERSEDED disposition linking the accepted successor, and never deletes anything. Then both
campaigns are reconciled and a write-once receipt is written beside REPORT.json.

    python tools/df_d3_report_supersede.py --root RUN_ROOT --run-id ID --collect COLLECT.x.json
        --api-key-file KEY --receipt REPORT.supersede-1.json --reason "..."
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


def refused_envelopes(outbox, campaigns: set) -> list:
    out = []
    for path in outbox._pending_files():
        failure = outbox._failure(path)
        if not failure or failure.get("class") != "REFUSED_BY_SERVER":
            continue
        envelope = json.loads(path.read_text(encoding="ascii"))
        if envelope["campaign_sha256"] in campaigns:
            out.append((path.name, envelope, failure))
    return out


def main(argv=None) -> int:
    GR = _load("governed_run")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collect", default="COLLECT.json")
    parser.add_argument("--report", default="REPORT.json")
    parser.add_argument("--receipt", default="REPORT.supersede-1.json")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reemit-envelope", action="store_true",
                        help="mark the run's pending OLAP envelope FAILED with the reason and "
                             "emit the corrected one from the same collect receipt")
    args = parser.parse_args(argv)

    prior = json.loads((args.root / args.report).read_text(encoding="utf-8"))
    campaigns = {c["campaign_sha256"]: name for name, c in prior["campaigns"].items()}
    units = {u["unit_id"]: u for u in report.collected_units(args.root, args.collect)}
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())

    def sender(envelope):
        status, receipt = gov.report_terminal(envelope["campaign_sha256"], envelope["unit_id"],
                                              envelope["terminal"])
        if status not in (200, 201):
            raise GR.GovernedRunError(
                f"terminal refused: http {status} {receipt.get('error', '')}".strip())
        return receipt

    receipt = {"schema": "d3_mechanics_supersede.v1", "run_id": args.run_id,
               "collect": args.collect, "reason": args.reason, "superseded": [],
               "not_superseded": [], "reconciliation": {}}
    for name, envelope, failure in refused_envelopes(outbox, set(campaigns)):
        unit = units.get(envelope["unit_id"])
        if unit is None:
            receipt["not_superseded"].append({"file": name, "unit_id": envelope["unit_id"],
                                              "why": "unit not in the collect receipt"})
            continue
        base = envelope["terminal"]
        corrected = campaign.unit_terminal(
            unit["rows"], status=base["status"], reason=base.get("reason"),
            wall=base["costs"]["wall_seconds"], cpu=base["costs"]["cpu_seconds"],
            deliveries=base.get("deliveries") or [], bank=unit["bank"],
            unit_id=unit["unit_id"], run_id=args.run_id)
        try:
            disposition = outbox.supersede(name, corrected, sender, args.reason)
        except GR.GovernedRunError as exc:
            receipt["not_superseded"].append({"file": name, "unit_id": envelope["unit_id"],
                                              "why": str(exc)[:300],
                                              "original_failure": failure})
            continue
        receipt["superseded"].append({"file": name, "unit_id": envelope["unit_id"],
                                      "campaign": campaigns[envelope["campaign_sha256"]],
                                      "successor_terminal_sha256":
                                          disposition["successor_terminal_sha256"],
                                      "successor_generation": disposition["successor_generation"],
                                      "original_failure": failure})
    for sha, name in campaigns.items():
        status, body = gov.reconcile_campaign(sha)
        receipt["reconciliation"][name] = {"http": status,
                                           "missing_units": body.get("missing_units"),
                                           "accounting_only": body.get("accounting_only"),
                                           "lake_only": body.get("lake_only")}
    if args.reemit_envelope:
        OB = _load("outbox", REPO / "olap")
        old_name = f"envelope-{prior['envelope']['digest']}.json"
        old_path = next((p for p in OB.pending_entries(None) if p.name == old_name), None)
        frozen = json.loads((args.root / "FREEZE.json").read_text(encoding="utf-8"))
        envelope = campaign.build_mechanics_envelope(
            campaign_key="d3-mechanics-v1", run_id=args.run_id,
            code_identity=GR.strict_code_identity(REPO), frozen=frozen,
            units=list(units.values()), wall_seconds=0.0)
        emitted = OB.emit(envelope, kind="envelope")
        marked = None
        if old_path is not None:
            OB.mark(old_path, OB.FAILED, root=None,
                    reason=f"malformed data_consumed items (bare strings); superseded by "
                           f"envelope {envelope['envelope_sha256']}"[:400])
            marked = old_name
        receipt["envelope"] = {"envelope_sha256": envelope["envelope_sha256"], **emitted,
                               "supersedes_outbox_entry": marked}
    receipt["outbox_status"] = {k: v for k, v in outbox.status().items()
                                if k not in ("pending", "adjudicated")}
    receipt["finished_utc"] = campaign.now_iso()
    campaign.write_once(args.root / args.receipt, receipt)
    summary = {"superseded": len(receipt["superseded"]), "envelope": receipt.get("envelope"),
               "not_superseded": receipt["not_superseded"],
               "reconciliation": receipt["reconciliation"],
               "outbox_status": receipt["outbox_status"]}
    print(json.dumps(summary, indent=1))
    return 0 if not receipt["not_superseded"] and all(
        not r["missing_units"] and not r["accounting_only"] and not r["lake_only"]
        for r in receipt["reconciliation"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
