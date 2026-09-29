"""CB-C: turn the retained classification projections into ACCEPTED warehouse rows.

A retained projection is a file.  An accepted warehouse row is a terminal the live
warehouse holds and will hand back.  The two are not the same claim, and this module
exists to close the distance between them through the EXISTING route and nothing else:

    register a campaign -> take a governed delivery per unit -> write the terminal to the
    O_EXCL outbox -> send it -> reconcile the campaign -> READ THE TERMINAL DIGEST BACK
    OUT OF THE LIVE WAREHOUSE and compare its stored rows, tags and costs against what
    was sent.

No service is started, stopped or restarted.  No schema, table or column is added.  The
metric rows are `app/classification_receipt.terminal_metrics` unmodified and the tags are
`terminal_tags` plus provenance; this module computes no score of its own.

What it deliberately does NOT write, and the reason, travels with the receipt bundle:
the author's PUBLISHED_REFERENCE value shares its metric identity with our measurement,
so it is carried as read-only tags rather than as metric rows an aggregate could reach.

  python tools/df_cbc_warehouse_20260929.py \
      --receipts docs/audits/evidence/cbc_reconcile_20260929/CBC_RECEIPTS.json \
      --out docs/audits/evidence/cbc_reconcile_20260929/CBC_WAREHOUSE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import df_public_lake_adopt as A     # noqa: E402  (endpoints, key file, cube token)
import governed_run as GR            # noqa: E402
import df_utility_run as R           # noqa: E402
import df_mod_e0_close as CL         # noqa: E402

LAKE = "sota_benchmarks"
RESOURCE = "agnews_zhang2015_test/test.parquet"
EXPECT_SHA = "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"
CUBE = "http://127.0.0.1:5057"

#: one warehouse unit per receipt that SHOULD become a row
UNITS = {
    "native-accuracy": "cbc_native_accuracy",
    "native-macro-f1": "cbc_native_macro_f1",
    "framework-accuracy": "cbc_framework_accuracy",
}

PUBLISHED_REFERENCE_TAGS = {
    "published_reference_value": "0.9525",
    "published_reference_metric": "accuracy",
    "published_reference_source": (
        "NandhaKishorM/laya research/results/app_benchmark_results.json, "
        "suites['jev.ag_news']['typed-decisions']; run stamped 2026-09-19 11:57:16, "
        "device cpu, laya 0.2.1, n_per_task 400"),
    "published_reference_in_training": "true",
    "published_reference_is_not_our_measurement": "true",
    "published_reference_not_written_as_metric_rows_because": (
        "it shares metric_identity_sha256 with our ACCURACY measurement, so a stored row "
        "would be reachable by an aggregate keyed on metric identity"),
}


def sha_file(path):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", default="docs/audits/evidence/cb03_20260929")
    ap.add_argument("--state", default=os.path.expanduser(
        "~/.local/state/crispdm-data-foundation/cbc-classification-20260929"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bundle = json.loads(Path(args.receipts).read_text())
    receipts = bundle["receipts"]
    projections = bundle["warehouse_projection"]
    for unit, name in UNITS.items():
        if name not in projections:
            raise SystemExit(f"REFUSED: no retained projection for {name}")

    ev = Path(args.evidence)
    artifacts = []
    for name in ("native_published400.json", "parity_400.json", "CB03_PARITY_ATTRIBUTION.json",
                 "MANIFEST.json"):
        digest, size = sha_file(ev / name)
        artifacts.append({"role": name.replace(".json", ""), "sha256": digest, "bytes": size})
    rdigest, rsize = sha_file(args.receipts)
    artifacts.append({"role": "cbc_receipts", "sha256": rdigest, "bytes": rsize})

    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    token = A.API_KEY_FILE.read_text().strip()
    run_id = f"cbc-classification-{int(time.time())}"
    key = f"{run_id}-agnews-published400"
    code_identity = GR.strict_code_identity(ROOT)
    config_sha = hashlib.sha256(json.dumps(
        {"route": "cbc-classification-receipts", "lake": LAKE, "resource": RESOURCE,
         "receipts": bundle["receipt_sha256"]}, sort_keys=True).encode()).hexdigest()
    campaign = {
        "schema": "governed_campaign.v1", "campaign_key": key,
        # honest: the scoring ran under CB03's own admitted children, not inside this
        # campaign, so this campaign transports and binds a measurement it did not run
        "classification": "NON_GOVERNING",
        "project": "predictor", "code_identity": code_identity, "config_sha256": config_sha,
        "input_mode": "DATASETS", "synthetic_spec_sha256": None,
        "units": sorted(UNITS), "terminal_lake": "olap_cube",
        "datasets": [{"lake": LAKE, "resource": RESOURCE, "role": "panel",
                      "from": None, "to": None}],
    }
    report = {"schema": "cbc_warehouse.v1", "campaign_key": key,
              "code_identity": code_identity, "config_sha256": config_sha,
              "units": sorted(UNITS), "artifacts_bound": artifacts,
              "dry_run": bool(args.dry_run)}
    if args.dry_run:
        report["campaign_body"] = campaign
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps({"dry_run": True, "campaign_key": key,
                          "units": sorted(UNITS)}, indent=1))
        return 0

    gov = GR.GovHttp(A.GOV_URL, token, key)
    status, receipt = gov.submit_campaign(campaign)
    report["campaign_http"] = status
    report["campaign_sha256"] = receipt.get("campaign_sha256")
    if status not in (200, 201):
        report["error"] = receipt
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        raise SystemExit(f"REFUSED by data-gov at campaign registration: {status} {receipt}")
    campaign_sha = receipt["campaign_sha256"]

    outbox = GR.TerminalOutbox((state / "outbox").resolve())
    deliveries, sent = {}, {}
    for unit in sorted(UNITS):
        _http, info = gov.governed_download(campaign_sha, unit, LAKE, RESOURCE, "panel",
                                            str(state / "cache"))
        on_disk, _ = sha_file(info["path"])
        deliveries[unit] = {
            "delivery_id": info.get("delivery_id"),
            "verification_state": info.get("verification_state"),
            "sha256": info.get("sha256"), "bytes": info.get("bytes"),
            "cached": info.get("cached"),
            "availability_use": info.get("availability_use"),
            "availability_label": info.get("availability_label"),
            "bytes_on_disk_sha256": on_disk,
            "delivered_bytes_are_the_pinned_bytes": on_disk == EXPECT_SHA,
        }
        if on_disk != EXPECT_SHA:
            report["deliveries"] = deliveries
            Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
            raise SystemExit("REFUSED: the delivered AG News bytes are not the pinned bytes")
    report["deliveries"] = deliveries

    for unit in sorted(UNITS):
        name = UNITS[unit]
        proj = projections[name]
        rec = receipts[name]
        tags = dict(proj["tags"])
        tags.update({
            "purpose": "CLASSIFICATION_BENCHMARK_RECEIPT",
            "classification": "NON_GOVERNING",
            "phase": "DEVELOPMENT",
            "lake": LAKE,
            "resource": RESOURCE,
            "receipt_name": name,
            "receipt_sha256": rec["receipt_sha256"],
            "measurement_provenance": (
                "RETROSPECTIVE_REPORT_OF_A_RETAINED_RUN: the score was produced by CB03's own "
                "admitted children and is recounted from their retained per-row arrays; this "
                "campaign binds a byte-identical fresh governed delivery of the same resource "
                "and does NOT claim the model ran inside it"),
            "recount_tool": "tools/df_cbc_recount_20260929.py",
            "receipt_tool": "tools/df_cbc_receipts_20260929.py",
            "paired_naive_tiebreak_range_low": "0.180000",
            "paired_naive_tiebreak_range_high": "0.307500",
            "paired_naive_tiebreak_is_a_four_way_tie": "true",
            "held_out": "false",
            "in_training": "true",
        })
        if name == "cbc_native_accuracy":
            tags.update(PUBLISHED_REFERENCE_TAGS)
        body = R._terminal(status="COMPLETED", reason=None,
                           cost={"wall_seconds": 0.0, "cpu_seconds": 0.0},
                           metrics=proj["metrics"], tags=tags,
                           started=R.now_iso(), finished=R.now_iso())
        body["deliveries"] = [deliveries[unit]["delivery_id"]]
        body["artifacts"] = artifacts
        outbox.put({"campaign_sha256": campaign_sha, "unit_id": unit, "terminal": body})
        sent[unit] = body

    flushed = GR._send_pending(gov, outbox)
    report["terminals"] = {"sent": flushed["sent"], "pending": flushed["pending"],
                           "failures": flushed["failures"],
                           "metric_rows_sent": {u: len(b["metrics"]) for u, b in sent.items()}}
    rstatus, rbody = gov.reconcile_campaign(campaign_sha)
    report["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                "accounting_only": rbody.get("accounting_only"),
                                "lake_only": rbody.get("lake_only")}
    report["campaign_closed"] = bool(rstatus == 200 and not rbody.get("missing_units")
                                     and not rbody.get("accounting_only")
                                     and not rbody.get("lake_only"))

    # --- acceptance, read back out of the LIVE warehouse ---------------------------------
    cube_token = A._cube_token()
    found = CL.warehouse_terminals(CUBE, cube_token, campaign_sha)
    current = found["current"]
    accepted = {}
    for unit, body in sent.items():
        row = current.get(unit)
        if row is None:
            accepted[unit] = {"accepted_warehouse_row": False,
                              "why": "the live warehouse holds no terminal row for this unit"}
            continue
        want = sorted(CL._metric_key(m) for m in body["metrics"])
        got = sorted(CL._metric_key(m) for m in (row.get("metrics") or []))
        tags = row.get("tags_json")
        tags = json.loads(tags) if isinstance(tags, str) else (tags or {})
        costs = row.get("costs_json")
        costs = json.loads(costs) if isinstance(costs, str) else (costs or {})
        stored_primary = {m["metric"]: float(m["value"]) for m in (row.get("metrics") or [])}
        accepted[unit] = {
            "accepted_warehouse_row": True,
            "terminal_sha256": row.get("terminal_sha256"),
            "generation": row.get("generation"),
            "status_stored": row.get("status"),
            "status_matches": str(row.get("status")) == "COMPLETED",
            "metric_rows": {"sent": len(body["metrics"]), "stored": len(row.get("metrics") or [])},
            "metric_rows_match": want == got,
            "costs_match": all(abs(float(costs.get(k, 0.0))
                                   - float(body["costs"].get(k, 0.0))) < 1e-9
                               for k in ("wall_seconds", "cpu_seconds")),
            "tags_roundtrip": all(str(tags.get(k)) == str(v) for k, v in body["tags"].items()),
            "artifacts_stored": len(row.get("artifacts") or []),
            "stored_primary_value": stored_primary,
            "receipt_sha256_in_stored_tags": tags.get("receipt_sha256"),
            "receipt_sha256_matches": tags.get("receipt_sha256") == body["tags"]["receipt_sha256"],
            "metric_identity_in_stored_tags": tags.get("metric_identity_sha256"),
        }
    report["warehouse_readback"] = {
        "cube": CUBE, "reader": "tools/df_mod_e0_close.warehouse_terminals",
        "rows_all_generations": found["rows_all_generations"], "per_unit": accepted}
    report["accepted_warehouse_rows"] = sum(1 for v in accepted.values()
                                            if v.get("accepted_warehouse_row")
                                            and v.get("metric_rows_match")
                                            and v.get("status_matches")
                                            and v.get("tags_roundtrip"))
    report["acceptance_proved_by"] = ("the terminal digest each row carries, read back out of the "
                                      "live warehouse and compared against the payload sent")
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({
        "campaign_key": key, "campaign_sha256": campaign_sha,
        "campaign_http": report["campaign_http"],
        "deliveries_verified": {u: v["delivered_bytes_are_the_pinned_bytes"]
                                for u, v in deliveries.items()},
        "terminals": report["terminals"], "reconciliation": report["reconciliation"],
        "campaign_closed": report["campaign_closed"],
        "accepted_warehouse_rows": report["accepted_warehouse_rows"],
        "terminal_sha256": {u: v.get("terminal_sha256") for u, v in accepted.items()},
        "stored_primary_value": {u: v.get("stored_primary_value", {})
                                 for u, v in accepted.items()},
        "written": args.out,
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
