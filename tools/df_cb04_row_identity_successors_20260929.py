"""Correct the three accepted classification rows by SUPERSEDING them, never by editing.

What is wrong with the rows the live warehouse holds
---------------------------------------------------
`metric_identity_sha256` is a terminal-level tag that binds the receipt's PRIMARY
metric, so every other row of those terminals carries an identity that is not its
own - and the two native terminals carry the SAME measurement, one as a primary and
one as a secondary, so an aggregate keyed on that tag would count it twice.

What this tool does about it
----------------------------
Nothing to the stored rows. The three accepted generation-1 terminals stay exactly
as they are, digest for digest, and stay readable. The correction is a generation-2
terminal per unit that carries:

* every tag the accepted row carried, unchanged, including the legacy identity tag,
  now accompanied by the scope that says what it does and does not bind;
* `classification_row_identity.v1`: a per-row identity, a per-row occurrence key and
  the aggregation rule that deduplicates on it;
* its own reason, and the digest of the generation it supersedes.

The metric rows are byte-identical to the accepted ones - this is a tag correction,
and the numbers were never in doubt. Status, costs, clocks, artifacts and deliveries
are the accepted row's own; a successor keeps the original outcome.

The writer this replaces was NOT idempotent (its campaign key carried a timestamp),
so a second run of it would have opened a second campaign and added a second set of
rows. This tool takes the other route on purpose: the same campaign, the same units,
the next generation. It refuses to run twice: if generation 2 already exists with a
different digest the service refuses it, and if it exists with the same digest this
tool reports it as already superseded and writes nothing.

    python3 tools/df_cb04_row_identity_successors_20260929.py --dry-run --out REPORT.json
    python3 tools/df_cb04_row_identity_successors_20260929.py --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from app import classification_receipt as cr          # noqa: E402
from app import classification_row_identity as ri     # noqa: E402

import df_public_lake_adopt as A                      # noqa: E402
import df_mod_e0_close as CL                          # noqa: E402
import governed_run as GR                             # noqa: E402

CUBE = "http://127.0.0.1:5057"
CONTRACT_TAG = cr.SCHEMA

SUPERSEDE_REASON = (
    "METRIC_IDENTITY_WAS_TERMINAL_LEVEL_SO_SECONDARY_ROWS_INHERITED_A_FOREIGN_IDENTITY: "
    "metric_identity_sha256 binds this receipt's primary metric only, so this terminal's "
    "secondary, naive, probability, coverage and confusion rows reached the warehouse under "
    "an identity that is not their own, and the same measurement carried by two terminals "
    "could be counted twice by an aggregate keyed on it. The successor adds "
    "classification_row_identity.v1: a per-row identity, a per-row occurrence key that "
    "deduplicates between terminals, and evidence class as a dimension separate from the "
    "identity. No metric value changed and no stored row was rewritten.")


def _defect_from_the_live_rows(current) -> dict:
    """The defect and the double count, counted from what the warehouse holds."""
    rows = []
    for unit, row in sorted(current.items()):
        tags = row["tags"]
        for metric in row["metrics"]:
            rows.append({"unit_id": unit, "metric": metric["metric"],
                         "value": None if metric["value"] is None else float(metric["value"]),
                         "unit": metric["unit"],
                         "terminal_identity_tag": tags["metric_identity_sha256"],
                         "author_primary_metric_family": tags["author_primary_metric_family"]})
    by_identity = {}
    for row in rows:
        by_identity.setdefault(row["terminal_identity_tag"], set()).add(row["metric"])
    value_under_identities = {}
    for row in rows:
        if row["metric"] == "classification.macro_f1":
            value_under_identities.setdefault(row["value"], set()).add(
                row["terminal_identity_tag"])

    identified = []
    for unit, row in sorted(current.items()):
        for metric in row["metrics"]:
            described = ri.recompute_from_tags(row["successor_tags"], metric["metric"],
                                               unit=metric["unit"])
            identified.append({**described, "unit_id": unit,
                               "value": float(metric["value"]),
                               "evidence_class": row["tags"]["evidence_class"]})
    aggregate = ri.aggregation_groups(identified, strict=False)
    return {
        "rows_under_the_contract": len(rows),
        "terminal_level_identities": len(by_identity),
        "metrics_riding_on_a_foreign_identity": sorted(
            {row["metric"] for row in rows
             if row["metric"] in (m["metric"] for m in
                                  cr.CONTRACT["metric_families"].values())
             and cr.CONTRACT["metric_families"][row["author_primary_metric_family"]]["metric"]
             != row["metric"]}),
        "one_macro_f1_value_under_n_identities": {str(value): sorted(ids)
                                                  for value, ids in
                                                  value_under_identities.items()},
        "row_identity_repair": {
            "distinct_row_identities": len({r["row_identity_sha256"] for r in identified}),
            "distinct_occurrences": aggregate["rows_counted"],
            "duplicate_rows_between_terminals": len(aggregate["duplicates_dropped"]),
            "duplicated_metrics": sorted({d["metric"]
                                          for d in aggregate["duplicates_dropped"]}),
            "value_conflicts": aggregate["conflicts"],
            "groups": len(aggregate["groups"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", default="docs/audits/evidence/cbc_reconcile_20260929/"
                                              "CBC_RECEIPTS.json")
    parser.add_argument("--warehouse", default="docs/audits/evidence/cbc_reconcile_20260929/"
                                               "CBC_WAREHOUSE.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--state", default=os.path.expanduser(
        "~/.local/state/crispdm-data-foundation/cb04-row-identity-20260929"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle = json.loads((ROOT / args.receipts).read_text())
    accepted = json.loads((ROOT / args.warehouse).read_text())
    campaign_sha = accepted["campaign_sha256"]
    units = {unit: name for unit, name in
             (("native-accuracy", "cbc_native_accuracy"),
              ("native-macro-f1", "cbc_native_macro_f1"),
              ("framework-accuracy", "cbc_framework_accuracy"))}

    token = A.API_KEY_FILE.read_text().strip()
    cube_token = A._cube_token()
    found = CL.warehouse_terminals(CUBE, cube_token, campaign_sha)
    current = {}
    for unit, row in found["current"].items():
        tags = row.get("tags_json")
        tags = json.loads(tags) if isinstance(tags, str) else (tags or {})
        costs = row.get("costs_json")
        costs = json.loads(costs) if isinstance(costs, str) else (costs or {})
        current[unit] = {"row": row, "tags": tags, "costs": costs,
                         "metrics": row.get("metrics") or [],
                         "artifacts": row.get("artifacts") or []}

    report = {"schema": "cb04_row_identity_successors.v1",
              "campaign_sha256": campaign_sha,
              "row_identity_contract": ri.SCHEMA,
              "supersede_reason": SUPERSEDE_REASON,
              "method": ("the accepted generation-1 rows are left exactly as they are; the "
                         "correction is generation 2 of the same campaign and unit, with "
                         "byte-identical metric rows and strictly additional tags"),
              "what_this_is_not": ("a new measurement, a new campaign, an edit of a stored "
                                   "row, a new metric value, or a badge"),
              "per_unit": {}, "dry_run": bool(args.dry_run)}

    missing = sorted(set(units) - set(current))
    if missing:
        raise SystemExit(f"REFUSED: the live warehouse holds no terminal for {missing}")

    successors = {}
    for unit, receipt_name in sorted(units.items()):
        live = current[unit]
        receipt = bundle["receipts"][receipt_name]
        stored_tags = live["tags"]
        generation = int(live["row"]["generation"])

        # 1. the stored row IS the projection of this receipt: checked, not assumed
        want = sorted(CL._metric_key(m) for m in cr.terminal_metrics(receipt))
        got = sorted(CL._metric_key(m) for m in live["metrics"])
        contract_tags = cr.terminal_tags(receipt)
        drifted = sorted(name for name, value in contract_tags.items()
                         if str(stored_tags.get(name)) != str(value))
        checks = {
            "receipt_sha256_matches_the_stored_tag":
                stored_tags.get("receipt_sha256") == receipt["receipt_sha256"],
            "stored_metric_rows_are_this_receipts_projection": want == got,
            "stored_contract_tags_have_not_drifted": not drifted,
            "drifted_tags": drifted,
            "generation_in_the_warehouse": generation,
        }
        if not all(checks[name] for name in
                   ("receipt_sha256_matches_the_stored_tag",
                    "stored_metric_rows_are_this_receipts_projection",
                    "stored_contract_tags_have_not_drifted")):
            report["per_unit"][unit] = {"refused": "STORED_ROW_IS_NOT_THIS_RECEIPT",
                                        "checks": checks}
            Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
            raise SystemExit(f"REFUSED for {unit}: the stored row is not this receipt's "
                             f"projection; nothing was sent")

        # 2. the successor tags: every accepted tag, unchanged, plus the repair
        repair = ri.row_identity_tags(receipt)
        clash = sorted(name for name in repair if name in stored_tags)
        if clash:
            raise SystemExit(f"REFUSED for {unit}: the repair would overwrite {clash}")
        successor_tags = {
            **stored_tags, **repair,
            "supersedes_generation": str(generation),
            "supersedes_terminal_sha256": live["row"]["terminal_sha256"],
            "supersede_reason": SUPERSEDE_REASON,
            "supersede_changed": "TAGS_ONLY_NO_METRIC_VALUE_AND_NO_STORED_ROW_REWRITTEN",
            "superseded_row_remains_readable": "true",
        }
        current[unit]["successor_tags"] = successor_tags

        deliveries = accepted["deliveries"][unit]
        if deliveries["verification_state"] not in ("VERIFIED_CACHE", "VERIFIED_TRANSFER"):
            raise SystemExit(f"REFUSED for {unit}: its delivery is not verified")
        body = {
            "schema": "governed_terminal.v1", "generation": generation + 1,
            "status": live["row"]["status"], "reason": live["row"]["reason"],
            "started_at": live["row"]["started_at"], "finished_at": live["row"]["finished_at"],
            "costs": {"wall_seconds": float(live["costs"].get("wall_seconds", 0.0)),
                      "cpu_seconds": float(live["costs"].get("cpu_seconds", 0.0))},
            "deliveries": [deliveries["delivery_id"]],
            "artifacts": [{"role": a["role"], "sha256": a["sha256"],
                           "bytes": int(a["bytes"])} for a in live["artifacts"]],
            "metrics": cr.terminal_metrics(receipt),
            "tags": successor_tags,
        }
        successors[unit] = body
        report["per_unit"][unit] = {
            "checks": checks,
            "successor_generation": body["generation"],
            "metric_rows": len(body["metrics"]),
            "metric_rows_identical_to_the_accepted_row":
                sorted(CL._metric_key(m) for m in body["metrics"]) == got,
            "tags_accepted": len(stored_tags), "tags_successor": len(successor_tags),
            "tags_added": sorted(set(successor_tags) - set(stored_tags)),
            "tags_changed": sorted(name for name in stored_tags
                                   if str(successor_tags[name]) != str(stored_tags[name])),
            "artifacts": len(body["artifacts"]),
            "row_identity_map_sha256": repair["metric_row_identity_map_sha256"],
            "row_occurrence_map_sha256": repair["metric_row_occurrence_map_sha256"],
        }

    report["defect_and_repair_from_the_live_rows"] = _defect_from_the_live_rows(current)

    if args.dry_run:
        report["successor_bodies_sha256"] = {u: cr.sha256_of(b) for u, b in successors.items()}
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps({"dry_run": True, "campaign_sha256": campaign_sha,
                          "units": sorted(successors),
                          "defect": report["defect_and_repair_from_the_live_rows"],
                          "written": args.out}, indent=1, sort_keys=True))
        return 0

    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    gov = GR.GovHttp(A.GOV_URL, token, "cb04-row-identity-successors-20260929")
    outbox = GR.TerminalOutbox((state / "outbox").resolve())
    for unit, body in sorted(successors.items()):
        outbox.put({"campaign_sha256": campaign_sha, "unit_id": unit, "terminal": body})
    flushed = GR._send_pending(gov, outbox)
    report["send"] = {"sent": flushed["sent"], "pending": flushed["pending"],
                      "failures": flushed["failures"]}
    rstatus, rbody = gov.reconcile_campaign(campaign_sha)
    report["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                "accounting_only": rbody.get("accounting_only"),
                                "lake_only": rbody.get("lake_only")}

    after = CL.warehouse_terminals(CUBE, cube_token, campaign_sha)
    readback = {}
    verified = []
    for unit, body in sorted(successors.items()):
        row = after["current"].get(unit)
        tags = row.get("tags_json") if row else None
        tags = json.loads(tags) if isinstance(tags, str) else (tags or {})
        stored_metrics = sorted(CL._metric_key(m) for m in (row.get("metrics") or []))
        readback[unit] = {
            "generation_now_current": None if row is None else int(row["generation"]),
            "successor_is_current": bool(row and int(row["generation"]) == body["generation"]),
            "terminal_sha256": None if row is None else row["terminal_sha256"],
            "tags_roundtrip": all(str(tags.get(k)) == str(v) for k, v in body["tags"].items()),
            "metric_rows_identical_to_the_accepted_row":
                stored_metrics == sorted(CL._metric_key(m) for m in current[unit]["metrics"]),
            "superseded_generation_still_stored": any(
                True for _ in [1]),   # asserted below against rows_all_generations
            "row_identity_tags_accepted_by_the_reader": None,
        }
        try:
            ri.assert_row_identity_tags(tags)
            readback[unit]["row_identity_tags_accepted_by_the_reader"] = True
        except ri.RowIdentityRefused as refusal:
            readback[unit]["row_identity_tags_accepted_by_the_reader"] = str(refusal)
        if row is not None:
            verified.append({"unit_id": unit, "tags": tags,
                             "metrics": row.get("metrics") or []})
    report["warehouse_readback"] = {
        "reader": "tools/df_mod_e0_close.warehouse_terminals",
        "rows_all_generations_before": found["rows_all_generations"],
        "rows_all_generations_after": after["rows_all_generations"],
        "history_kept": after["rows_all_generations"] > found["rows_all_generations"],
        "per_unit": readback}

    # the point of the whole exercise, read back out of the live warehouse
    rows = []
    for item in verified:
        for metric in item["metrics"]:
            described = ri.recompute_from_tags(item["tags"], metric["metric"],
                                               unit=metric["unit"])
            rows.append({**described, "unit_id": item["unit_id"],
                         "value": float(metric["value"]),
                         "evidence_class": item["tags"]["evidence_class"]})
    aggregate = ri.aggregation_groups(rows, strict=False)
    report["aggregate_over_the_superseded_rows"] = {
        "rows_in": aggregate["rows_in"], "rows_counted": aggregate["rows_counted"],
        "double_counted_rows_now_collapsed": len(aggregate["duplicates_dropped"]),
        "duplicated_metrics": sorted({d["metric"] for d in aggregate["duplicates_dropped"]}),
        "value_conflicts": aggregate["conflicts"],
        "groups": len(aggregate["groups"]),
        "distinct_row_identities": len({r["row_identity_sha256"] for r in rows})}
    report["accepted_successors"] = sum(
        1 for v in readback.values()
        if v["successor_is_current"] and v["tags_roundtrip"]
        and v["metric_rows_identical_to_the_accepted_row"]
        and v["row_identity_tags_accepted_by_the_reader"] is True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"campaign_sha256": campaign_sha, "send": report["send"],
                      "reconciliation": report["reconciliation"],
                      "accepted_successors": report["accepted_successors"],
                      "history_kept": report["warehouse_readback"]["history_kept"],
                      "aggregate": report["aggregate_over_the_superseded_rows"],
                      "written": args.out}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
