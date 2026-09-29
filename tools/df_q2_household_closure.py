#!/usr/bin/env python3
"""The closure table of the Q2 household governed round, generated from that round's own artifacts.

The owner's standing rule: every result carries a table with the model error and its metric and
scale, the paired naive ON THE SAME ROWS, the skill, a reference with its source or NOT_CARRIED,
and comparability with NOT_COMPARABLE and a reason.  A closure `null` is an ABSENT MEASUREMENT and
never a numeric zero, and NO_NEW_MEASUREMENT is an honest outcome.

This round fitted nothing.  Its units are a mechanical governed probe and a bounded memory pilot,
so every error cell is `null` with the reason that produced it, the round is NO_NEW_MEASUREMENT,
and the table's content is the part that WAS measured: the delivery, the reader, the bytes
re-verified inside the child, the accepted terminal, the warehouse row, and the scope peak.

    python tools/df_q2_household_closure.py --root ROOT --out TABLE.json --markdown TABLE.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "df_q2_household_closure.v1"
ESTIMAND = "MATCHED_BUDGET_DIFFERENCE"
ABSENT = "ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO"


def row_for(unit: dict) -> dict:
    """One closure row per governed unit, from that unit's own retained record."""
    child = unit.get("child_record") or {}
    delivery = unit.get("delivery") or {}
    terminal = unit.get("terminal") or {}
    warehouse = unit.get("warehouse") or {}
    memory = child.get("memory") or {}
    consumed = child.get("consumed") or {}
    peak = memory.get("scope_peak_bytes")
    return {
        "unit": unit.get("unit"),
        "role": (child.get("work") or {}).get("windows_shape") and "MEMORY_PILOT" or "MECHANICAL_PROBE",
        "design_sha256": unit.get("design_sha256"),
        "host_kind": child.get("host_kind"),

        # --- the science: absent, and said so ------------------------------------------------
        "estimand": ESTIMAND,
        "estimand_note": ("the block this lane serves censors every cell at 600 observed updates; "
                          "a fixed-600-update design answers the paired difference at a matched "
                          "budget and cannot answer converged accuracy"),
        "model_error": None,
        "model_error_metric": None,
        "model_error_scale": None,
        "model_error_absent_because": (f"{ABSENT}: this unit fits no model, holds no checkpoint and "
                                       "scores no evaluation population"),
        "naive_error": None,
        "naive_error_metric": None,
        "naive_error_scale": None,
        "naive_paired_on_the_same_rows": None,
        "naive_absent_because": f"{ABSENT}: there are no scored rows for a naive to be paired on",
        "skill": None,
        "skill_absent_because": f"{ABSENT}: skill is 1 - error_model/error_naive and both are absent",
        "reference": "NOT_CARRIED",
        "reference_source": "NOT_CARRIED",
        "comparability": "NOT_COMPARABLE",
        "comparability_reason": ("this unit reports a chain and a memory footprint, not an error; a "
                                 "published accuracy would be compared against nothing here, and "
                                 "the block's own estimand is a matched-budget difference which no "
                                 "converged-training publication shares"),

        # --- what WAS measured ---------------------------------------------------------------
        "delivery_id": delivery.get("delivery_id"),
        "delivery_verification_state": delivery.get("verification_state"),
        "delivery_served_from_cache": delivery.get("cached"),
        "delivered_sha256": delivery.get("sha256"),
        "delivered_bytes": delivery.get("bytes"),
        "availability_contract_sha256": delivery.get("availability_contract_sha256"),
        "consumed_sha256_reverified_in_child": consumed.get("sha256_reverified_in_child"),
        "consumed_matches_delivery": consumed.get("matches_delivery"),
        "terminal_status": terminal.get("status"),
        "terminal_accepted": terminal.get("accepted"),
        "terminal_sha256": terminal.get("receipt_terminal_sha256"),
        "warehouse_terminal_sha256": (warehouse.get("terminal") or {}).get("terminal_sha256"),
        "warehouse_digest_agrees": ((warehouse.get("terminal") or {}).get("terminal_sha256")
                                    == terminal.get("receipt_terminal_sha256")
                                    and terminal.get("receipt_terminal_sha256") is not None),
        "warehouse_metric_rows": warehouse.get("metric_rows"),
        "scope_peak_bytes": peak,
        "scope_peak_provenance": memory.get("scope_peak_provenance"),
        "scope_peak_absent_because": None if peak is not None else
            "NULL_PEAK_MEANS_NOT_MEASURED_NEVER_SMALL",
        "rss_self_peak_bytes": memory.get("rss_self_peak_bytes"),
        "rss_is_not_a_cgroup_peak": ("one process's resident set; it may not size a cap and it is "
                                     "not interchangeable with memory.peak in either direction"),
        "declared_cap_bytes": memory.get("declared_cap_bytes_as_the_kernel_holds_it"),
        "custody": "ACCEPTED_TERMINAL_AND_WAREHOUSE_ROW" if (
            terminal.get("accepted") and warehouse.get("ok")) else "UNCHECKED",
    }


def build(root: Path) -> dict:
    root = Path(root).expanduser()
    units = sorted(root.glob("UNIT.*.json"))
    rows = [row_for(json.loads(p.read_text(encoding="utf-8"))) for p in units]
    scored = [r for r in rows if r["model_error"] is not None]
    return {
        "schema": SCHEMA,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "root_basename": root.name,
        "no_new_measurement": not scored,
        "rows": rows,
        "scored_rows": len(scored),
        "rules": {
            "null": "a closure null is an ABSENT MEASUREMENT and never a numeric zero",
            "no_new_measurement": "an honest and preferred outcome when nothing was scored",
            "estimand": f"{ESTIMAND}, never CONVERGED_ACCURACY",
            "null_peak": "a null peak means NOT MEASURED, never small",
            "short_child": ("memory.peak is a kernel high-watermark rather than a sample, so it is a "
                            "measurement even for a child shorter than the launcher's sampling "
                            "interval; a SAMPLED tree peak for such a child is a floor"),
        },
        "what_this_round_does_not_establish": [
            "no successor cap for any W1440 cell: this round built no model, no gradients, no "
            "optimizer slots and no Keras graph, so its peak is a data-stage floor",
            "no custody for any of the twelve historical Q2 fits: they stay NOT_BOUND_TO_A_SEAL",
            "no promotion, selection or ranking of anything",
        ],
    }


def markdown(table: dict) -> str:
    head = ("| unit | estimand | model error (metric, scale) | paired naive, same rows | skill | "
            "reference (source) | comparability | scope peak | custody |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [f"# Closure table — {table['root_basename']}", "",
             f"Generated from artifacts at {table['at']}.  "
             f"**{'NO_NEW_MEASUREMENT' if table['no_new_measurement'] else 'MEASURED'}**: "
             f"{table['scored_rows']} scored rows.", "", head, sep]
    for r in table["rows"]:
        def cell(value, absent_key):
            return "`null`" if value is None else str(value)
        peak = (f"{r['scope_peak_bytes']} B" if r["scope_peak_bytes"] is not None
                else "`null` — NOT MEASURED, never small")
        lines.append("| {unit} | {est} | {me} | {na} | {sk} | {ref} | {cmp} | {pk} | {cu} |".format(
            unit=r["unit"], est=r["estimand"],
            me=cell(r["model_error"], "model_error_absent_because"),
            na=cell(r["naive_error"], "naive_absent_because"),
            sk=cell(r["skill"], "skill_absent_because"),
            ref=f"{r['reference']} ({r['reference_source']})",
            cmp=r["comparability"], pk=peak, cu=r["custody"]))
    lines += ["", "## Why each null is there", ""]
    for r in table["rows"]:
        lines.append(f"* **{r['unit']}** — model error: {r['model_error_absent_because']}  "
                     f"naive: {r['naive_absent_because']}  skill: {r['skill_absent_because']}  "
                     f"comparability: {r['comparability_reason']}")
    lines += ["", "## What this round does not establish", ""]
    lines += [f"* {item}" for item in table["what_this_round_does_not_establish"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    table = build(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(table, indent=1, sort_keys=True, default=str), encoding="utf-8")
    if a.markdown:
        a.markdown.write_text(markdown(table), encoding="utf-8")
    print(json.dumps({"rows": len(table["rows"]), "scored_rows": table["scored_rows"],
                      "no_new_measurement": table["no_new_measurement"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
