#!/usr/bin/env python3
"""RR04: the closure table of a GOVERNED UNIT, built from that unit's own artifacts.

`tools/df_closure_table.py` closes an E1 block: a sealed design, a registry, run roots with
`attempts/<unit>/arrays.npz`. A unit produced by `tools/governed_run.py` has none of those; it has
an out-dir with `GOVERNED_RUN.json`, the predictions CSV and the results CSV, and a terminal that
either is in the warehouse or is not. This closes THAT shape, and it refuses in the same places.

What it walks, per unit, and what refuses a row:

    GOVERNED_RUN.json       -> the unit's status, classification, code identity, campaign and the
                               terminal payload the client actually sent
    artifact digests        -> the predictions CSV on disk must hash to the terminal's `predictions`
                               artifact; otherwise the row's binding is NOT_BOUND, which is a
                               reported fact and never a pass
    live warehouse          -> the terminal_sha256 must appear in the warehouse verification
                               document for that campaign, and the warehouse's artifact rows must
                               equal the local terminal's; otherwise NOT_IN_WAREHOUSE
    independent recomputation -> model MAE recomputed in float64 from the retained rows, and the
                               matched naive persistence on THE SAME rows and the same scale; the
                               recomputed value must equal the warehouse's metric for that split and
                               horizon, or the row carries a PROBLEM and is not verified
    splits with no rows     -> train/validation metrics exist in the warehouse but their rows are not
                               retained, so they are emitted with a null recomputation and
                               `NOT_RECOMPUTABLE_NO_RETAINED_ROWS`. A null is an ABSENT measurement.

A `null` is never a numeric zero, and no row is ever marked verified because a receipt says so.

    python tools/df_governed_unit_closure.py --unit LABEL=OUT_DIR [--unit ...] \\
        --warehouse-verify WAREHOUSE_VERIFY.json --out TABLE.json [--markdown TABLE.md] \\
        [--disposition MECHANICAL_TRANSPORT_ONLY] [--reference-source TEXT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "governed_unit_closure_table.v1"
DEFAULT_DISPOSITION = "MECHANICAL_TRANSPORT_ONLY"
HORIZON_RE = re.compile(r"^Prediction_H(\d+)$")


class TableRefusal(ValueError):
    """A table that would mislead is not emitted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- pure parts


def read_predictions(path: Path) -> dict:
    """{'origin_column': name, 'rows': [{col: float|None}], 'horizons': [int]} from the CSV."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    horizons = sorted(int(m.group(1)) for m in map(HORIZON_RE.match, fields) if m)
    base = next((f for f in fields if f.endswith("_CLOSE")), None)
    return {"fields": fields, "rows": rows, "horizons": horizons, "base_column": base}


def _float(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def recompute(pred: dict, horizon: int) -> dict:
    """Model MAE and the matched naive on EXACTLY the same rows, in the target's own units.

    The naive is persistence: the last observed base value at the origin, carried forward to the
    horizon. Rows where any of the three values is absent are dropped from BOTH errors, so the
    populations cannot differ.
    """
    base_col, target_col, pred_col = pred["base_column"], f"Target_H{horizon}", f"Prediction_H{horizon}"
    model_abs, naive_abs = [], []
    for row in pred["rows"]:
        target = _float(row.get(target_col))
        predicted = _float(row.get(pred_col))
        base = _float(row.get(base_col)) if base_col else None
        if target is None or predicted is None or base is None:
            continue
        model_abs.append(abs(predicted - target))
        naive_abs.append(abs(base - target))
    if not model_abs:
        return {"n": 0, "model_mae": None, "naive_mae": None, "skill_vs_naive": None,
                "note": "no row carries a finite prediction, target and base together"}
    model = math.fsum(model_abs) / len(model_abs)
    naive = math.fsum(naive_abs) / len(naive_abs)
    skill = None if naive == 0.0 else 1.0 - model / naive
    return {"n": len(model_abs), "model_mae": model, "naive_mae": naive,
            "skill_vs_naive": skill,
            "note": None if skill is not None else "the naive error is exactly zero; skill is undefined"}


def warehouse_index(doc: dict) -> dict:
    """terminal_sha256 -> {'terminal': row, 'artifacts': {role: (sha256, bytes)}, 'metrics': {...}}"""
    out = {}
    for row in (doc.get("gov_terminal") or {}).get("rows", []) or []:
        out[row["terminal_sha256"]] = {"terminal": row, "artifacts": {}, "metrics": {}}
    for row in (doc.get("artifacts") or {}).get("rows", []) or []:
        entry = out.setdefault(row["terminal_sha256"], {"terminal": None, "artifacts": {}, "metrics": {}})
        entry["artifacts"][row["role"]] = (row["sha256"], row["bytes"])
    for key in ("metric_mae", "metrics"):
        for row in (doc.get(key) or {}).get("rows", []) or []:
            entry = out.setdefault(row["terminal_sha256"], {"terminal": None, "artifacts": {}, "metrics": {}})
            entry["metrics"][(row["metric"], row["split"], row.get("horizon"))] = row["value"]
    return out


# ---------------------------------------------------------------- the table


def unit_rows(label: str, out_dir: Path, wh: dict, *, disposition: str, reference_source: str,
              tolerance: float = 1e-6) -> list:
    receipt_path = out_dir / "GOVERNED_RUN.json"
    if not receipt_path.is_file():
        raise TableRefusal(f"{label}: no GOVERNED_RUN.json in {out_dir}; there is no unit to close")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    terminal = receipt.get("terminal") or {}
    status = receipt.get("status")
    local_artifacts = {a["role"]: (a["sha256"], a["bytes"]) for a in terminal.get("artifacts") or []}

    terminal_sha = next((sha for sha, entry in wh.items()
                         if (entry.get("terminal") or {}).get("campaign_sha256") == receipt.get("campaign_sha256")),
                        None)
    wh_entry = wh.get(terminal_sha) if terminal_sha else None
    if wh_entry is None:
        warehouse_state = "NOT_IN_WAREHOUSE"
    elif {k: v for k, v in wh_entry["artifacts"].items()} != local_artifacts and wh_entry["artifacts"]:
        warehouse_state = "WAREHOUSE_ARTIFACTS_DIFFER"
    else:
        warehouse_state = "WAREHOUSE_ACCEPTED"

    pred_files = sorted(out_dir.glob("*_prediction.csv"))
    if status == "COMPLETED" and not pred_files:
        raise TableRefusal(f"{label}: status COMPLETED but no *_prediction.csv is retained; "
                           "a forecast unit with no rows cannot be closed")
    if not pred_files:
        return [{
            "disposition": disposition, "unit": label, "task_horizon_split": f"{label}.NO_FORECAST_ROWS",
            "metric_and_scale": None, "binding": "NOT_BOUND", "warehouse": warehouse_state,
            "model_error": None, "naive_error": None, "model_population": None, "naive_population": None,
            "skill_vs_naive": None, "reference_value_and_source": "NOT_CARRIED",
            "comparability": "NOT_COMPARABLE", "comparability_reason":
                f"the unit ended {status} before producing forecast rows; there is no error to compare",
            "verified": False,
            "verified_reason": f"{label}: unit status {status}; no forecast rows exist (absent measurement, not zero)",
            "problems": [],
        }]

    pred_path = pred_files[0]
    on_disk = sha256_file(pred_path)
    declared = (local_artifacts.get("predictions") or (None, None))[0]
    if declared is None:
        binding = "NOT_BOUND"
    elif declared == on_disk:
        binding = "TERMINAL_ARTIFACT"
    else:
        binding = "DIGEST_MISMATCH"

    pred = read_predictions(pred_path)
    if pred["base_column"] is None:
        raise TableRefusal(f"{label}: the predictions CSV carries no base column, so no matched naive "
                           "can be built on the same rows")

    rows = []
    for horizon in pred["horizons"]:
        measured = recompute(pred, horizon)
        reported = None
        if wh_entry:
            reported = wh_entry["metrics"].get(("MAE", "test", horizon))
        problems = []
        if reported is not None and measured["model_mae"] is not None:
            if abs(reported - measured["model_mae"]) > max(tolerance, tolerance * abs(reported)) * 10:
                problems.append(f"warehouse MAE {reported!r} != recomputed {measured['model_mae']!r}")
        elif measured["model_mae"] is not None and wh_entry is not None:
            problems.append("the warehouse holds no MAE for this split and horizon")
        verified = (binding == "TERMINAL_ARTIFACT" and warehouse_state == "WAREHOUSE_ACCEPTED"
                    and not problems and measured["model_mae"] is not None)
        reason = "TRANSPORT_AND_RECOMPUTATION" if verified else "; ".join(
            problems or [f"binding {binding}, warehouse {warehouse_state}"])
        rows.append({
            "disposition": disposition, "unit": label,
            "task_horizon_split": f"{label}.H{horizon}.test",
            "metric_and_scale": f"MAE, target units of {pred['base_column']} (same scale for model and naive)",
            "binding": binding, "warehouse": warehouse_state,
            "model_error": measured["model_mae"], "naive_error": measured["naive_mae"],
            "model_population": measured["n"], "naive_population": measured["n"],
            "skill_vs_naive": measured["skill_vs_naive"],
            "warehouse_reported_model_error": reported,
            "reference_value_and_source": reference_source,
            "comparability": "NOT_COMPARABLE",
            "comparability_reason": (
                "a bounded mechanical unit: 2 epochs over 300 steps with all columns read as features "
                "by an explicit legacy migration. It proves custody and transport, not accuracy"),
            "verified": verified, "verified_reason": reason, "problems": problems,
            "note": measured["note"],
        })

    for split in ("train", "validation"):
        for horizon in pred["horizons"]:
            reported = wh_entry["metrics"].get(("MAE", split, horizon)) if wh_entry else None
            if reported is None:
                continue
            rows.append({
                "disposition": disposition, "unit": label,
                "task_horizon_split": f"{label}.H{horizon}.{split}",
                "metric_and_scale": "MAE, target units (as the run reported it)",
                "binding": "RECORD_DIGEST", "warehouse": warehouse_state,
                "model_error": None, "naive_error": None,
                "model_population": None, "naive_population": None, "skill_vs_naive": None,
                "warehouse_reported_model_error": reported,
                "reference_value_and_source": reference_source,
                "comparability": "NOT_COMPARABLE",
                "comparability_reason": "this split's rows are not retained, so no matched naive exists",
                "verified": False,
                "verified_reason": "NOT_RECOMPUTABLE_NO_RETAINED_ROWS: the warehouse carries a value "
                                   "whose rows this unit did not keep; the recomputation is ABSENT, not zero",
                "problems": [], "note": None,
            })
    return rows


def build(units: list, warehouse_doc: dict, *, disposition: str, reference_source: str) -> dict:
    wh = warehouse_index(warehouse_doc)
    rows = []
    for label, out_dir in units:
        rows.extend(unit_rows(label, Path(out_dir), wh, disposition=disposition,
                              reference_source=reference_source))
    if not rows:
        raise TableRefusal("no unit produced a row; an empty table is not a closure")
    verified = [r for r in rows if r["verified"]]
    return {
        "schema": SCHEMA, "generated_at": now_iso(), "disposition": disposition,
        "units": [{"label": label, "out_dir": str(out_dir)} for label, out_dir in units],
        "rows": rows,
        "summary": {"rows": len(rows), "verified": len(verified),
                    "rows_with_problems": sum(1 for r in rows if r["problems"]),
                    "null_is_absent_measurement": True},
        "reading": "skill = 1 - model_error/naive_error on identical rows, horizon and scale; a null is an "
                   "ABSENT measurement and never a zero. verified means the predictions digest binds to the "
                   "accepted terminal, the terminal is in the live warehouse, and the error was recomputed "
                   "here and agreed - it does not mean the number is scientifically usable.",
    }


def markdown(table: dict) -> str:
    head = ("| disposition | unit · horizon · split | metric & scale | binding | warehouse | model error | "
            "naive error (same rows, n) | skill vs naive | reference & source | comparability | verified |")
    sep = "|---|---|---|---|---|---:|---:|---:|---|---|---|"

    def cell(value):
        if value is None:
            return "null (absent)"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [f"# Closure table — {table['disposition']} — generated from artifacts", "",
             f"Generated {table['generated_at']}. {table['summary']['rows']} rows, "
             f"{table['summary']['verified']} verified, "
             f"{table['summary']['rows_with_problems']} with problems.", "",
             table["reading"], "", head, sep]
    for row in table["rows"]:
        naive = cell(row["naive_error"])
        if row["naive_error"] is not None:
            naive += f" (n={row['naive_population']})"
        lines.append("| " + " | ".join([
            row["disposition"], row["task_horizon_split"], cell(row["metric_and_scale"]),
            row["binding"], row["warehouse"], cell(row["model_error"]), naive,
            cell(row["skill_vs_naive"]), row["reference_value_and_source"],
            f"{row['comparability']}: {row['comparability_reason']}",
            ("YES (" + row["verified_reason"] + ")") if row["verified"] else ("NO (" + row["verified_reason"] + ")"),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", action="append", required=True, metavar="LABEL=OUT_DIR")
    parser.add_argument("--warehouse-verify", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--disposition", default=DEFAULT_DISPOSITION)
    parser.add_argument("--reference-source", default="NOT_CARRIED")
    args = parser.parse_args(argv)

    units = []
    for item in args.unit:
        if "=" not in item:
            parser.error(f"--unit expects LABEL=OUT_DIR, got {item!r}")
        label, out_dir = item.split("=", 1)
        units.append((label, out_dir))
    doc = json.loads(args.warehouse_verify.read_text(encoding="utf-8"))
    table = build(units, doc, disposition=args.disposition, reference_source=args.reference_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(table), encoding="utf-8")
    print(f"{args.out}: {table['summary']['rows']} rows, {table['summary']['verified']} verified, "
          f"{table['summary']['rows_with_problems']} with problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
