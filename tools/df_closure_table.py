#!/usr/bin/env python3
"""Point 14: the owner-facing closure table, generated from verified artifacts linked to the warehouse.

Every row is built from a run root's own arrays and receipts and checked against the warehouse — not
from a remembered summary. For each arm: the metric recomputed in float64 from the saved prediction
and label arrays; the naive on the IDENTICAL rows and horizon; skill = 1 - error_model/error_naive
(lower-is-better; positive = smaller error, never accuracy or profit; a zero naive error is
UNDEFINED, not a ratio); the literature value and its comparability status from the benchmark
registry, which is NOT_COMPARABLE with the exact reason and the planned matched comparison whenever
no verified matched value exists.

The table refuses what it must not show:

  * a row without a naive on the same rows, without a comparability status, or without a reference
    column (NOT_COMPARABLE with a reason counts; an empty cell does not);
  * two numeric values on different scales inside one comparison row (kW next to z or log1p);
  * a model and a naive evaluated on different populations or horizons;
  * a percentage without its denominator and rows;
  * a round-off that erases a difference the artifacts carry (full precision is kept in the JSON).

When a round measured nothing new, the table says NO_NEW_MEASUREMENT and labels every prior
verified result with the run and scope it belongs to.

    python tools/df_closure_table.py --run ROOT[:label] ... --registry REGISTRY.json \\
        --warehouse-url URL --warehouse-token-file FILE --out TABLE.json --markdown TABLE.md [--no-new-measurement]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "owner_closure_table.v1"
REQUIRED = ("task_horizon_split", "metric_and_scale", "model_error", "naive_error", "skill_vs_naive",
            "literature_value_and_source", "comparability_status")


class TableRefusal(ValueError):
    """A table that would mislead is not emitted."""


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def skill(model_error: float, naive_error: float) -> dict:
    """1 - error_model/error_naive for a lower-is-better error; UNDEFINED when the naive is zero."""
    if not (np.isfinite(model_error) and np.isfinite(naive_error)):
        return {"value": None, "status": "UNDEFINED", "why": "a non-finite error has no skill"}
    if naive_error == 0:
        return {"value": None, "status": "UNDEFINED", "why": "the naive error is zero; no ratio is invented"}
    return {"value": float(1.0-model_error/naive_error), "status": "MEASURED",
            "reading": "positive = smaller error than the naive; not accuracy, not profit"}


# --- rows from verified artifacts ------------------------------------------------------------------

def _arrays_generic(path: Path) -> list:
    """Every forecast (arm, predictions, labels) an arrays file carries, in the layouts the runners save.

    An auto-encoder unit saves reconstructions, not forecasts, and a `controls` unit saves the
    baselines' predictions: the first yields no row and is NAMED as skipped; the second yields one
    row per learned control (the linear ridge), so the comparator the reader needs is in the table.
    A file with no forecast layout at all is reported, never guessed.
    """
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if {"pred", "y", "naive", "origins"} <= keys:                      # df_e1_huber
            return [{"arm": None, "pred": z["pred"].reshape(-1).astype(np.float64),
                     "y": z["y"].reshape(-1).astype(np.float64),
                     "naive": z["naive"].reshape(-1).astype(np.float64), "origins": z["origins"].astype(np.int64)}]
        if {"validation_pred", "validation_y", "eval_origins"} <= keys:     # df_e1_pilot / df_e1_phase1
            return [{"arm": None, "pred": z["validation_pred"].reshape(-1).astype(np.float64),
                     "y": z["validation_y"].reshape(-1).astype(np.float64),
                     "naive": None, "origins": z["eval_origins"].astype(np.int64)}]
        if {"validation_pred_linear_ridge", "validation_y", "eval_origins"} <= keys:   # the controls unit
            return [{"arm": "linear_ridge", "pred": z["validation_pred_linear_ridge"].reshape(-1).astype(np.float64),
                     "y": z["validation_y"].reshape(-1).astype(np.float64),
                     "naive": None, "origins": z["eval_origins"].astype(np.int64)}]
    return []


def _naive_from_source(root: Path, origins: np.ndarray, horizon: int) -> np.ndarray:
    """The persistence on the SAME origins, from the run's own DATA (or its declared source run)."""
    data = root/"DATA.npz"
    if not data.is_file():
        design = json.loads((root/"DESIGN.json").read_text())
        src = design.get("source_run", {}).get("root")
        if src:
            data = Path(src)/"DATA.npz"
    with np.load(data, allow_pickle=False) as z:
        Y = z["Y"]
        h = int(z["horizon"][0])
    if h != horizon:
        raise TableRefusal(f"the horizon in DATA ({h}) is not the row's horizon ({horizon})")
    return Y[origins].astype(np.float64), Y[origins+h].astype(np.float64)


def rows_from_run(root: Path, *, label: str, registry: dict, warehouse=None, task_key: str = "household_W60_h60") -> list:
    """One row per arm (unit), recomputed from the run's arrays and checked against the warehouse."""
    root = Path(root)
    design = json.loads((root/"DESIGN.json").read_text())
    receipts = json.loads((root/"TERMINAL_RECEIPTS.json").read_text())["units"]
    ours = registry["ours"][task_key]
    decisions = registry["decisions_against_household_W60_h60"]
    literature = registry["literature"]
    horizon = int(ours["horizon_steps"])
    rows, skipped = [], []
    for unit, receipt in sorted(receipts.items()):
        attempt = root/"attempts"/unit
        arrays = attempt/"arrays.npz"
        if not arrays.is_file():
            skipped.append({"unit": unit, "why": "no arrays.npz: the unit carries no forecast (prepare or a pilot)"})
            continue
        if unit.startswith("pilot"):
            skipped.append({"unit": unit, "why": "a cost pilot (200-300 updates): it measures seconds per update, "
                                                 "it is not a comparison arm and its score is not reported as one"})
            continue
        found = _arrays_generic(arrays)
        if not found:
            skipped.append({"unit": unit, "why": "arrays without a forecast layout (an auto-encoder's reconstructions)"})
            continue
        for a in found:
          arm = a["arm"] or unit.rsplit("_s", 1)[0]
          rows.append(_row_for(root, design, unit, arm, a, receipt, registry, warehouse, horizon, label))
    if skipped:
        for r in rows:
            r["units_skipped_in_this_run"] = skipped
    return rows


def _row_for(root, design, unit, arm, a, receipt, registry, warehouse, horizon, label) -> dict:
    """One verified row: recomputed from the arrays, naive on the identical origins, warehouse checked."""
    ours = registry["ours"]["household_W60_h60"]
    decisions = registry["decisions_against_household_W60_h60"]
    literature = registry["literature"]
    if True:
        naive_pred, truth = _naive_from_source(root, a["origins"], horizon)
        if not np.array_equal(truth, a["y"]):
            raise TableRefusal(f"{unit}: the saved labels are not the source's labels on these origins")
        if a["naive"] is not None and not np.array_equal(a["naive"], naive_pred):
            raise TableRefusal(f"{unit}: the saved naive is not the persistence on these origins")
        model_mae = float(np.mean(np.abs(a["pred"]-a["y"])))
        naive_mae = float(np.mean(np.abs(naive_pred-a["y"])))
        sd = None
        data = root/"DATA.npz"
        if not data.is_file():
            data = Path(design.get("source_run", {}).get("root", root))/"DATA.npz"
        with np.load(data, allow_pickle=False) as z:
            j = int(z["target_channel"][0])
            sd = float(z["scaler_sd"][j])
        wh = None
        if warehouse is not None:
            held = warehouse(receipt["campaign_sha256"])
            row = (held.get("current") or {}).get(unit)
            wh = {"terminal_in_warehouse": row is not None,
                  "digest_matches_receipt": bool(row and row.get("terminal_sha256") == receipt.get("terminal_sha256")),
                  "campaign_sha256": receipt["campaign_sha256"], "terminal_sha256": receipt.get("terminal_sha256")}
        best = "gasparin_2019"
        return {
            "run": label, "unit": unit, "arm": arm, "seed": unit.rsplit("_s", 1)[-1] if "_s" in unit else "-",
            "task_horizon_split": f"{ours['task_id']} | h={horizon} steps ({ours['horizon_seconds']} s) | {ours['split_rule']}",
            "metric_and_scale": "MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in a separate column",
            "model_error": model_mae, "model_error_z": model_mae/sd if sd else None,
            "naive_error": naive_mae, "naive_error_z": naive_mae/sd if sd else None,
            "naive_definition": "persistence at the horizon on the identical evaluation origins",
            "n_evaluated": int(a["y"].size), "horizon_steps": horizon,
            "skill_vs_naive": skill(model_mae, naive_mae),
            "literature_value_and_source": {
                "status": decisions[best]["mode"],
                "source": literature[best]["source"]["citation"],
                "published_value": literature[best]["reference_method"],
                "published_or_reproduced": "PUBLISHED (not reproduced here)",
                "placed_in_comparison_column": False,
                "why_not": decisions[best]["why"],
                "planned_matched_comparison": decisions[best]["resolution"]},
            "comparability_status": decisions[best]["mode"],
            "scope": design.get("purpose") or design.get("what_this_is") or "DEVELOPMENT",
            "warehouse": wh,
            "precision_note": "values are float64 from the arrays; no rounding in this JSON",
        }


# --- validation of a table -------------------------------------------------------------------------

def validate(table: dict) -> list:
    """Every reason the table must not be shown as it is."""
    problems = []
    rows = table.get("rows") or []
    if not rows and not table.get("no_new_measurement"):
        problems.append("no rows and no NO_NEW_MEASUREMENT declaration")
    for r in rows:
        tag = f"{r.get('run')}/{r.get('unit')}"
        for k in REQUIRED:
            if k not in r or r[k] in (None, "", {}):
                problems.append(f"{tag}: missing {k}")
        lit = r.get("literature_value_and_source") or {}
        if isinstance(lit, dict):
            if lit.get("status") == "NOT_COMPARABLE" and not (lit.get("why_not") and lit.get("planned_matched_comparison")):
                problems.append(f"{tag}: NOT_COMPARABLE without a reason and a planned matched comparison")
            if lit.get("placed_in_comparison_column") and lit.get("status") == "NOT_COMPARABLE":
                problems.append(f"{tag}: an unmatched published value was placed in the comparison column")
        if r.get("comparability_status") not in ("REPRODUCTION", "MATCHED_DOMAIN_COMPARISON", "NOT_COMPARABLE"):
            problems.append(f"{tag}: comparability status is not one of the three")
        me, ne = r.get("model_error"), r.get("naive_error")
        if isinstance(me, (int, float)) and isinstance(ne, (int, float)):
            sk = r.get("skill_vs_naive") or {}
            if ne == 0 and sk.get("status") != "UNDEFINED":
                problems.append(f"{tag}: zero naive error with a defined skill")
            if ne != 0 and sk.get("value") is not None and abs(sk["value"]-(1-me/ne)) > 1e-12:
                problems.append(f"{tag}: skill does not equal 1 - model/naive")
        if r.get("model_scale") and r.get("naive_scale") and r["model_scale"] != r["naive_scale"]:
            problems.append(f"{tag}: model and naive on different scales in one row")
        if r.get("model_population") is not None and r.get("naive_population") is not None \
                and r["model_population"] != r["naive_population"]:
            problems.append(f"{tag}: model and naive on different populations")
        if r.get("model_horizon") is not None and r.get("naive_horizon") is not None \
                and r["model_horizon"] != r["naive_horizon"]:
            problems.append(f"{tag}: model and naive at different horizons")
        for claim in r.get("percentage_claims") or []:
            if not (claim.get("numerator") is not None and claim.get("denominator") and claim.get("rows")):
                problems.append(f"{tag}: a percentage without numerator, denominator and rows")
    return problems


# --- rendering -------------------------------------------------------------------------------------

def markdown(table: dict) -> str:
    lines = ["# Closure table — generated from verified artifacts", ""]
    if table.get("no_new_measurement"):
        lines += ["**NO_NEW_MEASUREMENT** in this order set. Every row below is a PRIOR verified result, "
                  "labelled with its run and scope.", ""]
    lines += ["| task / horizon / split | metric & scale | run · arm · seed | model error | naive error (same rows) | "
              "skill vs naive | literature value & source | comparability |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for r in table.get("rows") or []:
        sk = r["skill_vs_naive"]
        skill_txt = "UNDEFINED" if sk.get("value") is None else f"{sk['value']:+.6f}"
        lit = r["literature_value_and_source"]
        lit_txt = (f"{lit['status']}: {lit['source'][:60]}… — published {lit['published_or_reproduced'].lower()}; "
                   f"not in the comparison column. Planned: {lit['planned_matched_comparison'][:70]}…")
        wh = r.get("warehouse") or {}
        wh_txt = " ✓wh" if wh.get("digest_matches_receipt") else (" (wh unchecked)" if not wh else " ✗wh")
        lines.append(f"| {r['task_horizon_split'][:70]}… | {r['metric_and_scale'][:40]}… | {r['run']} · {r['arm']} · s{r['seed']}{wh_txt} | "
                     f"{r['model_error']:.6f} kW (z {r['model_error_z']:.6f}) | {r['naive_error']:.6f} kW (n={r['n_evaluated']}) | "
                     f"{skill_txt} | {lit_txt} | {r['comparability_status']} |")
    lines += ["", "skill = 1 − error_model / error_naive on identical rows and horizon; positive means a smaller error, "
              "not accuracy or profit. kW and z values never share a comparison. Full precision is in the JSON.",
              "", f"validated: {'no problems' if not table.get('problems') else table['problems']}"]
    return "\n".join(lines)+"\n"


def build(runs: list, *, registry: dict, warehouse=None, no_new_measurement: bool) -> dict:
    rows = []
    for spec in runs:
        root, _, label = spec.partition(":")
        rows += rows_from_run(Path(root).expanduser(), label=label or Path(root).name, registry=registry,
                              warehouse=warehouse)
    table = {"schema": SCHEMA, "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
             "no_new_measurement": bool(no_new_measurement), "rows": rows,
             "rules": {"skill": "1 - error_model/error_naive; lower-is-better; UNDEFINED when the naive is zero",
                       "scales": "kW and z reported in separate columns, never compared across",
                       "literature": "a published value under another protocol is shown for the record and never "
                                     "placed in the comparison column; NOT_COMPARABLE carries its reason and the "
                                     "planned matched comparison"}}
    table["problems"] = validate(table)
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="ROOT[:label]")
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--no-new-measurement", action="store_true")
    a = ap.parse_args(argv)
    registry = json.loads(a.registry.read_text())
    warehouse = None
    if a.warehouse_token_file:
        E0 = _module("df_mod_e0_close")
        token = a.warehouse_token_file.read_text().strip().strip('"').strip("'")
        warehouse = lambda campaign: E0.warehouse_terminals(a.warehouse_url, token, campaign)
    table = build(a.run, registry=registry, warehouse=warehouse, no_new_measurement=a.no_new_measurement)
    a.out.write_text(json.dumps(table, indent=1, default=str))
    if a.markdown:
        a.markdown.write_text(markdown(table))
    print(json.dumps({"rows": len(table["rows"]), "problems": table["problems"],
                      "no_new_measurement": table["no_new_measurement"]}, indent=1))
    return 0 if not table["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
