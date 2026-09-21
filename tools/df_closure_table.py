#!/usr/bin/env python3
"""RP66: the owner-facing closure table, built along the WHOLE chain — never from a receipt badge.

Musashi's probe against the previous version: on fabricated files, changing the predictions moved
the MAE from 2 to 0 with the same receipt, both tables had zero problems and a matching warehouse
badge; a missing warehouse terminal, NaN predictions and a missing forecast file also passed, the
last one producing an empty NO_NEW_MEASUREMENT table although a forecast unit was declared. The
verifier trusted a terminal digest and read nothing else.

The chain this version walks, per unit, and what refuses it:

    registered design      -> the unit's ROLE (forecast / pretraining / controls / cost pilot /
                              preparation) comes from the sealed design, never from a file name
    attempt + terminal     -> the terminal payload the client sent, and its accepted receipt
    artifact digests       -> the arrays' sha256 must equal the terminal's `predictions` artifact
                              (huber-style terminals) or the record's `arrays_sha256`; a terminal
                              that carries neither is reported as NOT_BOUND — a fact, not a pass
    arrays and source DATA -> finite, nonempty, exactly the expected evaluation population, unique
                              origins equal to DATA's, labels identical to Y[origins+h], horizon and
                              target identities, scaler sd finite and > 0
    independent metrics    -> MAE recomputed in float64 and, when the record carries a score, equal
                              to it (the existing verifier's rule); the naive on the same origins
    accounting/warehouse   -> the terminal must exist in the warehouse with the receipt's digest, and
                              its artifact rows (when any) must equal the local ones

A declared forecast unit with no arrays is a PROBLEM, whatever NO_NEW_MEASUREMENT says about the
round. Model/naive population, horizon and scale are PRODUCED on every row and checked, not optional
keys a test might add. Retained history is re-verified without fitting anything.

    python tools/df_closure_table.py --run ROOT[:label] ... --registry REGISTRY.json \\
        --warehouse-url URL --warehouse-token-file FILE --out TABLE.json --markdown TABLE.md [--no-new-measurement]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "owner_closure_table.v2"
REQUIRED = ("task_horizon_split", "metric_and_scale", "model_error", "naive_error", "skill_vs_naive",
            "literature_value_and_source", "comparability_status", "model_population", "naive_population",
            "model_horizon", "naive_horizon", "model_scale", "naive_scale", "binding")
FORECAST_ROLES = ("forecast", "control_forecast")


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


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def skill(model_error, naive_error) -> dict:
    if not (isinstance(model_error, (int, float)) and isinstance(naive_error, (int, float))) \
            or not (np.isfinite(model_error) and np.isfinite(naive_error)):
        return {"value": None, "status": "UNDEFINED", "why": "a non-finite error has no skill"}
    if naive_error == 0:
        return {"value": None, "status": "UNDEFINED", "why": "the naive error is zero; no ratio is invented"}
    return {"value": float(1.0-model_error/naive_error), "status": "MEASURED",
            "reading": "positive = smaller error than the naive; not accuracy, not profit"}


# --- roles from the registered design -------------------------------------------------------------

def unit_roles(design: dict) -> dict:
    """Every registered unit with its role — from the sealed design's own vocabulary."""
    roles = {"prepare": "preparation"}
    for c in design.get("pilots") or []:
        roles[c["cell_id"]] = "cost_pilot"
    for c in design.get("cells") or []:
        kind = c.get("kind")
        if kind == "ae":
            roles[c["cell_id"]] = "pretraining"
        elif kind == "controls":
            roles[c["cell_id"]] = "control_forecast"
        elif kind == "fit" or c.get("arm") is not None:
            roles[c["cell_id"]] = "forecast"
        else:
            roles[c["cell_id"]] = "unknown"
    return roles


def _source_data(root: Path, design: dict) -> tuple:
    data = root/"DATA.npz"
    data_json = root/"DATA.json"
    if not data.is_file():
        src = (design.get("source_run") or {}).get("root")
        if not src:
            raise TableRefusal(f"{root}: no DATA.npz and no source_run to read it from")
        data, data_json = Path(src)/"DATA.npz", Path(src)/"DATA.json"
    with np.load(data, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(data_json.read_text()) if data_json.is_file() else {}
    return d, meta, data


def _arrays(path: Path) -> list:
    """Every forecast the arrays file carries, with its arm name, in the layouts the runners save."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if {"pred", "y", "naive", "origins"} <= keys:                      # df_e1_huber
            return [{"arm": None, "pred": z["pred"], "y": z["y"], "naive": z["naive"], "origins": z["origins"]}]
        if {"validation_pred", "validation_y", "eval_origins"} <= keys:     # df_e1_pilot / df_e1_phase1 / df_e1_block
            return [{"arm": None, "pred": z["validation_pred"], "y": z["validation_y"], "naive": None,
                     "origins": z["eval_origins"]}]
        if {"validation_pred_linear_ridge", "validation_y", "eval_origins"} <= keys:   # the controls unit
            return [{"arm": "linear_ridge", "pred": z["validation_pred_linear_ridge"], "y": z["validation_y"],
                     "naive": None, "origins": z["eval_origins"]}]
    return []


def _record(attempt: Path) -> dict | None:
    p = attempt/"cell.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _terminal_payload(root: Path, unit: str) -> dict | None:
    p = root/"TERMINALS"/f"{unit}.json"
    if not p.is_file():
        return None
    body = json.loads(p.read_text())
    return body.get("terminal") if "terminal" in body and "artifacts" not in body else body


# --- one unit along the chain -------------------------------------------------------------------------

def verify_unit(root: Path, design: dict, unit: str, role: str, receipt: dict, data: dict, meta: dict,
                *, warehouse=None) -> list:
    """Every forecast row a unit yields, each carrying its problems — a row with problems is not clean."""
    h = int(data["horizon"][0])
    j = int(data["target_channel"][0])
    expected_origins = data["eval_origins"].astype(np.int64)
    Y = data["Y"].astype(np.float64)
    sd = float(data["scaler_sd"][j])
    target_name = (meta.get("input_columns") or [None]*(j+1))[j]
    attempt = root/"attempts"/unit
    problems = []
    arrays_path = attempt/"arrays.npz"
    if not arrays_path.is_file():
        return [_row(root, design, unit, role, None, receipt, problems+[f"{unit}: a declared {role} unit has no arrays.npz — "
                                                                       "an expected forecast is missing, not absent"],
                     data, meta, None, None, None)]
    found = _arrays(arrays_path)
    if not found:
        return [_row(root, design, unit, role, None, receipt, problems+[f"{unit}: arrays.npz carries no forecast layout"],
                     data, meta, None, None, None)]
    arrays_sha = sha_file(arrays_path)
    record = _record(attempt)
    terminal = _terminal_payload(root, unit)
    # --- artifact binding: arrays -> terminal (or record) -----------------------------------------------
    binding = {"level": "NOT_BOUND", "arrays_sha256": arrays_sha}
    term_pred = next((a for a in (terminal or {}).get("artifacts") or [] if a.get("role") == "predictions"), None)
    if term_pred:
        binding["level"] = "TERMINAL_ARTIFACT"
        binding["terminal_predictions_sha256"] = term_pred.get("sha256")
        if term_pred.get("sha256") != arrays_sha:
            problems.append(f"{unit}: the arrays on disk ({arrays_sha[:12]}) are not the predictions artifact the terminal "
                            f"declared ({str(term_pred.get('sha256'))[:12]}): CHANGED ARRAYS")
    elif record and record.get("arrays_sha256"):
        binding["level"] = "RECORD_DIGEST"
        binding["record_arrays_sha256"] = record["arrays_sha256"]
        if record["arrays_sha256"] != arrays_sha:
            problems.append(f"{unit}: the arrays on disk are not the ones the record digested: CHANGED ARRAYS")
    else:
        problems.append(f"{unit}: neither the terminal nor the record carries a digest of the arrays: the scored "
                        f"arrays are NOT BOUND to the accepted terminal (binding level NOT_BOUND)")
    # --- warehouse: terminal present with the receipt's digest; artifact rows equal to the local ones ----
    wh = {"checked": warehouse is not None}
    if warehouse is not None:
        held = warehouse(receipt["campaign_sha256"]) or {}
        row = (held.get("current") or {}).get(unit)
        if row is None:
            problems.append(f"{unit}: the warehouse holds NO terminal for this unit under campaign {receipt['campaign_sha256'][:12]}")
            wh.update(terminal_in_warehouse=False)
        else:
            wh.update(terminal_in_warehouse=True, digest_matches_receipt=row.get("terminal_sha256") == receipt.get("terminal_sha256"),
                      status=row.get("status"), artifact_rows=len(row.get("artifacts") or []))
            if not wh["digest_matches_receipt"]:
                problems.append(f"{unit}: the warehouse terminal digest differs from the client's receipt")
            wh_pred = next((a for a in row.get("artifacts") or [] if a.get("role") == "predictions"), None)
            if wh_pred and wh_pred.get("sha256") != arrays_sha:
                problems.append(f"{unit}: the warehouse's predictions artifact digest is not the arrays on disk")
            if row.get("status") != "COMPLETED":
                problems.append(f"{unit}: the warehouse terminal is {row.get('status')}, not COMPLETED")
    rows = []
    for a in found:
        arm = a["arm"] or unit.rsplit("_s", 1)[0]
        pred = np.asarray(a["pred"], dtype=np.float64).reshape(-1)
        y = np.asarray(a["y"], dtype=np.float64).reshape(-1)
        origins = np.asarray(a["origins"]).reshape(-1).astype(np.int64)
        p = list(problems)
        # --- arrays and source DATA -----------------------------------------------------------------------
        if pred.size == 0 or y.size == 0:
            p.append(f"{unit}: empty arrays")
        if pred.shape != y.shape or origins.shape != y.shape:
            p.append(f"{unit}: predictions, labels and origins differ in length ({pred.size}, {y.size}, {origins.size})")
        if pred.size and not np.isfinite(pred).all():
            p.append(f"{unit}: {int((~np.isfinite(pred)).sum())} non-finite predictions")
        if y.size and not np.isfinite(y).all():
            p.append(f"{unit}: non-finite labels")
        if np.unique(origins).size != origins.size:
            p.append(f"{unit}: duplicated evaluation origins")
        if origins.size != expected_origins.size or not np.array_equal(np.sort(origins), np.sort(expected_origins)):
            p.append(f"{unit}: the evaluation population is not DATA's ({origins.size} vs {expected_origins.size} origins)")
        elif not np.array_equal(origins, expected_origins):
            p.append(f"{unit}: the evaluation origins are DATA's but in another order")
        truth = Y[origins+h] if origins.size and origins.max()+h < Y.size else None
        if truth is None:
            p.append(f"{unit}: origins reach beyond the source series at horizon {h}")
        elif y.shape == truth.shape and not np.array_equal(truth, y):
            p.append(f"{unit}: the saved labels are not Y[origin + {h}] of the source DATA (label/horizon identity)")
        naive = Y[origins] if origins.size and origins.max() < Y.size else None
        if a["naive"] is not None and naive is not None and not np.array_equal(np.asarray(a["naive"], dtype=np.float64).reshape(-1), naive):
            p.append(f"{unit}: the saved naive is not the persistence on these origins")
        if not (np.isfinite(sd) and sd > 0):
            p.append(f"{unit}: the scaler sd for the target is not finite and positive ({sd})")
        clean = not p and truth is not None and naive is not None
        model_mae = float(np.mean(np.abs(pred-y))) if clean else None
        naive_mae = float(np.mean(np.abs(naive-y))) if clean else None
        # --- independent metric against the record's own score, when it carries one --------------------
        if clean and record:
            scores = ((record.get("scores") or {}).get("validation") or {}).get("model") or {}
            stored = scores.get("mae_kw", scores.get("mae_mean"))
            if a["arm"] is None and isinstance(stored, (int, float)) and abs(stored-model_mae) > 1e-9:
                p.append(f"{unit}: the recomputed MAE {model_mae:.9f} differs from the record's {stored:.9f}")
                clean = False
        rows.append(_row(root, design, unit, role, arm, receipt, p, data, meta,
                         model_mae if clean else None, naive_mae if clean else None,
                         {"binding": binding, "warehouse": wh, "n": int(y.size), "target": target_name,
                          "record_score_checked": bool(record and a["arm"] is None)}))
    return rows


def _row(root, design, unit, role, arm, receipt, problems, data, meta, model_mae, naive_mae, extra) -> dict:
    B = _module("df_benchmark_contract")
    h = int(data["horizon"][0])
    j = int(data["target_channel"][0])
    sd = float(data["scaler_sd"][j])
    reg = _registry_cache()
    ours = reg["ours"]["household_W60_h60"]
    best = "gasparin_2019"
    decision = reg["decisions_against_household_W60_h60"][best]
    lit = reg["literature"][best]
    row = {"run": root.name, "unit": unit, "role": role, "arm": arm or unit.rsplit("_s", 1)[0],
           "seed": unit.rsplit("_s", 1)[-1] if "_s" in unit else "-",
           "task_horizon_split": f"{ours['task_id']} | h={h} steps ({ours['horizon_seconds']} s) | {ours['split_rule']}",
           "metric_and_scale": "MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column",
           "model_error": model_mae, "model_error_z": (model_mae/sd) if (model_mae is not None and sd > 0) else None,
           "naive_error": naive_mae, "naive_error_z": (naive_mae/sd) if (naive_mae is not None and sd > 0) else None,
           "naive_definition": "persistence at the horizon on the identical evaluation origins",
           "model_population": (extra or {}).get("n"), "naive_population": (extra or {}).get("n"),
           "model_horizon": h, "naive_horizon": h, "model_scale": "kW", "naive_scale": "kW",
           "target": (extra or {}).get("target"), "n_evaluated": (extra or {}).get("n"), "horizon_steps": h,
           "skill_vs_naive": skill(model_mae, naive_mae),
           "literature_value_and_source": {"status": decision["mode"], "comparator_state": decision.get("comparator_state", "NONE"),
                                           "source": lit["source"]["citation"], "published_value": lit["reference_method"],
                                           "published_or_reproduced": "PUBLISHED (not reproduced here)",
                                           "placed_in_comparison_column": False, "why_not": decision["why"],
                                           "planned_matched_comparison": decision["resolution"]},
           "comparability_status": decision["mode"],
           "scope": design.get("purpose") or design.get("what_this_is") or "DEVELOPMENT",
           "binding": (extra or {}).get("binding", {"level": "NOT_BOUND"}),
           "warehouse": (extra or {}).get("warehouse", {"checked": False}),
           "record_score_checked": (extra or {}).get("record_score_checked", False),
           "problems": problems, "verified": not problems,
           "precision_note": "values are float64 from the arrays; no rounding in this JSON"}
    return row


_REG = {}


def _registry_cache() -> dict:
    if not _REG:
        _REG.update(_module("df_benchmark_contract").registry())
    return _REG


def rows_from_run(root: Path, *, label: str, registry: dict, warehouse=None) -> list:
    root = Path(root)
    _REG.clear(); _REG.update(registry)
    design = json.loads((root/"DESIGN.json").read_text())
    receipts = (json.loads((root/"TERMINAL_RECEIPTS.json").read_text()) or {}).get("units") or {}
    roles = unit_roles(design)
    data, meta, _ = _source_data(root, design)
    rows, skipped = [], []
    for unit, role in sorted(roles.items()):
        if role not in FORECAST_ROLES:
            skipped.append({"unit": unit, "role": role, "why": "not a forecast unit by its registered role"})
            continue
        receipt = receipts.get(unit)
        if receipt is None:
            rows.append(_row(root, design, unit, role, None, {"campaign_sha256": ""}, [f"{unit}: a registered {role} unit "
                             "has NO accepted terminal receipt"], data, meta, None, None, None))
            continue
        rows += verify_unit(root, design, unit, role, receipt, data, meta, warehouse=warehouse)
    strangers = sorted(set(receipts)-set(roles))
    for u in strangers:
        rows.append(_row(root, design, u, "unregistered", None, receipts[u], [f"{u}: a terminal exists for a unit the "
                         "design does not register"], data, meta, None, None, None))
    for r in rows:
        r["run"] = label
        r["units_not_scored_by_role"] = skipped
    return rows


# --- validation and rendering ---------------------------------------------------------------------------

def validate(table: dict) -> list:
    problems = []
    rows = table.get("rows") or []
    if not rows and not table.get("no_new_measurement"):
        problems.append("no rows and no NO_NEW_MEASUREMENT declaration")
    for r in rows:
        tag = f"{r.get('run')}/{r.get('unit')}"
        problems += [f"{tag}: {p}" if not p.startswith(r.get("unit", "")) else f"{r.get('run')}/{p}" for p in r.get("problems") or []]
        for k in REQUIRED:
            if k not in r or r[k] in (None, "", {}):
                if k in ("model_error", "naive_error", "model_population", "naive_population") and r.get("problems"):
                    continue                                  # already refused above, with its reason
                problems.append(f"{tag}: missing {k}")
        lit = r.get("literature_value_and_source") or {}
        if isinstance(lit, dict):
            if lit.get("status") == "NOT_COMPARABLE" and not (lit.get("why_not") and lit.get("planned_matched_comparison")):
                problems.append(f"{tag}: NOT_COMPARABLE without a reason and a planned matched comparison")
            if lit.get("placed_in_comparison_column") and lit.get("status") != "MATCHED_DOMAIN_COMPARISON":
                problems.append(f"{tag}: a value that is not a verified matched comparator was placed in the comparison column")
        if r.get("comparability_status") not in ("REPRODUCTION", "MATCHED_DOMAIN_COMPARISON", "NOT_COMPARABLE"):
            problems.append(f"{tag}: comparability status is not one of the three")
        me, ne = r.get("model_error"), r.get("naive_error")
        if isinstance(me, (int, float)) and isinstance(ne, (int, float)):
            sk = r.get("skill_vs_naive") or {}
            if ne == 0 and sk.get("status") != "UNDEFINED":
                problems.append(f"{tag}: zero naive error with a defined skill")
            if ne != 0 and sk.get("value") is not None and abs(sk["value"]-(1-me/ne)) > 1e-12:
                problems.append(f"{tag}: skill does not equal 1 - model/naive")
        if r.get("model_scale") != r.get("naive_scale"):
            problems.append(f"{tag}: model and naive on different scales in one row")
        if r.get("model_population") != r.get("naive_population"):
            problems.append(f"{tag}: model and naive on different populations")
        if r.get("model_horizon") != r.get("naive_horizon"):
            problems.append(f"{tag}: model and naive at different horizons")
        for claim in r.get("percentage_claims") or []:
            if not (claim.get("numerator") is not None and claim.get("denominator") and claim.get("rows")):
                problems.append(f"{tag}: a percentage without numerator, denominator and rows")
    return problems


def markdown(table: dict) -> str:
    lines = ["# Closure table — generated from verified artifacts along the whole chain", ""]
    if table.get("no_new_measurement"):
        lines += ["**NO_NEW_MEASUREMENT** in this order set. Every row below is a PRIOR verified result, labelled with its run "
                  "and scope; an expected forecast that is missing is a PROBLEM, not an absence.", ""]
    lines += ["| task / horizon / split | metric & scale | run · arm · seed | binding | model error | naive error (same rows, n) | "
              "skill vs naive | literature value & source | comparability | verified |",
              "|---|---|---|---|---:|---:|---:|---|---|---|"]
    for r in table.get("rows") or []:
        sk = r["skill_vs_naive"]
        skill_txt = "UNDEFINED" if sk.get("value") is None else f"{sk['value']:+.6f}"
        lit = r["literature_value_and_source"]
        lit_txt = (f"{lit['status']} ({lit.get('comparator_state')}): {lit['source'][:50]}… — {lit['published_or_reproduced'].lower()}; "
                   f"not in the comparison column")
        me = "—" if r["model_error"] is None else f"{r['model_error']:.6f} kW (z {r['model_error_z']:.6f})"
        ne = "—" if r["naive_error"] is None else f"{r['naive_error']:.6f} kW (n={r['n_evaluated']})"
        ver = "yes" if r.get("verified") else "NO: " + "; ".join(r.get("problems") or [])[:120]
        lines.append(f"| {r['task_horizon_split'][:60]}… | {r['metric_and_scale'][:36]}… | {r['run']} · {r['arm']} · s{r['seed']} | "
                     f"{r['binding'].get('level')} | {me} | {ne} | {skill_txt} | {lit_txt} | {r['comparability_status']} | {ver} |")
    lines += ["", "skill = 1 − error_model / error_naive on identical rows and horizon; positive means a smaller error, not accuracy "
              "or profit. kW and z never share a comparison. Binding: TERMINAL_ARTIFACT = arrays digest in the accepted terminal; "
              "RECORD_DIGEST = digest in the unit's record only; NOT_BOUND = no digest links the arrays to the terminal (reported, "
              "never passed). Full precision is in the JSON.",
              "", f"validated: {'no problems' if not table.get('problems') else table['problems']}"]
    return "\n".join(lines)+"\n"


def build(runs: list, *, registry: dict, warehouse=None, no_new_measurement: bool) -> dict:
    rows = []
    for spec in runs:
        root, _, label = spec.partition(":")
        rows += rows_from_run(Path(root).expanduser(), label=label or Path(root).name, registry=registry, warehouse=warehouse)
    table = {"schema": SCHEMA, "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
             "no_new_measurement": bool(no_new_measurement), "rows": rows,
             "chain": ["registered design -> unit role", "attempt + terminal payload", "artifact digests (arrays -> terminal/record)",
                       "arrays vs source DATA (finite, exact population, unique origins, label/horizon/target identity, scaler)",
                       "independent float64 metrics vs the record", "warehouse terminal by digest and artifact rows"],
             "rules": {"skill": "1 - error_model/error_naive; lower-is-better; UNDEFINED when the naive is zero",
                       "scales": "kW and z reported in separate columns, never compared across",
                       "literature": "a published value under another protocol is shown for the record and never placed in the "
                                     "comparison column; NOT_COMPARABLE carries its reason and the planned matched comparison",
                       "missing_forecast": "a registered forecast unit without arrays or without a warehouse terminal is a problem, "
                                           "whatever NO_NEW_MEASUREMENT says about the round"}}
    table["problems"] = validate(table)
    table["verified_rows"] = sum(1 for r in rows if r.get("verified"))
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
    print(json.dumps({"rows": len(table["rows"]), "verified_rows": table["verified_rows"],
                      "problems": table["problems"][:20], "no_new_measurement": table["no_new_measurement"]}, indent=1))
    return 0 if not table["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
