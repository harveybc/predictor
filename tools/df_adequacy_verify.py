#!/usr/bin/env python3
"""Independent verification of adequacy cells (S4): losses recomputed from the persisted arrays,
compared with the cell record, the parent's report and the warehouse content (live query);
learning-curve and coverage tables from the verified cells only.

    python tools/df_adequacy_verify.py --root RUN_ROOT [--warehouse-url URL] [--out VERIFY.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("df_adequacy_models")


def cube_rows(url: str, token: str, key: str) -> dict:
    sql = (f"SELECT t.unit_id, m.metric, m.value FROM \"main\".\"gov_terminal_metric\" m "
           f"JOIN \"main\".\"gov_terminal\" t ON m.terminal_sha256 = t.terminal_sha256 "
           f"WHERE t.campaign_key = '{key}' LIMIT 5000")
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        rows = json.loads(answer.read())["rows"]
    held = {}
    for r in rows:
        held.setdefault(r["unit_id"], {})[r["metric"]] = r["value"]
    return held


def verify_cell(attempt: Path) -> dict:
    entry = {"attempt": attempt.name, "problems": [], "record": None}
    if not (attempt / "outcome.json").is_file():
        entry["problems"].append("no outcome.json")
        return entry
    outcome = json.loads((attempt / "outcome.json").read_text())
    if outcome.get("status") != "COMPLETED":
        entry["carried"] = (outcome.get("summary") or {}).get("outcome")
        return entry
    result = json.loads((attempt / "result.json").read_text())
    body = (attempt / "cell.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (outcome.get("verified") or {}).get("output_sha256"):
        entry["problems"].append("cell.json bytes differ from the declared/verified digest")
        return entry
    rec = json.loads(body)
    arrays = attempt / "arrays.npz"
    if hashlib.sha256(arrays.read_bytes()).hexdigest() != rec["arrays_sha256"]:
        entry["problems"].append("arrays altered")
        return entry
    arr = np.load(arrays)
    recomputed = {}
    for part in ("train", "validation", "test"):
        y, p, b = arr[f"{part}_y"], arr[f"{part}_pred"], arr[f"{part}_baseline"]
        rows = arr[f"{part}_rows"]
        recomputed[part] = {"model": M.mae(p, y), "baseline": M.mae(b, y), "rows": int(y.size),
                            "first_row": int(rows[0]), "last_row": int(rows[-1])}
        for k in ("model", "baseline"):
            if abs(recomputed[part][k] - rec["losses"][part][k]) > 1e-9:
                entry["problems"].append(f"{part} {k}: record {rec['losses'][part][k]} vs recomputed {recomputed[part][k]}")
        if int(y.size) != rec["losses"][part]["rows"]:
            entry["problems"].append(f"{part} rows differ")
    if recomputed["test"]["first_row"] < rec["boundaries"]["test"][0] or recomputed["train"]["last_row"] >= rec["boundaries"]["validation"][0]:
        entry["problems"].append("rows outside the frozen boundaries")
    skill = 1.0 - recomputed["test"]["model"] / recomputed["test"]["baseline"] if recomputed["test"]["baseline"] > 0 else None
    if skill is not None and rec.get("skill_test") is not None and abs(skill - rec["skill_test"]) > 1e-9:
        entry["problems"].append("skill differs from the recomputed one")
    entry["recomputed"] = recomputed
    entry["record"] = {k: rec[k] for k in ("cell_id", "unit", "task", "model", "window", "train_length", "seed", "skill_test",
                                          "consumed_span_over_P", "effective_support", "diagnosis")}
    entry["record"]["updates"] = rec["training"]["updates"]
    entry["record"]["parameters"] = rec["graph"]["parameters"]
    entry["record"]["receptive_field"] = rec["graph"]["receptive_field"]
    entry["record"]["losses_test"] = rec["losses"]["test"]
    return entry


def verify(root: Path, warehouse_url: str | None, token: str | None) -> dict:
    report = json.loads((root / "REPORT.json").read_text())
    out = {"schema": "df_adequacy_verify.v1", "run_id": report["run_id"], "design_sha256": report["design_sha256"],
           "stopped": report.get("stopped"), "cells": {}, "parent_equal": True, "all_verified": True,
           "warehouse": None, "live_query": bool(warehouse_url and token)}
    for attempt in sorted(p for p in (root / "attempts").iterdir() if p.is_dir()):
        e = verify_cell(attempt)
        out["cells"][attempt.name] = e
        if e["problems"]:
            out["all_verified"] = False
        parent = (report.get("cells") or {}).get(attempt.name) or (report.get("cost_pilot") or {}).get(attempt.name.replace("pilot__", ""))
        if e.get("record") and parent and "skill_test" in parent and abs(parent["skill_test"] - e["record"]["skill_test"]) > 1e-9:
            out["parent_equal"] = False
            e["problems"].append("parent's skill differs from the file")
    if warehouse_url and token:
        keys = [k for k in ((report.get("campaign") or {}).get("key"), f"{report['run_id']}-adequacy-cost-pilot") if k]
        held = {}
        for k in keys:
            held.update(cube_rows(warehouse_url, token, k))
        wh = {"units": {}, "all_equal": True, "covered": 0}
        for name, e in out["cells"].items():
            if not e.get("record"):
                continue
            got = held.get(name, {})
            expected = {"adequacy.mae_test": e["recomputed"]["test"]["model"], "adequacy.mae_baseline_test": e["recomputed"]["test"]["baseline"],
                        "adequacy.rows_test": float(e["recomputed"]["test"]["rows"]), "adequacy.updates": float(e["record"]["updates"])}
            if e["record"]["skill_test"] is not None:
                expected["adequacy.skill_test"] = e["record"]["skill_test"]
            equal = bool(got) and all(k in got and abs(float(got[k]) - v) < 1e-9 for k, v in expected.items())
            wh["units"][name] = {"equal": equal, "cube": got, "expected": expected}
            wh["all_equal"] &= equal
            wh["covered"] += int(bool(got))
        out["warehouse"] = wh
        out["all_verified"] &= wh["all_equal"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--no-warehouse", action="store_true")
    parser.add_argument("--out", default="VERIFY.json")
    args = parser.parse_args(argv)
    token = None if args.no_warehouse else os.environ.get(args.token_env, "")
    out = verify(args.root, None if args.no_warehouse else args.warehouse_url, token or None)
    target = args.root / args.out
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; a verification is never written over")
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"all_verified": out["all_verified"], "parent_equal": out["parent_equal"], "live_query": out["live_query"],
                      "warehouse": {k: v for k, v in (out["warehouse"] or {}).items() if k != "units"},
                      "cells": {k: (v.get("record") or {}).get("skill_test") if not v["problems"] else v["problems"] for k, v in out["cells"].items()}},
                     indent=1, default=str))
    return 0 if out["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
