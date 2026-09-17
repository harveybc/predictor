#!/usr/bin/env python3
"""The D3 matrix outcome, aggregated from the collected rows of one run (J3).

Reads `COLLECT.json` (only units whose output the parent re-hashed and verified) and every
verified `rows.jsonl`, and writes `MATRIX.json` plus a markdown table with, per operator:
applicability (verdict counts), causal-test coverage (per-test outcome counts), the resource
contract exercised (bank/toy), cost (measured CPU s per 1000 samples: median and max), declared
vs measured availability (the `availability_emission` and `response_probe` outcomes), restart
evidence (`chunk_restart`) and the number of units/variables it saw. Nothing here ranks: the
counts are what the battery said, operator by operator.

    python tools/df_d3_matrix.py --root RUN_ROOT [--out MATRIX.json] [--markdown MATRIX.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TESTS = ("prefix_all_available", "future_perturbation", "warm_up_edge", "fit_scope_train_only",
         "fresh_state_per_branch", "chunk_restart", "response_probe", "non_causal_twin",
         "availability_emission", "cost_pilot", "applicability", "raw_branch")


def verified_row_files(root: Path, collect: dict) -> list:
    """One rows.jsonl per verified unit: the attempt the collect record verified."""
    files = []
    for unit in collect["units"]:
        if not unit.get("output_verified"):
            continue
        base = root / "collected" / unit["role"] / unit["shard"] / "attempts" / unit["unit"]
        attempt = base / f"attempt-{unit.get('attempt', 1)}" / "rows.jsonl"
        if attempt.is_file():
            files.append((unit, attempt))
    return files


def aggregate(root: Path) -> dict:
    collect = json.loads((root / "COLLECT.json").read_text(encoding="utf-8"))
    per_op = defaultdict(lambda: {"verdicts": Counter(), "tests": defaultdict(Counter),
                                  "cost": [], "units": set(), "variables": 0,
                                  "banks": Counter(), "families": Counter(),
                                  "probe_lags": Counter(), "refusals": Counter()})
    rows_total = 0
    units_seen = 0
    for unit, path in verified_row_files(root, collect):
        units_seen += 1
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                rows_total += 1
                op = per_op[r["operator_kind"]]
                op["group"] = r["operator_group"]
                op["units"].add(r["unit_id"])
                op["banks"][r["bank"]] += 1
                if r["test"] == "verdict":
                    op["verdicts"][r["outcome"]] += 1
                    op["variables"] += 1
                    op["families"][r["family"]] += 1
                    if r["outcome"] == "REFUSED":
                        op["refusals"][r["detail"][:80]] += 1
                elif r["test"] == "battery":
                    continue
                else:
                    op["tests"][r["test"]][r["outcome"]] += 1
                    if r["test"] == "cost_pilot" and r["value"] is not None:
                        op["cost"].append(float(r["value"]))
                    if r["test"] == "response_probe" and r["value"] is not None:
                        op["probe_lags"][str(r["value"])] += 1
    out = {"schema": "d3_mechanics_matrix.v1", "run_id": collect["run_id"],
           "units_verified": units_seen, "units_collected": len(collect["units"]),
           "mismatched": collect.get("mismatched"), "rows": rows_total, "operators": {}}
    for kind in sorted(per_op):
        op = per_op[kind]
        cost = sorted(op["cost"])
        out["operators"][kind] = {
            "group": op["group"], "units": len(op["units"]), "variables": op["variables"],
            "banks": dict(op["banks"]), "verdicts": dict(op["verdicts"]),
            "tests": {t: dict(op["tests"][t]) for t in TESTS if t in op["tests"]},
            "cost_cpu_s_per_1000": {"n": len(cost),
                                    "median": round(statistics.median(cost), 4) if cost else None,
                                    "max": round(cost[-1], 4) if cost else None},
            "response_probe_lags": dict(op["probe_lags"]),
            "refusals": dict(op["refusals"]),
        }
    return out


def _cell(counter: dict) -> str:
    if not counter:
        return "—"
    return " / ".join(f"{k} {v}" for k, v in sorted(counter.items()))


def markdown(m: dict) -> str:
    lines = [f"Run `{m['run_id']}`: **{m['units_verified']}** units verified of "
             f"{m['units_collected']} collected, {m['mismatched']} digest mismatches, "
             f"{m['rows']:,} rows.", "",
             "| operator | group | units × vars | verdicts | causal tests failed | restart | "
             "availability | probe onset | cost s/1k (median, max) |",
             "|---|---|---|---:|---|---|---|---|---|"]
    causal = ("prefix_all_available", "future_perturbation", "non_causal_twin", "warm_up_edge",
              "fresh_state_per_branch", "fit_scope_train_only")
    for kind, op in m["operators"].items():
        failed = {t: op["tests"].get(t, {}).get("FAILED", 0) for t in causal}
        failed_s = ", ".join(f"{t} {n}" for t, n in failed.items() if n) or "none"
        c = op["cost_cpu_s_per_1000"]
        lines.append(f"| `{kind}` | {op['group']} | {op['units']} × {op['variables']} | "
                     f"{_cell(op['verdicts'])} | {failed_s} | "
                     f"{_cell(op['tests'].get('chunk_restart', {}))} | "
                     f"{_cell(op['tests'].get('availability_emission', {}))} | "
                     f"{_cell(op['response_probe_lags'])} | {c['median']}, {c['max']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    m = aggregate(args.root)
    text = json.dumps(m, indent=1, sort_keys=True) + "\n"
    (args.out or args.root / "MATRIX.json").write_text(text, encoding="utf-8")
    md = markdown(m)
    (args.markdown or args.root / "MATRIX.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
