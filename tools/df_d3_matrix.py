#!/usr/bin/env python3
"""The D3 matrix outcome: a VERIFIED matrix sealed from the frozen population, and, apart from
it, an exploratory summary that says it is not a verification (J3, K1).

`verify(root, receipt)` derives the expected population from the frozen manifest — units and
their variable counts, operators and their spec digests, the twelve required tests, the
amendment digest — resolves each unit's attempt from the collected ledger (the terminals),
re-hashes every rows file against the terminal that produced it, binds every row to the
manifest (design, spec, code, contract digests), demands exactly one row per unit x variable
x operator x test and exactly one verdict, recomputes each verdict from its own tests under
the battery's state policy, and seals only when nothing is missing, duplicated, unexpected,
contradictory or changed. A unit that FAILED and was recorded stays in the denominator; a
unit with no terminal is missing scope. `output_verified` in the receipt is never taken as
sufficient: the bytes are read again.

`aggregate(root, receipt)` is the exploratory summary: it counts what it reads, refuses a
missing file, recomputes nothing about verdicts and carries `verified: false`.

    python tools/df_d3_matrix.py --root RUN_ROOT --collect COLLECT.x.json --verify
        [--out MATRIX.verified.json] [--markdown MATRIX.verified.md]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


design = _load("df_d3_design")

TESTS = tuple(design.REQUIRED_TESTS)
SCOPEABLE = set(design.SCOPEABLE_TESTS)
ROW_SCHEMA = "df_fact_d3_mechanics.v1"
_TERMINAL = re.compile(r"(.+)\.attempt-(\d+)\.json")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the frozen population ------------------------------------------------------------------

def expected_population(root: Path, frozen: dict) -> dict:
    """Units -> {variables: [names], contract_sha256, bank} from FREEZE.json, the bank's own
    unit records and the toys' delivery record. Nothing is inferred from the collected rows."""
    units = {}
    for rec in frozen["bank"]["units"]:
        unit_dir = Path(rec.get("path") or (Path(frozen["bank"]["root"]).expanduser()
                                            / rec["unit_id"]))
        contract = None
        if (unit_dir / "UNIT.json").is_file():
            unit = json.loads((unit_dir / "UNIT.json").read_text(encoding="utf-8"))
            contract = unit.get("digests", {}).get("observed_signal")
        units[rec["unit_id"]] = {"variables": [f"v{i}" for i in range(int(rec["n_variables"]))],
                                 "contract_sha256": contract, "bank": "SYNTHETIC"}
    toys_doc = root / "TOYS.json"
    if toys_doc.is_file():
        for toy in json.loads(toys_doc.read_text(encoding="utf-8"))["units"]:
            rec_path = root / "toys" / toy["unit_id"] / "TOY.json"
            variables, contract = None, toy.get("contract_sha256")
            if rec_path.is_file():
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                variables, contract = list(rec["variables"]), rec["contract_sha256"]
            units[toy["unit_id"]] = {"variables": variables, "contract_sha256": contract,
                                     "bank": "TOY"}
    operators = {op["kind"]: op["spec_sha256"] for op in frozen["operators"]}
    return {"units": units, "operators": operators, "tests": TESTS,
            "design_sha256": frozen["design_sha256"],
            "code_sha256s": set(frozen.get("code_sha256s", {}).values())}


# --- the ledger -----------------------------------------------------------------------------

def ledger(root: Path) -> dict:
    """unit -> sorted [(attempt, terminal_path, terminal)] from every collected terminal."""
    out = defaultdict(list)
    for path in sorted((root / "collected").glob("*/*/terminals/*.json")) \
            if (root / "collected").is_dir() else []:
        m = _TERMINAL.fullmatch(path.name)
        if not m:
            continue
        terminal = json.loads(path.read_text(encoding="utf-8"))
        out[m.group(1)].append((int(m.group(2)), path, terminal))
    return {unit: sorted(items, key=lambda t: t[0]) for unit, items in out.items()}


def recompute_verdict(outcomes: dict) -> str:
    """The battery's policy over recorded outcomes: any FAILED refuses; any outcome that is
    neither PASSED nor a scoped SCOPED is undecided and makes it INCONCLUSIVE."""
    failed = [t for t, o in outcomes.items() if o == "FAILED"]
    undecided = [t for t, o in outcomes.items()
                 if o != "PASSED" and o != "FAILED" and not (o == "SCOPED" and t in SCOPEABLE)]
    return ("MECHANICALLY_REFUSED" if failed else
            "INCONCLUSIVE" if undecided else "MECHANICALLY_ACCEPTED")


# --- verification ---------------------------------------------------------------------------

def verify(root: Path, receipt_name: str = "COLLECT.json") -> dict:
    root = Path(root)
    frozen = json.loads((root / "FREEZE.json").read_text(encoding="utf-8"))
    receipt = json.loads((root / receipt_name).read_text(encoding="utf-8"))
    expected = expected_population(root, frozen)
    book = ledger(root)
    refusals = []

    def refuse(kind, **where):
        refusals.append({"kind": kind, **where})

    # 1. the receipt against the population and the ledger
    seen_receipt = Counter(u["unit"] for u in receipt["units"])
    for unit, n in seen_receipt.items():
        if n > 1:
            refuse("DUPLICATE_RECEIPT", unit=unit, entries=n)
    by_receipt = {u["unit"]: u for u in receipt["units"]}
    for unit in by_receipt:
        if unit not in expected["units"]:
            refuse("UNEXPECTED_UNIT", unit=unit)
    for unit in book:
        if unit not in expected["units"]:
            refuse("UNEXPECTED_UNIT", unit=unit, where="ledger")

    # 2. each expected unit: its attempt from the ledger, its outcome, its bytes
    completed, failed_units, missing = [], [], []
    per_op = defaultdict(lambda: {"verdicts": Counter(), "tests": defaultdict(Counter),
                                  "cost": [], "units": set(), "variables": 0,
                                  "banks": Counter(), "probe_lags": Counter(),
                                  "refusals": Counter(), "group": None})
    rows_total = 0
    for unit, spec in expected["units"].items():
        attempts = book.get(unit, [])
        if not attempts:
            missing.append(unit)
            refuse("MISSING_UNIT", unit=unit)
            continue
        attempt, tpath, terminal = attempts[-1]
        entry = by_receipt.get(unit)
        if entry is None:
            refuse("LEDGER_DISAGREES", unit=unit, why="in the ledger, not in the receipt")
        elif int(entry.get("attempt", 1)) != attempt or entry.get("status") != terminal["status"]:
            refuse("LEDGER_DISAGREES", unit=unit, receipt_attempt=entry.get("attempt", 1),
                   ledger_attempt=attempt, receipt_status=entry.get("status"),
                   ledger_status=terminal["status"])
        if terminal.get("code_sha256") not in expected["code_sha256s"]:
            refuse("CODE_UNBOUND", unit=unit, terminal=terminal.get("code_sha256"))
        if terminal["status"] != "COMPLETED":
            failed_units.append({"unit": unit, "attempt": attempt, "status": terminal["status"],
                                 "reason": (terminal.get("reason") or "")[:120]})
            continue
        rows_path = tpath.parents[1] / "attempts" / unit / f"attempt-{attempt}" / "rows.jsonl"
        if not rows_path.is_file():
            missing.append(unit)
            refuse("MISSING_FILE", unit=unit, path=str(rows_path))
            continue
        body = rows_path.read_bytes()
        n_lines = sum(1 for _ in body.splitlines())
        if hashlib.sha256(body).hexdigest() != terminal.get("output_sha256") \
                or n_lines != int(terminal.get("rows_written") or -1):
            refuse("DIGEST_MISMATCH", unit=unit, attempt=attempt,
                   rows_file=n_lines, rows_written=terminal.get("rows_written"))
            continue
        rows = [json.loads(line) for line in body.splitlines()]
        rows_total += len(rows)
        ok = _check_rows(unit, spec, rows, expected, receipt["run_id"], refuse, per_op)
        if ok:
            completed.append(unit)

    verified = not refusals
    operators = {}
    for kind in sorted(per_op):
        op = per_op[kind]
        cost = sorted(op["cost"])
        operators[kind] = {
            "group": op["group"], "units": len(op["units"]), "variables": op["variables"],
            "banks": dict(op["banks"]), "verdicts": dict(op["verdicts"]),
            "tests": {t: dict(op["tests"][t]) for t in TESTS if t in op["tests"]},
            "cost_cpu_s_per_1000": {"n": len(cost),
                                    "median": round(statistics.median(cost), 4) if cost else None,
                                    "max": round(cost[-1], 4) if cost else None},
            "response_probe_lags": dict(op["probe_lags"]),
            "refusals": dict(op["refusals"])}
    n_vars = sum(len(u["variables"] or []) for u in expected["units"].values())
    return {"schema": "d3_mechanics_matrix_verified.v1", "run_id": receipt["run_id"],
            "receipt": receipt_name, "freeze_sha256": frozen.get("freeze_sha256"),
            "verified": verified,
            "population": {"units": len(expected["units"]), "variables": n_vars,
                           "operators": len(expected["operators"]), "tests": len(TESTS),
                           "design_sha256": expected["design_sha256"]},
            "units": {"expected": len(expected["units"]), "completed": len(completed),
                      "failed": len(failed_units), "missing": sorted(set(missing))},
            "failed_units": failed_units, "rows": rows_total,
            "refusals": refusals, "operators": operators if verified else operators}


def _check_rows(unit, spec, rows, expected, run_id, refuse, per_op) -> bool:
    """Identity, uniqueness, coverage and verdict recomputation for one unit's rows."""
    clean = True
    variables = spec["variables"]
    cells = defaultdict(dict)          # (variable, operator) -> test -> row
    seen = Counter()
    for r in rows:
        key = (r.get("variable"), r.get("operator_kind"), r.get("test"))
        seen[key] += 1
        if seen[key] > 1:
            refuse("DUPLICATE_ROW", unit=unit, variable=key[0], operator=key[1], test=key[2])
            clean = False
            continue
        if r.get("schema") != ROW_SCHEMA or r.get("run_id") != run_id \
                or r.get("unit_id") != unit:
            refuse("ROW_IDENTITY", unit=unit, row=key)
            clean = False
            continue
        if variables is not None and r.get("variable") not in variables:
            refuse("UNEXPECTED_VARIABLE", unit=unit, variable=r.get("variable"))
            clean = False
            continue
        if r.get("operator_kind") not in expected["operators"]:
            refuse("UNEXPECTED_OPERATOR", unit=unit, operator=r.get("operator_kind"))
            clean = False
            continue
        if r.get("design_sha256") != expected["design_sha256"]:
            refuse("DESIGN_MISMATCH", unit=unit, row=key)
            clean = False
            continue
        if r.get("spec_sha256") != expected["operators"][r["operator_kind"]]:
            refuse("SPEC_MISMATCH", unit=unit, operator=r["operator_kind"])
            clean = False
            continue
        if r.get("code_sha256") not in expected["code_sha256s"]:
            refuse("CODE_UNBOUND", unit=unit, row=key)
            clean = False
            continue
        if spec["contract_sha256"] is not None and r.get("contract_sha256") != spec["contract_sha256"]:
            refuse("CONTRACT_MISMATCH", unit=unit, row=key)
            clean = False
            continue
        if r.get("test") not in TESTS and r.get("test") not in ("verdict", "battery"):
            refuse("UNEXPECTED_TEST", unit=unit, test=r.get("test"))
            clean = False
            continue
        cells[(r["variable"], r["operator_kind"])][r["test"]] = r
    if variables is not None:
        for v in variables:
            for op in expected["operators"]:
                if (v, op) not in cells:
                    refuse("MISSING_CELL", unit=unit, variable=v, operator=op)
                    clean = False
    for (variable, op), tests in cells.items():
        verdict = tests.get("verdict")
        if verdict is None:
            refuse("MISSING_VERDICT", unit=unit, variable=variable, operator=op)
            clean = False
            continue
        if "battery" in tests:
            # a refusal recorded at the contract: no tests ran, the verdict says REFUSED
            if verdict["outcome"] != "REFUSED" or any(t in tests for t in TESTS):
                refuse("VERDICT_CONTRADICTION", unit=unit, variable=variable, operator=op,
                       recorded=verdict["outcome"], recomputed="REFUSED")
                clean = False
            else:
                _count(per_op, op, tests, verdict, spec["bank"])
            continue
        absent = [t for t in TESTS if t not in tests]
        if absent:
            refuse("MISSING_TESTS", unit=unit, variable=variable, operator=op, tests=absent)
            clean = False
            continue
        recomputed = recompute_verdict({t: tests[t]["outcome"] for t in TESTS})
        expected_value = 1.0 if recomputed == "MECHANICALLY_ACCEPTED" else 0.0
        if verdict["outcome"] != recomputed or float(verdict.get("value") or 0.0) != expected_value:
            refuse("VERDICT_CONTRADICTION", unit=unit, variable=variable, operator=op,
                   recorded=verdict["outcome"], recomputed=recomputed)
            clean = False
            continue
        _count(per_op, op, tests, verdict, spec["bank"])
    return clean


def _count(per_op, kind, tests, verdict, bank):
    op = per_op[kind]
    op["group"] = verdict.get("operator_group")
    op["units"].add(verdict["unit_id"])
    op["banks"][bank] += 1
    op["verdicts"][verdict["outcome"]] += 1
    op["variables"] += 1
    if verdict["outcome"] == "REFUSED":
        op["refusals"][(verdict.get("detail") or "")[:80]] += 1
    for t in TESTS:
        r = tests.get(t)
        if r is None:
            continue
        op["tests"][t][r["outcome"]] += 1
        if t == "cost_pilot" and r.get("value") is not None:
            op["cost"].append(float(r["value"]))
        if t == "response_probe" and r.get("value") is not None:
            op["probe_lags"][str(r["value"])] += 1


# --- the exploratory summary ----------------------------------------------------------------

def aggregate(root: Path, receipt: str = "COLLECT.json") -> dict:
    """Counts what it reads. It refuses a missing file and never repeats the receipt's own
    counts as if it had checked them. It is not a verification; `verify` is."""
    root = Path(root)
    collect = json.loads((root / receipt).read_text(encoding="utf-8"))
    per_op = defaultdict(lambda: {"verdicts": Counter(), "tests": defaultdict(Counter),
                                  "cost": [], "units": set(), "variables": 0,
                                  "banks": Counter(), "probe_lags": Counter(),
                                  "refusals": Counter(), "group": None})
    rows_total, units_read = 0, 0
    cells = Counter()
    for unit in collect["units"]:
        if unit.get("status", "COMPLETED") != "COMPLETED":
            continue
        path = (root / "collected" / unit["role"] / unit["shard"] / "attempts" / unit["unit"]
                / f"attempt-{unit.get('attempt', 1)}" / "rows.jsonl")
        if not path.is_file():
            raise SystemExit(f"REFUSED: {unit['unit']}: the receipt names {path} and it is "
                             "not there; a summary over missing evidence is not a summary")
        units_read += 1
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                rows_total += 1
                op = per_op[r["operator_kind"]]
                op["group"] = r["operator_group"]
                op["units"].add(r["unit_id"])
                if r["test"] == "verdict":
                    op["verdicts"][r["outcome"]] += 1
                    op["variables"] += 1
                    op["banks"][r["bank"]] += 1
                elif r["test"] != "battery":
                    cells[(r["unit_id"], r["variable"], r["operator_kind"])] += 1
                    op["tests"][r["test"]][r["outcome"]] += 1
                    if r["test"] == "cost_pilot" and r["value"] is not None:
                        op["cost"].append(float(r["value"]))
                    if r["test"] == "response_probe" and r["value"] is not None:
                        op["probe_lags"][str(r["value"])] += 1
    verdict_cells = sum(op["variables"] for op in per_op.values())
    complete = sum(1 for n in cells.values() if n == len(TESTS))
    out = {"schema": "d3_mechanics_summary.v1", "run_id": collect["run_id"], "verified": False,
           "note": "exploratory summary of the rows the receipt names; not a verification — "
                   "verdicts are counted as recorded, not recomputed",
           "units_read": units_read, "units_in_receipt": len(collect["units"]),
           "rows": rows_total, "verdict_cells": verdict_cells,
           "cells_with_all_tests": complete,
           "cells_incomplete": verdict_cells - complete, "operators": {}}
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
            "refusals": dict(op["refusals"])}
    return out


# --- rendering ------------------------------------------------------------------------------

def _cell(counter: dict) -> str:
    if not counter:
        return "—"
    return " / ".join(f"{k} {v}" for k, v in sorted(counter.items()))


def markdown(m: dict) -> str:
    if m.get("schema") == "d3_mechanics_matrix_verified.v1":
        u = m["units"]
        head = (f"Run `{m['run_id']}` — **{'VERIFIED' if m['verified'] else 'NOT VERIFIED'}** "
                f"({len(m['refusals'])} refusals) over receipt `{m['receipt']}`: population "
                f"{m['population']['units']} units × {m['population']['variables']} variables × "
                f"{m['population']['operators']} operators × {m['population']['tests']} tests; "
                f"{u['completed']} completed, {u['failed']} failed (recorded), "
                f"{len(u['missing'])} missing; {m['rows']:,} rows re-read and bound.")
    else:
        head = (f"Run `{m['run_id']}` — exploratory summary, not a verification: "
                f"{m['units_read']} units read of {m['units_in_receipt']} in the receipt, "
                f"{m['rows']:,} rows.")
    lines = [head, "",
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
    if m.get("refusals"):
        kinds = Counter(r["kind"] for r in m["refusals"])
        lines += ["", "Refusals: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collect", default="COLLECT.json")
    parser.add_argument("--verify", action="store_true",
                        help="seal the verified matrix from the frozen population (K1); "
                             "without it, the exploratory summary")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    m = verify(args.root, args.collect) if args.verify else aggregate(args.root, args.collect)
    stem = "MATRIX.verified" if args.verify else "MATRIX.summary"
    out = args.out or args.root / f"{stem}.json"
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists; a matrix is never written over")
    out.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md = markdown(m)
    (args.markdown or args.root / f"{stem}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0 if m.get("verified") or not args.verify else 1


if __name__ == "__main__":
    raise SystemExit(main())
