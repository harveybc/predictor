#!/usr/bin/env python3
"""The delta between two VERIFIED D3 matrices, per operator and test, with its causes (K5).

Both inputs must be sealed by `df_d3_matrix.verify` (`verified: true`); a delta over an
unverified matrix is refused. For each operator the verdict counts and every test's outcome
counts are compared; each difference is attributed to a declared cause when one applies —
the amendment changed (`design_sha256`), the operator's declaration changed (`spec_sha256`),
the population changed (units/variables) — and left `UNATTRIBUTED` otherwise, so an unexplained
movement is visible rather than absorbed.

    python tools/df_d3_matrix_delta.py --before MATRIX.verified.json --after MATRIX.verified.json
        --before-freeze FREEZE.json --after-freeze FREEZE.json --out DELTA.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: tests whose rule the successor amendment (07B) changed, and what it changed
AMENDED_TESTS = {"response_probe": "07B: excitation from the training fit and the declared "
                                   "resolution; three facts apart; abstention only by declaration",
                 "non_causal_twin": "07B/07C: a twin without observable comparisons is "
                                    "INSUFFICIENT_TEST; 07C counts only comparisons whose "
                                    "reach crosses the cut as sensitive"}


def _counts_delta(before: dict, after: dict) -> dict:
    keys = sorted(set(before) | set(after))
    return {k: {"before": before.get(k, 0), "after": after.get(k, 0),
                "delta": after.get(k, 0) - before.get(k, 0)}
            for k in keys if before.get(k, 0) != after.get(k, 0)}


def delta(before: dict, after: dict, before_freeze: dict, after_freeze: dict) -> dict:
    for name, m in (("before", before), ("after", after)):
        if m.get("schema") != "d3_mechanics_matrix_verified.v1" or m.get("verified") is not True:
            raise SystemExit(f"REFUSED: the {name} matrix is not a verified matrix")
    specs_before = {op["kind"]: op["spec_sha256"] for op in before_freeze["operators"]}
    specs_after = {op["kind"]: op["spec_sha256"] for op in after_freeze["operators"]}
    design_changed = before_freeze["design_sha256"] != after_freeze["design_sha256"]
    strip = lambda pop: {k: v for k, v in pop.items() if k != "design_sha256"}
    population_changed = strip(before["population"]) != strip(after["population"])
    out = {"schema": "d3_mechanics_matrix_delta.v1",
           "before": {"run_id": before["run_id"], "freeze_sha256": before["freeze_sha256"],
                      "design_sha256": before_freeze["design_sha256"]},
           "after": {"run_id": after["run_id"], "freeze_sha256": after["freeze_sha256"],
                     "design_sha256": after_freeze["design_sha256"]},
           "design_changed": design_changed, "population_changed": population_changed,
           "population": {"before": before["population"], "after": after["population"]},
           "units": {"before": before["units"], "after": after["units"]},
           "operators": {}, "unattributed": []}
    for kind in sorted(set(before["operators"]) | set(after["operators"])):
        b = before["operators"].get(kind, {"verdicts": {}, "tests": {}})
        a = after["operators"].get(kind, {"verdicts": {}, "tests": {}})
        spec_changed = specs_before.get(kind) != specs_after.get(kind)
        entry = {"spec_changed": spec_changed, "verdicts": _counts_delta(b["verdicts"], a["verdicts"]),
                 "tests": {}, "causes": []}
        for test in sorted(set(b["tests"]) | set(a["tests"])):
            d = _counts_delta(b["tests"].get(test, {}), a["tests"].get(test, {}))
            if not d:
                continue
            causes = []
            if test in AMENDED_TESTS and design_changed:
                causes.append(AMENDED_TESTS[test])
            if spec_changed and test == "response_probe":
                causes.append("declaration changed: response_probe.scale = TRAIN_FIT (v3)")
            if population_changed:
                causes.append("population changed")
            entry["tests"][test] = {"delta": d, "causes": causes or ["UNATTRIBUTED"]}
            if not causes:
                out["unattributed"].append({"operator": kind, "test": test, "delta": d})
        if entry["verdicts"]:
            moved = [t for t, v in entry["tests"].items()]
            entry["causes"] = sorted({c for t in entry["tests"].values() for c in t["causes"]}) \
                or ["UNATTRIBUTED: verdicts moved with no test moving"]
            if not moved:
                out["unattributed"].append({"operator": kind, "test": "verdict",
                                            "delta": entry["verdicts"]})
        out["operators"][kind] = entry
    out["all_attributed"] = not out["unattributed"]
    return out


def markdown(d: dict) -> str:
    lines = [f"Delta `{d['before']['run_id']}` → `{d['after']['run_id']}` "
             f"(design changed: {d['design_changed']}, population changed: "
             f"{d['population_changed']}, all attributed: **{d['all_attributed']}**)", "",
             "| operator | verdict movements | test movements | causes |", "|---|---|---|---|"]
    for kind, e in d["operators"].items():
        if not e["verdicts"] and not e["tests"]:
            continue
        v = "; ".join(f"{k} {x['before']}→{x['after']}" for k, x in e["verdicts"].items()) or "—"
        t = "; ".join(f"{test}: " + ", ".join(f"{k} {x['before']}→{x['after']}"
                                              for k, x in td["delta"].items())
                      for test, td in e["tests"].items()) or "—"
        lines.append(f"| `{kind}` | {v} | {t} | {'; '.join(e['causes']) or '—'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-freeze", type=Path, required=True)
    parser.add_argument("--after-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    d = delta(load(args.before), load(args.after), load(args.before_freeze), load(args.after_freeze))
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a delta is never written over")
    args.out.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(d), encoding="utf-8")
    print(markdown(d))
    return 0 if d["all_attributed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
