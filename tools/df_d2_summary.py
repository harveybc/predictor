#!/usr/bin/env python3
"""D2 operator totals, derived from the decision records rather than from prose.

Musashi's B1 (2026-09-14): report totals by `arm_role` and decision, distinguish **rows**,
**regimes** and **unique methods**, and derive the transition counts from the records. The
earlier prose said "58 favourable or limited → 48 calibrated plus 6 limited" and "seven lose
a pass"; both were wrong, and a summary that cannot be recomputed is how that survived.

The grain of a decision is (subject_kind, subject, operator_params, regime). Comparing
published and successor rows on anything less than that grain is a cross product: on
(subject, regime) alone the same 3,591 rows appear to lose 36 passes, where the real number
is 5 candidate losses plus 2 identity-control losses and no gain at all.

usage:
  df_d2_summary.py --published df_fact_d2_decision.jsonl --successor DECISIONS_SUCCESSOR.jsonl
                   --out SUMMARY.json [--markdown SUMMARY.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PASSES = ("LAB_CALIBRATED", "REGIME_LIMITED")
SNR_PASSES = ("SNR_CALIBRATED_FOR_REGIME", "SNR_REGIME_LIMITED")


def grain(row: dict) -> tuple:
    """The identity of a decision. Anything coarser is a different question."""
    return (row["subject_kind"], row["subject"],
            json.dumps(row.get("operator_params") or {}, sort_keys=True),
            json.dumps(row["regime"], sort_keys=True))


def read(path: Path) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = grain(row)
            if key in rows:
                raise SystemExit(f"REFUSED: {path} repeats a decision grain: {key}")
            rows[key] = row
    return rows


def totals(rows: dict) -> dict:
    """Rows, distinct regimes and distinct methods, by arm role and decision."""
    counts: dict = defaultdict(lambda: {"rows": 0, "regimes": set(), "methods": set()})
    for key, row in rows.items():
        bucket = counts[(row.get("arm_role") or "(none)", row["subject_kind"], row["decision"])]
        bucket["rows"] += 1
        bucket["regimes"].add(key[3])
        bucket["methods"].add(row["subject"])
    out = {}
    for (arm, kind, decision), bucket in sorted(counts.items()):
        out.setdefault(arm, {}).setdefault(kind, {})[decision] = {
            "rows": bucket["rows"], "regimes": len(bucket["regimes"]), "methods": len(bucket["methods"])}
    return out


def passes(rows: dict, kind: str, arm: str | None = None, names=PASSES) -> dict:
    selected = [r for r in rows.values() if r["subject_kind"] == kind
                and (arm is None or (r.get("arm_role") or "(none)") == arm)]
    return {name: sum(1 for r in selected if r["decision"] == name) for name in names}


def transitions(published: dict, successor: dict) -> dict:
    missing = sorted(set(published) - set(successor))
    added = sorted(set(successor) - set(published))
    moves: dict = defaultdict(int)
    lost, gained = [], []
    for key in sorted(set(published) & set(successor)):
        old, new = published[key], successor[key]
        moves[(old.get("arm_role") or "(none)", old["decision"], new["decision"])] += 1
        was, is_now = old["decision"] in PASSES, new["decision"] in PASSES
        if was and not is_now:
            lost.append({"arm_role": old.get("arm_role"), "subject": old["subject"],
                         "operator_params": old.get("operator_params"), "regime": old["regime"],
                         "old": old["decision"], "new": new["decision"],
                         "n_seeds_valid_old": old.get("n_seeds_valid"),
                         "n_seeds_valid_new": new.get("n_seeds_valid")})
        if is_now and not was:
            gained.append({"arm_role": old.get("arm_role"), "subject": old["subject"],
                           "old": old["decision"], "new": new["decision"]})
    return {"compared_grains": len(set(published) & set(successor)),
            "only_in_published": missing, "only_in_successor": added,
            "moves": [{"arm_role": a, "from": f, "to": t, "rows": n}
                      for (a, f, t), n in sorted(moves.items()) if f != t],
            "changed_rows": sum(n for (_a, f, t), n in moves.items() if f != t),
            "passes_lost": lost, "passes_gained": gained,
            "passes_lost_by_arm": {arm: sum(1 for x in lost if (x["arm_role"] or "(none)") == arm)
                                   for arm in sorted({(x["arm_role"] or "(none)") for x in lost})}}


def summarise(published: dict, successor: dict) -> dict:
    return {
        "schema": "d2_operator_totals.v1",
        "grain": "subject_kind + subject + operator_params + regime",
        "rows": {"published": len(published), "successor": len(successor)},
        "totals_published": totals(published),
        "totals_successor": totals(successor),
        "operator_passes": {
            "published": {arm: passes(published, "OPERATOR", arm)
                          for arm in sorted({(r.get("arm_role") or "(none)") for r in published.values()
                                             if r["subject_kind"] == "OPERATOR"})},
            "successor": {arm: passes(successor, "OPERATOR", arm)
                          for arm in sorted({(r.get("arm_role") or "(none)") for r in successor.values()
                                             if r["subject_kind"] == "OPERATOR"})}},
        "snr_passes": {"published": passes(published, "SNR_ESTIMATOR", None, SNR_PASSES),
                       "successor": passes(successor, "SNR_ESTIMATOR", None, SNR_PASSES)},
        "transitions": transitions(published, successor),
        "supersedes": ("the prose of the earlier reports: '48 LAB_CALIBRATED plus 6 REGIME_LIMITED' "
                       "and 'seven lose a pass' are withdrawn; the records say 47 + 6 for the "
                       "CANDIDATE arm, and five candidate losses plus two identity-control losses"),
    }


def markdown(summary: dict) -> str:
    lines = ["# D2 operator totals (derived from the decision records)", "",
             f"Grain: `{summary['grain']}`. Rows: published {summary['rows']['published']}, "
             f"successor {summary['rows']['successor']}.", "",
             "| arm role | decision | rows | regimes | methods |", "|---|---|---|---|---|"]
    for arm, kinds in sorted(summary["totals_successor"].items()):
        for _kind, decisions in sorted(kinds.items()):
            for decision, values in sorted(decisions.items()):
                lines.append(f"| {arm} | {decision} | {values['rows']} | {values['regimes']} | "
                             f"{values['methods']} |")
    transitions = summary["transitions"]
    lines += ["", "## What moved", "",
              f"{transitions['changed_rows']} rows change decision; "
              f"{len(transitions['passes_lost'])} lose a pass and "
              f"{len(transitions['passes_gained'])} gain one.", "",
              "| arm role | from | to | rows |", "|---|---|---|---|"]
    for move in transitions["moves"]:
        lines.append(f"| {move['arm_role']} | {move['from']} | {move['to']} | {move['rows']} |")
    lines += ["", "## Superseded prose", "", summary["supersedes"], ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--successor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    summary = summarise(read(a.published), read(a.successor))
    a.out.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    if a.markdown:
        a.markdown.write_text(markdown(summary) + "\n", encoding="utf-8")
    operators = summary["operator_passes"]
    print(json.dumps({"candidate_published": operators["published"].get("CANDIDATE"),
                      "candidate_successor": operators["successor"].get("CANDIDATE"),
                      "snr": summary["snr_passes"],
                      "passes_lost_by_arm": summary["transitions"]["passes_lost_by_arm"],
                      "passes_gained": len(summary["transitions"]["passes_gained"]),
                      "changed_rows": summary["transitions"]["changed_rows"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
