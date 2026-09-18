#!/usr/bin/env python3
"""The next DEVELOPMENT design for utility, per variable (O3). Design only: nothing here runs.

For every (unit, variable) a sealed family of contrasts answers two declared hypotheses per
operator R:

    H_T  raw      vs transformed   does R alone predict at least as well as raw lags?
    H_A  raw_wide vs augmented     does raw+R beat raw lags of the SAME total width?
                                   (capacity control: never raw+R against raw alone)

Target, horizon, probe model, window, blocks, margin and alpha are inherited from the sealed
pilot protocol and frozen here — no threshold is chosen after the results seen. Multiplicity is
per family (one variable): alpha / |family|; replicates are independent units with their own
identical family and are COUNTED as replicates, never pooled as more tests. The calibration
plan per family is derived from alpha_adjusted and the length (0 advances in n_sims must give
an upper bound <= alpha_adjusted at the bound confidence). Stages are separated by name: this
design is DEVELOPMENT_SELECTION; flow diagnostics, public confirmation (reserve) and financial
revalidation are out of its scope and need their own orders.

    python tools/df_utility_next_design.py --pilot-root ROOT --cells CELLS.json \\
        --replication-units UNIT [UNIT ...] --out DESIGN.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
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


H = _load("df_utility_harness")

DESIGN_SCHEMA = "df_utility_dev_design.v1"
STAGES = ("FLOW_DIAGNOSTIC", "DEVELOPMENT_SELECTION", "PUBLIC_CONFIRMATION", "FINANCIAL_REVALIDATION")
THIS_STAGE = "DEVELOPMENT_SELECTION"
HYPOTHESES = {
    "H_T": {"branch_a": "raw", "branch_b": "transformed",
            "question": "does the representation alone predict at least as well as raw lags of width `window`?"},
    "H_A": {"branch_a": "raw_wide", "branch_b": "augmented",
            "question": "does raw+representation beat raw lags of the same total width (capacity control)?"},
}
CAPACITY_CONTROL = {"augmented": "raw_wide"}
INHERITED = ("target", "horizon", "model", "window", "n_blocks", "margin", "alpha", "min_rows_per_block")


class DesignRefusal(ValueError):
    """A design that cannot be sealed."""


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def family_for(unit: str, variable: str, operators: list, hypotheses: tuple) -> list:
    return [{"contrast_id": f"{unit}__{variable}__{k}__{HYPOTHESES[h]['branch_b']}",
             "unit": unit, "variable": variable, "operator": k, "hypothesis": h,
             "branch_a": HYPOTHESES[h]["branch_a"], "branch_b": HYPOTHESES[h]["branch_b"]}
            for k in operators for h in hypotheses]


def build_design(*, variables: list, replication: list, operators: list, inherited_protocol: dict,
                 pilot_units: list, cells_record: str, hypotheses: tuple = ("H_T", "H_A"),
                 bound_confidence: float = 0.95, seed: int = 3, stage: str = THIS_STAGE,
                 eligibility_state: str = "SYNTHETIC_DEVELOPMENT") -> dict:
    """`variables`: [{"unit", "variable", "n"}] — the selection set; `replication`: the same
    shape, disjoint units with their own identical families; `inherited_protocol`: the sealed
    pilot protocol base (thresholds are taken from it, never chosen here)."""
    if stage != THIS_STAGE:
        raise DesignRefusal(f"this design is {THIS_STAGE}; {stage} needs its own order and design")
    base = {k: inherited_protocol[k] for k in INHERITED}
    families = []
    for group, entries in (("selection", variables), ("replication", replication)):
        for v in entries:
            members = family_for(v["unit"], v["variable"], list(operators), tuple(hypotheses))
            m = len(members)
            alpha_adjusted = base["alpha"] / m
            plan = {"generator": "white_null", "n": int(v["n"]), "bound_confidence": float(bound_confidence),
                    "n_sims": H.sims_required_for_zero(alpha_adjusted, bound_confidence)}
            protocol = H.Protocol(**base, seed=seed, family=tuple(x["contrast_id"] for x in members),
                                  branches=("raw", "transformed", "augmented", "raw_wide"),
                                  calibration_plan=plan)
            families.append({"role": group, "unit": v["unit"], "variable": v["variable"], "n": int(v["n"]),
                             "members": members, "comparisons": m, "alpha_adjusted": alpha_adjusted,
                             "calibration_plan": plan, "protocol": protocol.sealed(),
                             "protocol_base_sha256": protocol.base_sha256()})
    doc = {"schema": DESIGN_SCHEMA, "stage": stage,
           "stages_out_of_scope": [s for s in STAGES if s != stage],
           "hypotheses": {h: HYPOTHESES[h] for h in hypotheses},
           "capacity_control": CAPACITY_CONTROL,
           "inherited_protocol_base": base, "inherited_from_sha256": sha_obj(inherited_protocol),
           "thresholds_policy": "margin, alpha, target, horizon, model, window and blocks are inherited "
                                "from the sealed pilot protocol; none is chosen from results seen",
           "multiplicity_policy": "alpha / |family| per variable; contrasts of one family are "
                                  "correlated (same series) and are never treated as independent "
                                  "experiments; replicates are counted, not pooled",
           "replication_rule": {"replicates": len(replication), "required_advances": len(replication),
                                "of": len(replication),
                                "rule": "an operator is proposed for confirmation on a variable only if "
                                        "the same hypothesis ADVANCES in the selection family AND in "
                                        "every independent replicate"},
           "operators": list(operators), "pilot_units": sorted(pilot_units),
           "eligibility": {"cells_record": str(cells_record), "state": eligibility_state},
           "families": families, "execution": "NONE — design only; a governed run needs its own order",
           "design_sha256": ""}
    doc["design_sha256"] = sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    validate_design(doc, inherited_protocol=inherited_protocol, pilot_units=pilot_units)
    return doc


def validate_design(doc: dict, *, inherited_protocol: dict, pilot_units: list) -> None:
    problems = []
    if doc.get("schema") != DESIGN_SCHEMA:
        problems.append("schema")
    if doc.get("stage") != THIS_STAGE:
        problems.append(f"stage must be {THIS_STAGE}")
    if sha_obj({k: v for k, v in doc.items() if k != "design_sha256"}) != doc.get("design_sha256"):
        problems.append("design digest does not seal the document")
    base = doc.get("inherited_protocol_base") or {}
    for k in INHERITED:
        if base.get(k) != inherited_protocol.get(k):
            problems.append(f"{k} {base.get(k)!r} is not the inherited {inherited_protocol.get(k)!r} "
                            "(thresholds are never chosen after results)")
    for h, spec in (doc.get("hypotheses") or {}).items():
        if h not in HYPOTHESES or spec != HYPOTHESES[h]:
            problems.append(f"hypothesis {h} is not a declared one")
        if spec.get("branch_b") in CAPACITY_CONTROL and spec.get("branch_a") != CAPACITY_CONTROL[spec["branch_b"]]:
            problems.append(f"{spec.get('branch_b')} without its capacity control {CAPACITY_CONTROL[spec['branch_b']]}")
    selection = {f["unit"] for f in doc.get("families", []) if f["role"] == "selection"}
    replication = {f["unit"] for f in doc.get("families", []) if f["role"] == "replication"}
    if selection & set(pilot_units):
        problems.append("selection units repeat pilot units")
    if replication & (selection | set(pilot_units)):
        problems.append("replication units are not independent of selection/pilot units")
    if not replication:
        problems.append("no independent replication")
    for f in doc.get("families", []):
        units = {m["unit"] for m in f["members"]}
        if len(units) != 1 or len({m["variable"] for m in f["members"]}) != 1:
            problems.append(f"family {f['unit']}/{f['variable']} pools several units/variables as one test family")
        if f["comparisons"] != len(f["members"]) or abs(f["alpha_adjusted"] - base.get("alpha", 0) / max(1, len(f["members"]))) > 1e-15:
            problems.append(f"family {f['unit']}: alpha is not alpha / |family|")
        plan = f["calibration_plan"]
        need = H.sims_required_for_zero(f["alpha_adjusted"], plan["bound_confidence"])
        if plan["n_sims"] < need:
            problems.append(f"family {f['unit']}: {plan['n_sims']} simulations cannot bound alpha_adjusted (need {need})")
        for m in f["members"]:
            if m["branch_b"] in CAPACITY_CONTROL and m["branch_a"] != CAPACITY_CONTROL[m["branch_b"]]:
                problems.append(f"{m['contrast_id']}: augmented without capacity control")
            if m["branch_a"] not in H.BRANCHES or m["branch_b"] not in H.BRANCHES:
                problems.append(f"{m['contrast_id']}: unknown branch")
        if f["protocol"].get("horizon") != inherited_protocol.get("horizon") \
                or f["protocol"].get("target") != inherited_protocol.get("target"):
            problems.append(f"family {f['unit']}: target/horizon not frozen to the inherited ones")
    if doc.get("execution", "").split(" ")[0] != "NONE":
        problems.append("a design executes nothing")
    if problems:
        raise DesignRefusal("; ".join(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-root", type=Path, required=True, help="the sealed pilot (FREEZE.pre.json)")
    parser.add_argument("--cells", required=True, help="the verified matrix cells record")
    parser.add_argument("--selection-units", nargs="+", required=True, help="unit ids for selection (n from the bank)")
    parser.add_argument("--replication-units", nargs="+", required=True)
    parser.add_argument("--n", type=int, required=True, help="series length of the bank units")
    parser.add_argument("--variable", default="v0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pre = json.loads((args.pilot_root / "FREEZE.pre.json").read_text())
    doc = build_design(variables=[{"unit": u, "variable": args.variable, "n": args.n} for u in args.selection_units],
                       replication=[{"unit": u, "variable": args.variable, "n": args.n} for u in args.replication_units],
                       operators=pre["operators"], inherited_protocol=pre["protocol_base"],
                       pilot_units=[u["unit"] for u in pre["units"]], cells_record=args.cells)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"design_sha256": doc["design_sha256"], "families": len(doc["families"]),
                      "contrasts_per_family": doc["families"][0]["comparisons"],
                      "sims_per_family": doc["families"][0]["calibration_plan"]["n_sims"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
