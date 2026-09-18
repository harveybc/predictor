#!/usr/bin/env python3
"""The next DEVELOPMENT design for utility, per variable (O3, corrected under P1–P2). Design only.

For every (unit, variable) a sealed family of contrasts answers two declared hypotheses per
operator R, each with its OWN calibration contract (branch pair, widths, rows policy):

    H_T  raw      vs transformed   does R alone IMPROVE on raw lags of width `window`?
                                   (superiority at the sealed margin — not non-inferiority)
    H_A  raw_wide vs augmented     does raw+R improve on raw lags of the SAME total width?
                                   (capacity control: equal number of inputs; it does not
                                   equalise information, effective range or model difficulty)

Thresholds (target, horizon, model, window, blocks, margin, alpha, ridge lambda) are inherited
from the sealed pilot protocol and checked recursively on every family protocol. Multiplicity is
per family: alpha / |family| — NOT a global control over the campaign's discoveries. Replication
is an explicit map selection unit -> replica unit of the SAME generator family, perturbation,
SNR, missingness and variable, with a different seed and different data digest, both bound to
their bank resources (length, digest, eligibility re-derived from the resources, never from a
flag). Stages are separated by name; this design executes nothing.

    python tools/df_utility_next_design.py --pilot-root ROOT --bank BANK --cells CELLS.json \\
        --map SELECTION=REPLICA [...] --out DESIGN.json [--predecessor SHA]
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

DESIGN_SCHEMA = "df_utility_dev_design.v2"
STAGES = ("FLOW_DIAGNOSTIC", "DEVELOPMENT_SELECTION", "PUBLIC_CONFIRMATION", "FINANCIAL_REVALIDATION")
THIS_STAGE = "DEVELOPMENT_SELECTION"
HYPOTHESES = {
    "H_T": {"branch_a": "raw", "branch_b": "transformed",
            "question": "does the representation alone IMPROVE on raw lags of width `window` "
                        "(superiority at the sealed margin; not non-inferiority, not equivalence)?"},
    "H_A": {"branch_a": "raw_wide", "branch_b": "augmented",
            "question": "does raw+representation improve on raw lags of the same total width? "
                        "(capacity control equalises the number of inputs only — not information, "
                        "effective range, history or model difficulty)"},
}
CAPACITY_CONTROL = {"augmented": "raw_wide"}
INHERITED = ("target", "horizon", "model", "window", "n_blocks", "margin", "alpha", "min_rows_per_block",
             "ridge_lambda")
BRANCHES = ("raw", "transformed", "augmented", "raw_wide")
REGIME_KEYS = ("family", "perturbation", "snr_db", "missingness", "length", "n_variables")


class DesignRefusal(ValueError):
    """A design that cannot be sealed."""


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def unit_resource(bank_root: Path, unit_id: str) -> dict:
    """What the design binds from the bank: regime, seed, length, data digest, variables."""
    rec = json.loads((Path(bank_root) / unit_id / "UNIT.json").read_text())
    if rec.get("unit_id") != unit_id:
        raise DesignRefusal(f"{unit_id}: UNIT.json names {rec.get('unit_id')!r}")
    params = rec.get("unit_params") or {}
    regime = {"family": rec.get("family"), "perturbation": rec.get("perturbation"),
              "snr_db": str(params.get("snr_db", rec.get("declared_snr_db"))),
              "missingness": (rec.get("missingness") or {}).get("kind"),
              "length": int(rec.get("n_samples")), "n_variables": int(rec.get("n_variables", 1))}
    return {"unit": unit_id, "regime": regime, "seed": int(rec["seed"]),
            "data_sha256": rec["digests"]["observed_signal"], "n": int(rec["n_samples"]),
            "variables": list(rec.get("variable_names") or ["v0"]),
            "generator": (rec.get("generator") or {}).get("version")}


def eligibility_of(cells_record: Path, unit: str, variable: str, operators: list) -> dict:
    ops = _load("df_d3_operators")
    contract = _load("df_d3_contract")
    record = H.eligibility_record(Path(cells_record))
    out = {}
    for k in operators:
        ok, why = H.eligible(record, unit=unit, variable=variable, operator=ops.build(k))
        out[k] = {"eligible": bool(ok), "why": None if ok else why,
                  "spec_sha256": contract.spec_sha256(ops.build(k).describe())}
    return {"cells_sha256": hashlib.sha256(Path(cells_record).read_bytes()).hexdigest(),
            "freeze_sha256": record["freeze_sha256"], "design_sha256": record["design_sha256"],
            "operators": out}


def family_for(unit: str, variable: str, operators: list, hypotheses: tuple) -> list:
    return [{"contrast_id": f"{unit}__{variable}__{k}__{HYPOTHESES[h]['branch_b']}",
             "unit": unit, "variable": variable, "operator": k, "hypothesis": h,
             "branch_a": HYPOTHESES[h]["branch_a"], "branch_b": HYPOTHESES[h]["branch_b"]}
            for k in operators for h in hypotheses]


def _family(role, res, variable, operators, hypotheses, base, seed, bound_confidence, cells_record, replica_of=None):
    members = family_for(res["unit"], variable, list(operators), tuple(hypotheses))
    m = len(members)
    alpha_adjusted = base["alpha"] / m
    plan = {"generator": "white_null", "n": int(res["n"]), "bound_confidence": float(bound_confidence),
            "n_sims": H.sims_required_for_zero(alpha_adjusted, bound_confidence)}
    protocol = H.Protocol(**base, seed=seed, family=tuple(x["contrast_id"] for x in members),
                          branches=BRANCHES, calibration_plan=plan)
    return {"role": role, "unit": res["unit"], "variable": variable, "n": int(res["n"]),
            "resource": res, "eligibility": eligibility_of(cells_record, res["unit"], variable, operators),
            "members": members, "comparisons": m, "alpha_adjusted": alpha_adjusted,
            "calibration_plan": plan,
            "calibration_contracts": [{"operator": x["operator"], "hypothesis": x["hypothesis"],
                                       "branch_a": x["branch_a"], "branch_b": x["branch_b"],
                                       "widths": {"a": H.branch_width(x["branch_a"], base["window"]),
                                                  "b": H.branch_width(x["branch_b"], base["window"])},
                                       "n_sims": plan["n_sims"], "n": plan["n"],
                                       "rows_policy": H.ROWS_POLICY} for x in members],
            "protocol": protocol.sealed(), "protocol_base_sha256": protocol.base_sha256(),
            "replica_of": replica_of}


def build_design(*, bank_root, cells_record, replication_map: dict, operators: list, inherited_protocol: dict,
                 pilot_units: list, variable: str = "v0", hypotheses: tuple = ("H_T", "H_A"),
                 bound_confidence: float = 0.95, seed: int = 3, stage: str = THIS_STAGE,
                 eligibility_state: str = "SYNTHETIC_DEVELOPMENT", predecessor: str | None = None) -> dict:
    """`replication_map`: {selection unit id: replica unit id}; every unit is read from the bank."""
    if stage != THIS_STAGE:
        raise DesignRefusal(f"this design is {THIS_STAGE}; {stage} needs its own order and design")
    base = {k: inherited_protocol[k] for k in INHERITED}
    families = []
    for sel, rep in replication_map.items():
        rs, rr = unit_resource(bank_root, sel), unit_resource(bank_root, rep)
        families.append(_family("selection", rs, variable, operators, hypotheses, base, seed, bound_confidence, cells_record))
        families.append(_family("replication", rr, variable, operators, hypotheses, base, seed, bound_confidence,
                                cells_record, replica_of=sel))
    contracts = sum(len(f["calibration_contracts"]) for f in families)
    doc = {"schema": DESIGN_SCHEMA, "stage": stage, "predecessor_design_sha256": predecessor,
           "stages_out_of_scope": [s for s in STAGES if s != stage],
           "hypotheses": {h: HYPOTHESES[h] for h in hypotheses},
           "capacity_control": CAPACITY_CONTROL, "branches": list(BRANCHES),
           "inherited_protocol_base": base, "inherited_from_sha256": sha_obj(inherited_protocol),
           "thresholds_policy": "target, horizon, model, window, blocks, margin, alpha, ridge lambda and "
                                "min rows are inherited from the sealed pilot protocol and checked on every "
                                "family protocol; none is chosen from results seen",
           "multiplicity_policy": "alpha / |family| per (unit, variable); contrasts of one family are "
                                  "correlated (same series) and are never treated as independent "
                                  "experiments; this is NOT a global control over the campaign's "
                                  "discoveries; replicates are counted per explicit map, never pooled",
           "calibration_policy": f"one calibration contract per (family, operator, hypothesis): {contracts} "
                                 "contracts; a record never transfers across pairs, operators, protocols, "
                                 "lengths or harness code; generator, margin, confidence and plan are "
                                 "predeclared and no unfavourable bound licenses more simulations",
           "replication_map": dict(replication_map),
           "replication_rule": {"rule": "an (operator, hypothesis) on a variable is PROPOSED FOR REVIEW only "
                                        "if it ADVANCES in the selection family AND in its mapped replica "
                                        "(same generator family, perturbation, SNR, missingness, variable; "
                                        "different seed and data digest); nothing is confirmed, publicly "
                                        "eligible or licensed by this design"},
           "operators": list(operators), "variable": variable, "pilot_units": sorted(pilot_units),
           "eligibility_state": eligibility_state, "cells_record": str(cells_record),
           "families": families, "calibration_contracts_total": contracts,
           "execution": "NONE — design only; a governed run needs its own order", "design_sha256": ""}
    doc["design_sha256"] = sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    validate_design(doc, inherited_protocol=inherited_protocol, pilot_units=pilot_units, bank_root=bank_root)
    return doc


def _protocol_from(doc: dict) -> "H.Protocol":
    return H.Protocol(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in doc.items()
                         if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})


def validate_design(doc: dict, *, inherited_protocol: dict, pilot_units: list, bank_root=None) -> None:
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
    hyps = doc.get("hypotheses") or {}
    for h, spec in hyps.items():
        if h not in HYPOTHESES or spec != HYPOTHESES[h]:
            problems.append(f"hypothesis {h} is not a declared one")
        if spec.get("branch_b") in CAPACITY_CONTROL and spec.get("branch_a") != CAPACITY_CONTROL[spec["branch_b"]]:
            problems.append(f"{spec.get('branch_b')} without its capacity control {CAPACITY_CONTROL[spec['branch_b']]}")
    operators = list(doc.get("operators") or [])
    fams = doc.get("families") or []
    selection = {f["unit"] for f in fams if f["role"] == "selection"}
    replication = {f["unit"] for f in fams if f["role"] == "replication"}
    rmap = doc.get("replication_map") or {}
    if selection & set(pilot_units) or replication & set(pilot_units):
        problems.append("units repeat pilot units")
    if selection & replication or not replication:
        problems.append("replication units are not independent of selection units, or absent")
    if set(rmap) != selection or set(rmap.values()) != replication or len(set(rmap.values())) != len(rmap):
        problems.append("replication map does not pair every selection unit with one distinct replica")
    by_unit = {f["unit"]: f for f in fams}
    for sel, rep in rmap.items():
        fs, fr = by_unit.get(sel), by_unit.get(rep)
        if not fs or not fr:
            continue
        if fr.get("replica_of") != sel:
            problems.append(f"{rep} is not recorded as the replica of {sel}")
        if fs["resource"]["regime"] != fr["resource"]["regime"] or fs["variable"] != fr["variable"]:
            problems.append(f"{sel} -> {rep}: not the same generator family/perturbation/SNR/missingness/length/variable")
        if fs["resource"]["seed"] == fr["resource"]["seed"] or fs["resource"]["data_sha256"] == fr["resource"]["data_sha256"]:
            problems.append(f"{sel} -> {rep}: same seed or same data")
    expected_members = {(k, h) for k in operators for h in hyps}
    for f in fams:
        tag = f"family {f.get('unit')}/{f.get('variable')}"
        members = f.get("members") or []
        pairs = [(m["operator"], m["hypothesis"]) for m in members]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected_members:
            problems.append(f"{tag}: members are not exactly operators × hypotheses (duplicate or missing)")
        if f.get("comparisons") != len(members):
            problems.append(f"{tag}: cardinality")
        if abs(f.get("alpha_adjusted", -1) - base.get("alpha", 0) / max(1, len(members))) > 1e-15:
            problems.append(f"{tag}: alpha is not alpha / |family|")
        for m in members:
            spec = HYPOTHESES.get(m["hypothesis"], {})
            if (m["branch_a"], m["branch_b"]) != (spec.get("branch_a"), spec.get("branch_b")):
                problems.append(f"{m['contrast_id']}: branch pair is not its hypothesis's")
            if m["unit"] != f["unit"] or m["variable"] != f["variable"]:
                problems.append(f"{tag}: pools another unit/variable")
            if m["contrast_id"] != f"{m['unit']}__{m['variable']}__{m['operator']}__{m['branch_b']}":
                problems.append(f"{m['contrast_id']}: id does not name its unit, variable, operator and branch")
        # the family protocol, rebuilt and checked field by field
        pdoc = f.get("protocol") or {}
        try:
            protocol = _protocol_from(pdoc)
            if protocol.sealed()["protocol_sha256"] != pdoc.get("protocol_sha256"):
                problems.append(f"{tag}: protocol digest does not seal the protocol")
            if protocol.base_sha256() != f.get("protocol_base_sha256"):
                problems.append(f"{tag}: protocol base digest differs")
            for k in INHERITED:
                if getattr(protocol, k) != inherited_protocol.get(k):
                    problems.append(f"{tag}: protocol {k} {getattr(protocol, k)!r} is not the inherited {inherited_protocol.get(k)!r}")
            if list(protocol.family) != [m["contrast_id"] for m in members]:
                problems.append(f"{tag}: protocol family is not the member list")
            if tuple(protocol.branches) != BRANCHES:
                problems.append(f"{tag}: protocol branches")
            if protocol.calibration_plan != f.get("calibration_plan"):
                problems.append(f"{tag}: protocol plan is not the family plan")
            if protocol.calibration is not None:
                problems.append(f"{tag}: a design protocol carries no calibration record")
        except (H.ProtocolRefusal, TypeError, KeyError) as e:
            problems.append(f"{tag}: protocol cannot be rebuilt: {e}")
        plan = f.get("calibration_plan") or {}
        if plan.get("n") != f.get("n") or f.get("n") != (f.get("resource") or {}).get("n"):
            problems.append(f"{tag}: length is not the bound resource's")
        need = H.sims_required_for_zero(f["alpha_adjusted"], plan.get("bound_confidence", 0.95)) if members else 0
        if plan.get("n_sims") != need:
            problems.append(f"{tag}: simulations {plan.get('n_sims')} are not the derived {need}")
        contracts = f.get("calibration_contracts") or []
        if len(contracts) != len(members) or any(
                (c["operator"], c["hypothesis"], c["branch_a"], c["branch_b"]) !=
                (m["operator"], m["hypothesis"], m["branch_a"], m["branch_b"]) for c, m in zip(contracts, members)):
            problems.append(f"{tag}: one calibration contract per member, in order")
        for c in contracts:
            if c.get("widths") != {"a": H.branch_width(c["branch_a"], base.get("window", 0)),
                                   "b": H.branch_width(c["branch_b"], base.get("window", 0))} \
                    or c.get("rows_policy") != H.ROWS_POLICY or c.get("n_sims") != plan.get("n_sims") or c.get("n") != plan.get("n"):
                problems.append(f"{tag}: contract {c['operator']}/{c['hypothesis']} widths/policy/plan")
        elig = (f.get("eligibility") or {}).get("operators") or {}
        for k in operators:
            if not elig.get(k, {}).get("eligible"):
                problems.append(f"{tag}: {k} is not MECHANICALLY_ACCEPTED in the bound cells record")
        if bank_root is not None:
            try:
                fresh = unit_resource(bank_root, f["unit"])
                if fresh != f.get("resource"):
                    problems.append(f"{tag}: bound resource differs from the bank")
                if f["variable"] not in fresh["variables"]:
                    problems.append(f"{tag}: variable not in the resource")
            except (OSError, KeyError, DesignRefusal) as e:
                problems.append(f"{tag}: resource unreadable: {e}")
    if sum(len(f.get("calibration_contracts") or []) for f in fams) != doc.get("calibration_contracts_total"):
        problems.append("calibration contract count")
    if not str(doc.get("execution", "")).startswith("NONE"):
        problems.append("a design executes nothing")
    if problems:
        raise DesignRefusal("; ".join(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-root", type=Path, required=True, help="the sealed pilot (FREEZE.pre.json)")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--cells", required=True, help="the verified matrix cells record")
    parser.add_argument("--map", nargs="+", required=True, help="SELECTION=REPLICA unit ids")
    parser.add_argument("--variable", default="v0")
    parser.add_argument("--predecessor", default=None)
    parser.add_argument("--operators", nargs="+", default=None, help="default: the pilot's operators")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pre = json.loads((args.pilot_root / "FREEZE.pre.json").read_text())
    rmap = dict(x.split("=", 1) for x in args.map)
    doc = build_design(bank_root=args.bank, cells_record=args.cells, replication_map=rmap,
                       operators=args.operators or pre["operators"], inherited_protocol=pre["protocol_base"],
                       pilot_units=[u["unit"] for u in pre["units"]], variable=args.variable,
                       predecessor=args.predecessor)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists; a design is never written over")
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"design_sha256": doc["design_sha256"], "families": len(doc["families"]),
                      "contrasts": sum(f["comparisons"] for f in doc["families"]),
                      "calibration_contracts": doc["calibration_contracts_total"],
                      "sims_per_contract": doc["families"][0]["calibration_plan"]["n_sims"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
