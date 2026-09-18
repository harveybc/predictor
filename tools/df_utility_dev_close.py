#!/usr/bin/env python3
"""Reproducible closure of a development campaign (P4, corrected under Q1).

The closure is bound to the registered population: the sealed design is validated against the
pilot's inherited protocol, the campaign report must name that design's identity, and the exact
families, members, calibration contracts, pairs and replication map are DERIVED from the design.
An empty design, an omitted or unexpected family, a duplicated or unforeseen member, a receipt of
another run, or a content check that does not cover the whole population is a typed refusal or a
failed check — never all([]) == True.

Checks per family (all mandatory): files (re-verification from conserved results), parent (the
parent's recorded outcome per member equals the file's), accounting (both campaigns keyed to this
run, reconciled, every member with a terminal) and warehouse (the content check names this run,
covers every member and contract, and is all equal). Closure policy: PARTIAL closure is allowed;
a pair (selection, replica) yields a verdict only when BOTH families passed every check; otherwise
its rows are UNVERIFIED_PAIR and any advancing candidate is listed apart as an unverified
candidate, never under PROPOSED_FOR_REVIEW. Nothing is promoted.

    python tools/df_utility_dev_close.py --root DEV_ROOT --design DESIGN.json --pilot-root PILOT \\
        [--repo .] [--content-checks NAME] [--out CLOSE.json]
"""

from __future__ import annotations

import argparse
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


RV = _load("df_utility_reverify")
H = _load("df_utility_harness")
D = _load("df_utility_next_design")

PROPOSED = "PROPOSED_FOR_REVIEW"
NOT_PROPOSED = "NOT_PROPOSED"
UNVERIFIED_PAIR = "UNVERIFIED_PAIR"
CHECKS = ("files_verified", "parent_equals_files", "accounting_reconciled", "warehouse_content_equal")


class ClosureRefusal(SystemExit):
    """A closure that cannot be made: the population is not the registered one."""


def _population(design: dict) -> dict:
    fams = design.get("families") or []
    if not fams or not design.get("operators") or not design.get("hypotheses") or not design.get("replication_map"):
        raise ClosureRefusal("REFUSED: the design carries no population (families, operators, hypotheses or map empty)")
    units = [f["unit"] for f in fams]
    if len(set(units)) != len(units):
        raise ClosureRefusal("REFUSED: the design repeats a family")
    return {"units": units,
            "members": {f["unit"]: [m["contrast_id"] for m in f["members"]] for f in fams},
            "contracts": {f["unit"]: [f"{c['operator']}__{c['hypothesis']}" for c in f["calibration_contracts"]] for f in fams},
            "pairs": {(m["operator"], m["hypothesis"]): (m["branch_a"], m["branch_b"]) for f in fams for m in f["members"]},
            "map": dict(design["replication_map"])}


def _family_checks(froot: Path, fam: dict, members: list, contracts: list, run_id: str, repo: Path,
                   content_check_name: str | None) -> dict:
    entry = {"role": fam["role"], "replica_of": fam.get("replica_of"), "run_id": run_id, "root": str(froot),
             "checks": {k: False for k in CHECKS}, "problems": [], "reverify": None, "content_check": None, "parent": None,
             "accounting": None}
    freport = json.loads((froot / "REPORT.json").read_text())
    if freport.get("run_id") != run_id:
        raise ClosureRefusal(f"REFUSED: family {fam['unit']}: its report is run {freport.get('run_id')!r}, the campaign registered {run_id!r} (identity)")
    # --- the family's terminals: exactly the members, once each ---
    terminal_ids = [t.get("unit_id") for t in freport.get("terminals") or []]
    if len(set(terminal_ids)) != len(terminal_ids):
        raise ClosureRefusal(f"REFUSED: family {fam['unit']}: duplicate terminal in the report")
    strangers = sorted(set(terminal_ids) - set(members))
    if strangers:
        raise ClosureRefusal(f"REFUSED: family {fam['unit']}: unexpected terminal(s) not a member of the design: {strangers}")
    missing_terminals = sorted(set(members) - set(terminal_ids))
    cal_keys = sorted(k for k in (freport.get("calibration") or {}) if k != "campaign")
    if set(cal_keys) - set(contracts):
        raise ClosureRefusal(f"REFUSED: family {fam['unit']}: unexpected calibration unit(s) {sorted(set(cal_keys) - set(contracts))}")
    # --- files ---
    rv = RV.reverify(froot, repo)
    entry["reverify"] = rv
    rv_members = set(rv.get("contrasts") or {})
    rv_contracts = set(rv.get("calibrations") or {})
    files_ok = rv.get("all_verified") is True and set(members) <= rv_members and set(contracts) <= rv_contracts
    if not (set(members) <= rv_members):
        entry["problems"].append(f"files: members without a verified attempt: {sorted(set(members) - rv_members)}")
    if not (set(contracts) <= rv_contracts):
        entry["problems"].append(f"files: contracts without a verified record: {sorted(set(contracts) - rv_contracts)}")
    if rv.get("all_verified") is not True:
        entry["problems"].append("files: re-verification found problems")
    entry["checks"]["files_verified"] = files_ok
    # --- parent ---
    recorded = {k: v.get("outcome") for k, v in ((freport.get("contrasts") or {}).get("outcomes") or {}).items()}
    parent = {c: {"parent": recorded.get(c), "file": (rv.get("contrasts") or {}).get(c, {}).get("original_outcome"),
                  "equal": recorded.get(c) is not None and recorded.get(c) == (rv.get("contrasts") or {}).get(c, {}).get("original_outcome")}
              for c in members}
    entry["parent"] = parent
    entry["checks"]["parent_equals_files"] = all(x["equal"] for x in parent.values())
    if not entry["checks"]["parent_equals_files"]:
        entry["problems"].append("parent: recorded outcome differs from the file (or absent) for " +
                                 str(sorted(c for c, x in parent.items() if not x["equal"])))
    # --- accounting ---
    recon = freport.get("reconciliation") or {}
    keys = {"contrasts": ((freport.get("contrasts") or {}).get("campaign") or {}).get("key"),
            "calibration": ((freport.get("calibration") or {}).get("campaign") or {}).get("key")}
    acc_ok = True
    for which, expected_key in (("contrasts", f"{run_id}-utility-contrasts"), ("calibration", f"{run_id}-utility-calibration")):
        r = recon.get(which) or {}
        if keys[which] != expected_key:
            entry["problems"].append(f"accounting: {which} campaign key {keys[which]!r} is not this run's {expected_key!r}")
            acc_ok = False
        if r.get("http") != 200 or r.get("missing_units") != [] or r.get("accounting_only") or r.get("lake_only"):
            entry["problems"].append(f"accounting: {which} not reconciled: {r}")
            acc_ok = False
    if missing_terminals:
        entry["problems"].append(f"accounting: members without a terminal: {missing_terminals}")
        acc_ok = False
    if set(cal_keys) != set(contracts):
        entry["problems"].append(f"accounting: contracts without a calibration unit: {sorted(set(contracts) - set(cal_keys))}")
        acc_ok = False
    entry["accounting"] = {"keys": keys, "reconciliation": recon}
    entry["checks"]["accounting_reconciled"] = acc_ok
    # --- warehouse ---
    if content_check_name:
        path = froot / content_check_name
        if not path.is_file():
            entry["problems"].append("warehouse: no content check")
        else:
            cc = json.loads(path.read_text())
            entry["content_check"] = {"run_id": cc.get("run_id"), "all_equal": cc.get("all_equal"),
                                      "units": len(cc.get("units") or {}), "refused": cc.get("refused")}
            expected_units = set(members) | {f"calibrate__{k}" for k in contracts}
            covered = set(cc.get("units") or {})
            wh_ok = cc.get("run_id") == run_id and cc.get("all_equal") is True and not cc.get("refused") \
                and expected_units <= covered and all((cc["units"][u].get("equal") is True) for u in expected_units)
            if cc.get("run_id") != run_id:
                entry["problems"].append(f"warehouse: content check is run {cc.get('run_id')!r}, not {run_id!r}")
            if not (expected_units <= covered):
                entry["problems"].append(f"warehouse: population not covered: {sorted(expected_units - covered)}")
            if cc.get("all_equal") is not True or cc.get("refused"):
                entry["problems"].append("warehouse: content not all equal or refused entries")
            entry["checks"]["warehouse_content_equal"] = wh_ok
    else:
        entry["checks"]["warehouse_content_equal"] = None
    entry["verified"] = all(v is True for v in entry["checks"].values() if v is not None) and \
        entry["checks"]["warehouse_content_equal"] is not None
    return entry


def close(root: Path, design: dict, repo: Path, content_check_name: str | None = None, *,
          inherited_protocol: dict | None = None, pilot_units: list | None = None) -> dict:
    """`inherited_protocol`/`pilot_units`: the pilot's, as the CLI passes them; without them the
    design's own declared base is used (its inner recursion is still validated)."""
    root = Path(root)
    if inherited_protocol is None:
        inherited_protocol = design.get("inherited_protocol_base") or {}
    if pilot_units is None:
        pilot_units = design.get("pilot_units") or []
    try:
        D.validate_design(design, inherited_protocol=inherited_protocol, pilot_units=pilot_units)
    except D.DesignRefusal as e:
        raise ClosureRefusal(f"REFUSED: the design is not a sealed valid one: {e}") from e
    pop = _population(design)
    report = json.loads((root / "REPORT.json").read_text())
    if report.get("design_sha256") != design.get("design_sha256"):
        raise ClosureRefusal(f"REFUSED: the campaign report names design {str(report.get('design_sha256'))[:12]!r}, "
                             f"the closure was asked for {design['design_sha256'][:12]!r} (identity)")
    reported = report.get("families") or {}
    omitted = sorted(set(pop["units"]) - set(reported))
    unexpected = sorted(set(reported) - set(pop["units"]))
    if omitted:
        raise ClosureRefusal(f"REFUSED: families of the design omitted from the campaign report (missing): {omitted}")
    if unexpected:
        raise ClosureRefusal(f"REFUSED: families in the report not in the design (unexpected): {unexpected}")
    families = {}
    for fam in design["families"]:
        u = fam["unit"]
        rf = reported[u]
        froot = Path(rf.get("root") or root / "families" / u)
        if rf.get("incomplete") or not (froot / "REPORT.json").is_file() or not (froot / "FREEZE.json").is_file():
            families[u] = {"role": fam["role"], "replica_of": fam.get("replica_of"), "root": str(froot),
                           "status": rf.get("incomplete") or "NOT_RUN", "checks": {k: False for k in CHECKS},
                           "problems": [rf.get("incomplete") or "no family report"], "verified": False, "reverify": None}
            continue
        families[u] = _family_checks(froot, fam, pop["members"][u], pop["contracts"][u], rf.get("run_id"), repo, content_check_name)
    # --- the table: selection rows with their mapped replica; verdict only for verified pairs ---
    rows, proposals, unverified = [], [], []
    for sel, rep in pop["map"].items():
        fs, fr = families[sel], families[rep]
        pair_verified = fs.get("verified") is True and fr.get("verified") is True
        ts = {(t["operator"], t["hypothesis"]): t for t in ((fs.get("reverify") or {}).get("table") or [])}
        tr = {(t["operator"], t["hypothesis"]): t for t in ((fr.get("reverify") or {}).get("table") or [])}
        for k in design["operators"]:
            for h in design["hypotheses"]:
                a, b = ts.get((k, h)), tr.get((k, h))
                sel_out = (a or {}).get("reverified_outcome") or fs.get("status") or "MISSING"
                rep_out = (b or {}).get("reverified_outcome") or fr.get("status") or "MISSING"
                if not pair_verified:
                    verdict = UNVERIFIED_PAIR
                    if sel_out == H.ADVANCES or rep_out == H.ADVANCES:
                        unverified.append({"operator": k, "hypothesis": h, "selection": sel, "replica": rep,
                                           "selection_outcome": sel_out, "replica_outcome": rep_out,
                                           "why": "a mandatory check of this pair's scope failed; diagnostic only"})
                else:
                    verdict = PROPOSED if sel_out == H.ADVANCES and rep_out == H.ADVANCES else NOT_PROPOSED
                    if verdict == PROPOSED:
                        proposals.append({"operator": k, "hypothesis": h, "selection": sel, "replica": rep})
                rows.append({"operator": k, "hypothesis": h, "pair": list(pop["pairs"][(k, h)]),
                             "selection_unit": sel, "replica_unit": rep, "selection": _cells(a), "replica": _cells(b),
                             "selection_outcome": sel_out, "replica_outcome": rep_out, "verdict": verdict})
    checks = {c: all(f["checks"].get(c) is True for f in families.values()) if any(f["checks"].get(c) is not None for f in families.values()) else None
              for c in CHECKS}
    return {"schema": "df_utility_dev_close.v2", "run_id": report["run_id"], "design_sha256": design["design_sha256"],
            "population": {"families": len(pop["units"]), "members": sum(len(v) for v in pop["members"].values()),
                           "contracts": sum(len(v) for v in pop["contracts"].values()), "pairs": len(pop["map"])},
            "closure": "TOTAL" if all(f.get("verified") for f in families.values()) else "PARTIAL",
            "closure_policy": "a pair yields a verdict only when both its families passed every mandatory check "
                              "(files, parent, accounting, warehouse) over their whole population; otherwise "
                              "UNVERIFIED_PAIR and no proposal",
            "stopped": report.get("stopped"), "spent_cpu_seconds": report.get("spent_cpu_seconds"),
            "cap_seconds": report.get("cap_seconds"), "projection": (report.get("projection") or {}).get("projected_cpu_seconds"),
            "families": families, "table": rows, "proposed_for_review": proposals, "unverified_candidates": unverified,
            "checks": checks,
            "reading": "DOES_NOT_ADVANCE is not equivalence and does not show an operator useless elsewhere "
                       "(other domains, horizons, models, widths); INCONCLUSIVE_UNCALIBRATED is descriptive; "
                       "a favourable bound covers only the measured null and scope; the capacity control equalises "
                       "the number of inputs, not the available history or information; PROPOSED_FOR_REVIEW is not "
                       "confirmation, public eligibility or any licence"}


def _cells(t):
    if not t:
        return None
    return {"loss_a": t["loss_raw_mean"], "loss_b": t["loss_transformed_mean"], "delta": t["delta_mean"],
            "lower": t["delta_lower"], "se": t["delta_se"], "rows_paired": t["rows_paired"], "n": t["n"],
            "blocks": t["blocks_used"], "cpu_seconds": t["cpu_seconds"],
            "support": (t.get("null_scope") or {}).get("derived_decision_bound"),
            "outcome": t["reverified_outcome"]}


def markdown(out: dict) -> str:
    f = lambda v, d=5: ("—" if v is None else f"{v:+.{d}f}" if isinstance(v, float) else str(v))
    lines = [f"# Development campaign `{out['run_id']}` — closure ({out['closure']})", "",
             f"Design `{out['design_sha256'][:12]}…`; population {out['population']}; CPU spent {f(out['spent_cpu_seconds'], 0)} s "
             f"of {f(out['cap_seconds'], 0)} s (projected {f(out['projection'], 0)} s); stopped: {out['stopped']}.", "",
             "Checks: " + ", ".join(f"{k} = {v}" for k, v in out["checks"].items()) + f". Policy: {out['closure_policy']}.", "",
             "| operator | hyp | pair | selection unit | A loss | B loss | Δ | lower | rows | bound | outcome | replica unit | Δ | lower | bound | outcome | verdict |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|"]
    for r in out["table"]:
        s, p = r["selection"] or {}, r["replica"] or {}
        lines.append(f"| `{r['operator']}` | {r['hypothesis']} | {'/'.join(r['pair'])} | `{r['selection_unit']}` | {f(s.get('loss_a'))} | {f(s.get('loss_b'))} | "
                     f"{f(s.get('delta'))} | {f(s.get('lower'))} | {s.get('rows_paired', '—')}/{s.get('n', '—')} | {f(s.get('support'))} | "
                     f"{r['selection_outcome']} | `{r['replica_unit']}` | {f(p.get('delta'))} | {f(p.get('lower'))} | {f(p.get('support'))} | "
                     f"{r['replica_outcome']} | {r['verdict']} |")
    lines += ["", f"Proposed for review: {len(out['proposed_for_review'])}. Unverified candidates (diagnostic, not proposed): "
              f"{len(out['unverified_candidates'])}.", ""]
    for fam, e in out["families"].items():
        if e.get("problems"):
            lines.append(f"* `{fam}`: " + "; ".join(str(p) for p in e["problems"]))
    lines += ["", out["reading"], ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True, help="the sealed pilot the design inherits from")
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--content-checks", default=None, help="name of the per-family content check file")
    parser.add_argument("--out", default="CLOSE.json")
    args = parser.parse_args(argv)
    pre = json.loads((args.pilot_root / "FREEZE.pre.json").read_text())
    design = json.loads(args.design.read_text())
    out = close(args.root, design, args.repo, args.content_checks,
                inherited_protocol=pre["protocol_base"], pilot_units=[u["unit"] for u in pre["units"]])
    target = args.root / args.out
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; a closure is never written over")
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    target.with_suffix(".md").write_text(markdown(out))
    print(json.dumps({"closure": out["closure"], "checks": out["checks"], "proposed_for_review": out["proposed_for_review"],
                      "unverified_candidates": len(out["unverified_candidates"]),
                      "verdicts": {f"{r['operator']}/{r['hypothesis']}/{r['selection_unit']}": (r["selection_outcome"], r["replica_outcome"], r["verdict"])
                                   for r in out["table"]}}, indent=1))
    return 0 if out["closure"] == "TOTAL" and all(v is True for v in out["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
