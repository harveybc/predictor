#!/usr/bin/env python3
"""Reproducible closure of a development campaign (P4): files -> parent -> accounting ->
warehouse, per family, then one table per hypothesis / family / operator with the mapped
replica beside each selection row. Nothing is promoted: a pair that ADVANCES in the selection
family and in its mapped replica is PROPOSED_FOR_REVIEW; everything else is reported as it is.

    python tools/df_utility_dev_close.py --root DEV_ROOT --design DESIGN.json [--repo .] \\
        [--content-checks NAME] [--out CLOSE.json]
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

PROPOSED = "PROPOSED_FOR_REVIEW"
NOT_PROPOSED = "NOT_PROPOSED"


def close(root: Path, design: dict, repo: Path, content_check_name: str | None = None) -> dict:
    report = json.loads((root / "REPORT.json").read_text())
    families = {}
    for fam in design["families"]:
        froot = root / "families" / fam["unit"]
        entry = {"role": fam["role"], "replica_of": fam.get("replica_of"), "root": str(froot),
                 "reverify": None, "content_check": None, "parent": None, "accounting": None}
        if (froot / "FREEZE.json").is_file() and (froot / "REPORT.json").is_file():
            rv = RV.reverify(froot, repo)
            entry["reverify"] = rv
            freport = json.loads((froot / "REPORT.json").read_text())
            # parent: the parent's recorded outcome per contrast equals the file's re-verified one
            recorded = {k: v["outcome"] for k, v in (freport.get("contrasts") or {}).get("outcomes", {}).items()}
            entry["parent"] = {c: {"parent": recorded.get(c), "file": e.get("original_outcome"),
                                   "equal": recorded.get(c) == e.get("original_outcome")}
                               for c, e in rv["contrasts"].items()}
            entry["accounting"] = freport.get("reconciliation")
            if content_check_name and (froot / content_check_name).is_file():
                entry["content_check"] = json.loads((froot / content_check_name).read_text())
        else:
            entry["status"] = (report.get("families") or {}).get(fam["unit"], {}).get("incomplete") or "NOT_RUN"
        families[fam["unit"]] = entry
    # --- the table: selection rows with their mapped replica --------------------------------------
    rmap = design["replication_map"]
    rows = []
    proposals = []
    for sel, rep in rmap.items():
        fs, fr = families.get(sel) or {}, families.get(rep) or {}
        ts = {(t["operator"], t["hypothesis"]): t for t in ((fs.get("reverify") or {}).get("table") or [])}
        tr = {(t["operator"], t["hypothesis"]): t for t in ((fr.get("reverify") or {}).get("table") or [])}
        for k in design["operators"]:
            for h in design["hypotheses"]:
                a, b = ts.get((k, h)), tr.get((k, h))
                sel_out = (a or {}).get("reverified_outcome") or _status_of(fs, sel, k, h)
                rep_out = (b or {}).get("reverified_outcome") or _status_of(fr, rep, k, h)
                verdict = PROPOSED if sel_out == H.ADVANCES and rep_out == H.ADVANCES else NOT_PROPOSED
                if verdict == PROPOSED:
                    proposals.append({"operator": k, "hypothesis": h, "selection": sel, "replica": rep})
                rows.append({"operator": k, "hypothesis": h, "selection_unit": sel, "replica_unit": rep,
                             "selection": _cells(a), "replica": _cells(b),
                             "selection_outcome": sel_out, "replica_outcome": rep_out, "verdict": verdict})
    verified = all((f.get("reverify") or {}).get("all_verified") is True for f in families.values())
    content_ok = all(((f.get("content_check") or {}).get("all_equal") is True) for f in families.values()) \
        if content_check_name else None
    parent_ok = all(all(x["equal"] for x in (f.get("parent") or {}).values()) for f in families.values())
    accounting_ok = all(((f.get("accounting") or {}).get("contrasts") or {}).get("missing_units") == []
                        and ((f.get("accounting") or {}).get("calibration") or {}).get("missing_units") == []
                        for f in families.values())
    return {"schema": "df_utility_dev_close.v1", "run_id": report["run_id"], "design_sha256": design["design_sha256"],
            "stopped": report.get("stopped"), "spent_cpu_seconds": report.get("spent_cpu_seconds"),
            "cap_seconds": report.get("cap_seconds"), "projection": (report.get("projection") or {}).get("projected_cpu_seconds"),
            "families": families, "table": rows, "proposed_for_review": proposals,
            "checks": {"files_verified": verified, "parent_equals_files": parent_ok,
                       "accounting_reconciled": accounting_ok, "warehouse_content_equal": content_ok},
            "reading": "DOES_NOT_ADVANCE is not equivalence and does not show an operator useless elsewhere "
                       "(other domains, horizons, models, widths); INCONCLUSIVE_UNCALIBRATED is descriptive; "
                       "a favourable bound covers only the measured null and scope; PROPOSED_FOR_REVIEW is not "
                       "confirmation, public eligibility or any licence"}


def _status_of(fam, unit, k, h):
    if fam.get("status"):
        return fam["status"]
    return "MISSING"


def _cells(t):
    if not t:
        return None
    return {"loss_a": t["loss_raw_mean"], "loss_b": t["loss_transformed_mean"], "delta": t["delta_mean"],
            "lower": t["delta_lower"], "se": t["delta_se"], "rows_paired": t["rows_paired"], "n": t["n"],
            "blocks": t["blocks_used"], "cpu_seconds": t["cpu_seconds"],
            "support": (t.get("null_scope") or {}).get("derived_decision_bound"),
            "outcome": t["reverified_outcome"]}


def markdown(out: dict) -> str:
    lines = [f"# Development campaign `{out['run_id']}` — closure", "",
             f"Design `{out['design_sha256'][:12]}…`; CPU spent {out['spent_cpu_seconds']:.0f} s of a "
             f"{out['cap_seconds']:.0f} s ceiling (projected {out['projection']:.0f} s); stopped: {out['stopped']}.", "",
             "Checks: " + ", ".join(f"{k} = {v}" for k, v in out["checks"].items()), "",
             "| operator | hyp | selection unit | A loss | B loss | Δ | lower | rows | bound | outcome | replica unit | Δ | lower | bound | outcome | verdict |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|"]
    for r in out["table"]:
        s, p = r["selection"] or {}, r["replica"] or {}
        f = lambda v, d=5: ("—" if v is None else f"{v:+.{d}f}" if isinstance(v, float) else str(v))
        lines.append(f"| `{r['operator']}` | {r['hypothesis']} | `{r['selection_unit']}` | {f(s.get('loss_a'))} | {f(s.get('loss_b'))} | "
                     f"{f(s.get('delta'))} | {f(s.get('lower'))} | {s.get('rows_paired', '—')}/{s.get('n', '—')} | {f(s.get('support'))} | "
                     f"{r['selection_outcome']} | `{r['replica_unit']}` | {f(p.get('delta'))} | {f(p.get('lower'))} | {f(p.get('support'))} | "
                     f"{r['replica_outcome']} | {r['verdict']} |")
    lines += ["", f"Proposed for review: {len(out['proposed_for_review'])}.", "", out["reading"], ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--content-checks", default=None, help="name of the per-family content check file")
    parser.add_argument("--out", default="CLOSE.json")
    args = parser.parse_args(argv)
    out = close(args.root, json.loads(args.design.read_text()), args.repo, args.content_checks)
    target = args.root / args.out
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; a closure is never written over")
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    target.with_suffix(".md").write_text(markdown(out))
    print(json.dumps({"checks": out["checks"], "proposed_for_review": out["proposed_for_review"],
                      "verdicts": {f"{r['operator']}/{r['hypothesis']}/{r['selection_unit']}": (r["selection_outcome"], r["replica_outcome"], r["verdict"])
                                   for r in out["table"]}}, indent=1))
    return 0 if all(v in (True, None) for v in out["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
