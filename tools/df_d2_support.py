#!/usr/bin/env python3
"""D2-R1/R2/R3 preparation: support table, universe check and re-adjudication preview
from the CONSERVED fresh confirmation rows (no operator is re-executed).

Per (arm, unit, variable, metric) on the confirmation partition the support table says
whether the metric applies by contract (regime noise, event kinds present in the window
according to the unit's own event list, extreme geometry), what the evaluator recorded
(OBSERVED | INCONCLUSIVE | UNAVAILABLE | REFUSED | FAILED | MISSING_ROW | NOT_APPLICABLE),
the reason, whether that leaves the metric SUPPORTED or UNSUPPORTED, and the component
responsible. The universe check compares the expected units x arms x variables x
metrics of the sealed design and reserve with what physically arrived. The preview runs
the repaired adjudicator over the same rows and tabulates every old -> new decision.

The preview is NOT a governed re-adjudication: D2-R3 registers it as a Flow v3 review
campaign once the production micro-run is reconciled; until then it is evidence for
review, it grants nothing, and it never overwrites a published row.

usage: df_d2_support.py --design DESIGN.json --tables TABLES_DIR --reserve RESERVE_ROOT
                        --decisions DECISIONS.jsonl --out OUT_DIR [--evidence EVIDENCE_DIR]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _load("df_d2_design")
A = _load("df_d2_adjudicate")

STATUS_MAP = {"COMPLETED": "OBSERVED", "INCONCLUSIVE": "INCONCLUSIVE", "UNAVAILABLE": "UNAVAILABLE",
              "REFUSED": "REFUSED", "FAILED": "FAILED", "NOT_APPLICABLE": "NOT_APPLICABLE"}
COMPONENT = {"INCONCLUSIVE": "df_lab_evaluation.partition_metrics (metric undefined on this window)",
             "UNAVAILABLE": "df_lab_evaluation.partition_metrics (fewer available samples than MIN_SUPPORT)",
             "REFUSED": "df_d2_unit_worker missing-data rule (arm refused whole)",
             "FAILED": "df_d2_unit_worker / operator", "MISSING_ROW": "df_d2_unit_worker (row never emitted)",
             "OBSERVED": "", "NOT_APPLICABLE": ""}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_by_regime(table: Path, work: Path) -> dict:
    """One pass over the conserved denoising rows -> one file per regime (confirmation and
    COST rows only, as the adjudicator consumes them). Returns regime_key -> path."""
    work.mkdir(parents=True, exist_ok=True)
    handles, paths, counts = {}, {}, collections.Counter()
    kept = dropped = 0
    try:
        with open(table, "rb") as source:
            for raw in source:
                row = json.loads(raw)
                if row["partition"] != "confirmation" and not (row["branch"] == "COST" and row["partition"] == "all"):
                    dropped += 1
                    continue
                key = D.regime_key(row["regime"])
                if key not in handles:
                    paths[key] = work / f"regime_{hashlib.sha256(key.encode()).hexdigest()[:16]}.jsonl"
                    handles[key] = open(paths[key], "wb")
                handles[key].write(raw if raw.endswith(b"\n") else raw + b"\n")
                counts[key] += 1
                kept += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {"paths": paths, "counts": dict(counts), "rows_kept": kept, "rows_dropped_other_partitions": dropped}


LAB = _load("df_lab_evaluation")


def evaluable(ev: dict, lo: int, hi: int) -> bool:
    """Whether the evaluator can measure this event's floor metric on the window by
    index geometry alone (mirrors df_lab_evaluation.event_metrics; sample finiteness is
    the evaluator's own business and shows up as a disagreement in MCAR regimes)."""
    i = int(ev.get("index", -1))
    if not (lo <= i < hi):
        return False
    t = ev.get("type")
    if t == "impulse":
        return i - max(lo, i - LAB.EVENT_PRE) >= 3
    if t == "bump":
        hw = int(ev.get("support_halfwidth", 0))
        a, b = max(lo, i - hw), min(hi, i + hw + 1)
        return (b - a) >= 3 and (a - max(lo, a - LAB.EVENT_PRE)) >= 3
    if t == "motif":
        return i + int(ev.get("length", 0)) <= hi
    if t == "step":
        return True
    if t == "regime_boundary":
        return ev.get("regime") == "mean"
    return False


def unit_events(reserve: Path, unit_id: str) -> tuple[dict, dict]:
    """Event kinds per variable that are evaluable inside the confirmation window, from
    the unit's own truth."""
    udir = reserve / unit_id
    unit = json.loads((udir / "UNIT.json").read_text(encoding="utf-8"))
    lo, hi = unit["partitions"]["confirmation"]
    events = json.loads((udir / "events.json").read_text(encoding="utf-8"))["events"]
    per_var: dict = {}
    for ev in events:
        if evaluable(ev, lo, hi):
            v = int(ev.get("variable", 0))
            kind = "regime_mean" if ev.get("type") == "regime_boundary" else str(ev.get("type"))
            per_var.setdefault(v, collections.Counter())[kind] += 1
    return unit, per_var


EVENT_METRIC_OF_TYPE = {"bump": "bump_retention", "impulse": "impulse_retention", "motif": "motif_corr",
                        "regime_mean": "regime_mean_latency_samples", "step": "step_delay_samples"}


def support_rows(regime_rows: list, design: dict, reserve: Path, truth_cache: dict) -> tuple[list, dict]:
    """Support table rows for one regime, plus the universe check for it."""
    R = design["denoising_rules"]
    ops = {o["spec_sha256"]: o for o in design["operators"]}
    by_arm: dict = {}
    for r in regime_rows:
        by_arm.setdefault(r["spec_sha256"], []).append(r)
    regime = regime_rows[0]["regime"]
    noise_free = str(regime["declared_snr_db"]) == "inf"
    required = [R["improvement"]["metric"], *A.PRIMARY_METRICS, "extreme_retention", "extreme_retention_raw",
                *R["event_floors"], *sum(([m, raw] for m, raw in R["non_inferiority"]["pairs"].items()), [])]
    required = list(dict.fromkeys(required))
    out, universe = [], {"expected_rows": 0, "observed_rows": 0, "missing_rows": 0, "events_disagree": 0,
                         "events_disagree_missingness_none": 0}
    for spec_sha, rs in sorted(by_arm.items()):
        op = ops[spec_sha]
        seeds = A._seed_table(rs)
        for unit_id, s in sorted(seeds.items()):
            if unit_id not in truth_cache:
                truth_cache[unit_id] = unit_events(reserve, unit_id)
            unit, truth_events = truth_cache[unit_id]
            variables = sorted(s["variables"]) or list(range(int(unit.get("n_variables", 1))))
            for v in variables:
                truth = truth_events.get(v, {})
                # the evaluator's event counts must agree with the unit's truth
                for ev_type, n in truth.items():
                    metric = EVENT_METRIC_OF_TYPE.get(ev_type)
                    if metric and float(s["events"].get(v, {}).get(metric, 0)) != float(n):
                        universe["events_disagree"] += 1
                        if str(regime.get("missingness")) == "none":
                            universe["events_disagree_missingness_none"] += 1
                for metric in required:
                    base = metric[:-4] if metric.endswith("_raw") else metric
                    if metric == R["improvement"]["metric"]:
                        applicable = not noise_free
                    elif metric in A.PRIMARY_METRICS:
                        applicable = True
                    elif base in ("extreme_retention",):
                        applicable = s["states"].get(v, {}).get("extreme_retention_raw") != "INCONCLUSIVE"
                    else:
                        applicable = any(EVENT_METRIC_OF_TYPE.get(t) == base for t in truth)
                    state = s["states"].get(v, {}).get(metric)
                    if s["abstained"]:
                        status = "REFUSED"
                    elif s["states"].get(v, {}).get("partition_support") == "UNAVAILABLE":
                        status = "UNAVAILABLE"
                    elif not applicable:
                        status = "NOT_APPLICABLE"
                    elif state is None:
                        status = "MISSING_ROW"
                    else:
                        status = STATUS_MAP.get(state, state)
                    universe["expected_rows"] += 1 if applicable else 0
                    universe["observed_rows"] += 1 if (applicable and state is not None) else 0
                    universe["missing_rows"] += 1 if (applicable and status == "MISSING_ROW") else 0
                    support = ("NOT_APPLICABLE" if not applicable else
                               "SUPPORTED" if status == "OBSERVED" else "UNSUPPORTED")
                    out.append({"spec_sha256": spec_sha, "operator": op["kind"], "arm_role": op["arm_role"],
                                "regime_key": D.regime_key(regime), "unit_id": unit_id, "variable_index": v,
                                "metric": metric, "applicable": applicable, "status": status,
                                "reason": ("" if status in ("OBSERVED", "NOT_APPLICABLE") else
                                           (s.get("abstain_reason") if s["abstained"] else "undefined for this unit"
                                            if status == "INCONCLUSIVE" else status)),
                                "support": support, "component": COMPONENT.get(status, "")})
    return out, universe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True, help="collected tables dir (df_fact_d2_unit_*.jsonl)")
    ap.add_argument("--reserve", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True, help="published DECISIONS.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="work/output dir (created)")
    ap.add_argument("--evidence", type=Path, help="where to copy the compact summaries")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"out dir exists: {out}")
    out.mkdir(parents=True)
    design = json.loads(a.design.read_text(encoding="utf-8"))
    D.require_valid(design)
    published_sha = sha256_file(a.decisions)
    published = [json.loads(l) for l in open(a.decisions, encoding="utf-8")]
    den_table = a.tables / "df_fact_d2_unit_denoising.jsonl"
    snr_table = a.tables / "df_fact_d2_unit_snr.jsonl"
    split = split_by_regime(den_table, out / "by_regime")
    truth_cache: dict = {}
    summary_universe = collections.Counter()
    support_summary: dict = {}
    new_decisions = []
    with open(out / "SUPPORT_TABLE.jsonl", "w", encoding="utf-8") as table:
        for key, path in sorted(split["paths"].items()):
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            srows, universe = support_rows(rows, design, a.reserve, truth_cache)
            for r in srows:
                table.write(json.dumps(r, sort_keys=True) + "\n")
            for k, v in universe.items():
                summary_universe[k] += v
            per = collections.Counter((r["operator"], r["metric"], r["support"]) for r in srows)
            support_summary[key] = {f"{o}|{m}|{s}": n for (o, m, s), n in sorted(per.items())}
            new_decisions += A.decide_denoising(rows, design)
    snr_rows = [json.loads(l) for l in open(snr_table, encoding="utf-8")]
    snr_rows = [r for r in snr_rows if r["partition"] == "confirmation"]
    new_decisions += A.decide_snr(snr_rows, design)

    def ident(d):
        return (d["subject_kind"], d["subject"], json.dumps(d.get("operator_params"), sort_keys=True),
                d.get("spec_sha256"), D.regime_key(d["regime"]))
    old = {ident(d): d for d in published}
    new = {ident(d): d for d in new_decisions}
    impact = []
    flips = collections.Counter()
    for k in sorted(set(old) | set(new), key=str):
        o, n = old.get(k), new.get(k)
        row = {"subject_kind": k[0], "subject": k[1], "operator_params": json.loads(k[2]), "spec_sha256": k[3],
               "regime_key": k[4], "old": o["decision"] if o else None, "new": n["decision"] if n else None,
               "old_n_valid": o["n_seeds_valid"] if o else None, "new_n_valid": n["n_seeds_valid"] if n else None,
               "new_reasons": n["reasons"] if n else None,
               "support": (n["evidence"].get("support") if n else None)}
        row["changed"] = row["old"] != row["new"]
        flips[(row["old"], row["new"])] += 1
        impact.append(row)
    with open(out / "DECISIONS_PREVIEW.jsonl", "w", encoding="utf-8") as handle:
        for d in new_decisions:
            handle.write(json.dumps(dict(d, preview="NON_GOVERNING_PREVIEW_R3_PENDING"), sort_keys=True, default=float) + "\n")
    impact_doc = {"schema": "d2_support_readjudication_preview.v1", "governing": False,
                  "note": "preview from conserved rows with the repaired adjudicator; D2-R3 registers the governed re-adjudication",
                  "published_decisions_sha256": published_sha, "design_sha256": design["design_sha256"],
                  "adjudicator_code_sha256": sha256_file(HERE / "df_d2_adjudicate.py"), "support_code_sha256": sha256_file(Path(__file__)),
                  "decisions_published": len(published), "decisions_preview": len(new_decisions),
                  "changed": sum(r["changed"] for r in impact),
                  "transitions": {f"{o} -> {n}": c for (o, n), c in sorted(flips.items(), key=lambda x: (-x[1], str(x[0])))},
                  "universe": dict(summary_universe), "split": {k: v for k, v in split.items() if k != "paths"},
                  "changed_rows": [r for r in impact if r["changed"]]}
    (out / "IMPACT_TABLE.json").write_text(json.dumps(impact_doc, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "IMPACT_ROWS.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=float) + "\n" for r in impact), encoding="utf-8")
    (out / "SUPPORT_SUMMARY.json").write_text(json.dumps({"universe": dict(summary_universe), "by_regime": support_summary},
                                                          indent=1) + "\n", encoding="utf-8")
    (out / "UNIVERSE_CHECK.json").write_text(json.dumps({"schema": "d2_universe_check.v1", **dict(summary_universe),
                                                         "ok": summary_universe["missing_rows"] == 0
                                                         and summary_universe["events_disagree_missingness_none"] == 0},
                                                        indent=1) + "\n", encoding="utf-8")
    if a.evidence:
        a.evidence.mkdir(parents=True, exist_ok=True)
        for name in ("IMPACT_TABLE.json", "UNIVERSE_CHECK.json"):
            (a.evidence / name).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({k: impact_doc[k] for k in ("decisions_published", "decisions_preview", "changed", "transitions", "universe")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
