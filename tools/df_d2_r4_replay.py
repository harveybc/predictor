#!/usr/bin/env python3
"""D2-R4: numerical portability of the SNR estimators, measured on a frozen subset.

The subset, the seeds and the comparison criterion are fixed BEFORE any replay
(`R4_DIAGNOSTIC_SUBSET.json`, produced by `df_d2_r4_margins.py`). This tool never
modifies the code that produced the historical observations: it calls the same
entry point (`df_snr.py --d2-unit-facts`) in a child process, one unit at a time,
one linear-algebra thread, and records the environment of the role that ran it.

    --replay   recompute the SNR facts of every unit of the subset on this role
    --compare  put replays and the conserved rows side by side and separate
               byte equality, numerical tolerance and decision stability

Decision stability is measured, not inferred: for every regime of the subset the
conserved rows are re-adjudicated with the replayed values substituted, and the
resulting decision is compared with the published one. The historical tolerance
is never widened; the observed deviation is reported against it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
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


D = _load("df_d2_design")


def environment() -> dict:
    import numpy
    import scipy
    try:
        import statsmodels
        sm = statsmodels.__version__
    except Exception:
        sm = None
    blas = {}
    try:
        cfg = numpy.show_config(mode="dicts")
        blas = {k: v.get("name") for k, v in cfg.get("Build Dependencies", {}).items() if k in ("blas", "lapack")}
    except Exception:
        blas = {"unavailable": True}
    cpu = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu_model": cpu, "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "statsmodels": sm, "blas_lapack": blas,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "float_precision": "float64", "libc": platform.libc_ver()[1], "kernel": platform.release()}


def replay(design: dict, reserve: Path, subset: dict, out: Path, role: str, limit=None) -> dict:
    units = sorted({u for group in subset["units"].values() for u in group})
    if limit:
        units = units[:limit]
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    snr = design["snr"]
    facts, timings = {}, {}
    for unit in units:
        job_path, out_path = work / f"{unit}.job.json", work / f"{unit}.facts.json"
        if out_path.exists():
            facts[unit] = json.loads(out_path.read_text(encoding="utf-8"))
            continue
        job_path.write_text(json.dumps({"unit_dir": str(reserve / unit), "estimators": snr["estimators"],
                                        "partitions": snr["partitions"], "bootstrap": snr.get("bootstrap"),
                                        "out": str(out_path)}, sort_keys=True), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.monotonic()
        proc = subprocess.run([sys.executable, "-B", str(HERE / "df_snr.py"), "--d2-unit-facts", str(job_path)],
                              capture_output=True, text=True, env=env, timeout=3600)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        if proc.returncode != 0 or not out_path.is_file():
            raise SystemExit(f"REFUSED: df_snr failed on {unit} (exit {proc.returncode}): {proc.stderr[-400:]}")
        facts[unit] = json.loads(out_path.read_text(encoding="utf-8"))
        timings[unit] = {"wall_seconds": time.monotonic() - start,
                         "cpu_seconds": (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
                         "max_rss_bytes_children": after.ru_maxrss * 1024}
    doc = {"schema": "d2_r4_replay.v1", "role": role, "environment": environment(),
           "design_sha256": design["design_sha256"], "subset_rule": subset["rule"],
           "df_snr_code_sha256": {u: f["code_sha256"] for u, f in facts.items()},
           "units": {u: f["facts"] for u, f in facts.items()}, "timings": timings,
           "cpu_seconds_total": sum(t["cpu_seconds"] for t in timings.values()),
           "peak_rss_bytes": max((t["max_rss_bytes_children"] for t in timings.values()), default=0)}
    path = out / f"R4_REPLAY_{role}.json"
    path.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return {"role": role, "units": len(facts), "path": str(path), "cpu_seconds": doc["cpu_seconds_total"],
            "peak_rss_bytes": doc["peak_rss_bytes"]}


def _key(fact):
    return (fact["variable_index"], fact["estimator"], fact["partition"])


def _hat(fact):
    result = fact.get("result") or {}
    return result.get("snr_db"), result.get("status"), (result.get("reason") or ""), (result.get("bootstrap") or {})


def compare(design: dict, replays: list, snr_table: Path, decisions: Path, out: Path, tolerance_db: float,
            subset: dict | None = None) -> dict:
    A = _load("df_d2_adjudicate")
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in replays]
    roles = [d["role"] for d in docs]
    units = sorted({u for d in docs for u in d["units"]})
    requested_units = sorted({u for group in (subset or {}).get("units", {}).values() for u in group})
    replayed_facts = sum(len(f) for d in docs for f in d["units"].values())
    historical = {}
    regimes = {}
    with open(snr_table, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["unit_id"] in set(units):
                historical[(row["unit_id"], row["variable_index"], row["estimator"], row["partition"])] = row
                regimes.setdefault(D.regime_key(row["regime"]), set()).add(row["unit_id"])
    published = {}
    with open(decisions, encoding="utf-8") as handle:
        for line in handle:
            d = json.loads(line)
            if d["subject_kind"] == "SNR_ESTIMATOR":
                published[(d["subject"], D.regime_key(d["regime"]))] = d
    cells, byte_equal, within, outside, state_changed, convergence = [], 0, 0, 0, 0, []
    unmatched_facts = 0
    per_estimator: dict[str, dict] = {}
    for unit in units:
        for doc in docs:
            for fact in doc["units"].get(unit, []):
                v, est, part = _key(fact)
                hat, status, reason, boot = _hat(fact)
                row = historical.get((unit, v, est, part))
                if row is None:
                    unmatched_facts += 1
                    continue
                ref = row["snr_db_hat"]
                same_bytes = (hat is not None and ref is not None
                              and json.dumps(hat, sort_keys=True) == json.dumps(ref, sort_keys=True))
                delta = None if (hat is None or ref is None) else abs(float(hat) - float(ref))
                ident_now = ("ESTIMATED" if status == "ESTIMATED" else
                             ("NOT_APPLICABLE" if fact["status"] == "NOT_APPLICABLE" else "NOT_IDENTIFIABLE"))
                changed = ident_now != row["identifiability"]
                cells.append({"role": doc["role"], "unit_id": unit, "variable_index": v, "estimator": est,
                              "partition": part, "historical": ref, "replay": hat, "abs_delta_db": delta,
                              "bytes_equal": same_bytes, "identifiability_historical": row["identifiability"],
                              "identifiability_replay": ident_now, "identifiability_changed": changed,
                              "ci_historical": [row.get("ci_low_db"), row.get("ci_high_db")],
                              "ci_replay": [boot.get("ci_low_db"), boot.get("ci_high_db")],
                              "reason_replay": reason})
                byte_equal += 1 if same_bytes else 0
                stats = per_estimator.setdefault(est, {"cells": 0, "bytes_equal": 0, "comparable": 0,
                                                       "within_tolerance": 0, "outside_tolerance": 0,
                                                       "max_abs_delta_db": None})
                stats["cells"] += 1
                stats["bytes_equal"] += 1 if same_bytes else 0
                if delta is not None:
                    within += 1 if delta <= tolerance_db else 0
                    outside += 0 if delta <= tolerance_db else 1
                    stats["comparable"] += 1
                    stats["within_tolerance"] += 1 if delta <= tolerance_db else 0
                    stats["outside_tolerance"] += 0 if delta <= tolerance_db else 1
                    stats["max_abs_delta_db"] = delta if stats["max_abs_delta_db"] is None \
                        else max(stats["max_abs_delta_db"], delta)
                state_changed += 1 if changed else 0
                if reason:
                    convergence.append({"role": doc["role"], "unit_id": unit, "estimator": est, "reason": reason})
    # decision stability: re-adjudicate the subset's regimes with the replayed values substituted
    stability = []
    by_regime = {}
    with open(snr_table, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = D.regime_key(row["regime"])
            if key in regimes and row["partition"] == "confirmation":
                by_regime.setdefault(key, []).append(row)
    substituted_facts = 0
    regimes_skipped_no_substitution = set()
    decisions_unmatched = 0
    for doc in docs:
        for key, rows in by_regime.items():
            patched = []
            touched = 0
            for row in rows:
                fact = next((f for f in doc["units"].get(row["unit_id"], [])
                             if _key(f) == (row["variable_index"], row["estimator"], row["partition"])), None)
                if fact is None:
                    patched.append(row)
                    continue
                hat, status, _reason, boot = _hat(fact)
                new = dict(row)
                if status == "ESTIMATED" and hat is not None and row["true_snr_db"] is not None:
                    new.update(snr_db_hat=hat, error_db=hat - row["true_snr_db"],
                               abs_error_db=abs(hat - row["true_snr_db"]), identifiability="ESTIMATED",
                               ci_low_db=boot.get("ci_low_db"), ci_high_db=boot.get("ci_high_db"))
                    low, high = boot.get("ci_low_db"), boot.get("ci_high_db")
                    if low is not None and high is not None:
                        new["ci_covers_true"] = 1.0 if low <= row["true_snr_db"] <= high else 0.0
                    touched += 1
                    substituted_facts += 1
                patched.append(new)
            if not touched:
                regimes_skipped_no_substitution.add(key)
                continue
            for decision in A.decide_snr(patched, design):
                old = published.get((decision["subject"], key))
                if old is None:
                    decisions_unmatched += 1
                    continue
                stability.append({"role": doc["role"], "estimator": decision["subject"], "regime_key": key,
                                  "published": old["decision"], "replayed": decision["decision"],
                                  "changed": old["decision"] != decision["decision"],
                                  "published_upper": (old["evidence"].get("ci95_abs_error_db") or [None, None])[1],
                                  "replayed_upper": (decision["evidence"].get("ci95_abs_error_db") or [None, None])[1]})
    coverage = {"units_requested": len(requested_units), "units_replayed": len(units),
                "units_missing": sorted(set(requested_units) - set(units)),
                "replayed_facts": replayed_facts, "compared_cells": len(cells),
                "facts_without_historical_row": unmatched_facts,
                "regimes_in_scope": len(by_regime), "substituted_facts": substituted_facts,
                "regimes_without_substitution": sorted(regimes_skipped_no_substitution),
                "decisions_without_published_counterpart": decisions_unmatched,
                "decisions_compared": len(stability)}
    refusals = []
    if not docs:
        refusals.append("NO_REPLAY_FILES")
    if replayed_facts == 0:
        refusals.append("NO_REPLAYED_FACTS")
    if not cells:
        refusals.append("NO_COMPARED_CELLS")
    if substituted_facts == 0:
        refusals.append("NO_SUBSTITUTED_FACTS")
    if not stability:
        refusals.append("NO_DECISIONS_COMPARED")
    if requested_units and coverage["units_missing"]:
        refusals.append("UNITS_MISSING_FROM_REPLAY")
    if coverage["regimes_without_substitution"]:
        refusals.append("REGIMES_WITHOUT_SUBSTITUTION")
    verdict = "MEASURED" if not refusals else "INCONCLUSIVE"
    report = {"schema": "d2_r4_portability_report.v2", "roles": roles, "units": len(units),
              "coverage": coverage, "verdict": verdict, "inconclusive_reasons": refusals,
              "per_estimator": per_estimator,
              "tolerance_db_original": tolerance_db,
              "cells": len(cells), "bytes_equal": byte_equal, "within_tolerance": within,
              "outside_tolerance": outside, "identifiability_changed": state_changed,
              "max_abs_delta_db": max((c["abs_delta_db"] for c in cells if c["abs_delta_db"] is not None), default=None),
              "convergence_reasons": convergence[:50],
              "decision_stability": {"verdict": verdict,
                                     "compared": len(stability),
                                     "changed": sum(s["changed"] for s in stability) if stability else None,
                                     "changed_rows": [s for s in stability if s["changed"]],
                                     "note": ("stability is asserted only over the compared scope"
                                              if verdict == "MEASURED"
                                              else "no stability conclusion: " + ", ".join(refusals))},
              "environments": {d["role"]: d["environment"] for d in docs},
              "cpu_seconds": {d["role"]: d["cpu_seconds_total"] for d in docs},
              "peak_rss_bytes": {d["role"]: d["peak_rss_bytes"] for d in docs},
              "outside_tolerance_cells": [c for c in cells if c["abs_delta_db"] is not None
                                          and c["abs_delta_db"] > tolerance_db][:200]}
    (out / "R4_PORTABILITY_REPORT.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with open(out / "R4_CELLS.jsonl", "w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
    return {k: report[k] for k in ("roles", "units", "verdict", "inconclusive_reasons", "coverage",
                                   "cells", "bytes_equal", "within_tolerance",
                                   "outside_tolerance", "identifiability_changed", "max_abs_delta_db",
                                   "per_estimator", "decision_stability")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--reserve", type=Path)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", default=os.environ.get("CRISPDM_ROLE", "COORDINATOR"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--replay-file", action="append", default=[])
    ap.add_argument("--snr-table", type=Path)
    ap.add_argument("--decisions", type=Path)
    ap.add_argument("--tolerance-db", type=float, default=1e-9,
                    help="the original AT9 criterion: bytes/float equality of the replayed estimate")
    a = ap.parse_args(argv)
    design = json.loads(a.design.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    subset = json.loads(a.subset.read_text(encoding="utf-8"))
    result = {}
    if a.replay:
        if not a.reserve:
            raise SystemExit("--replay needs --reserve")
        result["replay"] = replay(design, a.reserve, subset, a.out, a.role, a.limit)
    if a.compare:
        if not (a.replay_file and a.snr_table and a.decisions):
            raise SystemExit("--compare needs --replay-file, --snr-table and --decisions")
        result["compare"] = compare(design, a.replay_file, a.snr_table, a.decisions, a.out, a.tolerance_db,
                                    subset=subset)
    print(json.dumps(result, indent=1, default=float))
    if result.get("compare", {}).get("verdict") == "INCONCLUSIVE":
        print("REFUSED: the comparison covers nothing it claims to measure: "
              + ", ".join(result["compare"]["inconclusive_reasons"]), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
