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
import math
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


def _state_of(fact):
    """What a replayed fact is, for the decision input: an estimate, or explicitly not one."""
    result = fact.get("result") or {}
    status = result.get("status")
    value = result.get("snr_db")
    if status != "ESTIMATED":
        return ("NOT_IDENTIFIABLE", None)
    if value is None:
        return ("NOT_IDENTIFIABLE", None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ("INVALID_VALUE", None)
    if not math.isfinite(number):
        return ("INVALID_VALUE", None)
    return ("ESTIMATED", number)


def compare(design: dict, replays: list, snr_table: Path, decisions: Path, out: Path, tolerance_db: float,
            subset: dict | None = None) -> dict:
    """Compare replays against the conserved rows over a *declared* population.

    The population is the frozen subset, not whatever the replay files happen to contain.
    Every selected (unit, variable, estimator, partition) key must be replayed exactly once
    by every role; anything else — an omitted unit, a missing fact, a duplicate, a second
    file for the same role, a replay of another design — makes the comparison INCONCLUSIVE
    instead of a zero-change summary. Rows outside the subset are carried untouched and
    counted separately from selected facts that are missing.
    """
    A = _load("df_d2_adjudicate")
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in replays]
    roles = [d["role"] for d in docs]
    duplicated_roles = sorted({r for r in roles if roles.count(r) > 1})
    wrong_design = sorted({d["role"] for d in docs
                           if design.get("design_sha256") and d.get("design_sha256")
                           and d["design_sha256"] != design["design_sha256"]})
    units = sorted({u for d in docs for u in d["units"]})
    selected_units = sorted({u for group in (subset or {}).get("units", {}).values() for u in group})
    population_units = set(selected_units) if selected_units else set(units)
    replayed_facts = sum(len(f) for d in docs for f in d["units"].values())

    historical, regimes, carried_unselected = {}, {}, 0
    with open(snr_table, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["unit_id"], row["variable_index"], row["estimator"], row["partition"])
            if row["unit_id"] in population_units:
                historical[key] = row
                regimes.setdefault(D.regime_key(row["regime"]), set()).add(row["unit_id"])
            else:
                carried_unselected += 1
    expected_keys = sorted(historical)
    published = {}
    with open(decisions, encoding="utf-8") as handle:
        for line in handle:
            d = json.loads(line)
            if d["subject_kind"] == "SNR_ESTIMATOR":
                published[(d["subject"], D.regime_key(d["regime"]))] = d

    # index every role's facts by key, and refuse a key a role reports twice
    by_role, duplicated_facts, unmatched_facts = {}, [], 0
    for doc in docs:
        index = {}
        for unit, facts in doc["units"].items():
            for fact in facts:
                key = (unit, *_key(fact))
                if key in index:
                    duplicated_facts.append({"role": doc["role"], "key": list(key)})
                    continue
                index[key] = fact
                if key not in historical:
                    unmatched_facts += 1
        by_role.setdefault(doc["role"], {}).update(index)

    units_missing_by_role = {role: sorted(u for u in population_units
                                          if not any(k[0] == u for k in index))
                             for role, index in by_role.items()}
    missing_selected = {role: [list(k) for k in expected_keys if k not in index]
                        for role, index in by_role.items()}
    missing_selected_facts = sum(len(v) for v in missing_selected.values())

    cells, byte_equal, within, outside, state_changed, convergence = [], 0, 0, 0, 0, []
    per_estimator: dict = {}
    for doc in docs:
        index = by_role[doc["role"]]
        for key in expected_keys:
            fact = index.get(key)
            if fact is None:
                continue
            unit, v, est, part = key
            hat, status, reason, boot = _hat(fact)
            if isinstance(hat, float) and not math.isfinite(hat):
                hat = None          # a non-finite replay is not a value; it is a state
            row = historical[key]
            ref = row["snr_db_hat"]
            same_bytes = (hat is not None and ref is not None
                          and json.dumps(hat, sort_keys=True) == json.dumps(ref, sort_keys=True))
            delta = None
            if hat is not None and ref is not None:
                try:
                    candidate = abs(float(hat) - float(ref))
                    delta = candidate if math.isfinite(candidate) else None
                except (TypeError, ValueError):
                    delta = None
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

    # decision stability: re-adjudicate each regime of the population with the replayed
    # values substituted. A replayed non-estimate replaces the published estimate; it never
    # leaves it standing.
    stability, substituted_facts, substituted_states = [], 0, {}
    adjudication_refusals = []
    regimes_skipped_no_substitution, decisions_unmatched = set(), 0
    by_regime: dict = {}
    with open(snr_table, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = D.regime_key(row["regime"])
            if key in regimes and row["partition"] == "confirmation":
                by_regime.setdefault(key, []).append(row)
    for doc in docs:
        index = by_role[doc["role"]]
        for key, rows in by_regime.items():
            patched, touched = [], 0
            for row in rows:
                cell = (row["unit_id"], row["variable_index"], row["estimator"], row["partition"])
                if row["unit_id"] not in population_units:
                    patched.append(row)               # untouched, and counted as carried
                    continue
                fact = index.get(cell)
                new = dict(row)
                if fact is None:
                    substituted_states["MISSING"] = substituted_states.get("MISSING", 0) + 1
                    new.update(snr_db_hat=None, error_db=None, abs_error_db=None,
                               identifiability="NOT_IDENTIFIABLE", ci_low_db=None, ci_high_db=None,
                               ci_covers_true=None)
                    touched += 1
                    substituted_facts += 1
                    patched.append(new)
                    continue
                state, value = _state_of(fact)
                boot = (fact.get("result") or {}).get("bootstrap") or {}
                substituted_states[state] = substituted_states.get(state, 0) + 1
                if state == "ESTIMATED" and row["true_snr_db"] is not None:
                    new.update(snr_db_hat=value, error_db=value - row["true_snr_db"],
                               abs_error_db=abs(value - row["true_snr_db"]), identifiability="ESTIMATED",
                               ci_low_db=boot.get("ci_low_db"), ci_high_db=boot.get("ci_high_db"))
                    low, high = boot.get("ci_low_db"), boot.get("ci_high_db")
                    if low is not None and high is not None:
                        new["ci_covers_true"] = 1.0 if low <= row["true_snr_db"] <= high else 0.0
                else:
                    new.update(snr_db_hat=None, error_db=None, abs_error_db=None,
                               identifiability="NOT_IDENTIFIABLE", ci_low_db=None, ci_high_db=None,
                               ci_covers_true=None)
                touched += 1
                substituted_facts += 1
                patched.append(new)
            if not touched:
                regimes_skipped_no_substitution.add(key)
                continue
            try:
                decided = A.decide_snr(patched, design)
            except Exception as exc:               # a refusing adjudicator is not "no change"
                adjudication_refusals.append({"role": doc["role"], "regime_key": key,
                                              "error": f"{type(exc).__name__}: {exc}"})
                continue
            for decision in decided:
                old = published.get((decision["subject"], key))
                if old is None:
                    decisions_unmatched += 1
                    continue
                stability.append({"role": doc["role"], "estimator": decision["subject"], "regime_key": key,
                                  "published": old["decision"], "replayed": decision["decision"],
                                  "changed": old["decision"] != decision["decision"],
                                  "published_upper": (old["evidence"].get("ci95_abs_error_db") or [None, None])[1],
                                  "replayed_upper": (decision["evidence"].get("ci95_abs_error_db") or [None, None])[1]})

    non_estimates = sum(n for state, n in substituted_states.items() if state != "ESTIMATED")
    coverage = {"units_selected": len(selected_units), "units_replayed": len(units),
                "units_missing": sorted(set(selected_units) - set(units)),
                "units_missing_by_role": units_missing_by_role,
                "replayed_facts": replayed_facts, "compared_cells": len(cells),
                "facts_without_historical_row": unmatched_facts,
                "duplicated_facts": duplicated_facts, "duplicated_roles": duplicated_roles,
                "missing_selected_facts": missing_selected_facts,
                "missing_selected_by_role": missing_selected,
                "carried_unselected_rows": carried_unselected,
                "regimes_in_scope": len(by_regime), "substituted_facts": substituted_facts,
                "substituted_non_estimates": non_estimates,
                "regimes_without_substitution": sorted(regimes_skipped_no_substitution),
                "decisions_without_published_counterpart": decisions_unmatched,
                "decisions_compared": len(stability),
                "adjudication_refusals": adjudication_refusals[:20],
                "denominators": {"selected_units": len(selected_units) or len(units),
                                 "expected_facts_per_role": len(expected_keys),
                                 "roles": len(set(roles)),
                                 "expected_cells": len(expected_keys) * len(set(roles)),
                                 "compared_cells": len(cells),
                                 "regimes": len(by_regime),
                                 "decisions_compared": len(stability)}}
    refusals = []
    if not docs:
        refusals.append("NO_REPLAY_FILES")
    if duplicated_roles:
        refusals.append("DUPLICATED_ROLES")
    if wrong_design:
        refusals.append("REPLAY_OF_ANOTHER_DESIGN")
    if not selected_units:
        refusals.append("NO_FROZEN_SUBSET")
    if replayed_facts == 0:
        refusals.append("NO_REPLAYED_FACTS")
    if not cells:
        refusals.append("NO_COMPARED_CELLS")
    if duplicated_facts:
        refusals.append("DUPLICATED_FACTS")
    if any(units_missing_by_role.values()):
        refusals.append("ROLE_MISSING_SELECTED_UNITS")
    if missing_selected_facts:
        refusals.append("MISSING_SELECTED_FACTS")
    if unmatched_facts:
        refusals.append("FACTS_WITHOUT_HISTORICAL_ROW")
    if substituted_facts == 0:
        refusals.append("NO_SUBSTITUTED_FACTS")
    if not stability:
        refusals.append("NO_DECISIONS_COMPARED")
    if coverage["regimes_without_substitution"]:
        refusals.append("REGIMES_WITHOUT_SUBSTITUTION")
    if adjudication_refusals:
        refusals.append("ADJUDICATION_REFUSED")
    verdict = "MEASURED" if not refusals else "INCONCLUSIVE"
    report = {"schema": "d2_r4_portability_report.v3", "roles": roles, "units": len(units),
              "coverage": coverage, "verdict": verdict, "inconclusive_reasons": refusals,
              "per_estimator": per_estimator, "substituted_states": substituted_states,
              "tolerance_db_original": tolerance_db,
              "cells": len(cells), "bytes_equal": byte_equal, "within_tolerance": within,
              "outside_tolerance": outside, "identifiability_changed": state_changed,
              "max_abs_delta_db": max((c["abs_delta_db"] for c in cells if c["abs_delta_db"] is not None),
                                      default=None),
              "convergence_reasons": convergence[:50],
              "decision_stability": {"verdict": verdict,
                                     "compared": len(stability),
                                     "changed": (sum(s["changed"] for s in stability)
                                                 if stability and verdict == "MEASURED" else None),
                                     "changed_rows": [s for s in stability if s["changed"]],
                                     "note": ("stability is asserted only over the compared scope"
                                              if verdict == "MEASURED"
                                              else "no stability conclusion: " + ", ".join(refusals))},
              "environments": {d["role"]: d["environment"] for d in docs},
              "cpu_seconds": {d["role"]: d.get("cpu_seconds_total") for d in docs},
              "peak_rss_bytes": {d["role"]: d.get("peak_rss_bytes") for d in docs},
              "outside_tolerance_cells": [c for c in cells if c["abs_delta_db"] is not None
                                          and c["abs_delta_db"] > tolerance_db][:200]}
    (out / "R4_PORTABILITY_REPORT.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n",
                                                    encoding="utf-8")
    with open(out / "R4_CELLS.jsonl", "w", encoding="utf-8") as handle:
        for cell in cells:
            handle.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
    return {k: report[k] for k in ("roles", "units", "verdict", "inconclusive_reasons", "coverage",
                                   "cells", "bytes_equal", "within_tolerance",
                                   "outside_tolerance", "identifiability_changed", "max_abs_delta_db",
                                   "per_estimator", "substituted_states", "decision_stability")}


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
