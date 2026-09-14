#!/usr/bin/env python3
"""D2-R4 preparation: margins of every SNR decision against ALL its conditions, the
environment inventory of this role, and a frozen diagnostic subset chosen by fixed rule.

Nothing is re-executed here. From the published decisions (read by digest) each SNR
decision gets its distance to every threshold of the sealed rules: upper bound of the
mean absolute error vs 1 dB, coverage vs 0.90, not-identifiable rate vs 0.10 and
identifiable seeds vs the design. The diagnostic subset is fixed BEFORE any replay by
rule, not by looking at a new replay: the AT9 case, one non-iterative estimator control
per Kalman regime, the calibrated Kalman regimes, and the k decisions closest to each
limit; seeds are the two lowest seed ids of each regime in the reserve.

usage: df_d2_r4_margins.py --decisions DECISIONS.jsonl --design DESIGN.json --reserve ROOT --out DIR [--k 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_inventory() -> dict:
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
    return {"role": os.environ.get("CRISPDM_ROLE", "COORDINATOR"), "cpu_model": cpu, "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "statsmodels": sm, "blas_lapack": blas,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "float_precision": "float64 (numpy default)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--reserve", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--at9-unit", default="", help="unit id of the AT9 cross-host case, if known")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    design = json.loads(a.design.read_text(encoding="utf-8"))
    R = design["snr_rules"]
    rows = [json.loads(l) for l in open(a.decisions, encoding="utf-8")]
    snr = [d for d in rows if d["subject_kind"] == "SNR_ESTIMATOR"]
    margins = []
    for d in snr:
        ev = d["evidence"]
        upper = (ev.get("ci95_abs_error_db") or [None, None])[1]
        lower = (ev.get("ci95_abs_error_db") or [None, None])[0]
        margins.append({
            "estimator": d["subject"], "regime": d["regime"], "decision": d["decision"],
            "upper_minus_limit_db": None if upper is None else upper - R["mean_abs_error_ci_upper_max_db"],
            "lower_minus_limit_db": None if lower is None else lower - R["mean_abs_error_ci_upper_max_db"],
            "coverage_minus_min": None if ev.get("coverage") is None else ev["coverage"] - R["coverage_min"],
            "ni_rate_minus_max": None if ev.get("not_identifiable_rate") is None else ev["not_identifiable_rate"] - R["not_identifiable_rate_max"],
            "identifiable_minus_design": ev.get("n_seeds_identifiable", 0) - ev.get("n_seeds_design", 0),
            "n_seeds_identifiable": ev.get("n_seeds_identifiable"), "n_seeds_design": ev.get("n_seeds_design"),
        })
    closest = {}
    for key in ("upper_minus_limit_db", "lower_minus_limit_db", "coverage_minus_min", "ni_rate_minus_max"):
        ranked = sorted((m for m in margins if m[key] is not None), key=lambda m: abs(m[key]))
        closest[key] = ranked[:a.k]

    def regime_key(regime):
        return json.dumps(regime, sort_keys=True)

    kalman_calibrated = [m for m in margins if "kalman" in m["estimator"] and m["decision"] == "SNR_CALIBRATED_FOR_REGIME"]
    non_iterative = {"mad_first_difference", "wavelet_mad", "welch_noise_floor", "ar_residual"}
    controls = [m for m in margins if m["estimator"] in non_iterative and regime_key(m["regime"]) in
                {regime_key(k["regime"]) for k in kalman_calibrated}]
    subset_regimes = {}
    for m in kalman_calibrated + controls + sum(closest.values(), []):
        subset_regimes.setdefault(regime_key(m["regime"]), set()).add(m["estimator"])
    # seeds by fixed rule: the two lowest seed ids of each regime in the reserve
    reserve_units = sorted(p.name for p in a.reserve.iterdir() if p.is_dir())
    chosen_units = {}
    for rk in subset_regimes:
        regime = json.loads(rk)
        prefix = f"{regime['family']}__{regime['perturbation']}__snr{regime['declared_snr_db']}__{regime['missingness']}__n{regime['length']}__"
        units = sorted(u for u in reserve_units if u.startswith(prefix))
        units = sorted(units, key=lambda u: int(u.rsplit("seed", 1)[-1]))[:2]
        chosen_units[rk] = units
    subset = {"schema": "d2_r4_diagnostic_subset.v1", "rule": "AT9 case + non-iterative controls of the calibrated Kalman regimes + "
              "the k closest decisions to each limit; two lowest seed ids per regime; frozen before any replay",
              "k": a.k, "at9_unit": a.at9_unit or None, "kalman_calibrated": kalman_calibrated,
              "controls": controls, "closest": closest,
              "units": chosen_units, "n_units": sum(len(v) for v in chosen_units.values()),
              "estimated_cost": {"cpu_hours_max": 6.0, "wall_hours_max": 4.0, "per_unit_process": 1, "hard_limit_gib": 2},
              "not_executed": True}
    out = {"schema": "d2_r4_snr_margins.v1", "decisions_sha256": sha256_file(a.decisions), "design_sha256": design["design_sha256"],
           "thresholds": {k: R[k] for k in ("mean_abs_error_ci_upper_max_db", "coverage_min", "not_identifiable_rate_max")},
           "n_snr_decisions": len(snr), "by_decision": {k: sum(m["decision"] == k for m in margins) for k in
                                                        ("SNR_CALIBRATED_FOR_REGIME", "SNR_REGIME_LIMITED", "SNR_REJECTED", "SNR_NOT_IDENTIFIABLE")},
           "margins": margins}
    (a.out / "R4_SNR_MARGINS.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    (a.out / "R4_DIAGNOSTIC_SUBSET.json").write_text(json.dumps(subset, indent=1, default=float) + "\n", encoding="utf-8")
    (a.out / "R4_ENVIRONMENT_INVENTORY.json").write_text(json.dumps(environment_inventory(), indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n_snr_decisions": len(snr), "by_decision": out["by_decision"], "subset_regimes": len(subset_regimes),
                      "subset_units": subset["n_units"], "kalman_calibrated": len(kalman_calibrated)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
