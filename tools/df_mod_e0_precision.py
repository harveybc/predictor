#!/usr/bin/env python3
"""RP12: precision / power analysis DESIGN for the MOD-E0 estimands, by simulation, fed with the
replicate dispersion observed in the executed DEV pilot and with declared sensitivity multipliers.

It answers, for a number of replicates n (independent trajectories): what half-width of a 95 %
interval the design would give for e(h) per level, the H2 slope and the H3 gamma, and what effect
size would be detected with 80 % power at a two-sided 5 % test, if the replicate SD were the pilot's
(x1), half of it (x0.5) or double (x2). It uses the estimator the verifier uses (least-squares slope
over levels with per-level replicate means; gamma = d_1 - d_0 paired by replicate). It does NOT
propose confirmatory sizes: those follow the E0-CONF margin, which is not fixed here, and the
observed dispersion of three trajectories is itself uncertain (its own interval is reported).

    python tools/df_mod_e0_precision.py --close CLOSE.json --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPLICATES = (3, 5, 8, 10, 15, 20, 30, 50)
MULTIPLIERS = (0.5, 1.0, 2.0)
N_SIM = 4000
Z975, Z80 = 1.959964, 0.841621


def slope(levels, values):
    A = np.vstack([np.asarray(levels, dtype=float), np.ones(len(levels))]).T
    return float(np.linalg.lstsq(A, np.asarray(values), rcond=None)[0][0])


def sd_interval(sd: float, n: int) -> list:
    """95 % interval of a normal SD estimated from n replicates (chi-square)."""
    from scipy.stats import chi2
    df = n - 1
    return [float(sd * math.sqrt(df / chi2.ppf(0.975, df))), float(sd * math.sqrt(df / chi2.ppf(0.025, df)))]


def design(effects: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h2, h3 = effects.get("H2") or {}, effects.get("H3") or {}
    levels = [int(h) for h in (h2.get("levels") or [])]
    sd_e = {int(k): v for k, v in (h2.get("sd_replicates") or {}).items() if v is not None}
    sd_d = {int(k): v for k, v in (h3.get("sd_replicates") or {}).items() if v is not None}
    n_pilot = int(h2.get("replicates") or h3.get("n_units") or 3)
    out = {"schema": "df_mod_e0_precision_design.v1", "inputs": {"levels": levels, "sd_e_by_level": sd_e, "sd_d_by_r": sd_d, "pilot_replicates": n_pilot,
                                                              "sd_uncertainty_95": {f"e_h{k}": sd_interval(v, n_pilot) for k, v in sd_e.items()}
                                                              | {f"d_r{k}": sd_interval(v, n_pilot) for k, v in sd_d.items()}},
           "estimators": {"e(h)": "mean over replicates of MASE(profiles) - mean(random)", "slope": "least squares over levels of the per-level means",
                          "gamma": "d_1 - d_0 with d_r the mean over replicates; the replicate of r=1 and r=0 share the seed (paired)"},
           "assumptions": "replicate differences approximately normal with the pilot SD; the slope SD is derived analytically and by "
                          "simulation; gamma treats d_1 and d_0 as independent (conservative if they are positively correlated across seeds)",
           "sensitivity_multipliers": list(MULTIPLIERS), "rows": [], "not_a_confirmatory_size": True,
           "why_not": "the E0-CONF margin is not fixed; three trajectories give a dispersion whose own interval spans a factor ~4 (see sd_uncertainty_95)"}
    for mult in MULTIPLIERS:
        for n in REPLICATES:
            row = {"sd_multiplier": mult, "replicates": n}
            if sd_e and levels:
                half = {h: Z975 * mult * sd_e[h] / math.sqrt(n) for h in sd_e}
                row["e_h_half_width_95"] = {str(h): round(v, 5) for h, v in half.items()}
                # analytic slope SD: slope = sum(w_h * ebar_h) with w from least squares; ebar_h ~ N(., (m sd_h)^2 / n)
                L = np.asarray(levels, dtype=float)
                w = (L - L.mean()) / float(np.sum((L - L.mean()) ** 2))
                var_slope = float(sum((w[i] ** 2) * (mult * sd_e[h]) ** 2 / n for i, h in enumerate(levels) if h in sd_e))
                sims = []
                for _ in range(N_SIM):
                    ebar = [rng.normal(0.0, mult * sd_e[h], n).mean() for h in levels]
                    sims.append(slope(levels, ebar))
                row["slope_sd_analytic"] = round(math.sqrt(var_slope), 6)
                row["slope_sd_simulated"] = round(float(np.std(sims)), 6)
                row["slope_half_width_95"] = round(Z975 * math.sqrt(var_slope), 6)
                row["slope_detectable_at_80pct_power"] = round((Z975 + Z80) * math.sqrt(var_slope), 6)
            if 0 in sd_d and 1 in sd_d:
                var_gamma = ((mult * sd_d[0]) ** 2 + (mult * sd_d[1]) ** 2) / n
                row["gamma_half_width_95"] = round(Z975 * math.sqrt(var_gamma), 6)
                row["gamma_detectable_at_80pct_power"] = round((Z975 + Z80) * math.sqrt(var_gamma), 6)
                row["d1_half_width_95"] = round(Z975 * mult * sd_d[1] / math.sqrt(n), 6)
            out["rows"].append(row)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", type=Path, required=True, help="CLOSE.json (local closure) whose effects feed the dispersion")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = json.loads(args.close.read_text())
    effects = (doc.get("local") or doc).get("effects") or {}
    out = design(effects)
    out["source"] = str(args.close)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    rows = [r for r in out["rows"] if r["sd_multiplier"] == 1.0]
    print(json.dumps({"inputs": out["inputs"], "x1": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
