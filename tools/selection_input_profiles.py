#!/usr/bin/env python3
"""TRAIN-only temporal/redundancy diagnostics; never transforms model inputs or fits a policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import pacf

from expert_package_worker import atomic, digest


FIT = ("2020-01-01", "2023-07-01")


def fit_frame(frame):
    available = pd.to_datetime(frame["timestamp"], utc=True) + pd.Timedelta(hours=1)
    return frame[(available >= pd.Timestamp(FIT[0], tz="UTC")) &
                 (available < pd.Timestamp(FIT[1], tz="UTC"))].copy()


def purged_folds(n, k=6, embargo=96):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        val = np.arange(edges[i], edges[i + 1])
        train = np.r_[np.arange(0, max(edges[i] - embargo, 0)),
                      np.arange(min(edges[i + 1] + embargo, n), n)]
        yield i, train, val


def finite_run(x):
    finite = np.isfinite(x)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def temporal_profile(x, max_lag=48):
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    corrs, counts, significant = [], [], []
    z = float(norm.ppf(1 - .05 / (2 * max_lag)))
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        n = len(a)
        rho = float(np.corrcoef(a, b)[0, 1]) if n >= 3 and np.std(a) > 0 and np.std(b) > 0 else None
        corrs.append(rho)
        counts.append(n)
        significant.append(rho is not None and abs(rho) > z / np.sqrt(n))
    start, end = finite_run(x)
    run = x[start:end]
    nlags = min(max_lag, (len(run) - 1) // 2)
    ps, pacf_status = [], "INSUFFICIENT_OR_CONSTANT"
    if nlags >= 1 and np.std(run) > 0:
        try:
            values = pacf(run, nlags=nlags, method="ywmle")
            if not np.isfinite(values).all():
                raise np.linalg.LinAlgError("nonfinite PACF")
            ps = [int(lag) for lag in range(1, len(values)) if abs(values[lag]) > z / np.sqrt(len(run))]
            pacf_status = "COMPUTED"
        except np.linalg.LinAlgError:
            pacf_status = "NUMERICAL_FAILURE"
    return {"n_rows_used": int(len(finite)), "missing_share_fit": float(1 - len(finite) / len(x)) if len(x) else None,
            "constant_fit": bool(len(finite) and np.std(finite) == 0),
            "acf_lag1": corrs[0], "acf_lag24": corrs[23] if max_lag >= 24 else None,
            "acf_first_insignificant_lag": next((i + 1 for i, (r, sig) in enumerate(zip(corrs, significant)) if r is not None and not sig), None),
            "acf_values": corrs, "acf_n_pairs": counts,
            "partial_acf_lags_significant": ps, "pacf_n_rows": len(run), "pacf_status": pacf_status,
            "pacf_run_start_slot": start, "pacf_run_end_slot_exclusive": end}


def summarize_folds(x, folds):
    def stats(indices):
        v = x[indices]
        v = v[np.isfinite(v)]
        return {"n": len(v), "mean": float(np.mean(v)) if len(v) else None,
                "std": float(np.std(v)) if len(v) else None}
    vals, trains = [stats(v) for _, _, v in folds], [stats(t) for _, t, _ in folds]
    means = [v["mean"] for v in vals if v["mean"] is not None]
    finite = x[np.isfinite(x)]
    sd = float(np.std(finite)) if len(finite) else 0.
    return {"fold_mean": [v["mean"] for v in vals], "fold_std": [v["std"] for v in vals],
            "fold_n": [v["n"] for v in vals], "fold_train_stats": trains,
            "fold_stability_ratio": float(np.std(means) / sd) if len(means) and sd > 0 else None}


def run(store, admitted, selection, templates, output):
    store, admitted, selection, templates, output = map(Path, (store, admitted, selection, templates, output))
    man = json.loads((store.parent / "MANIFEST.json").read_text())
    if digest(store) != man["file_sha256"] or digest(admitted) != man["source_sha256"]:
        raise ValueError("PROFILE_INPUT_IDENTITY")
    lists = {n: json.loads((selection / ("FEATURE_LIST_" + n + ".json")).read_text())["features"]
             for n in ("core15", "shortlist_strict")}
    columns = list(dict.fromkeys(c for cols in lists.values() for c in cols))
    df = fit_frame(pd.read_parquet(store, columns=["timestamp"] + columns)).reset_index(drop=True)
    observed = pd.read_parquet(admitted, columns=["timestamp"])
    observed = fit_frame(observed)
    ts = pd.DatetimeIndex(pd.to_datetime(df.timestamp, utc=True))
    obs = pd.DatetimeIndex(pd.to_datetime(observed.timestamp, utc=True))
    expected = obs[(obs >= ts[0]) & (obs <= ts[-1])]
    if ts.has_duplicates or not ts.is_monotonic_increasing or not ts.equals(expected):
        raise ValueError("PROFILE_OBSERVED_GRID_MISMATCH")
    folds = list(purged_folds(len(df)))
    profiles = {c: {**temporal_profile(df[c].to_numpy(float)),
                    **summarize_folds(df[c].to_numpy(float), folds)} for c in columns}
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name, cols in lists.items():
        template = pd.read_csv(templates / ("INPUT_JUSTIFICATION_" + name + ".csv"))
        if template.column.tolist() != cols:
            raise ValueError("PROFILE_TEMPLATE_ORDER")
        values = df[cols].replace([np.inf, -np.inf], np.nan)
        corr = values.corr(method="spearman", min_periods=100)
        present = np.isfinite(values.to_numpy(float)).astype(np.int64)
        paired_n = present.T @ present
        families = dict(zip(template.column, template.family.fillna("")))
        added = []
        for i, c in enumerate(cols):
            row = {"column": c, **profiles[c]}
            rho = corr.loc[c].drop(c).abs().dropna()
            partner = rho.idxmax() if len(rho) else None
            same = rho[[families.get(k) == families[c] for k in rho.index]] if len(rho) else rho
            row.update(redundancy_partner=partner,
                       redundancy_max_abs_spearman=float(rho[partner]) if partner else None,
                       redundancy_partner_n=int(paired_n[i, cols.index(partner)]) if partner else 0,
                       redundancy_within_family_max=float(same.max()) if len(same) else None)
            added.append(row)
        result = template.merge(pd.DataFrame(added), on="column", how="left", validate="one_to_one", sort=False)
        for c in result.columns:
            if c in profiles[cols[0]] and any(isinstance(v, (list, dict)) for v in result[c]):
                result[c] = result[c].map(lambda v: json.dumps(v, allow_nan=False))
        target = output / ("PROFILES_" + name + ".csv")
        tmp = target.with_suffix(".csv.tmp")
        result.to_csv(tmp, index=False)
        tmp.replace(target)
        files.append(target)
    metadata = {"schema": "selection_input_profiles.v1", "scope": "diagnostics only; no feature deletion or model-input transform",
                "input_sha256": digest(store), "admitted_sha256": digest(admitted),
                "lists_sha256": {n: digest(selection / ("FEATURE_LIST_" + n + ".json")) for n in lists},
                "fit": list(FIT), "timestamp_semantics": "bar start; FIT membership by available_at = start + 1h",
                "rows": len(df), "max_lag": 48, "lags_unit": "observed bars, no missing-value compression",
                "acf_estimator": "pairwise-finite Pearson at each lag; n_pairs retained",
                "significance": "white-noise normal approximation with Bonferroni alpha=.05 / 48 per feature, not joint calibrated inference across features",
                "pacf": "Yule-Walker MLE, longest contiguous finite run; run and n logged, no fill",
                "folds": 6, "embargo_observed_rows": 96,
                "fold_stability_ratio": "std of six validation-block means / global std, descriptive",
                "missing_grid_rows": 0, "files": {p.name: digest(p) for p in files},
                "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
    atomic(output / "PROFILES_MANIFEST.json", metadata)
    print(json.dumps({"status": "COMPLETE", "rows": len(df), "profiles": {n: len(v) for n, v in lists.items()}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("store", "admitted", "selection", "templates", "output"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    run(a.store, a.admitted, a.selection, a.templates, a.output)
