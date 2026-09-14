#!/usr/bin/env python3
"""Flow v3 adoption, P0.2: derive the availability contract of the toy resources from bytes.

The resources the governed micro-run consumes are
`phase_1/normalized_d{4,5,6}.csv` of the `predictor_examples` lake
(predictor `examples/data_downsampled`). Nothing here is taken from a column's
name. Every claim is a comparison of physical values across the producer chain:

  preprocessor examples/data/phase_3b.csv (hourly, typical_price)
    -> preprocessor examples/data/phase_3b_downsampled.csv (4h grid)
    -> preprocessor examples/data_downsampled/phase_1b/base_d*.csv (plugin_default)
    == predictor examples/data_downsampled/phase_1/base_d*.csv (byte-identical)
    -> predictor examples/data_downsampled/phase_1/normalized_d*.csv (stripped to
       DATE_TIME + typical_price, commit f1bb65d)

The decisive test is which window of hourly values reproduces the 4h value at
DATE_TIME = t exactly: (t-4h, t] (right label: information complete by t + 1h
at the latest under either hourly-label hypothesis), [t, t+4h) (left label:
complete only at t + 4h), or the single hourly row at t (decimation).

Run from any cwd; sibling checkouts are located relative to this file.
Output: JSON on stdout. Read-only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PREDICTOR = HERE.parents[5]  # docs/audits/evidence/repro_runs/flow_v3_tools/<file>
GITHUB = PREDICTOR.parent
PREPROCESSOR = GITHUB / "preprocessor"
FEATURE_ENG = GITHUB / "feature-eng"

TOY = ["phase_1/normalized_d4.csv", "phase_1/normalized_d5.csv", "phase_1/normalized_d6.csv"]
EXACT = 1e-12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_bytes() -> dict:
    out = {}
    for rel in TOY:
        path = PREDICTOR / "examples/data_downsampled" / rel
        df = pd.read_csv(path)
        ts = pd.to_datetime(df["DATE_TIME"], format="ISO8601")
        gaps = collections.Counter(ts.diff().dropna().dt.total_seconds().astype(int))
        big = ts.index[ts.diff().dt.total_seconds() > 14400]
        out[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "t_min": str(ts.min()),
            "t_max": str(ts.max()),
            "monotonic_increasing": bool(ts.is_monotonic_increasing),
            "duplicate_timestamps": int(ts.duplicated().sum()),
            "hours_of_day": sorted(set(ts.dt.hour.tolist())),
            "weekdays": sorted(collections.Counter(ts.dt.day_name()).items()),
            "spacing_seconds_top3": gaps.most_common(3),
            "last_bar_before_gap_top3": collections.Counter(
                (ts[i - 1].day_name(), int(ts[i - 1].hour)) for i in big).most_common(3),
            "first_bar_after_gap_top3": collections.Counter(
                (ts[i].day_name(), int(ts[i].hour)) for i in big).most_common(3),
            "last_day_bar_hour_max": int(ts.dt.hour.max()),
        }
    return out


def lineage() -> dict:
    out = {"base_identity": {}, "normalized_vs_preprocessor": {}}
    for n in (4, 5, 6):
        a = PREDICTOR / f"examples/data_downsampled/phase_1/base_d{n}.csv"
        b = PREPROCESSOR / f"examples/data_downsampled/phase_1b/base_d{n}.csv"
        out["base_identity"][f"base_d{n}"] = {
            "predictor_sha256": sha256_file(a), "preprocessor_sha256": sha256_file(b),
            "identical": sha256_file(a) == sha256_file(b),
        }
        na = PREDICTOR / f"examples/data_downsampled/phase_1/normalized_d{n}.csv"
        nb = PREPROCESSOR / f"examples/data_downsampled/phase_1b/normalized_d{n}.csv"
        out["normalized_vs_preprocessor"][f"normalized_d{n}"] = {
            "predictor_sha256": sha256_file(na), "preprocessor_sha256": sha256_file(nb),
            "identical": sha256_file(na) == sha256_file(nb),
        }
    base4 = pd.read_csv(PREDICTOR / "examples/data_downsampled/phase_1/base_d4.csv")
    tb4 = pd.to_datetime(base4["DATE_TIME"], format="ISO8601")
    norm4 = pd.read_csv(PREDICTOR / "examples/data_downsampled/phase_1/normalized_d4.csv")
    tn4 = pd.to_datetime(norm4["DATE_TIME"], format="ISO8601")
    out["normalized_d4_timestamps_equal_base_d4"] = bool(len(tb4) == len(tn4) and (tb4.values == tn4.values).all())

    ds = pd.read_csv(PREPROCESSOR / "examples/data/phase_3b_downsampled.csv", usecols=["DATE_TIME", "typical_price"])
    tds = pd.to_datetime(ds["DATE_TIME"], format="ISO8601")
    m4h = ds.set_index(tds)["typical_price"]
    common = sorted(set(tb4) & set(tds))
    out["phase_3b_downsampled"] = {
        "sha256": sha256_file(PREPROCESSOR / "examples/data/phase_3b_downsampled.csv"),
        "rows": int(len(ds)), "t_min": str(tds.min()), "t_max": str(tds.max()),
        "hours_of_day": sorted(set(tds.dt.hour.tolist())),
        "base_d4_timestamps_present": f"{len(common)}/{len(tb4)}",
        "base_d4_typical_price_exact_equal": int((m4h.loc[common].values == base4.set_index(tb4).loc[common, "typical_price"].values).sum()),
    }

    hourly = pd.read_csv(PREPROCESSOR / "examples/data/phase_3b.csv", usecols=["DATE_TIME", "typical_price"])
    th = pd.to_datetime(hourly["DATE_TIME"], format="ISO8601")
    hh = hourly.set_index(th)["typical_price"].sort_index()
    hh = hh[~hh.index.duplicated()]
    spacing = collections.Counter(th.diff().dropna().dt.total_seconds().astype(int))
    g = hh.index.to_series().diff().dt.total_seconds()
    big = np.where(g.values > 3600)[0]
    tests = {}
    c = sorted(set(m4h.index) & set(hh.index))
    tests["decimation_same_hour_row"] = {"common": len(c), "exact_equal": int((np.abs(m4h.loc[c].values - hh.loc[c].values) < EXACT).sum())}
    for label in ("left", "right"):
        agg = hh.resample("4h", label=label, closed=label).mean().dropna()
        cc = sorted(set(m4h.index) & set(agg.index))
        tests[f"mean_of_hourly_label_{label}"] = {
            "window": "[t, t+4h)" if label == "left" else "(t-4h, t]",
            "common": len(cc),
            "exact_equal": int((np.abs(m4h.loc[cc].values - agg.loc[cc].values) < 1e-9).sum()),
        }
    right = hh.resample("4h", label="right", closed="right")
    counts = right.count()
    full = counts.reindex(m4h.index).fillna(0)
    agg_r = right.mean().reindex(m4h.index)
    diff = np.abs(m4h.values - agg_r.values)
    full_mask = (full.values == 4)
    tests["right_label_restricted_to_full_windows"] = {
        "full_4_row_windows": int(full_mask.sum()),
        "exact_equal_in_full_windows": int((diff[full_mask] < 1e-9).sum()),
        "partial_windows": int((~full_mask).sum()),
    }
    out["phase_3b_hourly"] = {
        "sha256": sha256_file(PREPROCESSOR / "examples/data/phase_3b.csv"),
        "rows": int(len(hourly)), "t_min": str(th.min()), "t_max": str(th.max()),
        "spacing_seconds_top3": spacing.most_common(3),
        "hours_of_day": sorted(set(hh.index.hour.tolist())),
        "last_row_before_gap_top3": collections.Counter((hh.index[i - 1].day_name(), int(hh.index[i - 1].hour)) for i in big).most_common(3),
        "first_row_after_gap_top3": collections.Counter((hh.index[i].day_name(), int(hh.index[i].hour)) for i in big).most_common(3),
        "window_tests": tests,
    }

    raw = FEATURE_ENG / "tests/data/EURUSD_ForexTrading_4hrs_05.05.2003_to_16.10.2021.csv"
    if raw.is_file():
        r = pd.read_csv(raw, usecols=["Gmt time"])
        rt = pd.to_datetime(r["Gmt time"], format="%d.%m.%Y %H:%M:%S.%f")
        out["dukascopy_4h_raw_not_in_lineage"] = {
            "sha256": sha256_file(raw), "rows": int(len(r)),
            "hours_of_day": sorted(set(rt.dt.hour.tolist())),
            "base_d4_timestamps_present": f"{len(set(tb4) & set(rt))}/{len(tb4)}",
        }
    return out


def main() -> int:
    result = {
        "schema": "flow_v3_p02_contract_derivation.v1",
        "code_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
        "toy_resources": toy_bytes(),
        "lineage": lineage(),
    }
    json.dump(result, sys.stdout, indent=1, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
