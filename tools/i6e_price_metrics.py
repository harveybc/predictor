"""Reconcile retained returns against original hourly closes, then report prices.

Clock/grouping matches feature-eng tools/eurusd_ps/sources.py: NEW_YORK_LOCAL /
BAR_START, hourly buckets indexed by UTC bar END. No smoothing, interpolation,
or synthetic market bars. Only pre-2025 source observations are decoded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.i6e_compact_run import atomic_json, file_hash


def hourly_close(raw):
    """Convert the measured source clock and aggregate the actual last 5m close."""
    naive = pd.DatetimeIndex(raw["timestamp"]).tz_localize(None)
    starts = naive.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
    valid = ~starts.isna()
    series = pd.Series(raw["close"].to_numpy()[valid], index=starts[valid]).sort_index()
    result = series.groupby(series.index.floor("h")).last()
    result.index += pd.Timedelta(hours=1)
    if result.index.has_duplicates or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("invalid original hourly price support")
    return result


def load_prices(path, start="2012-04-01", end="2025-01-01"):
    """Apply a source filter before decoding; never return an observation in TEST."""
    end_utc = pd.Timestamp(end, tz="UTC")
    if end_utc > pd.Timestamp("2025-01-01", tz="UTC"):
        raise ValueError("TEST price access prohibited")
    raw = pd.read_parquet(path, columns=["timestamp", "close"], filters=[
        ("timestamp", ">=", pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)),
        ("timestamp", "<", end_utc - pd.Timedelta(hours=5))])
    result = hourly_close(raw)
    return result[(result.index >= pd.Timestamp(start, tz="UTC")) & (result.index < end_utc)]


def price_errors(current, future, actual_return, predicted_return):
    """Price MAE/MSE paired to persistence, after independently checking targets."""
    arrays = [np.asarray(a, dtype=float) for a in (current, future, actual_return, predicted_return)]
    current, future, actual_return, prediction = arrays
    if (current.ndim != 1 or not len(current) or any(a.shape != current.shape for a in arrays)
            or not all(np.isfinite(a).all() for a in arrays) or np.any(current <= 0) or np.any(future <= 0)):
        raise ValueError("nonempty finite positive same-row price arrays required")
    if not np.allclose(np.log(future/current), actual_return, rtol=0, atol=1e-12):
        raise ValueError("original price / retained target mismatch")
    predicted_price = current * np.exp(prediction)
    if not np.isfinite(predicted_price).all():
        raise ValueError("nonfinite predicted price")
    error, naive = predicted_price - future, current - future
    mae, nm = float(np.abs(error).mean()), float(np.abs(naive).mean())
    mse, ns = float(np.square(error).mean()), float(np.square(naive).mean())
    return dict(n=len(current), mae=mae, mse=mse, naive_mae=nm, naive_mse=ns,
                skill_mae=None if nm == 0 else 1-mae/nm,
                skill_mse=None if ns == 0 else 1-mse/ns, unit="USD_per_EUR")


def exact_price_values(series, timestamps):
    """Exact clock lookup; absent endpoints are errors, never as-of replacements."""
    values = series.reindex(pd.to_datetime(timestamps, unit="s", utc=True)).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("missing exact original price endpoint")
    return values


def enrich(root, source, output):
    """Verify existing neural terminals and add a separate, non-mutating metric file."""
    from tools.i6e_campaign import close, validate
    closure = close(root)
    prices = load_prices(source)
    by_arm = {}
    weekly = []
    for samples in (3, 5, 9):
        pieces = []
        for week in range(52):
            cell = validate(root, week, samples)
            path = root / "cells" / f"week_{week:03d}_samples_{samples}" / "PREDICTIONS.npz"
            with np.load(path, allow_pickle=False) as data:
                ts, actual, pred = data["timestamps"], data["actual"], data["prediction"]
            current = exact_price_values(prices, ts)
            future = exact_price_values(prices, ts + 72*3600)
            metrics = price_errors(current, future, actual, pred)
            weekly.append(dict(week=week, samples=samples, cell_sha256=cell["cell_sha256"], metrics=metrics))
            pieces.append((current, future, actual, pred))
        by_arm[str(samples)] = price_errors(*(np.concatenate([p[i] for p in pieces]) for i in range(4)))
    report = dict(status="COMPLETE", test_read=False, original_price_sha256=file_hash(source),
                  source_clock="NEW_YORK_LOCAL/BAR_START", exact_target_reconciled=True,
                  metrics_by_samples=by_arm, weekly=weekly,
                  neural_closure_sha256=file_hash(root / "NEURAL_CLOSURE.json"))
    atomic_json(output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "weekly"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enrich(args.campaign, args.source, args.output)


if __name__ == "__main__":
    main()
