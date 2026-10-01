"""Generalized FX builder: derived features, elapsed-second labels over weekend gaps, purge by time."""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import fin_forecast_dataset as fd  # noqa: E402


def make_fx(path, days=30, seed=0):
    """Hourly OHLC with weekends removed (Fri 22:00 -> Mon 00:00 gap); one flat bar (HIGH == LOW)."""
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(2020, 1, 6, tzinfo=dt.timezone.utc)  # a Monday
    rows, price = [], 1.1
    t = t0
    while len(rows) < days * 24:
        if t.weekday() < 5:
            o = price
            c = o * math.exp(rng.normal(0, 0.001))
            hi, lo = max(o, c) * (1 + abs(rng.normal(0, 0.0005))), min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
            if len(rows) == 100:
                hi = lo = o = c
            rows.append((t, o, lo, hi, c))
            price = c
        t += dt.timedelta(hours=1)
    lines = ["DATE_TIME,OPEN,LOW,HIGH,CLOSE"] + [f"{r[0].strftime('%Y-%m-%d %H:%M:%S')},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}"
                                               for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")
    return rows


def test_elapsed_second_labels_derived_features_and_purge(tmp_path):
    rows = make_fx(tmp_path / "fx.csv")
    n = len(rows)
    split = {"declared_by": "t", "train_rows": [0, 400], "validation_rows": [400, 600], "test_rows": [600, n]}
    m = fd.build(tmp_path / "fx.csv", tmp_path / "npz", features=["log_high_low", "close_location", "log_close_open"],
                 window=24, horizons=[1, 2, 3], sample_hours=1.0, split=split, purge_seconds=6 * 3600,
                 expected_sha=None, dataset_id="fx:test")
    assert m["feature_names"] == ["log_high_low", "close_location", "log_close_open", "log_return_1"]
    assert m["target_support_channel_appended"] is True
    va = m["splits"]["validation"]
    assert va["dropped_missing_label"] > 0  # Friday-evening origins have no t+1h bar
    assert va["irregular_windows"] > 0 and va["irregular_excluded"] is False
    with np.load(tmp_path / "npz" / "validation.npz") as z:
        x, y, ts = z["windows"], z["targets"], z["timestamps"]
        assert z["feature_names"].tolist()[-1] == "log_return_1" and z["target_names"].tolist() == ["log_return_1"]
    sup = np.load(tmp_path / "npz" / "label_support.npz")
    origins, labels = sup["validation_origins"], sup["validation_label_rows"]
    times = np.array([int(r[0].timestamp()) for r in rows])
    for k, h in enumerate([1, 2, 3]):
        assert np.all(times[labels[:, k]] == ts + h * 3600)  # labels by elapsed seconds
    assert np.all(sup["validation_n_rows"] == labels - origins[:, None])
    mu, sigma = m["target"]["mu"], m["target"]["sigma"]
    close = np.array([r[4] for r in rows])
    for i in range(5):
        o, lab = origins[i], labels[i, 2]
        expect = (math.log(close[lab] / close[o]) - (lab - o) * mu) / sigma
        assert y[i, 2, 0] == pytest.approx(expect, abs=1e-4)
    # purge by time at the train boundary: last train label time + 6h < first validation input time
    tr = m["splits"]["train"]
    first_val_input = times[origins.min() - 23]
    assert times[tr["label_rows"][1]] + 6 * 3600 < first_val_input
    # the flat bar (HIGH == LOW) gives a NaN close_location: every window touching row 100 is dropped
    assert tr["dropped_nonfinite_support"] >= 1
    split_doc = json.loads((tmp_path / "npz" / "SPLIT.json").read_text())
    assert split_doc["test"]["status"] == "PROTECTED_NEVER_READ"


def test_regular_series_matches_row_offsets_and_excludes_irregular(tmp_path):
    rows = make_fx(tmp_path / "fx.csv", days=20)
    n = len(rows)
    split = {"declared_by": "t", "train_rows": [0, 250], "validation_rows": [250, 400], "test_rows": [400, n]}
    m = fd.build(tmp_path / "fx.csv", tmp_path / "npz", features=["log_close_open"], window=24, horizons=[1, 6],
                 sample_hours=1.0, split=split, purge_seconds=0, expected_sha=None, dataset_id="fx:test",
                 exclude_irregular_windows=True)
    va = m["splits"]["validation"]
    assert va["irregular_excluded"] is True and va["irregular_windows"] > 0
    sup = np.load(tmp_path / "npz" / "label_support.npz")
    assert np.all(sup["validation_n_rows"] == np.array([1, 6]))  # kept windows are regular: n_rows == h


def test_refuses_unknown_feature(tmp_path):
    make_fx(tmp_path / "fx.csv", days=10)
    split = {"declared_by": "t", "train_rows": [0, 100], "validation_rows": [100, 150], "test_rows": [150, 168]}
    with pytest.raises(ValueError, match="neither a view column"):
        fd.build(tmp_path / "fx.csv", tmp_path / "x", features=["rsi_14"], window=24, horizons=[1], sample_hours=1.0,
                 split=split, purge_seconds=0, expected_sha=None, dataset_id="fx:test")
