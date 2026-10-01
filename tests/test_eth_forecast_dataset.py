"""Tests for the lane F2 ETH 4h NPZ builder (synthetic view, no TensorFlow)."""
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eth_forecast_dataset as ds  # noqa: E402

FEATURES = ["return_1", "log_return_1", "rsi_14", "flag"]


def make_view(path, rows=400, gap_at=None, nan_test_from=None, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000
    times = [t0 + 4 * 3600 * i for i in range(rows)]
    if gap_at is not None:  # one 8-hour step: every window spanning it is irregular
        times = times[:gap_at] + [t + 4 * 3600 for t in times[gap_at:]]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    lr = np.concatenate([[0.0], np.log(close[1:] / close[:-1])])
    lines = ["DATE_TIME,CLOSE," + ",".join(FEATURES)]
    import datetime as dt
    for i in range(rows):
        stamp = dt.datetime.fromtimestamp(times[i], dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        feats = [math.expm1(lr[i]), lr[i], rng.uniform(20, 80), float(i % 2)]
        if nan_test_from is not None and i >= nan_test_from:
            feats = [float("nan")] * 4
        lines.append(f"{stamp},{close[i]:.8f}," + ",".join(f"{v:.10f}" for v in feats))
    Path(path).write_text("\n".join(lines) + "\n")
    return np.array(times), close, lr


SPLIT = {"declared_by": "test", "train_rows": [0, 300], "validation_rows": [300, 360], "test_rows": [360, 400]}


def test_split_ranges_purge_and_target_definition(tmp_path):
    view = tmp_path / "view.csv"
    times, close, lr = make_view(view)
    out = tmp_path / "npz"
    m = ds.build(view, out, features=FEATURES, window=24, horizons=[1, 2, 3, 4, 5, 6], split=SPLIT, expected_sha=None)
    tr, va = m["splits"]["train"], m["splits"]["validation"]
    # validation origins: inside [300, 360), targets never touch row 360
    assert va["origin_rows"] == [300, 353]
    assert va["purged_origins"] == [354, 355, 356, 357, 358, 359]
    # first validation input row = 300 - 23 = 277; train origin + 6 < 277 -> origin <= 270
    assert tr["origin_rows"] == [23, 270]
    assert tr["purged_origins"] == list(range(271, 300))
    assert tr["purge_components"] == {"h_max_purge": [271, 276], "validation_input_support": [277, 299]}
    assert tr["target_rows"][1] < va["input_rows"][0]
    with np.load(out / "validation.npz") as z:
        x, y, ts, ids = z["windows"], z["targets"], z["timestamps"], z["row_ids"]
        assert x.shape == (54, 24, 4) and y.shape == (54, 6, 1)
        assert z["target_names"].tolist() == ["log_return_1"]
        assert float(z["scaler_scale"][0]) == pytest.approx(m["scaler"]["target_sigma"])
        assert np.array_equal(z["target_timestamps"][:, 0], ts + 4 * 3600)
    mu, sigma = m["target"]["mu"], m["target"]["sigma"]
    assert mu == pytest.approx(lr[:300].mean()) and sigma == pytest.approx(lr[:300].std())
    # Y_h at origin o = (log(C[o+h]/C[o]) - h*mu)/sigma, and the window's last step is z(lr[o])
    o = 300
    for k, h in enumerate(range(1, 7)):
        expect = (math.log(close[o + h] / close[o]) - h * mu) / sigma
        assert y[0, k, 0] == pytest.approx(expect, abs=1e-4)
    assert x[0, -1, 1] == pytest.approx((lr[o] - mu) / sigma, abs=1e-5)
    assert ids[0] == f"eth4h:row300:{times[300]}"
    split = json.loads((out / "SPLIT.json").read_text())
    assert split["test"] == {"rows": [360, 400], "status": "PROTECTED_NEVER_READ"}
    assert split["splits"]["validation"]["row_ids_sha256"] == va["row_ids_sha256"]
    assert m["split_sha256"] == ds.sha_file(out / "SPLIT.json")


def test_test_rows_are_never_read_for_statistics(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view, nan_test_from=360)
    m = ds.build(view, tmp_path / "npz", features=FEATURES, window=24, horizons=[1, 6], split=SPLIT, expected_sha=None)
    assert m["test_rows_read_for_statistics"] is False
    with np.load(tmp_path / "npz" / "train.npz") as z:
        assert np.all(np.isfinite(z["windows"])) and np.all(np.isfinite(z["targets"]))


def test_irregular_step_windows_are_excluded_and_counted(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view, gap_at=320)
    m = ds.build(view, tmp_path / "npz", features=FEATURES, window=24, horizons=[1, 2, 3, 4, 5, 6], split=SPLIT,
                 expected_sha=None)
    va = m["splits"]["validation"]
    # a window with input rows o-23..o and targets to o+6 spans the step at 319->320 iff o-23 <= 319 < o+6
    assert va["gap_excluded_windows"] == len([o for o in range(300, 354) if o - 23 <= 319 < o + 6])
    assert va["windows"] == 54 - va["gap_excluded_windows"]


def test_refuses_wrong_digest_and_bad_log_return(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view)
    with pytest.raises(ValueError, match="digest mismatch"):
        ds.build(view, tmp_path / "a", features=FEATURES, window=24, horizons=[1], split=SPLIT, expected_sha="0" * 64)
    text = view.read_text().splitlines()
    parts = text[5].split(","); parts[3] = "0.5"; text[5] = ",".join(parts)
    view.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="log_return_1"):
        ds.build(view, tmp_path / "b", features=FEATURES, window=24, horizons=[1], split=SPLIT, expected_sha=None)


def test_split_must_tile_the_view(tmp_path):
    view = tmp_path / "view.csv"
    make_view(view)
    bad = {**SPLIT, "test_rows": [360, 399]}
    with pytest.raises(ValueError, match="tile"):
        ds.build(view, tmp_path / "c", features=FEATURES, window=24, horizons=[1], split=bad, expected_sha=None)
