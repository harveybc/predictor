"""Lane H: EURUSD 1h loader (OHLC, irregular weekend gaps, targets located by elapsed seconds, 144 h purge)."""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS.parent))


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eu = _load("h_kalman_eurusd_run")


def tiny_csv(path, rows=1500, seed=0):
    rng = np.random.RandomState(seed)
    t = dt.datetime(2005, 5, 2, 0, 0, 0, tzinfo=dt.timezone.utc)
    close, out = 1.2, []
    while len(out) < rows:
        if t.weekday() == 5 or (t.weekday() == 4 and t.hour >= 22):   # weekend gap
            t += dt.timedelta(hours=1)
            continue
        close *= float(np.exp(0.0005 * rng.standard_normal()))
        o = close * (1 + 0.0001 * rng.standard_normal())
        out.append((t.strftime("%Y-%m-%d %H:%M:%S"), o, min(o, close) * 0.9998, max(o, close) * 1.0002, close))
        t += dt.timedelta(hours=1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DATE_TIME", "OPEN", "LOW", "HIGH", "CLOSE"])
        for r in out:
            w.writerow([r[0]] + [f"{v:.6f}" for v in r[1:]])


@pytest.fixture()
def panel(tmp_path):
    p = tmp_path / "eu.csv"
    tiny_csv(p)
    split = {"train_rows": [0, 1000], "validation_rows": [1000, 1300], "test_rows": [1300, 1500]}
    d = eu.load_eurusd(str(p), split=split, expected_sha=None, window=24, horizons=(1, 2, 3, 4, 5, 6), purge=24)
    return p, split, d


def test_panel_shapes_and_split_and_test_rows_never_loaded(panel):
    p, split, d = panel
    assert d["Z"].shape == (1300, 4) and d["names"] == ["OPEN", "LOW", "HIGH", "CLOSE"]
    assert d["train_rows"] == [0, 1000] and d["val_rows"] == [1000, 1300]
    assert np.allclose(d["Z"][:1000].mean(axis=0), 0, atol=1e-9) and np.allclose(d["Z"][:1000].std(axis=0), 1, atol=1e-9)
    assert d["origins"]["train"].max() < d["origins"]["validation"].min() - 24 + 1 - 24 + 24      # purge preserved
    assert d["origins"]["validation"].max() + 6 < 1300


def test_targets_are_located_by_elapsed_seconds_and_windows_are_regular(panel):
    p, split, d = panel
    raw = np.genfromtxt(p, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4))
    times = np.array([int(dt.datetime.strptime(l.split(",")[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
                      for l in open(p).read().splitlines()[1:]])
    for s in ("train", "validation"):
        for o in d["origins"][s][::37]:
            assert np.all(np.diff(times[o - 23:o + 7]) == 3600)                  # regular window and target support
    o = d["origins"]["validation"][5]
    for k, h in enumerate(d["horizons"]):
        want = (np.log(raw[o + h, 3] / raw[o, 3]) - h * d["mu"]) / d["sigma"]
        assert abs(d["Y"]["validation"][5, k] - want) < 1e-9
    # irregular windows were excluded and counted
    assert d["gap_excluded"]["train"] > 0


def test_loader_refuses_digest_mismatch(panel):
    p, split, d = panel
    with pytest.raises(ValueError):
        eu.load_eurusd(str(p), split=split, expected_sha="0" * 64, window=24, horizons=(1, 2, 3, 4, 5, 6), purge=24)
