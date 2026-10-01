"""Lane H tests for tools/h_kalman_eth_run.py: the panel loader reproduces the lane F2 NPZ windows exactly."""
from __future__ import annotations

import csv
import importlib.util
import json
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


run = _load("h_kalman_eth_run")
eth = run.eth


def tiny_view(path, rows=420, seed=0):
    rng = np.random.RandomState(seed)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(rows)))
    lr = np.zeros(rows)
    lr[1:] = np.log(close[1:] / close[:-1])
    sma = np.convolve(close, np.ones(5) / 5, mode="same")
    t0 = 1_600_000_000 - 1_600_000_000 % 14400
    import datetime as dt
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DATE_TIME", "CLOSE", "log_return_1", "sma_5", "noise"])
        for i in range(rows):
            d = dt.datetime.fromtimestamp(t0 + 14400 * i, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([d, f"{close[i]:.10f}", f"{lr[i]:.12f}", f"{sma[i]:.10f}", f"{rng.standard_normal():.10f}"])


@pytest.fixture()
def built(tmp_path):
    view = tmp_path / "view.csv"
    tiny_view(view)
    split = {"declared_by": "test", "train_rows": [0, 300], "validation_rows": [300, 400], "test_rows": [400, 420]}
    feats = ["log_return_1", "sma_5", "noise"]
    out = tmp_path / "data"
    eth.build(str(view), out, features=feats, window=6, horizons=[1, 2, 3], split=split, expected_sha=None,
              feature_manifest=None)
    return view, out, split, feats


def test_loader_reproduces_npz_windows_targets_and_never_holds_test_rows(built):
    view, out, split, feats = built
    d = run.load_data(str(view), feats, str(out), window=6, horizons=[1, 2, 3], split=split, expected_sha=None)
    assert d["Z"].shape == (400, 3) and d["train_rows"] == [0, 300] and d["val_rows"] == [300, 400]
    assert d["names"] == feats and d["window"] == 6 and d["horizons"] == [1, 2, 3]
    with np.load(out / "validation.npz") as z:
        want = z["windows"]
    got = np.stack([d["Z"][o - 5:o + 1] for o in d["origins"]["validation"]]).astype(np.float32)
    assert np.array_equal(got, want)
    assert d["origins"]["train"].max() + 3 < d["origins"]["validation"].min() - 5          # purge preserved
    assert d["split_sha256"] == run.eth.sha_file(out / "SPLIT.json")
    # z is the TRAIN-standardised panel: train columns have mean 0 and std 1
    assert np.allclose(d["Z"][:300].mean(axis=0), 0, atol=1e-9) and np.allclose(d["Z"][:300].std(axis=0), 1, atol=1e-9)


def test_loader_refuses_a_digest_or_tiling_mismatch(built, tmp_path):
    view, out, split, feats = built
    with pytest.raises(ValueError):
        run.load_data(str(view), feats, str(out), window=6, horizons=[1, 2, 3], split=split, expected_sha="0" * 64)


def test_candidate_groups_exclude_binary_flags_and_unknown_features():
    cands = {"datasets": [{"dataset": "x", "manifest_canonical": "m", "features": [
        {"feature": "a", "form": "LOCAL_LEVEL"}, {"feature": "b", "form": "LEVEL_PLUS_SLOPE"},
        {"feature": "ema_cross_10_50", "form": "LEVEL_PLUS_SLOPE"}, {"feature": "c", "form": "STATIONARY_NOT_A_LEVEL_STATE"},
        {"feature": "zz", "form": "LOCAL_LEVEL"}]}]}
    g = run.groups_from_candidates(cands, 0, ["a", "b", "c", "ema_cross_10_50"], exclude_prefixes=("ema_cross",))
    assert g == {"local_level": ["a"], "local_linear_trend": ["b"]}


def test_heartbeat_file_is_written_and_advances(tmp_path):
    hb = run.Heartbeat(tmp_path, "unit", interval=0.05)
    hb.start()
    hb.stage("s1")
    import time
    time.sleep(0.3)
    first = json.loads((tmp_path / "heartbeat.json").read_text())
    time.sleep(0.3)
    second = json.loads((tmp_path / "heartbeat.json").read_text())
    hb.stop()
    assert first["stage"] == "s1" and second["tick"] > first["tick"] and "max_rss_kib" in second
