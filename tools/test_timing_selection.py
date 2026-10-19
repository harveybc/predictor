"""RED acceptance tests for the timing-target selection extension (targets, price pool, driver).

Each test is one acceptance point of docs/TIMING_SELECTION.md.  Capabilities are loaded lazily
through ``_cap`` so a PRE run fails with ``capability absent: <module>``, never a broken import.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _cap(name: str):
    try:
        return importlib.import_module(f"tools.{name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith(name):
            pytest.fail(f"capability absent: tools/{name}.py")
        raise


# ----------------------------------------------------------------------------- synthetic bars
def _bars(n=600, seed=0, start="2020-01-01"):
    """Hourly OHLC with a deliberate saw-tooth so a ZigZag has well-separated pivots."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 1.10 + 0.01 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 2e-4, n).cumsum() * 0.05
    close = base
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1.5e-4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1.5e-4, n))
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=idx.rename("DATE_TIME"))


class _FakeZZ:
    """A stand-in for the pinned zigzag module: pivots at fixed bars, confirmation 3 bars later."""
    PEAK, VALLEY = 1, -1

    class Params:
        def __init__(self, depth=3, deviation_pips=36.3, backstep=2):
            self.depth, self.deviation_pips, self.backstep = depth, deviation_pips, backstep

    def hindsight_pivots(self, high, low, p):
        n = len(high)
        out = []
        kind = self.VALLEY
        for b in range(24, n - 30, 24):
            out.append((b, kind, float(low[b] if kind == self.VALLEY else high[b])))
            kind = -kind
        return out

    def confirmation_bars(self, high, low, pivots, p, max_lag=400):
        return [b + 3 for b, _, _ in pivots]


# ----------------------------------------------------------------------------- targets
def test_t1_pivot_within_k_is_a_future_label_and_purged_at_cuts():
    tp = _cap("timing_pivot_targets")
    bars = _bars(300)
    zz = _FakeZZ()
    piv = zz.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), zz.Params())
    frame = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[pd.Timestamp("2020-01-06", tz="UTC")])
    y = frame["timing_pivot_k2"].to_numpy()
    cut_bar = int(bars.index.searchsorted(pd.Timestamp("2020-01-06", tz="UTC")))
    for b, kind, _ in piv:
        if b == cut_bar:            # its two label bars lie just before the cut and are purged (checked below)
            continue
        # the label lives on the k bars BEFORE the pivot, never on the pivot bar itself
        assert all(y[b - j] == (1 if kind == zz.VALLEY else 2) for j in (1, 2))
        assert y[b] == 0
    # the two bars before each cut carry NaN (their (t, t+k] window crosses the cut)
    cut = pd.Timestamp("2020-01-06", tz="UTC")
    before = (frame.timestamp >= cut - pd.Timedelta(hours=2)) & (frame.timestamp < cut)
    assert before.sum() == 2 and frame.loc[before, "timing_pivot_k2"].isna().all()
    assert not frame.loc[~before, "timing_pivot_k2"].iloc[:-2].isna().any()
    assert set(frame.columns) >= {"row_id", "timestamp", "timing_pivot_k2", "timing_valley_k2", "timing_peak_k2",
                                  "timing_action_confirmed", "timing_expert_leg_pnl_pips"}


def test_t2_action_label_sits_on_the_confirmation_bar_not_the_pivot():
    tp = _cap("timing_pivot_targets")
    bars = _bars(300)
    zz = _FakeZZ()
    piv = zz.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), zz.Params())
    frame = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[])
    a = frame["timing_action_confirmed"].to_numpy()
    for b, kind, _ in piv:
        assert a[b + 3] == (1 if kind == zz.VALLEY else -1)
        assert a[b] == 0


def test_t3_leg_pnl_is_signed_distance_to_the_next_pivot_level():
    tp = _cap("timing_pivot_targets")
    bars = _bars(300)
    zz = _FakeZZ()
    piv = zz.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), zz.Params())
    frame = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[])
    pnl = frame["timing_expert_leg_pnl_pips"].to_numpy()
    b0, k0, _ = piv[0]
    b1, k1, lvl1 = piv[1]
    t = b0 + 5
    d = 1 if k0 == zz.VALLEY else -1       # after a valley the expert is long until the next peak
    assert pnl[t] == pytest.approx(d * (lvl1 - bars.CLOSE.iloc[t]) / 1e-4)
    assert np.isnan(pnl[piv[-1][0] + 1])   # beyond the last pivot there is no next level


def test_target_manifest_entries_declare_the_timing_family_and_heads():
    tp = _cap("timing_pivot_targets")
    entries = tp.manifest_targets(k_bars=2)
    ids = {e["target_id"]: e for e in entries}
    assert ids["timing_pivot_k2"]["head"] == "classification" and ids["timing_pivot_k2"]["horizon_hours"] == 2
    assert ids["timing_expert_leg_pnl_pips"]["head"] == "regression"
    assert all(e["family"] == "timing_pivot" for e in entries)


def test_zigzag_module_loader_pins_the_sha256(tmp_path):
    tp = _cap("timing_pivot_targets")
    mod = tmp_path / "zz.py"
    mod.write_text("PEAK, VALLEY = 1, -1\n")
    import hashlib
    good = hashlib.sha256(mod.read_bytes()).hexdigest()
    assert tp.load_zigzag_module(mod, good).PEAK == 1
    with pytest.raises(tp.TimingError, match="ZIGZAG_MODULE_DIGEST"):
        tp.load_zigzag_module(mod, "0" * 64)


def test_engine_mi_codes_keep_three_classes_distinct():
    """The phase-3 engine codes a 3-class target without collapsing classes (no engine change needed)."""
    ff = _cap("feature_filter_selection")
    y = np.array([0, 1, 2] * 100, dtype=float)
    codes = ff._codes(y, 8)
    assert len(np.unique(codes)) == 3


# ----------------------------------------------------------------------------- price pool
def test_price_pool_is_prefix_invariant_and_declares_availability():
    pp = _cap("timing_price_pool")
    bars = _bars(400)
    full = pp.price_pool(bars)
    prefix = pp.price_pool(bars.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250], prefix, check_exact=True)
    meta = pp.pool_metadata(full)
    assert all(m["clock"] in ("OBSERVED", "KNOWN_IN_ADVANCE") for m in meta)
    assert {"px.logret_1", "px.rv_24", "px.er_12", "px.close_loc_24", "px.atr_ratio_14_120", "cal.hod_sin",
            "cal.session_us"} <= set(full.columns)


def test_availability_gate_refuses_centred_filters_and_future_availability():
    pp = _cap("timing_price_pool")
    with pytest.raises(pp.AvailabilityError, match="FORBIDDEN_TRANSFORM"):
        pp.availability_gate([{"feature_id": "x_stl_trend", "transform": "stl", "availability_time": "timestamp"}])
    with pytest.raises(pp.AvailabilityError, match="FUTURE_AVAILABILITY"):
        pp.availability_gate([{"feature_id": "x", "transform": "identity", "availability_time": "timestamp+1h"}])
    assert pp.availability_gate([{"feature_id": "x", "transform": "identity", "availability_time": "timestamp"}]) == ["x"]


def test_within_day_variance_flags_daily_constant_features():
    pp = _cap("timing_price_pool")
    idx = pd.date_range("2020-01-01", periods=240, freq="h", tz="UTC")
    daily = np.repeat(np.arange(10), 24).astype(float)
    hourly = np.arange(240, dtype=float)
    rep = pp.within_day_variance(pd.DataFrame({"d": daily, "h": hourly}, index=idx))
    assert rep["d"]["constant_within_day"] is True and rep["h"]["constant_within_day"] is False


# ----------------------------------------------------------------------------- driver
def test_fdr_screen_controls_the_null_and_keeps_a_real_signal():
    ts = _cap("timing_selection")
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 3, n)
    X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(20)})
    X["sig"] = (y == 1).astype(float) + rng.normal(0, 0.5, n)
    rep = ts.univariate_screen(X, y, head="classification", q=0.1)
    assert rep["sig"]["pass_fdr"] is True
    assert sum(r["pass_fdr"] for k, r in rep.items() if k != "sig") <= 3


def test_driver_runs_end_to_end_on_synthetic_bars_and_seals_a_manifest(tmp_path, monkeypatch):
    ts = _cap("timing_selection")
    bars = _bars(2000, start="2020-01-01")
    lake = tmp_path / "lake.parquet"
    pd.DataFrame({"datetime": bars.index.tz_convert(None), "open": bars.OPEN, "high": bars.HIGH, "low": bars.LOW,
                  "close": bars.CLOSE}).to_parquet(lake, index=False)
    zz = tmp_path / "zz.py"
    zz.write_text(
        "PEAK, VALLEY = 1, -1\n"
        "class Params:\n    def __init__(self, depth=3, deviation_pips=36.3, backstep=2):\n"
        "        self.depth, self.deviation_pips, self.backstep = depth, deviation_pips, backstep\n"
        "def hindsight_pivots(high, low, p):\n    out=[]; kind=-1\n"
        "    for b in range(24, len(high)-30, 24):\n        out.append((b, kind, float(low[b] if kind==-1 else high[b]))); kind=-kind\n"
        "    return out\n"
        "def confirmation_bars(high, low, pivots, p, max_lag=400):\n    return [b+3 for b,_,_ in pivots]\n")
    import hashlib
    config = {
        "schema": "predictor.timing_selection.v1", "population_id": "SYNTH_TIMING",
        "lake": {"path": str(lake), "sha256": hashlib.sha256(lake.read_bytes()).hexdigest(), "load": ["2020-01-01", "2020-03-15"]},
        "zigzag": {"module": str(zz), "sha256": hashlib.sha256(zz.read_bytes()).hexdigest(),
                   "params": {"depth": 3, "deviation_pips": 36.3, "backstep": 2}, "k_bars": 2},
        "splits": {"train": ["2020-01-01", "2020-02-15"], "inner": ["2020-02-15", "2020-03-01"],
                   "confirm": ["2020-03-01", "2020-03-15"], "folds": ["2020-01-20", "2020-02-05"]},
        "candidates": {"inventory": None, "price_pool": True, "fractal": None},
        "pairwise_params": {"lags_hours": [0, 1, 2, 6], "min_support": 50, "dcor_max_rows": 128},
        "filter": {"seed": 1, "k_grid": [4, 8], "recipe": {"method": "MRMR", "k": 8, "prune_abs_spearman": 0.9}},
        "causal": {"enabled": False},
        "n_shards": 1, "seed": 1,
    }
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "out"
    rep = ts.run(cfg, out)
    assert (out / "SELECTION_MANIFEST.json").exists() and (out / "REPORT.json").exists()
    man = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    assert man["schema"] == "feature_selection.manifest.v1" and man["final_selection"] is False
    assert len(man["selected_features"]) >= 1 and man["targets"][0]["family"] == "timing_pivot"
    assert man["test_read"] is False and rep["phase3"]["methods"]
    # the manifest digest re-derives from the canonical body without the digest field
    from tools.fs_phase23_manifest import digest
    assert digest({k: v for k, v in man.items() if k != "manifest_sha256"}) == man["manifest_sha256"]
