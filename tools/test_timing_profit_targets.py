"""timing_profit family: order resolution, SL-first, forced flat, horizon, censoring, purge, finality, ledger."""
import numpy as np
import pandas as pd
import pytest

from tools import timing_profit_targets as tpf


class _ZZ:
    """Only what the family needs from the expert module: the causal sigma (constant here, strictly causal)."""

    def __init__(self, s=0.002):
        self.s = s

    def causal_sigma(self, bars):
        out = np.full(len(bars), self.s)
        out[:5] = np.nan                                           # warm-up: no sigma -> not labellable
        return out


def _bars(close, index=None):
    close = np.asarray(close, float)
    n = len(close)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1e-5
    low = np.minimum(open_, close) - 1e-5
    idx = index if index is not None else pd.date_range("2020-01-06", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=pd.DatetimeIndex(idx).rename("DATE_TIME"))


def test_first_touch_decides_sl_first_in_one_bar_and_both_directions():
    c = np.full(60, 1.0)
    c[20:25] = [1.001, 1.002, 1.003, 1.0045, 1.006]               # a long move: +0.006 > 2 sigma (0.004)
    c[40:45] = [0.999, 0.998, 0.997, 0.9955, 0.994]               # a short move
    b = _bars(c)
    sp = {"horizon_bars": 30, "m_tp": 2.0, "m_sl": 2.0, "cost_sigma": 0.0}
    a = tpf.arrays_from_bars(b, _ZZ(), None, k_bars=30, spec=sp)
    assert a["t1"][19] == 1.0 and a["long_wins"][19] and not a["short_wins"][19]      # order filled at open(20)
    assert a["t1"][39] == 2.0
    assert a["s1"][19] == 23 and a["t2"][19] == 1.0 and a["t3"][19] == pytest.approx(2.0)   # both resolve at bar 23
    # one bar touching both levels: SL first (conservative)
    b2 = _bars(np.full(30, 1.0))
    b2.loc[b2.index[11], ["HIGH", "LOW"]] = [1.01, 0.99]
    a2 = tpf.arrays_from_bars(b2, _ZZ(), None, k_bars=10, spec=sp)
    assert a2["t1"][10] == 0.0 and not a2["long_wins"][10] and not a2["short_wins"][10] and a2["s1"][10] == 11
    assert a2["t3"][10] == pytest.approx(-2.0)


def test_no_sigma_and_unresolved_rows_are_censored_never_zero_and_purged_at_cuts():
    b = _bars(np.full(40, 1.0) + np.r_[np.zeros(30), np.linspace(0, 0.0001, 10)])
    sp = {"horizon_bars": 8, "m_tp": 2.0, "m_sl": 2.0, "cost_sigma": 0.0}
    a = tpf.arrays_from_bars(b, _ZZ(), None, k_bars=8, spec=sp)
    assert (a["s1"][:4] == -1).all()                                  # no causal sigma at t+1 -> censored
    assert a["s1"][35] >= 40                                          # horizon beyond the data -> censored
    f, led = tpf.build_targets(b, _ZZ(), None, k_bars=8, cuts=["2020-01-07T00:00:00Z"], arrays=a, with_ledger=True)
    col = tpf.primary(8)
    assert np.isnan(f[col].iloc[0]) and led.t1_status.iloc[0] == "CENSORED_UNCONFIRMED"
    assert np.isnan(f[col].iloc[35]) and led.t1_status.iloc[35] == "CENSORED_END_OF_DATA"
    cut = int(b.index.searchsorted(pd.Timestamp("2020-01-07", tz="UTC")))
    assert led.t1_status.iloc[cut - 2] == "PURGED_SUPPORT_CROSSES_CUT" and np.isnan(f[col].iloc[cut - 2])
    assert led.t1_status.iloc[cut - 12] == "LABELLED" and f[col].iloc[cut - 12] == 0.0   # flat path: timeout, pnl 0 -> no win
    st = tpf.status_table(led, {"ALL": ["2020-01-01", "2020-02-01"]})
    assert sum(st["ALL"]["t1"].values()) == 40


def test_forced_flat_closes_the_order_at_the_preclose_bar():
    idx = pd.date_range("2020-01-06", periods=48, freq="h", tz="UTC")
    c = np.full(48, 1.0)
    c[30:] = 1.0 + np.linspace(0.0005, 0.01, 18)                     # the long would reach TP at ~bar 33
    b = _bars(c, idx)
    closures = [["2020-01-07T08:00:00Z", "2020-01-08T00:00:00Z"]]      # session close 08:00 on the 7th = bar 32
    sp = {"horizon_bars": 40, "m_tp": 2.0, "m_sl": 2.0, "cost_sigma": 0.0, "preclose_hours": 12.0}
    flat = tpf.flat_bar_positions(b.index, closures, 12.0)
    assert flat[10] == 19                                             # last bar complete at or before 20:00 on the 6th
    a = tpf.arrays_from_bars(b, _ZZ(), closures, k_bars=40, spec=sp)
    assert a["s1"][10] == 19 and a["t1"][10] == 0.0                   # closed flat at the forced flat, before the move
    assert a["s1"][19] == -1 and a["s1"][20] == -1                    # fill would land after the forced flat: censored


def test_finality_holds_under_truncation_and_future_perturbation():
    rng = np.random.default_rng(3)
    c = 1.1 + rng.normal(0, 3e-4, 500).cumsum()
    b = _bars(c)
    sp = {"horizon_bars": 24, "m_tp": 1.5, "m_sl": 1.5, "cost_sigma": 0.1}
    a = tpf.arrays_from_bars(b, _ZZ(0.0015), None, k_bars=24, spec=sp)
    fin = tpf.finality_check(b, _ZZ(0.0015), None, k_bars=24, cut_positions=[120, 300, 450], full=a)
    assert fin["all_final_rows_unchanged"]
    assert all(r[m]["t1"]["rows_declared_final"] > 50 for r in fin["cuts"] for m in ("truncation", "future_perturbation"))


def test_manifest_identity_and_ledger_mapping():
    assert [t["target_id"] for t in tpf.manifest_targets(72)][:3] == ["timing_profit_h72", "timing_profit_long_h72", "timing_profit_short_h72"]
    i1 = tpf.target_identity(k_bars=72, zigzag_sha256="a", lake_sha256="b", spec={"m_tp": 2.0})
    i2 = tpf.target_identity(k_bars=72, zigzag_sha256="a", lake_sha256="b", spec={"m_tp": 2.5})
    assert i1 != i2 and i1 == tpf.target_identity(k_bars=72, zigzag_sha256="a", lake_sha256="b", spec={"m_tp": 2.0, "cost_sigma": 0.1})
    assert {k.format(k=72): v for k, v in tpf.TARGET_LEDGER.items()}["timing_best_order_pnl_sigma_h72"] == "t3"
