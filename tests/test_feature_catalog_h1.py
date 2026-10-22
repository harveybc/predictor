"""Tests for tools/feature_catalog_h1.py on small synthetic stores (layer-1-like, layer-2-like, optional r4).

Run only this file: the repository's legacy suite is stale.
    python -m pytest tests/test_feature_catalog_h1.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import feature_catalog_h1 as FC  # noqa: E402

N = 3000
TS = pd.date_range("2022-10-01", periods=N, freq="h", tz="UTC")       # 125 days: spans TRAIN (2022) and 2023


def _diag(cols, decisions):
    return pd.DataFrame([{"feature": c, "group": "g", "n": 100, "min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1, "skew": 0.1, "kurtosis": 1.0,
                          "zero_share": 0.0, "adf_p": 0.01, "kpss_p": 0.1, "decision": decisions.get(c, "NONE"), "reason": "test"} for c in cols])


def make_layer1(d: Path, with_test_rows=False):
    rng = np.random.default_rng(0)
    ts = TS if not with_test_rows else pd.date_range("2024-12-20", periods=N, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "CLOSE": 1.1 + rng.normal(0, 0.01, N).cumsum() * 0.01, "RSI": rng.uniform(0, 100, N),
                       "fx.gbpusd.ret_1": rng.normal(0, 1e-3, N), "fac.fx_f1.ret_1": rng.normal(0, 1, N), "ssa.c2_pips": rng.normal(0, 5, N),
                       "act.p_hit": rng.uniform(0, 1, N)})
    df.loc[:300, "ssa.c2_pips"] = np.nan                                   # warm-up
    df["fx.gbpusd.ret_1"] = df["fx.gbpusd.ret_1"].where(rng.uniform(size=N) > 0.1)   # 10 % missing
    df["CLOSE@diff"] = df["CLOSE"].diff()
    d.mkdir(parents=True)
    df.to_parquet(d / "layer1_pool_eurusd_h1.parquet", index=False)
    groups = {"CLOSE": "raw_price", "RSI": "inventory", "fx.gbpusd.ret_1": "cross_asset", "fac.fx_f1.ret_1": "cross_asset", "ssa.c2_pips": "ssa", "act.p_hit": "activity", "CLOSE@diff": "raw_price"}
    (d / "MANIFEST.json").write_text(json.dumps({"schema": "layer1_pool_h1.v1", "groups": groups, "features": list(df.columns[1:]), "file_sha256": "ab" * 32, "inputs": {"v4_eval_input": "cd" * 32}}))
    _diag([c for c in df.columns[1:] if "@" not in c], {"CLOSE": "DIFF"}).to_csv(d / "FEATURE_DIAGNOSTICS.csv", index=False)
    return df


def make_layer2(d: Path, suffix=""):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"timestamp": TS, "wv.lc.A4": rng.normal(size=N), "kf.lc.slope": rng.normal(size=N) * 1e-5, "fd.lc.d0.3": rng.normal(size=N)})
    df.loc[:255, ["wv.lc.A4", "fd.lc.d0.3"]] = np.nan
    df["fd.lc.d0.3@diff"] = df["fd.lc.d0.3"].diff()
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"layer2_pool_eurusd_h1{suffix}.parquet", index=False)
    cols = {"wv.lc.A4": {"parent": "log_close", "component": "A4", "unit": "u", "algorithm": "DWT_DB4_TRAILING_WINDOW", "proxy_of": None, "warmup_bars": 255, "family": "wavelet"},
            "kf.lc.slope": {"parent": "log_close", "component": "filtered_slope", "unit": "u", "algorithm": "KALMAN_LOCAL_LINEAR_TREND_FILTER", "proxy_of": None, "warmup_bars": 100, "family": "kalman"},
            "fd.lc.d0.3": {"parent": "log_close", "component": "fracdiff", "unit": "u", "algorithm": "FRACTIONAL_DIFFERENCE_FIXED_WINDOW", "proxy_of": None, "warmup_bars": 255, "family": "fracdiff"}}
    cols["fd.lc.d0.3@diff"] = cols["fd.lc.d0.3"] | {"derived_from": "fd.lc.d0.3"}
    fit_end = "2023-01-01" if suffix else "2024-01-01"
    man = {"schema": "layer2_pool_h1.v1", "source_sha256": "ef" * 32, "file_sha256": "01" * 32, "test_start_closed": "2025-01-01",
           "fit_period": {"start": "2020-01-01", "end_exclusive": fit_end, "applies_to": ["kalman", "fracdiff"]}, "columns": cols,
           "parents": {"log_close": "log(BidClose)"},
           "family_meta": {"wavelet": {"config": {"window": 256}, "library": "pywt"},
                           "kalman": {"params": {"q_level": 1e-6, "q_slope": 1e-24, "r": 2e-8, "negloglik": -1.0, "converged": True, "boundary_hits": ["q_slope"], "boundary_note": "n"}},
                           "fracdiff": {"params": {"d": 0.3, "window": 256, "adf_table": [{"d": 0.3, "adf_stat": -3.4, "adf_p": 0.009}], "rule": "r"}}},
           "future_perturbation": {"wavelet": {"passed": True}, "kalman": {"passed": True}, "fracdiff": {"passed": True}}}
    (d / f"MANIFEST{suffix}.json").write_text(json.dumps(man))
    _diag(["wv.lc.A4", "kf.lc.slope", "fd.lc.d0.3"], {"fd.lc.d0.3": "DIFF"}).to_csv(d / f"FEATURE_DIAGNOSTICS{suffix}.csv", index=False)


def make_r3_manifest(d: Path):
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text(json.dumps({"features": [{"feature_id": "fx.gbpusd.ret_1", "family": "cross_asset_fx", "availability_rule": "source bar at the grid bar", "transform": "log return", "unit": "log return"}]}))


def make_r4(d: Path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"timestamp": TS, "fx.gbpusd.ret_1": rng.normal(size=N)})
    d.mkdir(parents=True)
    df.to_parquet(d / "r4.parquet", index=False)
    (d / "MANIFEST.json").write_text(json.dumps({"file": "r4.parquet", "test_start_closed": "2025-01-01", "revision": "r4", "features": [{"feature_id": "fx.gbpusd.ret_1", "family": "cross_asset_fx", "unit": "log return", "transform": "log return"}]}))


def run(tmp_path, r4=False, suffixes=",_fit_end_2023-01-01"):
    make_layer1(tmp_path / "l1")
    make_layer2(tmp_path / "l2")
    make_layer2(tmp_path / "l2", "_fit_end_2023-01-01")
    make_r3_manifest(tmp_path / "r3")
    if r4:
        make_r4(tmp_path / "r4")
    out = tmp_path / "cat"
    FC.main(["--layer1", str(tmp_path / "l1"), "--layer2", str(tmp_path / "l2"), "--layer2-suffixes", suffixes, "--cross-asset-r3", str(tmp_path / "r3"),
             "--cross-asset-r4", str(tmp_path / "r4"), "--out", str(out)])
    return pd.read_csv(out / "FEATURE_CATALOG.csv"), json.loads((out / "FEATURE_CATALOG.json").read_text()), (out / "CATALOG_SUMMARY.md").read_text()


def test_catalog_columns_exact_and_rows(tmp_path):
    cat, js, md = run(tmp_path)
    assert list(cat.columns) == FC.COLUMNS
    assert js["columns"] == FC.COLUMNS and js["rows"] == len(cat) and js["test_read"] is False
    # 7 layer-1 columns + 4 layer-2 x 2 fit sets + 3 PENDING
    assert len(cat) == 7 + 8 + len(FC.PENDING_ROWS)
    assert cat.feature_id.is_unique
    assert (cat.selection_decision == "PENDING").all() and (cat.consumed_by == "PENDING").all()
    assert cat.proxy_of.isna().all()
    assert "## Counts per family x generation_status" in md


def test_status_assignment_without_r4(tmp_path):
    cat, _, md = run(tmp_path)
    r = cat.set_index("feature_id")
    assert r.loc["l1:fx.gbpusd.ret_1", "generation_status"] == "REPAIR_IN_PROGRESS"
    assert r.loc["l1:fac.fx_f1.ret_1", "generation_status"] == "REPAIR_IN_PROGRESS"
    assert FC.R3_REPAIR in r.loc["l1:fx.gbpusd.ret_1", "params_json"]
    assert r.loc["l1:fx.gbpusd.ret_1", "algorithm"] == "log return"           # from the r3 manifest
    assert r.loc["l1:fac.fx_f1.ret_1", "fit_period"].startswith("2015-01-01")
    assert r.loc["l1:RSI", "generation_status"] == "GENERATED"
    assert (cat[cat.generation_status == "PENDING"].feature_id.str.startswith("pending:")).all()
    assert "ev.calendar" in " ".join(cat[cat.generation_status == "PENDING"].column)
    assert "owner decision" in md


def test_status_assignment_with_r4(tmp_path):
    cat, _, _ = run(tmp_path, r4=True)
    r = cat.set_index("feature_id")
    assert r.loc["l1:fx.gbpusd.ret_1", "generation_status"] == "SUPERSEDED_BY_R4"
    assert r.loc["l1:fac.fx_f1.ret_1", "generation_status"] == "SUPERSEDED_BY_R4"
    assert r.loc["r4:fx.gbpusd.ret_1", "store"] == "cross_asset_r4" and r.loc["r4:fx.gbpusd.ret_1", "generation_status"] == "GENERATED"
    assert r.loc["r4:fx.gbpusd.ret_1", "preprocessing_decision"] in ("NONE", "DIFF", "ASINH")
    assert "adf_p" in json.loads(r.loc["r4:fx.gbpusd.ret_1", "preprocessing_metric_json"])
    # the strength PENDING rows stay while no store produces the strength family
    assert set(cat[cat.generation_status == "PENDING"].family) == {"event_calendar", "strength"}


def test_data_derived_fields(tmp_path):
    cat, _, _ = run(tmp_path)
    r = cat.set_index("feature_id")
    assert r.loc["l1:ssa.c2_pips", "warmup_bars"] == 301
    assert r.loc["l2:wv.lc.A4", "warmup_bars"] == 256
    assert 0.05 < r.loc["l1:fx.gbpusd.ret_1", "missing_share_train"] < 0.15
    assert 0.05 < r.loc["l1:fx.gbpusd.ret_1", "missing_share_2023"] < 0.15
    assert r.loc["l1:RSI", "missing_share_train"] == 0.0
    cov = json.loads(r.loc["l1:ssa.c2_pips", "coverage_monthly_json"])
    assert cov["2022-10"] < 1.0 and cov["2022-12"] == 1.0
    assert r.loc["l1:RSI", "period_start"].startswith("2022-10-01") and r.loc["l1:RSI", "period_end"].startswith("2023-02-02 23:00")
    # outlier share: uniform RSI has no 6-sigma outliers; a normal has ~0
    assert r.loc["l1:RSI", "outlier_share_train"] == 0.0


def test_outlier_share_definition():
    x = np.r_[np.random.default_rng(0).normal(size=5000), 50.0, -50.0]
    s = pd.Series(x, index=pd.date_range("2021-01-01", periods=len(x), freq="h", tz="UTC"))
    st = FC.column_stats(s)
    assert st["outlier_share_train"] == pytest.approx(2 / len(x), rel=0.5)
    const = pd.Series(np.ones(500), index=pd.date_range("2021-01-01", periods=500, freq="h", tz="UTC"))
    assert np.isnan(FC.column_stats(const)["outlier_share_train"])        # MAD = 0 -> undefined, not zero


def test_layer2_metadata_and_fit_sets(tmp_path):
    cat, _, md = run(tmp_path)
    r = cat.set_index("feature_id")
    assert r.loc["l2:kf.lc.slope", "fit_period"] == "2020-01-01 .. 2024-01-01 (exclusive)"
    assert r.loc["l2_fit20230101:kf.lc.slope", "fit_period"] == "2020-01-01 .. 2023-01-01 (exclusive)"
    assert r.loc["l2:wv.lc.A4", "fit_period"] == "NONE (no fitted parameter)"
    f = json.loads(r.loc["l2:kf.lc.slope", "fitted_stats_json"])["fitted"]
    assert f["q_slope"] == 1e-24 and f["boundary_hits"] == ["q_slope"]
    assert json.loads(r.loc["l2:fd.lc.d0.3", "fitted_stats_json"])["fitted"]["adf_table"][0]["adf_p"] == 0.009
    assert r.loc["l2:fd.lc.d0.3", "preprocessing_decision"] == "DIFF"
    assert r.loc["l2:fd.lc.d0.3@diff", "preprocessing_decision"] == "DERIVED_DIFF_OF_fd.lc.d0.3"
    assert r.loc["l2:fd.lc.d0.3@diff", "parent"] == "fd.lc.d0.3"
    assert r.loc["l2:wv.lc.A4", "algorithm"] == "DWT_DB4_TRAILING_WINDOW"
    assert r.loc["l2:wv.lc.A4", "producer_path"] == "tools/layer2_pool_h1.py"
    assert r.loc["l2:wv.lc.A4", "producer_sha256"] == hashlib.sha256((ROOT / "tools/layer2_pool_h1.py").read_bytes()).hexdigest()
    assert "native_wavelet.py" in json.loads(r.loc["l2:wv.lc.A4", "params_json"])["lib_sha256"]
    assert "## Fitted parameters" in md and "q_slope" in md


def test_pending_strength_rows_drop_once_produced():
    rows = FC.pending_rows([], {"strength"})
    assert [r["feature_id"] for r in rows] == ["pending:ev.calendar.consensus_surprise"]
    rows = FC.pending_rows([], set())
    assert len(rows) == len(FC.PENDING_ROWS) and all("produced_as" not in r for r in rows)


def test_refuses_test_rows(tmp_path):
    make_layer1(tmp_path / "l1", with_test_rows=True)
    make_layer2(tmp_path / "l2")
    with pytest.raises(SystemExit, match="TEST_ROWS_FORBIDDEN"):
        FC.main(["--layer1", str(tmp_path / "l1"), "--layer2", str(tmp_path / "l2"), "--layer2-suffixes", "", "--cross-asset-r3", str(tmp_path / "none"),
                 "--cross-asset-r4", str(tmp_path / "none"), "--out", str(tmp_path / "cat")])
