"""C130: univariate raw profile per partition, train-frozen references, exact counts,
row format, forbidden vocabulary and a performance smoke test. Synthetic arrays only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("df_contract")
U = _load("df_profile_univariate")

ROW_KEYS = {"dataset_id", "partition", "metric", "estimator", "value", "status", "reason", "code_sha256",
            "cpu_seconds"}
FORBIDDEN = ("intelligence", "noise-free information", "kolmogorov complexity", "publicly_eligible")


def contract(T, V, meaning="SAMPLE_INDEX", period="NOT_APPLICABLE"):
    ds = "synthetic.c130.test.v1"
    doc = {"schema": C.DATASET_SCHEMA, "dataset_id": ds, "version": "1", "bank": "SYNTHETIC", "files": [],
           "content_sha256": "", "contract_sha256": "",
           "source": {"provider": "generator", "official_url": C.UNKNOWN, "citation": C.UNKNOWN, "doi": C.UNKNOWN,
                      "upstream_owner": C.UNKNOWN},
           "license": {"state": "NOT_APPLICABLE_GENERATED", "id": C.UNKNOWN, "url": C.UNKNOWN,
                       "text_sha256": "UNAVAILABLE", "attribution_required": C.UNKNOWN,
                       "redistribution": C.UNKNOWN, "derivatives": C.UNKNOWN, "evidence": []},
           "time": {"frequency_nominal_seconds": period, "timezone": "UTC", "timestamp_meaning": meaning,
                    "range_start": "0", "range_end": str(T), "availability_rule": "SAMPLE_INDEX",
                    "availability_delay_seconds": 0},
           "panel": {"aligned_common_grid": True, "n_series": V, "alignment_rule": "generated on one grid"},
           "partitions": {"scheme": "CHRONOLOGICAL_FRACTIONS",
                          "fractions": {"train": 0.6, "calibration": 0.2, "confirmation": 0.2},
                          "boundaries": C.chronological_partitions(T), "sealed_periods_excluded": [],
                          "frozen_before_profile": True},
           "dependence": [],
           "variables": [C.variable(ds, f"v{i}", license_state="NOT_APPLICABLE_GENERATED") for i in range(V)],
           "original_fields": {}}
    return C.seal(doc)


def ar1(n, phi, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def pick(rows, metric, partition, vid=None):
    out = [r for r in rows if r["metric"] == metric and r["partition"] == partition
           and (vid is None or r.get("variable_id") == vid)]
    assert len(out) == 1, (metric, partition, len(out))
    return out[0]


def check_rows(rows, module_path):
    sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    for r in rows:
        ident = set(r) - ROW_KEYS
        assert len(ident) == 1 and ident <= {"variable_id", "pair", "group_id"}, r
        assert set(r) == ROW_KEYS | ident
        assert set(r["estimator"]) == {"name", "params", "assumptions"}
        assert r["status"] in {"COMPLETED", "UNAVAILABLE", "INCONCLUSIVE", "FAILED", "NOT_RUN"}
        if r["status"] == "COMPLETED":
            assert isinstance(r["value"], float) and np.isfinite(r["value"])
        else:
            assert r["value"] is None and r["reason"]
        assert r["code_sha256"] == sha
        assert isinstance(r["cpu_seconds"], float) and r["cpu_seconds"] >= 0
    text = json.dumps(rows, allow_nan=False).lower()
    for word in FORBIDDEN:
        assert word not in text


def test_ar1_lag1_acf_and_correlation_time():
    rng = np.random.default_rng(1)
    x = ar1(30000, 0.7, rng)
    ct = contract(x.size, 1)
    rows = U.run_univariate(ct, x[:, None])
    vid = ct["variables"][0]["variable_id"]
    for p in ("train", "calibration", "confirmation"):
        assert abs(pick(rows, "acf_lag_1", p, vid)["value"] - 0.7) < 0.03
        assert abs(pick(rows, "acf_lag_2", p, vid)["value"] - 0.49) < 0.05
        # 0.7^k < 1/e first at k = 3
        assert pick(rows, "correlation_time", p, vid)["value"] in (3.0, 4.0)
    check_rows(rows, ROOT / "tools/df_profile_univariate.py")


def test_fft_acf_matches_direct_pairwise_with_gaps():
    rng = np.random.default_rng(2)
    x = ar1(500, 0.5, rng)
    x[rng.choice(500, 60, replace=False)] = np.nan
    acf, cnt = U.acf_pairwise(x, 10)
    m = np.isfinite(x)
    mu = x[m].mean()
    var = ((x[m] - mu) ** 2).mean()
    for k in (1, 3, 10):
        a, b = x[:-k], x[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        assert cnt[k] == ok.sum()
        assert abs(acf[k] - ((a[ok] - mu) * (b[ok] - mu)).mean() / var) < 1e-9


def test_missing_and_non_finite_counts_are_exact():
    rng = np.random.default_rng(3)
    T = 1000
    x = rng.standard_normal(T)
    ct = contract(T, 1)
    tr = ct["partitions"]["boundaries"]["train"]
    x[[1, 5, 7]] = np.nan
    x[[10, 11]] = np.inf
    x[12] = -np.inf
    x[20] = x[21] = x[22] = 4.0          # two consecutive duplicates
    x[700] = np.nan                        # calibration
    rows = U.run_univariate(ct, x[:, None])
    assert tr == [0, 600]
    assert pick(rows, "missing_count", "train")["value"] == 3.0
    assert pick(rows, "non_finite_count", "train")["value"] == 3.0
    assert pick(rows, "coverage", "train")["value"] == (600 - 3) / 600
    assert pick(rows, "duplicate_consecutive_count", "train")["value"] == 2.0
    assert pick(rows, "n", "train")["value"] == 600.0
    assert pick(rows, "missing_count", "calibration")["value"] == 1.0
    assert pick(rows, "non_finite_count", "calibration")["value"] == 0.0
    assert pick(rows, "missing_count", "confirmation")["value"] == 0.0
    check_rows(rows, ROOT / "tools/df_profile_univariate.py")


def test_mean_shift_in_confirmation_is_detected_by_ks_and_psi():
    rng = np.random.default_rng(4)
    T = 10000
    x = rng.standard_normal(T)
    ct = contract(T, 1)
    s = ct["partitions"]["boundaries"]["confirmation"][0]
    x[s:] += 1.0
    rows = U.run_univariate(ct, x[:, None])
    assert pick(rows, "ks_pvalue_vs_train", "calibration")["value"] > 1e-3
    assert pick(rows, "psi_vs_train", "calibration")["value"] < 0.05
    assert pick(rows, "ks_statistic_vs_train", "confirmation")["value"] > 0.3
    assert pick(rows, "ks_pvalue_vs_train", "confirmation")["value"] < 1e-10
    assert pick(rows, "psi_vs_train", "confirmation")["value"] > 0.5
    assert not [r for r in rows if r["partition"] == "train" and "vs_train" in r["metric"]]


def test_train_rows_do_not_change_when_confirmation_changes():
    rng = np.random.default_rng(5)
    T = 3000
    X = rng.standard_normal((T, 2))
    ct = contract(T, 2)
    s = ct["partitions"]["boundaries"]["confirmation"][0]
    Y = X.copy()
    Y[s:] = Y[s:] * 7 + 3
    Y[s + 5:s + 50, 0] = np.nan
    strip = lambda rows: [{k: v for k, v in r.items() if k != "cpu_seconds"} for r in rows if r["partition"] == "train"]
    a, b = strip(U.run_univariate(ct, X)), strip(U.run_univariate(ct, Y))
    assert a == b and len(a) > 50
    # calibration rows depend on the frozen train reference only, so they are unchanged too
    cal = lambda rows: [{k: v for k, v in r.items() if k != "cpu_seconds"} for r in rows
                        if r["partition"] == "calibration"]
    assert cal(U.run_univariate(ct, X)) == cal(U.run_univariate(ct, Y))


def test_robust_outliers_use_train_center_and_never_delete():
    rng = np.random.default_rng(6)
    T = 2000
    x = rng.standard_normal(T)
    ct = contract(T, 1)
    s = ct["partitions"]["boundaries"]["confirmation"][0]
    x[s + 3] = 50.0
    x[s + 9] = -40.0
    before = x.copy()
    rows = U.run_univariate(ct, x[:, None])
    assert pick(rows, "robust_z_outlier_count", "confirmation")["value"] == 2.0
    assert np.array_equal(before, x)


def test_constant_and_empty_partitions_report_reasons():
    T = 300
    X = np.column_stack([np.ones(T), np.full(T, np.nan)])
    rows = U.run_univariate(contract(T, 2), X)
    check_rows(rows, ROOT / "tools/df_profile_univariate.py")
    reasons = {r["reason"] for r in rows if r["status"] != "COMPLETED"}
    assert {"ZERO_VARIANCE", "NO_FINITE_VALUES", "SAMPLE_INDEX_HAS_NO_TIMESTAMPS"} <= reasons
    const = [r for r in rows if r["metric"] == "constant_flag" and r["status"] == "COMPLETED"]
    assert const and all(r["value"] == 1.0 for r in const)


def test_duplicate_timestamps_are_counted_per_partition():
    T = 100
    ts = (np.arange(T, dtype=np.int64) * 60_000_000_000)
    ts[10] = ts[9]
    ts[70] = ts[69]
    ct = contract(T, 1, meaning="PERIOD_START", period=60)
    rows = U.run_univariate(ct, np.random.default_rng(7).standard_normal((T, 1)), ts)
    assert pick(rows, "duplicate_timestamp_count", "train")["value"] == 1.0
    assert pick(rows, "duplicate_timestamp_count", "calibration")["value"] == 1.0
    assert pick(rows, "duplicate_timestamp_count", "confirmation")["value"] == 0.0


def test_performance_smoke_20_variables_20000_samples():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((20000, 20))
    t = time.perf_counter()
    rows = U.run_univariate(contract(20000, 20), X)
    elapsed = time.perf_counter() - t
    assert elapsed < 8.0, elapsed
    assert len(rows) > 2000
