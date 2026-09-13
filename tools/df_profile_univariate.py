#!/usr/bin/env python3
"""C130 (order 2026-09-12): raw univariate profile per partition, without targets.

For every contract variable and every chronological partition (train,
calibration, confirmation) separately: counts, coverage, duplicates,
cardinality, robust quantiles and tail descriptors, autocorrelation at
predeclared lags and a correlation time, ADF/KPSS unit-root diagnostics,
distribution shift of each later partition against train (two-sample KS and
PSI on train-frozen quantile bins), and a robust-z outlier count using the
train median/MAD.

Everything that is fitted (PSI bins, the outlier median/MAD) is fitted on
the train partition only and applied frozen. No statistic mixes rows of two
partitions except the declared two-sample shift comparisons, which read the
train sample as the frozen reference. Nothing is deleted, selected or
transformed; every output is a diagnostic, not a decision.

Entry point: run_univariate(contract, X, timestamps=None) -> list of rows.
"""
from __future__ import annotations

import hashlib
import math
import time
import warnings
from pathlib import Path

import numpy as np

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARTITIONS = ("train", "calibration", "confirmation")
STATUSES = ("COMPLETED", "UNAVAILABLE", "INCONCLUSIVE", "FAILED", "NOT_RUN")

QUANTILES = (0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995)
OCTILES = (0.125, 0.375, 0.625, 0.875)
ACF_LAGS = (1, 2, 5, 10, 20, 50)
CORR_TIME_MAX_LAG = 200
ACF_MIN_PAIRS = 30
CARDINALITY_CAP = 10000
PSI_BINS = 10
PSI_EPS = 1e-6
ROBUST_Z_THRESHOLD = 5.0
MAD_TO_SIGMA = 1.4826
UNIT_ROOT_MIN_N = 50
SHIFT_MIN_N = 20


# ------------------------------------------------------------------ rows
def make_row(dataset_id, key, partition, metric, estimator, value, status="COMPLETED", reason="", cpu=0.0):
    """One output row with exactly the shared keys. `key` is
    {"variable_id": id} or {"pair": [a, b]} or {"group_id": g}."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if status == "COMPLETED":
        if value is None or not math.isfinite(float(value)):
            status, value, reason = "INCONCLUSIVE", None, "NON_FINITE_RESULT"
        else:
            value = float(value)
    else:
        value = None
        if not reason:
            raise ValueError("a row that is not COMPLETED needs a reason")
    (k, v), = key.items()
    return {"dataset_id": dataset_id, k: v, "partition": partition, "metric": metric,
            "estimator": estimator, "value": value, "status": status, "reason": reason,
            "code_sha256": CODE_SHA256, "cpu_seconds": round(float(cpu), 6)}


def est(name, params=None, assumptions=()):
    return {"name": name, "params": dict(params or {}), "assumptions": list(assumptions)}


def contract_layout(contract, X, timestamps):
    """Partitions as (name, start, end) and variable ids, checked against X."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a (T, V) array")
    T, V = X.shape
    b = contract["partitions"]["boundaries"]
    parts = [(p, int(b[p][0]), int(b[p][1])) for p in PARTITIONS]
    if parts[0][1] != 0 or parts[0][2] != parts[1][1] or parts[1][2] != parts[2][1] or parts[2][2] != T \
            or any(s >= e for _, s, e in parts):
        raise ValueError("partition boundaries must be contiguous [start, end) blocks covering X")
    vids = [v["variable_id"] for v in contract["variables"]]
    if len(vids) != V:
        raise ValueError(f"contract declares {len(vids)} variables, X has {V} columns")
    meaning = contract["time"]["timestamp_meaning"]
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.shape != (T,) or timestamps.dtype != np.int64:
            raise ValueError("timestamps must be int64 nanoseconds of shape (T,)")
        if meaning == "SAMPLE_INDEX":
            raise ValueError("the contract says SAMPLE_INDEX; timestamps must be None")
    return X, parts, vids, timestamps


# --------------------------------------------------------------- helpers
def _q_sorted(s, q):
    """Linear interpolation quantile (Hyndman-Fan type 7) on a sorted array."""
    h = (s.size - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, s.size - 1)
    return float(s[lo] + (h - lo) * (s[hi] - s[lo]))


def acf_pairwise(x, max_lag):
    """ACF with pairwise-available products: mean over available pairs of
    (x_t - m)(x_{t+k} - m), divided by the mean-square about m, where m and the
    variance use every finite value of the partition. Returns (acf, pair_counts)."""
    m = np.isfinite(x)
    n = x.size
    max_lag = min(max_lag, n - 1)
    mu = x[m].mean()
    var = float(((x[m] - mu) ** 2).mean())
    z = np.where(m, x - mu, 0.0)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))

    def corr(u):
        F = np.fft.rfft(u, nfft)
        return np.fft.irfft((F * np.conj(F)).real, nfft)[: max_lag + 1]

    num = corr(z)
    cnt = np.rint(corr(m.astype(float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = np.where(cnt > 0, num / np.maximum(cnt, 1) / var, np.nan) if var > 0 else np.full(max_lag + 1, np.nan)
    return acf, cnt.astype(np.int64)


def longest_finite_run(x):
    m = np.isfinite(x).astype(np.int8)
    if not m.any():
        return x[:0]
    d = np.diff(np.concatenate(([0], m, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(ends - starts))
    return x[starts[i]:ends[i]]


def frozen_quantile_edges(train_finite, n_bins):
    """Interior bin edges from train quantiles k/n_bins; ties collapse."""
    return np.unique(np.quantile(train_finite, np.arange(1, n_bins) / n_bins))


# ------------------------------------------------------------ estimators
E_COUNT = est("count", {}, ["n counts every row of the partition"])
E_MISSING = est("nan_count", {}, ["missing means NaN"])
E_NONFINITE = est("inf_count", {}, ["non-finite counts +inf and -inf; NaN is counted as missing, not here"])
E_COVERAGE = est("non_missing_fraction", {}, ["coverage = (n - NaN count) / n"])
E_DUP_CONSEC = est("consecutive_identical_count", {},
                   ["counts t where x_t == x_{t-1}, both finite, inside the partition"])
E_DUP_TS = est("duplicate_timestamp_count", {}, ["n - number of distinct int64 timestamps inside the partition"])
E_CONSTANT = est("constant_flag", {}, ["1 if every finite value is equal, else 0"])
E_CARD = est("distinct_finite_values_capped", {"cap": CARDINALITY_CAP},
             ["exact equality of float64 values", "value is min(distinct, cap); at cap it is a lower bound"])
E_QUANT = est("empirical_quantile", {"method": "linear_type7", "levels": list(QUANTILES)},
              ["finite values only"])
E_MAD = est("median_absolute_deviation", {"scale": 1.0}, ["unscaled; multiply by 1.4826 for a normal-consistent sigma",
                                                          "finite values only"])
E_IQR = est("interquartile_range", {"method": "linear_type7"}, ["finite values only"])
E_RANGE = est("max_minus_min", {}, ["finite values only"])
E_UTR = est("upper_tail_ratio", {"definition": "(q0.99 - q0.5) / (q0.75 - q0.5)"},
            ["equals about 3.45 for a normal distribution", "finite values only"])
E_LTR = est("lower_tail_ratio", {"definition": "(q0.5 - q0.01) / (q0.5 - q0.25)"},
            ["equals about 3.45 for a normal distribution", "finite values only"])
E_MOORS = est("moors_octile_kurtosis", {"definition": "((q7/8 - q5/8) + (q3/8 - q1/8)) / (q6/8 - q2/8)",
                                        "reference": "Moors (1988), The Statistician 37:25-32"},
              ["equals about 1.233 for a normal distribution", "finite values only"])
E_BOWLEY = est("bowley_quartile_skewness", {"definition": "(q0.75 + q0.25 - 2 q0.5) / (q0.75 - q0.25)"},
               ["finite values only"])
E_ACF = est("acf_pairwise_available", {"lags": list(ACF_LAGS), "min_pairs": ACF_MIN_PAIRS},
            ["mean and variance from all finite values of the partition",
             "lag-k product averaged over pairs where both x_t and x_{t+k} are finite",
             "pairs never cross a partition boundary"])
E_CTIME = est("first_lag_below_inverse_e", {"threshold": "1/e", "max_lag": CORR_TIME_MAX_LAG,
                                             "min_pairs": ACF_MIN_PAIRS},
              ["value is the first lag k>=1 with ACF(k) < 1/e, in samples",
               "INCONCLUSIVE when no lag up to max_lag crosses, or a lag before crossing lacks min_pairs"])
_UR_ASSUME = ["computed on the longest contiguous run of finite values in the partition",
              "diagnostic, not a decision; no differencing or transformation is applied"]
E_ADF = est("augmented_dickey_fuller", {"library": "statsmodels.tsa.stattools.adfuller", "regression": "c",
                                        "autolag": None, "maxlag_rule": "floor(12 * (n/100)^(1/4)) (Schwert 1989)",
                                        "min_n": UNIT_ROOT_MIN_N},
            ["null hypothesis: unit root", "MacKinnon approximate p-value"] + _UR_ASSUME)
E_KPSS = est("kpss", {"library": "statsmodels.tsa.stattools.kpss", "regression": "c", "nlags": "auto",
                      "nlags_rule": "Hobijn et al. (1998) data-dependent bandwidth", "min_n": UNIT_ROOT_MIN_N},
             ["null hypothesis: level stationarity",
              "p-value interpolated from a table and truncated to [0.01, 0.10]"] + _UR_ASSUME)
_SHIFT_ASSUME = ["reference sample is the train partition, read frozen",
                 "two-sample comparison of marginal distributions; ignores serial dependence, so p-values are optimistic",
                 "diagnostic, not a decision"]
E_KS = est("two_sample_kolmogorov_smirnov", {"library": "scipy.stats.ks_2samp", "method": "asymp",
                                             "reference_partition": "train", "min_n": SHIFT_MIN_N}, _SHIFT_ASSUME)
E_PSI = est("population_stability_index", {"bins": PSI_BINS, "bin_rule": "train quantiles k/10, ties collapsed",
                                           "epsilon_floor": PSI_EPS, "log": "natural",
                                           "reference_partition": "train", "min_n": SHIFT_MIN_N},
            ["PSI = sum (p_part - p_train) * ln(p_part / p_train)"] + _SHIFT_ASSUME)
E_OUT = est("robust_z_exceedance_count", {"threshold": ROBUST_Z_THRESHOLD, "center": "train median",
                                          "scale": "1.4826 * train MAD"},
            ["diagnostic only; nothing is deleted or modified", "finite values only"])
E_OUT_F = est("robust_z_exceedance_fraction", E_OUT["params"],
              ["count divided by the number of finite values of the partition"] + E_OUT["assumptions"])


def blas_single_thread():
    """Single-threaded BLAS so cpu_seconds are honest and small regressions do
    not oversubscribe cores; a no-op when threadpoolctl is absent."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


# ------------------------------------------------------------------- run
def run_univariate(contract, X, timestamps=None):
    with blas_single_thread():
        return _run_univariate(contract, X, timestamps)


def _run_univariate(contract, X, timestamps=None):
    X, parts, vids, timestamps = contract_layout(contract, X, timestamps)
    ds = contract["dataset_id"]
    rows = []
    meaning = contract["time"]["timestamp_meaning"]

    for pname, s, e in parts:
        t0 = time.process_time()
        g = {"group_id": "dataset_timestamps"}
        if timestamps is None:
            why = "SAMPLE_INDEX_HAS_NO_TIMESTAMPS" if meaning == "SAMPLE_INDEX" else "TIMESTAMPS_NOT_PROVIDED"
            rows.append(make_row(ds, g, pname, "duplicate_timestamp_count", E_DUP_TS, None, "UNAVAILABLE", why))
        else:
            ts = timestamps[s:e]
            rows.append(make_row(ds, g, pname, "duplicate_timestamp_count", E_DUP_TS,
                                 ts.size - np.unique(ts).size, cpu=time.process_time() - t0))

    train_ref = {}
    for j, vid in enumerate(vids):
        xt = X[parts[0][1]:parts[0][2], j]
        tf = np.sort(xt[np.isfinite(xt)])
        train_ref[j] = tf

    for j, vid in enumerate(vids):
        key = {"variable_id": vid}
        tf = train_ref[j]
        if tf.size:
            t_med = _q_sorted(tf, 0.5)
            t_mad = float(np.median(np.abs(tf - t_med)))
            t_edges = frozen_quantile_edges(tf, PSI_BINS)
        for pname, s, e in parts:
            x = X[s:e, j]
            add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: rows.append(
                make_row(ds, key, pname, metric, estimator, value, status, reason, cpu))

            t0 = time.process_time()
            n = x.size
            nan = np.isnan(x)
            inf = np.isinf(x)
            fin = ~(nan | inf)
            xf = x[fin]
            sx = np.sort(xf)
            both = fin[1:] & fin[:-1]
            dup = int(np.count_nonzero((x[1:] == x[:-1]) & both))
            c = time.process_time() - t0
            add("n", E_COUNT, n, cpu=c)
            add("missing_count", E_MISSING, int(nan.sum()), cpu=c)
            add("non_finite_count", E_NONFINITE, int(inf.sum()), cpu=c)
            add("coverage", E_COVERAGE, (n - int(nan.sum())) / n, cpu=c)
            add("duplicate_consecutive_count", E_DUP_CONSEC, dup, cpu=c)

            if sx.size == 0:
                for metric, estimator in [("constant_flag", E_CONSTANT), ("cardinality", E_CARD)] + \
                        [(f"quantile_{q}", E_QUANT) for q in QUANTILES] + \
                        [("mad", E_MAD), ("iqr", E_IQR), ("range", E_RANGE), ("upper_tail_ratio", E_UTR),
                         ("lower_tail_ratio", E_LTR), ("moors_kurtosis", E_MOORS), ("bowley_skewness", E_BOWLEY)] + \
                        [(f"acf_lag_{k}", E_ACF) for k in ACF_LAGS] + [("correlation_time", E_CTIME)] + \
                        [("adf_statistic", E_ADF), ("adf_pvalue", E_ADF), ("kpss_statistic", E_KPSS),
                         ("kpss_pvalue", E_KPSS)]:
                    add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_VALUES")
            else:
                t0 = time.process_time()
                distinct = 1 + int(np.count_nonzero(np.diff(sx)))
                c = time.process_time() - t0
                add("constant_flag", E_CONSTANT, 1.0 if distinct == 1 else 0.0, cpu=c)
                add("cardinality", E_CARD, min(distinct, CARDINALITY_CAP),
                    reason="AT_CAP_LOWER_BOUND" if distinct >= CARDINALITY_CAP else "", cpu=c)

                t0 = time.process_time()
                qv = {q: _q_sorted(sx, q) for q in QUANTILES + OCTILES}
                q25, q50, q75 = qv[0.25], qv[0.5], qv[0.75]
                mad = float(np.median(np.abs(sx - q50)))
                c = time.process_time() - t0
                for q in QUANTILES:
                    add(f"quantile_{q}", E_QUANT, qv[q], cpu=c)
                add("mad", E_MAD, mad, cpu=c)
                add("iqr", E_IQR, q75 - q25, cpu=c)
                add("range", E_RANGE, float(sx[-1] - sx[0]), cpu=c)

                def ratio(metric, estimator, num, den):
                    if den > 0:
                        add(metric, estimator, num / den, cpu=c)
                    else:
                        add(metric, estimator, None, "INCONCLUSIVE", "ZERO_DENOMINATOR", cpu=c)
                ratio("upper_tail_ratio", E_UTR, qv[0.99] - q50, q75 - q50)
                ratio("lower_tail_ratio", E_LTR, q50 - qv[0.01], q50 - q25)
                ratio("moors_kurtosis", E_MOORS, (qv[0.875] - qv[0.625]) + (qv[0.375] - qv[0.125]),
                      qv[0.75] - qv[0.25])
                ratio("bowley_skewness", E_BOWLEY, q75 + q25 - 2 * q50, q75 - q25)

                # autocorrelation and correlation time
                t0 = time.process_time()
                if distinct == 1 or n < 2:
                    c = time.process_time() - t0
                    for k in ACF_LAGS:
                        add(f"acf_lag_{k}", E_ACF, None, "INCONCLUSIVE", "ZERO_VARIANCE", cpu=c)
                    add("correlation_time", E_CTIME, None, "INCONCLUSIVE", "ZERO_VARIANCE", cpu=c)
                else:
                    acf, cnt = acf_pairwise(x, max(CORR_TIME_MAX_LAG, max(ACF_LAGS)))
                    c = time.process_time() - t0
                    for k in ACF_LAGS:
                        if k >= acf.size:
                            add(f"acf_lag_{k}", E_ACF, None, "UNAVAILABLE", "LAG_EXCEEDS_PARTITION", cpu=c)
                        elif cnt[k] < ACF_MIN_PAIRS:
                            add(f"acf_lag_{k}", E_ACF, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
                        else:
                            add(f"acf_lag_{k}", E_ACF, acf[k], cpu=c)
                    ctime, why = None, "NO_CROSSING_WITHIN_MAX_LAG"
                    for k in range(1, min(CORR_TIME_MAX_LAG, acf.size - 1) + 1):
                        if cnt[k] < ACF_MIN_PAIRS:
                            why = "INSUFFICIENT_SAMPLE"
                            break
                        if acf[k] < 1 / math.e:
                            ctime = k
                            break
                    if ctime is None:
                        add("correlation_time", E_CTIME, None, "INCONCLUSIVE", why, cpu=c)
                    else:
                        add("correlation_time", E_CTIME, ctime, cpu=c)

                # unit-root diagnostics
                run = longest_finite_run(x)
                rows.extend(_unit_root_rows(ds, key, pname, run))

            # distribution shift against the frozen train reference
            if pname != "train":
                t0 = time.process_time()
                if tf.size < SHIFT_MIN_N or xf.size < SHIFT_MIN_N:
                    for metric, estimator in (("ks_statistic_vs_train", E_KS), ("ks_pvalue_vs_train", E_KS),
                                              ("psi_vs_train", E_PSI)):
                        add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE")
                else:
                    from scipy.stats import ks_2samp
                    r = ks_2samp(tf, xf, method="asymp")
                    c = time.process_time() - t0
                    add("ks_statistic_vs_train", E_KS, r.statistic, cpu=c)
                    add("ks_pvalue_vs_train", E_KS, r.pvalue, cpu=c)
                    t0 = time.process_time()
                    k = t_edges.size + 1
                    p_tr = np.bincount(np.searchsorted(t_edges, tf, side="right"), minlength=k) / tf.size
                    p_pa = np.bincount(np.searchsorted(t_edges, xf, side="right"), minlength=k) / xf.size
                    p_tr, p_pa = np.maximum(p_tr, PSI_EPS), np.maximum(p_pa, PSI_EPS)
                    add("psi_vs_train", E_PSI, float(np.sum((p_pa - p_tr) * np.log(p_pa / p_tr))),
                        cpu=time.process_time() - t0)

            # robust-z outliers with the frozen train center and scale
            t0 = time.process_time()
            if tf.size == 0:
                for metric, estimator in (("robust_z_outlier_count", E_OUT), ("robust_z_outlier_fraction", E_OUT_F)):
                    add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_TRAIN_VALUES")
            elif t_mad == 0:
                for metric, estimator in (("robust_z_outlier_count", E_OUT), ("robust_z_outlier_fraction", E_OUT_F)):
                    add(metric, estimator, None, "INCONCLUSIVE", "ZERO_TRAIN_MAD")
            elif xf.size == 0:
                for metric, estimator in (("robust_z_outlier_count", E_OUT), ("robust_z_outlier_fraction", E_OUT_F)):
                    add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_VALUES")
            else:
                cnt_out = int(np.count_nonzero(np.abs(xf - t_med) / (MAD_TO_SIGMA * t_mad) > ROBUST_Z_THRESHOLD))
                c = time.process_time() - t0
                add("robust_z_outlier_count", E_OUT, cnt_out, cpu=c)
                add("robust_z_outlier_fraction", E_OUT_F, cnt_out / xf.size, cpu=c)
    return rows


def _unit_root_rows(ds, key, pname, run):
    from statsmodels.tsa.stattools import adfuller, kpss
    out = []
    n = run.size
    names = (("adf_statistic", "adf_pvalue", E_ADF), ("kpss_statistic", "kpss_pvalue", E_KPSS))
    if n < UNIT_ROOT_MIN_N or np.all(run == run[0]):
        why = "INSUFFICIENT_SAMPLE" if n < UNIT_ROOT_MIN_N else "ZERO_VARIANCE"
        for a, b, estimator in names:
            out.append(make_row(ds, key, pname, a, estimator, None, "INCONCLUSIVE", why))
            out.append(make_row(ds, key, pname, b, estimator, None, "INCONCLUSIVE", why))
        return out
    t0 = time.process_time()
    try:
        lag = int(math.floor(12 * (n / 100.0) ** 0.25))
        lag = max(0, min(lag, n // 2 - 2))
        stat, pval = adfuller(run, maxlag=lag, regression="c", autolag=None)[:2]
        c = time.process_time() - t0
        out.append(make_row(ds, key, pname, "adf_statistic", E_ADF, stat, cpu=c))
        out.append(make_row(ds, key, pname, "adf_pvalue", E_ADF, pval, cpu=c))
    except Exception as exc:  # recorded, never raised
        c = time.process_time() - t0
        out.append(make_row(ds, key, pname, "adf_statistic", E_ADF, None, "FAILED", type(exc).__name__, cpu=c))
        out.append(make_row(ds, key, pname, "adf_pvalue", E_ADF, None, "FAILED", type(exc).__name__, cpu=c))
    t0 = time.process_time()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pval = kpss(run, regression="c", nlags="auto")[:2]
        bound = any("outside of the range" in str(w.message) for w in caught)
        c = time.process_time() - t0
        out.append(make_row(ds, key, pname, "kpss_statistic", E_KPSS, stat, cpu=c))
        out.append(make_row(ds, key, pname, "kpss_pvalue", E_KPSS, pval,
                            reason="PVALUE_AT_TABLE_BOUND_VALUE_IS_A_BOUND" if bound else "", cpu=c))
    except Exception as exc:
        c = time.process_time() - t0
        out.append(make_row(ds, key, pname, "kpss_statistic", E_KPSS, None, "FAILED", type(exc).__name__, cpu=c))
        out.append(make_row(ds, key, pname, "kpss_pvalue", E_KPSS, None, "FAILED", type(exc).__name__, cpu=c))
    return out
