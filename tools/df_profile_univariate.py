#!/usr/bin/env python3
"""C130 (order 2026-09-12), C147-C148 (order 2026-09-13): raw univariate profile per partition, without targets.

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

ADF and KPSS are descriptors of one contiguous stretch of one partition. They
are never a causality gate and never an eligibility gate, alone or together.

C148 finite-memory unit-root policy (UNIT_ROOT_POLICY, frozen before any
measurement, bound by UNIT_ROOT_POLICY_SHA256): the unit-root tests read the
longest contiguous run of finite values of the partition, of length n.

* n <= UNIT_ROOT_EXACT_MAX_N (200,000): EXACT, the whole run, Schwert lag on n.
* n >  UNIT_ROOT_EXACT_MAX_N: BLOCK_APPROX. Three contiguous blocks of exactly
  UNIT_ROOT_EXACT_MAX_N consecutive observations at deterministic offsets
  (start, middle, end of the run), the Schwert lag computed on the block
  length, one row per offset, plus the spread (max - min) across offsets. The
  exact rows are emitted NOT_RUN with that reason, so n is never reduced
  silently. Blocks are never thinned: thinning would change the sampling rate
  and therefore the estimand.

Why 200,000: the ADF OLS design is n * (lag + 2) float64 and statsmodels
peaks at about 5.04 times that (C146 PRE, measured 100k-400k). At n = 200,000
the Schwert lag is 80, the design is 125 MiB and the declared peak (factor 5.5)
is about 690 MiB, which fits one task on the smallest worker with room for the
rest of the profile. 200,000 observations is far past the sample sizes at
which MacKinnon's response-surface critical values and the KPSS table are
asymptotically accurate, so the block statistic is a full-power descriptor of
its stretch; the spread across the three offsets reports how much the answer
depends on which stretch was read. Every unit-root row declares its temporal
universe as [start, end) in partition row coordinates.

C147: every metric group asks an optional `gate` before allocating. A group
the gate does not admit is emitted NOT_RUN with reason NOT_RUN_RESOURCE_BOUND
and its library is never invoked. Without a gate every group runs, exactly as
before.

Entry points: run_univariate(contract, X, timestamps=None, gate=None, policy=None)
-> list of rows; timestamp_rows(...) and variable_rows(...) are the column-wise
generators used by the bounded runner.
"""
from __future__ import annotations

import hashlib
import json
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

# ------------------------------------------------ C148 unit-root policy
UNIT_ROOT_EXACT_MAX_N = 200_000
UNIT_ROOT_BLOCK_OFFSETS = ("start", "middle", "end")
UNIT_ROOT_POLICY = {
    "schema": "crispdm.data_foundation.unit_root_policy.v1",
    "exact_max_n": UNIT_ROOT_EXACT_MAX_N,
    "min_n": UNIT_ROOT_MIN_N,
    "series": "longest contiguous run of finite values inside the partition",
    "exact_rule": "EXACT on the whole run when n <= exact_max_n",
    "approx_rule": "BLOCK_APPROX when n > exact_max_n: contiguous blocks of exactly exact_max_n consecutive "
                   "observations, never thinned",
    "block_offsets": {"start": "run_start", "middle": "run_start + floor((n - exact_max_n) / 2)",
                      "end": "run_end - exact_max_n"},
    "lag_rule": "floor(12 * (m/100)^(1/4)), clipped to [0, m//2 - 2], m = length actually tested",
    "spread": "max - min across the three offsets, only when all three completed",
    "role": "descriptor; never a causality gate and never an eligibility gate",
}
MUTATION_EXACT_MAX_N = 10 ** 12   # test-only policy override used by the preflight-bypass mutation


def policy_sha256(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


UNIT_ROOT_POLICY_SHA256 = policy_sha256(UNIT_ROOT_POLICY)
RESOURCE_REASON = "NOT_RUN_RESOURCE_BOUND"
PREREQUISITE_REASON = "NOT_RUN_RESOURCE_BOUND_PREREQUISITE"


def schwert_lag(m: int) -> int:
    lag = int(math.floor(12 * (m / 100.0) ** 0.25))
    return max(0, min(lag, m // 2 - 2))


def unit_root_blocks(run_start: int, run_end: int, policy=None):
    """-> (mode, [(label, start, end), ...]) in partition row coordinates."""
    policy = policy or UNIT_ROOT_POLICY
    n = run_end - run_start
    B = int(policy["exact_max_n"])
    if n <= B:
        return "EXACT", [("exact", run_start, run_end)]
    mid = run_start + (n - B) // 2
    return "BLOCK_APPROX", [("start", run_start, run_start + B), ("middle", mid, mid + B),
                            ("end", run_end - B, run_end)]


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


def admit_all(group, key=None, partition=None, **sizes):
    """The default gate: every group runs exactly."""
    return {"decision": "RUN_EXACT", "window": None}


def contract_layout(contract, X, timestamps):
    """Partitions as (name, start, end) and variable ids, checked against X."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a (T, V) array")
    T, V = X.shape
    parts, vids, timestamps = layout(contract, T, V, timestamps)
    return X, parts, vids, timestamps


def layout(contract, T, V, timestamps):
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
    return parts, vids, timestamps


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


def longest_finite_run_bounds(x):
    """[start, end) of the longest run of finite values; (0, 0) when none."""
    m = np.isfinite(x).astype(np.int8)
    if not m.any():
        return 0, 0
    d = np.diff(np.concatenate(([0], m, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def longest_finite_run(x):
    s, e = longest_finite_run_bounds(x)
    return x[s:e]


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
              "diagnostic, not a decision; no differencing or transformation is applied",
              "a descriptor only: never a causality gate and never an eligibility gate"]
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

COUNT_METRICS = (("n", E_COUNT), ("missing_count", E_MISSING), ("non_finite_count", E_NONFINITE),
                 ("coverage", E_COVERAGE), ("duplicate_consecutive_count", E_DUP_CONSEC))
QUANTILE_METRICS = ([("constant_flag", E_CONSTANT), ("cardinality", E_CARD)]
                    + [(f"quantile_{q}", E_QUANT) for q in QUANTILES]
                    + [("mad", E_MAD), ("iqr", E_IQR), ("range", E_RANGE), ("upper_tail_ratio", E_UTR),
                       ("lower_tail_ratio", E_LTR), ("moors_kurtosis", E_MOORS), ("bowley_skewness", E_BOWLEY)])
ACF_METRICS = [(f"acf_lag_{k}", E_ACF) for k in ACF_LAGS] + [("correlation_time", E_CTIME)]
ADF_METRICS = [("adf_statistic", E_ADF), ("adf_pvalue", E_ADF)]
KPSS_METRICS = [("kpss_statistic", E_KPSS), ("kpss_pvalue", E_KPSS)]
SHIFT_METRICS = (("ks_statistic_vs_train", E_KS), ("ks_pvalue_vs_train", E_KS), ("psi_vs_train", E_PSI))
OUTLIER_METRICS = (("robust_z_outlier_count", E_OUT), ("robust_z_outlier_fraction", E_OUT_F))


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
def run_univariate(contract, X, timestamps=None, gate=None, policy=None):
    with blas_single_thread():
        return _run_univariate(contract, X, timestamps, gate, policy)


def _run_univariate(contract, X, timestamps=None, gate=None, policy=None):
    X, parts, vids, timestamps = contract_layout(contract, X, timestamps)
    ds = contract["dataset_id"]
    rows = list(timestamp_rows(ds, parts, timestamps, contract["time"]["timestamp_meaning"], gate))
    for j, vid in enumerate(vids):
        rows.extend(variable_rows(ds, vid, X[:, j], parts, gate, policy))
    return rows


def timestamp_rows(ds, parts, timestamps, meaning, gate=None):
    gate = gate or admit_all
    for pname, s, e in parts:
        t0 = time.process_time()
        g = {"group_id": "dataset_timestamps"}
        if timestamps is None:
            why = "SAMPLE_INDEX_HAS_NO_TIMESTAMPS" if meaning == "SAMPLE_INDEX" else "TIMESTAMPS_NOT_PROVIDED"
            yield make_row(ds, g, pname, "duplicate_timestamp_count", E_DUP_TS, None, "UNAVAILABLE", why)
        elif gate("timestamps_duplicates", None, pname, n=e - s)["decision"] == "NOT_RUN_RESOURCE_BOUND":
            yield make_row(ds, g, pname, "duplicate_timestamp_count", E_DUP_TS, None, "NOT_RUN", RESOURCE_REASON)
        else:
            ts = timestamps[s:e]
            yield make_row(ds, g, pname, "duplicate_timestamp_count", E_DUP_TS,
                           ts.size - np.unique(ts).size, cpu=time.process_time() - t0)


def variable_rows(ds, vid, xcol, parts, gate=None, policy=None):
    """Every row of one variable, partition by partition, from its full column."""
    gate = gate or admit_all
    key = {"variable_id": vid}
    n_train = parts[0][2] - parts[0][1]
    have_ref = gate("train_reference", vid, "train", n=n_train)["decision"] != "NOT_RUN_RESOURCE_BOUND"
    tf = None
    if have_ref:
        xt = xcol[parts[0][1]:parts[0][2]]
        tf = np.sort(xt[np.isfinite(xt)])
        del xt
        if tf.size:
            t_med = _q_sorted(tf, 0.5)
            t_mad = float(np.median(np.abs(tf - t_med)))
            t_edges = frozen_quantile_edges(tf, PSI_BINS)
    for pname, s, e in parts:
        x = xcol[s:e]
        out = []
        add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: out.append(
            make_row(ds, key, pname, metric, estimator, value, status, reason, cpu))

        def not_run(metrics, reason=RESOURCE_REASON):
            for metric, estimator in metrics:
                add(metric, estimator, None, "NOT_RUN", reason)

        n = x.size
        counted = gate("counts", vid, pname, n=n)["decision"] != "NOT_RUN_RESOURCE_BOUND"
        if not counted:
            not_run(COUNT_METRICS)
            not_run(QUANTILE_METRICS + ACF_METRICS + ADF_METRICS + KPSS_METRICS, PREREQUISITE_REASON)
            if pname != "train":
                not_run(SHIFT_METRICS, PREREQUISITE_REASON)
            not_run(OUTLIER_METRICS, PREREQUISITE_REASON)
            yield from out
            continue
        t0 = time.process_time()
        nan = np.isnan(x)
        inf = np.isinf(x)
        fin = ~(nan | inf)
        xf = x[fin]
        both = fin[1:] & fin[:-1]
        dup = int(np.count_nonzero((x[1:] == x[:-1]) & both))
        n_nan, n_inf = int(nan.sum()), int(inf.sum())
        del nan, inf, fin, both
        c = time.process_time() - t0
        add("n", E_COUNT, n, cpu=c)
        add("missing_count", E_MISSING, n_nan, cpu=c)
        add("non_finite_count", E_NONFINITE, n_inf, cpu=c)
        add("coverage", E_COVERAGE, (n - n_nan) / n, cpu=c)
        add("duplicate_consecutive_count", E_DUP_CONSEC, dup, cpu=c)

        if xf.size == 0:
            for metric, estimator in QUANTILE_METRICS + ACF_METRICS + ADF_METRICS + KPSS_METRICS:
                add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_VALUES")
        else:
            distinct = None
            if gate("quantiles", vid, pname, n=n)["decision"] == "NOT_RUN_RESOURCE_BOUND":
                not_run(QUANTILE_METRICS)
            else:
                t0 = time.process_time()
                sx = np.sort(xf)
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
                del sx

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
            if distinct is None:
                not_run(ACF_METRICS, PREREQUISITE_REASON)
            elif distinct == 1 or n < 2:
                c = time.process_time() - t0
                for k in ACF_LAGS:
                    add(f"acf_lag_{k}", E_ACF, None, "INCONCLUSIVE", "ZERO_VARIANCE", cpu=c)
                add("correlation_time", E_CTIME, None, "INCONCLUSIVE", "ZERO_VARIANCE", cpu=c)
            elif gate("acf", vid, pname, n=n)["decision"] == "NOT_RUN_RESOURCE_BOUND":
                not_run(ACF_METRICS)
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
                del acf, cnt

            # unit-root diagnostics under the declared policy
            out.extend(unit_root_rows(ds, key, pname, x, gate, policy))

        # distribution shift against the frozen train reference
        if pname != "train":
            t0 = time.process_time()
            if tf is None:
                not_run(SHIFT_METRICS, PREREQUISITE_REASON)
            elif tf.size < SHIFT_MIN_N or xf.size < SHIFT_MIN_N:
                for metric, estimator in SHIFT_METRICS:
                    add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE")
            elif gate("shift_ks_psi", vid, pname, n=n, n_train=n_train)["decision"] == "NOT_RUN_RESOURCE_BOUND":
                not_run(SHIFT_METRICS)
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
        if tf is None:
            not_run(OUTLIER_METRICS, PREREQUISITE_REASON)
        elif tf.size == 0:
            for metric, estimator in OUTLIER_METRICS:
                add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_TRAIN_VALUES")
        elif t_mad == 0:
            for metric, estimator in OUTLIER_METRICS:
                add(metric, estimator, None, "INCONCLUSIVE", "ZERO_TRAIN_MAD")
        elif xf.size == 0:
            for metric, estimator in OUTLIER_METRICS:
                add(metric, estimator, None, "UNAVAILABLE", "NO_FINITE_VALUES")
        elif gate("robust_z", vid, pname, n=n)["decision"] == "NOT_RUN_RESOURCE_BOUND":
            not_run(OUTLIER_METRICS)
        else:
            cnt_out = int(np.count_nonzero(np.abs(xf - t_med) / (MAD_TO_SIGMA * t_mad) > ROBUST_Z_THRESHOLD))
            c = time.process_time() - t0
            add("robust_z_outlier_count", E_OUT, cnt_out, cpu=c)
            add("robust_z_outlier_fraction", E_OUT_F, cnt_out / xf.size, cpu=c)
        del xf
        yield from out


def _ur_est(base, policy, mode, universe, run, lag=None, offset=None):
    params = dict(base["params"], policy=mode, policy_sha256=policy_sha256(policy),
                  exact_max_n=int(policy["exact_max_n"]), temporal_universe=list(universe),
                  run_universe=list(run), run_length=run[1] - run[0])
    if lag is not None:
        params["maxlag"] = lag
    if offset is not None:
        params["block_offset"] = offset
        params["block_length"] = universe[1] - universe[0]
    extra = [] if mode == "EXACT" else [
        "BLOCK_APPROX: the tested stretch is one contiguous block of the run, not the whole run",
        "the run was never thinned; the block keeps the partition's sampling rate"]
    return est(base["name"], params, base["assumptions"] + extra)


def unit_root_rows(ds, key, pname, x, gate=None, policy=None):
    """ADF and KPSS on the longest finite run of the partition slice `x`, under the policy."""
    gate = gate or admit_all
    policy = policy or UNIT_ROOT_POLICY
    vid = key.get("variable_id")
    out = []
    rs, re_ = longest_finite_run_bounds(x)
    run = x[rs:re_]
    n = run.size
    mode, blocks = unit_root_blocks(rs, re_, policy)
    if n < UNIT_ROOT_MIN_N or np.all(run == run[0]):
        why = "INSUFFICIENT_SAMPLE" if n < UNIT_ROOT_MIN_N else "ZERO_VARIANCE"
        for base, metrics in ((E_ADF, ADF_METRICS), (E_KPSS, KPSS_METRICS)):
            e = _ur_est(base, policy, "EXACT", (rs, re_), (rs, re_))
            for metric, _ in metrics:
                out.append(make_row(ds, key, pname, metric, e, None, "INCONCLUSIVE", why))
        return out
    variant = "EXACT" if mode == "EXACT" else "BLOCK_APPROX"
    if mode == "BLOCK_APPROX":
        for base, metrics in ((E_ADF, ADF_METRICS), (E_KPSS, KPSS_METRICS)):
            e = _ur_est(base, policy, "EXACT", (rs, re_), (rs, re_))
            for metric, _ in metrics:
                out.append(make_row(ds, key, pname, metric, e, None, "NOT_RUN",
                                    "RUN_LENGTH_EXCEEDS_EXACT_MAX_N_BLOCK_APPROX_ROWS_REPORTED"))
    results = {"adf": {}, "kpss": {}}
    for label, bs, be in blocks:
        seg = x[bs:be]
        m = be - bs
        lag = schwert_lag(m)
        sfx = "" if mode == "EXACT" else f"_block_{label}"
        offset = None if mode == "EXACT" else label
        e_adf = _ur_est(E_ADF, policy, variant, (bs, be), (rs, re_), lag, offset)
        e_kpss = _ur_est(E_KPSS, policy, variant, (bs, be), (rs, re_), None, offset)
        # ADF
        if gate("unit_root_adf", vid, pname, n_run=m, lag=lag, variant=variant)["decision"] == "NOT_RUN_RESOURCE_BOUND":
            out.append(make_row(ds, key, pname, f"adf_statistic{sfx}", e_adf, None, "NOT_RUN", RESOURCE_REASON))
            out.append(make_row(ds, key, pname, f"adf_pvalue{sfx}", e_adf, None, "NOT_RUN", RESOURCE_REASON))
        else:
            from statsmodels.tsa.stattools import adfuller
            t0 = time.process_time()
            try:
                stat, pval = adfuller(seg, maxlag=lag, regression="c", autolag=None)[:2]
                c = time.process_time() - t0
                out.append(make_row(ds, key, pname, f"adf_statistic{sfx}", e_adf, stat, cpu=c))
                out.append(make_row(ds, key, pname, f"adf_pvalue{sfx}", e_adf, pval, cpu=c))
                results["adf"][label] = (stat, pval)
            except Exception as exc:  # recorded, never raised
                c = time.process_time() - t0
                out.append(make_row(ds, key, pname, f"adf_statistic{sfx}", e_adf, None, "FAILED", type(exc).__name__, cpu=c))
                out.append(make_row(ds, key, pname, f"adf_pvalue{sfx}", e_adf, None, "FAILED", type(exc).__name__, cpu=c))
        # KPSS
        if gate("unit_root_kpss", vid, pname, n_run=m, variant=variant)["decision"] == "NOT_RUN_RESOURCE_BOUND":
            out.append(make_row(ds, key, pname, f"kpss_statistic{sfx}", e_kpss, None, "NOT_RUN", RESOURCE_REASON))
            out.append(make_row(ds, key, pname, f"kpss_pvalue{sfx}", e_kpss, None, "NOT_RUN", RESOURCE_REASON))
        else:
            from statsmodels.tsa.stattools import kpss
            t0 = time.process_time()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    stat, pval = kpss(seg, regression="c", nlags="auto")[:2]
                bound = any("outside of the range" in str(w.message) for w in caught)
                c = time.process_time() - t0
                out.append(make_row(ds, key, pname, f"kpss_statistic{sfx}", e_kpss, stat, cpu=c))
                out.append(make_row(ds, key, pname, f"kpss_pvalue{sfx}", e_kpss, pval,
                                    reason="PVALUE_AT_TABLE_BOUND_VALUE_IS_A_BOUND" if bound else "", cpu=c))
                results["kpss"][label] = (stat, pval)
            except Exception as exc:
                c = time.process_time() - t0
                out.append(make_row(ds, key, pname, f"kpss_statistic{sfx}", e_kpss, None, "FAILED", type(exc).__name__, cpu=c))
                out.append(make_row(ds, key, pname, f"kpss_pvalue{sfx}", e_kpss, None, "FAILED", type(exc).__name__, cpu=c))
    if mode == "BLOCK_APPROX":
        for test, base in (("adf", E_ADF), ("kpss", E_KPSS)):
            got = results[test]
            e = est(f"{base['name']}_block_offset_spread",
                    dict(_ur_est(base, policy, variant, (blocks[0][1], blocks[-1][2]), (rs, re_))["params"],
                         offsets=[[lb, bs, be] for lb, bs, be in blocks], definition="max - min across offsets"),
                    base["assumptions"] + ["sensitivity of the block descriptor to the stretch that was read"])
            for i, what in enumerate(("statistic", "pvalue")):
                if len(got) == len(blocks):
                    vals = [got[lb][i] for lb, _, _ in blocks]
                    out.append(make_row(ds, key, pname, f"{test}_{what}_block_spread", e, max(vals) - min(vals)))
                else:
                    out.append(make_row(ds, key, pname, f"{test}_{what}_block_spread", e, None, "INCONCLUSIVE",
                                        "NOT_EVERY_OFFSET_COMPLETED"))
    return out


def _unit_root_rows(ds, key, pname, run):
    """Backward-compatible name: the unit-root rows of an already-extracted run."""
    return unit_root_rows(ds, key, pname, run)
