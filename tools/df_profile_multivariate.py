#!/usr/bin/env python3
"""C132 (order 2026-09-12): multivariate profile inside TRAIN only, without targets.

Only the train partition is read. Pairwise statistics use temporally common
samples (pairwise complete). Every lag is causal: "A leads B by k" compares the
past of A (t - k) with the present of B (t); the other direction is computed the
same way with the roles swapped. Nothing is transformed, selected or kept:
PCA, effective rank, clusters and common/private components are diagnostics
and candidates only.

Pair cap (deterministic, data-independent): variables are ordered by
variable_id (lexicographic). The expensive pairwise diagnostics run on the
first k variables of that order, k the largest integer with k(k-1)/2 <=
MAX_PAIRS; the correlation-matrix diagnostics (PCA, clusters, common/private)
run on the first MAX_VARIABLES_MATRIX variables of the same order. The counts of
excluded pairs and variables are reported as rows.

Entry point: run_multivariate(contract, X, timestamps=None) -> list of rows.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import time
from pathlib import Path

import numpy as np

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARTITIONS = ("train", "calibration", "confirmation")
STATUSES = ("COMPLETED", "UNAVAILABLE", "INCONCLUSIVE", "FAILED", "NOT_RUN")

MAX_PAIRS = 1225
MAX_VARIABLES_MATRIX = 400
MIN_PAIR_N = 30
BICOR_C = 9.0
MAX_LAG = 20
LAG_SUBWINDOWS = 8
LAG_STABLE_FRACTION = 0.75
COH_NPERSEG = 256
COH_SURROGATES = 39
COH_SEED = 20260912
MI_BINS = 16
MI_SHUFFLES = 20
MI_SEED = 20260913
PCA_COMPONENTS_REPORTED = 5
CLUSTER_LINKAGE = "average"
CLUSTER_CUT_DISTANCE = 0.5
CLUSTER_BOOTSTRAPS = 30
CLUSTER_SEED = 20260914
CLUSTER_STABLE_JACCARD = 0.75
EIG_TOL_REL = 1e-10


# ------------------------------------------------------------------ rows
def make_row(dataset_id, key, partition, metric, estimator, value, status="COMPLETED", reason="", cpu=0.0):
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
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.shape != (T,) or timestamps.dtype != np.int64:
            raise ValueError("timestamps must be int64 nanoseconds of shape (T,)")
        if contract["time"]["timestamp_meaning"] == "SAMPLE_INDEX":
            raise ValueError("the contract says SAMPLE_INDEX; timestamps must be None")
    return X, parts, vids, timestamps


def blas_single_thread():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover
        return contextlib.nullcontext()


def pair_cap(vids):
    """(order, pair_variables, matrix_variables): indices ordered by variable_id."""
    order = sorted(range(len(vids)), key=lambda i: vids[i])
    k = int(math.floor((1 + math.sqrt(1 + 8 * MAX_PAIRS)) / 2))
    while k * (k - 1) // 2 > MAX_PAIRS:
        k -= 1
    return order, order[:k], order[:MAX_VARIABLES_MATRIX]


# ------------------------------------------------------------ estimators
def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else None


def spearman(a, b):
    from scipy.stats import rankdata
    return pearson(rankdata(a), rankdata(b))


def biweight_midcorrelation(a, b, c=BICOR_C):
    def weighted(u):
        med = np.median(u)
        mad = np.median(np.abs(u - med))
        if mad == 0:
            return None
        v = (u - med) / (c * mad)
        w = (1 - v ** 2) ** 2 * (np.abs(v) < 1)
        return (u - med) * w
    ua, ub = weighted(a), weighted(b)
    if ua is None or ub is None:
        return None
    d = math.sqrt(float(ua @ ua) * float(ub @ ub))
    return float(ua @ ub) / d if d > 0 else None


def pairwise_pearson_matrix(X):
    """Exact pairwise-complete Pearson matrix with pair counts; NaN where a pair
    has fewer than MIN_PAIR_N common samples or zero variance."""
    M = np.isfinite(X).astype(float)
    Z = np.where(M > 0, X, 0.0)
    n = M.T @ M
    Sa = Z.T @ M
    Saa = (Z * Z).T @ M
    Sab = Z.T @ Z
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Sab - Sa * Sa.T / n
        va = Saa - Sa ** 2 / n
        vb = va.T
        r = cov / np.sqrt(va * vb)
    r[(n < MIN_PAIR_N) | ~(va > 1e-12 * np.maximum(Saa, 1e-300)) | ~(vb > 1e-12 * np.maximum(Saa.T, 1e-300))] = np.nan
    r = np.clip(r, -1.0, 1.0)
    np.fill_diagonal(r, 1.0)
    return r, n


def causal_lagged_corr(a, b, k):
    """corr(a_{t-k}, b_t) over common finite samples."""
    x, y = (a[:a.size - k], b[k:]) if k > 0 else (a, b)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_PAIR_N:
        return None
    return pearson(x[m], y[m])


def best_signed_lag(a, b):
    """+k: A's past at t-k with B's present; -k: B's past with A's present."""
    ab = [causal_lagged_corr(a, b, k) for k in range(MAX_LAG + 1)]
    ba = [causal_lagged_corr(b, a, k) for k in range(MAX_LAG + 1)]
    cands = [(k, ab[k]) for k in range(MAX_LAG + 1)] + [(-k, ba[k]) for k in range(1, MAX_LAG + 1)]
    cands = [(lag, c) for lag, c in cands if c is not None]
    if not cands:
        return ab, ba, None, None
    lag, c = max(cands, key=lambda t: (abs(t[1]), -abs(t[0]), t[0]))
    return ab, ba, lag, c


def longest_common_finite_run(a, b):
    m = (np.isfinite(a) & np.isfinite(b)).astype(np.int8)
    if not m.any():
        return 0, 0
    d = np.diff(np.concatenate(([0], m, [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    i = int(np.argmax(e - s))
    return int(s[i]), int(e[i])


def phase_randomized(b, count, rng):
    """Fourier phase-randomized surrogates: same amplitude spectrum, random phases."""
    n = b.size
    F = np.fft.rfft(b - b.mean())
    ph = rng.uniform(0, 2 * np.pi, size=(count, F.size))
    ph[:, 0] = 0.0
    S = np.fft.irfft(np.abs(F)[None, :] * np.exp(1j * ph), n, axis=-1)
    return S + b.mean()


def miller_madow_entropy_nats(counts):
    c = counts[counts > 0].astype(float)
    N = c.sum()
    p = c / N
    return float(-(p * np.log(p)).sum() + (c.size - 1) / (2 * N))


def mutual_information_bits(sa, sb, k):
    ha = miller_madow_entropy_nats(np.bincount(sa, minlength=k))
    hb = miller_madow_entropy_nats(np.bincount(sb, minlength=k))
    hab = miller_madow_entropy_nats(np.bincount(sa * k + sb, minlength=k * k))
    return (ha + hb - hab) / math.log(2)


def frozen_symbols(x):
    edges = np.unique(np.quantile(x, np.arange(1, MI_BINS) / MI_BINS))
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def corr_eigen(R):
    """Eigen-decomposition of a correlation matrix with NaN off-diagonals set to 0
    and negative eigenvalues clipped at 0. Returns (eigenvalues desc, vectors, negatives)."""
    R = np.where(np.isfinite(R), R, 0.0)
    w, U = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    w, U = w[idx], U[:, idx]
    neg = int((w < -EIG_TOL_REL * max(w.max(), 1.0)).sum())
    return np.clip(w, 0.0, None), U, neg


def cluster_labels(R):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - np.abs(np.where(np.isfinite(R), R, 0.0))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Zl = linkage(squareform(np.clip(D, 0, None), checks=False), method=CLUSTER_LINKAGE)
    return fcluster(Zl, t=CLUSTER_CUT_DISTANCE, criterion="distance")


def moving_block_indices(T, block, rng):
    nb = int(math.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]


_TRAIN = ["train partition only"]
_PAIR = ["pairwise complete: samples where both variables are finite", f"min common samples {MIN_PAIR_N}"] + _TRAIN
E_PEARSON = est("pearson_correlation", {}, _PAIR)
E_SPEARMAN = est("spearman_correlation", {"ties": "average ranks"}, _PAIR)
E_BICOR = est("biweight_midcorrelation", {"c": BICOR_C, "reference": "Wilcox (2012); Langfelder & Horvath (2012)"},
              _PAIR + ["INCONCLUSIVE when a variable has zero MAD"])
_LAG_A = _PAIR + ["causal: only the past of one variable with the present of the other",
                  "lagged pairs never leave the train partition"]
E_XAB = est("causal_cross_correlation_a_past_b_present", {"max_lag": MAX_LAG,
                                                          "definition": "corr(a_{t-k}, b_t)"}, _LAG_A)
E_XBA = est("causal_cross_correlation_b_past_a_present", {"max_lag": MAX_LAG,
                                                          "definition": "corr(b_{t-k}, a_t)"}, _LAG_A)
_BEST_P = {"max_lag": MAX_LAG, "sign": "+k means a leads b by k; -k means b leads a by k",
           "selection": "max |corr|; ties to smaller |lag|, then positive"}
E_BEST = est("best_signed_causal_lag", _BEST_P, _LAG_A)
E_BEST_C = est("best_signed_causal_lag_correlation", _BEST_P, _LAG_A)
_STAB_P = {**_BEST_P, "subwindows": LAG_SUBWINDOWS, "rule": "contiguous equal train sub-windows",
           "stable_if_modal_fraction_at_least": LAG_STABLE_FRACTION}
E_LSTAB = est("best_lag_modal_fraction_across_subwindows", _STAB_P,
              _LAG_A + ["fraction = windows whose best lag equals the modal lag / windows with a computable lag"])
E_LMODE = est("best_lag_modal_value_across_subwindows", _STAB_P, _LAG_A)
E_LSTABLE = est("best_lag_stable_flag", _STAB_P, _LAG_A + ["1 when the modal fraction reaches the threshold"])
_COH_P = {"library": "scipy.signal.coherence", "fs": 1.0, "nperseg": COH_NPERSEG, "window": "hann",
          "band": "all frequencies > 0 (excludes DC)", "surrogates": COH_SURROGATES, "seed": COH_SEED,
          "surrogate": "Fourier phase randomization of b (amplitude spectrum kept)", "min_length": 4 * COH_NPERSEG}
_COH_A = ["computed on the longest contiguous run where both variables are finite"] + _TRAIN + \
         ["null percentile = 100 * fraction of surrogate band means <= observed"]
E_COH = est("band_mean_magnitude_squared_coherence", _COH_P, _COH_A)
E_COH_P = est("coherence_null_percentile", _COH_P, _COH_A)
E_COH_LO = est("coherence_null_quantile_0.025", _COH_P, _COH_A)
E_COH_HI = est("coherence_null_quantile_0.975", _COH_P, _COH_A)
_MI_P = {"bins": MI_BINS, "bin_rule": "train quantiles k/16 per variable over the pair's common samples",
         "correction": "Miller-Madow on each entropy", "unit": "bits", "shuffles": MI_SHUFFLES, "seed": MI_SEED}
_MI_A = _PAIR + ["binned estimator; the shuffle baseline permutes b and destroys both pairing and temporal "
                 "dependence, so it is a lenient baseline under autocorrelation"]
E_MI = est("mutual_information_binned_miller_madow", _MI_P, _MI_A)
E_MI_SM = est("mutual_information_shuffle_baseline_mean", _MI_P, _MI_A)
E_MI_SQ = est("mutual_information_shuffle_baseline_q95", _MI_P, _MI_A)
E_MI_EX = est("mutual_information_excess_over_shuffle_mean", _MI_P, _MI_A)
_MAT_A = ["pairwise-complete Pearson correlation matrix; undefined entries set to 0",
          "negative eigenvalues clipped at 0", "diagnostic only; never a transformation"] + _TRAIN
E_PCA = est("pca_explained_variance_ratio", {"source": "eigenvalues of the correlation matrix",
                                             "components_reported": PCA_COMPONENTS_REPORTED}, _MAT_A)
E_ERANK = est("effective_rank_roy_vetterli",
              {"definition": "exp(-sum p_i ln p_i), p_i = sqrt(lambda_i) / sum sqrt(lambda_j)",
               "equivalence": "sigma_i of the z-scored matrix are proportional to sqrt(lambda_i) of its correlation",
               "reference": "Roy & Vetterli (2007), EUSIPCO 2007, pp. 606-610"}, _MAT_A)
E_NEG = est("negative_eigenvalue_count", {"tolerance": f"{EIG_TOL_REL} * max(lambda_max, 1)"}, _MAT_A)
_CL_P = {"distance": "1 - |pearson|", "linkage": CLUSTER_LINKAGE, "cut_distance": CLUSTER_CUT_DISTANCE,
         "bootstrap": "moving block", "block_length_rule": "max(10, ceil(sqrt(T_train)))",
         "bootstraps": CLUSTER_BOOTSTRAPS, "seed": CLUSTER_SEED,
         "stability": "mean over bootstraps of the best Jaccard between the cluster and any bootstrap cluster "
                      "(Hennig 2007)", "identified_if_stability_at_least": CLUSTER_STABLE_JACCARD}
_CL_A = _MAT_A + ["only clusters with at least two members are candidate groups",
                  "a group below the stability threshold is GROUP_NOT_IDENTIFIED"]
E_CL_COUNT = est("identified_group_count", _CL_P, _CL_A)
E_CL_STAB = est("cluster_bootstrap_jaccard_mean", _CL_P, _CL_A)
_CP_A = _MAT_A + ["CANDIDATE_NOT_A_TRANSFORMATION: a descriptor of shared variation, not a component to use",
                  "loading sign fixed so the loadings sum to a non-negative number"]
E_PC1L = est("pc1_loading", {"definition": "u1_i * sqrt(lambda_1)"}, _CP_A)
E_PC1S = est("pc1_common_variance_share", {"definition": "u1_i^2 * lambda_1"}, _CP_A)
E_PRIV = est("private_residual_variance_share", {"definition": "1 - u1_i^2 * lambda_1"}, _CP_A)
E_CAP = est("pair_cap", {"max_pairs": MAX_PAIRS, "max_variables_matrix": MAX_VARIABLES_MATRIX,
                         "rule": "variables ordered by variable_id; first k with k(k-1)/2 <= max_pairs for pairwise "
                                 "diagnostics; first max_variables_matrix for matrix diagnostics"},
             ["data-independent and deterministic; not a selection by any statistic"])


# ------------------------------------------------------------------- run
def run_multivariate(contract, X, timestamps=None):
    with blas_single_thread():
        return _run_multivariate(contract, X, timestamps)


def _run_multivariate(contract, X, timestamps=None):
    X, parts, vids, timestamps = contract_layout(contract, X, timestamps)
    ds = contract["dataset_id"]
    tr = X[parts[0][1]:parts[0][2]]
    T, V = tr.shape
    rows = []
    P = "train"
    order, pvars, mvars = pair_cap(vids)
    g = {"group_id": "pair_cap"}
    rows.append(make_row(ds, g, P, "pairs_total", E_CAP, V * (V - 1) // 2))
    rows.append(make_row(ds, g, P, "pairs_evaluated", E_CAP, len(pvars) * (len(pvars) - 1) // 2))
    rows.append(make_row(ds, g, P, "pairs_excluded_by_cap", E_CAP,
                         V * (V - 1) // 2 - len(pvars) * (len(pvars) - 1) // 2))
    rows.append(make_row(ds, g, P, "variables_excluded_from_matrix_diagnostics", E_CAP, V - len(mvars)))
    if V < 2:
        rows.append(make_row(ds, {"group_id": "train_matrix"}, P, "multivariate_profile", E_CAP, None,
                             "NOT_RUN", "SINGLE_VARIABLE"))
        return rows

    for ia in range(len(pvars)):
        for ib in range(ia + 1, len(pvars)):
            i, j = pvars[ia], pvars[ib]
            rows.extend(_pair_rows(ds, [vids[i], vids[j]], tr[:, i], tr[:, j]))

    rows.extend(_matrix_rows(ds, [vids[i] for i in mvars], tr[:, mvars]))
    return rows


def _pair_rows(ds, pair, a, b):
    out = []
    key = {"pair": list(pair)}
    add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: out.append(
        make_row(ds, key, "train", metric, estimator, value, status, reason, cpu))

    t0 = time.process_time()
    m = np.isfinite(a) & np.isfinite(b)
    xa, xb = a[m], b[m]
    if xa.size < MIN_PAIR_N:
        for metric, estimator in (("pearson", E_PEARSON), ("spearman", E_SPEARMAN), ("biweight_midcorrelation", E_BICOR),
                                  ("mutual_information_bits", E_MI), ("mi_shuffle_mean_bits", E_MI_SM),
                                  ("mi_shuffle_q95_bits", E_MI_SQ), ("mi_excess_over_shuffle_bits", E_MI_EX)):
            add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_COMMON_SAMPLES")
    else:
        for metric, estimator, fn in (("pearson", E_PEARSON, pearson), ("spearman", E_SPEARMAN, spearman),
                                      ("biweight_midcorrelation", E_BICOR, biweight_midcorrelation)):
            t0 = time.process_time()
            v = fn(xa, xb)
            c = time.process_time() - t0
            if v is None:
                add(metric, estimator, None, "INCONCLUSIVE", "ZERO_SPREAD", cpu=c)
            else:
                add(metric, estimator, v, cpu=c)
        t0 = time.process_time()
        sa, sb = frozen_symbols(xa), frozen_symbols(xb)
        mi = mutual_information_bits(sa, sb, MI_BINS)
        rng = np.random.default_rng(MI_SEED)
        null = np.array([mutual_information_bits(sa, sb[rng.permutation(sb.size)], MI_BINS)
                         for _ in range(MI_SHUFFLES)])
        c = time.process_time() - t0
        add("mutual_information_bits", E_MI, mi, cpu=c)
        add("mi_shuffle_mean_bits", E_MI_SM, null.mean(), cpu=c)
        add("mi_shuffle_q95_bits", E_MI_SQ, np.quantile(null, 0.95), cpu=c)
        add("mi_excess_over_shuffle_bits", E_MI_EX, mi - null.mean(), cpu=c)

    # causal lead-lag
    t0 = time.process_time()
    ab, ba, lag, corr = best_signed_lag(a, b)
    c = time.process_time() - t0
    for k in range(MAX_LAG + 1):
        for metric, estimator, vals in ((f"causal_xcorr_a_past_b_present_lag_{k}", E_XAB, ab),
                                        (f"causal_xcorr_b_past_a_present_lag_{k}", E_XBA, ba)):
            if vals[k] is None:
                add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_COMMON_SAMPLES", cpu=c)
            else:
                add(metric, estimator, vals[k], cpu=c)
    if lag is None:
        for metric, estimator in (("best_signed_lag", E_BEST), ("best_signed_lag_correlation", E_BEST_C),
                                  ("best_lag_modal_fraction", E_LSTAB), ("best_lag_modal_value", E_LMODE),
                                  ("best_lag_stable", E_LSTABLE)):
            add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_COMMON_SAMPLES", cpu=c)
    else:
        add("best_signed_lag", E_BEST, lag, cpu=c)
        add("best_signed_lag_correlation", E_BEST_C, corr, cpu=c)
        t0 = time.process_time()
        w = a.size // LAG_SUBWINDOWS
        lags = []
        for s in range(LAG_SUBWINDOWS):
            _, _, lg, _ = best_signed_lag(a[s * w:(s + 1) * w], b[s * w:(s + 1) * w]) if w > MAX_LAG else (0, 0, None, 0)
            if lg is not None:
                lags.append(lg)
        c = time.process_time() - t0
        if len(lags) < 2:
            for metric, estimator in (("best_lag_modal_fraction", E_LSTAB), ("best_lag_modal_value", E_LMODE),
                                      ("best_lag_stable", E_LSTABLE)):
                add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_SUBWINDOWS", cpu=c)
        else:
            vals, cnts = np.unique(lags, return_counts=True)
            mode = int(vals[np.argmax(cnts)])
            frac = float(cnts.max()) / len(lags)
            add("best_lag_modal_fraction", E_LSTAB, frac, cpu=c)
            add("best_lag_modal_value", E_LMODE, mode, cpu=c)
            add("best_lag_stable", E_LSTABLE, 1.0 if frac >= LAG_STABLE_FRACTION else 0.0, cpu=c)

    # coherence with a phase-randomized null
    t0 = time.process_time()
    s, e = longest_common_finite_run(a, b)
    names = (("coherence_band_mean", E_COH), ("coherence_null_percentile", E_COH_P),
             ("coherence_null_q025", E_COH_LO), ("coherence_null_q975", E_COH_HI))
    if e - s < 4 * COH_NPERSEG:
        for metric, estimator in names:
            add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_CONTIGUOUS_COMMON_SAMPLES")
    elif np.all(a[s:e] == a[s]) or np.all(b[s:e] == b[s]):
        for metric, estimator in names:
            add(metric, estimator, None, "INCONCLUSIVE", "ZERO_VARIANCE")
    else:
        from scipy.signal import coherence
        ra, rb = a[s:e], b[s:e]
        f, C = coherence(ra, rb, fs=1.0, nperseg=COH_NPERSEG)
        obs = float(C[f > 0].mean())
        S = phase_randomized(rb, COH_SURROGATES, np.random.default_rng(COH_SEED))
        f2, C2 = coherence(ra[None, :], S, fs=1.0, nperseg=COH_NPERSEG, axis=-1)
        null = C2[:, f2 > 0].mean(axis=1)
        c = time.process_time() - t0
        add("coherence_band_mean", E_COH, obs, cpu=c)
        add("coherence_null_percentile", E_COH_P, 100.0 * float((null <= obs).mean()), cpu=c)
        add("coherence_null_q025", E_COH_LO, np.quantile(null, 0.025), cpu=c)
        add("coherence_null_q975", E_COH_HI, np.quantile(null, 0.975), cpu=c)
    return out


def _matrix_rows(ds, ids, M):
    out = []
    T, V = M.shape
    gm = {"group_id": "train_matrix"}
    t0 = time.process_time()
    R, _ = pairwise_pearson_matrix(M)
    w, U, neg = corr_eigen(R)
    c = time.process_time() - t0
    tot = w.sum()
    if tot <= 0:
        out.append(make_row(ds, gm, "train", "effective_rank", E_ERANK, None, "INCONCLUSIVE", "ZERO_SPECTRUM", cpu=c))
        return out
    for i in range(min(PCA_COMPONENTS_REPORTED, V)):
        out.append(make_row(ds, gm, "train", f"pca_explained_variance_ratio_pc{i + 1}", E_PCA, w[i] / tot, cpu=c))
    sv = np.sqrt(w)
    p = sv / sv.sum()
    p = p[p > 0]
    out.append(make_row(ds, gm, "train", "effective_rank", E_ERANK, math.exp(-(p * np.log(p)).sum()), cpu=c))
    out.append(make_row(ds, gm, "train", "negative_eigenvalue_count", E_NEG, neg, cpu=c))

    # common/private candidates
    u1 = U[:, 0] * (1.0 if U[:, 0].sum() >= 0 else -1.0)
    for i, vid in enumerate(ids):
        key = {"variable_id": vid}
        share = u1[i] ** 2 * w[0]
        out.append(make_row(ds, key, "train", "pc1_loading", E_PC1L, u1[i] * math.sqrt(w[0]),
                            reason="CANDIDATE_NOT_A_TRANSFORMATION", cpu=c))
        out.append(make_row(ds, key, "train", "pc1_common_variance_share", E_PC1S, share,
                            reason="CANDIDATE_NOT_A_TRANSFORMATION", cpu=c))
        out.append(make_row(ds, key, "train", "private_residual_variance_share", E_PRIV, 1.0 - share,
                            reason="CANDIDATE_NOT_A_TRANSFORMATION", cpu=c))

    # hierarchical clusters with moving-block bootstrap stability
    t0 = time.process_time()
    labels = cluster_labels(R)
    clusters = [np.flatnonzero(labels == l) for l in np.unique(labels)]
    clusters = [cl for cl in clusters if cl.size >= 2]
    rng = np.random.default_rng(CLUSTER_SEED)
    block = max(10, int(math.ceil(math.sqrt(T))))
    jacc = np.zeros(len(clusters))
    if clusters and T > block:
        for _ in range(CLUSTER_BOOTSTRAPS):
            idx = moving_block_indices(T, block, rng)
            Rb, _ = pairwise_pearson_matrix(M[idx])
            lb = cluster_labels(Rb)
            bsets = [set(np.flatnonzero(lb == l)) for l in np.unique(lb)]
            for ci, cl in enumerate(clusters):
                s = set(cl)
                jacc[ci] += max(len(s & bs) / len(s | bs) for bs in bsets)
        jacc /= CLUSTER_BOOTSTRAPS
    c = time.process_time() - t0
    identified = 0
    for ci, cl in enumerate(clusters):
        members = sorted(ids[i] for i in cl)
        gid = "train_cluster:" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        e_members = est(E_CL_STAB["name"], {**E_CL_STAB["params"], "members": members}, E_CL_STAB["assumptions"])
        if T <= block:
            out.append(make_row(ds, {"group_id": gid}, "train", "cluster_stability", e_members, None,
                                "INCONCLUSIVE", "INSUFFICIENT_SAMPLE_FOR_BOOTSTRAP", cpu=c))
        elif jacc[ci] >= CLUSTER_STABLE_JACCARD:
            identified += 1
            out.append(make_row(ds, {"group_id": gid}, "train", "cluster_stability", e_members, jacc[ci],
                                reason="GROUP_IDENTIFIED", cpu=c))
        else:
            out.append(make_row(ds, {"group_id": gid}, "train", "cluster_stability", e_members, None,
                                "INCONCLUSIVE", "GROUP_NOT_IDENTIFIED", cpu=c))
    out.append(make_row(ds, {"group_id": "train_clusters"}, "train", "identified_group_count", E_CL_COUNT,
                        identified, reason="" if identified else "GROUP_NOT_IDENTIFIED", cpu=c))
    return out
