#!/usr/bin/env python3
"""C131 (order 2026-09-12): operational information and compression descriptors, without targets.

Every quantity here is an operational descriptor or a bound under a declared
estimator, never a property of the data-generating process:

- discrete entropy (bits) under train-frozen quantile quantization;
- normalized permutation entropy (Bandt & Pompe 2002), orders 3, 4, 5, delay 1;
- spectral entropy in trailing windows (Welch inside each window), median and IQR;
- compressed length per sample with fixed compressors (zlib level 9, lzma
  preset 6) on train-frozen symbols and on raw float64 bytes; gain versus the
  uncompressed stream, and the temporal-structure gain versus a seeded shuffle;
- effective rank (Roy & Vetterli 2007) of the train matrix only;
- lag-1 conditional redundancy H(X_t) - H(X_t | X_{t-1}) under frozen quantization.

Compressed length is an upper-bound descriptor for the declared compressor
only. Bins are fitted on train and applied frozen; no statistic mixes partitions.

Entry point: run_information(contract, X, timestamps=None) -> list of rows.
"""
from __future__ import annotations

import contextlib
import hashlib
import lzma
import math
import time
import zlib
from pathlib import Path

import numpy as np

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARTITIONS = ("train", "calibration", "confirmation")
STATUSES = ("COMPLETED", "UNAVAILABLE", "INCONCLUSIVE", "FAILED", "NOT_RUN")

N_BINS = 16
MISSING_SYMBOL = N_BINS
PE_ORDERS = (3, 4, 5)
PE_DELAY = 1
PE_MIN_WINDOWS_PER_PATTERN = 10
SPEC_WINDOW = 256
SPEC_HOP = 128
WELCH_NPERSEG = 64
WELCH_NOVERLAP = 32
SPEC_MIN_WINDOWS = 3
ZLIB_LEVEL = 9
LZMA_PRESET = 6
SHUFFLE_SEED = 20260912
COND_MIN_COUNT = 20
ERANK_MIN_ROWS_PER_VARIABLE = 10
ERANK_MIN_ROWS = 50


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


# ----------------------------------------------------------- estimators
def frozen_edges(train_values):
    """Interior edges at train quantiles k/N_BINS (linear), ties collapsed."""
    f = train_values[np.isfinite(train_values)]
    return np.unique(np.quantile(f, np.arange(1, N_BINS) / N_BINS))


def quantize(x, edges):
    """Symbols 0..len(edges) for finite values, MISSING_SYMBOL otherwise."""
    fin = np.isfinite(x)
    sym = np.full(x.shape, MISSING_SYMBOL, dtype=np.uint8)
    sym[fin] = np.searchsorted(edges, x[fin], side="right")
    return sym


def entropy_bits(counts):
    c = counts[counts > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def permutation_entropy(x, order, delay=1):
    """Normalized permutation entropy over windows whose values are all finite.
    Ties are ranked by position (stable argsort). Returns (value, n_windows)."""
    span = (order - 1) * delay + 1
    if x.size < span:
        return None, 0
    w = np.lib.stride_tricks.sliding_window_view(x, span)[:, ::delay]
    w = w[np.all(np.isfinite(w), axis=1)]
    if w.shape[0] == 0:
        return None, 0
    perm = np.argsort(w, axis=1, kind="stable")
    codes = perm @ (order ** np.arange(order))
    counts = np.bincount(codes)
    h = entropy_bits(counts) * math.log(2)
    return h / math.log(math.factorial(order)), int(w.shape[0])


def spectral_entropy_windows(x):
    """Normalized spectral entropy per trailing window [t - W, t) fully inside the
    partition and fully finite with non-zero power; Welch inside the window."""
    from scipy.signal import welch
    if x.size < SPEC_WINDOW:
        return np.array([])
    w = np.lib.stride_tricks.sliding_window_view(x, SPEC_WINDOW)[::SPEC_HOP]
    w = w[np.all(np.isfinite(w), axis=1)]
    if w.shape[0] == 0:
        return np.array([])
    _, pxx = welch(w, fs=1.0, window="hann", nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP,
                   detrend="constant", axis=-1)
    pxx = pxx[:, 1:]
    tot = pxx.sum(axis=1)
    keep = tot > 0
    p = pxx[keep] / tot[keep, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return h / math.log(pxx.shape[1])


def effective_rank(matrix):
    """Roy & Vetterli (2007): exp(-sum p_i ln p_i), p_i = sigma_i / sum sigma.
    Returns (effective_rank, numerical_rank, singular_values)."""
    s = np.linalg.svd(matrix, compute_uv=False)
    if s.size == 0 or s.sum() <= 0:
        return None, 0, s
    p = s / s.sum()
    p = p[p > 0]
    tol = s.max() * max(matrix.shape) * np.finfo(float).eps
    return float(math.exp(-(p * np.log(p)).sum())), int((s > tol).sum()), s


def conditional_redundancy(sym):
    """Plug-in H(X_t), H(X_t | X_{t-1}) over consecutive finite symbol pairs.
    Returns (h_marginal, h_conditional, min_state_count) or None."""
    a, b = sym[:-1], sym[1:]
    ok = (a != MISSING_SYMBOL) & (b != MISSING_SYMBOL)
    a, b = a[ok].astype(np.int64), b[ok].astype(np.int64)
    if a.size == 0:
        return None
    k = MISSING_SYMBOL
    joint = np.bincount(a * k + b, minlength=k * k)
    state = np.bincount(a, minlength=k)
    h_b = entropy_bits(np.bincount(b, minlength=k))
    h_ab = entropy_bits(joint)
    h_a = entropy_bits(state)
    return h_b, h_ab - h_a, int(state[state > 0].min())


_Q_ASSUME = ["bins fitted on finite train values at quantiles k/16 and applied frozen to every partition",
             "tied edges collapse, so fewer than 16 symbols may exist"]
E_ENT = est("discrete_entropy_plugin", {"bins": N_BINS, "bin_rule": "train quantiles k/16", "unit": "bits"},
            ["plug-in (maximum likelihood) estimate, biased low", "finite values only"] + _Q_ASSUME)
E_PE = {m: est("permutation_entropy_normalized",
               {"order": m, "delay": PE_DELAY, "normalization": f"ln({m}!)",
                "min_windows": PE_MIN_WINDOWS_PER_PATTERN * math.factorial(m),
                "reference": "Bandt & Pompe (2002), Phys. Rev. Lett. 88:174102"},
               ["windows containing any non-finite value are skipped",
                "ties ranked by position (stable sort)", "windows never cross a partition boundary"])
        for m in PE_ORDERS}
_SE_P = {"window": SPEC_WINDOW, "hop": SPEC_HOP, "window_alignment": "trailing [t - window, t)",
         "welch": {"window": "hann", "nperseg": WELCH_NPERSEG, "noverlap": WELCH_NOVERLAP, "detrend": "constant"},
         "normalization": "ln(number of non-DC frequency bins)", "min_windows": SPEC_MIN_WINDOWS}
_SE_A = ["only windows fully inside the partition, fully finite and with non-zero power",
         "the DC bin is excluded", "fs is one cycle per sample"]
E_SE_MED = est("spectral_entropy_trailing_windows_median", _SE_P, _SE_A)
E_SE_IQR = est("spectral_entropy_trailing_windows_iqr", _SE_P, _SE_A)
E_SE_N = est("spectral_entropy_valid_window_count", _SE_P, _SE_A)
_COMP = {"zlib9": ("zlib", {"level": ZLIB_LEVEL}, lambda b: len(zlib.compress(b, ZLIB_LEVEL))),
         "lzma6": ("lzma", {"preset": LZMA_PRESET, "format": "FORMAT_XZ"},
                   lambda b: len(lzma.compress(b, preset=LZMA_PRESET)))}
_STREAMS = {"symbols": ["train-frozen symbols as one uint8 per sample; non-finite samples map to reserved symbol 16",
                        "temporal order kept"] + _Q_ASSUME,
            "raw_float64": ["raw little-endian float64 bytes of the partition, NaN bytes kept in place"]}
_C_ASSUME = ["compressed length is an operational upper-bound descriptor for this compressor only",
             "container headers are included in the length"]
E_BPS = {(c, s): est(f"compressed_bits_per_sample_{c}_{s}", {"compressor": _COMP[c][0], **_COMP[c][1], "stream": s},
                     _C_ASSUME + _STREAMS[s]) for c in _COMP for s in _STREAMS}
E_GAIN = {(c, s): est(f"compression_gain_vs_uncompressed_{c}_{s}",
                      {"compressor": _COMP[c][0], **_COMP[c][1], "stream": s,
                       "definition": "1 - compressed_bytes / uncompressed_bytes"}, _C_ASSUME + _STREAMS[s])
          for c in _COMP for s in _STREAMS}
E_TGAIN = {(c, s): est(f"temporal_structure_gain_{c}_{s}",
                       {"compressor": _COMP[c][0], **_COMP[c][1], "stream": s, "shuffle_seed": SHUFFLE_SEED,
                        "shuffle": "numpy.random.default_rng(seed).permutation(n), fresh generator per stream",
                        "definition": "(compressed(shuffled) - compressed(ordered)) / compressed(shuffled)"},
                       _C_ASSUME + _STREAMS[s] + ["the shuffle keeps the marginal and destroys temporal order; "
                                                  "the difference is the operational temporal-structure gain"])
           for c in _COMP for s in _STREAMS}
E_ERANK = est("effective_rank_roy_vetterli",
              {"definition": "exp(-sum_i p_i ln p_i), p_i = sigma_i / sum_j sigma_j",
               "reference": "Roy & Vetterli (2007), The effective rank: a measure of effective dimensionality, "
                            "EUSIPCO 2007, pp. 606-610",
               "matrix": "train rows finite in every variable, columns z-scored with train mean and std",
               "min_rows": f"max({ERANK_MIN_ROWS}, {ERANK_MIN_ROWS_PER_VARIABLE} * variables)"},
              ["train partition only", "zero-variance columns are dropped", "diagnostic only, never a transformation"])
E_NRANK = est("numerical_rank", {"tolerance": "sigma_max * max(rows, cols) * float64 eps",
                                 "matrix": E_ERANK["params"]["matrix"]}, E_ERANK["assumptions"])
_CR_P = {"bins": N_BINS, "bin_rule": "train quantiles k/16", "lag": 1, "unit": "bits",
         "min_count_per_conditioning_state": COND_MIN_COUNT}
_CR_A = ["plug-in entropies over consecutive pairs where both samples are finite",
         "every occurring conditioning state must have at least the minimum count, else INCONCLUSIVE"] + _Q_ASSUME
E_CRED = est("conditional_redundancy_lag1", {**_CR_P, "definition": "H(X_t) - H(X_t | X_{t-1})"}, _CR_A)
E_CSUR = est("conditional_surprisal_lag1", {**_CR_P, "definition": "H(X_t | X_{t-1})"}, _CR_A)


# ------------------------------------------------------------------- run
def run_information(contract, X, timestamps=None):
    with blas_single_thread():
        return _run_information(contract, X, timestamps)


def _run_information(contract, X, timestamps=None):
    X, parts, vids, timestamps = contract_layout(contract, X, timestamps)
    ds = contract["dataset_id"]
    rows = []
    tr_s, tr_e = parts[0][1], parts[0][2]

    for j, vid in enumerate(vids):
        key = {"variable_id": vid}
        xtr = X[tr_s:tr_e, j]
        have_bins = int(np.isfinite(xtr).sum()) >= N_BINS
        edges = frozen_edges(xtr) if have_bins else None
        for pname, s, e in parts:
            x = X[s:e, j]
            n = x.size
            add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: rows.append(
                make_row(ds, key, pname, metric, estimator, value, status, reason, cpu))

            # discrete entropy and conditional redundancy under frozen bins
            t0 = time.process_time()
            sym = quantize(x, edges) if have_bins else None
            if not have_bins:
                add("discrete_entropy_bits", E_ENT, None, "UNAVAILABLE", "FEWER_FINITE_TRAIN_VALUES_THAN_BINS")
                add("conditional_redundancy_bits_lag1", E_CRED, None, "UNAVAILABLE",
                    "FEWER_FINITE_TRAIN_VALUES_THAN_BINS")
                add("conditional_surprisal_bits_lag1", E_CSUR, None, "UNAVAILABLE",
                    "FEWER_FINITE_TRAIN_VALUES_THAN_BINS")
            else:
                fin_sym = sym[sym != MISSING_SYMBOL]
                if fin_sym.size == 0:
                    add("discrete_entropy_bits", E_ENT, None, "UNAVAILABLE", "NO_FINITE_VALUES")
                else:
                    add("discrete_entropy_bits", E_ENT, entropy_bits(np.bincount(fin_sym)),
                        cpu=time.process_time() - t0)
                t0 = time.process_time()
                cr = conditional_redundancy(sym)
                c = time.process_time() - t0
                if cr is None:
                    for metric, estimator in (("conditional_redundancy_bits_lag1", E_CRED),
                                              ("conditional_surprisal_bits_lag1", E_CSUR)):
                        add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
                elif cr[2] < COND_MIN_COUNT:
                    for metric, estimator in (("conditional_redundancy_bits_lag1", E_CRED),
                                              ("conditional_surprisal_bits_lag1", E_CSUR)):
                        add(metric, estimator, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
                else:
                    add("conditional_redundancy_bits_lag1", E_CRED, cr[0] - cr[1], cpu=c)
                    add("conditional_surprisal_bits_lag1", E_CSUR, cr[1], cpu=c)

            # permutation entropy
            for m in PE_ORDERS:
                t0 = time.process_time()
                val, nw = permutation_entropy(x, m, PE_DELAY)
                c = time.process_time() - t0
                need = PE_MIN_WINDOWS_PER_PATTERN * math.factorial(m)
                if val is None or nw < need:
                    add(f"permutation_entropy_order_{m}", E_PE[m], None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
                else:
                    add(f"permutation_entropy_order_{m}", E_PE[m], val, cpu=c)

            # spectral entropy in trailing windows
            t0 = time.process_time()
            h = spectral_entropy_windows(x)
            c = time.process_time() - t0
            add("spectral_entropy_window_count", E_SE_N, h.size, cpu=c)
            if h.size < SPEC_MIN_WINDOWS:
                add("spectral_entropy_median", E_SE_MED, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
                add("spectral_entropy_iqr", E_SE_IQR, None, "INCONCLUSIVE", "INSUFFICIENT_SAMPLE", cpu=c)
            else:
                q25, q50, q75 = np.quantile(h, [0.25, 0.5, 0.75])
                add("spectral_entropy_median", E_SE_MED, q50, cpu=c)
                add("spectral_entropy_iqr", E_SE_IQR, q75 - q25, cpu=c)

            # compression
            streams = {"raw_float64": np.ascontiguousarray(x, dtype="<f8")}
            if have_bins:
                streams["symbols"] = sym
            for sname in _STREAMS:
                for cname, (_, _, fn) in _COMP.items():
                    metric_sfx = f"{cname}_{sname}"
                    if sname not in streams:
                        for metric, estimator in ((f"compressed_bits_per_sample_{metric_sfx}", E_BPS),
                                                  (f"compression_gain_vs_uncompressed_{metric_sfx}", E_GAIN),
                                                  (f"temporal_structure_gain_{metric_sfx}", E_TGAIN)):
                            add(metric, estimator[(cname, sname)], None, "UNAVAILABLE",
                                "FEWER_FINITE_TRAIN_VALUES_THAN_BINS")
                        continue
                    arr = streams[sname]
                    t0 = time.process_time()
                    raw = arr.tobytes()
                    ordered = fn(raw)
                    c = time.process_time() - t0
                    add(f"compressed_bits_per_sample_{metric_sfx}", E_BPS[(cname, sname)], 8.0 * ordered / n, cpu=c)
                    add(f"compression_gain_vs_uncompressed_{metric_sfx}", E_GAIN[(cname, sname)],
                        1.0 - ordered / len(raw), cpu=c)
                    t0 = time.process_time()
                    perm = np.random.default_rng(SHUFFLE_SEED).permutation(n)
                    shuffled = fn(arr[perm].tobytes())
                    add(f"temporal_structure_gain_{metric_sfx}", E_TGAIN[(cname, sname)],
                        (shuffled - ordered) / shuffled, cpu=c + time.process_time() - t0)

    # effective rank of the train matrix only
    g = {"group_id": "train_matrix_all_variables"}
    t0 = time.process_time()
    V = len(vids)
    if V < 2:
        rows.append(make_row(ds, g, "train", "effective_rank", E_ERANK, None, "NOT_RUN", "SINGLE_VARIABLE"))
        rows.append(make_row(ds, g, "train", "numerical_rank", E_NRANK, None, "NOT_RUN", "SINGLE_VARIABLE"))
    else:
        M = X[tr_s:tr_e]
        M = M[np.all(np.isfinite(M), axis=1)]
        sd = M.std(axis=0) if M.shape[0] else np.zeros(V)
        M = M[:, sd > 0]
        need = max(ERANK_MIN_ROWS, ERANK_MIN_ROWS_PER_VARIABLE * V)
        if M.shape[0] < need or M.shape[1] < 2:
            why = "INSUFFICIENT_COMPLETE_ROWS" if M.shape[0] < need else "FEWER_THAN_TWO_NON_CONSTANT_VARIABLES"
            rows.append(make_row(ds, g, "train", "effective_rank", E_ERANK, None, "INCONCLUSIVE", why))
            rows.append(make_row(ds, g, "train", "numerical_rank", E_NRANK, None, "INCONCLUSIVE", why))
        else:
            Z = (M - M.mean(axis=0)) / M.std(axis=0)
            er, nr, _ = effective_rank(Z)
            c = time.process_time() - t0
            rows.append(make_row(ds, g, "train", "effective_rank", E_ERANK, er, cpu=c))
            rows.append(make_row(ds, g, "train", "numerical_rank", E_NRANK, nr, cpu=c))
    return rows
