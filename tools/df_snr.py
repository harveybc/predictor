#!/usr/bin/env python
"""C134: noise and SNR estimators, calibrated against known truth.

Every estimator is fitted on the TRAIN partition only and returns a noise
variance, a signal variance (var(x_train) - noise variance), an SNR in dB, a
moving-block bootstrap interval and a status.  Every estimate is conditional on
a named decomposition model and on the assumptions declared as data in
ESTIMATORS; nothing here is a model-free "SNR".

Missing data policy (no silent interpolation): an estimator only ever sees the
longest complete contiguous segment of the train partition (NaN or a
missing_mask entry breaks a segment).  If that segment is shorter than
MIN_SEGMENT_LENGTH the estimate is refused as NOT_IDENTIFIABLE.

Status values:
  ESTIMATED         all values present
  NOT_IDENTIFIABLE  values null; `reason` says why.  Reasons beginning with
                    "fit_failed" count as failures in the calibration table.

CLI (calibration against a known-truth bank, write-once):
  python tools/df_snr.py --bank DIR --out FILE [--limit N] [--bootstrap-b B]
writes FILE (schema crispdm.data_foundation.snr_calibration.v1) and
FILE-with-suffix ".olap_rows.jsonl" (OLAP-ready rows).  Both must not exist.
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL = "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"
TRAIN_SEGMENT_AGGREGATE = "TRAIN_SEGMENT_AGGREGATE"
# True only inside estimate_offline_train_diagnostic (C160).
_OFFLINE_AGGREGATE = contextvars.ContextVar("df_snr_offline_aggregate", default=False)


class SnrRefusal(Exception):
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


def _snapshot_module():
    name = "df_snapshot"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]

CALIBRATION_SCHEMA = "crispdm.data_foundation.snr_calibration.v1"
REAL_LABEL = "MODEL_CONDITIONAL_SNR_ESTIMATE"
MIN_SEGMENT_LENGTH = 64
MAD_TO_SIGMA = 0.6745  # median(|Z|) for Z ~ N(0, 1)
STATUS_OK = "ESTIMATED"
STATUS_NI = "NOT_IDENTIFIABLE"
DEFAULT_BOOTSTRAP = {"B": 200, "block_length": 50, "seed": 20260912,
                     "interval": "percentile", "alpha": 0.05}
MISSING_POLICY = ("longest_complete_contiguous_segment_of_train; NaN or "
                  "missing_mask breaks segments; no interpolation; refuse "
                  f"if segment length < {MIN_SEGMENT_LENGTH}")


def code_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class _FitFailed(Exception):
    pass


# --------------------------------------------------------------------------
# noise-variance kernels: 1-D finite array -> noise variance (float)
# --------------------------------------------------------------------------

def _nv_mad_first_difference(x: np.ndarray, p: dict) -> float:
    d = np.diff(x)
    sigma = np.median(np.abs(d)) / (MAD_TO_SIGMA * math.sqrt(2.0))
    return float(sigma ** 2)


def _offline_wavelet_mad_kernel(x: np.ndarray, p: dict) -> float:
    """PRIVATE. db4 with periodization over the whole segment: the last sample
    moves the first coefficients, so it is never causal. It runs only inside the
    aggregate path that returns one figure per training segment."""
    if not _OFFLINE_AGGREGATE.get():
        raise SnrRefusal("wavelet_mad is OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL; it runs only inside "
                         "estimate_offline_train_diagnostic and never yields per-timestamp values")
    import pywt
    _, detail = pywt.dwt(x, p["wavelet"], mode=p["mode"])
    sigma = np.median(np.abs(detail)) / MAD_TO_SIGMA
    return float(sigma ** 2)


def _ols_sigma2(x: np.ndarray, p_lag: int, start: int) -> tuple[float, int]:
    y = x[start:]
    cols = [np.ones_like(y)] + [x[start - k: len(x) - k] for k in range(1, p_lag + 1)]
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(np.mean(resid ** 2)), len(y)


def _nv_ar_residual(x: np.ndarray, p: dict) -> float:
    max_p = int(p["max_p"])
    if len(x) <= 3 * (max_p + 1):
        raise _FitFailed("fit_failed: series too short for declared max_p")
    best = None
    for lag in range(0, max_p + 1):  # common sample start=max_p for AIC comparability
        s2, n = _ols_sigma2(x, lag, max_p)
        if not np.isfinite(s2) or s2 <= 0:
            continue
        aic = n * math.log(s2) + 2 * (lag + 1)
        if best is None or aic < best[0]:
            best = (aic, lag)
    if best is None:
        raise _FitFailed("fit_failed: no AR order produced a positive innovation variance")
    s2, _ = _ols_sigma2(x, best[1], best[1])  # refit selected order on its own sample
    p["_selected_p"] = best[1]
    return s2


def _nv_spectral_floor(x: np.ndarray, p: dict) -> float:
    from scipy import signal, stats
    nperseg = int(min(p["nperseg"], len(x)))
    freqs, psd = signal.welch(x, fs=1.0, nperseg=nperseg, detrend="constant",
                              scaling="density", return_onesided=True)
    nyq = 0.5
    lo, hi = p["upper_band_fraction_of_nyquist"]
    band = (freqs >= lo * nyq) & (freqs < hi * nyq)  # excludes the undoubled Nyquist bin
    if band.sum() < 3:
        raise _FitFailed("fit_failed: fewer than 3 PSD bins in the declared upper band")
    med = float(np.median(psd[band]))
    if p["median_bias_correction"]:
        n_seg = max(1, (len(x) - nperseg) // (nperseg // 2) + 1)
        dof = 2 * n_seg
        med = med / (stats.chi2.median(dof) / dof)
    # one-sided density of white noise with variance s2 is 2*s2 over [0, 0.5]
    return med * nyq


def _nv_local_level_kalman(x: np.ndarray, p: dict) -> float:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = UnobservedComponents(x, "llevel").fit(disp=False, maxiter=int(p["maxiter"]))
        except Exception as exc:  # noqa: BLE001 - any MLE failure is a failed fit
            raise _FitFailed(f"fit_failed: {type(exc).__name__}: {exc}") from exc
    if not res.mle_retvals.get("converged", False):
        raise _FitFailed("fit_failed: MLE did not converge")
    names = list(res.model.param_names)
    val = float(np.asarray(res.params)[names.index("sigma2.irregular")])
    if not np.isfinite(val):
        raise _FitFailed("fit_failed: non-finite irregular variance")
    return val


def _nv_trailing_median_residual(x: np.ndarray, p: dict) -> float:
    w = int(p["window"])
    if len(x) < w + 1:
        raise _FitFailed("fit_failed: series shorter than window")
    from numpy.lib.stride_tricks import sliding_window_view
    med = np.median(sliding_window_view(x, w), axis=1)  # med[i] uses x[i..i+w-1], causal at t=i+w-1
    r = x[w - 1:] - med
    sigma = np.median(np.abs(r - np.median(r))) / MAD_TO_SIGMA
    return float(sigma ** 2)


ESTIMATORS: dict[str, dict[str, Any]] = {
    "mad_first_difference": {
        "name": "mad_first_difference",
        "parameters": {},
        "decomposition_model": "x_t = s_t + e_t, s smooth (|s_t - s_{t-1}| negligible), e iid Gaussian",
        "formula": "sigma = median|diff(x)| / (0.6745*sqrt(2))",
        "assumptions": ["white (serially uncorrelated) noise", "Gaussian noise for the MAD constant",
                        "signal smooth at the sampling scale"],
        "known_biases": ["colored noise with positive autocorrelation -> noise variance biased LOW, SNR HIGH",
                         "rough/jumpy signal -> noise variance biased HIGH"],
        "contract_state": TRAIN_SEGMENT_AGGREGATE,
        "bootstrap_B": None,
    },
    "wavelet_mad": {
        "name": "wavelet_mad",
        "parameters": {"wavelet": "db4", "level": 1, "mode": "periodization"},
        "decomposition_model": "x_t = s_t + e_t, s sparse in db4 wavelet basis, e iid Gaussian (Donoho-Johnstone)",
        "formula": "sigma = median|d1| / 0.6745, d1 = finest db4 detail coefficients",
        "assumptions": ["white Gaussian noise", "finest-scale detail dominated by noise",
                        "analysis-only offline estimator (uses the whole train segment, not causal)"],
        "known_biases": ["colored noise -> finest-scale energy misrepresents total noise variance"],
        "contract_state": OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL,
        "bootstrap_B": None,
    },
    "ar_residual": {
        "name": "ar_residual",
        "parameters": {"max_p": 20, "order_selection": "AIC on common sample", "trend": "constant",
                       "fit": "conditional OLS"},
        "decomposition_model": "AR-predictable signal plus white innovation: x_t = c + sum_k a_k x_{t-k} + u_t",
        "formula": "noise variance = innovation variance of AR(p*), p* = argmin AIC, p <= max_p",
        "assumptions": ["MODEL-CONDITIONAL: 'noise' is whatever the selected AR model cannot predict one step ahead",
                        "colored measurement noise is partly counted as signal",
                        "white measurement noise on an AR signal is NOT separated from signal innovations"],
        "known_biases": ["p*=0 gives noise variance = var(x) -> NOT_IDENTIFIABLE"],
        "contract_state": TRAIN_SEGMENT_AGGREGATE,
        "bootstrap_B": None,
    },
    "spectral_floor": {
        "name": "spectral_floor",
        "parameters": {"method": "welch", "nperseg": 256, "overlap": 0.5, "detrend": "constant",
                       "upper_band_fraction_of_nyquist": [0.7, 1.0], "median_bias_correction": True},
        "decomposition_model": "x = s + e, s band-limited below the upper band, e white with flat PSD",
        "formula": "noise variance = median(PSD in upper band) / chi2_median_correction * 0.5 (full one-sided band)",
        "assumptions": ["white noise (flat spectrum)", "no signal power in the declared upper band",
                        "chi2 median correction uses 2*n_segments dof, ignoring overlap correlation"],
        "known_biases": ["signal energy in upper band -> HIGH", "red noise -> LOW"],
        "contract_state": TRAIN_SEGMENT_AGGREGATE,
        "bootstrap_B": None,
    },
    "local_level_kalman": {
        "name": "local_level_kalman",
        "parameters": {"model": "statsmodels UnobservedComponents('llevel')", "fit": "MLE", "maxiter": 200},
        "decomposition_model": "local level: x_t = mu_t + eps_t, mu_t = mu_{t-1} + eta_t",
        "formula": "noise variance = sigma2.irregular",
        "assumptions": ["signal is a random walk", "irregular is white Gaussian",
                        "non-convergence is reported as fit_failed"],
        "known_biases": ["deterministic smooth signals are approximated by a random walk"],
        "contract_state": TRAIN_SEGMENT_AGGREGATE,
        "bootstrap_B": 50,
    },
    "trailing_median_residual": {
        "name": "trailing_median_residual",
        "parameters": {"window": 9, "window_includes_current_sample": True, "causal": True},
        "decomposition_model": "x_t = m_t + e_t, m_t approximated by the causal trailing median of the last 9 samples",
        "formula": "sigma = MAD(x_t - median(x_{t-8..t})) / 0.6745",
        "assumptions": ["white Gaussian noise", "signal locally constant over 9 samples",
                        "causal (uses only past and current samples)"],
        "known_biases": ["current sample is inside its own window -> residual shrunk, noise variance LOW "
                         "(no finite-sample correction applied)", "trending signal -> HIGH"],
        "contract_state": TRAIN_SEGMENT_AGGREGATE,
        "bootstrap_B": None,
    },
}


# Private registry: kernels are not reachable through ESTIMATORS (C160).
_KERNELS: dict[str, Callable] = {
    "mad_first_difference": _nv_mad_first_difference,
    "wavelet_mad": _offline_wavelet_mad_kernel,
    "ar_residual": _nv_ar_residual,
    "spectral_floor": _nv_spectral_floor,
    "local_level_kalman": _nv_local_level_kalman,
    "trailing_median_residual": _nv_trailing_median_residual,
}


def estimator_declarations(bootstrap: dict | None = None) -> dict[str, dict]:
    bs = dict(DEFAULT_BOOTSTRAP, **(bootstrap or {}))
    out = {}
    for name, spec in ESTIMATORS.items():
        d = {k: v for k, v in spec.items() if k != "bootstrap_B"}
        b = dict(bs)
        if spec["bootstrap_B"] is not None and bootstrap is None:
            b["B"] = spec["bootstrap_B"]
        d["bootstrap"] = {"scheme": "moving_block", **b}
        d["fitted_on"] = "train partition only"
        d["missing_policy"] = MISSING_POLICY
        out[name] = d
    return out


# --------------------------------------------------------------------------
# core estimation
# --------------------------------------------------------------------------

def longest_complete_segment(x: np.ndarray, missing: np.ndarray | None = None) -> tuple[int, int]:
    """Return [start, end) of the longest run where x is finite and not masked."""
    ok = np.isfinite(x)
    if missing is not None:
        ok &= ~np.asarray(missing, dtype=bool)
    best = (0, 0)
    start = None
    for i, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _point(x: np.ndarray, name: str) -> dict:
    """Point estimate on a complete 1-D array.  Never raises."""
    spec = ESTIMATORS[name]
    params = dict(spec["parameters"])
    res = {"noise_variance": None, "signal_variance": None, "snr_db": None,
           "status": STATUS_NI, "reason": None}
    if len(x) < MIN_SEGMENT_LENGTH:
        res["reason"] = f"insufficient_complete_segment: length {len(x)} < {MIN_SEGMENT_LENGTH}"
        return res
    total_var = float(np.var(x))
    if not np.isfinite(total_var) or total_var <= 0:
        res["reason"] = "total_variance_nonpositive"
        return res
    try:
        with np.errstate(all="ignore"):
            nv = _KERNELS[name](x, params)
    except SnrRefusal:
        raise
    except _FitFailed as exc:
        res["reason"] = str(exc)
        return res
    except Exception as exc:  # noqa: BLE001
        res["reason"] = f"fit_failed: {type(exc).__name__}: {exc}"
        return res
    if "_selected_p" in params:
        res["selected_ar_order"] = params["_selected_p"]
    if not np.isfinite(nv):
        res["reason"] = "fit_failed: non-finite noise variance"
        return res
    sv = total_var - nv
    floor = total_var * 1e-12
    if nv <= floor:
        res["reason"] = f"noise_variance_at_or_below_numerical_floor ({nv:.3e} <= {floor:.3e})"
        return res
    if sv <= 0:
        res["reason"] = f"signal_variance_nonpositive ({sv:.6g})"
        res["noise_variance_unclipped"] = nv
        return res
    res.update(noise_variance=float(nv), signal_variance=float(sv),
               snr_db=float(10.0 * math.log10(sv / nv)), status=STATUS_OK)
    return res


def _mbb_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def _bootstrap(x: np.ndarray, name: str, bs: dict) -> dict:
    rng = np.random.default_rng(int(bs["seed"]))
    snrs = []
    n_ni = 0
    for _ in range(int(bs["B"])):
        xb = x[_mbb_indices(len(x), int(bs["block_length"]), rng)]
        r = _point(xb, name)
        if r["status"] == STATUS_OK:
            snrs.append(r["snr_db"])
        elif r["reason"] and r["reason"].startswith("signal_variance_nonpositive"):
            snrs.append(-math.inf)  # SNR below any finite value
            n_ni += 1
        elif r["reason"] and r["reason"].startswith("noise_variance_at_or_below"):
            snrs.append(math.inf)
            n_ni += 1
        else:
            n_ni += 1
    out = {"B": int(bs["B"]), "block_length": int(bs["block_length"]), "seed": int(bs["seed"]),
           "alpha": bs["alpha"], "n_replicates_used": len(snrs),
           "n_replicates_not_identifiable": n_ni, "ci_low_db": None, "ci_high_db": None,
           "lower_unbounded": False, "upper_unbounded": False}
    if len(snrs) < max(10, int(bs["B"]) // 2):
        out["reason"] = "too_few_usable_bootstrap_replicates"
        return out
    a = np.asarray(snrs)
    lo = float(np.quantile(a, bs["alpha"] / 2, method="lower"))
    hi = float(np.quantile(a, 1 - bs["alpha"] / 2, method="higher"))
    out["lower_unbounded"] = lo == -math.inf
    out["upper_unbounded"] = hi == math.inf
    out["ci_low_db"] = None if not np.isfinite(lo) else lo
    out["ci_high_db"] = None if not np.isfinite(hi) else hi
    return out


def estimate(x_full: np.ndarray, train: tuple[int, int] | list, name: str,
             missing_mask: np.ndarray | None = None, bootstrap: dict | None = None,
             do_bootstrap: bool = True) -> dict:
    """Estimate on x_full[train[0]:train[1]] only.  `bootstrap` overrides the declared config.
    An OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL estimator refuses here: it is reachable only
    through estimate_offline_train_diagnostic."""
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}")
    if ESTIMATORS[name]["contract_state"] == OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL:
        raise SnrRefusal(f"{name} is OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL: a bare array never grants it; use "
                         "estimate_offline_train_diagnostic with a TRAIN FitSnapshot of the whole train partition")
    s, e = int(train[0]), int(train[1])
    xt = np.asarray(x_full, dtype=float)[s:e]
    mt = None if missing_mask is None else np.asarray(missing_mask, dtype=bool)[s:e]
    return _estimate_segment(xt, mt, s, e, name, bootstrap, do_bootstrap)


def estimate_offline_train_diagnostic(snapshot, column: int, name: str = "wavelet_mad",
                                      missing_mask: np.ndarray | None = None, bootstrap: dict | None = None,
                                      do_bootstrap: bool = True) -> dict:
    """The only path to an OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL estimator: ONE figure for one column
    of a verified TRAIN FitSnapshot covering the whole train partition of its contract. `missing_mask`
    has the length of the train partition."""
    if name not in ESTIMATORS or ESTIMATORS[name]["contract_state"] != OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL:
        raise SnrRefusal(f"{name!r} is not an offline train diagnostic")
    snap = _snapshot_module()
    if not isinstance(snapshot, snap.FitSnapshot):
        raise SnrRefusal("an offline train diagnostic requires a FitSnapshot built from a contract")
    try:
        snap.verify_fit_snapshot(snapshot)
    except snap.SnapshotRefusal as exc:
        raise SnrRefusal(f"fit snapshot does not verify: {exc}") from exc
    lo, hi = snap._partition_ranges(snapshot.contract)["TRAIN"]
    if snapshot.role != "TRAIN" or (snapshot.start, snapshot.end) != (lo, hi):
        raise SnrRefusal(f"one figure per training segment: the snapshot [{snapshot.start}, {snapshot.end}) "
                         f"with role {snapshot.role} is not the whole TRAIN partition [{lo}, {hi})")
    if type(column) is not int or not 0 <= column < snapshot.matrix.shape[1]:
        raise SnrRefusal(f"column {column!r} is not a column of the snapshot")
    xt = np.array(snapshot.matrix[:, column], dtype=float)
    mt = None if missing_mask is None else np.asarray(missing_mask, dtype=bool)
    if mt is not None and mt.shape != xt.shape:
        raise SnrRefusal("missing_mask must have the length of the train partition")
    token = _OFFLINE_AGGREGATE.set(True)
    try:
        return _estimate_segment(xt, mt, lo, hi, name, bootstrap, do_bootstrap)
    finally:
        _OFFLINE_AGGREGATE.reset(token)


def calibration_unit_contract(unit_dir: str, meta: dict, n_variables: int) -> tuple[dict, Callable]:
    """A sealed contract over a calibration unit's on-disk observed bytes, with its layout's
    train / calibration (or validation) / confirmation boundaries, and a bytes loader."""
    snap = _snapshot_module()
    C = snap.C
    parts = meta.get("partitions") or meta.get("temporal_roles")
    n = int(meta["n_samples"])
    second = "calibration" if "calibration" in parts else "validation"
    bounds = {"train": [int(v) for v in parts["train"]], "calibration": [int(v) for v in parts[second]],
              "confirmation": [int(v) for v in parts["confirmation"]]}
    name = "observed_signal.npy"
    blob = Path(unit_dir, name).read_bytes()
    unit_id = str(meta.get("unit_id") or os.path.basename(os.path.normpath(unit_dir)))
    dataset_id = f"snr_calibration_unit.{unit_id}"
    ev = [{"source": "calibration unit record", "sha256": "UNAVAILABLE"}]
    variables = [C.variable(dataset_id, f"v{i}", semantics={"type": "CALIBRATION_UNIT_OBSERVED",
                                                           "description": "observed signal", "evidence": ev},
                            available_time_rule="SAMPLE_INDEX", role="INPUT_CANDIDATE",
                            license_state="NOT_APPLICABLE_GENERATED", original_fields={"variable_index": i})
                 for i in range(n_variables)]
    na = "NOT_APPLICABLE"
    lengths = {k: b[1] - b[0] for k, b in bounds.items()}
    doc = {"schema": C.DATASET_SCHEMA, "dataset_id": dataset_id, "version": "calibration_unit.v1",
           "bank": "SYNTHETIC",
           "files": [{"name": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                      "role": "OBSERVED"}],
           "content_sha256": "", "contract_sha256": "",
           "source": {"provider": "calibration unit", "official_url": na, "citation": na, "doi": na,
                      "upstream_owner": na},
           "license": {"state": "NOT_APPLICABLE_GENERATED", "id": na, "url": na, "text_sha256": "UNAVAILABLE",
                       "attribution_required": "NO", "redistribution": na, "derivatives": na, "evidence": []},
           "time": {"frequency_nominal_seconds": na, "timezone": na, "timestamp_meaning": "SAMPLE_INDEX",
                    "range_start": "0", "range_end": str(n - 1), "availability_rule": "SAMPLE_INDEX",
                    "availability_delay_seconds": na},
           "panel": {"aligned_common_grid": True, "n_series": n_variables, "alignment_rule": "unit grid"},
           "partitions": {"scheme": "UNIT_DECLARED_BOUNDARIES",
                          "fractions": {k: lengths[k] / n if n else 0.0 for k in bounds},
                          "boundaries": bounds, "sealed_periods_excluded": [], "frozen_before_profile": True},
           "dependence": [], "variables": variables, "original_fields": {"unit_record": meta}}
    try:
        contract = C.seal(doc)
    except C.ContractRefusal as exc:
        raise SnrRefusal(f"calibration unit does not seal as a contract: {exc}") from exc
    return contract, (lambda requested: Path(unit_dir, requested).read_bytes() if requested == name else None)


def _estimate_segment(xt: np.ndarray, mt, s: int, e: int, name: str, bootstrap, do_bootstrap) -> dict:
    a, b = longest_complete_segment(xt, mt)
    n_missing = int((~np.isfinite(xt)).sum() + (0 if mt is None else (mt & np.isfinite(xt)).sum()))
    seg = xt[a:b]
    res = _point(seg, name)
    res["estimator"] = name
    res["train"] = [s, e]
    res["segment_used"] = [s + a, s + b]
    res["n_missing_in_train"] = n_missing
    bs = estimator_declarations(bootstrap)[name]["bootstrap"]
    if res["status"] == STATUS_OK and do_bootstrap and bs["B"] > 0:
        res["bootstrap"] = _bootstrap(seg, name, bs)
    else:
        res["bootstrap"] = None
    return res


def estimate_real(x: np.ndarray, train: tuple[int, int] | list, variable_names: list[str] | None = None,
                  estimators: list[str] | None = None, bootstrap: dict | None = None,
                  train_snapshot=None) -> list[dict]:
    """Real-data estimates.  x is (T,) or (V, T).  Rows are always labelled model-conditional.
    An offline train diagnostic is computed only from `train_snapshot` (a TRAIN FitSnapshot whose
    column v is variable v); without one its rows are NOT_IDENTIFIABLE with a refusal reason."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    names = estimators or list(ESTIMATORS)
    decl = estimator_declarations(bootstrap)
    sha = code_sha256()
    rows = []
    for v in range(arr.shape[0]):
        for name in names:
            if ESTIMATORS[name]["contract_state"] == OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL:
                if train_snapshot is None:
                    r = {"snr_db": None, "noise_variance": None, "signal_variance": None,
                         "status": STATUS_NI, "train": [int(train[0]), int(train[1])], "segment_used": None,
                         "bootstrap": None,
                         "reason": "refused: offline train diagnostic requires a TRAIN FitSnapshot"}
                else:
                    r = estimate_offline_train_diagnostic(train_snapshot, v, name, bootstrap=bootstrap)
            else:
                r = estimate(arr[v], train, name, bootstrap=bootstrap)
            bsr = r.get("bootstrap") or {}
            rows.append({
                "label": REAL_LABEL,
                "variable_index": v,
                "variable_name": None if variable_names is None else variable_names[v],
                "estimator": name,
                "decomposition_model": decl[name]["decomposition_model"],
                "assumptions": decl[name]["assumptions"],
                "known_biases": decl[name]["known_biases"],
                "model_conditional_snr_db": r["snr_db"],
                "noise_variance": r["noise_variance"],
                "signal_variance": r["signal_variance"],
                "ci_low_db": bsr.get("ci_low_db"), "ci_high_db": bsr.get("ci_high_db"),
                "lower_unbounded": bsr.get("lower_unbounded"),
                "status": r["status"], "reason": r["reason"],
                "train": r["train"], "segment_used": r["segment_used"],
                "bootstrap": decl[name]["bootstrap"],
                "code_sha256": sha,
            })
    return rows


# --------------------------------------------------------------------------
# calibration against a known-truth bank
# --------------------------------------------------------------------------

def _orient(arr: np.ndarray, n_samples: int, what: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != n_samples:
            raise ValueError(f"{what}: length {arr.shape[0]} != n_samples {n_samples}")
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{what}: expected 2-D array, got shape {arr.shape}")
    r, c = arr.shape
    if c == n_samples and r != n_samples:
        return arr
    if r == n_samples and c != n_samples:
        return arr.T
    if r == n_samples and c == n_samples:
        raise ValueError(f"{what}: orientation ambiguous, shape {arr.shape} with n_samples {n_samples}")
    raise ValueError(f"{what}: shape {arr.shape} does not contain n_samples {n_samples}")


def load_unit(unit_dir: str) -> dict:
    with open(os.path.join(unit_dir, "UNIT.json")) as fh:
        meta = json.load(fh)
    parts = meta.get("partitions") or meta.get("temporal_roles")
    if not parts or "train" not in parts:
        raise ValueError(f"{unit_dir}: no partitions/temporal_roles.train")
    n = int(meta["n_samples"])
    arrays = {}
    for k in ("clean_signal", "additive_noise", "observed_signal"):
        arrays[k] = _orient(np.load(os.path.join(unit_dir, k + ".npy")), n, k).astype(float)
    mm_path = os.path.join(unit_dir, "missing_mask.npy")
    mask = _orient(np.load(mm_path), n, "missing_mask").astype(bool) if os.path.exists(mm_path) else None
    return {"unit_id": meta.get("unit_id") or os.path.basename(os.path.normpath(unit_dir)),
            "meta": meta, "train": [int(parts["train"][0]), int(parts["train"][1])],
            "layout": "partitions" if meta.get("partitions") else "temporal_roles",
            "missing_mask": mask, **arrays}


def discover_units(bank: str) -> list[str]:
    return sorted(os.path.join(bank, d) for d in os.listdir(bank)
                  if os.path.isfile(os.path.join(bank, d, "UNIT.json")))


def _declared_for_var(declared: Any, v: int) -> Any:
    if isinstance(declared, list):
        return declared[v] if v < len(declared) else None
    return declared


def _truth(clean: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
    nv = float(np.var(noise))
    sv = float(np.var(clean))
    if nv <= 0:
        snr = math.inf
    elif sv <= 0:
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(sv / nv)
    return nv, snr


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def calibrate(bank: str, limit: int | None = None, estimators: list[str] | None = None,
              bootstrap: dict | None = None) -> tuple[dict, list[dict]]:
    names = estimators or list(ESTIMATORS)
    sha = code_sha256()
    unit_dirs = discover_units(bank)
    if limit is not None:
        unit_dirs = unit_dirs[:int(limit)]
    records, rows = [], []
    for ud in unit_dirs:
        u = load_unit(ud)
        meta = u["meta"]
        s, e = u["train"]
        train_snapshot = None
        if any(ESTIMATORS[nm]["contract_state"] == OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL for nm in names):
            snap = _snapshot_module()
            contract, loader = calibration_unit_contract(ud, meta, u["observed_signal"].shape[0])
            train_snapshot = snap.FitSnapshot.from_contract(contract, "TRAIN", loader)
        for v in range(u["observed_signal"].shape[0]):
            obs = u["observed_signal"][v]
            mask = None if u["missing_mask"] is None else u["missing_mask"][v]
            mt = None if mask is None else mask[s:e]
            a, b = longest_complete_segment(obs[s:e], mt)
            # truth on the same complete train segment the estimators see
            t_nv, t_snr = _truth(u["clean_signal"][v][s + a:s + b], u["additive_noise"][v][s + a:s + b])
            group = {"family": meta.get("family"), "perturbation": meta.get("perturbation"),
                     "declared_snr_db": _declared_for_var(meta.get("declared_snr_db"), v),
                     "length": int(meta["n_samples"])}
            for name in names:
                if ESTIMATORS[name]["contract_state"] == OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL:
                    r = estimate_offline_train_diagnostic(train_snapshot, v, name, missing_mask=mt,
                                                          bootstrap=bootstrap)
                else:
                    r = estimate(obs, u["train"], name, missing_mask=mask, bootstrap=bootstrap)
                bsr = r["bootstrap"] or {}
                log_err = snr_err = covers = None
                if r["status"] == STATUS_OK:
                    if t_nv > 0:
                        log_err = math.log(r["noise_variance"]) - math.log(t_nv)
                    if math.isfinite(t_snr):
                        snr_err = r["snr_db"] - t_snr
                    if bsr.get("n_replicates_used"):
                        lo = -math.inf if bsr["lower_unbounded"] else bsr["ci_low_db"]
                        hi = math.inf if bsr["upper_unbounded"] else bsr["ci_high_db"]
                        if lo is not None and hi is not None:
                            covers = bool(lo <= t_snr <= hi)
                rec = {"unit_id": u["unit_id"], "variable_index": v, "estimator": name, **group,
                       "status": r["status"], "reason": r["reason"],
                       "log_noise_var_error": log_err, "snr_db_error": snr_err, "ci_covers_true": covers}
                records.append(rec)
                base = {"unit_id": u["unit_id"], "variable_index": v, "estimator": name,
                        "status": r["status"], "reason": r["reason"], "code_sha256": sha}
                for metric, value in (("noise_variance", r["noise_variance"]),
                                      ("signal_variance", r["signal_variance"]),
                                      ("snr_db", r["snr_db"]),
                                      ("snr_db_ci_low", bsr.get("ci_low_db")),
                                      ("snr_db_ci_high", bsr.get("ci_high_db")),
                                      ("true_noise_variance", t_nv),
                                      ("true_snr_db", t_snr if math.isfinite(t_snr) else None),
                                      ("log_noise_var_error", log_err),
                                      ("snr_db_error", snr_err),
                                      ("ci_covers_true", None if covers is None else float(covers))):
                    rows.append({**base, "metric": metric, "value": value})
    table = _group_table(records)
    doc = {
        "schema": CALIBRATION_SCHEMA,
        "code_sha256": sha,
        "bank": os.path.abspath(bank),
        "unit_count": len(unit_dirs),
        "record_count": len(records),
        "estimators": estimator_declarations(bootstrap),
        "truth_definition": {
            "true_noise_variance": "var(additive_noise) over the complete train segment used by the estimators (ddof=0)",
            "true_snr_db": "10*log10(var(clean_signal)/var(additive_noise)) over the same segment",
            "log": "natural log",
        },
        "group_keys": ["estimator", "family", "perturbation", "declared_snr_db", "length"],
        "grouped_table": table,
        "least_biased_per_perturbation": {
            "label": "DESCRIPTIVE_ONLY_NOT_A_SELECTION",
            "note": "estimator with the smallest |mean SNR dB error| pooled over units of each perturbation; "
                    "not used to select anything anywhere",
            "by_perturbation": _least_biased(records),
        },
    }
    return doc, rows


def _stats(recs: list[dict]) -> dict:
    n = len(recs)
    le = np.array([r["log_noise_var_error"] for r in recs if _finite(r["log_noise_var_error"])])
    se = np.array([r["snr_db_error"] for r in recs if _finite(r["snr_db_error"])])
    cov = [r["ci_covers_true"] for r in recs if r["ci_covers_true"] is not None]
    ni = sum(r["status"] == STATUS_NI for r in recs)
    fail = sum(bool(r["reason"]) and r["reason"].startswith("fit_failed") for r in recs)
    return {
        "n": n,
        "n_log_noise_var": int(le.size),
        "bias_log_noise_var": float(le.mean()) if le.size else None,
        "rmse_log_noise_var": float(np.sqrt((le ** 2).mean())) if le.size else None,
        "n_snr_db": int(se.size),
        "bias_snr_db": float(se.mean()) if se.size else None,
        "rmse_snr_db": float(np.sqrt((se ** 2).mean())) if se.size else None,
        "n_coverage": len(cov),
        "ci_coverage_true_snr": float(np.mean(cov)) if cov else None,
        "not_identifiable_rate": ni / n if n else None,
        "failure_rate": fail / n if n else None,
    }


def _group_table(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["estimator"], r["family"], r["perturbation"], json.dumps(r["declared_snr_db"]), r["length"])
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups):
        est, fam, pert, dec, length = key
        out.append({"estimator": est, "family": fam, "perturbation": pert,
                    "declared_snr_db": json.loads(dec), "length": length, **_stats(groups[key])})
    return out


def _least_biased(records: list[dict]) -> dict:
    by: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by.setdefault(str(r["perturbation"]), {}).setdefault(r["estimator"], []).append(r)
    out = {}
    for pert, ests in sorted(by.items()):
        cand = []
        for est, recs in ests.items():
            st = _stats(recs)
            if st["bias_snr_db"] is not None:
                cand.append((abs(st["bias_snr_db"]), est, st))
        if cand:
            cand.sort()
            out[pert] = {"estimator": cand[0][1], "abs_bias_snr_db": cand[0][0],
                         "all": {c[1]: {"bias_snr_db": c[2]["bias_snr_db"], "n_snr_db": c[2]["n_snr_db"],
                                        "not_identifiable_rate": c[2]["not_identifiable_rate"]}
                                 for c in cand}}
        else:
            out[pert] = {"estimator": None, "reason": "no identifiable estimates with finite truth"}
    return out


def rows_path_for(out: str) -> str:
    root, _ = os.path.splitext(out)
    return root + ".olap_rows.jsonl"


def write_calibration(doc: dict, rows: list[dict], out: str) -> tuple[str, str]:
    rows_path = rows_path_for(out)
    for p in (out, rows_path):
        if os.path.exists(p):
            raise FileExistsError(f"write-once: {p} already exists")
    rows_blob = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode()
    doc = dict(doc, olap_rows={"path": os.path.abspath(rows_path), "count": len(rows),
                               "sha256": hashlib.sha256(rows_blob).hexdigest()})
    with open(rows_path, "xb") as fh:
        fh.write(rows_blob)
    with open(out, "x") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, allow_nan=False)
    return out, rows_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bootstrap-b", type=int, default=None,
                    help="override declared bootstrap B for all estimators (smoke runs)")
    args = ap.parse_args(argv)
    for p in (args.out, rows_path_for(args.out)):
        if os.path.exists(p):
            print(f"write-once: {p} already exists", file=sys.stderr)
            return 2
    bs = None if args.bootstrap_b is None else dict(DEFAULT_BOOTSTRAP, B=args.bootstrap_b)
    doc, rows = calibrate(args.bank, limit=args.limit, bootstrap=bs)
    out, rp = write_calibration(doc, rows, args.out)
    print(json.dumps({"out": out, "rows": rp, "units": doc["unit_count"], "records": doc["record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
