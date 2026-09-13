"""D2 causal operator bank (C135 contract, C136 bank; C152-C155 boundary).

Contract, per operator kind:

* spec ``{"kind": str, "params": dict}`` validated strictly: exact keys,
  exact primitive types (bool is never a number, int is never float),
  finite values, and membership in the small PREDECLARED grid of the
  bank. Nothing outside the grid is an operator of this bank.
* ``fit(spec, snapshot, mode)``: ``snapshot`` is a ``FitSnapshot`` built by
  ``FitSnapshot.from_contract`` (tools/df_snapshot.py) and re-derived here at
  the last point of use (contract digest, matrix digest, source bytes, range
  inside the declared partition, role allowed, later partitions excluded). A
  bare array or a role string refuses. ``mode`` is one of FIT_MODES and must
  be declared for the kind (KIND_FIT_MODES). The artifact records the
  snapshot digest, dataset, contract, role, range and columns it was fitted
  on, and is bound by the SHA-256 of its canonical JSON.
* ``transform_batch(fitted, snapshot)`` consumes a ``TransformSnapshot``:
  strictly increasing timestamps, no row available after its decision instant,
  the fitted dataset, contract and columns, and a range the fit mode licenses.
* ``init_state(fitted, snapshot)`` binds a stream to a licensed snapshot;
  ``step(fitted, state, row)`` takes a ``TransformRow`` of a verified snapshot
  and enforces the same facts per row (next row index, strictly later
  timestamp, availability, dataset and columns); ``transform_chunk`` does the
  same for a contiguous later snapshot. ``save_state``/``load_state`` keep the
  binding.
* The numeric kernels (``_fit_kernel``, ``_transform_kernel``,
  ``_kernel_init_state``, ``_kernel_step``, ``_transform_chunk_kernel``) take
  bare arrays and are private: tests and the causal battery only. Their
  artifacts are unbound and every public path refuses them.
* ``probe_transform``/``probe_stream`` run an artifact on a declared generated
  probe (zero, impulse, step, random walk) for delay and cost measurement;
  they cannot carry caller data.

Temporal fit modes (C154):

``FROZEN_PREVIOUS_PARTITION``  parameters estimated on the fit snapshot are
    applied only to partitions strictly after the fit role's partition: the
    transform refuses any row before the end of that partition.
``EXPANDING_PREFIX``  the output at row t is a function of rows <= t and of
    parameters estimated from rows < t only; the state is updated after the
    output of t is emitted. For kinds whose parameters do not depend on data
    (DATA_INDEPENDENT_KINDS) this holds trivially and any partition may be
    transformed. ``causal_decomposition`` implements it: phase means are
    estimated from the stream prefix, and every column is WARMUP until each
    phase has at least one finite observation.
``OFFLINE_ANALYSIS_ONLY_NON_CAUSAL``  diagnostic artifact; it never emits
    per-timestamp values: every transform, state and probe path refuses it.

Kinds whose parameters are estimated from data and which have no expanding
implementation (Kalman MLE, Hampel fallback scale, trailing Haar thresholds)
declare FROZEN_PREVIOUS_PARTITION only; EXPANDING_PREFIX refuses for them.

``trailing_haar_threshold`` (C155; previously ``wavelet_haar_atrous``). With
J = levels and c_0(t) = x_t, for j = 0..J-1:
    c_{j+1}(t) = (c_j(t) + c_j(t - 2^j)) / 2,   d_{j+1}(t) = c_j(t) - c_{j+1}(t)
    y_t = c_J(t) + sum_j soft(d_j(t), lambda_j),  soft(d, l) = sign(d) max(|d| - l, 0)
Lookback 2^J - 1 samples; no padding of any kind: rows 0..2^J-2 of a stream
are WARMUP, and an output whose trailing window holds a NaN is MISSING_INPUT.
Without thresholding y_t = x_t (telescoping). lambda_j = threshold_k * MAD/0.6745
of d_j over the fit snapshot (FROZEN only). It is NOT claimed to equal a
standard a trous / stationary wavelet transform; ``T06_CAUSAL_SWT`` is a
separate design arm, DESIGN_ONLY_NOT_IMPLEMENTED.

Zero look-ahead for every causal kind; delay is measured
(tools/df_operator_measure.py), never compensated. Missing input (NaN) is
typed per kind; nothing is forward-filled; +/-inf is refused.

Negative controls: ``centered_mean_oracle`` is NON_CAUSAL, has no incremental
path and its transform refuses unless ``oracle_mode=True``; ``trailing_median``
window 5 carries PREVIOUSLY_LAB_REJECTED_CONTROL.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, signal

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SNAP = _load("df_snapshot")
FitSnapshot = SNAP.FitSnapshot
TransformSnapshot = SNAP.TransformSnapshot
TransformRow = SNAP.TransformRow

ARTIFACT_SCHEMA = "df_operator_artifact.v2"
STATE_SCHEMA = "df_operator_state.v2"

FROZEN_PREVIOUS_PARTITION = "FROZEN_PREVIOUS_PARTITION"
EXPANDING_PREFIX = "EXPANDING_PREFIX"
OFFLINE_ANALYSIS_ONLY_NON_CAUSAL = "OFFLINE_ANALYSIS_ONLY_NON_CAUSAL"
FIT_MODES = (FROZEN_PREVIOUS_PARTITION, EXPANDING_PREFIX, OFFLINE_ANALYSIS_ONLY_NON_CAUSAL)

LOOKBACK_UNBOUNDED = -1
LOOKBACK_NON_CAUSAL = -2

AVAILABLE = "AVAILABLE"
WARMUP = "WARMUP"
MISSING_INPUT = "MISSING_INPUT"
FUTURE_UNAVAILABLE = "FUTURE_UNAVAILABLE"      # oracle edges only
REASONS = (AVAILABLE, WARMUP, MISSING_INPUT, FUTURE_UNAVAILABLE)
_REASON_DTYPE = "<U18"

PREVIOUSLY_LAB_REJECTED_CONTROL = "PREVIOUSLY_LAB_REJECTED_CONTROL"
NON_CAUSAL_NEGATIVE_CONTROL = "NON_CAUSAL_NEGATIVE_CONTROL"

# Guards added by C152-C154, switched off only by mutation tests (C159).
GUARDS = {name: True for name in (
    "artifact_bound", "dataset_binding", "column_identity", "transform_partition_license",
    "fit_mode_enforcement", "stream_binding")}

NAMING_DECISIONS = (
    {"subject": "trailing_haar_threshold", "subject_kind": "OPERATOR", "previous_name": "wavelet_haar_atrous",
     "decision": "RENAMED",
     "evidence": "C155: a trailing multiscale Haar recurrence with soft thresholds (recurrence, edge convention "
                 "and warm-up in the module docstring); no equivalence with a standard a trous or stationary "
                 "wavelet transform was demonstrated, so the name no longer claims one. Historical records keep "
                 "the old name."},
    {"subject": "T06_CAUSAL_SWT", "subject_kind": "DESIGN_ARM", "previous_name": None,
     "decision": "DESIGN_ONLY_NOT_IMPLEMENTED",
     "evidence": "C155: named by the D3 design; no implementation and no proof exist; never a synonym of "
                 "trailing_haar_threshold."},
)


class OperatorRefusal(Exception):
    """Typed refusal; every message starts with REFUSED."""

    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


class OperatorAbstain(OperatorRefusal):
    """The artifact is a typed ABSTAIN: no output exists for it."""


def _guard(name: str, bad: bool, msg: str) -> None:
    if GUARDS[name] and bad:
        raise OperatorRefusal(msg)


def _snap_call(fn, *a):
    try:
        return fn(*a)
    except SNAP.SnapshotRefusal as exc:
        raise OperatorRefusal(str(exc)[len("REFUSED: "):]) from exc


# ----------------------------------------------------------------------
# Bank: the predeclared parameter grids (C136). (type, allowed values)
# ----------------------------------------------------------------------
BANK_GRID = {
    "identity": {},
    "ewma": {"alpha": (float, (0.1, 0.3, 0.5))},
    "local_level_kalman": {},
    "local_linear_trend_kalman": {},
    "trailing_mean": {"window": (int, (5, 9))},
    "trailing_median": {"window": (int, (5, 9, 21))},
    "trailing_hampel": {"window": (int, (9, 21)), "k": (float, (3.0,))},
    "fir_sinc_lowpass": {"cutoff": (float, (0.1, 0.25)),
                         "taps": (int, (21,))},
    "butterworth2_lowpass": {"cutoff": (float, (0.1, 0.25))},
    "trailing_haar_threshold": {"levels": (int, (2, 3)),
                                "threshold_k": (float, (3.0,))},
    "causal_decomposition": {"period": (int, (5, 24)),
                             "trend_alpha": (float, (0.1,)),
                             "season_alpha": (float, (0.1,))},
    "centered_mean_oracle": {"window": (int, (5,))},
}
KINDS = tuple(BANK_GRID)
WINDOWED_KINDS = ("identity", "trailing_mean", "trailing_median",
                  "trailing_hampel", "fir_sinc_lowpass",
                  "trailing_haar_threshold")
RECURSIVE_KINDS = ("ewma", "local_level_kalman",
                   "local_linear_trend_kalman", "butterworth2_lowpass",
                   "causal_decomposition")
NON_CAUSAL_KINDS = ("centered_mean_oracle",)
CAUSAL_KINDS = WINDOWED_KINDS + RECURSIVE_KINDS
DATA_INDEPENDENT_KINDS = ("identity", "ewma", "trailing_mean", "trailing_median", "fir_sinc_lowpass",
                          "butterworth2_lowpass", "centered_mean_oracle")
KIND_FIT_MODES = {k: ((EXPANDING_PREFIX, FROZEN_PREVIOUS_PARTITION, OFFLINE_ANALYSIS_ONLY_NON_CAUSAL)
                      if k in DATA_INDEPENDENT_KINDS or k == "causal_decomposition"
                      else (FROZEN_PREVIOUS_PARTITION, OFFLINE_ANALYSIS_ONLY_NON_CAUSAL))
                  for k in KINDS}

_WINDOW_MISSING = ("WINDOW_MUST_BE_COMPLETE: an output whose window holds "
                   "a NaN is MISSING_INPUT; no imputation")
_CARRY_MISSING = ("CARRY_STATE: a NaN sample is not consumed, state is "
                  "carried unchanged, output MISSING_INPUT")
_KALMAN_MISSING = ("PREDICT_ONLY: a NaN sample runs the prediction step "
                   "only (variance grows), output MISSING_INPUT")

# Declared, data-level assumptions and properties of every kind.
KIND_META = {
    "identity": {
        "linearity": "LTI", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": False,
                        "impulses_preserved": True}},
    "ewma": {
        "linearity": "LTI_STEADY_STATE", "missing_policy": _CARRY_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "initialized_at_first_finite_sample": True}},
    "local_level_kalman": {
        "linearity": "LTI_STEADY_STATE", "missing_policy": _KALMAN_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True, "gaussian_noise": True,
                        "random_walk_level": True,
                        "impulses_preserved": False,
                        "noise_ratio_identified_on_fit_snapshot": True}},
    "local_linear_trend_kalman": {
        "linearity": "LTI_STEADY_STATE", "missing_policy": _KALMAN_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True, "gaussian_noise": True,
                        "random_walk_level_and_slope": True,
                        "impulses_preserved": False,
                        "noise_ratios_identified_on_fit_snapshot": True}},
    "trailing_mean": {
        "linearity": "LTI", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "is_causal_fir_moving_average": True}},
    "trailing_median": {
        "linearity": "NONLINEAR", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "steps_preserved_with_delay": True}},
    "trailing_hampel": {
        "linearity": "NONLINEAR", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "inliers_passed_unchanged": True,
                        "fallback_scale_identified_on_fit_snapshot": True}},
    "fir_sinc_lowpass": {
        "linearity": "LTI", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "signal_band_below_cutoff": True}},
    "butterworth2_lowpass": {
        "linearity": "LTI_STEADY_STATE", "missing_policy": _CARRY_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "impulses_preserved": False,
                        "signal_band_below_cutoff": True,
                        "steady_state_init_at_first_finite_sample": True}},
    "trailing_haar_threshold": {
        "linearity": "NONLINEAR", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "gaussian_detail_noise": True,
                        "impulses_preserved": False,
                        "threshold_sigma_identified_on_fit_snapshot": True,
                        "trailing_haar_frontier": True,
                        "claims_standard_swt_or_a_trous": False}},
    "causal_decomposition": {
        "linearity": "LINEAR_PERIODICALLY_TIME_VARYING",
        "missing_policy": _CARRY_MISSING,
        "outputs": ["denoised", "trend", "seasonal", "residual"],
        "assumptions": {"stationary_noise": True,
                        "additive_seasonality": True,
                        "fixed_declared_period": True,
                        "phase_origin_is_dataset_row_index": True,
                        "impulses_preserved": False}},
    "centered_mean_oracle": {
        "linearity": "LTI_NON_CAUSAL", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"uses_future_samples": True,
                        "deployable": False}},
}


def code_sha256() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(SNAP.__file__).read_bytes())
    return h.hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _sha(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


# ----------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------
def validate_spec(spec) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"kind", "params"}:
        raise OperatorRefusal("spec keys must be exactly ['kind', "
                              "'params']")
    kind = spec["kind"]
    if type(kind) is not str or kind not in BANK_GRID:
        raise OperatorRefusal(f"unknown operator kind {kind!r}")
    grid = BANK_GRID[kind]
    params = spec["params"]
    if not isinstance(params, dict) or set(params) != set(grid):
        raise OperatorRefusal(f"params for {kind} must be exactly "
                              f"{sorted(grid)}")
    for name, (typ, allowed) in grid.items():
        v = params[name]
        if type(v) is not typ:          # bool is not int, int not float
            raise OperatorRefusal(
                f"param {name!r} must be {typ.__name__}, got "
                f"{type(v).__name__}")
        if not math.isfinite(float(v)):
            raise OperatorRefusal(f"param {name!r} must be finite")
        if v not in allowed:
            raise OperatorRefusal(
                f"param {name!r}={v!r} is outside the predeclared grid "
                f"{list(allowed)}")
    return spec


def bank_specs() -> list:
    """Every operator of the bank, in a fixed order."""
    out = []
    for kind, grid in BANK_GRID.items():
        names = sorted(grid)
        combos = [{}]
        for n in names:
            combos = [dict(c, **{n: v}) for c in combos
                      for v in grid[n][1]]
        out.extend({"kind": kind, "params": c} for c in combos)
    return out


def derived_lookback(kind: str, params: dict) -> int:
    """Algorithmic lookback derived from the spec."""
    if kind == "identity":
        return 0
    if kind in ("trailing_mean", "trailing_median", "trailing_hampel"):
        return int(params["window"]) - 1
    if kind == "fir_sinc_lowpass":
        return int(params["taps"]) - 1
    if kind == "trailing_haar_threshold":
        return 2 ** int(params["levels"]) - 1
    if kind in RECURSIVE_KINDS:
        return LOOKBACK_UNBOUNDED
    if kind == "centered_mean_oracle":
        return LOOKBACK_NON_CAUSAL
    raise OperatorRefusal(f"kind {kind!r} has no derived lookback")


def look_ahead(kind: str, params: dict) -> int:
    """Samples of future reach. Zero for every causal kind."""
    if kind == "centered_mean_oracle":
        return (int(params["window"]) - 1) // 2
    return 0


def warmup_length(kind: str, params: dict, mode: str = FROZEN_PREVIOUS_PARTITION) -> int:
    """Rows from stream start typed WARMUP on a complete stream (windowed
    kinds need a full window; the expanding decomposition needs one
    observation of every phase; other recursive kinds initialize at the first
    finite sample)."""
    lb = derived_lookback(kind, params)
    if kind in WINDOWED_KINDS:
        return lb
    if kind == "centered_mean_oracle":
        return (int(params["window"]) - 1) // 2
    if kind == "causal_decomposition" and mode == EXPANDING_PREFIX:
        return int(params["period"])
    return 0


def control_label(spec: dict):
    if spec["kind"] == "centered_mean_oracle":
        return NON_CAUSAL_NEGATIVE_CONTROL
    if spec["kind"] == "trailing_median" and spec["params"]["window"] == 5:
        return PREVIOUSLY_LAB_REJECTED_CONTROL
    return None


_LICENSE_TEXT = {
    FROZEN_PREVIOUS_PARTITION: "partitions strictly after the fit role's partition only",
    EXPANDING_PREFIX: "any partition; output at t uses rows <= t and parameters from rows < t",
    OFFLINE_ANALYSIS_ONLY_NON_CAUSAL: "none: never emits per-timestamp values",
}


def _meta(spec: dict, mode: str) -> dict:
    kind, params = spec["kind"], spec["params"]
    m = copy.deepcopy(KIND_META[kind])
    m.update({
        "causality": ("NON_CAUSAL" if kind in NON_CAUSAL_KINDS
                      else "CAUSAL"),
        "control_label": control_label(spec),
        "lookback": derived_lookback(kind, params),
        "look_ahead": look_ahead(kind, params),
        "warmup": warmup_length(kind, params, mode),
        "future_shift_compensation": 0,
        "fit_mode": mode,
        "parameters_depend_on_data": kind not in DATA_INDEPENDENT_KINDS,
        "licensed_transform": _LICENSE_TEXT[mode],
    })
    return m


# ----------------------------------------------------------------------
# input checks
# ----------------------------------------------------------------------
def _real_array(x, where: str, allow_nan: bool) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in ("f", "i", "u"):
        raise OperatorRefusal(f"{where}: dtype {arr.dtype} is not a real "
                              "numeric dtype (bool/str/object/complex "
                              "refused)")
    out = arr.astype(np.float64)
    if np.isinf(out).any():
        raise OperatorRefusal(f"{where}: infinite value refused")
    if not allow_nan and np.isnan(out).any():
        raise OperatorRefusal(f"{where}: NaN refused (the fit snapshot must "
                              "be complete; missingness is typed only at "
                              "transform time)")
    return out


# ----------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------
_MIN_TRAIN_ROWS = 50
_LOG_RATIO_BOUNDS = (-12.0, 4.0)


def _ll_loglik(y: np.ndarray, log_ratio: float):
    """Concentrated Gaussian log-likelihood of the local level model
    with r=1, q=exp(log_ratio); same recursion and initialization as
    the transform (level=y0, var=r at the first sample)."""
    q = math.exp(log_ratio)
    lvl, var = float(y[0]), 1.0
    s_logf, s_v2f, m = 0.0, 0.0, 0
    for t in range(len(y)):
        var_p = var + q
        f = var_p + 1.0
        v = float(y[t]) - lvl
        if t >= 1:
            s_logf += math.log(f)
            s_v2f += v * v / f
            m += 1
        k = var_p / f
        lvl = lvl + k * v
        var = (1.0 - k) * var_p
    sigma2 = s_v2f / m
    return -0.5 * (s_logf + m * math.log(max(sigma2, 1e-300))), sigma2


def _llt_loglik(y: np.ndarray, log_ratios):
    ql, qs = math.exp(log_ratios[0]), math.exp(log_ratios[1])
    lvl, slope = float(y[0]), 0.0
    p11, p12, p22 = 1.0, 0.0, 1.0
    s_logf, s_v2f, m = 0.0, 0.0, 0
    for t in range(len(y)):
        a1 = lvl + slope
        q11 = p11 + 2.0 * p12 + p22 + ql
        q12 = p12 + p22
        q22 = p22 + qs
        f = q11 + 1.0
        v = float(y[t]) - a1
        if t >= 2:
            s_logf += math.log(f)
            s_v2f += v * v / f
            m += 1
        k1, k2 = q11 / f, q12 / f
        lvl = a1 + k1 * v
        slope = slope + k2 * v
        p11 = q11 - k1 * q11
        p12 = q12 - k1 * q12
        p22 = q22 - k2 * q12
    sigma2 = s_v2f / m
    return -0.5 * (s_logf + m * math.log(max(sigma2, 1e-300))), sigma2


def _abstain(reason: str) -> dict:
    return {"__abstain__": reason}


def _haar_details(c: np.ndarray, levels: int):
    """Yield (detail_j, approx_j) for j = 1..levels on a window tensor
    (..., L) whose last element is time t; past-only pairs."""
    for j in range(levels):
        s = 2 ** j
        cn = (c[..., s:] + c[..., :-s]) * 0.5
        yield c[..., -1] - cn[..., -1], cn
        c = cn


def _fit_kind(kind: str, params: dict, x: np.ndarray, t0: int) -> dict:
    T, V = x.shape
    lo, hi = _LOG_RATIO_BOUNDS
    if kind == "local_level_kalman":
        cols = []
        for j in range(V):
            y = x[:, j]
            if float(np.var(np.diff(y))) <= 0.0:
                return _abstain("CONSTANT_TRAIN_SERIES: noise ratio not "
                                "identifiable")
            res = optimize.minimize_scalar(
                lambda lr: -_ll_loglik(y, lr)[0], bounds=(lo, hi),
                method="bounded", options={"xatol": 1e-6})
            lr = float(res.x)
            if not res.success or lr > hi - 0.1:
                return _abstain("MLE_AT_UPPER_BOUND: observation noise "
                                "not identifiable (series behaves as a "
                                "pure random walk)")
            _, sigma2 = _ll_loglik(y, lr)
            cols.append({"obs_var": float(sigma2),
                         "level_var": float(sigma2 * math.exp(lr)),
                         "log_ratio": lr})
        return {"per_column": cols, "estimator": "concentrated_mle"}
    if kind == "local_linear_trend_kalman":
        cols = []
        for j in range(V):
            y = x[:, j]
            if float(np.var(np.diff(y))) <= 0.0:
                return _abstain("CONSTANT_TRAIN_SERIES: noise ratios not "
                                "identifiable")

            def nll(p, y=y):
                pc = np.clip(p, lo, hi)
                pen = float(np.sum((p - pc) ** 2)) * 1e3
                return -_llt_loglik(y, pc)[0] + pen
            best = None
            for start in ((-2.0, -8.0), (0.0, -4.0)):
                r = optimize.minimize(nll, np.array(start),
                                      method="Nelder-Mead",
                                      options={"xatol": 1e-5,
                                               "fatol": 1e-8,
                                               "maxiter": 800})
                if best is None or r.fun < best.fun:
                    best = r
            lr = np.clip(best.x, lo, hi)
            if not best.success or float(lr[0]) > hi - 0.1:
                return _abstain("MLE_NOT_IDENTIFIED: level/slope noise "
                                "ratios did not converge inside bounds")
            _, sigma2 = _llt_loglik(y, lr)
            cols.append({"obs_var": float(sigma2),
                         "level_var": float(sigma2 * math.exp(lr[0])),
                         "slope_var": float(sigma2 * math.exp(lr[1])),
                         "log_ratios": [float(lr[0]), float(lr[1])]})
        return {"per_column": cols, "estimator": "concentrated_mle"}
    if kind == "trailing_hampel":
        w = params["window"]
        W = sliding_window_view(x, w, axis=0)
        med = np.median(W, axis=-1)
        resid = x[w - 1:] - med
        scale = 1.4826 * np.median(np.abs(resid), axis=0)
        if (scale <= 0).any():
            return _abstain("ZERO_TRAIN_ROBUST_SCALE: fallback scale not "
                            "identifiable")
        return {"fallback_scale": [float(s) for s in scale]}
    if kind == "fir_sinc_lowpass":
        h = signal.firwin(params["taps"], params["cutoff"], fs=1.0)
        return {"taps": [float(v) for v in h], "window": "hamming"}
    if kind == "butterworth2_lowpass":
        b, a = signal.butter(2, 2.0 * params["cutoff"])
        zi = signal.lfilter_zi(b, a)
        return {"b": [float(v) for v in b], "a": [float(v) for v in a],
                "zi_unit": [float(v) for v in zi]}
    if kind == "trailing_haar_threshold":
        W = sliding_window_view(x, 2 ** params["levels"], axis=0)
        lam = np.empty((V, params["levels"]))
        for j, (d, _) in enumerate(_haar_details(W, params["levels"])):
            sigma = np.median(np.abs(d - np.median(d, axis=0)),
                              axis=0) / 0.6745
            if (sigma <= 0).any():
                return _abstain("ZERO_DETAIL_MAD: detail noise sigma not "
                                f"identifiable at level {j + 1}")
            lam[:, j] = params["threshold_k"] * sigma
        return {"thresholds": lam.tolist(), "sigma_estimator":
                "MAD/0.6745 per level on the fit snapshot's details"}
    if kind == "causal_decomposition":
        P = params["period"]
        if T < 3 * P:
            return _abstain("TRAIN_SHORTER_THAN_3_PERIODS: seasonal phase "
                            "means not identifiable")
        a = params["trend_alpha"]
        trend = np.empty_like(x)
        trend[0] = x[0]
        for t in range(1, T):
            trend[t] = a * x[t] + (1.0 - a) * trend[t - 1]
        d = x - trend
        phase = (t0 + np.arange(T)) % P
        S = np.stack([d[phase == p].mean(axis=0) for p in range(P)],
                     axis=1)                             # (V, P)
        S = S - S.mean(axis=1, keepdims=True)
        return {"seasonal_init": S.tolist(),
                "phase_origin": "dataset row index modulo period"}
    return {}


def _seal(doc: dict) -> dict:
    body = {k: doc[k] for k in doc if k != "artifact_sha256"}
    doc["artifact_sha256"] = _sha(body)
    return doc


def _build_artifact(spec: dict, x, mode, binding, t0: int) -> dict:
    validate_spec(spec)
    if mode not in FIT_MODES:
        raise OperatorRefusal(f"unknown fit mode {mode!r}; declared modes are {list(FIT_MODES)}")
    _guard("fit_mode_enforcement", mode not in KIND_FIT_MODES[spec["kind"]],
           f"fit mode {mode} is not implemented for kind {spec['kind']} "
           f"(declared {list(KIND_FIT_MODES[spec['kind']])})")
    x = _real_array(x, "fit matrix", allow_nan=False)
    if x.ndim != 2 or x.shape[1] < 1:
        raise OperatorRefusal("fit matrix must be 2-D (T, V)")
    if x.shape[0] < _MIN_TRAIN_ROWS:
        raise OperatorRefusal(f"fit matrix needs >= {_MIN_TRAIN_ROWS} rows")
    if (spec["kind"] == "causal_decomposition" and mode == EXPANDING_PREFIX
            and GUARDS["fit_mode_enforcement"]):
        fitted = {"seasonal": "ESTIMATED_ONLINE_FROM_THE_STREAM_PREFIX",
                  "phase_origin": "dataset row index modulo period"}
    else:
        fitted = _fit_kind(spec["kind"], spec["params"], x, t0)
    status, reason = "FITTED", None
    if "__abstain__" in fitted:
        status, reason, fitted = "ABSTAIN", fitted["__abstain__"], {}
    doc = {"schema": ARTIFACT_SCHEMA, "spec": copy.deepcopy(spec),
           "status": status, "abstain_reason": reason, "fit_mode": mode,
           "fit_binding": binding, "fit_rows": int(x.shape[0]),
           "n_columns": int(x.shape[1]), "fitted": fitted,
           "meta": _meta(spec, mode), "code_sha256": code_sha256()}
    return _seal(doc)


def fit(spec: dict, snapshot, mode: str) -> dict:
    """Fit on a verified FitSnapshot under a declared temporal fit mode."""
    validate_spec(spec)
    if not isinstance(snapshot, FitSnapshot):
        raise OperatorRefusal("fit requires a FitSnapshot built from a contract; a bare array or a "
                              f"self-declared role string never grants a fit (got {type(snapshot).__name__})")
    _snap_call(SNAP.verify_fit_snapshot, snapshot)
    ranges = SNAP._partition_ranges(snapshot.contract)
    binding = {"snapshot_sha256": snapshot.snapshot_sha256, "dataset_id": snapshot.dataset_id,
               "contract_sha256": snapshot.contract_sha256, "role": snapshot.role,
               "range": [snapshot.start, snapshot.end], "column_ids": list(snapshot.column_ids),
               "matrix_sha256": snapshot.matrix_sha256,
               "excluded_partitions": [list(p) for p in snapshot.excluded_partitions],
               "licensed_min_index": (ranges[snapshot.role][1] if mode == FROZEN_PREVIOUS_PARTITION else 0),
               "licensed_partitions": (list(SNAP.PARTITION_ORDER[SNAP.PARTITION_ORDER.index(snapshot.role) + 1:])
                                       if mode == FROZEN_PREVIOUS_PARTITION else
                                       [] if mode == OFFLINE_ANALYSIS_ONLY_NON_CAUSAL
                                       else list(SNAP.PARTITION_ORDER))}
    return _build_artifact(spec, snapshot.matrix, mode, binding, snapshot.start)


def _fit_kernel(spec: dict, train_array, mode: str = FROZEN_PREVIOUS_PARTITION, t0: int = 0) -> dict:
    """PRIVATE numeric kernel for tests and the causal battery. The artifact
    is unbound; every public transform, state or step path refuses it."""
    return _build_artifact(spec, train_array, mode, None, t0)


_ARTIFACT_KEYS = {"schema", "spec", "status", "abstain_reason", "fit_mode",
                  "fit_binding", "fit_rows", "n_columns", "fitted", "meta",
                  "code_sha256", "artifact_sha256"}


def verify_artifact(fitted) -> dict:
    if not isinstance(fitted, dict) or set(fitted) != _ARTIFACT_KEYS:
        raise OperatorRefusal("artifact keys are not the exact schema")
    if fitted["schema"] != ARTIFACT_SCHEMA:
        raise OperatorRefusal("unknown artifact schema")
    validate_spec(fitted["spec"])
    if fitted["fit_mode"] not in FIT_MODES:
        raise OperatorRefusal("artifact fit mode is not declared")
    if fitted["meta"] != _meta(fitted["spec"], fitted["fit_mode"]):
        raise OperatorRefusal("artifact meta differs from the kind's "
                              "declared meta")
    if fitted["code_sha256"] != code_sha256():
        raise OperatorRefusal("artifact was produced by different operator "
                              "code")
    body = {k: fitted[k] for k in fitted if k != "artifact_sha256"}
    try:
        digest = _sha(body)
    except (TypeError, ValueError) as exc:
        raise OperatorRefusal(f"artifact is not canonical JSON: {exc}")
    if digest != fitted["artifact_sha256"]:
        raise OperatorRefusal("artifact digest does not re-derive "
                              "(mutated artifact)")
    return fitted


def _usable(fitted: dict) -> dict:
    verify_artifact(fitted)
    if fitted["fit_mode"] == OFFLINE_ANALYSIS_ONLY_NON_CAUSAL:
        raise OperatorRefusal("OFFLINE_ANALYSIS_ONLY_NON_CAUSAL artifact never emits per-timestamp values")
    if fitted["status"] == "ABSTAIN":
        raise OperatorAbstain(f"ABSTAIN artifact has no output: "
                              f"{fitted['abstain_reason']}")
    return fitted


def _check_license(fitted: dict, snap) -> None:
    """Dataset, columns and partition license of a verified TransformSnapshot."""
    b = fitted["fit_binding"]
    _guard("artifact_bound", b is None, "artifact is not bound to a FitSnapshot (kernel test artifact)")
    if b is None:
        return
    _guard("dataset_binding", snap.dataset_id != b["dataset_id"] or snap.contract_sha256 != b["contract_sha256"],
           "transform snapshot belongs to another dataset or contract than the fit")
    _guard("column_identity", list(snap.column_ids) != b["column_ids"],
           "transform columns differ from the fitted columns")
    _guard("transform_partition_license",
           snap.start < b["licensed_min_index"] or any(p not in b["licensed_partitions"] for p in snap.partitions),
           f"range [{snap.start}, {snap.end}) covering {list(snap.partitions)} is not licensed for a "
           f"{fitted['fit_mode']} fit on {b['role']} (licensed {b['licensed_partitions']} from row "
           f"{b['licensed_min_index']})")


# ----------------------------------------------------------------------
# windowed kinds: evaluation on a window tensor (..., V, L)
# ----------------------------------------------------------------------
def _window_len(spec: dict) -> int:
    return derived_lookback(spec["kind"], spec["params"]) + 1


def _soft(d, lam):
    return np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)


def _window_eval(fitted: dict, W: np.ndarray) -> np.ndarray:
    """Elementwise arithmetic in a fixed order (sums accumulate from the
    oldest sample), so batch, step, chunk and restart give identical bits
    whatever the memory layout of W."""
    kind, p, f = (fitted["spec"]["kind"], fitted["spec"]["params"],
                  fitted["fitted"])
    with np.errstate(invalid="ignore"):
        if kind == "identity":
            return W[..., -1].copy()
        if kind == "trailing_mean":
            L = W.shape[-1]
            acc = W[..., 0].copy()
            for k in range(1, L):
                acc = acc + W[..., k]
            return acc / L
        if kind == "trailing_median":
            return np.median(W, axis=-1)
        if kind == "trailing_hampel":
            med = np.median(W, axis=-1)
            mad = 1.4826 * np.median(np.abs(W - med[..., None]), axis=-1)
            scale = np.where(mad > 0, mad,
                             np.asarray(f["fallback_scale"]))
            xt = W[..., -1]
            return np.where(np.abs(xt - med) > p["k"] * scale, med, xt)
        if kind == "fir_sinc_lowpass":
            h = f["taps"]                # y_t = sum_k h[k] x[t-k]
            L = W.shape[-1]
            acc = h[L - 1] * W[..., 0]
            for k in range(L - 2, -1, -1):
                acc = acc + h[k] * W[..., L - 1 - k]
            return acc
        if kind == "trailing_haar_threshold":
            y, _ = _haar_eval(fitted, W)
            return y
    raise OperatorRefusal(f"kind {kind!r} is not windowed")


def _haar_eval(fitted: dict, W: np.ndarray):
    lam = np.asarray(fitted["fitted"]["thresholds"])            # (V, J)
    acc = np.zeros(W.shape[:-1])
    levels = []
    c = W
    with np.errstate(invalid="ignore"):
        for j, (d, cn) in enumerate(_haar_details(W, fitted["spec"]["params"]["levels"])):
            sd = _soft(d, lam[:, j])
            acc = acc + sd
            levels.append({"detail": sd, "approx": cn[..., -1].copy()})
            c = cn
        return c[..., -1] + acc, levels


# ----------------------------------------------------------------------
# recursive kinds: one shared update over V columns
# ----------------------------------------------------------------------
def _fresh_payload(fitted: dict) -> dict:
    kind, V = fitted["spec"]["kind"], fitted["n_columns"]
    nan = np.full(V, np.nan)
    if kind in WINDOWED_KINDS:
        return {"buffer": np.full((V, _window_len(fitted["spec"])),
                                  np.nan)}
    if kind == "ewma":
        return {"level": nan.copy()}
    if kind == "local_level_kalman":
        return {"level": nan.copy(), "var": nan.copy()}
    if kind == "local_linear_trend_kalman":
        return {k: nan.copy() for k in ("level", "slope", "p11", "p12",
                                        "p22")}
    if kind == "butterworth2_lowpass":
        return {"z0": nan.copy(), "z1": nan.copy()}
    if kind == "causal_decomposition":
        if fitted["fit_mode"] == EXPANDING_PREFIX:
            P = fitted["spec"]["params"]["period"]
            return {"trend": nan.copy(), "seasonal": np.full((V, P), np.nan),
                    "raw_level": nan.copy(), "phase_sum": np.zeros((V, P)),
                    "phase_count": np.zeros((V, P)), "ready": np.zeros(V)}
        return {"trend": nan.copy(),
                "seasonal": np.asarray(fitted["fitted"]["seasonal_init"],
                                       dtype=np.float64).copy()}
    raise OperatorRefusal(f"kind {kind!r} has no state")


def _decomp_ready_step(p, pay, ph, x, fin, ready):
    aT, aS = p["trend_alpha"], p["season_alpha"]
    S = pay["seasonal"]
    s_prev = S[:, ph].copy()            # learned from rows < t
    tr = pay["trend"]
    des = x - s_prev
    tr_new = np.where(np.isnan(tr), des, tr + aT * (des - tr))
    s_new = s_prev + aS * ((x - tr_new) - s_prev)
    use = fin & ready
    nanv = np.full_like(x, np.nan)
    outs = {"denoised": np.where(use, tr_new + s_prev, nanv),
            "trend": np.where(use, tr_new, nanv),
            "seasonal": np.where(use, s_prev, nanv),
            "residual": np.where(use, x - tr_new - s_prev, nanv)}
    pay["trend"] = np.where(use, tr_new, tr)
    S[:, ph] = np.where(use, s_new, s_prev)
    return outs


def _rec_update(fitted: dict, pay: dict, idx: int, x: np.ndarray) -> tuple:
    """Consume row x (V,) at dataset row idx; mutate pay after the outputs
    are computed; return (outputs, reason)."""
    kind, p, f = (fitted["spec"]["kind"], fitted["spec"]["params"],
                  fitted["fitted"])
    fin = np.isfinite(x)
    reason = np.where(fin, AVAILABLE, MISSING_INPUT).astype(_REASON_DTYPE)
    with np.errstate(invalid="ignore"):
        if kind == "ewma":
            a = p["alpha"]
            lvl = pay["level"]
            init = np.isnan(lvl)
            new = np.where(init, x, a * x + (1.0 - a) * lvl)
            pay["level"] = np.where(fin, new, lvl)
            return {"y": np.where(fin, pay["level"], np.nan)}, reason
        if kind == "local_level_kalman":
            r = np.array([c["obs_var"] for c in f["per_column"]])
            q = np.array([c["level_var"] for c in f["per_column"]])
            start = np.isnan(pay["level"]) & fin
            lvl = np.where(start, x, pay["level"])
            var = np.where(start, r, pay["var"])
            var_p = var + q
            k = var_p / (var_p + r)
            lvl_u = lvl + k * (x - lvl)
            var_u = (1.0 - k) * var_p
            pay["level"] = np.where(fin, lvl_u, lvl)
            pay["var"] = np.where(fin, var_u, var_p)
            return {"y": np.where(fin, pay["level"], np.nan)}, reason
        if kind == "local_linear_trend_kalman":
            r = np.array([c["obs_var"] for c in f["per_column"]])
            ql = np.array([c["level_var"] for c in f["per_column"]])
            qs = np.array([c["slope_var"] for c in f["per_column"]])
            start = np.isnan(pay["level"]) & fin
            lvl = np.where(start, x, pay["level"])
            slope = np.where(start, 0.0, pay["slope"])
            p11 = np.where(start, r, pay["p11"])
            p12 = np.where(start, 0.0, pay["p12"])
            p22 = np.where(start, r, pay["p22"])
            a1 = lvl + slope
            q11 = p11 + 2.0 * p12 + p22 + ql
            q12 = p12 + p22
            q22 = p22 + qs
            fv = q11 + r
            k1, k2 = q11 / fv, q12 / fv
            v = x - a1
            pay["level"] = np.where(fin, a1 + k1 * v, a1)
            pay["slope"] = np.where(fin, slope + k2 * v, slope)
            pay["p11"] = np.where(fin, q11 - k1 * q11, q11)
            pay["p12"] = np.where(fin, q12 - k1 * q12, q12)
            pay["p22"] = np.where(fin, q22 - k2 * q12, q22)
            return {"y": np.where(fin, pay["level"], np.nan)}, reason
        if kind == "butterworth2_lowpass":
            b0, b1, b2 = f["b"]
            _, a1, a2 = f["a"]
            zi0, zi1 = f["zi_unit"]
            start = np.isnan(pay["z0"]) & fin
            z0 = np.where(start, zi0 * x, pay["z0"])
            z1 = np.where(start, zi1 * x, pay["z1"])
            y = b0 * x + z0
            n0 = b1 * x - a1 * y + z1
            n1 = b2 * x - a2 * y
            pay["z0"] = np.where(fin, n0, z0)
            pay["z1"] = np.where(fin, n1, z1)
            return {"y": np.where(fin, y, np.nan)}, reason
        if kind == "causal_decomposition":
            P = p["period"]
            ph = idx % P
            if fitted["fit_mode"] != EXPANDING_PREFIX:
                return _decomp_ready_step(p, pay, ph, x, fin, np.ones_like(fin)), reason
            ready = pay["ready"] > 0
            outs = _decomp_ready_step(p, pay, ph, x, fin, ready)
            warm = fin & ~ready
            reason[warm] = WARMUP
            # state update after emission: expanding phase means of the
            # detrended prefix, the same estimator as the frozen fit
            aT = p["trend_alpha"]
            rl = pay["raw_level"]
            new_rl = np.where(np.isnan(rl), x, aT * x + (1.0 - aT) * rl)
            pay["raw_level"] = np.where(warm, new_rl, rl)
            pay["phase_sum"][:, ph] = np.where(warm, pay["phase_sum"][:, ph] + (x - new_rl),
                                               pay["phase_sum"][:, ph])
            pay["phase_count"][:, ph] = pay["phase_count"][:, ph] + warm
            now = warm & np.all(pay["phase_count"] > 0, axis=1)
            if now.any():
                means = pay["phase_sum"] / np.where(pay["phase_count"] > 0, pay["phase_count"], 1.0)
                acc = means[:, 0].copy()
                for q in range(1, P):
                    acc = acc + means[:, q]
                centred = means - (acc / P)[:, None]
                pay["seasonal"] = np.where(now[:, None], centred, pay["seasonal"])
                pay["trend"] = np.where(now, pay["raw_level"], pay["trend"])
                pay["ready"] = np.where(now, 1.0, pay["ready"])
            return outs, reason
    raise OperatorRefusal(f"kind {kind!r} is not recursive")


# ----------------------------------------------------------------------
# batch (kernel)
# ----------------------------------------------------------------------
def _check_X(fitted: dict, X, where: str) -> np.ndarray:
    x = _real_array(X, where, allow_nan=True)
    if x.ndim != 2 or x.shape[1] != fitted["n_columns"]:
        raise OperatorRefusal(f"{where} must be 2-D with "
                              f"{fitted['n_columns']} columns")
    return x


def _windowed_reasons(x: np.ndarray, L: int, warm: int) -> np.ndarray:
    T = x.shape[0]
    isn = np.isnan(x)
    c = np.concatenate([np.zeros((1, x.shape[1])),
                        np.cumsum(isn, axis=0)])
    idx = np.arange(T)
    lo = np.maximum(idx - L + 1, 0)
    win_nan = (c[idx + 1] - c[lo]) > 0
    tt = idx[:, None]
    reason = np.full(x.shape, AVAILABLE, dtype=_REASON_DTYPE)
    reason[win_nan] = MISSING_INPUT
    reason[(tt < warm) & ~isn] = WARMUP
    reason[isn] = MISSING_INPUT
    return reason


def _transform_kernel_components(fitted: dict, X, t0: int = 0,
                                 oracle_mode: bool = False) -> tuple:
    """PRIVATE: -> (outputs: dict name -> (T, V), available, reason)."""
    _usable(fitted)
    spec = fitted["spec"]
    kind = spec["kind"]
    x = _check_X(fitted, X, "batch X")
    T, V = x.shape
    if kind in NON_CAUSAL_KINDS:
        if oracle_mode is not True:
            raise OperatorRefusal("NON_CAUSAL oracle: transform requires "
                                  "explicit oracle_mode=True")
        w = spec["params"]["window"]
        h = (w - 1) // 2
        y = np.full(x.shape, np.nan)
        reason = np.full(x.shape, AVAILABLE, dtype=_REASON_DTYPE)
        if T >= w:
            y[h:T - h] = np.mean(sliding_window_view(x, w, axis=0),
                                 axis=-1)
        reason[:h] = WARMUP
        reason[T - h:] = FUTURE_UNAVAILABLE
        reason[(reason == AVAILABLE) & np.isnan(y)] = MISSING_INPUT
        avail = reason == AVAILABLE
        y[~avail] = np.nan
        return {"y": y}, avail, reason
    if oracle_mode:
        raise OperatorRefusal("oracle_mode is only meaningful for the "
                              "NON_CAUSAL oracle")
    if kind in WINDOWED_KINDS:
        L = _window_len(spec)
        reason = _windowed_reasons(x, L, fitted["meta"]["warmup"])
        y = np.full(x.shape, np.nan)
        if T >= L:
            y[L - 1:] = _window_eval(fitted,
                                     sliding_window_view(x, L, axis=0))
        avail = reason == AVAILABLE
        y[~avail] = np.nan
        return {"y": y}, avail, reason
    pay = _fresh_payload(fitted)
    names = fitted["meta"]["outputs"]
    outs = {n: np.empty(x.shape) for n in names}
    reason = np.empty(x.shape, dtype=_REASON_DTYPE)
    for t in range(T):
        o, r = _rec_update(fitted, pay, t0 + t, x[t])
        for n in names:
            outs[n][t] = o[n]
        reason[t] = r
    avail = reason == AVAILABLE
    return outs, avail, reason


def _transform_kernel(fitted: dict, X, t0: int = 0, oracle_mode: bool = False) -> tuple:
    outs, avail, reason = _transform_kernel_components(fitted, X, t0, oracle_mode)
    return outs[fitted["meta"]["outputs"][0]], avail, reason


def _transform_levels_kernel(fitted: dict, X) -> list:
    """PRIVATE hook (C156): per-level outputs of trailing_haar_threshold,
    [{"detail": (T, V), "approx": (T, V)}, ...], NaN where y is unavailable."""
    _usable(fitted)
    if fitted["spec"]["kind"] != "trailing_haar_threshold":
        raise OperatorRefusal("levels exist only for trailing_haar_threshold")
    x = _check_X(fitted, X, "batch X")
    L = _window_len(fitted["spec"])
    reason = _windowed_reasons(x, L, fitted["meta"]["warmup"])
    avail = reason == AVAILABLE
    out = [{"detail": np.full(x.shape, np.nan), "approx": np.full(x.shape, np.nan)}
           for _ in range(fitted["spec"]["params"]["levels"])]
    if x.shape[0] >= L:
        _, levels = _haar_eval(fitted, sliding_window_view(x, L, axis=0))
        for j, lv in enumerate(levels):
            for k in ("detail", "approx"):
                out[j][k][L - 1:] = lv[k]
                out[j][k][~avail] = np.nan
    return out


# ----------------------------------------------------------------------
# public batch
# ----------------------------------------------------------------------
def transform_batch_components(fitted: dict, snapshot, oracle_mode: bool = False) -> tuple:
    """-> (outputs: dict name -> (T, V), available, reason) for the rows of a
    verified, licensed TransformSnapshot."""
    _usable(fitted)
    if not isinstance(snapshot, TransformSnapshot):
        raise OperatorRefusal("transform requires a TransformSnapshot built from a contract; a bare array "
                              f"carries no timestamps, availability or partition (got {type(snapshot).__name__})")
    _snap_call(SNAP.verify_transform_snapshot, snapshot)
    _check_license(fitted, snapshot)
    return _transform_kernel_components(fitted, snapshot.matrix, snapshot.start, oracle_mode)


def transform_batch(fitted: dict, snapshot, oracle_mode: bool = False) -> tuple:
    """-> (Y (T, V), available (T, V) bool, reason (T, V) str)."""
    outs, avail, reason = transform_batch_components(fitted, snapshot, oracle_mode)
    return outs[fitted["meta"]["outputs"][0]], avail, reason


# ----------------------------------------------------------------------
# incremental
# ----------------------------------------------------------------------
def _new_state(fitted: dict, next_index: int, binding) -> dict:
    if fitted["spec"]["kind"] in NON_CAUSAL_KINDS:
        raise OperatorRefusal("NON_CAUSAL oracle has no incremental path")
    return {"schema": STATE_SCHEMA,
            "artifact_sha256": fitted["artifact_sha256"],
            "kind": fitted["spec"]["kind"], "t": 0, "next_index": int(next_index),
            "last_timestamp": None, "binding": binding,
            "payload": _fresh_payload(fitted)}


def _kernel_init_state(fitted: dict, t0: int = 0) -> dict:
    """PRIVATE: an unbound state for the numeric kernel."""
    _usable(fitted)
    return _new_state(fitted, t0, None)


def init_state(fitted: dict, snapshot) -> dict:
    """A state bound to the dataset, contract and columns of a verified,
    licensed TransformSnapshot; the stream starts at the snapshot's first row."""
    _usable(fitted)
    if not isinstance(snapshot, TransformSnapshot):
        raise OperatorRefusal("init_state requires a TransformSnapshot built from a contract")
    _snap_call(SNAP.verify_transform_snapshot, snapshot)
    _check_license(fitted, snapshot)
    return _new_state(fitted, snapshot.start, {"dataset_id": snapshot.dataset_id,
                                               "contract_sha256": snapshot.contract_sha256,
                                               "column_ids": list(snapshot.column_ids)})


def _check_state(fitted: dict, state) -> None:
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise OperatorRefusal("not a df_operator state")
    if state["artifact_sha256"] != fitted.get("artifact_sha256"):
        raise OperatorRefusal("state belongs to a different artifact")


def _step_core(fitted: dict, state: dict, x: np.ndarray) -> tuple:
    t = state["t"]
    pay = state["payload"]
    fin = np.isfinite(x)
    if fitted["spec"]["kind"] in WINDOWED_KINDS:
        buf = pay["buffer"]
        buf[:, :-1] = buf[:, 1:]
        buf[:, -1] = x
        val = _window_eval(fitted, buf[None])[0]
        win_nan = np.isnan(buf).any(axis=1)
        reason = np.full(x.shape, AVAILABLE, dtype=_REASON_DTYPE)
        if t < fitted["meta"]["warmup"]:
            reason[:] = WARMUP
        else:
            reason[win_nan] = MISSING_INPUT
        reason[~fin] = MISSING_INPUT
        avail = reason == AVAILABLE
        outs = {"y": np.where(avail, val, np.nan)}
    else:
        outs, reason = _rec_update(fitted, pay, state["next_index"], x)
        avail = reason == AVAILABLE
    state["t"] = t + 1
    state["next_index"] += 1
    return outs, avail, reason, state


def _kernel_step_components(fitted: dict, state: dict, x_t) -> tuple:
    """PRIVATE: step on a bare row of an unbound state."""
    _check_state(fitted, state)
    if state["binding"] is not None:
        raise OperatorRefusal("a bound state is stepped only with TransformRows")
    x = _real_array(x_t, "step x_t", allow_nan=True)
    if x.shape != (fitted["n_columns"],):
        raise OperatorRefusal(f"x_t must have shape ({fitted['n_columns']},)")
    return _step_core(fitted, state, x)


def _kernel_step(fitted: dict, state: dict, x_t) -> tuple:
    outs, avail, reason, state = _kernel_step_components(fitted, state, x_t)
    return outs[fitted["meta"]["outputs"][0]], avail, reason, state


def _kernel_state_levels(fitted: dict, state: dict) -> list:
    """PRIVATE hook (C156): per-level outputs for the row just stepped."""
    if fitted["spec"]["kind"] != "trailing_haar_threshold":
        raise OperatorRefusal("levels exist only for trailing_haar_threshold")
    buf = state["payload"]["buffer"]
    _, levels = _haar_eval(fitted, buf[None])
    ok = (state["t"] > fitted["meta"]["warmup"]) & ~np.isnan(buf).any(axis=1)
    return [{k: np.where(ok, lv[k][0], np.nan) for k in ("detail", "approx")} for lv in levels]


def _check_bound_stream(fitted: dict, state: dict, dataset_id, contract_sha256, column_ids,
                        first_index: int, first_ts: int) -> None:
    b = state["binding"]
    _guard("stream_binding", b is None, "state is not bound to a stream")
    if b is not None:
        _guard("stream_binding", b["dataset_id"] != dataset_id or b["contract_sha256"] != contract_sha256,
               "state belongs to another series (dataset or contract differs)")
        _guard("column_identity", b["column_ids"] != list(column_ids),
               "row columns differ from the stream's columns")
    fb = fitted["fit_binding"]
    _guard("artifact_bound", fb is None, "artifact is not bound to a FitSnapshot (kernel test artifact)")
    if fb is not None:
        _guard("column_identity", list(column_ids) != fb["column_ids"],
               "row columns differ from the fitted columns")
        _guard("transform_partition_license", first_index < fb["licensed_min_index"],
               f"row {first_index} is not licensed for a {fitted['fit_mode']} fit on {fb['role']} "
               f"(licensed from row {fb['licensed_min_index']})")
    if SNAP.GUARDS["monotonic_timestamps"]:
        if first_index != state["next_index"]:
            raise OperatorRefusal(f"row index {first_index} is not the next row {state['next_index']} of the stream")
        if state["last_timestamp"] is not None and not first_ts > state["last_timestamp"]:
            raise OperatorRefusal("timestamps are not strictly increasing along the stream")


def step_components(fitted: dict, state: dict, row) -> tuple:
    """-> (outputs dict name -> (V,), available (V,), reason (V,), state)."""
    _check_state(fitted, state)
    if not isinstance(row, TransformRow):
        raise OperatorRefusal("step requires a TransformRow of a verified TransformSnapshot, never a bare row")
    _snap_call(SNAP.verify_row, row)
    _check_bound_stream(fitted, state, row.dataset_id, row.contract_sha256, row.column_ids, row.index,
                        row.timestamp)
    if SNAP.GUARDS["availability"] and row.available_at > row.decision_at:
        raise OperatorRefusal("a row is available after its decision instant")
    x = _real_array(row.values, "step row", allow_nan=True)
    if x.shape != (fitted["n_columns"],):
        raise OperatorRefusal(f"row must have shape ({fitted['n_columns']},)")
    out = _step_core(fitted, state, x)
    state["last_timestamp"] = int(row.timestamp)
    return out


def step(fitted: dict, state: dict, row) -> tuple:
    """-> (y_t (V,), available_t (V,), reason_t (V,), state)."""
    outs, avail, reason, state = step_components(fitted, state, row)
    return outs[fitted["meta"]["outputs"][0]], avail, reason, state


def _chunk_core(fitted: dict, state: dict, x: np.ndarray) -> tuple:
    kind = fitted["spec"]["kind"]
    n, V = x.shape
    if kind not in WINDOWED_KINDS:
        names = fitted["meta"]["outputs"]
        outs = {k: np.empty(x.shape) for k in names}
        reason = np.empty(x.shape, dtype=_REASON_DTYPE)
        for i in range(n):
            o, _, r, _ = _step_core(fitted, state, x[i])
            for k in names:
                outs[k][i] = o[k]
            reason[i] = r
        return outs, reason == AVAILABLE, reason
    pay, t = state["payload"], state["t"]
    L = _window_len(fitted["spec"])
    full = np.concatenate([pay["buffer"].T, x], axis=0)           # (L + n, V)
    W = sliding_window_view(full, L, axis=0)[1:]                  # windows ending at chunk rows
    val = _window_eval(fitted, W)
    isn = np.isnan(x)
    win_nan = np.isnan(W).any(axis=-1)
    consumed = (t + np.arange(n))[:, None]
    reason = np.full(x.shape, AVAILABLE, dtype=_REASON_DTYPE)
    reason[win_nan] = MISSING_INPUT
    reason[(consumed < fitted["meta"]["warmup"]) & ~isn] = WARMUP
    reason[isn] = MISSING_INPUT
    avail = reason == AVAILABLE
    pay["buffer"] = np.ascontiguousarray(full[-L:].T)
    state["t"] = t + n
    state["next_index"] += n
    return {"y": np.where(avail, val, np.nan)}, avail, reason


def _transform_chunk_kernel(fitted: dict, state: dict, X) -> tuple:
    """PRIVATE: consume a chunk of bare rows on an unbound state."""
    _check_state(fitted, state)
    if state["binding"] is not None:
        raise OperatorRefusal("a bound state consumes only TransformSnapshots")
    x = _check_X(fitted, X, "chunk X")
    outs, avail, reason = _chunk_core(fitted, state, x)
    return outs[fitted["meta"]["outputs"][0]], avail, reason, state


def transform_chunk(fitted: dict, state: dict, snapshot) -> tuple:
    """Consume a contiguous later TransformSnapshot on a bound stream.
    -> (Y, available, reason, state)."""
    _check_state(fitted, state)
    if not isinstance(snapshot, TransformSnapshot):
        raise OperatorRefusal("transform_chunk requires a TransformSnapshot built from a contract")
    _snap_call(SNAP.verify_transform_snapshot, snapshot)
    _check_license(fitted, snapshot)
    _check_bound_stream(fitted, state, snapshot.dataset_id, snapshot.contract_sha256, snapshot.column_ids,
                        snapshot.start, int(snapshot.timestamps[0]))
    outs, avail, reason = _chunk_core(fitted, state, snapshot.matrix)
    state["last_timestamp"] = int(snapshot.timestamps[-1])
    return outs[fitted["meta"]["outputs"][0]], avail, reason, state


# ----------------------------------------------------------------------
# state persistence
# ----------------------------------------------------------------------
_STATE_KEYS = {"schema", "artifact_sha256", "kind", "t", "next_index", "last_timestamp", "binding", "payload"}


def _enc(a: np.ndarray):
    return [None if not math.isfinite(v) else float(v)
            for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def _payload_shapes(fitted: dict) -> dict:
    return {k: v.shape for k, v in _fresh_payload(fitted).items()}


def save_state(state: dict) -> bytes:
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise OperatorRefusal("state keys are not the exact schema")
    pay = {}
    for k, v in state["payload"].items():
        arr = np.asarray(v, dtype=np.float64)
        if k == "buffer":
            keep = min(state["t"], arr.shape[1])
            arr = arr[:, arr.shape[1] - keep:]
        pay[k] = {"shape": list(arr.shape), "values": _enc(arr)}
    body = {"schema": STATE_SCHEMA,
            "artifact_sha256": state["artifact_sha256"],
            "kind": state["kind"], "t": int(state["t"]), "next_index": int(state["next_index"]),
            "last_timestamp": state["last_timestamp"], "binding": state["binding"], "payload": pay}
    doc = dict(body, state_sha256=_sha(body))
    return _canonical(doc)


def load_state(blob: bytes, fitted: dict) -> dict:
    _usable(fitted)
    if not isinstance(blob, (bytes, bytearray)):
        raise OperatorRefusal("state must be bytes")

    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise OperatorRefusal("duplicate JSON key in state")
        return dict(pairs)
    try:
        doc = json.loads(blob.decode(), object_pairs_hook=_no_dupes,
                         parse_constant=lambda c: (_ for _ in ()).throw(
                             OperatorRefusal(f"non-finite literal {c}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorRefusal(f"malformed state bytes: {exc}")
    if not isinstance(doc, dict) or set(doc) != _STATE_KEYS | {"state_sha256"}:
        raise OperatorRefusal("state keys are not the exact schema")
    body = {k: doc[k] for k in doc if k != "state_sha256"}
    if _sha(body) != doc["state_sha256"]:
        raise OperatorRefusal("state digest does not re-derive")
    if _canonical(doc) != bytes(blob):
        raise OperatorRefusal("state bytes are not canonical")
    if doc["schema"] != STATE_SCHEMA:
        raise OperatorRefusal("unknown state schema")
    if doc["artifact_sha256"] != fitted["artifact_sha256"]:
        raise OperatorRefusal("state was produced under a different "
                              "artifact")
    if doc["kind"] != fitted["spec"]["kind"]:
        raise OperatorRefusal("state kind differs from the artifact")
    t, nxt, last = doc["t"], doc["next_index"], doc["last_timestamp"]
    if type(t) is not int or t < 0 or type(nxt) is not int or nxt < t:
        raise OperatorRefusal("state t or next_index invalid")
    if not (last is None or type(last) is int):
        raise OperatorRefusal("state last_timestamp invalid")
    b = doc["binding"]
    if not (b is None or (isinstance(b, dict) and set(b) == {"dataset_id", "contract_sha256", "column_ids"})):
        raise OperatorRefusal("state binding malformed")
    if fitted["spec"]["kind"] in NON_CAUSAL_KINDS:
        raise OperatorRefusal("NON_CAUSAL oracle has no state")
    shapes = _payload_shapes(fitted)
    pay_in = doc["payload"]
    if not isinstance(pay_in, dict) or set(pay_in) != set(shapes):
        raise OperatorRefusal("state payload keys are not the exact "
                              "schema for this kind")
    payload = {}
    for k, full_shape in shapes.items():
        ent = pay_in[k]
        if not isinstance(ent, dict) or set(ent) != {"shape", "values"}:
            raise OperatorRefusal(f"state payload {k!r} malformed")
        want = list(full_shape)
        if k == "buffer":
            want[1] = min(t, full_shape[1])
        if ent["shape"] != want:
            raise OperatorRefusal(f"state payload {k!r} shape "
                                  f"{ent['shape']} is incoherent with "
                                  f"t={t} (expected {want})")
        vals = ent["values"]
        n = int(np.prod(want))
        if not isinstance(vals, list) or len(vals) != n or any(
                v is not None and (type(v) is not float
                                   or not math.isfinite(v))
                for v in vals):
            raise OperatorRefusal(f"state payload {k!r} values invalid")
        arr = np.array([np.nan if v is None else v for v in vals],
                       dtype=np.float64).reshape(want)
        if k == "buffer":
            full = np.full(full_shape, np.nan)
            if want[1]:
                full[:, full_shape[1] - want[1]:] = arr
            arr = full
        elif (k == "seasonal" and fitted["fit_mode"] != EXPANDING_PREFIX
              and np.isnan(arr).any()):
            raise OperatorRefusal("seasonal means must be finite")
        payload[k] = arr
    if t == 0:
        fresh = _fresh_payload(fitted)
        for k in payload:
            if not np.array_equal(payload[k], fresh[k], equal_nan=True):
                raise OperatorRefusal("a t=0 state must equal the fresh "
                                      "state")
    return {"schema": STATE_SCHEMA,
            "artifact_sha256": fitted["artifact_sha256"],
            "kind": doc["kind"], "t": t, "next_index": nxt, "last_timestamp": last,
            "binding": b, "payload": payload}


# ----------------------------------------------------------------------
# declared probes (measurement of responses and cost, never data)
# ----------------------------------------------------------------------
PROBE_KINDS = ("ZERO", "IMPULSE", "STEP", "RANDOM_WALK")
_PROBE_TOKEN = object()


@dataclass(frozen=True, eq=False)
class ProbeSignal:
    token: object = field(repr=False)
    kind: str
    n: int
    V: int
    amplitude: float
    t0: int
    seed: int
    matrix: np.ndarray = field(repr=False)

    def __post_init__(self):
        if self.token is not _PROBE_TOKEN:
            raise OperatorRefusal("a probe is generated by probe_signal, never built by hand")


def _probe_matrix(kind, n, V, amplitude, t0, seed) -> np.ndarray:
    x = np.zeros((n, V))
    if kind == "IMPULSE":
        x[t0] = amplitude
    elif kind == "STEP":
        x[t0:] = amplitude
    elif kind == "RANDOM_WALK":
        x = np.cumsum(np.random.default_rng(seed).normal(size=(n, V)), axis=0)
    return x


def probe_signal(kind: str, n: int, V: int, amplitude: float = 1.0, t0: int = 64, seed: int = 0) -> ProbeSignal:
    if kind not in PROBE_KINDS or type(n) is not int or type(V) is not int or n < 1 or V < 1:
        raise OperatorRefusal(f"probe must be one of {list(PROBE_KINDS)} with positive integer n, V")
    m = _probe_matrix(kind, n, V, float(amplitude), int(t0), int(seed))
    m.flags.writeable = False
    return ProbeSignal(_PROBE_TOKEN, kind, n, V, float(amplitude), int(t0), int(seed), m)


def _verify_probe(p) -> np.ndarray:
    if not isinstance(p, ProbeSignal) or p.token is not _PROBE_TOKEN:
        raise OperatorRefusal("probe paths take only a ProbeSignal from probe_signal")
    again = _probe_matrix(p.kind, p.n, p.V, p.amplitude, p.t0, p.seed)
    if again.tobytes() != np.ascontiguousarray(p.matrix).tobytes():
        raise OperatorRefusal("probe matrix does not re-derive from its declared parameters")
    return again


def probe_transform(fitted: dict, probe, oracle_mode: bool = False) -> tuple:
    return _transform_kernel(fitted, _verify_probe(probe), 0, oracle_mode)


def probe_stream(fitted: dict, probe):
    """Yield (y, available, reason) row by row on a declared probe."""
    x = _verify_probe(probe)
    st = _kernel_init_state(fitted)
    for i in range(x.shape[0]):
        y, a, r, st = _kernel_step(fitted, st, x[i])
        yield y, a, r
