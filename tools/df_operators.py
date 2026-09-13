"""D2 causal operator bank (C135 contract, C136 bank).

Contract, per operator kind:

* spec ``{"kind": str, "params": dict}`` validated strictly: exact keys,
  exact primitive types (bool is never a number, int is never float),
  finite values, and membership in the small PREDECLARED grid of the
  bank. Nothing outside the grid is an operator of this bank.
* ``fit(spec, train_array, role)``: role must be ``"train"``; estimation
  (Kalman noise by maximum likelihood, robust scales, wavelet thresholds,
  seasonal phase means) uses the train array only. Returns a JSON-only
  artifact bound by the SHA-256 of its canonical JSON. The artifact is
  immutable by digest: every consumer re-derives the digest and refuses a
  mutated artifact. A kind whose assumptions cannot be identified on the
  train input returns ``status == "ABSTAIN"`` with a typed reason; such an
  artifact refuses to transform (``OperatorAbstain``).
* ``transform_batch(fitted, X)`` -> ``(Y, available, reason)``;
  ``init_state(fitted)``; ``step(fitted, state, x_t)`` ->
  ``(y_t, available_t, reason_t, state)``. ``reason`` is a string array
  with values in ``REASONS``. Unavailable samples are NaN.
* ``save_state(state)`` -> canonical JSON bytes (digest embedded);
  ``load_state(bytes, fitted)`` validates schema, artifact binding,
  shapes, types and the digest before returning a state.
* Zero look-ahead for every causal kind: Y[:t+1] is a function of
  X[:t+1] only. No kind shifts its output with future samples to
  "compensate" a delay; delay is measured (tools/df_operator_measure.py),
  never hidden.
* Missing input (NaN) is typed per kind (``missing_policy``): windowed
  kinds mark every output whose window holds a NaN as MISSING_INPUT;
  recursive kinds carry their state without consuming the sample (the
  Kalman kinds run the prediction step only) and mark it MISSING_INPUT.
  Nothing is forward-filled. +/-inf is refused.

Negative controls: ``centered_mean_oracle`` is NON_CAUSAL, has no
incremental path and its batch transform refuses unless
``oracle_mode=True``; ``trailing_median`` window 5 carries the label
PREVIOUSLY_LAB_REJECTED_CONTROL.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, signal

ARTIFACT_SCHEMA = "df_operator_artifact.v1"
STATE_SCHEMA = "df_operator_state.v1"
FIT_ROLE = "train"

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


class OperatorRefusal(Exception):
    """Typed refusal; every message starts with REFUSED."""

    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


class OperatorAbstain(OperatorRefusal):
    """The artifact is a typed ABSTAIN: no output exists for it."""


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
    "wavelet_haar_atrous": {"levels": (int, (2, 3)),
                            "threshold_k": (float, (3.0,))},
    "causal_decomposition": {"period": (int, (5, 24)),
                             "trend_alpha": (float, (0.1,)),
                             "season_alpha": (float, (0.1,))},
    "centered_mean_oracle": {"window": (int, (5,))},
}
KINDS = tuple(BANK_GRID)
WINDOWED_KINDS = ("identity", "trailing_mean", "trailing_median",
                  "trailing_hampel", "fir_sinc_lowpass",
                  "wavelet_haar_atrous")
RECURSIVE_KINDS = ("ewma", "local_level_kalman",
                   "local_linear_trend_kalman", "butterworth2_lowpass",
                   "causal_decomposition")
NON_CAUSAL_KINDS = ("centered_mean_oracle",)
CAUSAL_KINDS = WINDOWED_KINDS + RECURSIVE_KINDS

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
                        "noise_ratio_identified_on_train": True}},
    "local_linear_trend_kalman": {
        "linearity": "LTI_STEADY_STATE", "missing_policy": _KALMAN_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True, "gaussian_noise": True,
                        "random_walk_level_and_slope": True,
                        "impulses_preserved": False,
                        "noise_ratios_identified_on_train": True}},
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
                        "fallback_scale_identified_on_train": True}},
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
    "wavelet_haar_atrous": {
        "linearity": "NONLINEAR", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"stationary_noise": True,
                        "gaussian_detail_noise": True,
                        "impulses_preserved": False,
                        "threshold_sigma_identified_on_train": True,
                        "trailing_haar_frontier": True}},
    "causal_decomposition": {
        "linearity": "LINEAR_PERIODICALLY_TIME_VARYING",
        "missing_policy": _CARRY_MISSING,
        "outputs": ["denoised", "trend", "seasonal", "residual"],
        "assumptions": {"stationary_noise": True,
                        "additive_seasonality": True,
                        "fixed_declared_period": True,
                        "phase_origin_is_stream_row_0": True,
                        "impulses_preserved": False}},
    "centered_mean_oracle": {
        "linearity": "LTI_NON_CAUSAL", "missing_policy": _WINDOW_MISSING,
        "outputs": ["y"],
        "assumptions": {"uses_future_samples": True,
                        "deployable": False}},
}


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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
    """Algorithmic lookback derived from the spec. Same values as
    causal_operators.derived_lookback for the shared kinds."""
    if kind == "identity":
        return 0
    if kind in ("trailing_mean", "trailing_median", "trailing_hampel"):
        return int(params["window"]) - 1
    if kind == "fir_sinc_lowpass":
        return int(params["taps"]) - 1
    if kind == "wavelet_haar_atrous":
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


def warmup_length(kind: str, params: dict) -> int:
    """Rows from stream start typed WARMUP (windowed kinds need a full
    window; recursive kinds initialize at the first finite sample)."""
    lb = derived_lookback(kind, params)
    if kind in WINDOWED_KINDS:
        return lb
    if kind == "centered_mean_oracle":
        return (int(params["window"]) - 1) // 2
    return 0


def control_label(spec: dict):
    if spec["kind"] == "centered_mean_oracle":
        return NON_CAUSAL_NEGATIVE_CONTROL
    if spec["kind"] == "trailing_median" and spec["params"]["window"] == 5:
        return PREVIOUSLY_LAB_REJECTED_CONTROL
    return None


def _meta(spec: dict) -> dict:
    kind, params = spec["kind"], spec["params"]
    m = copy.deepcopy(KIND_META[kind])
    m.update({
        "causality": ("NON_CAUSAL" if kind in NON_CAUSAL_KINDS
                      else "CAUSAL"),
        "control_label": control_label(spec),
        "lookback": derived_lookback(kind, params),
        "look_ahead": look_ahead(kind, params),
        "warmup": warmup_length(kind, params),
        "future_shift_compensation": 0,
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
        raise OperatorRefusal(f"{where}: NaN refused (train must be "
                              "complete; missingness is typed only at "
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


def _fit_kind(kind: str, params: dict, x: np.ndarray) -> dict:
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
    if kind == "wavelet_haar_atrous":
        L = 2 ** params["levels"]
        W = sliding_window_view(x, L, axis=0)            # (T', V, L)
        c = W
        lam = np.empty((V, params["levels"]))
        for j in range(params["levels"]):
            s = 2 ** j
            cn = (c[..., s:] + c[..., :-s]) * 0.5
            d = c[..., -1] - cn[..., -1]                 # (T', V)
            sigma = np.median(np.abs(d - np.median(d, axis=0)),
                              axis=0) / 0.6745
            if (sigma <= 0).any():
                return _abstain("ZERO_DETAIL_MAD: wavelet noise sigma not "
                                "identifiable at level "
                                f"{j + 1}")
            lam[:, j] = params["threshold_k"] * sigma
            c = cn
        return {"thresholds": lam.tolist(), "sigma_estimator":
                "MAD/0.6745 per level on train details"}
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
        phase = np.arange(T) % P
        S = np.stack([d[phase == p].mean(axis=0) for p in range(P)],
                     axis=1)                             # (V, P)
        S = S - S.mean(axis=1, keepdims=True)
        return {"seasonal_init": S.tolist(),
                "phase_origin": "row 0 of the train array and of every "
                                "transformed stream"}
    return {}


def _seal(doc: dict) -> dict:
    body = {k: doc[k] for k in doc if k != "artifact_sha256"}
    doc["artifact_sha256"] = _sha(body)
    return doc


def fit(spec: dict, train_array, role: str) -> dict:
    validate_spec(spec)
    if role != FIT_ROLE:
        raise OperatorRefusal(f"fit on role {role!r}; only 'train' is "
                              "licensed")
    x = _real_array(train_array, "train array", allow_nan=False)
    if x.ndim != 2 or x.shape[1] < 1:
        raise OperatorRefusal("train array must be 2-D (T, V)")
    if x.shape[0] < _MIN_TRAIN_ROWS:
        raise OperatorRefusal(f"train array needs >= {_MIN_TRAIN_ROWS} "
                              "rows")
    fitted = _fit_kind(spec["kind"], spec["params"], x)
    status, reason = "FITTED", None
    if "__abstain__" in fitted:
        status, reason, fitted = "ABSTAIN", fitted["__abstain__"], {}
    doc = {"schema": ARTIFACT_SCHEMA, "spec": copy.deepcopy(spec),
           "status": status, "abstain_reason": reason,
           "fit_role": FIT_ROLE, "fit_rows": int(x.shape[0]),
           "n_columns": int(x.shape[1]), "fitted": fitted,
           "meta": _meta(spec), "code_sha256": code_sha256()}
    return _seal(doc)


_ARTIFACT_KEYS = {"schema", "spec", "status", "abstain_reason",
                  "fit_role", "fit_rows", "n_columns", "fitted", "meta",
                  "code_sha256", "artifact_sha256"}


def verify_artifact(fitted) -> dict:
    if not isinstance(fitted, dict) or set(fitted) != _ARTIFACT_KEYS:
        raise OperatorRefusal("artifact keys are not the exact schema")
    if fitted["schema"] != ARTIFACT_SCHEMA:
        raise OperatorRefusal("unknown artifact schema")
    validate_spec(fitted["spec"])
    if fitted["fit_role"] != FIT_ROLE:
        raise OperatorRefusal("artifact was not fitted on the train role")
    if fitted["meta"] != _meta(fitted["spec"]):
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
    if fitted["status"] == "ABSTAIN":
        raise OperatorAbstain(f"ABSTAIN artifact has no output: "
                              f"{fitted['abstain_reason']}")
    return fitted


# ----------------------------------------------------------------------
# windowed kinds: evaluation on a window tensor (..., V, L)
# ----------------------------------------------------------------------
def _window_len(spec: dict) -> int:
    return derived_lookback(spec["kind"], spec["params"]) + 1


def _soft(d, lam):
    return np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)


def _window_eval(fitted: dict, W: np.ndarray) -> np.ndarray:
    kind, p, f = (fitted["spec"]["kind"], fitted["spec"]["params"],
                  fitted["fitted"])
    with np.errstate(invalid="ignore"):
        if kind == "identity":
            return W[..., -1].copy()
        if kind == "trailing_mean":
            return np.mean(W, axis=-1)
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
            h = np.asarray(f["taps"])
            return W @ h[::-1]          # y_t = sum_k h[k] x[t-k]
        if kind == "wavelet_haar_atrous":
            lam = np.asarray(f["thresholds"])            # (V, J)
            c = W
            acc = np.zeros(W.shape[:-1])
            for j in range(p["levels"]):
                s = 2 ** j
                cn = (c[..., s:] + c[..., :-s]) * 0.5    # past-only pairs
                acc = acc + _soft(c[..., -1] - cn[..., -1], lam[:, j])
                c = cn
            return c[..., -1] + acc
    raise OperatorRefusal(f"kind {kind!r} is not windowed")


# ----------------------------------------------------------------------
# recursive kinds: one shared update over V columns (used by batch and
# step alike, so batch/incremental parity is bitwise)
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
        return {"trend": nan.copy(),
                "seasonal": np.asarray(fitted["fitted"]["seasonal_init"],
                                       dtype=np.float64).copy()}
    raise OperatorRefusal(f"kind {kind!r} has no state")


def _rec_update(fitted: dict, pay: dict, t: int, x: np.ndarray) -> dict:
    """Consume row x (V,) at stream row t; mutate pay; return outputs."""
    kind, p, f = (fitted["spec"]["kind"], fitted["spec"]["params"],
                  fitted["fitted"])
    fin = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        if kind == "ewma":
            a = p["alpha"]
            lvl = pay["level"]
            init = np.isnan(lvl)
            new = np.where(init, x, a * x + (1.0 - a) * lvl)
            pay["level"] = np.where(fin, new, lvl)
            return {"y": np.where(fin, pay["level"], np.nan)}
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
            return {"y": np.where(fin, pay["level"], np.nan)}
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
            return {"y": np.where(fin, pay["level"], np.nan)}
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
            return {"y": np.where(fin, y, np.nan)}
        if kind == "causal_decomposition":
            P, aT, aS = p["period"], p["trend_alpha"], p["season_alpha"]
            ph = t % P
            S = pay["seasonal"]
            s_prev = S[:, ph].copy()            # learned from the past
            tr = pay["trend"]
            des = x - s_prev
            tr_new = np.where(np.isnan(tr), des,
                              tr + aT * (des - tr))
            s_new = s_prev + aS * ((x - tr_new) - s_prev)
            pay["trend"] = np.where(fin, tr_new, tr)
            S[:, ph] = np.where(fin, s_new, s_prev)
            nanv = np.full_like(x, np.nan)
            return {"denoised": np.where(fin, tr_new + s_prev, nanv),
                    "trend": np.where(fin, tr_new, nanv),
                    "seasonal": np.where(fin, s_prev, nanv),
                    "residual": np.where(fin, x - tr_new - s_prev, nanv)}
    raise OperatorRefusal(f"kind {kind!r} is not recursive")


# ----------------------------------------------------------------------
# batch
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


def transform_batch_components(fitted: dict, X,
                               oracle_mode: bool = False) -> tuple:
    """-> (outputs: dict name -> (T, V), available, reason)."""
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
    for t in range(T):
        o = _rec_update(fitted, pay, t, x[t])
        for n in names:
            outs[n][t] = o[n]
    avail = np.isfinite(x)
    reason = np.where(avail, AVAILABLE, MISSING_INPUT).astype(
        _REASON_DTYPE)
    return outs, avail, reason


def transform_batch(fitted: dict, X, oracle_mode: bool = False) -> tuple:
    """-> (Y (T, V), available (T, V) bool, reason (T, V) str)."""
    outs, avail, reason = transform_batch_components(fitted, X,
                                                     oracle_mode)
    return outs[fitted["meta"]["outputs"][0]], avail, reason


# ----------------------------------------------------------------------
# incremental
# ----------------------------------------------------------------------
def init_state(fitted: dict) -> dict:
    _usable(fitted)
    if fitted["spec"]["kind"] in NON_CAUSAL_KINDS:
        raise OperatorRefusal("NON_CAUSAL oracle has no incremental path")
    return {"schema": STATE_SCHEMA,
            "artifact_sha256": fitted["artifact_sha256"],
            "kind": fitted["spec"]["kind"], "t": 0,
            "payload": _fresh_payload(fitted)}


def step_components(fitted: dict, state: dict, x_t) -> tuple:
    """-> (outputs dict name -> (V,), available (V,), reason (V,),
    state). The state is updated in place and returned. The artifact
    digest is verified at init_state/load_state; step checks the
    binding digest only (cheap)."""
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise OperatorRefusal("not a df_operator state")
    if state["artifact_sha256"] != fitted.get("artifact_sha256"):
        raise OperatorRefusal("state belongs to a different artifact")
    kind = fitted["spec"]["kind"]
    x = _real_array(x_t, "step x_t", allow_nan=True)
    if x.shape != (fitted["n_columns"],):
        raise OperatorRefusal(f"x_t must have shape "
                              f"({fitted['n_columns']},)")
    t = state["t"]
    pay = state["payload"]
    fin = np.isfinite(x)
    if kind in WINDOWED_KINDS:
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
        outs = _rec_update(fitted, pay, t, x)
        avail = fin
        reason = np.where(fin, AVAILABLE, MISSING_INPUT).astype(
            _REASON_DTYPE)
    state["t"] = t + 1
    return outs, avail, reason, state


def step(fitted: dict, state: dict, x_t) -> tuple:
    """-> (y_t (V,), available_t (V,), reason_t (V,), state)."""
    outs, avail, reason, state = step_components(fitted, state, x_t)
    return outs[fitted["meta"]["outputs"][0]], avail, reason, state


# ----------------------------------------------------------------------
# state persistence
# ----------------------------------------------------------------------
def _enc(a: np.ndarray):
    return [None if not math.isfinite(v) else float(v)
            for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def _payload_shapes(fitted: dict) -> dict:
    return {k: v.shape for k, v in _fresh_payload(fitted).items()}


def save_state(state: dict) -> bytes:
    if not isinstance(state, dict) or set(state) != {
            "schema", "artifact_sha256", "kind", "t", "payload"}:
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
            "kind": state["kind"], "t": int(state["t"]), "payload": pay}
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
    if not isinstance(doc, dict) or set(doc) != {
            "schema", "artifact_sha256", "kind", "t", "payload",
            "state_sha256"}:
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
    t = doc["t"]
    if type(t) is not int or t < 0:
        raise OperatorRefusal("state t invalid")
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
        elif k == "seasonal" and np.isnan(arr).any():
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
            "kind": doc["kind"], "t": t, "payload": payload}
