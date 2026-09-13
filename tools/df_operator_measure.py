"""Measurement of D2 operators (C135), reported as separate quantities.

* algorithmic lookback: derived from the spec (``derived_lookback``);
* group delay and phase delay: only for kinds with a transfer function
  (LTI, or LTI in steady state for the recursive kinds initialized at
  the first sample and for the Kalman filters at their steady-state
  gain), at declared normalized frequencies in cycles/sample, DC
  included. For non-LTI kinds they are ``UNDEFINED`` and the empirical
  impulse-response peak lag and step-response 50 % rise lag are reported
  instead (these are also reported for LTI kinds as a cross-check);
* computational latency: mean and p95 wall seconds per ``step``;
* CPU seconds of a batch transform and its tracemalloc peak bytes;
* warm-up length (declared, observed, and 99 % impulse-energy settling
  for kinds with a transfer function).

"No delay" means zero look-ahead. Delay is measured, never compensated
by shifting outputs with future samples; ``anticipation_max_abs``
reports any response before the stimulus and is 0 for causal kinds.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
from scipy import signal


def _load_operators():
    if "df_operators" in sys.modules:
        return sys.modules["df_operators"]
    path = Path(__file__).with_name("df_operators.py")
    spec = importlib.util.spec_from_file_location("df_operators", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["df_operators"] = mod
    spec.loader.exec_module(mod)
    return mod


ops = _load_operators()

DECLARED_FREQS = (0.0, 0.01, 0.02, 0.05)      # cycles/sample
UNDEFINED = "UNDEFINED"


def _steady_state_gain_ll(q: float, r: float) -> float:
    var = r
    k = 0.0
    for _ in range(100000):
        vp = var + q
        k_new = vp / (vp + r)
        var_new = (1.0 - k_new) * vp
        if abs(var_new - var) <= 1e-15 * max(1.0, var):
            return k_new
        var, k = var_new, k_new
    return k


def transfer_function(fitted: dict):
    """-> (b, a, basis) in powers of z^-1, or None when the kind has no
    transfer function. Column 0 is used for per-column fitted kinds."""
    kind = fitted["spec"]["kind"]
    p, f = fitted["spec"]["params"], fitted["fitted"]
    if fitted["status"] != "FITTED":
        return None
    if kind == "identity":
        return np.array([1.0]), np.array([1.0]), "LTI"
    if kind == "trailing_mean":
        w = p["window"]
        return np.ones(w) / w, np.array([1.0]), "LTI"
    if kind == "fir_sinc_lowpass":
        return np.asarray(f["taps"]), np.array([1.0]), "LTI"
    if kind == "ewma":
        a = p["alpha"]
        return (np.array([a]), np.array([1.0, -(1.0 - a)]),
                "LTI_STEADY_STATE")
    if kind == "butterworth2_lowpass":
        return (np.asarray(f["b"]), np.asarray(f["a"]),
                "LTI_STEADY_STATE")
    if kind == "local_level_kalman":
        c = f["per_column"][0]
        k = _steady_state_gain_ll(c["level_var"], c["obs_var"])
        return (np.array([k]), np.array([1.0, -(1.0 - k)]),
                "STEADY_STATE_KALMAN_GAIN")
    if kind == "local_linear_trend_kalman":
        c = f["per_column"][0]
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([c["level_var"], c["slope_var"]])
        r = c["obs_var"]
        P = np.diag([r, r])
        K = np.zeros((2, 1))
        for _ in range(100000):
            Pp = F @ P @ F.T + Q
            K_new = Pp @ H.T / (H @ Pp @ H.T + r)
            P_new = (np.eye(2) - K_new @ H) @ Pp
            done = np.max(np.abs(P_new - P)) <= 1e-15 * max(
                1.0, float(np.max(np.abs(P))))
            P, K = P_new, K_new
            if done:
                break
        A = (np.eye(2) - K @ H) @ F        # s_{t+1} = A s_t + K x_t
        num, den = signal.ss2tf(A, K, H @ A, H @ K)
        return (np.asarray(num).reshape(-1), np.asarray(den),
                "STEADY_STATE_KALMAN_GAIN")
    return None


def lti_delays(b, a, freqs=DECLARED_FREQS) -> dict:
    freqs = np.asarray(freqs, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, gd = signal.group_delay((b, a), w=freqs, fs=1.0)
        dense = np.unique(np.concatenate(
            [np.linspace(0.0, max(float(freqs.max()), 1e-3), 4097),
             freqs]))
        _, H = signal.freqz(b, a, worN=dense, fs=1.0)
    phase = np.unwrap(np.angle(H))
    phase = phase - 2 * np.pi * np.round(phase[0] / (2 * np.pi))
    out_gd, out_pd = [], []
    for fr, g in zip(freqs, gd):
        out_gd.append(float(g) if np.isfinite(g) else None)
        if fr == 0.0:
            # phase delay limit at DC equals the DC group delay when the
            # DC gain is real positive
            out_pd.append(float(g) if np.isfinite(g) else None)
        else:
            ph = float(phase[np.searchsorted(dense, fr)])
            out_pd.append(float(-ph / (2 * np.pi * fr)))
    dc_gain = float(np.sum(b) / np.sum(a))
    return {"freqs_cycles_per_sample": [float(v) for v in freqs],
            "group_delay_samples": out_gd,
            "phase_delay_samples": out_pd,
            "dc_gain": dc_gain,
            "phase_delay_at_dc": "LIMIT_EQUALS_DC_GROUP_DELAY"}


def empirical_responses(fitted: dict, amplitude: float = 1.0,
                        t0: int = 64, horizon: int = 256,
                        oracle_mode: bool = False) -> dict:
    """Differential response y(stimulus) - y(zero baseline), column 0.
    For non-linear kinds the result depends on the declared amplitude."""
    V = fitted["n_columns"]
    N = t0 + horizon
    base = np.zeros((N, V))
    imp = base.copy()
    imp[t0] = amplitude
    stp = base.copy()
    stp[t0:] = amplitude
    y0, _, _ = ops.transform_batch(fitted, base, oracle_mode=oracle_mode)
    yi, _, _ = ops.transform_batch(fitted, imp, oracle_mode=oracle_mode)
    ys, _, _ = ops.transform_batch(fitted, stp, oracle_mode=oracle_mode)
    di_raw = yi[:, 0] - y0[:, 0]
    ds_raw = ys[:, 0] - y0[:, 0]
    di = np.nan_to_num(di_raw)
    ds = np.nan_to_num(ds_raw)
    last_valid = np.flatnonzero(np.isfinite(ds_raw))
    antic = float(max(np.max(np.abs(di[:t0])), np.max(np.abs(ds[:t0]))))
    post = di[t0:]
    if np.max(np.abs(post)) <= 1e-12:
        peak = "NO_RESPONSE"
    else:
        peak = int(np.argmax(np.abs(post)))
    # the oracle anticipates: the earliest response lag can be negative
    nz = np.flatnonzero(np.abs(di) > 1e-12)
    first_lag = int(nz[0] - t0) if len(nz) else "NO_RESPONSE"
    final = ds[last_valid[-1]] if len(last_valid) else 0.0
    if abs(final) <= 1e-12:
        rise = "NO_RESPONSE"
    else:
        hit = np.flatnonzero(ds / final >= 0.5)
        rise = int(hit[0] - t0) if len(hit) else "NO_RESPONSE"
    return {"amplitude": float(amplitude), "t0": t0, "horizon": horizon,
            "impulse_peak_lag": peak, "impulse_first_response_lag":
            first_lag, "step_rise_50pct_lag": rise,
            "anticipation_max_abs": antic}


def settling_99(b, a, n: int = 4096):
    x = np.zeros(n)
    x[0] = 1.0
    h = signal.lfilter(b, a, x)
    e = np.cumsum(h * h)
    if e[-1] <= 0:
        return None
    return int(np.searchsorted(e, 0.99 * e[-1]))


def cost(fitted: dict, latency_samples: int = 2000,
         batch_rows: int = 4000, seed: int = 0) -> dict:
    V = fitted["n_columns"]
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(batch_rows, V)), axis=0)
    oracle = fitted["spec"]["kind"] in ops.NON_CAUSAL_KINDS
    c0 = time.process_time()
    ops.transform_batch(fitted, X, oracle_mode=oracle)
    cpu = time.process_time() - c0
    tracemalloc.start()
    try:
        ops.transform_batch(fitted, X, oracle_mode=oracle)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    out = {"batch_rows": batch_rows, "batch_cpu_seconds": float(cpu),
           "batch_peak_bytes": int(peak),
           "latency_samples": latency_samples}
    if oracle:
        out.update({"step_mean_seconds": "REFUSED_NON_CAUSAL",
                    "step_p95_seconds": "REFUSED_NON_CAUSAL"})
        return out
    st = ops.init_state(fitted)
    xs = X[:latency_samples] if latency_samples <= batch_rows else \
        np.cumsum(rng.normal(size=(latency_samples, V)), axis=0)
    dt = np.empty(latency_samples)
    for i in range(latency_samples):
        s = time.perf_counter()
        ops.step(fitted, st, xs[i])
        dt[i] = time.perf_counter() - s
    out.update({"step_mean_seconds": float(dt.mean()),
                "step_p95_seconds": float(np.percentile(dt, 95))})
    return out


def future_access_invariant(fitted: dict, X, cuts, seed: int = 0,
                            oracle_mode: bool = False) -> bool:
    """True iff for every cut t, replacing X[t+1:] leaves Y[:t+1],
    available[:t+1] and reason[:t+1] bitwise unchanged."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    y, a, r = ops.transform_batch(fitted, X, oracle_mode=oracle_mode)
    for t in cuts:
        X2 = X.copy()
        X2[t + 1:] = rng.normal(loc=50.0, scale=25.0,
                                size=X2[t + 1:].shape)
        y2, a2, r2 = ops.transform_batch(fitted, X2,
                                         oracle_mode=oracle_mode)
        if not (np.array_equal(y[:t + 1], y2[:t + 1], equal_nan=True)
                and np.array_equal(a[:t + 1], a2[:t + 1])
                and np.array_equal(r[:t + 1], r2[:t + 1])):
            return False
    return True


def measure(fitted: dict, freqs=DECLARED_FREQS, latency_samples=2000,
            batch_rows=4000, amplitude=1.0) -> dict:
    ops.verify_artifact(fitted)
    spec = fitted["spec"]
    meta = fitted["meta"]
    rep = {"spec": spec, "status": fitted["status"],
           "causality": meta["causality"],
           "control_label": meta["control_label"],
           "algorithmic_lookback": ops.derived_lookback(spec["kind"],
                                                        spec["params"]),
           "look_ahead": meta["look_ahead"],
           "future_shift_compensation": meta["future_shift_compensation"]}
    if fitted["status"] != "FITTED":
        rep["abstain_reason"] = fitted["abstain_reason"]
        return rep
    tf = transfer_function(fitted)
    oracle = spec["kind"] in ops.NON_CAUSAL_KINDS
    if tf is None:
        rep["delay_basis"] = "NON_LTI"
        rep["group_delay"] = UNDEFINED
        rep["phase_delay"] = UNDEFINED
        rep["settling_99_samples"] = UNDEFINED
    else:
        b, a, basis = tf
        rep["delay_basis"] = basis
        rep["lti"] = lti_delays(b, a, freqs)
        rep["group_delay"] = rep["lti"]["group_delay_samples"]
        rep["phase_delay"] = rep["lti"]["phase_delay_samples"]
        rep["settling_99_samples"] = settling_99(b, a)
    if oracle:
        rep["delay_basis"] = "NON_CAUSAL"
    rep["empirical"] = empirical_responses(fitted, amplitude,
                                           oracle_mode=oracle)
    rep["cost"] = cost(fitted, latency_samples, batch_rows)
    probe = np.cumsum(np.random.default_rng(1).normal(size=(200, fitted[
        "n_columns"])), axis=0)
    _, av, _ = ops.transform_batch(fitted, probe, oracle_mode=oracle)
    first = np.flatnonzero(av[:, 0])
    rep["warmup"] = {"declared": meta["warmup"],
                     "observed_first_available_row":
                         int(first[0]) if len(first) else None}
    return rep


def _demo_train(n=600, V=2, seed=7):
    rng = np.random.default_rng(seed)
    lvl = np.cumsum(rng.normal(scale=0.3, size=(n, V)), axis=0)
    t = np.arange(n)[:, None]
    return lvl + 0.8 * np.sin(2 * np.pi * t / 24) + rng.normal(size=(n, V))


def main():
    train = _demo_train()
    rows = []
    for spec in ops.bank_specs():
        fitted = ops.fit(spec, train, "train")
        rep = measure(fitted, latency_samples=300, batch_rows=1000)
        rows.append(rep)
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
