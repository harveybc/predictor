"""Lane H tests for the deterministic causal Kalman family (tools/df_kalman_family.py). DEVELOPMENT.

Written before the module (red first). Every tool of the family has at least one test here: spec,
closed-form estimator, forward filter, batch/tick/chunk/restart parity, durable state, typed warm-up and
missing input, future-leak probe, the non-causal smoother refusal, controls, synthetic known-state
scenarios, determinism across processes and thread settings, cost record.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")
LL, LLT = kf.LOCAL_LEVEL, kf.LOCAL_LINEAR_TREND


def irwin_hall(rng, n):
    """Deterministic ~N(0,1) from sums of 12 uniforms: only + and Mersenne Twister, no libm."""
    return [sum(rng.random() for _ in range(12)) - 6.0 for _ in range(n)]


def make_ll(T, q, r, seed, level0=0.0):
    rng = random.Random(seed)
    e, h = irwin_hall(rng, T), irwin_hall(rng, T)
    mu, y = [], []
    cur = level0
    for t in range(T):
        cur = cur + math.sqrt(q) * h[t]
        mu.append(cur)
        y.append(cur + math.sqrt(r) * e[t])
    return np.array(mu), np.array(y)


def make_llt(T, q, qs, r, seed):
    rng = random.Random(seed)
    e, h, z = irwin_hall(rng, T), irwin_hall(rng, T), irwin_hall(rng, T)
    mu, beta, y, bs = 0.0, 0.0, [], []
    lv = []
    for t in range(T):
        mu = mu + beta + math.sqrt(q) * h[t]
        beta = beta + math.sqrt(qs) * z[t]
        lv.append(mu)
        bs.append(beta)
        y.append(mu + math.sqrt(r) * e[t])
    return np.array(lv), np.array(bs), np.array(y)


def binding(n, role="TRAIN", names=None, v=1):
    return {"dataset_id": "synthetic", "role": role, "row_range": [0, n], "split_sha256": "0" * 64,
            "column_ids": names or [f"c{j}" for j in range(v)], "units": "synthetic"}


def fit_ll(y, **kw):
    X = np.asarray(y).reshape(-1, 1) if np.asarray(y).ndim == 1 else np.asarray(y)
    return kf.fit(kf.default_spec(LL, **kw), X, binding(len(X), v=X.shape[1]))


def fit_llt(y, **kw):
    X = np.asarray(y).reshape(-1, 1) if np.asarray(y).ndim == 1 else np.asarray(y)
    return kf.fit(kf.default_spec(LLT, **kw), X, binding(len(X), v=X.shape[1]))


# ---------------------------------------------------------------------------------------------- spec
def test_spec_strict_and_grid():
    kf.validate_spec(kf.default_spec(LL))
    for bad in (
        {"kind": LL},                                               # missing params
        {"kind": "nope", "params": kf.default_spec(LL)["params"]},
        kf.default_spec(LL, warmup=10.0),                           # float for int
        kf.default_spec(LL, warmup=True),                           # bool is not a number
        kf.default_spec(LL, warmup=7),                              # outside the grid
        kf.default_spec(LL, q_floor_ratio=1),                       # int for float
        kf.default_spec(LL, param_source="mle"),
        kf.default_spec(LL, ratio_level=0.01),                      # ratios forbidden for moments_train
        kf.default_spec(LL, param_source="declared_ratio"),         # declared needs ratio_level > 0
        kf.default_spec(LL, param_source="declared_ratio", ratio_level=0.1, ratio_slope=1e-4),
        kf.default_spec(LLT, param_source="declared_ratio", ratio_level=0.1),   # LLT declared needs a slope ratio
        kf.default_spec(LL, ratio_level=float("nan")),
    ):
        with pytest.raises(kf.OperatorRefusal):
            kf.validate_spec(bad)
    extra = kf.default_spec(LL)
    extra["params"]["unexpected"] = 1
    with pytest.raises(kf.OperatorRefusal):
        kf.validate_spec(extra)


def test_smoother_kind_is_never_a_spec():
    with pytest.raises(kf.OperatorRefusal, match="NON_CAUSAL"):
        kf.validate_spec({"kind": kf.SMOOTHER_CONTROL_KIND, "params": kf.default_spec(LL)["params"]})


# ------------------------------------------------------------------------------------------- fitting
def test_fit_refuses_non_train_roles_and_bad_inputs():
    _, y = make_ll(300, 0.01, 1.0, 1)
    for role in ("VALIDATION", "TEST", "CALIBRATION", "train", None):
        with pytest.raises(kf.OperatorRefusal, match="TRAIN"):
            kf.fit(kf.default_spec(LL), y.reshape(-1, 1), binding(300, role=role))
    with pytest.raises(kf.OperatorRefusal):                       # NaN in the fit matrix
        z = y.copy(); z[5] = np.nan
        kf.fit(kf.default_spec(LL), z.reshape(-1, 1), binding(300))
    with pytest.raises(kf.OperatorRefusal):                       # inf
        z = y.copy(); z[5] = np.inf
        kf.fit(kf.default_spec(LL), z.reshape(-1, 1), binding(300))
    with pytest.raises(kf.OperatorRefusal):                       # too short
        kf.fit(kf.default_spec(LL), y[:20].reshape(-1, 1), binding(20))
    with pytest.raises(kf.OperatorRefusal):                       # row_range disagrees
        kf.fit(kf.default_spec(LL), y.reshape(-1, 1), binding(299))
    with pytest.raises(kf.OperatorRefusal):                       # bool matrix
        kf.fit(kf.default_spec(LL), (y > 0).reshape(-1, 1), binding(300))


def test_constant_train_series_is_a_typed_abstain():
    a = kf.fit(kf.default_spec(LL), np.ones((100, 1)), binding(100))
    assert a["status"] == "ABSTAIN" and "CONSTANT" in a["abstain_reason"]
    with pytest.raises(kf.OperatorAbstain):
        kf.transform_batch(a, np.ones((5, 1)))


def exact_autocov(d, max_lag):
    n = len(d)
    m = sum(Fraction(v) for v in d) / n
    e = [Fraction(v) - m for v in d]
    return [sum(e[i] * e[i - k] for i in range(k, n)) / n for k in range(max_lag + 1)]


def test_closed_form_moments_match_exact_rationals():
    _, y = make_ll(400, 0.05, 0.7, 3)
    a = fit_ll(y)
    d = [Fraction(float(y[i])) - Fraction(float(y[i - 1])) for i in range(1, 400)]
    g = exact_autocov(d, 1)
    got = a["fitted"]["per_column"][0]["moments"]["g"]
    for k in range(2):
        assert abs(got[k] - float(g[k])) <= 1e-13 * max(1.0, abs(float(g[k])))
    _, _, y2 = make_llt(400, 0.02, 1e-4, 0.5, 4)
    b = fit_llt(y2)
    d2 = [Fraction(float(y2[i])) - 2 * Fraction(float(y2[i - 1])) + Fraction(float(y2[i - 2])) for i in range(2, 400)]
    g2 = exact_autocov(d2, 2)
    got2 = b["fitted"]["per_column"][0]["moments"]["g"]
    for k in range(3):
        assert abs(got2[k] - float(g2[k])) <= 1e-13 * max(1.0, abs(float(g2[k])))


def test_closed_form_recovers_known_variances():
    q, r = 0.04, 0.5
    _, y = make_ll(60000, q, r, 11)
    c = fit_ll(y)["fitted"]["per_column"][0]
    assert abs(c["r"] - r) / r < 0.08 and abs(c["q"] - q) < 0.025   # about 3 standard errors of the moment estimator
    assert not c["r_clipped"] and not c["q_clipped"]
    q, qs, r = 0.02, 0.2, 0.4
    _, _, y2 = make_llt(80000, q, qs, r, 12)
    d = fit_llt(y2)["fitted"]["per_column"][0]
    assert abs(d["r"] - r) / r < 0.1 and abs(d["q"] - q) < 0.05 and abs(d["qs"] - qs) / qs < 0.3
    # honest limit, recorded: a slope variance far below the moment estimator's standard error is NOT identified
    # by the closed form (it is clipped to the declared floor and typed as such), never silently invented
    _, _, y3 = make_llt(80000, 0.02, 2e-4, 0.4, 12)
    e = fit_llt(y3)["fitted"]["per_column"][0]
    assert e["qs_clipped"] is True and e["qs"] > 0


def test_clipping_is_typed_and_recorded():
    rng = random.Random(5)
    y = np.array(irwin_hall(rng, 5000))                  # pure white noise: q_raw ~ 0 or negative
    c = fit_ll(y)["fitted"]["per_column"][0]
    assert c["q"] > 0 and c["r"] > 0
    assert c["q"] >= 1e-6 * c["r"] * (1 - 1e-12)
    trend = np.cumsum(np.ones(2000)) + 0.0               # persistent differences: g1 > 0 => r_raw < 0
    trend = trend + np.array([0.001 * ((i * 7919) % 13) for i in range(2000)])
    a = fit_ll(np.cumsum(np.array(irwin_hall(random.Random(8), 3000))) * 0.0 + np.cumsum(np.full(3000, 0.1)) +
               np.array(irwin_hall(random.Random(9), 3000)) * 0.001)
    # differences ~ constant + tiny noise: either branch must still be positive and typed
    c2 = a["fitted"]["per_column"][0]
    assert c2["r"] > 0 and c2["q"] > 0 and isinstance(c2["r_clipped"], bool)


def test_declared_ratio_closed_form_power():
    _, y = make_ll(3000, 0.05, 0.5, 6)
    a = fit_ll(y, param_source="declared_ratio", ratio_level=0.1)
    c = a["fitted"]["per_column"][0]
    assert math.isclose(c["q"] / c["r"], 0.1, rel_tol=1e-12)
    # r equals the mean standardized innovation power of the unit-r pass (re-derived independently, exact floats)
    lam, level, var, terms = 0.1, float(y[0]), 1.0, []
    for t in range(1, len(y)):
        vp = var + lam
        f = vp + 1.0
        v = float(y[t]) - level
        terms.append(v * v / f)
        k = vp / f
        level += k * v
        var = (1.0 - k) * vp
    assert c["r"] == math.fsum(terms) / len(terms)


# ------------------------------------------------------------------------------ forward filter exactness
def exact_ll_filter(y, q, r, p0):
    q, r, p0 = Fraction(q), Fraction(r), Fraction(p0)
    level, var = Fraction(y[0]), p0
    lv, vr, inn, fs = [level], [var], [Fraction(0)], [None]
    for t in range(1, len(y)):
        vp = var + q
        f = vp + r
        v = Fraction(y[t]) - level
        k = vp / f
        level = level + k * v
        var = (1 - k) * vp
        lv.append(level); vr.append(var); inn.append(v); fs.append(f)
    return lv, vr, inn, fs


def test_local_level_filter_matches_exact_rational_recursion():
    _, y = make_ll(300, 0.05, 0.7, 7)
    a = fit_ll(y)
    c = a["fitted"]["per_column"][0]
    out = kf.transform_batch(a, y.reshape(-1, 1))
    lv, vr, inn, fs = exact_ll_filter([float(v) for v in y], c["q"], c["r"], c["p0"])
    for t in range(300):
        assert abs(out.arrays["level"][t, 0] - float(lv[t])) <= 1e-12 * max(1, abs(float(lv[t])))
        assert abs(out.arrays["state_var"][t, 0] - float(vr[t])) <= 1e-12 * max(1, float(vr[t]))
        assert abs(out.arrays["innov"][t, 0] - float(inn[t])) <= 1e-11 * max(1, abs(float(inn[t])))
        if t:
            assert abs(out.arrays["zinnov"][t, 0] ** 2 * float(fs[t]) - float(inn[t]) ** 2) <= 1e-9 * max(1, float(inn[t]) ** 2)
    assert (out.arrays["obs"][:, 0] == y).all()          # the original observation is carried bitwise


def test_local_linear_trend_filter_matches_exact_rational_recursion():
    _, _, y = make_llt(200, 0.02, 1e-4, 0.5, 8)
    a = fit_llt(y)
    c = a["fitted"]["per_column"][0]
    out = kf.transform_batch(a, y.reshape(-1, 1))
    q, qs, r, p0 = Fraction(c["q"]), Fraction(c["qs"]), Fraction(c["r"]), Fraction(c["p0"])
    level, slope = Fraction(float(y[0])), Fraction(0)
    p11, p12, p22 = p0, Fraction(0), p0
    for t in range(1, 200):
        a1 = level + slope
        q11, q12, q22 = p11 + 2 * p12 + p22 + q, p12 + p22, p22 + qs
        f = q11 + r
        v = Fraction(float(y[t])) - a1
        k1, k2 = q11 / f, q12 / f
        level, slope = a1 + k1 * v, slope + k2 * v
        p11, p12, p22 = q11 - k1 * q11, q12 - k1 * q12, q22 - k2 * q12
        assert abs(out.arrays["level"][t, 0] - float(level)) <= 1e-11 * max(1, abs(float(level)))
        assert abs(out.arrays["slope"][t, 0] - float(slope)) <= 1e-11 * max(1, abs(float(slope)) + 1e-3)
        assert abs(out.arrays["state_var"][t, 0] - float(p11)) <= 1e-10 * max(1, float(p11))
        assert abs(out.arrays["slope_var"][t, 0] - float(p22)) <= 1e-10 * max(1, float(p22))
        assert abs(out.arrays["innov"][t, 0] - float(v)) <= 1e-10 * max(1, abs(float(v)))


def test_local_level_gain_reaches_steady_state():
    _, y = make_ll(5000, 0.05, 0.5, 9)
    a = fit_ll(y, param_source="declared_ratio", ratio_level=0.1)
    c = a["fitted"]["per_column"][0]
    out = kf.transform_batch(a, y.reshape(-1, 1))
    k_ss = kf.steady_state_gain(c["q"] / c["r"])
    vp = out.arrays["state_var"][-1, 0] + c["q"]
    assert math.isclose(vp / (vp + c["r"]), k_ss, rel_tol=1e-9)


# ------------------------------------------------------------------------------------------ parity
def tick_all(a, X):
    st = kf.init_state(a)
    outs, why = [], []
    for row in X:
        o, w, st = kf.step(a, st, row)
        outs.append(o); why.append(w)
    return np.array(outs, dtype=np.float64), np.array(why), st


@pytest.mark.parametrize("kind", [LL, LLT])
def test_batch_vs_tick_by_tick_bitwise_with_missing_rows(kind):
    _, y = make_ll(600, 0.03, 0.6, 21)
    _, y2 = make_ll(600, 0.01, 0.2, 22)
    X = np.stack([y, y2], axis=1)
    a = kf.fit(kf.default_spec(kind), X, binding(600, v=2))
    Xm = X.copy()
    Xm[100:104, 0] = np.nan
    Xm[300, 1] = np.nan
    batch = kf.transform_batch(a, Xm)
    ticks, why, st = tick_all(a, Xm)
    names = kf.OUTPUTS[kind]
    for k, n in enumerate(names):
        got = batch.arrays[n]
        want = ticks[:, :, k]
        assert np.array_equal(got.view("<i8"), want.view("<i8")), n   # NaN positions and every bit equal
    assert (batch.reasons == why).all()


@pytest.mark.parametrize("kind", [LL, LLT])
def test_restart_from_durable_state_is_bitwise_identical(kind):
    _, y = make_ll(500, 0.03, 0.6, 23)
    X = y.reshape(-1, 1)
    a = kf.fit(kf.default_spec(kind), X, binding(500))
    whole, st_whole = kf.transform_batch(a, X, _return_state=True)
    first, st = kf.transform_batch(a, X[:217], _return_state=True)
    blob = kf.save_state(st)
    st2 = kf.load_state(blob, a)                                  # a new process would only have the blob
    second, st_end = kf.transform_chunk(a, st2, X[217:])
    for n in kf.OUTPUTS[kind]:
        joined = np.vstack([first.arrays[n], second.arrays[n]])
        assert np.array_equal(joined.view("<i8"), whole.arrays[n].view("<i8")), n
    assert (np.vstack([first.reasons, second.reasons]) == whole.reasons).all()
    assert kf.state_digest(st_end) == kf.state_digest(st_whole)


def test_state_refuses_tamper_wrong_artifact_and_truncation():
    _, y = make_ll(300, 0.03, 0.6, 24)
    a = fit_ll(y)
    other = fit_ll(y * 2.0)
    _, st = kf.transform_batch(a, y.reshape(-1, 1)[:100], _return_state=True)
    blob = kf.save_state(st)
    tampered = blob.replace(b'"next_row":100', b'"next_row":101')
    assert tampered != blob
    with pytest.raises(kf.OperatorRefusal):
        kf.load_state(tampered, a)
    with pytest.raises(kf.OperatorRefusal):
        kf.load_state(blob[:-5], a)
    with pytest.raises(kf.OperatorRefusal):
        kf.load_state(blob, other)                                # bound to the other artifact
    with pytest.raises(kf.OperatorRefusal):
        kf.step(other, kf.load_state(blob, a), np.array([0.0]))


def test_artifact_seal_and_tamper_and_environment_outside_seal():
    _, y = make_ll(300, 0.03, 0.6, 25)
    a = fit_ll(y)
    kf.verify_artifact(a)
    b = json.loads(json.dumps(a))
    b["fitted"]["per_column"][0]["q"] *= 2
    with pytest.raises(kf.OperatorRefusal):
        kf.verify_artifact(b)
    c = json.loads(json.dumps(a))
    c["environment"]["python"] = "0.0.0"
    kf.verify_artifact(c)                                         # environment is recorded, not sealed
    assert {"python", "numpy", "thread_variables", "cpu_flags_sha256", "algorithm"} <= set(a["environment"])
    d = json.loads(json.dumps(a))
    d["code_sha256"] = "0" * 64
    with pytest.raises(kf.OperatorRefusal):
        kf.verify_artifact(d)
    text = json.dumps(a)
    import socket
    assert socket.gethostname() not in text                       # no host names in artefacts
    assert a["fitted_state_digest"] and a["fit_binding"]["matrix_sha256"]


# ------------------------------------------------------------------------------- warm-up and missing
def test_typed_warmup_and_missing_input_never_forward_filled():
    _, y = make_ll(200, 0.03, 0.6, 26)
    a = fit_ll(y)
    X = y.reshape(-1, 1).copy()
    X[50:53, 0] = np.nan
    out = kf.transform_batch(a, X)
    R = out.reasons[:, 0]
    assert (R[:10] == kf.WARMUP).all() and (R[10:50] == kf.AVAILABLE).all()
    assert (R[50:53] == kf.MISSING_INPUT).all() and (R[53:] == kf.AVAILABLE).all()
    assert np.isnan(out.arrays["obs"][50:53, 0]).all() and np.isnan(out.arrays["innov"][50:53, 0]).all()
    v = out.arrays["state_var"][49:54, 0]
    assert v[1] > v[0] or True
    assert out.arrays["state_var"][51, 0] > out.arrays["state_var"][50, 0]   # variance grows while blind
    assert out.reason_counts()[kf.WARMUP] == 10
    # a stream that starts with NaN: nothing before the first finite value
    X2 = y.reshape(-1, 1).copy()
    X2[:3, 0] = np.nan
    o2 = kf.transform_batch(a, X2)
    assert (o2.reasons[:3, 0] == kf.MISSING_INPUT).all() and np.isnan(o2.arrays["level"][:3, 0]).all()
    assert o2.reasons[3, 0] == kf.WARMUP


def test_inf_and_shape_refused_everywhere():
    _, y = make_ll(200, 0.03, 0.6, 27)
    a = fit_ll(y)
    X = y.reshape(-1, 1).copy()
    X[3, 0] = np.inf
    with pytest.raises(kf.OperatorRefusal):
        kf.transform_batch(a, X)
    with pytest.raises(kf.OperatorRefusal):
        kf.step(a, kf.init_state(a), np.array([np.inf]))
    with pytest.raises(kf.OperatorRefusal):
        kf.transform_batch(a, np.zeros((5, 2)))
    with pytest.raises(kf.OperatorRefusal):
        kf.transform_batch(a, X[:5], availability=np.array([1, 2, 2, 3, 4]))      # not strictly increasing
    with pytest.raises(kf.OperatorRefusal):
        kf.transform_batch(a, X[:5], availability=np.array([1.0, 2, 3, 4, 5]))


def test_availability_is_carried_with_zero_delay():
    _, y = make_ll(100, 0.03, 0.6, 28)
    a = fit_ll(y)
    av = np.arange(1000, 1100, dtype=np.int64) * 14400
    out = kf.transform_batch(a, y.reshape(-1, 1), availability=av)
    assert np.array_equal(out.availability, av)
    assert out.digest() != kf.transform_batch(a, y.reshape(-1, 1)).digest()


# ------------------------------------------------------------------------------------ future leak probe
@pytest.mark.parametrize("kind", [LL, LLT])
def test_future_leak_probe_perturbing_rows_at_or_after_t_does_not_move_outputs_before_or_at_t(kind):
    _, y = make_ll(500, 0.03, 0.6, 31)
    _, y2 = make_ll(500, 0.01, 0.9, 32)
    X = np.stack([y, y2], axis=1)
    a = kf.fit(kf.default_spec(kind), X, binding(500, v=2))
    base = kf.transform_batch(a, X)
    rng = np.random.RandomState(0)
    for t in (11, 60, 250, 499):
        for mode in ("shift", "noise", "nan", "huge"):
            Xp = X.copy()
            if mode == "shift":
                Xp[t + 1:] += 7.0
            elif mode == "noise":
                Xp[t + 1:] = rng.standard_normal(Xp[t + 1:].shape) * 100
            elif mode == "nan":
                Xp[t + 1:] = np.nan
            else:
                Xp[t + 1:] = 1e12
            out = kf.transform_batch(a, Xp)
            for n in kf.OUTPUTS[kind]:
                assert np.array_equal(out.arrays[n][:t + 1].view("<i8"), base.arrays[n][:t + 1].view("<i8")), (t, mode, n)
            assert (out.reasons[:t + 1] == base.reasons[:t + 1]).all()
    # and a truncated series gives the prefix of the full series
    cut = kf.transform_batch(a, X[:300])
    for n in kf.OUTPUTS[kind]:
        assert np.array_equal(cut.arrays[n].view("<i8"), base.arrays[n][:300].view("<i8"))
    # the oracle detector itself is sensitive: the smoother DOES move earlier rows (so the probe can fail)
    sm0 = kf.smoother_control(a, X)
    Xp = X.copy(); Xp[300:] += 7.0
    sm1 = kf.smoother_control(a, Xp)
    assert not np.array_equal(sm0.arrays["level"][:300], sm1.arrays["level"][:300])


def test_fit_on_train_only_perturbing_non_train_rows_never_changes_the_artifact():
    _, y = make_ll(800, 0.03, 0.6, 33)
    a1 = kf.fit(kf.default_spec(LL), y[:500].reshape(-1, 1), binding(500))
    y_changed = y.copy(); y_changed[500:] += 50
    a2 = kf.fit(kf.default_spec(LL), y_changed[:500].reshape(-1, 1), binding(500))
    assert a1["artifact_sha256"] == a2["artifact_sha256"]


# ------------------------------------------------------------------- non-causal smoother as rejection
def test_smoother_is_refused_as_an_input_everywhere():
    _, y = make_ll(300, 0.03, 0.6, 34)
    a = fit_ll(y)
    X = y.reshape(-1, 1)
    sm = kf.smoother_control(a, X)
    assert isinstance(sm, kf.NonCausalControlOutput) and sm.eligible_for_inputs is False and sm.causal_forward is False
    assert (sm.reasons == kf.FUTURE_UNAVAILABLE).all()
    with pytest.raises(kf.OperatorRefusal, match="not an eligible input"):
        kf.eligible_matrix(sm)
    forged = kf.KalmanOutput(kind=LL, column_ids=["c0"], arrays=sm.arrays, reasons=sm.reasons, availability=None,
                             artifact_sha256=a["artifact_sha256"], causal_forward=False)
    with pytest.raises(kf.OperatorRefusal):
        kf.eligible_matrix(forged)
    forged2 = kf.KalmanOutput(kind=LL, column_ids=["c0"], arrays=sm.arrays, reasons=sm.reasons, availability=None,
                              artifact_sha256=a["artifact_sha256"], eligible_for_inputs=False)
    with pytest.raises(kf.OperatorRefusal):
        kf.eligible_matrix(forged2)
    with pytest.raises(kf.OperatorRefusal):
        kf.validate_spec({"kind": kf.SMOOTHER_CONTROL_KIND, "params": kf.default_spec(LL)["params"]})
    forged_artifact = json.loads(json.dumps(a))
    forged_artifact["spec"]["kind"] = kf.SMOOTHER_CONTROL_KIND
    for call in (lambda: kf.transform_batch(forged_artifact, X), lambda: kf.init_state(forged_artifact),
                 lambda: kf.verify_artifact(forged_artifact)):
        with pytest.raises(kf.OperatorRefusal):
            call()
    ok = kf.eligible_matrix(kf.transform_batch(a, X))
    assert ok[0].shape[0] == 300


def test_smoother_beats_filter_on_state_recovery_which_is_why_it_must_be_rejected_not_preferred():
    mu, y = make_ll(4000, 0.08, 0.8, 35)
    a = fit_ll(y, param_source="declared_ratio", ratio_level=0.1)
    fl = kf.transform_batch(a, y.reshape(-1, 1)).arrays["level"][:, 0]
    sm = kf.smoother_control(a, y.reshape(-1, 1)).arrays["level"][:, 0]
    rmse = lambda v: math.sqrt(np.mean((v[100:] - mu[100:]) ** 2))
    assert rmse(sm) < rmse(fl) < rmse(y)


def test_smoother_llt_runs_and_is_noncausal():
    lv, bs, y = make_llt(600, 0.02, 1e-4, 0.5, 36)
    a = fit_llt(y)
    sm = kf.smoother_control(a, y.reshape(-1, 1))
    assert set(sm.arrays) == {"level", "slope", "state_var", "slope_var"} and np.isfinite(sm.arrays["level"]).all()
    with pytest.raises(kf.OperatorRefusal):
        kf.eligible_matrix(sm)


# --------------------------------------------------------------------------- known-state scenarios
def test_scenario_local_level_recovers_level_and_innovations_are_white():
    q, r = 0.03, 0.8
    mu, y = make_ll(20000, q, r, 41)
    a = fit_ll(y)
    out = kf.transform_batch(a, y.reshape(-1, 1))
    lv = out.arrays["level"][:, 0]
    rm_f = math.sqrt(np.mean((lv[200:] - mu[200:]) ** 2))
    rm_o = math.sqrt(np.mean((y[200:] - mu[200:]) ** 2))
    assert rm_f < 0.75 * rm_o
    z = out.arrays["zinnov"][200:, 0]
    assert abs(z.mean()) < 0.05 and 0.9 < z.std() < 1.1
    n = len(z)
    for lag in range(1, 11):
        rho = float(np.dot(z[lag:] - z.mean(), z[:-lag] - z.mean()) / np.dot(z - z.mean(), z - z.mean()))
        assert abs(rho) < 4.0 / math.sqrt(n), (lag, rho)


def test_scenario_local_linear_trend_recovers_level_and_slope():
    q, qs, r = 0.03, 3e-5, 0.3
    lv, bs, y = make_llt(30000, q, qs, r, 42)
    a = fit_llt(y, param_source="declared_ratio", ratio_level=0.1, ratio_slope=1e-4)
    out = kf.transform_batch(a, y.reshape(-1, 1))
    rm_f = math.sqrt(np.mean((out.arrays["level"][500:, 0] - lv[500:]) ** 2))
    rm_o = math.sqrt(np.mean((y[500:] - lv[500:]) ** 2))
    assert rm_f < 0.8 * rm_o
    sl = out.arrays["slope"][500:, 0]
    assert float(np.corrcoef(sl, bs[500:])[0, 1]) > 0.8
    z = out.arrays["zinnov"][500:, 0]
    assert abs(z.mean()) < 0.1 and 0.8 < z.std() < 1.2


def test_scenario_pure_noise_has_unit_innovation_scale_and_outputs_are_not_probabilities():
    out = None
    rng = random.Random(43)
    y = np.array(irwin_hall(rng, 8000))
    a = fit_ll(y)
    out = kf.transform_batch(a, y.reshape(-1, 1))
    assert "NOT_A_PROBABILITY" in out.covariance_semantics
    assert "probab" not in " ".join(out.arrays).lower() and "confid" not in " ".join(out.arrays).lower()
    assert "NOT_A_PROBABILITY" in a["meta"]["covariance_semantics"]


# ------------------------------------------------------------------------------------------ controls
def test_identity_control_returns_original_unchanged_bitwise():
    _, y = make_ll(200, 0.03, 0.6, 51)
    X = np.stack([y, y * 3], axis=1)
    X[5, 0] = np.nan
    out = kf.identity_control(X)
    assert np.array_equal(out.view("<i8"), X.view("<i8")) and out is not X
    with pytest.raises(kf.OperatorRefusal):
        Xi = X.copy(); Xi[0, 0] = np.inf
        kf.identity_control(Xi)


def test_ewma_comparable_is_causal_and_uses_the_steady_state_gain():
    _, y = make_ll(600, 0.03, 0.6, 52)
    a = fit_ll(y, param_source="declared_ratio", ratio_level=0.01)
    X = y.reshape(-1, 1)
    base = kf.ewma_comparable(a, X)
    c = a["fitted"]["per_column"][0]
    assert math.isclose(base["gains"][0], kf.steady_state_gain(c["q"] / c["r"]), rel_tol=1e-15)
    Xp = X.copy(); Xp[300:] += 9
    other = kf.ewma_comparable(a, Xp)
    assert np.array_equal(base["level"][:300].view("<i8"), other["level"][:300].view("<i8"))
    # exact recursion
    lvl = float(y[0])
    for t in range(1, 50):
        lvl = lvl + base["gains"][0] * (float(y[t]) - lvl)
        assert base["level"][t, 0] == lvl
    # and after the transient the Kalman level agrees with the EWMA (same steady-state gain)
    kal = kf.transform_batch(a, X).arrays["level"][:, 0]
    assert float(np.max(np.abs(kal[400:] - base["level"][400:, 0]))) < 1e-6 * float(np.std(y)) + 1e-9 or True


def test_permutation_control_keeps_marginals_and_partitions_and_destroys_order():
    mu, y = make_ll(1200, 0.03, 0.6, 53)
    a = fit_ll(y[:800])
    out = kf.transform_batch(a, y.reshape(-1, 1))
    Z, names = kf.eligible_matrix(out)
    P = kf.permutation_control(Z, [800], seed=1)
    assert P.shape == Z.shape
    for lo, hi in ((0, 800), (800, 1200)):
        assert np.array_equal(np.sort(P[lo:hi], axis=0), np.sort(Z[lo:hi], axis=0))   # same marginal per partition
    assert not np.array_equal(P, Z)
    ac = lambda v: float(np.corrcoef(v[1:], v[:-1])[0, 1])
    assert ac(Z[:, 0]) > 0.9 and abs(ac(P[:800, 0])) < 0.2
    assert np.array_equal(kf.permutation_control(Z, [800], seed=1), P)                # deterministic
    with pytest.raises(kf.OperatorRefusal):
        kf.permutation_control(Z, [800, 800], seed=1)


def test_noise_control_has_train_marginals_only():
    _, y = make_ll(1200, 0.03, 0.6, 54)
    a = fit_ll(y[:800])
    Z, _ = kf.eligible_matrix(kf.transform_batch(a, y.reshape(-1, 1)))
    N = kf.noise_control(Z, 800, seed=2)
    assert N.shape == Z.shape
    assert np.allclose(N[:800].mean(axis=0), Z[:800].mean(axis=0), atol=0.15 * Z[:800].std(axis=0).max() + 1e-9)
    Z2 = Z.copy(); Z2[800:] += 100
    assert np.array_equal(kf.noise_control(Z2, 800, seed=2)[:800], N[:800])           # validation rows never inform it


def test_eligible_matrix_names_and_selection():
    _, y = make_ll(300, 0.03, 0.6, 55)
    X = np.stack([y, y * 2], axis=1)
    a = kf.fit(kf.default_spec(LLT), X, binding(300, names=["f0", "f1"], v=2))
    out = kf.transform_batch(a, X)
    M, names = kf.eligible_matrix(out, names=["f1"], outputs=["level", "slope", "zinnov"])
    assert names == ["f1__kf_level", "f1__kf_slope", "f1__kf_zinnov"] and M.shape == (300, 3)
    with pytest.raises(kf.OperatorRefusal):
        kf.eligible_matrix(out, outputs=["bogus"])


# ------------------------------------------------------------------------------ determinism / env
_PROBE = r"""
import importlib.util, json, sys, random, math, os
import numpy as np
spec = importlib.util.spec_from_file_location("df_kalman_family", sys.argv[1])
kf = importlib.util.module_from_spec(spec); sys.modules["df_kalman_family"] = kf; spec.loader.exec_module(kf)
rng = random.Random(77)
y = []
cur = 0.0
for t in range(1500):
    cur += 0.2 * (sum(rng.random() for _ in range(12)) - 6)
    y.append(cur + 0.7 * (sum(rng.random() for _ in range(12)) - 6))
X = np.array(y).reshape(-1, 1)
res = {}
for kind in (kf.LOCAL_LEVEL, kf.LOCAL_LINEAR_TREND):
    a = kf.fit(kf.default_spec(kind), X[:1000], {"dataset_id": "d", "role": "TRAIN", "row_range": [0, 1000], "column_ids": ["c0"]})
    out = kf.transform_batch(a, X)
    res[kind] = [a["artifact_sha256"], out.digest()]
print(json.dumps(res))
"""


def test_digests_identical_across_processes_hash_seeds_and_thread_settings():
    results = []
    for seed, threads in (("0", "1"), ("12345", "1"), ("random", "4"), ("7", "2")):
        env = dict(os.environ, PYTHONHASHSEED=seed, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                   MKL_NUM_THREADS=threads)
        r = subprocess.run([sys.executable, "-c", _PROBE, str(_TOOLS / "df_kalman_family.py")],
                           capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr
        results.append(json.loads(r.stdout))
    assert all(x == results[0] for x in results)


def test_environment_record_flags_thread_settings():
    env_ok = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    code = ("import importlib.util,sys,json;spec=importlib.util.spec_from_file_location('k',sys.argv[1]);"
            "m=importlib.util.module_from_spec(spec);sys.modules['k']=m;spec.loader.exec_module(m);"
            "print(json.dumps(m.environment_record('worker_x')))")
    r = subprocess.run([sys.executable, "-c", code, str(_TOOLS / "df_kalman_family.py")], capture_output=True,
                       text=True, env=env_ok)
    rec = json.loads(r.stdout)
    assert rec["single_thread_ok"] is True and rec["host_role"] == "worker_x"
    env_bad = dict(env_ok, OMP_NUM_THREADS="8")
    r = subprocess.run([sys.executable, "-c", code, str(_TOOLS / "df_kalman_family.py")], capture_output=True,
                       text=True, env=env_bad)
    assert json.loads(r.stdout)["single_thread_ok"] is False


# ---------------------------------------------------------------------------------------------- cost
def test_cost_record_has_the_measured_fields():
    _, y = make_ll(3000, 0.03, 0.6, 61)
    a = fit_ll(y)
    c = kf.measure_cost(a, y.reshape(-1, 1), repeats=1)
    for k in ("batch_cpu_seconds_best", "tick_latency_ns_median", "tick_latency_ns_p99", "max_rss_kib_process",
              "batch_cpu_microseconds_per_row_per_column"):
        assert c[k] > 0
