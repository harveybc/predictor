"""Tests for the D2 causal operator bank (C135 contract, C136 bank) and its
snapshot boundary (C152-C155). Numerics are exercised through the private
kernels; the public API is exercised through contracts and snapshots."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ops = _load("df_operators")
meas = _load("df_operator_measure")
snapm = _load("df_snapshot")
sync = _load("df_synthetic_contract")
bat = _load("df_causal_battery")

ALL_SPECS = ops.bank_specs()
CAUSAL_SPECS = [s for s in ALL_SPECS if s["kind"] in ops.CAUSAL_KINDS]
ORACLE_SPEC = {"kind": "centered_mean_oracle", "params": {"window": 5}}
FROZEN, EXPANDING = ops.FROZEN_PREVIOUS_PARTITION, ops.EXPANDING_PREFIX


def _sid(s):
    return s["kind"] + "-" + "-".join(f"{k}{v}" for k, v in
                                      sorted(s["params"].items()))


def _series(n, V=2, seed=0):
    rng = np.random.default_rng(seed)
    lvl = np.cumsum(rng.normal(scale=0.3, size=(n, V)), axis=0)
    t = np.arange(n)[:, None]
    return lvl + 0.8 * np.sin(2 * np.pi * t / 24) + rng.normal(size=(n, V))


TRAIN = _series(300, seed=1)
TEST = _series(240, seed=2)
_FIT_CACHE = {}


def _fitted(spec):
    key = json.dumps(spec, sort_keys=True)
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = ops._fit_kernel(spec, TRAIN)
    f = _FIT_CACHE[key]
    assert f["status"] == "FITTED", f["abstain_reason"]
    return copy.deepcopy(f)


def _same(a, b):
    """Bitwise equality; NaN == NaN for float arrays only."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def _with_nans(X):
    X = X.copy()
    X[40, 0] = np.nan
    X[100:104, 1] = np.nan
    X[0, 1] = np.nan
    return X


# ------------------------------------------------------------ spec ----
def test_bank_grid_is_the_predeclared_one():
    counts = {}
    for s in ALL_SPECS:
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
    assert counts == {"identity": 1, "ewma": 3, "local_level_kalman": 1,
                      "local_linear_trend_kalman": 1, "trailing_mean": 2,
                      "trailing_median": 3, "trailing_hampel": 2,
                      "fir_sinc_lowpass": 2, "butterworth2_lowpass": 2,
                      "trailing_haar_threshold": 2, "causal_decomposition": 2,
                      "centered_mean_oracle": 1}
    for s in ALL_SPECS:
        assert ops.validate_spec(s) is s


BAD_SPECS = [
    None, [], {"kind": "ewma"},
    {"kind": "ewma", "params": {"alpha": 0.1}, "extra": 1},
    {"kind": "nope", "params": {}},
    {"kind": "ewma", "params": {}},
    {"kind": "ewma", "params": {"alpha": 0.1, "beta": 0.2}},
    {"kind": "ewma", "params": {"alpha": 1}},              # int for float
    {"kind": "ewma", "params": {"alpha": "0.1"}},
    {"kind": "ewma", "params": {"alpha": float("nan")}},
    {"kind": "ewma", "params": {"alpha": 0.2}},            # off grid
    {"kind": "trailing_mean", "params": {"window": True}},  # bool as int
    {"kind": "trailing_mean", "params": {"window": 5.0}},  # float for int
    {"kind": "trailing_mean", "params": {"window": 7}},    # off grid
    {"kind": "trailing_hampel", "params": {"window": 9, "k": 2.0}},
    {"kind": "fir_sinc_lowpass", "params": {"cutoff": 0.1, "taps": 31}},
    {"kind": "butterworth2_lowpass", "params": {"cutoff": float("inf")}},
    {"kind": "causal_decomposition",
     "params": {"period": 5, "trend_alpha": 0.1}},
    {"kind": "centered_mean_oracle", "params": {"window": 7}},
    {"kind": "identity", "params": {"shift": -1}},
]


@pytest.mark.parametrize("bad", BAD_SPECS)
def test_spec_validation_refuses(bad):
    with pytest.raises(ops.OperatorRefusal, match="^REFUSED"):
        ops.validate_spec(bad)
    with pytest.raises(ops.OperatorRefusal):
        ops._fit_kernel(bad, TRAIN)


def test_no_spec_parameter_can_shift_or_lead():
    for kind, grid in ops.BANK_GRID.items():
        for name in grid:
            assert not any(tok in name for tok in
                           ("shift", "lead", "offset", "advance", "delay"))
    for s in CAUSAL_SPECS:
        m = _fitted(s)["meta"]
        assert m["look_ahead"] == 0
        assert m["future_shift_compensation"] == 0
        assert m["causality"] == "CAUSAL"


def test_lookback_matches_derivation():
    want = {"identity": 0, "ewma": -1, "local_level_kalman": -1,
            "local_linear_trend_kalman": -1, "butterworth2_lowpass": -1,
            "causal_decomposition": -1, "centered_mean_oracle": -2}
    for s in ALL_SPECS:
        k, p = s["kind"], s["params"]
        lb = ops.derived_lookback(k, p)
        if k in want:
            assert lb == want[k]
        elif k in ("trailing_mean", "trailing_median", "trailing_hampel"):
            assert lb == p["window"] - 1
        elif k == "fir_sinc_lowpass":
            assert lb == 20
        elif k == "trailing_haar_threshold":
            assert lb == 2 ** p["levels"] - 1
    assert ops.LOOKBACK_UNBOUNDED == -1 and ops.LOOKBACK_NON_CAUSAL == -2


def test_controls_are_labelled():
    f = ops._fit_kernel(ORACLE_SPEC, TRAIN)
    assert f["meta"]["causality"] == "NON_CAUSAL"
    assert f["meta"]["control_label"] == ops.NON_CAUSAL_NEGATIVE_CONTROL
    med5 = _fitted({"kind": "trailing_median", "params": {"window": 5}})
    assert med5["meta"]["control_label"] == \
        ops.PREVIOUSLY_LAB_REJECTED_CONTROL
    med9 = _fitted({"kind": "trailing_median", "params": {"window": 9}})
    assert med9["meta"]["control_label"] is None


def test_every_kind_declares_assumptions_and_fit_modes_as_data():
    for s in ALL_SPECS:
        a = ops.KIND_META[s["kind"]]["assumptions"]
        assert isinstance(a, dict) and a
        assert all(type(v) is bool for v in a.values())
        modes = ops.KIND_FIT_MODES[s["kind"]]
        assert ops.FROZEN_PREVIOUS_PARTITION in modes and ops.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL in modes
        assert (ops.EXPANDING_PREFIX in modes) == (s["kind"] in ops.DATA_INDEPENDENT_KINDS
                                                   or s["kind"] == "causal_decomposition")


# ------------------------------------------------------------- fit ----
def _contract(X=None, **kw):
    X = _series(100, seed=5) if X is None else X
    return sync.in_memory_contract(X, **kw)


@pytest.mark.parametrize("role", ["validation", "test", "Train", "train", "", None])
def test_fit_refuses_a_role_string_or_a_bare_array(role):
    spec = {"kind": "ewma", "params": {"alpha": 0.1}}
    with pytest.raises(ops.OperatorRefusal, match="requires a FitSnapshot"):
        ops.fit(spec, role, FROZEN)
    with pytest.raises(ops.OperatorRefusal, match="requires a FitSnapshot"):
        ops.fit(spec, TRAIN, FROZEN)
    c, L = _contract()
    with pytest.raises(snapm.SnapshotRefusal, match="unknown role"):
        snapm.FitSnapshot.from_contract(c, role, L)


def test_fit_refuses_calibration_or_confirmation_rows_under_train():
    spec = {"kind": "ewma", "params": {"alpha": 0.1}}
    c, L = _contract()
    with pytest.raises(ops.OperatorRefusal, match="not inside the TRAIN partition"):
        ops.fit(spec, snapm.FitSnapshot.from_contract(c, "TRAIN", L, end=100), FROZEN)
    with pytest.raises(ops.OperatorRefusal, match="not allowed by the design"):
        ops.fit(spec, snapm.FitSnapshot.from_contract(c, "CONFIRMATION", L), FROZEN)
    c, L = _contract(_series(300, seed=6))            # a calibration partition of 60 rows
    f = ops.fit(spec, snapm.FitSnapshot.from_contract(c, "CALIBRATION", L), FROZEN)
    assert f["fit_binding"]["role"] == "CALIBRATION" and f["fit_binding"]["range"] == [180, 240]
    assert f["fit_binding"]["licensed_partitions"] == ["CONFIRMATION"]


def test_a_hand_built_snapshot_refuses():
    c, L = _contract()
    s = snapm.FitSnapshot.from_contract(c, "TRAIN", L)
    with pytest.raises(snapm.SnapshotRefusal, match="built only by from_contract"):
        snapm.FitSnapshot(**{k: getattr(s, k) for k in s.__dataclass_fields__ if k != "token"}, token=None)


def test_fit_refuses_bad_train_arrays():
    s = {"kind": "local_level_kalman", "params": {}}
    bad = [TRAIN[:10], TRAIN[:, 0], TRAIN.astype(bool), TRAIN.astype(str),
           np.where(np.arange(300)[:, None] == 3, np.nan, TRAIN),
           np.where(np.arange(300)[:, None] == 3, np.inf, TRAIN)]
    for b in bad:
        with pytest.raises(ops.OperatorRefusal):
            ops._fit_kernel(s, b)


def test_artifact_binds_what_it_was_fitted_on():
    c, L = _contract()
    s = snapm.FitSnapshot.from_contract(c, "TRAIN", L)
    f = ops.fit({"kind": "trailing_mean", "params": {"window": 5}}, s, FROZEN)
    b = f["fit_binding"]
    assert b["snapshot_sha256"] == s.snapshot_sha256 and b["matrix_sha256"] == s.matrix_sha256
    assert b["dataset_id"] == c["dataset_id"] and b["contract_sha256"] == c["contract_sha256"]
    assert b["range"] == [0, 60] and b["role"] == "TRAIN" and b["licensed_min_index"] == 60
    assert b["excluded_partitions"] == [["CALIBRATION", 60, 80], ["CONFIRMATION", 80, 100]]
    ops.verify_artifact(f)
    g = copy.deepcopy(f)
    g["fit_binding"]["range"] = [0, 100]
    with pytest.raises(ops.OperatorRefusal, match="digest"):
        ops.verify_artifact(g)


@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_fit_is_digest_bound(spec):
    f = _fitted(spec)
    again = ops._fit_kernel(spec, TRAIN)
    assert again == f                                     # deterministic
    before = json.dumps(f, sort_keys=True)
    ops._transform_kernel(f, TEST * 10.0 + 3.0)            # other data
    assert json.dumps(f, sort_keys=True) == before         # no refit
    body = {k: f[k] for k in f if k != "artifact_sha256"}
    assert ops._sha(body) == f["artifact_sha256"]
    if f["fitted"]:
        tampered = copy.deepcopy(f)
        first = next(iter(tampered["fitted"]))
        tampered["fitted"][first] = "x"
        with pytest.raises(ops.OperatorRefusal, match="digest"):
            ops._transform_kernel(tampered, TEST)
    tampered = copy.deepcopy(f)
    tampered["meta"]["warmup"] = 999                # understate/overstate
    with pytest.raises(ops.OperatorRefusal):
        ops._kernel_init_state(tampered)


def test_fitted_parameters_depend_on_fit_rows_only():
    s = {"kind": "trailing_hampel", "params": {"window": 9, "k": 3.0}}
    a = ops._fit_kernel(s, TRAIN)
    b = ops._fit_kernel(s, TRAIN * 2.0)
    assert np.allclose(np.array(b["fitted"]["fallback_scale"]),
                       2.0 * np.array(a["fitted"]["fallback_scale"]))
    X = _series(500, seed=8)
    X2 = X.copy()
    X2[300:] += 100.0
    c1, L1 = _contract(X)
    c2, L2 = _contract(X2)
    f1 = ops.fit(s, snapm.FitSnapshot.from_contract(c1, "TRAIN", L1), FROZEN)
    f2 = ops.fit(s, snapm.FitSnapshot.from_contract(c2, "TRAIN", L2), FROZEN)
    assert f1["fitted"] == f2["fitted"] == ops._fit_kernel(s, X[:300])["fitted"]


def test_kalman_mle_agrees_with_statsmodels():
    sm = pytest.importorskip("statsmodels.tsa.statespace.structural")
    import warnings
    rng = np.random.default_rng(11)
    y = np.cumsum(rng.normal(scale=0.5, size=300)) + rng.normal(size=300)
    f = ops._fit_kernel({"kind": "local_level_kalman", "params": {}}, y[:, None])
    c = f["fitted"]["per_column"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.UnobservedComponents(y, "llevel").fit(disp=False)
    p = dict(zip(res.model.param_names, res.params))
    sm_lr = math.log(p["sigma2.level"] / p["sigma2.irregular"])
    assert abs(c["log_ratio"] - sm_lr) < 0.1
    assert abs(c["obs_var"] / p["sigma2.irregular"] - 1.0) < 0.1


# --------------------------------------------------------- abstain ----
def test_typed_abstain_artifacts_refuse_to_transform():
    const = np.ones((120, 2))
    cases = [
        ({"kind": "local_level_kalman", "params": {}}, const),
        ({"kind": "local_linear_trend_kalman", "params": {}}, const),
        ({"kind": "trailing_hampel", "params": {"window": 9, "k": 3.0}},
         const),
        ({"kind": "trailing_haar_threshold",
          "params": {"levels": 2, "threshold_k": 3.0}}, const),
        ({"kind": "causal_decomposition",
          "params": {"period": 24, "trend_alpha": 0.1,
                     "season_alpha": 0.1}}, TRAIN[:60]),
        # a pure random walk whose MLE reaches the ratio upper bound
        ({"kind": "local_level_kalman", "params": {}},
         np.cumsum(np.random.default_rng(0).normal(size=(300, 1)), 0)),
    ]
    for spec, train in cases:
        f = ops._fit_kernel(spec, train)
        assert f["status"] == "ABSTAIN", spec
        assert isinstance(f["abstain_reason"], str) and f["abstain_reason"]
        ops.verify_artifact(f)
        with pytest.raises(ops.OperatorAbstain, match="ABSTAIN"):
            ops._transform_kernel(f, np.zeros((10, train.shape[1])))
        with pytest.raises(ops.OperatorAbstain):
            ops._kernel_init_state(f)
        assert meas.measure(f)["status"] == "ABSTAIN"


# -------------------------------------------------- batch / step ------
def _run_steps(f, X, reload_at=None):
    st = ops._kernel_init_state(f)
    ys, av, rs = [], [], []
    for i in range(X.shape[0]):
        if reload_at is not None and i == reload_at:
            blob = ops.save_state(st)
            assert isinstance(blob, bytes)
            st = ops.load_state(blob, f)
        y, a, r, st = ops._kernel_step(f, st, X[i])
        ys.append(y.copy())
        av.append(a.copy())
        rs.append(r.copy())
    return np.array(ys), np.array(av), np.array(rs)


@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_batch_incremental_parity_is_bitwise(spec):
    f = _fitted(spec)
    for X in (TEST, _with_nans(TEST)):
        Y, A, R = ops._transform_kernel(f, X)
        ys, av, rs = _run_steps(f, X)
        assert Y.shape == X.shape and A.dtype == bool
        assert np.array_equal(A, av) and np.array_equal(R, rs)
        assert np.array_equal(np.isnan(Y), ~A)
        assert np.array_equal(Y, ys, equal_nan=True)


@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_save_reload_mid_stream_is_identical(spec):
    f = _fitted(spec)
    X = _with_nans(TEST)
    ref = _run_steps(f, X)
    for cut in (0, 3, 102, 170):
        got = _run_steps(f, X, reload_at=cut)
        for a, b in zip(ref, got):
            assert _same(a, b)


def test_decomposition_components_parity_and_identity():
    s = {"kind": "causal_decomposition",
         "params": {"period": 24, "trend_alpha": 0.1, "season_alpha": 0.1}}
    for mode in (FROZEN, EXPANDING):
        f = ops._fit_kernel(s, TRAIN, mode)
        X = _with_nans(TEST)
        outs, A, R = ops._transform_kernel_components(f, X)
        assert set(outs) == {"denoised", "trend", "seasonal", "residual"}
        fin = A
        assert np.allclose((outs["trend"] + outs["seasonal"])[fin],
                           outs["denoised"][fin], atol=0, rtol=0)
        assert np.allclose((outs["denoised"] + outs["residual"])[fin], X[fin],
                           atol=1e-12)
        Y, _, _ = ops._transform_kernel(f, X)
        assert np.array_equal(Y, outs["denoised"], equal_nan=True)
        st = ops._kernel_init_state(f)
        for i in range(X.shape[0]):
            o, a, r, st = ops._kernel_step_components(f, st, X[i])
            for n in outs:
                assert np.array_equal(o[n], outs[n][i], equal_nan=True)


def test_butterworth_matches_scipy_lfilter_with_carried_zi():
    for cutoff in (0.1, 0.25):
        f = _fitted({"kind": "butterworth2_lowpass",
                     "params": {"cutoff": cutoff}})
        b, a = signal.butter(2, 2 * cutoff)
        Y, _, _ = ops._transform_kernel(f, TEST)
        for j in range(TEST.shape[1]):
            zi = signal.lfilter_zi(b, a) * TEST[0, j]
            ref, _ = signal.lfilter(b, a, TEST[:, j], zi=zi)
            assert np.max(np.abs(ref - Y[:, j])) <= 1e-12
            y1, z = signal.lfilter(b, a, TEST[:100, j], zi=zi)
            y2, _ = signal.lfilter(b, a, TEST[100:, j], zi=z)
            assert np.max(np.abs(np.r_[y1, y2] - Y[:, j])) <= 1e-12


def test_fir_sinc_is_causal_21_tap_convolution():
    for cutoff in (0.1, 0.25):
        f = _fitted({"kind": "fir_sinc_lowpass",
                     "params": {"cutoff": cutoff, "taps": 21}})
        h = signal.firwin(21, cutoff, fs=1.0)
        Y, A, _ = ops._transform_kernel(f, TEST)
        ref = signal.lfilter(h, [1.0], TEST, axis=0)
        assert not A[:20].any() and A[20:].all()
        assert np.max(np.abs(ref[20:] - Y[20:])) <= 1e-12


def test_trailing_haar_threshold_recurrence_without_threshold_is_identity():
    s = {"kind": "trailing_haar_threshold", "params": {"levels": 3, "threshold_k": 3.0}}
    f = copy.deepcopy(_fitted(s))
    f["fitted"]["thresholds"] = [[0.0] * 3 for _ in range(2)]
    f = ops._seal(f)
    Y, A, _ = ops._transform_kernel(f, TEST)
    assert np.allclose(Y[A], TEST[A], atol=1e-12) and not A[:7].any() and A[7:].all()
    lv = ops._transform_levels_kernel(f, TEST)
    t = 50
    c1 = (TEST[t] + TEST[t - 1]) / 2
    c2 = (c1 + (TEST[t - 2] + TEST[t - 3]) / 2) / 2
    assert np.allclose(lv[0]["approx"][t], c1) and np.allclose(lv[1]["approx"][t], c2)
    assert np.allclose(lv[0]["detail"][t], TEST[t] - c1)


# ------------------------------------------------ public snapshot path ----
def _bank_contract():
    X = _with_nans(_series(500, seed=9))
    X[100:104, 1] = np.nan
    return X, *_contract(X, name="bank")


@pytest.mark.parametrize("spec", CAUSAL_SPECS + [ORACLE_SPEC], ids=_sid)
def test_snapshot_path_equals_the_kernel_bitwise(spec):
    X, c, L = _bank_contract()
    fs = snapm.FitSnapshot.from_contract(c, "TRAIN", L, start=104, end=300)   # complete stretch after the NaNs
    oracle = spec["kind"] in ops.NON_CAUSAL_KINDS
    for mode in [m for m in ops.KIND_FIT_MODES[spec["kind"]] if m != ops.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL]:
        f = ops.fit(spec, fs, mode)
        if f["status"] != "FITTED":
            continue
        start = 0 if mode == EXPANDING else 300
        ts = snapm.TransformSnapshot.from_contract(c, L, start=start, end=500)
        outs, A, R = ops.transform_batch_components(f, ts, oracle_mode=oracle)
        k = ops._fit_kernel(spec, X[104:300], mode, t0=104)
        assert k["fitted"] == f["fitted"]
        kouts, kA, kR = ops._transform_kernel_components(f, X[start:], start, oracle_mode=oracle)
        assert all(_same(outs[n], kouts[n]) for n in outs) and _same(A, kA) and _same(R, kR)
        if oracle:
            continue
        st = ops.init_state(f, ts)
        rows = [ops.step(f, st, ts.row(i)) for i in range(500 - start)]
        assert _same(np.array([r[0] for r in rows]), outs[f["meta"]["outputs"][0]])
        assert _same(np.array([r[2] for r in rows]), R)
        # fragmented public stream over two contiguous snapshots
        mid = start + 37
        st = ops.init_state(f, snapm.TransformSnapshot.from_contract(c, L, start=start, end=mid))
        y1, _, r1, st = ops.transform_chunk(f, st, snapm.TransformSnapshot.from_contract(c, L, start=start, end=mid))
        st = ops.load_state(ops.save_state(st), f)
        y2, _, r2, st = ops.transform_chunk(f, st, snapm.TransformSnapshot.from_contract(c, L, start=mid, end=500))
        assert _same(np.concatenate([y1, y2]), outs[f["meta"]["outputs"][0]])
        assert _same(np.concatenate([r1, r2]), R)


def test_public_stream_refuses_gaps_repeats_and_foreign_series():
    X, c, L = _bank_contract()
    f = ops.fit({"kind": "ewma", "params": {"alpha": 0.3}}, snapm.FitSnapshot.from_contract(c, "TRAIN", L,
                                                                                            start=104), FROZEN)
    ts = snapm.TransformSnapshot.from_contract(c, L, start=300, end=500)
    st = ops.init_state(f, ts)
    ops.step(f, st, ts.row(0))
    with pytest.raises(ops.OperatorRefusal, match="not the next row"):
        ops.step(f, st, ts.row(0))
    with pytest.raises(ops.OperatorRefusal, match="not the next row"):
        ops.step(f, st, ts.row(5))
    with pytest.raises(ops.OperatorRefusal, match="not the next row"):
        ops.transform_chunk(f, st, snapm.TransformSnapshot.from_contract(c, L, start=310, end=320))
    c2, L2 = _contract(_series(500, seed=10), name="other")
    with pytest.raises(ops.OperatorRefusal, match="another series"):
        ops.step(f, st, snapm.TransformSnapshot.from_contract(c2, L2, start=301, end=500).row(0))
    with pytest.raises(ops.OperatorRefusal, match="stepped only with TransformRows"):
        ops._kernel_step(f, st, X[301])


# ------------------------------------------------ state validation ----
def test_load_state_refuses_foreign_or_corrupt_state():
    f = _fitted({"kind": "trailing_mean", "params": {"window": 5}})
    g = _fitted({"kind": "trailing_mean", "params": {"window": 9}})
    st = ops._kernel_init_state(f)
    for i in range(7):
        _, _, _, st = ops._kernel_step(f, st, TEST[i])
    blob = ops.save_state(st)
    assert ops.load_state(blob, f)["t"] == 7
    with pytest.raises(ops.OperatorRefusal, match="different artifact"):
        ops.load_state(blob, g)
    doc = json.loads(blob)
    for mutate in (
            lambda d: d.__setitem__("t", 8),
            lambda d: d["payload"]["buffer"]["values"].__setitem__(0, 1e9),
            lambda d: d.__setitem__("extra", 1),
            lambda d: d.__setitem__("binding", {"dataset_id": "x"}),
            lambda d: d.__setitem__("schema", "other")):
        d = copy.deepcopy(doc)
        mutate(d)
        with pytest.raises(ops.OperatorRefusal):
            ops.load_state(ops._canonical(d), f)
    d = copy.deepcopy(doc)
    d["t"] = 2
    body = {k: d[k] for k in d if k != "state_sha256"}
    d["state_sha256"] = ops._sha(body)
    with pytest.raises(ops.OperatorRefusal, match="incoherent"):
        ops.load_state(ops._canonical(d), f)
    with pytest.raises(ops.OperatorRefusal):
        ops.load_state(blob.replace(b":", b": ", 1), f)
    with pytest.raises(ops.OperatorRefusal):
        ops.load_state(b"{not json", f)
    with pytest.raises(ops.OperatorRefusal):
        ops._kernel_step(g, st, TEST[7])


# -------------------------------------------------- future access -----
@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_future_access_invariance_at_every_t(spec):
    f = _fitted(spec)
    run = bat.kernel_runner(f)
    assert bat.prefix_all_t(run, TEST, range(TEST.shape[0])) == {}
    assert bat.prefix_all_t(run, _with_nans(TEST), range(TEST.shape[0])) == {}
    rng = np.random.default_rng(9)
    X2 = TEST.copy()
    X2[121:] = rng.normal(100.0, 30.0, size=X2[121:].shape)
    a = _run_steps(f, TEST[:121])
    b = _run_steps(f, X2)
    for u, v in zip(a, b):
        assert _same(u, v[:121])


def test_oracle_fails_future_access_and_refuses_without_oracle_mode():
    f = ops._fit_kernel(ORACLE_SPEC, TRAIN)
    assert f["status"] == "FITTED"
    with pytest.raises(ops.OperatorRefusal, match="oracle_mode"):
        ops._transform_kernel(f, TEST)
    with pytest.raises(ops.OperatorRefusal, match="oracle_mode"):
        ops._transform_kernel(f, TEST, oracle_mode=1)
    with pytest.raises(ops.OperatorRefusal):
        ops._kernel_init_state(f)

    def run(X):
        outs, a, r = ops._transform_kernel_components(f, X, oracle_mode=True)
        return outs, a, r
    assert bat.prefix_all_t(run, TEST, range(TEST.shape[0]))
    Y, A, R = ops._transform_kernel(f, TEST, oracle_mode=True)
    assert set(R[:2].ravel()) == {"WARMUP"}
    assert set(R[-2:].ravel()) == {"FUTURE_UNAVAILABLE"}
    assert np.allclose(Y[2:-2], (TEST[:-4] + TEST[1:-3] + TEST[2:-2]
                                 + TEST[3:-1] + TEST[4:]) / 5)
    with pytest.raises(ops.OperatorRefusal, match="oracle_mode"):
        ops._transform_kernel(_fitted({"kind": "ewma", "params": {"alpha": 0.1}}), TEST, oracle_mode=True)


# ----------------------------------------------- warm-up and NaN ------
@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_warmup_typing(spec):
    f = _fitted(spec)
    Y, A, R = ops._transform_kernel(f, TEST)
    w = f["meta"]["warmup"]
    assert w == (ops.derived_lookback(spec["kind"], spec["params"])
                 if spec["kind"] in ops.WINDOWED_KINDS else 0)
    assert set(np.unique(R[:w])) <= {"WARMUP"}
    assert not A[:w].any() and np.isnan(Y[:w]).all()
    assert A[w:].all() and np.isfinite(Y[w:]).all()
    assert set(np.unique(R)) <= set(ops.REASONS)


@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_nan_is_typed_never_forward_filled(spec):
    f = _fitted(spec)
    X = _with_nans(TEST)
    Y, A, R = ops._transform_kernel(f, X)
    nan = np.isnan(X)
    assert (R[nan] == "MISSING_INPUT").all()
    assert not A[nan].any() and np.isnan(Y[nan]).all()
    kind = spec["kind"]
    if kind in ops.WINDOWED_KINDS:
        L = ops.derived_lookback(kind, spec["params"]) + 1
        for (i, j) in zip(*np.nonzero(nan)):
            hi = min(i + L, X.shape[0])
            lo = max(i, f["meta"]["warmup"])
            assert (R[lo:hi, j] != "AVAILABLE").all()
            assert (R[i + 1:max(i + 1, min(hi, f["meta"]["warmup"])), j]
                    == "WARMUP").all()
    else:
        if kind in ("ewma", "butterworth2_lowpass"):
            for j in range(X.shape[1]):
                keep = ~nan[:, j]
                sub = X[keep][:, [j, j]]
                fj = copy.deepcopy(f)
                ys, _, _ = ops._transform_kernel(
                    ops._seal(dict(fj, n_columns=2)), sub)
                assert np.array_equal(ys[:, 0], Y[keep, j])
        if kind in ("local_level_kalman", "local_linear_trend_kalman"):
            st = ops._kernel_init_state(f)
            for i in range(100):
                _, _, _, st = ops._kernel_step(f, st, X[i])
            var_key = "var" if kind == "local_level_kalman" else "p11"
            v0 = st["payload"][var_key][1]
            lv0 = st["payload"]["level"][1]
            _, a, r, st = ops._kernel_step(f, st, X[100])
            assert r[1] == "MISSING_INPUT" and not a[1]
            assert st["payload"][var_key][1] > v0
            if kind == "local_level_kalman":
                assert st["payload"]["level"][1] == lv0
    for (i, j) in zip(*np.nonzero(nan)):
        assert np.isnan(Y[i, j])
    assert np.isfinite(Y[A]).all()


def test_inf_is_refused_in_batch_and_step():
    f = _fitted({"kind": "ewma", "params": {"alpha": 0.3}})
    X = TEST.copy()
    X[5, 0] = np.inf
    with pytest.raises(ops.OperatorRefusal, match="infinite"):
        ops._transform_kernel(f, X)
    st = ops._kernel_init_state(f)
    with pytest.raises(ops.OperatorRefusal, match="infinite"):
        ops._kernel_step(f, st, np.array([np.inf, 0.0]))
    with pytest.raises(ops.OperatorRefusal):
        ops._kernel_step(f, st, np.array([True, False]))
    with pytest.raises(ops.OperatorRefusal):
        ops._transform_kernel(f, TEST[:, :1])


# ------------------------------------------------------ delays --------
@pytest.mark.parametrize("w", [5, 9])
def test_fir_moving_average_group_delay_at_dc(w):
    f = _fitted({"kind": "trailing_mean", "params": {"window": w}})
    rep = meas.measure(f, latency_samples=50, batch_rows=200)
    assert rep["delay_basis"] == "LTI"
    assert rep["lti"]["freqs_cycles_per_sample"][0] == 0.0
    assert abs(rep["group_delay"][0] - (w - 1) / 2) < 1e-9
    assert np.allclose(rep["phase_delay"], (w - 1) / 2, atol=1e-6)
    assert rep["algorithmic_lookback"] == w - 1
    assert abs(rep["lti"]["dc_gain"] - 1.0) < 1e-12


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5])
def test_ewma_group_delay_at_dc(alpha):
    f = _fitted({"kind": "ewma", "params": {"alpha": alpha}})
    rep = meas.measure(f, latency_samples=50, batch_rows=200)
    assert rep["delay_basis"] == "LTI_STEADY_STATE"
    assert abs(rep["group_delay"][0] - (1 - alpha) / alpha) < 1e-9
    assert rep["algorithmic_lookback"] == ops.LOOKBACK_UNBOUNDED


def test_other_lti_delays():
    sinc = _fitted({"kind": "fir_sinc_lowpass",
                    "params": {"cutoff": 0.1, "taps": 21}})
    kw = {"latency_samples": 50, "batch_rows": 200}
    assert abs(meas.measure(sinc, **kw)["group_delay"][0] - 10.0) < 1e-9
    ident = meas.measure(_fitted({"kind": "identity", "params": {}}), **kw)
    assert ident["group_delay"] == [0.0, 0.0, 0.0, 0.0]
    for cutoff in (0.1, 0.25):
        f = _fitted({"kind": "butterworth2_lowpass",
                     "params": {"cutoff": cutoff}})
        b, a = signal.butter(2, 2 * cutoff)
        _, gd = signal.group_delay((b, a), w=[0.0], fs=1.0)
        rep = meas.measure(f, **kw)
        assert abs(rep["group_delay"][0] - gd[0]) < 1e-9
        assert rep["group_delay"][0] > 0
    ll = _fitted({"kind": "local_level_kalman", "params": {}})
    rep = meas.measure(ll, **kw)
    assert rep["delay_basis"] == "STEADY_STATE_KALMAN_GAIN"
    b, a, _ = meas.transfer_function(ll)
    k = b[0]
    assert abs(rep["group_delay"][0] - (1 - k) / k) < 1e-9
    c = ll["fitted"]["per_column"][0]
    st = ops._kernel_init_state(ll)
    for i in range(200):
        _, _, _, st = ops._kernel_step(ll, st, TEST[i])
    vp = st["payload"]["var"][0] + c["level_var"]
    assert abs(vp / (vp + c["obs_var"]) - k) < 1e-9
    llt = meas.measure(_fitted({"kind": "local_linear_trend_kalman",
                                "params": {}}), **kw)
    assert llt["delay_basis"] == "STEADY_STATE_KALMAN_GAIN"
    assert abs(llt["lti"]["dc_gain"] - 1.0) < 1e-9


NONLINEAR = [s for s in CAUSAL_SPECS if s["kind"] in
             ("trailing_median", "trailing_hampel", "trailing_haar_threshold",
              "causal_decomposition")]


@pytest.mark.parametrize("spec", NONLINEAR, ids=_sid)
def test_non_lti_delays_are_undefined_and_empirical_is_reported(spec):
    rep = meas.measure(_fitted(spec), latency_samples=50, batch_rows=200,
                       amplitude=10.0)
    assert rep["group_delay"] == "UNDEFINED"
    assert rep["phase_delay"] == "UNDEFINED"
    e = rep["empirical"]
    for key in ("impulse_peak_lag", "step_rise_50pct_lag"):
        assert e[key] == "NO_RESPONSE" or (type(e[key]) is int
                                           and e[key] >= 0)
    if spec["kind"] == "trailing_median":
        w = spec["params"]["window"]
        assert e["impulse_peak_lag"] == "NO_RESPONSE"
        assert e["step_rise_50pct_lag"] == w // 2


def test_probes_are_declared_signals_only():
    f = _fitted({"kind": "ewma", "params": {"alpha": 0.3}})
    with pytest.raises(ops.OperatorRefusal, match="ProbeSignal"):
        ops.probe_transform(f, TEST)
    p = ops.probe_signal("STEP", 100, 2, 3.0, 10)
    object.__setattr__(p, "matrix", np.asarray(TEST[:100]))
    with pytest.raises(ops.OperatorRefusal, match="does not re-derive"):
        ops.probe_transform(f, p)
    with pytest.raises(ops.OperatorRefusal):
        ops.probe_signal("DATA", 10, 2)


# ------------------------------------------- no shift compensation ----
@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_no_future_shift_compensation(spec):
    f = _fitted(spec)
    e = meas.empirical_responses(f, amplitude=10.0)
    assert e["anticipation_max_abs"] == 0.0
    lag = e["impulse_first_response_lag"]
    assert lag == "NO_RESPONSE" or lag >= 0
    if spec["kind"] not in ("identity", "trailing_hampel",
                            "trailing_median", "trailing_haar_threshold"):
        assert e["step_rise_50pct_lag"] >= 0


def test_oracle_anticipates_in_measurement():
    f = ops._fit_kernel(ORACLE_SPEC, TRAIN)
    rep = meas.measure(f, latency_samples=10, batch_rows=100)
    assert rep["causality"] == "NON_CAUSAL"
    assert rep["look_ahead"] == 2
    assert rep["empirical"]["anticipation_max_abs"] > 0
    assert rep["empirical"]["impulse_first_response_lag"] == -2
    assert rep["cost"]["step_mean_seconds"] == "REFUSED_NON_CAUSAL"


# --------------------------------------------------------- cost -------
@pytest.mark.parametrize("spec", CAUSAL_SPECS, ids=_sid)
def test_cost_measurement_returns_numbers(spec):
    rep = meas.measure(_fitted(spec), latency_samples=40, batch_rows=120)
    c = rep["cost"]
    for k in ("batch_cpu_seconds", "step_mean_seconds", "step_p95_seconds"):
        assert type(c[k]) is float and math.isfinite(c[k]) and c[k] >= 0
    assert c["step_p95_seconds"] > 0
    assert type(c["batch_peak_bytes"]) is int and c["batch_peak_bytes"] > 0
    assert rep["warmup"]["declared"] == \
        rep["warmup"]["observed_first_available_row"]
    json.dumps(rep, default=str)
    if rep["delay_basis"] in ("LTI", "LTI_STEADY_STATE",
                              "STEADY_STATE_KALMAN_GAIN"):
        assert type(rep["settling_99_samples"]) is int


# ----------------------------------- parity with causal_operators.py --
def _reference_path():
    env = os.environ.get("DF_CAUSAL_OPERATORS_REF")
    if env and Path(env).is_file():
        return Path(env)
    rel = Path(".worktrees") / "prep-t0t1" / "app" / "causal_operators.py"
    for parent in Path(__file__).resolve().parents:
        if (parent / rel).is_file():
            return parent / rel
    return None


@pytest.fixture(scope="module")
def ref():
    path = _reference_path()
    if path is None:
        pytest.skip("causal_operators.py reference is absent")
    spec = importlib.util.spec_from_file_location("ref_causal_operators",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_fit(ref, kind, params, train):
    cols = [f"c{j}" for j in range(train.shape[1])]
    oid = ("NON_CAUSAL_ORACLE_ONLY-probe" if kind == "centered_mean_oracle"
           else "probe")
    spec = {"schema": ref.SCHEMA_VERSION, "operator_id": oid, "kind": kind,
            "version": "1", "params": params, "columns": cols,
            "fit_role": "train",
            "lookback": ref.derived_lookback(kind, params),
            "availability_rule": "bar_close"}
    tc = ref.make_train_contract(train, train.shape[0])
    return ref.fit(spec, train, cols, "train", tc), cols


SHARED = [{"kind": "identity", "params": {}},
          {"kind": "trailing_mean", "params": {"window": 5}},
          {"kind": "trailing_mean", "params": {"window": 9}},
          {"kind": "trailing_median", "params": {"window": 5}},
          {"kind": "trailing_median", "params": {"window": 9}},
          {"kind": "trailing_median", "params": {"window": 21}},
          {"kind": "ewma", "params": {"alpha": 0.1}},
          {"kind": "ewma", "params": {"alpha": 0.3}},
          {"kind": "ewma", "params": {"alpha": 0.5}},
          {"kind": "local_level_kalman", "params": {}},
          ORACLE_SPEC]


@pytest.mark.parametrize("spec", SHARED, ids=_sid)
def test_parity_with_causal_operators(ref, spec):
    kind, params = spec["kind"], spec["params"]
    assert ops.derived_lookback(kind, params) == \
        ref.derived_lookback(kind, params)
    rng = np.random.default_rng(21)
    X = np.cumsum(rng.normal(size=(200, 3)), axis=0) + rng.normal(
        size=(200, 3))
    train = np.cumsum(rng.normal(size=(120, 3)), axis=0) + rng.normal(
        size=(120, 3))
    train[0] = X[0]
    rart, cols = _ref_fit(ref, kind, params, train)
    ry = ref.transform_batch(rart, X, cols,
                             ref.make_bar_close_contract(X.shape[0]))
    mine = ops._fit_kernel(spec, train)
    oracle = kind == "centered_mean_oracle"
    if kind == "local_level_kalman":
        mine = copy.deepcopy(mine)
        mine["fitted"]["per_column"] = [
            {"obs_var": rart["fitted"]["per_column"][c]["obs_var"],
             "level_var": rart["fitted"]["per_column"][c]["level_var"],
             "log_ratio": 0.0} for c in cols]
        mine = ops._seal(mine)
    Y, A, R = ops._transform_kernel(mine, X, oracle_mode=oracle)
    assert A.any()
    assert np.max(np.abs(Y[A] - ry[A])) <= 1e-12
    if kind in ("ewma", "local_level_kalman", "identity"):
        assert np.array_equal(Y, ry)
        assert A.all()
    if not oracle:
        ys, _, _ = _run_steps(mine, X)
        assert np.max(np.abs(ys[A] - ry[A])) <= 1e-12
