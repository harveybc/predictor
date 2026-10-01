"""Lane H: ECL testbed (M04 v2 NPZ, L24 -> H1..24, seasonal-residual target, channel-independent pooled learners)."""
from __future__ import annotations

import importlib.util
import sys
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


ecl = _load("h_kalman_ecl")
kf = ecl.kf


def synthetic_npz(tmp_path, R=900, C=5, L=24, H=24, n_tr=600, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(R)
    level = np.cumsum(0.02 * rng.standard_normal((R, C)), axis=0)
    Z = (np.sin(2 * np.pi * t / 24)[:, None] * (1 + np.arange(C))[None, :] * 0.5 + level
         + 0.3 * rng.standard_normal((R, C))).astype(np.float32)
    def build(lo, hi, name):
        origins = np.arange(lo, hi)
        w = np.stack([Z[o - L + 1:o + 1] for o in origins])
        y = np.stack([Z[o + 1:o + H + 1] for o in origins])
        np.savez(tmp_path / f"{name}.npz", windows=w, targets=y,
                 row_ids=np.array([f"ecl:row{o}:{o * 3600}" for o in origins]), horizons=np.arange(1, H + 1))
        return origins
    o_tr = build(L - 1, n_tr - H - L, "train")
    o_va = build(n_tr + L, R - H, "validation")
    return Z, o_tr, o_va


def test_memmap_reads_the_stored_npz_members_exactly(tmp_path):
    Z, o_tr, _ = synthetic_npz(tmp_path)
    mm = ecl.npz_memmap(tmp_path / "train.npz", "windows")
    with np.load(tmp_path / "train.npz") as z:
        assert np.array_equal(np.asarray(mm), z["windows"])


def test_rows_reconstructed_from_windows_and_targets_are_exact_and_contiguous(tmp_path):
    Z, o_tr, o_va = synthetic_npz(tmp_path)
    rows, origins = ecl.reconstruct_rows(tmp_path / "train.npz", tmp_path / "validation.npz", horizons=24, window=24)
    assert np.array_equal(origins["train"], o_tr) and np.array_equal(origins["validation"], o_va)
    covered = ~np.isnan(rows[:, 0])
    assert np.array_equal(rows[covered], Z[:len(rows)][covered].astype(np.float64))
    assert covered[:o_tr.max() + 25].all() and covered[o_va.min() - 23:o_va.max() + 25].all()


def test_seasonal_residual_target_and_seasonal_naive():
    rng = np.random.RandomState(1)
    Z = rng.standard_normal((300, 3))
    o = np.arange(30, 200)
    R = ecl.residual_targets(Z, o, 24)
    for h in (1, 7, 24):
        assert np.allclose(R[:, h - 1, :], Z[o + h] - Z[o + h - 24])
    nv = ecl.seasonal_naive(Z, o, 24)
    assert np.allclose(nv[:, 23, :], Z[o])                    # h = 24 uses the origin row itself, never a later row
    assert all((o + h - 24 <= o).all() for h in range(1, 25))


def test_feature_rows_are_origin_major_channel_minor_and_causal():
    rng = np.random.RandomState(2)
    M1, M2 = rng.standard_normal((100, 4)), rng.standard_normal((100, 4))
    o = np.array([40, 41])
    X = ecl.channel_features([M1, M2], o, lags=3)
    assert X.shape == (2 * 4, 2 * 3)
    assert np.allclose(X[1], [M1[40, 1], M1[39, 1], M1[38, 1], M2[40, 1], M2[39, 1], M2[38, 1]])
    M1b = M1.copy(); M1b[42:] = 999
    assert np.array_equal(ecl.channel_features([M1b, M2], o, lags=3), X)


def test_pooled_ridge_equals_a_direct_solve_and_selects_alpha_inside_train():
    rng = np.random.RandomState(3)
    Z = rng.standard_normal((400, 6))
    o = np.arange(30, 340)
    Y = rng.standard_normal((len(o) * 6, 4))
    w = rng.standard_normal((3, 4))
    feats = lambda oo: ecl.channel_features([Z], oo, lags=3)
    Xall = feats(o)
    Yall = Xall @ w + 0.1 * rng.standard_normal((len(Xall), 4))
    tgt = lambda oo: Yall[np.concatenate([np.arange(i * 6, i * 6 + 6) for i in np.searchsorted(o, oo)])]
    fit = ecl.pooled_ridge(feats, tgt, o, alphas=(1e-3, 1.0, 1e3), block=50, holdout_frac=0.2, purge=5)
    mu, sd = Xall.mean(0), Xall.std(0)
    A = (Xall - mu) / sd
    ym = Yall.mean(0)
    W = np.linalg.solve(A.T @ A + fit["alpha"] * np.eye(3), A.T @ (Yall - ym))
    assert np.allclose(fit["W"], W, atol=1e-8) and np.allclose(fit["mu"], mu) and np.allclose(fit["ym"], ym)
    assert fit["alpha"] in (1e-3, 1.0)


def test_arm_specs_have_the_declared_channel_blocks(tmp_path):
    Z, o_tr, o_va = synthetic_npz(tmp_path)
    rows, origins = ecl.reconstruct_rows(tmp_path / "train.npz", tmp_path / "validation.npz", horizons=24, window=24)
    d = ecl.panel(rows, origins, train_rows=(0, 600))
    kal = ecl.kalman_blocks(d, ecl.pipe.VARIANTS["moments_train"])
    arms = ecl.arm_blocks(d, kal)
    assert len(arms["A"]["blocks"]) == 1 and len(arms["B"]["blocks"]) == 5 and len(arms["C"]["blocks"]) == 4
    assert len(arms["C_EWMA"]["blocks"]) == 2 and arms["C_SMOOTHER_NONCAUSAL"]["eligible"] is False
    assert len(arms["B_PERMUTED"]["blocks"]) == 5 and len(arms["B_NOISE"]["blocks"]) == 5
    # the Kalman fit never sees rows at or after the train border
    d2 = ecl.panel(rows.copy(), origins, train_rows=(0, 600))
    d2["Z"][600:] += 5.0
    kal2 = ecl.kalman_blocks(d2, ecl.pipe.VARIANTS["moments_train"])
    assert kal["artifact"]["artifact_sha256"] == kal2["artifact"]["artifact_sha256"]
