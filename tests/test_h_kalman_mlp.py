"""Lane H: flatten+MLP control learner (same architecture family as lane F2's control), CPU, inner hold-out selection."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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


mlp = _load("h_kalman_mlp")


def task(N=1500, P=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((N, P))
    w = rng.standard_normal((P, 2))
    Y = np.tanh(X @ w * 0.5) + 0.2 * rng.standard_normal((N, 2))
    return X, Y


def test_learns_a_nonlinear_signal_and_selects_on_the_inner_holdout_only():
    X, Y = task()
    Xte, Yte = task(400, seed=1)
    Xte = Xte[:, :20]
    out = mlp.mlp_fit_predict(X[:1200], Y[:1200], [Xte], seed=2021, hidden=(32, 32), max_epochs=40, patience=6)
    pred = out["predictions"][0]
    assert np.mean(np.abs(pred - Yte)) < np.mean(np.abs(Y.mean(axis=0) - Yte)) * 0.9
    assert out["selected_epoch"] >= 1 and out["epochs_run"] <= 40 and out["selection"] == "inner_chronological_holdout_of_train"
    assert np.isfinite(out["inner_best_loss"])


def test_evaluation_rows_cannot_influence_training():
    X, Y = task()
    Xte, _ = task(300, seed=5)
    a = mlp.mlp_fit_predict(X, Y, [Xte], seed=2021, hidden=(16,), max_epochs=5, patience=3)
    b = mlp.mlp_fit_predict(X, Y, [Xte * 0 + 50.0], seed=2021, hidden=(16,), max_epochs=5, patience=3)
    assert a["weights_sha256"] == b["weights_sha256"] and a["selected_epoch"] == b["selected_epoch"]


def test_same_seed_same_result_different_seed_different_result():
    X, Y = task()
    Xte, _ = task(100, seed=6)
    a = mlp.mlp_fit_predict(X, Y, [Xte], seed=2021, hidden=(16,), max_epochs=4, patience=3)
    b = mlp.mlp_fit_predict(X, Y, [Xte], seed=2021, hidden=(16,), max_epochs=4, patience=3)
    c = mlp.mlp_fit_predict(X, Y, [Xte], seed=2022, hidden=(16,), max_epochs=4, patience=3)
    assert a["weights_sha256"] == b["weights_sha256"]
    assert a["weights_sha256"] != c["weights_sha256"]


def test_refuses_nonfinite_and_bad_shapes():
    X, Y = task(300)
    Xb = X.copy(); Xb[0, 0] = np.nan
    with pytest.raises(ValueError):
        mlp.mlp_fit_predict(Xb, Y, [X], seed=1, hidden=(8,), max_epochs=1, patience=1)
    with pytest.raises(ValueError):
        mlp.mlp_fit_predict(X, Y[:-1], [X], seed=1, hidden=(8,), max_epochs=1, patience=1)
    with pytest.raises(ValueError):
        mlp.mlp_fit_predict(X, Y, [X[:, :5]], seed=1, hidden=(8,), max_epochs=1, patience=1)
