"""R0 modular temporal predictor for the phase-4 weekly comparison (plan section 4, FS4-02/03/05).

Written before the implementation. The spec/identity tests are pure python and run on the
coordinator; the TensorFlow fits run only where FS4_TF_TESTS=1 (a worker under crispdm-run).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from tools import fs4_temporal_predictor as P

TF = os.environ.get("FS4_TF_TESTS") == "1"
needs_tf = pytest.mark.skipif(not TF, reason="TensorFlow fits run on a worker with FS4_TF_TESTS=1")


# ----------------------------------------------------------------------------- pure python
def test_spec_is_fixed_and_budget_is_identical_for_every_subset():
    spec = P.PredictorSpec()
    assert spec.window == 24 and spec.latent_steps == 6
    assert spec.sha256() == P.PredictorSpec().sha256()
    a = P.input_identity(spec, ["f_b", "f_a", "f_c"], "RAW", None)
    b = P.input_identity(spec, ["f_c", "f_a", "f_b"], "RAW", None)
    assert a == b                                                        # FS4-02 column permutation
    assert a != P.input_identity(spec, ["f_a", "f_b"], "RAW", None)
    assert P.budget_sha256(spec) == hashlib.sha256(P.canonical(spec.budget()).encode()).hexdigest()
    assert "n_features" not in spec.budget() and "epochs" in P.canonical(spec.budget())


def test_input_modes_and_encoder_identity_bind_the_latent_source():
    spec = P.PredictorSpec()
    assert P.INPUT_MODES == ("RAW", "TRAINED_ENCODER", "RANDOM_ENCODER")
    with pytest.raises(P.Refusal, match="ENCODER_IDENTITY_REQUIRED"):
        P.input_identity(spec, ["f_a"], "TRAINED_ENCODER", None)
    with pytest.raises(P.Refusal, match="UNKNOWN_INPUT_MODE"):
        P.input_identity(spec, ["f_a"], "MLP", None)
    assert P.input_identity(spec, ["f_a"], "RANDOM_ENCODER", "e" * 64) != P.input_identity(spec, ["f_a"], "TRAINED_ENCODER", "e" * 64)


def test_windows_use_only_preceding_rows_and_drop_short_histories():
    X = np.arange(40, dtype="float64").reshape(20, 2)
    win, kept = P.make_windows(X, np.array([3, 5, 19]), window=4)
    assert kept.tolist() == [3, 5, 19] and win.shape == (3, 4, 2)
    assert win[0, -1, 0] == X[3, 0] and win[0, 0, 0] == X[0, 0]            # ends at the origin, earlier rows before
    win2, kept2 = P.make_windows(X, np.array([1, 3]), window=4)
    assert kept2.tolist() == [3] and win2.shape == (1, 4, 2)                 # origin 1 has no 4-row history
    assert np.all(np.diff(win[2, :, 0]) > 0)                                 # time order preserved


def test_future_rows_never_enter_a_window():
    X = np.random.default_rng(0).normal(size=(50, 3))
    win, _ = P.make_windows(X, np.array([10, 20]), window=5)
    Y = X.copy()
    Y[21:] = 1e9                                                             # perturb everything after the last origin
    win2, _ = P.make_windows(Y, np.array([10, 20]), window=5)
    assert np.array_equal(win, win2)


def test_standardiser_fits_on_fit_rows_only():
    X = np.vstack([np.zeros((10, 2)), np.full((5, 2), 100.0)])
    st = P.Standardiser.fit(X[:10])
    Z = st.apply(X)
    assert np.allclose(Z[:10], 0.0) and Z[10, 0] > 0
    assert st.sha256() == P.Standardiser.fit(X[:10]).sha256()


# ----------------------------------------------------------------------------- tensorflow
def _synthetic(n=600, f=3, window=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = 0.6 * np.roll(X[:, 0], 1) - 0.3 * np.roll(X[:, 1], 2) + 0.05 * rng.normal(size=n)
    return X, y


@needs_tf
def test_architecture_keeps_time_until_the_head_and_fuses_channels():
    spec = P.PredictorSpec(max_epochs=1)
    model = P.build_predictor(spec, n_features=3, input_mode="RAW", latent_dim=None)
    names = [type(l).__name__ for l in model.layers]
    assert "Flatten" not in names and "GlobalAveragePooling1D" not in names and "GlobalMaxPooling1D" not in names
    core = model.get_layer(P.CORE_OUTPUT_NAME)
    assert tuple(core.output.shape[1:]) == (spec.latent_steps, spec.core_filters)   # rank-3 until the head
    assert model.get_layer(P.HEAD_INPUT_NAME).output.shape[1:] == (spec.core_filters,)
    assert len([l for l in model.layers if l.name.startswith("branch_")]) >= 3          # one causal branch per feature
    assert all(getattr(l, "padding", "causal") == "causal" for l in model.layers if type(l).__name__ == "Conv1D")
    other = P.build_predictor(spec, n_features=7, input_mode="RAW", latent_dim=None)
    assert P.architecture_sha256(model) == P.architecture_sha256(other)          # identical architecture family
    assert P.count_params(model) != P.count_params(other)


@needs_tf
def test_fit_is_deterministic_and_future_perturbation_leaves_weights_unchanged(tmp_path):
    X, y = _synthetic()
    spec = P.PredictorSpec(max_epochs=3, batch_size=64, patience=2)
    fit_idx = np.arange(24, 400)
    inner_idx = np.arange(400, 480)
    rep1 = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    rep2 = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep1.weights_sha256 == rep2.weights_sha256
    Xp, yp = X.copy(), y.copy()
    Xp[480:] = 1e6
    yp[480:] = -1e6
    rep3 = P.fit_predictor(spec, Xp, yp, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep3.weights_sha256 == rep1.weights_sha256                                   # FS4-03 / BW02
    assert rep1.epochs_run <= spec.max_epochs and rep1.updates > 0 and rep1.best_epoch >= 1
    assert rep1.budget_sha256 == P.budget_sha256(spec)
    pred = P.predict(rep1, X, np.arange(480, 600))
    assert pred.shape == (120,) and np.all(np.isfinite(pred))
    rep_seed = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=1)
    assert rep_seed.weights_sha256 != rep1.weights_sha256                               # FS4-04


@needs_tf
def test_column_permutation_gives_identical_fit_and_budget_is_the_same_for_any_subset():
    X, y = _synthetic(f=4)
    spec = P.PredictorSpec(max_epochs=2, batch_size=64)
    fit_idx, inner_idx = np.arange(24, 400), np.arange(400, 480)
    names = ["f_c", "f_a", "f_d", "f_b"]
    rep = P.fit_named(spec, X, names, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    perm = [3, 1, 0, 2]
    rep_p = P.fit_named(spec, X[:, perm], [names[i] for i in perm], y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep.input_identity == rep_p.input_identity and rep.weights_sha256 == rep_p.weights_sha256
    small = P.fit_named(spec, X[:, :2], names[:2], y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert small.budget_sha256 == rep.budget_sha256 and small.architecture_sha256 == rep.architecture_sha256


@needs_tf
def test_random_encoder_never_updates_and_trained_encoder_loads_chosen_weights(tmp_path):
    X, y = _synthetic(f=2)
    enc_spec = P.EncoderSpec()
    rnd = P.FrozenEncoder.from_random(enc_spec, n_features=2, seed=0)
    before = rnd.weights_sha256
    lat = rnd.transform(P.make_windows(X, np.arange(24, 100), window=enc_spec.window)[0])
    assert lat.shape == (76, enc_spec.latent_steps, 2 * enc_spec.latent_dim)              # (B, 6, F*D) time kept
    assert rnd.weights_sha256 == before and rnd.optimizer_steps == 0                         # FS4-05
    spec = P.PredictorSpec(max_epochs=2, batch_size=64)
    rep = P.fit_predictor(spec, X, y, np.arange(24, 400), np.arange(400, 480), input_mode="RANDOM_ENCODER", encoder=rnd, seed=0)
    assert rnd.weights_sha256 == before and rep.encoder_sha256 == before
    path = tmp_path / "chosen.npz"
    rnd.save(path)
    loaded = P.FrozenEncoder.from_npz(enc_spec, path, expected_sha256=before)
    assert loaded.weights_sha256 == before
    with pytest.raises(P.Refusal, match="ENCODER_IDENTITY_MISMATCH"):
        P.FrozenEncoder.from_npz(enc_spec, path, expected_sha256="0" * 64)
