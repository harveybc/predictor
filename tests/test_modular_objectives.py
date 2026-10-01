"""PS3-R decoder-free branch objectives and the Delta_probe criterion (written before the mechanism).

Architecture and objective are separate axes: the branch architecture keeps its
donor manifest whatever objective trains it; the objective has its own entry
point, version, effective parameters and identity, recorded beside the donor.
The TS2Vec-style objective is a hierarchical instance + temporal contrastive
loss with contextual (crop) consistency and timestamp masking on the
(B, 24, 16) branch latent; it has no decoder, so reconstruction is
NOT_APPLICABLE. The probe battery compares a frozen trained encoder, the same
architecture untrained, and the raw input, with the same probe and budget, on
Y_s / Y_l / Y_b with a paired naive on identical rows.
Synthetic component checks, not results.
"""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import objectives as ob
from predictor_plugins.modular_temporal import probes as pb


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(3)
    yield


def config(f=2):
    return mt.default_config([f"x{i}" for i in range(f)])


def windows(n, seed=0, f=1):
    """AR(1) windows with a slow regime: structure a contrastive objective can use."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 24, f), dtype="float32")
    level = rng.normal(size=(n, 1, f)).astype("float32")
    for t in range(24):
        prev = out[:, t - 1] if t else level[:, 0]
        out[:, t] = 0.8 * prev + 0.3 * rng.normal(size=(n, f))
    return out


FIT = {"max_epochs": 3, "patience": 3, "batch_size": 16, "learning_rate": 3e-3, "seed": 1}


# ------------------------------------------------------------------ identity
def test_objectives_resolve_with_versions_and_unknown_names_fail():
    for name in ("autoencoder_reconstruction", "ts2vec_contrastive"):
        d = ob.describe_objective(name)
        assert d["version"] == "1.0.0" and d["group"] == "modular.objective" and d["contract"]
    with pytest.raises(ValueError, match="objective"):
        ob.resolve_objective({"plugin": "no_such_objective"})


def test_objective_identity_is_separate_from_architecture_identity():
    b = mt.build_modular(config())
    before = b.donor_manifest("branch", "branch_0")
    ae = ob.objective_identity({"plugin": "autoencoder_reconstruction"})
    cl = ob.objective_identity({"plugin": "ts2vec_contrastive"})
    explicit = ob.objective_identity({"plugin": "ts2vec_contrastive",
                                      "params": dict(ob.describe_objective("ts2vec_contrastive")["defaults"])})
    other = ob.objective_identity({"plugin": "ts2vec_contrastive", "params": {"alpha": 0.25}})
    assert ae["sha256"] != cl["sha256"] and cl["sha256"] == explicit["sha256"] != other["sha256"]
    assert b.donor_manifest("branch", "branch_0") == before            # objective never enters the manifest
    assert "objective" not in json.dumps(before)
    with pytest.raises(ValueError, match="params"):
        ob.objective_identity({"plugin": "ts2vec_contrastive", "params": {"temperature": 2}})


# ------------------------------------------------------------- contrastive fit
def test_contrastive_objective_trains_the_branch_without_a_decoder():
    b = mt.build_modular(config())
    enc = b.branch_models["branch_0"]
    x, vx = windows(64, 1), windows(32, 2)
    before = mt.weights_hash(enc)
    receipt = ob.fit_objective({"plugin": "ts2vec_contrastive"}, enc, x, vx, FIT)
    assert receipt["reconstruction"] == {"state": "NOT_APPLICABLE"}
    assert receipt["observed_updates"] == receipt["optimizer_iterations"] - receipt["initial_optimizer_iterations"]
    assert receipt["observed_updates"] == 4 * receipt["epochs_completed"]           # 64 rows / batch 16
    assert receipt["restored_best_weights"] and mt.weights_hash(enc) != before
    assert receipt["identity"] == ob.objective_identity({"plugin": "ts2vec_contrastive"})
    assert np.isfinite(receipt["best_validation_loss"])
    assert receipt["best_validation_loss"] < receipt["initial_validation_loss"]      # it learned something
    assert enc.output_shape == (None, 24, 16)                                        # latent stays temporal
    mt.probe_alignment(enc, b.branch_time_grid, b.branch_time_grid, label="trained branch")  # still causal


def test_contrastive_loss_mechanics():
    z = tf.random.normal((4, 24, 16))
    same = float(ob.hierarchical_contrastive_loss(z, z))
    shuffled = float(ob.hierarchical_contrastive_loss(z, tf.random.shuffle(z)))
    assert np.isfinite(same) and same < shuffled                     # matched views score better
    with pytest.raises(ValueError, match="rank-three"):
        ob.hierarchical_contrastive_loss(tf.reduce_mean(z, axis=1), tf.reduce_mean(z, axis=1))


def test_autoencoder_control_uses_the_same_interface_and_measures_reconstruction():
    b = mt.build_modular(config())
    enc = b.branch_models["branch_0"]
    receipt = ob.fit_objective({"plugin": "autoencoder_reconstruction"}, enc, windows(64, 1), windows(32, 2), FIT)
    assert receipt["reconstruction"]["state"] == "MEASURED"
    assert receipt["reconstruction"]["mse_z"] >= 0 and receipt["reconstruction"]["mae_z"] >= 0
    assert receipt["observed_updates"] > 0


def test_donor_records_objective_beside_an_unchanged_manifest(tmp_path):
    b = mt.build_modular(config())
    enc = b.branch_models["branch_0"]
    receipt = ob.fit_objective({"plugin": "ts2vec_contrastive"}, enc, windows(32, 1), windows(16, 2),
                               dict(FIT, max_epochs=1))
    doc = mt.save_donor(enc, tmp_path / "b0.keras", b.donor_manifest("branch", "branch_0"),
                        objective=receipt["identity"])
    assert doc["provenance"]["objective"] == receipt["identity"]
    assert doc["manifest"] == b.donor_manifest("branch", "branch_0")
    mt.load_donor(tmp_path / "b0.keras", b.donor_manifest("branch", "branch_0"))
    assert ob.donor_objective(tmp_path / "b0.keras") == receipt["identity"]


# ------------------------------------------------------------------- probes
class Fixed:
    """A deterministic 'encoder' for probe arithmetic: (B,24,1) -> (B,24,C)."""

    def __init__(self, fn, channels):
        self.fn, self.output_shape = fn, (None, 24, channels)

    def __call__(self, x, training=False):
        return self.fn(np.asarray(x))


def probe_data(n=400, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 24, 1)).astype("float32")
    signal = np.tanh(3 * x[:, -1, 0]) * np.sign(x[:, -2, 0])      # nonlinear: a linear raw probe misses it
    y_s = (signal + 0.1 * rng.normal(size=n)).astype("float32")
    y_l = (0.5 * signal + 0.3 * rng.normal(size=n)).astype("float32")
    return x, {"Y_s@4h": y_s, "Y_l@24h": y_l}


def informative(x):
    feat = np.tanh(3 * x[:, :, :1]) * np.sign(np.roll(x[:, :, :1], 1, axis=1))
    return np.concatenate([feat, np.zeros_like(feat)], axis=-1)


def uninformative(x):
    return np.random.default_rng(0).normal(size=(*x.shape[:2], 2)).astype("float32")


def test_delta_probe_sign_arithmetic_and_card_rows():
    x, y = probe_data()
    rows = pb.probe_battery(trained=Fixed(informative, 2), random=Fixed(uninformative, 2), x=x, targets=y,
                            fit_rows=np.arange(300), eval_rows=np.arange(300, 400), fold="inner_1", seed=0)
    assert {(r["target"], r["horizon"]) for r in rows} == {("Y_s", 4), ("Y_l", 24)}
    for r in rows:
        assert r["delta_probe"] == pytest.approx(r["loss_random"] - r["loss_trained"], abs=1e-12)
        assert r["preservation"] == pytest.approx(r["loss_raw"] - r["loss_trained"], abs=1e-12)
        assert r["delta_probe"] > 0 and r["loss_trained"] < r["naive"]
        assert r["raw_dimension_differs"] is True and r["eval_rows"] == 100
        assert set(pb.CARD_ROW_KEYS) <= set(r)
    swapped = pb.probe_battery(trained=Fixed(uninformative, 2), random=Fixed(informative, 2), x=x, targets=y,
                               fit_rows=np.arange(300), eval_rows=np.arange(300, 400), fold="inner_1", seed=0)
    assert all(r["delta_probe"] < 0 for r in swapped)


def test_paired_naive_uses_identical_rows_and_binary_base_rate():
    x, y = probe_data()
    y_b = (y["Y_s@4h"] > 0).astype("float32")
    rows = pb.probe_battery(trained=Fixed(informative, 2), random=Fixed(uninformative, 2), x=x,
                            targets={"Y_s@4h": y["Y_s@4h"], "Y_b@4h": y_b},
                            fit_rows=np.arange(300), eval_rows=np.arange(300, 400), fold="f", seed=0)
    ys = next(r for r in rows if r["target"] == "Y_s")
    assert ys["naive"] == pytest.approx(float(np.mean(y["Y_s@4h"][300:] ** 2)))   # persistence: zero return
    yb = next(r for r in rows if r["target"] == "Y_b")
    p = float(np.clip(y_b[:300].mean(), 1e-6, 1 - 1e-6))
    t = y_b[300:]
    assert yb["loss_name"] == "log_loss"
    assert yb["naive"] == pytest.approx(float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))))


def test_probe_refusals():
    x, y = probe_data()
    kwargs = dict(x=x, fit_rows=np.arange(300), eval_rows=np.arange(300, 400), fold="f", seed=0)
    with pytest.raises(ValueError, match="SELF_FORECAST_REFUSED"):
        pb.probe_battery(trained=Fixed(informative, 2), random=Fixed(uninformative, 2),
                         targets={"x0@4h": y["Y_s@4h"]}, **kwargs)
    pooled = Fixed(lambda a: informative(a)[:, -1:, :], 2)
    pooled.output_shape = (None, 1, 2)
    with pytest.raises(ValueError, match="POOLED"):
        pb.probe_battery(trained=pooled, random=Fixed(uninformative, 2), targets=y, **kwargs)
    with pytest.raises(ValueError, match="overlap"):
        pb.probe_battery(trained=Fixed(informative, 2), random=Fixed(uninformative, 2), targets=y,
                         x=x, fit_rows=np.arange(300), eval_rows=np.arange(250, 400), fold="f", seed=0)


def test_random_arm_is_the_same_architecture_untrained():
    b = mt.build_modular(config())
    enc = b.branch_models["branch_0"]
    random = pb.untrained_twin(enc, seed=11)
    assert random.output_shape == enc.output_shape and random is not enc
    assert [w.shape for w in random.get_weights()] == [w.shape for w in enc.get_weights()]
    assert mt.weights_hash(random) != mt.weights_hash(enc)
    again = pb.untrained_twin(enc, seed=11)
    assert mt.weights_hash(again) == mt.weights_hash(random)                       # seeded, reproducible


def test_latent_diagnostics_detect_collapse():
    x, _ = probe_data()
    healthy = pb.latent_diagnostics(Fixed(lambda a: np.repeat(a, 4, axis=-1) + np.random.default_rng(1).normal(
        size=(*a.shape[:2], 4)), 4), x)
    collapsed = pb.latent_diagnostics(Fixed(lambda a: np.ones((*a.shape[:2], 4), "float32"), 4), x)
    assert healthy["effective_dimension"] > 1.5 and collapsed["effective_dimension"] < 1e-6 + 1
    assert collapsed["collapsed"] is True and healthy["collapsed"] is False
