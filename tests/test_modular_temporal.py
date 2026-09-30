"""Synthetic contract checks; deliberately no model-quality claims."""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt


@pytest.fixture(autouse=True)
def deterministic():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(417)
    yield
    tf.keras.backend.clear_session()


def config():
    return {
        "window": 24, "sample_hours": 1, "feature_names": ["a", "b", "c"],
        "branch_steps": 12,
        "branches": [
            {"name": "price", "features": ["b", "a"], "plugin": "causal_conv1d",
             "params": {"channels": 8}, "regime": "R0"},
            {"name": "volume", "features": ["c"], "regime": "R0"},
        ],
        "core": {"plugin": "transformer_conv", "params": {}, "regime": "R0"},
    }


def data(channels=3, steps=24):
    return np.random.default_rng(7).normal(size=(2, steps, channels)).astype("float32")


def test_shapes_raw_fusion_and_architecture():
    b = mt.build_modular(config())
    x = data()
    assert set(b.branch_models) == {"price", "volume"}
    raw = np.concatenate([b.branch_models["price"](x[:, :, [1, 0]]),
                          b.branch_models["volume"](x[:, :, [2]])], axis=-1)
    np.testing.assert_allclose(b.fusion_model(x), raw, atol=1e-6)
    assert raw.shape == (2, 12, 24)
    assert b.core_model(raw).shape == (2, 6, 8)
    assert b.encoder_model(x).shape == (2, 6, 8)
    assert b.forecast_model(x).shape == (2, 1, 1)
    np.testing.assert_allclose(b.encoder_model(x), b.core_model(raw), atol=1e-6)
    layers = b.core_model.layers
    assert isinstance(layers[1], mt.PositionalEncoding)
    attention = [l for l in layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    assert len(attention) == 2
    assert all(l.get_config()["num_heads"] == 4 for l in attention)
    assert len([l for l in layers if isinstance(l, tf.keras.layers.LayerNormalization)]) == 4
    conv = [l for l in layers if isinstance(l, tf.keras.layers.Conv1D)]
    assert [l.filters for l in conv] == [32, 16, 8]
    assert all(l.padding == "causal" for l in conv)
    assert not any(isinstance(l, tf.keras.layers.Flatten) for l in layers)
    assert b.branch_time_grid == tuple(range(2, 25, 2))
    assert b.core_time_grid == tuple(range(4, 25, 4))


def test_causal_prefix_and_final_sample_coverage():
    b = mt.build_modular(config())
    x = data()
    changed = x.copy()
    changed[:, 12:] += 20
    np.testing.assert_allclose(b.fusion_model(x)[:, :6],
                               b.fusion_model(changed)[:, :6], atol=1e-6)
    np.testing.assert_allclose(b.encoder_model(x)[:, :3],
                               b.encoder_model(changed)[:, :3], atol=1e-6)
    changed = x.copy()
    changed[:, -1] += 20
    assert not np.allclose(b.fusion_model(x)[:, -1], b.fusion_model(changed)[:, -1])
    assert not np.allclose(b.encoder_model(x)[:, -1], b.encoder_model(changed)[:, -1])
    v = tf.Variable(x)
    with tf.GradientTape() as tape:
        objective = tf.reduce_sum(tf.square(b.encoder_model(v)))
    gradient = tape.gradient(objective, v).numpy()
    assert np.all(np.max(np.abs(gradient), axis=(0, 2)) > 1e-10)


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("sample_hours"),
    lambda c: c.update(sample_hours=0.5),
    lambda c: c.update(sample_hours=float("nan")),
    lambda c: c.update(window=25),
    lambda c: c.update(branch_steps=13),
    lambda c: c.update(output_steps=5),
    lambda c: c["branches"][0].update(features=["missing"]),
    lambda c: c["branches"][0].update(features=["a", "a"]),
    lambda c: c["branches"][1].update(name="price"),
    lambda c: c["branches"][0].update(regime="R1"),
    lambda c: c["core"].update(regime="R2", donor="missing.keras"),
    lambda c: c["core"].update(donor="ignored.keras"),
])
def test_invalid_configs_fail_closed(mutation):
    c = config()
    mutation(c)
    with pytest.raises((ValueError, FileNotFoundError)):
        mt.build_modular(c)


def test_subhour_window_and_nondefault_output():
    c = config()
    c.update(window=48, sample_hours=0.5, output_steps=3, output_channels=4)
    b = mt.build_modular(c)
    assert b.branch_time_grid[-1] == 24
    assert b.core_time_grid == (8, 16, 24)
    assert b.encoder_model(data(steps=48)).shape == (2, 3, 4)
    assert b.forecast_model(data(steps=48)).shape == (2, 1, 1)


def test_pe_and_full_model_reload(tmp_path):
    b = mt.build_modular(config())
    pe = b.core_model.layers[1]
    z = pe(tf.zeros((1, 12, 24))).numpy()
    assert not np.allclose(z[:, 0], z[:, 1])
    path = tmp_path / "forecast.keras"
    b.forecast_model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    np.testing.assert_allclose(loaded(data()), b.forecast_model(data()), atol=1e-6)


def donors(tmp_path):
    c = config()
    b = mt.build_modular(c)
    for spec in c["branches"]:
        name = spec["name"]
        path = tmp_path / (name + ".keras")
        mt.save_donor(b.branch_models[name], path, b.donor_manifest("branch", name))
        spec.update(regime="R1", donor=str(path))
    core = tmp_path / "core.keras"
    mt.save_donor(b.core_model, core, b.donor_manifest("core"))
    c["core"].update(regime="R1", donor=str(core))
    return c, b


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_donor_parity_and_real_weight_updates(tmp_path, regime):
    c, original = donors(tmp_path)
    for spec in [*c["branches"], c["core"]]:
        spec["regime"] = regime
    b = mt.build_modular(c)
    np.testing.assert_allclose(b.encoder_model(data()), original.encoder_model(data()), atol=1e-6)
    models = [*b.branch_models.values(), b.core_model]
    before = [[w.copy() for w in m.get_weights()] for m in models]
    b.forecast_model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss="mse")
    b.forecast_model.train_on_batch(data(), np.ones((2, 1, 1), dtype="float32"))
    for m, old in zip(models, before):
        moved = any(not np.array_equal(a, v) for a, v in zip(old, m.get_weights()))
        assert moved == (regime == "R2")
        assert bool(m.trainable_weights) == (regime == "R2")


def test_strict_feature_config_and_upstream_binding(tmp_path):
    c, _ = donors(tmp_path)
    wrong = copy.deepcopy(c)
    wrong["branches"][0]["features"].reverse()
    with pytest.raises(ValueError, match="manifest"):
        mt.build_modular(wrong)
    wrong = copy.deepcopy(c)
    wrong["branches"][0]["params"]["channels"] = 7
    with pytest.raises(ValueError, match="manifest"):
        mt.build_modular(wrong)
    # A fresh branch has the same schema, but different weights from core pretraining.
    wrong = copy.deepcopy(c)
    wrong["branches"][0].update(regime="R0", donor=None)
    with pytest.raises(ValueError, match="manifest"):
        mt.build_modular(wrong)


@pytest.mark.parametrize("target", ["archive", "manifest"])
def test_tampered_donor_is_rejected(tmp_path, target):
    c, _ = donors(tmp_path)
    path = tmp_path / "price.keras"
    if target == "archive":
        with path.open("ab") as f:
            f.write(b"tampered")
    else:
        path = path.with_suffix(".manifest.json")
        doc = json.loads(path.read_text())
        doc["manifest"]["features"] = ["a", "b"]
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash"):
        mt.build_modular(c)


def test_decoder_autoencoder_pretrain_roundtrip(tmp_path):
    b = mt.build_modular(config())
    branch = b.branch_models["price"]
    ae = mt.build_autoencoder(branch)
    assert ae(data(channels=2)).shape == (2, 24, 2)
    before = branch.get_weights()
    ae.compile(optimizer="adam", loss="mse")
    ae.train_on_batch(data(channels=2), data(channels=2))
    assert any(not np.array_equal(a, v) for a, v in zip(before, branch.get_weights()))
    path = tmp_path / "pretrained.keras"
    manifest = b.donor_manifest("branch", "price")
    mt.save_donor(branch, path, manifest)
    restored = mt.load_donor(path, manifest)
    np.testing.assert_allclose(restored(data(channels=2)), branch(data(channels=2)), atol=1e-6)
    core_ae = mt.build_autoencoder(b.core_model)
    raw = b.fusion_model(data()).numpy()
    assert core_ae(raw).shape == raw.shape
    core_ae.compile(optimizer="adam", loss="mse")
    core_ae.train_on_batch(raw, raw)
    assert mt.build_decoder((6, 8), 24, 3)(np.ones((2, 6, 8))).shape == (2, 24, 3)


def test_entry_point_groups_and_grid_rejection(monkeypatch):
    seen = []

    class EP:
        def __init__(self, group):
            self.group = group
            self.value = "test:factory"
            self.dist = None

        def load(self):
            def factory(**kwargs):
                seen.append(self.group)
                result = mt.BUILTINS[self.group][mt.DEFAULTS[self.group]](**kwargs)
                if self.group == "modular.branch" and kwargs["name"] == "volume":
                    return mt.TemporalComponent(result.model, tuple(t + 1 for t in result.time_grid))
                return result
            return factory

    monkeypatch.setattr(mt, "entry_points", lambda *, group, name: [EP(group)])
    c = config()
    c["branches"][0]["plugin"] = "external"
    c["core"]["plugin"] = "external"
    c["fusion"] = {"plugin": "external"}
    c["head"] = {"plugin": "external"}
    mt.build_modular(c)
    assert set(seen) == {"modular.branch", "modular.core", "modular.fusion", "modular.head"}
    c["branches"][1]["plugin"] = "external"
    with pytest.raises(ValueError, match="grid"):
        mt.build_modular(c)


def test_default_config_and_independent_forecast_dimensions():
    c = mt.default_config(["a", "b", "c"])
    c.update(horizons=[1, 6, 24], target_count=2)
    b = mt.build_modular(c)
    assert b.config["horizons"] == [1, 6, 24]
    assert b.config["core"]["regime"] == "R0"
    assert b.encoder_model(data()).shape == (2, 6, 8)
    assert b.forecast_model(data()).shape == (2, 3, 2)
    assert mt.branch_autoencoder(b, c["branches"][0]["name"]).output_shape == (None, 24, 1)
    assert mt.core_autoencoder(b).output_shape == b.core_model.input_shape
    head = b.forecast_model.get_layer("forecast_head")
    latent = tf.Variable(np.ones((2, 6, 8), dtype="float32"))
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(head(latent))
    assert np.all(np.max(np.abs(tape.gradient(value, latent)), axis=(0, 2)) > 0)
