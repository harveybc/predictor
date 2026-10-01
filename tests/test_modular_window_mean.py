"""Window-mean normalization (M04 request item 3, under the owner's standing order).

``input_normalization = {"kind": "window_mean", "length": L, "target_features": [...]}``: in the graph,
each window's mean over its last L steps (per channel, all observed by t) is subtracted from the inputs
before the branches, and the target channels' mean is added back to every horizon of the output; the
head learns y - mean_L(target). Absent, graph and digest are unchanged. Exclusive with target_residual.
Synthetic component checks.
"""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(31)
    yield


def cfg(norm=True, length=24):
    c = mt.default_config(["load", "temp"])
    c["horizons"] = [1, 6, 24]
    if norm:
        c["input_normalization"] = {"kind": "window_mean", "length": length, "target_features": ["load"]}
    return c


def x(n=4):
    rng = np.random.default_rng(5)
    return (rng.normal(size=(n, 24, 2)) + rng.normal(size=(n, 1, 2)) * 3).astype("float32")


def test_absent_is_identity_and_digest_neutral():
    plain = cfg(norm=False)
    before = mt.config_digest(plain)
    b = mt.build_modular(plain)
    assert mt.config_digest(plain) == before and "input_normalization" not in mt._normalize(plain)
    assert not any(isinstance(l, mt.WindowMean) for l in b.forecast_model.layers)


def test_zero_weight_head_returns_the_target_window_mean_and_inputs_are_centered():
    b = mt.build_modular(cfg(length=12))
    head = b.forecast_model.get_layer("forecast_head")
    head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    xs = x()
    out = np.asarray(b.forecast_model(xs))
    expected = xs[:, -12:, 0].mean(axis=1)
    for i in range(3):
        np.testing.assert_allclose(out[:, i, 0], expected, rtol=1e-6, atol=1e-6)
    centered = np.asarray(b.forecast_model.get_layer("window_mean_centered")(xs))
    np.testing.assert_allclose(centered[:, -12:, :].mean(axis=1), 0.0, atol=1e-5)
    assert b.component_manifests()["input_normalization"]["length"] == 12


def test_round_trip_flat_grammar_and_refusals(tmp_path):
    c = cfg()
    b = mt.build_modular(c)
    mt.save_bundle(b, tmp_path / "b")
    rebuilt, _ = mt.load_bundle(tmp_path / "b")
    np.testing.assert_array_equal(rebuilt.forecast_model(x()), b.forecast_model(x()))
    assert mc.unflatten(mc.flatten(c)) == mt._normalize(c)
    for bad, code in (({"kind": "window_mean", "length": 25, "target_features": ["load"]}, "length"),
                      ({"kind": "zscore", "length": 24, "target_features": ["load"]}, "kind"),
                      ({"kind": "window_mean", "length": 24, "target_features": ["x"]}, "target_features")):
        with pytest.raises(ValueError, match=code):
            mt.build_modular(dict(copy.deepcopy(c), input_normalization=bad))
    both = dict(copy.deepcopy(c), target_residual={"kind": "seasonal_naive", "period": 24, "target_features": ["load"]})
    with pytest.raises(ValueError, match="exclusive"):
        mt.build_modular(both)
