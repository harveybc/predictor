"""Seasonal-residual option (owner standing order 2026-10-01, item 1).

``target_residual = {"kind": "seasonal_naive", "period": P, "target_features": [...]}`` makes the forecast
model output head(latent) + seasonal_naive, where for horizon h the naive reads the target channel at
t + h - P INSIDE the input window (h > P is refused: the naive would need its own forecast), so the head learns y - seasonal_naive and the bundle
adds the naive back in the graph. Absent, the graph and the config digest are exactly as before.
Synthetic component checks.
"""
import copy
import json
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
    tf.keras.utils.set_random_seed(21)
    yield


def cfg(horizons=(1, 6, 24), residual=True, period=24):
    c = mt.default_config(["load", "temp"])
    c["horizons"] = list(horizons)
    c["target_count"] = 1
    if residual:
        c["target_residual"] = {"kind": "seasonal_naive", "period": period, "target_features": ["load"]}
    return c


def x(n=4):
    return np.random.default_rng(3).normal(size=(n, 24, 2)).astype("float32")


def test_absent_option_is_forward_identity_and_digest_neutral():
    plain = cfg(residual=False)
    before = mt.config_digest(plain)
    b = mt.build_modular(plain)
    assert mt.config_digest(plain) == before and "target_residual" not in mt._normalize(plain)
    assert not any(isinstance(l, mt.SeasonalNaiveBaseline) for l in b.forecast_model.layers)
    head = b.forecast_model.get_layer("forecast_head")
    np.testing.assert_array_equal(b.forecast_model(x()), head(b.encoder_model(x())))


def test_zero_weight_head_reproduces_the_seasonal_naive_exactly_per_horizon():
    b = mt.build_modular(cfg())
    head = b.forecast_model.get_layer("forecast_head")
    head.set_weights([np.zeros_like(w) for w in head.get_weights()])
    xs = x()
    out = np.asarray(b.forecast_model(xs))
    for i, h in enumerate((1, 6, 24)):
        lag = 24 - h                                       # steps back from the last observed row
        np.testing.assert_array_equal(out[:, i, 0], xs[:, 23 - lag, 0])
    receipt = b.component_manifests()["target_residual"]
    assert receipt["kind"] == "seasonal_naive" and receipt["period"] == 24
    assert receipt["per_horizon"] == [{"horizon": 1, "source_offset_steps": -23},
                                      {"horizon": 6, "source_offset_steps": -18},
                                      {"horizon": 24, "source_offset_steps": 0}]


def test_head_learns_the_residual_only():
    b = mt.build_modular(cfg())
    xs = x()
    baseline = b.forecast_model.get_layer("seasonal_naive_baseline")(xs)
    head = b.forecast_model.get_layer("forecast_head")(b.encoder_model(xs))
    np.testing.assert_allclose(b.forecast_model(xs), np.asarray(head) + np.asarray(baseline), atol=1e-6)


@pytest.mark.parametrize("bad, code", [
    ({"kind": "seasonal_naive", "period": 24, "target_features": ["load"]}, None),        # valid
    ({"kind": "seasonal_naive", "period": 12, "target_features": ["load"]}, "SEASONAL_HORIZON_EXCEEDS_PERIOD"),
    ({"kind": "seasonal_naive", "period": 48, "target_features": ["load"]}, "SEASONAL_REFERENCE_OUTSIDE_WINDOW"),
    ({"kind": "seasonal_naive", "period": 24, "target_features": ["nope"]}, "target_features"),
    ({"kind": "seasonal_naive", "period": 24, "target_features": ["load", "temp"]}, "target_count"),
    ({"kind": "mean", "period": 24, "target_features": ["load"]}, "kind"),
    ({"kind": "seasonal_naive", "period": 0, "target_features": ["load"]}, "period"),
])
def test_option_validation(bad, code):
    c = cfg(residual=False)
    c["target_residual"] = bad
    if code is None:
        mt.build_modular(c)
    else:
        with pytest.raises(ValueError, match=code):
            mt.build_modular(c)


def test_round_trip_save_load_and_flat_grammar(tmp_path):
    c = cfg()
    b = mt.build_modular(c)
    mt.save_bundle(b, tmp_path / "bundle")
    rebuilt, doc = mt.load_bundle(tmp_path / "bundle")
    np.testing.assert_array_equal(rebuilt.forecast_model(x()), b.forecast_model(x()))   # exact match
    assert doc["config"]["target_residual"]["period"] == 24
    plain = tf.keras.models.load_model(tmp_path / "bundle" / "forecast_model.keras", compile=False)
    np.testing.assert_array_equal(plain(x()), b.forecast_model(x()))
    assert mc.unflatten(mc.flatten(c)) == mt._normalize(c)
    assert mt.config_digest(c) != mt.config_digest(cfg(residual=False))
