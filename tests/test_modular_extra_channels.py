"""Per-feature extra channels (front H, Kalman arms B and C) and the no-temporal-collapse graph contract.

``extra_channels = {"<feature>": ["<feature>__kf_level", "<feature>__kf_slope", ...]}``: each declared
input may carry companion channels (e.g. a causal Kalman filter's filtered state, slope, innovation,
standardized innovation and log state variance, computed upstream at bar t from data up to t). They are
ordinary named columns of ``feature_names``; the key only fixes the layout: the parent's branch receives
[parent, *extras] in declared order (FeatureSelect), so the branch sees (window, 1 + k) and keeps the
time axis. Absent, graph, flat keys and digest are unchanged. Synthetic checks; no data.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import itertools

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt

KF = ["kf_level", "kf_slope", "kf_innov", "kf_zinnov", "kf_logvar"]


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)
    yield


def extras(*parents):
    return {p: [f"{p}__{k}" for k in KF] for p in parents}


def kcfg(parents=("close", "vol"), with_extras=("close",)):
    return mt.config_with_extra_channels(list(parents), extras(*with_extras))


def test_layout_routes_parent_then_its_extras_into_one_branch():
    c = kcfg()
    assert c["feature_names"] == ["close", *extras("close")["close"], "vol"]
    assert [b["features"] for b in c["branches"]] == [["close", *extras("close")["close"]], ["vol"]]
    bundle = mt.build_modular({**c, "sample_hours": 4})
    assert bundle.branch_models["branch_0"].input_shape == (None, 24, 6)
    assert bundle.branch_models["branch_0"].output_shape[1] == 24          # time axis kept
    assert bundle.branch_models["branch_1"].input_shape == (None, 24, 1)
    x = np.random.default_rng(0).normal(size=(3, 24, 7)).astype("float32")
    assert bundle.forecast_model(x).shape == (3, 1, 1)
    assert bundle.component_manifests()["extra_channels"] == extras("close")


def test_feature_select_feeds_the_declared_columns():
    c = {**kcfg(), "sample_hours": 4}
    bundle = mt.build_modular(c)
    sel = bundle.forecast_model.get_layer("select_branch_0")
    x = np.arange(2 * 24 * 7, dtype="float32").reshape(2, 24, 7)
    np.testing.assert_array_equal(np.asarray(sel(x)), x[:, :, :6])


def test_absent_key_is_digest_neutral_and_flat_round_trips():
    plain = mt.default_config(["close", "vol"])
    assert "extra_channels" not in plain
    assert not any("extra_channels" in k for k in mc.flatten(plain))
    same = mt.config_with_extra_channels(["close", "vol"], {})
    assert "extra_channels" not in same
    assert mt.config_digest(same) == mt.config_digest(plain)
    c = kcfg()
    flat = mc.flatten(c)
    assert flat["modular.extra_channels"] == extras("close")
    assert mt.config_digest(mc.unflatten(flat)) == mt.config_digest(c)
    assert mt.config_digest(c) != mt.config_digest(plain)


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["extra_channels"].update(ghost=["x"]), "parent"),
    (lambda c: c["extra_channels"]["close"].append("nope"), "feature_names"),
    (lambda c: c["extra_channels"].update(vol=["close__kf_level"]), "once"),
    (lambda c: c["extra_channels"].update(vol=["close"]), "parent"),
    (lambda c: c["branches"][0]["features"].reverse(), "layout"),
    (lambda c: c["branches"][1]["features"].append("close__kf_slope"), "layout"),
    (lambda c: c["extra_channels"].update(close=[]), "nonempty"),
])
def test_malformed_extra_channels_are_refused(mutate, match):
    c = kcfg()
    mutate(c)
    with pytest.raises(ValueError, match=match):
        mt.build_modular({**c, "sample_hours": 4})


COLLAPSE = (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D, tf.keras.layers.GlobalMaxPooling1D)


def _walk(layer):
    yield layer
    for sub in getattr(layer, "layers", []):
        yield from _walk(sub)


def _variants():
    keys = {"target_residual": {"kind": "seasonal_naive_cumulative", "period": 6,
                                "target_features": ["close"]},
            "input_normalization": {"kind": "window_mean", "length": 24, "target_features": ["close"]},
            "extra_channels": extras("close")}
    for r in range(len(keys) + 1):
        for combo in itertools.combinations(sorted(keys), r):
            if {"target_residual", "input_normalization"} <= set(combo):
                continue                                       # exclusive by contract
            yield combo, keys


@pytest.mark.parametrize("combo", [c for c, _ in _variants()], ids=lambda c: "+".join(c) or "plain")
def test_no_layer_collapses_time_before_the_head(combo):
    keys = dict(next(k for c, k in _variants() if c == combo))
    c = (mt.config_with_extra_channels(["close", "vol"], keys["extra_channels"]) if "extra_channels" in combo
         else mt.default_config(["close", "vol"]))
    c.update(sample_hours=4, horizons=[1, 6])
    for key in combo:
        if key != "extra_channels":
            c[key] = keys[key]
    bundle = mt.build_modular(c)
    window, steps = c["window"], c["output_steps"]
    for model in [*bundle.branch_models.values(), bundle.core_model]:
        for layer in _walk(model):
            assert not isinstance(layer, COLLAPSE), f"{model.name}/{layer.name} collapses time"
        for layer in model.layers:
            for out in tf.nest.flatten(layer.output):
                assert len(out.shape) == 3 and out.shape[1] >= steps, (
                    f"{model.name}/{layer.name} output {out.shape} lost the time axis")
    assert bundle.encoder_model.output_shape == (None, steps, c["output_channels"])
    head_seen = False
    for layer in bundle.forecast_model.layers:
        head_seen |= layer.name == "forecast_head"
        if not head_seen:
            assert not isinstance(layer, COLLAPSE + (tf.keras.layers.Dense,)), layer.name
            for out in tf.nest.flatten(layer.output):
                assert len(out.shape) == 3, f"{layer.name} output {out.shape} before the head"
    assert head_seen
    x = np.random.default_rng(1).normal(size=(2, window, len(c["feature_names"]))).astype("float32")
    assert bundle.forecast_model(x).shape == (2, 2, 1)
