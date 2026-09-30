"""M01 behavioural checks of the opt-in modular predictor (MS01-MS08).

Synthetic component checks on tiny random arrays: they establish mechanics
(resolution, shapes, time alignment, donor identity, regime update counts,
serialization), NOT forecasting quality. No result here is a forecasting result.
"""
import copy
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_config as mc
from predictor_plugins import modular_temporal as mt
from predictor_plugins.predictor_plugin_modular import Plugin

SMALL_CORE = {"d_model": 16, "heads": 2, "ff_dim": 16, "stage_channels": [12, 10, 8]}


@pytest.fixture(autouse=True)
def deterministic():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(20260930)
    yield
    tf.keras.backend.clear_session()


def nested(**top):
    c = {"schema": "predictor.modular.v1", "window": 24, "sample_hours": 1,
         "feature_names": ["close", "volume", "spread"], "horizons": [1, 3],
         "branches": [
             {"name": "price", "features": ["close"], "params": {"channels": 4}},
             {"name": "volume", "features": ["volume"], "params": {"channels": 3}},
             {"name": "spread", "features": ["spread"], "params": {"channels": 2, "kernel_size": 2}},
         ],
         "core": {"params": dict(SMALL_CORE)}}
    c.update(top)
    return c


def x_data(n=32, seed=3, window=24, features=3):
    return np.random.default_rng(seed).normal(size=(n, window, features)).astype("float32")


def y_dict(x, horizons=(1, 3)):
    # A learnable synthetic target: a linear function of the last observations.
    return {f"output_horizon_{h}": (x[:, -1, :1] * 0.5 + x[:, -h, 1:2] * 0.1).astype("float32")
            for h in horizons}


def run_config(model, **extra):
    c = {"predicted_horizons": model.get("horizons", [1]), "window_size": model["window"],
         "modular": model, "modular_training": {"max_epochs": 2, "patience": 5, "batch_size": 8,
                                                 "learning_rate": 1e-2, "seed": 1}}
    c.update(extra)
    return c


# ----------------------------------------------------------------- MS01
def test_builtins_resolve_with_versions_and_unknown_names_fail():
    for role, name in (("branch", "causal_conv1d"), ("fusion", "sequence_concat"),
                       ("core", "transformer_conv"), ("head", "forecast")):
        d = mt.describe_component(role, name)
        assert d["version"] == "1.0.0" and d["contract"] and d["name"] == name
    with pytest.raises(ValueError, match="exactly one plugin"):
        mt.describe_component("branch", "no_such_branch")
    c = nested()
    c["core"]["plugin"] = "no_such_core"
    with pytest.raises(ValueError, match="exactly one plugin"):
        mt.build_modular(c)


def test_external_entry_points_undeclared_ambiguous_and_shadowing_fail(monkeypatch):
    @mt.component("branch", "2.1.0", {"channels", "kernel_size"}, "same as causal_conv1d")
    def external(**kwargs):
        return mt.causal_conv1d(**kwargs)

    def undeclared(**kwargs):
        return mt.causal_conv1d(**kwargs)

    class EP:
        def __init__(self, obj, value="ext:factory"):
            self.obj, self.value, self.dist = obj, value, None

        def load(self):
            return self.obj

    table = {"external": [EP(external)], "undeclared": [EP(undeclared)],
             "twice": [EP(external), EP(external, "other:factory")],
             "causal_conv1d": [EP(external)]}
    monkeypatch.setattr(mt, "entry_points", lambda *, group, name: table.get(name, []))
    c = nested()
    c["branches"][0]["plugin"] = "external"
    b = mt.build_modular(c)
    assert b.donor_manifest("branch", "price")["plugin"]["version"] == "2.1.0"
    for bad, message in (("undeclared", "component"), ("twice", "exactly one"),
                         ("causal_conv1d", "shadows")):
        c = nested()
        c["branches"][0]["plugin"] = bad
        with pytest.raises(ValueError, match=message):
            mt.build_modular(c)


def test_flat_mapping_is_reversible_and_overrides_reach_the_graph():
    c = mt._normalize(nested())
    flat = mc.flatten(c)
    assert flat["core.params.d_model"] == 16 and flat["branches.price.params.channels"] == 4
    assert mc.unflatten(flat) == c
    assert mc.dumps(c) == mc.dumps(json.loads(json.dumps(c)[::-1][::-1]))
    resolved, applied = mc.apply_flat_overrides(
        c, {"core.params.blocks": 1, "branches.volume.params.channels": 6, "epochs": 99})
    assert applied == {"branches.volume.params.channels": 6, "core.params.blocks": 1}
    b = mt.build_modular(resolved)
    attention = [l for l in b.core_model.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    assert len(attention) == 1
    assert b.branch_models["volume"].output_shape == (None, 12, 6)
    for key in ("branches.nope.params.channels", "core.bogus", "modular.branch_order", "head.params"):
        with pytest.raises(ValueError):
            mc.apply_flat_overrides(c, {key: 1})


# ------------------------------------------------- MS02, conditional params
@pytest.mark.parametrize("mutation, message", [
    (lambda c: c.update(sample_hours=1 / 60), "24h"),              # 24 minute observations
    (lambda c: c.update(sample_hours=0.25), "24h"),               # 24 x 15 min = 6 h
    (lambda c: c.update(output_steps=5), "divisib"),
    (lambda c: c["core"]["params"].update(heads=3), "divisib"),
    (lambda c: c["core"]["params"].update(stage_channels=[12, 10, 9, 8]), "stages"),
    (lambda c: c["core"]["params"].update(time_factors=[3, 1, 1]), "reduce"),
    (lambda c: c["branches"][0]["params"].update(dilation=2), "branch params"),
    (lambda c: c.update(schema="predictor.modular.v0"), "schema"),
    (lambda c: c.update(regime="R1"), "donor"),
])
def test_invalid_combinations_fail_in_build_model_before_fit(mutation, message):
    model = nested()
    mutation(model)
    plugin = Plugin()
    with pytest.raises(ValueError, match=message):
        plugin.build_model((24, 3), x_data(4), run_config(model))
    assert plugin.model is None and plugin.training_receipt is None


def test_legacy_keys_must_agree_with_the_modular_config():
    for extra, message in (({"window_size": 33}, "window_size"),
                           ({"predicted_horizons": [1, 6]}, "predicted_horizons"),
                           ({"feature_names": ["volume", "close", "spread"]}, "feature order")):
        with pytest.raises(ValueError, match=message):
            Plugin().build_model((24, 3), x_data(4), run_config(nested(), **extra))
    with pytest.raises(ValueError, match="Input shape"):
        Plugin().build_model((24, 4), x_data(4, features=4), run_config(nested()))


def test_four_stage_schedule_and_subhour_sampling():
    c = nested(window=48, sample_hours=0.5)
    c["core"]["params"].update(stage_channels=[14, 12, 10, 8], time_factors=[1, 2, 1, 1])
    b = mt.build_modular(c)
    assert b.branch_time_grid[-1] == 24.0 and b.encoder_model(x_data(2, window=48)).shape == (2, 6, 8)


# ------------------------------------------------------------ MS03-MS06
def test_default_plan_shapes_rank_three_and_independent_head():
    c = mt.default_config(["a", "b", "c"])
    c.update(horizons=[1, 6, 24], target_count=2)
    b = mt.build_modular(c)
    x = x_data(2)
    assert [m.output_shape for m in b.branch_models.values()] == [(None, 12, 16)] * 3
    assert b.fusion_model(x).shape == (2, 12, 48)
    assert b.encoder_model(x).shape == (2, 6, 8)
    assert b.forecast_model(x).shape == (2, 3, 2)
    core = b.core_model
    assert isinstance(core.layers[1], mt.PositionalEncoding)
    assert core.get_layer("model_projection").units == 64
    attention = [l for l in core.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    assert len(attention) == 2 and all(l.get_config()["num_heads"] == 4 for l in attention)
    assert len([l for l in core.layers if l.name.startswith("stage_") and l.name.endswith("_projection")]) == 3
    for model in [*b.branch_models.values(), b.fusion_model, core]:
        assert all(len(l.output.shape) == 3 for l in model.layers if hasattr(l, "output")
                   and not isinstance(l, tf.keras.layers.Reshape) or len(l.output.shape) == 3)
        assert not any(isinstance(l, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D))
                       for l in model.layers)


def test_different_branch_architectures_per_branch():
    b = mt.build_modular(nested())
    kernels = {n: [l.kernel_size[0] for l in m.layers if isinstance(l, tf.keras.layers.Conv1D)]
               for n, m in b.branch_models.items()}
    widths = {n: m.output_shape[-1] for n, m in b.branch_models.items()}
    assert kernels == {"price": [3], "volume": [3], "spread": [2]}
    assert widths == {"price": 4, "volume": 3, "spread": 2}


def _first_moved(model, x, index, step):
    changed = x.copy()
    changed[:, index, :] += 5.0
    moved = np.max(np.abs(np.asarray(model(changed)) - np.asarray(model(x))), axis=(0, 2))
    return int(np.argmax(moved > 1e-6)), moved


def test_time_alignment_is_behavioural_on_the_common_right_edge_grid():
    b = mt.build_modular(nested())
    x = x_data(2)
    for i in range(24):
        first, moved = _first_moved(b.fusion_model, x, i, 2)
        assert first == i // 2 and np.all(moved[:i // 2] <= 1e-6)   # branch grid 2,4,...,24 h
        # every branch, not only the concatenation, moves at the same block
        for name, m in b.branch_models.items():
            column = ["close", "volume", "spread"].index(b.config["branches"][
                [s["name"] for s in b.config["branches"]].index(name)]["features"][0])
            f, mv = _first_moved(m, x[:, :, [column]], i, 2)
            assert f == i // 2
        first, moved = _first_moved(b.encoder_model, x, i, 4)
        assert first == i // 4 and np.all(moved[:i // 4] <= 1e-6)   # latent grid 4,...,24 h
    assert b.branch_time_grid == tuple(range(2, 25, 2)) and b.core_time_grid == tuple(range(4, 25, 4))


def test_same_shape_but_misaligned_plugins_are_rejected(monkeypatch):
    @mt.component("branch", "1.0.0", {"channels", "kernel_size"}, "time reversed (defective)")
    def reversed_branch(**kwargs):
        good = mt.causal_conv1d(**kwargs)
        inputs = tf.keras.Input(kwargs["input_shape"])
        flipped = tf.keras.layers.Lambda(lambda t: tf.reverse(t, axis=[1]))(inputs)
        return mt.TemporalComponent(tf.keras.Model(inputs, good.model(flipped)), good.time_grid)

    @mt.component("branch", "1.0.0", {"channels", "kernel_size"}, "drops newest sample (defective)")
    def lagged_branch(**kwargs):
        good = mt.causal_conv1d(**kwargs)
        inputs = tf.keras.Input(kwargs["input_shape"])
        shifted = tf.keras.layers.Lambda(lambda t: tf.pad(t[:, :-1], [[0, 0], [1, 0], [0, 0]]))(inputs)
        return mt.TemporalComponent(tf.keras.Model(inputs, good.model(shifted)), good.time_grid)

    class EP:
        def __init__(self, f):
            self.f, self.value, self.dist = f, "test:f", None

        def load(self):
            return self.f

    table = {"reversed": [EP(reversed_branch)], "lagged": [EP(lagged_branch)]}
    monkeypatch.setattr(mt, "entry_points", lambda *, group, name: table.get(name, []))
    for name in table:
        c = nested()
        c["branches"][1]["plugin"] = name
        with pytest.raises(ValueError, match="time alignment"):
            mt.build_modular(c)


# ------------------------------------------------------------ MS07, MS08
def _pretrained_donors(tmp_path):
    """Donors whose weights differ from any fresh initialization (one AE step each)."""
    base = mt.build_modular(nested())
    x = x_data(16, seed=11)
    cols = {"price": [0], "volume": [1], "spread": [2]}
    specs = copy.deepcopy(base.config)
    for spec in specs["branches"]:
        name = spec["name"]
        ae = mt.build_autoencoder(base.branch_models[name])
        ae.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
        ae.train_on_batch(x[:, :, cols[name]], x[:, :, cols[name]])
        path = tmp_path / f"{name}.keras"
        mt.save_donor(base.branch_models[name], path, base.donor_manifest("branch", name))
        spec.update(regime=None, donor=str(path))
    raw = base.fusion_model(x).numpy()
    core_ae = mt.build_autoencoder(base.core_model)
    core_ae.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss="mse")
    core_ae.train_on_batch(raw, raw)
    core = tmp_path / "core.keras"
    mt.save_donor(base.core_model, core, base.donor_manifest("core"))
    specs["core"].update(regime=None, donor=str(core))
    donor_hashes = {n: mt.weights_hash(m) for n, m in base.branch_models.items()}
    donor_hashes["core"] = mt.weights_hash(base.core_model)
    return specs, donor_hashes


def _arm(specs, regime):
    c = copy.deepcopy(specs)
    c["regime"] = regime
    for spec in [*c["branches"], c["core"]]:
        spec.pop("regime", None)
        if regime == "R0":
            spec["donor"] = None
    return c


def test_three_arm_regimes_observe_actual_optimizer_updates(tmp_path):
    specs, donor = _pretrained_donors(tmp_path)
    x, vx = x_data(32, seed=5), x_data(16, seed=6)
    receipts, initial = {}, {}
    for regime in ("R0", "R1", "R2"):
        tf.keras.utils.set_random_seed(7)
        plugin = Plugin()
        plugin.build_model((24, 3), x, run_config(_arm(specs, regime)))
        initial[regime] = {n: mt.weights_hash(m) for n, m in plugin.bundle.branch_models.items()}
        initial[regime]["core"] = mt.weights_hash(plugin.bundle.core_model)
        history, tp, tu, vp, vu = plugin.train(x, y_dict(x), x_val=vx, y_val=y_dict(vx))
        r = plugin.training_receipt
        receipts[regime] = r
        assert r["regimes"]["common"] == regime
        # 32 rows / batch 8 = 4 observed updates per completed epoch, counted from the optimizer
        assert r["observed_updates"] == 4 * r["epochs_completed"]
        assert r["optimizer_iterations"] - r["initial_optimizer_iterations"] == r["observed_updates"]
        assert len(history.history["loss"]) == r["epochs_completed"]
        assert [p.shape for p in tp] == [(32, 1), (32, 1)] and [p.shape for p in vp] == [(16, 1)] * 2
        assert r["components"]["head"]["weights_changed"]
    # R0 starts from fresh weights, not the donor
    assert all(initial["R0"][k] != donor[k] for k in donor)
    # R1 and R2 start from the identical verified donor weights
    assert initial["R1"] == initial["R2"] == donor
    for name, comp in receipts["R0"]["components"].items():
        assert comp["weights_changed"] and comp["trainable_parameters"] > 0
    for name, comp in receipts["R1"]["components"].items():
        if name != "head":
            assert comp["regime"] == "R1" and comp["trainable_parameters"] == 0
            assert not comp["weights_changed"] and comp["weights_sha256_after"] == donor[name.split(":")[-1]]
    for name, comp in receipts["R2"]["components"].items():
        assert comp["weights_changed"] and comp["trainable_parameters"] > 0


def test_mixed_regimes_are_recorded_not_conflated(tmp_path):
    specs, _ = _pretrained_donors(tmp_path)
    c = _arm(specs, "R2")
    c["regime"] = None
    for spec in c["branches"]:
        spec["regime"] = "R1"
    c["core"]["regime"] = "R2"
    summary = mt.regime_summary(c)
    assert summary["common"] == "MIXED" and summary["declared"] is False
    contradicting = copy.deepcopy(c)
    contradicting["regime"] = "R2"
    with pytest.raises(ValueError, match="common regime"):
        mt.build_modular(contradicting)


def _bad(specs, tmp_path, kind):
    c = _arm(specs, "R2")
    if kind == "missing_file":
        c["branches"][0]["donor"] = str(tmp_path / "absent.keras")
    elif kind == "archive_digest":
        with open(c["branches"][1]["donor"], "ab") as f:
            f.write(b"x")
    elif kind == "manifest_digest":
        side = tmp_path / "price.manifest.json"
        doc = json.loads(side.read_text())
        doc["manifest"]["params"]["channels"] = 5
        side.write_text(json.dumps(doc))
    elif kind == "feature_order":
        c["feature_names"] = ["volume", "close", "spread"]
    elif kind == "branch_feature_swap":
        c["branches"][0]["features"], c["branches"][1]["features"] = ["volume"], ["close"]
    elif kind == "input_time_shape":
        c.update(window=48, sample_hours=0.5)
    elif kind == "sampling_period":
        c.update(window=12, sample_hours=2, branch_steps=6, output_steps=3)
    elif kind == "upstream_identity":
        c["regime"] = None
        for spec in c["branches"]:
            spec.update(regime="R0", donor=None)
        c["core"]["regime"] = "R2"
    elif kind == "core_for_other_branch_params":
        c["branches"][2]["params"]["channels"] = 5
        c["branches"][2].update(donor=None)
        c["regime"] = None
        for spec in c["branches"]:
            spec["regime"] = "R0" if spec["donor"] is None else "R2"
        c["core"]["regime"] = "R2"
    return c


@pytest.mark.parametrize("kind", ["missing_file", "archive_digest", "manifest_digest", "feature_order",
                                  "branch_feature_swap", "input_time_shape", "sampling_period",
                                  "upstream_identity", "core_for_other_branch_params"])
def test_explicit_bad_donors_are_rejected_before_fit(tmp_path, monkeypatch, kind):
    specs, _ = _pretrained_donors(tmp_path)
    fits = []
    import tools.modular_candidate_evaluator as ev
    monkeypatch.setattr(ev, "fit_with_early_stopping", lambda *a, **k: fits.append(1))
    c = _bad(specs, tmp_path, kind)
    plugin = Plugin()
    window = c["window"]
    with pytest.raises(ValueError):
        plugin.build_model((window, 3), x_data(4, window=window), run_config(c))
    assert plugin.model is None and fits == []


# ------------------------------------------------------ serialization
def test_facade_save_load_reproduces_outputs_and_detects_tampering(tmp_path):
    specs, _ = _pretrained_donors(tmp_path)
    x, vx = x_data(16, seed=5), x_data(8, seed=6)
    plugin = Plugin()
    plugin.build_model((24, 3), x, run_config(_arm(specs, "R1")))
    plugin.train(x, y_dict(x), x_val=vx, y_val=y_dict(vx))
    path = tmp_path / "saved" / "model.keras"
    path.parent.mkdir()
    plugin.save(str(path))
    expected, _ = plugin.predict_with_uncertainty(vx)
    restored = Plugin()
    restored.load(str(path))
    got, unc = restored.predict_with_uncertainty(vx)
    for a, b in zip(expected, got):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert restored.modular_config == plugin.modular_config
    assert not restored.bundle.core_model.trainable          # R1 regime survives reload
    receipt = json.loads((tmp_path / "saved" / "model.keras.bundle" / "training_receipt.json").read_text())
    assert receipt["observed_updates"] > 0
    # the plain Keras path the legacy pipelines use also reproduces the outputs
    plain = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(plain.predict(vx, verbose=0)[0], expected[0], atol=1e-6)
    archive = tmp_path / "saved" / "model.keras.bundle" / "forecast_model.keras"
    with archive.open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        Plugin().load(str(path))


def test_legacy_stl_pipeline_build_and_train_contract():
    from pipeline_plugins.stl_pipeline import STLPipelinePlugin
    x, vx = x_data(16, seed=5), x_data(8, seed=6)
    plugin = Plugin()
    history, tp, tu, vp, vu = STLPipelinePlugin()._build_and_train(
        plugin, x, y_dict(x), vx, y_dict(vx), run_config(nested()))
    assert plugin.output_names == ["output_horizon_1", "output_horizon_3"]
    assert [o.name.split("/")[0] for o in plugin.model.outputs] == plugin.output_names
    assert "loss" in history.history and "val_loss" in history.history
    assert all(np.all(u == 0) for u in vu) and len(vp) == 2
