"""Evaluator acceptance trace (synthetic calibration, not domain acceptance).

E1 identity/leakage -> test_boundary; E2 metrics -> test_metrics;
E3 bounded real updates/best restore/reload -> test_training;
E4 real modular integration -> test_real_engine_smoke (must run, not just skip).
Design: local NPZ -> validated arrays -> lazy builder -> manual AdamW loop ->
restored forecast artifact -> reload -> measured validation-only objective.
No plan/state files are changed: this lane owns exactly these two Python files.
"""

import copy
import importlib.util
import sys
import types

import numpy as np
import pytest

from tools import modular_candidate_evaluator as evaluator


def candidate():
    return {
        "model": {"window": 24, "sample_hours": 1,
                  "feature_names": ["close", "volume"],
                  "horizons": [1, 3], "target_count": 1},
        "target_feature_indices": [0],
        "evaluator": {"max_epochs": 3, "patience": 1, "batch_size": 4,
                      "learning_rate": 0.001, "seed": 17},
    }


def arrays(split):
    rng = np.random.default_rng(12 if split == "train" else 34)
    x = rng.normal(size=(8, 24, 2)).astype("float32")
    timestamps = np.arange(8, dtype="int64") * 3600 + (100000 if split == "train" else 500000)
    return dict(windows=x, targets=np.repeat(x[:, -1:, :1], 2, axis=1) + 0.3,
                row_ids=np.array([f"{split}-{i}" for i in range(8)]),
                timestamps=timestamps,
                target_timestamps=timestamps[:, None] + np.array([1, 3]) * 3600,
                dataset_id=np.array("synthetic-v1"), split=np.array(split),
                feature_names=np.array(["close", "volume"]),
                target_names=np.array(["close"]), horizons=np.array([1, 3]),
                timestamp_unit=np.array("seconds"), metric_space=np.array("z_train"),
                scaler_identity=np.array("fixture-train-scaler-v1"), scaler_scale=np.array([2.]))


def inputs(tmp_path, mutate=None):
    train, validation = arrays("train"), arrays("validation")
    if mutate:
        mutate(train, validation)
    paths = tmp_path / "train.npz", tmp_path / "validation.npz"
    for path, data in zip(paths, (train, validation)):
        np.savez(path, **data)
    return paths


@pytest.mark.parametrize("mutation,match", [
    (lambda t, v: v.pop("row_ids"), "missing"),
    (lambda t, v: v.update(split=np.array("test")), "split"),
    (lambda t, v: v.update(row_ids=t["row_ids"]), "overlap"),
    (lambda t, v: t["targets"].fill(np.nan), "finite"),
    (lambda t, v: v.update(timestamps=t["timestamps"], target_timestamps=t["target_timestamps"]), "chronolog"),
    (lambda t, v: v.update(horizons=np.array([3, 1])), "horizons"),
    (lambda t, v: v.update(dataset_id=np.array("different")), "dataset"),
    (lambda t, v: v.update(target_names=np.array(["volume"])), "target"),
    (lambda t, v: t.update(test_targets=np.zeros(1)), "unexpected"),
    (lambda t, v: v.update(scaler_identity=np.array("validation-fit")), "scaler_identity"),
    (lambda t, v: v.update(scaler_scale=np.array([3.])), "scaler_scale"),
    (lambda t, v: v.update(metric_space=np.array("price")), "metric_space"),
    (lambda t, v: v.update(timestamp_unit=np.array("milliseconds")), "seconds"),
    (lambda t, v: v.update(target_timestamps=v["target_timestamps"] + 1), "target_timestamps"),
    (lambda t, v: v.update(horizons=np.array(1)), "horizons"),
])
def test_boundary(tmp_path, mutation, match):
    paths = inputs(tmp_path, mutation)
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate_candidate(candidate(), *paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("key,value", [("max_epochs", 0), ("max_epochs", 101),
                                       ("patience", 0), ("patience", 51),
                                       ("batch_size", True), ("loss", "bogus"),
                                       ("learning_rate", float("inf"))])
def test_budget_refusal(tmp_path, key, value):
    config = candidate()
    config["evaluator"][key] = value
    with pytest.raises(ValueError):
        evaluator.evaluate_candidate(config, *inputs(tmp_path), tmp_path / "out")


def test_metrics():
    y = np.array([[[3., 6.], [5., 10.]]])
    p = y - 1
    baseline = y - 2
    scores = evaluator._metrics(y, p, baseline)
    assert scores["MAE"] == 1
    assert scores["MSE"] == 1
    assert scores["skill_MAE"] == 0.5
    assert scores["skill_MSE"] == 0.75
    assert evaluator._metrics(y, p, y)["skill_MAE"] is None


@pytest.fixture
def fake_builder(monkeypatch):
    tf = pytest.importorskip("tensorflow")
    built = []

    def build(config):
        assert "evaluator" not in config and "target_feature_indices" not in config
        x = tf.keras.Input((24, 2))
        y = tf.keras.layers.Dense(2)(tf.keras.layers.Flatten()(x))
        model = tf.keras.Model(x, tf.keras.layers.Reshape((2, 1))(y))
        built.append(model)
        return types.SimpleNamespace(forecast_model=model)

    monkeypatch.setitem(sys.modules, "predictor_plugins.modular_temporal",
                        types.SimpleNamespace(build_modular=build))
    return built


@pytest.mark.parametrize("loss", ["huber", "mae", "mse"])
def test_training(tmp_path, fake_builder, loss):
    config = candidate()
    config["evaluator"].update(loss=loss, min_delta=1e6)
    original = copy.deepcopy(config)
    paths = inputs(tmp_path)
    result = evaluator.evaluate_candidate(config, *paths, tmp_path / "out")
    assert config == original
    assert result["training"]["epochs_completed"] == 2
    assert result["training"]["best_epoch"] == 1
    assert result["training"]["observed_updates"] == 4
    assert result["training"]["optimizer_iterations"] == 4
    assert result["reload_parity"]["passed"]
    assert result["objective"]["split"] == "validation"
    assert result["objective"]["value"] == result["metrics"]["MAE"]
    assert result["data"]["metric_space"] == "z_train"
    assert result["training"]["stop_reason"] == "patience"
    assert result["training"]["selected_updates"] == 2
    assert set(result["per_horizon"]) == {"1", "3"}
    assert result["metrics"]["baseline_MAE"] == pytest.approx(0.3, abs=1e-6)
    assert all(len(v) == 64 for v in result["digests"].values())
    assert (tmp_path / "out" / "best.keras").exists()
    assert (tmp_path / "out" / "evaluation.json").exists()


def test_real_engine_smoke(tmp_path):
    if importlib.util.find_spec("predictor_plugins.modular_temporal") is None:
        pytest.skip("real modular engine not installed in this worktree")
    from predictor_plugins import modular_temporal as mt

    config = candidate()
    model = mt.default_config(["close", "volume"])
    model.update(window=24, sample_hours=1, horizons=[1, 3], target_count=1)
    config["model"] = model
    result = evaluator.evaluate_candidate(config, *inputs(tmp_path), tmp_path / "real")
    assert result["training"]["observed_updates"] > 0
    assert result["reload_parity"]["passed"]
    assert np.isfinite(result["objective"]["value"])


def test_purge_against_input_not_origin(tmp_path):
    def overlap(t, v):
        v["timestamps"] = t["timestamps"] + 12 * 3600
        v["target_timestamps"] = v["timestamps"][:, None] + np.array([1, 3]) * 3600

    with pytest.raises(ValueError, match="input support"):
        evaluator.evaluate_candidate(candidate(), *inputs(tmp_path, overlap), tmp_path / "out")


def test_fit_lazy_batches_custom_compile_and_restore(fake_builder, monkeypatch):
    import tensorflow as tf

    class View:
        def __init__(self, value):
            self.value, self.shape, self.dtype = value, value.shape, value.dtype
            self.largest_slice = 0

        def __len__(self):
            return len(self.value)

        def __array__(self, *args, **kwargs):
            raise AssertionError("must never materialize whole view")

        def __getitem__(self, key):
            result = self.value[key]
            self.largest_slice = max(self.largest_slice, len(result))
            return result

    from predictor_plugins.modular_temporal import build_modular

    model = build_modular(candidate()["model"]).forecast_model
    x, y = View(arrays("train")["windows"]), View(arrays("train")["targets"])
    selected = []
    calls = 0

    def validation_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            selected.extend(w.copy() for w in model.get_weights())
        return 1.0 if calls <= 2 else 2.0

    monkeypatch.setattr(model, "test_on_batch", validation_loss)
    result = evaluator.fit_with_early_stopping(
        model, x, y, x, y,
        dict(max_epochs=3, patience=1, batch_size=4,
             optimizer=tf.keras.optimizers.AdamW(0.001), loss=tf.keras.losses.MeanSquaredError()))
    assert result["observed_updates"] == 4
    assert result["selected_updates"] == 2
    assert result["selected_epoch"] == 1
    assert result["stop_reason"] == "patience"
    assert x.largest_slice <= 4 and y.largest_slice <= 4
    for actual, expected in zip(model.get_weights(), selected):
        np.testing.assert_array_equal(actual, expected)


def test_fit_precompiled_and_update_budget(fake_builder):
    from predictor_plugins.modular_temporal import build_modular

    model = build_modular(candidate()["model"]).forecast_model
    model.compile(optimizer="adam", loss="mse")
    x, y = arrays("train")["windows"], arrays("train")["targets"]
    result = evaluator.fit_with_early_stopping(
        model, x, y, x, y,
        dict(compile=False, batch_size=4, max_epochs=5, max_updates=3))
    assert result["observed_updates"] == 3
    assert result["selected_updates"] == 2
    assert result["epochs_completed"] == 1
    assert result["stop_reason"] == "max_updates"
