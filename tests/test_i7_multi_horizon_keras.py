"""Real Keras vector head, checkpoint and joint-monitor integration checks."""

import numpy as np
import pytest


def test_vector_head_roundtrip_and_joint_monitor(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from predictor_plugins.modular_temporal import build_modular, default_config
    from tools.i7_multi_horizon import HORIZONS, check_prediction
    from tools.modular_candidate_evaluator import fit_with_early_stopping

    tf.keras.utils.set_random_seed(0)
    config = default_config(["a", "b"])
    config["horizons"] = list(HORIZONS)
    bundle = build_modular(config)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 24, 2)).astype("float32")
    y = rng.normal(size=(8, 11, 1)).astype("float32")
    fit = fit_with_early_stopping(bundle.forecast_model, x[:6], y[:6], x[6:], y[6:],
                                   dict(max_epochs=2, batch_size=2, patience=2,
                                        monitor="train_validation_mean", max_seconds=120))
    monitored = [r for r in fit["history"] if r["monitored"]]
    for row in monitored:
        assert row["monitored_loss"] == pytest.approx(
            (row["train_loss"] + row["validation_loss"]) / 2)
    assert fit["best_monitored_loss"] == min(r["monitored_loss"] for r in monitored)
    assert fit["observed_updates"] == 6
    expected = bundle.forecast_model(x, training=False).numpy()
    assert check_prediction(expected, len(x)).shape == (8, 11)
    path = tmp_path / "multi.keras"
    bundle.forecast_model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_array_equal(expected, loaded(x, training=False).numpy())
