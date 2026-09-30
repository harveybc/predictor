"""Behavioral specification for staged, train-only representation pretraining."""
import copy
import numpy as np
import pytest

from tools.modular_pretrain import pretrain_components


def populations():
    return ({"split": "train", "dataset_id": "synthetic", "support_start": 0,
             "support_end": 86400, "time_unit": "seconds"},
            {"split": "train_validation", "dataset_id": "synthetic",
             "support_start": 90000, "support_end": 172800, "time_unit": "seconds"})


@pytest.mark.parametrize("mutation", [
    lambda t, v: v.update(split="test"),
    lambda t, v: v.update(split="validation"),
    lambda t, v: v.update(support_start=86400),
    lambda t, v: v.update(dataset_id="other"),
    lambda t, v: t.update(support_end=True),
])
def test_rejects_bad_split_before_model_load(tmp_path, mutation):
    t, v = populations()
    mutation(t, v)
    with pytest.raises(ValueError):
        pretrain_components({}, np.zeros((2, 24, 2)), np.zeros((2, 24, 2)),
                            tmp_path / "run", {}, t, v)
    assert not (tmp_path / "run").exists()


def test_all_stages_export_reload_and_real_updates(tmp_path):
    from predictor_plugins.modular_temporal import build_modular, default_config
    config = default_config(["a", "b"])
    original = copy.deepcopy(config)
    rng = np.random.default_rng(91)
    train = rng.normal(size=(8, 24, 2)).astype("float32")
    val = rng.normal(size=(4, 24, 2)).astype("float32")
    t, v = populations()
    result = pretrain_components(config, train, val, tmp_path / "run",
                                {"max_epochs": 2, "patience": 1, "batch_size": 4,
                                 "learning_rate": 0.001, "loss": "mse"}, t, v)
    assert config == original
    assert len(result["branches"]) == 2
    assert result["core"]["training"]["observed_updates"] > 0
    assert all(s["training"]["observed_updates"] > 0 for s in result["branches"])
    fitted = build_modular(result["fine_tune_config"])
    assert tuple(fitted.encoder_model(train[:1]).shape) == (1, 6, 8)
    frozen_config = copy.deepcopy(result["fine_tune_config"])
    for component in [*frozen_config["branches"], frozen_config["core"]]:
        component["regime"] = "R1"
    frozen = build_modular(frozen_config)
    np.testing.assert_allclose(fitted.encoder_model(val), frozen.encoder_model(val), atol=1e-6)
    assert not frozen.core_model.trainable_weights
    assert (tmp_path / "run" / "PRETRAIN.json").exists()
    assert result["fused_train"]["shape"][:2] == [8, 24]
