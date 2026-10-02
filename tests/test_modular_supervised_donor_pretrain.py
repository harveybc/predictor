"""Contract tests for supervised seasonal-residual donor pretraining."""

import json

import numpy as np
import pytest

from tools import modular_supervised_donor_pretrain as subject


def _timestamps(rows, horizons=(1, 2)):
    origins = np.arange(rows, dtype=np.int64) * 3600
    targets = origins[:, None] + np.asarray(horizons, dtype=np.int64)[None, :] * 3600
    return origins, targets


def test_internal_split_is_chronological_purged_and_label_disjoint():
    origins, targets = _timestamps(100, horizons=(1, 2, 3, 4))
    split = subject.internal_split_indices(origins, targets, validation_fraction=0.2, purge_origins=4)

    assert split["train"].tolist() == list(range(76))
    assert split["validation"].tolist() == list(range(80, 100))
    assert targets[split["train"]].max() < origins[split["validation"]].min()


def test_internal_split_rejects_insufficient_purge_and_bad_fraction():
    origins, targets = _timestamps(100, horizons=(1, 2, 3, 4))
    with pytest.raises(ValueError, match="LABEL_OVERLAP"):
        subject.internal_split_indices(origins, targets, validation_fraction=0.2, purge_origins=3)
    with pytest.raises(ValueError, match="validation_fraction"):
        subject.internal_split_indices(origins, targets, validation_fraction=1.0, purge_origins=4)


def test_train_loader_refuses_non_train_and_nonfinite_arrays(tmp_path):
    origins, target_times = _timestamps(12)
    common = dict(
        windows=np.zeros((12, 4, 2), dtype=np.float32),
        targets=np.zeros((12, 2, 2), dtype=np.float32),
        timestamps=origins,
        target_timestamps=target_times,
        row_ids=np.asarray([f"r{i}" for i in range(12)]),
        feature_names=np.asarray(["a", "b"]),
        target_names=np.asarray(["a", "b"]),
        horizons=np.asarray([1, 2]),
        split=np.asarray("validation"),
        dataset_id=np.asarray("fixture"),
        timestamp_unit=np.asarray("seconds"),
        metric_space=np.asarray("z_train"),
        scaler_identity=np.asarray("fixture-scaler"),
        scaler_scale=np.ones(2),
    )
    path = tmp_path / "not_train.npz"
    np.savez(path, **common)
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        subject.load_train_npz(path)

    common["split"] = np.asarray("train")
    common["windows"][0, 0, 0] = np.nan
    path = tmp_path / "nonfinite.npz"
    np.savez(path, **common)
    with pytest.raises(ValueError, match="finite"):
        subject.load_train_npz(path)


def test_expand_recipe_preserves_time_and_builds_one_branch_per_feature(tmp_path):
    recipe = {
        "schema": "modular.supervised_donor.recipe.v1",
        "window": 24,
        "sample_hours": 1,
        "horizons": list(range(1, 25)),
        "branch": {"plugin": "causal_conv1d", "params": {"channels": 16, "kernel_size": 3}},
        "core": {"plugin": "transformer_conv", "params": {"blocks": 2, "d_model": 64,
                 "heads": 4, "ff_dim": 128, "dropout": 0.0, "kernel_size": 3,
                 "stage_channels": [32, 16, 8], "time_factors": [2, 2, 1]}},
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "head": {"plugin": "forecast", "params": {}},
        "output_steps": 6,
        "output_channels": 8,
        "target_residual_period": 24,
    }
    model = subject.expand_recipe(recipe, ["a", "b"])
    assert model["window"] == model["branch_steps"] == 24
    assert model["output_steps"] == 6
    assert [b["features"] for b in model["branches"]] == [["a"], ["b"]]
    assert all(b["regime"] == "R0" and b["donor"] is None for b in model["branches"])
    assert model["target_residual"]["target_features"] == ["a", "b"]


def test_expand_recipe_accepts_partitioned_branches_and_distinct_target():
    recipe = {
        "schema": "modular.supervised_donor.recipe.v1", "window": 24,
        "sample_hours": 4, "horizons": [6, 12],
        "branches": [
            {"name": "returns", "features": ["target", "r5"],
             "plugin": "causal_conv1d", "params": {"channels": 16}},
            {"name": "volume", "features": ["volume"],
             "plugin": "causal_conv1d", "params": {"channels": 16}},
        ],
        "core": {"plugin": "transformer_conv", "params": {}},
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "head": {"plugin": "forecast", "params": {}},
        "output_steps": 6, "output_channels": 8, "target_residual_period": 6,
    }
    model = subject.expand_recipe(recipe, ["target", "r5", "volume"], ["target"])
    assert [branch["name"] for branch in model["branches"]] == ["returns", "volume"]
    assert model["target_count"] == 1
    assert model["target_residual"] == {
        "kind": "seasonal_naive", "period": 6, "target_features": ["target"]
    }


def test_expand_recipe_rejects_non_partition_and_unknown_target():
    recipe = {
        "schema": "modular.supervised_donor.recipe.v1", "window": 24,
        "sample_hours": 4, "horizons": [6],
        "branches": [{"name": "bad", "features": ["a"],
                      "plugin": "causal_conv1d", "params": {}}],
        "core": {"plugin": "transformer_conv", "params": {}},
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "head": {"plugin": "forecast", "params": {}},
        "output_steps": 6, "output_channels": 8, "target_residual_period": 6,
    }
    with pytest.raises(ValueError, match="partition"):
        subject.expand_recipe(recipe, ["a", "b"], ["a"])
    recipe["branches"][0]["features"] = ["a", "b"]
    with pytest.raises(ValueError, match="not input features"):
        subject.expand_recipe(recipe, ["a", "b"], ["missing"])


def test_regime_configs_bind_every_exported_donor(tmp_path):
    base = subject.expand_recipe({
        "schema": "modular.supervised_donor.recipe.v1", "window": 24, "sample_hours": 1,
        "horizons": [1], "branch": {"plugin": "causal_conv1d", "params": {}},
        "core": {"plugin": "transformer_conv", "params": {}},
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "head": {"plugin": "forecast", "params": {}}, "output_steps": 6,
        "output_channels": 8, "target_residual_period": 24,
    }, ["a", "b"])
    donors = {"branches": {"branch_0": "/d/b0.keras", "branch_1": "/d/b1.keras"},
              "core": "/d/core.keras"}
    configs = subject.regime_configs(base, donors, freeze_epochs=3, unfreeze_learning_rate=1e-4)
    assert set(configs) == {"R1", "R2", "R3"}
    assert all(b["regime"] == "R1" and b["donor"] for b in configs["R1"]["branches"])
    assert configs["R2"]["core"]["regime"] == "R2"
    assert configs["R3"]["core"]["freeze_epochs"] == 3
    assert configs["R3"]["core"]["unfreeze_learning_rate"] == 1e-4


def test_receipt_identity_changes_with_split_or_training_bytes():
    base = {"objective": "seasonal_residual_forecast_P24", "train_sha256": "a" * 64,
            "train_rows": [0, 70], "validation_rows": [80, 100]}
    first = subject.identity(base)
    assert first == subject.identity(dict(base))
    assert first != subject.identity({**base, "train_rows": [0, 69]})
    assert first != subject.identity({**base, "train_sha256": "b" * 64})
