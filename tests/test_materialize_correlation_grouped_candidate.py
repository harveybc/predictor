"""Tests for deterministic TRAIN-only correlation grouping."""

import json

import numpy as np
import pytest

from tools import materialize_correlation_grouped_candidate as subject


def test_correlation_groups_are_deterministic_partition():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=200)
    observations = np.column_stack([
        latent, latent + rng.normal(0, 0.01, 200),
        -latent + rng.normal(0, 0.01, 200), rng.normal(size=200),
    ])
    windows = np.repeat(observations[:, None, :], 3, axis=1)
    first = subject.correlation_groups(windows, ["a", "b", "c", "d"], 2)
    second = subject.correlation_groups(windows, ["a", "b", "c", "d"], 2)
    assert first == second
    assert sorted(sum(first, [])) == ["a", "b", "c", "d"]
    assert set(["a", "b", "c"]).issubset(next(set(group) for group in first if "a" in group))


def test_materialize_preserves_branch_count_and_rejects_validation(tmp_path):
    model = {
        "feature_names": ["a", "b", "c", "d"],
        "branches": [
            {"name": "b0", "features": ["a", "b"], "plugin": "causal_conv1d",
             "params": {"channels": 16}, "regime": "R0", "donor": None},
            {"name": "b1", "features": ["c", "d"], "plugin": "causal_conv1d",
             "params": {"channels": 16}, "regime": "R0", "donor": None},
        ],
        "core": {"plugin": "transformer_conv", "params": {}, "regime": "R0", "donor": None},
    }
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"model": model, "modular_candidate": {}}))
    data = tmp_path / "train.npz"
    np.savez(data, windows=np.random.default_rng(1).normal(size=(30, 3, 4)),
             feature_names=np.asarray(model["feature_names"]), split=np.asarray("train"))

    receipt = subject.materialize(candidate, data, tmp_path / "out", 2)

    assert receipt["group_sizes"] and sum(receipt["group_sizes"]) == 4
    generated = json.loads((tmp_path / "out/CANDIDATE_correlation.json").read_text())
    assert len(generated["model"]["branches"]) == 2
    assert generated["modular_candidate"]["grouping"]["train_npz"]["sha256"]

    validation = tmp_path / "validation.npz"
    np.savez(validation, windows=np.zeros((2, 3, 4)),
             feature_names=np.asarray(model["feature_names"]), split=np.asarray("validation"))
    with pytest.raises(ValueError, match="TRAIN"):
        subject.materialize(candidate, validation, tmp_path / "bad", 2)
