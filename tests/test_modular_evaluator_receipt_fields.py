"""Every evaluator path writes the receipt fields M04's checkpoint scorer reads (finding D-R3-VERIFY-01).

The scorer (tools/modular_checkpoint_scorer.verify, M04) reads: schema_version, status,
artifacts.best_model, digests.model_sha256, digests.validation_sha256, metrics, objective, per_horizon,
and training.settings.batch_size (the inference batch it replays). R3 omitted training.settings at
3b073d2e; this pins it for R0, R1, R2, R3, target_residual, window_mean and extra_channels.
Synthetic component checks through the real evaluate_candidate.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json

import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from tests.test_modular_candidate_evaluator import inputs
from tests.test_modular_evaluator_r3 import OPERATIONAL, candidate, model_config
from tools import modular_candidate_evaluator as evaluator


def _donors(c, tmp_path, regime, **extra):
    tf.keras.utils.set_random_seed(4)
    base = mt.build_modular(c)
    for spec in c["branches"]:
        p = tmp_path / f"{spec['name']}.keras"
        mt.save_donor(base.branch_models[spec["name"]], p, base.donor_manifest("branch", spec["name"]),
                      provenance=OPERATIONAL)
        spec.update(regime=regime, donor=str(p), **extra)
    return c


def build(path, tmp_path):
    c = model_config()
    if path == "R0":
        return c
    if path in ("R1", "R2"):
        return _donors(c, tmp_path, path)
    if path == "R3":
        return _donors(c, tmp_path, "R3", freeze_epochs=1, unfreeze_learning_rate=1e-3)
    if path == "target_residual":
        c["target_residual"] = {"kind": "seasonal_naive_cumulative", "period": 6, "target_features": ["close"]}
        return c
    if path == "window_mean":
        c["input_normalization"] = {"kind": "window_mean", "length": 24, "target_features": ["close"]}
        return c
    if path == "extra_channels":
        e = mt.config_with_extra_channels(["close"], {"close": ["volume"]})
        e.update(horizons=c["horizons"], target_count=1, sample_hours=c["sample_hours"])
        return e
    raise AssertionError(path)


@pytest.mark.parametrize("path", ["R0", "R1", "R2", "R3", "target_residual", "window_mean", "extra_channels"])
def test_every_evaluator_path_writes_the_fields_the_scorer_reads(tmp_path, path):
    tf.keras.backend.clear_session()
    cand = candidate(build(path, tmp_path))
    result = evaluator.evaluate_candidate(cand, *inputs(tmp_path), tmp_path / "out")
    written = json.loads((tmp_path / "out" / "evaluation.json").read_text())
    assert written == json.loads(json.dumps(result))
    assert written["schema_version"] == "modular.candidate.evaluation.v1" and written["status"] == "completed"
    assert os.path.isfile(written["artifacts"]["best_model"])
    assert len(written["digests"]["model_sha256"]) == 64 and len(written["digests"]["validation_sha256"]) == 64
    assert written["metrics"]["MAE"] > 0 and written["objective"]["value"] is not None
    assert set(written["per_horizon"]) == {"1", "3"}
    settings = written["training"]["settings"]
    assert settings["batch_size"] == cand["evaluator"]["batch_size"] and "progress" not in settings
