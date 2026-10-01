"""R3 dispatch in the DOIN evaluator (M04 gap on 9223391d; coordinator order).

``evaluate_candidate`` routes any candidate with an R3 component to ``warm.fit_warm`` and keeps every
existing receipt field (observed_updates, selected_epoch, stop_reason, early_stop, reload parity,
digests) while adding the warm fields (unfreeze epoch, per-phase component hashes, per-phase updates).
R0/R1/R2 candidates never reach fit_warm and keep their exact training receipt layout.
Synthetic component checks.
"""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import warm
from tools import modular_candidate_evaluator as evaluator
from tests.test_modular_candidate_evaluator import inputs

OPERATIONAL = {"conditioning_contract": "OPERATIONAL",
               "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": "fixture:train", "data_sha256": "b" * 64,
                                  "support": "fixture rows", "pretrained_weights_source": None},
               "reconstruction": {"state": "NOT_EVALUATED"}}


def model_config():
    c = mt.default_config(["close", "volume"])
    c.update(horizons=[1, 3], target_count=1)
    return c


def candidate(model):
    return {"model": model, "target_feature_indices": [0],
            "evaluator": {"max_epochs": 3, "patience": 3, "batch_size": 4, "learning_rate": 0.001, "seed": 17}}


def r3_model(tmp_path):
    c = model_config()
    tf.keras.utils.set_random_seed(4)
    base = mt.build_modular(c)
    for spec in c["branches"]:
        p = tmp_path / f"{spec['name']}.keras"
        mt.save_donor(base.branch_models[spec["name"]], p, base.donor_manifest("branch", spec["name"]),
                      provenance=OPERATIONAL)
        spec.update(regime="R3", donor=str(p), freeze_epochs=1, unfreeze_learning_rate=1e-3)
    return c


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = warm.fit_warm

    def wrapper(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(warm, "fit_warm", wrapper)
    return calls


def test_r3_candidate_reaches_fit_warm_and_keeps_every_receipt_field(tmp_path, spy):
    paths = inputs(tmp_path)
    result = evaluator.evaluate_candidate(candidate(r3_model(tmp_path)), *paths, tmp_path / "out")
    assert spy == [1]
    t = result["training"]
    for key in ("observed_updates", "selected_epoch", "stop_reason", "early_stop", "optimizer_iterations"):
        assert key in t
    w = t["warm"]
    assert w["unfreeze_epoch"] == 2 and w["freeze_epochs"] == 1 and w["optimizer_reset_at_unfreeze"] is True
    assert w["phase_1"]["component_hashes_after"] == w["component_hashes_start"]
    assert t["observed_updates"] == w["phase_1"]["observed_updates"] + w["phase_2"]["observed_updates"] > 0
    assert t["selected_epoch"] == (w["phase_1"]["selected_epoch"] if w["selected_phase"] == 1
                                   else w["phase_1"]["epochs_completed"] + w["phase_2"]["selected_epoch"])
    assert result["reload_parity"]["passed"] is True and result["digests"]["weights_sha256"]


def test_r0_candidate_never_reaches_fit_warm_and_keeps_its_layout(tmp_path, spy):
    paths = inputs(tmp_path)
    result = evaluator.evaluate_candidate(candidate(model_config()), *paths, tmp_path / "out")
    assert spy == [] and "warm" not in result["training"]
    assert result["status"] == "completed"
