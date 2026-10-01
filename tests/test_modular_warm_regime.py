"""R3 warm regime (owner standing order 2026-10-01 item 2; M04 request).

R3 loads the donor like R2 (through the same donor-contract check), keeps the donor-loaded components
FROZEN for ``freeze_epochs`` epochs, then unfreezes them and continues with a NEW AdamW at
``unfreeze_learning_rate`` (optimizer state is reset at the boundary, by design and recorded). The best
checkpoint is selected across the boundary. Synthetic component checks.
"""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import tensorflow as tf

from predictor_plugins import modular_temporal as mt
from predictor_plugins.modular_temporal import warm

OPERATIONAL = {"conditioning_contract": "OPERATIONAL",
               "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": "fixture:train", "data_sha256": "a" * 64,
                                  "support": "fixture rows", "pretrained_weights_source": None},
               "reconstruction": {"state": "NOT_EVALUATED"}}
FIT = {"max_epochs": 5, "patience": 10, "batch_size": 8, "learning_rate": 1e-2, "seed": 3}


@pytest.fixture(autouse=True)
def seeded():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(9)
    yield


def data(n=32, seed=1):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(n, 24, 2)).astype("float32")
    return xs, (0.5 * xs[:, -1:, :1]).astype("float32")


def r3_config(tmp_path, freeze=2, lr=1e-3):
    c = mt.default_config(["a", "b"])
    base = mt.build_modular(c)
    for spec in c["branches"]:
        path = tmp_path / f"{spec['name']}.keras"
        mt.save_donor(base.branch_models[spec["name"]], path, base.donor_manifest("branch", spec["name"]),
                      provenance=OPERATIONAL)
        spec.update(regime="R3", donor=str(path), freeze_epochs=freeze, unfreeze_learning_rate=lr)
    core = tmp_path / "core.keras"
    mt.save_donor(base.core_model, core, base.donor_manifest("core"), provenance=OPERATIONAL)
    c["core"].update(regime="R3", donor=str(core), freeze_epochs=freeze, unfreeze_learning_rate=lr)
    donor_hashes = {n: mt.weights_hash(m) for n, m in base.branch_models.items()}
    donor_hashes["core"] = mt.weights_hash(base.core_model)
    return c, donor_hashes


def test_frozen_during_warm_epochs_moves_after_and_records_the_boundary(tmp_path):
    c, donor = r3_config(tmp_path)
    b = mt.build_modular(c)
    assert not b.core_model.trainable and not any(m.trainable for m in b.branch_models.values())
    x, y = data()
    vx, vy = data(16, 2)
    receipt = warm.fit_warm(b, x, y, vx, vy, FIT)
    assert receipt["unfreeze_epoch"] == 3 and receipt["freeze_epochs"] == 2
    assert receipt["phase_1"]["component_hashes_after"] == donor           # byte-stable while frozen
    after = receipt["phase_2"]["component_hashes_after"]
    assert all(after[k] != donor[k] for k in donor)                        # moved after unfreezing
    assert receipt["phase_1"]["epochs_completed"] == 2
    assert receipt["observed_updates"] == receipt["phase_1"]["observed_updates"] + \
        receipt["phase_2"]["observed_updates"] > 0
    assert receipt["phase_1"]["observed_updates"] == 2 * 4                 # 32 rows / batch 8 x 2 epochs
    assert receipt["optimizer_reset_at_unfreeze"] is True
    assert receipt["phase_2"]["learning_rate"] == 1e-3
    assert receipt["selected_phase"] in (1, 2)
    if receipt["selected_phase"] == 1:
        assert {n: mt.weights_hash(m) for n, m in b.branch_models.items()} == \
            {k: v for k, v in donor.items() if k != "core"}


def test_best_checkpoint_is_selected_across_the_boundary(tmp_path):
    c, donor = r3_config(tmp_path, lr=10.0)                                 # phase 2 diverges on purpose
    b = mt.build_modular(c)
    x, y = data()
    vx, vy = data(16, 2)
    receipt = warm.fit_warm(b, x, y, vx, vy, dict(FIT, max_epochs=4))
    best = min(receipt["phase_1"]["best_validation_loss"], receipt["phase_2"]["best_validation_loss"])
    assert receipt["best_validation_loss"] == best
    assert receipt["selected_phase"] == (1 if receipt["phase_1"]["best_validation_loss"] <= best else 2)


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.update(freeze_epochs=0), "freeze_epochs"),
    (lambda s: s.update(unfreeze_learning_rate=-1.0), "unfreeze_learning_rate"),
    (lambda s: s.pop("freeze_epochs"), "R3"),
    (lambda s: s.update(regime="R2"), "R3"),                                 # params only with R3
    (lambda s: s.update(donor=None), "donor"),
])
def test_r3_declaration_is_validated(tmp_path, mutate, message):
    c, _ = r3_config(tmp_path)
    mutate(c["core"])
    with pytest.raises(ValueError, match=message):
        mt.build_modular(c)


def test_r3_goes_through_the_donor_contract_and_reports_its_regime(tmp_path):
    c, _ = r3_config(tmp_path)
    assert mt.regime_summary(c)["common"] == "R3"
    import json
    side = tmp_path / "core.manifest.json"
    doc = json.loads(side.read_text())
    doc["provenance"]["conditioning_contract"] = "UNKNOWN"
    side.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CONDITIONING_CONTRACT_NOT_OPERATIONAL"):
        mt.build_modular(c)


def test_fit_warm_refuses_a_bundle_without_r3_components():
    b = mt.build_modular(mt.default_config(["a", "b"]))
    x, y = data()
    with pytest.raises(ValueError, match="R3"):
        warm.fit_warm(b, x, y, x, y, FIT)
