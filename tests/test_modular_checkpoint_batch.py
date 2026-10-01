"""Historical R3 scoring recovery: no training, no changed identity or pin."""
import hashlib
import json
import sys

import numpy as np
import pytest

from tools import modular_checkpoint_scorer as scorer


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@pytest.fixture
def historical(tmp_path):
    tf = pytest.importorskip("tensorflow")
    tf.keras.backend.clear_session()
    x = np.arange(6, dtype="float32").reshape(3, 2, 1)
    y = x[:, -1:] + 1
    inputs = tf.keras.Input((2, 1))
    model = tf.keras.Model(inputs, tf.keras.layers.Cropping1D((1, 0))(inputs))
    model_path = tmp_path / "best.keras"
    model.save(model_path)
    validation = tmp_path / "validation.npz"
    fields = {name: np.array(0) for name in scorer.VALIDATION_FIELDS}
    fields.update(windows=x, targets=y, split=np.array("validation"),
                  feature_names=np.array(["x"]), target_names=np.array(["x"]),
                  horizons=np.array([1]))
    np.savez(validation, **fields)
    config = {"model": {"name": "fixture"}, "evaluator": {"batch_size": 2, "seed": 2021}}
    metrics = scorer.score_metrics(y, x[:, -1:], x[:, -1:])
    receipt = {"schema_version": "modular.candidate.evaluation.v1", "status": "completed",
               "training": {"observed_updates": 0}, "candidate": {"cid": digest(config)},
               "bridge": {"predictor_revision": "historical-pin"},
               "digests": {"config_sha256": digest(config),
                           "model_sha256": scorer._sha_file(model_path),
                           "validation_sha256": scorer._sha_file(validation)},
               "artifacts": {"best_model": str(model_path)}, "metrics": metrics,
               "per_horizon": {"1": metrics},
               "objective": {"metric": "MAE", "split": "validation",
                             "unit": "z_train", "higher_is_better": False, "value": metrics["MAE"]}}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(receipt))
    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(config))
    # Any accidental fit path, even on this tiny fixture, fails the test.
    return path, validation, config_path, receipt, config


def test_historical_scoring_authenticated_config_no_fit(historical, tmp_path, monkeypatch):
    import tensorflow as tf
    monkeypatch.setattr(tf.keras.Model, "fit", lambda *a, **k: pytest.fail("fit forbidden"))
    path, validation, config_path, receipt, config = historical
    before = path.read_bytes()
    result = scorer.verify(path, validation, tmp_path / "recovery.json", candidate_config=config)
    assert result["exact_match"] and result["verdict"] == "VERIFIED"
    assert result["batch_size"] == 2
    assert result["batch_size_binding"]["config_sha256"] == digest(config)
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["bridge"] == receipt["bridge"]


def test_cli_authenticated_batch(historical, tmp_path, monkeypatch):
    path, validation, config_path, _, _ = historical
    output = tmp_path / "cli-recovery.json"
    monkeypatch.setattr(sys, "argv", ["scorer", "--receipt", str(path), "--validation", str(validation),
                                     "--output", str(output), "--candidate-config", str(config_path),
                                     "--batch-size", "2"])
    with pytest.raises(SystemExit) as exc:
        scorer.main()
    assert exc.value.code == 0
    assert json.loads(output.read_text())["exact_match"]


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, "2", 3])
def test_invalid_or_conflicting_cli_batch_fails_before_loading(historical, tmp_path, monkeypatch, batch):
    path, validation, _, _, config = historical
    monkeypatch.setattr(scorer, "_load_validation", lambda *a: pytest.fail("must reject before data loading"))
    with pytest.raises(ValueError, match="batch"):
        scorer.verify(path, validation, tmp_path / "refused.json", batch_size=batch, candidate_config=config)
    assert not (tmp_path / "refused.json").exists()


def test_tampered_config_rejected_before_loading(historical, tmp_path, monkeypatch):
    path, validation, _, _, config = historical
    config["evaluator"]["seed"] = 2022
    monkeypatch.setattr(scorer, "_load_validation", lambda *a: pytest.fail("must authenticate first"))
    with pytest.raises(ValueError, match="config.*sha256"):
        scorer.verify(path, validation, tmp_path / "bad.json", candidate_config=config)


def test_unauthenticated_cli_cannot_supply_missing_batch(historical, tmp_path):
    path, validation, _, _, _ = historical
    with pytest.raises(ValueError, match="authenticated"):
        scorer.verify(path, validation, tmp_path / "bad.json", batch_size=2)


def test_settings_conflict_and_existing_output_refused(historical, tmp_path):
    path, validation, _, receipt, config = historical
    receipt["training"]["settings"] = {"batch_size": 3}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="batch"):
        scorer.verify(path, validation, tmp_path / "bad.json", candidate_config=config)
    receipt["training"]["settings"]["batch_size"] = 2
    path.write_text(json.dumps(receipt))
    output = tmp_path / "existing.json"
    output.write_text("historical evidence")
    with pytest.raises(FileExistsError):
        scorer.verify(path, validation, output)
    assert output.read_text() == "historical evidence"


def test_modern_receipt_and_authenticated_embedded_config(historical, tmp_path):
    path, validation, _, receipt, config = historical
    receipt["training"]["settings"] = {"batch_size": 2}
    path.write_text(json.dumps(receipt))
    assert scorer.verify(path, validation, tmp_path / "modern.json")["exact_match"]
    receipt["training"].pop("settings")
    receipt["candidate"] = config
    path.write_text(json.dumps(receipt))
    assert scorer.verify(path, validation, tmp_path / "embedded.json")["exact_match"]


@pytest.mark.parametrize("field", ["model_sha256", "validation_sha256"])
def test_batch_recovery_does_not_relax_artifact_identity(historical, tmp_path, field):
    path, validation, _, receipt, config = historical
    receipt["digests"][field] = "0" * 64
    path.write_text(json.dumps(receipt))
    result = scorer.verify(path, validation, tmp_path / "refuted.json", candidate_config=config)
    assert result["verdict"] == "REFUTED"
    assert field + " mismatch" in result["problems"]


def test_incomplete_embedded_config_is_not_an_authenticated_fallback(historical, tmp_path):
    path, validation, _, receipt, _ = historical
    receipt["candidate"] = {"evaluator": {"batch_size": 2}}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="config_sha256"):
        scorer.verify(path, validation, tmp_path / "bad.json")
