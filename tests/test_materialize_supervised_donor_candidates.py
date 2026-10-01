"""Consumable candidate bridge for supervised modular donors."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import materialize_supervised_donor_candidates as subject


SETTINGS = {
    "loss": "mae",
    "seed": 2021,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 3,
    "patience": 1,
    "min_delta": 0.0,
    "max_updates": 100,
    "max_seconds": 30.0,
}
OBJECTIVE = {"metric": "MAE", "split": "validation", "higher_is_better": False, "unit": "z_train"}


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")


def _fixture(root: Path, monkeypatch, marker="a"):
    root.mkdir()
    donors = root / "donors"
    donors.mkdir()
    records = []
    paths = {}
    for name, role in (("branch_0", "branch"), ("branch_1", "branch"), ("core", "core")):
        model = donors / f"{name}.keras"
        model.write_bytes(f"keras-{marker}-{name}".encode())
        sidecar = model.with_suffix(".manifest.json")
        manifest = {"schema": 1, "role": role, "name": name}
        weights = subject._sha256_bytes(f"weights-{marker}-{name}".encode())
        document = {
            "schema": 2,
            "manifest": manifest,
            "manifest_sha256": subject._canonical_sha256(manifest),
            "model_sha256": subject._sha256_file(model),
            "weights_sha256": weights,
            "provenance": {"conditioning_contract": "OPERATIONAL"},
        }
        _write(sidecar, document)
        records.append({"role": role, "name": name, "path": str(model),
                        "model_sha256": document["model_sha256"], "weights_sha256": weights})
        paths[name] = str(model)

    base = {
        "schema": "predictor.modular.v1",
        "window": 4,
        "sample_hours": 1,
        "feature_names": ["a", "b"],
        "horizons": [1],
        "target_count": 2,
        "branch_steps": 4,
        "output_steps": 2,
        "output_channels": 2,
        "branches": [
            {"name": "branch_0", "features": ["a"], "plugin": "causal_conv1d", "params": {},
             "regime": "R0", "donor": None},
            {"name": "branch_1", "features": ["b"], "plugin": "causal_conv1d", "params": {},
             "regime": "R0", "donor": None},
        ],
        "fusion": {"plugin": "sequence_concat", "params": {}},
        "core": {"plugin": "transformer_conv", "params": {}, "regime": "R0", "donor": None},
        "head": {"plugin": "forecast", "params": {}},
    }
    configs = {}
    for regime in ("R1", "R2", "R3"):
        config = copy.deepcopy(base)
        for branch in config["branches"]:
            branch.update(regime=regime, donor=paths[branch["name"]])
        config["core"].update(regime=regime, donor=paths["core"])
        if regime == "R3":
            for component in [*config["branches"], config["core"]]:
                component.update(freeze_epochs=1, unfreeze_learning_rate=1e-4)
        path = root / f"MODEL_{regime}.json"
        _write(path, config)
        configs[regime] = str(path)

    objective = {"group": "modular.supervised_donor", "name": "seasonal_residual_forecast_P24",
                 "version": "1.0.0", "params": {"loss": "mae"},
                 "train_sha256": marker * 64, "recipe_sha256": marker * 64,
                 "support": {"fit": [0, 7], "validation": [9, 9], "purged_origins": 1}}
    objective["sha256"] = subject._canonical_sha256(objective)
    receipt = {"schema": "modular.supervised_donor.pretrain.v1", "status": "COMPLETE",
               "objective": objective, "donors": records, "regime_configs": configs}
    _write(root / "PRETRAIN.json", receipt)

    def verify(path, expected_manifest, require_contract=None):
        sidecar = json.loads(Path(path).with_suffix(".manifest.json").read_text())
        assert expected_manifest == sidecar["manifest"]
        assert require_contract == "OPERATIONAL"
        return sidecar["weights_sha256"]

    monkeypatch.setattr(subject, "_load_and_hash_weights", verify)
    return root / "PRETRAIN.json"


def test_materializes_complete_r1_r2_r3_candidates_and_direct_bindings(tmp_path, monkeypatch):
    pretrain = _fixture(tmp_path / "one", monkeypatch)

    candidates = subject.materialize(pretrain, SETTINGS, OBJECTIVE)

    assert set(candidates) == {"R1", "R2", "R3"}
    for regime, candidate in candidates.items():
        assert set(candidate) == {"modular_candidate", "model", "evaluator",
                                  "target_feature_indices", "objective"}
        assert all(candidate["evaluator"][key] == value for key, value in SETTINGS.items())
        assert candidate["evaluator"]["monitor"] == "validation_loss"
        assert candidate["target_feature_indices"] == [0, 1]
        assert candidate["objective"] == OBJECTIVE
        binding = candidate["modular_candidate"]["donor_binding"]
        assert binding["schema"] == subject.BINDING_SCHEMA
        assert binding["pretrain"]["sha256"] == subject._sha256_file(pretrain)
        assert binding["regime_config"]["sha256"] == subject._sha256_file(pretrain.parent / f"MODEL_{regime}.json")
        assert len(binding["donors"]) == 3
        assert all(set(item) == {"keras", "manifest", "weights"}
                   for item in binding["donors"].values())


def test_rejects_incomplete_receipt(tmp_path, monkeypatch):
    pretrain = _fixture(tmp_path / "one", monkeypatch)
    document = json.loads(pretrain.read_text())
    document["status"] = "RUNNING"
    _write(pretrain, document)
    with pytest.raises(ValueError, match="PRETRAIN_NOT_COMPLETE"):
        subject.materialize(pretrain, SETTINGS, OBJECTIVE)


def test_rejects_altered_donor_bytes(tmp_path, monkeypatch):
    pretrain = _fixture(tmp_path / "one", monkeypatch)
    document = json.loads(pretrain.read_text())
    Path(document["donors"][0]["path"]).write_bytes(b"altered")
    with pytest.raises(ValueError, match="DONOR_MODEL_SHA256_MISMATCH"):
        subject.materialize(pretrain, SETTINGS, OBJECTIVE)


@pytest.mark.parametrize("mode", ["missing", "unknown"])
def test_rejects_missing_or_unknown_manifest(tmp_path, monkeypatch, mode):
    pretrain = _fixture(tmp_path / "one", monkeypatch)
    document = json.loads(pretrain.read_text())
    sidecar = Path(document["donors"][0]["path"]).with_suffix(".manifest.json")
    if mode == "missing":
        sidecar.unlink()
    else:
        manifest = json.loads(sidecar.read_text())
        manifest["provenance"]["conditioning_contract"] = "UNKNOWN"
        _write(sidecar, manifest)
    with pytest.raises(ValueError, match="DONOR_(MANIFEST_MISSING|CONTRACT_NOT_OPERATIONAL)"):
        subject.materialize(pretrain, SETTINGS, OBJECTIVE)


def test_rejects_model_config_from_another_pretrain(tmp_path, monkeypatch):
    first = _fixture(tmp_path / "one", monkeypatch, marker="a")
    second = _fixture(tmp_path / "two", monkeypatch, marker="b")
    first_document = json.loads(first.read_text())
    first_document["regime_configs"]["R1"] = str(second.parent / "MODEL_R1.json")
    _write(first, first_document)
    with pytest.raises(ValueError, match="MODEL_R1_PATH_INVALID"):
        subject.materialize(first, SETTINGS, OBJECTIVE)


@pytest.mark.parametrize("key,value", [("learning_rate", float("nan")), ("max_seconds", float("inf")),
                                        ("batch_size", True), ("weight_decay", False)])
def test_rejects_nonfinite_and_boolean_settings(tmp_path, monkeypatch, key, value):
    pretrain = _fixture(tmp_path / "one", monkeypatch)
    with pytest.raises(ValueError, match=key):
        subject.materialize(pretrain, {**SETTINGS, key: value}, OBJECTIVE)


def test_heartbeat_accepts_direct_schema2_binding_without_alongside_manifest(tmp_path, monkeypatch):
    from predictor_plugins.modular_temporal import provenance
    from tools import modular_heartbeat

    pretrain = _fixture(tmp_path / "one", monkeypatch)
    candidate = subject.materialize(pretrain, SETTINGS, OBJECTIVE)["R1"]
    monkeypatch.setattr(provenance, "donor_provenance",
                        lambda path: {"conditioning_contract": "OPERATIONAL"})

    result = modular_heartbeat.verify_donor_binding(candidate)

    assert result["result"] == "VERIFIED_BEFORE_BUILD"
    assert result["binding_schema"] == subject.BINDING_SCHEMA
    assert result["donors_checked"] == 3


def test_heartbeat_rejects_changed_regime_config_bound_by_direct_binding(tmp_path, monkeypatch):
    from predictor_plugins.modular_temporal import provenance
    from tools import modular_heartbeat

    pretrain = _fixture(tmp_path / "one", monkeypatch)
    candidate = subject.materialize(pretrain, SETTINGS, OBJECTIVE)["R2"]
    Path(candidate["modular_candidate"]["donor_binding"]["regime_config"]["path"]).write_text("{}\n")
    monkeypatch.setattr(provenance, "donor_provenance",
                        lambda path: {"conditioning_contract": "OPERATIONAL"})

    with pytest.raises(ValueError, match="DONOR_BINDING_MISMATCH: regime_config"):
        modular_heartbeat.verify_donor_binding(candidate)


def test_heartbeat_rejects_inline_model_changed_while_bound_file_is_intact(tmp_path, monkeypatch):
    from tools import modular_heartbeat

    pretrain = _fixture(tmp_path / "one", monkeypatch)
    candidate = subject.materialize(pretrain, SETTINGS, OBJECTIVE)["R1"]
    candidate["model"]["output_channels"] += 1

    with pytest.raises(ValueError, match="DONOR_BINDING_MODEL_MISMATCH"):
        modular_heartbeat.verify_donor_binding(candidate)


@pytest.mark.parametrize("mutation", ["extra", "omitted", "substituted"])
def test_heartbeat_rejects_nonexact_bound_donor_set(tmp_path, monkeypatch, mutation):
    from tools import modular_heartbeat

    pretrain = _fixture(tmp_path / "one", monkeypatch)
    candidate = subject.materialize(pretrain, SETTINGS, OBJECTIVE)["R2"]
    donors = candidate["modular_candidate"]["donor_binding"]["donors"]
    first = next(iter(donors))
    if mutation == "extra":
        donors[str(tmp_path / "extra.keras")] = copy.deepcopy(donors[first])
    elif mutation == "omitted":
        donors.pop(first)
    else:
        record = donors.pop(first)
        donors[str(tmp_path / "replacement.keras")] = record

    with pytest.raises(ValueError, match="DONOR_BINDING_DONOR_SET_MISMATCH"):
        modular_heartbeat.verify_donor_binding(candidate)
