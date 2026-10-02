import copy
import json

import pytest

from tools import materialize_donor_ablation as subject


def _source(tmp_path, monkeypatch):
    branches = []
    donors = {}
    for name in ("a", "b", "core"):
        path = tmp_path / f"{name}.keras"
        path.write_bytes(name.encode())
        manifest = path.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({"schema": 2, "weights_sha256": name, "provenance": {
            "conditioning_contract": "OPERATIONAL"}}))
        donors[str(path)] = {"keras": subject._sha256(path), "manifest": subject._sha256(manifest),
                             "weights": name}
        if name != "core":
            branches.append({"name": name, "regime": "R2", "donor": str(path), "freeze_epochs": 2})
    model = {"branches": branches, "core": {"name": "core", "regime": "R2",
             "donor": str(tmp_path / "core.keras"), "freeze_epochs": 2}}
    model_path = tmp_path / "MODEL_R2.json"
    model_path.write_text(json.dumps(model))
    pretrain = tmp_path / "PRETRAIN.json"
    pretrain.write_text(json.dumps({"status": "COMPLETE", "objective": {"sha256": "objective"}}))
    candidate = {"model": model, "modular_candidate": {"donor_binding": {
        "schema": "predictor.modular.donor_binding.v2", "required_contract": "OPERATIONAL",
        "pretrain": {"path": str(pretrain), "sha256": subject._sha256(pretrain)},
        "regime_config": {"path": str(model_path), "sha256": subject._sha256(model_path)},
        "donor_objective_sha256": "objective", "donors": donors}}}
    monkeypatch.setattr("predictor_plugins.modular_temporal.provenance.donor_provenance",
                        lambda path: {"conditioning_contract": "OPERATIONAL"})
    source = tmp_path / "candidate.json"
    source.write_text(json.dumps(candidate))
    return source


@pytest.mark.parametrize(("mode", "retained"),
                         [("branches_only", 2), ("core_only", 1), ("random_all", 0)])
def test_materializes_exact_partial_binding(tmp_path, monkeypatch, mode, retained):
    from tools import modular_heartbeat

    source = _source(tmp_path, monkeypatch)
    receipt = subject.materialize(source, mode, tmp_path / mode)
    candidate = json.loads(open(receipt["candidate"]["path"]).read())
    assert receipt["retained_donors"] == retained
    assert modular_heartbeat.verify_donor_binding(candidate)["donors_checked"] == retained
    randomized = ([candidate["model"]["core"]] if mode == "branches_only" else
                  candidate["model"]["branches"] if mode == "core_only" else
                  [*candidate["model"]["branches"], candidate["model"]["core"]])
    assert all(component["regime"] == "R0" and component["donor"] is None for component in randomized)


def test_rejects_existing_destination(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        subject.materialize(source, "branches_only", destination)
