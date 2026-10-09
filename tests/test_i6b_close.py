import hashlib
import json
from pathlib import Path

import pytest

from tools import i6b_close as subject


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_branch(root, name, feature, config_sha="c" * 64, train_sha="t" * 64):
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"{name}.keras"
    model.write_bytes(name.encode())
    decoder = root / f"{name}.decoder.keras"
    decoder.write_bytes((name + "-decoder").encode())
    manifest = root / f"{name}.manifest.json"
    manifest_identity = {"name": name, "features": [feature], "role": "branch"}
    manifest_doc = {
        "schema": 2,
        "model_sha256": sha(model),
        "manifest": manifest_identity,
        "manifest_sha256": subject.config_sha256(manifest_identity),
        "provenance": {"conditioning_contract": "OPERATIONAL", "learned_corpus": {
            "kind": "TRAIN_ONLY", "data_sha256": train_sha}},
    }
    manifest.write_text(json.dumps(manifest_doc))
    (root / f"{name}.provenance.json").write_text(json.dumps({
        "name": name,
        "source_config_sha256": config_sha,
        "train_input_sha256": train_sha,
        "input_provenance": "local_file",
    }))
    (root / f"{name}.record.json").write_text(json.dumps({
        "stage": "branch_ae",
        "name": name,
        "features": [feature],
        "donor_sha256": sha(model),
        "decoder_sha256": sha(decoder),
        "donor_manifest_sha256": subject.config_sha256(manifest_identity),
        "source_config_sha256": config_sha,
        "train_input_sha256": train_sha,
        "reload_parity": {"passed": True, "max_abs_error": 0.0},
        "training": {"settings": {"seed": 42}, "best_validation_loss": 0.1},
    }))


def config(path):
    document = {"branches": [
        {"name": "b00_a", "features": ["a"]},
        {"name": "b01_b", "features": ["b"]},
    ]}
    path.write_text(json.dumps(document))
    return document


def test_close_accepts_complete_disjoint_population(tmp_path):
    cfg = tmp_path / "config.json"
    document = config(cfg)
    one, two = tmp_path / "one", tmp_path / "two"
    one_config = {**document, "branches": [document["branches"][0]]}
    two_config = {**document, "branches": [document["branches"][1]]}
    write_branch(one, "b00_a", "a", subject.config_sha256(one_config))
    write_branch(two, "b01_b", "b", subject.config_sha256(two_config))

    result = subject.close(cfg, [one, two], tmp_path / "closed")

    assert result["status"] == "COMPLETE"
    assert result["completed_branches"] == 2
    assert result["validation_read"] is False
    assert result["test_read"] is False
    assert (tmp_path / "closed" / "BRANCH_PRETRAIN.json").exists()


def test_close_rejects_missing_duplicate_and_tampered_donor(tmp_path):
    cfg = tmp_path / "config.json"
    document = config(cfg)
    one_config = {**document, "branches": [document["branches"][0]]}
    two_config = {**document, "branches": [document["branches"][1]]}
    one, two = tmp_path / "one", tmp_path / "two"
    write_branch(one, "b00_a", "a", subject.config_sha256(one_config))
    with pytest.raises(ValueError, match="INCOMPLETE_BRANCH_POPULATION"):
        subject.close(cfg, [one], tmp_path / "closed")

    write_branch(two, "b00_a", "a", subject.config_sha256(one_config))
    with pytest.raises(ValueError, match="DUPLICATE_BRANCH"):
        subject.close(cfg, [one, two], tmp_path / "closed")

    for path in two.glob("*"):
        path.unlink()
    write_branch(two, "b01_b", "b", subject.config_sha256(two_config))
    (two / "b01_b.keras").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="DONOR_DIGEST_MISMATCH"):
        subject.close(cfg, [one, two], tmp_path / "closed")
