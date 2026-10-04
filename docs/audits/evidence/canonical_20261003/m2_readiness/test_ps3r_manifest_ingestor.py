"""Discovery adopts a terminal only when its manifest and results agree."""

import hashlib
import json
import shutil
from pathlib import Path

import ps3r_manifest_ingestor as ingest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CONFIG = json.loads((HERE / "ps3r_ingest_config.json").read_text(encoding="utf-8"))
BASE_NAME = "fred.stress.vixcls.logret_5d"
SOURCE = ROOT / "docs/audits/evidence/canonical_20261003/ps3r_ingest/baseline" / BASE_NAME
ALTERNATIVES = ("tv.hilbert_amp", "tv.kalman_dev", "tv.stl_dev")


def _place(tmp_path):
    repo = tmp_path / "repo"
    cell = repo / "cells" / BASE_NAME
    shutil.copytree(SOURCE, cell)
    return repo, cell


def _config(roots):
    rules = json.loads(json.dumps(CONFIG["baseline"]))
    rules["roots"] = list(roots)
    path = None
    return {"baseline": rules}, path


def _save(repo, payload):
    path = repo / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _only(repo, feature):
    payload, _ = _config(["cells"])
    found = ingest.discover(repo, _save(repo, payload))
    matches = [item for item in found if item["feature_id"] == feature]
    assert len(matches) == 1
    return matches[0]


def _manifest(cell):
    path = cell / "run_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_bad_hash_is_rejected(tmp_path):
    repo, cell = _place(tmp_path)
    path, manifest = _manifest(cell)
    manifest["results_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "RESULTS_HASH"


def test_wrong_feature_is_rejected(tmp_path):
    repo, cell = _place(tmp_path)
    path, manifest = _manifest(cell)
    manifest["features"] = ["other.feature"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "FEATURE"


def test_missing_family_is_rejected(tmp_path):
    repo, cell = _place(tmp_path)
    path, manifest = _manifest(cell)
    manifest["families"] = ["identity", "random", "ae"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "FAMILY"


def test_different_seed_is_rejected(tmp_path):
    repo, cell = _place(tmp_path)
    path, manifest = _manifest(cell)
    manifest["seed"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "SEED"


def test_partial_directory_is_not_terminal(tmp_path):
    repo, cell = _place(tmp_path)
    (cell / "results.jsonl").unlink()
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "NOT_TERMINAL"
    assert item["reason"] == "PARTIAL_DIRECTORY"
    assert item["ps3r_status"] == ""


def test_contradictory_terminals_are_not_adopted(tmp_path):
    repo, cell = _place(tmp_path)
    other = repo / "cells_b" / BASE_NAME
    shutil.copytree(cell, other)
    results = other / "results.jsonl"
    original = results.read_text(encoding="utf-8")
    mutated = original.replace("0.0004340630841538771", "0.0004340630841538772", 1)
    assert mutated != original
    results.write_text(mutated, encoding="utf-8")
    path, manifest = _manifest(other)
    manifest["results_sha256"] = hashlib.sha256(results.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    payload, _ = _config(["cells", "cells_b"])
    found = ingest.discover(repo, _save(repo, payload))
    same = [item for item in found if item["feature_id"] == BASE_NAME]
    assert [item for item in same if item["disposition"] == "ADOPTED"] == []
    assert [item["reason"] for item in same] == [
        "CONTRADICTORY_TERMINAL",
        "CONTRADICTORY_TERMINAL",
    ]


def test_three_alternative_artifacts_are_adopted_without_replacing_baseline():
    found = ingest.discover(ROOT, HERE / "ps3r_ingest_config.json")
    adopted = {
        item["feature_id"]: item
        for item in found
        if item["role"] == "alternative" and item["disposition"] == "ADOPTED"
    }
    baseline = {
        item["feature_id"]
        for item in found
        if item["role"] == "baseline" and item["disposition"] == "ADOPTED"
    }
    assert set(ALTERNATIVES).isdisjoint(baseline)
    for feature_id in ALTERNATIVES:
        item = adopted[feature_id]
        assert item["utility"] == "mixed"
        assert item["ps3r_status"] == ""
        assert item["locator"] == (
            "docs/audits/evidence/canonical_20261003/ps3r_ingest/alternative/"
            + feature_id
            + "/results.jsonl"
        )
        assert "/" + "home" + "/" not in item["locator"]
