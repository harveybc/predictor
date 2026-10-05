"""Discovery adopts a terminal only when its manifest and results agree."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import ps3r_manifest_ingestor as ingest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CONFIG = json.loads((HERE / "ps3r_ingest_config.json").read_text(encoding="utf-8"))
BASE_NAME = "fred.stress.vixcls.logret_5d"
SOURCE = ROOT / "docs/audits/evidence/canonical_20261003/ps3r_ingest/baseline" / BASE_NAME
ALTERNATIVES = ("tv.hilbert_amp", "tv.kalman_dev", "tv.stl_dev")
ALTERNATIVE_DIGESTS = {
    "tv.hilbert_amp": "2cb5445214590c7aa68bb654c1f44828d2e8371c6bf68db6dd2bcc2259309eab",
    "tv.kalman_dev": "3746ee3effd0ff7c5b9737b5a7da77b0ada34843cd97a36a3fbb14a4f4436cf1",
    "tv.stl_dev": "8c182faa5a4c862f49b691545739773a48b37a2b7e21d398c62f1aa6a86ef0af",
}


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


def _mutate_result_and_reseal(cell, mutate):
    results = cell / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    path, manifest = _manifest(cell)
    manifest["results_sha256"] = hashlib.sha256(results.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


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


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("feature_id", "other.feature", "ROW_FEATURE"),
        ("family", "other_family", "ROW_FAMILY"),
        ("seed", 7, "ROW_SEED"),
        ("fold_id", "other_fold", "ROW_FOLD"),
    ),
)
def test_resealed_result_row_identity_mutations_are_rejected(tmp_path, field, value, reason):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(item for item in rows if (item.get("row_kind") or item.get("kind")) == "fold_family")
        row[field] = value

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == reason


@pytest.mark.parametrize("kind", ("fold_family", "probe", "probe_delta", "feature_summary"))
def test_every_result_kind_is_bound_to_manifest_feature(tmp_path, kind):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(item for item in rows if (item.get("row_kind") or item.get("kind")) == kind)
        row["feature_id"] = "other.feature"

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "ROW_FEATURE"


@pytest.mark.parametrize(
    ("kind", "field"),
    (("fold_family", "family"), ("probe", "representation"), ("probe_delta", "trained")),
)
def test_every_family_bearing_result_kind_is_bound_to_allowed_families(tmp_path, kind, field):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(item for item in rows if (item.get("row_kind") or item.get("kind")) == kind)
        row[field] = "other_family"

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "ROW_FAMILY"


@pytest.mark.parametrize("kind", ("fold_family", "probe", "probe_delta"))
def test_every_fold_bearing_result_kind_is_bound_to_expected_folds(tmp_path, kind):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(item for item in rows if (item.get("row_kind") or item.get("kind")) == kind)
        row["fold_id"] = "other_fold"

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "ROW_FOLD"


@pytest.mark.parametrize(
    ("kind", "field", "value", "reason"),
    (
        ("probe", "target", "FORGED_TARGET", "ROW_PROBE_COVERAGE"),
        ("probe", "horizon_index", 999, "ROW_PROBE_COVERAGE"),
        ("probe_delta", "target", "FORGED_TARGET", "ROW_DELTA_COVERAGE"),
        ("probe_delta", "horizon_index", 999, "ROW_DELTA_COVERAGE"),
        ("probe_delta", "loss", "forged_loss", "ROW_DELTA_COVERAGE"),
    ),
)
def test_resealed_probe_domain_mutations_are_rejected(
    tmp_path, kind, field, value, reason
):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(item for item in rows if (item.get("row_kind") or item.get("kind")) == kind)
        row[field] = value

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == reason


@pytest.mark.parametrize(
    ("target", "missing_metric"),
    (("Y_s", "mae"), ("Y_b", "log_loss")),
)
def test_resealed_probe_missing_required_metric_is_rejected(
    tmp_path, target, missing_metric
):
    repo, cell = _place(tmp_path)

    def mutate(rows):
        row = next(
            item
            for item in rows
            if (item.get("row_kind") or item.get("kind")) == "probe"
            and item.get("target") == target
        )
        del row[missing_metric]

    _mutate_result_and_reseal(cell, mutate)
    item = _only(repo, BASE_NAME)
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "ROW_METRIC"


def test_probe_contract_is_required_and_not_inferred_from_rows(tmp_path):
    repo, _ = _place(tmp_path)
    payload, _ = _config(["cells"])
    del payload["baseline"]["probe_contract"]
    item = ingest.discover(repo, _save(repo, payload))[0]
    assert item["disposition"] == "REJECTED"
    assert item["reason"] == "PROBE_CONTRACT"


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
        assert item["results_sha256"] == ALTERNATIVE_DIGESTS[feature_id]
        assert item["families"] == ["identity", "random", "past_to_current_siamese"]
        assert item["utility"] == "mixed"
        assert item["ps3r_status"] == ""
        assert item["locator"] == (
            "docs/audits/evidence/canonical_20261003/ps3r_ingest/alternative/"
            + feature_id
            + "/results.jsonl"
        )
        assert "/" + "home" + "/" not in item["locator"]
