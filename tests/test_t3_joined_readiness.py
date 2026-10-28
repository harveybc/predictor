import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import t3_joined_readiness as j


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "mirror"
    j.w.atomic(root / "T3_MANIFEST.json", {})
    support, columns = {"TRAIN": "T", "INNER_A": "V"}, ["x", "y"]
    coverage = {"done": 21, "failed": 0, "pending": 0, "total": 21, "units": {}}
    for i in range(1, 43, 2):
        name = f"t3_unit_{i:02d}"
        p = root / "ops" / (name + ".RESULT.json")
        j.w.atomic(p, {"job": {"name": name}, "config_sha256": "cfg", "support_hash": support,
                       "consumed_features": columns, "metrics": {str(s): {} for s in (20261019, 20261020, 20261021)},
                       "hashes": {"E3M.json": "h"}})
        coverage["units"][name] = {"receipt": str(p), "receipt_sha256": j.w.digest(p),
                                    "origin": "own", "config_sha256": "cfg"}
    peer = {"schema": "t3_peer_preflight.v1", "state": "ARTIFACTS_VERIFIED", "count": 22,
            "manifest_sha256": j.w.digest(root / "T3_MANIFEST.json"), "support_hash": support,
            "consumed_features": columns, "units": {f"t3_unit_{i:02d}": {
                "hashes": {"E3M.json": "h"}, "unit_identity": "u", "command_identity": "c"} for i in range(0, 43, 2)}}
    monkeypatch.setattr(j, "validate_mirror", lambda root: {})
    return coverage, peer


def test_all_43_paired_artifacts_are_readiness_not_release(tmp_path, monkeypatch):
    own, peer = fixture(tmp_path, monkeypatch)
    result = j.assemble(own, peer)
    assert result["count"] == 43
    assert not result["scientific_release"] and not result["decision_executed"]
    assert not result["checkpoint_forward_replayed"] and not result["combined_fit_started"]


@pytest.mark.parametrize("fault", ["unfinished", "missing", "duplicate", "support", "columns", "manifest", "receipt"])
def test_join_refuses_incomplete_or_unpaired_inputs(tmp_path, monkeypatch, fault):
    own, peer = fixture(tmp_path, monkeypatch)
    peer = copy.deepcopy(peer)
    if fault == "unfinished": own["done"] = 20
    if fault == "missing": peer["units"].pop("t3_unit_00")
    if fault == "duplicate": peer["units"]["t3_unit_01"] = peer["units"].pop("t3_unit_00")
    if fault == "support": peer["support_hash"]["TRAIN"] = "different"
    if fault == "columns": peer["consumed_features"].reverse()
    if fault == "manifest": peer["manifest_sha256"] = "changed"
    if fault == "receipt": own["units"]["t3_unit_01"]["receipt_sha256"] = "changed"
    with pytest.raises(ValueError):
        j.assemble(own, peer)


def test_waiting_for_training_does_not_execute_any_command(monkeypatch):
    monkeypatch.setattr(j.w, "read", lambda p: {"done": 18, "failed": 0})
    monkeypatch.setattr(j.subprocess, "run", lambda *a, **k: pytest.fail("started before completion"))
    assert j.tick({"coverage": "coverage"})["state"] == "WAITING_FOR_OWN_SHARD"
