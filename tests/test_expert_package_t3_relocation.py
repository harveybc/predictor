import importlib.util
from pathlib import Path
import sys

import pytest


def module():
    root = Path(__file__).parents[1] / "tools"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("prepare_t3_relocation", root / "prepare_expert_package_t3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(m, root):
    p = root / "data/input"
    p.parent.mkdir()
    p.write_bytes(b"same science")
    cfg = {"campaign": "v4_t3", "root": "/old", "runtime_versions": {"torch": "same"},
           "pins": {"/old/data/input": m.digest(p)}}
    stop = {"state": "STOPPED_FOR_OWNER_REQUESTED_RELOCATION", "parent_config_sha256": m.identity(cfg),
            "remaining_scientific_children": [], "prior_status": {"done": 0, "failed": 5}}
    m.atomic(root / "source/ops/CONFIG.private.json", cfg)
    m.atomic(root / "source/ops/RELOCATION_STOP.private.json", stop)
    m.atomic(root / "source/ops/STATUS.json", {"state": "STOPPED_FOR_RELOCATION"})
    return cfg, stop


def test_t3_relocation_authenticates_stopped_source_and_preserves_science(tmp_path):
    m = module()
    fixture(m, tmp_path)
    p = m.validate_relocation(tmp_path, {"torch": "same"})
    assert p["verified_input_files"] == 1
    assert p["source_done"] == 0 and p["source_failed"] == 5
    assert p["science_flags_unchanged"] is True
    assert p["cross_hardware_bitwise_equivalence_claimed"] is False
    (tmp_path / "data/input").write_bytes(b"changed")
    with pytest.raises(ValueError, match="INPUT"):
        m.validate_relocation(tmp_path, {"torch": "same"})


@pytest.mark.parametrize("fault", ["live", "completed", "runtime", "identity"])
def test_t3_relocation_rejects_duplicate_or_unmatched_provenance(tmp_path, fault):
    m = module()
    cfg, stop = fixture(m, tmp_path)
    versions = {"torch": "same"}
    if fault == "live":
        stop["remaining_scientific_children"] = [123]
    elif fault == "completed":
        stop["prior_status"]["done"] = 1
    elif fault == "runtime":
        versions = {"torch": "other"}
    else:
        stop["parent_config_sha256"] = "changed"
    m.atomic(tmp_path / "source/ops/RELOCATION_STOP.private.json", stop)
    with pytest.raises(ValueError):
        m.validate_relocation(tmp_path, versions)


def test_shard_relocation_excludes_every_completed_receipted_unit(tmp_path):
    m = module()
    cfg, stop = fixture(m, tmp_path)
    stop["prior_status"]["done"] = 1
    m.atomic(tmp_path / "source/ops/RELOCATION_STOP.private.json", stop)
    m.atomic(tmp_path / "source/ops/first.RESULT.json", {
        "config_sha256": m.identity(cfg), "job": {"name": "t3_unit_01"}})
    assert m.validate_relocation(tmp_path, {"torch": "same"}, ["t3_unit_03"])["source_done"] == 1
    with pytest.raises(ValueError, match="REPEAT"):
        m.validate_relocation(tmp_path, {"torch": "same"}, ["t3_unit_01"])
