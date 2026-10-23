import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import t3_peer_preflight as audit


def test_preflight_pairs_columns_and_support_and_never_attests_historical_execution(tmp_path, monkeypatch):
    audit.w.atomic(tmp_path / "reference.json", {"seeds": [1, 2, 3]})
    monkeypatch.setattr(audit, "verify_inputs", lambda *a: 162)
    monkeypatch.setattr(audit, "jobs", lambda *a, **kw: [
        {"name": n, "peer_unit_identity": n, "peer_command_identity": "command"} for n in ("control", "contrast")])
    rows = {"control": {"support_hash": {"TRAIN": "same"}, "consumed_features": ["x", "y"],
                        "hashes": {"E3M.json": "hash"}, "wall_seconds": 10, "metrics": {}}}
    rows["contrast"] = dict(rows["control"])
    monkeypatch.setattr(audit.w, "validate_result", lambda cfg, j: rows[j["name"]])
    result = audit.check(tmp_path / "peer", tmp_path / "reference.json")
    assert result["count"] == 2
    assert result["historical_execution_attested"] is False
    assert result["checkpoint_forward_replayed"] is False
    assert result["ranks_or_promotes"] is False
    rows["contrast"] = dict(rows["contrast"], support_hash={"TRAIN": "other"})
    with pytest.raises(ValueError, match="SUPPORT_OR_COLUMN_ORDER"):
        audit.check(tmp_path / "peer", tmp_path / "reference.json")
    rows["contrast"] = dict(rows["control"], consumed_features=["y", "x"])
    with pytest.raises(ValueError, match="SUPPORT_OR_COLUMN_ORDER"):
        audit.check(tmp_path / "peer", tmp_path / "reference.json")


def test_input_audit_rejects_changes_and_extra_python_modules(tmp_path, monkeypatch):
    for name in ("T3_MANIFEST.json", "T3_UNITS_satoshi.json"):
        audit.w.atomic(tmp_path / name, {})
    (tmp_path / "code/app").mkdir(parents=True)
    module = tmp_path / "code/app/model.py"
    module.touch()
    monkeypatch.setitem(audit.PINS, "T3_MANIFEST.json", audit.w.digest(tmp_path / "T3_MANIFEST.json"))
    monkeypatch.setattr(audit, "ASSIGNMENT", audit.w.digest(tmp_path / "T3_UNITS_satoshi.json"))
    cfg = {"root": str(tmp_path / "source"), "pins": {
        str(tmp_path / "source/code/app/model.py"): audit.w.digest(module)}}
    assert audit.verify_inputs(tmp_path, cfg) == 1
    (tmp_path / "code/app/extra.py").touch()
    with pytest.raises(ValueError, match="CODE_FILE_SET_CHANGED"):
        audit.verify_inputs(tmp_path, cfg)
    cfg["pins"][str(tmp_path / "source/code/app/model.py")] = "wrong"
    with pytest.raises(ValueError, match="INPUT_CHANGED"):
        audit.verify_inputs(tmp_path, cfg)
