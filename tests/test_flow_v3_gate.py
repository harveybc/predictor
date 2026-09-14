"""Flow v3 work-plan gate and coverage SQL."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_subject", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_subject"] = module
    spec.loader.exec_module(module)
    return module


G3 = _load("flow_v3_gate")
COV = _load("flow_v3_coverage")


def _manifest(**changes):
    body = {
        "schema": "governed_campaign.v1", "campaign_key": "d3-screen-001", "classification": "GOVERNING",
        "project": "predictor", "code_identity": {"kind": "git_commit", "value": "a" * 40},
        "config_sha256": "b" * 64, "input_mode": "DATASETS", "synthetic_spec_sha256": None,
        "units": ["u001", "u002"],
        "datasets": [{"lake": "financial_files", "resource": "features/x.parquet", "role": "x", "from": None, "to": None}],
        "terminal_lake": "olap_cube",
    }
    body.update(changes)
    return body


def test_manifest_validation_seals_a_digest_and_refuses_missing_declarations():
    ok = G3.validate_manifest(_manifest())
    assert ok["manifest_sha256"] == G3.sha256_text(G3.canonical(_manifest()))
    assert G3.validate_manifest({**_manifest(), "manifest_sha256": ok["manifest_sha256"]}) == ok
    with pytest.raises(ValueError, match="digest"):
        G3.validate_manifest({**_manifest(), "manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="terminal destination"):
        G3.validate_manifest(_manifest(terminal_lake=""))
    with pytest.raises(ValueError, match="declares its deliveries"):
        G3.validate_manifest(_manifest(datasets=[]))
    with pytest.raises(ValueError, match="no deliveries"):
        G3.validate_manifest(_manifest(input_mode="SYNTHETIC", synthetic_spec_sha256="c" * 64))
    with pytest.raises(ValueError, match="40-hex"):
        G3.validate_manifest(_manifest(code_identity={"kind": "git_commit", "value": "abc-dirty"}))
    assert G3.validate_manifest(_manifest(classification="NON_GOVERNING",
                                          code_identity={"kind": "git_commit", "value": "abc-dirty"}))
    with pytest.raises(ValueError, match="exactly the governed_campaign.v1 keys"):
        G3.validate_manifest(_manifest(extra=1))
    with pytest.raises(ValueError, match="no valid units"):
        G3.validate_manifest(_manifest(units=[]))


def test_governing_dispatch_requires_a_governing_manifest(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(G3.DispatchRefusal) as refusal:
        G3.require_governed_dispatch("GOVERNING", None, root=root, jobs_sha256="1" * 64)
    assert refusal.value.code == 4 and "requires --campaign-manifest" in refusal.value.reason
    doc = json.loads((root / G3.REFUSAL_FILE).read_text())
    assert doc["reason"] == refusal.value.reason and not (root / G3.GATE_FILE).exists()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(classification="NON_GOVERNING")))
    with pytest.raises(G3.DispatchRefusal, match="not GOVERNING"):
        G3.require_governed_dispatch("GOVERNING", manifest, root=tmp_path / "root2")
    manifest.write_text(json.dumps(_manifest()))
    gate = G3.require_governed_dispatch("GOVERNING", manifest, root=tmp_path / "root3", jobs_sha256="1" * 64)
    assert gate["classification"] == "GOVERNING" and gate["units"] == 2 and gate["datasets"] == 1
    assert gate["terminal_lake"] == "olap_cube" and gate["manifest_sha256"] == G3.validate_manifest(_manifest())["manifest_sha256"]
    assert json.loads((tmp_path / "root3" / G3.GATE_FILE).read_text()) == gate
    assert G3.require_governed_dispatch("GOVERNING", manifest, root=tmp_path / "root3", jobs_sha256="1" * 64) == gate
    with pytest.raises(G3.DispatchRefusal, match="different gate"):
        G3.require_governed_dispatch("GOVERNING", manifest, root=tmp_path / "root3", jobs_sha256="2" * 64)


def test_non_governing_dispatch_states_its_reason(tmp_path):
    with pytest.raises(G3.DispatchRefusal, match="states its reason"):
        G3.require_governed_dispatch("NON_GOVERNING", None, root=tmp_path / "r")
    gate = G3.require_governed_dispatch("NON_GOVERNING", None, root=tmp_path / "r2", non_governing_reason="synthetic")
    assert gate == {"schema": "flow_v3_dispatch_gate.v1", "classification": "NON_GOVERNING", "jobs_sha256": None,
                    "non_governing_reason": "synthetic"}
    with pytest.raises(G3.DispatchRefusal, match="unknown classification"):
        G3.require_governed_dispatch("DECISION", None)


def test_dispatcher_refuses_a_governing_dispatch_without_manifest(tmp_path):
    D = _load("df_dispatch")
    jobs = tmp_path / "jobs.json"
    jobs.write_text("[]")
    with pytest.raises(SystemExit) as stop:
        D.main(["--jobs-file", str(jobs), "--root", str(tmp_path / "root"), "--classification", "GOVERNING"])
    assert stop.value.code == 4
    assert json.loads((tmp_path / "root" / G3.REFUSAL_FILE).read_text())["classification"] == "GOVERNING"


def test_coverage_sql_names_the_project_and_bounds_the_limit():
    sql = COV.coverage_sql("predictor")
    assert sql.startswith("SELECT project, classification, status, COUNT(*)") and "project = 'predictor'" in sql
    latest = COV.latest_sql("predictor", 5)
    assert latest.endswith("ORDER BY finished_at DESC LIMIT 5") and "gov_terminal" in latest
    with pytest.raises(ValueError):
        COV.coverage_sql("predictor'; DROP TABLE gov_terminal; --")
    with pytest.raises(ValueError):
        COV.latest_sql("predictor", 0)
    assert COV.main(["--project", "x y", "--sql-only"]) == 2
