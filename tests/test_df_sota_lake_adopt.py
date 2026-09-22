"""RP92: the benchmark lake adopter is the public-panel procedure REBOUND to the official ECL store — every rule of that
procedure (additive change, rehearsal-bound bytes, rollback on a failed post-check, no second adoption, inspection modes never
write production) must hold for it. All rules run in a sandbox: a copy of the configuration and a service that does nothing."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load("df_sota_lake_adopt")
P = _load("df_public_lake_adopt")
_PRESENT = P.RUNTIME_CONFIG.is_file() and (L.STORE_ROOT / "BUILD_RECEIPT.json").is_file()
pytestmark = pytest.mark.skipif(not _PRESENT, reason="the deployed data-gov configuration or the benchmark store is not present here")
# the adopter's PRIVATE instance of the public-panel procedure (the public module itself stays untouched); binding reads the store
# receipt, so on a host without it the module must still import (and every rule skips)
A = L.bind() if _PRESENT else None


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cfg = json.loads(A.RUNTIME_CONFIG.read_text())
    cfg["lakes"] = [l for l in cfg["lakes"] if l.get("lake_id") != L.LAKE_ID]
    cfg["policies"] = [p for p in cfg["policies"] if p.get("lake") != L.LAKE_ID]
    config = tmp_path / "5055.runtime.json"
    config.write_text(json.dumps(cfg, indent=1))
    monkeypatch.setattr(A, "RUNTIME_CONFIG", config)
    calls = []
    monkeypatch.setattr(A, "run", lambda argv, **kw: (calls.append(list(argv)), subprocess.CompletedProcess(argv, 0, stdout="", stderr=""))[1])
    monkeypatch.setattr(A, "_service_healthy", lambda url, tries=120: True)
    monkeypatch.setattr(A, "HOME", tmp_path)
    (tmp_path / ".config/systemd/user").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "service_state", lambda unit: {"ActiveState": "active", "SubState": "running", "NRestarts": "0", "returncode": 0})
    monkeypatch.setattr(A, "inventory", lambda: {"services": {"crispdm-data-gov.service": {"ActiveState": "active"}}, "public_panel_lake_registered": True,
                                                 "benchmark_lake_registered": True, "lakes": [], "at": "now"})
    monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": True, "delivered_bytes_verified": True, "refusals_hold": True, "campaign_closed": True})
    return {"config": config, "calls": calls, "before": config.read_text()}


def _rehearsal(tmp_path, sandbox, *, ok=True, bind=True):
    cfg = json.loads(sandbox["config"].read_text())
    deployed = A.deployed_external(cfg, ("predictor",), port=L.LAKE_HOST_PORT)
    path = tmp_path / f"REHEARSAL-{ok}-{bind}.json"
    binding = A.rehearsal_binding(None, deployed) if bind else {"deployed_config_sha256": "0" * 64}
    path.write_text(json.dumps({"route_ok": ok, "binding": binding}))
    return path


def test_RP92_the_contract_binds_the_official_bytes_and_withholds_every_date_range():
    built = L.lake_entry()
    entry, declared = built["entry"], built["declared"]["thuml_tsl_electricity/electricity.csv"]
    assert entry["lake_id"] == "sota_benchmarks" and entry["untimed"] == entry["include_globs"] == ["thuml_tsl_electricity/electricity.csv"]
    assert declared["sha256"] == declared["provenance"]["lfs_oid_sha256"] == "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"
    assert declared["rows"] == 26304 and declared["channels"] == 321 and entry["holdout_start"] == "2016-07-01" == L.first_day()
    assert entry["resource_contracts"]["thuml_tsl_electricity/electricity.csv"]["frequency"] == "3600s"
    assert A.LAKE_ID == "sota_benchmarks" and A.LAKE_HOST_PORT == 5060 and A.LAKE_HOST_UNIT == "crispdm-data-lake-sota-benchmarks.service"
    assert A.HOLDOUT_START == "2016-07-01" and A.policy_entries(["predictor"])[0]["deny_from"] == "2016-07-01"
    assert P.LAKE_ID == "public_panels" and P.LAKE_HOST_PORT == 5059 and P.HOLDOUT_START == "2006-12-16"     # the public adopter is untouched
    # defaults bound at the public module's definition time are rebound: the route registers on THIS lake, the adoption on THIS port
    assert A.route_checks.keywords == {"lake": "sota_benchmarks"} and A.adopt.keywords == {"lake_port": 5060} and A.deployed_external.keywords == {"port": 5060}


def test_RP92_a_bound_rehearsal_adopts_additively_and_an_unbound_one_never_touches_the_configuration(tmp_path, sandbox):
    before = json.loads(sandbox["before"])
    receipt = A.adopt(tmp_path / "state-unbound", principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox, bind=False))
    assert not receipt["adopted"] and "does not bind" in receipt["refused"] and sandbox["config"].read_text() == sandbox["before"]
    receipt = A.adopt(tmp_path / "state", principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    assert receipt["adopted"] and receipt["change"]["additive"] and receipt["change"]["lakes_added"] == ["sota_benchmarks"]
    after = json.loads(sandbox["config"].read_text())
    assert [l["lake_id"] for l in before["lakes"]] + ["sota_benchmarks"] == [l["lake_id"] for l in after["lakes"]]
    assert {p["lake"] for p in after["policies"]} - {p["lake"] for p in before["policies"]} == {"sota_benchmarks"}
    unit = tmp_path / ".config/systemd/user" / L.LAKE_HOST_UNIT
    assert unit.is_file() and "5060" in unit.read_text() and str(L.STORE_ROOT) in (tmp_path / "state" / "public-panels.host.json").read_text()
    assert any("restart" in c for c in sandbox["calls"]) and (tmp_path / "state" / "RECEIPT.json").is_file()
    # a second adoption of a live entry is refused
    again = A.adopt(tmp_path / "state2", principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    assert not again["adopted"] and again.get("already_adopted")


def test_RP92_a_failed_post_check_restores_the_previous_configuration(tmp_path, sandbox, monkeypatch):
    monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": False, "why": "no terminal"})
    receipt = A.adopt(tmp_path / "state", principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    assert not receipt["adopted"] and receipt["rollback"]["rolled_back"] and sandbox["config"].read_text() == sandbox["before"]
    assert not (tmp_path / ".config/systemd/user" / L.LAKE_HOST_UNIT).exists()


def test_RP92_the_cli_inspection_modes_never_write_and_adopt_exits_non_zero_when_refused(tmp_path, sandbox, capsys):
    assert L.main(["contract", "--out", str(tmp_path / "c.json")]) == 0 and json.loads((tmp_path / "c.json").read_text())["entry"]["lake_id"] == "sota_benchmarks"
    assert sandbox["config"].read_text() == sandbox["before"]
    assert L.main(["adopt", "--state-dir", str(tmp_path / "s"), "--rehearsal", str(_rehearsal(tmp_path, sandbox, bind=False))]) == 1
    assert sandbox["config"].read_text() == sandbox["before"]
