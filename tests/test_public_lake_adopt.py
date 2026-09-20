"""RP41 (dictum F1): an adoption either succeeds and says so, or leaves the host as it found it.

The defect: the configuration was replaced and the service restarted BEFORE the recovery block, so a
timeout inside the restart left the successor configuration in place with no receipt and no rollback,
and the CLI returned 0 even when `adopted` was false. Every mutation below is exercised on a COPY of
the real configuration; no production service or file is touched by these rules.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("df_public_lake_adopt")

pytestmark = pytest.mark.skipif(not A.RUNTIME_CONFIG.is_file(),
                                reason="the deployed data-gov configuration is not present here")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of the configuration and a service that does nothing: the adopter must never reach the
    real one in these rules."""
    config = tmp_path / "5055.runtime.json"
    config.write_text(A.RUNTIME_CONFIG.read_text())
    monkeypatch.setattr(A, "RUNTIME_CONFIG", config)
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(A, "run", fake_run)
    monkeypatch.setattr(A, "_service_healthy", lambda url, tries=120: True)
    # the unit file of the disposable lake host goes into the sandbox, never into the user's units
    units = tmp_path / "systemd"
    units.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "HOME", tmp_path)
    (tmp_path / ".config/systemd/user").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(A, "service_state", lambda unit: {"ActiveState": "active", "SubState": "running",
                                                          "NRestarts": "0", "returncode": 0})
    monkeypatch.setattr(A, "inventory", lambda: {"services": {"crispdm-data-gov.service": {"ActiveState": "active"}},
                                                 "public_panel_lake_registered": True, "lakes": [], "at": "now"})
    monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": True, "delivered_bytes_verified": True,
                                                             "refusals_hold": True, "campaign_closed": True})
    return {"config": config, "calls": calls, "before": config.read_text()}


def _rehearsal(tmp_path, sandbox, *, ok=True, bind=True, principals=("predictor",)):
    cfg = json.loads(sandbox["config"].read_text())
    deployed = A.deployed_external(cfg, principals, port=A.LAKE_HOST_PORT)
    path = tmp_path / f"REHEARSAL-{ok}-{bind}.json"
    binding = (A.rehearsal_binding(None, deployed) if bind else
               {"code_identity": "0" * 40, "deployed_config_sha256": "0" * 64,
                "panels_sha256": {}, "provider_sha256": None})
    path.write_text(json.dumps({"route_ok": ok, "binding": binding}))
    return path


def test_RP41_a_successful_adoption_writes_its_receipt_and_reports_success(tmp_path, sandbox):
    state = tmp_path / "state"
    receipt = A.adopt(state, principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    assert receipt["adopted"] is True
    assert (state / "RECEIPT.json").is_file() and json.loads((state / "RECEIPT.json").read_text())["adopted"]
    assert any("restart" in " ".join(c) for c in sandbox["calls"])
    assert json.loads(sandbox["config"].read_text())["lakes"][-1]["lake_id"] == A.LAKE_ID


@pytest.mark.parametrize("failure", ["restart_timeout", "restart_nonzero", "unhealthy", "post_check_failed",
                                     "route_raises"])
def test_RP41_every_failure_leaves_the_configuration_restored_and_a_receipt_behind(tmp_path, sandbox, monkeypatch, failure):
    if failure == "restart_timeout":
        def boom(argv, **kw):
            if "restart" in argv and "restored" not in str(argv):
                raise subprocess.TimeoutExpired(argv, 180)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        monkeypatch.setattr(A, "run", boom)
    elif failure == "restart_nonzero":
        monkeypatch.setattr(A, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="no"))
    elif failure == "unhealthy":
        monkeypatch.setattr(A, "_service_healthy", lambda url, tries=120: False)
    elif failure == "post_check_failed":
        monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": False, "why": "no terminal"})
    else:
        def raiser(*a, **kw):
            raise RuntimeError("the route blew up")
        monkeypatch.setattr(A, "route_checks", raiser)
    state = tmp_path / "state"
    receipt = A.adopt(state, principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    assert receipt["adopted"] is False
    stored = json.loads((state / "RECEIPT.json").read_text())
    assert stored["adopted"] is False and ("failure" in stored or "rollback" in stored)
    assert "rollback" in stored, "a failed adoption must record its rollback"
    assert sandbox["config"].read_text() == sandbox["before"], "the configuration was left changed"
    if failure != "restart_timeout":
        assert stored["rollback"].get("config_restored") is True
        assert "lake_absent_again" in stored["rollback"]


def test_RP41_a_rollback_that_fails_is_recorded_as_a_failure(tmp_path, sandbox, monkeypatch):
    monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": False})
    real_copy = A.shutil.copy2

    def bad_copy(src, dst, *a, **kw):
        if str(dst).endswith("5055.runtime.json"):
            raise PermissionError("cannot restore")
        return real_copy(src, dst, *a, **kw)
    monkeypatch.setattr(A.shutil, "copy2", bad_copy)
    state = tmp_path / "state"
    receipt = A.adopt(state, principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox))
    stored = json.loads((state / "RECEIPT.json").read_text())
    assert stored["adopted"] is False
    assert stored["rollback"]["rolled_back"] is False and "PermissionError" in stored["rollback"]["error"]


def test_RP41_an_adoption_without_a_bound_rehearsal_never_touches_the_configuration(tmp_path, sandbox):
    state = tmp_path / "no-rehearsal"
    receipt = A.adopt(state, principals=["predictor"], rehearsal=None)
    assert receipt["adopted"] is False and "rehearsal" in receipt["refused"]
    assert sandbox["config"].read_text() == sandbox["before"] and not sandbox["calls"]
    state2 = tmp_path / "failed-rehearsal"
    receipt2 = A.adopt(state2, principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox, ok=False))
    assert receipt2["adopted"] is False and "did not pass" in receipt2["binding"]["why"]
    state3 = tmp_path / "foreign-rehearsal"
    receipt3 = A.adopt(state3, principals=["predictor"], rehearsal=_rehearsal(tmp_path, sandbox, bind=False))
    assert receipt3["adopted"] is False and "differs in" in receipt3["binding"]["why"]
    assert sandbox["config"].read_text() == sandbox["before"]


def test_RP41_the_cli_exits_non_zero_when_it_did_not_adopt(tmp_path, sandbox, monkeypatch, capsys):
    monkeypatch.setattr(A, "route_checks", lambda *a, **kw: {"route_complete": False})
    code = A.main(["adopt", "--state-dir", str(tmp_path / "state"), "--rehearsal", str(_rehearsal(tmp_path, sandbox))])
    assert code == 2
    capsys.readouterr()
    ok = A.main(["adopt", "--state-dir", str(tmp_path / "ok")])            # no rehearsal: refused, still non-zero
    assert ok == 2


def test_RP41_the_inspection_modes_never_write_production(tmp_path, sandbox, capsys):
    before = sandbox["config"].read_text()
    assert A.main(["inventory"]) == 0
    assert A.main(["contract"]) == 0
    capsys.readouterr()
    assert sandbox["config"].read_text() == before and not sandbox["calls"]


def test_RP41_the_archive_holdout_is_the_archives_own_first_day(tmp_path):
    assert A.HOLDOUT_START == min(A.ARCHIVE_FIRST_DAY.values()) == "2006-12-16"
    assert "retrospective archive" in A.HOLDOUT_REASON
    entry = A.lake_entry()["entry"]
    assert entry["holdout_start"] == A.HOLDOUT_START and sorted(entry["include_globs"]) == sorted(A.RESOURCES)
    assert all("availability" not in c for c in entry["resource_contracts"].values())
    assert "governed_download" in A.EXTERNAL_HOST_DIVERGENCE["correction"]
    assert A.EXTERNAL_HOST_DIVERGENCE["status"].startswith("CORRECTED_AND_REHEARSED")
