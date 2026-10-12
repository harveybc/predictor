"""Startup regressions from real deployment failures; no services or fits."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import i6n_historical_evaluator as evaluator
from tools import i6n_swarm_deploy as deploy
from tools.i6n_swarm import agent


@pytest.mark.parametrize("family", [None, "short", "long"])
def test_freeze_cli_forwards_portable_context_and_family(monkeypatch, capsys, family):
    seen = {}

    def freeze(config, output, experiment, fold, score_outer=False, family=None, data_root=None):
        seen.update(config=config, output=output, experiment=experiment, fold=fold,
                    score_outer=score_outer, family=family, data_root=data_root)
        return {"schema": evaluator.SCHEMA_V3}

    argv = ["evaluator", "freeze", "--config", "/data/cell.json", "--output", "/data/out",
            "--experiment", "campaign", "--fold", "fold1", "--data-root", "/data", "--score-outer"]
    if family:
        argv += ["--family", family]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluator, "freeze", freeze)
    evaluator.main()
    assert seen == dict(config=Path("/data/cell.json"), output=Path("/data/out"),
                        experiment="campaign", fold="fold1", score_outer=True,
                        family=family, data_root=Path("/data"))
    assert json.loads(capsys.readouterr().out)["schema"] == evaluator.SCHEMA_V3


def test_exported_runtime_refuses_missing_deployer_before_start(tmp_path):
    spec = {"code_dir": str(tmp_path)}
    with pytest.raises(deploy.DeployError, match="RUNTIME_ENTRYPOINT_MISSING"):
        deploy.validate_runtime_entrypoints(spec)


def test_exported_runtime_checks_actual_cli_flags(tmp_path):
    base = tmp_path / "predictor" / "tools"
    base.mkdir(parents=True)
    for name in ("i6n_swarm_deploy", "i6n_swarm_campaign", "i6n_historical_evaluator"):
        (base / (name + ".py")).write_text("# test fixture\n")
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "--data-root --family", "")

    deploy.validate_runtime_entrypoints({"code_dir": str(tmp_path)}, run=command)
    assert len(calls) == 3
    assert all(argv[-1] == "--help" for argv in calls)

    def old_cli(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "--config --output", "")

    with pytest.raises(deploy.DeployError, match="RUNTIME_CLI_INCOMPATIBLE"):
        deploy.validate_runtime_entrypoints({"code_dir": str(tmp_path)}, run=old_cli)


def test_fixed_preparation_yields_to_node_sync_after_bounded_batch(monkeypatch):
    class Ledger:
        def next_claimable(self, stages):
            return [{"stage": "freeze", "fit_id": str(i)} for i in range(50)]

    called = []
    monkeypatch.setattr(agent, "_freeze", lambda *args: called.append(args[2]["fit_id"]) or "COMPLETE")
    result = agent.run_fixed_units({}, Ledger(), boot_id="boot", admit=lambda _: True, max_units=4)
    assert called == ["0", "1", "2", "3"]
    assert result["completed"] == 4


def test_partial_export_retains_plugin_registry_from_exact_pin(tmp_path):
    target = tmp_path / "predictor"
    target.mkdir()
    (target / "tools").mkdir()
    (tmp_path / ".predictor.pin").write_text("pinned\n")
    calls = []

    def git(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"# pinned setup registry\n", b"")

    files = deploy.export_code({"code_dir": str(tmp_path), "repos": {
        "predictor": {"path": "/repo", "pin": "pinned", "paths": ["tools"]}}}, run=git)
    assert (target / "setup.py").read_bytes() == b"# pinned setup registry\n"
    assert str(target / "setup.py") in files
    assert calls == [["git", "-C", "/repo", "show", "pinned:setup.py"]]
