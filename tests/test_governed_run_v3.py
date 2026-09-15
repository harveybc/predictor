"""Fail-safe Flow v3 contract for the predictor consumer."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "governed_run_v3_test_subject"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "governed_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GR = _load()


def _args(tmp_path):
    source = tmp_path / "lake" / "x.csv"
    source.parent.mkdir()
    source.write_text("available_time,x\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "x_train_file": str(source),
        "y_train_file": str(source),
        "epochs": 1,
    }), encoding="utf-8")
    key_file = tmp_path / "key"
    key_file.write_text("test-key\n", encoding="utf-8")
    return Namespace(
        load_config=str(config), experiment_key="run-001", experiment_set_key="set-1",
        gov_url="http://data-gov.invalid", api_key_file=str(key_file), lake="features",
        lake_root=str(source.parent), metrics_lake="cube", out_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"), outbox_dir=str(tmp_path / "outbox"),
        range_from=None, range_to=None, project="predictor", phase="test",
        # `--classification` arrived with the mechanical-run flag and this fixture was not
        # updated with it, so every rule here raised AttributeError before reaching its
        # subject. The default matches the parser's.
        classification="GOVERNING",
    )


class FakeGov:
    instances = []
    fail_terminal = False
    predictor_exit = 0

    def __init__(self, base_url, api_key, experiment_key):
        self.calls = []
        self.terminals = []
        self.finished = False
        type(self).instances.append(self)

    def submit_campaign(self, body):
        self.calls.append(("campaign", body))
        return 201, {"campaign_sha256": "c" * 64, "stored": True}

    def governed_download(self, campaign, unit, lake, resource, role, cache, start=None, end=None):
        self.calls.append(("download", role))
        path = Path(cache) / f"{role}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("available_time,x\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
        return 200, {
            "path": str(path), "sha256": "d" * 64, "bytes": path.stat().st_size,
            "source_sha256": "e" * 64, "delivery": "FULL",
            "time_column": "available_time", "availability_contract_sha256": "a" * 64,
            "delivery_id": ("1" if role.startswith("x") else "2") * 32,
            "verification_state": "VERIFIED_TRANSFER", "cached": False,
            "resource": resource, "role": role,
        }

    def report_terminal(self, campaign, unit, terminal):
        self.calls.append(("terminal", terminal["status"]))
        if type(self).fail_terminal:
            raise GR.GovernedRunError("cube unavailable")
        self.terminals.append(terminal)
        self.finished = True
        return 201, {"terminal_sha256": "f" * 64, "stored": True}

    def reconcile_campaign(self, campaign):
        self.calls.append(("reconcile", campaign))
        return 200, {
            "campaign_sha256": campaign,
            "missing_units": [] if self.finished else ["run-001"],
            "accounting_only": [], "lake_only": [],
        }


def _fake_predictor_run(command, cwd=None, env=None):
    cfg_path = Path(command[command.index("--load_config") + 1])
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    if FakeGov.predictor_exit == 0:
        Path(config["save_config"]).write_text(json.dumps(config), encoding="utf-8")
        Path(config["results_file"]).write_text(
            "Metric,Average,Std Dev,Min,Max\nTest MAE H1,0.1,0.0,0.1,0.1\n",
            encoding="utf-8",
        )
    return Namespace(returncode=FakeGov.predictor_exit)


@pytest.fixture(autouse=True)
def reset_fake():
    FakeGov.instances.clear()
    FakeGov.fail_terminal = False
    FakeGov.predictor_exit = 0


def _patch(monkeypatch):
    monkeypatch.setattr(GR, "GovHttp", FakeGov)
    monkeypatch.setattr(GR, "strict_code_identity", lambda root: {"kind": "git_commit", "value": "a" * 40})
    monkeypatch.setattr(GR.subprocess, "run", _fake_predictor_run)


def test_execution_spec_is_stable_across_local_paths(tmp_path):
    config = {"x_train_file": "/one/x.csv", "results_file": "/one/results.csv", "epochs": 2}
    datasets = [{"lake": "features", "resource": "x.csv", "role": "x_train_file",
                 "from": None, "to": None}]
    one = GR.execution_spec(config, datasets, ["--epochs", "3"])
    config["x_train_file"] = "/elsewhere/x.csv"
    config["results_file"] = "/elsewhere/results.csv"
    assert GR.execution_spec(config, datasets, ["--epochs", "3"]) == one


def test_strict_code_identity_refuses_dirty_checkout(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "run.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "add", "run.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    assert GR.strict_code_identity(tmp_path)["kind"] == "git_commit"
    (tmp_path / "run.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(GR.GovernedRunError, match="clean checkout"):
        GR.strict_code_identity(tmp_path)


def test_outbox_keeps_failure_and_retries_exactly_once(tmp_path):
    outbox = GR.TerminalOutbox(tmp_path / "outbox")
    envelope = {"campaign_sha256": "c" * 64, "unit_id": "u", "terminal": {"schema": "x"}}
    first = outbox.put(envelope)
    assert first.path.is_file()
    assert outbox.flush(lambda payload: (_ for _ in ()).throw(RuntimeError("down"))) == {
        "sent": 0, "pending": 1, "failures": {first.path.name: "RuntimeError: down"},
    }
    calls = []
    assert outbox.flush(lambda payload: calls.append(payload) or {"terminal_sha256": "f" * 64}) == {
        "sent": 1, "pending": 0, "failures": {},
    }
    assert calls == [envelope]
    assert outbox.flush(lambda payload: calls.append(payload)) == {
        "sent": 0, "pending": 0, "failures": {},
    }


def test_metric_labels_become_terminal_keys():
    """predictor writes `Train Naive MAE H9`; governed_terminal.v1 refuses a
    metric name with a space (400 for the whole terminal, pending forever)."""
    rows = GR.parse_results_rows([
        {"Metric": "Train Naive MAE H9", "Average": "1", "Std Dev": "0", "Min": "1", "Max": "1"},
        {"Metric": "Test MAE H9", "Average": "2", "Std Dev": "0", "Min": "2", "Max": "2"},
        {"Metric": "Sharpe (annual)", "Average": "3", "Std Dev": "0", "Min": "3", "Max": "3"},
    ])
    assert [(r["metric"], r["split"], r["horizon"]) for r in rows] == [
        ("Naive_MAE", "train", 9), ("MAE", "test", 9), ("Sharpe_annual", None, None),
    ]
    for row in rows:
        assert GR.re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", row["metric"])
    assert GR.metric_key("   ") == "metric"


def test_success_registers_downloads_and_terminal_before_return(tmp_path, monkeypatch):
    _patch(monkeypatch)
    result = GR.run(_args(tmp_path), ["--epochs", "2"])
    gov = FakeGov.instances[-1]
    assert result["status"] == "COMPLETED"
    assert [call[0] for call in gov.calls] == [
        "campaign", "reconcile", "download", "download", "terminal", "reconcile", "reconcile",
    ]
    assert gov.terminals[0]["status"] == "COMPLETED"
    assert len(gov.terminals[0]["deliveries"]) == 2
    assert gov.terminals[0]["metrics"][0]["metric"] == "MAE"
    assert not list((tmp_path / "outbox" / "pending").glob("*.json"))


def test_predictor_failure_still_persists_and_reports_terminal(tmp_path, monkeypatch):
    _patch(monkeypatch)
    FakeGov.predictor_exit = 7
    with pytest.raises(GR.GovernedRunError, match="predictor exited 7"):
        GR.run(_args(tmp_path), [])
    gov = FakeGov.instances[-1]
    assert gov.terminals[0]["status"] == "FAILED"
    assert gov.terminals[0]["reason"] == "PREDICTOR_EXIT_7"
    receipt = json.loads((tmp_path / "out" / "GOVERNED_RUN.json").read_text())
    assert receipt["status"] == "FAILED"


def test_stale_output_refuses_without_overwriting_or_downloading(tmp_path, monkeypatch):
    _patch(monkeypatch)
    args = _args(tmp_path)
    stale = Path(args.out_dir) / "results.csv"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"prior scientific result\n")

    with pytest.raises(GR.GovernedRunError, match="output namespace is not fresh"):
        GR.run(args, [])

    gov = FakeGov.instances[-1]
    assert stale.read_bytes() == b"prior scientific result\n"
    assert not [call for call in gov.calls if call[0] == "download"]
    assert gov.terminals[0]["status"] == "REFUSED"


def test_terminal_outage_leaves_durable_pending_result(tmp_path, monkeypatch):
    _patch(monkeypatch)
    FakeGov.fail_terminal = True
    with pytest.raises(GR.GovernedRunError, match="terminal remains pending"):
        GR.run(_args(tmp_path), [])
    pending = list((tmp_path / "outbox" / "pending").glob("*.json"))
    assert len(pending) == 1
    payload = json.loads(pending[0].read_text())
    assert payload["terminal"]["status"] == "COMPLETED"
