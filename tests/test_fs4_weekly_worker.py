"""Durable weekly worker (extends tools/fs4_worker.py): terminal first, delivery rejection = FAILED, restart adopts the terminal."""
from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import pytest

from tools import fs4_weekly_worker as WK


class Recorder:
    def __init__(self, reject=False, claim=None):
        self.calls, self.reject, self.claim = [], reject, claim

    def __call__(self, args, action, extra=None, input_value=None):
        self.calls.append((action, extra or [], input_value))
        if action == "complete" and self.reject:
            raise RuntimeError("controller complete rc=2: PAIRED_ROWS_OR_NAIVE_MISMATCH")
        if action == "claim":
            return self.claim
        return {"ok": True}


def _shim(tmp_path):
    p = tmp_path / "crispdm-run"
    p.write_text('#!/usr/bin/env bash\nwhile [[ "$1" != "--" ]]; do shift; done; shift; exec "$@"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _wrapper(tmp_path, body):
    p = tmp_path / "wrapper.py"
    p.write_text(body)
    return p


def _args(tmp_path, wrapper, mode="RAW", **kw):
    base = dict(coordinator="coord", controller="/c/fs4_weekly_campaign.py", python="python3", python_local="python3", wrapper=str(wrapper),
                db="/c/q.sqlite", owner="worker_a-weekly-raw", crispdm_run=str(_shim(tmp_path)), cap="1G", wall="5m", timeout=60,
                heartbeat=3600, output_root=str(tmp_path / "out"), max_tasks=1, population="EURUSD", bar_hours=1,
                train_features=["/d/tf.parquet"], train_targets="/d/tt.parquet", val_features=["/d/vf1.parquet", "/d/vf2.parquet"],
                val_targets="/d/vt.parquet", input_mode=mode, split="validation", runner_results=None, extractor_code=None,
                test_freeze=None, gpu_uuid=None, max_gpu_temp=75, task_id=None, arm=mode)
    base.update(kw)
    return SimpleNamespace(**base)


TASK = {"task_id": "t" * 64, "set_id": "s", "week": {"start": "2024-01-01T00:00:00Z"}, "input_mode": "RAW"}
GOOD = 'import json,sys\nprint(json.dumps({"task_id": "%s", "status": "COMPLETE", "disposition": "COMPLETED"}))\n' % ("t" * 64)


def test_success_persists_the_terminal_before_delivery(tmp_path, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(WK.BASE, "controller", rec)
    args = _args(tmp_path, _wrapper(tmp_path, GOOD))
    out = WK.run_one(args, TASK)
    assert (tmp_path / "out" / f"{'t' * 64}.json").is_file()
    assert [c[0] for c in rec.calls] == ["complete"] and rec.calls[0][2]["task_id"] == TASK["task_id"]
    assert out == {"ok": True}


def test_rejected_delivery_is_failed_and_the_terminal_is_set_aside(tmp_path, monkeypatch):
    rec = Recorder(reject=True)
    monkeypatch.setattr(WK.BASE, "controller", rec)
    out = WK.run_one(_args(tmp_path, _wrapper(tmp_path, GOOD)), TASK)
    assert out["failed"] is True and "DELIVERY_REJECTED" in out["reason"]
    assert [c[0] for c in rec.calls] == ["complete", "fail"]
    assert not (tmp_path / "out" / f"{'t' * 64}.json").exists() and list((tmp_path / "out").glob("*.rejected.*"))


@pytest.mark.parametrize("body,needle", [
    ("import sys; sys.exit(2)\n", "wrapper rc=2"),
    ('print("not json")\n', "INVALID_WRAPPER_RESULT"),
    ('import json; print(json.dumps({"task_id": "other", "status": "COMPLETE"}))\n', "WRAPPER_RESULT_MISMATCH"),
    ('import json; print(json.dumps({"task_id": "%s", "status": "REFUSED"}))\n' % ("t" * 64), "WRAPPER_RESULT_MISMATCH"),
])
def test_bad_wrapper_outcomes_are_failed_never_complete(tmp_path, monkeypatch, body, needle):
    rec = Recorder()
    monkeypatch.setattr(WK.BASE, "controller", rec)
    out = WK.run_one(_args(tmp_path, _wrapper(tmp_path, body)), TASK)
    assert out["failed"] is True and needle in out["reason"]
    assert [c[0] for c in rec.calls] == ["fail"]
    assert not (tmp_path / "out" / f"{'t' * 64}.json").exists()


def test_restart_adopts_the_retained_terminal_without_running_the_wrapper(tmp_path, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(WK.BASE, "controller", rec)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / f"{'t' * 64}.json").write_text(json.dumps({"task_id": "t" * 64, "status": "COMPLETE"}))
    marker = tmp_path / "ran"
    WK.run_one(_args(tmp_path, _wrapper(tmp_path, f'open({str(marker)!r}, "w").write("x")\n' + GOOD)), TASK)
    assert not marker.exists() and [c[0] for c in rec.calls] == ["complete"]


def test_wrapper_command_and_child_env_for_each_mode(tmp_path):
    raw = WK.wrapper_command(_args(tmp_path, "w.py"), tmp_path / "claim.json")
    assert raw[1:3] == ["w.py", "run-task"] and "--runner-results" not in raw and "--task-file" in raw and "--val-features" in raw
    enc = WK.wrapper_command(_args(tmp_path, "w.py", mode="TRAINED_ENCODER", runner_results="/r", extractor_code="/fe"), tmp_path / "c.json")
    assert enc[enc.index("--runner-results") + 1] == "/r" and enc[enc.index("--extractor-code") + 1] == "/fe"
    assert WK.child_env(_args(tmp_path, "w"))["CUDA_VISIBLE_DEVICES"] == ""             # CPU slot: the child sees no GPU at all
    gpu = "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"
    env = WK.child_env(_args(tmp_path, "w", gpu_uuid=gpu))
    assert env["CUDA_VISIBLE_DEVICES"] == gpu and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_main_idles_skips_unhealthy_hosts_and_filters_by_slot(tmp_path, monkeypatch, capsys):
    rec = Recorder(claim=None)
    monkeypatch.setattr(WK.BASE, "controller", rec)
    monkeypatch.setattr(WK.BASE, "host_health", lambda args, **k: {})
    argv = ["--coordinator", "c", "--controller", "/c/x.py", "--db", "/c/q", "--owner", "worker_a-weekly-raw", "--cap", "1G",
            "--output-root", str(tmp_path / "o"), "--population", "EURUSD", "--bar-hours", "1", "--train-features", "/a", "--train-targets", "/b",
            "--val-features", "/c", "--val-targets", "/d", "--input-mode", "RAW"]
    WK.main(argv)
    assert json.loads(capsys.readouterr().out)["idle"] is True
    action, extra, _ = rec.calls[0]
    assert action == "claim" and extra[extra.index("--input-mode") + 1] == "RAW" and extra[extra.index("--population") + 1] == "EURUSD"
    assert "--min-features" not in extra
    rec.calls.clear()
    WK.main(argv + ["--max-features", "32"])
    capsys.readouterr()
    assert rec.calls[0][1][rec.calls[0][1].index("--max-features") + 1] == "32"

    def unhealthy(args, **k):
        raise WK.BASE.Unhealthy("MEM_AVAILABLE_BELOW_CAP_PLUS_RESERVE: 1 < 2")
    monkeypatch.setattr(WK.BASE, "host_health", unhealthy)
    rec.calls.clear()
    WK.main(argv)
    assert json.loads(capsys.readouterr().out)["skipped"] is True and rec.calls == []     # no claim on an unfit host
    with pytest.raises(SystemExit):
        WK.main(argv[:-2] + ["--input-mode", "TRAINED_ENCODER"])                           # encoder slot without runner results
