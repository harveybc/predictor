import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import fs4_worker as worker

TASK_ID = "a" * 64
UUID = "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"


def make_args(tmp_path, runner, arm="RAW", gpu_uuid=None, cap="1G"):
    passthrough = tmp_path / "crispdm-run"
    passthrough.write_text("#!/usr/bin/env bash\nwhile [[ $1 != -- ]]; do shift; done; shift; exec \"$@\"\n")
    passthrough.chmod(passthrough.stat().st_mode | stat.S_IEXEC)
    return SimpleNamespace(coordinator="coord", controller="/x/fs4_campaign.py", python="python3",
                           db="/x/queue.sqlite", owner="worker_b-test", runner=str(runner),
                           crispdm_run=str(passthrough), cap=cap, wall="10m", timeout=60,
                           output_root=str(tmp_path / "out"), max_tasks=1, task_id=None,
                           arm=arm, gpu_uuid=gpu_uuid, max_gpu_temp=75)


def script(tmp_path, body):
    path = tmp_path / "runner.sh"
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def task(arm="RAW"):
    return {"task_id": TASK_ID, "arm": arm, "feature_id": "px.rv5", "fold_id": "inner_2023", "seed": 0}


def good_result():
    return {"task_id": TASK_ID, "status": "COMPLETE", "seed": 0, "metrics": {"mae": 0.1, "naive_mae": 0.2}}


class FakeController:
    def __init__(self, reject_complete=False):
        self.calls = []
        self.reject_complete = reject_complete

    def __call__(self, args, action, extra=None, input_value=None):
        self.calls.append((action, extra, input_value))
        if action == "complete" and self.reject_complete:
            raise RuntimeError("controller complete rc=2: PAIRED_INPUT_ROWS_MASK_OR_NAIVE_MISMATCH")
        if action == "complete":
            return {"task_id": input_value["task_id"], "accepted": True}
        return {"task_id": extra[1] if extra else None, action: True}


def test_rejected_delivery_is_failed_and_terminal_set_aside(tmp_path, monkeypatch):
    fake = FakeController(reject_complete=True)
    monkeypatch.setattr(worker, "controller", fake)
    runner = script(tmp_path, f"cat >/dev/null; echo '{json.dumps(good_result())}'")
    out = worker.run_one(make_args(tmp_path, runner), task())
    assert out["failed"] is True and out["reason"].startswith("DELIVERY_REJECTED")
    actions = [c[0] for c in fake.calls]
    assert actions == ["complete", "fail"]
    assert not (tmp_path / "out" / f"{TASK_ID}.json").exists()
    assert list((tmp_path / "out").glob(f"{TASK_ID}.json.rejected.*"))


def test_exit_zero_without_json_is_failed_not_complete(tmp_path, monkeypatch):
    fake = FakeController()
    monkeypatch.setattr(worker, "controller", fake)
    runner = script(tmp_path, "cat >/dev/null; echo 'not json'; exit 0")
    out = worker.run_one(make_args(tmp_path, runner), task())
    assert out["failed"] is True
    assert [c[0] for c in fake.calls] == ["fail"]


def test_runner_refusal_is_failed_with_typed_reason(tmp_path, monkeypatch):
    fake = FakeController()
    monkeypatch.setattr(worker, "controller", fake)
    refusal = {"task_id": TASK_ID, "status": "REFUSED", "reason": "NOT_AVAILABLE_FOR_TRAIN"}
    runner = script(tmp_path, f"cat >/dev/null; echo '{json.dumps(refusal)}'")
    out = worker.run_one(make_args(tmp_path, runner), task())
    assert out["failed"] is True and "NOT_AVAILABLE_FOR_TRAIN" in out["reason"]
    assert [c[0] for c in fake.calls] == ["fail"]
    assert (tmp_path / "out" / f"{TASK_ID}.json.refusal").exists()


def test_accepted_delivery_keeps_local_terminal_first(tmp_path, monkeypatch):
    fake = FakeController()
    monkeypatch.setattr(worker, "controller", fake)
    runner = script(tmp_path, f"cat >/dev/null; echo '{json.dumps(good_result())}'")
    out = worker.run_one(make_args(tmp_path, runner), task())
    assert out == {"task_id": TASK_ID, "accepted": True}
    assert json.loads((tmp_path / "out" / f"{TASK_ID}.json").read_text())["status"] == "COMPLETE"


def test_retained_terminal_is_resubmitted_without_rerunning(tmp_path, monkeypatch):
    fake = FakeController()
    monkeypatch.setattr(worker, "controller", fake)
    runner = script(tmp_path, "echo MUST_NOT_RUN >&2; exit 9")
    args = make_args(tmp_path, runner)
    worker.atomic_json(Path(args.output_root) / f"{TASK_ID}.json", good_result())
    out = worker.run_one(args, task())
    assert out["accepted"] is True
    assert [c[0] for c in fake.calls] == ["complete"]


def test_trained_arm_requires_uuid_and_child_sees_expected_uuid(tmp_path):
    args = make_args(tmp_path, "/bin/true", arm="TRAINED_ENCODER")
    with pytest.raises(RuntimeError, match="TRAINED_ENCODER_REQUIRES_PHYSICAL_GPU_UUID"):
        worker.child_env(args, task("TRAINED_ENCODER"))
    args.gpu_uuid = UUID
    env = worker.child_env(args, task("TRAINED_ENCODER"))
    assert env["CUDA_VISIBLE_DEVICES"] == UUID and env["FS4_EXPECTED_GPU_UUID"] == UUID
    cpu = worker.child_env(make_args(tmp_path, "/bin/true"), task())
    assert cpu["CUDA_VISIBLE_DEVICES"] == "" and cpu["FS4_EXPECTED_GPU_UUID"] == ""


def test_health_refuses_low_memory_missing_gpu_heat_and_tmp(tmp_path):
    args = make_args(tmp_path, "/bin/true", cap="4G")
    args.output_root = str(Path.home() / ".local/state/fs4-test-durable-root")  # pytest tmp_path is under /tmp
    enough = {"MemAvailable": worker.cap_bytes("4G") + worker.DESKTOP_RESERVE_BYTES}
    assert worker.host_health(args, meminfo=enough)["cap_bytes"] == 4 * 1024 ** 3
    with pytest.raises(worker.Unhealthy, match="MEM_AVAILABLE_BELOW_CAP_PLUS_RESERVE"):
        worker.host_health(args, meminfo={"MemAvailable": enough["MemAvailable"] - 1})
    args.gpu_uuid = UUID
    with pytest.raises(worker.Unhealthy, match="GPU_UUID_NOT_VISIBLE"):
        worker.host_health(args, meminfo=enough, gpu=(False, 40, "Unknown Error"))
    with pytest.raises(worker.Unhealthy, match="GPU_TOO_HOT"):
        worker.host_health(args, meminfo=enough, gpu=(True, 80, ""))
    report = worker.host_health(args, meminfo=enough, gpu=(True, 50, ""))
    assert report["gpu_visible"] is True and report["gpu_temperature_c"] == 50
    args.gpu_uuid = "0"
    with pytest.raises(worker.Unhealthy, match="GPU_UUID_NOT_PHYSICAL_FORMAT"):
        worker.host_health(args, meminfo=enough, gpu=(True, 50, ""))
    args.gpu_uuid = None
    args.output_root = "/tmp/fs4"
    with pytest.raises(worker.Unhealthy, match="OUTPUT_ROOT_NOT_DURABLE"):
        worker.host_health(args, meminfo=enough)


def test_unhealthy_host_claims_nothing(tmp_path, monkeypatch, capsys):
    fake = FakeController()
    monkeypatch.setattr(worker, "controller", fake)
    monkeypatch.setattr(worker, "read_meminfo", lambda: {"MemAvailable": 0})
    worker.main(["--coordinator", "c", "--controller", "/x", "--db", "/q", "--owner", "o",
                 "--runner", "/bin/true", "--cap", "1G", "--output-root", str(tmp_path / "out"),
                 "--arm", "RAW"])
    assert fake.calls == []
    assert json.loads(capsys.readouterr().out)["skipped"] is True


def test_cap_parsing():
    assert worker.cap_bytes("4G") == 4 * 1024 ** 3
    assert worker.cap_bytes("4650M") == 4650 * 1024 ** 2
    with pytest.raises(ValueError):
        worker.cap_bytes("four")
