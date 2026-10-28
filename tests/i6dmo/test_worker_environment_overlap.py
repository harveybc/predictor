"""Actual worker claim/launch path, with no process, service, GPU access or fit."""
import json
import time

import pytest

from tests.i6dmo.ops_fakes import facts
from tools.i6dmo import campaign as C, worker as W


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_worker_locked_merges_overlapping_env_and_preserves_role_hardening(tmp_path, monkeypatch, device):
    role = "synthetic"
    role_env = dict(APP_MARKER="role", ROLE_ONLY=17, OMP_NUM_THREADS=2, PYTHONPATH="synthetic-shared-path",
                    CUDA_VISIBLE_DEVICES="role-must-not-select-device", PYTHONDONTWRITEBYTECODE="0",
                    TF_CPP_MIN_LOG_LEVEL="0", I6DMO_FORWARD_ENV="role-must-not-replace-forward-list")
    extra = dict(APP_MARKER="extra", EXTRA_ONLY="extra", OMP_NUM_THREADS="3", PYTHONPATH="synthetic-shared-path",
                 CUDA_VISIBLE_DEVICES="extra-must-not-select-device", PYTHONDONTWRITEBYTECODE="0",
                 TF_CPP_MIN_LOG_LEVEL="1", I6DMO_FORWARD_ENV="extra-must-not-replace-forward-list")
    monkeypatch.setenv("APP_MARKER", "parent")
    monkeypatch.setenv("PARENT_ONLY", "parent")
    gpu = dict(uuid="GPU-fake-preferred-0", **{"class": "rtx5090"}, heavy=True)
    deployment = dict(roles={role: dict(host_key="synthetic-host", cpu_units=True,
        environment=role_env, gpus=[gpu] if device == "gpu" else [], order=0)},
        launcher=dict(crispdm_run=None, initial_cap_bytes=2 << 30, cpu_cap_bytes=2 << 30))
    unit = dict(unit_id="comparison", kind="comparison", family="comparison", device=device,
                depends_on=[], identity_sha256="a" * 64)
    plan = dict(campaign_id="synthetic-env-overlap", effective_config_sha256="b" * 64,
                units=[unit])
    effective = dict(mode=C.MODE_ELAPSED, prior_roots=[], data_pins={},
                     entrypoints={"comparison": ["synthetic-never-executed"]})
    (tmp_path / "private").mkdir()
    (tmp_path / "private/deployment.json").write_text(json.dumps(deployment))
    (tmp_path / "PLAN.json").write_text(json.dumps(plan))
    queue = C.open_queue(tmp_path)
    try:
        queue.ensure_unit("comparison", unit, state="QUEUED")
    finally:
        queue.close()

    class CaptureOnlyExecutor:
        def __init__(self):
            self.calls = []

        def start(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("synthetic capture only; no process created")

    executor = CaptureOnlyExecutor()
    result = W._worker_locked(tmp_path, role, effective, deployment, lambda _: facts(),
                              executor, 0, 60, extra, time.time)
    assert result["result"] == "INTERRUPTED" and result["cause"] == "LAUNCH_FAILED"
    assert len(executor.calls) == 1
    launched = executor.calls[0]
    env = launched["env"]
    assert env["APP_MARKER"] == "role"
    assert env["PARENT_ONLY"] == "parent" and env["EXTRA_ONLY"] == "extra"
    assert env["ROLE_ONLY"] == "17" and env["OMP_NUM_THREADS"] == "2"
    assert env["PYTHONPATH"] == "synthetic-shared-path"
    assert env["CUDA_VISIBLE_DEVICES"] == (gpu["uuid"] if device == "gpu" else "")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert env["I6DMO_FORWARD_ENV"] == ",".join(sorted(set(role_env) - {"CUDA_VISIBLE_DEVICES"}))
    spec = json.loads((launched["folder"] / "UNIT_SPEC.json").read_text())
    assert spec["device"]["gpu_uuid"] == (gpu["uuid"] if device == "gpu" else None)
    queue = C.open_queue(tmp_path)
    try:
        assert queue.unit("comparison")["state"] == "INTERRUPTED"
        assert len(queue.attempts("comparison")) == 1
    finally:
        queue.close()
