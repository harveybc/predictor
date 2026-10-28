import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import recover_expert_package as recovery
from expert_package_worker import atomic, jobs


def partial(tmp_path):
    cfg = {"root": str(tmp_path), "pins": {}}
    job = jobs()[0]
    out = tmp_path / "out" / job["name"]
    out.mkdir(parents=True)
    (out / "E3M.json").write_text('{"meta":')
    (out / "model.pt").write_bytes(b"preserved")
    atomic(tmp_path / "ops/JOBS.json", {job["name"]: {"state": "RUNNING", "entered": 1}})
    return cfg, job, out


def test_partial_final_write_is_preserved_and_retryable(tmp_path, monkeypatch):
    c, job, out = partial(tmp_path)
    monkeypatch.setattr(recovery, "live_children", lambda root: [])
    assert recovery.recover(c) == [job["name"]]
    state = json.loads((tmp_path / "ops/JOBS.json").read_text())[job["name"]]
    assert state["state"] == "RETRY_PENDING" and state["entered"] == 1
    assert Path(state["preserved_attempt"]).joinpath("model.pt").read_bytes() == b"preserved"
    assert not out.exists()
    assert recovery.recover(c) == []


def test_live_child_is_never_touched(tmp_path, monkeypatch):
    c, job, out = partial(tmp_path)
    monkeypatch.setattr(recovery, "live_children", lambda root: [123])
    assert recovery.recover(c) == []
    assert out.exists()


def test_accepted_receipt_prevents_automatic_repair(tmp_path, monkeypatch):
    c, job, out = partial(tmp_path)
    monkeypatch.setattr(recovery, "live_children", lambda root: [])
    atomic(tmp_path / "ops" / (job["name"] + ".RESULT.json"), {"accepted": True})
    assert recovery.recover(c) == []
    assert out.exists()


def test_recovery_uses_the_active_campaign_plan(tmp_path, monkeypatch):
    name = "t3_unit_01"
    c = {"root": str(tmp_path), "pins": {}, "campaign": "v4_t3"}
    out = tmp_path / "out" / name
    out.mkdir(parents=True)
    (out / "E3M.json").write_text('{"meta":')
    atomic(tmp_path / "ops/JOBS.json", {name: {"state": "RUNNING", "entered": 1}})
    monkeypatch.setattr(recovery, "live_children", lambda root: [])
    monkeypatch.setattr(recovery, "planned_jobs", lambda cfg: [{"name": name}], raising=False)
    assert recovery.recover(c) == [name]
    assert not out.exists()
