import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


def test_partitions_cover_the_21_odd_units_without_duplicates():
    from t3_parallel_handoff import partition
    slow, fast = partition()
    assert len(slow) == 7 and len(fast) == 14
    assert slow[0] == "t3_unit_01"
    assert not set(slow) & set(fast)
    assert set(slow + fast) == {f"t3_unit_{i:02d}" for i in range(1, 43, 2)}


def test_manager_pause_requires_the_exact_owned_worker_command():
    from t3_parallel_handoff import owns_manager
    assert owns_manager(["/usr/bin/python3", "/job/expert_package_worker.py", "run", "--root", "/job"], "/job")
    for argv in (["python3", "/other/expert_package_worker.py", "run", "--root", "/job"],
                 ["python3", "/job/expert_package_worker.py", "status", "--root", "/job"],
                 ["python3", "/job/expert_package_worker.py", "run", "--root", "/other"]):
        assert not owns_manager(argv, "/job")


def test_duplicate_receipts_are_not_silently_chosen():
    from t3_parallel_handoff import checked_rows
    one = {"unit": "t3_unit_01", "origin": "first"}
    assert checked_rows([one]) == {"t3_unit_01": one}
    with pytest.raises(ValueError, match="DUPLICATE"):
        checked_rows([one, dict(one, origin="second")])
    with pytest.raises(ValueError, match="UNEXPECTED"):
        checked_rows([{"unit": "t3_unit_02"}])


def test_handoff_waits_without_signalling_an_active_scientific_child(tmp_path, monkeypatch):
    import t3_parallel_handoff as h
    cfg = {"control": str(tmp_path / "control"), "source": str(tmp_path / "source"),
           "slow_root": str(tmp_path / "target")}
    h.w.atomic(Path(cfg["control"]) / "SOURCE_PAUSED.private.json", {"manager_pid": 123})
    monkeypatch.setattr(h.w, "live_children", lambda root: [456])
    monkeypatch.setattr(h.os, "kill", lambda *args: pytest.fail("scientific child was signalled"))
    monkeypatch.setattr(h.w, "seal_result", lambda *args: pytest.fail("in-flight output was sealed"))
    assert h.handoff(cfg) is False


@pytest.mark.parametrize("complete", [True, False])
def test_handoff_preserves_finished_unit_and_only_schedules_disjoint_tail(tmp_path, monkeypatch, complete):
    import t3_parallel_handoff as h
    root, control, target = (tmp_path / n for n in ("source", "control", "target"))
    cfg = {"control": str(control), "source": str(root), "slow_root": str(target),
           "source_service": "test-source.service", "python": "/python", "launcher": "/admit",
           "templates": "/templates", "source_collector": "test-old", "slow_collector": "test-tail"}
    original = {"root": str(root), "pins": {}, "gpu_uuid": "GPU-test", "memory": "4G"}
    job = {"name": "t3_unit_01"}
    h.w.atomic(root / "ops/CONFIG.private.json", original)
    h.w.atomic(root / "ops/JOBS.json", {job["name"]: {"state": "RUNNING"}})
    h.w.atomic(control / "SOURCE_PAUSED.private.json", {
        "source_config_sha256": h.w.identity(original), "manager_pid": 99999999})
    monkeypatch.setattr(h.w, "live_children", lambda root: [])
    monkeypatch.setattr(h, "launcher_live", lambda context: False)
    monkeypatch.setattr(h.w, "planned_jobs", lambda cfg: [job])
    def seal(cfg, selected):
        if not complete:
            raise ValueError("native output incomplete")
        h.w.atomic(root / "ops/first.RESULT.json", {"job": job, "config_sha256": h.w.identity(original)})
    monkeypatch.setattr(h.w, "seal_result", seal)
    commands = []
    monkeypatch.setattr(h.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    assert h.handoff(cfg) is True
    install = next(cmd for cmd in commands if "--jobs" in cmd)
    selected = install[install.index("--jobs") + 1].split(",")
    assert ("t3_unit_01" in selected) is not complete
    assert not set(selected) & set(h.partition()[1])
    assert install[install.index("--memory") + 1] == "5G"
    assert h.w.read(control / "HANDOFF.json")["slow_selected"] == selected


def test_retrying_an_incomplete_first_unit_is_pending_not_a_final_failure(tmp_path, monkeypatch):
    import t3_parallel_handoff as h
    source, tail, control = (tmp_path / n for n in ("source", "tail", "control"))
    h.w.atomic(source / "ops/CONFIG.private.json", {})
    h.w.atomic(source / "ops/JOBS.json", {"t3_unit_01": {"state": "FAILED"}})
    h.w.atomic(tail / "ops/CONFIG.private.json", {"only_jobs": ["t3_unit_01", "t3_unit_03"]})
    monkeypatch.setattr(h, "validate_mirror", lambda root: {})
    result = h.index({"source": str(source), "slow_root": str(tail), "fast_mirror": str(tmp_path / "fast"),
                      "control": str(control), "timer": "test.timer"})
    assert result["failed"] == 0 and result["pending"] == 21
