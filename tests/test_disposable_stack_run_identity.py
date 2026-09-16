"""Two disposable stacks at the same destination must not be able to lose each other.

Block 1 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`:

    "Prevent the STACK.json overwrite recurrence. Give every disposable invocation a unique
     immutable run identity and process manifest. Refuse reuse of an occupied directory before
     starting children, or allocate a unique child directory. Keep the old ownership record
     rather than overwriting it."

The defect is mine and it stranded a process for a day. A second stack started in the same
`--work` directory wrote its own `STACK.json` over the first one's, so the first run's children
were recorded nowhere and its own teardown answered `ALREADY_GONE` for three PIDs that had
belonged to the newer run. The scope stayed alive until the reviewer stopped it by hand.

Two further holes in the same place, both exercised below:

* the record was written only AFTER all three children were healthy, so a launch that failed
  part-way left running children and no record of them at all;
* the marker that authorises a signal was the `--work` directory the caller named, which two
  runs share by definition. A marker that two runs share cannot tell them apart.

Everything here uses stand-in children. No production service, database or port is touched,
and no test in this file starts a store host.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tools" / "disposable_route_stack.py"

spec = importlib.util.spec_from_file_location("disposable_stack_identity_under_test", HARNESS)
harness = importlib.util.module_from_spec(spec)
sys.modules["disposable_stack_identity_under_test"] = harness
spec.loader.exec_module(harness)


def sleeper(cwd: Path, marker: str, name: str):
    """A stand-in child whose command line carries the run's marker, as a real one does."""
    script = cwd / f"{name}.py"
    script.write_text("import time\ntime.sleep(600)\n", encoding="utf-8")
    return harness.start([sys.executable, str(script), "--load_config", marker],
                         cwd, dict(os.environ), cwd / f"{name}.log")


def identifiable(process, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if harness.cmdline_of(process.pid):
            return process.pid
        time.sleep(0.05)
    raise AssertionError(f"pid {process.pid} never became identifiable")


def alive(pid: int) -> bool:
    return harness.cmdline_of(pid) is not None


@pytest.fixture
def reaper():
    started = []
    yield started
    for process in started:
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def launch(work: Path, reaper, roles=("lake", "warehouse", "gov")) -> dict:
    """One run's worth of children under a fresh identity, recorded as the harness records it."""
    run = harness.begin_run(work)
    for index, role in enumerate(roles):
        process = sleeper(Path(run["directory"]), run["marker"], f"{role}-{run['run_id'][:8]}")
        reaper.append(process)
        identifiable(process)
        harness.record_child(run, f"{role}_pid", process.pid,
                             [sys.executable, "-", "--load_config", run["marker"]])
    return run


# --- a run identity that two invocations cannot share -------------------------------------

def test_every_invocation_gets_its_own_identity_and_directory(tmp_path, reaper):
    first = launch(tmp_path / "work", reaper)
    second = launch(tmp_path / "work", reaper)
    assert first["run_id"] != second["run_id"]
    assert first["directory"] != second["directory"]
    assert first["run_id"] in first["directory"]
    assert len(first["run_id"]) >= 32


def test_a_second_run_never_overwrites_the_first_ones_record(tmp_path, reaper):
    work = tmp_path / "work"
    first = launch(work, reaper)
    second = launch(work, reaper)
    held = harness.recorded_runs(work)
    assert {run["run_id"] for run in held} == {first["run_id"], second["run_id"]}
    kept = next(run for run in held if run["run_id"] == first["run_id"])
    assert kept["lake_pid"] == first["lake_pid"]


def test_the_marker_cannot_be_the_directory_the_caller_named(tmp_path, reaper):
    """Two runs share `--work` by definition, so it can never be what authorises a signal."""
    work = tmp_path / "work"
    first = launch(work, reaper)
    second = launch(work, reaper)
    assert str(work) != first["marker"]
    assert first["marker"] not in second["marker"]
    assert second["marker"] not in first["marker"]


def test_tearing_down_one_run_leaves_the_other_running(tmp_path, reaper):
    work = tmp_path / "work"
    first = launch(work, reaper)
    second = launch(work, reaper)
    report = harness.teardown(first)
    assert report["stopped"] == 3 and report["refused"] == 0
    assert not any(alive(first[f"{role}_pid"]) for role in ("lake", "warehouse", "gov"))
    assert all(alive(second[f"{role}_pid"]) for role in ("lake", "warehouse", "gov"))


def test_tearing_down_a_directory_reaches_every_run_recorded_in_it(tmp_path, reaper):
    """The case that stranded a process: the older run must still be findable."""
    work = tmp_path / "work"
    first = launch(work, reaper)
    second = launch(work, reaper)
    report = harness.teardown_directory(work)
    assert report["runs"] == 2
    assert report["stopped"] == 6 and report["refused"] == 0
    assert not any(alive(run[f"{role}_pid"])
                   for run in (first, second)
                   for role in ("lake", "warehouse", "gov"))


# --- a launch that fails part-way ----------------------------------------------------------

def test_children_are_recorded_before_the_launch_can_fail(tmp_path, reaper):
    """The record used to be written only once all three were healthy."""
    work = tmp_path / "work"
    run = launch(work, reaper, roles=("lake",))
    held = harness.recorded_runs(work)
    assert len(held) == 1
    assert held[0]["lake_pid"] == run["lake_pid"]
    assert held[0].get("warehouse_pid") is None
    assert held[0]["complete"] is False


def test_a_partly_launched_run_is_torn_down_from_its_own_record(tmp_path, reaper):
    work = tmp_path / "work"
    run = launch(work, reaper, roles=("lake", "warehouse"))
    report = harness.teardown_directory(work)
    assert report["stopped"] == 2 and report["refused"] == 0
    assert not alive(run["lake_pid"]) and not alive(run["warehouse_pid"])


def test_a_run_is_marked_complete_only_when_it_says_so(tmp_path, reaper):
    work = tmp_path / "work"
    run = launch(work, reaper)
    assert harness.recorded_runs(work)[0]["complete"] is False
    harness.finish_run(run)
    assert harness.recorded_runs(work)[0]["complete"] is True


# --- teardown after the parent has gone ----------------------------------------------------

def test_teardown_works_from_the_record_alone_after_the_parent_exits(tmp_path, reaper):
    """No handle, no parent, no directory name: the record and the marker are enough."""
    work = tmp_path / "work"
    run = launch(work, reaper)
    reloaded = json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))
    report = harness.teardown(reloaded)
    assert report["stopped"] == 3
    assert not any(alive(run[f"{role}_pid"]) for role in ("lake", "warehouse", "gov"))


def test_a_reused_pid_is_still_refused_under_the_new_identity(tmp_path, reaper):
    work = tmp_path / "work"
    run = launch(work, reaper)
    stranger = sleeper(tmp_path, "somebody-elses-config", "stranger")
    reaper.append(stranger)
    identifiable(stranger)
    run = dict(run, lake_pid=stranger.pid)
    report = harness.teardown(run)
    outcomes = {entry["role"]: entry["outcome"] for entry in report["processes"]}
    assert outcomes["lake_pid"] == "PID_REUSED_REFUSED"
    assert report["refused"] == 1
    assert alive(stranger.pid)


def test_no_process_is_selected_by_name_anywhere_in_the_harness():
    body = HARNESS.read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "pgrep", "-f data_", "psutil.process_iter"):
        assert forbidden not in body, forbidden


def test_the_directory_name_alone_never_authorises_a_signal(tmp_path, reaper):
    """A child whose argv merely mentions the work directory is not this run's child."""
    work = tmp_path / "work"
    run = launch(work, reaper)
    impostor = sleeper(tmp_path, str(work), "impostor")
    reaper.append(impostor)
    identifiable(impostor)
    report = harness.teardown(dict(run, gov_pid=impostor.pid))
    outcomes = {entry["role"]: entry["outcome"] for entry in report["processes"]}
    assert outcomes["gov_pid"] == "PID_REUSED_REFUSED"
    assert alive(impostor.pid)


# --- the real launch path, without launching a store host ----------------------------------

@pytest.fixture
def fixtures(tmp_path):
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "MANIFEST.json").write_text(json.dumps({"contracts": {}}), encoding="utf-8")
    return root


def fake_launch(monkeypatch, *, fail_at=None):
    """Run `main()` for real, with the children replaced by recorded stand-ins.

    The point under test is what the launcher WRITES and WHEN, which is the thing that
    stranded a process. Starting three store hosts to observe a JSON file would test the
    hosts.
    """
    started = []

    class Fake:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    def start(argv, cwd, env, log):
        started.append({"argv": [str(item) for item in argv], "cwd": str(cwd)})
        return Fake(900000 + len(started))

    def wait_for(url, processes, deadline=60.0):
        if fail_at is not None and len(started) >= fail_at:
            raise RuntimeError("health check failed")

    monkeypatch.setattr(harness, "start", start)
    monkeypatch.setattr(harness, "wait_for", wait_for)
    monkeypatch.setattr(harness, "stop", lambda process: None)
    return started


def run_main(work: Path, fixtures: Path) -> int:
    return harness.main(["--fixtures", str(fixtures), "--work", str(work)])


def test_the_launcher_gives_each_invocation_its_own_directory(tmp_path, fixtures, monkeypatch):
    work = tmp_path / "work"
    fake_launch(monkeypatch)
    assert run_main(work, fixtures) == 0
    assert run_main(work, fixtures) == 0
    held = harness.recorded_runs(work)
    assert len(held) == 2
    assert held[0]["directory"] != held[1]["directory"]
    assert all(Path(run["directory"]).is_dir() for run in held)
    assert all((Path(run["directory"]) / "governance.json").is_file() for run in held)


def test_two_launches_at_one_destination_keep_both_records(tmp_path, fixtures, monkeypatch):
    """The exact recurrence: the first run's ownership record must survive the second run."""
    work = tmp_path / "work"
    fake_launch(monkeypatch)
    run_main(work, fixtures)
    first = harness.recorded_runs(work)[0]
    run_main(work, fixtures)
    held = harness.recorded_runs(work)
    assert len(held) == 2
    assert held[0] == first
    assert held[0]["lake_pid"] != held[1]["lake_pid"]


def test_the_launcher_records_children_before_the_first_health_check(tmp_path, fixtures,
                                                                    monkeypatch):
    work = tmp_path / "work"
    fake_launch(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError):
        run_main(work, fixtures)
    held = harness.recorded_runs(work)
    assert len(held) == 1
    assert held[0]["lake_pid"] and held[0]["warehouse_pid"]
    assert held[0].get("gov_pid") is None
    assert held[0]["complete"] is False
    assert [child["role"] for child in held[0]["children"]] == ["lake_pid", "warehouse_pid"]


def test_a_failed_launch_still_records_the_argv_each_child_was_given(tmp_path, fixtures,
                                                                    monkeypatch):
    work = tmp_path / "work"
    fake_launch(monkeypatch, fail_at=1)
    with pytest.raises(RuntimeError):
        run_main(work, fixtures)
    child = harness.recorded_runs(work)[0]["children"][0]
    assert child["role"] == "lake_pid"
    assert "data_lake_service.main" in " ".join(child["argv"])
    assert harness.recorded_runs(work)[0]["marker"] in " ".join(child["argv"])


def test_concurrent_launches_at_one_destination_do_not_collide(tmp_path, fixtures, monkeypatch):
    import threading

    work = tmp_path / "work"
    fake_launch(monkeypatch)
    errors = []

    def go():
        try:
            run_main(work, fixtures)
        except Exception as exc:                        # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    held = harness.recorded_runs(work)
    assert len(held) == 4
    assert len({run["run_id"] for run in held}) == 4
    assert len({run["directory"] for run in held}) == 4


def test_the_pointer_at_the_destination_never_replaces_the_record(tmp_path, fixtures,
                                                                  monkeypatch):
    """`STACK.json` stays, for whoever reads it, but it is no longer what teardown depends on."""
    work = tmp_path / "work"
    fake_launch(monkeypatch)
    run_main(work, fixtures)
    first = harness.recorded_runs(work)[0]
    run_main(work, fixtures)
    pointer = json.loads((work / "STACK.json").read_text(encoding="utf-8"))
    assert pointer["run_id"] == harness.recorded_runs(work)[1]["run_id"]
    assert json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))["run_id"] == \
        first["run_id"]


def test_teardown_by_directory_runs_as_a_command(tmp_path, reaper):
    import subprocess

    work = tmp_path / "work"
    launch(work, reaper)
    finished = subprocess.run(
        [sys.executable, str(HARNESS), "--teardown", str(work)],
        capture_output=True, text=True, timeout=120)
    report = json.loads(finished.stdout)
    assert finished.returncode == 0
    assert report["runs"] == 1 and report["stopped"] == 3 and report["refused"] == 0
