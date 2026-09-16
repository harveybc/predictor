"""Tearing down one disposable stack must not touch anything else that shares its name.

U1 of `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md`:

    "Replace broad process-name teardown in the disposable-stack harness with ownership of the
     actual subprocess handles it created, or an isolated service group belonging only to that
     stack. [...] start two disposable stacks using the SAME module names; tear down one and
     prove the other still answers and retains its data. [...] No pkill/killall by module name."

This comes from a real outage I caused: `pkill -f "data_warehouse_service.main"` matched the
PRODUCTION warehouse host as well as the disposable one, and :5057 went down for nearly three
hours. The module name is shared by design — the whole point of the store hosts is that the
same module serves any backend — so it can never be the thing teardown selects on.

The survivor here is a **second disposable stack**, never production: the order says so, and
using production as a survivor test would be the same mistake in a nicer costume.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "tools" / "disposable_route_stack.py"

spec = importlib.util.spec_from_file_location("disposable_route_stack_under_test", HARNESS)
harness = importlib.util.module_from_spec(spec)
sys.modules["disposable_route_stack_under_test"] = harness
spec.loader.exec_module(harness)


def sleeper(tmp_path: Path, marker: str, name: str):
    """A stand-in service whose command line carries the stack's marker, as a real one does.

    Deliberately NOT a store host: this file is about which process is selected, not about what
    it serves, and a test that needs PostgreSQL to prove a signal went to the right PID is
    testing the wrong thing.
    """
    script = tmp_path / f"{name}.py"
    script.write_text("import sys, time\ntime.sleep(600)\n", encoding="utf-8")
    log = tmp_path / f"{name}.log"
    return harness.start([sys.executable, str(script), "--load_config", marker],
                         tmp_path, dict(os.environ), log)


def alive(pid: int) -> bool:
    return harness.cmdline_of(pid) is not None


@pytest.fixture
def two_stacks(tmp_path):
    """Two stacks, same module names, different work directories."""
    started = []
    state = {}
    for name in ("alpha", "beta"):
        work = tmp_path / name
        work.mkdir()
        processes = {role: sleeper(work, str(work), f"{role}_{name}")
                     for role in ("lake", "warehouse", "gov")}
        started.extend(processes.values())
        state[name] = {"work": str(work),
                       **{f"{role}_pid": process.pid for role, process in processes.items()}}
    yield state
    for process in started:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), 9)


def test_tearing_down_one_stack_leaves_the_other_running(two_stacks):
    """The rule the outage broke. Both stacks run the same module name."""
    alpha, beta = two_stacks["alpha"], two_stacks["beta"]
    assert all(alive(alpha[f"{r}_pid"]) for r in ("lake", "warehouse", "gov"))
    assert all(alive(beta[f"{r}_pid"]) for r in ("lake", "warehouse", "gov"))

    report = harness.teardown(alpha, grace=10)

    assert report["complete"] is True
    assert report["stopped"] == 3
    assert report["refused"] == 0
    assert not any(alive(alpha[f"{r}_pid"]) for r in ("lake", "warehouse", "gov"))
    assert all(alive(beta[f"{r}_pid"]) for r in ("lake", "warehouse", "gov")), (
        "the survivor runs the same module name and must be untouched")


def test_the_survivor_still_answers_and_keeps_its_data(tmp_path):
    """Not just alive: still serving, and its state still there after the neighbour dies."""
    port = harness.free_port()
    server_dir = tmp_path / "survivor"
    server_dir.mkdir()
    (server_dir / "data.txt").write_text("kept", encoding="utf-8")
    script = server_dir / "server.py"
    script.write_text(
        "import http.server, sys, os\n"
        "os.chdir(sys.argv[sys.argv.index('--root') + 1])\n"
        "http.server.HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])),\n"
        "    http.server.SimpleHTTPRequestHandler).serve_forever()\n", encoding="utf-8")
    survivor = harness.start(
        [sys.executable, str(script), "--port", str(port), "--root", str(server_dir),
         "--load_config", str(server_dir)], server_dir, dict(os.environ),
        tmp_path / "survivor.log")

    doomed_work = tmp_path / "doomed"
    doomed_work.mkdir()
    doomed = sleeper(doomed_work, str(doomed_work), "doomed")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/data.txt", timeout=2)
            break
        except Exception:
            time.sleep(0.2)

    try:
        report = harness.teardown({"work": str(doomed_work), "lake_pid": doomed.pid}, grace=10)
        assert report["stopped"] == 1
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/data.txt", timeout=5) as answer:
            assert answer.status == 200
            assert answer.read().decode() == "kept", "the survivor kept its data"
    finally:
        os.killpg(os.getpgid(survivor.pid), 9)


def test_a_reused_pid_is_refused_rather_than_signalled(tmp_path):
    """Stale teardown metadata: the PID is alive, but it is somebody else's.

    A live process whose command line does not carry this stack's marker must be reported and
    left alone. This is the exact shape of the production incident, reduced to one process.
    """
    stranger_dir = tmp_path / "stranger"
    stranger_dir.mkdir()
    stranger = sleeper(stranger_dir, str(stranger_dir), "stranger")
    stale = {"work": str(tmp_path / "a-stack-that-is-gone"), "lake_pid": stranger.pid}
    try:
        report = harness.teardown(stale, grace=5)
        entry = {item["role"]: item for item in report["processes"]}["lake_pid"]
        assert entry["outcome"] == "PID_REUSED_REFUSED"
        assert report["refused"] == 1
        assert report["stopped"] == 0
        assert alive(stranger.pid), "a process that is not ours must survive teardown"
        assert str(stranger_dir) in entry["observed_cmdline"], (
            "the report must say WHOSE process it refused to kill")
    finally:
        os.killpg(os.getpgid(stranger.pid), 9)


def test_a_process_that_is_already_gone_is_not_an_error(tmp_path):
    work = tmp_path / "finished"
    work.mkdir()
    process = sleeper(work, str(work), "short")
    os.killpg(os.getpgid(process.pid), 9)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and alive(process.pid):
        time.sleep(0.1)

    report = harness.teardown({"work": str(work), "gov_pid": process.pid}, grace=5)
    outcomes = {entry["role"]: entry["outcome"] for entry in report["processes"]}
    assert outcomes["gov_pid"] == "ALREADY_GONE"
    assert report["complete"] is True
    assert report["stopped"] == 0


def test_a_partial_setup_records_what_was_never_started(tmp_path):
    """A stack that failed halfway has PIDs for some roles and none for others."""
    work = tmp_path / "partial"
    work.mkdir()
    process = sleeper(work, str(work), "only_lake")
    report = harness.teardown({"work": str(work), "lake_pid": process.pid,
                               "warehouse_pid": None}, grace=10)
    outcomes = {entry["role"]: entry["outcome"] for entry in report["processes"]}
    assert outcomes["lake_pid"] == "TERMINATED"
    assert outcomes["warehouse_pid"] == "NOT_RECORDED"
    assert outcomes["gov_pid"] == "NOT_RECORDED"
    assert report["complete"] is True


def test_the_harness_offers_no_way_to_select_a_process_by_name():
    """The regression that matters: no pkill, no killall, no matching on a module name."""
    source = HARNESS.read_text(encoding="utf-8")
    for forbidden in ("pkill", "killall", "pgrep"):
        assert forbidden not in source, f"{forbidden} is how the production host was killed"


def test_teardown_runs_as_a_command_and_reports_as_json(tmp_path):
    """The path an operator actually uses, exercised through the real entry point."""
    work = tmp_path / "cli"
    work.mkdir()
    process = sleeper(work, str(work), "cli_lake")
    state = work / "STACK.json"
    state.write_text(json.dumps({"work": str(work), "lake_pid": process.pid}), encoding="utf-8")

    result = subprocess.run([sys.executable, str(HARNESS), "--teardown", str(state)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["stopped"] == 1
    assert not alive(process.pid)
