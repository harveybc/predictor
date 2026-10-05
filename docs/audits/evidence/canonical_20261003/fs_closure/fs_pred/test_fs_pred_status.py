"""Tests for the FS-PRED watchdog (run: python -m pytest docs/audits/evidence/canonical_20261003/fs_closure/fs_pred/test_fs_pred_status.py)."""
from __future__ import annotations

import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("fs_pred_status", pathlib.Path(__file__).with_name("fs_pred_status.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PROGRESS = {"done": 555, "total": 4900, "method_cells_done": 111, "method_cells_total": 980, "observed_p90_method_wall_seconds": 51.2}


def test_live_pid_without_progress_under_pressure_is_stalled_not_running() -> None:
    probe = {"scope_active": True, "d_state_pids": ["3593336"], "high_events_rising": True, "cgroup_some_avg10": 12.0}
    verdict = mod.watchdog_state(PROGRESS, progress_age_seconds=2 * 3600 + 32 * 60, probe=probe, blocked=False)
    assert verdict["state"] == "STALLED"
    assert verdict["threshold_seconds"] == 600.0  # max(10 min, 3 x 51.2 s)


def test_threshold_is_three_times_p90_when_cells_are_long() -> None:
    progress = dict(PROGRESS, observed_p90_method_wall_seconds=400.0)
    probe = {"scope_active": True, "d_state_pids": ["1"], "high_events_rising": False, "cgroup_some_avg10": 0.0}
    assert mod.watchdog_state(progress, 900.0, probe, False)["state"] == "RUNNING"  # 900 < 1200
    assert mod.watchdog_state(progress, 1300.0, probe, False)["state"] == "STALLED"


def test_running_requires_recent_progress_and_an_active_scope() -> None:
    probe = {"scope_active": True, "d_state_pids": [], "high_events_rising": False, "cgroup_some_avg10": 0.0}
    assert mod.watchdog_state(PROGRESS, 120.0, probe, False)["state"] == "RUNNING"
    assert mod.watchdog_state(PROGRESS, 120.0, dict(probe, scope_active=False), False)["state"] == "IDLE"
    assert mod.watchdog_state(PROGRESS, 5000.0, probe, False)["state"] == "SUSPECT"


def test_done_and_blocked_dominate() -> None:
    assert mod.watchdog_state(dict(PROGRESS, done=4900), 0.0, None, False)["state"] == "DONE"
    assert mod.watchdog_state(PROGRESS, 0.0, None, True)["state"] == "BLOCKED"


def test_eta_withheld_until_three_post_cells() -> None:
    assert mod.eta_after_post_cells(PROGRESS, [10.0, 12.0], 1.0)["eta_utc"] is None
    eta = mod.eta_after_post_cells(PROGRESS, [10.0, 12.0, 14.0], 1.0)
    assert eta["eta_utc"] is not None and eta["post_median_wall_seconds"] == 12.0
