import json
from pathlib import Path

from tools.render_modular_progress import OUTSIDE_TOLERANCE_COLOR, VERIFIED_COLOR, status_color

EV = Path(__file__).resolve().parents[1] / "docs/audits/evidence/modular_architecture_20261001"
SUMMARY = json.loads((EV / "SUMMARY.json").read_text())
FUTURE = ("Full causal ladder", "DOIN search", "RL + heuristic")


def _row(block, label):
    return next(r for r in SUMMARY[block]["rows"] if r["label"] == label)


def test_top64_mi_verified():
    assert _row("eth", "R0 MI top64")["status"] == "verified"


def test_r3_measured_replay_outside_tolerance():
    for block in ("ecl", "eth"):
        status = _row(block, "R3 warm")["status"]
        assert status.startswith("measured") and "outside" in status
        assert status_color(status) == OUTSIDE_TOLERANCE_COLOR != VERIFIED_COLOR


def test_no_future_item_complete():
    for m in SUMMARY["milestones"]:
        if m["label"].startswith(FUTURE):
            assert m["state"] == "pending"
        if m["label"].startswith("R1 / R2"):
            assert m["state"] != "complete"


def test_pending_jobs_fields():
    jobs = json.loads((EV / "PENDING_JOBS.json").read_text())["jobs"]
    for j in jobs:
        assert {"owner", "gpu", "dependency", "next_artifact"} <= set(j)
