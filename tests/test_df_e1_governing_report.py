"""RP55: the table describes the run that FINISHED, and says which report it read.

The defect. A report is written once. When a run stopped and a later run resumed the same root and
completed it, the second report landed beside the first as REPORT.<epoch>.json — and the closure
took REPORT.json, so the published table would have described the ABORTED attempt (in the successor
run: seven units and a refusal) while the root held a complete run of sixteen closed units.

The rule now picks the report of the run that ended without stopping, refuses when two runs claim
that, falls back to the latest attempt when none did, and records the choice in the results document
so a reader never has to guess which run a table belongs to.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_e1_pilot")
DESIGN = {"design_sha256": "a" * 64}


def _write(root: Path, name: str, *, stopped=None, design_sha=DESIGN["design_sha256"], units=0):
    (root / name).write_text(json.dumps({"design_sha256": design_sha, "stopped": stopped,
                                         "terminals": [{"unit_id": f"u{i}"} for i in range(units)]}))


def test_RP55_the_report_of_the_run_that_finished_governs(tmp_path):
    _write(tmp_path, "REPORT.json", stopped="REFUSED: unit 'R0_s2' has no delivery", units=7)
    _write(tmp_path, "REPORT.1789931301.json", stopped=None, units=16)
    _write(tmp_path, "REPORT.pilot.json", stopped=None, units=2)          # the cost pilot is not the run
    path, doc, choice = P._governing_report(tmp_path, DESIGN)
    assert path.name == "REPORT.1789931301.json" and len(doc["terminals"]) == 16
    assert choice["file"] == "REPORT.1789931301.json" and "REPORT.json" in choice["others"]
    assert "REPORT.pilot.json" not in choice["others"]


def test_RP55_two_finished_runs_in_one_root_are_a_refusal(tmp_path):
    _write(tmp_path, "REPORT.json", stopped=None, units=16)
    _write(tmp_path, "REPORT.1789931301.json", stopped=None, units=16)
    with pytest.raises(SystemExit, match="not decidable"):
        P._governing_report(tmp_path, DESIGN)


def test_RP55_when_nothing_finished_the_latest_attempt_is_read_and_said_so(tmp_path):
    _write(tmp_path, "REPORT.json", stopped="COST_PILOT_FAILED: pilot_fit UNCERTAIN", units=2)
    _write(tmp_path, "REPORT.1789931301.json", stopped="REFUSED: no delivery", units=7)
    import os
    os.utime(tmp_path / "REPORT.1789931301.json", (2 << 30, 2 << 30))     # plainly the later attempt
    os.utime(tmp_path / "REPORT.json", (1 << 30, 1 << 30))
    path, doc, choice = P._governing_report(tmp_path, DESIGN)
    assert path.name == "REPORT.1789931301.json"
    assert choice["why"].startswith("no run in this root finished")
    assert choice["stopped"] == "REFUSED: no delivery"


def test_RP55_a_report_of_another_design_is_not_this_runs_report(tmp_path):
    _write(tmp_path, "REPORT.json", stopped=None, units=16, design_sha="b" * 64)
    with pytest.raises(SystemExit, match="no REPORT"):
        P._governing_report(tmp_path, DESIGN)
