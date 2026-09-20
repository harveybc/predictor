"""RP55: the governance chronology is a comparison of INSTANTS, not of text.

The defect, found by closing the successor run. The delivery receipt is written by the acquisition
("2026-09-20T19:01:29Z") and the terminal receipt records when the work started as the runner's cost
record spells it ("2026-09-20T19:01:29+00:00"). Compared as strings, '+' sorts before 'Z', so the
SAME instant read as "the work started before its delivery" and ten units of a fully governed run
were demoted to HISTORICAL_UNGOVERNED — on a spelling, with every receipt in order.

The rules below fix the instant and vary only its spelling, and they keep the direction the rule
exists for: a delivery that really follows the work is still refused.
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


C = _load("df_e1_close")
DESIGN_SHA = "a" * 64
UNIT = "R1_s1"


def _root(tmp_path, *, delivered_at, started_at, accepted_at):
    root = tmp_path / "root"
    root.mkdir()
    (root / "DELIVERIES.json").write_text(json.dumps({
        "design_sha256": DESIGN_SHA, "lake": "public_panels", "resource": "panel.parquet",
        "units": {UNIT: {"campaign_key": "k", "campaign_sha256": "c" * 64, "delivery_id": "d" * 32,
                         "sha256": "b" * 64, "at": delivered_at, "host": "omega",
                         "code_identity": {"kind": "git_commit", "value": "f" * 40},
                         "cached": True, "verification_state": "VERIFIED_CACHE"}}}))
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps({
        "design_sha256": DESIGN_SHA,
        "units": {UNIT: {"campaign_sha256": "c" * 64, "terminal_sha256": "e" * 64, "status": "COMPLETED",
                         "accepted_at": accepted_at, "work_started_at": started_at,
                         "reconciliation": {"http": 200, "missing_units": [], "accounting_only": [],
                                            "lake_only": []}}}}))
    return root


@pytest.mark.parametrize("delivered,started", [
    ("2026-09-20T19:01:29Z", "2026-09-20T19:01:29+00:00"),      # the same instant, two spellings
    ("2026-09-20T19:01:29+00:00", "2026-09-20T19:01:29Z"),
    ("2026-09-20T19:01:29Z", "2026-09-20T19:01:30Z"),           # a second later, plainly after
])
def test_RP55_the_same_instant_spelled_two_ways_is_not_a_violation(tmp_path, delivered, started):
    root = _root(tmp_path, delivered_at=delivered, started_at=started, accepted_at="2026-09-20T19:03:27Z")
    state, facts = C._governance(root, UNIT, {"design_sha256": DESIGN_SHA})
    assert facts["problems"] == []
    assert state == C.GOVERNED


@pytest.mark.parametrize("case", ["delivery_after_the_work", "terminal_before_the_delivery",
                                  "terminal_before_the_work"])
def test_RP55_a_real_inversion_is_still_refused(tmp_path, case):
    stamps = {"delivered_at": "2026-09-20T19:01:29Z", "started_at": "2026-09-20T19:02:00+00:00",
              "accepted_at": "2026-09-20T19:03:27Z"}
    if case == "delivery_after_the_work":
        stamps["delivered_at"] = "2026-09-20T19:02:30Z"          # delivered after the work began
    elif case == "terminal_before_the_delivery":
        stamps.update(delivered_at="2026-09-20T19:04:00Z", started_at="2026-09-20T19:04:10Z")
    else:
        stamps["accepted_at"] = "2026-09-20T19:01:45+00:00"      # accepted before the work started
    root = _root(tmp_path, **stamps)
    state, facts = C._governance(root, UNIT, {"design_sha256": DESIGN_SHA})
    assert state == C.HISTORICAL and facts["problems"], facts
    assert "governed delivery and accepted terminal" in facts["reading"]


def test_RP55_an_unreadable_stamp_is_not_silently_ordered(tmp_path):
    """A stamp that cannot be read is not turned into an ordering: it is simply not a violation,
    and the receipt's own required fields keep the unit honest."""
    assert C._instant("not a time") is None and C._instant(None) is None
    assert not C._before("not a time", "2026-09-20T19:01:29Z")
    assert C._before("2026-09-20T19:01:28Z", "2026-09-20T19:01:29+00:00")
