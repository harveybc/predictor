#!/usr/bin/env python3
"""Musashi's probe, frozen: the OLD gate accepted these, the NEW gate refuses them BY NAME.

`docs/audits/qrm_scope_probe_20260929.py` (dictamen of request 23b2efa3, 2026-09-29) built three
record shapes and drove `supervise` over a launcher double.  This file replays the same three
shapes against BOTH sides of the repair in one run:

  * the instrument as it stood at c1033dc6, loaded from git -- so the defect is demonstrated
    rather than asserted from memory;
  * the instrument as it stands now.

It therefore is the "red before, green after" evidence itself, and it stays in the repository as
the standing proof: if anyone loosens the gate again, this file fails.

The three shapes and what each one proved:

  stale_foreign_record   a record of ANOTHER cell, clock 1, no observed scope, no lease.
                         OLD: usable_for_costing=true.  A false POSITIVE.
  negative_peak_record   status MEASURED, bytes -1.
                         OLD: usable_for_costing=true.  A domain that was never checked.
  production_nested_record  the REAL producer shape, `host_ram` nested under `cell_scope`.
                         OLD: UNKNOWN and usable_for_costing=false WHILE CONTAINING THE PEAK.
                         A false NEGATIVE at a schema boundary, and the only one of the three
                         whose repaired behaviour is ACCEPTANCE.

No launcher, no service, no GPU, no kernel measurement: process and observer doubles isolate the
gate, exactly as the auditor's probe did.  Nothing here simulates having measured anything.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
AUDITED_COMMIT = "c1033dc6"

GOOD = {"cell_id": "requested", "stage": "TRAIN", "recorded_at": 1,
        "scope": {"inode": 123, "cgroup": "old.scope"},
        "host_ram": {"cgroup_peak": {"status": "MEASURED", "bytes": 123456}},
        "usable_for_costing": True}
CASES = {
    "production_nested_record": {"cell_scope": GOOD},
    "stale_foreign_record": dict(GOOD, cell_id="another-cell"),
    "negative_peak_record": dict(GOOD, host_ram={"cgroup_peak": {"status": "MEASURED",
                                                                 "bytes": -1}}),
}


def _load_current():
    spec = importlib.util.spec_from_file_location("df_cell_scope_now",
                                                  TOOLS / "df_cell_scope.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["df_cell_scope_now"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_audited():
    """The instrument as the auditor read it, executed from the git object and never written to
    the working tree."""
    try:
        source = subprocess.check_output(
            ["git", "-C", str(HERE.parent), "show", f"{AUDITED_COMMIT}:tools/df_cell_scope.py"],
            text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"the audited commit {AUDITED_COMMIT} is not reachable here: {e}")
    mod = types.ModuleType("df_cell_scope_audited")
    mod.__file__ = str(TOOLS / "df_cell_scope.py")
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    return mod


class _Watcher:
    """The auditor's observer double: it sees NOTHING, which is the ordinary state for a child
    shorter than one observation interval -- and the state the old gate treated as harmless."""

    identity = None
    last_peak = None
    samples = 0

    def __init__(self, *a):
        self.stop = types.SimpleNamespace(set=lambda: None)

    def start(self):
        pass

    def join(self, **kw):
        pass


class _Child:
    def __init__(self, *a, **kw):
        pass

    def wait(self):
        return 0

    def poll(self):
        return 0


def _drive(module, record: dict) -> dict:
    """Drive `supervise` over the doubles, with the record already lying at record_path."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        path = root / "cell.json"
        path.write_text(json.dumps(record))
        with patch.object(module, "_ScopeWatcher", _Watcher), \
             patch.object(module, "_slice_cgroup", return_value=None), \
             patch.object(module, "_find_lease", return_value=(None, "not observed")), \
             patch.object(module.subprocess, "Popen", _Child):
            return module.supervise(cell_id="requested", argv=["unused"], launcher="/bin/true",
                                    cap_bytes=1048576, wall_seconds=5,
                                    supervisor_dir=root / "supervisor", log_path=root / "log",
                                    record_path=path, stage="TRAIN")


@pytest.fixture(scope="module")
def sides():
    return {"audited": _load_audited(), "now": _load_current()}


def test_the_audited_gate_accepted_a_record_belonging_to_another_cell(sides):
    """The defect, demonstrated from the git object: F1's first counterexample was ACCEPTED."""
    old = _drive(sides["audited"], CASES["stale_foreign_record"])
    assert old["usable_for_costing"] is True
    assert old["lease_confirmed"] is False                  # no reservation was ever confirmed
    assert old["host_ram"]["cgroup_peak"]["bytes"] == 123456


def test_the_repaired_gate_refuses_a_record_belonging_to_another_cell(sides):
    new = _drive(sides["now"], CASES["stale_foreign_record"])
    assert new["usable_for_costing"] is False
    assert new["refused_by"], new
    assert new["host_ram"]["cgroup_peak"]["status"] == "UNKNOWN"
    assert new["host_ram"]["cgroup_peak"]["bytes"] is None


def test_the_audited_gate_accepted_a_peak_of_minus_one(sides):
    old = _drive(sides["audited"], CASES["negative_peak_record"])
    assert old["usable_for_costing"] is True
    assert old["host_ram"]["cgroup_peak"]["bytes"] == -1


def test_the_repaired_gate_refuses_a_peak_of_minus_one(sides):
    new = _drive(sides["now"], CASES["negative_peak_record"])
    assert new["usable_for_costing"] is False
    assert new["host_ram"]["cgroup_peak"]["bytes"] is None
    assert new["refused_by"]


def test_the_audited_consumer_could_not_read_the_real_producers_document(sides):
    """F2, demonstrated: the production shape yielded UNKNOWN although it contained the peak."""
    old = _drive(sides["audited"], CASES["production_nested_record"])
    assert old["usable_for_costing"] is False
    assert old["host_ram"]["cgroup_peak"]["status"] == "UNKNOWN"
    # the peak was physically present in the document the gate had just read
    assert CASES["production_nested_record"]["cell_scope"][
        "host_ram"]["cgroup_peak"]["bytes"] == 123456


def test_the_repaired_consumer_reads_a_nested_record_only_through_the_declared_envelope(sides):
    """The repair of F2 is a DECLARED envelope, so the auditor's undeclared nesting is still
    refused -- and now it is refused by name instead of silently producing UNKNOWN.  The
    accepted-and-costable case for the real producer is exercised against a complete v2 record in
    tests/test_df_cell_scope_fresh_attempt.py, which the producer itself writes."""
    now = sides["now"]
    record, why = now.extract_record(CASES["production_nested_record"])
    assert record is None and why.startswith("ENVELOPE_MISSING")
    new = _drive(now, CASES["production_nested_record"])
    assert new["usable_for_costing"] is False
    assert new["refused_by"] == ["ENVELOPE_MISSING"]
    # and the same nesting, DECLARED, is found -- shape is agreed by version, not guessed
    declared = now.embed_record({"schema": "df_e1_block_cell.v1"},
                                dict(GOOD, schema=now.RECORD_SCHEMA))
    found, why2 = now.extract_record(declared)
    assert why2 is None and found["host_ram"]["cgroup_peak"]["bytes"] == 123456


def test_none_of_the_three_counterexamples_survives_the_repaired_gate(sides):
    """The summary assertion, in the auditor's own terms."""
    verdicts = {name: _drive(sides["now"], rec)["usable_for_costing"]
                for name, rec in CASES.items()}
    assert verdicts == {"production_nested_record": False, "stale_foreign_record": False,
                        "negative_peak_record": False}
    audited = {name: _drive(sides["audited"], rec)["usable_for_costing"]
               for name, rec in CASES.items()}
    assert audited == {"production_nested_record": False,   # the false negative
                       "stale_foreign_record": True,        # the false positives
                       "negative_peak_record": True}
