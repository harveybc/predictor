#!/usr/bin/env python3
"""QRM01 F2, end to end on a real workload: producer -> cell.json -> supervisor -> scope record.

The dictamen of request 23b2efa3 says the isolated-scopes test never crossed the producer/consumer
boundary, and that the repair must be proved with a small real workload rather than with doubles.
So this file runs the REAL chain:

  * the DEPLOYED `crispdm-run` and its atomic admission module give the child a fresh transient
    scope inside `crispdm-batch.slice`, that scope's own `MemoryMax`, and its own reservation;
  * the child claims that scope through the real `require_fresh_exclusive_scope`, loads a synthetic
    panel through the real loader and TRAINS a real (tiny) Keras model through the real
    `df_e1_block.run_cell`, which writes the real `cell.json`;
  * the real `df_cell_scope.supervise` reads that document through the declared envelope and puts
    it through the whole fresh-attempt contract;
  * the peak is a real `memory.peak` of a real cgroup, read from inside the scope by the child.

What this does NOT cover, stated so nobody reads more into it: the governed delivery check at the
top of `df_e1_block.child` (covered by tests/test_df_e1_governed_route.py) and governance
reporting are not exercised here, the panel is SYNTHETIC and declared as such, and nothing here is
an experiment or a measurement of any model's quality.

Resources.  CPU only (`CUDA_VISIBLE_DEVICES=""`, as `run_units` sets), one child, one 4 GiB cap
declared once and never re-asked smaller, inside a slice whose own ceiling is untouched.  The test
skips rather than improvising when the launcher or a user systemd is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
GIB = 1024 ** 3
CAP_BYTES = 4 * GIB              # declared ONCE, at the size a Keras cell needs on this host
# the first cell of the sealed design; its id is read from the design, not guessed

sys.path.insert(0, str(HERE))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load("df_cell_scope")

pytestmark = pytest.mark.skipif(
    not CS.launcher_available(),
    reason="the deployed crispdm-run or a user systemd able to make a transient scope is absent")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A synthetic block, prepared through the same loader path the real preparation uses."""
    import test_df_e1_block as BT                                   # the block suite's fixtures

    tmp = tmp_path_factory.mktemp("qrm01-e2e")
    lo = BT.DAY - BT.H
    frame = BT._panel(lo + 35 * BT.DAY)
    source = BT._source(tmp, frame, lo)
    design = BT._design("DEV_MATCHED", source,
                        recipe={"max_updates": 6, "validate_every_updates": 3,
                                "patience_events": 3})
    root = tmp / "root"
    BT.K.prepare(design, root, frame=frame)
    (root / "DESIGN.json").write_text(json.dumps(design))
    unit = design["cells"][0]["cell_id"]
    cell = design["cells"][0]
    child = tmp / "e2e_child.py"
    child.write_text(f'''
"""The cell child: the same steps df_e1_block.child performs, minus the governed delivery."""
import importlib.util, json, os, sys
from pathlib import Path

TOOLS = {str(TOOLS)!r}


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(TOOLS) / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CS = load("df_cell_scope")
K = load("df_e1_block")
root = Path({str(root)!r})
unit = {unit!r}
parent = os.environ.get("CRISPDM_CELL_SCOPE_PARENT_CGROUP")
if parent is None:
    raise SystemExit("REFUSED CELL_NOT_LAUNCHED_BY_THE_SUPERVISOR")
claim = CS.require_fresh_exclusive_scope(os.environ["CRISPDM_CELL_SCOPE_CLAIMS"], unit,
                                         parent_cgroup=parent or None)
design = json.loads((root / "DESIGN.json").read_text())
data = K.load_data(root, design)
cell = next(c for c in design["cells"] if c["cell_id"] == unit)
K.run_cell(design, data, cell, root / "attempts" / unit, pilot=False, cell_scope=claim)
''')
    return {"root": root, "design": design, "cell": cell, "child": child,
            "tmp": tmp, "unit": unit}


@pytest.fixture(scope="module")
def supervised(prepared):
    """One real cell, launched through the deployed launcher, supervised by the real supervisor."""
    root, unit = prepared["root"], prepared["unit"]
    log = root / f"{unit}.log"
    log.touch(exist_ok=False)
    return CS.supervise(
        cell_id=unit, argv=[sys.executable, str(prepared["child"])],
        cap_bytes=CAP_BYTES, wall_seconds=600,
        supervisor_dir=root / "SUPERVISOR", log_path=log,
        claims_dir=root / "SCOPE_CLAIMS",
        record_path=root / "attempts" / unit / "cell.json",
        stage="CELL_TRAIN_AND_SCORE",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
             "OPENBLAS_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"})


def test_the_real_child_completes_in_a_scope_of_its_own(supervised, prepared):
    r = supervised
    if r["termination"]["status"] != "COMPLETED":
        pytest.fail(f"the child did not complete: {r['termination']}\n"
                    f"{(prepared['root'] / (prepared['unit'] + '.log')).read_text()[-4000:]}")
    assert r["termination"]["cap_asked_once_at_the_declared_size"] is True
    assert r["termination"]["cap_bytes_declared"] == CAP_BYTES
    assert (prepared["root"] / "attempts" / prepared["unit"] / "cell.json").is_file()


def test_the_producers_document_declares_the_envelope_and_the_consumer_reads_it(prepared,
                                                                                supervised):
    """The exact F2 boundary, on the real document this workload produced."""
    document = json.loads((prepared["root"] / "attempts" / prepared["unit"] / "cell.json").read_text())
    assert document["schema"] == "df_e1_block_cell.v1"          # the PRODUCER's own document
    assert "host_ram" not in document                           # the peak is not at the root...
    env = document["cell_scope_envelope"]
    assert env["schema"] == CS.ENVELOPE_SCHEMA
    assert env["record_schema"] == CS.RECORD_SCHEMA
    assert env["record_at"] == "cell_scope"                      # ...it is HERE, and it says so
    record, why = CS.extract_record(document)
    assert why is None and record["host_ram"]["cgroup_peak"]["status"] == CS.MEASURED


def test_the_whole_fresh_attempt_contract_accepts_this_attempt(supervised):
    """F1's contract, satisfied by a real run rather than by a fixture."""
    c = supervised["fresh_attempt_contract"]
    assert c["accepted"] is True, c["refusals"]
    assert c["schema"] == CS.CONTRACT_SCHEMA
    assert supervised["refused_by"] == []
    assert supervised["usable_for_costing"] is True
    assert supervised["why_not_usable"] is None


def test_the_attempt_token_was_minted_by_the_supervisor_and_echoed_by_the_child(prepared,
                                                                                supervised):
    document = json.loads((prepared["root"] / "attempts" / prepared["unit"] / "cell.json").read_text())
    token = supervised["attempt_id"]
    assert token and document["cell_scope"]["attempt_id"] == token
    assert supervised["fresh_attempt_contract"]["checks"]["attempt_id"] == token


def test_the_peak_is_a_real_kernel_watermark_of_this_cells_own_scope(supervised):
    peak = supervised["host_ram"]["cgroup_peak"]
    assert peak["status"] == CS.MEASURED
    assert isinstance(peak["bytes"], int) and 0 < peak["bytes"] <= CAP_BYTES
    assert peak["read_by"] == "THE_CHILD_INSIDE_ITS_OWN_SCOPE_BEFORE_THE_SCOPE_WAS_REMOVED"
    assert peak["accepted_by"] == CS.CONTRACT_SCHEMA
    assert peak["basis"] == CS.CGROUP_BASIS
    # host RAM, the resident set and device memory stay in separate fields and are never added
    record = supervised["fresh_attempt_contract"]["record"]
    assert record["host_ram"]["process_rss_peak"]["basis"] == CS.RSS_BASIS
    assert record["gpu"]["status"] == CS.UNKNOWN                 # no device was selected: ABSENT
    assert "not zero" in record["gpu"]["why"]


def test_the_kernel_limit_in_force_is_the_declared_cap(supervised):
    record = supervised["fresh_attempt_contract"]["record"]
    assert record["kernel_limit"]["status"] == CS.MEASURED
    assert abs(record["kernel_limit"]["bytes"] - CAP_BYTES) <= CS.LIMIT_PAGE_TOLERANCE_BYTES
    assert record["kernel_limit"]["unlimited"] is False
    assert supervised["fresh_attempt_contract"]["checks"]["kernel_limit_bytes"] > 0


def test_the_reservation_is_confirmed_in_the_launchers_own_store_and_re_verified(supervised):
    record = supervised["fresh_attempt_contract"]["record"]
    res = record["reservation"]
    assert res["confirmed"] is True and res["lease_id"]
    assert res["source"] in ("LIVE_ADMISSION_STORE", "RETAINED_LEASE_BODY")
    assert res["cap_agrees_with_the_declaration"] is True
    v = supervised["fresh_attempt_contract"]["checks"]["reservation_verification"]
    assert v["verified"] is True and v["lease_id"] == res["lease_id"]
    # the supervisor's own live sighting of the same reservation, independently of the child
    assert supervised["lease_confirmed"] is True


def test_the_scope_is_a_fresh_transient_scope_and_both_sides_name_the_same_one(supervised):
    record = supervised["fresh_attempt_contract"]["record"]
    scope = record["scope"]
    assert scope["is_scope"] is True and scope["unit"].endswith(".scope")
    assert scope["unit"].startswith("crispdm-")
    assert isinstance(scope["inode"], int) and scope["inode"] > 0
    assert scope["parent"].endswith("crispdm-batch.slice")
    # the supervisor identified the scope itself and the two identifications agree BY INODE.
    # Identity, membership, lifetime and reservation are the proof here; a difference between two
    # peaks would not be (dictamen 23b2efa3 retires that inference).
    assert supervised["scope"]["inode"] == scope["inode"]
    assert supervised["scope_identity_agrees_with_the_child"] is True


def test_the_retained_supervision_file_carries_the_verdict_and_its_reasons(prepared, supervised):
    held = json.loads((prepared["root"] / "SUPERVISOR" / f"{prepared['unit']}.json").read_text())
    assert held["schema"] == "df_cell_scope_supervision.v2"
    assert held["usable_for_costing"] is True and held["refused_by"] == []
    assert held["attempt_id"] == supervised["attempt_id"]
    assert held["declared_cap_bytes"] == CAP_BYTES
    assert "ABSENCE_IS_NOT_COINCIDENCE" in held["absence_is_not_coincidence"]
    assert held["fresh_attempt_contract"]["accepted"] is True


def test_a_sampled_observation_and_an_in_scope_read_are_not_ordered_in_either_direction(supervised):
    """A finding of this very run, recorded rather than smoothed over.

    The supervisor's sampled value came out ABOVE the child's in-scope read here -- 559,525,888 B
    against 557,654,016 B on the first real pass -- because `run_cell` reads its own memory.peak
    before the child has finished and the kernel watermark keeps rising afterwards.  The
    classification lane had seen the opposite, by 9.5x.  So the sampled series bounds only the
    scope's LIFETIME watermark, the comparison between the two is unsigned, and neither number may
    stand in for the other.  The retained lifetime watermark is what bounds both.
    """
    cmp = supervised["host_ram"].get("floor_against_in_scope_peak")
    if cmp is None:
        pytest.skip("the supervisor observed no sample on this pass, so there is no pair")
    assert cmp["direction"] in ("SAMPLED_BELOW_THE_IN_SCOPE_READ",
                                "SAMPLED_ABOVE_THE_IN_SCOPE_READ", "EQUAL")
    expected = ("SAMPLED_BELOW_THE_IN_SCOPE_READ" if cmp["floor_bytes"] < cmp["in_scope_peak_bytes"]
                else "SAMPLED_ABOVE_THE_IN_SCOPE_READ" if
                cmp["floor_bytes"] > cmp["in_scope_peak_bytes"] else "EQUAL")
    assert cmp["direction"] == expected
    assert "UNSIGNED" in cmp["reading"] and "not a correction factor" in cmp["reading"]
    # the one figure that is at least as large as both, retained under its own name
    life = supervised["fresh_attempt_contract"]["record"]["host_ram"]["cgroup_lifetime_peak"]
    assert life["reset"] is False and life["watermark"] == "LIFETIME_OF_THE_SCOPE"
    assert life["bytes"] >= cmp["in_scope_peak_bytes"]
    assert "lifetime" in cmp["what_bounds_both"].lower()


def test_a_second_attempt_would_not_be_certified_by_this_attempts_record(prepared, supervised):
    """The record this real workload left behind is itself refused for a LATER attempt: the file
    is complete, correct and nested exactly as the producer writes it, and it still does not
    certify a different attempt.  This is the defect F1 named, checked against real bytes."""
    document = json.loads((prepared["root"] / "attempts" / prepared["unit"] / "cell.json").read_text())
    later = CS.new_attempt_id()
    now = supervised["finished_at"]
    v = CS.verify_fresh_attempt(document, cell_id=prepared["unit"], attempt_id=later,
                                stage="CELL_TRAIN_AND_SCORE", declared_cap_bytes=CAP_BYTES,
                                started_at=now, finished_at=now + 1,
                                observed_scope=supervised["scope"])
    assert v["accepted"] is False
    assert "ATTEMPT_IDENTITY_MISMATCH" in v["refused_by"]
    assert "CLOCK_OUTSIDE_THIS_ATTEMPT" in v["refused_by"]
