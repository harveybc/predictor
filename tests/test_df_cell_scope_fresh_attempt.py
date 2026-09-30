#!/usr/bin/env python3
"""QRM01 F1/F2: the auditor's counterexamples, frozen as the standing proof.

Musashi's dictamen of request 23b2efa3 (docs/audits/work_plan/, 2026-09-29) proved two high
findings against the shared runner with an independent probe.  His three cases are the first three
tests in this file and they are the specification:

  1. `production_nested_record` -- the REAL producer's shape.  `df_e1_block` wrote `host_ram`
     nested inside `cell_scope`; `supervise` read `host_ram` at the ROOT.  So the genuine document
     yielded UNKNOWN and `usable_for_costing=false` WHILE CONTAINING THE PEAK.  F2 is a false
     negative at a schema boundary, and the test asserts the real shape is now costable.
  2. `stale_foreign_record` -- a record of ANOTHER cell, clock 1, no observed scope, no lease, was
     ACCEPTED as this cell's cost.  F1 is a false positive, and the test asserts each missing thing
     refuses BY NAME.
  3. `negative_peak_record` -- a peak of MINUS ONE with status MEASURED was ACCEPTED.  A typed
     field with a checked domain is what makes that impossible.

Before the repair each of these three failed against `tools/df_cell_scope.py` at c1033dc6; the
auditor's own reproduction, `docs/audits/qrm_scope_probe_20260929.py`, is retained beside this file
as the record of that state.

The rule the dictamen states, carried into the code and asserted here: ABSENCE IS NOT COINCIDENCE.
A missing identity, clock, inode, limit, peak or reservation refuses.  Nothing passes by default,
and in particular the supervisor NOT having observed the scope -- the ordinary case for a child
shorter than one observation interval, which is exactly where the old gate fell open -- is never
read as agreement.

One inference the dictamen retires and this file does not use: two different peaks do not by
themselves prove two distinct scopes, because one scope can show different maxima at two instants.
Every assertion here is about identity, membership, scope lifetime or reservation instead.

Bounded fixtures only.  The cgroup tree is a directory of ordinary files, the admission store is a
tmp directory, and no test pressures host memory or provokes an OOM kill.  The one test that needs
a real transient scope is named `real_launcher` and skips when systemd --user cannot make one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
GIB = 1024 ** 3
MIB = 1024 ** 2
SLICE_REL = "user.slice/crispdm-batch.slice"


def _load(name: str, where: Path = TOOLS):
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load("df_cell_scope")
ADM = _load("crispdm_admission")


# ---- bounded fixtures --------------------------------------------------------------------------

@pytest.fixture()
def world(tmp_path, monkeypatch):
    """A simulated cgroup tree and a real (but empty, tmp) admission store.

    Nothing here charges a byte of real memory: `memory.peak` and `memory.max` are text files and
    the leases are JSON written by this fixture, through the admission module's own dataclass.
    """
    root = tmp_path / "cgroup"
    scope_rel = f"{SLICE_REL}/crispdm-requested-1-1.scope"
    d = root / scope_rel
    d.mkdir(parents=True)
    (d / "memory.peak").write_text("123456\n")
    (d / "memory.max").write_text(f"{6 * GIB}\n")
    (d / "cgroup.procs").write_text("")
    monkeypatch.setenv("CRISPDM_CGROUP_ROOT", str(root))
    monkeypatch.setenv("CRISPDM_CELL_SCOPE_SLICE_CGROUP", SLICE_REL)
    monkeypatch.setenv("CRISPDM_ADMISSION_DIR", str(tmp_path / "admission"))
    monkeypatch.delenv(CS.ATTEMPT_ENV, raising=False)
    return {"root": root, "scope_rel": scope_rel, "scope_dir": d, "tmp": tmp_path,
            "inode": d.stat().st_ino, "limit": 6 * GIB}


def put_lease(tmp_path: Path, *, cgroup: str, cap_bytes: int, unit: str | None = None,
              lease_id: str = "lease-bounded-fixture", retired: bool = False) -> str:
    """A reservation in the store, written by the admission module's own Lease dataclass.

    `retired=True` writes the RETAINED body instead of a live lease: that is the state a SHORT
    child leaves behind, and the path the gate must still be able to verify.
    """
    store = ADM.Store(tmp_path / "admission")
    store.prepare()
    now = time.time()
    lease = ADM.Lease(lease_id=lease_id, name="cell", label="cell:requested", cap_bytes=cap_bytes,
                      wall_seconds=600, slice_name="crispdm-batch.slice", created_at=now,
                      expires_at=now + 600, armed=True, pid=os.getpid(),
                      pid_starttime=ADM.SystemResources().pid_starttime(os.getpid()),
                      cgroup=cgroup, unit=unit or cgroup.rsplit("/", 1)[-1])
    if retired:
        body = {"schema": "crispdm.retained_lease.v1", "reclaim_cause": "CHILD_ENDED",
                "retired_at_epoch": now, "lease": asdict(lease)}
        (store.retained / f"{lease_id}.json").write_text(json.dumps(body))
    else:
        store.write(lease)
    return lease_id


def fresh_record(world, *, attempt: str, cell_id: str = "requested",
                 stage: str = "TRAIN", peak: int = 123456, lease_id: str = "lease-bounded-fixture",
                 cap: int = 6 * GIB, **over) -> dict:
    """A record as the instrument itself writes one: full identity, typed peak, limit, lease."""
    rec = {
        "schema": CS.RECORD_SCHEMA, "cell_id": cell_id, "attempt_id": attempt, "stage": stage,
        "recorded_at": time.time(), "boot_id": CS._boot_id(),
        "scope": {"cgroup": world["scope_rel"], "unit": "crispdm-requested-1-1.scope",
                  "is_scope": True, "inode": world["inode"], "path_exists": True,
                  "parent": SLICE_REL, "boot_id": CS._boot_id(), "read_at": time.time()},
        "kernel_limit": {"bytes": cap, "status": CS.MEASURED, "source": "memory.max",
                         "unlimited": False},
        "declared_cap_bytes": cap,
        "reservation": {"schema": "df_cell_scope_reservation.v1", "confirmed": True,
                        "lease_id": lease_id, "source": "LIVE_ADMISSION_STORE",
                        "cgroup": world["scope_rel"], "cap_bytes": cap,
                        "declared_cap_bytes": cap, "cap_agrees_with_the_declaration": True},
        "host_ram": {"cgroup_peak": {"bytes": peak, "status": CS.MEASURED,
                                     "basis": CS.CGROUP_BASIS, "cgroup": world["scope_rel"]},
                     "process_rss_peak": {"bytes": 1, "status": CS.MEASURED,
                                          "basis": CS.RSS_BASIS},
                     "comparability": CS.COMPARABILITY},
        "usable_for_costing": True,
    }
    rec.update(over)
    return rec


def verdict(world, document, *, attempt: str, cell_id: str = "requested", stage: str = "TRAIN",
            cap: int = 6 * GIB, observed=None, window: float = 60.0) -> dict:
    now = time.time()
    return CS.verify_fresh_attempt(document, cell_id=cell_id, attempt_id=attempt, stage=stage,
                                   declared_cap_bytes=cap, started_at=now - window,
                                   finished_at=now + 1, observed_scope=observed)


# ---- 1. the auditor's `production_nested_record`: F2, a false NEGATIVE --------------------------

def test_musashi_production_nested_record_is_read_by_its_declared_envelope_and_is_costable(world):
    """Counterexample 1.  `df_e1_block.py:820` nested the record under `cell_scope` and
    `df_cell_scope.py:604` read `host_ram` at the root, so the REAL producer document was refused
    while containing the peak.  The envelope now binds the two sides by schema version."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    document = CS.embed_record({"schema": "df_e1_block_cell.v1", "cell": {"cell_id": "requested"}},
                               fresh_record(world, attempt=attempt, lease_id=lease))
    # the producer's document is NOT a scope record and must not be mistaken for one...
    assert document["schema"] == "df_e1_block_cell.v1"
    assert "host_ram" not in document
    # ...and the consumer still finds the record, by the declaration and not by the name
    record, why = CS.extract_record(document)
    assert why is None and record["host_ram"]["cgroup_peak"]["bytes"] == 123456
    v = verdict(world, document, attempt=attempt)
    assert v["accepted"] is True, v["refusals"]
    assert v["refused_by"] == []
    assert v["checks"]["peak_bytes"] == 123456


def test_a_field_that_merely_shares_a_name_is_never_accepted(world):
    """The repair is explicit, by schema version.  A document with a root `host_ram` that looks
    right but declares no envelope is refused: replacing one silent mismatch with another -- a
    consumer that tries both places -- is what the order forbids."""
    attempt = CS.new_attempt_id()
    look_alike = {"cell_id": "requested",
                  "host_ram": {"cgroup_peak": {"bytes": 123456, "status": CS.MEASURED,
                                               "basis": CS.CGROUP_BASIS}}}
    record, why = CS.extract_record(look_alike)
    assert record is None and why.startswith("ENVELOPE_MISSING")
    assert verdict(world, look_alike, attempt=attempt)["refused_by"] == ["ENVELOPE_MISSING"]


def test_a_v1_record_is_refused_by_its_version_and_not_by_a_field_by_field_guess(world):
    """v1 carried no attempt identity, no reservation and no checked domains.  It is refused by its
    VERSION, so no reader has to decide which of its fields might still be trustworthy."""
    attempt = CS.new_attempt_id()
    v1 = {"schema": "df_cell_scope_record.v1", "cell_id": "requested", "stage": "TRAIN",
          "host_ram": {"cgroup_peak": {"bytes": 123456, "status": CS.MEASURED}}}
    assert verdict(world, v1, attempt=attempt)["refused_by"] == ["RECORD_SCHEMA_SUPERSEDED"]
    wrapped = dict(v1)
    env = CS.embed_record({}, v1)
    assert env["cell_scope_envelope"]["record_schema"] == "df_cell_scope_record.v1"
    assert CS.extract_record(env)[1].startswith("RECORD_SCHEMA_SUPERSEDED")
    assert wrapped["schema"] == "df_cell_scope_record.v1"


# ---- 2. the auditor's `stale_foreign_record`: F1, a false POSITIVE ------------------------------

def test_musashi_stale_foreign_record_is_refused_by_name(world):
    """Counterexample 2, exactly as the probe built it: a record of ANOTHER cell, clock 1, no
    observed scope and no lease.  The old gate returned usable_for_costing=true.  Every one of the
    five things it never asked for now refuses by its own name."""
    attempt = CS.new_attempt_id()
    stale = {"cell_id": "another-cell", "stage": "TRAIN", "recorded_at": 1,
             "scope": {"inode": 123, "cgroup": "old.scope"},
             "host_ram": {"cgroup_peak": {"status": CS.MEASURED, "bytes": 123456}},
             "usable_for_costing": True}
    v = verdict(world, stale, attempt=attempt, observed=None)
    assert v["accepted"] is False
    # the record does not even reach the field checks: it declares no supported schema
    assert v["refused_by"] == ["ENVELOPE_MISSING"]
    # and with the envelope supplied, so that the FIELDS are what is judged, each gap is named
    v2 = verdict(world, CS.embed_record({}, dict(stale, schema=CS.RECORD_SCHEMA)),
                 attempt=attempt, observed=None)
    assert v2["accepted"] is False
    for code in ("CELL_IDENTITY_MISMATCH",          # it belongs to another cell
                 "ATTEMPT_IDENTITY_MISSING",        # it predates this attempt
                 "BOOT_IDENTITY_MISSING",
                 "NOT_A_SCOPE",                     # "old.scope" was never shown to be a scope
                 "CLOCK_OUTSIDE_THIS_ATTEMPT",      # clock 1 is not a moment of this run
                 "KERNEL_LIMIT_MISSING",            # no limit was in force
                 "RESERVATION_MISSING",             # no confirmed reservation
                 "PEAK_BASIS_IS_NOT_THE_TREE_PEAK"):
        assert code in v2["refused_by"], (code, v2["refused_by"])
    assert "another-cell" in [r["detail"] for r in v2["refusals"] if
                              r["code"] == "CELL_IDENTITY_MISMATCH"][0]


def test_a_record_of_the_right_cell_from_a_previous_attempt_is_refused(world):
    """Freshness, isolated.  Same cell, same stage, same scope, valid lease, in-window clock --
    and the token of an EARLIER attempt.  This is the case a pre-existing file at record_path
    actually presents, and it is the one the old gate could not see at all."""
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    old, new = CS.new_attempt_id(), CS.new_attempt_id()
    document = CS.embed_record({}, fresh_record(world, attempt=old, lease_id=lease))
    v = verdict(world, document, attempt=new)
    assert v["accepted"] is False and v["refused_by"] == ["ATTEMPT_IDENTITY_MISMATCH"]
    assert verdict(world, document, attempt=old)["accepted"] is True


def test_a_record_with_no_reservation_is_refused_even_when_the_peak_is_real(world):
    """A cell with no reservation was never admitted, so its charge is not a governed cost.  The
    lease is absent from the store here; the record's claim is not taken on trust."""
    attempt = CS.new_attempt_id()
    document = CS.embed_record({}, fresh_record(world, attempt=attempt, lease_id="no-such-lease"))
    v = verdict(world, document, attempt=attempt)
    assert v["refused_by"] == ["RESERVATION_NOT_VERIFIABLE_IN_THE_STORE"]
    assert "no-such-lease" in v["refusals"][0]["detail"]


def test_a_reservation_bound_to_another_scope_is_refused(world):
    """A lease that exists but belongs to a different cgroup does not admit THIS scope."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=f"{SLICE_REL}/crispdm-other-9-9.scope",
                      cap_bytes=world["limit"], lease_id="lease-elsewhere")
    document = CS.embed_record({}, fresh_record(world, attempt=attempt, lease_id=lease))
    v = verdict(world, document, attempt=attempt)
    assert v["refused_by"] == ["RESERVATION_NOT_VERIFIABLE_IN_THE_STORE"]
    assert "crispdm-other-9-9.scope" in v["refusals"][0]["detail"]


def test_a_reservation_smaller_than_the_declared_cap_is_refused_and_never_accepted_instead(world):
    """Never shrink a declared cap to evade a refusal, and never reduce a reservation to pass: a
    record measured under a lowered reservation is refused rather than costed at the smaller size."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=2 * GIB,
                      lease_id="lease-lowered")
    rec = fresh_record(world, attempt=attempt, lease_id=lease)
    rec["reservation"]["cap_bytes"] = 2 * GIB
    v = verdict(world, CS.embed_record({}, rec), attempt=attempt, cap=world["limit"])
    assert "RESERVATION_CAP_IS_NOT_THE_DECLARED_CAP" in v["refused_by"]
    assert v["accepted"] is False


def test_a_kernel_limit_below_the_declared_cap_is_refused_by_name(world):
    """The kernel limit in force is part of the contract.  A scope limited under the declaration
    is not the declared attempt, and the smaller number is not quietly adopted."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec = fresh_record(world, attempt=attempt, lease_id=lease)
    rec["kernel_limit"]["bytes"] = 2 * GIB
    v = verdict(world, CS.embed_record({}, rec), attempt=attempt, cap=world["limit"])
    assert "KERNEL_LIMIT_BELOW_THE_DECLARED_CAP" in v["refused_by"]


def test_an_unlimited_scope_is_refused_because_it_encloses_nothing(world):
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec = fresh_record(world, attempt=attempt, lease_id=lease)
    rec["kernel_limit"] = {"bytes": None, "status": CS.UNKNOWN, "unlimited": True,
                           "why": "memory.max is 'max'"}
    v = verdict(world, CS.embed_record({}, rec), attempt=attempt)
    assert "KERNEL_LIMIT_NOT_MEASURED" in v["refused_by"]


def test_a_missing_child_inode_refuses_instead_of_being_skipped(world):
    """The old comparison was CONDITIONAL: a child with no inode did not prevent acceptance.  A
    missing inode is now a refusal of its own, whether or not the supervisor saw anything."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec = fresh_record(world, attempt=attempt, lease_id=lease)
    rec["scope"]["inode"] = None
    v = verdict(world, CS.embed_record({}, rec), attempt=attempt, observed=None)
    assert v["refused_by"] == ["SCOPE_INODE_MISSING"]
    assert "REFUSES" in v["refusals"][0]["detail"]


def test_a_scope_the_supervisor_saw_differently_is_refused_and_absence_is_not_agreement(world):
    """When an external observation exists it must AGREE.  When it does not exist, the record is
    not thereby accepted: the check is recorded as absent with its reason, and acceptance rests on
    identity, clock, limit and reservation -- the path a short child also has."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    document = CS.embed_record({}, fresh_record(world, attempt=attempt, lease_id=lease))
    disagreeing = verdict(world, document, attempt=attempt,
                          observed={"inode": world["inode"] + 1, "cgroup": "other"})
    assert disagreeing["refused_by"] == ["SCOPE_IS_NOT_THE_SCOPE_THE_SUPERVISOR_OBSERVED"]
    agreeing = verdict(world, document, attempt=attempt, observed={"inode": world["inode"]})
    assert agreeing["accepted"] and agreeing["checks"][
        "scope_identity_agrees_with_the_supervisor"] is True
    absent = verdict(world, document, attempt=attempt, observed=None)
    assert absent["checks"]["scope_identity_agrees_with_the_supervisor"] is None
    assert "NOT read as agreement" in absent["checks"]["why_no_external_observation"]
    assert "ABSENCE_IS_NOT_COINCIDENCE" in absent["absence_is_not_coincidence"]


def test_a_short_child_whose_scope_is_already_gone_is_still_costable_through_the_retained_lease(world):
    """The path that ALSO works for short-lived children -- the case where the old gate fell open.

    The scope is removed and the supervisor observed nothing, so there is no live lease and no
    external sighting.  What remains is the retained lease BODY the store keeps after a load ends,
    and it binds the lease to this cgroup and this cap.  That, with the minted token, the boot and
    the clock, is enough -- and it is evidence from the launcher's store, not the child's word.
    """
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"],
                      lease_id="lease-short-child", retired=True)
    document = CS.embed_record({}, fresh_record(world, attempt=attempt, lease_id=lease))
    # the scope directory is gone, as it is for any child whose transient unit has been released
    for f in sorted(world["scope_dir"].iterdir()):
        f.unlink()
    world["scope_dir"].rmdir()
    v = verdict(world, document, attempt=attempt, observed=None)
    assert v["accepted"] is True, v["refusals"]
    assert v["checks"]["reservation_verification"]["source"] == "RETAINED_LEASE_BODY"


# ---- 3. the auditor's `negative_peak_record`: F1, a domain that was never checked ---------------

def test_musashi_negative_peak_record_is_refused_by_name(world):
    """Counterexample 3, exactly as the probe built it: status MEASURED, bytes -1.  The old gate
    accepted it.  A high-watermark cannot be negative."""
    attempt = CS.new_attempt_id()
    negative = {"cell_id": "requested", "stage": "TRAIN", "recorded_at": 1,
                "scope": {"inode": 123, "cgroup": "old.scope"},
                "host_ram": {"cgroup_peak": {"status": CS.MEASURED, "bytes": -1}},
                "usable_for_costing": True}
    assert verdict(world, negative, attempt=attempt)["refused_by"] == ["ENVELOPE_MISSING"]
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec = fresh_record(world, attempt=attempt, lease_id=lease, peak=-1)
    v = verdict(world, CS.embed_record({}, rec), attempt=attempt)
    assert v["accepted"] is False and v["refused_by"] == ["PEAK_BYTES_NEGATIVE"]
    assert "cannot be negative" in v["refusals"][0]["detail"]


@pytest.mark.parametrize("value,code", [
    (0, "PEAK_BYTES_ZERO"),
    (-123456, "PEAK_BYTES_NEGATIVE"),
    ("123456", "PEAK_BYTES_OUT_OF_DOMAIN"),
    (123456.5, "PEAK_BYTES_OUT_OF_DOMAIN"),
    (True, "PEAK_BYTES_OUT_OF_DOMAIN"),
    (None, "PEAK_BYTES_MISSING"),
    (7 * GIB, "PEAK_ABOVE_THE_KERNEL_LIMIT"),
])
def test_the_peak_field_is_typed_and_its_domain_is_checked(world, value, code):
    """Typed fields with checked domains, so a peak that is not a positive count of bytes inside
    the enforced limit cannot be accepted whatever its status field says."""
    attempt = CS.new_attempt_id()
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec = fresh_record(world, attempt=attempt, lease_id=lease, peak=value)
    assert code in verdict(world, CS.embed_record({}, rec), attempt=attempt)["refused_by"]


def test_an_absent_record_refuses_and_says_that_absence_is_not_coincidence(world):
    v = verdict(world, None, attempt=CS.new_attempt_id())
    assert v["accepted"] is False and v["refused_by"] == ["RECORD_ABSENT"]
    assert "ABSENCE_IS_NOT_COINCIDENCE" in v["absence_is_not_coincidence"]


# ---- the supervisor, end to end over a stub launcher -------------------------------------------

def _record_writing_launcher(tmp_path: Path, cell: str, *, record_path: Path, scope_rel: str,
                             peak: int, limit: int, hold: float = 0.0, exit_code: int = 0,
                             write_record: bool = True) -> Path:
    """A bounded stand-in for the deployed launcher: it makes the scope directory, writes the
    child's record through THIS MODULE's own producer helper, holds for `hold` seconds and then
    removes the scope.  `hold=0` is the short child the old gate could not certify."""
    script = tmp_path / f"launcher-{cell}"
    script.write_text(f"""#!/usr/bin/env bash
while getopts "m:t:n:W:L:E:P:S:qh" o; do :; done
shift $((OPTIND - 1)); [ "${{1:-}}" = "--" ] && shift
d="$CRISPDM_CGROUP_ROOT/{scope_rel}"
mkdir -p "$d"; echo {peak} > "$d/memory.peak"; echo {limit} > "$d/memory.max"; : > "$d/cgroup.procs"
{'python3 ' + str(tmp_path / f'writer-{cell}.py') if write_record else 'true'}
sleep {hold}
rm -rf "$d"
exit {exit_code}
""")
    script.chmod(0o755)
    (tmp_path / f"writer-{cell}.py").write_text(f"""
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("df_cell_scope", {str(TOOLS / 'df_cell_scope.py')!r})
CS = importlib.util.module_from_spec(spec); sys.modules["df_cell_scope"] = CS
spec.loader.exec_module(CS)
rec = CS.cell_scope_record({cell!r}, stage=CS.os.environ["CRISPDM_CELL_SCOPE_STAGE"],
                           rel={scope_rel!r})
doc = CS.embed_record({{"schema": "df_e1_block_cell.v1"}}, rec)
Path({str(record_path)!r}).parent.mkdir(parents=True, exist_ok=True)
Path({str(record_path)!r}).write_text(json.dumps(doc))
""")
    return script


def test_the_supervisor_accepts_the_producers_document_over_the_whole_path(world, tmp_path):
    """producer -> cell.json -> supervisor -> scope record, across the F2 boundary, with the F1
    contract in force.  The launcher is a stub; the producer, the envelope, the record and the gate
    are the real ones, and the child here is SHORT: it holds no scope for the watcher to see."""
    rec_path = tmp_path / "attempts" / "requested" / "cell.json"
    put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"],
              lease_id="lease-e2e")
    launcher = _record_writing_launcher(tmp_path, "requested", record_path=rec_path,
                                       scope_rel=world["scope_rel"], peak=258_584_576,
                                       limit=world["limit"])
    r = CS.supervise(cell_id="requested", argv=[sys.executable, "-c", "pass"],
                     cap_bytes=world["limit"], wall_seconds=60,
                     supervisor_dir=tmp_path / "SUP", log_path=tmp_path / "e2e.log",
                     launcher=launcher, record_path=rec_path, stage="CELL_TRAIN_AND_SCORE")
    assert r["termination"]["status"] == "COMPLETED"
    assert r["fresh_attempt_contract"]["accepted"] is True, r["refused_by"]
    assert r["usable_for_costing"] is True
    assert r["host_ram"]["cgroup_peak"]["bytes"] == 258_584_576
    assert r["host_ram"]["cgroup_peak"]["accepted_by"] == CS.CONTRACT_SCHEMA
    assert r["attempt_id"] and json.loads(
        rec_path.read_text())["cell_scope"]["attempt_id"] == r["attempt_id"]
    held = json.loads((tmp_path / "SUP" / "requested.json").read_text())
    assert held["usable_for_costing"] is True and held["refused_by"] == []


def test_a_record_left_by_an_earlier_attempt_is_refused_by_the_supervisor(world, tmp_path):
    """The whole of F1 in one run: a complete, valid, correctly nested record of the SAME cell is
    already lying at record_path, and this attempt's launcher writes nothing.  The old gate read
    that file and returned usable_for_costing=true."""
    rec_path = tmp_path / "attempts" / "requested" / "cell.json"
    rec_path.parent.mkdir(parents=True)
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec_path.write_text(json.dumps(CS.embed_record(
        {}, fresh_record(world, attempt=CS.new_attempt_id(), lease_id=lease))))
    launcher = _record_writing_launcher(tmp_path, "requested", record_path=rec_path,
                                       scope_rel=world["scope_rel"], peak=1,
                                       limit=world["limit"], write_record=False)
    r = CS.supervise(cell_id="requested", argv=[sys.executable, "-c", "pass"],
                     cap_bytes=world["limit"], wall_seconds=60, supervisor_dir=tmp_path / "SUP",
                     log_path=tmp_path / "stale.log", launcher=launcher, record_path=rec_path,
                     stage="CELL_TRAIN_AND_SCORE")
    assert r["termination"]["status"] == "COMPLETED"        # the child "succeeded"...
    assert r["usable_for_costing"] is False                 # ...and still costs nothing
    assert "ATTEMPT_IDENTITY_MISMATCH" in r["refused_by"]
    assert r["host_ram"]["cgroup_peak"]["status"] == CS.UNKNOWN
    assert "did NOT satisfy the fresh-attempt contract" in r["host_ram"]["cgroup_peak"]["why"]


def test_a_foreign_cells_record_is_refused_by_the_supervisor(world, tmp_path):
    rec_path = tmp_path / "attempts" / "requested" / "cell.json"
    rec_path.parent.mkdir(parents=True)
    lease = put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    rec_path.write_text(json.dumps(CS.embed_record({}, fresh_record(
        world, attempt=CS.new_attempt_id(), cell_id="another-cell", lease_id=lease))))
    launcher = _record_writing_launcher(tmp_path, "requested", record_path=rec_path,
                                       scope_rel=world["scope_rel"], peak=1,
                                       limit=world["limit"], write_record=False)
    r = CS.supervise(cell_id="requested", argv=[sys.executable, "-c", "pass"],
                     cap_bytes=world["limit"], wall_seconds=60, supervisor_dir=tmp_path / "SUP",
                     log_path=tmp_path / "foreign.log", launcher=launcher, record_path=rec_path,
                     stage="CELL_TRAIN_AND_SCORE")
    assert r["usable_for_costing"] is False
    assert "CELL_IDENTITY_MISMATCH" in r["refused_by"]
    assert "ATTEMPT_IDENTITY_MISMATCH" in r["refused_by"]


def test_the_supervisor_mints_a_fresh_token_per_attempt_and_hands_it_to_the_child(world, tmp_path):
    """The token is minted by the supervisor, is different every attempt, and reaches the child in
    the environment -- which is why a file already on disk cannot carry it."""
    seen = []
    rec_path = tmp_path / "attempts" / "requested" / "cell.json"
    launcher = _record_writing_launcher(tmp_path, "requested", record_path=rec_path,
                                        scope_rel=world["scope_rel"], peak=99,
                                        limit=world["limit"])
    put_lease(world["tmp"], cgroup=world["scope_rel"], cap_bytes=world["limit"])
    for i in range(2):
        r = CS.supervise(cell_id="requested", argv=[sys.executable, "-c", "pass"],
                         cap_bytes=world["limit"], wall_seconds=60,
                         supervisor_dir=tmp_path / "SUP", log_path=tmp_path / f"t{i}.log",
                         launcher=launcher, record_path=rec_path, stage="CELL_TRAIN_AND_SCORE")
        seen.append(r["attempt_id"])
        assert json.loads(rec_path.read_text())["cell_scope"]["attempt_id"] == r["attempt_id"]
    assert seen[0] != seen[1] and all(seen)


# ---- the mechanism we both got wrong: memory.peak resets PER DESCRIPTOR -------------------------

def test_a_write_text_then_read_text_does_not_establish_the_reset_experiment(world, monkeypatch):
    """The kernel documents the reset of `memory.peak` as applying to subsequent reads THROUGH THE
    SAME open file descriptor (https://docs.kernel.org/admin-guide/cgroup-v2.html).  A `write_text`
    followed by a `read_text` opens two different descriptors, so that sequence establishes nothing
    -- an earlier report of ours claimed it did.  This test pins the descriptor DISCIPLINE, which
    is what code can be held to: the reset and the read happen on ONE `os.open`.  It deliberately
    does not claim to reproduce kernel semantics on a regular file, which cannot show them.
    """
    opened = []
    real_open = os.open

    def counting_open(path, flags, *a, **k):
        if str(path).endswith("memory.peak"):
            opened.append((str(path), flags))
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(CS.os, "open", counting_open)
    out = CS.stage_peak_one_descriptor(world["scope_rel"], reset_after=True)
    assert out["status"] == CS.MEASURED and out["bytes"] == 123456
    assert len(opened) == 1, opened                      # one descriptor for the read AND the reset
    assert opened[0][1] & os.O_RDWR
    assert out["descriptor"] == "SAME_OPEN_FILE_DESCRIPTOR" and out["reset_on_the_same_fd"] is True
    assert out["watermark"] == "SINCE_THE_LAST_RESET_ON_THIS_FD"
    assert "SAME open file descriptor" in out["why_one_descriptor"]
    src = (TOOLS / "df_cell_scope.py").read_text()
    assert "memory.peak\").write_text" not in src and '"memory.peak").write_text' not in src


def test_the_lifetime_watermark_is_kept_separately_and_is_never_reset(world):
    """Whatever is done per stage, the lifetime high-watermark is retained under its own name."""
    life = CS.lifetime_peak(world["scope_rel"])
    assert life["bytes"] == 123456 and life["status"] == CS.MEASURED
    assert life["reset"] is False and life["watermark"] == "LIFETIME_OF_THE_SCOPE"
    rec = CS.cell_scope_record("requested", stage="TRAIN", rel=world["scope_rel"])
    assert rec["host_ram"]["cgroup_lifetime_peak"]["watermark"] == "LIFETIME_OF_THE_SCOPE"
    assert rec["host_ram"]["cgroup_lifetime_peak"]["reset"] is False
    assert rec["host_ram"]["cgroup_peak"]["bytes"] == 123456


def test_two_different_peaks_are_not_offered_as_proof_of_two_distinct_scopes():
    """The inference the dictamen retires.  The instrument's own words must not claim that two
    different charges demonstrate distinct scopes: identity, membership, lifetime and reservation
    are the proof, and those are what the record carries."""
    src = (TOOLS / "df_cell_scope.py").read_text()
    assert "a fresh scope is a fresh directory" in src          # inode identity, not a magnitude
    for field in ("attempt_id", "reservation", "kernel_limit", "boot_id"):
        assert field in CS.cell_scope_record("c", stage="TRAIN", rel=None)
    v = CS.verify_fresh_attempt(None, cell_id="c", attempt_id="a", stage="TRAIN",
                                declared_cap_bytes=1, started_at=0.0, finished_at=1.0)
    assert v["accepted"] is False


# ---- the producer really does declare the envelope ---------------------------------------------

def test_the_block_producer_declares_the_envelope_and_never_assigns_the_record_positionally():
    """Source-level, against the exact lines the dictamen cites: `df_e1_block` must embed through
    the declared envelope, and the consumer must not reach for a root field by name."""
    prod = (TOOLS / "df_e1_block.py").read_text()
    assert 'CSC.embed_record(record,' in prod
    assert 'record["cell_scope"] = CSC.cell_scope_record' not in prod
    cons = (TOOLS / "df_cell_scope.py").read_text()
    assert 'child_record.get("host_ram"' not in cons
    assert "extract_record(document)" in cons
    assert 'CSC.STAGE_ENV' in prod                     # producer honours the declared stage
    assert '"refused_by": sup["refused_by"]' in prod or 'sup["refused_by"]' in prod
