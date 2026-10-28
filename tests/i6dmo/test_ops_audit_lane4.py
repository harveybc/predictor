"""Audit correction lane 4 regressions (AUD-DMO-05/06 operational parts).

PRE: these tests target behaviour absent at the audited commit `fdaae9d2` (see the lane-4 PRE run
recorded in OPS_RETURN.md). POST: they pass on the corrected code.
"""
import json
import math

from tests.i6dmo.ops_fakes import GIB, facts, fake_adapter, make_campaign, private_spec, run_worker, write_design

MEASURED_SHORT_PEAK = 2431946752  # bytes, the audited completed ASOF short fit (engineering cost)


def test_audit_hold_root_is_held_never_complete_and_hold_is_visible(tmp_path):
    from tools.i6dmo import status
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")  # one completed unit
    (root / "AUDIT_HOLD.json").write_text(json.dumps(dict(
        schema="i6dmo.audit_hold.v1", verdict="REVISE_BEFORE_CONTINUE",
        reason="sampling drift", publication_of_retained_cost_facts_allowed=True)))
    camp = status.compute(root, level="campaign")
    assert camp["state"] == "HELD"
    assert camp["hold"]["verdict"] == "REVISE_BEFORE_CONTINUE" and camp["hold"]["publication_continues"] is True


def test_fit_eta_never_extrapolates_a_short_fit_to_the_long_family(tmp_path):
    from tools.i6dmo import status
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")  # short.fit COMPLETE; long.fit remains
    eta = status.compute(root, level="campaign")["eta"]
    assert eta["fit"]["kind"] == "UNMEASURED" and "long" in eta["fit"]["cause"]
    assert eta["fit"]["families"]["short"]["kind"] == "DONE"  # no short fit remains
    assert eta["fit"]["families"]["short"]["basis_n"] == 1
    assert eta["fit"]["families"]["long"]["kind"] == "UNMEASURED"
    costs = eta["measured_completed_costs"]["short.fit"]
    assert costs["fit_seconds"] == 0.5 and costs["cgroup_peak_bytes"] == 1 << 30


def test_launcher_measured_class_peak_caps_a_fresh_campaign(tmp_path):
    spec = private_spec(tmp_path)
    spec["launcher"]["measured_class_peaks"] = {"short.fit": MEASURED_SHORT_PEAK}
    (tmp_path / "private_spec.json").write_text(json.dumps(spec))
    root = make_campaign(tmp_path)
    out = run_worker(root, "preferred", tmp_path / "c.txt")
    assert out["result"] == "COMPLETE"
    unit_spec = json.loads((root / out["attempt_dir"] / "UNIT_SPEC.json").read_text())
    assert unit_spec["cap_bytes"] == math.ceil(1.25 * MEASURED_SHORT_PEAK)  # 3039933440, not the 7936 MiB default
    # the long family has no measured class: it keeps the declared initial cap
    from tools.i6dmo import campaign
    campaign.tick(root)
    run_worker(root, "coordinator", tmp_path / "c.txt")
    campaign.tick(root)
    out = run_worker(root, "preferred", tmp_path / "c.txt")
    assert out["unit_id"] == "long.fit"
    unit_spec = json.loads((root / out["attempt_dir"] / "UNIT_SPEC.json").read_text())
    assert unit_spec["cap_bytes"] == spec["launcher"]["initial_cap_bytes"]


def test_measured_class_cap_is_never_below_a_refused_reservation(tmp_path):
    from tests.i6dmo.ops_fakes import facts_provider
    from tools.i6dmo import queue
    spec = private_spec(tmp_path)
    spec["launcher"]["measured_class_peaks"] = {"short.fit": MEASURED_SHORT_PEAK}
    (tmp_path / "private_spec.json").write_text(json.dumps(spec))
    root = make_campaign(tmp_path)
    low = facts_provider(mem_available=4 * GIB)
    refused = run_worker(root, "preferred", tmp_path / "c.txt", provider=low)
    assert refused["result"] == "WAIT_RESOURCE"
    q = queue.Queue(root / "state" / "queue.sqlite3")
    assert q.min_cap("short.fit") == math.ceil(1.25 * MEASURED_SHORT_PEAK)
    q.close()
    out = run_worker(root, "preferred", tmp_path / "c.txt")  # healthy facts: cap unchanged, not trimmed
    unit_spec = json.loads((root / out["attempt_dir"] / "UNIT_SPEC.json").read_text())
    assert unit_spec["cap_bytes"] == math.ceil(1.25 * MEASURED_SHORT_PEAK)


def test_preflight_disk_check_blocks_a_small_filesystem_per_role(tmp_path):
    from tools.i6dmo import preflight
    spec = private_spec(tmp_path)
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    def probe(role, files):
        f = facts(disk_free=(16 if role == "coordinator" else 500) * GIB)
        f["legacy_active_units"] = []
        f["strategy_pin"] = dict(state="PASS", identity={})
        return f
    doc = preflight.preflight(write_design(tmp_path), tmp_path / "spec.json", adapter=fake_adapter(), probe=probe,
                              data_pins={"short_bundle": {"relpath": "b", "sha256": "0" * 64}},
                              out=tmp_path / "PRECHECK.json")
    assert "disk:coordinator" in doc["launch_blockers"]
    disk = {c["check"]: c for c in doc["checks"] if c["check"].startswith("disk:")}
    assert disk["disk:coordinator"]["state"] == "FAIL" and disk["disk:preferred"]["state"] == "PASS"
    assert disk["disk:coordinator"]["detail"]["disk_free_bytes"] == 16 * GIB
