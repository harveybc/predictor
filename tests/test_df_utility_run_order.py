"""N3: governance comes before the work. The campaign is registered before any child, before_run
precedes each child, a refusal at registration starts nothing, instants and costs are the
children's. Observed through callbacks on the real entry point with a stub governance and a
stub child runner (no services, no processes)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("df_utility_run")
H = _load("df_utility_harness")
GR = _load("governed_run")


class StubGov:
    def __init__(self, refuse_registration=False, units=None):
        self.refuse = refuse_registration
        self.calls = []
        self.terminals = {}
        self.units = dict(units or {})

    def submit_campaign(self, body):
        self.calls.append(("submit_campaign", body["campaign_key"]))
        if self.refuse:
            return 403, {"error": "refused by the stub"}
        sha = "c" * 60 + body["campaign_key"][-4:]
        self.units = getattr(self, "units", {})
        self.units[sha] = list(body["units"])
        return 201, {"campaign_sha256": sha}

    def reconcile_campaign(self, sha):
        """As data-gov: a unit is missing until its terminal is stored."""
        self.calls.append(("reconcile", sha))
        missing = [u for u in self.units.get(sha, []) if (sha, u) not in self.terminals]
        return 200, {"missing_units": missing, "accounting_only": [], "lake_only": []}

    def report_terminal(self, sha, unit_id, terminal):
        self.calls.append(("terminal", unit_id))
        self.terminals[(sha, unit_id)] = terminal
        return 201, {"terminal_sha256": "t" * 64}


class StubOutbox:
    def __init__(self, gov):
        self.gov = gov
        self.pending = []

    def put(self, envelope):
        self.pending.append(envelope)

    def flush(self, sender):
        for env in list(self.pending):
            sender(env)
            self.pending.remove(env)
        return {"pending": 0}


def stub_isolated(children):
    def run(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        marker = Path(attempt_dir) / "outcome.json"
        if marker.is_file():                                   # resumed: recorded, not re-run
            recorded = json.loads(marker.read_text())
            return {**recorded, "resumed": True}
        children.append(job["kind"])
        cost = {"cpu_seconds": 1.5, "wall_seconds": 2.0, "peak_rss_bytes": 1,
                "cgroup_memory_peak": None, "started_at": "2026-09-17T10:00:00Z",
                "ended_at": "2026-09-17T10:00:02Z"}
        def record(summary):
            Path(attempt_dir).mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(summary))
            return summary
        if job["kind"] == "calibrate":
            rec = {"schema": "df_utility_calibration.v1", "generator": "white_null", "null": True,
                   "plan": job["plan"], "n_sims": 2, "scored": 2, "failed": 0, "advances": 0,
                   "false_advance_rate": 0.0, "upper_bound": 0.5, "bound_confidence": 0.5,
                   "alpha_adjusted": 0.05 / 4, "seed": 1, "n": 40,
                   "operator": {"kind": job["operator"], "spec_sha256": "s" * 64, "params": {}},
                   "protocol_base_sha256": "b" * 64, "family": [], "margin": 0.0, "n_blocks": 4,
                   "window": 4, "target": "return", "model": "ridge",
                   "per_sim": [{"index": 0}, {"index": 1}], "per_sim_sha256": H.sha_obj([{"index": 0}, {"index": 1}]),
                   "harness_sha256": "h" * 64, "cost": {"cpu_seconds": 1.0}}
            return record({"outcome": "COMPLETED", "reason": "", "cost": cost, "score": rec, "output_sha256": "o" * 64})
        if job["kind"] == "mechanics":
            cells = {"schema": "d3_mechanics_cells.v1", "verified": True, "freeze_sha256": "f",
                     "design_sha256": "d", "cells": [{"unit": job["unit"], "variable": "v0",
                                                      "operator": k, "verdict": "MECHANICALLY_ACCEPTED",
                                                      "spec_sha256": "s" * 64} for k in job["operators"]]}
            Path(attempt_dir).mkdir(parents=True, exist_ok=True)
            (Path(attempt_dir) / "cells.json").write_text(json.dumps(cells))
            return record({"outcome": "COMPLETED", "reason": "", "cost": cost, "score": cells, "output_sha256": "m" * 64})
        return record({"outcome": H.INCONCLUSIVE_UNCALIBRATED, "reason": "", "cost": cost,
                       "score": {"schema": H.CONTRAST_SCHEMA, "outcome": H.INCONCLUSIVE_UNCALIBRATED,
                                 "delta_mean": 0.25, "delta_lower": 0.1, "delta_se": 0.05, "blocks_used": 4,
                                 "loss_name": "mae"}, "output_sha256": "x" * 64})
    return run


def cfg_for(tmp_path):
    return {"root": str(tmp_path / "run"), "run_id": "n3", "code_identity": {"kind": "git_commit", "value": "0" * 40},
            "units": [{"unit": "fab", "variable": "v0", "values": [float(i % 7) for i in range(60)]}],
            "operators": ["delta_run_length"], "purpose": "UTILITY_HARNESS_REHEARSAL",
            "eligibility_state": "FABRICATED_REHEARSAL", "exposure": "DEVELOPMENT_REHEARSAL_NO_RESERVE",
            "slow_control": False,
            "plan": {"generator": "white_null", "n_sims": 2, "n": 60, "bound_confidence": 0.5},
            "protocol": {"target": "return", "horizon": 1, "model": "ridge", "window": 4, "n_blocks": 4,
                         "margin": 0.0, "seed": 7, "min_rows_per_block": 5},
            "budgets": {"task_memory_bytes": 1 << 20, "wall_seconds": 5.0, "cpu_seconds": 5,
                        "calibration_wall_seconds": 5.0, "calibration_cpu_seconds": 5,
                        "mechanics_wall_seconds": 5.0, "mechanics_cpu_seconds": 5,
                        "slow_control": {"slow_seconds": 1.0, "wall_seconds": 1.0}}}


def test_N3_a_registration_refusal_starts_no_child(tmp_path):
    gov = StubGov(refuse_registration=True)
    trace, children = [], []
    with pytest.raises(SystemExit, match="no child was started"):
        R.run_rehearsal(cfg_for(tmp_path), gov, lambda e, **f: trace.append((e, f)),
                        GR=GR, outbox=StubOutbox(gov), isolated=stub_isolated(children))
    assert children == [] and not [e for e, _ in trace if e == "child"]
    assert [e for e, _ in trace] == ["freeze-pre", "register"]


def test_N3_registration_precedes_every_child_and_before_run_precedes_each(tmp_path):
    gov = StubGov()
    trace, children = [], []
    receipt, outcomes, frozen, pre, sha = R.run_rehearsal(
        cfg_for(tmp_path), gov, lambda e, **f: trace.append((e, f)), GR=GR,
        outbox=StubOutbox(gov), isolated=stub_isolated(children))
    events = [e for e, _ in trace]
    first_child = events.index("child")
    assert events[:first_child] == ["freeze-pre", "register", "before_run"]
    assert events.index("register") < first_child
    second_register = [i for i, e in enumerate(events) if e == "register"][1]
    contrast_child = [i for i, (e, f) in enumerate(trace) if e == "child" and f["kind"] == "contrast"][0]
    assert second_register < contrast_child
    for i, (e, f) in enumerate(trace):
        if e == "child":
            assert trace[i - 1][0] in ("before_run", "child-done", "freeze") or f["kind"] == "mechanics"
    assert children == ["calibrate", "mechanics", "contrast"]
    # instants and costs are the children's, not the reporter's
    t = gov.terminals[(sha, "fab__v0__delta_run_length__transformed")]
    assert t["started_at"] == "2026-09-17T10:00:00Z" and t["costs"]["cpu_seconds"] == 1.5
    assert [m["metric"] for m in t["metrics"]][:2] == ["utility.delta_mean", "utility.delta_lower"]
    assert receipt["reconciliation"]["contrasts"]["missing_units"] == []
    assert (tmp_path / "run" / "FREEZE.pre.json").is_file() and (tmp_path / "run" / "FREEZE.json").is_file()


def test_N3_a_resumed_run_registers_again_and_rebuilds_identical_terminals(tmp_path):
    gov = StubGov()
    children = []
    R.run_rehearsal(cfg_for(tmp_path), gov, lambda e, **f: None, GR=GR, outbox=StubOutbox(gov),
                    isolated=stub_isolated(children))
    first = dict(gov.terminals)
    # the resumed run reuses the persisted registrations: the second governance knows the same
    # campaigns and already holds their terminals (as data-gov would)
    gov2 = StubGov(units=gov.units)
    gov2.terminals = dict(first)
    R.run_rehearsal(cfg_for(tmp_path), gov2, lambda e, **f: None, GR=GR, outbox=StubOutbox(gov2),
                    isolated=stub_isolated(children))
    assert not [c for c in gov2.calls if c[0] == "submit_campaign"]   # nothing re-registered
    assert first == gov2.terminals                     # the same bytes, no duplicate identity


def test_N3_a_root_frozen_under_another_code_identity_is_not_resumed(tmp_path):
    gov = StubGov()
    children = []
    R.run_rehearsal(cfg_for(tmp_path), gov, lambda e, **f: None, GR=GR, outbox=StubOutbox(gov),
                    isolated=stub_isolated(children))
    other = dict(cfg_for(tmp_path), code_identity={"kind": "git_commit", "value": "1" * 40})
    with pytest.raises(SystemExit, match="another code identity"):
        R.run_rehearsal(other, StubGov(), lambda e, **f: None, GR=GR, outbox=StubOutbox(gov),
                        isolated=stub_isolated(children))
    assert (tmp_path / "run" / "CAMPAIGNS.json").is_file()
    # asked to, it resumes and records both identities
    trace = []
    gov2 = StubGov(units=gov.units)
    gov2.terminals = dict(gov.terminals)
    receipt, *_ = R.run_rehearsal(dict(other, resume_under_new_code=True), gov2,
                                  lambda e, **f: trace.append(e), GR=GR, outbox=StubOutbox(gov2),
                                  isolated=stub_isolated(children))
    assert receipt["resumed_under_new_code"] is True and "resume-under-new-code" in trace
    assert receipt["code_identity_frozen"]["value"] != receipt["code_identity_now"]["value"]


def test_N4_contrast_ids_of_bank_units_with_the_separator_inside_are_never_re_parsed(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg["units"] = [{"unit": "bumps__white__snr10__none__n2048__v1__seed11", "variable": "v0",
                     "values": [float(i % 7) for i in range(60)]}]
    gov = StubGov()
    receipt, outcomes, *_ = R.run_rehearsal(cfg, gov, lambda e, **f: None, GR=GR,
                                            outbox=StubOutbox(gov), isolated=stub_isolated([]))
    cid = next(iter(outcomes))
    assert cid.startswith("bumps__white__snr10__none__n2048__v1__seed11__v0__delta_run_length")
    assert outcomes[cid]["operator"] == "delta_run_length"


def test_N3_terminal_instants_are_the_childrens_in_data_govs_form():
    """utilreh-v6: data-gov refused every terminal as 'invalid started_at' because the runner
    records +00:00 while the service takes Z."""
    t = R._terminal(status="COMPLETED", reason=None, cost={}, metrics=[], tags={},
                    started="2026-09-17T15:05:50+00:00", finished="2026-09-17T15:07:35+00:00")
    assert t["started_at"] == "2026-09-17T15:05:50Z" and t["finished_at"] == "2026-09-17T15:07:35Z"
    assert R._z("2026-09-17T15:05:50Z") == "2026-09-17T15:05:50Z"
