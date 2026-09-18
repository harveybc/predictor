"""P3: the development runner projects the campaign from a governed cost pilot, refuses to launch
beyond the aggregate CPU ceiling, runs families in the design's order with per-hypothesis
contracts, and keeps incomplete attempts when the ceiling is exhausted mid-way."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DEV = _load("df_utility_dev_run")
R = _load("df_utility_run")
H = _load("df_utility_harness")
GR = _load("governed_run")
ORDER = _load("test_df_utility_run_order", HERE)
DESIGN = _load("test_df_utility_next_design", HERE)


class StubOB:
    def __init__(self):
        self.emitted = []

    def emit(self, envelope, kind):
        self.emitted.append(envelope)
        return {"kind": kind, "sha": envelope["envelope_sha256"]}


class StubCE:
    @staticmethod
    def build_envelope(**kw):
        return {"envelope_sha256": "e" * 64, **kw}


def _budgets():
    return {"task_memory_bytes": 1 << 20, "wall_seconds": 5.0, "cpu_seconds": 5,
            "calibration_wall_seconds": 100.0, "calibration_cpu_seconds": 100,
            "mechanics_wall_seconds": 5.0, "mechanics_cpu_seconds": 5,
            "slow_control": {"slow_seconds": 0.0, "wall_seconds": 1.0}}


def _run(tmp_path, cap, isolated=None, contrast_seconds=0.5, cost_pilot_sims=2):
    design = DESIGN.design()
    gov = ORDER.StubGov()
    trace = []
    children = []
    report = DEV.run_development(
        design, root=tmp_path / "dev", run_id="dev", gov=gov, trace=lambda e, **f: trace.append((e, f)),
        GR=GR, outbox=ORDER.StubOutbox(gov), OB=StubOB(), CE=StubCE(),
        code_identity={"kind": "git_commit", "value": "0" * 40},
        load_values=lambda unit: [float(i % 7) for i in range(400)], budgets=_budgets(),
        cap_seconds=cap, cost_pilot_sims=cost_pilot_sims, contrast_seconds=contrast_seconds,
        isolated=isolated or ORDER.stub_isolated(children))
    return report, gov, trace, children


def test_P3_the_cost_pilot_is_governed_and_the_projection_counts_every_contract_and_contrast(tmp_path):
    report, gov, trace, children = _run(tmp_path, cap=10 ** 9)
    events = [e for e, _ in trace]
    assert events[:3] == ["freeze-pre", "register", "before_run"]           # the cost pilot first
    assert sorted(k for k in report["cost_pilot"] if k != "campaign") == sorted(
        f"{k}__{h}" for k in DESIGN.OPS for h in ("H_T", "H_A"))
    proj = report["projection"]
    assert proj["calibration_contracts"] == 24 and proj["contrasts"] == 24
    assert proj["per_sim_seconds"]["cusum_causal__H_A"] == pytest.approx(1.5 / 2)   # stub cost / sims
    expected = proj["cost_pilot_seconds"] + sum(c["seconds"] for c in proj["contracts"]) + 0.5 * 24
    assert proj["projected_cpu_seconds"] == pytest.approx(expected)
    assert report["stopped"] is None and len(report["families"]) == 4
    fam = report["families"]["bumps__s12"]
    assert fam["role"] == "selection" and fam["replica_of"] is None
    assert report["families"]["bumps__s13"]["replica_of"] == "bumps__s12"
    assert {o["hypothesis"] for o in fam["outcomes"].values()} == {"H_T", "H_A"}
    assert len(fam["outcomes"]) == 6 and fam["reconciliation"]["contrasts"]["missing_units"] == []
    assert (tmp_path / "dev" / "families" / "bumps__s12" / "FREEZE.json").is_file()
    assert (tmp_path / "dev" / "REPORT.json").is_file()
    # every family registered its own calibration units per contract
    keys = [k for k, _ in gov.calls if k == "submit_campaign"]
    assert len([c for c in gov.calls if c[0] == "submit_campaign"]) == 1 + 2 * 4


def test_P3_a_projection_beyond_the_ceiling_launches_nothing_and_writes_a_feasible_plan(tmp_path):
    report, gov, trace, children = _run(tmp_path, cap=20.0)
    assert report["stopped"] == "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED" and report["families"] == {}
    plan = json.loads((tmp_path / "dev" / "PLAN.json").read_text())
    assert plan["verdict"] == "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED" and plan["feasible_families_in_order"] == []
    assert children.count("calibrate") == 6 and "contrast" not in children


def test_P3_a_ceiling_exhausted_mid_way_stops_and_keeps_incomplete_attempts(tmp_path):
    children = []
    stub = ORDER.stub_isolated(children)

    def costly(job, *, attempt_dir, **kw):
        """the stub's recorded 1.5 s for calibrations; contrasts turn out far costlier than
        projected (the projection is not the measurement)"""
        out = stub(job, attempt_dir=attempt_dir, **kw)
        if job.get("kind") == "contrast" and not out.get("resumed"):
            out["cost"]["cpu_seconds"] = 1000.0
            marker = Path(attempt_dir) / "outcome.json"
            marker.write_text(json.dumps({**json.loads(marker.read_text()), "cost": out["cost"]}))
        return out
    # projection: pilot 9 s + 24 contracts × 358 × 0.75 s + 24 × 0.5 s ≈ 6465 s fits under 7000 s;
    # the first family's contrasts alone then spend 6000 s
    report, gov, trace, _ = _run(tmp_path, cap=7000.0, isolated=costly, contrast_seconds=0.5)
    assert report["projection"]["projected_cpu_seconds"] < 7000
    assert report["stopped"] and report["stopped"].startswith(DEV.CAP_EXHAUSTED)
    assert len(report["families"]["bumps__s12"]["outcomes"]) == 6
    second = report["families"]["bumps__s13"]
    assert "incomplete" in second
    assert (tmp_path / "dev" / "families" / "bumps__s13" / "attempts").is_dir()      # kept
    assert "sinusoid__s12" not in report["families"]
    assert report["spent_cpu_seconds"] == pytest.approx(DEV.spent_cpu(tmp_path / "dev"))
    assert report["spent_cpu_seconds"] > 6000
