"""R3: the instrument validation is frozen before any outcome, governed, budget-enforced through
the isolated runner (calibrations included), judged with complete denominators and intervals,
and classified PASS / DOES_NOT_SEPARATE / INCONCLUSIVE / BUDGET_LIMITED."""
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


INS = _load("df_utility_instrument_run")
H = _load("df_utility_harness")
GR = _load("governed_run")
ORDER = _load("test_df_utility_run_order", HERE)
DEVT = _load("test_df_utility_dev_run", HERE)


def _design():
    return INS.sealed_design(n=400, replicates=3, seed0=200)


def _supportable_record(job, cal_supports=True):
    """A record the CURRENT verifier accepts and that supports THIS job's contract: built from
    the real computation key, with fabricated (not simulated) per-simulation rows."""
    ops = _load("df_d3_operators")
    contract = _load("df_d3_contract")
    p = H.Protocol(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in job["protocol"].items()
                      if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")})
    op = ops.build(job["operator"])
    plan = job["plan"]
    key = H.computation_key(p, op, plan, branch_a=job["branch_a"], branch_b=job["branch_b"], seed=job["seed"])
    sims = [{"index": i, "seed": 1000 + i, "outcome": "DOES_NOT_ADVANCE", "delta_mean": -0.01, "delta_lower": -0.05}
            for i in range(plan["n_sims"])]
    advances = 0
    if not cal_supports:
        sims[0].update(outcome="ADVANCES", delta_mean=0.6, delta_lower=0.5)
        advances = 1
    n = plan["n_sims"]
    return {"schema": "df_utility_calibration.v4", "generator": plan["generator"], "null": True,
            "computation": key, "computation_sha256": H.computation_sha256(key), "numeric_dependencies": key["numeric_dependencies"],
            "branch_a": job["branch_a"], "branch_b": job["branch_b"], "widths": key["widths"], "rows_policy": H.ROWS_POLICY,
            "plan": dict(plan), "n_sims": n, "scored": n, "failed": 0, "advances": advances, "false_advance_rate": advances / n,
            "upper_bound": H.clopper_pearson_upper(advances, n, plan["bound_confidence"]), "bound_confidence": plan["bound_confidence"],
            "alpha_adjusted": p.alpha_adjusted, "seed": job["seed"], "n": plan["n"],
            "operator": {"kind": op.KIND, "spec_sha256": contract.spec_sha256(op.describe()), "params": dict(op.params)},
            "protocol_base_sha256": p.base_sha256(), "family": list(p.family), "margin": p.margin, "n_blocks": p.n_blocks,
            "window": p.window, "target": p.target, "model": p.model, "per_sim": sims, "per_sim_sha256": H.sha_obj(sims),
            "harness_sha256": key["harness_sha256"], "cost": {"cpu_seconds": 1.5}}


def _stub(children, outcome_of, cal_supports=True, contrast_cost=1.0):
    def run(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        attempt_dir = Path(attempt_dir)
        marker = attempt_dir / "outcome.json"
        if marker.is_file():
            return {**json.loads(marker.read_text()), "resumed": True}
        children.append(job["kind"])
        cost = {"cpu_seconds": 1.5 if job["kind"] == "calibrate" else contrast_cost, "wall_seconds": 1.0, "peak_rss_bytes": 1,
                "cgroup_memory_peak": None, "started_at": "2026-09-18T10:00:00Z", "ended_at": "2026-09-18T10:00:01Z"}
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if job["kind"] == "calibrate":
            rec = _supportable_record(job, cal_supports)
            body = json.dumps(rec, sort_keys=True).encode()
            (attempt_dir / "calibration.json").write_bytes(body)
            summary = {"outcome": "COMPLETED", "reason": "", "cost": cost, "score": rec, "output_sha256": "o" * 64}
            marker.write_text(json.dumps(summary))
            return summary
        outcome, delta, why = outcome_of(job)
        score = {"schema": H.CONTRAST_SCHEMA, "outcome": outcome, "loss_name": "mae", "blocks_used": 4,
                 "delta_mean": delta, "delta_lower": None if delta is None else delta - 0.05, "delta_se": 0.01, "why": why}
        summary = {"outcome": outcome, "reason": "", "cost": cost, "score": score, "output_sha256": "x" * 64}
        marker.write_text(json.dumps(summary))
        return summary
    return run


def _run(tmp_path, outcome_of, cap=10 ** 9, already=0.0, cal_supports=True, contrast_seconds=0.5, contrast_cost=1.0):
    gov = ORDER.StubGov()
    trace, children = [], []
    report = INS.run_instrument(_design(), root=tmp_path / "ins", run_id="ins", gov=gov, trace=lambda e, **f: trace.append((e, f)),
                                GR=GR, outbox=ORDER.StubOutbox(gov), OB=DEVT.StubOB(), CE=DEVT.StubCE(),
                                code_identity={"kind": "git_commit", "value": "0" * 40}, budgets=DEVT._budgets(),
                                cap_seconds=cap, already_spent=already, cost_pilot_sims=2, contrast_seconds=contrast_seconds,
                                isolated=_stub(children, outcome_of, cal_supports, contrast_cost))
    return report, gov, trace, children


def _ideal(job):
    name = job["control"]
    if name in ("positive_H_T", "positive_H_A"):
        return H.ADVANCES, 0.1, None
    if name == "info_loss":
        return H.DOES_NOT_ADVANCE, -0.1, None
    if name == "future_leak":
        return H.REFUSED, None, "representation is not causal: its prefix outputs changed when the series was cut"
    return H.DOES_NOT_ADVANCE, 0.001, None


def test_R3_the_design_is_sealed_first_and_derives_simulations_from_the_campaigns_alpha_and_confidence():
    d = _design()
    assert d["multiplicity"]["alpha_adjusted"] == 0.025 and d["calibration_plan"]["bound_confidence"] == 0.95
    assert d["calibration_plan"]["n_sims"] == H.sims_required_for_zero(0.025, 0.95) == 119
    assert d["seeds"]["list"] == [200, 201, 202] and min(d["seeds"]["list"]) > 105          # disjoint from Q3
    assert d["controls"]["positive_H_T"]["criterion"]["min_count"] == 10 or d["replicates"] != 12
    assert all("why" in c["criterion"] for c in d["controls"].values())
    assert d["controls"]["positive_H_A"]["pair"] == ["raw_wide", "augmented"]
    assert d["design_sha256"] == INS.campaign.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})


def test_R3_a_governed_run_freezes_then_pilots_then_calibrates_then_controls_and_judges_with_intervals(tmp_path):
    report, gov, trace, children = _run(tmp_path, _ideal)
    events = [e for e, _ in trace]
    assert events[0] == "design-frozen" and events.index("projection") < events.index("before_run", events.index("projection"))
    assert (tmp_path / "ins" / "DESIGN.json").is_file() and (tmp_path / "ins" / "FREEZE.json").is_file()
    assert children.count("contrast") == 5 * 3
    keys = [k for k, _ in gov.calls if k == "submit_campaign"]
    assert len(keys) == 3                                             # cost pilot, calibration, controls
    assert sorted(k for k in report["calibration"] if k != "campaign") == ["mad_extremes_trailing__H_A", "mad_extremes_trailing__H_T"]
    assert report["calibration"]["mad_extremes_trailing__H_T"]["supports"]["decision"] is True
    assert report["reconciliation"]["contrasts"]["missing_units"] == []
    pc = report["controls"]["per_control"]
    assert pc["positive_H_T"]["hits"] == 3 and pc["positive_H_T"]["ci95"][0] > 0.29 and pc["positive_H_T"]["ci95"][1] == 1.0
    assert pc["future_leak"]["status"] == "MET" and pc["null_contrast"]["advances"] == 0
    assert report["instrument_outcome"] == "PASS" and report["envelope"]["sha"] == "e" * 64
    leak_job = json.loads((tmp_path / "ins" / "attempts" / "future_leak__r00" / "outcome.json").read_text())
    assert leak_job["outcome"] == H.REFUSED
    assert report["projection"]["n_controls"] == 15 and report["spent_cpu_seconds"] > 0


def test_R3_positives_below_criterion_do_not_separate_and_a_failed_calibration_is_inconclusive(tmp_path):
    def weak(job):
        if job["control"] == "positive_H_T" and job["replicate"] > 0:
            return H.DOES_NOT_ADVANCE, 0.02, None
        return _ideal(job)
    report, *_ = _run(tmp_path, weak)
    assert report["controls"]["per_control"]["positive_H_T"]["status"] == "NOT_MET"
    assert report["instrument_outcome"] == "DOES_NOT_SEPARATE"
    report, *_ = _run(tmp_path / "b", _ideal, cal_supports=False)
    assert report["calibration"]["mad_extremes_trailing__H_T"]["supports"]["decision"] is False
    assert all(v["status"] in ("INCONCLUSIVE_CALIBRATION", "MET") for v in report["controls"]["per_control"].values())
    assert report["instrument_outcome"] == "INCONCLUSIVE"


def test_R3_the_budget_is_checked_before_calibration_and_during_controls(tmp_path):
    report, gov, trace, children = _run(tmp_path, _ideal, cap=1.0)
    assert report["instrument_outcome"] == "BUDGET_LIMITED" and report["stopped"]
    assert (tmp_path / "ins" / "DESIGN.json").is_file() and "contrast" not in children
    # projection fits, but the already-spent CPU of the order plus the run exhausts the ceiling mid-way
    # projection ≈ 3 + 2·(0.75·119) + 15·0.5 ≈ 189 s fits under 200 s; the contrasts then cost 50 s each
    report, gov, trace, children = _run(tmp_path / "b", _ideal, cap=200.0, already=0.0, contrast_seconds=0.5, contrast_cost=50.0)
    assert report["instrument_outcome"] == "BUDGET_LIMITED" and report["stopped"].startswith(INS.DEV.CAP_EXHAUSTED)
    assert report["controls"]["incomplete"]
    assert (tmp_path / "b" / "ins" / "attempts").is_dir()
