"""S3: the governed adequacy runner freezes the design, pilots cost, projects the whole factorial,
refuses beyond the ceiling with the exact need, registers before any cell, runs every cell as an
isolated child under the aggregate ceiling, keeps incompletes, and never claims selection."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


D = _load("df_adequacy_design")
M = _load("df_adequacy_models")
RUN = _load("df_adequacy_run")
GR = _load("governed_run")
ORDER = _load("test_df_utility_run_order", HERE)
DEVT = _load("test_df_utility_dev_run", HERE)
ADQ = _load("test_df_adequacy", HERE)


def _design(tmp_path):
    bank = ADQ._fake_bank(tmp_path)
    (bank / D.UNITS[1]).mkdir()
    for f in ("clean_signal.npy", "observed_signal.npy", "UNIT.json"):
        (bank / D.UNITS[1] / f).write_bytes((bank / D.UNITS[0] / f).read_bytes())
    doc = D.build(bank)
    # a small factorial for the stub run: the same schema, sealed again
    doc.update(units=[D.UNITS[0]], contexts=[4, 8], train_lengths=[256], models=["ridge", "lstm"])
    doc["cells_total"] = len(D.cells(doc))
    doc["design_sha256"] = D.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc, bank


def _stub(children, cell_cost=1.0, fail=(), role_cost=None):
    def run(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        attempt_dir = Path(attempt_dir)
        marker = attempt_dir / "outcome.json"
        if marker.is_file():
            return {**json.loads(marker.read_text()), "resumed": True}
        children.append(job["cell_id"])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cost = {"cpu_seconds": (role_cost or {}).get(job.get("role"), cell_cost), "wall_seconds": 1.0, "peak_rss_bytes": 1, "cgroup_memory_peak": None,
                "started_at": "2026-09-18T12:00:00Z", "ended_at": "2026-09-18T12:00:01Z"}
        if job["cell_id"] in fail:
            summary = {"outcome": RUN.H.RESOURCE_EXCEEDED, "reason": "WALL", "cost": cost, "score": None, "output_sha256": None}
            marker.write_text(json.dumps(summary))
            return summary
        updates = 40 if job["model"] != "ridge" else 1
        rec = {"schema": M.CELL_SCHEMA, "cell_id": job["cell_id"], "diagnosis": {"class": M.FITTED, "why": "stub"},
               "losses": {"test": {"model": 0.1, "baseline": 0.2, "oracle": 0.05, "rows": 383}}, "skill_test": 0.5, "block_skill_test": [0.4, 0.5, 0.6, 0.5],
               "training": {"updates": updates}, "graph": {"receptive_field": job["window"]}, "consumed_span_over_P": (job["window"] - 1) / 40.0,
               "cost": {"cpu_seconds": cell_cost}}
        summary = {"outcome": M.FITTED, "reason": "", "cost": cost, "score": rec, "output_sha256": "o" * 64}
        marker.write_text(json.dumps(summary))
        return summary
    return run


def _run(tmp_path, cap, already=0.0, cell_cost=1.0, fail=(), role_cost=None):
    design, bank = _design(tmp_path)
    gov = ORDER.StubGov()
    trace, children = [], []
    report = RUN.run_adequacy(design, root=tmp_path / "adq", run_id="adq", bank=bank, gov=gov, trace=lambda e, **f: trace.append((e, f)),
                              GR=GR, outbox=ORDER.StubOutbox(gov), OB=DEVT.StubOB(), CE=DEVT.StubCE(),
                              code_identity={"kind": "git_commit", "value": "0" * 40},
                              budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5}, cap_seconds=cap,
                              already_spent=already, pilot_updates=20, isolated=_stub(children, cell_cost, fail, role_cost))
    return report, gov, trace, children, design


def test_S3_freeze_pilot_projection_register_then_cells_with_terminals_and_reconciliation(tmp_path):
    report, gov, trace, children, design = _run(tmp_path, cap=10 ** 9)
    events = [e for e, _ in trace]
    assert events[0] == "design-frozen" and (tmp_path / "adq" / "DESIGN.json").is_file()
    assert children[:2] == ["pilot__ridge", "pilot__lstm"]
    assert events.index("projection") < events.index("register", events.index("projection"))
    assert report["stopped"] is None and len(report["cells"]) == design["cells_total"] == 3 * 2 * 2
    assert report["projection"]["cells"] == 12 and report["projection"]["projected_cpu_seconds"] > 0
    assert report["reconciliation"]["missing_units"] == []
    assert len([c for c in gov.calls if c[0] == "submit_campaign"]) == 2
    t = gov.terminals[(report["campaign"]["campaign_sha256"], children[2])]
    assert t["tags"]["role"] == "CELL" and t["metrics"][0]["metric"] == "adequacy.mae_test" and t["costs"]["cpu_seconds"] == 1.0
    assert report["envelope"]["sha"] == "e" * 64 and report["spent_cpu_seconds"] == pytest.approx(14.0)


def test_S3_projection_beyond_the_ceiling_writes_the_exact_need_and_launches_nothing(tmp_path):
    report, gov, trace, children, design = _run(tmp_path, cap=3.0, cell_cost=1.0)
    assert report["stopped"] == "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED" and report["cells"] == {}
    plan = json.loads((tmp_path / "adq" / "PLAN.json").read_text())
    assert plan["exact_need_cpu_seconds"] > 3.0 and plan["projection"]["cells"] == 12
    assert children == ["pilot__ridge", "pilot__lstm"]


def test_S3_the_ceiling_exhausted_mid_way_keeps_incompletes_and_a_failed_cell_is_a_complete_failure(tmp_path):
    # the pilot projects ~150 s for the factorial; the cells then cost 100 s each: the ceiling of
    # 260 s is exhausted after two cells (2 pilot + 2 x 100 + the next cell's projected need > 260)
    report, gov, trace, children, design = _run(tmp_path, cap=260.0, cell_cost=1.0, fail=("seed12__clean_next_level__ridge__W4__L256__s1",),
                                                role_cost={"CELL": 100.0, "COST_PILOT": 1.0})
    assert report["stopped"] and report["stopped"].startswith(RUN.DEV.CAP_EXHAUSTED)
    assert 0 < len(report["cells"]) < design["cells_total"]
    failed = report["cells"].get("seed12__clean_next_level__ridge__W4__L256__s1")
    assert failed is None or (failed["outcome"] == RUN.H.RESOURCE_EXCEEDED and "skill_test" not in failed)
    assert report["reconciliation"]["missing_units"]                          # the rest is reported missing, not hidden
