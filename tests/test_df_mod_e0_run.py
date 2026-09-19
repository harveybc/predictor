"""RP4 / ML11-ML12 runner rules with stubbed governance and children: design frozen, pilots without
test access, projection with headroom, registration before any cell, H3 arms only after their
extractor, terminals with the metrics contract states, incompletes kept, failures preserved."""
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


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
RUN = _load("df_mod_e0_run")
GR = _load("governed_run")
ORDER = _load("test_df_utility_run_order", HERE)
DEVT = _load("test_df_utility_dev_run", HERE)


def _design():
    doc = D.build()
    doc.update(levels=[0, 3], replicates=[1], random_assignments=2)
    doc["cells"] = D.cells(doc)
    doc["cells_total"] = len(doc["cells"])
    doc["design_sha256"] = E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def _stub(children, fail=(), cost=1.0, cell_cost=None):
    def run(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        attempt_dir = Path(attempt_dir)
        marker = attempt_dir / "outcome.json"
        if marker.is_file():
            return {**json.loads(marker.read_text()), "resumed": True}
        children.append(job["cell_id"])
        attempt_dir.mkdir(parents=True, exist_ok=True)
        c = {"cpu_seconds": (cell_cost if (cell_cost is not None and job.get("role") == "CELL") else cost), "wall_seconds": 1.0, "peak_rss_bytes": 1, "cgroup_memory_peak": None,
             "started_at": "2026-09-18T12:00:00Z", "ended_at": "2026-09-18T12:00:01Z"}
        if job["cell_id"] in fail:
            summary = {"outcome": RUN.H.RESOURCE_EXCEEDED, "reason": "WALL", "cost": c, "score": None, "output_sha256": None}
            marker.write_text(json.dumps(summary))
            return summary
        if job["arm"] in ("sequence", "summary") and job["hypothesis"] == "H3":
            assert job.get("extractor_weights"), "an H3 arm needs its extractor"
        pilot = job.get("role") == "COST_PILOT"
        (attempt_dir / "weights.weights.h5").write_bytes(b"w")
        rec = {"schema": E.CELL_SCHEMA, "cell_id": job["cell_id"], "exposure": "NO_TEST_ACCESS" if pilot else "TEST_SCORED_DESCRIPTIVE",
               "scores": {"validation": {"model": {"mase_mean": 0.5, "mae_mean": 0.4}, "naive": {"mase_mean": 0.7}, "oracle": {"mase_mean": 0.4},
                                         "linear_window": {"mase_mean": 0.45}},
                          **({} if pilot else {"test": {"model": {"mase_mean": 0.52, "mae_mean": 0.41}}})},
               "training": {"updates": 30, "stop_reason": "EARLY_STOPPING"}, "parameters": {"trainable": 900}, "profiles": {"ari_vs_latent": 1.0},
               "extractor_weight_change": 0.0, "cost": {"cpu_seconds": cost, "fit_seconds": 0.8}}
        summary = {"outcome": "COMPLETED", "reason": "", "cost": c, "score": rec, "output_sha256": "o" * 64}
        marker.write_text(json.dumps(summary))
        return summary
    return run


def _run(tmp_path, cap=10 ** 9, fail=(), pilot_only=False, cost=1.0, cell_cost=None):
    design = _design()
    gov = ORDER.StubGov()
    trace, children = [], []
    report = RUN.run_mod_e0(design, root=tmp_path / "e0", run_id="e0", gov=gov, trace=lambda e, **f: trace.append((e, f)), GR=GR,
                            outbox=ORDER.StubOutbox(gov), OB=DEVT.StubOB(), CE=DEVT.StubCE(),
                            code_identity={"kind": "git_commit", "value": "0" * 40},
                            budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5}, cap_seconds=cap, already_spent=0.0,
                            pilot_updates=10, isolated=_stub(children, fail, cost, cell_cost), pilot_only=pilot_only)
    return report, gov, trace, children, design


def test_RP4_freeze_pilots_without_test_project_with_headroom_register_then_cells_in_dependency_order(tmp_path):
    report, gov, trace, children, design = _run(tmp_path)
    events = [e for e, _ in trace]
    assert events[0] == "design-frozen" and (tmp_path / "e0" / "DESIGN.json").is_file()
    assert children[:3] == ["pilot__H2_profiles", "pilot__H3_extractor", "pilot__H3_sequence"]
    assert events.index("projection") < events.index("register", events.index("projection"))
    assert report["stopped"] is None and len(report["cells"]) == design["cells_total"] == 2 * 1 * 3 + 2 * 1 * 3
    for c in design["cells"]:
        if c.get("depends_on"):
            assert children.index(c["depends_on"]) < children.index(c["cell_id"])
    assert report["projection"]["headroom"] == 0.25 and report["projection"]["projected_with_headroom"] > 0
    assert report["reconciliation"]["missing_units"] == []
    t = gov.terminals[(report["campaign"]["campaign_sha256"], "H2__h3__s1__profiles")]
    names = {m["metric"]: m for m in t["metrics"]}
    assert names["mod_e0.mase_validation"]["status"] == "MEDIDO" and "mod_e0.mase_test" in names
    assert t["tags"]["hypothesis"] == "H2" and t["tags"]["condition"] == "h3_r1" and t["tags"]["phase"] == "DEVELOPMENT"
    pt = gov.terminals[(list(gov.terminals)[0][0], "pilot__H2_profiles")]
    assert not any(m["metric"].endswith("_test") for m in pt["metrics"])
    assert report["envelope"]["sha"] == "e" * 64


def test_RP4_projection_beyond_the_ceiling_or_pilot_only_launches_no_cell(tmp_path):
    report, gov, trace, children, design = _run(tmp_path, cap=3.5)
    assert report["stopped"] == "PROJECTION_EXCEEDS_CAP_NOT_LAUNCHED" and report["cells"] == {}
    assert json.loads((tmp_path / "e0" / "PLAN.json").read_text())["measured_costs"]
    report, gov, trace, children, design = _run(tmp_path / "b", pilot_only=True)
    assert report["stopped"] == "PILOT_ONLY" and len(children) == 3 and (tmp_path / "b" / "e0" / "REPORT.pilot.json").is_file()


def test_ML12_a_failed_extractor_leaves_its_terminal_and_makes_its_arms_inconclusive_not_rerun(tmp_path):
    report, gov, trace, children, design = _run(tmp_path, fail=("H3__r1__s1__extractor",))
    assert report["cells"]["H3__r1__s1__extractor"]["outcome"] == RUN.H.RESOURCE_EXCEEDED
    assert report["cells"]["H3__r1__s1__sequence"]["outcome"] == "INCONCLUSIVE_DEPENDENCY"
    assert "H3__r1__s1__sequence" not in children
    sha = report["campaign"]["campaign_sha256"]
    assert gov.terminals[(sha, "H3__r1__s1__extractor")]["status"] == "FAILED"
    assert "H3__r1__s1__sequence" in report["reconciliation"]["missing_units"]
    assert report["cells"]["H3__r0__s1__sequence"]["mase_validation"] == 0.5


def test_ML11_the_ceiling_is_enforced_before_each_child_and_incompletes_are_kept(tmp_path):
    # pilots cost 1 s (projection ~167 s per cell, 12 cells -> ~2 508 s with headroom); the cells then cost 400 s each
    report, gov, trace, children, design = _run(tmp_path, cap=2600.0, cost=1.0, cell_cost=400.0)
    assert report["stopped"] and report["stopped"].startswith(RUN.DEV.CAP_EXHAUSTED)
    assert 0 < len(report["cells"]) < design["cells_total"] and report["reconciliation"]["missing_units"]


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_ML11_a_real_cost_pilot_child_completes_and_verifies_without_a_test_split(tmp_path):
    job = {"kind": "mod_e0_cell", "cell_id": "pilot__H2_profiles", "hypothesis": "H2", "level": 3, "r": 1, "seed": 1, "arm": "profiles",
           "window": E.WINDOW, "training": E.TRAINING, "design_sha256": "d" * 64, "run_id": "t", "max_updates_override": 8, "role": "COST_PILOT"}
    out = RUN.run_isolated(job, attempt_dir=tmp_path / "p", assigned_bytes=3 << 30, wall_seconds=600.0, cpu_seconds=600)
    assert out["outcome"] == "COMPLETED", out
    rec = out["score"]
    assert rec["exposure"] == "NO_TEST_ACCESS" and "test" not in rec["scores"] and rec["cost"]["fit_seconds"] > 0
    again = RUN.run_isolated(job, attempt_dir=tmp_path / "p", assigned_bytes=3 << 30, wall_seconds=600.0, cpu_seconds=600)
    assert again["resumed"] is True and again["score"]["scores"]["validation"]["model"]["mase_mean"] == rec["scores"]["validation"]["model"]["mase_mean"]
