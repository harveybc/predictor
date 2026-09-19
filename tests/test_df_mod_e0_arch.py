"""RP14: the ARCH-A/B/C/0 design and its runner. Design: deterministic enumeration, every dependency
group on one host, pilots per architecture, self-digest. Architectures: parameters, extractor
layers, declared reach TESTED by perturbation, readout controls. Runner (stubbed governance and
children): pilots per architecture, projection by role, delegation, a worker's execute-only run,
the coordinator's report of collected attempts, concurrent waves that respect dependencies."""
import importlib.util
import json
import shutil
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


E = _load("df_mod_e0")
AD = _load("df_mod_e0_arch_design")
RUN = _load("df_mod_e0_run")
CLOSE = _load("df_mod_e0_close")
GR = _load("governed_run")
ORDER = _load("test_df_utility_run_order", HERE)
DEVT = _load("test_df_utility_dev_run", HERE)
RUNT = _load("test_df_mod_e0_run", HERE)


def _design(**kw):
    return AD.build(levels=[0, 3], replicates=[1], random_assignments=1, max_updates=6, **kw)


def test_RP14_design_enumerates_every_architecture_mechanism_control_and_diagnostic_on_one_host_per_group():
    d = _design()
    assert d["schema"] == AD.DESIGN_SCHEMA and E.sha_obj({k: v for k, v in d.items() if k != "design_sha256"}) == d["design_sha256"]
    ids = [c["cell_id"] for c in d["cells"]]
    assert len(ids) == len(set(ids)) == d["cells_total"]
    per_arch = {a: [c for c in d["cells"] if c["arch"] == a] for a in AD.ARCHS}
    for a in AD.ARCHS:
        h2 = [c for c in per_arch[a] if c["hypothesis"] == "H2"]
        h3 = [c for c in per_arch[a] if c["hypothesis"] == "H3" and c.get("donor") != "summary" and c["arm"] != "extractor_summary"]
        assert len(h2) == 2 * 1 * 2 and len(h3) == 2 * 1 * (1 + 4)                # levels x seeds x (profiles + 1 random); r x seeds x (extractor + 4 readouts)
        assert {c["arm"] for c in h3} == {"extractor", *AD.READOUT_ARMS}
        assert [c for c in per_arch[a] if c["hypothesis"] == "DX"] and all(c["diagnostic"] == "trend_event" for c in per_arch[a] if c["hypothesis"] == "DX")
    dsum = [c for c in d["cells"] if c.get("donor") == "summary"]
    assert {c["arch"] for c in dsum} == {"A", "B"} and all(c["depends_on"].endswith("extractor_summary") for c in dsum)
    by_id = {c["cell_id"]: c for c in d["cells"]}
    for c in d["cells"]:
        if c.get("depends_on"):
            assert by_id[c["depends_on"]]["host_role"] == c["host_role"]          # dependency locality
    assert set(d["cells_by_role"]) == set(AD.ROLES) and min(d["cells_by_role"].values()) > 0
    assert len(d["pilots"]) == 12 and {p["arch"] for p in d["pilots"]} == set(AD.ARCHS)
    assert d["reference"] == "A" and "NO_WINNER" in d["classification"]
    # the closure derives the same population
    pop = CLOSE.population(d)
    assert pop["v2"] and pop["members"] == ids and len(pop["pilot_ids"]) == 12
    assert d["common"]["reach"]["0"]["sequence"] == 3 >= E.TAU                    # ARCH-0 reaches the planted lag through the core
    assert AD.pilot_key_for(by_id[f"H3__r1__s1__C__summary_last"]) == "pilot__C__H3_sequence"
    assert AD.pilot_key_for(by_id[f"DX__trend_event__s1__0__profiles"]) == "pilot__0__H2_profiles"


@pytest.mark.parametrize("arch", AD.ARCHS)
def test_RP14_each_architecture_builds_with_the_declared_reach_and_extractor_layers(arch):
    assign = [0] * 4 + [1] * 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, E.WINDOW, 8)).astype(np.float32)
    for fusion in AD.READOUT_ARMS:
        m = E.build_modular(assign, E.WINDOW, 8, fusion=fusion, seed=1, arch=arch)
        ext = E.extractor_layer_names(m)
        assert (len(ext) == 0) == (arch == "0")
        base = m.predict(X, verbose=0)
        declared = E.support_reach(arch, fusion, E.WINDOW)
        # a perturbation beyond the declared reach never changes the output
        if declared < E.WINDOW:
            Xp = X.copy()
            Xp[:, :E.WINDOW - declared, :] += 5.0
            assert np.allclose(m.predict(Xp, verbose=0), base, atol=1e-6), (arch, fusion)
        # a perturbation at the last reachable position does
        Xq = X.copy()
        Xq[:, E.WINDOW - min(declared, 3), :] += 5.0
        assert not np.allclose(m.predict(Xq, verbose=0), base, atol=1e-6), (arch, fusion)
        pc = E.count_params(m)
        assert pc["trainable"] == pc["total"] > 0
    assert E.build_modular(assign, E.WINDOW, 8, fusion="sequence", seed=1, arch="B").count_params() == 9608   # the executed pilot's receiver unchanged


def test_RP14_the_readout_controls_share_the_fusion_and_differ_only_in_the_readout():
    assign = [0] * 4 + [1] * 4
    seq = E.build_modular(assign, E.WINDOW, 8, fusion="sequence", seed=1, arch="A")
    gap = E.build_modular(assign, E.WINDOW, 8, fusion="sequence_gap", seed=1, arch="A")
    assert E.count_params(seq) == E.count_params(gap) and "core_gap" in [l.name for l in gap.layers] and "core_last" in [l.name for l in seq.layers]
    summ = E.build_modular(assign, E.WINDOW, 8, fusion="summary", seed=1, arch="A")
    last = E.build_modular(assign, E.WINDOW, 8, fusion="summary_last", seed=1, arch="A")
    assert E.count_params(summ) == E.count_params(last) and "last_g0" in [l.name for l in last.layers] and "summary_g0" in [l.name for l in summ.layers]
    with pytest.raises(ValueError):
        E.build_modular(assign, E.WINDOW, 8, fusion="nope", seed=1, arch="A")
    with pytest.raises(ValueError):
        E.build_modular(assign, E.WINDOW, 8, fusion="sequence", seed=1, arch="Z")


def test_RP14_a_DX_cell_runs_the_profiles_arm_under_the_diagnostic_and_a_successor_can_hold_DX_only(tmp_path):
    rec = E.run_cell({"cell_id": "dx", "hypothesis": "DX", "level": 3, "r": 1, "seed": 1, "arm": "profiles", "role": "CELL", "arch": "0",
                      "diagnostic": "trend_event", "max_updates_override": 4, "descriptors": False}, tmp_path / "dx")
    assert rec["diagnostic"] == "trend_event" and rec["arch"] == "0" and rec["assignment_is_profile"] and rec["scores"]["validation"]["model"]["status"] == "MEDIDO"
    d = AD.build(levels=[0, 3], replicates=[1], random_assignments=1, max_updates=6, only_hypotheses=["DX"], hosts=["COORDINATOR"], successor_of="p" * 64)
    assert d["cells_total"] == 8 and all(c["hypothesis"] == "DX" and c["host_role"] == "COORDINATOR" for c in d["cells"]) and d["pilots"] == []
    assert CLOSE.population(d)["members"] == [c["cell_id"] for c in d["cells"]] and CLOSE.population(d)["pilot_ids"] == []
    # the successor inherits the parent's measured pilot costs, checked by identity; without them it refuses
    parent = _design()
    report, gov, trace, children = _run(tmp_path, parent, root=tmp_path / "parent")
    d2 = AD.build(levels=[0, 3], replicates=[1], random_assignments=1, max_updates=6, only_hypotheses=["DX"], hosts=["COORDINATOR"], successor_of=parent["design_sha256"])
    with pytest.raises(RUN.R.Refusal):
        _run(tmp_path, d2, root=tmp_path / "dx_norep")
    gov2 = ORDER.StubGov()
    trace2, children2 = [], []
    rep2 = RUN.run_mod_e0(d2, root=tmp_path / "dx", run_id="dx", gov=gov2, trace=lambda e, **f: trace2.append((e, f)), GR=GR, outbox=ORDER.StubOutbox(gov2),
                          OB=DEVT.StubOB(), CE=DEVT.StubCE(), code_identity={"kind": "git_commit", "value": "0" * 40},
                          budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5}, cap_seconds=10 ** 9, already_spent=0.0, pilot_updates=10,
                          isolated=RUNT._stub(children2), pilot_costs_from={"run_id": "e0", "design_sha256": parent["design_sha256"], "cost_pilot": report["cost_pilot"]})
    assert rep2["stopped"] is None and len(rep2["cells"]) == 8 and all(k.startswith("DX__") for k in children2) and len(children2) == 8
    assert rep2["cost_pilot"]["inherited_from_run"] == "e0" and "pilots-inherited" in [e for e, _ in trace2]


def test_RP14_the_diagnostic_condition_adds_a_known_deterministic_term_the_oracle_sees():
    g = E.generate(3, 1, 1, diagnostic="trend_event")
    g0 = E.generate(3, 1, 1)
    assert np.allclose(g["x"] - g0["x"], g["deterministic"][:, None])
    assert np.nanmax(np.abs((g["oracle"] - g0["oracle"])[:-1] - g["deterministic"][1:, None])) < 1e-9
    assert g["deterministic"][1300] - g["deterministic"][1299] > 1.9 and g["params"]["diagnostic"] == "trend_event"
    with pytest.raises(ValueError):
        E.generate(3, 1, 1, diagnostic="unknown")


# --- runner with stubs ---------------------------------------------------------------------------------------

def _run(tmp_path, design, role="COORDINATOR", execute_only=False, report_collected=False, parallel=1, cap=10 ** 9, fail=(), cell_cost=None, root=None, gov=None):
    gov = gov or ORDER.StubGov()
    trace, children = [], []
    report = RUN.run_mod_e0(design, root=root or (tmp_path / "e0"), run_id="e0", gov=gov, trace=lambda e, **f: trace.append((e, f)), GR=GR,
                            outbox=ORDER.StubOutbox(gov), OB=DEVT.StubOB(), CE=DEVT.StubCE(), code_identity={"kind": "git_commit", "value": "0" * 40},
                            budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5}, cap_seconds=cap, already_spent=0.0,
                            pilot_updates=10, isolated=RUNT._stub(children, fail, 1.0, cell_cost), pilot_only=False, role=role,
                            execute_only=execute_only, report_collected=report_collected, parallel=parallel)
    return report, gov, trace, children


def test_RP14_the_coordinator_runs_pilots_per_architecture_projects_by_role_and_delegates_the_other_roles(tmp_path):
    design = _design()
    report, gov, trace, children = _run(tmp_path, design)
    assert len(report["cost_pilot"]) == 12 and all(k.startswith("pilot__") for k in report["cost_pilot"])
    assert set(report["projection"]["by_role_with_headroom"]) == set(AD.ROLES)
    mine = [c["cell_id"] for c in design["cells"] if c["host_role"] == "COORDINATOR"]
    others = [c["cell_id"] for c in design["cells"] if c["host_role"] != "COORDINATOR"]
    assert all(report["cells"][c]["outcome"] == "COMPLETED" for c in mine) and all(report["cells"][c]["outcome"] == RUN.DELEGATED for c in others)
    assert set(children) >= set(mine) and not (set(children) & set(others))
    sha = report["campaign"]["campaign_sha256"]
    assert all((sha, c) in gov.terminals for c in mine) and not any((sha, c) in gov.terminals for c in others)
    t = gov.terminals[(sha, mine[0])]
    assert t["tags"]["host_role"] == "COORDINATOR" and t["tags"].get("arch") in AD.ARCHS
    assert set(report["reconciliation"]["missing_units"]) == set(others)


def test_RP14_a_worker_executes_only_its_role_without_governance_and_the_coordinator_reports_the_collected_attempts(tmp_path):
    design = _design()
    coord_root = tmp_path / "coord"
    report, gov, trace, children = _run(tmp_path, design, root=coord_root)
    sha = report["campaign"]["campaign_sha256"]
    worker_root = tmp_path / "worker_a"
    wreport, wgov, wtrace, wchildren = _run(tmp_path, design, role="WORKER_A", execute_only=True, root=worker_root)
    a_cells = [c["cell_id"] for c in design["cells"] if c["host_role"] == "WORKER_A"]
    assert sorted(wchildren) == sorted(a_cells) and (worker_root / "REPORT.WORKER_A.json").is_file()
    assert wgov.terminals == {} and all(wreport["cells"][c]["outcome"] == "COMPLETED" for c in a_cells)
    for c in a_cells:                                                       # arms found their extractor on the same host
        if "__extractor" not in c and design["cells"][[x["cell_id"] for x in design["cells"]].index(c)].get("depends_on"):
            assert wreport["cells"][c]["mase_validation"] == 0.5
    # collect: the worker's attempts land in the coordinator's root; then the coordinator reports them
    for c in a_cells:
        shutil.copytree(worker_root / "attempts" / c, coord_root / "attempts" / c)
    report2, gov2, trace2, children2 = _run(tmp_path, design, role="COORDINATOR", report_collected=True, root=coord_root, gov=gov)
    # the same governance (the server persists): registration is resumed from CAMPAIGNS.json; missing units are those never reported
    assert (coord_root / "REPORT.collected.json").is_file()
    reported = {u for (s, u) in gov2.terminals if s == sha}
    assert set(a_cells) <= reported and children2 == []                     # nothing re-run
    assert all(report2["cells"][c]["outcome"] == "COMPLETED" for c in a_cells)
    assert gov2.terminals[(sha, a_cells[0])]["tags"]["collected"] == "true" and gov2.terminals[(sha, a_cells[0])]["tags"]["host_role"] == "WORKER_A"
    b_cells = [c["cell_id"] for c in design["cells"] if c["host_role"] == "WORKER_B"]
    assert all(report2["cells"][c]["outcome"] == RUN.DELEGATED for c in b_cells)


def test_RP14_concurrent_waves_respect_dependencies_and_the_ceiling(tmp_path):
    design = _design()
    report, gov, trace, children = _run(tmp_path, design, parallel=3)
    order = {c: i for i, c in enumerate(children)}
    for c in design["cells"]:
        if c["host_role"] == "COORDINATOR" and c.get("depends_on"):
            assert order[c["depends_on"]] < order[c["cell_id"]]
    assert report["stopped"] is None
    # a failed extractor makes its arms INCONCLUSIVE_DEPENDENCY in the parallel path too
    ext = next(c["cell_id"] for c in design["cells"] if c["host_role"] == "COORDINATOR" and c["arm"] == "extractor")
    report, gov, trace, children = _run(tmp_path / "f", design, parallel=3, fail=(ext,))
    arms = [c["cell_id"] for c in design["cells"] if c.get("depends_on") == ext]
    assert all(report["cells"][a]["outcome"] == "INCONCLUSIVE_DEPENDENCY" for a in arms) and not (set(arms) & set(children))
    # the ceiling stops dispatch with incompletes kept
    report, gov, trace, children = _run(tmp_path / "c", design, parallel=3, cap=2000.0, cell_cost=400.0)
    assert report["stopped"] and report["stopped"].startswith(RUN.DEV.CAP_EXHAUSTED) and report["reconciliation"]["missing_units"]


# The RP16 effects test that encoded the "average of the surviving fusion arms" (dictum F1) was removed with
# the v1 estimator; the contrast-bound estimator is tested in tests/test_df_mod_e0_arch_effects.py (RP18).


def test_RP22_readout_completion_successor_inherits_the_parent_donors_never_retrains_and_closes_with_them(tmp_path):
    """The parent stage (stubbed) executes; the successor enumerates only the 16 r=0 readout controls, links the
    parent's r=0 extractors read-only, runs 16 children (no extractor), inherits the parent's measured pilot costs,
    and the closure verifies the inherited donors against the PARENT's job while its population is the 16 cells."""
    parent = _design()                                               # levels [0,3], replicates [1], readout controls at r in {0,1} by default
    parent = AD.build(levels=[0, 3], replicates=[1], random_assignments=1, max_updates=6, readout_controls_r=[1])
    parent_root = tmp_path / "parent"
    report, gov, trace, children = _run(tmp_path, parent, root=parent_root)
    # make the parent's delegated cells 'collected' for the test: run the workers and copy their attempts
    for role in ("WORKER_A", "WORKER_B"):
        wroot = tmp_path / role
        _run(tmp_path, parent, role=role, execute_only=True, root=wroot)
        for a in (wroot / "attempts").iterdir():
            if not (parent_root / "attempts" / a.name).exists():
                shutil.copytree(a, parent_root / "attempts" / a.name)
    succ = AD.readout_completion(parent, "e0", str(parent_root), "test: complete the r=0 readout controls", {"equivalent_training_path": True})
    assert succ["cells_total"] == 4 * 1 * 2 and len(succ["inherited"]) == 4 and succ["pilots"] == [] and succ["successor_of"] == parent["design_sha256"]
    assert all(c["r"] == 0 and c["arm"] in ("sequence_gap", "summary_last") and c["depends_on"].endswith("__extractor") for c in succ["cells"])
    assert {c["depends_on"] for c in succ["cells"]} == {i["cell_id"] for i in succ["inherited"]}
    pop = CLOSE.population(succ)
    assert pop["members"] == [c["cell_id"] for c in succ["cells"]] and len(pop["inherited"]) == 4
    # the runner: inherited donors linked, only the 8 arms run, pilots inherited by identity
    gov2 = ORDER.StubGov()
    trace2, children2 = [], []
    sroot = tmp_path / "succ"
    rep2 = RUN.run_mod_e0(succ, root=sroot, run_id="succ", gov=gov2, trace=lambda e, **f: trace2.append((e, f)), GR=GR, outbox=ORDER.StubOutbox(gov2),
                          OB=DEVT.StubOB(), CE=DEVT.StubCE(), code_identity={"kind": "git_commit", "value": "0" * 40},
                          budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5}, cap_seconds=10 ** 9, already_spent=0.0, pilot_updates=10,
                          isolated=RUNT._stub(children2), pilot_costs_from={"run_id": "e0", "design_sha256": parent["design_sha256"], "cost_pilot": report["cost_pilot"]})
    assert rep2["stopped"] is None and sorted(children2) == sorted(c["cell_id"] for c in succ["cells"]) and len(children2) == 8
    assert all(v["outcome"] == "INHERITED" for v in rep2["inherited"].values()) and len(rep2["inherited"]) == 4
    for inh in succ["inherited"]:
        link = sroot / "attempts" / inh["cell_id"]
        assert link.is_symlink() and link.resolve() == (parent_root / "attempts" / inh["cell_id"]).resolve()
    assert rep2["reconciliation"]["missing_units"] == [] and set(u for (s, u) in gov2.terminals) == set(children2)
    # a second run refuses to replace the inherited link with something else
    (sroot / "attempts" / succ["inherited"][0]["cell_id"]).unlink()
    (sroot / "attempts" / succ["inherited"][0]["cell_id"]).mkdir()
    with pytest.raises(RUN.R.Refusal):
        RUN.run_mod_e0(succ, root=sroot, run_id="succ", gov=gov2, trace=lambda e, **f: None, GR=GR, outbox=ORDER.StubOutbox(gov2), OB=DEVT.StubOB(), CE=DEVT.StubCE(),
                       code_identity={"kind": "git_commit", "value": "0" * 40}, budgets={"task_memory_bytes": 1, "wall_seconds": 5.0, "cpu_seconds": 5},
                       cap_seconds=10 ** 9, already_spent=0.0, pilot_updates=10, isolated=RUNT._stub([]),
                       pilot_costs_from={"run_id": "e0", "design_sha256": parent["design_sha256"], "cost_pilot": report["cost_pilot"]})
