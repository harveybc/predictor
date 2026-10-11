"""Lane B campaign: sealed plan, preflight, install/start, DAG activation, status/ETA, collect.

Generated fixtures only; systemctl is a recorder; no unit is installed on this host.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import i6n_swarm_fixtures as F
from tools import i6n_search as search
from tools.i6n_swarm import bridge as B, collect as C, common, coordinator as K, plan as P
from tools.i6n_swarm import status as S, units as U, ledger as L

@pytest.fixture(autouse=True)
def _reference_aggregator(monkeypatch):
    """These tests exercise B; E's score_folds is exercised by E's suite and the integration check."""
    from tools.i6n_swarm import bridge as _b
    monkeypatch.setattr(_b, "resolve_score_folds", lambda: _b._reference_score_folds)



@pytest.fixture
def root(tmp_path):
    cfg = F.private_config(tmp_path)
    out = tmp_path / "campaign"
    P.build(cfg, out)
    return out


class Recorder:
    def __init__(self, active=()):
        self.calls, self.active = [], set(active)

    def __call__(self, *args):
        self.calls.append(list(args))
        if args[:1] == ("is-active",):
            return 0 if args[1] in self.active else 3
        if args[:1] == ("start",):
            self.active.update(args[1:])
        return 0


def receipt(swarm, fold, wire, config, factor):
    sup = fold["support"]
    return dict(status="ok", counts=sup["counts"], naive_sums=sup["naive_sums"], support_digest=sup["support_digest"],
                error_sums={k: v * factor for k, v in sup["naive_sums"].items()},
                metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1",
                    experiment=swarm["experiment"], fold=fold["fold"], effective_config_sha256=common.sha256_json(config),
                    horizon_hours=swarm["horizon"], genotype_canonical_sha256=wire["identity"], seed=6102026, rows=[])))


def complete_candidate(root, horizon, genome, factor, node=None, seconds=100.0, fail_fold=None):
    """Drive the real ledger exactly as the bridge would, with a fabricated receipt."""
    swarm = K.swarm_context(root, horizon)
    led = L.Ledger(root / "state" / "ledger.sqlite3", clock=lambda: 1000.0)
    genome = search.normalize_genome(genome)
    wire = dict(schema="i6n-shared-deap-v1", genome=genome, identity=search.effective_identity(genome),
                context="0" * 64, generation=0)
    config = search.effective_config(genome)
    node = node or dict(device=dict(cls="cpu_fixture"), fixture=True)
    for fold in swarm["folds"]:
        fit = B.fold_fit_id(swarm, fold, config)
        led.ensure_unit(fit, B.unit_spec(swarm, fold, wire, config, node))
        claim = led.claim(fit, "n1", "preferred", "boot", node["device"]["cls"], 600)
        if claim["result"] != "CLAIMED":
            continue
        if fold["fold"] == fail_fold:
            led.report(fit, claim["token"], "FAILED", "EVALUATOR_REFUSED")
            continue
        led.clock = lambda: 1000.0 + seconds
        led.complete(fit, claim["token"], B.receipt_document(fit, swarm, fold, wire, config,
                                                             receipt(swarm, fold, wire, config, factor)))
        led.clock = lambda: 1000.0
    return wire


def other_genome(h, transform="difference", width=None):
    g = search.default_genome(h)
    g["input_transform"] = transform
    if width:
        g["core"]["width"] = width
    return g


def test_plan_is_sealed_and_tamper_evident(root):
    plan = P.load(root)
    payload = plan["payload"]
    assert payload["seed"] == 6102026 and payload["direction"] == "minimize"
    assert payload["search"]["population_size"] <= 8 and payload["search"]["generations"] == 3
    assert payload["search"]["tolerance"] == 1e-4 and payload["search"]["patience"] == 3
    assert [n["node_id"] for n in payload["dag"] if not n["node_id"].startswith("freeze")] == [
        "search:h12", "search:h72", "annual:h12", "annual:h72", "multi_output:short", "multi_output:long"]
    assert {n["node_id"] for n in payload["dag"] if n["node_id"].startswith("freeze")} == {
        "freeze_folds:h12", "freeze_folds:h72", "freeze_weeks:h12", "freeze_weeks:h72"}
    assert payload["budget"]["search_new_fits_max"] == 4 * 2 * 2
    for role in ("preferred", "secondary"):
        node = json.loads((root / "private" / "hosts" / role / "doin_node.json").read_text())
        domain = node["domains"][0]
        assert domain["higher_is_better"] is False
        oc = domain["optimization_config"]
        assert oc["shared_population_seed"] == 6102026 and oc["improvement_tolerance"] == 1e-4
        assert oc["optimizer_plugin"] == "i6n_optimizer" and oc["reproduction_failure_policy"] == "fail_closed"
        assert oc["i6n_swarm"]["node_config"].endswith("node.json")
    text = (root / "PLAN.json").read_text().replace("6102026", "6102027", 1)
    (root / "PLAN.json").write_text(text)
    with pytest.raises(ValueError, match="seal"):
        P.load(root)


def test_plan_refuses_existing_root_and_wrong_seed(tmp_path):
    cfg = F.private_config(tmp_path)
    out = tmp_path / "c"
    P.build(cfg, out)
    with pytest.raises(FileExistsError):
        P.build(cfg, out)
    bad = F.private_config(tmp_path / "x", seed=42)
    with pytest.raises(ValueError, match="6102026"):
        P.build(bad, tmp_path / "d")


def test_plan_refuses_test_period_inputs(tmp_path):
    cfg_path = F.private_config(tmp_path)
    cfg = json.loads(Path(cfg_path).read_text())
    views = json.loads(Path(cfg["weekly_views"]["path"]).read_text())
    extra = dict(views["weeks"][0], week_id="W2025-01-05", unit_id="W2025-01-05", cutoff_utc="2025-01-03T22:00:00Z")
    views["weeks"].append(extra)
    cfg["weekly_views"] = F.pin(cfg["weekly_views"]["path"], views)
    with pytest.raises(ValueError, match="TEST"):
        P.build(F.pin(cfg_path, cfg)["path"], tmp_path / "c")


def test_preflight_never_trains_and_reports_checks(root, monkeypatch):
    def forbid(*a, **k):
        raise AssertionError("preflight launched a process")
    monkeypatch.setattr(subprocess, "Popen", forbid)
    report = K.preflight(root, which=lambda name: None, linger=lambda: False)
    names = {c["name"]: c for c in report["checks"]}
    assert names["plan_seal"]["ok"] and names["input_digests"]["ok"]
    assert names["seed_direction"]["ok"]
    assert not names["user_linger"]["ok"]
    assert report["state"] == "BLOCKED"


def test_install_writes_only_this_campaign_units_and_enables_them(root, tmp_path):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / "warehouse.service").write_text("[Unit]\nDescription=foreign\n")
    rec = Recorder()
    result = U.install(root, roles=["coordinator", "preferred"], unit_dir=unit_dir, systemctl=rec)
    names = sorted(p.name for p in unit_dir.iterdir())
    assert "warehouse.service" in names
    ours = [n for n in names if n != "warehouse.service"]
    assert ours and all(n.startswith("i6n-swarm-fx1-") for n in ours)
    assert (unit_dir / "warehouse.service").read_text() == "[Unit]\nDescription=foreign\n"
    for name in ours:
        body = (unit_dir / name).read_text()
        assert "X-I6N-Swarm-Campaign=fx1" in body
        if name.endswith(".service") and "tick" not in name:
            assert "Restart=on-failure" in body
    enabled = [c for c in rec.calls if c[0] == "enable"]
    assert enabled and all(n.startswith("i6n-swarm-fx1-") for c in enabled for n in c[1:])
    assert not any(c[0] in ("stop", "restart", "disable", "kill") for c in rec.calls)
    assert set(result["units"]) == set(ours)
    again = U.install(root, roles=["coordinator", "preferred"], unit_dir=unit_dir, systemctl=Recorder())
    assert again["unchanged"] == sorted(ours)


def test_install_refuses_to_overwrite_a_foreign_file_with_our_name(root, tmp_path):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    (unit_dir / "i6n-swarm-fx1-tick.service").write_text("[Unit]\nDescription=someone else\n")
    with pytest.raises(ValueError, match="foreign"):
        U.install(root, roles=["coordinator"], unit_dir=unit_dir, systemctl=Recorder())


def test_start_is_idempotent(root, tmp_path):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    U.install(root, roles=["coordinator", "preferred"], unit_dir=unit_dir, systemctl=Recorder())
    rec = Recorder()
    first = K.start(root, roles=["coordinator", "preferred"], unit_dir=unit_dir, systemctl=rec)
    second = K.start(root, roles=["coordinator", "preferred"], unit_dir=unit_dir, systemctl=rec)
    starts = [c for c in rec.calls if c[0] == "start"]
    assert len(starts) == 1 and first["started"] and second["started"] == []
    led = L.Ledger.open_readonly(root / "state" / "ledger.sqlite3")
    assert len(led.dag_nodes()) == 10


def test_status_reads_only_and_reports_every_level(root, monkeypatch):
    K.start(root, roles=[], systemctl=Recorder())
    complete_candidate(root, 12, search.default_genome(12), 0.9)

    def forbid(*a, **k):
        raise AssertionError("status launched a process")
    monkeypatch.setattr(subprocess, "Popen", forbid)
    st = S.compute(root, now=2000.0)
    assert set(st["levels"]) == {"swarm", "horizon", "candidate", "fold", "publication"}
    assert st["levels"]["swarm"]["state"] in S.STATES
    h12 = next(n for n in st["levels"]["horizon"] if n["node_id"] == "search:h12")
    assert h12["counts"]["complete"] == 2
    assert h12["eta"]["kind"] == "unknown"  # fixture fits never feed production ETA
    assert all(f["state"] in S.STATES for f in st["levels"]["fold"])
    assert st["levels"]["publication"]["pending"] == 2
    annual = next(n for n in st["levels"]["horizon"] if n["node_id"] == "annual:h12")
    assert annual["state"] == "WAIT_DEPENDENCY" and "search:h12" in annual["cause"]


def test_status_cli_does_not_import_tensorflow(root):
    K.start(root, roles=[], systemctl=Recorder())
    code = ("import runpy, sys\n"
            f"sys.argv=['x','status','--root',{str(root)!r},'--json']\n"
            "try:\n    runpy.run_module('tools.i6n_swarm_campaign', run_name='__main__')\n"
            "except SystemExit as e:\n    assert e.code in (0, None), e.code\n"
            "assert 'tensorflow' not in sys.modules and 'keras' not in sys.modules\n"
            "assert 'tools.i6n_swarm.runner' not in sys.modules\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path.cwd(),
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["levels"]["swarm"]["state"]


def test_eta_measured_projection_excludes_fixture(root):
    K.start(root, roles=[], systemctl=Recorder())
    prod = dict(device=dict(cls="rtx5090"), fixture=False)
    complete_candidate(root, 12, search.default_genome(12), 0.9, node=prod, seconds=100.0)
    complete_candidate(root, 12, other_genome(12), 0.8, node=prod, seconds=300.0)
    st = S.compute(root, now=5000.0)
    h12 = next(n for n in st["levels"]["horizon"] if n["node_id"] == "search:h12")
    assert h12["eta"]["kind"] == "projected" and h12["eta"]["basis_n"] == 4
    remaining = h12["counts"]["expected"] - h12["counts"]["complete"]
    assert h12["eta"]["seconds"] == pytest.approx(remaining * 200.0)
    assert h12["eta"]["low"] <= h12["eta"]["seconds"] <= h12["eta"]["high"]


def test_successors_activate_automatically_and_failures_stay_local(root):
    K.start(root, roles=[], systemctl=Recorder())
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    complete_candidate(root, 72, search.default_genome(72), 0.95)
    complete_candidate(root, 72, other_genome(72), 0.90)
    complete_candidate(root, 12, search.default_genome(12), 0.9, fail_fold="2023Q2")
    led.dag_event("search:h72", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    led.dag_event("search:h12", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    result = K.tick(root, now=3000.0)
    states = {n["node_id"]: n for n in led.dag_nodes()}
    assert states["search:h72"]["state"] == "COMPLETE"
    assert states["search:h12"]["state"] == "FAILED" and "NO_COMPLETE_CANDIDATES" in states["search:h12"]["cause"]
    assert states["annual:h72"]["state"] == "QUEUED"
    assert states["annual:h12"]["state"] == "WAIT_DEPENDENCY"
    annual = [u for u in led.units() if u["dag_node"] == "annual:h72"]
    assert len(annual) == 2 * 2  # two finalists x two eligible weeks; the ineligible week is not a fit
    finalists = {u["genotype_canonical_sha256"] for u in annual}
    assert finalists == {search.effective_identity(search.default_genome(72)),
                         search.effective_identity(other_genome(72))}
    assert result["activated"] == ["annual:h72"]
    again = K.tick(root, now=3100.0)
    assert again["activated"] == [] and len([u for u in led.units() if u["dag_node"] == "annual:h72"]) == 4


def test_finalists_are_best_plus_minimal_control(root):
    K.start(root, roles=[], systemctl=Recorder())
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    complete_candidate(root, 12, other_genome(12, "difference"), 0.80)
    complete_candidate(root, 12, other_genome(12, "difference", width=32), 0.70)
    complete_candidate(root, 12, search.default_genome(12), 0.99)
    led.dag_event("search:h12", "SEARCH_TERMINAL", {"stop_reason": "patience"})
    K.tick(root, now=3000.0)
    node = next(n for n in led.dag_nodes() if n["node_id"] == "search:h12")
    summary = node["summary"]
    assert summary["finalists"][0]["role"] == "BEST"
    assert summary["finalists"][0]["fitness"] == pytest.approx(0.70)
    assert summary["finalists"][1]["role"] == "MINIMAL_CONTROL"
    assert summary["finalists"][1]["genotype_canonical_sha256"] == search.effective_identity(search.default_genome(12))


def test_annual_completion_activates_multi_output(root):
    K.start(root, roles=[], systemctl=Recorder())
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    for h in (12, 72):
        complete_candidate(root, h, search.default_genome(h), 0.9)
        complete_candidate(root, h, other_genome(h), 0.8)
        led.dag_event(f"search:h{h}", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    K.tick(root, now=3000.0)
    for unit in [u for u in led.units() if u["stage"] == "annual"]:
        c = led.claim(unit["fit_id"], "w", "preferred", "b", "cpu_fixture", 600)
        spec = unit["spec"]
        led.complete(unit["fit_id"], c["token"], dict(
            fit_id=unit["fit_id"], fold=spec["fold"], experiment=spec["experiment"],
            effective_config_sha256=spec["effective_config_sha256"],
            genotype_canonical_sha256=spec["genotype_canonical_sha256"],
            shared_receipt=dict(status="ok"), candidate_receipt=dict(seed=6102026, rows=[])))
    K.tick(root, now=3100.0)
    states = {n["node_id"]: n["state"] for n in led.dag_nodes()}
    assert states["annual:h12"] == "COMPLETE" and states["annual:h72"] == "COMPLETE"
    assert states["multi_output:short"] == "QUEUED"
    # the fixture defines no long-family arms: typed dependency, not a silent skip
    assert states["multi_output:long"] == "WAIT_DEPENDENCY"
    assert "MULTI_OUTPUT_ARMS_UNDEFINED" in {n["node_id"]: n for n in led.dag_nodes()}["multi_output:long"]["cause"]
    assert len([u for u in led.units() if u["stage"] == "multi_output"]) == 2


def test_tick_recovers_expired_leases(root):
    K.start(root, roles=[], systemctl=Recorder())
    swarm = K.swarm_context(root, 12)
    led = L.Ledger(root / "state" / "ledger.sqlite3", clock=lambda: 1000.0)
    genome = search.normalize_genome(search.default_genome(12))
    wire = dict(schema="i6n-shared-deap-v1", genome=genome, identity=search.effective_identity(genome), context="0" * 64, generation=0)
    config = search.effective_config(genome)
    fit = B.fold_fit_id(swarm, swarm["folds"][0], config)
    led.ensure_unit(fit, B.unit_spec(swarm, swarm["folds"][0], wire, config, dict(device=dict(cls="rtx5090"))))
    led.claim(fit, "n1", "preferred", "b", "rtx5090", 60)
    K.tick(root, now=5000.0)
    assert L.Ledger(root / "state" / "ledger.sqlite3").unit(fit)["state"] == "INTERRUPTED"


def test_collect_publishes_once_and_writes_results(root):
    K.start(root, roles=[], systemctl=Recorder())
    complete_candidate(root, 12, search.default_genome(12), 0.9)
    complete_candidate(root, 12, other_genome(12), 0.8, fail_fold="2023Q2")
    published = []

    class FakePublisher:
        def record_candidate(self, candidate, fold_receipts):
            published.append((candidate, fold_receipts))
            return dict(item_id=common.sha256_json(candidate), states={"LOCAL_COMPLETE": {"ok": True}})

    first = C.collect(root, publisher=FakePublisher())
    second = C.collect(root, publisher=FakePublisher())
    assert first["published"] == 2 and second["published"] == 0
    docs = {c["disposition"]: c for c, _ in published}
    assert set(docs) == {"COMPLETE", "FAILED"}
    done = docs["COMPLETE"]
    assert done["schema"] == "i6n.archive.candidate.v1" and done["seed"] == 6102026
    assert done["fitness"] == pytest.approx(0.9) and docs["FAILED"]["fitness"] is None
    assert len(done["attempts"]) == 2 and all(a["index"] == 1 for a in done["attempts"])
    rows = list(csv.DictReader(open(root / "collect" / "RESULTS.csv")))
    train = [r for r in rows if r["split"] == "TRAIN" and r["disposition"] == "COMPLETE"]
    assert train and float(train[0]["mae"]) == pytest.approx(float(train[0]["naive_mae"]) * 0.9)
    assert json.loads((root / "collect" / "ARTIFACTS.json").read_text())["plan_sha256"]


def test_collect_without_publisher_keeps_outbox_pending(root):
    K.start(root, roles=[], systemctl=Recorder())
    complete_candidate(root, 12, search.default_genome(12), 0.9)
    result = C.collect(root, publisher=None)
    assert result["publication_state"] == "WAIT_DEPENDENCY"
    st = S.compute(root, now=2000.0)
    assert st["levels"]["publication"]["state"] == "WAIT_DEPENDENCY"


def test_unfrozen_folds_become_freeze_units_and_wait_dependency_is_typed(tmp_path):
    cfg = F.private_config(tmp_path, frozen=False, wait_folds=("2023Q1",))
    out = tmp_path / "campaign"
    P.build(cfg, out)
    K.start(out, roles=[], systemctl=Recorder())
    K.tick(out, now=3000.0)
    led = L.Ledger(out / "state" / "ledger.sqlite3")
    states = {n["node_id"]: n for n in led.dag_nodes()}
    assert states["freeze_folds:h12"]["state"] == "WAIT_DEPENDENCY"
    assert "2023Q1" in states["freeze_folds:h12"]["cause"] and "CLOCK_SCOPE" in states["freeze_folds:h12"]["cause"]
    assert states["search:h12"]["state"] == "WAIT_DEPENDENCY"
    freeze = [u for u in led.units() if u["stage"] == "freeze"]
    # the READY fold is frozen (no fit) by a fixed-unit worker; the waiting fold never gets a unit
    assert sorted(u["fold"] for u in freeze if u["horizon"] == 12 and u["dag_node"] == "freeze_folds:h12") == ["2023Q2"]
    assert states["freeze_weeks:h12"]["state"] == "QUEUED"
    with pytest.raises(LookupError):
        K.swarm_context(out, 12)


def test_freeze_completion_builds_swarm_context(tmp_path):
    cfg = F.private_config(tmp_path, frozen=False)
    out = tmp_path / "campaign"
    P.build(cfg, out)
    K.start(out, roles=[], systemctl=Recorder())
    led = L.Ledger(out / "state" / "ledger.sqlite3")
    for unit in [u for u in led.units() if u["dag_node"] == "freeze_folds:h12"]:
        c = led.claim(unit["fit_id"], "w", "preferred", "b", "cpu", 60, exclusive_host=False)
        h, fold = unit["horizon"], unit["fold"]
        bind = F.binding(tmp_path / "frozen", f"{h}-{fold}", h, 1)
        led.complete(unit["fit_id"], c["token"], K.freeze_receipt_document(unit, json.loads(Path(bind["path"]).read_text()), bind))
    K.tick(out, now=3000.0)
    states = {n["node_id"]: n["state"] for n in led.dag_nodes()}
    assert states["freeze_folds:h12"] == "COMPLETE" and states["search:h12"] == "QUEUED"
    swarm = K.swarm_context(out, 12)
    assert [f["fold"] for f in swarm["folds"]] == ["2023Q1", "2023Q2"]
    assert swarm["plan_sha256"] == json.loads(Path(json.loads(Path(cfg).read_text())["search_plan"]["path"]).read_text())["plan_sha256"]


def test_declared_clock_assumption_folds_are_runnable_and_keep_their_ids(tmp_path):
    cfg = F.private_config(tmp_path, dca=("2023Q1",))
    out = tmp_path / "campaign"
    payload = P.build(cfg, out)
    assert payload["folds"] == ["2023Q1_DCA", "2023Q2"]
    assert payload["fold_map"] == {"2023Q1": "2023Q1_DCA", "2023Q2": "2023Q2"}
    K.start(out, roles=[], systemctl=Recorder())
    swarm = K.swarm_context(out, 12)
    assert [f["fold"] for f in swarm["folds"]] == ["2023Q1_DCA", "2023Q2"]
    complete_candidate(out, 12, search.default_genome(12), 0.9)
    complete_candidate(out, 12, other_genome(12), 0.8)
    led = L.Ledger(out / "state" / "ledger.sqlite3")
    led.dag_event("search:h12", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    K.tick(out, now=3000.0)
    node = {n["node_id"]: n for n in led.dag_nodes()}["search:h12"]
    assert node["state"] == "COMPLETE" and node["summary"]["finalists"][0]["fitness"] == pytest.approx(0.8)
    assert {u["fold"] for u in led.units(dag_node="search:h12")} == {"2023Q1_DCA", "2023Q2"}


def test_c_origins_travel_to_the_aggregator(tmp_path, monkeypatch):
    cfg_path = F.private_config(tmp_path)
    cfg = json.loads(Path(cfg_path).read_text())
    units = [dict(unit_id=f, support={"MANAGEMENT": {str(h): {"TRAIN": {"origins_sha256": common.sha256_json([f, h, "T"])},
                                                           "INNER": {"origins_sha256": common.sha256_json([f, h, "I"])}}
                                          for h in (12, 72)}}) for f in ("2023Q1", "2023Q2")]
    cfg["weekly_plan"] = F.pin(tmp_path / "inputs" / "c_plan.json", dict(fit_role="MANAGEMENT", units=units))
    out = tmp_path / "campaign"
    P.build(F.pin(cfg_path, cfg)["path"], out)
    K.start(out, roles=[], systemctl=Recorder())
    swarm = K.swarm_context(out, 12)
    assert swarm["folds"][0]["origins_sha256"] == dict(train=common.sha256_json(["2023Q1", 12, "T"]),
                                                       inner=common.sha256_json(["2023Q1", 12, "I"]))
    seen = []

    def spy(plan, receipts, supports, **kw):
        seen.append(kw)
        return B._reference_score_folds(plan, receipts, supports, **kw)
    monkeypatch.setattr(B, "resolve_score_folds", lambda: spy)
    complete_candidate(out, 12, search.default_genome(12), 0.9)
    L.Ledger(out / "state" / "ledger.sqlite3").dag_event("search:h12", "SEARCH_TERMINAL", {"stop_reason": "patience"})
    K.tick(out, now=3000.0)
    assert seen and seen[0]["origins_sha256"]["2023Q2"]["inner"] == common.sha256_json(["2023Q2", 12, "I"])
    assert seen[0]["genotype_canonical_sha256"] == search.effective_identity(search.default_genome(12))
    assert seen[0]["horizon_hours"] == 12


def test_production_plan_requires_c_plan_for_origins(tmp_path):
    cfg = F.private_config(tmp_path, mode="production", launcher=dict(crispdm_run="/usr/bin/true"))
    with pytest.raises(ValueError, match="origins"):
        P.build(cfg, tmp_path / "c")


def test_e_score_folds_signature_when_installed():
    try:
        from tools.i6n_search_plan import score_folds
    except ImportError:
        pytest.skip("lane E module not merged in this checkout")
    import inspect
    params = inspect.signature(score_folds).parameters
    assert {"genotype_canonical_sha256", "horizon_hours", "origins_sha256"} <= set(params)
