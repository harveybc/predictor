"""Regression tests for lane E's review of lane B (REVIEW_E_OF_B.md, E-B1..E-B9, risk 5).

Real lane E modules (merged), the sealed plan v1.2, fabricated receipts; no fit, no data.
"""
import copy
import json
from pathlib import Path

import pytest

from tests import i6n_swarm_fixtures as F
from tests.test_i6n_swarm_campaign import Recorder, complete_candidate, other_genome
from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_search as v1, i6n_search_plan as sp, i6n_search_space as space
from tools.i6n_swarm import bridge as B, collect as C, common, coordinator as K, ledger as L, plan as P

E_PLAN = json.loads(Path("docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json").read_text())
FOLDS = ["2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4"]


def support(i):
    return dict(counts=dict(train=100 + i, inner=20 + i), naive_sums=dict(train=10.0, inner=4.0),
                support_digest=dict(train=format(i + 1, "x") * 64, inner=format(i + 6, "x") * 64))


def origins(i):
    return dict(train=format(i + 1, "x") * 64, inner=format(i + 9, "x") * 64)


def context(with_origins=True):
    folds = [dict(fold=f, context={"fold": f, "fixture": "review"}, support=support(i), binding_sha256="0" * 64,
                  origins_sha256=origins(i) if with_origins else None) for i, f in enumerate(FOLDS)]
    return dict(schema=P.CONTEXT_SCHEMA, experiment="review", plan_sha256=E_PLAN["plan_sha256"], horizon=12,
                dag_node="search:h12", fixture=True, budget_new_candidates=24,
                classes=[dict(class_id=c["class_id"], max_samples=c["max_samples"], max_width=c["max_width"],
                              min_delta_x_hours=c["min_delta_x_hours"]) for c in E_PLAN["classes"]], folds=folds)


def receipt(fold, i, genotype, h=12, config=None, rows=True):
    r = copy.deepcopy(support(i))
    cr = dict(schema_version="i6n.candidate_receipt.v1", experiment="review", fold=fold, horizon_hours=h,
              genotype_canonical_sha256=genotype, seed=6102026,
              effective_config_sha256=common.sha256_json(config) if config else None)
    if rows:
        cr["rows"] = [dict(head_offset_hours=h, head_role="ENDPOINT", comparison=True, split=s.upper(),
                           support_sha256=origins(i)[s]) for s in ("train", "inner")]
    r.update(status="ok", error_sums=dict(train=9.0, inner=3.9), metadata=dict(candidate_receipt=cr))
    return r


def template():
    return dict(campaign_id="rev", node_label="x", port=1, data_dir="/nonexistent", bootstrap_peers=[],
                shared_min_peers=2, claim_timeout=600, predictor_root=".", node_config="/nonexistent",
                search=dict(population_size=8, generations=3, new_candidates_per_horizon=24, mutation=0.8,
                            immigration=0.2, tolerance=1e-4, patience=3), e_plan=E_PLAN, admitted_classes=["S1"])


# ---------------------------------------------------------------- E-B1
def test_e_b1_rendered_doin_config_runs_e_v2_path_with_minimal_seed():
    oc = P.render_doin_config(template(), {"12": context()})["domains"][0]["optimization_config"]
    assert oc["i6n_plan"]["plan_sha256"] == E_PLAN["plan_sha256"]
    assert oc["i6n_admitted_classes"] == ["S1"] and "i6n_support" not in oc
    assert oc["i6n_swarm"]["context"]["plan_sha256"] == E_PLAN["plan_sha256"]
    cfg = {k: v for k, v in oc.items() if k not in ("predictor_root", "i6n_swarm")}
    pop = Plugin().create_shared_population(8, cfg, seed=6102026)
    assert {p["schema"] for p in pop["population"]} == {"i6n-shared-deap-v2"}
    assert space.genotype_sha256(space.seed_genotype(12)) in [p["identity"] for p in pop["population"]]
    assert {space.class_of(p["genotype"]) for p in pop["population"]} == {"S1"}


# ---------------------------------------------------------------- E-B3 / E-B4
def _node(fixture):
    return dict(owner="o", host_key="h", device=dict(cls="rtx5090", cuda_visible_devices="GPU-test"),
                fixture=fixture, exclusive_host=True, evaluator_argv=["false"], workdir=".", launcher=None,
                class_caps={}, lease=dict(ttl_seconds=5, renew_seconds=1, max_wait_seconds=1, poll_seconds=0.1),
                logs_dir="/nonexistent")


def test_e_b3_production_bridge_refuses_without_the_sealed_plan(tmp_path):
    bridge = B.Bridge(_node(False), L.Ledger(tmp_path / "l.sqlite3"))
    genome = space.to_evaluator_genome(space.seed_genotype(12))
    wire = dict(schema="i6n-shared-deap-v2", genotype=space.seed_genotype(12), genome=genome,
                identity=space.genotype_sha256(space.seed_genotype(12)), context="c" * 64, generation=0)
    with pytest.raises(v1.ScoreRejected, match="sealed"):
        bridge.evaluate_wire(wire, 0, context())


def test_e_b3_real_plan_scores_through_e():
    gid = space.genotype_sha256(space.seed_genotype(12))
    receipts = {f: receipt(f, i, gid) for i, f in enumerate(FOLDS)}
    supports = {f: support(i) for i, f in enumerate(FOLDS)}
    allo = {f: origins(i) for i, f in enumerate(FOLDS)}
    out = B.score_candidate(sp.score_folds, E_PLAN, receipts, supports, gid, 12, allo, production=True)
    assert out["origins_binding"] == "VERIFIED" and out["aggregator"] == "E.score_folds"


def test_e_b4_production_refuses_partial_origins_and_the_reference_scorer():
    gid = space.genotype_sha256(space.seed_genotype(12))
    receipts = {f: receipt(f, i, gid) for i, f in enumerate(FOLDS)}
    supports = {f: support(i) for i, f in enumerate(FOLDS)}
    partial = {f: origins(i) for i, f in enumerate(FOLDS[:3])}
    with pytest.raises(ValueError, match="every fold"):
        B.score_candidate(sp.score_folds, E_PLAN, receipts, supports, gid, 12, partial)
    with pytest.raises(ValueError, match="origins"):
        B.score_candidate(sp.score_folds, E_PLAN, receipts, supports, gid, 12, None, production=True)
    allo = {f: origins(i) for i, f in enumerate(FOLDS)}
    with pytest.raises(ValueError, match="E.score_folds"):
        B.score_candidate(B._reference_score_folds, E_PLAN, receipts, supports, gid, 12, allo, production=True)
    assert B.resolve_score_folds() is sp.score_folds


# ---------------------------------------------------------------- E-B5
def test_e_b5_ledger_refuses_inner_receipt_of_another_candidate(tmp_path):
    led = L.Ledger(tmp_path / "l.sqlite3")
    ctx = context()
    fold = ctx["folds"][0]
    genome = space.to_evaluator_genome(space.seed_genotype(12))
    config = v1.effective_config(genome)
    gid = space.genotype_sha256(space.seed_genotype(12))
    wire = dict(schema="i6n-shared-deap-v1", genome=genome, identity=gid, context="c", generation=0)
    fit = B.fold_fit_id(ctx, fold, config)
    led.ensure_unit(fit, B.unit_spec(ctx, fold, wire, config))
    claim = led.claim(fit, "o", "h", "b", "cpu_fixture", 60)
    base = dict(fit_id=fit, fold=fold["fold"], experiment="review", effective_config_sha256=common.sha256_json(config),
                genotype_canonical_sha256=gid)
    foreign = dict(base, shared_receipt=receipt(FOLDS[1], 1, "b" * 64, config=config),
                   candidate_receipt=receipt(FOLDS[1], 1, "b" * 64, config=config)["metadata"]["candidate_receipt"])
    with pytest.raises(L.ReceiptMismatch):
        led.complete(fit, claim["token"], foreign)
    wrong_support = receipt(fold["fold"], 1, gid, config=config)  # support of fold 1 under fold 0
    wrong = dict(base, shared_receipt=wrong_support, candidate_receipt=wrong_support["metadata"]["candidate_receipt"])
    with pytest.raises(L.ReceiptMismatch):
        led.complete(fit, claim["token"], wrong)
    good = receipt(fold["fold"], 0, gid, config=config)
    assert led.complete(fit, claim["token"], dict(base, shared_receipt=good,
                                                   candidate_receipt=good["metadata"]["candidate_receipt"])) == "ACCEPTED"


# ---------------------------------------------------------------- E-B6 / E-B7 / E-B8
@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "resolve_score_folds", lambda: B._reference_score_folds)
    cfg = F.private_config(tmp_path)
    out = tmp_path / "campaign"
    P.build(cfg, out)
    K.start(out, roles=[], systemctl=Recorder())
    return out


def refusing(plan, receipts, supports, **kw):
    if kw["horizon_hours"] == 12:
        raise sp.ReceiptRefused("RECEIPT_ORIGINS_MISMATCH", "2023Q1", "fabricated")
    return B._reference_score_folds(plan, receipts, supports, **kw)


def test_e_b7_refused_candidate_does_not_abort_tick_for_other_horizons(root, monkeypatch):
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    for h in (12, 72):
        complete_candidate(root, h, v1.default_genome(h), 0.9)
        complete_candidate(root, h, other_genome(h), 0.8)
        led.dag_event(f"search:h{h}", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    monkeypatch.setattr(B, "resolve_score_folds", lambda: refusing)
    K.tick(root, now=3000.0)
    nodes = {n["node_id"]: n for n in led.dag_nodes()}
    assert nodes["search:h12"]["state"] == "FAILED" and "NO_COMPLETE_CANDIDATES" in nodes["search:h12"]["cause"]
    assert all(c["score_refused"].startswith("SCORE_REFUSED:RECEIPT_ORIGINS_MISMATCH")
               for c in nodes["search:h12"]["summary"]["candidates"])
    assert nodes["search:h72"]["state"] == "COMPLETE" and nodes["annual:h72"]["state"] == "QUEUED"


def test_e_b6_collect_never_publishes_a_refused_candidate_as_complete(root, monkeypatch):
    complete_candidate(root, 12, v1.default_genome(12), 0.9)
    monkeypatch.setattr(B, "resolve_score_folds", lambda: refusing)
    docs = []

    class Pub:
        def record_candidate(self, candidate, fold_receipts):
            docs.append(candidate)
            return dict(item_id="x" * 64)

        def record_decision(self, decision):
            return dict(item_id="y" * 64)
    C.collect(root, publisher=Pub())
    assert len(docs) == 1
    assert docs[0]["disposition"] == "FAILED" and docs[0]["fitness"] is None
    assert docs[0]["disposition_reason"].startswith("SCORE_REFUSED:RECEIPT_ORIGINS_MISMATCH")


def test_e_b8_archive_record_carries_inactive_genes_and_lineage(root):
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    g = space.seed_genotype(12)
    wire = complete_candidate(root, 12, space.to_evaluator_genome(g), 0.9, genotype_v2=g)
    state = dict(candidates=[dict(genotype_canonical_sha256=wire["identity"], generation=0, origin="seed",
                                  parent_genotype_sha256=None)])
    led.put_search_state("search:h12", 0, dict(innovation_tracker=dict(state=state)))
    docs = []

    class Pub:
        def record_candidate(self, candidate, fold_receipts):
            docs.append(candidate)
            return dict(item_id="x" * 64)

        def record_decision(self, decision):
            return dict(item_id="y" * 64)
    C.collect(root, publisher=Pub())
    doc = docs[0]
    assert doc["genotype"]["inactive"] == space.inactive_genes(g)
    assert set(doc["genotype"]["active"]) == set(space.effective_config(g)["active_genes"])
    assert doc["lineage"] == dict(generation=0, population_index=0, operator="seed", parents=[])


# ---------------------------------------------------------------- E-B9
def test_e_b9_non_admitted_class_is_typed_wait_resource(tmp_path):
    led = L.Ledger(tmp_path / "l.sqlite3")
    node = dict(_node(False), launcher=dict(crispdm_run="/nonexistent", label_prefix="x"), class_caps={})
    bridge = B.Bridge(node, led, plan=E_PLAN, score_folds=sp.score_folds)
    g = space.seed_genotype(12)
    wire = dict(schema="i6n-shared-deap-v2", genotype=g, genome=space.to_evaluator_genome(g),
                identity=space.genotype_sha256(g), context="c" * 64, generation=0)
    with pytest.raises(v1.ScoreRejected, match="CLASS_NOT_ADMITTED"):
        bridge.evaluate_wire(wire, 0, context())
    assert {u["state"] for u in led.units()} == {"WAIT_RESOURCE"}
    assert all(led.attempts(u["fit_id"]) == [] for u in led.units())


# ---------------------------------------------------------------- risk 5
def test_root_relative_bindings_resolve_on_a_host_with_another_root(tmp_path):
    shared = tmp_path / "hostA"
    pin = F.pin(shared / "bindings" / "b.json", dict(i6n_horizon=12))
    rel = common.relative_pin(pin, shared)
    assert rel == dict(relpath="bindings/b.json", sha256=pin["sha256"])
    other = tmp_path / "hostB"
    (other / "bindings").mkdir(parents=True)
    (other / "bindings" / "b.json").write_bytes(Path(pin["path"]).read_bytes())
    assert common.read_pinned(rel, "b", root=other) == dict(i6n_horizon=12)
    with pytest.raises(ValueError):
        common.read_pinned(rel, "b")  # a relative pin needs its host's root
    with pytest.raises(ValueError):
        common.relative_pin(dict(path="/elsewhere/x.json", sha256="0" * 64), shared)
