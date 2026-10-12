"""Lane E review of lane B, round 2: the round-1 reproducers adapted to B.3 (predictor f7a05838,
doin-plugins e24cf6b). Run from B's predictor worktree. Invented fixtures; no fit, data, TEST or services."""
import copy, json, tempfile
from pathlib import Path

out = {}
from tools.i6n_swarm import plan as P, bridge as BR, ledger as L, common
from tools import i6n_search_plan as sp, i6n_search_space as space, i6n_search as v1
from optimizer_plugins.i6n_optimizer import Plugin

e_plan = json.loads(Path("docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json").read_text())
FOLDS = ["2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4"]

def support(i):
    return dict(counts=dict(train=100 + i, inner=20 + i), naive_sums=dict(train=10.0, inner=4.0),
                support_digest=dict(train=format(i + 1, "x") * 64, inner=format(i + 6, "x") * 64))

ORIG = {f: dict(train=format(i + 1, "x") * 64, inner=format(i + 9, "x") * 64) for i, f in enumerate(FOLDS)}
folds = [dict(fold=f, context={"fold": f, "fixture": "review"}, support=support(i), binding_sha256="0" * 64,
              origins_sha256=ORIG[f]) for i, f in enumerate(FOLDS)]
context = dict(schema=P.CONTEXT_SCHEMA, experiment="review", plan_sha256=e_plan["plan_sha256"], horizon=12,
               dag_node="search:h12", fixture=False, budget_new_candidates=24,
               classes=[dict(class_id=c["class_id"], max_samples=c["max_samples"], max_width=c["max_width"],
                             min_delta_x_hours=c["min_delta_x_hours"]) for c in e_plan["classes"]], folds=folds)

# B1 (adapted): B.3 template carries the sealed plan and admitted classes.
template = dict(campaign_id="rev", node_label="x", port=1, data_dir="/nonexistent", bootstrap_peers=[],
                shared_min_peers=2, claim_timeout=600, predictor_root=".", node_config="/nonexistent",
                search=dict(population_size=8, generations=3, new_candidates_per_horizon=24, mutation=0.8,
                            immigration=0.2, tolerance=1e-4, patience=3),
                e_plan=e_plan, admitted_classes=["S1"])
oc = P.render_doin_config(template, {"12": context})["domains"][0]["optimization_config"]
cfg = {k: v for k, v in oc.items() if k not in ("predictor_root", "i6n_swarm")}
cfg["i6n_evaluator"] = lambda request: None
p = Plugin()
pop = p.create_shared_population(8, cfg, seed=6102026)
seed_id = space.genotype_sha256(space.seed_genotype(12))
out["B1_rendered_config_has_i6n_plan"] = "i6n_plan" in oc
out["B1_population_wire_schemas"] = sorted({x["schema"] for x in pop["population"]})
out["B1_minimal_seed_in_initial_population"] = seed_id in [x["identity"] for x in pop["population"]]
out["B1_initial_population_classes"] = [space.class_of(x["genotype"]) for x in pop["population"]]
out["B1_origins_passed_per_fold"] = sorted(k for k, v in oc["i6n_folds"].items() if "i6n_origins_sha256" in v)

# B2 (adapted): reproduce with the seed the wrapper now passes (I6N_SEED, doin-plugins e24cf6b).
import re
wrapper = Path(__import__("doin_plugins.predictor.optimizer", fromlist=["x"]).__file__).read_text()
out["B2_wrapper_passes_campaign_seed_unchanged"] = bool(re.search(r"evaluated_pop, generation, I6N_SEED, opt_config", wrapper))
evaluated = [dict(i, fitness=1.0 + k / 100) for k, i in enumerate(pop["population"])]
try:
    r = p.reproduce_shared(evaluated, 0, 6102026, cfg, pop["innovation_tracker"], pop["stage_schedule"],
                           pop["param_defaults"], 0, 0)
    out["B2_v2_reproduce_with_wrapper_seed"] = f"ACCEPTED generation={r['generation']} population={len(r['population'])}"
except ValueError as exc:
    out["B2_v2_reproduce_with_wrapper_seed"] = f"REFUSED: {exc}"

# B3 (adapted): production Bridge without the sealed plan, and with it.
gid = space.genotype_sha256(space.seed_genotype(12))
genome = space.to_evaluator_genome(space.seed_genotype(12))
config = space.evaluator_config(space.seed_genotype(12))
def receipt(f, i, geno, rows=True):
    r = copy.deepcopy(support(i))
    cr = dict(genotype_canonical_sha256=geno, fold=f, horizon_hours=12)
    if rows:
        cr["rows"] = [dict(head_offset_hours=12, head_role="ENDPOINT", comparison=True, split=s.upper(),
                           support_sha256=ORIG[f][s]) for s in ("train", "inner")]
    r.update(status="ok", error_sums=dict(train=9.0, inner=3.9), metadata=dict(candidate_receipt=cr))
    return r
receipts = {f: receipt(f, i, gid) for i, f in enumerate(FOLDS)}
supports = {f: support(i) for i, f in enumerate(FOLDS)}
node = dict(fixture=False, lease=dict(ttl_seconds=60, renew_seconds=10, max_wait_seconds=0, poll_seconds=1),
            device=dict(cls="rtx5090", cuda_visible_devices="GPU-x"), owner="o", host_key="h",
            evaluator_argv=["false"], logs_dir=tempfile.mkdtemp())
wire = dict(schema="i6n-shared-deap-v2", genotype=space.seed_genotype(12), genome=genome, identity=gid,
            context="c", generation=0)
with tempfile.TemporaryDirectory() as tmp:
    led = L.Ledger(Path(tmp) / "l.sqlite3")
    try:
        BR.Bridge(node, led, plan=None).evaluate_wire(wire, 0, context)
        out["B3_production_bridge_without_plan"] = "EVALUATED"
    except Exception as exc:
        out["B3_production_bridge_without_plan"] = f"REFUSED {type(exc).__name__}: {str(exc)[:120]}"
out["B3_real_plan_production_score"] = BR.score_candidate(sp.score_folds, e_plan, receipts, supports, gid, 12,
                                                          ORIG, production=True)["origins_binding"]

# B4 (adapted): partial origins and the reference aggregator in production.
def attempt(label, **kw):
    try:
        BR.score_candidate(**kw)
        out[label] = "ACCEPTED"
    except ValueError as exc:
        out[label] = f"REFUSED: {str(exc)[:110]}"
attempt("B4_partial_origins_production", score_folds=sp.score_folds, plan=e_plan, receipts=receipts,
        supports=supports, genotype=gid, horizon=12, origins={f: ORIG[f] for f in FOLDS[:3]}, production=True)
attempt("B4_partial_origins_fixture", score_folds=sp.score_folds, plan=e_plan, receipts=receipts,
        supports=supports, genotype=gid, horizon=12, origins={f: ORIG[f] for f in FOLDS[:3]}, production=False)
foreign = {f: receipt(f, i, "b" * 64) for i, f in enumerate(FOLDS)}
attempt("B4_reference_scorer_production_foreign", score_folds=BR._reference_score_folds, plan=e_plan,
        receipts=foreign, supports=supports, genotype=gid, horizon=12, origins=ORIG, production=True)
attempt("B4_no_origins_production", score_folds=sp.score_folds, plan=e_plan, receipts=receipts,
        supports=supports, genotype=gid, horizon=12, origins=None, production=True)
out["B4_resolve_score_folds_is_E"] = BR.resolve_score_folds() is sp.score_folds

# B5 (adapted): ledger with inner receipts of another candidate/fold/horizon.
with tempfile.TemporaryDirectory() as tmp:
    led = L.Ledger(Path(tmp) / "l.sqlite3")
    fold = folds[0]
    fit = BR.fold_fit_id(context, fold, config)
    led.ensure_unit(fit, BR.unit_spec(context, fold, wire, config))
    claim = led.claim(fit, "o", "h", "b", "cpu_fixture", 60)
    def doc(shared, inner):
        return dict(fit_id=fit, fold=fold["fold"], experiment="review", effective_config_sha256=common.sha256_json(config),
                    genotype_canonical_sha256=gid, shared_receipt=shared, candidate_receipt=inner)
    cases = {
        "foreign_inner": doc(receipt(FOLDS[1], 1, "b" * 64), dict(genotype_canonical_sha256="b" * 64, fold=FOLDS[1], horizon_hours=24)),
        "own_inner_foreign_support": doc(receipt(FOLDS[0], 1, gid), dict(
            genotype_canonical_sha256=gid, fold=FOLDS[0], horizon_hours=12, experiment="review",
            effective_config_sha256=common.sha256_json(config), seed=6102026)),
    }
    for name, d in cases.items():
        try:
            out[f"B5_ledger_{name}"] = led.complete(fit, claim["token"], d)
        except Exception as exc:
            out[f"B5_ledger_{name}"] = f"REFUSED {type(exc).__name__}: {str(exc)[:90]}"
    # own inner receipt, consistent metadata, but the support of ANOTHER fold
    own_inner = dict(genotype_canonical_sha256=gid, fold=FOLDS[0], horizon_hours=12, experiment="review",
                     effective_config_sha256=common.sha256_json(config), seed=6102026)
    other_support = receipt(FOLDS[0], 1, gid)
    other_support["metadata"]["candidate_receipt"] = own_inner
    try:
        out["B5_ledger_own_inner_other_fold_support"] = led.complete(fit, claim["token"], doc(other_support, own_inner))
    except Exception as exc:
        out["B5_ledger_own_inner_other_fold_support"] = f"REFUSED {type(exc).__name__}: {str(exc)[:90]}"
    good_shared = receipt(FOLDS[0], 0, gid)
    good_inner = dict(good_shared["metadata"]["candidate_receipt"], experiment="review",
                      effective_config_sha256=common.sha256_json(config), seed=6102026)
    good_shared["metadata"]["candidate_receipt"] = good_inner
    try:
        out["B5_ledger_own_receipt_control"] = led.complete(fit, claim["token"], doc(good_shared, good_inner))
    except Exception as exc:
        out["B5_ledger_own_receipt_control"] = f"REFUSED {type(exc).__name__}: {str(exc)[:90]}"

out["B6_fit_id_B_equals_E"] = BR.fold_fit_id(context, folds[0], config) == space.fit_id(
    space.seed_genotype(12), dict(experiment="review", fold_id=folds[0]["fold"],
                                  context_sha256=common.sha256_json(folds[0]["context"]),
                                  support_sha256=common.sha256_json(folds[0]["support"])))
print(json.dumps(out, indent=1, default=str))
