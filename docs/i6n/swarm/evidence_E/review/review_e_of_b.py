"""Lane E review of lane B: reproducers on B@2b0b6e25 overlaid with E@5e400352 modules.

Fabricated fixtures only; no fit, no data, no TEST, no services. Run from the overlay root.
"""
import copy, json, sys, tempfile
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

folds = [dict(fold=f, context={"fold": f, "fixture": "review"}, support=support(i), binding_sha256="0" * 64,
              origins_sha256=dict(train=format(i + 1, "x") * 64, inner=format(i + 9, "x") * 64))
         for i, f in enumerate(FOLDS)]
context = dict(schema=P.CONTEXT_SCHEMA, experiment="review", plan_sha256=e_plan["plan_sha256"], horizon=12,
               dag_node="search:h12", fixture=True, budget_new_candidates=24,
               classes=[dict(class_id=c["class_id"], max_samples=c["max_samples"], max_width=c["max_width"],
                             min_delta_x_hours=c["min_delta_x_hours"]) for c in e_plan["classes"]], folds=folds)

# B1: B's rendered DOIN optimization_config drives E's plugin down the LEGACY v1 path.
template = dict(campaign_id="rev", node_label="x", port=1, data_dir="/nonexistent", bootstrap_peers=[],
                shared_min_peers=2, claim_timeout=600, predictor_root=".", node_config="/nonexistent",
                search=dict(population_size=8, generations=3, new_candidates_per_horizon=24, mutation=0.8,
                            immigration=0.2, tolerance=1e-4, patience=3))
doin = P.render_doin_config(template, {"12": context})
oc = doin["domains"][0]["optimization_config"]
cfg = {k: v for k, v in oc.items() if k not in ("predictor_root", "i6n_swarm")}
cfg["i6n_evaluator"] = lambda request: None  # never called here
pop = Plugin().create_shared_population(8, cfg, seed=6102026)
schemas = sorted({p["schema"] for p in pop["population"]})
seed_id = space.genotype_sha256(space.seed_genotype(12))
ids = [p["identity"] for p in pop["population"]]
classes = []
for p in pop["population"]:
    chk = space.check(space.from_evaluator_genome(p["genome"]))
    classes.append(space.class_of(chk["genotype"]) if chk["status"] == "FEASIBLE" else "OUT_OF_V2:" + ",".join(r["code"] for r in chk["reasons"]))
out["B1_rendered_config_has_i6n_plan"] = "i6n_plan" in oc
out["B1_population_wire_schemas"] = schemas
out["B1_minimal_seed_in_initial_population"] = seed_id in ids
out["B1_initial_population_v2_classes"] = classes

# B2: wrapper reproduction seed vs E v2 path.
cfg2 = dict(i6n_plan=e_plan, i6n_horizon=12, i6n_admitted_classes=["S1"],
            i6n_folds={f["fold"]: {"i6n_support": f["support"], "i6n_evaluation_context": f["context"]} for f in folds},
            i6n_evaluator=lambda r: None)
p2 = Plugin()
st = p2.create_shared_population(8, cfg2, seed=6102026)
evaluated = [dict(i, fitness=1.0 + k / 100) for k, i in enumerate(st["population"])]
wrapper_seed = 6102026 + 0 + 1  # doin-plugins 9564c6f reproduce_shared: I6N_SEED + generation + 1
try:
    p2.reproduce_shared(evaluated, 0, wrapper_seed, cfg2, st["innovation_tracker"], st["stage_schedule"],
                        st["param_defaults"], 0, 0)
    out["B2_v2_reproduce_with_wrapper_seed"] = "ACCEPTED"
except ValueError as exc:
    out["B2_v2_reproduce_with_wrapper_seed"] = f"REFUSED: {exc}"

# B3: Bridge without plan (as constructed by the doin-plugins wrapper) + E's real score_folds.
genome = space.to_evaluator_genome(space.seed_genotype(12))
config = v1.effective_config(genome)
def receipt(f, i, geno):
    r = copy.deepcopy(support(i))
    r.update(status="ok", error_sums=dict(train=9.0, inner=3.9), metadata=dict(candidate_receipt=dict(
        genotype_canonical_sha256=geno, fold=f, horizon_hours=12)))
    return r
gid = space.genotype_sha256(space.seed_genotype(12))
receipts = {f: receipt(f, i, gid) for i, f in enumerate(FOLDS)}
supports = {f: support(i) for i, f in enumerate(FOLDS)}
fabricated = dict(folds=dict(fold_ids=FOLDS), fitness=dict(aggregation="unweighted_mean_over_folds"), horizons_hours=[12])
try:
    BR.score_candidate(sp.score_folds, fabricated, receipts, supports, gid, 12, None)
    out["B3_bridge_default_plan_with_E_score_folds"] = "ACCEPTED"
except Exception as exc:
    out["B3_bridge_default_plan_with_E_score_folds"] = f"REFUSED {type(exc).__name__}: {exc}"
# with the real plan it works (control)
out["B3_control_with_real_plan"] = BR.score_candidate(sp.score_folds, e_plan, receipts, supports, gid, 12, None)["origins_binding"]

# B4: origins missing on one fold silently degrade to NOT_CHECKED; the fallback aggregator ranks without bindings.
origins = {f["fold"]: f["origins_sha256"] for f in folds[:3]}
partial = origins if len(origins) == len(supports) else None
out["B4_partial_origins_binding"] = BR.score_candidate(sp.score_folds, e_plan, receipts, supports, gid, 12, partial)["origins_binding"]
foreign = {f: receipt(f, i, "b" * 64) for i, f in enumerate(FOLDS)}
r = BR._reference_score_folds(e_plan, foreign, supports, genotype_canonical_sha256=gid, horizon_hours=12)
out["B4_reference_fallback_on_foreign_receipts"] = dict(fitness=r["fitness"], aggregator=r["aggregator"])

# B5: ledger.complete trusts the wrapper keys; inner receipts of another candidate are stored as this fit's terminal.
with tempfile.TemporaryDirectory() as tmp:
    led = L.Ledger(Path(tmp) / "l.sqlite3")
    fold = folds[0]
    fit = BR.fold_fit_id(context, fold, config)
    wire = dict(schema="i6n-shared-deap-v1", genome=genome, identity=gid, context="c", generation=0)
    led.ensure_unit(fit, BR.unit_spec(context, fold, wire, config))
    claim = led.claim(fit, "o", "h", "b", "cpu_fixture", 60)
    doc = dict(fit_id=fit, fold=fold["fold"], experiment="review", effective_config_sha256=common.sha256_json(config),
               genotype_canonical_sha256=gid, shared_receipt=receipt(FOLDS[1], 1, "b" * 64),
               candidate_receipt=dict(genotype_canonical_sha256="b" * 64, fold=FOLDS[1], horizon_hours=24))
    try:
        out["B5_ledger_complete_foreign_inner_receipt"] = led.complete(fit, claim["token"], doc)
    except Exception as exc:
        out["B5_ledger_complete_foreign_inner_receipt"] = f"REFUSED {type(exc).__name__}: {exc}"

# B6: fit_id agreement B common.fit_id vs E space.fit_id
b_fit = BR.fold_fit_id(context, folds[0], config)
e_fit = space.fit_id(space.seed_genotype(12), dict(experiment="review", fold_id=folds[0]["fold"],
                     context_sha256=common.sha256_json(folds[0]["context"]),
                     support_sha256=common.sha256_json(folds[0]["support"])))
out["B6_fit_id_B_equals_E"] = b_fit == e_fit
print(json.dumps(out, indent=1, default=str))
