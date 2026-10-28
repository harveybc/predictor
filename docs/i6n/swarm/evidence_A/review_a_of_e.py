"""Lane A review of lane E: dry contract checks, no fits, no data beyond invented fixtures."""
import copy, json, random, sys, tempfile, time
from pathlib import Path

from tests._i6n_trajectory_fixture import build, write_json
from tools import i6n_search as v1, i6n_search_space as space, i6n_search_plan as planmod
from tools import i6n_historical_evaluator as ev, i6n_historical_cell as cell
from tools.i6n_contracts.temporal_ops import reduction_plan
from tools.i6n_model import normalize_config

out = {}
t0 = time.monotonic()
plan = json.loads(Path("docs/i6n/swarm/I6N_SEARCH_PLAN_v1_1.json").read_text())
planmod.verify_plan(plan)
out["plan_sha256"] = plan["plan_sha256"]
out["plan_training_matches_A"] = {k: plan["training"][k] for k in ("max_epochs", "patience", "min_delta", "seed")} == ev.PLAN_STOPPING

# 1. domain exactness vs plan section 3 (independent recomputation)
dom = {}
for h in space.HORIZONS:
    Ls = sorted({max(24, m*h) for m in (1, 2, 4, 6)})
    assert list(space.history_domain(h)) == Ls, h
    for L in Ls:
        assert list(space.delta_x_domain(h, L)) == [d for d in range(1, h+1) if h % d == 0 and L % d == 0]
    assert list(space.trajectory_k_domain(h)) == [k for k in (2, 3, 4) if h % k == 0]
    dom[h] = dict(L=Ls, k=list(space.trajectory_k_domain(h)),
                  max_n=max(1 + L//dx for L in Ls for dx in space.delta_x_domain(h, L)))
out["domain"] = dom
assert max(d["max_n"] for d in dom.values()) == 433

tmp = Path(tempfile.mkdtemp(dir=sys.argv[1]))
rng = random.Random(6102026)
checked = dict(gen0=0, sampled=0, mutated=0)
problems = []

def check_against_A(g, base, tag):
    eff = space.evaluator_config(g)
    der = space.derived(g)
    cfg = ev.candidate_config(base, eff)                    # A's mapping (raises on mismatch)
    cell.authenticated(cfg, trajectory=True)               # A's cell contract incl. equal head weights
    genome = {k: eff[k] for k in v1._GENOME_KEYS}
    if v1.effective_identity(genome) != space.genotype_sha256(g):
        problems.append((tag, "genotype sha mismatch"))
    if ev.effective_config_sha256(eff) != space.effective_config_sha256(g):
        problems.append((tag, "effective_config_sha256 mismatch"))
    if list(cell.output_offsets(cfg)) != der["output_offsets_hours"]:
        problems.append((tag, "offsets mismatch"))
    if (cfg["samples"], cfg["spacing_hours"]) != (der["n_samples"], der["delta_x_hours"]):
        problems.append((tag, "n/dx mismatch"))
    plan_r = reduction_plan(cfg["samples"], cfg["model"]["reducer_strides"])  # t kept, nonempty
    if len(plan_r[-1]) != der["final_steps"] or plan_r[-1][-1] != cfg["samples"] - 1:
        problems.append((tag, "final steps / t"))
    normalize_config(cfg["model"])
    back = space.canonicalize(space.from_evaluator_genome(genome))
    if back != g:
        problems.append((tag, "v1->v2 roundtrip not injective"))
    return cfg

gen0 = {}
for h in space.HORIZONS:
    hdir = tmp / f"h{h}"; hdir.mkdir()
    seed = space.seed_genotype(h)
    base = build(hdir, horizon=h, samples=1 + seed["history_hours"]//seed["delta_x_hours"],
                 spacing=seed["delta_x_hours"])
    pop = planmod.initial_population(plan, h, ["S1"])
    first = pop[0]
    gen0[h] = dict(n=len(pop), seed_is_first=first == seed,
                   seed=dict(entrance=first["entrance"], width=first["width"], reducers=first["reducer_filters"],
                             strides=first["reducer_strides"], adapter=first["adapter"],
                             supervision=first["supervision"], L=first["history_hours"], dx=first["delta_x_hours"]),
                   classes=sorted({space.class_of(g) for g in pop}),
                   trajectories=sum(g["supervision"] == "trajectory" for g in pop))
    for g in pop:
        check_against_A(g, base, f"gen0 h{h}"); checked["gen0"] += 1
    for _ in range(60):
        g = space.sample_genotype(h, rng, classes=["S1", "S2", "S3"])
        check_against_A(g, base, f"sample h{h}"); checked["sampled"] += 1
        c = space.mutate(g, rng, classes=["S1", "S2", "S3"])
        if c is not None:
            check_against_A(c, base, f"mutate h{h}"); checked["mutated"] += 1
    gen0[h]["base"] = base
out["gen0"] = {h: {k: v for k, v in d.items() if k != "base"} for h, d in gen0.items()}
out["checked"] = checked

# 1b. exact 433 sample genotype
g433 = space.canonicalize(dict(space.seed_genotype(72), history_hours=432, delta_x_hours=1, width=8))
cfg433 = check_against_A(g433, gen0[72]["base"], "433")
out["n433"] = dict(samples=cfg433["samples"], class_=space.class_of(g433))

# 2. inactive canonicalization and endpoint alias
s = space.seed_genotype(12)
alias = space.canonicalize(dict(s, supervision="trajectory", trajectory_k=1))
inactive = space.canonicalize(dict(s, transformer_blocks=2, adapter_width=32, trajectory_k=3))
out["endpoint_alias_same_identity"] = space.genotype_sha256(alias) == space.genotype_sha256(s) and alias["trajectory_k"] is None
out["inactive_same_identity"] = space.genotype_sha256(inactive) == space.genotype_sha256(s)
active_changes = dict(history_hours=48, delta_x_hours=3, input_transform="difference", supervision="trajectory",
                      width=16, reducer_filters=[8], reducer_strides=[2], adapter="lstm")
diff = {}
for k, v in active_changes.items():
    raw = dict(s, **{k: v})
    if k == "supervision": raw["trajectory_k"] = 2
    if k == "adapter": raw["adapter_width"] = 8
    diff[k] = space.genotype_sha256(space.canonicalize(raw)) != space.genotype_sha256(s)
out["active_genes_change_identity"] = diff

# 3. typed infeasibility, no repair, input untouched
cases = {
    "dx_not_divisor": dict(s, delta_x_hours=5),
    "k_out_of_domain": dict(s, supervision="trajectory", trajectory_k=6),
    "dy_not_integer": dict(space.seed_genotype(5), supervision="trajectory", trajectory_k=2),
    "filters_increasing": dict(s, reducer_filters=[4, 8], reducer_strides=[1, 1]),
    "stride_3": dict(s, reducer_strides=[3]),
    "final_steps": dict(space.seed_genotype(72), history_hours=72, delta_x_hours=36, reducer_filters=[4, 4],
                        reducer_strides=[2, 2]),
    "multi": dict(s, delta_x_hours=7, width=12, adapter="gru"),
}
typed = {}
for name, raw in cases.items():
    before = copy.deepcopy(raw)
    try:
        space.canonicalize(raw)
        typed[name] = "ACCEPTED"
    except space.Infeasible as exc:
        typed[name] = [r["code"] for r in exc.reasons]
    assert raw == before, name
out["typed_infeasibility"] = typed

# 4. score_folds over four folds with A-shaped receipts
folds = plan["folds"]["fold_ids"]
def support(i):
    return dict(counts=dict(train=100+i, inner=10+i), naive_sums=dict(train=1.+i, inner=.5+i),
                support_digest=dict(train=f"{i:064x}", inner=f"{i+10:064x}"))
def receipt(i, err, geno="a"*64, fold=None):
    sp = support(i)
    r = dict(sp, status="ok", error_sums=dict(train=err*sp["naive_sums"]["train"], inner=err*sp["naive_sums"]["inner"]),
             metadata=dict(candidate_receipt=dict(genotype_canonical_sha256=geno, fold=fold or folds[i])))
    r["fitness"] = v1._score(r, sp)[0]
    return r
supports = {f: support(i) for i, f in enumerate(folds)}
res = planmod.score_folds(plan, {f: receipt(i, .9 + i/100) for i, f in enumerate(folds)}, supports)
out["score_folds_mean"] = res["fitness"]
try:
    planmod.score_folds(plan, {f: receipt(i, 1.) for i, f in enumerate(folds[:3])}, supports)
    out["score_folds_three"] = "ACCEPTED"
except planmod.Incomplete:
    out["score_folds_three"] = "INCOMPLETE"
mixed = {f: receipt(i, 1., geno=("a" if i else "b")*64, fold=folds[(i+1) % 4]) for i, f in enumerate(folds)}
try:
    planmod.score_folds(plan, mixed, supports)
    out["score_folds_mixed_candidates_and_fold_labels"] = "ACCEPTED"
except ValueError as exc:
    out["score_folds_mixed_candidates_and_fold_labels"] = f"REFUSED: {exc}"

# 5. A's freeze + load_request on two horizons (dry: no fit)
dry = {}
for h in (12, 72):
    base = gen0[h]["base"]
    path = tmp / f"h{h}" / "base.json"
    write_json(path, base)
    binding = ev.freeze(path, tmp / f"h{h}" / "bridge", experiment="review", fold="2023Q4")
    ok = 0
    for g in planmod.initial_population(plan, h, ["S1"]):
        req = dict(config=space.evaluator_config(g), generation=0, context=binding["i6n_evaluation_context"],
                   support=binding["i6n_support"])
        ev.load_request(req); ok += 1
    dry[h] = ok
out["freeze_load_request_gen0"] = dry
out["problems"] = problems
out["elapsed_seconds"] = round(time.monotonic() - t0, 1)
print(json.dumps(out, indent=1, default=str))
