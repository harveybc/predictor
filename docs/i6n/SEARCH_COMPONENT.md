# I6N Search Component

Status, 2026-10-11: local component verified with **stub scores only**. No
historical measurements, TensorFlow training, GPU, TEST, deployment, commit,
or DOIN runtime execution. This is not production-ready or the final approved
DEAP search. Files owned by this change are only
`tools/i6n_search.py`, `tests/test_i6n_search.py`, and this document.

## Contract

`default_genome(h)` returns the complete mutable genome schema:

```python
{
    "horizon": 12, "multiplier": 1, "delta_x": 1,
    "input_transform": "identity", "output_mode": "endpoint", "delta_y": None,
    "core": {
        "entrance": "conv", "width": 16, "transformer_blocks": 0,
        "reducer_filters": [8, 4], "reducer_strides": [1, 1],
        "adapter": "none", "adapter_width": None, "n_horizons": 1,
    },
}
```

- Horizons: `2,3,4,5,6,12,18,24,36,48,60,72`, each searched separately.
- History in physical hours: `max(24, multiplier*h)`, multiplier in `1,2,4,6`.
  Equal histories collapse to the smallest equivalent multiplier.
- `delta_x` is a positive integer divisor of h AND history. Integer 1 is
  always available. Samples are `1 + history // delta_x`, including t.
  Input offsets run from `-history` through `0`; no compressed market clock.
- Endpoint supervision has exactly `[h]`, `delta_y=None`, and one output.
  A trajectory uses `[delta_y, ..., h]`, where `delta_y` independently divides
  h. A trajectory with `delta_y=h` canonicalizes to endpoint, never an alias
  deserving another fit. Input and output spacing need not coincide.
- Core validation delegates to the real `i6n_model.normalize_config` without
  importing TensorFlow. Conv deactivates blocks; adapter `none` deactivates
  adapter width; `n_horizons` is derived from output offsets. These inactive
  values are canonicalized. Missing/unknown schema keys still fail.
- Reducer filters cannot increase, including from entrance width to first
  reducer. One to three stages independently use strides 1/2. Right-anchored
  length is repeatedly `ceil(steps/stride)` and must end at >=2 before the
  model's last-tick head. Invalid active genes are rejected, not repaired.
- Sampling widths are 8/16/32/64/128; reducer widths also allow 4. Explicit
  genomes may use any width accepted by the core validator (1..128, even for
  Transformer entrances), subject to the additional search constraints.

`effective_config(genome)` adds `history_hours`, `samples`, `final_steps`,
`input_offsets`, `output_offsets`. Its `core` can be supplied to the model
factory with `samples` and `spacing_hours=delta_x`. It is an expanded evaluator
config, not the raw genome accepted by `normalize_genome`.

`effective_identity(genome)` hashes the canonical, versioned effective config.
This is a **config identity**, not a dataset/model/checkpoint/run identity.
Across runs the caller must also bind feature order, data/calendar/clock,
frozen comparison support, evaluator/transform versions, training settings
and model seed. No persistent deduplication or distributed claim store exists
here. Inactive genes cannot consume multiple fits within a Search instance.

`input_transform` choices are `identity` and `difference`. They are evaluator
instructions, not transformations implemented by this component. The intended
difference is backward first difference on the selected input grid, with the
first slot and unavailable operand pairs masked, never filled using the future.
The evaluator must implement and version this behavior, preserve market/source
mask semantics, and reject unsupported transforms rather than silently ignore
the gene. A train-only scaling contract also belongs to that evaluator.

## Search API

```python
from tools.i6n_search import Search, per_horizon_configs

# Deterministic detached configs in horizon order; performs no evaluation.
configs = per_horizon_configs(seed=6102026, count=4)

# frozen_support and evaluate_config must be supplied by the integration owner.
search = Search(12, seed=6102026, budget=20, population_size=8,
                max_proposals=640, support=frozen_support)
proposals = search.propose(2)  # each is {"identity": ..., "config": ...}
for proposal in proposals:
    result = search.evaluate(proposal, evaluate_config)
# Or search.run(evaluate_config): handles pending work and records rejections.
```

The owner's primary seed, `6102026`, is the default for `per_horizon_configs`.
`Search` still requires an explicit seed.

Real DEAP `Toolbox`, `Fitness(weights=(-1,))` and `selBest` implement a
steady-state (mu+1) survivor population. Parent tournaments sample three
individuals using the instance-local `random.Random`; 80% of proposals with
parents use mutation, 20% use random immigrants. No crossover is currently
implemented. This mutation-plus-immigrants component is not the final approved
DEAP search and is not the default optimizer's staged/crossover GA.

Joint conditional mutation covers family, window/history/input spacing, width,
Transformer blocks, reducer count/filters, independent strides, adapter/type
and width, input transform, and output supervision/spacing. `mutate(g, rng,
group=...)` permits directed testing; its finite neighborhood may return the
canonical parent at an immovable boundary. Unguided mutation tries each group
in local-random order until a valid effective change exists.

There is no global RNG seeding or `deap.creator` registry mutation. Stable
seed, support, callback behavior, and propose/evaluate ordering reproduce a
run. Batching proposals differently can change the evolutionary history.
Evaluation callbacks own their own RNG and resource behavior; arbitrary user
callbacks are not sandboxed by this class.

Reservations, including failures, consume the unique-candidate budget.
Deduplication precedes evaluation; every raw proposal attempt also consumes
the lifetime `max_proposals` bound (default `32*budget`). Exhaustion returns
an empty proposal list, not polling/retry. Each reserved proposal can be
evaluated once. Modified/unreserved/cross-horizon proposals are rejected.
Exceptions and invalid receipts become `ScoreRejected`, recorded separately
from successful results; `run` continues only within the original budget.
Callback copies and returned records are detached. Internal population and
toolbox are coordinator-owned, not untrusted wire inputs. The class is
synchronous, non-thread-safe, in-memory, and supplies no callback timeout.

## Score Receipts

Frozen support supplied before proposing:

```python
{
    "counts": {"train": 10, "inner": 5},
    "naive_sums": {"train": 20.0, "inner": 15.0},
    "support_digest": {"train": "a" * 64, "inner": "b" * 64},
}
```

The values above and in the tests are **fabricated fixtures**, not measurements.
The callback receives an effective config and returns the same three mappings
plus `status="ok"`, `error_sums={"train": ..., "inner": ...}` and JSON
`metadata`. Tests explicitly set `metadata.evidence_kind="stub"`. An optional
`fitness` must agree with the recomputed fitness. Missing/unknown receipt keys,
failed statuses, booleans, nonfinite/negative sums, zero counts/naive sums,
malformed digests, extra splits, and mismatched frozen support are rejected.

Counts are eligible **terminal-horizon** origin/target pairs, not batch counts
or all trajectory heads. Error sums are absolute endpoint forecast errors in
original price units; naive sums are absolute errors of that horizon's paired
naive over exactly those same rows. Trajectory intermediate heads affect
training supervision, not the comparison population. Freeze support across
the admitted search space before scoring; a failed prediction cannot remove a
row. Digest construction must bind horizon, split, ordered origins/endpoints,
masks and governing data/calendar/clock identity. This component compares
caller-supplied receipts; it cannot authenticate source rows or verify that
the callback really used their claimed support or performed training.

Lower-is-better callback fitness is an explicit local candidate protocol:

```text
0.5 * (train_error_sum/train_naive_sum + inner_error_sum/inner_naive_sum)
```

On each fixed support this equals the ratio of paired MAEs. The two splits
have equal weight, regardless of their sizes. It is not a mean of batch means,
not the trainer's Huber early-stopping loss, and not a validated campaign
selection policy. Raw sums/counts/digests/metadata survive in result `metrics`.
This TRAIN/inner mean of ratios is not a chosen outer annual-validation
objective or selection policy. TRAIN/inner are the only accepted score splits;
no outer or TEST scores enter this component. The callback still owns split
provenance and best-weight restoration. No trainer or weekly-data adapter is
claimed wired here.

## Actual DOIN Interface

Read-only inspection on 2026-10-11, without importing the modules:

- Sibling `doin-plugins/src/doin_plugins/predictor/optimizer.py`, SHA-256
  `9ea82e2943d3f40ececaeea0c1c267833da1e1c43d5571f67385a7705b8eb2fc`.
- `predictor/optimizer_plugins/default_optimizer.py`, SHA-256
  `2b4322b911eaceac4bbff5ad3a155c9f96bbb877876132b57984da852e4817d4`.
  Primary and shared checkouts had identical bytes for this file.

`PredictorOptimizer.configure(config)` loads predictor/preprocessor/target
plugins and the selected optimizer, calling its `set_params(**config)`.
Its `optimize(current_best_params, current_best_performance)` invokes the full
`Plugin.optimize(predictor_plugin, preprocessor_plugin, opt_config)` and returns
`(best_hyper, best_fitness_so_far)` or None. This is not a one-candidate API.
It installs these six `optimization_callbacks`:

```text
on_generation_start(population, hof, hyper_keys, gen, stage_info)
    -> champion parameter dict | {"_force_stage_advance": True} | None
on_new_champion(champion_params, fitness, metrics, gen, stage_info)
on_between_candidates(gen, candidate_num, stage_info)
    -> {"_force_stage_advance": True} | None
on_generation_end(population, hof, hyper_keys, gen, stage_info, stats)
on_stage_end(stage, total_stages, champion_params, champion_fitness, metrics)
on_candidate_evaluated(candidate_info)
```

The wrapper's shared API, at lines 403-455, is exactly:

```text
setup_shared_mode()                         # changes cwd to predictor root
get_shared_opt_config()                     # deep copy of merged config
create_shared_population(population_size, seed=42)
  -> optimizer.create_shared_population(population_size, opt_config, seed=seed)
evaluate_candidate(genome_serialized, generation)
  -> optimizer.evaluate_single_genome(genome_serialized, generation, opt_config)
reproduce_shared(evaluated_pop, generation, seed, innovation_tracker_data,
                 stage_schedule, param_defaults, current_stage_idx=0,
                 no_improve_count=0)
  -> optimizer.reproduce_shared(evaluated_pop, generation, seed, opt_config,
       innovation_tracker_data, stage_schedule, param_defaults,
       current_stage_idx, no_improve_count)
```

Population documentation names `population`, `innovation_tracker`,
`stage_schedule`, `config_snapshot`; evaluation documentation names `fitness`,
`metrics`, `hyper_dict`, optional `_model_b64`. These names alone do not define
a DEAP wire genome. **Neither inspected default optimizer implements any of
the three shared methods.** Its existing flat `hyperparameter_bounds` genome,
TensorFlow import, global RNG usage, and whole-GA callback hooks are not this
hierarchical search contract. NEAT shared methods in a different optimizer
are not evidence of default-DEAP compatibility.

Therefore no speculative real-DOIN adapter is supplied. Remaining work:

1. Specify/version the hierarchical shared wire genome, population/reproduction
   state, stage barriers and resume identity in the owning optimizer/runtime.
2. Implement those three optimizer methods and route the joint genes into the
   real model, transform, label/mask, trainer and paired-endpoint evaluator.
3. Bind actual support/data identity, cost admission, claims/deduplication,
   stage advancement and score receipts; reject unsupported genes end-to-end.
4. Test those real boundaries, including migration and failed-score handling,
   before claiming runtime compatibility. No sibling edits were made here.

## Verification And Traceability

Scoped method state: requirements/test design preceded implementation; unit
and local component verification complete. Real runtime integration, data
acceptance, alpha and operation remain unverified. No initiative-specific
method-state file was present; this in-scope document records the handoff
without creating files outside the three-file ownership restriction.

| Requirement | Test evidence in `test_i6n_search.py` |
| --- | --- |
| Horizon/window/endpoint rules | `test_window_and_output_contracts`, independent-resolution and invalid-gene tests |
| No aliases, core schema authority | canonical-inactive, real-core-validator, pending/canonical-dedup tests |
| Joint bounded mutation | every-group test, per-horizon mutation chains |
| Determinism, actual DEAP, no global RNG/TF | import/config, seeded-search, selection, Python/NumPy RNG tests |
| Finite budgets and failures | duplicate/attempt-budget, pending reservations, stub exception tests |
| Comparable auditable scores | support mismatch, invalid frozen support, missing/nonfinite receipt tests |
| Isolation | detached callback/config/receipt, JSON roundtrip, repeated-run tests |
| Real DOIN compatibility | source inspection only; deliberately NOT asserted by stub tests |

Finite CPU command (no cache/bytecode/report files):

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  crispdm-run -m 512M -t 120s -n i6n-search-component -- \
  python -B -m pytest -p no:cacheprovider tests/test_i6n_search.py -q
```

Observed test-first RED: missing `tools.i6n_search`, exit 2, 0.17 s.
Initial GREEN: 64 passed, 0.46 s. Extended and final GREEN: 97 passed, 0.78 s
each. All four used CPU-only 512 MiB / 120 s admission, no queue/polling loop.
These are software contract checks with fabricated score receipts, not model
quality, workload-memory measurements, or evidence of a historical campaign.
