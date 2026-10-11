# I6N Shared DOIN Adapter

Status (2026-10-11): implemented shared-only DEAP adapter, verified with
fabricated CPU receipts through the **actual wrapper forwarding bodies**.
No historical fitting, TEST access, full wrapper configuration, node launch,
swarm, deployment, model-quality result, or distributed acceptance is claimed.
Only `optimizer_plugins/i6n_optimizer.py`, `tests/test_i6n_optimizer.py` and
this document are owned by this change. Registration in `setup.py` is the
parent's responsibility.

## Real Interface And Direction

Read-only graph discovery and source inspection covered:

- `doin-plugins/src/doin_plugins/predictor/optimizer.py`:
  `PredictorOptimizer` shared bridge and `get_domain_metadata`.
- `doin-node/src/doin_node/unified.py`: `_shared_optimizer_loop`,
  `_run_shared_optimization`, `_candidate_rejection_reason`.
- Local `tools/i6n_search.py`, `docs/i6n/SEARCH_COMPONENT.md`, and
  `app/plugin_loader.py` (requires the class's `plugin_params` mapping).

Adapter signatures, including the extra `opt_config` inserted by the wrapper:

```python
Plugin.set_params(self, **kwargs)
Plugin.create_shared_population(self, population_size, opt_config, seed=6102026)
Plugin.evaluate_single_genome(self, genome_serialized, generation, opt_config)
Plugin.reproduce_shared(
    self, evaluated_pop, generation, seed, opt_config,
    innovation_tracker_data, stage_schedule, param_defaults,
    current_stage_idx=0, no_improve_count=0,
)
```

`set_params` copies settings; per-call `opt_config` overrides them. The wrapper
still defaults its own bootstrap seed to **42**. Pass `seed=6102026` through
the wrapper or configure the node's bootstrap seed explicitly.

Predictor metadata declares `higher_is_better=False`. The node actually
compares according to **its domain role's** `higher_is_better`, using `min`
when false and `max` when true. Its generic role default is true, so ensure
the resolved predictor role is false; this adapter cannot inspect that role.
It rejects an explicit true setting in `opt_config`. The returned fitness is
the **nonnegative raw loss ratio**, never its negation. DEAP uses
`Fitness(weights=(-1.,))` and `tools.selBest`. Binary predictor maximization
is a different objective and is not supported.

## Configuration

Configure one horizon per domain/run. Required settings:

| Key | Contract |
| --- | --- |
| `i6n_horizon` | One horizon admitted by `i6n_search.HORIZONS` |
| `i6n_support` | Frozen `counts`, `naive_sums`, `support_digest`, exactly TRAIN/inner |
| `i6n_evaluation_context` | Nonempty JSON mapping binding the evaluator and experiment context |
| `i6n_evaluator` | Python callable or `module:callable`, mutually exclusive with command |
| `i6n_evaluator_command` | Alternative argv list, never a shell string |

An evaluator may be absent during population creation, but evaluation then
raises `ScoreRejected`; no success receipt or substitute score is generated.
The context must bind data/calendar/clock, feature order, frozen support
construction, transforms, evaluator version, training settings, model seed,
and resource admission. The adapter hashes the supplied context, horizon,
support and objective into each genome. It cannot authenticate these claims
or infer a callable's closure, code version, or historical provenance.
Changing an evaluator/command or any governing setting requires a new context.

Optional controls: `n_generations=10`, `optimization_patience=5`,
`i6n_budget=population_size*n_generations`,
`i6n_max_proposals=32*i6n_budget`, `i6n_evaluator_timeout=120.0` seconds
(command only). Counts and budgets must be positive integers. Settings are
pinned into shared state; changing them on resume is rejected. Population
and generation seeds are local `random.Random` instances; no global RNG or
`deap.creator` mutation occurs.

## Evaluation Boundary

A callable receives **one detached request mapping**, not a raw genome:

```python
{
    "config": effective_config,  # includes grids, samples, core, transform
    "generation": 0,
    "context": evaluation_context,
    "support": frozen_support,
}
```

It returns the `i6n_search` score receipt:

```python
{
    "status": "ok",
    "counts": {"train": ..., "inner": ...},
    "naive_sums": {"train": ..., "inner": ...},
    "support_digest": {"train": ..., "inner": ...},
    "error_sums": {"train": ..., "inner": ...},
    "metadata": {...},
    # Optional fitness must match the recomputed nonnegative ratio.
}
```

The JSON-command protocol is the same request as a single JSON document on
stdin, and exactly one receipt document on stdout. Logs belong on stderr.
Nonzero exit, timeout, invalid JSON, missing import, or invalid receipt raises
`ScoreRejected`. No module named historical_cell is imported by default.
A future executable bridge can be configured, for example, with
`["python", "-m", "<receipt_bridge_module>"]`; this is a protocol example,
not evidence that the pending historical cell implements stdin/stdout scoring.

The adapter reuses `i6n_search._support` and `_score` to keep a single receipt
authority. This private-helper coupling is intentional and covered by the
combined regression run. Counts, digests and naive sums must match frozen
support exactly. Boolean, negative, nonfinite, missing and extra-split values
fail closed. Endpoint errors are summed in original price units; trajectory
intermediate heads do not enter the endpoint comparison. Fitness is:

```text
0.5 * (train_error_sum/train_naive_sum + inner_error_sum/inner_naive_sum)
```

This is equal-weighted across splits, not pooled across rows/batches. The full
receipt is retained in `result["metrics"]`; `hyper_dict` is the expanded
effective config. `train_mae`, `train_naive_mae`, `val_mae`, `val_naive_mae`
are also emitted for the existing node dashboard. **`val_*` means inner**,
not outer validation or TEST. No `_model_b64` artifact is fabricated.

Callables run in-process without timeout or sandbox. Commands have a direct
process timeout, but no independent memory limit, descendant process cleanup,
or output quota. Production evaluator resource isolation remains the runner's
responsibility. Commands are trusted operator configuration, not remote genome
instructions. No data/model-training imports occur in this adapter itself.

## Population And Reproduction

The runtime treats each population item as opaque except for assigning its
top-level `fitness`. It passes the entire item to `evaluate_candidate`, then
hands reproduction the same items **with fitness only**, not full results.
Therefore this plugin defines an I6N-specific envelope, not a supposed common
NEAT/DEAP genome format:

```python
{
    "schema": "i6n-shared-deap-v1",
    "genome": canonical_i6n_search_genome,
    "identity": effective_config_sha256,
    "context": frozen_context_sha256,
    "generation": 0,
    # "fitness": raw_loss_ratio  # assigned by the node after evaluation
}
```

Initial state has `population`, `innovation_tracker`, `stage_schedule`,
`param_defaults`, `config_snapshot`, `generation`, `stage_idx` and
`no_improve_count`. `innovation_tracker` is the existing opaque runtime slot
for this plugin's versioned state, **not NEAT innovations**. It holds sorted
seen identities (including failed candidates), ordered pending identities,
lifetime proposal attempts, generation, pinned settings and context digest.
It survives the node's next-generation checkpoint; `config_snapshot` does
not survive that node reconstruction and is not relied on for resume.

Hierarchical sampling/normalization/mutation is delegated to `i6n_search`.
Parent tournaments draw three candidates with local RNG and select using real
DEAP minimization. Each proposal is 80% conditional mutation, 20% random
immigrant when eligible parents exist. There is no crossover. Unlike the local
steady-state Search, this adapter emits **generational novel offspring**;
elite genomes are not re-evaluated. The node retains its champion separately.

Canonical deduplication is checked before admission, across all generations
in this tracker. Every attempt, including aliases, consumes the lifetime
attempt bound. A partially filled population is valid and evaluated as-is;
no duplicate padding or unbounded retry is used. Empty initial admission
raises; exhausted reproduction returns `converged=True` with an explicit
`stop_reason` so the runtime does not retry the same generation indefinitely.
Budget/patience/generation completion also returns convergence. Nonfinite,
negative and nonnumeric parent fitness is excluded. With no eligible parents
the result reports `no_eligible_parents`, not scientific success.

Only one fixed-support stage is implemented. Other stage indices/schedules
fail closed. Reproduction preserves caller-supplied no-improvement count;
the node already updates it. State, callbacks and results are detached.

## Remaining Runtime Gaps

1. Parent must register `i6n_optimizer = optimizer_plugins.i6n_optimizer:Plugin`
   in `optimizer.plugins` and resolve/reinstall it. A safe real-resolver check
   found it in **neither installed metadata nor this checkout's setup.py**.
   This resolver raises `PluginResolutionRefusal`, a `SystemExit` subclass,
   which bypasses the wrapper's `except Exception`: missing registration fails
   closed here. Ordinary import/constructor/attribute exceptions instead fall
   back to the default optimizer, which lacks this shared interface. Verify
   the loaded class, not merely the configured name. The resolver prefers
   installed metadata, otherwise parses this checkout's setup declarations,
   and refuses repository plugins resolved outside this checkout. Use the
   shared checkout as `predictor_root` and ensure its `app` package is loaded.
2. Full `PredictorOptimizer.configure` eagerly loads predictor/preprocessor/
   target plugins before this adapter. Those imports, their configuration and
   a full runtime startup were not exercised. Shared-only operation is required;
   this adapter intentionally does not implement island `optimize()`.
3. The historical cell owner must translate the effective genome/request to
   real training and return this exact frozen-support receipt. The concurrently
   developing cell has a separate explicit config and `run/status --config
   --output` CLI, not this stdin/receipt protocol. Its current evaluation
   support is for the configured evaluation split, not yet this frozen
   TRAIN/inner receipt pair. A separate bridge is necessary.
   All admitted transforms, trajectory supervision and hierarchical genes
   must be implemented or explicitly rejected there. No silent gene dropping.
   The newly reported EST-versus-retained-NY source-clock correction must enter
   historical inputs, support digests and `i6n_evaluation_context`; do not reuse
   a support/context computed before correction.
4. The node retains receipts in results but passes **only scalar fitness** to
   reproduction. Reproduction cannot independently revalidate sums/support or
   distinguish a finite score whose rejection flag was discarded. This adapter
   raises on failures, which the current node converts to an error plus worst
   nonfinite fitness; those parents are excluded. Peer result authentication,
   malicious finite-score rejection and exact receipt persistence/readback
   remain node/integration-owner concerns.
5. Canonical proposal dedup is not a distributed exactly-once fit guarantee.
   Claims, lease recovery, re-evaluation retries and durable result reuse belong
   to DOIN. Node reproduction exceptions clear results and re-evaluate the same
   generation; corrupted state/config can therefore still induce runtime
   retries. Honest exhaustion is returned without an exception. Runtime-level
   handling of persistent corruption was not changed.
6. Swarm stage barriers, migrations, model artifact publication, workload cost
   admission and deployment were not tested. The objective remains the local
   TRAIN/inner search protocol, not an approved annual-validation selection
   policy. No sibling files, services, registration or commits were changed.

## Verification And Traceability

Requirements and tests preceded implementation. No applicable initiative
method state or canonical methodology file was present in this shared checkout;
this scoped document records the method state within the three-file ownership
limit. Unit/source-boundary integration is complete; historical/system/alpha
acceptance and operation are not. Next action: parent registration and a real
historical receipt bridge under frozen support, then resource-admitted runtime
acceptance before any operational claim.

| Requirement | Evidence in `tests/test_i6n_optimizer.py` |
| --- | --- |
| Real signatures and runtime forwarding | `test_real_wrapper_roundtrip_and_raw_minimization` executes AST-extracted real wrapper methods unchanged |
| Runtime fitness direction and rejection | actual wrapper metadata, `test_node_consumer_contract_and_rejection_source` |
| Determinism, resume, no global RNG/TF | `test_determinism_resume_and_global_rng_untouched` |
| Hierarchical validity and aliases | `test_all_horizons_valid`, canonical-dedup test |
| Exact comparable receipts | receipt rejection parameterization, unequal-split-ratio test |
| Real configured evaluator boundary | callable, import-callable, JSON subprocess and missing-evaluator tests |
| Bounded failure/stopping | finite-stopping and zero-offspring tests |
| State isolation and admission | context tampering, pending, precedence and invalid-admission tests |

The source-boundary tests extract actual methods from read-only sibling files
to avoid importing training/runtime dependencies. They do not mock forwarding
bodies and do not import/start a node. Set `I6N_SIBLING_ROOT` to the directory
containing both sibling repositories on other machines. Missing source causes
explicit skips, not a false runtime-pass claim; the verification below had no
skips.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  crispdm-run -m 512M -t 120s -n i6n-adapter-regression -- \
  python -B -m pytest -p no:cacheprovider \
  tests/test_i6n_optimizer.py tests/test_i6n_search.py -q
```

Observed RED: missing adapter, one collection error (0.16 s). Initial GREEN:
32 adapter tests passed (0.44 s). Extended combined GREEN: **142 passed in
1.08 s**, including 45 adapter and 97 existing search tests. Final GREEN after
an explicit real-DEAP tournament-direction check: **143 passed in 1.16 s**
(46 adapter, 97 search). Safe real plugin-resolution inspection independently
reported missing registration without importing TensorFlow. These are CPU-only
software checks under 512 MiB / 120 s admission, with fabricated receipts,
not historical/model-quality measurements or evidence of a running swarm.
