# I6N DOIN Island Sequence Plan

## Decision

Use the established DOIN island topology, not the experimental centralized
shared-population path. Four nodes each evolve a local DEAP population of 20
and exchange champions through one blockchain. Run one governed H4 search to
completion, close and archive it, then automatically start one governed H72
search.

## Fixed Search Contract

| Setting | Value |
|---|---:|
| Local population per node | 20 |
| Maximum generations | 100 |
| Generation patience | 10 |
| Minimum relative improvement | `1e-8`, strict `>` |
| Seed | 6102026 on every node |
| Replicas | 1 |
| Folds per candidate | 4 distinct quarters |
| Direction | Minimize |
| Reported fitness | mean ratio to paired naive minus 1 |
| Search order | H4, then H72 |

The values above are admission minima for this sequence. An implementation may
accept larger populations, generation ceilings, or patience only after a newly
sealed plan changes them; it may never silently lower them. The improvement
threshold is not a lower bound: it is exactly `1e-8`.

## Stage Plan

### 1. Unit implementation

Add `Plugin.optimize(predictor_plugin, preprocessor_plugin, config)` behind the
explicit `i6n_island_mode=true` flag. Keep shared methods unchanged. Validate
the complete contract before the first evaluator call.

The optimizer must:

1. construct a deterministic local population containing the minimal seed and
   small alternative;
2. evaluate each candidate on all four folds through the existing receipt
   verifier;
3. convert the established ratio to `ratio - 1` for reporting and minimization;
4. receive and inject valid network champions through existing callbacks;
5. perform deterministic selection, mutation and immigration within admitted
   complexity classes;
6. checkpoint population, fitness, RNG state, generation, patience and best
   candidate atomically;
7. emit candidate and generation callbacks with complete objective fields;
8. stop at 100 generations or after 10 completed generations without an
   improvement strictly greater than `1e-8`.

Group-safe crossover remains an open decision. It is not required to implement
the first correct island path and must not be improvised across conditional
genes.

### 2. Durable DOIN terminal

Before an island exits, persist a deterministic domain terminal containing:

- campaign/domain/config/plan identities;
- horizon and seed;
- terminal generation and reason;
- final population fingerprint and champion identity/fitness;
- last candidate ledger identity;
- canonical finalized chain checkpoint.

Restore this terminal after restart and expose it through the read-only API.

### 3. Automated successor

Implement a versioned oneshot service plus timer with this monotonic state
machine:

```text
H4_PREPARED
  -> H4_RUNNING
  -> H4_FLEET_TERMINAL
  -> H4_CHAIN_SHARED
  -> H4_ARCHIVE_VERIFIED
  -> H4_QUIESCED
  -> H72_RUNNING
  -> H72_FLEET_TERMINAL
  -> H72_CHAIN_SHARED
  -> H72_ARCHIVE_VERIFIED
  -> COMPLETE
```

Every transition is atomic and idempotent. No transition depends on a chat
agent remaining alive or reading logs.

### 4. Archive and OLAP

For every evaluated candidate, preserve configuration/genome inputs and metric
outputs, not training data or transformed feature matrices. The authoritative
analytical row includes:

- campaign, horizon, node, generation, candidate and genotype identities;
- exact repository commits and configuration digest;
- seed and admitted complexity class;
- all optimized genes and effective model parameters;
- per-fold train/inner error and naive sums, counts and support digests;
- mean ratio, `ratio - 1`, timing, resource peak and terminal disposition.

The blockchain may retain content hashes and durable references while the
object store/warehouse retain full metric payloads. H72 cannot start until H4
has chain verification and OLAP readback for the complete expected identity
set.

### 5. Fleet preflight

Query all four node APIs immediately before H4 launch and again before H72.
Refuse launch unless every node reports:

- seed 6102026;
- the same network/genesis identity;
- the same finalized chain height and hash after synchronization;
- exact matching commits for `predictor`, `doin-node`, `doin-core`, and
  `doin-plugins`;
- the same sealed plan/config identity for the active horizon;
- healthy evaluator dependencies and fresh resource admission.

Host-specific endpoints, device assignments and private addresses belong in a
private deployment spec, never in the public repository.

## Required Test Order

1. `tests/test_i6n_island_optimizer.py`
2. retained I6N optimizer/search tests
3. DOIN wrapper callback integration tests
4. successor and terminal falsification tests
5. four-node disposable system test
6. production read-only preflight
7. H4 launch and first-five-candidate alpha gate

## Stop Conditions

- Any identity, seed, commit, fold, support, chain, or archive mismatch refuses
  the affected transition.
- A resource refusal does not lower the scientific configuration or fall back
  to CPU.
- A partial four-fold candidate is terminally non-rankable.
- TEST access invalidates the run.
- The sequence never starts H72 from a merely local H4 completion.

## Current Boundary

This commit is complete at test and methodology design only. The tests are
expected to fail because `optimizer_plugins.i6n_optimizer.Plugin` intentionally
has no island `optimize` implementation yet. No runtime code or campaign state
is changed in this stage.
