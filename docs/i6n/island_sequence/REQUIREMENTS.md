# I6N DOIN Island Sequence Requirements

## Scope

This initiative replaces neither the retained shared-population evidence nor the
general optimizer. It adds an explicit local-island execution mode for exactly
two governed I6N searches: 4-hour first, then 72-hour. Each DOIN node evolves a
local population and exchanges champions through one canonical blockchain.

The work in this stage is specification only. No campaign is authorized by
these documents or by the red tests.

## Optimizer Contract

| ID | Requirement | Observable acceptance criterion |
|---|---|---|
| ISL-001 | Explicit mode | `Plugin.optimize` enters this path only when `i6n_island_mode` is exactly `true`; missing or false is rejected. |
| ISL-002 | Local population | Every node uses its own population. `population_size < 20` is rejected before any evaluator call. |
| ISL-003 | Search duration | `n_generations < 100` is rejected. One hundred is a ceiling, not a requirement to ignore early stopping. |
| ISL-004 | Generation patience | `optimization_patience < 10` is rejected. Patience counts completed generations, not candidates or folds. |
| ISL-005 | Exact improvement threshold | `min_relative_improvement` must equal `1e-8`. A reset requires `old_best - new_best > 1e-8`; equality does not reset patience. |
| ISL-006 | Horizons | The island path accepts only `i6n_horizon=4` or `i6n_horizon=72`. |
| ISL-007 | Four-fold evaluation | Every candidate must have one verified receipt from each of four distinct quarterly folds before it can be ranked. A partial candidate is not assigned a fallback fitness. |
| ISL-008 | Objective | Compute the established fold ratio, average all four folds, then report `fitness = mean_ratio - 1`. Lower is better; a negative value beats the paired naive. |
| ISL-009 | No TEST access | TEST is unavailable to candidate evaluation, ranking, patience, promotion, and H4-to-H72 transition. |
| ISL-010 | Determinism | The same configuration, input identities, and seed must reproduce the same local proposal sequence and result. |
| ISL-011 | Backward compatibility | Existing shared methods and retained historical evidence remain byte-for-byte interpretable. Island behavior is never selected implicitly. |

The established ratio for fold `f` is:

```text
ratio_f = 0.5 * (
    train_error_sum_f / train_naive_sum_f
  + inner_error_sum_f / inner_naive_sum_f
)

fitness = mean(ratio_f for all four folds) - 1
```

Counts, support digests, candidate identity, fold identity, horizon, business
calendar, feature manifest, and protocol identity must be bound by each receipt.

## Sequence Contract

| ID | Requirement | Observable acceptance criterion |
|---|---|---|
| SEQ-001 | Ordered searches | H4 is the only initial search. H72 cannot start concurrently or first. |
| SEQ-002 | Durable H4 closure | All four nodes publish the same terminal identity, stop reason, champion identity, configuration digest, and common finalized chain checkpoint. |
| SEQ-003 | Verified archive | Every terminal candidate/metric identity is durable, chain-finalized, projected into OLAP, and read back before the H72 transition receipt is committed. |
| SEQ-004 | Exactly-once transition | A restart at any transition state cannot launch H72 twice or reopen H4. |
| SEQ-005 | Independent H72 contract | H72 uses a separately sealed predictor/optimization configuration while retaining the same business, feature, fold, and objective contracts. |

## Fleet And Operations

| ID | Requirement | Observable acceptance criterion |
|---|---|---|
| OPS-001 | Exact fleet identity | Before launch, all four nodes report the same seed, network/genesis identity, canonical chain tip, and exact commits for `predictor`, `doin-node`, `doin-core`, and `doin-plugins`. Any mismatch refuses launch. |
| OPS-002 | Autonomous operation | Versioned services and timers perform evaluation, archive pumping, OLAP ETL, status generation, recovery, and H4-to-H72 succession without an agent polling logs. |
| OPS-003 | Monitorable status | A read-only status command/API reports phase, generation, evaluated candidates, patience, champion fitness, node health, chain checkpoint, archive state, and evidence-based ETA. |
| OPS-004 | Power-loss recovery | A restart resumes from a configuration-bound atomic checkpoint without repeating a completed candidate or changing the proposal sequence. |
| OPS-005 | Resource admission | GPU/host placement uses fresh measured admission. A node that cannot admit its configured workload is reported, never silently replaced by CPU or a smaller scientific configuration. |

## Business And Data Invariants

- Each candidate uses the frozen selected RAW features; feature identity is not
  an optimization gene.
- Four quarterly 2023 inner folds are distinct periods, not repeated seeds.
- Each fold trains only on its preceding four-year TRAIN support.
- Weekly market eligibility and forced-close boundaries are applied before
  scoring; model and naive use exactly the same eligible rows per horizon.
- Refit/evaluation behavior follows the governed weekly business contract.
- No model is promoted to the heuristic strategy unless its final governed
  prediction result beats the naive for its own horizon and matched support.

## Exclusions

- No 1-hour, 2-hour, 3-hour, 5-hour, 6-hour, 12-hour, 24-hour, 48-hour,
  96-hour, or 120-hour optimization in this sequence.
- No NEAT, RL, live trading, broker execution, feature reselection, denoising,
  or TEST evaluation.
- No claim that a red test, stub receipt, or configuration constitutes a model
  measurement.
