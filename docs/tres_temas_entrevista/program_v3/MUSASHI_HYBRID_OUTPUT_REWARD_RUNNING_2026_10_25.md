# Output and reward automation: verified launch

Owner authorized implementation and sequential execution on 2026-10-25.
Agent-multi implementation: `ae04322b`, branch
`codex/rl-static-temporal-20261018`; detailed design and verification scope:
`docs/work_plan/70_HYBRID_AUTOMATION_EXECUTION.md` in that repository.

The new source-pinned campaign spec is
`bcd180f16a548dd9e402adecfdb9cb578066c611c5dd1b1340896ff0cf8df9f6`.
CAT_FIXED_s7 started at 12:21:44 UTC; at 12:25:48 it had completed 80384
of 102400 decisions. GPU synthetic admission passed both fixed and learned
risk. The focused CPU suite passed 39 tests. This is actual execution, not
just a prepared specification; final scientific metrics were not yet available
at this observation cut.

## Finite sequence

Eight treatments, seeds 7/19/43 each, sequential on the primary GPU:

1. Categorical commands, fixed feasible SL/TP and full allowed order budget.
2. Conditional learned SL/TP/volume, same equity-delta reward.
3. Conditional risk, log-equity reward.
4. Log-equity minus incremental maximum drawdown, penalty 0.25.
5. Same with penalty 1.
6. Same with penalty 4.
7. Differential Sharpe with declared causal moment update and variance floor.
8. Fixed equal-weight log-return/differential-Sharpe mixture.

This is **PPO with a categorical and conditional Beta policy**, not SAC.
Commands are not discretized Gaussian outputs. The encoder, input recipe and
business rules are retained, but encoder dropout is explicitly disabled for
consistent PPO collection/optimization likelihoods. Thus this is a new
algorithm package, not a one-variable contrast with the previous SAC models.

Routing masks enforce feasible commands without adding account-state features
to the encoder. There is no expert, rolling scaler, TEST access, early stopping
or periodic refit. Models, effective parameters and native trading reports are
retained. Eight post-training ranking rules compare eligible three-seed results
using TRAIN-only common scales and INNER validation; 2024 does not select.
Report profit, Sharpe, activity, win rate, drawdown and Calmar separately.

## Recovery and limits

Worker and independent checksum-copy/warehouse timers are enabled. Reboots or
resource waits do not need an interactive dispatch. At most two fresh attempts
per fit, with distinct attempt identities; completed authenticated results are
adopted without repeating fits. Failed/inactive results remain explicit.
Initial collection succeeded before any final report existed. Publication
of this campaign's actual financial report still awaits its first completion.

Adaptive coefficient refinement and a robust Sharpe-uncertainty estimator are
deferred, not silently scheduled. Costs remain assumed rather than broker
verified. No changes or extra tasks are required from the independent expert
lane for this batch.
