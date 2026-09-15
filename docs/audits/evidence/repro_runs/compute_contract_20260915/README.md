# The compute contract, and one successor run under it (S1)

## The correction

The reviewed `_observed_work` published `_n_updates` as "gradient updates". On the installed
Stable-Baselines3 **2.9.0** that is false for PPO: `PPO.train` increments it once per
optimization **epoch**, outside the minibatch loop where the optimizer actually steps.

Measured here, not asserted (`agent-multi/tests/test_compute_contract.py`, CartPole, CPU):

| configuration | transitions | `_n_updates` | optimizer calls |
|---|---|---|---|
| n_steps 64, batch 32, epochs 2, learn(256) | 256 | 8 | **16** |
| n_steps 64, batch 64, epochs 1, learn(64) | 64 | 1 | 1 |
| n_steps 64, batch 64, epochs 1, learn(64), **4 envs** | 256 | 1 | 4 |

Eight epochs and sixteen optimizer calls are different numbers, and the ratio is the minibatch
count — it grows as the batch shrinks. Optimizer calls are now **counted by instrumenting the
real optimizer**, and the instrumentation is removed afterwards so it cannot count a later run
into the same meter. An algorithm whose counter has no measured meaning reports the raw delta,
`sb3_n_updates_meaning: UNKNOWN`, and makes **no** epoch or gradient-update claim at all.

## Two traps the rules exist for

**A resumed model resets its own step counter.** `learn()` defaults to
`reset_num_timesteps=True`, so a model that had already run 64 transitions goes 64 → 64: the
naive after-minus-before is **zero** for a call that really trained. Measured, and detected
rather than assumed; `_n_updates` is not reset, which is how it is caught.

**A partial rollout has not trained.** A run stopped inside its first rollout reports
`collected_rollouts: 0`, `optimization_epochs_completed: 0` and `optimizer_step_calls: 0`
while its transitions are still counted. It is not a short training run.

## The ceiling

A target is not a ceiling, and they are separate fields. Before a single transition is
collected, a configuration whose minimum rollout cannot fit the declared cap is **refused**:
the exact `n_steps 256` against a cap of 64 raises, and `model.num_timesteps` is still 0. A
target above the cap is refused too. The cap is never widened to fit the settings.

## The successor run

`doin-offline-replay-successor-1`, NON_GOVERNING, CPU, against the live services, from a
**clean** git worktree so `code_identity` is `ddbff4d6…` with no `-dirty`. Declared
mechanically, as S1 specifies: one environment, `n_steps 64`, `batch_size 64`, `n_epochs 1`,
target 64, hard cap 64, evaluation cap 384.

| quantity | value |
|---|---|
| requested training transitions | 64 |
| training transition cap | 64 |
| **training transitions observed** | **64** |
| collected rollouts | 1 |
| optimization epochs completed | 1 |
| **optimizer step calls** | **1** |
| environment count | 1 |
| evaluation transitions / cap | 384 / 384 |
| cap respected | true |
| wall seconds | 3.81 |

Terminal COMPLETED, reconciliation empty on all three lists, the input delivered
`VERIFIED_CACHE` — the same bytes as before, nothing re-downloaded. **21 counters** are in the
production cube under this campaign (`CUBE_PERSISTED_COUNTERS.txt`), each under the name of
what it is.

Process success is not acceptance of the resource contract; this run shows the contract is
executable and enforced, and says nothing about any scientific question.

`prod-12` and `prod-13` were not re-run and their rows are untouched. The historical reading
stays additive: their `observed_updates` of 400 and 10 are **PPO epochs**, and no optimizer-call
count is claimed for them, because none was measured at the time.
