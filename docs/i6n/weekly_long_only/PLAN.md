# Weekly Long Only: Delete Short Outputs From The Twelve-Head Parent

Owner approval 2026-10-17. Parent runtime469eef41, completed twelve-head
weekly campaign. SINGLE scientific intervention: remove [2,3,4,5,6,12] from
the output array, keeping [18,24,36,48,60,72]. Do not touch the active secondary
GPU window/core sweep. Use the free primary rtx5090, one heavy job only.

## Frozen Recipe

19 identical RAW inputs, DS2/13 observations including t, same market grid
and actual elapsed-offset/closure context; branches Conv1D4(k3), same weightless
fusion, core Conv1D2, each retained head LSTM4 -> Dense8 -> Dense1. Seed6102026,
AdamW .001/.0001, batch256, max200, patience10, min_delta1e-8. Original monitor
M=(TRAIN loss + current-week VALIDATION loss)/2 and lowest-monitor checkpoint.
No ANY_HEAD, optimizer search, transformed targets, new features or donors.
All53 original calendar periods: 52 weekly fits, one original empty skip,
same four rolling years and the exact same 2024 evaluation weeks. TEST closed.

## Strict Contrast Contracts

| ID | Requirement and test |
|---|---|
| LO01 | Model config differs ONLY in horizons_hours; original scientific source bytes unchanged. |
| LO02 | Target/mask columns are exact parent [:,6:] views; EVERY other array, origin and week remains identical, including rows whose retained-head masks are all false. Never intersect masks or use the separate sweep population. |
| LO03 | Input scales unchanged; six FIT-naive scales are exact parent suffix, not refitted. Recompute/check cached own-naive/count/pair hashes against parent per period and split, then annual totals. |
| LO04 | Preserve all remaining initial tensors and initial forwards exactly. Dropping preceding heads must NOT silently shift random initialization of the remaining heads. |
| LO05 | Same original trainer and loss formula, cadence, masks, seed and stopping settings. Removing six losses necessarily changes gradients, mean loss/monitor and chosen epoch; do not promise identical training trajectory. |
| LO06 | Existing admitted finite worker, bounded RAM, source pins, immutable terminals, durable outbox/readback, two mirrors and scoped close. No refit of completed parent or interrupted new fits. |
| LO07 | Per-head annual metrics paired against parent long heads, complete-year best registry, separate original artifacts and per-week stopping ledger. No partial-year scientific winner. |

## Initialization Control

The existing builder consumes seeded initialization in head-list order. Merely
building a shorter list can therefore change the retained initial head tensors.
Create the exact freshly SEEDED parent graph, then construct a Keras forecast
model from its existing inputs and only its six long outputs. Disconnected
short heads are not trainable parameters of that model and contribute no loss.
No TRAINED parent weights are imported. The resulting config differs only in
horizons; all82 retained tensors/initial forwards must match the fresh parent.
This scoped model factory is used consistently for fitting AND fresh reload;
the underlying builder and trainer are not edited. Native shorter-list
initialization is a negative control, not another scientific fit.

## Automation And Scope

Preparation authenticates parent closure, physical inputs and terminal digests;
fit-free GPU acceptance precedes launch. Timers run all53 periods, aggregate,
publish/read back, verify two copies and stop themselves. Status is queryable
at campaign and cell level without agents polling. No intervention between
weeks. Source-population tag `short` is inherited from the parent harness; it
does not define forecast horizons, which are the six explicitly LONG outputs.

These are retrospective DEVELOPMENT2024 results: each scored week also selects
its checkpoint, as explicitly requested. They do not demonstrate prospective
weekly deployment, independent OOS performance or profitability. Demo policy
remains the owner's separate net-positive-simulation rule, without TEST gate.

The parent52/52 fits stopped by patience (epochs11..75, median16), not max200.
Shared-core task competition and shared-checkpoint selection are hypotheses;
this ablation can test removing short tasks, not prove a unique mechanism.

## Execution

```bash
python -B -m tools.i6dmo.weekly_long prepare --spec "$SPEC"
python -B -m tools.i6dmo.weekly_long launch --spec "$SPEC"
python -B -m tools.i6dmo.weekly_long status --spec "$RUN_SPEC"
```

SPEC/RUN_SPEC are private operator paths. No host names, account identifiers,
tokens, source bytes or large analytic snapshots belong in git.
