# I7-M: direct multi-output strategy horizon sweep

## Purpose and priority

The completed 1h comparison does not answer whether the prediction-based
strategy is viable at its operating horizons. Run one vector-output R2 model
per week before the I6-E resampling contrast. Horizons are explicitly fixed by
the user's 2026-10-10 instruction: 1,2,3,4,5,6,24,48,72,96,120 physical hours.
One hour is diagnostic only. No broker, TEST or trading simulation is invoked.

## Fixed model and data

Reuse the selected 20 EURUSD features and the 1040 authenticated weekly branch
donors from I6-B. This is a fixed transfer set, not a fresh horizon-specific
feature selection. Each branch preserves the 24-hour input sequence. Reuse
the temporal fusion, positional encoding, Transformer/Conv1D core and direct
forecast head; enlarge only the head's output grid to 11 horizons.
Branches load their own weekly donor and update in R2. The core initializes
from scratch: this experiment does not claim core pretraining.

Seed 0 is retained. Each of the 52 validation weeks of the inherited 2024
calendar gets exactly one fit on the preceding four calendar years. Purge
labels by the maximum 120h support, with four chronological inner-validation weeks;
do not use outer-week outcomes for early stopping. Input normalization and
each output's robust scale are fitted on the fit population alone.
Missing physical future timestamps and stale long-target endpoints are
excluded explicitly, never substituted with the next row.
The FX calendar makes the intersection of all eleven exact endpoints empty.
Fit rows therefore need at least one finite label. Missing labels have zero
sample weight; each horizon's weights sum to its population's row count, so
the loss is the equally weighted mean of per-horizon mean losses. Fit and
inner-validation counts are retained separately for every horizon.
The preliminary one-week inner-validation pilot was retired after a holiday
week left a horizon with no labels. Four inner weeks provide the same fixed
selection rule for all weekly fits; preliminary cells are not mixed into the
annual result. This is a support repair, not selection based on forecast skill.

Early stopping monitors half the sum of sample-weighted batch training loss
and inner-validation loss, using the inherited patience/budgets. Batch training
loss is measured during updates; it is not a post-epoch evaluation of the
selected weights. Restore the best monitored checkpoint. Retain both losses,
optimizer update counts, seed, input/config/donor/code digests and checkpoint.

## Automation and acceptance

`tools/i7_multi_horizon.py` provides `init`, `run-cell`, `worker`, `status`
and `close`. Workers use an exclusive per-week claim and a fresh subprocess
per fit; completed valid cells are reused, not trained again. A failed child
terminates its worker instead of silently creating a replacement result.
Production requires one visible, execution-tested GPU; no CPU fallback.

Measure the first week with the full worker/child tree on the preferred GPU.
Reuse it in the campaign. Reserve 1.25 times its measured cgroup peak if it
fits. Run even and odd weeks on two independent admissible hosts; never run
two GPU jobs concurrently on the memory-constrained preferred host.
The durable coordinator polls and merges terminals automatically, updates
`SWEEP_STATUS.json` and produces `SWEEP_CLOSURE.json` only at 52/52 without
identity/reduction problems. Preserve `.keras` files in local artifact storage,
not Git. A live OLAP readback is separate from local closure and must not be
reported successful merely because a JSON terminal exists.

```bash
cd <pinned-checkout>
python -m tools.i7_multi_horizon init --parent-design <I7_DESIGN.json> --output <sweep>
python -m tools.i7_multi_horizon worker --output <sweep> --shard-index 0 --shard-count 2 \
  --feature-parquet <train-1.parquet> --feature-parquet <train-2.parquet> \
  --feature-parquet <train-3.parquet> --target-parquet <train-targets.parquet> \
  --validation-feature-parquet <validation-1.parquet> \
  --validation-feature-parquet <validation-2.parquet> \
  --validation-feature-parquet <validation-3.parquet> \
  --validation-target-parquet <validation-targets.parquet>
python -m tools.i7_multi_horizon status --output <sweep>
python -m tools.i7_multi_horizon close --output <sweep>
```

Use the second shard index on the other host. Merge disjoint `cells/` and
`models/` before closing centrally. The pilot uses shard count 52/index 0
and then both production workers skip its authenticated terminal.

## Result and next decision

For every horizon publish annual row-pooled MAE/MSE, paired naive MAE/MSE,
skill, directional accuracy, scored/excluded counts and weekly diagnostics.
Targets are raw log returns, so price persistence is prediction zero.
Eligibility is annual MAE strictly below the naive on identical rows, with
1h always excluded. Do not discard negative-skill weeks or pass the strategy
using a selected profitable week. Compare required short/long arrays against
the actual strategy plugin before executing any reduced subset.

The downsampling contrast remains I6-E: origin-relative endpoint samples,
no smoothing, separately selected spacing/window inside TRAIN. It cannot be
claimed evaluated by this fixed 24-hour hourly-input multi-output sweep.

## Verification

Seven pure contract tests and one real Keras integration test cover:
physical-hour gaps/staleness, per-horizon fit-only scaling, vector shape,
same-row naive and pooled reduction, joint early-stopping arithmetic and
bitwise prediction equality after checkpoint save/reload. CPU tests establish
plumbing, not forecast quality; scientific results require the full campaign.
