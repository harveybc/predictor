# RP155: useful development scores, scoring closure changes required

Reviewed predictor `99a1630a` and lane return `ebe35fb3`. Scope: source, published
per-cell results, actual window adapter and scoring orchestration with bounded
CPU fixtures. No checkpoint inference, live service, accepted warehouse payload
or current GPU occupancy was independently inspected. No training was launched.

## Findings

1. **High: the advertised selection-free complement shares target labels with
   checkpoint selection.** `tools/df_ecl_modular.py:897` takes origins 640..2536
   after the first 640 monitor windows. The actual window adapter delivers 96
   hourly targets per origin. A row-identity probe through that adapter finds
   95 target timestamps shared between the monitor and the first 95 complement
   windows: 1,463,760 repeated target/channel elements in the complement tensor.
   Different origin indices do not prove untouched target support. The first
   label-disjoint local origin is 735, leaving **1,802**, not 1,897, windows.
   Preserve the original scores as descriptive, but withdraw "clean/never used"
   at target-support scope. Re-score the disjoint subset from existing selected
   checkpoints; no retraining follows from this finding. Disjoint labels still
   do not establish temporal statistical independence. Legitimate observed input
   context need not be purged just because it overlaps earlier context.

2. **High: scoring accepts an incomplete population and includes identity failures
   in its summaries.** `score_contrast`, `:932`, derives expected cells from whatever
   remains in the local run, not the registered design. Removing R2_s2023 gives
   8 expected/8 scored, no problems, and an R2 mean over two seeds alongside three
   for the others. An empty population reports all model identities match. At
   `:888`, score_cell hashes the checkpoint but does not refuse a mismatch before
   loading/scoring it. The parent (`:950-965`) aggregates a child with identity=false,
   still with no problems. These orchestration counterexamples do NOT show any of
   the nine retained real digests mismatched; their published flags are all true.
   Derive the population from accepted design/registration, enforce all identity
   and coverage checks before aggregation, and keep incomplete diagnostics separate.

3. **Medium: the new scoring reduction is not the sealed author's float32 metric.**
   `score_cell`, `:908-915`, casts predictions/targets to float64 and sums per batch.
   `seal_contrast` and the adapter's header declare the author's float32 reduction.
   Float64 is a valid separate diagnostic, not proof of exact metric reproduction.
   Reuse the existing independently checked bounded-memory author reducer; retain
   both named reductions and their numerical difference. Do not infer that this
   precision issue explains the regime effect. The paper's TEST scores are also
   not a direct comparator for these VALIDATION scores, regardless of dtype.

## Other qualifications to preserve

- A fresh-process score proves fresh loading/inference, not output replay parity
  by itself. The new scorer compares checkpoint hashes, not predictions against
  an independent retained reference. Name that scope unless the missing comparison
  is supplied. Reuse the recorded restoration/monitor evidence where applicable.
- `author_datasets` (`:108`) constructs train, val AND test dataset objects. There
  is no test-scoring call in the new scorer, but "test bytes never read" is not
  established by this code. Distinguish loading, target access, fit, selection and
  scoring. Add explicit split selection for this path without changing training
  data; do not claim this observation proves test-based fitting or score selection.
- The 11,415.44 CPU s in CONTRAST.json is the training-run measurement. The new
  scoring subprocesses are outside it and SCORING.json has no cost ledger. Report
  fit + scoring/replay + closure totals before claiming the whole delivery stayed
  within the allocation. This review has NOT demonstrated a budget overrun.
- The report has no matched TimeFilter score on this validation subset. Do not
  compare 0.369 against its published test MAE as if task population matched.
  Any existing reference checkpoint evaluated here must disclose whether these
  targets contributed to its own checkpoint selection.

## What the retained results actually show

Arithmetic independently recomputed from the nine published per-cell metrics,
NOT from the original prediction arrays:

| Regime | Reported-complement MAE mean | SD across three seeds |
|---|---:|---:|
| R0 | 0.371382775 | 0.001598938 |
| R1 | 0.374611839 | 0.000618879 |
| R2 | 0.368979984 | 0.000425052 |

R2 minus R0 per paired seed: -0.001278637, -0.001861149, -0.004068588.
Mean improvement: **0.002402791 MAE, about 0.647% relative to R0**. R1 is worse
than R0 in all three retained pairs. This is useful directional development
evidence worth preserving, not a financial utility estimate or a decision on H1.
The label-disjoint scores and a comparable reference are still outstanding.
Three initialization seeds are not three independent datasets or market regimes.
Equal downstream updates do not equal total pretraining-plus-fitting cost.

Evidence: [probe](../evidence/RP155_MUSASHI/probe.py) and
[results](../evidence/RP155_MUSASHI/results.json). The unchanged real scorer is
called with a mocked subprocess transport carrying published child fixtures;
no test is presented as actual model replay. The window check uses timestamp
identities encoded as values through the real adapter, not equality of real prices.

Bounded existing tests: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_df_ecl_contrast_gates.py
-k 'F2 or F5'`: **9 passed, 6 deselected**. This is not a whole-suite claim.

## Disposition and next actions

**Training retained; scoring closure requires a successor; no global hold.**
Follow [the execution continuation](../../handoffs/SATOSHI_RP155_REVIEW_AND_EXECUTION_2026_09_24.md).
The runtime repairs of M5PHET d9ffc4d and the non-ML application lanes remain
independent of this scoring correction. Do not manufacture a new training campaign
or repeat A/B to solve a reporting/inference issue.
