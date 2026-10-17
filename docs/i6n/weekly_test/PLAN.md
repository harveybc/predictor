# Weekly TEST Replication

Owner authorization: evaluate both retained winning families on 2025, preserving
the weekly four-calendar-year recipe. This supersedes the earlier execution hold
in the pair-retention plan; source limitations remain disclosed, not certified.

## Frozen Requirements

- Short: the actual seven-head winner (2, 3, 4, 5, 6, 12, 18 hours).
- Long: the six-head winner (18, 24, 36, 48, 60, 72 hours), with fresh seeded
  twelve-head parent initialization and projection, not a newly seeded six-head graph.
- Nineteen features, unchanged ordering and formulas; STLFSI4 remains excluded.
- Thirteen observations every two expected open-market hours, including t;
  closed hours do not count as slots, observed gaps are not filled.
- Keep each family's original conservative support and TRAIN-only scale recipe.
- Each weekly cutoff uses exactly four preceding calendar years, mature labels,
  fresh seed 6102026, and the original business mask (12 hours after opening,
  forced flat 12 hours before closing, exact t+h, separate masks per head).
- AdamW, batch 256, learning rate .001, weight decay .0001, 200 epochs,
  patience 10, min_delta 1e-8; the original mean TRAIN/evaluation-week stopping
  rule remains unchanged. No ANY_HEAD rule and no new INNER split.
- Original-price MAE and its own naive for each horizon, on identical rows;
  relative error = (MAE_h - naive_h) / naive_h. Pool error sums/counts annually.
- Retain every weekly .keras archive, weights, scales, effective configuration,
  predictions, training history and digests. Publish metrics via durable outbox
  with readback; archive on disk and a second machine. No large binary in git.
- One heavy process on the preferred GPU; do not use the second GPU sharing RAM.
  A finite timer advances weeks and families without an agent watching logs.

## Interpretation

Only the evaluated year changes. The original stopping rule consults that
week's evaluation targets; therefore TEST participates in checkpoint selection.
Label results TEST_2025_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS, never independent
confirmatory test performance. Earlier TEST weeks can enter later four-year
TRAIN windows after their labels mature, as explicitly requested.

Inherited limitations: current-vintage FRED, declared publication clocks and an
assumed rather than broker-certified calendar. Extending the source-clock rule
to 2025 is not an independent quote-clock certification.

## Acceptance And Traceability

| ID | Requirement | Evidence |
| --- | --- | --- |
| WT01 | Same feature recipe, intact retained prefix | source generation prefix comparison |
| WT02 | Four-year weekly intervals, mature targets | prepared period contracts and tests |
| WT03 | Exact origin, masks and horizon-specific naive | preparation invariants and tests |
| WT04 | Unchanged model, seed and stopping | configuration and initial tensor comparison |
| WT05 | Honest TEST identity | result/report split tests |
| WT06 | Autonomous progression, durable artifacts | timer units, generated status and close |
| WT07 | Paired annual metrics | replay from retained prediction arrays |

Implementation order: source reconstruction, CPU contract tests, data-only
preparation, unchanged GPU trainer, first real fit check, unattended progression.
