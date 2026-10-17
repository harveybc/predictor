# Separate Forecast Families: Retention And Strategy Handoff

Owner directive 2026-10-17. Preserve running window/core experiments unchanged.
No new fits, no DOIN edits, no live-money execution. Use the retained weekly
checkpoints; an annual recipe is a sequence of models, not one final model.

## Requirements And Test Design

| ID | Contract | Acceptance evidence |
|---|---|---|
| PR01 | Family promotion requires STRICTLY negative native AND common-support error for EVERY output of the actual model. Never promote by mean alone. | One failing, equal, missing or nonfinite head rejects promotion, even with an excellent mean. |
| PR02 | Preserve each checkpoint's weights, config, scales, seed, period, monitor, runtime and digest. Export additional full .keras archives without modifying immutable original terminals or fitting. | Rebuilt weights and archive-reloaded tensor hashes agree with the retained training digest. |
| PR03 | A full-year recipe requires all declared periods. A failed zero-label period may acquire a supplemental omission ONLY from digest-pinned prepared arrays proving no supported label. Keep original failure intact. | Positive-label mutation and changed prepared digest are rejected. |
| PR04 | CSVs use UTC ORIGIN timestamps and explicit physical-hour column names. Each head has an availability column; missing values are blank, never current-price filled. | Duplicate/shifted origins, invalid flags, missing heads and invalid values fail closed. |
| PR05 | Long forecasts open; short and long manage in horizon order, SL-before-TP. Exactly one concurrent position per asset. | Existing pinned business policy and executor regression tests plus CSV adapter replay. |
| PR06 | Mandatory market-open+12h and market-close-12h boundaries, versioned calendar, no new entry after cutoff; forced exit does not depend on forecasts. | Boundary, missing-forecast, weekend-gap and early-close tests. No calendar inference or weekend forward fill. |
| PR07 | Freeze recipes before TEST. Existing point-in-time source gate remains technical prerequisite, not another chat approval. Historical weekly checkpoints selected on 2024 are NOT a valid 2025 weekly evaluation ensemble. | TEST remains unread; missing PIT receipt or prospective weekly recipe returns explicit blocked state. |
| PR08 | Generated status, durable manifests, two physical copies, OLAP outbox/readback. Code in git, weights outside git on artifact storage. | Idempotent operation, original digest checks, no refit, monitorable status and publication receipts. |

## Sequence

1. Verify saved checkpoints and complete annual families; retain descriptive
   per-head records but separately enforce all-output eligibility.
2. Certify the last long-only zero-label period through a supplemental proof.
   Never edit the original FAILED receipt to make it look successful.
3. Seal eligible short/long references. Keep historical best-by-horizon and
   recipe-level records separately; a mosaic is not one trained model.
4. Export forecast CSVs and full Keras compatibility archives from saved weights.
   Preserve canonical rebuild+load_weights, which remains the scoring path.
5. Verify the strict CSV-to-business-policy route on synthetic replay before
   any real strategy run. Retain offsets, masks and original price scale.
6. TEST evaluation follows the frozen recipe and availability gate. If blocked,
   report the exact missing object; do not invent source availability or use a
   checkpoint selected on its own TEST week. Demo remains separately authorized
   after positive costed DEVELOPMENT simulation, never real-money deployment.

Research scope remains retrospective DEVELOPMENT 2024. All-head negative MAE
does not establish profitability or independent predictive ability.
