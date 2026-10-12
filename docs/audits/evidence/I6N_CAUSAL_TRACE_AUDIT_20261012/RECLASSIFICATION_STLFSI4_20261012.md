# STLFSI4 reclassification and H72 correction (owner ruling 2026-10-12)

## Ruling

STLFSI4 is reclassified from PROVEN_SAFE (base audit) / RISK_UNRESOLVED (review)
to **PROVEN_POINT_IN_TIME_LEAK**. The Friday-dated weekly value was consumed
from about +23 h, while FRED publishes it the following Thursday: 313 of 313
Friday values in the retained data were used before publication, about 136 h
early (test `tests/test_i6n_stlfsi4_exclusion.py`, branch
`satoshi/i6n-stlfsi4-exclusion-20261012`). The row precedes t, but the value was
not publicly observable at t. Missing evidence still to retain as a reproducible
artifact: official release metadata (ALFRED vintages or FRED release calendar);
the release weekday currently comes from the source's published schedule, not
from a retained file.

## Consequences

- All prior H4 results (config fb70328d, plan 1b83490b, champion 356e4acc) are
  **DEVELOPMENT_TUNING_POINT_IN_TIME_CONTAMINATED**: retained as engineering
  evidence; their metrics are not promoted nor compared as valid skill.
- STLFSI4 is excluded from the input manifest (no assumed-lag shift). New
  identity: 19 features, optimization_config 3d922c07..., plan 614dc5f2...
  (v2.1.0), feature manifest 2ff34fa2..., availability contract 9668898c...
  No candidate whose input contained STLFSI4 is reused under it (tests refuse
  old checkpoints, migrants, fit caches and contexts).
- General availability gate: every external feature declares event_time,
  available_time, timezone, availability source and revision policy; a window
  consumes a value only when available_time <= t. Retained features: 12
  DECLARED_CONSERVATIVE (FRED daily rates rely on the Treasury same-day release
  as the availability source), 7 UNVERIFIED (producer code missing or as-of
  match below 0.95) -- mandatory gate before TEST, not a blocker for H72.
- H72 start is held (successor timer disabled) until it can start under the new
  identity only. Deploy commit: predictor e9747fb8.
- Re-enabling STLFSI4 later requires verifiable point-in-time availability.

## Categories (Musashi's addendum vocabulary, updated)

1. Direct input/target leak from >t: none found; four fail-closed mutations
   confirm (REVIEW.md).
2. Retrospective selection vs 2023 folds: FUTURE_RELATIVE_TO_2023_FOLDS.
3. In-sample ranking bias: TRAIN half of fitness labelled as regularizer.
4. Point-in-time availability: STLFSI4 PROVEN_POINT_IN_TIME_LEAK (new).
5. Unresolved provenance: 7 features, minute archives, champion row artifacts.
