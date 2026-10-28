# Data integrity incident and recovery

Status: DATA_INTEGRITY_HOLD. No new fit is authorized by a source download.

## Confirmed incident

The retained source has missing open-session observations in 2023. The selected
GBPUSD one-hour-return column also contains 395 nonfinite values in March-July
2023. Window construction amplifies these losses: the inspected four-year FIT
packet has no retained origins in those five months. The source's date range
therefore did not establish adequate training coverage. Earlier code tests and
metrics arithmetic do not certify this data requirement. Historical results
are being deleted under the owner's subsequent explicit instruction, without
replacement archives. The incident alone does not establish the unique cause
of their predictive performance. Original quote sources and recovery inputs
are not model experiment outputs and are kept for reconstruction.

The monitor-contrast and successor stopping-probe timers were disabled and their
training services stopped. Both workers had no remaining GPU compute process.
This hold supersedes earlier RUNNING/WAIT_PARENT declarations. The affected
predictive artifacts, mirrors, best registries and local outboxes were deleted
on the three roles. Scoped warehouse cleanup removes affected reports, metric
rows, causal/selection outputs and their dependent records, preserving
independent populations. This is not a
claim that Git history or every separate legacy archive has been purged.

## Requirements And Acceptance

| ID | Requirement | Test/evidence required |
|---|---|---|
| DQ01 | TRAIN 2020-2023, VALIDATION 2024, TEST 2025; separate rolling fit intervals | Explicit split boundaries, per-month counts, no substituted years |
| DQ02 | Observed input quotes only; no manufactured weekend bars or interpolated prices | Source bytes, hashes, documented clock; expected-open-grid comparison |
| DQ03 | Recover actual EURUSD/GBPUSD observations, never interpolate or preserve invalid experiments as backups | New acquisition roots; finite positive OHLC, duplicate/conflict tests |
| DQ04 | Audit every configured candidate and raw dependency, not just the previously selected inputs | Per-split/month missing/nonfinite/constant/outlier counts |
| DQ05 | Outliers are flags, not automatic deletions | Threshold fitted on TRAIN only, validation/test perturbation invariance; retain extremes |
| DQ06 | Account for missing source rows separately from business/target/window exclusions | Source coverage and prepared coverage both mandatory; months lost must block |
| DQ07 | Trace selection admission | Locate actual profile/selector versions, recorded metrics and thresholds; do not assume a profile score enforced data quality |
| DQ08 | No training before data acceptance | Disabled old timers; source/prepared checks and identity required before redeployment |

## Recovery Order

1. Stop affected work, then delete its model outputs under the owner's updated
   instruction. Do not create an alternative archive or retain a best registry.
2. Download authentic source candidates. HistData annual re-download confirms
   the same 2023 shortage, so it is not itself a repair. Yahoo's hourly endpoint
   returns 422 for 2023 because of its 730-day restriction. Authenticated
   FXMacroData reads succeed, but the inspected historical intraday-reference
   response contains official fixings, not a continuous hourly quote series.
   Do not upsample fixings or daily prices into missing hourly observations.
3. Use an independent quote feed for recovery and comparison; retain bid/ask
   and provider identity. A feed change needs a new dataset identity and overlap
   comparison, not an undocumented splice into old data.
4. Audit selected arrays and their raw upstream sources; separate calendar
   assumptions from observations and point-in-time availability.
5. Rebuild features from admitted raw sources, then verify aligned and window
   support. Recompute individual, cross-variable and causal evidence and the
   feature selection; no affected ranking or selected manifest is reused.
6. Re-run source and prepared-data admission; only then construct new model
   experiments. No numerical performance target is an acceptance shortcut.

## Unit And System Test Design

Acquisition: fixed-EST versus summer, bad period/pair, finite/OHLC checks,
identical versus conflicting duplicates, immutable reuse and failed downloads.
Quality: missing month, missing open hour versus weekend, all-null column,
constant column, zero MAD, real extreme retained, train-only thresholds,
reordered/duplicated timestamps, wrong feature count, split/month totals.
System: the actual defective source must be refused; downloaded candidates
must not be reported as admitted training data. Recovery writes outside the
admitted lake/repository. Closed affected campaigns are deleted, not modified
or restored.

## Limits

Coverage is measured against a versioned assumed FX calendar, not a certified
broker holiday schedule. Unexplained expected-open gaps remain unresolved;
neither dropping them nor marking every missing hour as closed is permitted.
Quality profiling is not proof of publication-time availability or causality.
Selection admission trace and repaired-dataset acceptance remain pending. See
[the full quality standard](DATA_QUALITY_AND_SELECTION_STANDARD.md) for the
primary-source review and control matrix beyond the first located defect.
