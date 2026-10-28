# Data Quality And Feature Selection Standard

Status: implementation and real-data acceptance incomplete. This document is
the corrective contract, not a declaration that every control already runs.
Recovery of real observations precedes new profiling, selection and training.

## Evidence-Based Practice

There is no universal best selector or preprocessing recipe. Separate data
fitness from predictive relevance, and test the complete temporal pipeline.
The practices below are supported by primary documentation and research:

- [TFDV guide](https://tensorflow.github.io/tfx/guide/tfdv/): validate every
  split and both pre/post-transform data; check schema, missingness, drift and
  training-serving skew. A successful statistic computation is not admission.
- [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls):
  fit transformations and feature selection without leaking evaluation data.
- [Rolling-origin evaluation](https://otexts.com/fpp3/tscv.html): respect
  forecast origin and horizon rather than random row shuffles.
- [Feast point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins):
  historical joins respect decision-time support and TTL. Our contract also
  requires actual availability/publication time and revision identity; an
  event-time-only join is insufficient for revised macroeconomic releases.
- [NIST outliers](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)
  and [robust scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm):
  robust scale and potential-outlier flags aid inspection, not automatic proof
  that a financial extreme is erroneous. Distributional assumptions matter.
- [Feature-selection stability](https://www.jmlr.org/beta/papers/v18/17-514.html):
  measure sensitivity to sampling, not just one selected list.
- [Selection bias](https://www.jmlr.org/beta/papers/v11/cawley10a.html): tuning
  the selection criterion can overfit; development scores are not independent
  performance estimates.
- [PCMCI paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6881151/): temporal
  dependence and autocorrelation matter for conditional causal discovery.
  Observational associations are not unrestricted intervention/counterfactual
  identification. Record assumptions; NOT_IDENTIFIED stays neutral.

These references guide the controls; installing an entire new framework is
not required to repair the data. Use existing engines with tested adapters.

## Mandatory Control Matrix

| Stage | Required controls | Refusal / disposition |
|---|---|---|
| Acquisition | Instrument, venue/provider, bid/ask/mid, units, timezone, bar start/end, UTC availability, license, raw hashes, retries/rate limits, pagination and source coverage | Wrong schema/period or unexplained absence: recover before downstream work |
| Raw integrity | Finite positive prices, coherent OHLC, conflicting duplicates, ordering, source anomalies, known corporate actions/roll adjustments | Verify anomalies against independent observed quotes; never synthesize prices |
| Calendar | Versioned sessions/DST/holidays per source; observed versus expected open hours by month/year | Do not count weekends as gaps or redefine unexplained gaps as closures |
| Alignment | Event and available times, release lags, vintage, backward-only joins, staleness and no future/backfill | Missing availability evidence remains unresolved; do not certify causality |
| Coverage | Original rows, absent expected rows, nonfinite cells and retained prepared windows separately; longest gaps and consecutive loss | Aggregate finite percentage cannot excuse a lost month |
| Outliers | TRAIN-fitted MAD/IQR diagnostics, source validity checks, independent-source comparison, documented action | Statistical extreme alone is not corruption; zero MAD is undefined, not zero outliers |
| Transform | Source-appropriate returns and units, gaps not compressed into false fixed lags, causal rolling windows, scalers fitted on TRAIN, no centered filters/future smoothing | Perturb future rows; every transformed input at t must remain unchanged |
| Partitions | TRAIN2020-2023, VALIDATION2024, TEST2025; declared rolling intervals; target maturity/purge, no duplicate leakage | Date endpoints alone do not prove four complete usable years |
| Individual metrics | Quality admitted first; stationarity, spectral and nonlinear diagnostics with actual cadence and effective support | Do not apply regular-grid estimators to compressed gaps without qualification |
| Relationships | Common support per pair, pairwise n, missingness patterns, lag semantics, redundancy plus nonlinear dependence | Pair-specific support must not masquerade as one common complete population |
| Selection | Existing filter/wrapper controls, subsets by target, temporal stability, redundancy and relevance jointly; causal assumptions explicit | No causal absence inferred from NOT_IDENTIFIED; no tuning on a claimed holdout |
| Predictive validation | Native/common support, own naive per horizon and same rows/units; business masks, reproducible checkpoints and prospective scoring | No quality exception because a candidate beats naive |
| Operations | Gates at acquisition, preparation, deployment and worker claim; immutable data identity, bounded resources, status, physical copy, OLAP readback | No automatic successor trained on rejected/changed inputs |

## Outlier Policy

1. Reject impossible prices/OHLC and conflicting duplicates as source defects;
   recover observed replacements with provider/time identity. Retain the
   original quote source separately while repairing; do not retain invalid
   model experiments against the owner's deletion instruction.
2. Flag modified absolute z-score >3.5 and IQR fences as diagnostic screens.
   Compute reference parameters on TRAIN only. For zero MAD/IQR report an
   undefined diagnostic rather than manufacturing a zero score.
3. Real crashes, jumps and regime changes remain in the signal and targets.
   Corroborate extreme timestamps with independent sources before treating them
   as bad ticks. Daily macro steps and repeated as-of values are not hourly
   stale-quote defects merely because they repeat.
4. Any later robust scaling/clipping comparison is a separately identified,
   TRAIN-fitted experiment. No target denoising, default winsorization, centered
   Hampel window, global interpolation or future-informed thresholds.

## Actual Admission Defect Located

The retained GBPUSD profile reported 71,302 finite values out of 71,734 existing
decision rows (99.3978%). Its metadata said ADMISSIBLE. That denominator did not
measure the independent expected-open-hour grid or the eventual prepared-window
loss. The downstream phase-2 manifest copies train_coverage and an admissibility
label; the inspected construction does not enforce a monthly coverage gate.
The old profile did contain tail/MAD diagnostics; it is inaccurate to say no
statistics existed. It is equally inaccurate to treat their existence as proof
that source completeness and prepared coverage were accepted.

This is a demonstrated gap in the inspected path, not proof that every source
and every selection result has been fully traced. Audit source identities and
all candidates before recomputing phase-1/2/3 evidence on repaired data. Keep
ETH and unrelated work outside an EURUSD cleanup unless lineage implicates it.

## Current Implementation

- Real-data selected-input audit: 19 inputs, 72 months and 1,368 feature/month
  records, with TRAIN-only outlier thresholds; source clock and raw-source
  availability remain separate checks. Structural pass never authorizes fits.
- Tested acquisition scripts save independent raw quotes and typed candidates
  outside the repository, without modifying the lake or filling missing bars.
- Whole-inventory re-profiling, warehouse replacement of selection metrics,
  calibrated per-source calendars, complete prepared-window admission and
  end-to-end deployment guards are NOT YET ACCEPTED. They precede new modeling.
