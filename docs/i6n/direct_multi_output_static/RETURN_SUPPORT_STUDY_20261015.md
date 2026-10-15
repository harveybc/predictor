# I6-DMO return: baseline closure, market-time support study, next-experiment options

Order `85ea134e` (`MUSASHI_TO_SATOSHI_STATIC_BASELINE_CLOSE_AND_MARKET_GRID_20261015.md`),
section 4. CPU only. No fit, no GPU work, no TEST read, no DOIN change in this round.
`tests/i6dmo`: **319 passed** on the merged tree (one bounded CPU job).

## 1. Commits and sealed identities

| Package | Commit |
|---|---|
| ops / integration (closure and status) | `d46f61ed` |
| data (market-grid support study, FRED attempt) | `f320cac2` |
| model (sealed hourly control) | `e6bbaf8a` |
| strategy (versioned instrument configuration) | `8870a853` |
| integration merge of all four | this commit, on `satoshi/i6dmo-static-20261014` |

| Artifact | Identity |
|---|---|
| Short exact-DS2 effective config (measured baseline) | `287e4f11…` — **unchanged**; design `541ce40a` |
| Hourly control design (`EXPERIMENT.short_hourly.json`) | design `971ba9c4`, seal `d29a766d`, 25 observations |
| Frozen comparison rules (`COMMON_SUPPORT_RULES.json`) | `9d61117f` |
| Market-grid support study (`market_grid/SUPPORT_STUDY.json`) | report `d2b6a238` |
| Instrument/cost configuration (`tools/i6dmo/instrument_eurusd.v1.json`) | `f2443141`, version `i6dmo-eurusd-costs-2026-10-15.v1` |
| Calendar | `782603fd`, `FX_NY_17_ASSUMED_NOT_BROKER` (ASSUMED status preserved) |

## 2. Candidate support study, as measured

**This is candidate support preparation, not a predictive measurement.** The study's own status
field says so (`SUPPORT_STUDY_NOT_A_PREDICTIVE_MEASUREMENT`), it uses no forecast outcome to
select rows, and **no sampling rule is declared optimal from these counts.**

Usable origins (FIT / INNER / VALIDATION):

| Candidate | FIT | INNER | VALIDATION |
|---|---|---|---|
| MG2_MARKET_EXACT (13 slots, 2 open-market hours) | 16,496 | 715 | 4,772 |
| MG12_MARKET_EXACT (13 slots, 12 open-market hours) | 13,174 | 546 | 3,804 |
| DS2_EXACT (retained elapsed grid, measured baseline) | 14,279 | 624 | 4,122 |
| DS12_EXACT (retained elapsed grid) | **0** | **0** | **0** |

- MG2 is a **strict superset** of DS2_EXACT: `ds2_only_vs_mg2 = 0` in every split, with
  `mg2_only_vs_ds2` = 2,217 / 91 / 650 additional origins.
- MG12 recovers a usable long population (13,174 / 546 / 3,804) where the sealed elapsed DS12
  grid has **empty support in every split** — the typed `LONG_EXACT_EMPTY_SUPPORT` result stands
  unchanged and is preserved, not overwritten.
- Shared short+long usable origins: **13,123 / 546 / 3,760**.
- Expected open-market hours vs unplanned missing bars: **24,072 / 960 / 6,288 expected** with
  **919 / 16 / 45 unplanned missing**. Calendar closures are not counted as expected hours;
  vendor gaps inside an open session stay missing observations with counted exclusions, never
  compressed closures.
- Effective elapsed spans are **not fixed**: MG2's 24 open-market hours span 24 h normally and
  71–73 h across a weekend closure (worked Monday-origin example: origin 2020-01-06T10:00Z,
  actual offsets −72 … 0 h, one closure-crossing slot; the DST cases give distinct closure spans
  of 71.0 and 72.0 h). MG12's 144 open-market hours span 192 h typically and 240–241 h across a
  double weekend. Per-window UTC timestamps, actual offsets and closure flags are retained, and
  the candidates are labelled in open-market hours — never as fixed 24/144 elapsed hours.
- Worked cases included: Monday origin crossing the weekend, a DST case, an unexpected midweek
  gap, a current-`t` positive control and a `t+1` perturbation control.

## 3. Finished vs awaiting design vs computing

**Nothing is computing.** No fit is running on any host; the retained ASOF root stays HELD with
only its publication timer; campaign timers cannot resolve design decisions by polling.

- **FINISHED (measured and closed).** The exact-DS2 short baseline: `short.fit` and
  `short.report` COMPLETE; six heads NOT_ELIGIBLE versus the per-horizon naive on 2024
  development — scoped to **one minimal static R0 exact-DS2 model**, with no claim about other
  predictors, modular extraction, R2, other retraining policies or the strategy, and no causal
  attribution to ASOF, in-sample fitness or the 2024 selection (this was not a controlled
  ablation). Publication closure: **55 delivered reports, 2,011 metric rows, 0 pending,
  0 mismatch, 28 files verified on each physical mirror** (reports are not metric rows).
- **FINISHED (preparation).** The market-time support study, the sealed hourly control design
  with frozen common-support rules, and the versioned instrument/cost configuration.
- **AWAITING A NEW SCIENTIFIC DESIGN.** The long family (held `WAIT_DESIGN_RULING`), the pair
  and the strategy stage, and any further short contrast — all need an owner/Musashi decision
  among the options below, not more compute.

## 4. Next-experiment options (for the owner / Musashi)

**(a) Seal MG2 and MG12 market-time designs.** Requires a model change: the encoder must consume
actual elapsed offsets and closure flags instead of assuming uniform wall-time spacing; new
identities and bundles; targets stay exact elapsed `t+h` under the original business masks.
*Would establish*: whether a market-time short/long pair is fittable and eligible, and it
unblocks the long family. *Would not establish*: that market-time sampling is superior — the
support counts alone cannot decide that, and the comparison against DS2_EXACT is confounded by
the extra 2,217/91/650 origins.

**(b) Fit the sealed 25-observation hourly control** under the frozen common-support rules
(`9d61117f`). Requires no new design work. *Would establish*: the downsampling contrast for the
**short family only**, on the declared common population. *Would not establish*: anything about
the long family, encoders, R2 or cadence; and a reused model scored on a subset is explicitly
not the same as a model trained on that common TRAIN population.

**(c) Long-family alternatives if market-time is declined.** Either a declared ASOF arm under its
own identity, or another grid design. *Would establish*: a fittable long family. *Would not
establish*: comparability with the exact elapsed treatment — an ASOF arm is a different
scientific treatment and must stay separately identified, never a substitution.

**(d) Authentic-encoder/R2 and cadence contrasts** as separate later designs, each with its own
identity. *Would establish*: the specific effect of that one change. *Would not establish*:
anything attributable to downsampling — combining changes and crediting one of them is exactly
the inference this return refuses to make.

## 5. Blockers

- **FRED point-in-time evidence is still missing.** No governed ALFRED path exists: the only FRED
  adapter in the fleet queries current-vintage observations and needs a `FRED_API_KEY` from a
  governed EnvironmentFile that does not exist for this campaign. Required objects: ALFRED
  per-observation `realtime_start` for DGS3MO and DGS7 over 2019–2024, **or** the H.15 release
  timestamps, **and** a governed key file. Until then confirmatory **TEST stays closed**
  (`PROVENANCE_UNRESOLVED`).
- **All four strategy costs remain `ASSUMED_NOT_BROKER_VERIFIED`** (spread, slippage, commission,
  swap) in instrument configuration `f2443141`. MT5 lab live-spread rows are retained but
  untranscribed; no cost is presented as broker-verified.
- The 19-feature set still used 2024, so every number here is development, not an untouched
  holdout. The calendar remains ASSUMED, not a verified broker calendar.

## 6. M1–M8

Statuses and percentages are **preserved unchanged**. Only M6's descriptive text reflects the
actual state (short closure / long support redesign); no scientific percentage moved in this
round or the previous one.
