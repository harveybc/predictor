# I6-DMO data lane return (2026-10-14)

Branch `satoshi/i6dmo-data-20261014`, base `2a440265`. Tests were written
first (`feb3d2b1`) and the implementation followed (`58d489fe`). After Satoshi's
ruling of 2026-10-14, the as-of sampling tests came first (`59319218`, 8 failing)
and the implementation followed (`e77cbbd2`). Interface:
`INTERFACE_DATA.json`. Nothing was trained, no TEST bar was read, no service
was touched.

## DMO traceability

| DMO | Test (tests/i6dmo/test_data_dmo.py) | PRE | POST @e77cbbd2 |
|---|---|---|---|
| DMO02 | `test_dmo02_window_is_13_slot_instants_including_t_asof_sampled`, `..._perturbing_t_plus_1_and_later_changes_no_input`, `..._absent_slot_takes_previous_bar_never_next`, `..._weekend_slots_take_friday_close_asof`, `..._no_slot_reads_an_observation_after_its_instant`, `test_asof_refuses_only_when_no_observation_exists`, `test_asof_stale_beyond_weekend_closure_is_counted_not_refused` | FAIL (ImportError @feb3d2b1; as-of tests FAIL @59319218) | PASS |
| DMO03 | `test_dmo03_exact_h3_h5_targets_and_hourly_origins`, `..._missing_endpoint_is_not_bridged_by_next_row` | FAIL | PASS |
| DMO05 | `test_dmo05_masks_equal_direct_c1_calls_including_dst_weeks`, `..._business_rules`, `..._split_boundaries_purge_cross_boundary_labels` | FAIL | PASS |
| DMO06 | `test_dmo06_outer_and_inner_perturbation_leave_fit_preprocessing_unchanged` (with a positive control), `..._fit_statistics_zero_scale_is_typed` | FAIL | PASS |
| DMO08 | `test_dmo08_mae_and_naive_identical_rows_and_manual_replay`, `..._negative_control_replay_detects_tampered_target`, `..._missing_prediction_is_failure_not_dropped` | FAIL | PASS |
| DMO09 | `test_dmo09_weekly_sums_counts_and_explicit_absences`, `..._weekly_from_real_shapes` | FAIL | PASS |
| DMO16 | `test_dmo16_four_calendar_years_and_all_validation_origins` (independent recount), `..._no_sampling_or_shortening_knobs`, `..._incomplete_coverage_is_refused_not_shortened` | FAIL | PASS |
| supporting DMO13/18 | `test_test_rows_are_refused_before_materialization`, `test_stlfsi4_and_aliases_rejected`, `test_availability_gate_reports_unverified_and_blocks_confirmatory_test`, `test_frozen_bundle_digests_and_tamper_detection`, `test_successor_availability_contract_prepared_not_adopted` | FAIL | PASS |

The POST suite has 27 tests, all passed (65 s, CPU, 2G cap). Two test assertions were corrected during the first POST run:

- The primary C1 reason at a moved split boundary is `TARGET_AVAILABLE_AFTER_LABEL_CUTOFF`. The test now requires `TARGET_AT_OR_AFTER_SPLIT_END` to be among the reasons, rather than first.
- The late-start fixture now keeps the calendar and drops the bars, so the test separates `COVERAGE_INCOMPLETE` from `CALENDAR_INCOMPLETE`.

Command:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-data -- python -m pytest -q tests/i6dmo/test_data_dmo.py
```

## Input sampling rule

The rule is `DS2_ASOF_LAST_AVAILABLE`, from Satoshi's ruling of 2026-10-14 under PLAN section 4. It is recorded in `BUNDLE.json` under `input_sampling` and is part of the bundle identity.

- **Slot value.** Each of the 13 slot instants `s = t-24h..t` takes, per feature, the last retained finite observation with available_time (its UTC bar end) `<= s`. It never takes the next row and never interpolates.
- **Staleness.** The staleness `s - observation_time` is recorded per slot and feature in `<split>_aux.npz` as `slot_staleness_s`. It is not a model input.
- **Counted, not refused.** Origins stale beyond the longest calendar closure (49 h) and slots that fell back from a non-finite exact bar are counted.
- **Refusal.** An origin is refused only when some feature has no observation `<= s`, reported as `INPUT_NO_OBSERVATION_AVAILABLE`; the real bundle has none.

## Real frozen bundle

The bundle was built at `e77cbbd2` and re-verified, but it is not in Git.

- bundle_sha256 `3e42c3f5...c205`; `BUNDLE.json` file `0140dfbc...e03a`.
- Design `541ce40a...d381`.
- Input `ef22ddb5...b8` (no-STLFSI4, ends 2024-12-31T22:00Z, no TEST bar).
- Calendar `782603fd...265`, `FX_NY_17_ASSUMED_NOT_BROKER/i6n-v1`, labelled ASSUMED.
- Manifest `2ff34fa2...470`, availability contract `9668898c...bc9` (v1, sealed).

The model lane's `load_data_bundle` (@12e13752) accepted this bundle. For every head, the fit-only naive MAE equals the independent disk replay.

| Split | Rows | Valid labels h2/h3/h4/h5/h6/h12 | row_ids_sha256 | arrays_sha256 |
|---|---|---|---|---|
| fit [2020-01-01, 2023-11-06) | 18,144 | 17909/17440/17477/17051/17004/15761 | a7965dbb... | 112a66a7... |
| inner [2023-11-06, 2024-01-01) | 739 | 739/731/723/715/707/659 | c367a1a2... | bf5bb1f3... |
| validation [2024-01-01, 2025-01-01) | 4,889 | 4885/4830/4777/4723/4669/4345 | cef0d08e... | 91eb48fe... |

As-of statistics for materialized rows:

| Split | Origins with any as-of slot | Fell back from a NaN exact bar | Stale beyond 49 h | Max staleness |
|---|---|---|---|---|
| fit | 3,865 | 1,087 | 185 | 72 h |
| inner | 115 | 1 | 1 | 50 h |
| validation | 767 | 7 | 24 | 72 h |

Fit-only naive MAE per head: h2 0.00104006, h3 0.00128384, h4 0.00149475, h5 0.00168486, h6 0.00186241, h12 0.00276441.

Before the ruling, the exact-slot build (`5221615d`) had 14,279 / 624 / 4,122 rows. It is superseded and must not be used.

## Seven producers and clocks

Evidence is in `tools/i6dmo/producer_recovery.v1.json`. The producer is feature-eng `tools/eurusd_ps/`, the PS1 lane A code, at commits 79bb09d, 18cabb5 and 42272ad. It is on origin/master but missing from the stale local master. Its source is the 5m EURUSD lake file, not the 1h file the earlier audit used.

Reproduction covered the TRAIN rows only:

- The producer matched the retained PS1 batches at 100% (71,734 rows, relative error 1e-9).
- It matched the I6N INPUT at 100% of finite rows (29,155 rows before 2024).

| Feature | Status |
|---|---|
| px.close_loc, px.log_co, px.rv5, rg.atr_ratio, fx.gbpusd.logret_1h | RECOVERED (formula, producer@commit, bar-end clock) |
| fred.rates.dgs3mo.diff_5d, fred.rates.dgs7.logret_5d | STILL_UNRESOLVED. The producer and formula are recovered and reproduced exactly. The publication clock is not proven. Missing object: ALFRED vintage history (`realtime_start` per observation) for DGS3MO/DGS7 for 2019-2024, or the H.15 release timestamps. The retained parquet holds only the 2026-05-01 vintage. |

The successor availability contract `tools/i6dmo/feature_availability.successor.v2.json` (sha `c1c00b29...48d6`, version 2.0.0) is prepared but NOT adopted. It promotes the five recovered hourly features to VERIFIED_EVIDENCE and keeps both FRED features UNVERIFIED. It is a TEST gate input, not a training blocker. Running `gate(require_verified=True)` against it still refuses exactly dgs3mo.diff_5d and dgs7.logret_5d.

## Risks and owner decisions

1. **Weekend gap: resolved by ruling.** Under as-of sampling, 0 origins are refused for input. Staleness up to 72 h is recorded and counted, not hidden.
2. **Vendor gap in 2023.** From 2023-02-19 to 2023-07-23 only about 85 of 120 bars per week exist. Inputs use as-of values for those weeks. Targets still require the exact bar, so missing endpoints are counted as `TARGET_NOT_OBSERVED`. As a result, the label counts across heads are not monotone (h3 < h4 in fit).
3. **GBPUSD NaN.** These slots now take as-of values. 3,238 fit slots are counted under `nonfinite_exact_slots_by_feature`.
4. **Clock disagreement in DST-gap weeks.** The producer clock is New York local, bar start. The I6N clock policy is EU DST minus 5 h. In the DST-gap weeks, about 1,966 TRAIN rows sit 1 h later than the producer's bar end. They are stale, not leaked.
5. **Confirmatory TEST still blocked.** The sealed contract `9668898c` still marks all seven features UNVERIFIED. The successor v2 is ready to adopt for the TEST gate, but the FRED clocks still block it (`PROVENANCE_UNRESOLVED`).
6. **Development use of 2024.** The 2024 year was used in development selection. The outer VALIDATION year is reported as development.

## Long-family bundle preparation (`I6-DMO-LONG-DS12-STATIC-R0`)

This is preparation only. Nothing was fit, and no short-family validation outcome was read.

**Commits.** Tests came first (`b675db8e`: 3 failing, 3 passing). The implementation followed (`0a6d9584`). The `long_data_view` helper was added together with its test. Both test files now pass, 34 tests in total.

**Design and rule.**
- The bundle was built from the long data view: the short design plus the PLAN section 5 fields (heads 18/24/36/48/60/72, 144 h span, 12 h spacing, 13 observations).
- `data_contract_sha256` covers only the data fields. A design sealed by the model lane therefore has the same data identity.
- The rule is `DS12_ASOF_LAST_AVAILABLE`.

**Real bundle.** It was built and re-verified, but it is not in Git.
- Identity: bundle_sha256 `e88f72aa...77a1`, `BUNDLE.json` file `afcef506...9a75`, data contract `6e354fe6...5e75`, input `ef22ddb5...b8`.
- Earliest consumed bar: 2019-12-26T23:00Z. This is after the `DCA_2019_EU_DST_POLICY` span, which ends 2019-12-01, and inside the declared, not quote-verified, December-2019 warmup. Both are recorded in `history_provenance`.
- TRAIN was not shortened, and no input refusals occurred.

| Split | Rows | Labels h18/24/36/48/60/72 | Origins with as-of slots / stale beyond 49 h | row_ids_sha256 |
|---|---|---|---|---|
| fit | 15,017 | 14690/13623/11198/9052/6706/4509 | 15,017 / 574 | bcd38518... |
| inner | 611 | 611/563/467/371/275/179 | 611 / 6 | 73ab7376... |
| validation | 4,043 | 4025/3707/3073/2464/1856/1234 | 4,043 / 65 | 7c146e1c... |

Fit-only naive MAE per head: h18 0.00352008, h24 0.00410667, h36 0.00496066, h48 0.00583283, h60 0.00658154, h72 0.00746189. These equal the disk replay.

h72 support is small because of the C1 rule `q < force-flat`: 4,509 fit rows and 179 inner rows. Every valid label falls inside its own split (tested), so the trainer's `PURGE_VIOLATION` check cannot fire.

## Audit correction AUD-DMO-01 (2026-10-14, lane 1)

Satoshi's ASOF ruling was audited and withdrawn: EXACT sampling is the default again. Tests first (`7ee32315`: 7 failed, 8 errors, probe 1 REPRODUCED), implementation `7601904f` (probe 1 NOT_REPRODUCED; `tests/i6dmo` 214 passed). The loader now authenticates the sampling rule (`SAMPLING_RULE_MISMATCH`), and the ASOF treatment survives only as a separately named optional arm with its own `data_contract_sha256`. The ASOF bundles `3e42c3f5` (short) and `e88f72aa` (long) are retained under their ASOF identities, not deleted and not usable as exact DS2. The two ops test fixtures pinning ASOF rule names were updated mechanically (lane 4 should review).

**Exact short bundle** (code `7601904f`, rule `DS2_EXACT`): bundle `5f3d6782...0122`, `BUNDLE.json` file `489ebb8d...d603`, input `ef22ddb5...`. Rows fit/inner/validation 14,279 / 624 / 4,122 — identical row-id digests to the pre-ruling exact build (`1d0a295c` / `582af6e3` / `b4ccd80c`), an exact-reproduction cross-check. Origins lost vs ASOF: 3,865 / 115 / 767. Counted exclusions: fit 5,941 slot-not-observed + 291 non-finite; inner 200; validation 1,330 + 17. Fit-only naive MAE (equals disk replay): h2 0.00105245, h3 0.00130245, h4 0.00152185, h5 0.00172175, h6 0.00191438, h12 0.00288577.

**Exact long bundle** (rule `DS12_EXACT`): bundle `9f14c2d2...6ffb`, file `48460378...b507` — **EMPTY SUPPORT**: 0 rows in every split; all 23,153 / 944 / 6,243 candidates are `INPUT_SLOT_NOT_OBSERVED`, because every 144 h / 12 h ladder crosses a >=47 h weekend closure. The sealed long family cannot fit under exact sampling; it needs an explicit ruling (ASOF arm under its own identity, or a market-time window redesign). This is reported, not improvised.

**FRED clocks, bounded attempt:** financial-data's only FRED adapter (`_scripts/workers/stage13_gamma_macro_onchain_worker.py::fetch_fred` @ef0ba6612) queries `fred/series/observations` without realtime parameters, so only the 2026-05-01 current vintage is retained; no ALFRED/vintage adapter exists in any sibling repository. Missing artifact: ALFRED per-observation `realtime_start` for DGS3MO/DGS7 over 2019-2024, or H.15 release timestamps. Confirmatory TEST stays blocked (`PROVENANCE_UNRESOLVED`).

## Market-time sampling support study (order 85ea134e section 2, 2026-10-15)

Tests first (`c9de424e`, module absent), implementation `23aa4e51`; `tests/i6dmo` data+market files 45 passed. Committed artifacts: `market_grid/SUPPORT_STUDY.json` (sha `d2b6a238...0475`, over the full real TRAIN/inner/development), `CANDIDATE.short.json`, `CANDIDATE.long.json`. No fit; no rule declared optimal; TEST untouched.

Candidates `MG2_MARKET_EXACT` / `MG12_MARKET_EXACT`: 13 slots every 2 / 12 EXPECTED open-market hours (24 / 144 open-market-hour spans; elapsed spans vary — short is mostly 24 h with 71-73 h weekend/DST crossings, long is mostly 192 h with 240-241 h double-weekend windows). Exact finite observations required at expected endpoints; the 919 / 16 / 45 unplanned missing expected hours (fit/inner/validation, mostly the 2023 vendor gap) stay counted exclusions. Actual UTC slots, elapsed offsets and closure flags are retained for a later model.

Usable rows (vs retained elapsed grids): MG2 16,496 / 715 / 4,772 — a strict superset of DS2_EXACT (14,279 / 624 / 4,122; ds2-only = 0). MG12 13,174 / 546 / 3,804 where DS12_EXACT has 0; long heads get real support (fit S_72 = 4,019). Shared usable short+long origins: 13,123 / 546 / 3,760. Worked examples executed: Monday 2020-01-06T10Z crossing the weekend (slots end Friday 22:00, one closure flag, 72 h elapsed), DST spans 71 vs 72-73 h, the midweek-gap and current-t / t+1 perturbation cases run in tests.

**FRED (section 3, bounded):** a governed ALFRED fetch is impossible on this host — the only FRED adapter requires `FRED_API_KEY` from its environment and no governed EnvironmentFile provisions it (the point-in-time capture env carries only FD_* variables); the 2026-05-01 acquisition's key was not retained. Missing artifacts recorded in `producer_recovery.v1.json`: ALFRED per-observation `realtime_start` 2019-2024 (or H.15 release timestamps) and a governed `FRED_API_KEY` EnvironmentFile. Confirmatory TEST stays closed.

## Market-time pair training adapter (order 50d20d01 §1.1, 2026-10-15)

Tests first (`8d18bc54`: 3 failed / 12 errors), implementation `08b9fb5c`; `tests/i6dmo/test_market_bundle.py` 15 passed, `test_market_grid.py` 9 passed. CPU only, bounded crispdm jobs, no GPU, no TEST bar, no DOIN. Interface published in `INTERFACE_DATA.json` → `market_time`.

Sealed data-lane designs `market_grid/DESIGN.mg2.json` and `DESIGN.mg12.json` (schema `i6dmo.market_design.v1`): market spacing replaces the elapsed `input` block entirely, so no fixed elapsed span can be read out of them; any elapsed field or `elapsed_span_hours_fixed: true` raises `CONTRADICTORY_TIME_FIELDS`. Bundles use schema `i6dmo.market_bundle.v1` (never `i6dmo.data_bundle.v1`), publish no fixed `offsets_hours` ladder, and refuse cross-adoption (`MARKET_RULE_MISMATCH`, `MARKET_DATA_IDENTITY_MISMATCH`, `SAMPLING_RULE_MISMATCH` against the DS2 contract).

Per-window model inputs, verified on the real bundles: `offsets_hours` signed from t with the last column exactly 0, `adjacent_delta_hours` in RAW hours with the first column 0 (the /24 conversion is the model's), `closure_crossed` 0/1 with the first column 0, plus `slot_instants` UTC. Value channels come from the exact slot source rows; all finite.

| Bundle | bundle_sha256 | BUNDLE.json | Rows fit/inner/validation | Elapsed span range | Rows crossing a closure |
|---|---|---|---|---|---|
| MG2 | `21400c51…a55da` | `471ea311…f047d` | 16,496 / 715 / 4,772 | 24–73 h | 2,217 |
| MG12 | `c3867b5c…7670b` | `e6595c0e…5d866` | 13,174 / 546 / 3,804 | 191–241 h | 13,174 |

Data identities `d5d97a6e…` (MG2) and `0449cd4e…` (MG12); input `ef22ddb5…`; calendar `782603fd…` ASSUMED_NOT_BROKER. Reconciliation against study `d2b6a238`: every split MATCH (difference 0) and shared usable origins 13,123 / 546 / 3,760 equal to the study — status RECONCILED, recorded in each manifest with `counts_are_not_quotas: true`.

Fit-only naive MAE (equals the independent disk replay on identical rows): MG2 h2 0.00104852 … h12 0.00276528; MG12 h18 0.00353112 … h72 0.00753410. Exclusions are typed and retained, e.g. MG2 fit 1,394 slot-not-observed + 730 non-finite + 186 outside-week + 4,358 no-valid-head, with 919 unplanned missing expected open-market hours never compressed.

## FRED point-in-time provenance after the reviewer's ALFRED recovery (2026-10-15)

Tests first (`79359d56`, 6 failing), record update in this commit; `tests/i6dmo/test_fred_provenance.py` 6 passed. Reviewer artifacts (`retsu/i6dmo-market-time-review-20261015` @3508e149, `market_time/provenance/`) are cited by path and digest; nothing was re-downloaded and no credential was used.

**Withdrawn as FALSE:** my earlier claim that no unauthenticated vintage path existed and that a governed `FRED_API_KEY` EnvironmentFile was a missing artifact. ALFRED's public Download Data form ("Observations, Initial Release Only", column `realtime_start_date`) needs no credential.

**Typed PARTIAL clock state** for `fred.rates.dgs3mo.diff_5d` and `fred.rates.dgs7.logret_5d`: `PARTIAL_COVERAGE_REFUTED_BY_ALFRED_NOT_PROVEN_LOOKAHEAD`, carrying the measurement — 1,587 per-observation first-release rows per series (2018-12-03..2024-12-31), none earlier than its observation, but 339 (21.4%; Fridays and the Thursday before a holiday, ALFRED lag 3-5 days) available LATER than the declared (D+2) 00:00 UTC instant. Because ALFRED `realtime_start` is an upper bound on availability, this is a bounded refutation of the declared clock's coverage by that artifact, **not** proven look-ahead; accepting H.15's documented 4:15pm ET posting as the availability instant is a ruling for Musashi/the owner. Per-release H.15 timestamps for 2019-2024 remain genuinely missing (dated archive pages 404; RSS carries only the current release).

**Value-level exception recorded** in the replay rule and in the successor availability contract (now sha `5f5de7d8...22ee`): observation 2020-08-19 was published EMPTY and back-filled at vintage 2020-08-25, so under the declared as-of rule the feature reads stale/NaN there and never 0.11 (DGS3MO) / 0.48 (DGS7). Any future TEST replay must honour it. The replay rule also records the quarter-chunking that defeats left-censoring and stays under ALFRED's 450-vintage cap.

**Gate unchanged:** `confirmatory_gate()` still refuses with `PROVENANCE_UNRESOLVED`, and every record states that incomplete provenance blocks the confirmatory TEST **only** — labelled development execution, including the MG2/MG12 development fits, is not blocked.
