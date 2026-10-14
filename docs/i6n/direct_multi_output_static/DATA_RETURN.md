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
