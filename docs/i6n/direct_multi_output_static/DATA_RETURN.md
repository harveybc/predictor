# I6-DMO data lane return (2026-10-14)

Branch `satoshi/i6dmo-data-20261014`, base `2a440265`. Tests were written
first (`feb3d2b1`) and the implementation followed (`58d489fe`). Interface:
`INTERFACE_DATA.json`. Nothing was trained, no TEST bar was read, no service
was touched.

## DMO traceability

| DMO | Test (tests/i6dmo/test_data_dmo.py) | PRE @feb3d2b1 | POST @58d489fe |
|---|---|---|---|
| DMO02 | `test_dmo02_window_is_13_exact_timestamps_including_t`, `..._perturbing_t_plus_1_and_later_changes_no_input`, `..._absent_slot_makes_origin_input_unavailable_never_filled` | FAIL (ImportError) | PASS |
| DMO03 | `test_dmo03_exact_h3_h5_targets_and_hourly_origins`, `..._missing_endpoint_is_not_bridged_by_next_row` | FAIL | PASS |
| DMO05 | `test_dmo05_masks_equal_direct_c1_calls_including_dst_weeks`, `..._business_rules`, `..._split_boundaries_purge_cross_boundary_labels` | FAIL | PASS |
| DMO06 | `test_dmo06_outer_and_inner_perturbation_leave_fit_preprocessing_unchanged` (with a positive control), `..._fit_statistics_zero_scale_is_typed` | FAIL | PASS |
| DMO08 | `test_dmo08_mae_and_naive_identical_rows_and_manual_replay`, `..._negative_control_replay_detects_tampered_target`, `..._missing_prediction_is_failure_not_dropped` | FAIL | PASS |
| DMO09 | `test_dmo09_weekly_sums_counts_and_explicit_absences`, `..._weekly_from_real_shapes` | FAIL | PASS |
| DMO16 | `test_dmo16_four_calendar_years_and_all_validation_origins` (independent recount), `..._no_sampling_or_shortening_knobs`, `..._incomplete_coverage_is_refused_not_shortened` | FAIL | PASS |
| supporting DMO13/18 | `test_test_rows_are_refused_before_materialization`, `test_stlfsi4_and_aliases_rejected`, `test_availability_gate_reports_unverified_and_blocks_confirmatory_test`, `test_frozen_bundle_digests_and_tamper_detection` | FAIL | PASS |

The POST suite has 22 tests, all passed (51 s, CPU, 2G cap). Two test assertions were corrected during POST:

- The primary C1 reason at a moved split boundary is `TARGET_AVAILABLE_AFTER_LABEL_CUTOFF`. The test now requires `TARGET_AT_OR_AFTER_SPLIT_END` to be among the reasons, rather than first.
- The late-start fixture now keeps the calendar and drops the bars, so the test separates `COVERAGE_INCOMPLETE` from `CALENDAR_INCOMPLETE`.

Command:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-data -- python -m pytest -q tests/i6dmo/test_data_dmo.py
```

## Real frozen bundle

The bundle was built and re-verified, but it is not in Git.

- Code `58d489fe`, bundle_sha256 `5221615d...73c2`, design `541ce40a...d381`.
- Input `ef22ddb5...b8` (no-STLFSI4, ends 2024-12-31T22:00Z, no TEST bar).
- Calendar `782603fd...265`, `FX_NY_17_ASSUMED_NOT_BROKER/i6n-v1`, labelled ASSUMED.
- Manifest `2ff34fa2...470`, availability contract `9668898c...bc9`.

The model lane's `load_data_bundle` (@12e13752) accepted this bundle.

| Split | Rows | Valid labels h2/h3/h4/h5/h6/h12 | row_ids_sha256 |
|---|---|---|---|
| fit [2020-01-01, 2023-11-06) | 14,279 | 14278/14089/13919/13731/13560/12487 | 1d0a295c... |
| inner [2023-11-06, 2024-01-01) | 624 | 624/616/608/600/592/544 | 582af6e3... |
| validation [2024-01-01, 2025-01-01) | 4,122 | 4121/4065/4014/3958/3907/3587 | b4ccd80c... |

For the fit split, the fit-only naive MAE in the bundle's `fit_statistics` equals the independent disk replay for every head. For example, h2 is 0.00105245 and h12 is 0.00288577.

## Seven producers and clocks

Evidence is in `tools/i6dmo/producer_recovery.v1.json`. The producer is feature-eng `tools/eurusd_ps/`, the PS1 lane A code, at commits 79bb09d, 18cabb5 and 42272ad. It is on origin/master but missing from the stale local master. Its source is the 5m EURUSD lake file, not the 1h file the earlier audit used.

Reproduction covered the TRAIN rows only:

- The producer matched the retained PS1 batches at 100% (71,734 rows, relative error 1e-9).
- It matched the I6N INPUT at 100% of finite rows (29,155 rows before 2024).

| Feature | Status |
|---|---|
| px.close_loc, px.log_co, px.rv5, rg.atr_ratio, fx.gbpusd.logret_1h | RECOVERED (formula, producer@commit, bar-end clock) |
| fred.rates.dgs3mo.diff_5d, fred.rates.dgs7.logret_5d | STILL_UNRESOLVED. The producer and formula are recovered and reproduced exactly. The publication clock is not proven. Missing object: ALFRED vintage history (`realtime_start` per observation) for DGS3MO/DGS7 for 2019-2024, or the H.15 release timestamps. The retained parquet holds only the 2026-05-01 vintage. |

## Risks and owner decisions

1. **Exact slots and the weekend gap.** The 24 h window of the first ≈13 operational hours of each week falls on weekend instants where no bar exists. Those origins are refused as `INPUT_SLOT_NOT_OBSERVED`: 5,941 of the 23,153 fit candidates. This lane does not substitute the last Friday bars, because that would be an improvisation. If the order meant market-time windows, a ruling is needed.
2. **Vendor gap in 2023.** From 2023-02-19 to 2023-07-23 only about 85 of 120 bars per week exist. 2023 has 5,505 of 6,360 expected bars, and 39 weeks have missing bars. These are counted as exclusions, not filled.
3. **GBPUSD NaN.** 291 fit and 17 validation origins are refused as `INPUT_VALUE_NOT_FINITE`.
4. **Clock disagreement in DST-gap weeks.** The producer clock is New York local, bar start. The I6N clock policy is EU DST minus 5 h. In the DST-gap weeks, 1,966 TRAIN rows sit 1 h later than the producer's bar end. They are stale, not leaked.
5. **Sealed contract still says UNVERIFIED.** Contract `9668898c` still marks all seven features UNVERIFIED. Promoting the five recovered features needs a successor contract version. Confirmatory TEST stays blocked (`PROVENANCE_UNRESOLVED`).
6. **Development use of 2024.** The 2024 year was used in development selection. The outer VALIDATION year is reported as development.
