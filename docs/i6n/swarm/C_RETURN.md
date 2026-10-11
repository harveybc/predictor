# Lane C return: complete weekly units, 2023 folds, support and data

Branch `satoshi/i6n-swarm-C-weekly-20261011`, base `5c383d17`. Real plan
`plan_sha256 dd34986a46be92a0b3986be1be9b52081e922f266d22f5dde1f8f037b10ce9e7`
(PLAN.json file `2424c6a9...`), VERIFIED. No fit, no price read for the plan,
no TEST row, no download, no service.

## Narrative

The builder enumerates every calendar week whose operational band meets
DEVELOPMENT 2024 from the retained nominal calendar
`FX_NY_17_ASSUMED_NOT_BROKER`/`i6n-v1` (byte-identical to the calendar bound
by `CLOCK.json`, checked by digest; nominal, not certified): **53 weekly refit
units, 2023-12-31 .. 2024-12-29, none skipped**. Each refits at the previous
Friday close, trains on four calendar years, uses the last 28 days as purged
inner and is released at the following open. Support was generated from the
real timestamps (30,854 rows, 2019-12-01T23:00Z .. 2024-12-31T22:00Z) with C1
eligibility before any fit. **633 (week, h) dispositions per arm, 1,266 for
two arms**; h48/h60/h72 have no support in the final week, whose targets would
fall in 2025 (typed NO_ELIGIBLE_ROWS, never scored).

The counts reproduce the retained pilots exactly (W2024-01-07 MANAGEMENT:
h2 18,346/350/94; h3 17,873/346/93; h72 4,629/70/24; h12 TRAIN/inner
16,158/310). On real timestamps they also equal `build_weekly_partitions`
for both roles, every business head and split in W2024-01-07, W2024-03-10,
W2024-11-03, W2024-12-29 and 2023Q4. All 645 generated cell configs pass
`i6n_historical_cell.authenticated`.

Inner folds use the close of the last week entirely inside each 2023 quarter:
2023-03-31T21Z, 06-30T21Z, 09-29T21Z, 12-29T22Z. **2023Q4 is READY**.
**2023Q1-Q3 are WAIT_DEPENDENCY**: their context starts on 2019-03-13,
06-12 and 09-11, but the successor input and the clock scope certificate
start on 2019-12-01. Using only local files, I verified that 4,499 legacy
rows of 2019-03-12..2019-12-01 reconcile exactly with the digest-verified 2019
annual archive. Max price difference is 0. The successor correction shifts 432
rows and does not collide. The two in-span clock cases that passed, 2019-03-18
and 2019-07-15, replay identically. The **exact missing object** is the planned
independent case **2019-10-28** (INCONCLUSIVE, HTTP 429). The governed resume
tool does not admit 429, so I made no request. A scope certificate extension
is also missing. History is not shortened, and no 2024 week is used as a
substitute. Until this is resolved, B's rule means no candidate has a
four-fold fitness.

## Real counts (MANAGEMENT; sums over the 53 weeks)

| h | weeks READY | VALIDATION n (MGMT / ENTRY) | TRAIN n per week | INNER n per week | 2023Q4 TRAIN/INNER |
| --- | --- | --- | --- | --- | --- |
| 2 | 53 | 4885 / 3481 | 18311-18346 | 341-376 | 18320/363 |
| 3 | 53 | 4830 / 3478 | 17837-17873 | 336-372 | 17846/359 |
| 4 | 53 | 4777 / 3477 | 17871-17906 | 333-368 | 17878/355 |
| 5 | 53 | 4723 / 3475 | 17441-17476 | 329-364 | 17447/351 |
| 6 | 53 | 4669 / 3473 | 17390-17425 | 325-360 | 17395/347 |
| 12 | 53 | 4345 / 3461 | 16122-16158 | 301-336 | 16122/323 |
| 18 | 53 | 4025 / 3453 | 15021-15063 | 277-312 | 15021/299 |
| 24 | 53 | 3707 / 3447 | 13924-13972 | 253-288 | 13924/275 |
| 36 | 53 | 3073 / 3073 | 11451-11498 | 206-240 | 11451/227 |
| 48 | 52 | 2464 / 2464 | 9261-9292 | 161-192 | 9268/179 |
| 60 | 52 | 1856 / 1856 | 6858-6886 | 116-144 | 6885/131 |
| 72 | 52 | 1234 / 1234 | 4602-4629 | 69-96 | 4629/83 |

Per-unit/role/h/split rows: `evidence_C/SUPPORT_BUSINESS.csv`. Observed bars
per week: `evidence_C/WEEK_COVERAGE.csv`. 2023 has 86.6% of calendar bars
(2024 98.5%), which reduces support in every 2024 TRAIN window.

## Requirement -> test -> result

| Requirement | Test | Result |
| --- | --- | --- |
| Retained nominal calendar reproduced, id/version recorded | `test_nominal_calendar_reproduces_retained_identity` | PASS |
| All 2024 weeks, refit each week, release at open | `test_all_2024_weeks_enumerated_with_weekly_refit` | PASS |
| 4 calendar years + purged 28-day inner | `test_four_calendar_years_and_purged_inner_per_unit` | PASS |
| DST | `test_dst_weeks_keep_elapsed_hour_semantics` | PASS |
| Leap year | `test_leap_year_history_start_and_leap_day_origins` | PASS |
| 2024/2025 boundary, TEST never opened | `test_2024_2025_boundary_...`, `test_test_rows_in_source_are_refused` | PASS |
| Minimum data; never shorten | `test_min_data_coverage_...`, `test_minimum_label_support_threshold_is_typed` | PASS |
| Horizon without support | `test_horizon_without_calendar_support_is_never_scored` | PASS |
| Incomplete weeks | `test_incomplete_weeks_stay_listed_with_typed_exclusions` | PASS |
| Same counts as reference partitioner | `test_counts_match_weekly_partitions_reference_exactly` + real cross-check | PASS |
| Trajectory heads have own support | `test_trajectory_head_offsets_have_own_support` | PASS |
| 2023 quarter folds; WAIT with exact object | `test_2023_quarter_folds_...`, `test_folds_before_certificate_scope_...` | PASS |
| Sealed, redacted, tamper-evident artifacts | `test_plan_is_deterministic_and_sealed`, `test_file_cli_...` | PASS |
| Annual sum(err)/sum(n), identical naive | `test_annual_mae_is_sum_over_sum...`, `test_identical_naive_required_between_arms` | PASS |
| Accumulated state | `test_accumulated_state_survives_restart_and_duplicates` | PASS |
| Missing week means no annual score | `test_missing_week_is_incomplete_not_annual` | PASS |
| 70-row inner never compared with h12 annual | `test_inner_fold_score_never_compared_with_annual` | PASS |
| 2019 extension decision | `tests/test_i6n_weekly_history_extension.py` (4) | PASS |

PRE: collection ImportError (modules absent). POST: **73 passed in 33.93 s**,
including the existing `test_i6n_weekly_data.py`, under
`crispdm-run -m 2G -t 120s`, CPU-only.

## Risks

- The calendar is nominal (no holidays or early closes). Holiday hours are
  excluded because their bars are missing, not because the calendar knows them.
- Inherited limits remain: 2024-informed feature selection, current-vintage
  FRED, declared publication lags and the Dec-2019 winter-clock assumption.
- In the last week, 57 origins after the final 2024 bar are typed
  INPUT_UNAVAILABLE ahead of TEST_CLOSED. Rows from 2025 are never loaded.
- The 2023Q4 fold and W2023-12-31 share the same cutoff and TRAIN/inner data.
- A.1 does not score VALIDATION, so the annual stage needs the historical-cell
  run path or an A extension.

## Next artifact

An owner or parent decision on the 2019-10-28 clock case: either admit one
transport successor for that case, or accept a declared assumption. After
that, issue a certificate, materialize the extension and replan as `plan-v3`.
B then runs A's `freeze` on the READY cell configs. Lane D review is pending
on request.
