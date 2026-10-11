# Lane C return: complete weekly units, 2023 folds, support and data

Branch `satoshi/i6n-swarm-C-weekly-20261011`, base `5c383d17`. Current plan
`plan-v3`, `plan_sha256 be5750da0177ecd3a47d73b2682a22964e45d1a5c42b6209903022215bc9169f`
(PLAN.json file `8f2aef32...`), VERIFIED. It supersedes `plan-v2`
(`dd34986a...`); the support of all 54 units that were READY there is
unchanged. No fit, no price read for the plan, and no TEST row. There was
one authorized single-case download (below). No service was started.

## Update C.2: clock ruling outcome

The authorized single attempt for planned case 2019-10-28 used the existing
governed resume tool. That tool gained a minimal admission rule, with tests:
an exact `HTTP 429` reason counts as a transport failure and may be retried
only as an explicitly named single `--only-date`. The attempt returned
**HTTP 429 again (INCONCLUSIVE, report `7e1e34cb...`)**. I made no further
retry. Per the ruling, the span 2019-03-11..2019-12-01 is now a **declared
assumption** `DCA_2019_EU_DST_POLICY_2019-03-11_2019-12-01` (`f851cea1...`),
declared the same way as December 2019. The record names the limitation and
cites the four inconclusive attempt reports, the two in-span passing cases
(2019-03-18 and 2019-07-15) and their exact replay, and the five passing
October holdouts of 2020-2024. A new extended successor was materialized with
35,398 rows (4,544 new). Every row reconciles with its annual archive, and the
old successor is a byte-identical suffix. Its pins are input `bf8587b1...`,
CLOCK `43103443...` and scope `2d81f654...`; the clock_policy text carries the
assumption.

In `plan-v3`, **2023Q1_DCA, 2023Q2_DCA and 2023Q3_DCA are
READY_WITH_DECLARED_CLOCK_ASSUMPTION** on that source. The `_DCA` suffix is in
the fold id and therefore in every receipt row. The assumption also enters
`unit_sha256` and, through the cell's source clock, A's candidate identity.
**2023Q4 stays READY** on the certified source, and so do the 53 weekly units.
History was not shortened. On real timestamps the counts of the three DCA folds
equal `build_weekly_partitions` for both roles. All 681 cell configs pass
`authenticated` and carry the plan stopping rule, seed 6102026 and the endpoint
base model (A.2). Weekly units are marked `freeze --score-outer`; folds never.

| Fold (MANAGEMENT) | h2 TRAIN/INNER | h12 | h72 |
| --- | --- | --- | --- |
| 2023Q1_DCA | 19114/219 | 17023/161 | 4857/51 |
| 2023Q2_DCA | 18613/224 | 16457/157 | 4716/51 |
| 2023Q3_DCA | 18304/376 | 16096/336 | 4629/96 |
| 2023Q4 | 18320/363 | 16122/323 | 4629/83 |

Q1/Q2 inner support is low because 2023 February-July has many missing bars.

The C.1 narrative below is kept as history. Its WAIT_DEPENDENCY statements for
Q1-Q3 are superseded by this section.

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
| 2019 extension decision, exact successor suffix, declared assumption | `tests/test_i6n_weekly_history_extension.py` (6) | PASS |
| Declared assumption carried in fold identity; weekly support unchanged | `test_declared_clock_assumption_source_is_carried_in_fold_identity` | PASS |
| Cell configs carry plan stopping/seed/endpoint model | `test_file_cli_plan_verify_export_redacts_private_paths` | PASS |
| Single named 429 case only | 3 new tests in `test_i6n_external_clock_audit_v2_resume.py` | PASS |

PRE: collection ImportError (modules absent). For C.2, the 3 new tests failed
before implementation. POST C.2, under `crispdm-run -m 2G -t 120s`, CPU-only:
`test_i6n_weekly_campaign.py` **18 passed in 43.19 s**. The rollup, extension,
weekly_data and all external_clock_audit suites: **122 passed in 1.90 s**.

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

B runs A's `freeze` on the `plan-v3` cell configs: four folds per h, and the
weekly units with `--score-outer`. The lane D review is in `REVIEW_C_OF_D.md`.
