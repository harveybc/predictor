# I6-N C1 candidate return

Order: SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md (sha256 243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025). Base 269c2d13.
Status: CANDIDATE IMPLEMENTATION. Not deployed, not integrated into any executor, not a scientific measurement. No DOIN, fits, GPU, TEST, brokers or services.

Commits: initial 30804654; fix round 5e6fc9a7.

## Contract -> test -> result (initial round)

| Clause | Source | Test | Result |
|---|---|---|---|
| Horizon is a positive plain int in elapsed hours. Bool, 0, negative, float, str and None are rejected (INVALID_HORIZON) | order s4; task | test_horizon_must_be_positive_plain_int (8 parametrized cases) | PASS |
| Naive datetimes are rejected for origin, week bounds and split bounds (NAIVE_DATETIME). Comparison is by UTC instant | order s4 | test_naive_datetimes_are_rejected_everywhere; test_dst_change_uses_elapsed_hours_and_utc_instants (same instant written with a +02:00 offset gives identical decisions) | PASS |
| Role is typed ENTRY/MANAGEMENT. Duplicate heads are rejected | order s4 | test_role_must_be_typed; test_duplicate_heads_are_rejected | PASS |
| A=O+delay and B=C-flat in elapsed hours. A declared A/B that disagrees is CALENDAR_OFFSET_MISMATCH. O<C. Bool offsets are rejected | market-week contract s1 | test_week_offsets_and_order_are_validated | PASS |
| WMC02: h72 is valid with R=73 and invalid with R=72 (target == B) and R=48 | market-week s3, WMC02; order s4 | test_h72_against_remaining_hours[73/72/48] | PASS |
| An ineligible long head never drops the valid short heads. Endpoint is not moved. eligible_until=B-h | market-week s3; order s4 | test_h72_ineligible_head_never_drops_valid_short_heads (R=48: 2..12,18,24,36 eligible; all 12 heads returned) | PASS |
| Origin == A is allowed and A-1h is not. Origin == B is rejected (half-open [A,B)) | order s4 | test_origin_equal_to_A_is_allowed_and_equal_to_B_is_not | PASS |
| The Thursday cutoff closes ENTRY (t>=K) but not MANAGEMENT | market-week s1-2 | test_thursday_cutoff_closes_entry_but_not_management | PASS |
| A DST change uses elapsed hours. A/B are derived as elapsed offsets, and an offset change inside the week gives B-A=97h, not 96h | order s4; WMC01 | test_dst_change_uses_elapsed_hours_and_utc_instants (synthetic fixed-offset fixture; h20 eligible at R=21h although the wall clock shows 20h; h21 rejected) | PASS |
| A next-week target is rejected even when it lands inside the next week's [A,B) | market-week s2 | test_next_week_target_is_rejected_even_inside_next_operational_window | PASS |
| Removing a future bar changes M (TARGET_NOT_OBSERVED) and never E | WMC02; order s4 | test_removing_a_future_bar_changes_M_and_never_E | PASS |
| Past gaps do not move A/B, E counts or the eligible-origins digest | order s4; WMC01 | test_past_gaps_do_not_move_A_or_B (function level and table level) | PASS |
| A target on the exclusive split end is rejected; the split start is inclusive; membership is decided by the target instant | order s4; market-week s2 | test_target_on_exclusive_split_end_is_rejected_and_start_is_inclusive | PASS |
| Label cutoff (available_at <= label_available_by) and input availability affect M only. input_available must be a bool | market-week s2 | test_label_cutoff_and_input_availability_are_label_conditions_only | PASS |
| The label arguments (input_available, observed_targets, split) must be passed all together or not at all | design choice (no silent partial M) | test_label_arguments_are_all_or_nothing | PASS |
| WMC04: known inputs from non-trading hours (weekend, [O,A), prior force-flat band, t itself) remain admissible; a future input is rejected | market-week s2; WMC04 | test_known_non_trading_inputs_remain_admissible_and_future_ones_do_not | PASS |
| A missing calendar, id or version is a CALENDAR_REQUIRED typed error. There is no NY fallback and no inference from prices | order s4 | test_missing_calendar_is_a_typed_error_with_no_fallback; test_cli_missing_calendar_fails_typed_without_result | PASS |
| Overlapping calendar weeks and a mismatched declared B are rejected | market-week s1 | test_calendar_rejects_overlapping_weeks_and_mismatched_declared_bounds | PASS |
| The support table has one row per week x role x head. WMC03: 96/120h give NO_ELIGIBLE_ROWS, not a score | WMC03; order s4 | test_support_table_enumerates_every_week_role_head (MGMT h2=94, h72=24; ENTRY h2=62, h72=24) | PASS |
| Support comes from enumerated timestamps, not the closed form (B-A)-h. Origins outside every calendar week are counted, not dropped | order s3-4 | test_support_depends_on_enumerated_timestamps_not_a_closed_form (even-hour grid gives 12, not 24) | PASS |
| M counts are reported per split, with typed exclusions among E-eligible origins | order s4 | test_table_label_counts_by_split_and_input_availability (TRAIN 34, VALIDATION 36) | PASS |
| Splits must not overlap | design | test_splits_must_not_overlap | PASS |
| The synthetic example config runs | task | test_example_config_is_synthetic_and_runs | PASS |
| CLI writes an atomic STATUS.json with the required fields. An identical input with an intact terminal output is reused. A different input is refused (OUTPUT_HOLDS_DIFFERENT_INPUT) without overwriting | order s6 | test_cli_runs_reuses_and_refuses_a_different_input | PASS |
| --status is read-only, returns once and never creates or starts anything | order s6 | test_cli_status_is_read_only_and_returns_once | PASS |

## PRE / POST

PRE: Tests were written first and run before any implementation existed: `CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_eligibility.py`. Result: collection ERROR, `ModuleNotFoundError: No module named 'tools.i6n_contracts'`, "1 error in 0.22s".

POST (initial): Same command after implementation: `36 passed in 0.16s` (green on the first post-implementation run; it was not rerun afterwards). `git diff --cached --check` is clean. A one-off import check shows that importing the CLI loads no pandas, tensorflow, numpy, keras or DB drivers. It does load the stdlib `sqlite3` module through the reused `tools.i6m_queue`, but no database is opened.

## Independent review 1: FIX_REQUIRED

- Minor: per-head records carry no explicit `available` field, although market-week contract s3 lists one per prediction alongside horizon_hours, target_timestamp, eligible_until and reason. C1 emits operational_eligible and label_eligible plus reasons. Prediction availability (inputs/checkpoint/execution) arguably belongs to a later adapter, but the gap is not stated in the limits.
- Minor: input_unavailable_origins entries that match no enumerated origin are accepted silently. input_unavailable_origins_count reports how many were declared, not how many were applied. A config whose only unavailable origin is 2030-01-01 completes with count=1 and has no effect.
- Minor: a degenerate week with A >= B (for example offsets summing to O..C) is not rejected; only O < C is checked. Every row then becomes NO_ELIGIBLE_ROWS, which reads like horizon incompatibility rather than a calendar/offset error.
- Documented limits, not defects: the Thursday cutoff K must be supplied per week as a resolved instant (no tz-database resolution of THURSDAY 00:00 MARKET_CALENDAR); the table counts only the primary exclusion reason; split membership is by target instant with [start, end); input availability sits in M, not E (consistent with 'E uses only the calendar').
- Minor: the CLI pulls in stdlib sqlite3 through the reused tools.i6m_queue. No database is opened (no connect/Queue usage), but the order asked to avoid database dependency for these pure components.
- Mandatory 'never drop valid heads' clause on the M (label) path: test_h72_ineligible_head_never_drops_valid_short_heads calls evaluate_heads without label evidence, so only the E-only branch is covered. Mutant: filter `if d.operational_eligible` on the label branch of evaluate_heads (eligibility.py lines 539-541). Result: SURVIVED, 36 passed. Add a multi-head evaluate_heads call with input_available/observed_targets/split and assert all heads are returned.
- Mandatory DST clause: test_dst_change_uses_elapsed_hours_and_utc_instants uses fixed-offset tzinfo (-04:00/-05:00). With fixed offsets, wall-clock and elapsed arithmetic coincide, so the test cannot fail on a wall-clock implementation. Mutant: remove `.astimezone(UTC)` in utc_instant (line 167). Result: SURVIVED, 36 passed. On a real America/New_York week spanning 2026-11-01 the correct code gives B-A=145h elapsed and h=R-1 eligible; the mutant gives 144h and is off by one hour, so it rejects a valid head. Use zoneinfo (or an equivalent tz with a transition) in the fixture so the test discriminates.
- CLI TERMINAL_OUTPUT_DAMAGED path is untested. Mutant: always treat a COMPLETED result as intact (eligibility_cli.py line 177). Result: SURVIVED. A tampered SUPPORT_TABLE.json would be REUSED without any test noticing.
- CLI OUTPUT_HOLDS_DIFFERENT_CODE / code-digest identity is untested. Mutant: constant code_sha256. Result: SURVIVED.
- Atomic STATUS.json write is not exercised. Mutant: plain write_text instead of atomic_json. Result: SURVIVED. Acceptable because it reuses the existing i6m_queue.atomic_json helper (unique tmp + fsync + os.replace), but no test pins it.

## Fix round

- FIXED: (1) Each head's record now has an explicit `available` field driven by an input. The caller supplies prediction_available as None, one bool, or a {head: bool} mapping that names exactly the requested heads. available = E AND prediction_available, with typed AvailabilityReason (OPERATIONALLY_INELIGIBLE, PREDICTION_UNAVAILABLE). If it is not supplied, available is None and availability_evaluated is False; there is never a default True/False. E and M are unchanged. Malformed input raises PREDICTION_AVAILABILITY_INVALID. The limits state that deciding real availability (inputs, checkpoint, execution) belongs to a later adapter.
- FIXED: (2) input_unavailable_origins entries are now rejected with INPUT_AVAILABILITY_INVALID when they are duplicates, match no enumerated in-calendar origin (e.g. 2030-01-01, or a weekend origin outside every week), or are given without label_evidence (they would have no effect). The table field was renamed input_unavailable_origins_applied.
- FIXED: (3) A week with A >= B now raises the new typed error OPERATIONAL_WINDOW_EMPTY in OperationalWeek.__post_init__, so it is caught both by derive/direct construction and by the config path. A cutoff K <= A is still accepted and is documented as a week without new entries.
- FIXED: (4) The CLI no longer imports tools.i6m_queue, which pulls in sqlite3. It now uses a local atomic_json (unique temp file in the same directory, fsync, os.replace, temp cleaned up on failure) and eligibility.canonical_digest (same formula as i6m_queue.digest). i6m_queue.py was removed from the code-digest file set. A subprocess probe confirms that sqlite3, _sqlite3, pandas, tensorflow and numpy are not imported.
- FIXED: (5) Added test_label_path_never_drops_valid_heads: on the M path all 12 heads are kept in order, short heads stay label-eligible, and 48/60/72 report OPERATIONALLY_INELIGIBLE.
- FIXED: (6) Added test_dst_end_with_real_zoneinfo_uses_elapsed_hours. It uses a real ZoneInfo('America/New_York') for a synthetic week from 2026-10-31 12:00 to 2026-11-05 12:00 local, which spans the DST end on 2026-11-01 (a timezone rule only, not a broker calendar). From A, h=96 is eligible because 97 elapsed hours remain, while the wall clock shows 96; remaining_seconds is 97*3600, the target is 2026-11-05T04:00Z, and h=97 is rejected.
- FIXED: (7) Added test_cli_refuses_a_damaged_completed_result: after a byte is appended to SUPPORT_TABLE.json, the CLI exits 2 with TERMINAL_OUTPUT_DAMAGED, the damaged result is not rewritten, STATUS.json bytes are unchanged, and --status reports result_intact False.
- FIXED: (8) Added test_code_digest_tracks_code_bytes_and_refuses_different_code: the code files are copied and _CODE_FILES is monkeypatched to point at the copies. Changing one byte changes code_sha256, and a rerun on the same config is refused with OUTPUT_HOLDS_DIFFERENT_CODE while STATUS is left unchanged.
- FIXED: (9) Added test_status_and_result_are_replaced_atomically: a spy on os.replace checks that the staged file is in the same directory and is complete JSON before the swap, that STATUS.json is swapped at least 3 times and SUPPORT_TABLE.json once, and that no temp files are left over.

Mutation checks: The tests were written first. On the old code 30804654, the suite with the new tests gave 5 failed, 41 passed. The failures were items 1-4: no AvailabilityReason, the 2030 origin was accepted, A>=B was not rejected (2 parametrized cases), and the import probe printed ['_sqlite3','sqlite3']. The tests for items 5-9 passed on the old code, as expected, since they only pin existing behaviour.

POST in the worktree: 46 passed (CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_eligibility.py), and git diff --check is clean.

Mutants ran one at a time in a scratch copy at <session-scratch> (generator mutate.py; nothing was committed). Base: 46 passed. All 9 mutants were killed:
- M5 (label branch filters to operationally eligible heads): killed by test_label_path_never_drops_valid_heads and the availability test.
- M6 (utc_instant returns the value without .astimezone(UTC)): killed by test_dst_end_with_real_zoneinfo_uses_elapsed_hours.
- M7 (COMPLETED always treated as intact): killed by test_cli_refuses_a_damaged_completed_result.
- M8 (constant code_sha256): killed by test_code_digest_tracks_code_bytes_and_refuses_different_code.
- M9 (STATUS written with plain write_text): killed by test_status_and_result_are_replaced_atomically.
- M1 (available defaults to E when prediction_available is not supplied): killed by the availability test.
- M2 (unmatched input_unavailable origin accepted): killed by test_input_unavailable_origins_must_match_an_enumerated_origin.
- M3 (A<B check removed): killed by both test_week_with_empty_operational_window_is_a_typed_error cases.
- M4 (CLI re-imports tools.i6m_queue.digest): killed by test_cli_imports_no_database_dataframe_or_tensor_library.

POST (fix round): I fixed all nine review findings and declined none. The fixes are in commit 5e6fc9a7, pushed on satoshi/i6n-c1-eligibility-20261011. This is a candidate implementation only: it is not deployed and its outputs are not measurements. No DOIN, fits, GPU, TEST, brokers, services, network, pandas, TensorFlow or database were used.

**Tests**
- On the old code (30804654), the suite with the new tests gave 5 failed, 41 passed. The 5 failures were findings 1–4. The tests for 5–9 passed there, because they pin behaviour that was already correct.
- After the fixes: 46 passed. `git diff --check` is clean.
- All 9 mutants, run in a scratch copy, are killed: the five the reviewer named (5–9) plus four for my own fixes (1–4).

**What changed in the behaviour**
- **`available` per head (1):** you now pass `prediction_available`, either one bool for all heads or a `{head: bool}` mapping that names exactly the heads. `available` is true only when the head is operationally eligible (E) and you said the prediction is available, and each false comes with a typed reason. If you don't pass it, `available` stays `None`; there is no default. It never changes E or M. Deciding real availability (inputs, checkpoint, execution) is left to a later adapter, and the limits say so.
- **Unavailable-origin list (2):** entries that are duplicates, that match no origin inside a calendar week, or that are given without label evidence (where they would do nothing) are now rejected with `INPUT_AVAILABILITY_INVALID`. The table field is now `input_unavailable_origins_applied`.
- **Empty operational window (3):** a week where A >= B now raises the new typed error `OPERATIONAL_WINDOW_EMPTY` instead of producing `NO_ELIGIBLE_ROWS` everywhere. A week whose entry cutoff falls at or before A is still accepted, as a week with no new entries; this is documented.
- **No database import (4):** the CLI no longer imports `tools.i6m_queue`, which pulls in `sqlite3`. It uses a local digest and a local atomic file writer with the same formulas as that helper. A test confirms that `sqlite3`, pandas, TensorFlow and numpy are not loaded.
- **Tests only (5–9):**
  - Valid heads are not dropped on the label path.
  - DST is tested with a real America/New_York week crossing 2026-11-01. This uses only the timezone rule, not a broker calendar.
  - A damaged completed result is refused.
  - A code change is detected and refused.
  - STATUS.json and the result are written atomically.

The module docstring now has a short limits section covering these choices. I scanned the diff for host names, IPs, usernames and absolute home paths and found none.

To reproduce, from the worktree root:
```
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_eligibility.py
git diff --check
```

## Independent re-review: ACCEPT_AS_CANDIDATE

- Remaining (minor): Minor, test gap only: the fix report says atomic_json in tools/i6n_contracts/eligibility_cli.py removes its temp file when a write fails. No test covers that failure path. test_status_and_result_are_replaced_atomically only checks that no temp files are left after a successful run. Not a blocker: the order requires an atomic STATUS.json, and that is pinned.
- Surviving mutant: N10_atomic_no_cleanup: replacing the `finally: if tmp.exists(): tmp.unlink()` in eligibility_cli.atomic_json with `pass` -> 46 passed (cleanup on failure is not pinned; minor)

## Limits

- Candidate only: not wired into any executor, dataset builder, strategy adapter or OLAP. No training and no GPU. Keras masking was not validated (the table records keras_masking_validated=false).
- No real calendar was used. All fixtures are synthetic, and the DST case uses fixed offsets (-04:00 to -05:00) inside one week. It does not claim that this matches any real broker or NY rule.
- The entry cutoff K must be supplied per week as an already-resolved instant. The "THURSDAY 00:00 MARKET_CALENDAR" rule is not resolved here, because that would need a tz database and a calendar choice.
- Decisions I made where the contract leaves room, documented in the docstrings:
  - Split membership is decided by the target instant q, with [start, end) bounds.
  - label_available_by is mandatory and may be null. A label is accepted when available_at <= label_available_by.
  - The cutoff must lie within [O, C].
  - The table enumerates origins over [O, C). UTC_WHOLE_HOURS or an EXPLICIT list is a declared convention, and both are recorded in the output.
  - Reasons are listed in a fixed precedence order; the table counts only the primary reason.
- Not covered: purge beyond target membership and the label cutoff; S_h/U_h populations (that is C2); calendars for 24/7 assets.
- Reuse:
  - `_require_plain_int` is imported from the sealed `tools/business_weekly_protocol.py`, which was not modified.
  - The CLI imports `digest` and `atomic_json` from `tools/i6m_queue.py`, also unmodified.
  - The pure module mirrors the `i6m_queue.digest` formula locally so that it does not import sqlite3.
- The single writer per output directory is a contract, not enforced (no leases, as ordered). A same-identity RUNNING or FAILED record is recomputed once from scratch.
- Code digest covers eligibility.py, eligibility_cli.py, business_weekly_protocol.py and i6m_queue.py. If any of them changes, the existing output is refused as OUTPUT_HOLDS_DIFFERENT_CODE.
- The design documents are still uncommitted in predictor-i6d-weekly. CONTRACT_SOURCES records their digests as provenance only.
- Independence: Satoshi's line implemented this, so it cannot be independently verified by the same author.

## Reproduce

git -C <path> fetch origin satoshi/i6n-c1-eligibility-20261011
git -C <path> worktree add <dir> 3080465416ff92fe719fb8256a7131d926239422
cd <dir>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_eligibility.py
git diff --check 269c2d132b21a55e59cd6faa8242b1ac74f0f7ae HEAD
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m tools.i6n_contracts.eligibility_cli --config tools/i6n_contracts/examples/eligibility_synthetic.json --output <scratch>/out
python -m tools.i6n_contracts.eligibility_cli --status --output <scratch>/out
# rerun of the same --config prints REUSED; another config on the same --output exits 2 with OUTPUT_HOLDS_DIFFERENT_INPUT
