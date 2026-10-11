# I6-N C2 candidate return

Order: SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md (sha256 243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025). Base 269c2d13.
Status: CANDIDATE IMPLEMENTATION. Not deployed, not integrated into any executor, not a scientific measurement. No DOIN, fits, GPU, TEST, brokers or services.

Commits: initial a2d09fe6; fix round f338969f.

## Contract -> test -> result (initial round)

| Clause | Source | Test | Result |
|---|---|---|---|
| The naive is computed per h on that h's own rows (no naive shared between horizons); the naive differs at h2 and h72 | order s5; market-week contract s5 | test_naive_is_computed_per_horizon_and_differs_between_h2_and_h72 | PASS (h2 naive_mae 0.375 / mae 0.1875; h72 naive_mae 1.0 / mae 0.5; values computed by hand) |
| When the model equals the naive, skill = 0 | order s5 | test_model_equal_to_naive_gives_zero_skill | PASS |
| skill = (MAE_h - naive_h) / naive_h; a smaller error gives a negative skill | order s5; market-week contract s5 | test_smaller_error_gives_negative_skill_and_larger_error_positive_skill | PASS (-0.5 MAE, -0.75 MSE; worse model gives +1.0) |
| A zero naive leaves the skill undefined with a typed reason; JSON never carries NaN or Infinity | order s5 | test_zero_naive_leaves_skill_undefined_with_reason_and_json_has_no_nan_or_infinity | PASS (NAIVE_MAE_ZERO / NAIVE_MSE_ZERO) |
| Unit or scale differences are rejected | order s5 | test_scale_or_unit_difference_is_rejected[unit, scale] | PASS (SCALE_MISMATCH) |
| target_timestamp must equal origin + h, checked before computing | order s5 | test_target_timestamp_not_equal_origin_plus_h_is_rejected; test_cli_validation_failure_writes_typed_failed_status_and_no_result | PASS (TARGET_TIMESTAMP_MISMATCH; no RESULT written) |
| Timestamps without a zone are rejected; comparison is by UTC instant | order s4/s5 | test_timestamp_without_zone_is_rejected | PASS (TIMESTAMP_NOT_AWARE) |
| Duplicates are rejected, including the same instant written with another offset | order s5 | test_duplicate_forecast_is_rejected_even_when_written_in_another_offset; test_duplicate_population_key_is_rejected; test_duplicate_receipt_is_rejected | PASS (DUPLICATE_RECORD / DUPLICATE_POPULATION_KEY / DUPLICATE_RECEIPT) |
| Conflicting values for one forecast are rejected | order s5 | test_conflicting_values_for_one_forecast_across_roles_are_rejected | PASS (FORECAST_CONFLICT) |
| Finiteness and identities are validated before computing (NaN, inf and booleans rejected; h must be a positive int; identities non-empty) | order s5 | test_nonfinite_values_and_bad_identities_are_rejected (6 cases) | PASS |
| One missing forecast gives INCOMPLETE plus coverage, never a favourable row selection; an incomplete run cannot be approved | order s5; market-week contract s5 | test_one_missing_forecast_yields_incomplete_and_coverage_not_favourable_selection; test_declared_forecast_failure_is_counted_with_its_reason; test_cli_incomplete_result_has_its_own_exit_code | PASS (coverage 3/4, MISSING_FORECAST 1; the comparison returns NOT_COMPARABLE_INCOMPLETE with no deltas; exit code 4) |
| Configs are compared only on an identical, previously sealed common support, never an opportunistic intersection | order s5 | test_comparison_requires_the_identical_previously_sealed_support; test_tampered_sealed_population_is_rejected; test_complete_comparison_on_sealed_support_reports_paired_deltas | PASS (SUPPORT_MISMATCH / POPULATION_SEAL_INVALID; complete case delta_mae -0.25 on identical naive rows) |
| Aggregating sums and counts equals full scoring, not the unweighted mean of weeks of different size | order s5; I6N s7 | test_aggregating_sums_and_counts_equals_full_scoring_not_unweighted_week_mean | PASS (pooled 1.375 equals full; differs from the 2.25 mean of week means) |
| The same forecast is never counted twice (role aggregation; re-aggregating the same rows) | order s5 | test_aggregating_roles_counts_a_shared_forecast_once; test_aggregation_refuses_to_count_a_row_twice | PASS (ALL_ROLES n=2 from 3 role records; DOUBLE_COUNT) |
| S_h and U_h are distinct populations, each exactly paired (model rows digest == naive rows digest) | order s5; market-week contract s5 | test_usage_and_opportunity_populations_differ_but_each_is_exactly_paired | PASS (S n=3, U n=2 including a NO_ORDER receipt; separate row digests) |
| U_h comes only from receipts recorded before the outcome; never from predictions or trades | order s5 | test_usage_is_built_only_from_receipts_never_from_predictions; test_receipt_recorded_at_or_after_the_outcome_is_rejected; test_receipt_for_a_different_prediction_is_rejected; test_receipt_without_a_scored_forecast_is_incomplete_not_dropped; test_unobservable_label_is_excluded_from_usage_and_refused_in_opportunity | PASS (NO_RECEIPTS; RECEIPT_AFTER_OUTCOME / RECEIPT_BEFORE_ORIGIN / RECEIPT_CONFLICT; U INCOMPLETE 1/2; LABEL_NOT_OBSERVABLE excluded in U, DECLARED_LABEL_ABSENT refused in S) |
| An empty population gives no score but keeps its exclusion and failure counts | order s5 | test_empty_population_has_no_score_but_keeps_exclusion_counts | PASS (NO_ELIGIBLE_ROWS, EMPTY_POPULATION, exclusions preserved) |
| Output is JSON records ready for a future OLAP load, with no warehouse write; results are sealed and carry the design digests | order s5 | test_olap_records_are_flat_scalars_without_nan; test_canonical_digest_is_byte_identical_to_the_i6m_record_digest | PASS |
| CLI: atomic STATUS.json with the required fields; identical input reused; a different input never overwrites; --status read-only and returns once | order s6 | test_cli_run_writes_result_and_complete_status; test_cli_identical_input_reuses_intact_terminal_output; test_cli_different_input_never_overwrites_previous_output; test_cli_status_is_read_only_and_returns_once | PASS (bytes and mtimes unchanged; OUTPUT_IDENTITY_MISMATCH; STATUS_ABSENT creates nothing; no absolute path in STATUS) |
| API documented with docstrings and a synthetic example JSON | order s6 | test_synthetic_example_json_is_valid_and_complete | PASS |

## PRE / POST

PRE: Tests were written before any implementation. The run was CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_paired_metrics.py and exited with code 2: "ModuleNotFoundError: No module named 'tools.i6n_contracts' ... Interrupted: 1 error during collection ... 1 error in 0.19s".

POST (initial): The same command returned "42 passed in 0.25s", exit 0. I made one later code change (embedding the design digests and adding a seal-tamper assertion). Because the code changed, I reran the suite once: "42 passed in 0.33s", exit 0. git diff --cached --check exited 0. An import check found that loading the CLI does not load pandas, tensorflow, sqlite3, requests or urllib.request. numpy is loaded only through the reused tools.i6e_compact_run helper.

## Independent review 1: FIX_REQUIRED

- Inconsistent price(t) is accepted. Two forecasts with the same (asset, origin) at different horizons can carry different price_origin values: probe h2 p0=1.0 and h72 p0=1.75 scored COMPLETE. The same target instant q can also carry different price_target values across rows (probe P1b, COMPLETE). The market-week contract s5 (naive from the price observed at t) and order s5 (validate identities before computing) require one observed price per (asset, instant). A per-horizon naive can therefore be fed an arbitrary price(t). There is no PRICE_ORIGIN_CONFLICT or PRICE_TARGET_CONFLICT check (paired_metrics.py _forecasts, around line 289, keys conflicts on (asset, origin, h) only).
- aggregate_groups (paired_metrics.py lines 647-697) groups only by (population_kind, h, role, unit, scale). Groups carry no config_id, protocol_id or calendar_id, so week groups from different configs, protocols or calendars pool silently (probe P2: config-a plus config-b with another protocol and calendar gave n=2). The groups are also unsealed and n and sums are trusted unchecked: a tampered group with n=100 and sum_abs_error=0 was accepted (probe P2b). Order s5 aggregation must stay within one config and protocol.
- The ALL_ROLES groups drop the upstream sealed exclusion counts (exclusions={} at line 578; probe P11 returned {} while ENTRY carried CUTOFF=3). The implementer declares this. Contract s5 asks for exclusion counts per h; the role-deduplicated group publishes none and does not explicitly mark them as reported per role.
- Declared and accepted as candidate limits, not defects: E-eligibility of receipts needs C1; the real timing of sealed_at needs external custody (commit or OLAP readback); unit and scale are checked by label only.
- An untyped exception leaves STATUS.json at RUNNING forever, and --status then returns exit 0, the code documented as 'complete'. Repro (probe P6): price_origin=10**400 raises OverflowError ('int too large to convert to float') inside _finite via math.isfinite. Other untyped OverflowErrors: horizon 10**12 in timedelta (P3); origin 9999-12-31T23Z + h (P5). The CLI catches only PairedMetricsError (paired_metrics_cli.py line 194). status() maps every state other than FAILED or COMPLETED+INCOMPLETE to EXIT_COMPLETE (line 221). Order s6 requires a typed error in STATUS, and automation reading exit codes would treat a crashed run as complete. Fix: catch OverflowError, TypeError and similar into a typed failure; give RUNNING and unknown states their own non-zero exit code.
- PAIRING_BROKEN (paired_metrics.py lines 441-443) is unreachable. model_keys and naive_keys are built from the same `rows` list, so model_row_keys_sha256 == naive_row_keys_sha256 by construction. The 'exact model/naive row equality' digest therefore carries no independent evidence. Only naive_rows_sha256 and model_rows_sha256 (which include values) are informative. It is not harmful, but the reported pairing proof is tautological.
- Minor: run() reads the config with Path.read_bytes() before writing any status. A missing config path raises an uncaught FileNotFoundError traceback instead of a typed refusal. --status on a STATUS.json that is valid JSON but not an object raises AttributeError (value.get).
- Mutation removing the NAIVE_MISMATCH guard in compare_configs (line 739) SURVIVED: 42 passed. No test compares two configs on the same sealed support whose naive inputs or labels differ. A probe shows the guard works (P10 NAIVE_MISMATCH), but it is unprotected.
- Mutation removing CONTRACT_MISMATCH in compare_configs SURVIVED (configs with a different protocol, calendar, unit or horizons could be compared).
- Mutation removing AGGREGATE_MISMATCH SURVIVED (aggregating groups from different h, role or kind is untested).
- Mutation removing ROW_DIGEST_INVALID in aggregate_groups SURVIVED.
- Mutation removing RECEIPT_PROTOCOL_MISMATCH SURVIVED (receipts under another calendar or protocol are untested).
- Mutations removing HORIZON_UNDECLARED and ROLE_UNDECLARED SURVIVED.
- Mutations removing DUPLICATE_FAILURE and the failure-and-forecast FORECAST_CONFLICT in _failures SURVIVED.
- Mutation making coverage_denominator ignore exclusions (denominator = n_declared) SURVIVED. No test asserts coverage when U has LABEL_NOT_OBSERVABLE exclusions.
- Mutation removing pm.verify_result on a reused RESULT.json SURVIVED. Reuse of a corrupt or tampered terminal result (OUTPUT_CORRUPT) is untested.
- The two CLI identity guards (STATUS identity, RESULT identity) each survive removal on their own because the other catches the case. Together they hold, but no test isolates a case with STATUS only or RESULT only.
- No test covers the --status exit code for RUNNING or crashed state, or untyped exceptions in the CLI.
- Mandatory order s5 tests all KILL their mutants: per-h naive h2/h72; model=naive skill 0; negative skill; scale rejection (no_scale_check killed); timestamp mismatch (no_target_check killed); RECEIPT_AFTER_OUTCOME at equality (>= to > killed); duplicate; missing forecast as INCOMPLETE (missing_as_excluded killed); pooled sums vs mean of week means (killed); DOUBLE_COUNT (killed); ALL_ROLES dedupe (killed); zero naive returning inf (killed); compare on incomplete (killed); population seal (killed); receipt digest (killed); result state (killed).

## Fix round

- FIXED: (1) Inconsistent prices are now refused. A new _observe step builds one price series per (asset, instant) from every record's price_origin at its origin and price_target at its target. If two rows give different prices for the same instant, or one row says the price was observed and another says it was not, validation raises PRICE_CONFLICT before anything is computed. The naive error is read from this validated series. Tests: same origin at h2 vs h72 (p0 1.0 vs 1.75), same target instant across rows, a target price that clashes with an origin price at the same instant, an unobservable label clashing with an observed price, and a positive case where shared prices agree. To keep existing fixtures consistent, rec() now spaces origins 100 hours apart; no assertion was relaxed.
- FIXED: (2) Every group now carries config_id, protocol_id, calendar_id and contract_sha256, plus its own sha256 seal. aggregate_groups checks, in order: required fields (SCHEMA_INVALID), the group seal (GROUP_SEAL_INVALID), the row_keys digest (ROW_DIGEST_INVALID), and that n, sums and means agree (GROUP_INCONSISTENT: n must equal len(row_keys) and fit in declared minus excluded, and mae/mse/naive/skill must follow from the sums and n). It refuses groups whose identity differs (AGGREGATE_MISMATCH over kind, h, role, unit, scale, config, protocol, calendar and contract digest) and returns a sealed aggregate that lists its constituent group digests. The reviewer's tamper (n=100, sum_abs_error=0) is caught by GROUP_SEAL_INVALID; a resealed version is caught by GROUP_INCONSISTENT. The remaining limit is written in the docstring: digests are unkeyed, so someone who edits a group and reseals it with false but self-consistent sums cannot be detected without the rows.
- FIXED: (3) ALL_ROLES groups now keep the sealed exclusion counts. They are reported unpooled in upstream_exclusions_by_role, because the counts carry no row identities and could not be de-duplicated across roles. A new field exclusions_scope says what 'exclusions' holds: C2_ONLY_UPSTREAM_REPORTED_PER_ROLE for ALL_ROLES, UPSTREAM_AND_C2 for per-role groups. The field is also in the OLAP records, and aggregates pool the per-role counts.
- FIXED: (4) Overflow and crashes now end in a typed state. A price of 10**400 gives NONFINITE_VALUE. origin+h overflow (horizon 10**12, or 9999-12-31T23Z + 2h) and a zone conversion out of range (0001-01-01T00:00+01:00) give TIMESTAMP_OUT_OF_RANGE. An error sum that overflows gives NONFINITE_RESULT (via a typed _fsum wrapper). A skill that is not finite becomes None with reason SKILL_*_NOT_FINITE. In the CLI, any other exception becomes STATUS FAILED with code UNEXPECTED_ERROR and the exception type. KeyboardInterrupt or SystemExit is recorded as FAILED/INTERRUPTED and then re-raised. --status now exits 5 (EXIT_RUNNING) for RUNNING and 2 for FAILED, so neither returns 0. Docstring limit: without a lease, a RUNNING status cannot tell a live run from one killed by SIGKILL. Rerunning the same input recomputes from scratch.
- FIXED: (5) Took the 'remove the claim' option. PAIRING_BROKEN, model_row_keys_sha256 and naive_row_keys_sha256 are gone, from both the groups and the OLAP fields. The docstring now says pairing is structural: one sorted row list feeds both sums. row_keys_sha256, model_rows_sha256 and naive_rows_sha256 remain for external readers. The required S_h/U_h test no longer relies on the circular digest equality. It now checks the expected row_keys for each population and recomputes the model and naive MAE independently from the input rows.
- FIXED: (6) Typed errors for both cases. A missing or unreadable config returns CONFIG_UNREADABLE (exit 2) and creates no output directory. --status on a non-object JSON file, on null, or on an unknown state returns STATUS_UNREADABLE (exit 2). A non-object STATUS.json or RESULT.json found in the output directory during a run is OUTPUT_CORRUPT and is left untouched. A RecursionError while parsing becomes CONFIG_INVALID.
- FIXED: (7) Added a test for every named mutant: NAIVE_MISMATCH (same support, different price_origin), CONTRACT_MISMATCH (different calendar, and a resealed result with a group set that differs), AGGREGATE_MISMATCH (config, protocol, calendar and horizon), ROW_DIGEST_INVALID (resealed row_keys), RECEIPT_PROTOCOL_MISMATCH (calendar and protocol), HORIZON_UNDECLARED and ROLE_UNDECLARED at every site (forecast, population key, exclusions), DUPLICATE_FAILURE, failure-and-forecast FORECAST_CONFLICT, a coverage denominator with LABEL_NOT_OBSERVABLE in U (n=1, declared=3, excluded=1, so 1/2), verify_result on a reused RESULT.json (inner result tampered with the wrapper resealed, giving OUTPUT_CORRUPT) and the wrapper digest, the STATUS-only and RESULT-only identity guards tested separately, and the --status exit code for a RUNNING or crashed run (followed by a rerun that completes).
- DECLINED: Partial on item 2, by design and documented: a group that is edited and then resealed with false but self-consistent sums cannot be detected, because the digests are unkeyed SHA-256 and the order prescribes canonical digests, not keys or signatures. Every inconsistent tamper is refused.
- DECLINED: Kept tools.i6e_compact_run.atomic_json and file_hash instead of writing a local atomic writer. Order section 6 says to use the existing helpers, and this one does not import sqlite3 or pandas. It does import numpy, which the CLI docstring now states. The pure module paired_metrics.py imports only the standard library.

Mutation checks: Run with a driver (scratchpad/fix_C2/mutate.py) against a scratch copy at <session-scratch> Each mutant was a single exact text replacement in a fresh copy, followed by pytest -x. The driver ran once under crispdm-run -m 2G -t 120s. Nothing from it was committed.

First pass: 31 of 32 killed. M21 (the GROUP_INCONSISTENT n check) survived because the means check caught the same tamper first. I added a test with a resealed n whose means are self-consistent, and on the rerun all 32 of 32 were killed.

Mutants and their killing tests:
- M01 NAIVE_MISMATCH: test_comparison_refuses_a_different_naive_on_the_same_support
- M02 and M03 CONTRACT_MISMATCH (field loop, group set): test_comparison_refuses_a_different_calendar and test_comparison_refuses_different_group_sets
- M04 AGGREGATE_MISMATCH: test_aggregation_refuses_groups_from_different_config_protocol_or_calendar
- M05 ROW_DIGEST_INVALID: test_aggregation_refuses_row_keys_that_do_not_match_their_digest
- M06 RECEIPT_PROTOCOL_MISMATCH: test_receipt_under_another_calendar_or_protocol_is_rejected
- M07, M08 and M09 (HORIZON_UNDECLARED in identity and in exclusions, ROLE_UNDECLARED): test_undeclared_horizon_or_role_is_rejected_wherever_it_appears
- M10 DUPLICATE_FAILURE: test_duplicate_failure_is_rejected
- M11 FORECAST_CONFLICT on a failed-and-forecast identity: test_identity_both_failed_and_forecast_is_rejected
- M12 coverage denominator ignoring exclusions: test_coverage_denominator_excludes_unobservable_labels_but_keeps_failures
- M13 CLI verify_result on reuse: test_cli_reused_result_with_a_tampered_inner_result_is_corrupt
- M14 wrapper digest: test_cli_reused_result_file_with_a_broken_wrapper_digest_is_corrupt
- M15 STATUS-only identity guard: test_cli_status_only_identity_guard_refuses_a_different_input
- M16 RESULT-only identity guard: test_cli_result_only_identity_guard_refuses_a_different_input
- M17 --status RUNNING exiting 0: test_cli_status_of_a_running_or_crashed_run_is_nonzero_and_rerun_completes
- M18 PRICE_CONFLICT disabled: test_same_origin_price_must_agree_across_horizons
- M19 group seal check disabled: test_aggregation_refuses_a_tampered_unsealed_group
- M20 means-consistency check disabled: test_aggregation_refuses_a_resealed_group_whose_counts_or_sums_disagree
- M21 n check disabled: test_aggregation_refuses_a_resealed_n_that_disagrees_with_its_rows_even_with_consistent_means
- M22 CLI generic exception left untyped: test_cli_unexpected_exception_writes_typed_failed_status
- M23 _finite integer overflow: test_integer_price_too_large_for_a_float_is_a_typed_refusal
- M24 _target overflow: test_horizon_whose_target_overflows_the_calendar_is_a_typed_refusal
- M25 ALL_ROLES dropping exclusions: test_all_roles_group_reports_the_sealed_exclusions_per_role
- M26 missing config left untyped: test_cli_missing_config_is_a_typed_refusal_and_creates_nothing
- M27 non-object output accepted: test_cli_non_object_status_in_the_output_is_refused_as_corrupt
- M28 --status accepting an unknown state: test_cli_status_on_a_non_object_or_unknown_state_is_typed
- M29 interrupt not recorded: test_cli_interrupt_is_recorded_as_failed_and_reraised
- M30 fsum overflow: test_error_sums_that_overflow_are_a_typed_refusal
- M31 skill overflow: test_skill_that_overflows_is_undefined_with_a_reason
- M32 parse_utc overflow: test_origin_at_the_end_of_the_calendar_is_a_typed_refusal

PRE (new tests run against the old code a2d09fe6): 25 failed, 57 passed. One of the 25 failures came from a fixture TypeError in a test helper, which I then fixed. The tests that already passed on the old code (NAIVE_MISMATCH, ROW_DIGEST_INVALID and the like) were mutant-killers for checks that already existed; the driver confirmed that each one kills its mutant.

POST: CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_paired_metrics.py gave 83 passed in 0.46s, and git diff --check is clean.

POST (fix round): Fix round C2 is pushed: f338969f on satoshi/i6n-c2-paired-metrics-20261011 (a new commit on top of a2d09fe6). It is a candidate implementation only: not deployed, not a scientific measurement, and nothing was written to a warehouse or the OLAP schema.

All 7 review findings are addressed, with two partial declines:
- Item 2: someone who edits a group and reseals it with false but self-consistent sums cannot be detected, because the digests are unkeyed SHA-256. This limit is documented in the docstring.
- The CLI keeps the existing atomic writer helper in tools.i6e_compact_run, as order section 6 asks. That helper imports numpy but neither sqlite3 nor pandas.

Suite: 83 passed. Old code before the fix: 25 failed and 57 passed. git diff --check is clean, and all 32 mutants were killed in the scratch copy.

Reproduce from the worktree:
- CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_paired_metrics.py
- python -m tools.i6n_contracts.paired_metrics_cli --config tools/i6n_contracts/examples/paired_metrics_synthetic.json --output <dir>
- python -m tools.i6n_contracts.paired_metrics_cli --status --output <dir>

--status exit codes:
- 0: complete
- 2: refused, failed, or an unreadable or unknown status
- 3: no STATUS.json
- 4: finished with an INCOMPLETE result
- 5: RUNNING, which can mean a live run or a crashed one, since there is no lease

Interface changes for readers of the output:
- Groups now carry config_id, protocol_id, calendar_id, contract_sha256, sha256 and exclusions_scope.
- ALL_ROLES groups also carry upstream_exclusions_by_role.
- model_row_keys_sha256 and naive_row_keys_sha256 are removed; row_keys_sha256 remains.
- OLAP metric rows gain group_sha256 and exclusions_scope.

No DOIN, fits, GPU, TEST runs, network, pandas, TensorFlow or databases were used, and no host names or home paths are in the committed files.

## Independent re-review: ACCEPT_AS_CANDIDATE

- Remaining (minor): Minor, non-blocking: aggregate_groups does not type-check n_declared, n_excluded or failures in a group that has been resealed. A resealed group with n_declared='x' raises an untyped TypeError (probe agg_str_declared). failures=['a'] is accepted and counted through Counter.update. The CLI never calls aggregate_groups, so this only affects library callers. It falls inside the documented limit on resealed groups, but a typed GROUP_INCONSISTENT would be cheap.
- Remaining (minor): Minor, non-blocking: the consistency check only enforces n_declared - n_excluded >= n. It does not check the exact invariant n_declared == n + sum(failures) + n_excluded. A resealed group with phantom declared members and no failures is accepted as COMPLETE with coverage 0.2 (probe agg_phantom_declared). This is covered by the declined limit, which is documented and sound under unkeyed digests, but the exact equality would refuse more self-inconsistent reseals.
- Remaining (minor): Test gaps on claims made in this fix round (non-equivalent surviving mutants, listed separately). Nothing tests ALL_ROLES exclusions == {}, the aggregate pooling of upstream_exclusions_by_role, the aggregate's exclusions_scope, constituent_group_sha256, exclusions_scope in the OLAP records, the RecursionError -> CONFIG_INVALID path, the population_kind identity in aggregation (S mixed with U), the n <= n_declared - n_excluded check, or the mse/naive/skill parts of the means check. The code itself behaves correctly in every one of these cases (verified by direct probes).
- Surviving mutant: no_n_declared_check: removing the n_declared - n_excluded < n guard in _verified_group (untested)
- Surviving mutant: means_check_mae_only: the GROUP_INCONSISTENT means check restricted to mae; mse/naive_mae/naive_mse/skill are never tested alone
- Surviving mutant: identity_no_kind: dropping population_kind from _IDENTITY_FIELDS (aggregating an S group with a U group is not tested; the probe shows real code gives AGGREGATE_MISMATCH)
- Surviving mutant: all_roles_pool_upstream: the ALL_ROLES group pooling upstream exclusions into 'exclusions' again (the regression that finding 3 was about; no assertion that ALL_ROLES exclusions stay C2-only)
- Surviving mutant: agg_drop_upstream_by_role: aggregate_groups not pooling upstream_exclusions_by_role (the report claims it pools them; untested)
- Surviving mutant: agg_scope_fixed: aggregate exclusions_scope hardcoded to UPSTREAM_AND_C2 (untested)
- Surviving mutant: agg_no_constituents: aggregate without constituent_group_sha256 (the report claims it; untested)
- Surviving mutant: cli_recursion_parse: dropping RecursionError from run()'s parse except (the claimed RecursionError -> CONFIG_INVALID is untested)
- Surviving mutant: olap_no_scope: exclusions_scope missing from the OLAP records (claimed; untested)
- Surviving mutant: Equivalent or near-equivalent, not defects: naive_from_record (identical after PRICE_CONFLICT validation), identity_no_unit_scale (contract_sha256 already covers unit and scale), cli_status_no_object_check (AttributeError is already caught as STATUS_UNREADABLE), metrics_no_finite_check (still typed, via canonical_json NONFINITE_VALUE)

## Limits

- **Candidate only.** Not integrated into any live executor. No DOIN, fits, GPU, TEST, broker, service, network, database or OLAP write, and no OLAP schema change. Running it on synthetic fixtures is not a scientific measurement and is not end-to-end or trading acceptance.
- **Independent of C1.** The S support is accepted as declared and sealed. E and M are not recomputed and no calendar is read. Whether a receipt was E-eligible is not checked here, because that needs C1.
- **Sealing time.** "Sealed previously" is enforced by digest identity (a tampered or different support is refused; configs are compared only on the identical sealed sha256). This pure component cannot prove that the declared sealed_at came before anyone saw the errors; that needs external custody, for example a commit or OLAP readback of the seal.
- **Scale checks.** These compare the declared unit/scale labels against the contract. A value mislabelled with the right label (for example a return tagged ORIGINAL_PRICE) cannot be detected.
- **ALL_ROLES rule.** A failure in any role marks that forecast failed (conservative). Upstream exclusions are reported per role only, because roles can overlap.
- **Duplicate receipts.** Two receipts for the same (asset, origin, h, role) are refused. If the strategy can legitimately consult one forecast twice in one bar, this rule must be revised.
- **Identity binding is byte-exact.** An output directory is bound to the config file bytes plus the input, contract and code digests, so a reformatted but equivalent config is refused rather than reused.
- **Atomic writes.** The reused `atomic_json` writes a fixed `.tmp` name without fsync. That is safe for the single writer here, not for concurrent writers.
- **Merge conflict expected.** `tools/i6n_contracts/__init__.py` is a one-line docstring; C1 will likely create the same file, giving a trivial add/add conflict when the branches are merged.
- **Fixtures.** Prices are binary fractions so the exact equality assertions hold. Real prices need tolerance-aware aggregation checks, although `math.fsum` is used throughout.

## Reproduce

git -C <path> fetch origin satoshi/i6n-c2-paired-metrics-20261011
git -C <path> worktree add <dir> a2d09fe67df03d2ddb3d2567c2e621bbc4a03038
cd <dir>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q tests/test_i6n_paired_metrics.py
git diff --check 269c2d132b21a55e59cd6faa8242b1ac74f0f7ae HEAD
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m tools.i6n_contracts.paired_metrics_cli --config tools/i6n_contracts/examples/paired_metrics_synthetic.json --output <scratch_out>
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m tools.i6n_contracts.paired_metrics_cli --status --output <scratch_out>
Exit codes: 0 complete, 2 refused/failed, 3 status absent, 4 INCOMPLETE. Rerunning with the same config returns {"reused": true} and writes nothing.
