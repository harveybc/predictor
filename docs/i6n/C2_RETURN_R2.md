# I6-N C2 correction round 2 return (review I6N_CANDIDATE_REVIEW_20261011)

Status: CANDIDATE COMPONENT, correction round. Not integrated into any executor,
not deployed, not a scientific measurement, not READY_FOR_CAMPAIGN. Component
acceptance is for the reviewer to decide. No DOIN, fits, GPU, TEST, brokers,
services, network or databases were used.

Governing documents (read in full, digests verified before work):

- Review `docs/audits/evidence/I6N_CANDIDATE_REVIEW_20261011/REVIEW.md` (Musashi),
  sha256 `ab05e154f882ad0bf4ad15bac75f62a93a8d18f19fd574b855973f994540e069`,
  with its `probe_candidates.py` and `c2_observed.json`. It lives in the
  design worktree and is not committed on this branch.
- Order `docs/handoffs/SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md`,
  sha256 `243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025`.

Pins:

| Role | Commit |
|---|---|
| Audited commit (PRE) | `8dd38b963bc5c9b92f69a183e28d75d735cb90a2` |
| Successor: code, tests and probe replay (POST) | `cdc85fb59c0bc805fddd540232c809d409551dd8` |
| This document | the commit that adds this file, directly after `cdc85fb5` on `satoshi/i6n-c2-paired-metrics-20261011` |

The historical return `docs/i6n/C2_RETURN.md` is unchanged. It stays as evidence
of rounds 0 and 1. Where it disagrees with this file, this file governs (see
"Superseded statements" below).

## CR -> test -> PRE -> POST

PRE means the successor test file run against the audited code `8dd38b96`.
POST means the same test file run against `cdc85fb5`. The counts come from the
retained runs described in the next section. Every CR test demands rejection,
no overwrite, or non-success. None of them encodes the defective behaviour
that the probe observed.

| CR | Test(s) in `tests/test_i6n_paired_metrics.py` | PRE on 8dd38b96 | POST on cdc85fb5 |
|---|---|---|---|
| CR02 phantom declared members (n=1 of 5 declared, no failures) | `test_cr02_phantom_declared_members_are_refused` | FAIL (accepted as COMPLETE, coverage 0.2) | PASS (GROUP_INCONSISTENT) |
| CR02 negative error sums with recomputed means | `test_cr02_negative_error_sums_are_refused_even_with_recomputed_means` | FAIL (accepted, MAE -0.25) | PASS (GROUP_INCONSISTENT) |
| CR02 sums must be mutually possible: sqrt(sum e^2) <= sum\|e\| <= sqrt(n sum e^2) | `test_cr02_error_sums_must_be_mutually_possible` | FAIL | PASS |
| CR02 zero rows imply zero sums | `test_cr02_zero_rows_require_zero_error_sums` | FAIL | PASS (honest empty group still pools: control) |
| CR02 counts are non-bool non-negative ints (n, n_declared, n_excluded x True/False/-1/1.0/"1"/None) | `test_cr02_counts_must_be_non_bool_non_negative_integers` (18 cases) | FAIL 18/18 (accepted, wrong code or TypeError) | PASS 18/18 (SCHEMA_INVALID) |
| CR02 typed reason->count maps (failures, exclusions, upstream_exclusions; 8 bad shapes each) | `test_cr02_reason_count_maps_are_typed` (24 cases) | FAIL 24/24 | PASS 24/24 (SCHEMA_INVALID) |
| CR02 n_declared == n + n_excluded + sum(failures) | `test_cr02_declared_must_equal_rows_plus_exclusions_plus_failures` | FAIL | PASS (honest INCOMPLETE group pools: control) |
| CR02 n_excluded == sum(C2 exclusions) | `test_cr02_n_excluded_must_match_the_c2_exclusion_counts` | FAIL | PASS |
| CR02 state, coverage, reason, scope and skill reasons derived, never trusted | `test_cr02_state_coverage_and_reason_are_derived_never_trusted` | FAIL | PASS |
| CR02 row keys belong to the group's own h and role | `test_cr02_row_keys_must_belong_to_the_group_horizon_and_role` | FAIL | PASS |
| CR02 typed refusals, never TypeError (unhashable, wrong-type identity fields, nested row keys, non-JSON values) | `test_cr02_untyped_fields_are_typed_refusals_never_type_errors` (6), `test_cr02_non_json_values_are_typed_refusals_never_type_errors` | FAIL 7/7 | PASS 7/7 (SCHEMA_INVALID) |
| CR02 upstream exclusions kept separate: outside C2's declared denominator, published once, not repeated by ALL_ROLES | `test_cr02_upstream_exclusions_stay_out_of_the_c2_denominator_and_are_published_once`; amended `test_empty_population_has_no_score_but_keeps_exclusion_counts`, `test_all_roles_group_reports_the_sealed_exclusions_per_role` | FAIL 3/3 | PASS 3/3 |
| CR02 verify_result and compare_configs refuse a self-consistently resealed result with an impossible group | `test_cr02_results_with_an_impossible_group_are_refused_by_verify_and_compare` | FAIL (verify passed, compare returned COMPARABLE) | PASS (GROUP_INCONSISTENT; honest pair COMPARABLE: control) |
| CR02 result state derived from its groups | `test_cr02_result_state_is_derived_from_its_groups` | FAIL | PASS (RESULT_INCONSISTENT) |
| CR03 status with RESULT.json deleted | `test_cr03_status_with_a_missing_result_is_not_success_and_writes_nothing` | FAIL (exit 0, COMPLETED) | PASS (exit 6, RESULT_MISSING, nothing written) |
| CR03 status with a corrupt RESULT.json | `test_cr03_status_with_a_corrupt_result_is_not_success` | FAIL (exit 0) | PASS (exit 6, RESULT_UNREADABLE) |
| CR03 status with another identity's valid RESULT.json | `test_cr03_status_with_the_result_of_another_identity_is_not_success` | FAIL (exit 0) | PASS (exit 6, RESULT_IDENTITY_MISMATCH) |
| CR03 status with changed RESULT.json bytes | `test_cr03_status_with_changed_result_bytes_is_not_success` | FAIL (exit 0) | PASS (exit 6, RESULT_HASH_MISMATCH) |
| CR03 INCOMPLETE status also requires a verified result | `test_cr03_incomplete_status_also_requires_a_verified_result` | FAIL (exit 4 without RESULT) | PASS (4 with the result: control; 6 without it) |
| CR02+CR03 forged RESULT and STATUS (both resealed, summary rewritten): neither reused nor reported as success, nothing overwritten | `test_cr02_cr03_a_forged_self_consistent_result_is_neither_reused_nor_reported_as_success` | FAIL (status exit 0) | PASS (status exit 6 RESULT_INVALID; run OUTPUT_CORRUPT; bytes and mtimes unchanged) |
| CR03 damaged status distinguishable from FAILED; success printed only after verification | amended `test_cli_status_on_a_non_object_or_unknown_state_is_typed` (3), amended `test_cli_status_is_read_only_and_returns_once` | FAIL 4/4 (exit 2 like FAILED; no verification) | PASS 4/4 (exit 6 STATUS_UNREADABLE; COMPLETE_VERIFIED) |
| CR04 semantics: RESULT write failure ends in typed FAILED, no `.tmp` left | `test_cr04_result_write_failure_is_a_typed_failed_status_without_leftovers` | FAIL (untyped OSError, status RUNNING) | PASS (FAILED/OUTPUT_WRITE_FAILED, phase WRITING_RESULT) |
| CR04 semantics: a failure that cannot be recorded is non-success and is not claimed as durable FAILED | `test_cr04_unrecordable_failure_is_non_success_without_claiming_a_durable_failed` | FAIL (untyped OSError) | PASS (durable:false; disk RUNNING; --status 5) |
| CR04 semantics: status unwritable before start | `test_cr04_unwritable_status_before_start_is_a_typed_non_success` | FAIL (untyped OSError) | PASS (STATUS_WRITE_FAILED, durable:false; --status 3) |
| CR04 semantics: an interrupt propagates even when its status cannot be written | `test_cr04_interrupt_still_propagates_when_its_status_cannot_be_written` | FAIL (OSError replaced the KeyboardInterrupt) | PASS |
| Focused coverage: ALL_ROLES exclusions scope (C2 exclusion of a shared forecast counted once) | `test_all_roles_counts_a_c2_exclusion_of_a_shared_forecast_once` | PASS (coverage, no defect) | PASS |
| Focused coverage: ALL_ROLES aggregate pools upstream per role, keeps C2 scope and constituent digests; OLAP scope present | `test_all_roles_aggregate_pools_upstream_per_role_and_keeps_the_c2_scope` | FAIL (needs the new per-role `upstream_exclusions` / scope constant) | PASS |
| Focused coverage: S/U population identity in aggregation | `test_aggregation_refuses_to_pool_opportunity_with_usage` | PASS (coverage, no defect) | PASS (AGGREGATE_MISMATCH) |
| Focused coverage: MSE, naive and skill parts of the means check, each alone | `test_means_check_covers_each_mean_and_skill_alone` (6) | PASS 6/6 (coverage, no defect) | PASS 6/6 |
| Focused coverage: RecursionError in the config -> CONFIG_INVALID | `test_cli_recursion_in_the_config_is_a_typed_config_invalid` | PASS (coverage, no defect) | PASS |
| CR05 reproduction pinned, PRE/POST split, current limits | this document | n/a | n/a |

Totals: PRE 77 failed, 86 passed. The 77 failures are all CR02, CR03 and CR04
regressions, plus four pre-existing tests amended for the interface change.
The 86 passes are the unchanged earlier suite plus the 9 focused-coverage
cases that pass on the audited code. POST: 163 passed.

## PRE (retained)

Scratch checkout outside the repository: `git worktree add --detach <pre> 8dd38b96`.
The successor test file (sha256 `9d3e1c2e9716cfd68c92c62b966dc1ea6b072d5fb74f6cec8dfb0e7fbc524414`,
identical to the file in `cdc85fb5`) was copied over the audited one.

```
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py
-> exit 1: "77 failed, 86 passed in 4.64s"
```

The red PRE suite was run a second time, with `-rA` added, only to list which
new tests pass on the audited code (the coverage rows above). No green suite
was rerun.

Precision on the CR03 PRE column. On the audited code, several CR03 tests
first fail on the new `EXIT_DAMAGED` constant or the new `disposition` field,
before they reach their behavioural assertion. The defective behaviour itself
was measured separately, in the same PRE checkout, using a scratch script that
was not committed. That script ran the audited CLI, damaged its output in the
same way as the tests, and called `status()`. Results: corrupt RESULT.json
exit 0; another identity's RESULT.json exit 0; reformatted RESULT.json bytes
exit 0; INCOMPLETE run with RESULT.json deleted exit 4. The deleted-result
case (exit 0) comes from the probe replay below. The CR02 PRE failures are
behavioural (`DID NOT RAISE`, or an untyped `TypeError` for the typing cases).
The CR04 PRE failures are untyped `OSError`s, or an `OSError` that replaced
the `KeyboardInterrupt`.

## POST (retained)

In the C2 worktree at the successor code, before committing:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py
-> exit 0: "163 passed in 0.72s"
git diff --check   -> exit 0
```

One development run (163 passed) came before a final one-line hardening, which
typed an OSError while hashing a reused result. The POST run above is the
single run on the final code. The committed tree `cdc85fb5` is byte-identical
to that code. Importing the pure module and the CLI loads none of pandas,
tensorflow, sqlite3, requests or urllib.request.

## Review probe outcome

The unmodified `probe_candidates.py c2 --repo <C2 worktree at cdc85fb5>` now
exits 1 on its first scenario with
`PairedMetricsError: GROUP_INCONSISTENT: groups[0]: n_declared 5 != n 1 + n_excluded 0 + failures 0`.
The probe has no exception handling, so this typed refusal is the expected
observation. It is not a crash of the candidate.

To observe all three scenarios, `docs/i6n/evidence_r2/probe_c2_per_scenario.py`
builds each one exactly as the probe does and records the values or the typed
refusal:

| Scenario | PRE 8dd38b96 (matches `c2_observed.json`) | POST cdc85fb5 |
|---|---|---|
| phantom_population | NO_EXCEPTION: state COMPLETE, n 1, n_declared 5, coverage 0.2 | REFUSED: GROUP_INCONSISTENT (n_declared 5 != n 1 + n_excluded 0 + failures 0) |
| negative_error_sums | NO_EXCEPTION: state COMPLETE, mae -0.25, mse -0.0625, naive_mae -0.5 | REFUSED: GROUP_INCONSISTENT (sum_abs_error is negative) |
| missing_result_status | exit 0, state COMPLETED, result_state COMPLETE | exit 6, state RESULT_INVALID, disposition RESULT_MISSING, result_state null |

## Interface changes for readers of C2 output

- `RESULT_SCHEMA_VERSION` is now `i6n.paired_metrics.result.v2`. A v1 result
  file is refused on reuse (OUTPUT_CORRUPT) and reported damaged by `--status`.
- Per-role groups: `exclusions` holds only C2's own exclusions, so
  `n_excluded == sum(exclusions)`. New `upstream_exclusions` holds the sealed
  upstream counts. `exclusions_scope` is `C2_ONLY_UPSTREAM_SEPARATE`, and the
  constant `EXCLUSIONS_UPSTREAM_AND_C2` was removed. ALL_ROLES groups are
  unchanged: C2-only `exclusions`, per-role `upstream_exclusions_by_role`,
  scope `C2_ONLY_UPSTREAM_REPORTED_PER_ROLE`.
- OLAP count rows gain the category `UPSTREAM_EXCLUSION`, emitted from per-role
  groups only. ALL_ROLES groups never repeat upstream counts.
- Aggregates of per-role groups carry a pooled `upstream_exclusions`. All
  aggregates derive their state and coverage from counts.
- `verify_result` now verifies semantics as well as the seal, so
  `compare_configs` and CLI reuse refuse impossible groups. A row listed twice
  inside one group is DOUBLE_COUNT.
- CLI: new `EXIT_DAMAGED = 6`. An unreadable or unknown STATUS moved from exit
  2 to 6. `--status` payloads carry `disposition`, and verified successes carry
  `result_verification`. Run payloads carry `durable`, and typed errors carry
  `phase`.
- Exit codes: 0 verified COMPLETE; 2 refused/failed; 3 status absent;
  4 verified INCOMPLETE (never an approval); 5 RUNNING (live or dead, no
  lease); 6 damaged (status unreadable or unknown, or RESULT.json missing,
  unreadable, unsealed, of another identity, semantically invalid or different
  from the recorded digests).

## Current limits

- **Candidate only.** No integration with C1 or any executor. No warehouse or
  OLAP write, and no OLAP schema change. Synthetic fixtures are not a
  scientific measurement and not end-to-end or trading acceptance. Not
  READY_FOR_CAMPAIGN.
- **Unkeyed digests.** Arithmetic contradictions are now refused locally. A
  forgery that is both plausible and self-consistent (false but possible
  counts and sums, resealed RESULT and STATUS) remains undetectable without
  the rows and external custody. This round adds no signatures, leases or
  calendar system.
- **Sum relations are necessary conditions only.** The check
  sqrt(sum e^2) <= sum|e| <= sqrt(n sum e^2) uses a relative tolerance of
  1e-9 and an absolute slack of n x 1.5e-154 for squares that underflow. It
  rejects impossible sums. It does not prove that sums are true.
- **Independent of C1.** The S support is taken as declared and sealed. E/M
  eligibility of receipts is not checked here.
- **Sealing time.** The prior sealing of `sealed_at` needs external custody,
  for example a commit or an OLAP readback of the seal.
- **Scale checks.** Unit and scale are compared by label only.
- **ALL_ROLES rule.** A failure in any role marks the forecast failed. Upstream
  counts carry no row identities and stay unpooled per role.
- **Duplicate receipts.** A second receipt for the same (asset, origin, h, role)
  is refused.
- **No lease.** `RUNNING` cannot be told apart from a dead process (SIGKILL,
  power loss) or from a run whose failure could not be written
  (`durable:false`). `--status` exits 5 in all of these cases, never 0.
- **`--status` does not compare the recorded code digest with the current
  code.** It verifies the result against STATUS. A rerun with changed code is
  refused (OUTPUT_IDENTITY_MISMATCH).
- **Identity binding is byte-exact** (config file bytes plus input, contract
  and code digests).
- **Atomic writes.** The reused `atomic_json` writes a fixed `.tmp` without
  fsync, which is safe only for a single writer. A failed write now removes
  its `.tmp`.
- **Fixtures** use binary-fraction prices for exact equality. Real prices need
  tolerance-aware checks.
- **Merge conflict expected.** `tools/i6n_contracts/__init__.py` will give a
  trivial add/add conflict with C1.
- **No mutation run this round.** The killing tests from round 1 are
  unchanged. This round's guarantees rest on the PRE-failing regressions above.

## Superseded statements in C2_RETURN.md

- The Reproduce section pins `a2d09fe6`. That predates both correction rounds
  and must not be used. Use the commands below.
- "A reseal with self-consistent but false sums is not detectable" is
  narrowed: inconsistent or impossible reseals are now refused (CR02), and only
  plausible self-consistent forgeries remain (see limits).
- The re-review's minor items are fixed: untyped `n_declared` raising
  TypeError, `failures=['a']` accepted, and only `n_declared - n_excluded >= n`
  enforced.
- The re-review's untested items now have tests: ALL_ROLES exclusions, the
  aggregate's upstream pooling, scope and constituents, RecursionError, S/U
  identity and the parts of the means check.
- "Only a process killed without running Python code can leave RUNNING" now
  also covers a failure whose status write itself fails, which is reported as
  `durable:false`.
- The `--status` exit table (STATUS unreadable = 2) is replaced by the table
  above (6).
- The C1-specific limits that Musashi flagged (fixed offsets, sqlite3 through
  i6m_queue, code-digest composition) do not apply to C2. C2's code digest
  composition is unchanged: the pure module, the CLI and the reused atomic
  writer module.

## Reproduction

PRE (new regression tests against the audited code; expected exit 1, 77 failed / 86 passed):

```bash
git -C <repo> fetch origin satoshi/i6n-c2-paired-metrics-20261011
git -C <repo> worktree add --detach <pre> 8dd38b963bc5c9b92f69a183e28d75d735cb90a2
git -C <pre> checkout cdc85fb59c0bc805fddd540232c809d409551dd8 -- tests/test_i6n_paired_metrics.py
cd <pre>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py
git -C <pre> checkout 8dd38b963bc5c9b92f69a183e28d75d735cb90a2 -- tests/test_i6n_paired_metrics.py
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python <post>/docs/i6n/evidence_r2/probe_c2_per_scenario.py --repo <pre>
```

POST (successor; expected 163 passed, clean diff check, probe scenarios refused):

```bash
git -C <repo> worktree add --detach <post> cdc85fb59c0bc805fddd540232c809d409551dd8
cd <post>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py
git diff --check 269c2d132b21a55e59cd6faa8242b1ac74f0f7ae HEAD
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python <review-dir>/probe_candidates.py c2 --repo <post>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python docs/i6n/evidence_r2/probe_c2_per_scenario.py --repo <post>
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m tools.i6n_contracts.paired_metrics_cli --config tools/i6n_contracts/examples/paired_metrics_synthetic.json --output <scratch_out>
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m tools.i6n_contracts.paired_metrics_cli --status --output <scratch_out>
```

The unmodified review probe is expected to exit 1 with the typed
GROUP_INCONSISTENT refusal of its first scenario. On a scratch copy, deleting
`<scratch_out>/RESULT.json` makes `--status` exit 6 with disposition
RESULT_MISSING.
