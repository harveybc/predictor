# I6-N C1 correction round 2 return (Musashi review CR01-CR05)

Status: CANDIDATE IMPLEMENTATION, component acceptance only. It is not
integrated, not deployed and not a scientific measurement, and this return
does NOT claim READY_FOR_CAMPAIGN. No GPU, DOIN, fits, TEST, services,
network, pandas, TensorFlow or databases were used. Every test is CPU-only and
synthetic, run under `CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s`.

The historical return `docs/i6n/C1_RETURN.md` is kept unchanged as evidence.
Where the two disagree, this file supersedes it (see "Superseded limits").

## Pins

| Object | Value |
|---|---|
| Audited commit (PRE) | `a1e64e4cad333b468664e1deefcbad43be543a54` |
| Successor code commit (POST) | `b1f0750255c5da4009304ff41312e97f0f8f173a` |
| Branch | `satoshi/i6n-c1-eligibility-20261011` |
| Review `REVIEW.md` (I6N_CANDIDATE_REVIEW_20261011) | sha256 `ab05e154f882ad0bf4ad15bac75f62a93a8d18f19fd574b855973f994540e069` |
| Review probe `probe_candidates.py` | sha256 `7cb0947188553fabcebe1e4a5e0f61e61b94a03113d121273ddc94b37c366aea` |
| Owner order `SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md` | sha256 `243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025` |

This document was committed after the successor code commit and changes no
code. The code under test is exactly `b1f07502`.

## CR -> test -> PRE (a1e64e4c) -> POST (b1f07502)

The tests were written before the fix. Each one demands REJECTION, NO
OVERWRITE or NON-SUCCESS; none of them freezes the behaviour the review probe
observed. PRE = the successor test file run against the audited code. All
tests are in `tests/test_i6n_eligibility.py`.

| CR | Test | PRE on a1e64e4c | POST on b1f07502 |
|---|---|---|---|
| CR01 | `test_cr01_orphan_result_of_another_input_is_refused_without_writing` | FAIL: exit 0, overwrote the result (`assert 0 == 2`) | PASS: exit 2 `OUTPUT_HOLDS_DIFFERENT_INPUT`, with or without `--recover`; directory bytes unchanged, no STATUS written |
| CR01 | `test_cr01_orphan_result_of_other_code_is_refused_without_writing` | FAIL: exit 0 | PASS: exit 2 `OUTPUT_HOLDS_DIFFERENT_CODE`, nothing written |
| CR01 | `test_cr01_damaged_orphan_result_is_refused_without_writing` [garbage, truncated, edited_content] | FAIL x3: exit 0, damaged result replaced | PASS x3: exit 2 `RESULT_DAMAGED`, also with `--recover`; nothing written |
| CR01 | `test_cr01_identical_orphan_result_needs_explicit_recovery_never_adoption` | FAIL: exit 0 (recomputed and replaced on existence) | PASS: without the flag, exit 2 `RECOVERY_REQUIRED` and nothing written; `--recover` gives exit 0 `RECOVERED`, same inode, mtime and bytes, status `recovered: true` |
| CR01 | `test_cr01_recovery_refuses_bytes_that_differ_from_the_recomputation` | FAIL: no `RECOVERY_MISMATCH` (flag unknown) | PASS: exit 2 `RECOVERY_MISMATCH`, nothing written |
| CR01 | `test_cr01_failed_status_beside_an_intact_result_is_not_adopted` | FAIL: exit 0 | PASS: exit 2 `RECOVERY_REQUIRED` |
| CR01 (structural) | `test_status_is_replaced_atomically_and_result_is_created_never_replaced` (replaces `test_status_and_result_are_replaced_atomically`) | FAIL: `SUPPORT_TABLE.json` swapped with `os.replace` (can clobber) | PASS: result published with `os.link` exactly once (create-only); STATUS swapped atomically |
| CR03 | `test_cr03_status_exit_zero_only_for_an_intact_complete_terminal` [10 cases] | FAIL x10: no `disposition` field (KeyError). The old exits were 0 for absent, damaged and resealed outputs, and 2 for an unreadable status. The `complete` case is a control and fails PRE only on the missing field | PASS x10: complete 0; absent 10; failed 14; result garbage, extra byte or missing 16; status garbage 16; result edited with the status resealed 16; result of another identity with the status resealed 16; status missing 17. Directory bytes are unchanged in every case |
| CR03 | `test_cr03_status_exit_codes_are_distinct_and_only_success_is_zero` | FAIL: no `STATUS_EXIT_CODES` | PASS |
| CR03 | `test_cli_status_is_read_only_and_returns_once` (amended: absent dir gives 10, not 0) | FAIL: `assert 0 == 10` | PASS |
| CR03 | `test_cli_refuses_a_damaged_completed_result` (amended: `--status` gives 16, not 0) | FAIL: `assert 0 == 16` | PASS |
| CR04 | `test_cr04_ordinary_exception_is_recorded_as_typed_failed` | FAIL: `RuntimeError` escaped, state RUNNING | PASS: CLI exit 1; FAILED with `UNEXPECTED_ERROR` and `exception_type: RuntimeError`; `--status` 14; `run_command` re-raises after recording; control rerun recomputes, exit 0 |
| CR04 | `test_cr04_failure_while_writing_the_result_is_failed_without_result_or_temp` (fsync of the result temp raises ENOSPC) | FAIL: `OSError` escaped | PASS: exit 1, FAILED (`OSError`), only STATUS.json left (no result, no temp) |
| CR04 | `test_cr04_interrupt_is_recorded_and_propagated` | FAIL: state RUNNING | PASS: `KeyboardInterrupt` propagates; state INTERRUPTED; `--status` 15; control rerun exit 0 |
| CR04 | `test_cr04_unrecordable_terminal_status_exits_nonzero_without_claiming_failed` | FAIL: `RuntimeError` escaped | PASS: exit 3; `STATUS_NOT_RECORDED` with `durable: false`; no FAILED or COMPLETED event; persisted state stays RUNNING; no temp files |
| CR04 | `test_cr04_unwritable_output_exits_nonzero_without_status` | FAIL: `FileExistsError` escaped | PASS: exit 3, `STATUS_NOT_RECORDED`, nothing written |
| CR04 | `test_cr04_persisted_running_is_distinguished_from_a_live_writer` | FAIL: `RuntimeError` escaped (no persisted RUNNING could even be produced cleanly) | PASS: live writer gives 11 RUNNING and a second run is refused `RUN_IN_PROGRESS`; reused pid (start time differs) gives 12 INCOMPLETE; unverifiable writer gives 13 and the run is refused; exited writer gives 12 INCOMPLETE, then recomputed with `previous_disposition: INCOMPLETE` |
| re-review gap | `test_atomic_writer_removes_its_temp_file_on_failure` | PASS (cleanup already existed; it was untested) | PASS. A targeted check of reviewer mutant N10 (cleanup `finally` replaced by `pass`) on a scratch copy of b1f07502: this test FAILS, so N10 is killed |
| CR05 | this document (no automated test) | `C1_RETURN.md` reproduction pointed at 30804654 | PRE and POST commands below, each pinned to its exact commit |

PRE totals: exit 1, **28 failed, 44 passed in 1.48s**. POST totals: exit 0,
**72 passed in 0.74s**. `git diff --check` is clean.

The 44 PRE passes are the earlier contract tests plus the cleanup test. None
of the 28 failures were caused by the test harness. Every failure is a
required rejection or non-success that the audited code did not provide,
apart from the CR03 `complete` control described in the table.

POST attempt 1 gave 1 failed, 71 passed. The failure was a test bug: the
control rerun in `test_cr04_ordinary_exception_is_recorded_as_typed_failed`
did not drain its COMPLETED event before parsing `--status`. One
`capsys.readouterr()` line was added and the suite was run once more (72
passed). PRE is unaffected: that test fails on a1e64e4c at its first CLI
call, before the added line.

## Reproduction

PRE, the successor tests against the audited code. This is expected to FAIL:
28 failed, 44 passed.

```bash
git -C <repo> fetch origin satoshi/i6n-c1-eligibility-20261011
git -C <repo> worktree add --detach <pre-dir> a1e64e4cad333b468664e1deefcbad43be543a54
git -C <repo> show b1f0750255c5da4009304ff41312e97f0f8f173a:tests/test_i6n_eligibility.py \
  > <pre-dir>/tests/test_i6n_eligibility.py
cd <pre-dir>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_eligibility.py
```

POST, the successor code. This is expected to pass: 72 passed.

```bash
git -C <repo> worktree add --detach <post-dir> b1f0750255c5da4009304ff41312e97f0f8f173a
cd <post-dir>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_eligibility.py
git diff --check a1e64e4cad333b468664e1deefcbad43be543a54 b1f0750255c5da4009304ff41312e97f0f8f173a
```

Review probe, run unmodified from the review directory against either
worktree:

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python probe_candidates.py c1 --repo <post-dir>
```

CLI walk-through on the synthetic example, from `<post-dir>`. Each line was
run as a subprocess on b1f07502 and gave the result shown:

```bash
C=tools/i6n_contracts/examples/eligibility_synthetic.json
python -m tools.i6n_contracts.eligibility_cli --config $C --output <scratch>/out           # 0 COMPLETED
python -m tools.i6n_contracts.eligibility_cli --status --output <scratch>/out              # 0 COMPLETE_INTACT
python -m tools.i6n_contracts.eligibility_cli --config $C --output <scratch>/out           # 0 REUSED
rm <scratch>/out/STATUS.json
python -m tools.i6n_contracts.eligibility_cli --config <horizons [2]> --output <scratch>/out  # 2 OUTPUT_HOLDS_DIFFERENT_INPUT
python -m tools.i6n_contracts.eligibility_cli --status --output <scratch>/out              # 17 ORPHAN_RESULT
python -m tools.i6n_contracts.eligibility_cli --config $C --output <scratch>/out           # 2 RECOVERY_REQUIRED
python -m tools.i6n_contracts.eligibility_cli --config $C --output <scratch>/out --recover # 0 RECOVERED
python -m tools.i6n_contracts.eligibility_cli --status --output <scratch>/none             # 10 ABSENT
```

## Probe outcome

| Probe case | a1e64e4c (review `c1_observed.json`) | b1f07502 (rerun of the same probe) |
|---|---|---|
| orphan output, different config | exit 0, previous result overwritten | exit 2, previous result NOT overwritten |
| unexpected `RuntimeError` | exception, persisted RUNNING, `--status` exit 0 | exception re-raised after recording, persisted FAILED, `--status` exit 14 |
| damaged result, `--status` | exit 0, state COMPLETED, `result_intact` false | exit 16 (DAMAGED), state COMPLETED, `result_intact` false |

The probe's `state` field is the persisted state. The disposition is the new
`disposition` field, and the exit code follows the disposition.

## Contract now implemented (CLI only; the pure module is unchanged)

- **Result identity, independent of STATUS.** `SUPPORT_TABLE.json` carries
  `config_sha256`, `code_sha256` and `identity_sha256`, plus `table_sha256`,
  a canonical digest of the table without that field. The file is parsed and
  checked for identity consistency, its self digest and row completeness
  (every week x role x head exactly once, `units_total` agrees) before any
  decision is made.
- **No overwrite.** The result is created with a complete fsynced temp file
  and `os.link`, which fails if the name exists. It is never replaced.
  Another identity (in the status or in the result) and a damaged result are
  refused with exit 2, with or without `--recover`.
- **Explicit recovery.** A same-identity intact result without a matching
  COMPLETED status needs `--recover`. The table is recomputed from scratch in
  memory, and the existing bytes are accepted only if they are identical. The
  file is never rewritten on that path.
- **`--status`** (read-only, one return): `COMPLETE_INTACT 0`, `ABSENT 10`,
  `RUNNING 11`, `INCOMPLETE 12`, `RUNNING_UNVERIFIED 13`, `FAILED 14`,
  `INTERRUPTED 15`, `DAMAGED 16`, `ORPHAN_RESULT 17`. It reports
  `disposition`, `problem`, the persisted `state`, `result_intact`,
  `writer_liveness`, `code_matches_current` and the raw record.
- **Run exits**: 0 completed, reused or recovered; 1 FAILED recorded; 2
  refusal or usage error (nothing written); 3 status could not be recorded
  (nothing durable claimed).
- **Liveness**: a RUNNING record stores the writer pid and its process start
  time from `/proc/<pid>/stat`. It is checked once and never polled.

## Current limits

- Component acceptance only. There is no integration with C2 or any
  executor, and no READY_FOR_CAMPAIGN claim. The order requires a reviewer
  other than the author to check the counterexamples.
- The digests are unkeyed. The self digest and the status pin detect damage
  and inconsistent edits. A coherent forgery, with every digest recomputed,
  is caught only by `--recover` recomputation. Authenticity (signatures) is
  out of scope by order.
- Liveness evidence is valid only on the same host and pid namespace, and
  needs Linux `/proc`. ALIVE means the writer process exists, not that it is
  making progress. Where liveness cannot be checked, the record reads as
  `RUNNING_UNVERIFIED` and a new run is refused. SIGKILL and power loss
  cannot write a terminal state; they read as INCOMPLETE.
- There is no lease. One writer per directory is a contract. `os.link`
  guarantees the result is never clobbered, but two concurrent writers can
  still race on `STATUS.json`.
- Publishing the result needs hard-link support. On a filesystem without it
  the run fails as a typed FAILED (`UNEXPECTED_ERROR`, `OSError`).
- The directory entry is not fsynced after `os.replace` or `os.link`. A power
  loss right after publishing can lose an entry. The possible outcomes then
  read as ABSENT, INCOMPLETE, ORPHAN_RESULT or DAMAGED
  (`TERMINAL_OUTPUT_MISSING`), never as COMPLETE_INTACT.
- The STATUS schema is now `i6n.eligibility.cli_status.v2`, and the result
  gained identity and self-digest fields. Directories written by earlier
  commits read as DAMAGED and are refused, so use a new output directory.
- The pure-module limits are unchanged from `C1_RETURN.md`:
  - The entry cutoff K is supplied per week as an already resolved instant.
  - Split membership is decided by the target instant, `[start, end)`.
  - Only the primary exclusion reason is counted.
  - `prediction_available` is supplied by the caller.
  - No real calendar is used; the fixtures are synthetic.
  - Keras masking is not validated.
  - Purge beyond target membership and the label cutoff is not covered, nor
    are 24/7 calendars.

## Superseded limits and claims in `C1_RETURN.md`

- "The DST case uses fixed offsets (-04:00 to -05:00)": superseded since
  5e6fc9a7 by `test_dst_end_with_real_zoneinfo_uses_elapsed_hours` (real
  `America/New_York` transition, timezone rule only).
- "The CLI imports `digest` and `atomic_json` from `tools/i6m_queue.py`":
  superseded since 5e6fc9a7. The writers are local, the digest is
  `eligibility.canonical_digest`, and `sqlite3` is not loaded (see
  `test_cli_imports_no_database_dataframe_or_tensor_library`).
- "The code digest covers ... and i6m_queue.py": superseded. The code digest
  covers exactly `eligibility.py`, `eligibility_cli.py` and
  `business_weekly_protocol.py`.
- "A same-identity RUNNING or FAILED record is recomputed once from scratch":
  superseded. RUNNING with a live or unverifiable writer is refused. A FAILED,
  INTERRUPTED or stale RUNNING record is recomputed only when no result
  exists. If an intact result exists, recovery must be explicit.
- "Exit codes: 0 completed/reused/status": superseded. `--status` exits 0
  only for COMPLETE_INTACT.
- "Reproduce: worktree add 30804654": superseded by the pinned PRE and POST
  commands above.
