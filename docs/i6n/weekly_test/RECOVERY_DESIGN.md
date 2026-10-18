# Transient recovery for the prequential weekly TEST: design

Candidate design for part A of [ORDEN_SATOSHI_OPERACION.md](ORDEN_SATOSHI_OPERACION.md). It is
written and committed before the code. Scope: `tools/i6dmo/prequential_test.py` and the status
seam of `weekly_test.py`. The scientific files (`model.py`, `train.py`, `weekly_test_data.py`,
`continuation*.py`, the data modules) stay byte-identical to the base, and a test asserts it.

## 1. Classification by typed cause, never by message text

Every exception raised between the claim of a cell and its terminal is classified by its TYPE and,
for the module's own `ValueError` codes, by the code before the first `:`. The message text is
never inspected. Anything not listed is UNKNOWN and is NOT retried.

| Class | Typed cause | Examples | Action |
|---|---|---|---|
| `TRANSIENT_PROCESS` | `TransientProcessError` (raised by this module), `tensorflow.errors.ResourceExhaustedError`, `tensorflow.errors.UnavailableError`, `ConnectionError`, `TimeoutError`, `BlockingIOError` | `GPU_NOT_VISIBLE` (the retained runtime is right but no GPU is enumerated), `NVIDIA_SMI_UNAVAILABLE`, GPU OOM inside the fit, a transient I/O timeout | Initial attempt + at most 2 retries, cooldowns 30 s then 60 s, every attempt in the ledger. Exhausted: terminal `FAILED` with cause `TRANSIENT_EXHAUSTED` and the minimum required action. |
| `RESOURCE_BLOCKED` | The pre-launch gate of `run` (foreign GPU compute process, `MemAvailable` below the floor) | another process on the GPU; host memory below 7 GiB | No attempt is consumed. State `WAIT_RESOURCES` with the measured values. Unbounded in count; bounded in time only by the operator. The cap is never changed. |
| `PERMANENT` | `ValueError` whose code is in the integrity/contract/version/causality table below; `AssertionError`; `KeyError`; `MemoryError` | `SEALED_FORECAST_IDENTITY`, `DONOR_WEIGHTS_CHANGED`, `ARTIFACT_DIGEST`, `RUNTIME_VERSION_MISMATCH`, `DONOR_NOT_YET_AVAILABLE`, `TRAIN_CUTOFF_OR_LABEL_MATURITY` | Terminal `FAILED` on the first occurrence. Never retried. Requires audited recovery. |
| `UNKNOWN` | Everything else | an unexpected `TypeError` | Terminal `FAILED` on the first occurrence. Never retried. |

The version check and the GPU-visibility check share today one code, `RETAINED_GPU_RUNTIME_REQUIRED`.
They are split: `RUNTIME_VERSION_MISMATCH` (PERMANENT, `ValueError`) when TensorFlow or Keras differ
from the pin, and `GPU_NOT_VISIBLE` (`TransientProcessError`) when the pinned runtime enumerates no
GPU. The split is by the raised type, so a future shared message cannot merge them again.

PERMANENT `ValueError` codes (the module's own contract and integrity refusals):

`PREDICT_INPUT_KEYS`, `PREDICT_INPUT_NONFINITE_OR_FUTURE`, `DONOR_NOT_YET_AVAILABLE`,
`SEALED_FORECAST_IDENTITY`, `SEALED_ORIGINS_CHANGED`, `FORECAST_SHAPE_OR_NONFINITE`,
`OBSERVED_INPUT_CHANGED`, `NAIVE_CACHE_CHANGED`, `UPDATE_CHANGED_PREDICTION`,
`NOT_FOUR_CALENDAR_YEARS`, `TRAIN_CUTOFF_OR_LABEL_MATURITY`, `NO_OBSERVED_MONITOR_PAIRS`,
`OBSERVED_WEEK_NOT_FINISHED`, `RUNTIME_VERSION_MISMATCH`, `ARTIFACT_DIGEST`, `DONOR_PATH`,
`DONOR_WEIGHTS_CHANGED`, `WINNING_MODEL_CHANGED`, `UPDATE_DONOR_CHANGED`,
`WARM_START_WEIGHTS_CHANGED`, `SAVED_CHECKPOINT_RELOAD`, `CONTINUATION_NONFINITE_PREDICTION`,
`CONTINUATION_ZERO_NAIVE`, `PREPARED_CHANGED`, `CELLS_CHANGED`, `REPLICATION_RECIPE_CHANGED`,
`TEST_QUALIFICATION_CHANGED`, `TERMINAL_IDENTITY`.

A `ValueError` whose code is not in this table is UNKNOWN (terminal, not retried): an unlisted
code is never assumed to be transient.

## 2. Attempts, cooldowns and the durable ledger

- Ledger: `results/<cell>/ATTEMPTS.json`, append-only, written atomically. One entry per attempt:
  `index`, `started_at`, `ended_at`, `outcome` (`COMPLETE`, `TRANSIENT`, `PERMANENT`, `UNKNOWN`),
  `cause_class`, `cause_code`, `exception_type`, `next_attempt_at`, `evidence` (the relative path of
  `attempts/aNN/FAILURE.json`, which keeps the full message and traceback).
- Budget: `MAX_RETRIES = 2` (three attempts in total), `COOLDOWNS = (30, 60)` seconds, measured
  with an injected clock. The clock is only compared, never slept on.
- A transient outcome never writes `RESULT.json`. The terminal `RESULT.json` is written only on
  completion, on a PERMANENT or UNKNOWN cause, or when the transient budget is exhausted.
- An exhausted budget writes `RESULT.json{state: FAILED, cause_class: TRANSIENT_PROCESS,
  reason: TRANSIENT_EXHAUSTED:<last code>, attempts: [...], required_action: ...}`. The minimum
  required action is named in the terminal: restore the GPU/runtime, then audited recovery of this
  one cell. Nothing is restarted automatically.
- A pre-existing `RESULT.json` is never deleted or rewritten by the recovery path. The historical
  `FAILED` of a cell stays where it is; its successor stays blocked (`BLOCKED_DEPENDENCY`).

## 3. What is preserved across attempts

- `PREDICTED.npz` and `PREDICTION.json`: an existing seal is reused byte for byte; a retry never
  predicts again (`execute_week` already refuses `SEALED_FORECAST_IDENTITY` on any change).
- The donor, the seed, the data arrays, the scales and the configuration are the same objects on
  every attempt: they come from the stored predecessor terminal / `PERIOD.json`, never from the
  failed attempt.
- A completed update (`update/UPDATE.json` with authenticated files and the same `donor_id`) is
  adopted by the retry; the backend never fits a second time.
- `max_epochs`, `patience`, `min_delta`, the loss, the architecture, the columns, the windows, the
  masks and the memory caps are not parameters of the recovery path. A GPU OOM is retried at the
  SAME cap and the same model; it never lowers the cap. No GPU reset and no service restart.

## 4. Scheduling during a cooldown

`run` chooses the first pending cell whose predecessor is terminal `COMPLETE` or
`SKIPPED_NO_EVAL_ROWS` and that is not in cooldown. While the short family cools down, a ready long
cell runs, and the reverse. A family never advances past a predecessor that is not terminal, and
never past a `FAILED` predecessor. If every eligible cell is in cooldown, the process ends with
`RETRY_PENDING` and the earliest `next_attempt_at`.

## 5. Status

`PROGRESS.json` and the tick's `STATUS.json` distinguish, per cell and for the campaign:

| State | Meaning | Fields |
|---|---|---|
| `RUNNING` | a child is executing this cell now | `cell`, `stage`, `attempt` |
| `WAIT_RESOURCES` | the pre-launch gate refused; no attempt consumed | `cell`, `memory_available`, `foreign_gpu_processes` |
| `RETRY_PENDING` | transient attempts remain; waiting for `next_attempt_at` | `cell`, `attempt`, `cause_code`, `next_attempt_at` |
| `BLOCKED_TERMINAL` | a `FAILED` terminal exists (permanent, unknown or exhausted) | `cell`, `cause_class`, `cause_code`, `required_action`, `dependents` |
| `BLOCKED_DEPENDENCY` | a successor waits on a blocked predecessor | `cell`, `dependency` |

The campaign state is never `RUNNING` while a terminal block exists. The ETA is `NOT_ESTIMABLE`
(seconds `null`) whenever a cell is `RETRY_PENDING` or blocked; a measured mean is reported only
when every remaining cell is runnable.

## 6. Tests (fake backends, injected clock, no sleep, no poll)

| Requirement | Test |
|---|---|
| Transient after the seal → recovery; prediction bytes identical; donor/seed/config intact | `test_transient_after_seal_recovers_with_identical_prediction_and_same_donor` |
| Cooldown without real waiting | `test_cooldown_is_enforced_by_the_injected_clock_without_sleeping` |
| Exhaustion → explicit terminal with cause and required action | `test_transient_budget_exhaustion_is_an_explicit_terminal` |
| No retry for digest / causality / version / unknown | `test_permanent_and_unknown_causes_are_never_retried` |
| Adoption of a completed update without a second fit | `test_completed_update_is_adopted_on_retry_without_a_second_fit` |
| Handoff to the independent family during a cooldown | `test_other_family_runs_during_a_cooldown_and_never_skips_its_own_dependency` |
| Ledger resumption after a process restart | `test_ledger_resumes_attempt_numbering_after_a_simulated_restart` |
| Historical FAILED is never deleted; successor stays blocked | `test_historical_failed_terminal_is_never_deleted_or_retried` |
| Status states and no finite ETA under a block | `test_status_distinguishes_states_and_never_promises_an_eta_under_a_block` |
| Scientific files byte-identical to the base | `test_scientific_files_are_byte_identical_to_the_base` |

## Implementation notes (candidate)

- `tools/i6dmo/prequential_recovery.py`: `classify`, the ledger (`begin_attempt`, `record_failure`,
  `finish_attempt`), `choose`/`classify_pending` (scheduling during cooldowns) and `status`.
- `tools/i6dmo/prequential_test.py`: `GpuBackend.load` splits `RUNTIME_VERSION_MISMATCH`
  (`ValueError`, PERMANENT) from `GPU_NOT_VISIBLE` (`TransientProcessError`); `run` takes an
  injected `clock`, `backend_factory` and resource `gate`, consults the ledger and writes
  `RESULT.json` only for completion, PERMANENT/UNKNOWN causes or an exhausted transient budget;
  `enrich_status` is patched into the tick seam by `context()`.
- `tools/i6dmo/weekly_test.py`: `enrich_status` seam (no-op by default) and closure coverage.
- `PROGRESS.json` `RUNNING` is a last-written marker, not a liveness proof: a process that died
  mid-attempt leaves `RUNNING` until the next admitted process writes a new state. The tick's
  own liveness checks are unchanged by this candidate.
- A prior `FAILED` `RESULT.json` (written by the deployed code before this candidate) stays
  terminal; this candidate never removes it and its successor stays `BLOCKED_DEPENDENCY`.
