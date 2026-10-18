# Audit of the prequential weekly TEST (Retsu, 2026-10-17)

Target: `codex/i6dmo-prequential-test-20261017` at `811a7544`. Read-only audit, owner-mandated
method: code, synthetic CPU tests and EXISTING receipts only. No training, no change to the
campaign, no deploy, no restart, no monitoring. The live root and its SQLite were not opened; one
`STATUS.json` read was taken; receipts were read from the artifacts copy and the prepared data
root, and `PREDICTED.npz` was only hashed. No TEST row was loaded beyond what the receipts
already expose (per-head MAE/naive/n in `SCORE.json`/`RESULT.json`).

**Verdict: VERIFIED on every mandated item (1–6 and the edge cases). No defect against the
canonical procedure. Three non-blocking observations (O1–O3) and two coverage gaps.**

Evidence in this branch: `tests/i6dmo/test_retsu_prequential_audit.py` (11 probes, PS01–PS09 and
the edge cases; CPU, fake backends) and `audit/RECEIPT_CHAIN_811a7544.json` (the hash chain of
the 13 cells terminal at audit time, paths redacted). Commands and exit codes at the end.

Deployed code identity: the live spec records `code_commit fa47d2b2`; `git diff --stat
fa47d2b2 811a7544` touches only `docs/i6n/weekly_test/*`, the interview work plan and one
line of `tests/i6dmo/test_prequential_test.py`. `tools/` and `predictor_plugins/` are identical,
so the audited code is the running code.

TEST authorization: commit `353bb7e6` ("owner-authorized TEST 2025"),
`PROJECT_METHOD_STATE.json` `initiative: owner_authorized_weekly_test_replication`, and
`PLAN.md` ("binding owner specification"). Recorded: VERIFIED.

## Items

| # | Owner item | Verdict | Where (file:line) and evidence |
|---|---|---|---|
| 1 | First inference loads the weights AND scalers of the last useful VALIDATION checkpoint; short 7 heads, long 6; no random init, no training before the first prediction | VERIFIED | `prequential_test.py:188` picks the member with the latest `train_end`; `:189-192` authenticates its files and `ARCHIVE.json` (`VERIFIED`, archive hash); `:196-199` the donor body carries `model` from the member's `CONFIG.json`, `scales` from its `SCALES.json` and `weights_sha256` from the archive proof; `:200` refuses `INITIAL_DONOR_FROM_TEST`. `GpuBackend.load :116` `keras.models.load_model` (no build), `:117` `DONOR_WEIGHTS_CHANGED`. `execute_week :44-64` predicts before any reader of labels. Receipts: initial donors `member 913a485c…` (short, heads `[2,3,4,5,6,12,18]`) and `4b3f8112…` (long, `[18,24,36,48,60,72]`), `available_by 2024-12-27T22:00Z`, archive bytes re-hashed and files authenticated; first COMPLETE cells `short__WEEKLY-2025-01-05` / `long__WEEKLY-2025-01-05` have `inference_donor` = the initial id and `UPDATE.json.initial_weights_sha256` = the initial `weights_sha256` (`08ef4d5f…`, `4d9290d3…`). Probes `test_ps01_*` (2). |
| 2 | Prediction sealed (saved + hashed) before the week's outcomes are consulted | VERIFIED | `:44` reader `PREDICT` (keys `PREDICT_KEYS :24`, no target/mask; `:28` refuses extra keys); `:55-64` fsync + atomic rename + `PREDICTION.json{prediction_sha256}`; `:65` the `OBSERVED` reader runs only afterwards; `:68-70` `SCORE.json` binds `forecast_sha256`; `:79` `UPDATE_CHANGED_PREDICTION`. Receipts: every COMPLETE cell has `predicted_npz_hash_matches`, `score_binds_prediction` and `prediction.created_at < score.created_at < update/STARTED.at` (13/13 consistent). Probes `test_ps02_*`, `test_ps09_crash_*`. |
| 3 | Refit inherits weights (no reinit); TRAIN = four calendar years before the start of the evaluated week (exact weekly cut); M = (loss_TRAIN + loss_week)/2, patience 10, min_delta 1e-8, max 200; fresh optimizer per fit with original params | VERIFIED | `:141` `bundle=self.load(donor)` (saved weights); `:149-151` `train.fit_dmo` then `WARM_START_WEIGHTS_CHANGED` if `initial_weights_sha256 != donor.weights_sha256`. Window: `continuation_data.py:88-92` `cutoff = batch[0].market_open`, `train_start = cutoff − 4y`, `validation_end = market_close`; `prequential_test.py:92-101` re-checks four calendar years, origins in `[start, cutoff)`, labels `< cutoff`, observed week finished. Formula: `train.py:1327-1341` `MeanMonitorStopper` `0.5*(fit+inner)`, patience/min_delta/max from `TRAINING` (`resolution_batch.py:20-21`: 256/.001/.0001/10/1e-8/200); `train.py:1379-1380` a new `AdamW(lr, weight_decay)` is created inside every `fit_dmo` call (and `:1377` the seed is reset). `train.py` is byte-frozen against `658a7733` by `test_trainer_and_models_have_not_changed`. Receipts: all cells `period = (open−4y, open, close)` (e.g. `2021-01-05T22:00 → 2025-01-05T22:00 → 2025-01-10T22:00`), `early_stop {rule i6dmo.mean_monitor_stop.v1, monitor 0.5*(fit_loss+inner_loss), patience 10, min_delta 1e-08, max 200}`, `split_roles {fit: FOUR_YEARS_BEFORE_OBSERVED_WEEK, inner: ALREADY_OBSERVED_WEEK, use: NEXT_WEEK_ONLY}`, stop `patience`. Probe `test_ps05_*`. |
| 4 | The refit serves only the following week; later labels never modify earlier predictions | VERIFIED | `:237` the donor of cell n+1 is `results[predecessor].next_checkpoint`; `:165` its `available_by = validation_end` of week n and `:36-38` `DONOR_NOT_YET_AVAILABLE` if any origin precedes it; `:47-53` an existing seal is reused, never recomputed. Receipts: each `inference_donor` equals the predecessor's `next_checkpoint.id`, and each donor's `available_by` ≤ the week's `train_end`. Probe `test_ps04_*`: perturbing week-2 targets leaves every week-1 byte (seal, score, update receipt, next checkpoint) identical; only week-2's score and refit move. |
| 5 | Scalers fitted only on that refit's TRAIN; inference with the checkpoint's own scalers; coherent units | VERIFIED | `weekly_test_data.py:92-95` scaler mean/std over the union of FIT slot bars with `SCALER_NOT_TRAIN_ONLY` (any bar ≥ `train_end` refuses); `:117-118` `naive_mae` from FIT; `prequential_test.py:142-145` the refit transforms FIT and the observed week with `record['scales']` (that period's TRAIN scaler) and `:165` seals them into `next_checkpoint.scales`; `:123-125` inference uses `donor['scales']` for both the z-score and `to_price` (`train.py:715-731`: y = Δprice/naive, inverse = origin + pred·naive, so prediction returns to original USD/EUR); `:68` `measured_metrics` (`continuation.py:173-194`) computes MAE and naive on the same mask and original prices. Receipts: every `next_checkpoint.scales.scales_sha256` equals its cell's `PERIOD.scales.scales_sha256`, and every `inference_scaler_sha256` equals the donor's scaler. Probes `test_ps07_*` (2): a swapped scaler changes the model input and the checkpoint id, so resume refuses `SEALED_FORECAST_IDENTITY`. |
| 6 | Resumption never replaces a sealed prediction; the refit resumes from its previous donor; old results are not adopted | VERIFIED | `:47-53` seal reuse with identity/hash check; `:135-138` an `UPDATE.json` is reused only if `donor_id` matches (`UPDATE_DONOR_CHANGED`) and its files authenticate; `:140` `STARTED.json` then a fresh refit from the donor; `weekly18.py:282-292` `terminal` demands this batch's `batch_id`, the cell and a valid digest, so no older campaign result is adopted; `:176-179` `prepare` consumes only the parent's DATA (`FIT/TEST` arrays), never its models. Probes `test_ps09_*` (2). |

Edge cases from the procedure:

| Edge | Verdict | Evidence |
|---|---|---|
| Week without valid pairs → no fictitious MAE, no early stopping, checkpoint carried forward | VERIFIED | `:239-240` `SKIPPED_NO_EVAL_ROWS` with `next_checkpoint=donor`, no metrics; `:99` `NO_OBSERVED_MONITOR_PAIRS` refuses a refit on an empty week; `continuation.py:179-182` a head with n=0 reports `NO_VALID_TARGET_IN_PERIOD` with `mae None`. Receipts: both `WEEKLY-2024-12-29` cells SKIPPED carrying the initial donor ids; `long__WEEKLY-2025-01-05` head 72 n=0 (no MAE) while the other heads scored. Probes `test_edge_*` (2). The code label is `SKIPPED_NO_EVAL_ROWS`, not the doc's literal `SIN_PARES` (cosmetic, O3). |
| Initial checkpoint finished its VALIDATION week before any scoring TEST origin; never a checkpoint trained on TEST | VERIFIED | `:200` `INITIAL_DONOR_FROM_TEST` (`available_by ≤ 2025-01-01`); `:36-38` availability vs the first origins. Receipts: initial `available_by 2024-12-27T22:00Z`; first scored origins are in the week opening `2025-01-05T22:00Z`. |

PS01–PS10 against the code at `811a7544`: `git grep PS0` hits only docs. Coverage by the
committed tests (`tests/i6dmo/test_prequential_test.py`): PS02, PS03 (its "future targets" test
perturbs the current week), PS05, PS07 (inference side), PS08, PS09 (resume identity), plus the
frozen-trainer check. Missing as explicit tests: PS01, PS04, PS06, PS07 (refit side), PS10. This
branch adds PS01, PS04, PS07 (both sides), PS09 (donor-receipt guard) and the two edge probes.
PS06 and PS10 remain code/receipt-verified only (see coverage).

## Observations (non-blocking)

- **O1 — a typed failure is terminal and blocks the chain.** `run :258-261` writes a `FAILED`
  `RESULT.json` for any exception after the seal (e.g. a transient `RETAINED_GPU_RUNTIME_REQUIRED`
  or a lost GPU); `weekly18.terminal` treats it as terminal, so `:231-233` reports
  `BLOCKED_DEPENDENCY` and nothing retries. The sealed prediction is retained (correct), but the
  procedure's "resume the refit from its previous donor" only happens for a hard crash that left no
  `RESULT.json`. Probe `test_edge_sin_pares_*` shows the block. Operational risk, not a violation;
  a manual, audited removal of the failed `RESULT.json` would be the recovery path. Live status
  at audit time: `failed {}`.
- **O2 — boundary week coverage.** `WEEKLY-2024-12-29` (TEST origins 2 and 3 January) is
  `SKIPPED_NO_EVAL_ROWS` for both families: its origins are never predicted, so the annual pooled
  metrics will not include them. Consistent with the procedure (no pairs after the short family's
  common-window intersection), but it should be stated in the closure.
- **O3 — naming.** The procedure says `SIN_PARES`; the code and receipts say
  `SKIPPED_NO_EVAL_ROWS` and, per head, `NO_VALID_TARGET_IN_PERIOD`.

Out of scope but noted: `run :243-246` reads `MemAvailable` directly for the 7 GiB gate; the
systemd unit itself is launched by `crispdm_run` with `MemoryMax=6G`.

## Coverage not verified

- PS06 synthetically: `GpuBackend.update` requires the retained GPU runtime (`:110`); verified by
  the guard at `:151` and by the receipts (`initial_weights_sha256` == donor weights on all 11
  COMPLETE cells).
- PS10 end to end: `weekly_test.tick :133-196` needs ssh/rsync/warehouse. Verified by code and by
  the one status read: `publication {expected 11, delivered 11, pending 0, mismatch 0}`,
  `state RUNNING`, `failed {}`; mirrors and `CLOSE.json` only occur at annual closure, not reached.
- GPU-side checks (`RETAINED_GPU_RUNTIME_REQUIRED :110`, `SAVED_CHECKPOINT_RELOAD :160`) by code
  and receipts (`tensorflow 2.21.0`, `keras 3.13.2`, `restored_weights_sha256` present) only.
- Cells beyond the 13 terminal at audit time (campaign RUNNING; `long__WEEKLY-2025-02-02` was in
  progress).

## Privacy

`git diff --name-only 0a2f26e9 811a7544` (17 files) scanned for the coordinator host name, ssh
aliases, GPU UUIDs, IPs and home paths: the only hits are in
`docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md` (coordinator
host name and the secondary alias), both pre-existing since `128e7a1c` and not in the lines added
by this range. The `weekly_test` docs, tools and tests are clean. This branch's additions contain
no path, host, alias or UUID (the evidence JSON has 0 absolute paths).

## Commands and exit codes

| Step | Command | Exit |
|---|---|---|
| committed tests | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-preq-audit-tests -- python -m pytest -q -p no:cacheprovider tests/i6dmo/test_prequential_test.py tests/i6dmo/test_weekly_test.py tests/i6dmo/test_weekly_test_data.py tests/i6dmo/test_weekly_test_source.py` | 0 (24 passed) |
| audit probes | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-preq-audit-probes -- python -m pytest -q -p no:cacheprovider -rf tests/i6dmo/test_retsu_prequential_audit.py` | 0 (11 passed) |
| receipt chain | read-only script over `<artifacts>/results/*/{RESULT,PREDICTION,SCORE,update/*}.json` and `<data_root>/{PREPARED,<cell>/PERIOD}.json`; output `audit/RECEIPT_CHAIN_811a7544.json` (`all_complete_cells_consistent: true`) | 0 |
| status | one read of `<root>/STATUS.json` | 0 |

Receipt hashes (prefixes; full values in the evidence JSON): prepared `population_digest` matches
the spec; initial donors `3439e386…` (short) / `e7ff10b7…` (long); first checkpoints
`a7beea62…` (short, after `2025-01-05`) / `153568bf…` (long).
