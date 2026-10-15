# Follow-up review of 9f8bd67a: R2, R3 and R4 (Retsu, 2026-10-15)

| Item | Value |
|---|---|
| Subject | `codex/i6dmo-matched-followup-20261015` at `9f8bd67a19aeea9303ef85e4c317e3813776276f`: R2 dispatch fairness, R3 strict scientific defaults, R4 launcher-admission attempt accounting. None of the three is deployed. |
| Review branch | `retsu/i6dmo-followup-review-20261015` |
| Added by this review | `tests/i6dmo/test_retsu_followup_adversarial.py`, the orchestration-only child `tests/i6dmo/retsu_adversarial_child.py`, and this record |
| Work performed | CPU only. No fit, no TEST, no deploy, no systemd. Every invocation went through `crispdm-run`. |
| Left untouched | The closed campaign `i6dmo-6004b711281c` |

**Verdict: ACCEPT_WITH_CONDITIONS.**

- R3 is VERIFIED.
- R2 is VERIFIED for its stated scope, which is `INFRA_PREFIT` refusals. A gap remains for refusals by the launcher.
- R4 does not erase the budget of any real failure that the current children can produce.
- R4's rule nonetheless trusts the exit code 75 alone. An adversarial child that started work and then exited 75 is re-run without bound, and its written terminal is discarded (REFUTED as stated below).
- The suite contains one test that is not hermetic.

Conditions for the next release:

- **C1.** In `worker._finish`, treat exit 75 as a pre-execution refusal only when the attempt folder holds no child output. That means no `out/TERMINAL.json`, no progress file, and empty child stdout. With any of those present, the attempt must count against the budget.
- **C2.** `test_launcher_admission_budget.py::test_no_tensorflow_loaded` fails under the standard invocation, because `tests/i6dmo/conftest.py` imports TensorFlow. Either make the test hermetic or run it separately with `--noconftest` as its docstring requires.
- **C3 (recommended).** Extend R2's rotation to persistent launcher refusals, or type persistent `LAUNCHER_ADMISSION_REFUSED` in status after a bounded threshold.

## 1. Test replay (`out/followup_suite_9f8bd67a.txt`)

Every command was:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 600s -n <name> -- python -m pytest -q -p no:cacheprovider <files>
```

The exceptions are noted in the table. Every count is at revision `9f8bd67a`. Counts from different revisions are never merged.

| Piece | Files | Result |
|---|---|---|
| Focused | `test_launcher_admission_budget.py`, `test_matched_dispatch_fairness.py`, `test_matched_scientific_defaults.py`, `test_matched_fit_path.py`, `test_ops_matched_runner.py` | 142 passed, **1 failed**: `test_launcher_admission_budget.py::test_no_tensorflow_loaded` |
| R4 file alone, `--noconftest` (2G cap) | `test_launcher_admission_budget.py` | 12 passed |
| R4 file alone, standard invocation (2G cap) | the same | 11 passed, 1 failed (the same test) |
| p1, worker integration | `ops_automation_repair`, `ops_matched_real_chain`, `completed_fit_recovery`, `ops_dmo07_units`, `ops_dmo17_host_lease`, `ops_dmo11_identity` | 41 passed |
| p2, report integration | `market_time_reporting`, `ops_dmo08_09_reporting`, `ops_dmo12_outbox`, `ops_dmo13_test_closed`, `ops_storage_adapter`, `dmo_final`, `strategy_pair_unit`, `strategy_execution_policy` | 82 passed |
| p3 | `ops_dmo19_status`, `ops_audit_lane4`, `ops_market_time_pair`, `ops_deploy_runner`, `ops_pair_long_wait`, `ops_dmo20_cadence` | 48 passed |
| p5 | `audit_warehouse_contract`, `data_dmo`, `data_long`, `dmo_design`, `dmo_hourly`, `dmo_long`, `dmo_market_time`, `market_bundle`, `market_grid`, `matched_population`, `strategy_adapter`, `strategy_final_backtest`, `strategy_instrument_config` | 235 passed |
| p6 | `dmo10_isolation`, `dmo_model`, `dmo_train`, `market_time_fit_path` | 57 passed |

**Full suite at `9f8bd67a`:** the focused piece plus p1, p2, p3, p5 and p6 cover all 42 `tests/i6dmo` files. The result is 605 passed and 1 failed, and the failure is the non-hermetic TensorFlow test (C2).

No governed invocation was refused before launch. Every piece finished within the 600 s cap.

**Adversarial file of this review** (run with the `9f8bd67a` tools): 32 passed and 3 xfailed. Each xfail is a minimal reproduction, and each was re-run with `--runxfail` to confirm it fails on an assertion and not on a setup error.

## 2. R4 adversarial (`test_retsu_followup_adversarial.py`)

These tests drive a real queue and a real worker (claim, governed child, `_finish`). The child is an orchestration fixture.

| Case | Verdict | Evidence |
|---|---|---|
| A genuine crash after start (exit 1, with stdout and `out/` written) | VERIFIED | `FAILED EXIT_1`, terminal, never retried; 1 invocation; attempt row `(1, FAILED, EXIT_1)` |
| A timeout (exit 124) | VERIFIED | `FAILED_TIMEOUT`, terminal; 1 invocation |
| MAX + 2 genuine refusals (exit 75) followed by a crash | VERIFIED | Refusals spend no budget, indices are contiguous and preserved, and the crash still ends the unit `FAILED` |
| Mixed sequences on the queue: refusals interleaved with `CHILD_VANISHED`, `SUPERVISOR_LOST` and `LEASE_EXPIRED` (expired through the clock), or with `LEASE_EXPIRED`, `NOT_LAUNCHED` and `LAUNCH_FAILED` | VERIFIED | Exactly 3 real interruptions exhaust the budget (`ATTEMPTS_EXHAUSTED`). Rows 1–7 are preserved with 3 refusals among them. A later claim cannot reopen the unit |
| A child that STARTED, wrote a COMPLETE `out/TERMINAL.json` and stdout, then exited 75 | **REFUTED** | `test_r4_started_then_75_must_spend_budget_and_terminate`, xfail(strict): 6 invocations in 6 rounds against a maximum of 3, never terminal, every row `LAUNCHER_ADMISSION_REFUSED`. The terminal written by the child is ignored, because `_finish` checks `rc == 75` before it looks at `out/TERMINAL.json`. The observation test confirms the terminal file exists |
| stderr says "REFUSED nothing was started", but stdout and `out/` exist; exit 75 | **REFUTED** | `test_r4_claims_not_started_with_outputs_must_spend_budget`, xfail(strict): 6 invocations, unbounded. The exit code is the only evidence consulted |
| A persistent refusal shown in status | Observation (NON_BLOCKING) | It stays an untyped `WAIT_RESOURCE` without bound. There is no `INFRA_PREFIT_BLOCKED`-style typing |

### How reachable the R4 refutation is

In the shipped stack, exit code 75 comes from `crispdm-run` in two ways:

- **A pre-launch refusal.** Admission is refused, or the travel hold is on.
- **An arming failure after launch.** The child has been started and is stopped with TERM within about 3 s.

`unit_runner` passes a trainer's exit code through unchanged when it is not 0, 2, 3 or 4. The trainer itself exits only 0, 2, 3 or 4, or 1 on a crash. The report and comparison hooks exit 0 or raise.

So no current child does real work and then exits 75. That is why the refutation is NON_BLOCKING today. But R4's rule does not establish the "proven pre-execution" property it claims. A future child or wrapper that exits 75 after real work would re-execute without bound and discard a completed result (C1).

Minimal reproduction:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 600s -n i6dmo-fu-adv-runxfail -- \
  python -m pytest -q -p no:cacheprovider --runxfail tests/i6dmo/test_retsu_followup_adversarial.py \
  -k "started_then_75_must or claims_not_started"
```

Result: `assert 6 <= 3` in both tests, exit 1.

## 3. R2 adversarial

| Case | Verdict | Evidence |
|---|---|---|
| A real worker where mg1 gets a pre-fit `NO_GPU_VISIBLE` and mg2m is healthy | VERIFIED | Round 1 refuses mg1. Round 2 picks mg2m, which reaches `COMPLETE`. mg1 was invoked once |
| Both arms persistently refused | VERIFIED | The arms alternate: 6 rounds, with a difference of at most 1 |
| A tie | VERIFIED | Musashi's test, plus the alternation above (the declared preference wins a tie) |
| A post-training failure carrying an infrastructure code (`NO_GPU_VISIBLE` with `fit_seconds` > 0) | VERIFIED | Terminal `FAILED TYPED:NO_GPU_VISIBLE`, one invocation, not counted as a refusal. The other arm completes |
| A launcher refusal (75) followed by success | VERIFIED | Harmless; the declared order is kept |
| One arm's launch persistently refused by the launcher (exit 75, for example when its larger class cap does not fit) | **REFUTED for that case (outside R2's stated INFRA_PREFIT scope; NON_BLOCKING)** | `test_r2_persistent_launcher_refusal_of_one_arm_must_not_starve_the_other`, xfail(strict): mg1 is refused 4 times out of 4 and mg2m is never invoked (`QUEUED`). Rotation counts only `INFRA_PREFIT:` causes. Worker-side admission waits likewise do not rotate; that predates this patch (C3) |

## 4. R3 adversarial

| Case | Verdict | Evidence |
|---|---|---|
| `decide_matched(require_scientific=...)` with `"false"`, `"False"`, `"0"`, `0.0`, `np.False_`, `[]` or `{}` | VERIFIED | Strict in every case: `NON_SCIENTIFIC_FIT` |
| Musashi's cases (missing, `True`, `None`, `0`, `""`) | VERIFIED | Covered by his file, which passed in the replay |
| `scientific_fit` set to `"true"`, `1`, `"gpu"` or `None` | VERIFIED | Not treated as scientific; the decision refuses |
| The literal `False`, and JSON `false` | VERIFIED | The engineering opt-out still works, in both the decision and the real report |
| The real report (`reporting.run_unit` → `run_matched_unit` on the synthetic common population) with `require_scientific_fit` set to `"false"`, `"False"`, `0.0`, `np.False_`, `None` or `0` | VERIFIED | Strict in every case: `NON_SCIENTIFIC_FIT`, no `METRICS` |

The campaign path is unchanged: the worker writes a real boolean, and `matched_runner` passes `not engineering_cpu_fixture`.

## 5. Housekeeping

- Deleted `/var/tmp/retsu-review-scratch-20261015` (1.3 GiB).
- Deleted the worker-side replay scratch `/var/tmp/retsu-i6dmo-replay-20261015` (736 MiB). Its committed evidence stays in `review/out/p7_gpu_replay.json`.
- This pass's scratch is `/var/tmp/retsu-followup-scratch` (logs only, small). pytest's `tmp_path` used `/tmp`. Nothing was created under `/home`.
