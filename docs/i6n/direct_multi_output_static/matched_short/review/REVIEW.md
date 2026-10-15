# Independent review: I6-DMO matched short control (Retsu, 2026-10-15)

Candidate: `satoshi/i6dmo-matched-short-20261015` at `4a77bea8`, tree `bd402e94340a33d48870b250337a9de1cd07c7c6`.
Review branch: `retsu/i6dmo-matched-short-review-20261015`, which adds only this directory.

**Verdict: ACCEPT_WITH_CONDITIONS.** The verdict lists four conditions. Condition C1 is a hard
precondition for deployment, because nothing can run today on either GPU worker.

No scientific fit ran. Nothing read TEST: the retained input ends at `2024-12-31T22:00Z`. Nothing
was deployed, no timer or service was touched, and no live SQLite was opened. The closed campaigns
were not modified. They were read in only three places: the deployment spec, to resolve paths; the
effective-config pins; and one `grep` of the closed receipts' `tensorflow_version`.

Every probe below is in this directory. Its output, with paths, host names and UUIDs redacted, is
in `out/`.

## Conditions

- **C1 (hard precondition for deployment).** Each GPU role must satisfy all of the following before
  the timers are enabled:
  - its TensorFlow and Keras are the sealed 2.18.0 / 3.9.0;
  - its remote workdir is at the runtime pin;
  - it declares `experiment_storage_root`, with the frozen population replicated under it;
  - `matched-short preflight` reports PASS on both GPU roles immediately before the timers are
    enabled.

  Today both GPU roles run TF 2.21.0 / Keras 3.13.2, and no environment on either host has
  2.18.0 / 3.9.0. Preflight FAILs, and worker admission would hold both fits in `WAIT_RESOURCE`
  (D4).
- **C2.** The real `init` must not use `--prior-root` or `--entrypoints` until D1 is fixed. D1 is
  a BLOCKER before any reuse of a matched R0 receipt, and the R2 plan reuses one.
- **C3.** No `I6DMO_TEST_*` variable may appear in any role `environment` or in the service
  environment. The alternative is to fix D2, so that the report and the decision refuse
  non-scientific or CPU receipts.
- **C4 (recommended).** Fix D3, the TOCTOU between the verified load and the accessor.

## Per-item verdicts

| Item | Verdict | Evidence |
|---|---|---|
| SC01 shared exact masks; tampering refused | VERIFIED | `p1_tamper` (12 tampers × load / trainer CLI / report entry, all refused, nothing fitted); `p3_independent` |
| SC02 last input is t; perturbing > t | VERIFIED | `p2_perturb` gt_t (real input, T1 = 2024-07-10T13Z): inputs byte-identical for every t ≤ T1, both arms, every split; 32 labels with t+h > T1 changed, 14,261 with t+h ≤ T1 unchanged |
| SC03 weekend/DST, missing bars | VERIFIED | `p3`: every common row's slots equal the grid computed independently from `CALENDAR.json`; weekend gap 48 h at 2024-03-10 versus 49 h, and 50 h at 2024-11-03 versus 49 h; the unobserved expected 2024-10-27T22Z bar gives 24 typed `INPUT_SLOT_NOT_OBSERVED` exclusions; 3,297 slot-not-observed exclusions checked, 0 contradicted |
| SC04 FIT-only shared scaler and naive | VERIFIED | `p2` inner_dev (7,187 INNER+DEV bars perturbed): scaler mean, std and row digest bit-identical, naive scales identical, FIT arm arrays bitwise equal; `p3` recomputes mean, std and naive exactly (max diff 0.0); `p7` gives identical `scales_sha256` in both arms |
| SC05 same rows, mask and units; manual replay | VERIFIED | `p4`: real report hook on the real population; per-head MAE, naive and relative error recomputed here from the saved predictions plus `INPUT.npz`, max abs diff 0.0; naive identical across arms |
| SC06 consumes the frozen population; closed MG2 refused | VERIFIED | `p1`; `p4` (`MATCHED_IDENTITY_REQUIRED`, `NOT_A_MATCHED_FIT` ×3, `ROW_ID_MISMATCH` for full-population-shaped predictions); `p8` b |
| SC07 2304 delta; unreported change refused | VERIFIED | `p9`: counts taken independently from Keras give 7186 vs 4882 and 4902 vs 2598 heads; the only config difference is the slot count; a changed head or branch is refused (`FIXED_VALUE`); `p4`: decide/report refuse a changed capacity (`CAPACITY_UNREPORTED_CHANGE`) |
| SC08 negative and positive auto-close; no OHLC | VERIFIED | `p6`: `MG1_SELECTED` and `INCOMPLETE_COMPARISON` close automatically; snapshots are 0444 with matching digests; mirrors are bound to CLOSE and timers. The automation E2E re-run at `4a77bea8` gives PASS with `PAIR_NOT_ELIGIBLE` and no strategy pins (the committed receipt was made at `3ba826ae`) |
| SC09 restart, duplicate claim, outbox retry | VERIFIED | `p6` B: worker SIGKILLed mid-fit; a duplicate claim gets `HELD`; restart re-attaches to `COMPLETE`; 3 failed POSTs are retried and then CLOSE; post-CLOSE passes are inert; exactly one fit per arm. "Unavailable host" is covered by the lane test only |
| SC10 no TEST; changed donor cannot reuse | **REFUTED (partial)** | No TEST: VERIFIED (`p5`: plan units, rendered units and `enumerate_test_units` refused). Donor: REFUTED (`p6` A, D1) |
| SC11 bounded GPU replay | VERIFIED under the worker's actual TF 2.21.0; UNVERIFIED under the sealed TF 2.18.0 (not installed on any GPU host) | `p7_gpu_replay.json` |
| SC12 status never RUNNING indefinitely | VERIFIED | `p6` C: `STALE` after lease expiry; `SERVICE_SILENT` → `SERVICE_FAILED` at 1801 s; a failed arm shows `FAILED`; `p5` status at every level |
| engineering_cpu_fixture | **KEEP** | `p8` a–f |
| Seam | VERIFIED, with D5 | `p8` e (the real `unit_runner` passes the three flags to the real `train` CLI, which parses them; with no GPU the result is `NO_GPU_VISIBLE` and nothing is loaded); `p4`; `p6` |
| Recovery repairs | VERIFIED | `p6`, `p10`, E2E re-run, lane suite |
| FRED addendum | VERIFIED | File hashes match. 1176 + 72 = 1248 covered; 269 + 68 + 2 = 339 not covered. Friday lag ≥ 3 accounts for 317 of the 339 |
| Runbook commands | VERIFIED (D4 makes preflight FAIL) | `p5` |
| Privacy | VERIFIED for the 39 lane files | Three hits predate the lanes (see below) |

## Defects

**D1 — BLOCKER before any prior-root reuse; NON_BLOCKING for the first contrast under C2.**

- Adoption matches on `unit identity_sha256` only. That identity covers design + adapter, data
  pins, code commit, seed and upstream.
- It does not cover the fit procedure: `entrypoints`, `fit_device`, `engineering_cpu_fixture`,
  overrides, and the receipt's `scientific` flag.
- `matched_runner._check_donor` checks design seals and pin digests only.

Reproduction (`p6` part A):

1. Init root A on the REAL sealed designs and REAL pins, with the fake `ops_matched_fake`
   fit/report entrypoints, and drive it to CLOSE.
2. `_check_donor(A)` returns ACCEPTED.
3. Init root B with the default real entrypoints and `prior_roots=[A]`.
4. Workers on B ADOPT `mg1.fit`, `mg2m.fit`, both reports and the comparison (`MG1_SELECTED`).
   B makes 0 child invocations and creates no attempt directory.

`p8` c adds that unit identities are equal between a CPU-fixture campaign and a GPU campaign on
the same designs.

**D2 — NON_BLOCKING; defence in depth.**

- `reporting.run_matched_unit` records `scientific_fit` but never refuses
  `scientific is not True` or `device != "gpu"`.
- `decide_matched` checks neither.
- `train.py` stamps `scientific: true` on a `--device cpu` matched run of the REAL design: `p8` g
  shows the receipt `scientific: true, device: cpu`; it stopped only because the digest was wrong.

`p4` observations: receipts with `scientific: false, device: cpu`, or with neither field, report
`COMPLETE` and decide `TIE_RETAIN_CHEAPER`. Role `environment` variables are forwarded to children,
so a stray `I6DMO_TEST_TRAINING_OVERRIDES` would publish a decision built on 2-epoch fits.

**D3 — NON_BLOCKING; low.** `common_population.arm_tensors` takes the `source_dir` path when it
reads the arm arrays back from disk:

- It re-reads `ARM_*.npz`.
- It compares the file against the digest *strings* stored in `POPULATION.json`, not against the
  verified load.

`p8` h: after a verified load, a tampered `ARM_mg1_FIT.npz` with its stored digest string updated
is ACCEPTED; a fresh verified load refuses it with `POPULATION_DIGEST_MISMATCH`. The window is
in-process, between load and accessor. The FIT path is not replayed against `INPUT.npz`; the
DEVELOPMENT path is.

**D4 — deployment BLOCKER; environment, not code.**

- Both GPU roles run TF 2.21.0 / Keras 3.13.2, in their role python and in every candidate env.
- The remote workdirs' probe reports no runtime, which gives `RUNTIME_UNVERIFIED`.
- Neither GPU role declares `experiment_storage_root`.
- `PRECHECK.matched` is `FAIL` with blockers `{preferred, secondary} × {pin:common_population,
  runtime_pin}` (`p5`).

Observation: the closed MG2/MG12 receipts record TF 2.21.0 under the same 2.18.0 seal. They were
not re-graded.

**D5 — NON_BLOCKING; risk.** A remote fit child's technical failures are FAILED terminals:

- `POPULATION_PIN_MISMATCH` and `INPUT_PIN_MISMATCH` in `unit_runner`;
- `NO_GPU_VISIBLE` in `train`.

Such a terminal makes the arm FAILED, the comparison returns `INCOMPLETE_COMPARISON`, and the
campaign CLOSES automatically (`p6` C, `p8` e). `matched_unit_problem` skips file checks for ssh
roles. This follows the plan, but an infrastructure misconfiguration burns the single-seed
contrast, so C1 has to be checked right before enabling the timers.

**D6 — NON_BLOCKING; documentation.** The `status --arm` help says `(mg1|mg2)`. `--arm mg2`
returns `UNKNOWN_ARM`; the arm id is `mg2m` (`p5`).

**D7 — NON_BLOCKING.** `tools.i6dmo.replay_monitor` has no matched / common-population mode;
`p7` needed its own driver.

## GPU replay numbers (`out/p7_gpu_replay.json`)

The run used the secondary role's RTX 4090 Laptop GPU, pinned by the role's UUID; the UUID is
recorded privately.

- **Runtime:** TF 2.21.0 / Keras 3.13.2, which is NOT the sealed pin.
- **Code:** exact tree `4a77bea8`; the staged manifest is identical.
- **Data:** the real sealed designs and real population. Slice FIT 4096 / INNER 512 / DEV 512,
  2 epochs.
- **Memory:** the measurement run under a 4 GiB cap peaked at 2,250,780,672 B. The evidence cap
  was ceil(1.25 × peak) = 2,813,475,840 B. The fit peaked at 2,252,709,888 B and the replay at
  1,995,067,392 B.
- **Same process:** reloaded predictions are bitwise equal (max abs diff 0.0); monitor replay is
  EXACT; self-check PASS.
- **Cross-process GPU** (distinct process uuid): predictions are bitwise equal; max abs and max
  relative diff are 0.0, within the `i6dmo.monitor_numeric_equivalence.v2` floor. The monitor
  replay is EXACT in context `gpu_cross_process` (|Δ| = 0.0). The weights digests are equal.
- **Inputs:** CPU (coordinator) and GPU host are byte-identical for both arms. Weights were also
  identical between the measurement run and the evidence run.

## Not verified

- A GPU replay under the sealed TF 2.18.0 / Keras 3.9.0 runtime, which is not installed on any GPU
  worker.
- A CPU engineering fit on the real population. It was not run, because it is not allowed on real
  data.
- `deploy --no-start` and `deploy`, which call systemd.
- The storage gate, because it writes a probe file into the real storage root.
- An unavailable host, beyond the lane test.
- The strategy qualified path, beyond the lane tests and reading the code.
- A negative `EXPECTED_WEEKS_MISMATCH` (unreachable without breaking digests).
- The production warehouse and the two real physical mirrors.

## Privacy

The 39 files changed between `0245c1f8` and `4a77bea8` contain no real host name, ssh alias, GPU
UUID, IP or home path. The only hits are the placeholders `host-preferred` and `host-secondary`
in the tests.

Three pre-existing hits, present at `0245c1f8` and not the lanes' work:

| File | Hit | Introduced at |
|---|---|---|
| `docs/i6n/direct_multi_output_static/DATA_RETURN.md` | the preferred role's ssh alias and host name | `a1789633` |
| `tools/i6dmo/producer_recovery.v1.json` | the same | `58d489fe` |
| `tests/i6dmo/test_ops_storage_adapter.py` | the coordinator host name, in a comment | `b22a42c1` |

## Commands and exit codes

Every command was run from the review worktree.

| Probe | Command | Exit |
|---|---|---|
| lane suite | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1400s -n i6dmo-review -- python -m pytest -q -p no:cacheprovider tests/i6dmo/test_matched_population.py tests/i6dmo/test_matched_fit_path.py tests/i6dmo/test_ops_matched_runner.py tests/i6dmo/test_ops_matched_real_chain.py tests/i6dmo/test_ops_automation_repair.py` | 0 (92 passed) |
| full i6dmo suite | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 3600s -n i6dmo-review-suite -- python -m pytest -q -p no:cacheprovider tests/i6dmo` | 0 (506 passed; `out/full_suite.txt`) |
| p1 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 2400s -n i6dmo-review -- python <review>/p1_tamper.py` | 0 |
| p2 | `crispdm-run -q -W 1800 -m 6G -t 3600s -n i6dmo-review-p2 -- python <review>/p2_perturb.py` (*) | 0 |
| p3 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p3 -- python -I <review>/p3_independent.py` | 0 |
| p4 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p4 -- python <review>/p4_report_real.py` | 0 |
| p5 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 3600s -n i6dmo-review-p5 -- python <review>/p5_runbook.py` | 0 (inner exit codes in `out/p5_runbook.json`) |
| p6 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p6 -- python <review>/p6_orchestration.py` | 0 |
| p7 | `p7_stage.py check / stage / measure 4G / fit 2813475840 / replay 2813475840 / fetch`; `p7_gpu_replay.py --mode inputs` (CPU) | all 0 |
| p8 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p8 -- python <review>/p8_cpu_fixture_and_seam.py` | 0 |
| p9 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 900s -n i6dmo-review-p9 -- python <review>/p9_capacity.py` | 0 |
| p10 | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 600s -n i6dmo-review-p10 -- python <review>/p10_timers.py` | 0 |
| E2E | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-review-e2e -- python -m tests.i6dmo.e2e_automation_receipt --scratch $SCRATCH/e2e/s --out $SCRATCH/e2e/E2E_RECEIPT_at_4a77bea8.json` | 0 (PASS) |

(*) p2 first queued at 6G on the coordinator and was refused because of the slice aggregate
budget (`SLICE_AGGREGATE_BUDGET`). An attempt at 4G was then refused with
`CAP_LOWERED_AFTER_REFUSAL`, and the cap was not lowered again. p2 was moved instead, at the same
6G cap, to the secondary worker's CPU (CUDA hidden), running against the exact staged tree. The
base rebuild there reproduces the sealed digest `4ab41bd8…` exactly.

`<review>` = `docs/i6n/direct_multi_output_static/matched_short/review`
