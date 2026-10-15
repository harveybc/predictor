# Independent review: R2 runner (Retsu, 2026-10-15)

## Scope

| Item | Value |
|---|---|
| Subject | `satoshi/i6dmo-followup-20261015` at `a5ac7607`, tree `beef4ae5038b26255765a93f0b3d80604d5f71aa` |
| Changes in scope | `tools/i6dmo/r2_runner.py`; `R2_CONTRAST` mode in campaign, worker, unit_runner, matched_runner, status and cli; `train.py` (`r2_import_into`, `--r2-donors`); `RUN_R2.md`; ruled contract seal `0cad8817…` |
| Review branch | `retsu/i6dmo-r2-runner-review-20261015` |

The review branch adds only:

- `tests/i6dmo/test_retsu_r2_runner_review.py`
- `review/r2run_gpu_replay.py` and `review/r2run_stage.py`
- `review/out/r2run_gpu_replay_a5ac7607.json`
- this document

What was and was not run:

- No scientific fit, no TEST, no deploy, no real systemd call.
- `init` ran only into a scratch root.
- The only fit was the bounded engineering GPU replay of item 7.

## Verdict: ACCEPT_WITH_CONDITIONS

The seams, leakage controls, scientific integrity and automation all hold against the real adapters. One fairness gap remains, and C1 must be resolved before deployment.

**C1 (required before deployment): same-GPU-class placement.**

- The worker places `r2.fit` on the secondary (4090) role while `r0.fit` holds the preferred (5090) host.
- Nothing holds R2 for R0's GPU class.
- The mismatch is caught only at the decision. `decide_r2` raises `R2_FIT_PINNING`, the comparison returns FAILED, and the campaign ends at a typed technical terminal: no decision, no automatic CLOSE, both fits wasted.
- A cross-class decision is never accepted, which is correct, but a campaign could fail for this reason alone.
- Either:
  - add a claim-time hold: the second fit waits for the class the first fit recorded;
  - or deploy with exactly one heavy GPU role for this campaign, with `unit_roles` set to the preferred role plus the coordinator. Running serially costs little here: about 1.7 GB / 1–2 s per pretraining unit at 2 epochs on a slice, and 54.6 s per fit measured earlier.

**C2 (fix before deployment; one line each): TensorFlow is imported before the static-algorithm flags in `run_pretrain` and `run_probe`.**

- Both bodies run `import tensorflow as tf` before `from tools.i6dmo import model`.
- The donor process therefore records `INIT_ORDER.tf_imported_before_flags = True`, with no static flags set at import.
- The trainer's own init-order witness treats exactly this as `DETERMINISM_INIT_ORDER_INVALID`. My first import+fit driver hit that refusal when it imported TF first.
- Empirically the donor still reproduced bitwise on the 4090 across 3 separate processes (`2ec3006d…`). R2-14 should not rest on luck, so import `tools.i6dmo.model` first in both bodies.

**C3 (NON_BLOCKING; disclose or enforce): the probe runs on the coordinator's own runtime.**

- `r2.probe` builds models and imports donors on the coordinator, which has TF 2.18.0 / Keras 3.9.0, not the 2.21.0 / 3.13.2 pin.
- `matched_runtime_problem` covers only `fit` and `pretrain`.
- The probe is report-only, so the decision is unaffected.

## Verdicts

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Seam pretraining → donor terminal → stored digest → `--r2-donors` → `r2_import_into` → `import_branch_donors` → receipt `r2_import_sha256` → report → `run_comparison` → `decide_r2` | VERIFIED | Lane real chain: 117 passed. My tests listed in "Seam attacks" below |
| 2 | Leakage | VERIFIED | `out/r2run_gpu_replay_a5ac7607.json` (LEAK, PRETRAIN); details below |
| 3 | Scientific integrity | VERIFIED | Lane tests plus code reading; details below |
| 4 | Fairness: GPU-class placement | **REFUTED for placement (C1)**; cross-class decision VERIFIED refused | `test_cross_class_placement_is_not_held_at_claim_time`; strict xfail `test_r2_fit_must_wait_for_the_r0_gpu_class`; `test_r0_metrics_with_an_import_binding_and_cross_class_fits_are_refused` |
| 5 | Automation | VERIFIED | Details below |
| 6 | RUN_R2.md commands | VERIFIED | Details below |
| 7 | Bounded GPU engineering replay | VERIFIED (engineering only) | `out/r2run_gpu_replay_a5ac7607.json`; details below |
| 8 | Privacy | VERIFIED clean | No host names, aliases, GPU UUIDs, IPs, home paths, role paths or warehouse values in the 11 files changed `6c17976d..a5ac7607` or in this review's additions |

### 1. Seam attacks

Every attempt below was refused, typed, with no weight set:

| Attack | Result | Test |
|---|---|---|
| Donor set from another, independently initialized campaign | `R2_DONOR_PROVENANCE_MISMATCH` | `test_donor_from_another_campaign_is_refused_before_any_weight_is_set` |
| One substituted donor among our own | `R2_DONOR_PROVENANCE_MISMATCH` | same |
| Stale or rewritten donor terminal | `R2_DONOR_TERMINAL_DIGEST` | `test_stale_or_substituted_donor_bytes_are_refused` |
| Swapped `donor.npz` | `R2_DONOR_WEIGHTS_DIGEST` | same |
| 18 donors | the R2 fit is held (`R2_DONOR_REFERENCES_INCOMPLETE`); `r2_donor_references` is None; import refuses `R2_DONOR_SET` (lane) | `test_eighteen_donors_hold_the_r2_fit_and_never_launch_it` |
| R0 carrying donors | the adapter refuses `R2_R0_HAS_IMPORT` (lane); `decide_r2` refuses `R2_R0_HAS_IMPORT` | mine |
| R2 without donors | `R2_DONOR_REFERENCES_MISSING` | lane |

Trust boundary: the donor reference is the digest the OPS store retained. A coherent rewrite of the store (queue row, stored terminal, donor terminal and npz together) lies outside what unkeyed hashes can authenticate. That is stated and accepted.

### 2. Leakage

- **Accessor spy on the real population** (`population-tf2210`): computing all 19 learned corpora made 42 accessor calls, every one to FIT or INNER. DEVELOPMENT and TEST were never requested. `split_tensors` refuses both (`R2_SPLIT_FORBIDDEN`, lane).
- **DEVELOPMENT perturbation:** I perturbed the 6,243 DEVELOPMENT bars in a copy of the retained input and rebuilt the population with the real data-lane builder.
  - The population digest changed, and DEVELOPMENT x changed.
  - All 19 learned-corpus digests were bitwise unchanged.
  - The feature-0 FIT/selection slices were bitwise equal.
  - A real pretraining unit (4090, 2 epochs) produced the bitwise-identical donor (`donor.npz` `2ec3006d…`) on the original and on the perturbed data.
- **Byte-level nuance:** the population loader still reads the DEVELOPMENT npz bytes to verify the population digest as a whole. They are never requested as data.
- **INNER purge on the real population:** feature-0 counts are FIT 16,397, selection 701, purged 14.

### 3. Scientific integrity

All verified through the lane tests plus my reading of the code:

| Claim | Where it is enforced |
|---|---|
| CPU fits refused at init | a scientific init seals `fit_device` gpu; an engineering fixture with a real design is refused with `ENGINEERING_CPU_FIXTURE_REQUIRES_SYNTHETIC_CALENDAR`; `R2_OBJECTIVE_OVERRIDE` |
| CPU fits refused at the worker | `R2_SCIENTIFIC_CPU_FIT_REFUSED`, `TEST_OVERRIDE_IN_SCIENTIFIC_ENVIRONMENT` |
| CPU fits refused at the decision | `R2_NON_SCIENTIFIC_FIT`, strict unless the literal `require_scientific=False` on the fixture |
| Probe never gates or selects | no unit depends on `r2.probe`; corrupting it leaves `COMPARISON.json` byte-identical |
| Failed or budget-stopped pretraining | a typed FAILED terminal (it has `fit_seconds`, so it is never an infrastructure wait); `r2.fit`, `r2.report` and `r2.probe` become BLOCKED; the comparison closes `INCOMPLETE_COMPARISON`, never `R0_RETAINED` |

One observation: CLOSE waits for every unit to be terminal, the probe included. A stuck probe can therefore delay CLOSE, but it can never change the decision.

### 4. Fairness

R0 on the 5090 and R2 on the 4090 is refused at the decision, not held at placement. See C1.

### 5. Automation

- **No repeated work.** Restart, duplicate claim and outbox retry never repeat a completed pretraining or fit (lane, `NoLaunch` executor).
- **Admission classes.** The three classes (`r2_branch_pretrain.pretrain`, `r2_contrast_r0_fit.fit`, `r2_contrast_r2_fit.fit`) are separate. Each is sealed with its provisional cap of 3,584,604,160 B and its stated basis. The measured peak is then tracked per class.
- **Status.** Status works at campaign, stage, regime and unit level on a scratch init.
- **CLOSE and TIMERS_STOPPED.** Both work, scoped to the campaign (lane).
- **No starvation.** On one GPU role, `r0.fit` is the 20th claim, after the 19 pretraining units, so it is delayed but bounded (`test_r0_fit_is_reached_after_the_pretraining_claims_not_starved`).

### 6. RUN_R2.md commands

| Command | Result |
|---|---|
| `r2 prepare` against the real population and input | exit 0, contract `0cad8817…`, population `ba01f87a…`, design `416c1156…` |
| `r2 validate` | exit 0, VALID |
| `r2 preflight` with a private copy of the live matched deployment spec | exit 0, PASS: both GPU roles at TF 2.21.0 / Keras 3.13.2, population pins present; the coordinator runtime mismatch is marked INFO |
| `r2 preflight` with an older spec lacking storage roots | exit 3, FAIL; this is the expected typed refusal |
| `r2 prepare --test` | exit 2, `TEST_REQUEST` |
| `r2 prepare --population-dir` pointing at a missing directory | `UNAVAILABLE_MOUNT` |
| `r2 init` into a scratch root | exit 0, 26 units |
| `r2 status` at campaign / stage / arm `--regime R2_PRETRAIN` / unit | exit 0 each |
| `deploy --render-to` | exit 0 |

No init was run on a real root and nothing was deployed.

### 7. GPU engineering replay

| Item | Value |
|---|---|
| Placement | secondary role, RTX 4090 Laptop GPU pinned by its role UUID (recorded privately); TF 2.21.0 |
| Code | exact pin `a5ac7607` staged via git archive over ssh; tree manifest identical |
| Caps | each process admitted at the runner's own provisional class cap, 3,584,604,160 B |
| Pretraining | one real unit body, feature 0, FIT slice 2048 and purged INNER slice 256, at most 2 epochs |

Pretraining result:

- seed 6102026 applied; stop reason `max_epochs`;
- reconstruction mse_z 0.01965, mae_z 0.12494, which is report-only;
- **cgroup peak 1,733,713,920 B** (rerun 1,740,148,736 B). This is an engineering observation, not the scientific cap measurement.

Import and tiny fit (fresh process):

- `import_branch_donors` with 19 terminals. Feature 0 was really pretrained; the other 18 are engineering scaffolds at seed weights, labelled as such.
- The donor weights survived the import bitwise. The other 18 branches equal the seed build, and the non-branch digest equals the seed build.
- All branches stayed trainable, with 4,882 parameters.
- A 2-epoch fit followed. Monitor self-check PASS; saved weights equal the loaded weights; same-process reload predictions bitwise equal.
- Peak 1,987,870,720 B.

Fresh-process replay:

- inputs identical to the fit process;
- predictions bitwise equal, max abs diff 0.0, within the v2 floor;
- peak 1,691,197,440 B.

Cleanup: the worker-side scratch (736 MB) was deleted after fetching the JSON evidence. Nothing was written to any campaign root, queue or OLAP.

## Tests

| Run | Result |
|---|---|
| Lane files `test_r2_real_chain.py`, `test_r2_runner_ops.py`, `test_r2_design.py` at `a5ac7607` | 117 passed, exit 0 |
| `tests/i6dmo/test_retsu_r2_runner_review.py` | 6 passed, 1 xfailed (strict; the C1 reproduction), exit 0 |
| Full committed suite at `a5ac7607` | 49 `tests/i6dmo` files in 6 pieces, each under `crispdm-run -m 4G -t 600s`: **799 passed, 1 skipped**, every piece exit 0; this reproduces the OPS count (`out/r2run_full_suite_a5ac7607.txt`) |
