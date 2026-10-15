# Independent review: frozen R0-vs-R2 design contract (Retsu, 2026-10-15)

## Scope and how the review ran

- **Subject.** Branch `satoshi/i6dmo-followup-20261015` at `7488e899` (MODEL commit `96046c22`):
  - `tools/i6dmo/r2_design.py`;
  - the R2 donor checks in `tools/i6dmo/model.py`;
  - `tools/i6dmo/design.py`;
  - `tests/i6dmo/test_r2_design.py`;
  - `R2_CONTRAST_DESIGN.json` (seal `3e46e005…`, status `FROZEN_BEFORE_FIT_DESIGN_ONLY`, objective `UNDECIDED_REQUIRES_REVIEW`);
  - `R2_DESIGN_PROPOSAL.md`.
- **Authority.** MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP_20261015 §4 and NEXT_CONTRASTS_PLAN_20261015 §3.
- **Review branch.** `retsu/i6dmo-r2-design-review-20261015` adds only:
  - `tests/i6dmo/test_retsu_r2_design_review.py`;
  - this document.
- **What was and was not run.**
  - CPU only.
  - Models were built at their seed initialization and never fitted. No pretraining ran and no TEST data was read.
  - The real population (`population-tf2210`) was read once, for slot instants only, to count the INNER purge.
- **Command.**

  ```
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 600s -n i6dmo-r2-review -- python -m pytest -q -p no:cacheprovider \
    -rxXf tests/i6dmo/test_r2_design.py tests/i6dmo/test_retsu_r2_design_review.py
  ```

  Result: **71 passed, 4 xfailed**, exit 0. That is the lane's 63 tests plus my 8. All four xfails are strict and were rerun with `--runxfail`; each gave `DID NOT RAISE`, i.e. the stated defect is real.

## Verdict: ACCEPT_WITH_CONDITIONS

The design contract is sound in the following respects:

- **Shared inputs.** R0 and R2 share the population, scaler, masks, seed, architecture and recipe.
- **Arm difference.** The only difference between the arms is the initial weights of the 19 trainable branches.
- **R0 reuse.** The reuse predicate is strict.
- **Scope.** Out-of-scope fields are refused.
- **Execution gate.** An undecided objective blocks execution: `--for-execution` returns `R2_OBJECTIVE_UNDECIDED`.

Four defects and gaps must be fixed before the runner is built or the objective is ruled; the conditions are below.

### Conditions

- **C1. Make the ruling executable and fully fixed.** This applies before the ruling is sealed.
  - `max_epochs` must be ≤ 100, because the registered fit loop caps it there.
  - `max_updates` and `max_seconds` must be fixed explicitly. Neither may bind silently; a budget stop must be a typed failure.
  - `_validate_objective` must check executability against the registered component (D1).
- **C2. Bind the seed in the runner.** Call `keras.utils.set_random_seed(6102026 + feature_index)` immediately before building and fitting each autoencoder, and record it. The registered loop does not consume `seed` (D2).
- **C3. Bind donors to their provenance.** This is a BLOCKER before any R2 execution.
  - Import must accept only donors produced by this campaign's own pretraining units, verified through their authenticated stored terminals.
  - The donor identity must include the learned-corpus digest (the FIT window digest per feature and the selection-row digest), the run's objective identity and settings, the seed and the code pin.
  - Today a manifest copied from the target and carrying another population's real weights is imported (D3).
- **C4. Tighten `decide_r2`.**
  - Require scientific GPU fits, matching R3 parity with `decide_matched`.
  - Require the R2 import record (branch digests equal to the donor digests, non-branch digest equal to the R0 seed build) and, for R0, the absence of one. Today two R0 fits can be decided as "R0 vs R2" (D4).
- **Also.** Correct the proposal's claim about I6-B (item 1) and add the acceptance rows listed in item 6.

## 1. Objective options

Recommendation: **O1, registered, with executable settings.**

| Check | Verdict | Evidence |
|---|---|---|
| O1 `autoencoder_reconstruction` 1.0.0 exists as a registered component | VERIFIED | `predictor_plugins/modular_temporal/objectives.py:228`, defaults `{decoder_channels: 32, loss: "mse"}`, setup.py entry point. Decoder from `pretraining.build_decoder` / `build_autoencoder` (UpSampling → Conv1D(channels, 3, gelu) → Conv1D(F, 1)); 13-step latent, exact |
| O2 `ts2vec_contrastive` 1.0.1 exists | VERIFIED | `objectives.py:114`. The module docstring still says 1.0.0; the decorator, which is authoritative, says 1.0.1. Its declared deviations (crop by zeroing history; masking that zeroes INPUT steps) are as the proposal states |
| O3 and O4 are not in the repository | VERIFIED | Not registered; the dossier only describes them |
| "O1 is already used end to end by `tools/i6b_branch_pretraining.py`" | **REFUTED** | `test_o1_registered_component_has_no_end_to_end_caller_and_i6b_uses_its_own_loop`. I6-B never calls `fit_objective` or `AutoencoderReconstruction`; it runs its own Keras loop (see below). The registered O1 component has no end-to-end caller anywhere in `tools/`. What is shared is the decoder family (`branch_autoencoder` / `build_autoencoder`) |
| The recommended O1 ruling (max_epochs 200, …) is executable | **REFUTED (D1)** | `r2_design.validate_r2_design` returns `READY_FOR_EXECUTION`, but `fit_with_early_stopping` refuses `max_epochs` 200 with `max_epochs must be an integer in [1, 100]`. Test: `test_recommended_o1_ruling_is_not_executable_by_the_registered_fit_loop`; the strict xfail `test_a_non_executable_ruling_must_be_refused_at_validation` |
| Seed rule "6102026 + feature_index" is applied by the component | **REFUTED (D2)** | `seed` is passed in but `fit_with_early_stopping` seeds nothing. The decoder is built before the loop, so its initialization depends on the caller's global RNG state |
| The ruling fixes every stop rule | **REFUTED (D1)** | `max_updates` (100000) and a wall-clock `max_seconds` (3600) default silently. `OBJECTIVE_FIT_SETTINGS` does not contain them. A `max_seconds` stop would make the donor depend on the host |

The I6-B loop that the proposal cites differs from the registered O1 component in every respect that matters here:

| | I6-B loop | Registered O1 |
|---|---|---|
| Objective name | `branch_autoencoder_reconstruction` | `autoencoder_reconstruction` 1.0.0 |
| Optimizer | Adam, no weight decay | AdamW with lr / wd |
| Decoder channels | 16 | 32 |
| Monitor | `MeanTrainTailEarlyStopping`, mean(running train loss, tail) | validation loss on `vx` only |
| Shuffle | `shuffle=False` | fixed batch order (also unshuffled) |

**Is O1 a faithful use of existing reviewed work?** Partly.

- It *is* an existing, registered, identity-bearing component. Ruling it invents no objective and no architecture: the branch is unchanged, and the decoder is discarded.
- But it is *not* the reviewed end-to-end precedent. That precedent is a sibling loop with other parameters and another objective identity. Ruling O1 would be its first end-to-end donor use.

**Why O1 anyway.** O1 still beats every alternative:

- **O2:** zeroing z-scored inputs to mask them collides with the scaler mean, and its reconstruction is `NOT_APPLICABLE`.
- **O3 / O4:** would be inventions, which the order forbids.
- **The I6-B sibling loop:** has no registered identity, and its Adam/no-decay optimizer departs from the R0 recipe.

**Recommended ruling for O1:**

| Setting | Value |
|---|---|
| Objective params | `decoder_channels` 32, `loss` mse (the registered defaults) |
| Optimizer | AdamW (the registered loop compiles AdamW), lr 1e-3, weight decay 1e-4 |
| batch_size | 256 |
| max_epochs | **100**. This is the registered cap. It is a disclosed difference from R0's 200, and pretraining is a separate stage |
| patience / min_delta | 10 / 1e-8 |
| max_updates | 100000, set explicitly (non-binding: about 65 updates per epoch × 100 epochs) |
| max_seconds | 86400, set explicitly. Any budget stop is a typed failure, never a selected donor |
| Seed | `keras.utils.set_random_seed(6102026 + feature_index)` before the decoder is built (C2) |
| Restore | the best monitored epoch |

## 2. Pretraining selection monitor

Recommendation: **INNER-only (purged).**

| Option | Assessment |
|---|---|
| **INNER-only (purged)** | VERIFIED as the *only* monitor the registered O1 executes unchanged: `fit_with_early_stopping` monitors `validation_loss` on the `vx` it receives. The asymmetry is that INNER is used twice in R2 and once in R0. That use is **inputs only**: no labels, no targets. It never touches DEVELOPMENT, the scoring split, so it cannot leak outcome information into the R0-vs-R2 decision |
| mean(FIT, INNER) "like R0" | **Not implementable without new code.** `objectives._settings` rejects a `monitor` key (`test_o1_can_only_monitor_its_validation_argument`). The evaluator's `train_validation_mean` averages the *running in-epoch* train loss, not an end-of-epoch FIT evaluation at the same weights, so it would not be "like R0" anyway. I6-B's mean has the same running-loss form. It dilutes INNER's influence; it does not remove it |
| Fixed-epoch, no selection | Fairest in principle, because INNER would then enter both arms only through the fine-tune monitor. But the registered loop *always* restores the best monitored checkpoint, so this needs new code. It also adds a free hyperparameter (the epoch count) that no one has reviewed |

My recommendation is INNER-only, as ordered, with the asymmetry kept disclosed. The cleanest fallback, if the lead wants INNER fully out of pretraining, is to pass the FIT windows as the validation argument of the unchanged registered loop. That stops on FIT reconstruction, so INNER influences both arms only through the fine-tune. It is not one of the three listed options, so it would need an explicit ruling.

## 3. Shape-only refusal

| Attempt | Verdict | Evidence |
|---|---|---|
| Honest MG1 manifest (25 slots, same kernel shape) | VERIFIED refused, atomically | `R2_DONOR_ARM_MISMATCH`; no weight set |
| DS2 elapsed / hourly network | VERIFIED refused | `branch_donor_context` refuses with `R2_DONOR_ARM_MISMATCH`; the lane's rule, spacing and slots mutations are also refused |
| Closed full-population MG2 with honest population and scaler digests | VERIFIED refused | Lane tests: `R2_DONOR_POPULATION_MISMATCH` and `R2_DONOR_SCALER_MISMATCH` |
| FS4 / I6-B widths (3, 1, 16) | VERIFIED refused, atomically | `R2_DONOR_ARCHITECTURE_MISMATCH` |
| **A forged identity block** (manifest copied from the target, weight digests recomputed) carrying real-shaped weights from the closed full-population MG2 build or the MG1 build | **REFUTED (D3)** | Imported: `test_observation_forged_identity_with_foreign_real_shaped_weights_is_imported[closed_full_population_mg2, mg1_matched]`; strict xfail `test_a_forged_identity_block_must_be_refused_before_any_weight_is_set`. Every identity field is self-declared, and the weight digest is computed from the weights supplied, so it proves only internal consistency |

The import does verify all 19 donors before it sets any weight. If `set_weights` fails partway through, though, earlier branches stay modified. That is unlikely, but the import is not transactional; NON_BLOCKING.

## 4. R0 reuse predicate

**FRESH_R0_REQUIRED: VERIFIED.**

- **Why fresh.** `test_r0_reuse_is_fresh_because_trainer_code_files_changed_since_the_deployed_pin`: of the trainer's `CODE_FILES`, `tools/i6dmo/design.py` and `tools/i6dmo/model.py` differ between the deployed pin `796ec8f4` and `7488e899`. The other six are identical. The predicate fails on `code_files_sha256` alone.
- **Fairness.** A fresh R0 under the same code pin, runtime and monitor implementation as R2 is the fairer control anyway. Both fits then share the same code, the same determinism flags, the same admission and launcher path, and are exposed at the same time.
- **Recommendation (NON_BLOCKING).** Also pin both fits to the same GPU class or role. GPU class is not part of the contract, and cross-GPU numerical differences are a known effect, bounded by the v2 policy.

## 5. Leakage

| Check | Verdict | Evidence |
|---|---|---|
| FIT pretraining windows end before INNER | VERIFIED on the real population | Every MG2m FIT slot instant is before the INNER start, 2023-11-06Z |
| INNER selection windows | VERIFIED | 715 INNER origins; 14 are purged (earliest slot before the INNER start) and 701 kept. Every INNER slot is before the DEVELOPMENT start |
| Purge rule against label maturity | VERIFIED | Pretraining uses no labels, so label maturity does not apply to its selection. The purge removes input-bar overlap between FIT training windows and INNER selection windows, which is the correct rule for unlabeled selection. FIT's own labels already end before INNER (max origin + 12 h < INNER start) |
| Frozen reconstruction and probe never select or gate anything | UNVERIFIED (design text only; no runner) | The probe fits on FIT labels and evaluates on INNER labels as "REPORT_ONLY". Nothing in the code yet enforces that its output never reaches a decision or gate |
| DEVELOPMENT and TEST never opened by pretraining | UNVERIFIED (no runner) | The R2-04 DEVELOPMENT-perturbation test is specified and pending |

## 6. Acceptance table R2-01…R2-12 against plan §3

**Covered:**

- RAW recipe and no replacement inputs (R2-05);
- matching support, scaler and split (R2-01, R2-04);
- import and keep trainable (R2-02);
- exact R0 reuse (R2-08);
- no shape-only donors (R2-03);
- new donor → new identity (R2-06, partially);
- frozen reconstruction and predictive evaluation (R2-07);
- core pretraining downstream (R2-12);
- separate OLAP facts (R2-10);
- reconstruction is not utility (R2-07).

On the plan's "calendar/time inputs": the branch has no time inputs, and pretraining uses the identical MG2m slot windows. That is consistent.

**Missing — add these rows:**

| Row | Requirement |
|---|---|
| R2-13 | Donor provenance: each imported donor references this campaign's pretraining-unit stored terminal, authenticated, plus the learned-corpus digest (FIT windows and INNER selection rows), objective run, seed and code pin. A forged or copied identity is refused (D3) |
| R2-14 | Pretraining determinism: seed bound before the decoder is built; no wall-clock budget stop; bitwise-identical donor on CPU replay |
| R2-15 | The ruled objective is executable by the registered component at validation time (D1) |
| R2-16 | Decision integrity: `decide_r2` requires scientific GPU fits and the R2 import record, and refuses an import record for R0 (D4) |
| R2-17 | Automation (plan §4) for the new graph. Pretraining units get a measured-peak admission class (1.25× its own peak, never the fit's). Crash/restart, duplicate claim and outbox retry never repeat a completed pretraining or fit. Status at every level; CLOSE with mirrors. A failed pretraining unit gives a typed `INCOMPLETE_COMPARISON`, never an R0-only selection |
| R2-18 | R0 and R2 fits run on the same runtime and GPU class (or the class is recorded and disclosed) |
