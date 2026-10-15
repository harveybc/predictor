# Run the R0-versus-R2 branch-pretraining contrast

This runbook covers the finite R0-versus-R2 contrast ruled in
`R2_CONTRAST_DESIGN.ruled.json` (seal `0cad8817…`, status `FROZEN_BEFORE_FIT_OBJECTIVE_RULED`,
accepted by `r2_design.validate_r2_design(for_execution=True)`). The contrast
runs inside the existing i6dmo application and reuses its queue, governed worker, stored-terminal
verifier, outbox with exact readback, mirrors, automatic `CLOSE` and `TIMERS_STOPPED`.

Nothing in this document has been executed on a GPU, initialized or deployed. The only end-to-end
execution so far is the CPU engineering fixture listed under "What was verified".

## The graph

| Units | Kind | Device | Depends on | What it does |
|---|---|---|---|---|
| `r2.pretrain.00` … `r2.pretrain.18` | `pretrain` | GPU | none | One unit per feature. It runs `model.branch_pretraining_corpus` (FIT windows plus purged INNER selection rows, shared FIT-only scaler) and `model.pretrain_branch` with the ruled O1 objective. The seed `6102026 + feature_index` is bound immediately before the decoder is built. `model.write_branch_donor_terminal` writes the donor terminal, and the store retains its digest. |
| `r0.fit` | `fit` | GPU | none | Fresh R0 fit of the mg2m arm with no import binding. |
| `r2.fit` | `fit` | GPU | all 19 pretraining units | `train.py --r2-donors`. `train.r2_import_into` runs `model.import_branch_donors` with the 19 authenticated references and this campaign's provenance, then the R0 recipe runs. The receipt binds `identity.r2_import_sha256`, and `R2_IMPORT.json` is retained. |
| `r0.report`, `r2.report` | `report` | CPU | their fit | `reporting.run_matched_unit` on the common DEVELOPMENT rows. |
| `r2.probe` | `evaluation` | CPU | all 19 pretraining units | The frozen predictive probe. It is **report only**: no unit depends on it, it never gates and it never selects. |
| `comparison` | `comparison` | CPU | `r0.report`, `r2.report`, `r2.fit` | `r2_design.decide_r2`, which is strict unless the campaign is the engineering CPU fixture. |

After the comparison, the rest is automatic: publish with exact readback, then both mirrors, then
the automatic `CLOSE`, then this campaign's own timers are stopped.

Possible dispositions are `R2_SELECTED` (the R2 score is below the R0 score minus 1e-8),
`R0_RETAINED` (ties retain R0) and `INCOMPLETE_COMPARISON`.

A failed pretraining unit is a typed terminal. The `R2_PRETRAINING_BUDGET_STOP` stop is one
example. The failure blocks `r2.fit`, `r2.report` and `r2.probe`. The comparison then runs on what
is terminal and closes `INCOMPLETE_COMPARISON`. It never selects R0 from the R0 arm alone.

TEST is never enumerable (`TEST_NOT_ENUMERABLE_IN_MATCHED_CONTRAST`). Pretraining and the probe
only ever request the population's `FIT` and `INNER` splits (`r2_runner.FEATURE_SPLITS`). Any
other split is refused with `R2_SPLIT_FORBIDDEN`.

R0 policy: this runner always fits R0 fresh inside the graph. `r2_design.r0_reuse_decision` is not
evaluated here. The contract allows a fresh R0 whenever reuse is not proven (R2-08), and the R0
report is always recomputed.

## Admission classes (R2-17)

Each class is measured separately. The first unit of a class runs under the provisional cap and
**is** the measurement. After that, the cap is 1.25x that class's **own** measured peak
(`worker.cap_for`). A pretraining peak never sets a fit's cap, and a fit's peak never sets a
pretraining cap. The caps are sealed in `EFFECTIVE_CONFIG.json` under `class_caps`, each with its
basis.

| Class | Provisional cap | Basis |
|---|---|---|
| `r2_contrast_r0_fit.fit` | 3,584,604,160 B | 1.25x the measured cgroup peak of the closed matched `mg2m.fit` (2,867,683,328 B). R0 uses that same recipe, arm, population and runtime. |
| `r2_contrast_r2_fit.fit` | 3,584,604,160 B | Provisional. The R0 recipe on the identical network, after an in-place import of 19 small branch weight sets. Replaced by the R2 fit's own measured peak. |
| `r2_branch_pretrain.pretrain` | 3,584,604,160 B | Provisional upper bound. One branch plus a 32-channel decoder on one feature's windows, with the same population load and TF runtime as a fit. The first pretraining unit is the measurement. |

Pre-execution refusals keep the existing rules:

- `INFRA_PREFIT:*` and proven `LAUNCHER_ADMISSION_REFUSED` (exit 75 with no child output) are
  waits. They are not burned attempts, and they rotate the claim order.
- Once a refusal persists past the bounded threshold, `status` reports it as
  `INFRASTRUCTURE_BLOCKED` or `ADMISSION_BLOCKED`.

### Same GPU class for R0 and R2 (review C1)

A scientific `init` seals `fit_gpu_class` in `EFFECTIVE_CONFIG.json`. It is the class of the
preferred GPU role, the heavy GPU role with the lowest `order`, and `fit_gpu_class_role` names that
role. Both `r0.fit` and `r2.fit` carry it as `gpu_class`.

- The worker never claims either fit on a role of another class. The fit waits as `WAIT_RESOURCE`
  with cause `FIT_GPU_CLASS_HOLD:...`; it is never failed and never relabelled. The same check
  is repeated at admission.
- Pretraining units carry no class. They may run on any GPU role after their own admission, and
  each pretraining terminal records its `device_class` and the GPU device names.
- Holds are listed under `r2.fit_gpu_class_holds` in campaign status.
- If the preferred host stays unavailable, the existing bounded typed status applies (service,
  infrastructure or admission). There is never a fall-back to another class.
- `decide_r2` still refuses a mixed-class pair (`R2_FIT_PINNING`) as a second line of defence.

### Determinism init order (review C2)

`run_pretrain` and `run_probe` import `tools.i6dmo.model`, which sets the static-algorithm flags,
before TensorFlow. A process that had already imported TensorFlow is refused with the typed
terminal `DETERMINISM_INIT_ORDER_INVALID`, and no donor is written. Each pretraining terminal
records `tf_imported_before_flags` (false in a correct unit process).

### Probe runtime (review C3, disclosed only)

`r2.probe` runs on the executing role's own runtime. The coordinator has TF 2.18.0 / Keras 3.9.0,
not the 2.21.0 / 3.13.2 pin. `PROBE.json` and the `r2.pretraining.predictive_probe` fact record
the actual versions with the label `REPORT_ONLY_RUNTIME_NOT_PINNED`. This is not enforced, because
the probe never gates or selects.

A scientific R2 campaign never admits a CPU fit or pretraining unit
(`R2_SCIENTIFIC_CPU_FIT_REFUSED`). It never launches a child with an `I6DMO_TEST_*` override in
reach (`TEST_OVERRIDE_IN_SCIENTIFIC_ENVIRONMENT`). It requires the sealed TF/Keras runtime pin on
the executing worker. `decide_r2` additionally requires that both fits are scientific GPU fits on
the same runtime and the same recorded GPU class (R2-18).

## Placeholders

These are private and never committed:

| Name | Meaning |
|---|---|
| `$RUNTIME` | Staged predictor runtime worktree on ext4, under `/var/tmp`, never `/home` |
| `$PRIVATE` | Private directory outside git (`chmod 700`) |
| `$PRIVATE/deployment.json` | The existing private deployment spec |
| `$DATA_ROOT` | The role's data root |
| `$EXPERIMENT_STORAGE_ROOT` | The configured experiment storage root |
| `$I6DMO_ROOT` | A NEW campaign root on ext4 |

Every role must declare `experiment_storage_root`. This works exactly as for the matched contrast;
see `RUN_MATCHED_SHORT.md`.

```bash
cd "$RUNTIME"
export CUDA_VISIBLE_DEVICES=""
```

## 1. Prepare

`prepare` runs the matched preparation: pins by path, the population verified against the
committed `POPULATION_SEAL.json`, the memory and disk floors, and no TEST. It then binds the RULED
contract. The mg2m design seal, the population digest and the runtime must all equal the contract
(`R2_CONTRACT_BINDING` otherwise).

```bash
crispdm-run -m 3G -t 600s -n i6dmo-r2-prepare -- python -m tools.i6dmo r2 prepare \
  --deployment "$PRIVATE/deployment.json" \
  --input-dir "$DATA_ROOT/inputs/extended-no-stlfsi4" \
  --out "$PRIVATE/r2.prepared.json"
```

This writes `$PRIVATE/r2.prepared.json` and its companion `$PRIVATE/r2.prepared.json.matched.json`,
both with mode 0600. An output inside a git checkout is refused.

## 2. Validate and preflight (launch nothing)

```bash
crispdm-run -m 3G -t 600s -n i6dmo-r2-validate -- python -m tools.i6dmo r2 validate \
  --prepared "$PRIVATE/r2.prepared.json"
crispdm-run -m 1G -t 600s -n i6dmo-r2-preflight -- python -m tools.i6dmo r2 preflight \
  --prepared "$PRIVATE/r2.prepared.json" --out-dir "$PRIVATE/r2-preflight"
```

`preflight` is the matched per-role probe: pins, disk, the actual TF/Keras version against the
sealed pin, and initial admission on the preferred GPU role. It exits 3 when any check that blocks
launch fails.

## 3. Init, then deploy (not part of this pass)

```bash
crispdm-run -m 3G -t 600s -n i6dmo-r2-init -- python -m tools.i6dmo r2 init \
  --prepared "$PRIVATE/r2.prepared.json" --root "$I6DMO_ROOT"
python -m tools.i6dmo deploy --root "$I6DMO_ROOT" --deployment "$PRIVATE/deployment.json" \
  --render-to "$PRIVATE/r2-units-preview"
python -m tools.i6dmo deploy --root "$I6DMO_ROOT" --deployment "$PRIVATE/deployment.json" --no-start
```

`init` re-validates first. It refuses each of the following:

- an objective override (`R2_OBJECTIVE_OVERRIDE`)
- the engineering CPU fixture on any design not sealed on a synthetic calendar
- a non-mg2m design
- an existing root
- an R0 reuse decision other than `FRESH_R0_REQUIRED`

The ordinary `tick`, `worker`, `publish` and `status` services then drive the graph. At `CLOSE`,
or at a typed technical terminal, they stop only this campaign's units and record
`TIMERS_STOPPED.json`.

## 4. Status: campaign, stage, regime and unit

```bash
python -m tools.i6dmo r2 status --root "$I6DMO_ROOT" --level campaign
python -m tools.i6dmo r2 status --root "$I6DMO_ROOT" --level stage
python -m tools.i6dmo r2 status --root "$I6DMO_ROOT" --level arm --regime R2_PRETRAIN
python -m tools.i6dmo r2 status --root "$I6DMO_ROOT" --level unit --unit r2.pretrain.00
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign --probe-systemd --json
```

The campaign level carries the following:

- `r2.stage`, the first stage that is not yet terminal
- `r2.decision`
- `r2.pretraining`, giving complete / denominator / failed units
- per-regime views (`R2_PRETRAIN`, `R0`, `R2`, `R2_PROBE`, `DECISION`) with ETAs measured only
  from that regime's own completed units
- typed failures

## OLAP facts (separate, with readback)

| Fact tag | Source unit |
|---|---|
| `r2.pretraining.loss_curve` | each pretraining unit (per-epoch FIT/selection loss) |
| `r2.pretraining.reconstruction` | each pretraining unit, `use=REPORT_ONLY_NEVER_SELECTS` |
| `r2.pretraining.compute_cost` | each pretraining unit |
| `r2.pretraining.predictive_probe` | `r2.probe`, `use=REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS` |
| `r0.fit.fit_inner_diagnostics`, `r2.fit.fit_inner_diagnostics` | each fit (per-epoch FIT/INNER loss, best monitored) |
| `r0_r2.fit.compute_cost` | each fit |
| `r0_r2.development.per_head_metrics` | `comparison` |
| `r0_r2.decision` | `comparison` |

## What was verified, and how

All tests ran on CPU under `crispdm-run -m 3G -t 600s`.

| Claim | Test (`tests/i6dmo/test_r2_real_chain.py`) |
|---|---|
| Real chain end to end. Nothing is faked: a synthetic population is built by the real `build_common_population`, then 19 real pretraining units, the real trainer for R0 and for R2 (`--r2-donors`), real reports, the real probe, `decide_r2` with the literal `require_scientific=False`, publication to the test sink with exact readback, mirrors, `CLOSE` and `TIMERS_STOPPED` | `test_r2_contrast_real_chain_pretrain_import_fit_report_decide_publish_close_stop` |
| DEVELOPMENT/TEST are never opened by pretraining or by the probe | `test_pretraining_and_probe_never_open_development_or_test` |
| Pretraining replays bitwise on CPU (donor weights). A budget stop is a typed failure that leaves no donor | `test_pretraining_replay_is_bitwise_and_budget_stop_is_typed` |
| A failed pretraining unit blocks R2 and leads to `INCOMPLETE_COMPARISON`, never an R0-only selection | `test_failed_pretraining_blocks_r2_and_closes_incomplete_never_r0_only` |
| The probe never gates or selects. Corrupting it leaves the decision byte-identical | `test_probe_never_gates_or_selects` |
| The scientific path refuses CPU fits (init, decision, worker) | `test_scientific_path_refuses_cpu_fits` |
| Scientific init binds the contract, seals three separate admission classes, has no TEST, and the R2 fit waits for donors | `test_scientific_init_binds_contract_and_seals_separate_admission_classes` |
| Restart, duplicate claim and outbox retry never repeat a completed pretraining unit or fit | `test_restart_duplicate_claim_and_outbox_retry_never_repeat` |
| R2 imports only this campaign's authenticated donors, R0 never imports, and an R2 fit without donors never runs | `test_r2_fit_imports_only_this_campaigns_authenticated_donors` |
| C1: both fits are held to the sealed preferred-role GPU class while pretraining may use either GPU role; a mixed-class decision is never reached | `test_fits_are_held_to_the_sealed_gpu_class_and_a_mixed_decision_is_never_reached`; `test_retsu_r2_runner_review.py::test_r2_fit_must_wait_for_the_r0_gpu_class` |
| C2: the unit records `tf_imported_before_flags` = false; a process that imported TF first is refused with `DETERMINISM_INIT_ORDER_INVALID` | the real-chain test; `test_a_process_that_imported_tensorflow_first_is_refused` |
| C3: the probe's actual runtime is recorded as `REPORT_ONLY_RUNTIME_NOT_PINNED` | the real-chain test |
| Every `python -m tools.i6dmo` line in this document parses | `tests/i6dmo/test_r2_runner_ops.py::test_run_r2_document_names_only_existing_commands` |
