# I6-DMO lane `model`: return

Branch `satoshi/i6dmo-model-20261014`, base `2a440265`. Tests commit
`e9a3f43a`; implementation commit `02893770`. This return is software evidence
only. No scientific fit, GPU job or TEST read was performed.

## DMO traceability (model side)

| DMO | Test(s) in `tests/i6dmo/` | PRE (`e9a3f43a`) | POST (`02893770`) |
|---|---|---|---|
| DMO01 | `test_dmo_model.py::test_dmo01_*` (4), `test_causality_future_input_does_not_move_earlier_core_steps` | collection ERROR (module absent) | PASS |
| DMO04 | `test_dmo_train.py::test_dmo04_*` (3) | collection ERROR | PASS |
| DMO10 | `test_dmo_train.py::test_dmo10_*` (3), `test_dmo_model.py::test_dmo10_save_load_identical_predictions_and_tamper_refusal` | collection ERROR | PASS |
| DMO18 | `test_dmo_train.py::test_dmo18_*` (5 functions, 8 cases), `test_dmo_design.py::test_dmo18_*` (3 functions, 9 cases) | collection ERROR | PASS |
| design adapter | `test_dmo_design.py` (unknown/missing/duplicate/NaN fields, fixed section C values, pure `validate-design`, CLI) | collection ERROR | PASS |
| entry point | `test_cli_entry_point_typed_exit_and_refuses_existing_out`, `test_gpu_request_without_gpu_is_typed_no_cpu_fallback` | collection ERROR | PASS |

PRE: `3 errors in 3.92s`. POST: `60 passed in 31.06s`.

Command:
`CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 300s -n i6dmo-model -- python -m pytest -q tests/i6dmo/test_dmo_design.py tests/i6dmo/test_dmo_model.py tests/i6dmo/test_dmo_train.py`

## What exists

- `python -m tools.i6dmo.design validate-design --design <d> [--json]` is pure.
  It resolves the committed design to design sha `541ce40a…` and effective
  config sha `cd78cc6c…` (was `56ba5324…` before the `family` key was added in
  round 2), with 19 features (manifest `2ff34fa2…`). The fit
  window is `[2020-01-01, 2023-11-06)` and inner is `[2023-11-06, 2024-01-01)`.
  Seven features still have UNVERIFIED availability:
  dgs3mo.diff_5d, dgs7.logret_5d, gbpusd.logret_1h, px.close_loc, px.log_co,
  px.rv5 and rg.atr_ratio. `add_cli(subparsers)` registers this command on the
  shared CLI.
- `python -m tools.i6dmo.train --design --bundle --out --device {cpu,gpu} [--json]`
  writes a receipt. Exit codes: 0 completed, 2 contract error, 3 typed, 4 untyped.
  INTERFACE_MODEL.json describes the input bundle and the output artifacts.
- The model has 4,834 parameters: 19 branches (304), core (1,932) and six heads
  (433 each).

## Bridge, not fork

Components are declared with the `modular_temporal` `component()` protocol.
The model also reuses `FeatureSelect`, the registry-resolved `sequence_concat`
fusion, `_validate_component`, `probe_alignment` and `weights_hash`.
`PhysicalOffsetEncoding` subclasses the plugin's `PositionalEncoding`.

`build_modular` is not called, for three reasons:

1. Its built-in defaults (GELU, Transformer core, single Dense head) are
   historical identities and must stay unchanged.
2. Components that are not built in resolve only through installed entry points.
3. It labels horizons as multiples of `sample_hours`, so 3 h and 5 h would
   become 6 h and 10 h.

## Declared defaults (not fixed by section C; recorded in the effective config)

1. Seeded per-epoch shuffle: batch order is `np.random.default_rng(6102026)`
   permutation each epoch (`batch_order: seeded_permutation_per_epoch`).
2. ReLU in the core: both core Conv1D layers (8 then 4 filters) use ReLU; the
   plan names only filters/kernel/stride for the core.
3. Restore the lowest-loss epoch: the restored checkpoint is the epoch with the
   lowest `0.5*(fit+inner)` (first tie wins); `min_delta` only drives patience
   (stale unless `monitored < reference - min_delta`), the i6n convention.

## Round 2: long family (Satoshi request)

Tests `48c59ff0` (PRE: 10 failed, 11 errors in 6.64s), implementation in the
following commit; POST for all four model-lane files: `81 passed in 41.42s`.

- Long design adapter: `validate-design` accepts the short and the sealed long
  design (family by `experiment_id`, schema stays `i6dmo.design.v1` for ops).
  Long = short design with PLAN section 5 fields: heads 18/24/36/48/60/72,
  span 144 h, spacing 12 h; 13 observations and hourly issuance are equal to
  the short ones. Tamper is `LONG_DESIGN_SEAL`, an `UNDECIDED` field is
  `LONG_DESIGN_UNDECIDED: <field>`.
- `seal-long-design --short-design --short-terminal --out`: deterministic and
  idempotent. It accepts only a `short.report` terminal with status COMPLETE or
  NOT_ELIGIBLE and records unit/identity/file digest, never metrics. The long
  hyperparameters are identical whatever the short outcome (tested). The ops
  command `seal-long --root R --design EXPERIMENT.long.json` then seals it into
  the campaign; the names do not collide (tested).
- Open-field reading: PLAN.md leaves no long field open, so
  `LONG_PLAN_OPEN_FIELDS = ()`. "Freeze its support/cost design before
  fitting" is read as sealing the mirrored business (masks) and execution
  (cost/admission) blocks. If Satoshi rules otherwise, adding the path makes the
  seal stop with `LONG_DESIGN_UNDECIDED` (tested by monkeypatch).
- Trainer: same entry point and the same masked normalized loss on heads
  h18..h72; a short bundle fed to the long design is a typed bundle mismatch;
  72 h fit labels reaching the inner boundary are `PURGE_VIOLATION`.
- The data lane must build a long bundle (offsets -144..0 step 12, horizons
  18..72). Its 144 h windows at 2020-01-01 need late-2019 input history.

## Risks

- Op determinism is off. Same-seed CPU runs were bitwise equal in tests; the
  GPU behaviour is untested.
- `tf_gpu_memory_peak_bytes` is GPU-only and untested on CPU.
- Bundle schema: the data lane reports that its real short bundle conforms and
  was accepted by `load_data_bundle` (INTERFACE_DATA).
- A tiny CPU smoke run (200 epochs) took 27 s. This is not an ETA.

## Round 3: audit corrections AUD-DMO-02 / AUD-DMO-03 (lane 2)

Audit branch `98d83fcf` merged ordinarily; lane 3 `3268079d` merged before the
final hooks. Tests first at `4f8e32ac` (PRE: 15 failed). Probes (audit
`probes.py`, probe 3 now resolved through the registered `unit_runner` hook,
the real executor seam; strategy_adapter untouched):

| Probe | PRE | POST |
|---|---|---|
| AUD-DMO-01 (sampling) | REPRODUCED | REPRODUCED — lane 1 owns it |
| AUD-DMO-02 (final window) | REPRODUCED | NOT_REPRODUCED |
| AUD-DMO-03 (final placeholder) | REPRODUCED | NOT_REPRODUCED |

- Typed FINAL_REFIT: `final_refit_design` seals TRAIN `[2021-01-01,
  2025-01-01)` with the same 8-week inner tail (fit ends 2024-11-06); no outer
  validation split; every sample/label strictly before TEST; scales fit only on
  the final TRAIN window; `FINAL_DESIGN_UNDECIDED` for PLAN-open fields (none).
  Final and validation identities never match, and each refuses the other's
  bundle, so no TRAINED terminal crosses identities (tested both directions).
- `unit_runner._family_design` applies the final window for `final_refit`
  (FINAL_WINDOW_MISMATCH otherwise) and the REAL CLI path was exercised in a
  subprocess on synthetic dates/arrays: COMPLETED on a final-window bundle,
  no validation predictions, `scientific=false` via the env-only test override.
- `tools/i6dmo/final.py` (new, model lane) replaces the placeholder by
  registration (`HOOKS["final_test"]`): guarded TEST bundle construction,
  frozen-model prediction generation and the final hook. Guards tested as
  refusals: TEST_READINESS, PAIR_VALIDATION, PROCEDURE_FREEZE policy mismatch,
  FRED clocks unresolved, EXECUTION_POLICY absent/tampered (consumed via
  `strategy_adapter.execution_from_policy(load_execution_policy(...))`, never
  re-derived). Backtest delegates to the strategy export
  `run_final_backtest(spec, predictions, execution_params, out_dir)`; until the
  strategy lane publishes it, the hook returns the typed
  `FINAL_BLOCKED_GATE:FINAL_BACKTEST_EXPORT_MISSING` with predictions retained.
- POST: model-lane files 98 passed; neighboring strategy/ops guard tests 38
  passed; full `tests/i6dmo` 227 passed (pre-merge baseline) — code-only
  acceptance, not scientific success. AUD-DMO-01 and the real TEST year remain
  closed pending lane 1 and the external gates.

## Round 4: DMO10 GPU save/load identity (lane 4 replay)

Op determinism is now ENABLED for training and inference everywhere
(`enable_determinism()` in `tools/i6dmo/model.py`: `enable_op_determinism` plus
`TF_DETERMINISTIC_OPS`/`TF_CUDNN_DETERMINISTIC`, called by `build_dmo`,
`load_dmo`, `train_unit` and `generate_test_predictions`) and recorded in the
train receipt (`determinism`), `MODEL.json` and `PREDICTIONS.json`. No numeric
tolerance was declared: the bitwise same-device check stands. CPU POST: 99
passed (PRE: the two new recording tests failed). The GPU confirmation is lane
4's bounded replay, not this lane's. The earlier "op determinism is off" risk
line is superseded by this round.
