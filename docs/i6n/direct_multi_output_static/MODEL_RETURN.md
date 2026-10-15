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

## Round 5: DMO10 monitored-value identity (shared evaluation path)

The ~5e-6 GPU mismatch was a different computation path, not nondeterminism:
training monitored per predict_matrix batches while the replay recomputed in
one pass. Now ONE function, `train.evaluate_monitored` (predictions at the
training batch size, float64 global fsum sums/counts, equal-weight head mean),
is used by the training loop and by `train.replay_monitored`, which takes the
`monitor_evaluation` record from TRAINING.json and refuses any other procedure
or batch size as `MONITOR_PATH_MISMATCH`. The DMO10 replay now asserts bitwise
equality (`==`, no tolerance declared). The monitoring batch changed from the
implicit 4096 to the recorded training batch (256); tiny CPU fixtures are
unaffected (one batch either way), and no sealed scientific run depends on the
superseded implicit value. CPU POST: 56 passed across train/final/long (PRE:
the 2 shared-path tests failed). GPU confirmation is lane 4's rerun.

## Round 6: DMO10 in-session monitor identity (GPU replay 11/12 follow-up)

The remaining bit-stable ~5e-6 gap means the fit loop's monitored value came
from a different forward path than the shared inference path. Now the epoch-end
monitored value is computed ONLY through `evaluate_monitored` on the inference
graph (the exact call the replay makes), early stopping/patience/best-epoch
read that value, and `TRAINING.json.monitor_evaluation` records it with the
best-epoch per-head float64 sums/counts and `graph: "inference"`. At recording
time an in-session self-check recomputes the monitored value on the restored
weights through the same path and refuses the receipt bitwise-unequal as typed
`MONITOR_SELF_CHECK_FAILED`; `replay_monitored` refuses a training-graph record
(`MONITOR_PATH_MISMATCH`). No tolerance. This is the now-unambiguous
implementation of the design's "same-weight mean fit/inner" rule — selection
semantics unchanged, numeric path unified; the effective-config monitor text
says so, so the effective sha changed again (design sha unchanged). CPU POST:
94 passed (PRE: 2 failed). Lane 4 re-verifies on GPU; if its in-session gap
persists there, the self-check now fails typed instead of recording a
non-replayable value.

## Round 7: DMO10 canonical load path (GPU in-training vs loaded divergence)

Lane 4 localized the remaining bit-stable gap to the freshly LOADED instance:
keras full-model deserialization can reconstruct a differently fused/traced
graph on GPU. The canonical load path (`load_dmo`, used by the DMO10 replay,
validation/TEST predictions and the final hook) now REBUILDS the architecture
with `build_dmo` and `load_weights` from the saved `model.weights.h5`,
verifying every tensor's recorded digest (typed `WEIGHTS_DIGEST_MISMATCH`) and
file hashes; full-model deserialization remains only a secondary WEIGHTS
compatibility check and never scores. `tools/i6dmo/gpu_probe_dmo10.py` (for
lane 4's bounded GPU scope; logic CPU-tested) scores one identical batch
through in_session / rebuild / full_load in one session and names the diverging
pair. No tolerance. CPU POST: 66 passed across model/train/final/long (PRE: 3
failed). GPU re-verification is lane 4's.

## Round 8: DMO10 canonical monitor partition

With the three-instance probe green, the last comparable-value hazard is batch
COMPOSITION: on GPU a row's float32 forward value legitimately varies with its
batch (GEMM tiling). `evaluate_monitored` is now documented and recorded as the
CANONICAL FIXED SEQUENTIAL partition (row_id order, consecutive batches at the
recorded size) in both the fit loop and the replay; the seeded per-epoch
shuffle remains gradient-only and never reaches monitoring.
`monitor_evaluation.partition = "sequential_row_id_fixed"` is recorded and
`replay_monitored` refuses a permuted or unrecorded partition as typed
`MONITOR_PATH_MISMATCH`. The CPU-only probe fixture now skips when a GPU is
visible. No tolerance. CPU POST: 66 passed (PRE: 2 failed). Lane 4 reruns.

## Round 9: monitor witness + loader-path epoch monitoring (v2)

Ops' artifact diff shows every GPU round trained FRESH (distinct dirs/mtimes,
25 epochs, stop=max_epochs, best=last epoch, no restore involved) with perfect
within-revision determinism, isolating the divergence to the trained-in-session
instance versus a loaded instance on identical final weights. Honest caveats
recorded: the three-instance probe used an UNTRAINED in-session model, so it
could not see a trained-process effect (plausible cuDNN/compilation state
acquired during training altering later in-process forwards); and five
identical floats across rewrites also admits the innocent explanation that the
tiny fixture is single-batch and arithmetically insensitive to every rewrite so
far - the witness decides between "stale path" and "real forward divergence".

Changes (`i6dmo.monitor_eval.v2`, mixed/old records refused):

1. Epoch-end monitoring now evaluates through the CANONICAL LOADER PATH: each
   epoch saves weights and rebuilds via `build_dmo` + `load_weights` into a
   fresh instance (4.8k params; per-epoch overhead measured and recorded in
   `monitor_evaluation.loader_overhead_s`), and `evaluate_monitored` runs on
   THAT instance. The recorded value equals the replay by construction; early
   stopping reads it; self-check and witness stay.
2. Monitor WITNESS recorded at fit time (best epoch) and recomputed at replay:
   sha256 of the exact consumed row-id batch sequence, per-batch per-head
   float64 partial sums/counts, and a code witness (sha256 of the
   `evaluate_monitored` source plus a nonce from the partition parameters,
   computed INSIDE the function). Replay compares batch by batch and reports
   the FIRST divergent batch index with its row-id digest and the pair of
   partial sums (typed `MONITOR_WITNESS_DIVERGENCE`); a witness bound to row
   ids refuses a replay without them.
3. Self-check triviality audit: `monitor_self_check` recomputes on FRESHLY
   materialized input copies through the full path on a fresh loader instance;
   a deliberately corrupted history value fails typed (CPU regression).
4. The end-to-end DMO10 test prints out-dir, started/finished timestamps and
   update/epoch counts so a cached artifact is visible.

No tolerance. CPU POST: 104 passed across the five model-lane files (PRE: 3
failed). One lane-4 GPU round now runs witness + replay together.

## Round 10: static algorithm selection + pre-declared process-level tolerance

Witness verdict on pin 8f8c4c1f: REAL FORWARD DIVERGENCE, process-level —
identical rows/counts/weights, per-head partial sums differing at 1e-4..1e-5,
both sides individually deterministic. Interpretation: cuDNN/cuBLAS autotune
state is per-process; a process that trained selects different (repeatable)
kernels than one that only loads. Fix, in deciding-round form:

1. STATIC algorithm selection forced BEFORE TensorFlow initializes in every
   entrypoint: `TF_CUDNN_USE_AUTOTUNE=0`, `TF_CUDNN_DETERMINISTIC=1`,
   `TF_DETERMINISTIC_OPS=1`, XLA autoclustering off (`--tf_xla_auto_jit=0`,
   `set_jit(False)`), set at `tools/i6dmo/model.py` import and re-forced by
   `enable_determinism()`. All flags recorded in the receipt, MODEL.json,
   PREDICTIONS.json and `monitor_evaluation.determinism_flags`; a record whose
   flags differ from the running process is refused typed
   `DETERMINISM_FLAGS_MISMATCH`. The 4.8k-parameter model makes the
   performance impact irrelevant.
2. PRE-DECLARED tolerance, implemented now behind the flag check so one GPU
   round closes either way: |delta| <= 1e-5 RELATIVE per per-batch per-head
   partial sum (and the same relative bound on the monitored value), written
   justification "process-level GPU algorithm selection, library property".
   It activates ONLY when the static-algorithm flags are recorded and equal
   the running process's flags; tolerated batches are returned and reported,
   never silently absorbed. Reload and prediction identities stay bitwise; on
   CPU the comparison remains exact in practice (no witness fires).

CPU POST: 106 passed across the five model-lane files (PRE: 5 failed,
including flags-enforcement and tolerance-bound regressions). Lane 4 runs the
deciding GPU round: static flags fix it bitwise, or the witness fires within
the declared bound and the tolerance closes it with the deltas on record.

## Round 11: subprocess-path fidelity and the init-ordering witness

Lane 4 found the flags round bit-identical to the pre-flags round. HYPOTHESIS
(reviewer wording ruling: NOT demonstrated - the diff changes no
forward-execution code): the flags were set after TensorFlow had already
initialized, so earlier flag fixes may never have reached the GPU execution
path. The init-ordering witness found the same ordering defect in the
production CLI itself (`train_unit` imported TensorFlow before
`tools.i6dmo.model`), which is a proven import-order fact; whether that
ordering CAUSED the unchanged bits remains the hypothesis. Fixes:

1. `tools/i6dmo/train.py` imports `tools.i6dmo.model` FIRST (the module sets
   the static-algorithm env before importing TF itself), and `train_unit`
   enables determinism before importing TensorFlow.
2. Init-ordering witness: `model.INIT_ORDER` records whether `tensorflow` was
   already in `sys.modules` at the instant the flags were set, plus the flag
   values at that instant; it is stored in `determinism`/`determinism_flags`,
   and `replay_monitored`/the self-check refuse to score any record (or run in
   any process) with `tf_imported_before_flags=true` as typed
   `DETERMINISM_INIT_ORDER_INVALID`.
3. The end-to-end DMO10 test now exercises the REAL subprocess entrypoints:
   train via `python -m tools.i6dmo.train` and replay via the new
   `python -m tools.i6dmo.replay_monitor --design --bundle --unit --json`
   (fresh flags-first interpreters, faithful to unit_runner), asserting
   bitwise MATCH on CPU and `tf_imported_before_flags=false` on both sides.
   `tests/i6dmo/conftest.py` imports the model module before TF so the
   remaining in-process tests run under the production init order.
4. The 1e-5 tolerance stays exactly as declared; any revision would be a new
   transparent declaration, never retroactive.

CPU POST: 23 passed in train (PRE: 2 failed) plus the other four files. Lane 4
runs the next bounded round with governed kernel selection on both sides.

## Round 12: weights-at-witness, row-level forward witness, TF32 off

Three governed rounds with identical sums exclude kernel autotune and init
order. Two recordings now end the hunt in one GPU round:

1. WEIGHTS-AT-WITNESS: the epoch-end witness records the per-tensor weight
   digests of the exact instance it evaluated; the replay diffs them against
   the saved-file digests and a mismatch is typed
   `WEIGHTS_AT_WITNESS_MISMATCH` naming the tensors — proof of a
   post-evaluation weight mutation (optimizer slot/weight-decay/save-path).
   A CPU pass alongside such a mismatch would mean the mutation is GPU-only
   (e.g., a GPU-resident slot or a cast applied on save); the digests decide.
2. ROW-LEVEL FORWARD WITNESS: batch-0 per-row per-head predictions recorded as
   float64 hex at fit-monitor time; the replay records its own and diffs. A
   hex mismatch is a divergence even where sums agree; the typed message names
   the first divergent rows and the delta pattern — UNIFORM ~equal relative
   deltas on every row indict a different matmul/conv execution mode (e.g.,
   TF32 vs FP32), localized rows indict weights/data.
3. TF32 disabled explicitly in `enable_determinism()` (flags-first) and
   `tensor_float_32` recorded in the determinism flags on both sides — the
   prime execution-mode suspect on the 5090, applied now as the fix candidate.

Tolerance unchanged at the declared 1e-5 (any revision would be a new
transparent declaration). CPU POST: 25 passed in train (PRE: 2 failed) plus
the other four files. Lane 4's next bounded round names the mechanism.

## Round 13: input witness + one canonical monitor-input materialization

Lane 4's row witness excluded TF32 and weights (digests match) and showed a
LOCALIZED pattern across all 96 rows (6e-8..1.2e-4): the remaining suspect is
the input tensor itself. Changes:

1. INPUT WITNESS: `evaluate_monitored` digests the POST-NORMALIZATION float32
   bytes it actually feeds — full-split and per-batch sha256, batch-0 per-row
   digests, and the first eight rows byte-complete — before any forward. The
   replay recomputes and diffs them FIRST; a mismatch is typed
   `INPUT_WITNESS_MISMATCH` naming the divergent batches, the first batch-0
   rows and, for the first divergent row, the first differing float elements
   (index, recorded, recomputed).
2. ONE materialization: `materialize_monitor_inputs` (bundle arrays as the
   source of truth; normalization in float64 on CPU; exactly ONE float32 cast)
   is now consumed by BOTH the fit path and `replay_monitor`; the record pins
   `input_materialization = i6dmo.monitor_input.v1:...` and a record declaring
   any other derivation never scores (typed). A deliberately re-derived
   float32-pipeline input is refused by the witness (CPU regression).
3. This would also explain perfect CPU identity if the divergence were a
   GPU-side cast/normalization split: the witness now decides at the byte
   level. Tolerance unchanged at the declared 1e-5.

CPU POST: 26 passed in train (PRE: 1 failed) plus the other four files.

## Round 14: one dedicated monitor forward + grappler remapping off

Inputs proved byte-identical, so only the forward's EXECUTION GRAPH remained.
Both belts, as ordered:

1. The monitor now owns ONE dedicated forward, `train.monitor_forward` — pure
   eager calls on the canonical loader instance over the canonical sequential
   partition, created independently of any training step. The fit-time
   recording AND `replay_monitor` call exactly this function; its identity
   (`i6dmo.monitor_forward.v1` + source sha256, computed by
   `monitor_forward_identity()`) is recorded in `monitor_evaluation` and inside
   the witness, and a record produced by any other forward (e.g., a
   training-step graph) is refused typed `MONITOR_FORWARD_MISMATCH`.
2. Grappler remapping is disabled flags-first in `enable_determinism()`, with
   the sibling fusion passes judged relevant (`layout_optimizer`,
   `auto_mixed_precision`) also off; all three recorded in the determinism
   flags and enforced like the others (XLA autoclustering was already off).

Tolerance unchanged at the declared 1e-5. If lane 4's round STILL diverges
with same bytes, same weights, same function and remapping off, Satoshi has
pre-approved a transparent bound revision as a NEW declaration at the measured
maximum — to be recorded here only if that branch is taken.

## Round 15: declared monitor comparison floor (tolerance v2) — library floor reached

Satoshi approved the transparent bound revision as the single final round. The
honest endpoint, declared prospectively:

- Scope: ONLY the cross-process GPU comparison of the monitored value and the
  per-batch per-head float64 partial sums. Bound `|delta| <= 1e-9 + 5e-5*|recorded|`,
  calibrated from the recorded witness distribution (observed max 1.82e-5)
  with margin against flapping yet far below any scientific effect.
- Everything else stays BITWISE: CPU end-to-end, the in-session self-check
  (which now refuses any non-bitwise outcome by declaration), reload identity,
  saved-vs-reloaded predictions, input witness bytes, weight digests, forward
  identity and determinism flags.
- The row-level hex forward witness remains recorded as a DIAGNOSTIC: reported
  in `row_forward_diagnostics`, never pass/fail at row level.
- Justification, written into the effective config
  (`training.monitor_comparison_floor`) and here: eight rounds excluded
  cuDNN/cuBLAS autotune and init order, TF32, weights-at-witness digests,
  post-normalization input bytes, the dedicated forward's identity, and
  grappler remapping/layout/auto-mixed-precision; the remaining deltas are
  bit-stable within each side and are a property of in-session vs
  fresh-instance GPU execution contexts of this TensorFlow build. Applied
  prospectively: no scientific fit has run under the corrected protocol and no
  past run is re-graded.
- Typed outcomes: `WITHIN_DECLARED_FLOOR` (reported with max deltas and the
  allowance) vs `MONITOR_WITNESS_DIVERGENCE` beyond the floor (still hard);
  the floor activates only with the static-algorithm flags recorded and equal
  on both sides. The effective-config sha changes once more with this
  declaration (design sha unchanged).

## Round 16: Musashi numerical-replay corrections (binding)

Per `MUSASHI_TO_SATOSHI_DMO10_NUMERICAL_REPLAY_20261014.md` (sha cba02e7e),
with the owed retraction first: the round-15 "library floor" framing was an
OVERCLAIM. Exclusion of eight hypotheses is not proof of cause; the observed
discrepancy is specific to this TensorFlow build and these execution contexts;
autotune/trace/stream selection remains a HYPOTHESIS pending the controlled
isolation. The retained 8f8c4c1f witness is not bounded by 1e-5 (three heads
exceed it, max 1.8244613879748877e-5) and KEEPS its original FAIL: records
without the sealed protocol can never inherit the new rule (tested).

1. Sealed protocol identity (`floor_protocol_identity`,
   `i6dmo.monitor_numeric_equivalence.v2`): formula
   `abs(actual - reference) <= atol + rtol*abs(reference)` with the reference
   direction (recorded fit-time value), rtol 5e-5 / atol 1e-9, scope, dtype
   float64, label NUMERICALLY_EQUIVALENT (never BITWISE/PASS), required flags
   and runtime versions; a different or absent identity is typed
   (`MONITOR_PROTOCOL_MISMATCH` / hard divergence).
2. Scope PROVEN, not inferred: the record carries the recording device and
   process identity; the floor activates only for a GPU-recorded monitor
   replayed in a DIFFERENT process with a GPU visible. CPU records, same-process
   corruption, near-zero sums, boundary and over-bound are all covered by
   regressions; a GPU record replayed without a GPU is the separately typed
   `CROSS_DEVICE_NOT_COMPARABLE` (5090==4090 is never required).
3. Monitor forward tensors remain recorded DIAGNOSTICS and are never claimed
   bitwise; exported checkpoint reload/serving prediction equality stays EXACT
   in its named context (canonical loader, same process).
4. Decision impact is REPORTED, never claimed invariant:
   `stopping_decision_impact` (near-tie negative control at the measured
   1.82e-5 relative uncertainty vs min_delta 1e-8) is recorded per unit in
   `monitor_evaluation.decision_impact`; training min_delta/patience/selection
   untouched.
5. ONE bounded diagnostic, `tools/i6dmo/dmo10_isolation.py`: freeze (digests
   to Git, arrays local), element-by-element prediction diff before any
   reduction (tensor hashes, first differing element, abs/rel/max/mean),
   independent fsum reproduction, and a controlled flags-on/off comparison in
   fresh processes that can SUPPORT or NOT the autotune hypothesis — honestly
   NOT_INFORMATIVE_ON_CPU on this host; same-environment vs cross-device
   recorded separately. Reuses the retained checkpoint; no production fits.

Statuses renamed: EXACT / NUMERICALLY_EQUIVALENT (WITHIN_DECLARED_FLOOR and
"bitwise" claims removed from verdicts).

## Round 17: reviewer wording items (final pre-release identity)

Reviewer verdict on a33cfe70: ACCEPT_FOR_RELEASE with two non-blocking wording
items, both applied as the last pre-release change:

1. Round 11's stale-path explanation is now labelled a HYPOTHESIS (the diff
   changed no forward-execution code; the import-order defect is a proven fact,
   its causal role in the unchanged bits is not demonstrated).
2. `training.monitor_comparison_floor.justification` in the sealed effective
   config no longer says "library floor": it states a reproduced,
   build/context-specific numerical discrepancy, cause not demonstrated,
   bounded by the versioned NUMERICALLY_EQUIVALENT protocol.

FINAL effective config sha (the identity the scientific fit runs under):
`287e4f11e4ef385234becf02d9ed31864d8a3a58fc0b4c7f3e4c6a765a76e097`
(design sha unchanged `541ce40a…`). Nothing else changes after this commit.

## Round 18: paired short HOURLY-input control (prepared, NOT fitted)

Per MUSASHI_TO_SATOSHI_STATIC_BASELINE_CLOSE_AND_MARKET_GRID_20261015 section 3
(preparation only; no fit, no GPU, CPU tests only):

- `EXPERIMENT.short_hourly.json` is sealed and committed:
  `I6-DMO-SHORT-HOURLY-DS1-STATIC-R0`, the same 24 elapsed-hour span with 25
  observations at hourly spacing including t, six heads 2..12, same model
  recipe, same static cadence and splits, its OWN identity (design sha
  `971ba9c4…`). Derived deterministically from the validated short design by
  `seal-hourly-control` (typed refusals: FIXED_VALUE, HOURLY_DESIGN_SEAL,
  HOURLY_DESIGN_UNDECIDED, TEST_FORBIDDEN; idempotent; a final variant of the
  control is refused as a separate later decision).
- The ONLY forced model adjustment is declared: each head flattens a 25x4
  instead of a 13x4 core sequence (hidden Dense input 100 vs 52); branches
  (4/3/1), core (8->4) and all parameter counts outside the heads are
  identical (tested). The causality probe gained ReLU-aware retry seeds
  (look-ahead violations still fail on the first seed).
- TRAIN/validation availability differences are DECLARED inside the sealed
  design (`derived_from.availability_declaration`): hourly exact slots across
  weekend-adjacent windows differ from the two-hour grid; the per-split counts
  are owned by the data lane's market-grid support study and are not
  duplicated here.
- `COMMON_SUPPORT_RULES.json` (sha `9d61117f…`) is frozen BEFORE any scoring
  and bound into the sealed design: comparison population is the intersection
  of valid (row_id, head) pairs under both designs; shared naive on exactly
  those rows; a reused model scored on the subset is labelled
  REUSED_MODEL_SUBSET_SCORE and is NOT a model trained on the common TRAIN
  population; encoder/R2 and cadence contrasts remain distinct later sealed
  designs, never combined and attributed to downsampling.
- The short design's FINAL effective sha `287e4f11…` is unchanged (verified).

## Round 19: market-time pair MODEL ownership (order 1.2) - prepared, no fit

Plan sections 2/3/6 treated as FROZEN. CPU only, no fit, no GPU.

1. NEW typed schema `i6dmo.market_time_design.v1`, separate from
   `i6dmo.design.v1`: a `market_time` block (rule, observations, spacing and
   span in OPEN-MARKET hours, include_origin, `elapsed_span_fixed: false`,
   `require_exact_market_slots: true`). `history_span_hours` keeps its elapsed
   meaning in the old schema and is ABSENT here; any fixed-elapsed field
   (`input`, `history_span_hours`, `spacing_hours`, `observations`,
   `offsets_hours`) alongside `market_time` is typed
   `MARKET_TIME_CONTRADICTION`. Sealed designs (committed):
   - `EXPERIMENT.mg2.json` I6-DMO-SHORT-MG2-MARKET-EXACT-R0 design sha
     `169fbece…` seal `e7c2f3e7…` effective `5798baaa…`
   - `EXPERIMENT.mg12.json` I6-DMO-LONG-MG12-MARKET-EXACT-R0 design sha
     `8c10a9ca…` seal `ef562ba4…` effective `0b1b63e3…`
   The seal binds the time-encoder formula identity, feature manifest,
   availability contract, calendar (ASSUMED_NOT_BROKER), the data lane's
   candidate + support-study digests, seed 6102026 and runtime versions.
   `seal-market-time` is deterministic and idempotent; a market-time design is
   refused as a base for the final/hourly derivations.
2. The ONLY model extension: `DynamicPhysicalTimeEncoder` (serializable,
   version 1.0.0) immediately after branch fusion. It applies the same
   deterministic sinusoidal rule to the ACTUAL signed UTC offsets in hours
   (asserted different from the ordinal-index code, and equal to the plugin
   formula evaluated at those offsets), then appends exactly two context
   channels before the core: adjacent elapsed-hours/24 (fixed unit conversion,
   never fitted) and closure-crossed 0/1, with slot 0 delta 0 / flag 0 and the
   last offset exactly 0. With 19 branches the core input is 76 + 2 = 78.
   Per-example time inputs are mandatory (a fixed per-batch grid is refused),
   and model metadata records that slot order is not wall-clock time.
   Controlled-weight tests prove the actual time inputs reach the output
   (a 50 h weekend gap yields 50/24 + 1 against 2/24 for a regular window).
3. Everything else is the frozen recipe (branches 4/3/1, core 8->4, six
   Flatten->Dense(8,relu)->Dense(1,linear) heads, AdamW 1e-3/1e-4, batch 256,
   200 epochs, patience 10, min_delta 1e-8, equal-head masked normalized MAE,
   best-weight restore, mean end-of-epoch FIT/INNER monitor), and the
   already-reviewed numerical policy is UNCHANGED (rtol 5e-5 / atol 1e-9,
   `i6dmo.monitor_numeric_equivalence.v2`; DMO10 not reopened).
4. Old DS2 behaviour preserved under its own config: a replay test asserts the
   elapsed short/long/hourly families still build single-input models with the
   fixed-offset encoder, 76-channel core input and 4,834 parameters; the short
   FINAL effective sha `287e4f11…` is unchanged. Serialization/reload covers
   ALL inputs including the time/closure channels; identity refusals are typed
   (`MARKET_TIME_INCOMPATIBLE`, `WEIGHTS_DIGEST_MISMATCH`).

## Round 20: review B.1 repairs - the sealed market-time identity can now fit

Reviewer verdict FIX_REQUIRED on `96d4787b`: the temporal and metric paths all
PASS, but three seams refused the sealed identity. All three repaired against
the REAL seams (tests first, CPU only, no fit, no GPU, no TEST):

- **B-1 (RB04).** `design.effective_config()['model']` now builds:
  `normalize_model_config` accepts the sealed block's `offsets_hours: None`
  (refusing any non-null fixed grid) and the sealed `time_encoder` identity
  (verifying layer, version, the 24 h unit conversion, `fitted: false` and the
  two context channels), and is byte-equal to the `model_config(...,
  family=...)` positive control. Both sealed effective configs build 3-input
  78-channel networks.
- **B-2 (RB03 J1).** The trainer consumes market bundles: `load_data_bundle`
  routes market designs to a market loader that reads the declared schema
  BEFORE body verification (so a DS2 bundle is a typed `BUNDLE_SCHEMA`, not a
  digest error), authenticates through the data lane's
  `load_market_manifest` / `load_market_split` (rule and data-identity), and
  re-applies every model-side boundary check. `market_time_inputs` assembles
  `[observations, offsets_hours, closure_crossed]`, and the fit loop, monitor
  forward, witnesses and export paths slice all inputs consistently. The REAL
  CLI now completes on both sealed designs against REAL bundles built by the
  data lane from synthetic inputs (48-row subset, 2 epochs).
- **B-4 (RB02 E2).** Ordinal or impossible time inputs are now CAUGHT at both
  seams (encoder `tf.debugging.Assert` and the trainer assembly) as typed
  `ORDINAL_OR_INVALID_OFFSETS`: last offset exactly 0, all offsets <= 0,
  increasing towards the origin, adjacent elapsed hours >= the design's
  `spacing_open_market_hours` (so `arange(-12..0)` is refused), first adjacent
  delta 0, first closure flag 0. The reviewer's two accepted cases are
  regressions.

Because the encoder identity gained those guards, both designs were RESEALED:

| | design sha | seal sha | effective sha |
|---|---|---|---|
| MG2 | `1fa48222…` | `4b64b4ee…` | `920332ae…` |
| MG12 | `4da1d3e0…` | `97541764…` | `0853f6d9…` |

(superseding the pre-review `169fbece` / `8c10a9ca`). The elapsed DS2 path,
the frozen recipe and the numerical policy are untouched; the short FINAL
effective sha `287e4f11…` is unchanged. B-3 (campaign `_design_checks`
hard-requiring `i6dmo.design.v1`) is the ops lane's seam.
