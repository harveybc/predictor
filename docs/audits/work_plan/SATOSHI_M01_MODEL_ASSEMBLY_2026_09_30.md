# M01 — model assembly: return

Satoshi, successor technical lead · 2026-09-30
Orders: `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` §3–§4 (master `dc72170e`).
Plan: `docs/tres_temas_entrevista/program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md`, rows MS01–MS08.

```
M01 — complete hierarchical predictor behind one opt-in plugin
repo/branch/tip: predictor · satoshi/m01-model-assembly-20260930 · tested code tip 64a91a74
                 (the commit that adds this document and its evidence changes no code)
base: 556c5f3e (codex/modular-stack-20260930; engine 1b76981a, evaluator 1b9daf6c, pretraining 621570e1, all adopted)
files: predictor_plugins/modular_temporal.py (engine, extended) · predictor_plugins/modular_config.py (new)
       predictor_plugins/predictor_plugin_modular.py (new facade) · setup.py (5 entry points added)
       tests/test_modular_assembly.py · tests/test_modular_legacy_boundaries.py · tests/test_modular_flat_parity_m04.py
       tests/test_modular_temporal.py (one test adapted to component declarations)
       tests/fixtures/{legacy_entry_points_dc72170e.json, m04_3ceabfad/*}
       tools/m01_verify_installed.sh · tools/m01_verify_lts.sh · tools/m01_boundary_probe.py · tools/m01_legacy_e2e.py
       examples/config/modular_temporal/phase_1_modular_temporal_smoke_config.json
suites (CPU, worker_b, crispdm-run 4G, all at 64a91a74):
  installed isolated venv (Py 3.12.13, TF 2.21.0, Keras 3.13.2, no system/user site; package installed
  non-editable; tests run from a copy outside the checkout, M01_REQUIRE_INSTALLED=1):  69 passed · 1 xfailed · 0 skipped
  campaign env envs/tensorflow (Keras 3.13.2), checkout run:                           68 passed · 2 skipped (installed-only checks)
  LTS b05d01d broker/CSV/live-selection routes: baseline env 21 passed · same env + M01 installed 21 passed
acceptance: MS01–MS08 PASS with behavioural evidence (MS02 model side only, see below);
            every legacy boundary holds in an installed environment (master vs M01 probe below)
what is NOT done / refused / not measured: see the last two sections
```

The suites are **synthetic component checks, not forecasting results**. No number in this
return says anything about forecast quality.

## MS rows

| Row | Verdict | Behavioural evidence (test in `tests/test_modular_assembly.py` unless named) |
|---|---|---|
| MS01 plugins | PASS | Built-ins resolve through `modular.branch/fusion/core/head` entry points. Each carries `component()` metadata: role, semantic version, parameter names, tensor/time contract and defaults. An unknown name, an undeclared factory, a name published twice, or an entry point that reuses a built-in name with another factory all fail. In the installed venv each `modular.*` entry point loads the very built-in object. Tests: `test_builtins_resolve_with_versions_and_unknown_names_fail`, `test_external_entry_points_undeclared_ambiguous_and_shadowing_fail`, `test_modular_components_resolve_through_their_entry_points`. |
| MS02 ≥24 h | PASS (model side) | `sample_hours` is explicit. 24 one-minute samples (0.4 h) and 24×15 min (6 h) are refused before fit. 48 half-hour steps build with the grid's right edge at 24.0 h. Timestamp and gap checking of the actual rows belongs to data preparation and is **not** in this lane. |
| MS03 time preserved | PASS | Every layer of branch, fusion and core outputs rank 3; there is no Flatten or pooling before the head. The time alignment test is behavioural: perturbing input position *i* first moves branch step *i*//2 and latent step *i*//4, never an earlier one, for every branch and the encoder. `probe_alignment` runs at build. Two same-shape defective plugins are refused at build: one reverses time, one drops the newest sample. Tests: `test_time_alignment_is_behavioural_on_the_common_right_edge_grid`, `test_same_shape_but_misaligned_plugins_are_rejected`. |
| MS04 different branch archs | PASS | Each branch has its own plugin and parameters: kernels 3/3/2, widths 4/3/2. The CLI smoke ran 7 branches mixing 16ch/k3 and 4ch/k1. |
| MS05 full Transformer | PASS | Positional encoding, projection to 64, two blocks of 4-head attention plus FFN, each with residual and LayerNorm. Blocks, heads, width and dropout are configurable, and a flat override `core.params.blocks=1` changes the built graph. |
| MS06 learned bottleneck | PASS | The default is 3 learned stages to (6, 8). A 4-stage schedule builds when factors divide. A non-dividing factor, mismatched stage lists or non-decreasing widths fail; nothing is trimmed or padded. Reconstruction and utility quantification is MS10/MS11 (M02). |
| MS07 R0/R1/R2 | PASS | `test_three_arm_regimes_observe_actual_optimizer_updates`, against donors pre-trained one AE step so they differ from any fresh init. R0 starts off-donor. R1 and R2 start from weight hashes identical to the donor. Updates are counted from the optimizer (32 rows / batch 8 = 4 per epoch; `optimizer.iterations` delta equals observed updates). R1: branch and core have 0 trainable params and their weights equal the donor after fit, while the head moves. R0 and R2: every component's weights change. Mixed regimes report `MIXED`, and a component that contradicts the declared common regime fails. |
| MS08 bad donors | PASS | Nine explicitly requested bad donors are refused inside `build_model` with a spy on the fit helper at 0 calls: missing file, archive digest, manifest digest, feature order, branch feature swap, window/time shape, sampling period, upstream identity (fresh branches under a pretrained core), and a core donor for other branch params. |

Also delivered beyond the rows:

- **Effective-parameter identity** (M04 defect 1). Donor identity hashes declared defaults resolved, so `{}` and explicit defaults (including `dropout: 0` vs `0.0`) are one identity. The literal params are kept in the donor sidecar as provenance. Any changed effective value is refused. Test: `test_implicit_and_explicit_defaults_share_one_identity`.
- **Keras pin** (defect 2). `bundle.json` and every donor sidecar record `keras_version`. `load_bundle` and `load_donor` refuse a major.minor mismatch before deserializing. All suites and bundles here are Keras 3.13.2. Test: `test_keras_major_minor_mismatch_is_refused_before_deserialization`.
- **Serialization.** `save_bundle`/`load_bundle` rebuild the architecture from the canonical config, check the archive and weight digests, and require output parity with the archived graph; R1 trainability survives reload. The facade's `save`/`load` reproduce predictions, a tampered archive is refused, and plain `keras.models.load_model` on the facade archive also reproduces them. Test: `test_facade_save_load_reproduces_outputs_and_detects_tampering`.
- **Versioned nested config with reversible flat mapping.** Schema `predictor.modular.v1`, canonical JSON. `flatten`/`unflatten` are exact inverses. Unknown branch, field or malformed flat keys fail.
- **Conditional parameters fail in `build_model`.** Nine invalid combinations, plus disagreements between the legacy keys `window_size`, `predicted_horizons` and `feature_names` and the input channel count. Tests: `test_invalid_combinations_fail_in_build_model_before_fit`, `test_legacy_keys_must_agree_with_the_modular_config`.

## Flat grammar vs M04 (coordinator ruling recorded verbatim in `modular_config.py`)

M04's `tools/modular_search_space.py` is a tied search-space projection (`modular.candidate.v1`). M01's `modular_config.py` is the complete reversible encoding of `predictor.modular.v1`. They are layers that compose. **The facade does NOT accept M04's keys directly. The only integration path is M04 `from_flat` → nested candidate → its `model` → the facade's `modular` config.**

The shared `core.` prefix cannot shadow: M01 accepts only `core.plugin|regime|donor` and `core.params.<p>`, so a raw M04 key such as `core.d_model` fails loudly instead of being ignored. M04's `branch.`, `model.` and `train.` keys are outside M01's namespace.

`tests/test_modular_flat_parity_m04.py` pins this on M04's fixtures at `3ceabfad`, with sha256 recorded in `SOURCE.json`. Each M04 model key lands value-exact at its location in M01's grammar, at grouping sizes 1 and 3; both round trips hold; `train.*` is absent. M04's tip at return time, `05430e51`, has byte-identical fixtures, so no re-pin was needed.

## Legacy boundaries in an installed environment (master `dc72170e` vs M01, same isolated venv)

Driver: `tools/m01_verify_installed.sh`. Evidence: `docs/audits/evidence/m01_model_assembly_20260930/final/`.

- **Entry points.** Every legacy name and value in every group is byte-identical; nothing was removed. The only additions are `predictor.plugins: modular_temporal` and one built-in in each of `modular.branch`, `modular.fusion`, `modular.core` and `modular.head`.
- **Old JSON configs.** All 224 legacy example configs resolve identically: predictor, pipeline, preprocessor, target and optimizer, by value and by module. The 225th is the new opt-in smoke config; master refuses it as an unknown name, as it should.
- **Unmigrated old config.** Master refuses it (C-series column roles) with the same message under M01.
- **Result schema.** The legacy e2e run (`phase_1_ann_1575_1d`, 2 epochs, outputs redirected) produces an identical results CSV header, identical 90 ordered metric labels and an identical output file set.
- **prediction_provider.** Its `DirectionPredictor._load_direction_model` was run on the committed direction `.keras` + `_metadata.json` pairs (prediction_provider `78f0af5`). The route and outcome are identical under master and M01.
- **LTS.** b05d01d `test_backtrader_broker`, `test_csv_workflow_e2e` and `unit/test_live_model_selection` ran in the trading-stack env: 21/21 as is, and 21/21 in an overlay venv with M01 installed. LTS routes import nothing from predictor; this proves the top-level `predictor_plugins` name collision does not change them.

### Modular plugin through the unchanged legacy CLI

`app/main.py` → `stl_pipeline` → `modular_temporal`, on the phase-1 4-hourly data:

- Seven preprocessor channels, one branch each, `sample_hours: 4`, window 24 = 96 h, horizons 9–24.
- Exit 0. It wrote the legacy results CSV with the identical 90-label schema, predictions, plots and `model.keras`, plus `model.keras.bundle/` (bundle.json and training_receipt.json).
- Receipt: 16 observed updates equal 16 optimizer iterations, 2 epochs, stop `max_epochs`, every component's weights changed (R0), Keras 3.13.2.
- The facade first refused the run with `Input shape (24, 7) does not match (24, 1)`, because the declared feature list was incomplete. That is the intended behaviour: no silent adaptation.
- Smoke metrics (model MAE ≈10× naive after 2 epochs on 253 rows) are plumbing only and **not a result**.
- The legacy pipeline also scored its 300-row test slice, as it does for every plugin. That number was not used for anything.

## Incident (self-reported, resolved)

My compatibility smokes on worker_b wrote 11 OLAP envelopes into the shared `~/.local/share/predictor/olap_outbox/pending/`, between 19:34 and 19:52 host time. Legacy `app/main.py` emits an envelope on every exit path. None had been drained: there were no matches in `loaded/`, `failed/` or `receipts/`.

All 11 were moved (not deleted) to worker_b `~/.local/state/scratch/m01/olap_quarantine/`. Per the coordinator's ruling they stay there and never return to pending. Fix: `342f1fc5` points `CRISPDM_OLAP_OUTBOX` at each smoke's own output directory. After the fix, the pending queue stayed at 0 through the final runs.

Envelope ids (sha256 prefixes of the file names):

- `1cccd4242fc3`, `be313f48aa55`, `ee0ec039849f` (verify4/5, master)
- `76362bb2349a`, `9182701718ed` (verify5, M01)
- `73b99ec0e021`, `ab1b7e1faf11` (verify6, master)
- `cbe548888cc7`, `de4b22c22a79` (verify6, M01)
- `555f155717df`, `66de04556e42` (first modular smoke, REFUSED on the legacy `plugin` alias)

Earlier, on the coordinator, the admission monitor stopped my `m01-venv` pip install under sustained host memory pressure. It is recorded as INTERRUPTED_RESULT_NOT_RETAINED. Since then nothing above 1G runs on the coordinator, and every venv and pytest basetemp is on disk, never tmpfs. My only tmpfs residue was `/tmp/pytest-of-harveybc/pytest-12`, 2,987,725 bytes, now deleted. `/tmp/tmp3spjz0ia` (a partial torch wheel) is not mine and was left alone.

## Findings for the owner (pre-existing on master, not changed in this lane)

1. **Packaging gap.** `predictor_plugins/common/` and `olap/` have no `__init__.py`, so a non-editable install omits them. Every legacy Keras plugin then fails to import when installed (`No module named 'predictor_plugins.common'`), and so does prediction_provider's direction rebuild. The installed `app/main.py` cannot run (`config_merger` and `olap` not importable). This is identical on master and M01; it is the 1 xfail. Fixing it would change the prediction_provider route, so it needs its own decision.
2. **OLAP emission from ad-hoc runs.** Any run of the legacy CLI, by anyone, writes an envelope into the shared outbox unless `CRISPDM_OLAP_OUTBOX` is set, which pollutes the queue with smokes.
3. **Shared names across distributions.** prediction_provider also publishes `predictor.plugins: default_predictor`. Its loader takes the first match, so co-installing both packages makes that name order-dependent. `modular_temporal` collides with nothing.

## What is NOT done / limits

- **Not in this lane:** MS09–MS15 (early stopping is reused from the inherited evaluator, not re-audited); timestamp and gap validation of actual rows (MS02 data side); any forecasting, reconstruction or utility measurement.
- **Uncertainty.** The facade is deterministic, and `predict_with_uncertainty` returns zeros. The legacy `Uncertainty` and `SNR` columns are therefore meaningless for this plugin (SNR divides by zero spread).
- **Legacy-derived training settings** must lie within the shared fit loop's bounds (`max_epochs` ≤ 100, `patience` ≤ 50) and fail otherwise rather than being clamped. Legacy configs with `epochs: 10000` must declare `modular_training` explicitly. `threshold_error` and `mc_samples` are ignored.
- **External plugins** without declared defaults keep a literal-params identity, as documented in `effective_params`.
- **Alignment probe.** It requires causality inside the core as well, so a core with within-window bidirectional attention would be refused. That is stricter than leakage requires, and deliberate for now.
- **Interface change for M02 and M05.** Donors saved before `c1e035d7` (literal-params manifest, no Keras provenance) are refused by the new identity, by design.

## Commits

```
feedfd00 0111c264 cf7f0297 05b08dcd c1e035d7 8f38bfeb f04858a6 48a1bd85 ec662fd3 612ddf4b
7f12666f 7538c60b 7bf5de4c 08c0d37a 05af47ab 342f1fc5 64a91a74  (+ this document and evidence)
```

Re-run: `tools/m01_verify_installed.sh <isolated-venv-python> <M01 worktree> dc72170e <prediction_provider> <out>` and `tools/m01_verify_lts.sh <lts-env-python> <lts> <M01 worktree> <out>`, each under `crispdm-run -m 4G` on a CPU worker.

— Satoshi, successor technical lead, 2026-09-30
