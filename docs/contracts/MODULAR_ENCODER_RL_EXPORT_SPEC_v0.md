# Modular temporal encoder export for RL (SAC / DQN): adapter specification v0

Satoshi, successor technical lead · 2026-10-01 · orders b327b771 §3 row M01, §6, RL subplan RL02–RL05.

**Status: SPECIFICATION ONLY. NOT IMPLEMENTED.** The adapter is written only after lane G (the RL owner) states the installed algorithm interfaces. Specifically, lane G must state the following for agent-multi/gym-fx:

1. the SAC and DQN implementations, with their package, version and policy class;
2. the feature-extractor hook (an SB3-style `features_extractor_class` or equivalent);
3. the optimizer ownership of each network;
4. the target-network update code path;
5. the observation layout;
6. the save/load API.

The open questions in §7 are addressed to lane G.

## 1. What predictor exports

The export is one object per candidate: `ModularEncoderExport`. It is built from an engine bundle at the campaign pin (3ecdb256 or later) and contains:

| Field | Content |
|---|---|
| `encoder` | `bundle.encoder_model`: observations `(B, window, F)` → latent `(B, output_steps, output_channels)`, default `(B, 24, F)` → `(B, 6, 8)`. Rank three; time is preserved through branches, fusion and core (RL02). |
| `feature_order` | `config.feature_names`, exactly. The RL observation builder must supply channels in this order, and a mismatch is refused (RL01). |
| `time_contract` | `sample_hours`, `window`, the right-edge input grid and `core_time_grid`. The last input step is the decision time t. |
| `latent_contract` | `{"layout": "temporal", "shape": [6, 8], "grid": core_time_grid}`. Any reduction to a vector happens in an **explicit task-head adapter** (§3), never inside the encoder. |
| `identity` | `config_digest`; branch and core donor manifests (effective params, versions); the objective identity of any pretrained donor; donor provenance (item 6: conditioning contract, learned corpus, reconstruction state); Keras version. |
| `regimes` | Per component R0, R1 or R2 (`regime_summary`), declared separately for branches and core. |
| `scaler` | The feature normalization fitted on the TRAIN episodes only, as an explicit artifact with its fit rows. The encoder never refits it. |
| `budget` | `measure_budget(config)`: parameters per branch/core, fused width, bytes per row. Used in the parameter and compute disclosure against the baselines. |

**Refusals at export:**
- A donor whose `conditioning_contract` is not OPERATIONAL is refused (`load_donor(..., require_contract="OPERATIONAL")`). UNKNOWN is never served.
- A latent that is not rank three is refused.
- A feature order that differs from the declared selection manifest is refused.

## 2. Weight-sharing and optimizer ownership (RL03), declared per candidate

Lane G chooses one mode per candidate. The export carries the choice, and the tests check it.

| Mode | SAC | DQN |
|---|---|---|
| `separate` (default) | Actor and each critic get their own encoder copy, initialized identically from the same donor/seed. Each network's optimizer owns its copy. | The online network has its encoder copy. The target network gets a copy synchronized by the DQN target rule. |
| `shared_critic_owned` | One encoder updated only by the critic loss. The actor consumes it with gradients stopped (`tf.stop_gradient`, or the PyTorch equivalent if the RL library is torch; see §7). | Not applicable (one online network). |
| `frozen` | Every network shares one non-trainable encoder (R1). No optimizer contains its variables. | The same; the target copy equals the online copy forever. |

Targets:
- SAC target critics hold a **separate** encoder copy, updated by the same Polyak rule as the critic heads.
- The DQN target network holds a separate copy, synchronized at the declared interval. Encoder identity before and after each sync is recorded (RL04).

**No accidental double updates.** A variable may appear in exactly one optimizer's variable list. The adapter asserts this at construction and fails otherwise.

## 3. Task-head adapter (the only place time may be reduced)

`latent (B, 6, 8)` → `head_input`. Allowed choices, declared and versioned:
- `flatten`: 48 values, the default, matching the forecast head;
- `last_step`: `(B, 8)` at the decision time;
- `attention_pool`: learned, with its parameters counted.

The choice is part of the candidate identity. Policy and value heads are the library's own MLPs on `head_input`.

## 4. Regime contracts (RL04), checked behaviourally, not by flags

- **R0:** fresh initialization. All encoder variables update under their owning optimizer. Checked by comparing weight hashes before and after the first k updates.
- **R1:** loads the verified donor with OPERATIONAL provenance. Its weights hash is identical before and after training, it has 0 trainable encoder parameters, and no optimizer holds its variables.
- **R2:** starts from the same donor (identical starting hash) and its weights change under the owning optimizer.
- **Target copies:**
  - DQN: after a sync, target hash = online hash; between syncs the target hash is unchanged.
  - SAC: target hash = τ-mix of the online and previous target, verified on a fixture.

## 5. Save / reload (RL05)

The policy checkpoint stores the export identity, the encoder weights of every copy, the scaler, the head adapter and the action mapping. Reload in a fresh process:
1. rebuilds from the stored config;
2. verifies every weights hash and the Keras version (major.minor);
3. reproduces deterministic actions on a fixed observation batch within the declared device tolerance.

## 6. Baseline arms (RL-S0 / RL-D0)

The native baseline receives the same `feature_order`, window, scaler and information. Its architecture is the library's declared default, documented with its parameter count. It is never labelled modular. The export disclosure reports the parameter and compute differences and the pretraining cost (donor fits) against it.

## 7. Questions for lane G, to answer before implementation

1. Which SAC and DQN implementations are installed in the target environment (library, version)? Are they TensorFlow/Keras or PyTorch? If PyTorch, the Keras encoder needs either a verified weight-port with a numerical parity test, or a library-native re-implementation of the same graph with parity. This choice must be made explicitly, not by silent conversion.
2. Which feature-extractor hook is available, and does it allow distinct extractors for actor and critic?
3. Is the action space discrete for both arms? Is a tested discrete-SAC available for the shared discrete comparison?
4. What is the observation layout: a flat vector, or a `(window, F)` tensor? What normalization does the environment already apply?
5. What is the target-network update API (DQN hard sync interval; SAC τ)?

## 8. Tests to write first (once the interfaces are stated)

- feature-order mismatch refused;
- non-OPERATIONAL donor refused;
- pooled latent refused;
- every variable in exactly one optimizer;
- R0/R1/R2 weight-hash behaviour per network;
- target sync preserves encoder identity;
- fresh-process replay reproduces actions;
- baseline receives identical information.

These are synthetic mechanics; they establish no financial performance.

## 9. Verification against lane G's implementation (2026-10-01)

**Evidence reviewed:**
- agent-multi 02db0701: `rl_temporal/{keras_import,modular_torch,donor_contract,arms,checkpoint}.py`, tests RL01–RL08 (suite 42 passed on worker_a at 3G), and `docs/audits/evidence/g_rl_20261001/fixture_encoder_export.npz.receipt.json` (63 arrays, sha `a04c7e71…`, Keras 3.13.2, source `RANDOM_INIT_NOT_PRETRAINED`);
- predictor `tools/export_modular_encoder_npz.py` (600626c8, now carried on this branch).

| Spec item | Status | Evidence |
|---|---|---|
| §7 interfaces stated | SATISFIED | SB3 2.9.0 on torch 2.13; flat Box observation with an `ObservationLayout`; SAC τ 0.005, DQN hard sync 1000 |
| Weight port with numerical parity | SATISFIED on the fixture | `test_keras_bundle_weights_import_with_fidelity`: fused atol 1e-5, latent atol 1e-4, `unmatched_keras_weights == []`, config sha bound |
| Feature-order mismatch refused | SATISFIED | the import refuses a modular-config sha mismatch, and feature order is in the config |
| Every variable in exactly one optimizer | SATISFIED | RL03 `test_a_gradient_step_updates_each_extractor_parameter_at_most_once`; `optimizer_ownership()` |
| R0/R1/R2 by weight hash | SATISFIED | RL04 r1-fixed / r2-moves / donor refusals |
| Target copies (DQN sync, SAC Polyak) | SATISFIED | RL04 `test_dqn_target_sync…`, `test_sac_target_critic_tracks…` |
| Fresh-process replay | SATISFIED | RL05 subprocess round-trip, atol 1e-6, plus action-mapping refusal |
| Baseline receives identical information | SATISFIED | arm configs; `PARAMETERS_MEASURED.json` discloses parameters from built models |
| §3 explicit task-head reduction | SATISFIED | RL02 (time preserved through branches/fusion/core; the reduction is explicit in the extractor) |
| §2 weight-sharing modes | SATISFIED with a declared difference | Lane G uses SB3's `share_features_extractor`. Under SB3 SAC sharing, the shared extractor is updated by the actor loss and excluded from the critic optimizer: **actor-owned**, not the spec's `shared_critic_owned`. The spec now accepts `shared_actor_owned` (SB3 semantics) as the sharing mode, provided it is recorded per run, as RL03 already does. `separate` and `frozen` (R1) match. |
| §1 non-OPERATIONAL donors refused at export | **GAP, closed here** | The tool loaded bundles without a contract requirement. It now calls `load_bundle(require_contract="OPERATIONAL")`; `--allow-unknown-provenance` is an explicit, recorded bypass. |
| §1 export identity | **GAP, closed here** | The export now carries the config digest, provenance, regimes, per-component manifest digests and any head-only options (`identity` in `__meta__` and in the receipt). |
| Engine options after 3ecdb256 | **GAP, closed here** | `normalize_modular_config` on the torch side drops unknown keys silently, so an `input_normalization` encoder would import as a different model. The export now refuses it (`NOT_SUPPORTED_BY_TORCH_CONSUMER`). `target_residual` is head-only and does not change the encoder. |
| Parity scope | **GAP, open (lane G)** | Parity is proven only on a random-init FIXTURE (d_model 16, 1 block, 3 features). Before any R1/R2 RL cell, a parity receipt is needed on (a) the default architecture (d_model 64, 2 blocks, 4 heads, at the task's F) and (b) the actual donor bundle to be imported. Same test, different export. |
| §1 scaler artifact bound to the export | open, minor | Normalization lives in lane G's checkpoint (`_NORMALIZATION_KEYS`) and round-trips (RL05). It is not yet bound to the encoder export identity. |

**Verdict:** the spec is satisfied for R0 RL cells. For R1/R2 RL cells it is satisfied once lane G (i) re-exports with this tool, so the export carries identity and OPERATIONAL provenance, and (ii) adds the two parity receipts listed under "Parity scope". No other implementation is required.
