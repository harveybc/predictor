# Frozen-latent cache contract v0

Satoshi lane S3, 2026-10-02. Branch base: `codex/modular-neat-integration-20261001` @ `22685545`.

**Status: SPECIFICATION ONLY. NOT IMPLEMENTED.** No cache was built and no model-quality measurement exists. Code citations were read in the base worktree; anything not read is marked UNVERIFIED.

A latent cache stores the rank-three output of the modular encoder (`ModularBundle.encoder_model`, `predictor_plugins/modular_temporal/assembly.py`) for a fixed set of windows, so that several heads (the NEAT head and its Keras control, see `NEURAL_NEAT_HEAD_CONTRACT_v0.md`) are trained and scored on the same bytes. It is valid only when the producer cannot change between rows, runs and heads.

## 1. Eligibility: frozen producers only

A cache may be built only if ALL of the following hold, each checked and recorded, never assumed:

1. Every branch and the core has regime `R1`. `regime_summary(config)` (`config.py`) must return `common == "R1"`; `MIXED`, `R0`, `R2`, `R3` are refused. R3 starts frozen (`artifacts._apply_regime`: `trainable = regime not in ("R1","R3")`) but unfreezes in `warm.fit_warm`, so it is not a frozen producer.
2. Behavioral check, not flags: `weights_hash(bundle.encoder_model)` (`common.weights_hash`) measured immediately before and after the forward pass over all shards is identical, and `encoder_model` has zero trainable weights at build time.
3. The producer runs in inference mode (`training=False`); dropout in the core defaults to 0.0 (`components.transformer_conv`) and any nonzero value is still disabled by inference mode, but is recorded.
4. Provenance: the bundle's `conditioning_contract` is `OPERATIONAL` (`load_bundle(..., require_contract="OPERATIONAL")`, `bundle.py`). Each donor (branch and core) that fed the bundle has OPERATIONAL provenance in its sidecar (`provenance.donor_provenance`). `UNKNOWN` and `SYNTHETIC_OFFLINE` are refused. `donor_contract: UNKNOWN_ALLOWED` in the config is refused (it is the engine's explicit bypass, `artifacts.DONOR_CONTRACTS`).
5. Keras `major.minor` of the producing environment equals the bundle's (`bundle.load_bundle` enforces the same rule).
6. The config has no `input_normalization`: it changes the encoder input and adds a window-mean restoration at the output (`assembly.build_modular`: `forecast_plus_window_mean`), which a head over cached latents cannot reproduce. Refused. `target_residual` is allowed ONLY together with the residual-reference side-cache (section 2, `residual_reference`); without it the cache is refused with `LATENT_CACHE_HEAD_ONLY_OPTION`.
7. Donors are schema 2, or schema 1 with an ALONGSIDE schema-2 sidecar. A donor with only a schema-1 sidecar reads as UNKNOWN (`provenance.donor_provenance`) and is refused. Declared donor sets, as stated by the coordinator and not re-verified by me: M02's 322 ECL donors carry ALONGSIDE schema-2 sidecars (OPERATIONAL, TRAIN_ONLY; DONOR_INDEX amendment 1, sha prefix 2aaba33d; predictor branch `satoshi/a-engine-integration-20261001` @ `15f5c5d0`); M07's ETH donors (worker_a, `~/.local/state/scratch/m07/donors/eth_per_feature_v1`) are declared OPERATIONAL/TRAIN_ONLY. The cache binds each donor's sidecar bytes and provenance regardless.

The authority for producing latents is the Keras engine. The torch encoder in agent-multi (`rl_temporal/modular_torch.py`, branch `satoshi/g-rl-temporal-20261001` @ `27da2d96`; only its header was read, rest UNVERIFIED) is a parity-tested port for RL (`MODULAR_ENCODER_RL_EXPORT_SPEC_v0.md` §9) and must not produce a cache; it may only read one.

## 2. What the cache binds (manifest `LATENT_CACHE_MANIFEST.json`)

Schema string `predictor.latent_cache.v1`. Canonical JSON and digest follow `common._json`/`_digest` (sorted keys, compact separators, `allow_nan=False`, SHA-256 hex).

| Binding | Content | Source in code |
|---|---|---|
| `source_npz` | for each split: path-free `sha256` of the evaluator-format NPZ the windows came from, `dataset_id`, `split`, `metric_space` | `modular_candidate_evaluator._load` (streams sha, requires `windows,targets,row_ids,timestamps,target_timestamps,dataset_id,split,feature_names,target_names,horizons,timestamp_unit,metric_space,scaler_identity,scaler_scale`) |
| `row_order` | `row_ids_sha256` = SHA-256 over the concatenation of each `row_id` UTF-8 plus `\n`, in cache row order; `n_rows`; `timestamps_sha256` over little-endian int64 origin seconds; `target_timestamps_sha256` over `[N,H]` int64 | `row_ids` are unique strings (`_load`); origins strictly increasing (`_load`) |
| `scaler` | `scaler_identity` string, `scaler_scale` bytes sha, and, when the source has a MANIFEST, its `scaler.fit_rows`, `mean_sha256`, `scale_sha256`; fit rows must lie entirely inside the TRAIN rows | `modular_doin_ecl_npz.build` writes `scaler_identity = "ecl-train-standard:<sha>"` and `fit_rows = train_rows` |
| `feature_order` | `config.feature_names` exactly, `target_names`, `target_feature_indices` | `_load` checks features equal `model_config["feature_names"]` |
| `producer` | `config_sha256 = config_digest(bundle.config)`; `bundle_json_sha256`; `bundle.weights_sha256` and `archive_sha256` (from `bundle.json`); `encoder_weights_sha256 = weights_hash(encoder_model)`; per-component `weights_hash` for each branch and the core; `component_manifests()` digests for each branch, fusion, core | `assembly.ModularBundle.component_manifests`, `bundle.save_bundle` |
| `donors` | for each branch and the core: donor archive `model_sha256`, `weights_sha256`, `manifest_sha256` and the provenance block (`conditioning_contract`, `learned_corpus`, `reconstruction`, `objective`, `keras_version`, any `derivation`/`migration`); the core manifest's `upstream` (branch + fusion weight digests) is bound implicitly through `manifest_sha256` | `artifacts.save_donor`/`load_donor`, `provenance.donor_provenance` |
| `fusion` | fusion plugin identity and params, `input_shapes`, `output_shape`, `grid` (raw channel concatenation, no weights) | `assembly.component_manifests()["fusion"]`, `components.sequence_concat` |
| `grid` | `window`, `sample_hours`, `horizons`, `branch_time_grid`, `core_time_grid`, `latent_shape [6,8]`, purge rule, validation border rows, count of windows excluded for gaps and purged | `config._normalize`, `_partition`; purge in `modular_doin_ecl_npz.build` (`purge = target_end >= validation_input_start`) |
| `array` | `dtype "<f4"`, byte order little-endian, C order, layout `[N, T'=6, C'=8]` (time then channel, the Keras channels-last order; no transposition is ever applied inside the cache), per-shard shape | `bundle.encoder_model.output_shape[1:]`, `assembly._budget["latent_shape"]` |
| `residual_reference` | present iff the config has `target_residual`: float32 `[N,H,T]` of the in-window seasonal-naive terms per horizon, computed from the same source NPZ windows exactly as the engine does (`assembly._residual_receipt`: `source_window_positions` / `source_offset_steps` per horizon; layers `SeasonalNaiveBaseline`, `SeasonalCumulativeBaseline` in `layers.py`), with its own content hash, `kind`, `period`, `target_features`, the per-horizon window positions, the `row_ids_sha256` of the same row order and the source NPZ sha. The reported forecast is `head + residual_reference`, identical to the engine's `forecast_plus_seasonal_naive` | `assembly.py` `_residual_receipt`, `build_modular` |
| `companions` | `targets [N,H,T]` and `baseline [N,H,T]` float32 shards cut from the SAME source NPZ rows (baseline = each row's last observed target feature repeated across horizons, the evaluator's persistence naive; `evaluator._metrics`, module docstring), each with its own content hash | `modular_candidate_evaluator.py` docstring: "Persistence repeats each row's last observed target feature across horizons" |
| `environment` | Keras, TensorFlow, Python, numpy versions; device kind; batch size used (informational) | `tools/export_modular_encoder_npz.py` records `versions` the same way |
| `shards` | list of `{split, file, n_rows, row_start, row_stop, shape, dtype, content_sha256}` where `content_sha256` is SHA-256 of the raw array bytes (not of an `.npy` header) | new |
| `cache_id` | `_digest` of the manifest with `cache_id` and `created_utc` removed | new |

Shards are plain `.npy` (header-checked, `allow_pickle=False`) or raw `.bin` with the manifest as sole description; v0 prefers `.npy` for tooling. No compression is applied: the owner's standing rule (`predictions-metrics-vault-rule`) is that no stored evidence is compressed.

## 3. Refusal rules (each returns a named code, nothing is written)

`LATENT_CACHE_TRAINABLE_COMPONENT`, `LATENT_CACHE_REGIME_NOT_R1`, `LATENT_CACHE_PROVENANCE_UNKNOWN` (any UNKNOWN or SYNTHETIC_OFFLINE contract, or a missing donor sidecar), `LATENT_CACHE_SCALER_UNBOUND` (no `scaler_identity`, or fit rows not inside train rows, or train/validation scaler metadata differ; compare `evaluate_candidate` which already refuses mismatched `metric_space`, `scaler_identity`, `scaler_scale` across splits), `LATENT_CACHE_ROW_ORDER_MISMATCH`, `LATENT_CACHE_DIGEST_MISMATCH` (any bound digest recomputes differently on build or on read), `LATENT_CACHE_FEATURE_ORDER`, `LATENT_CACHE_NONFINITE`, `LATENT_CACHE_SHAPE`, `LATENT_CACHE_HEAD_ONLY_OPTION` (item 1.6), `LATENT_CACHE_TEST_SPLIT` (any split other than `train` or `validation`), `LATENT_CACHE_PRODUCER_MOVED` (item 1.2). A consumer re-verifies content hashes of every shard it reads and the manifest `cache_id`; a mismatch is a refusal, never a rebuild.

## 4. Versioning and invalidation

- `cache_id` is the identity. Any change to any bound value yields a different cache; a cache is never edited or extended in place. Adding rows is a new cache.
- A change of the engine's normalized config schema (`config.CONFIG_SCHEMA`), bundle schema (`provenance.BUNDLE_V2`) or this schema string invalidates by definition: the consumer refuses an unknown `schema`.
- Writes are atomic: shards and manifest are written to `<name>.tmp` and `os.replace`d, the manifest last (same pattern as `provenance._atomic`). A directory with shards but no manifest is garbage and never read. The manifest is written only after every shard hash was re-read from disk.

## 5. Size and memory arithmetic

Latent bytes per row: `6 * 8 * 4 = 192` B (float32). Per-row producer working set (default one branch per feature, 16 channels each, `components.causal_conv1d` default, and `default_config`): fused tensor `24 * (16 * F) * 4` B = `1536 F` B (`assembly._analytic_budget`: `materialization_bytes_per_row = branch_steps * sum(widths) * 4`).

| Quantity | ECL (F = 321) | ETH (F = 83) |
|---|---|---|
| Input window bytes per row (`24*F*4`) | 30,816 B | 7,968 B |
| Fused bytes per row (`1536*F`) | 493,056 B (~0.47 MiB) | 127,488 B |
| Latent per row | 192 B | 192 B |
| Validation, 2,609 windows (latent) | 500,928 B (~0.48 MiB); ECL is out of scope for the head in v0 | n.a. (window count not read) |
| Validation, 2,609 windows, fused if materialized at once | ~1.29 GB: do not; run in batches (a batch of 256 needs ~126 MB fused) | ~0.33 GB |
| Train, ~1.8e4 windows (latent) | ~3.5 MB (window count UNVERIFIED; from `n_train = int(0.7 * n_rows)` with ECL hourly rows, purge and gaps not counted) | n.a. |
| Companion `targets` + `baseline` per row | `2 * H * T * 4` B; with the coordinator's 24 ECL horizons and T = 321: 61,632 B per row, ~161 MB for 2,609 rows (H, T not independently read) | H = 6 (coordinator): `48 * T` B per row, tiny |

Consequences: the cache itself is tiny; its cost is the one-time producer forward, `N_rows * producer_forward_cost`, charged once to the campaign and reported beside the head arms. Memory ceiling during build is the producer batch, not the cache. Required: build in batches, one memory-heavy job per host, never on the coordinator (standing placement rule).

ECL targets are large (`T` may equal 321): the companions can exceed the latents by 50x. They are still stored, because both heads and the naive must read identical bytes.

## 6. Reference test plan (synthetic mechanics only; no performance claim)

Use a tiny config (3 features, window 24, d_model 16, one block) built with `build_modular`; synthetic NPZ with 40 train and 12 validation rows.

| Test | Assertion |
|---|---|
| `test_cache_builds_from_r1_bundle` | all-R1 bundle with OPERATIONAL provenance builds; shards have shape `[N,6,8]`, dtype `<f4`; manifest recomputes its `cache_id` |
| `test_refuses_r0_r2_r3_mixed` | each of R0, R2, R3, MIXED raises the named code; nothing written |
| `test_refuses_unknown_and_synthetic_provenance` | UNKNOWN and SYNTHETIC_OFFLINE bundle and donor sidecars refused; `UNKNOWN_ALLOWED` config refused |
| `test_producer_hash_unchanged` | `weights_hash(encoder_model)` equal before and after build; a hook that perturbs one weight mid-build raises `PRODUCER_MOVED` |
| `test_latents_match_direct_forward` | cache rows equal `encoder_model(x, training=False)` bitwise on the same batch size, and within 1e-6 across different batch sizes |
| `test_row_order_binding` | permuting rows in a shard or the NPZ changes `row_ids_sha256`; reading with a permuted order is refused |
| `test_scaler_binding` | scaler fit rows overlapping validation, or differing `scaler_identity` across splits, refused |
| `test_feature_order_binding` | swapping two feature names refused |
| `test_donor_bytes_binding` | altering one byte of a donor sidecar `weights_sha256` or the bundle archive refuses on read |
| `test_shard_hash_tamper` | flipping one byte in a shard refuses on read with `DIGEST_MISMATCH` |
| `test_no_test_split` | a split named `test` refused |
| `test_head_only_option_refused` | `input_normalization` refused; `target_residual` refused without the side-cache |
| `test_residual_reference_exact` | an all-zero head plus `residual_reference` equals the `forecast_model` output with a zero head (within 1e-6), for both residual kinds |
| `test_residual_reference_binding` | a side-cache row hash or window positions differing from the cache refuses on read |
| `test_schema1_only_donor_refused` | a donor with only a schema-1 sidecar is refused; the same donor with an ALONGSIDE schema-2 sidecar is accepted |
| `test_batch_size_sensitivity_measured` | ACCEPTANCE TEST TO RUN, not a claim: build the same rows at batch sizes 1, 32 and 256, record the maximum absolute latent difference, write it into the manifest `environment` block; the 1e-6 tolerance is accepted only if the measured value is within it |
| `test_atomic_write` | killing the writer after shards but before the manifest leaves no readable cache |
| `test_companions_same_rows` | `targets` and `baseline` row hashes equal the source NPZ rows; baseline equals last observed target repeated |
| `test_nonfinite_refused` | a NaN in the producer output refused |
| `test_replay_determinism` | two builds in one environment give identical `cache_id` (excluding `created_utc`) |

## 7. Open questions this document could not settle by reading

1. Whether the ECL cache stores joint or per-channel targets; ECL is out of scope for the head in v0, but the cache contract still permits it.
2. The measured batch-size and device sensitivity of the Keras forward pass: an acceptance test (section 6), not a claim.
3. The M02/M07 donor provenance statements are the coordinator's; the cache build re-reads them from the sidecars and refuses on mismatch.
