# Lane A — PS3-R pilot design (20-input stratified), for approval. NOT RUN.

Satoshi, successor technical lead · 2026-10-01 · orders ac125db9 (progressive selection and modular continuation)

**Status:** design only. No candidate has been fitted on real data. The only training so far is synthetic-noise timing and the unit tests.

## 1. What is ready (tests, CPU)

| Item | Where | Evidence |
|---|---|---|
| `app.representation_card` (FS17 green) and the cnn encoder's declared `latent_layout` | feature-extractor `satoshi/a-ps3r-card-20261001` `a0a2e21` | FS17 4/4 and 6 more card tests pass (coordinator, crispdm-run 1G, stdlib only) |
| Objectives as their own components (`modular.objective`, own version and identity, separate from the architecture manifest): `autoencoder_reconstruction` 1.0.0 (AE control) and `ts2vec_contrastive` 1.0.0 | predictor `satoshi/a-engine-integration-20261001`, `predictor_plugins/modular_temporal/objectives.py` | `tests/test_modular_objectives.py` 11/11, written first (973346c5) |
| Delta_probe battery (trained / untrained twin / raw, paired naive, card rows) | `predictor_plugins/modular_temporal/probes.py` | same suite |
| Regression of the whole lane A + M02 suite after these additions | worker_b, envs/tensorflow (TF 2.21.0, Keras 3.13.2), crispdm-run 4G | 139 passed, 2 skipped (installed-only checks) |

The campaign pin stays at 3ecdb256. The objectives and probes are additive commits after it. The architecture, the donor manifests and M02's files are unchanged; the objective only goes into donor provenance (`save_donor(..., objective=)`).

### Declared deviations from the TS2Vec reference (zhihanyue/ts2vec @ b0088e14)

1. **Cropping.** The branch input is fixed at 24 steps. A crop is therefore applied by zeroing the history before the crop start, and the loss is computed on the shared suffix. The reference feeds shorter sequences instead.
2. **Masking.** Timestamp masking zeroes input steps rather than the hidden projection, because the branch is an external component.
3. **Encoder.** The encoder is the approved causal Conv1D branch, not TS2Vec's dilated CNN. The objective is the axis under test; the architecture is held fixed.

These deviations are why any number from this pilot is NOT_COMPARABLE with the TS2Vec paper's tables.

## 2. Inputs: seeded stratified draw of 20 from lane B's ETH 4h work list

**Current draw.** Work list `397b67d6…` (feature-eng `1b22c64`, the CORRECTION after the label-unit defect). Rule: majority tier over the three inner folds, ties to the higher tier. A feature drawn EXPLORATORY in some fold but not ranked by majority forms the exploratory stratum. Quotas: PRIORITY 5, SYNERGY 4, REPRESENTATIVE 4, EXPLORATORY 3, DEFERRED 4. Seed 20261001.

Evidence: `docs/audits/evidence/lane_a_ps3r_20261001/PILOT_SELECTION_397b67d6.json`.

| Stratum (size) | Drawn (inclusion probability) | Caution (persistent input) |
|---|---|---|
| PRIORITY (10) | close_sma_ratio_100, close_sma_ratio_200, ema_cross_20_100, hurst_proxy_200, volume_sma_10 (0.5) | close_sma_ratio_200, ema_cross_20_100, volume_sma_10 |
| SYNERGY (7) | bb_middle, bb_width, roll_kurt_ret_20, sqret_autocorr_lag1_100 (0.571) | bb_middle |
| REPRESENTATIVE (11) | hist_vol_20, rsi_21, statistical__log_return_1, williams_r_14 (0.364) | — |
| EXPLORATORY (12) | log_return_10, return_1, roll_kurt_ret_60 (0.25) | — |
| DEFERRED (43) | atr_14, close_sma_ratio_20, log_return_60, mfi_14 (0.093) | atr_14 |

**Previous draw** on `f75260a6`, kept as `PILOT_SELECTION_f75260a6_PROVISIONAL.json`. It is `PROVISIONAL_SUPERSEDED_LABELS`: its strata were ranked against labels placed 1000× too far ahead. See the cross-lane finding in the same folder.

The persistent-input cautions are carried into the results, not resolved by this pilot.

## 3. Arms (budget-matched), targets, folds

- **Branch.** One feature per branch, (B, 24, 1) → (B, 24, 16), causal Conv1D 2.0.0. The window is 24 bars × 4 h = 96 h.
- **Scaling.** Each feature is standardized on the encoder-train rows of its own fold only.
- **Families**, per input × fold × seed:
  - AE control (`autoencoder_reconstruction`), trained;
  - contrastive (`ts2vec_contrastive`, alpha 0.5, mask 0.5, min overlap 8), trained;
  - random control: the same architecture untrained (`untrained_twin`, seed paired with the trained arm);
  - raw control: the 24-bar window itself. Its dimension differs, and this is declared.
- **Matched budget for both trained families:**
  - AdamW, learning rate 1e-3, weight decay 1e-4, batch 64;
  - max_epochs 30, patience 5, min_delta 0, `max_updates` 5 000 per fit;
  - early stopping on each family's own internal-validation objective, with best-checkpoint restore;
  - observed updates are recorded.
- **Seeds.** 2021 and 2022. Each seed sets encoder initialization, augmentation and the paired random twin.
- **Folds.** Lane B's three inner expanding folds inside TRAIN, row ranges as in their `run_summary.json`: inner_1 [0, 7474) / [7534, 9589), and so on, with a purge of 60 rows (24 context + 36 bars = 144 h).
  - The encoder fits on the first 85 % of a fold's train windows and early-stops on the last 15 %, after a 60-row purge.
  - The probe fits on all of the fold's train windows whose label lies inside the fold train, and is evaluated on the fold's validation windows.
  - No row of the outer validation (2024) or the protected test (2025) is read.
- **Targets**, from lane B's corrected constructions (`target_status.csv` `a8cb02f9`, CLOSE as the asset price, labels located by elapsed seconds):
  - Y_s@4h;
  - Y_l@24, 48, 72, 96, 120 and 144 h;
  - Y_b stays NOT_EVALUATED: there is no versioned barrier rule.
- **Probe.** Ridge on the flattened per-timestamp latent, standardized on fit rows. Alpha comes from a grid of 7 and is chosen on the last 20 % of the fit rows; the same grid and selection apply to every arm.
- **Paired naive.** A zero log return (price persistence) on the identical evaluation rows.

## 4. Outputs and decision rule (no winner is declared by the pilot)

- **Cards.** One card per input × family (`representation_candidate_card.v1`, validated by `app.representation_card`). Each carries probe rows per target × horizon × fold × seed:
  - loss_trained, loss_random, loss_raw, naive;
  - delta_probe and preservation;
  - effective dimension and the collapse flag;
  - cost: observed updates, seconds, peak RSS.
  - AE cards also record reconstruction as MEASURED. Contrastive cards record it as NOT_APPLICABLE.
- **Paired contrast.** The paired contrast on identical rows is loss_AE − loss_contrastive. It is reported per target × horizon at 1e-5 resolution, with fold × seed dispersion.
- **What the pilot can establish.** Whether either objective adds usable information (delta_probe > 0 against its untrained twin) and whether it beats the naive baseline. It also measures the real cost per fit.
  - It does NOT rank families globally.
  - It does NOT select inputs.
  - It does NOT touch the outer validation set.
- **Stop and report** on a nonfinite loss, a collapsed latent (effective dimension ≤ 1) in every seed, or a memory breach.

## 5. Cost, from measured rates (worker_b CPU, crispdm-run 2G; `PILOT_TIMING_synthetic_steady.json`)

The rates were measured on synthetic noise windows of the pilot's exact shape (24 × 1, batch 64, 4 096 rows, 3 epochs):

| Quantity | Measured |
|---|---|
| contrastive, steady state, last epoch with its validation pass | 0.00605 s per update |
| contrastive tracing, one graph per overlap start (≤ 17) + validation | ≈ 30 s per fit |
| AE control, including tracing | 0.00768 s per update |
| probe battery, 3 arms, 2 targets, inner_1 rows | 1.71 s, i.e. ≈ 0.86 s per target |

Updates per epoch at batch 64 follow from the fold sizes: 99 / 127 / 154 for inner_1 / inner_2 / inner_3. The worst case therefore runs all 30 epochs: 11 400 updates per input × seed × family.

| Component | Worst case per input × seed | × 40 (20 inputs × 2 seeds) |
|---|---|---|
| contrastive, 3 folds | 11 400 × 0.00605 + 3 × 30 s ≈ 159 s | ≈ 6 360 s |
| AE control, 3 folds | 11 400 × 0.00768 ≈ 88 s | ≈ 3 500 s |
| probes: 7 targets × 3 folds; raw and random arms computed once and shared by both families | ≈ 7 × 3 × 0.86 × 2 ≈ 36 s | ≈ 1 440 s |
| **total, one child** | | **≈ 11 300 s ≈ 3.1 h worst case** |

**Placement proposal.** Four CPU children on worker_b, each crispdm-run 2G, gives about 0.8 h wall in the worst case. Early stopping will usually end fits sooner. No GPU is needed for this pilot, so M04's GPU slots stay untouched. Peak RSS per child is expected below the 1.29 GB measured for TF plus a full fusion forward on the same host.

**Assumption to verify on the first fit.** Real windows have the same shape as the timing windows, so the per-update rate should carry over. The first real fit's measured rate replaces this estimate in the running report.

## 6. Not done / risks

- **Not run:** the pilot itself, any real-data fit, and Y_b.
- **Out of scope:** the PatchTST/CF-JEPA/MOMENT families.
- **Cautioned inputs.** Five of the 20 inputs carry the persistent-input caution, so their associations may be trend or regime confounding. The pilot reports them; it does not interpret them.
- **Contrastive loss scale.** The contrastive objective's loss scale differs from the AE's. The budget is matched by caps and early-stopping rules, not by loss values.
- **Comparability.** Pilot numbers are NOT_COMPARABLE with TS2Vec Table 7 (different data, protocol and encoder).

**Approval requested.** Run §3 as written on worker_b CPU, four children at 2G each, followed by an independent re-score of the cards.
