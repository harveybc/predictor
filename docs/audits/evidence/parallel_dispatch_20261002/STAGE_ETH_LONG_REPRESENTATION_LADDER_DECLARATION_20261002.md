# Declaration: ETH 4h long-horizon representation ladder, single seed 2021 (2026-10-02)

Declared before running. Tool: `tools/ladder_ae_forecast.py` (minimal new trainer; this branch). Data: `eth4h_long_l24_h6to36_v1` (train 13158 windows, validation 2160 windows; no test rows exist in these files; 2025 rows [15895,18085) untouched).

Grouping (table only): rule over H2 train-only per-feature columns `n_unique`, `acf_lag24`, `adf_pvalue`. No target-association column is read.
n_unique<=2 -> binary_regime (2); acf_lag24>=0.85 -> level_nonstationary (22); ADF p<0.05 and |acf_lag24|<0.10 -> fast_stationary (27); acf_lag24>=0.45 -> slow_state (23); else mid_state (9). Total 83, from `groups.json` produced by the `groups` subcommand.

Rung 1: one typed AE per group (`app/npz_encoder_adapter.py`, conv1d_ae_v1, filters 32, epochs 20, batch 128, seed 2021, latent_dim 8 except binary_regime = 2), fit on train windows of that group's channels only. Report recon MAE/MSE against the per-channel-mean naive on the same group tensor (diagnostic, train reconstruction, no selection).

Rung 2 (ONE cell): frozen encoders -> concatenated latents (34 dims), standardized with train statistics -> Dense(64,relu) -> Dense(6), output bias = train median per horizon, Huber(delta 1) / Adam 1e-3 / batch 64 / max 30 epochs / patience 5. Early stopping uses a train-tail holdout (last 15% of train windows; 40 windows purged between fit and holdout), NOT validation. Validation is evaluated once at the best-holdout weights. Seed 2021.

Reading: objective = mean validation MAE (z_train) over horizons 6,12,...,36 against the same-row intercept-only naive (2.37115, `naive_skill_validation.json`), train-mean (2.37263) and persistence (`naive_table_validation.json`); skill = 1 - model/naive, signed, single seed, no claim beyond that.

Caps: AE and forecast cells 4110M (1.25 x measured 3.29 GB peak RSS of the adapter pilot, never lower), sequential, one big job at a time, via crispdm-run.
