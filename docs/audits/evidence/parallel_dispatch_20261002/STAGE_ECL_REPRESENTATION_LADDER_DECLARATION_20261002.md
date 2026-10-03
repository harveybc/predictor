# Declaration: ECL (electricity) representation ladder, same ladder as ETH 4h (2026-10-02)

Declared before running. Decision made by the coordinator: move the ETH ladder to ECL because ETH 4h h6-36 is saturated at the intercept-only naive.
Data: `data_ecl_l24_h24_v1` of the M04 v2 modular campaign (worker_a `~/.local/state/crispdm-data-foundation/m04_doin_20260930/`), already the typed NPZ contract (windows float32 (N,24,321), string row_ids, split train/validation, metric_space z_train; train 18341 windows, validation 2609 windows; no test rows in these files). No conversion needed.

Per-feature table (train-only): taken, not recomputed, from the M03 ECL profile (feature-eng `docs/feature_metrics/m03/profiles/tsl_electricity/metrics_long.csv`, sha256 6db80e65..., TRAIN rows [0,18412), statsmodels autolag ADF): `unique_count`->n_unique, `acf_lag_24`->acf_lag24, `adf_c_aic_pvalue`->adf_pvalue. All 321 columns status OK. Result: `ladder_ecl_20261002/ecl_train_feature_table.csv` (sha256 5c0fc6ac...).
Grouping: identical rule as ETH (`tools/ladder_ae_forecast.py groups`): n_unique<=2 binary_regime; acf_lag24>=0.85 level_nonstationary; ADF p<0.05 & |acf_lag24|<0.10 fast_stationary; acf_lag24>=0.45 slow_state; else mid_state. Outcome (groups.json sha256 6c1b3091...): level_nonstationary 253, slow_state 61, mid_state 7 (binary_regime and fast_stationary empty). No target-association column is read.

Rung 1: one typed AE per non-empty group (`app/npz_encoder_adapter.py` conv1d_ae_v1, filters 32, epochs 20, batch 128, seed 2021, latent 8), train windows only; recon MAE/MSE vs per-channel-mean naive reported as diagnostic.

Rung 2 (ONE cell, `tools/ladder_ecl_forecast.py --mode ae`): frozen encoders -> 24 latents, standardized with fit-row statistics -> Dense(64,relu) -> Dense(24*321=7704), output bias = train(fit rows) median per (horizon, channel); Huber(delta 1), Adam 1e-3, batch 64, max 30 epochs, patience 5; early stopping on train-tail holdout (last 15% of train windows, 48 windows purged = window 24 + horizon 24 overlap; ETH used 40, deliberate change because purge must cover the overlap); validation evaluated once at best-holdout weights. Seed 2021.
Then: same cell seed 2022 (variance); ONE ablation: `--mode raw`, same head on flattened standardized raw windows (7704 inputs, standardized with fit-row statistics, no AE), seed 2021, matched control.

Reading: objective = validation MAE (z_train) averaged over all 24 horizons and 321 channels, against seasonal-24 naive (0.247966), persistence (0.851406) and train-median intercept, all recomputed in the tool from the same validation rows. skill = 1 - model/naive, signed. Reference only (not a gate): best verified modular model grouped32+residual 0.21288-0.21349. No claim beyond one config, two seeds, one control.

Caps: 1.25 x measured peak, never lowered. Peak measured by running the largest AE group (level_nonstationary, 253 ch) as the pilot under a 5G cap with /usr/bin/time maxrss recorded; the forecast cells are measured on their first run the same way and later cells reuse the derived cap. Sequential, one big job at a time on worker_a, crispdm-run, GPU 5090 first and 5070 Ti second.
