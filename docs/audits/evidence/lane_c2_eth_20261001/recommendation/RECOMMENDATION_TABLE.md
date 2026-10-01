# Lane C2: feature recommendation table for M03 (selection) and M07 (campaign) -- DEVELOPMENT

Population: `financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1`, view sha `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f` (predictor `b1f8a74f`), features: variant A, declaration `f3c0beca`, manifest file `fdff0c85`; split: M07 (agent afcf115024ffa1381), predictor satoshi/f2-eth-forecast-20261001 13ef175f, file `116a5b64`, TRAIN rows [0,13699), scored origins [23,13669] (13415 windows, 232 gap-excluded), train row-ids sha `3903daae`. Target: Y_h = sum_{k=1..h} z(log_return_1[t+k]) = (log(CLOSE[t+h]/CLOSE[t]) - h*mu)/sigma; mu, sigma fitted on train rows only (mu=0.000146097, sigma=0.0194728).

Units: MAE_z is in M07 z-units of the cumulative standardized 1-bar log return; MAE_log_return is the raw log return of CLOSE. Incremental utility delta = MAE(without feature) - MAE(all 83), positive = the feature helps the held-out ridge. Every row is TRAIN-only, held out within TRAIN.

## Reference rows on the identical held-out rows (blocks5 protocol, mean over the 5 blocks)

| horizon (bars / h) | n eval | naive zero | naive last-return | naive seasonal 24h | naive train-mean | ridge all-83 (alpha selected) | ridge all-83 (alpha=1, lane B) | HGB all-83 |
|---|---|---|---|---|---|---|---|---|
| 1 / 4 | 13415 | 0.63323 | 0.95261 | 0.96986 | 0.63327 | 0.63386 | 0.64680 | 0.65664 |
| 2 / 8 | 13415 | 0.90460 | 1.33501 | 1.38247 | 0.90474 | 0.90949 | 0.94100 | 0.97497 |
| 3 / 12 | 13415 | 1.12731 | 1.61810 | 1.71443 | 1.12778 | 1.13504 | 1.18647 | 1.22246 |
| 4 / 16 | 13415 | 1.32750 | 1.92242 | 1.99723 | 1.32800 | 1.33738 | 1.40799 | 1.51681 |
| 5 / 20 | 13415 | 1.51131 | 2.22084 | 2.26161 | 1.51225 | 1.52284 | 1.61324 | 1.70370 |
| 6 / 24 | 13415 | 1.67674 | 2.48560 | 2.48560 | 1.67767 | 1.68857 | 1.79532 | 1.86013 |
| 36 / 144 | 13181 | 4.36517 | 6.24954 | 6.24954 | 4.37126 | 4.43123 | 5.15689 | 5.73271 |

## Lane B comparability protocol (3 expanding inner folds, purge 60; MAE_z mean over folds)

| target | horizon bars | n eval | naive zero (ours) | ridge all-83 alpha=1 (ours) | ridge all-83 alpha selected (ours) | lane B C_ALL alpha=1 (its rows, log-return units) | lane B naive zero (its rows) |
|---|---|---|---|---|---|---|---|
| Y_s@4h | 1 | 6134 | 0.01087 (log-ret) / 0.55820 (z) | 0.01301 (log-ret) | 0.01114 (log-ret) | 0.01296 | 0.01088 |
| h2 | 2 | 6132 | 0.01558 (log-ret) / 0.80024 (z) | 0.02048 (log-ret) | 0.01624 (log-ret) | nan | nan |
| h3 | 3 | 6130 | 0.01943 (log-ret) / 0.99777 (z) | 0.02779 (log-ret) | 0.02047 (log-ret) | nan | nan |
| Y_l@144h | 36 | 6057 | 0.07270 (log-ret) / 3.73321 (z) | 0.15711 (log-ret) | 0.08578 (log-ret) | 0.15574 | 0.07270 |
| h4 | 4 | 6128 | 0.02288 (log-ret) / 1.17522 (z) | 0.03435 (log-ret) | 0.02441 (log-ret) | nan | nan |
| h5 | 5 | 6126 | 0.02592 (log-ret) / 1.33102 (z) | 0.04098 (log-ret) | 0.02789 (log-ret) | nan | nan |
| Y_l@24h | 6 | 6124 | 0.02853 (log-ret) / 1.46536 (z) | 0.04726 (log-ret) | 0.03093 (log-ret) | 0.04632 | 0.02853 |

## Lane B PS4 transform-family probe (feature-eng 2840e52), as reported by lane B

```
{
 "_source": {
  "file": "feature-eng docs/feature_metrics/laneB/f1/ETH_VARIANT_D_FAMILIES_PROBE.v1.json @ 2840e52",
  "sha256": "cfc77827d11908962b0403d54728384b4ac0b1ca97bd2368fe1cc05863dec623",
  "probe": "ridge alpha=1.0, fold-TRAIN standardization, naive zero return",
  "units": "raw log return, lane B rows"
 },
 "native_wavelet": {
  "columns": [
   "wavelet_native_A3",
   "wavelet_native_D3",
   "wavelet_native_D2",
   "wavelet_native_D1",
   "wavelet_native_energy_D3",
   "wavelet_native_energy_D2",
   "wavelet_native_energy_D1"
  ],
  "mean_mae_by_target": {
   "Y_s@4h": 0.010920799587774142,
   "Y_l@24h": 0.02875210538379926,
   "Y_l@144h": 0.07445193806345633
  },
  "naive_mean_by_target": {
   "Y_s@4h": 0.010880835431918345,
   "Y_l@24h": 0.02853350826199755,
   "Y_l@144h": 0.07269614178502508
  },
  "status": null
 },
 "rolling_zscore": {
  "columns": [
   "z_logclose_20",
   "z_logclose_60",
   "z_ret_60"
  ],
  "mean_mae_by_target": {
   "Y_s@4h": 0.010944270434528655,
   "Y_l@24h": 0.028802971395373463,
   "Y_l@144h": 0.07345488647075085
  },
  "naive_mean_by_target": {
   "Y_s@4h": 0.010880835431918345,
   "Y_l@24h": 0.02853350826199755,
   "Y_l@144h": 0.07269614178502508
  },
  "status": null
 },
 "realized_volatility": {
  "columns": [
   "rv_std_ret_6",
   "rv_std_ret_24",
   "rv_std_ret_42",
   "parkinson_6",
   "parkinson_24"
  ],
  "mean_mae_by_target": {
   "Y_s@4h": 0.01087376274596087,
   "Y_l@24h": 0.028571299554234302,
   "Y_l@144h": 0.07299795274311505
  },
  "naive_mean_by_target": {
   "Y_s@4h": 0.010880835431918345,
   "Y_l@24h": 0.02853350826199755,
   "Y_l@144h": 0.07269614178502508
  },
  "status": null
 },
 "range": {
  "columns": [
   "log_high_low",
   "close_location",
   "log_close_open"
  ],
  "mean_mae_by_target": {
   "Y_s@4h": 0.010834431617771624,
   "Y_l@24h": 0.02852618302617301,
   "Y_l@144h": 0.07251790332202081
  },
  "naive_mean_by_target": {
   "Y_s@4h": 0.010880835431918345,
   "Y_l@24h": 0.02853350826199755,
   "Y_l@144h": 0.07269614178502508
  },
  "status": null
 }
}
```

## Rank agreement across the 5 blocks (mean pairwise Spearman of per-block incremental-utility ranks)

- blocks5|h1: 0.116 (10 pairs)
- blocks5|h2: 0.049 (10 pairs)
- blocks5|h3: 0.051 (10 pairs)
- blocks5|h36: 0.035 (10 pairs)
- blocks5|h4: 0.029 (10 pairs)
- blocks5|h5: -0.023 (10 pairs)
- blocks5|h6: 0.001 (10 pairs)
- laneB|h1: 0.101 (3 pairs)
- laneB|h2: 0.038 (3 pairs)
- laneB|h3: 0.053 (3 pairs)
- laneB|h36: -0.165 (3 pairs)
- laneB|h4: 0.074 (3 pairs)
- laneB|h5: 0.071 (3 pairs)
- laneB|h6: 0.041 (3 pairs)

## Features with a positive median incremental utility and sign agreement >= 4/5 (blocks5)

| feature | h | delta MAE_z median | min | max | sign agreement | only-feature beats naive zero (blocks of 5) | leak verdict |
|---|---|---|---|---|---|---|---|
| close_sma_ratio_10 | 1 | 0.000182 | -0.000276 | 0.000250 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| return_5 | 1 | 0.000150 | -0.000486 | 0.000236 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| log_return_1 | 1 | 0.000142 | -0.000092 | 0.000234 | 0.8 | 5 | CAUSAL_BY_RECOMPUTATION |
| statistical__log_return_1 | 1 | 0.000142 | -0.000092 | 0.000234 | 0.8 | 5 | CAUSAL_BY_RECOMPUTATION |
| return_1 | 1 | 0.000133 | -0.000092 | 0.000203 | 0.8 | 5 | CAUSAL_BY_RECOMPUTATION |
| log_return_5 | 1 | 0.000132 | -0.000435 | 0.000198 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| obv | 1 | 0.000092 | -0.000284 | 0.000251 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_kurt_ret_60 | 1 | 0.000062 | 0.000007 | 0.000127 | 1.0 | 4 | CAUSAL_BY_RECOMPUTATION |
| hurst_proxy_200 | 1 | 0.000059 | -0.000106 | 0.000217 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| atr_14 | 1 | 0.000021 | -0.000071 | 0.000119 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_20 | 1 | 0.000021 | -0.000039 | 0.000054 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| roll_std_ret_20 | 1 | 0.000021 | -0.000039 | 0.000054 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| mom_10 | 1 | 0.000020 | -0.000208 | 0.000236 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 1 | 0.000014 | 0.000001 | 0.000031 | 1.0 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_50 | 1 | 0.000010 | -0.000015 | 0.000041 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| ema_cross_20_100 | 1 | 0.000009 | -0.000003 | 0.000051 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| volume_sma_20 | 1 | 0.000006 | -0.000181 | 0.000008 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| ema_cross_10_50 | 1 | 0.000003 | 0.000000 | 0.000008 | 1.0 | 2 | CAUSAL_BY_RECOMPUTATION |
| roc_20 | 1 | 0.000003 | -0.000000 | 0.000007 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| return_20 | 1 | 0.000003 | -0.000000 | 0.000007 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| trend_strength_50 | 1 | 0.000003 | -0.000006 | 0.000034 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| cci_14 | 2 | 0.000354 | -0.001910 | 0.000470 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_10 | 2 | 0.000163 | -0.000430 | 0.000245 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_kurt_ret_60 | 2 | 0.000123 | 0.000007 | 0.000179 | 1.0 | 2 | CAUSAL_BY_RECOMPUTATION |
| hurst_proxy_200 | 2 | 0.000101 | -0.000337 | 0.000518 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_252 | 2 | 0.000066 | -0.000034 | 0.000520 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 2 | 0.000047 | -0.000003 | 0.000092 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_50 | 2 | 0.000022 | -0.000061 | 0.000103 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| vol_regime_high | 2 | 0.000021 | -0.000080 | 0.000025 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| realized_var_48 | 2 | 0.000021 | -0.000237 | 0.000285 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| trend_strength_50 | 2 | 0.000013 | 0.000003 | 0.000094 | 1.0 | 2 | CAUSAL_BY_RECOMPUTATION |
| volume_sma_20 | 2 | 0.000009 | -0.000252 | 0.000029 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_100 | 2 | 0.000001 | -0.000006 | 0.000011 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| obv | 3 | 0.000353 | -0.000916 | 0.001000 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| cci_14 | 3 | 0.000242 | -0.002546 | 0.000930 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| mfi_14 | 3 | 0.000189 | 0.000029 | 0.000430 | 1.0 | 0 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_10 | 3 | 0.000165 | -0.000562 | 0.000207 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_10 | 3 | 0.000135 | -0.000553 | 0.000540 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_252 | 3 | 0.000107 | -0.000285 | 0.000475 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 3 | 0.000102 | 0.000005 | 0.000126 | 1.0 | 3 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_20 | 3 | 0.000063 | -0.000185 | 0.000166 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_std_ret_20 | 3 | 0.000063 | -0.000185 | 0.000166 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| trend_strength_50 | 3 | 0.000036 | 0.000001 | 0.000070 | 1.0 | 1 | CAUSAL_BY_RECOMPUTATION |
| realized_var_48 | 3 | 0.000032 | -0.000301 | 0.000413 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| vol_regime_high | 3 | 0.000029 | -0.000144 | 0.000071 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_50 | 3 | 0.000021 | -0.000062 | 0.000158 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| obv | 4 | 0.000708 | -0.001391 | 0.001356 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| cci_14 | 4 | 0.000372 | -0.003048 | 0.000937 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_10 | 4 | 0.000232 | -0.000877 | 0.000344 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_kurt_ret_60 | 4 | 0.000185 | -0.000290 | 0.000520 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| realized_var_48 | 4 | 0.000176 | -0.000539 | 0.000608 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_252 | 4 | 0.000173 | -0.000430 | 0.000492 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 4 | 0.000137 | 0.000007 | 0.000205 | 1.0 | 3 | CAUSAL_BY_RECOMPUTATION |
| mfi_14 | 4 | 0.000128 | 0.000008 | 0.000458 | 1.0 | 0 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_20 | 4 | 0.000057 | -0.000313 | 0.000186 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_std_ret_20 | 4 | 0.000057 | -0.000313 | 0.000186 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| volume_sma_20 | 4 | 0.000009 | -0.001030 | 0.000046 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| obv | 5 | 0.000969 | -0.001461 | 0.002021 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| roll_kurt_ret_60 | 5 | 0.000215 | -0.000356 | 0.000663 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| mfi_14 | 5 | 0.000179 | -0.000071 | 0.000408 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 5 | 0.000130 | 0.000020 | 0.000161 | 1.0 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_10 | 5 | 0.000118 | -0.001257 | 0.000713 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_20 | 5 | 0.000102 | -0.000280 | 0.000331 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| roll_std_ret_20 | 5 | 0.000102 | -0.000280 | 0.000331 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| return_1 | 5 | 0.000093 | -0.000222 | 0.000157 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| log_return_1 | 5 | 0.000071 | -0.000217 | 0.000120 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| statistical__log_return_1 | 5 | 0.000071 | -0.000217 | 0.000120 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_50 | 5 | 0.000003 | -0.000062 | 0.000228 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| obv | 6 | 0.001118 | -0.001239 | 0.003110 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| log_return_5 | 6 | 0.000433 | -0.001253 | 0.001169 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_20 | 6 | 0.000359 | -0.000710 | 0.000720 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| return_5 | 6 | 0.000315 | -0.001139 | 0.000933 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_252 | 6 | 0.000260 | -0.000772 | 0.000615 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| mfi_14 | 6 | 0.000255 | -0.000102 | 0.000532 | 0.8 | 0 | CAUSAL_BY_RECOMPUTATION |
| roll_kurt_ret_60 | 6 | 0.000137 | -0.000370 | 0.000625 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_10 | 6 | 0.000105 | -0.001410 | 0.000908 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| autocorr_lag1_100 | 6 | 0.000097 | -0.001452 | 0.000718 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 6 | 0.000037 | 0.000007 | 0.000204 | 1.0 | 2 | CAUSAL_BY_RECOMPUTATION |
| hist_vol_60 | 6 | 0.000025 | -0.000912 | 0.000081 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_std_ret_60 | 6 | 0.000025 | -0.000912 | 0.000081 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| ema_cross_10_50 | 6 | 0.000002 | -0.000010 | 0.000027 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| obv | 36 | 0.009330 | -0.014208 | 0.024468 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| realized_var_48 | 36 | 0.006504 | -0.006314 | 0.010608 | 0.8 | 3 | CAUSAL_BY_RECOMPUTATION |
| roll_skew_ret_252 | 36 | 0.002756 | -0.004332 | 0.026905 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_200 | 36 | 0.001839 | 0.000304 | 0.002578 | 1.0 | 4 | CAUSAL_BY_RECOMPUTATION |
| bb_width | 36 | 0.001548 | -0.000411 | 0.003231 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| trend_slope_50 | 36 | 0.000495 | -0.000136 | 0.002100 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_100 | 36 | 0.000486 | -0.001627 | 0.001379 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| volume_ratio_20 | 36 | 0.000348 | -0.000144 | 0.000576 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| macd_signal | 36 | 0.000208 | -0.001721 | 0.001263 | 0.8 | 2 | CAUSAL_BY_RECOMPUTATION |
| mfi_14 | 36 | 0.000154 | -0.003095 | 0.001119 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| close_sma_ratio_20 | 36 | 0.000141 | -0.000104 | 0.000487 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |
| cci_14 | 36 | 0.000100 | -0.000118 | 0.000285 | 0.8 | 1 | CAUSAL_BY_RECOMPUTATION |

## Does ANY single feature beat the zero-return naive on its own (one-feature ridge, >= 3 of 5 blocks)? YES: bb_lower, bb_middle, bb_upper, bb_width, close_sma_ratio_10, close_sma_ratio_200, ema_10, ema_100, ema_20, ema_200, ema_50, ema_cross_20_100, hist_vol_10, hist_vol_60, hurst_proxy_200, log_return_1, log_return_5, macd_hist, natr_14, realized_var_12, realized_var_48, return_1, return_5, roll_kurt_ret_20, roll_kurt_ret_60, roll_skew_ret_60, roll_std_ret_252, roll_std_ret_60, sma_10, sma_100, sma_20, sma_200, sma_50, statistical__log_return_1, vol_regime_low, volume_ratio_20, volume_sma_10, vwap_60

## Top 20 by stable incremental utility (mean over horizons 1..6 of the median relative delta)

| rank | feature | score (delta / naive-zero MAE, mean over h) | mean sign agreement | leak verdict | rho next-bar |
|---|---|---|---|---|---|
| 1 | obv | 0.00059 | 0.77 | CAUSAL_BY_RECOMPUTATION | -0.000 |
| 2 | hurst_proxy_200 | 0.00043 | 0.66 | CAUSAL_BY_RECOMPUTATION | -0.022 |
| 3 | realized_var_48 | 0.00025 | 0.71 | CAUSAL_BY_RECOMPUTATION | 0.013 |
| 4 | roll_skew_ret_252 | 0.00021 | 0.74 | CAUSAL_BY_RECOMPUTATION | -0.002 |
| 5 | cci_14 | 0.00015 | 0.74 | CAUSAL_BY_RECOMPUTATION | 0.001 |
| 6 | log_return_5 | 0.00015 | 0.69 | CAUSAL_BY_RECOMPUTATION | 0.006 |
| 7 | zscore_close_100 | 0.00015 | 0.63 | CAUSAL_BY_RECOMPUTATION | 0.015 |
| 8 | return_5 | 0.00013 | 0.69 | CAUSAL_BY_RECOMPUTATION | 0.006 |
| 9 | sqret_autocorr_lag1_100 | 0.00011 | 0.60 | CAUSAL_BY_RECOMPUTATION | -0.004 |
| 10 | close_sma_ratio_10 | 0.00011 | 0.80 | CAUSAL_BY_RECOMPUTATION | -0.028 |
| 11 | close_sma_ratio_200 | 0.00011 | 0.97 | CAUSAL_BY_RECOMPUTATION | 0.024 |
| 12 | hist_vol_10 | 0.00009 | 0.63 | CAUSAL_BY_RECOMPUTATION | 0.030 |
| 13 | mfi_14 | 0.00009 | 0.80 | CAUSAL_BY_RECOMPUTATION | 0.006 |
| 14 | bb_width | 0.00005 | 0.74 | CAUSAL_BY_RECOMPUTATION | 0.016 |
| 15 | hist_vol_60 | 0.00004 | 0.69 | CAUSAL_BY_RECOMPUTATION | 0.015 |
| 16 | roll_std_ret_60 | 0.00004 | 0.69 | CAUSAL_BY_RECOMPUTATION | 0.015 |
| 17 | roll_kurt_ret_60 | 0.00004 | 0.86 | CAUSAL_BY_RECOMPUTATION | -0.017 |
| 18 | log_return_1 | 0.00003 | 0.69 | CAUSAL_BY_RECOMPUTATION | -0.058 |
| 19 | statistical__log_return_1 | 0.00003 | 0.69 | CAUSAL_BY_RECOMPUTATION | -0.058 |
| 20 | autocorr_lag1_100 | 0.00003 | 0.63 | CAUSAL_BY_RECOMPUTATION | -0.010 |

## Leak probe summary

- counts: {'CAUSALITY_NOT_VERIFIED_NOT_PRODUCED_BY_PRODUCER': 0, 'CAUSALITY_NOT_VERIFIED_PRODUCER_MISMATCH': 0, 'CAUSAL_BY_RECOMPUTATION': 83, 'FUTURE_ROWS_INFLUENCE_PRODUCER': 0}; suspect next-bar correlation: 0
- producer replayed: financial-data _scripts/workers/stage22_trading_features_worker.py @ 19fe375a (sha 7495a0d9); probe steps [3000, 6000, 9000, 12000, 13600]; burn-in 1500 rows


## Recommendation

- M03 (selection): no feature in variant A is recommended for inclusion on predictive grounds alone; the table ranks by stable incremental utility so a bounded screen can take the top rows as candidates, each with its leak verdict and its dossier support state.
- M07 (campaign): the paired naive gate (zero-return) is the reference every candidate must beat on the same rows; the reference rows above are the numbers to pair against.
- All numbers are DEVELOPMENT (git-pinned view, undeclared timestamp semantics); nothing here is confirmatory.
