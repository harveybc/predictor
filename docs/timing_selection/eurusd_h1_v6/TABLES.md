# TIMING_PROXY_SELECTION tables (EURUSD_H1_PROFIT_CROSS_ASSET_ADMITTED, identity timing-selection:c483ca79a60c6e14)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `e13292fe62366109e8aa2dcba0761d104b9ceacd32c749fa6865ba5d710d36af`; TRAIN rows 18698; features 151 (30 inventory + 51 price_pool + 12 fractal + 58 external = 151).

## Targets and support ledger

Semantics `timing_targets.profit.v1`; pivots 0, confirmed 0, never confirmed 0; finality: all rows declared final unchanged under truncation and future perturbation = True.

| period | target | origins | LABELLED | CENSORED_UNCONFIRMED | CENSORED_END_OF_DATA | PURGED_SUPPORT_CROSSES_CUT | NO_LEG |
|---|---|---|---|---|---|---|---|
| INNER_A_2023H1 | t1 | 3111 | 2773 | 338 | 0 | 0 | 0 |
| INNER_A_2023H1 | t2 | 3111 | 2773 | 338 | 0 | 0 | 0 |
| INNER_A_2023H1 | t3 | 3111 | 2773 | 0 | 0 | 0 | 338 |
| INNER_B_2023H2 | t1 | 3086 | 2748 | 338 | 0 | 0 | 0 |
| INNER_B_2023H2 | t2 | 3086 | 2748 | 338 | 0 | 0 | 0 |
| INNER_B_2023H2 | t3 | 3086 | 2748 | 0 | 0 | 0 | 338 |
| REOBSERVED_2024 | t1 | 6221 | 5519 | 701 | 1 | 0 | 0 |
| REOBSERVED_2024 | t2 | 6221 | 5519 | 701 | 1 | 0 | 0 |
| REOBSERVED_2024 | t3 | 6221 | 5519 | 0 | 1 | 0 | 701 |
| TRAIN_2020_2022 | t1 | 18698 | 16528 | 2170 | 0 | 0 | 0 |
| TRAIN_2020_2022 | t2 | 18698 | 16528 | 2170 | 0 | 0 | 0 |
| TRAIN_2020_2022 | t3 | 18698 | 16528 | 0 | 0 | 0 | 2170 |

## Selection recipe (primary T1 task only)

| step | features |
|---|---|
| MRMR K12 union over timing_profit_h72, timing_profit_long_h72, timing_profit_short_h72 | 16: CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, dow_cos, fx.eurgbp.ret_4, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.hours_since_session_open, px.unexpected_gap_hours, variogram_hurst_w200 |
| dropped by the T1 FDR screen | CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, fx.eurgbp.ret_4, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.unexpected_gap_hours, variogram_hurst_w200 |
| pruned as redundant | dow_cos |
| SELECTED | 1: px.hours_since_session_open |

Comparator subsets (reported, not used): timing_best_order_pnl_sigma_h72: RAW_ELAPSED_HOURS, btc.ret_24, cot.eur.net_change, dly.dax.ret_1d, dly.us10y.chg_5d, fx.eurgbp.ret_4, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_24, px.hours_since_session_open, px.unexpected_gap_hours; timing_long_order_wins_h72: CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, dow_cos, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_4, px.er_48, px.unexpected_gap_hours

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| cal.utc_dow_cos | 301.4 | 3.516e-66 | True | 14823 |
| fx.eurjpy.ret_24 | 140.2 | 3.592e-31 | True | 14792 |
| hurst_rs_w200 | 64.72 | 8.837e-15 | True | 14790 |
| hurst_spectrum_width_w200 | 52.26 | 4.491e-12 | True | 14790 |
| fx.usdjpy.ret_24 | 51.53 | 6.46e-12 | True | 14792 |
| hurst_spectrum_width_w48 | 48.99 | 2.3e-11 | True | 14823 |
| fx.eurgbp.rv_24 | 44.41 | 2.277e-10 | True | 14792 |
| px.atr_14 | 43.02 | 4.543e-10 | True | 14823 |
| fx.usdjpy.rv_24 | 33.26 | 5.995e-08 | True | 14792 |
| hod_cos | 23.99 | 6.173e-06 | True | 14823 |
| hurst_spectrum_width_w96 | 20.86 | 2.954e-05 | True | 14823 |
| fx.gbpusd.rv_24 | 19.94 | 4.684e-05 | True | 14792 |
| fx.usdchf.rv_24 | 19.15 | 6.953e-05 | True | 14792 |
| hod_sin | 17.55 | 0.0001548 | True | 14823 |
| fx.eurjpy.ret_4 | 10.77 | 0.004591 | True | 14792 |
| fx.usdjpy.ret_4 | 9.889 | 0.007123 | True | 14792 |
| fx.eurjpy.rv_24 | 9.366 | 0.009249 | True | 14792 |
| fx.gbpusd.ret_4 | 9.294 | 0.009592 | True | 14792 |
| hurst_rs_w96 | 8.825 | 0.01212 | True | 14823 |
| price_minus_ema | 7.854 | 0.0197 | True | 14823 |
| px.close_loc_48 | 7.502 | 0.02349 | True | 14823 |
| fx.gbpusd.ret_24 | 6.23 | 0.04438 | True | 14792 |
| px.close_loc_24 | 6.179 | 0.04552 | True | 14823 |
| fx.usdchf.ret_24 | 5.923 | 0.05175 | True | 14792 |
| hurst_rs_w48 | 5.398 | 0.06727 | False | 14823 |
| px.atr_ratio_14_120 | 4.172 | 0.1242 | False | 14823 |
| fx.usdjpy.ret_1 | 3.311 | 0.191 | False | 14792 |
| fx.usd_strength.ret_4 | 3.176 | 0.2043 | False | 14792 |
| fx.eurjpy.ret_1 | 2.966 | 0.227 | False | 14792 |
| fx.usd_strength.ret_24 | 2.209 | 0.3313 | False | 14792 |
| px.close_loc_12 | 1.55 | 0.4607 | False | 14823 |
| fx.usdchf.ret_1 | 1.542 | 0.4625 | False | 14792 |
| px.close_loc_1 | 0.9524 | 0.6211 | False | 14823 |
| px.close_loc_6 | 0.7999 | 0.6704 | False | 14823 |
| fx.usdchf.ret_4 | 0.7818 | 0.6765 | False | 14792 |
| px.close_loc_2 | 0.2649 | 0.876 | False | 14823 |
| px.close_loc_4 | 0.178 | 0.9148 | False | 14823 |
| fx.usd_strength.ret_1 | 0.1739 | 0.9167 | False | 14792 |
| px.bar_body | 0.1573 | 0.9244 | False | 14823 |
| px.elapsed_hours | nan | nan | False | 14823 |
| px.er_1 | nan | nan | False | 14721 |
| px.hours_since_session_open | 323.5 | 5.767e-71 | True | 14823 |
| cal.utc_dow_sin | 78.43 | 9.295e-18 | True | 14823 |
| ATR | 52.75 | 3.509e-12 | True | 14823 |
| ATR_ratio | 52.75 | 3.509e-12 | True | 14823 |
| RAW_LOG_RANGE | 44.69 | 1.972e-10 | True | 14823 |
| ADX | 40.24 | 1.826e-09 | True | 14823 |
| btc.rv_24 | 40 | 2.063e-09 | True | 14819 |
| cal.utc_hod_cos | 30.51 | 2.374e-07 | True | 14823 |
| cal.london_hod_cos | 27.94 | 8.567e-07 | True | 14823 |
| BB_Width | 15.02 | 0.000548 | True | 14823 |
| cal.london_hod_sin | 14.69 | 0.0006453 | True | 14823 |
| MACD | 12.19 | 0.002259 | True | 14823 |
| MACD_Signal | 11.77 | 0.002784 | True | 14823 |
| box_counting_dimension_w48 | 11.5 | 0.003189 | True | 14823 |
| RSI | 10.96 | 0.004174 | True | 14823 |
| DI_minus | 10.91 | 0.00427 | True | 14823 |
| DI_plus | 8.802 | 0.01227 | True | 14823 |
| cal.utc_hod_sin | 8.041 | 0.01794 | True | 14823 |
| btc.taker_buy_share_24 | 5.913 | 0.05199 | True | 14794 |
| ROC | 5.685 | 0.05829 | True | 14823 |
| MACD_Histogram | 4.386 | 0.1116 | False | 14823 |
| box_counting_dimension_w200 | 4.151 | 0.1255 | False | 14790 |
| LOW | 3.316 | 0.1905 | False | 14823 |
| OPEN | 3.211 | 0.2008 | False | 14823 |
| CLOSE | 3.178 | 0.2041 | False | 14823 |
| btc.ret_24 | 3.148 | 0.2072 | False | 14819 |
| HIGH | 3.027 | 0.2201 | False | 14823 |
| btc.ret_4 | 2.588 | 0.2742 | False | 14819 |
| Stochastic_D | 2.549 | 0.2796 | False | 14823 |
| Stochastic_K | 1.863 | 0.394 | False | 14823 |
| box_counting_dimension_w96 | 1.745 | 0.4178 | False | 14823 |
| btc.ret_1 | 1.229 | 0.541 | False | 14819 |
| CCI | 1.116 | 0.5723 | False | 14823 |
| WilliamsR | 1.074 | 0.5844 | False | 14823 |
| BB_position | 0.9 | 0.6376 | False | 14823 |
| RAW_CLOSE_LOG_RETURN | 0.1834 | 0.9124 | False | 14823 |
| RAW_BAR_LOG_RETURN | 0.1573 | 0.9244 | False | 14823 |
| RAW_ELAPSED_HOURS | nan | nan | False | 14823 |
| dow_cos | 305.4 | 4.815e-67 | True | 14823 |
| cot.eur.net_share | 200.5 | 2.857e-44 | True | 14823 |
| cot.eur.net_long | 132 | 2.193e-29 | True | 14823 |
| cot.eur.net_change | 125.2 | 6.368e-28 | True | 14823 |
| cot.eur.net_z156 | 98.66 | 3.763e-22 | True | 14823 |
| dow_sin | 98.36 | 4.389e-22 | True | 14823 |
| dly.us10y2y.chg_1d | 82.7 | 1.101e-18 | True | 14823 |
| dly.spx.ret_5d | 81.29 | 2.232e-18 | True | 14823 |
| dly.oil.ret_5d | 79.83 | 4.634e-18 | True | 14775 |
| fx.eur_strength.ret_24 | 71.21 | 3.441e-16 | True | 14792 |
| px.range_6 | 70.75 | 4.324e-16 | True | 14823 |
| px.range_4 | 64.97 | 7.796e-15 | True | 14823 |
| px.range_12 | 61.88 | 3.649e-14 | True | 14823 |
| px.range_2 | 60.14 | 8.716e-14 | True | 14823 |
| px.rv_6 | 56.17 | 6.361e-13 | True | 14823 |
| px.rv_4 | 54.56 | 1.421e-12 | True | 14823 |
| px.er_24 | 48.8 | 2.532e-11 | True | 14823 |
| dly.vix.level | 47.01 | 6.189e-11 | True | 14823 |
| px.range_1 | 44.69 | 1.972e-10 | True | 14823 |
| dly.spx.ret_1d | 42.6 | 5.622e-10 | True | 14823 |
| px.rv_2 | 40.97 | 1.271e-09 | True | 14823 |
| px.rv_12 | 37.85 | 6.041e-09 | True | 14823 |
| px.range_24 | 31.14 | 1.731e-07 | True | 14823 |
| px.rv_1 | 23.8 | 6.803e-06 | True | 14823 |
| px.rv_24 | 23.43 | 8.151e-06 | True | 14823 |
| cal.utc_session_asia | 22.69 | 1.181e-05 | True | 14823 |
| dly.gold.ret_5d | 22.4 | 1.367e-05 | True | 14823 |
| px.rv_48 | 21.9 | 1.759e-05 | True | 14823 |
| variogram_hurst_w48 | 21.28 | 2.39e-05 | True | 14823 |
| fx.audusd.rv_24 | 20.65 | 3.283e-05 | True | 14792 |
| dly.oil.ret_1d | 19.46 | 5.952e-05 | True | 14775 |
| px.er_48 | 19.17 | 6.875e-05 | True | 14823 |
| dly.us2y.level | 18.71 | 8.65e-05 | True | 14823 |
| dly.gold.ret_1d | 17.16 | 0.000188 | True | 14823 |
| px.er_12 | 16.28 | 0.0002916 | True | 14823 |
| dly.us10y.chg_1d | 15.94 | 0.000345 | True | 14823 |
| dly.us10y.level | 15.32 | 0.0004703 | True | 14823 |
| px.range_48 | 14.46 | 0.0007244 | True | 14823 |
| dly.us10y2y.chg_5d | 13.41 | 0.001224 | True | 14823 |
| px.range_ratio_1_24 | 12.25 | 0.002184 | True | 14823 |
| rolling_std_24 | 12.19 | 0.002249 | True | 14823 |
| cal.utc_session_europe | 11.73 | 0.002844 | True | 14823 |
| dly.vix.chg_1d | 11.35 | 0.003438 | True | 14823 |
| px.logret_48 | 10.51 | 0.00521 | True | 14823 |
| px.logret_24 | 9.074 | 0.0107 | True | 14823 |
| dly.us10y2y.level | 8.451 | 0.01462 | True | 14823 |
| px.logret_6 | 7.321 | 0.02572 | True | 14823 |
| dly.dax.ret_1d | 7.005 | 0.03013 | True | 14823 |
| cal.utc_session_us | 6.252 | 0.0439 | True | 14823 |
| variogram_hurst_w200 | 5.925 | 0.05169 | True | 14790 |
| fx.audusd.ret_24 | 5.916 | 0.05191 | True | 14792 |
| px.er_6 | 5.395 | 0.06736 | False | 14823 |
| dly.dax.ret_5d | 4.967 | 0.08343 | False | 14823 |
| fx.eurgbp.ret_4 | 4.219 | 0.1213 | False | 14792 |
| px.logret_4 | 4.015 | 0.1343 | False | 14823 |
| fx.eur_strength.ret_4 | 3.963 | 0.1378 | False | 14792 |
| fx.audusd.ret_4 | 3.561 | 0.1685 | False | 14792 |
| px.logret_12 | 3.48 | 0.1755 | False | 14823 |
| dly.us2y.chg_1d | 3.216 | 0.2003 | False | 14823 |
| fx.eurgbp.ret_24 | 2.022 | 0.3639 | False | 14792 |
| variogram_hurst_w96 | 1.898 | 0.3871 | False | 14823 |
| dly.us10y.chg_5d | 1.142 | 0.565 | False | 14823 |
| px.er_4 | 0.9941 | 0.6083 | False | 14823 |
| dly.us2y.chg_5d | 0.962 | 0.6182 | False | 14823 |
| fx.eurgbp.ret_1 | 0.6497 | 0.7226 | False | 14792 |
| fx.eur_strength.ret_1 | 0.473 | 0.7894 | False | 14792 |
| px.logret_2 | 0.4712 | 0.7901 | False | 14823 |
| fx.audusd.ret_1 | 0.2436 | 0.8853 | False | 14792 |
| px.logret_1 | 0.1834 | 0.9124 | False | 14823 |
| fx.gbpusd.ret_1 | 0.1322 | 0.936 | False | 14792 |
| px.er_2 | 0.04442 | 0.978 | False | 14821 |
| px.unexpected_gap_hours | nan | nan | False | 14823 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.4429 | 120 |
| prefix_before_2022-01-01 | 0.8215 | 95 |

Selected: FDR pass count over (folds + TRAIN) = px.hours_since_session_open: 3

## Within-day variance

Constant within a day: cal.utc_dow_cos, cal.utc_dow_sin, cot.eur.net_change, cot.eur.net_long, cot.eur.net_share, cot.eur.net_z156.

| selected feature | within-day variance ratio |
|---|---|
| px.hours_since_session_open | 0.03203 |

## Redundancy

| selected feature | VIF |
|---|---|
| px.hours_since_session_open | nan |

TRAIN Spearman clusters (|rho| >= 0.9, average linkage): 124 clusters, 16 with > 1 member.

| cluster | members |
|---|---|
| 02b906b2f04daea4a423ee51f13a79c6ea3adf74d43b9a9f8146f61e8812a1c0 | cal.utc_dow_sin, dow_sin |
| 04939e0c2d4e1213a566d390d7c8d13190f0b6e2a62c1acc1d9ca48a239828ee | px.close_loc_4, px.close_loc_6 |
| 30c4af0eb6a5e9a74f6e7eac20b1d31547bb468af0ae970c0727f0383188205f | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| 49c3f07ef414d25293cda920a0b983bcc9ba595f2e07a9826cf9c86303fcc96e | BB_position, CCI |
| 4d82fc3d97eb096bb600fca22a0a200e9217e039094be1a379ad9144a9fc12ce | CLOSE, HIGH, LOW, OPEN |
| 6f7407bc4fc85bcc2e8f00874d8acd43a37fd291929485029501df00a31b1348 | cal.london_hod_sin, cal.utc_hod_sin, hod_sin |
| 752425754c635d37a792971544ab252939082a5304d10cc4b476cec0d8bcf119 | WilliamsR, px.close_loc_12 |
| 850139099bce57a0522fbcf87bbb641d961664235d00e25301ab356c5a3130a4 | px.range_4, px.range_6 |
| 8dad3a203d32506993c92e25d837c6978ac93976c0e9f2fb85a47e09cf35b416 | px.rv_24, px.rv_48 |
| 9774965fc477d910fac6ca171f355050975edd9f534819951a8e8beb35256162 | cal.london_hod_cos, cal.utc_hod_cos, hod_cos |
| 9f243b0e11e85c9a57470837e9301496ffc8a01d9118bf1d5c2030c6912cfeb0 | ATR, ATR_ratio, px.atr_14 |
| a7a8741d7390fed7402c9d1d494f39434f24c93919df401f5983370907f05fc1 | RSI, price_minus_ema, px.close_loc_24 |
| b30e818b7f686ffee6bd86f8be4df83761ff1177e2d3ef97720f324671ae6b97 | BB_Width, rolling_std_24 |
| ba5f0bc2bd8b3b9e5b82f0534c9825234cc32fd5e498d4d9a38a10d751267045 | Stochastic_D, Stochastic_K |
| c3b4ebee4082dbf2e935498255f91fb30b33f96dfba3107a35a34d72dfa8c200 | MACD, MACD_Signal |
| e594b8e8c31850cbb4e84dcf8cbc1c57b25685d70a67c1e3ed1a1b75f3a98cef | cal.utc_dow_cos, dow_cos, px.hours_since_session_open |

Fractal columns vs the realized-vol family (Spearman, TRAIN):

| fractal | ATR | px.atr_14 | px.rv_1 | px.rv_12 | px.rv_2 | px.rv_24 | px.rv_4 | px.rv_48 | px.rv_6 | rolling_std_24 |
|---|---|---|---|---|---|---|---|---|---|---|
| box_counting_dimension_w200 | 0.06023 | 0.05256 | 0.04163 | 0.05738 | 0.05574 | 0.04931 | 0.06795 | 0.05101 | 0.06966 | -0.03753 |
| box_counting_dimension_w48 | -0.02718 | -0.02436 | 0.02202 | -0.0413 | -0.005621 | -0.0326 | -0.028 | -0.03456 | -0.03912 | 0.03394 |
| box_counting_dimension_w96 | 0.02026 | -0.0006069 | 0.06264 | 0.005828 | 0.07861 | 0.003866 | 0.07497 | -0.005039 | 0.05311 | -0.1009 |
| hurst_rs_w200 | -0.1317 | -0.1062 | -0.05012 | -0.1018 | -0.06505 | -0.1344 | -0.07652 | -0.1563 | -0.08142 | -0.01473 |
| hurst_rs_w48 | -0.02839 | -0.02871 | -0.0219 | -0.04281 | -0.02593 | -0.03302 | -0.03011 | -0.04367 | -0.03425 | 0.008718 |
| hurst_rs_w96 | -0.09465 | -0.08429 | -0.04712 | -0.08004 | -0.06173 | -0.08473 | -0.06785 | -0.09421 | -0.07228 | -0.07815 |
| hurst_spectrum_width_w200 | 0.175 | 0.1445 | 0.03676 | 0.1265 | 0.04739 | 0.1896 | 0.06405 | 0.2337 | 0.07843 | 0.1351 |
| hurst_spectrum_width_w48 | 0.09292 | 0.06478 | 0.008069 | 0.0741 | 0.01194 | 0.1841 | 0.02102 | 0.2184 | 0.03089 | 0.2177 |
| hurst_spectrum_width_w96 | 0.1341 | 0.09969 | 0.02151 | 0.09126 | 0.03059 | 0.1653 | 0.04263 | 0.2252 | 0.05514 | 0.1308 |
| variogram_hurst_w200 | -0.09898 | -0.092 | -0.04146 | -0.1179 | -0.06143 | -0.1293 | -0.08529 | -0.1298 | -0.1012 | -0.01883 |
| variogram_hurst_w48 | 0.001941 | 0.02248 | -0.04133 | -0.02133 | -0.05579 | -0.01984 | -0.05785 | -0.02758 | -0.05193 | 0.2954 |
| variogram_hurst_w96 | -0.05384 | -0.04798 | -0.04376 | -0.09116 | -0.05967 | -0.1088 | -0.07147 | -0.08237 | -0.07961 | 0.05264 |

## Causal stage: CONDITIONAL_PREDICTIVE_ASSOCIATION (tigramite 5.2.7.0 RegressionCI, PCMCI+, tau_max 2)

History H = ['px.logret_24', 'px.rv_24']. lag 0 = the candidate's value on the COMPLETED bar t (available at bar end) against the label of the window (t, t+k]; the label's support starts at bar t+1, so a lag-0 link is time-ordered x -> y. Controls: {'NEGATIVE_CONTROL_CIRCULAR_SHIFT': 'NO_LINK', 'NEGATIVE_CONTROL_NOISE': 'NO_LINK', 'POSITIVE_CONTROL_LABEL_LEAK_LAG1': 'LAGGED_PARENT'} (behave: True).

| feature | verdict | lag | edge | p | verdict H1 | verdict H2 |
|---|---|---|---|---|---|---|
| px.logret_2 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 0 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.949e-214 |  |  |
| cal.utc_dow_cos | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| dow_cos | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.hours_since_session_open | NO_LINK |  |  |  | LAGGED_PARENT | NO_LINK |
| px.rv_24 | NO_LINK |  |  |  |  |  |

HAC lag screen (association comparator), timing_profit_long_h72: parents at lags 1..72 = 1: px.hours_since_session_open

HAC lag screen (association comparator), timing_profit_short_h72: parents at lags 1..72 = 1: px.hours_since_session_open

### Matched-control contrast (association comparator; outcome = pivot within k)

| feature | events | excess | ci lo | ci hi | significant |
|---|---|---|---|---|---|
| px.hours_since_session_open | 53 | 0.8151 | 0.7094 | 0.9094 | True |
| px.logret_2 | 1560 | 0.00687 | -0.007094 | 0.02062 | False |

## Predictive validation (proxy; identical rows)

| set | arm | cell | tau | period | role | PR-AUC valley | PR-AUC peak | base valley | log-loss | prec long | prec short | Sharpe | net USD | MDD % | entries | win % | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.4659 | 0.4378 | 0.4435 | 1.078 | 0.5882 | 0.25 | -0.4437 | 428.8 | 10.73 | 8.667 | 0.2 | 0 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3644 | 0.5632 | 0.3842 | 1.078 | 0.3077 | 0.75 | -0.2627 | -519.1 | 9.674 | 5.667 | 0.1176 | 0 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4847 | 0.4733 | 0.4605 | 1.082 | 0.5714 | 0.5 | -0.06402 | -328.2 | 11.03 | 11.33 | 0.1765 | 0 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.001 | 0.7 | INNER_A_2023H1 | CHOICE_BLOCK | 0.4505 | 0.4383 | 0.4435 | 1.072 | nan | nan | 0 | 0 | 0 | 0 | 0 | 0 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.001 | 0.7 | INNER_B_2023H2 | FROZEN_SCORE | 0.3617 | 0.54 | 0.3842 | 1.089 | nan | nan | 0 | 0 | 0 | 0 | 0 | 0 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.001 | 0.7 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4494 | 0.4424 | 0.4605 | 1.113 | nan | nan | 0 | 0 | 0 | 0 | 0 | 0 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.4807 | 0.4304 | 0.4435 | 1.044 | 0.5714 | 0.375 | 0.9996 | 2170 | 25.67 | 30 | 0.5 | 0 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.4509 | 0.5741 | 0.3842 | 1.031 | 0.4545 | 0.3333 | -1.662 | -3423 | 47.53 | 28 | 0.4286 | 0 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.49 | 0.4956 | 0.4605 | 1.024 | 0.4516 | 0.4231 | -0.5528 | -2598 | 40.53 | 57 | 0.4035 | 0 |
| timing_selected | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.5158 | 0.4489 | 0.4435 | 1.066 | 0.5846 | 1 | 0.8191 | 1783 | 32.48 | 26 | 0.5128 | 0 |
| timing_selected | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.3686 | 0.5953 | 0.3842 | 1.06 | 0.4154 | 0 | -1.771 | -3807 | 53.09 | 26 | 0.3718 | 0 |
| timing_selected | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4757 | 0.4793 | 0.4605 | 1.065 | 0.4194 | 1 | -1.158 | -4509 | 49.05 | 52.33 | 0.4266 | 0 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.521 | 0.4466 | 0.4435 | 1.069 | 0.5769 | nan | 0.3851 | 563.8 | 21.65 | 17.33 | 0.3333 | 0 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.374 | 0.5899 | 0.3842 | 1.061 | 0.4231 | nan | -1.021 | -2310 | 35.4 | 17.33 | 0.2564 | 0 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4657 | 0.4834 | 0.4605 | 1.068 | 0.4175 | 0.6667 | -0.79 | -3061 | 32.9 | 35.33 | 0.283 | 0 |
| timing_selected | m3_hgb | l2_regularization0 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.5093 | 0.4619 | 0.4435 | 1.064 | 0.5769 | nan | 0.5794 | 852.9 | 32.48 | 26 | 0.5 | 0 |
| timing_selected | m3_hgb | l2_regularization0 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.3655 | 0.5893 | 0.3842 | 1.057 | 0.4231 | nan | -1.532 | -3465 | 53.09 | 26 | 0.3846 | 0 |
| timing_selected | m3_hgb | l2_regularization0 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4734 | 0.4793 | 0.4605 | 1.063 | 0.4231 | nan | -1.221 | -4671 | 47.49 | 52 | 0.4231 | 0 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.477 | 0.4215 | 0.4435 | 1.075 | 0.6795 | 0.4375 | 0.9486 | 2632 | 18.49 | 13.67 | 0.3642 | 0 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3709 | 0.5754 | 0.3842 | 1.055 | 0.359 | 0.6291 | -0.1094 | -482.5 | 21.31 | 14.33 | 0.3303 | 0 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.5165 | 0.4845 | 0.4605 | 1.063 | 0.648 | 0.6168 | 0.2119 | 476.7 | 23.46 | 24.67 | 0.3748 | 0 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.4881 | 0.4257 | 0.4435 | 1.091 | 0.5977 | 0.381 | 0.2394 | 642.6 | 28.92 | 19 | 0.5546 | 0 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.3766 | 0.5354 | 0.3842 | 1.091 | 0.38 | 0.5693 | -1.21 | -2289 | 38.51 | 19.67 | 0.4459 | 0 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4324 | 0.4726 | 0.4605 | 1.097 | 0.3425 | 0.5111 | -0.2212 | -937.6 | 38.86 | 38.33 | 0.4405 | 0 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.4579 | 0.4338 | 0.4435 | 1.045 | 0.5909 | 0.5833 | 0.8983 | 1845 | 26.32 | 34 | 0.5882 | 0 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3515 | 0.5081 | 0.3842 | 1.047 | 0.3125 | 0.4667 | -1.907 | -3752 | 41.73 | 31 | 0.4194 | 0 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4936 | 0.4926 | 0.4605 | 1.033 | 0.3929 | 0.5556 | -0.6921 | -2956 | 48.23 | 64 | 0.4844 | 0 |
| v5_selected_calendar_one | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.5158 | 0.4489 | 0.4435 | 1.066 | 0.5846 | 1 | 0.8191 | 1783 | 32.48 | 26 | 0.5128 | 0 |
| v5_selected_calendar_one | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.3686 | 0.5953 | 0.3842 | 1.06 | 0.4154 | 0 | -1.771 | -3807 | 53.09 | 26 | 0.3718 | 0 |
| v5_selected_calendar_one | m1_dmo_class | dropout0.2_weight_decay0.0001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4757 | 0.4793 | 0.4605 | 1.065 | 0.4194 | 1 | -1.158 | -4509 | 49.05 | 52.33 | 0.4266 | 0 |
| v5_selected_calendar_one | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.521 | 0.4466 | 0.4435 | 1.069 | 0.5769 | nan | 0.3851 | 563.8 | 21.65 | 17.33 | 0.3333 | 0 |
| v5_selected_calendar_one | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.374 | 0.5899 | 0.3842 | 1.061 | 0.4231 | nan | -1.021 | -2310 | 35.4 | 17.33 | 0.2564 | 0 |
| v5_selected_calendar_one | m2_gru | dropout0_weight_decay0.0001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4657 | 0.4834 | 0.4605 | 1.068 | 0.4175 | 0.6667 | -0.79 | -3061 | 32.9 | 35.33 | 0.283 | 0 |
| v5_selected_calendar_one | m3_hgb | l2_regularization0 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.5093 | 0.4619 | 0.4435 | 1.064 | 0.5769 | nan | 0.5794 | 852.9 | 32.48 | 26 | 0.5 | 0 |
| v5_selected_calendar_one | m3_hgb | l2_regularization0 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.3655 | 0.5893 | 0.3842 | 1.057 | 0.4231 | nan | -1.532 | -3465 | 53.09 | 26 | 0.3846 | 0 |
| v5_selected_calendar_one | m3_hgb | l2_regularization0 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.4734 | 0.4793 | 0.4605 | 1.063 | 0.4231 | nan | -1.221 | -4671 | 47.49 | 52 | 0.4231 | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_A_2023H1 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_B_2023H2 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | REOBSERVED_2024 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
