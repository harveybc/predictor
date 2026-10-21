# TIMING_PROXY_SELECTION tables (EURUSD_H1_PROFIT_CROSS_ASSET_R2_ADMITTED, identity timing-selection:9d8aae907b3aa3c8)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `3ced12f6b5de0bd48927d1ce7162edc7dd1095d9dc264ce8fce9b6768cac82b4`; TRAIN rows 18698; features 151 (30 inventory + 51 price_pool + 12 fractal + 58 external = 151).

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
| MRMR K12 union over timing_profit_h72, timing_profit_long_h72, timing_profit_short_h72 | 17: CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, dow_cos, fx.eurgbp.ret_24, fx.eurgbp.ret_4, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.hours_since_session_open, px.unexpected_gap_hours, variogram_hurst_w200 |
| dropped by the T1 FDR screen | CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, fx.eurgbp.ret_24, fx.eurgbp.ret_4, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.unexpected_gap_hours, variogram_hurst_w200 |
| pruned as redundant | dow_cos |
| SELECTED | 1: px.hours_since_session_open |

Comparator subsets (reported, not used): timing_best_order_pnl_sigma_h72: RAW_ELAPSED_HOURS, btc.ret_24, cot.eur.net_change, dly.dax.ret_1d, dly.us10y.chg_5d, fx.eur_strength.ret_24, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.hours_since_session_open, px.unexpected_gap_hours; timing_long_order_wins_h72: CLOSE, RAW_ELAPSED_HOURS, btc.ret_24, dly.vix.chg_1d, dow_cos, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_4, px.er_48, px.unexpected_gap_hours

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| cal.utc_dow_cos | 301.4 | 3.516e-66 | True | 14823 |
| fx.eurjpy.ret_24 | 134.6 | 5.937e-30 | True | 14742 |
| hurst_rs_w200 | 64.72 | 8.837e-15 | True | 14790 |
| hurst_spectrum_width_w200 | 52.26 | 4.491e-12 | True | 14790 |
| hurst_spectrum_width_w48 | 48.99 | 2.3e-11 | True | 14823 |
| fx.usdjpy.ret_24 | 46.33 | 8.712e-11 | True | 14747 |
| px.atr_14 | 43.02 | 4.543e-10 | True | 14823 |
| fx.eurgbp.rv_24 | 41.46 | 9.943e-10 | True | 14685 |
| fx.usdjpy.rv_24 | 30.58 | 2.29e-07 | True | 14667 |
| hod_cos | 23.99 | 6.173e-06 | True | 14823 |
| fx.usdchf.rv_24 | 22.67 | 1.192e-05 | True | 14652 |
| fx.gbpusd.rv_24 | 22.16 | 1.539e-05 | True | 14665 |
| hurst_spectrum_width_w96 | 20.86 | 2.954e-05 | True | 14823 |
| hod_sin | 17.55 | 0.0001548 | True | 14823 |
| fx.eurjpy.ret_4 | 11.01 | 0.004072 | True | 14776 |
| fx.gbpusd.ret_4 | 9.78 | 0.00752 | True | 14780 |
| fx.usdjpy.ret_4 | 9.209 | 0.01001 | True | 14780 |
| hurst_rs_w96 | 8.825 | 0.01212 | True | 14823 |
| price_minus_ema | 7.854 | 0.0197 | True | 14823 |
| px.close_loc_48 | 7.502 | 0.02349 | True | 14823 |
| fx.eurjpy.rv_24 | 7.084 | 0.02896 | True | 14611 |
| px.close_loc_24 | 6.179 | 0.04552 | True | 14823 |
| fx.gbpusd.ret_24 | 5.886 | 0.05271 | True | 14746 |
| fx.usdchf.ret_24 | 5.635 | 0.05976 | True | 14744 |
| hurst_rs_w48 | 5.398 | 0.06727 | False | 14823 |
| px.atr_ratio_14_120 | 4.172 | 0.1242 | False | 14823 |
| fx.usdjpy.ret_1 | 3.205 | 0.2014 | False | 14786 |
| fx.usd_strength.ret_4 | 3.102 | 0.212 | False | 14764 |
| fx.eurjpy.ret_1 | 2.962 | 0.2274 | False | 14782 |
| fx.usd_strength.ret_24 | 2.314 | 0.3145 | False | 14731 |
| fx.usdchf.ret_1 | 1.556 | 0.4592 | False | 14784 |
| px.close_loc_12 | 1.55 | 0.4607 | False | 14823 |
| px.close_loc_1 | 0.9524 | 0.6211 | False | 14823 |
| px.close_loc_6 | 0.7999 | 0.6704 | False | 14823 |
| fx.usdchf.ret_4 | 0.7327 | 0.6933 | False | 14778 |
| px.close_loc_2 | 0.2649 | 0.876 | False | 14823 |
| px.close_loc_4 | 0.178 | 0.9148 | False | 14823 |
| fx.usd_strength.ret_1 | 0.1751 | 0.9162 | False | 14770 |
| px.bar_body | 0.1573 | 0.9244 | False | 14823 |
| px.elapsed_hours | nan | nan | False | 14823 |
| px.er_1 | nan | nan | False | 14721 |
| px.hours_since_session_open | 323.5 | 5.767e-71 | True | 14823 |
| cal.utc_dow_sin | 78.43 | 9.295e-18 | True | 14823 |
| ATR | 52.75 | 3.509e-12 | True | 14823 |
| ATR_ratio | 52.75 | 3.509e-12 | True | 14823 |
| btc.rv_24 | 47.28 | 5.42e-11 | True | 14636 |
| RAW_LOG_RANGE | 44.69 | 1.972e-10 | True | 14823 |
| ADX | 40.24 | 1.826e-09 | True | 14823 |
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
| btc.taker_buy_share_24 | 8.529 | 0.01406 | True | 14642 |
| cal.utc_hod_sin | 8.041 | 0.01794 | True | 14823 |
| ROC | 5.685 | 0.05829 | True | 14823 |
| btc.ret_24 | 5.475 | 0.06474 | True | 14788 |
| MACD_Histogram | 4.386 | 0.1116 | False | 14823 |
| box_counting_dimension_w200 | 4.151 | 0.1255 | False | 14790 |
| LOW | 3.316 | 0.1905 | False | 14823 |
| OPEN | 3.211 | 0.2008 | False | 14823 |
| CLOSE | 3.178 | 0.2041 | False | 14823 |
| HIGH | 3.027 | 0.2201 | False | 14823 |
| Stochastic_D | 2.549 | 0.2796 | False | 14823 |
| btc.ret_4 | 2.48 | 0.2894 | False | 14787 |
| Stochastic_K | 1.863 | 0.394 | False | 14823 |
| box_counting_dimension_w96 | 1.745 | 0.4178 | False | 14823 |
| CCI | 1.116 | 0.5723 | False | 14823 |
| WilliamsR | 1.074 | 0.5844 | False | 14823 |
| BB_position | 0.9 | 0.6376 | False | 14823 |
| btc.ret_1 | 0.8411 | 0.6567 | False | 14797 |
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
| px.range_6 | 70.75 | 4.324e-16 | True | 14823 |
| fx.eur_strength.ret_24 | 69.39 | 8.553e-16 | True | 14741 |
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
| dly.oil.ret_1d | 19.46 | 5.952e-05 | True | 14775 |
| px.er_48 | 19.17 | 6.875e-05 | True | 14823 |
| fx.audusd.rv_24 | 19.13 | 7.018e-05 | True | 14639 |
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
| fx.audusd.ret_24 | 7.233 | 0.02687 | True | 14744 |
| dly.dax.ret_1d | 7.005 | 0.03013 | True | 14823 |
| cal.utc_session_us | 6.252 | 0.0439 | True | 14823 |
| variogram_hurst_w200 | 5.925 | 0.05169 | True | 14790 |
| px.er_6 | 5.395 | 0.06736 | False | 14823 |
| dly.dax.ret_5d | 4.967 | 0.08343 | False | 14823 |
| fx.eur_strength.ret_4 | 4.215 | 0.1215 | False | 14776 |
| fx.eurgbp.ret_4 | 4.109 | 0.1282 | False | 14784 |
| fx.audusd.ret_4 | 4.019 | 0.1341 | False | 14778 |
| px.logret_4 | 4.015 | 0.1343 | False | 14823 |
| px.logret_12 | 3.48 | 0.1755 | False | 14823 |
| dly.us2y.chg_1d | 3.216 | 0.2003 | False | 14823 |
| variogram_hurst_w96 | 1.898 | 0.3871 | False | 14823 |
| fx.eurgbp.ret_24 | 1.885 | 0.3896 | False | 14749 |
| dly.us10y.chg_5d | 1.142 | 0.565 | False | 14823 |
| px.er_4 | 0.9941 | 0.6083 | False | 14823 |
| dly.us2y.chg_5d | 0.962 | 0.6182 | False | 14823 |
| fx.eurgbp.ret_1 | 0.6351 | 0.7279 | False | 14790 |
| fx.eur_strength.ret_1 | 0.49 | 0.7827 | False | 14782 |
| px.logret_2 | 0.4712 | 0.7901 | False | 14823 |
| fx.audusd.ret_1 | 0.2676 | 0.8748 | False | 14784 |
| px.logret_1 | 0.1834 | 0.9124 | False | 14823 |
| fx.gbpusd.ret_1 | 0.1317 | 0.9363 | False | 14786 |
| px.er_2 | 0.04442 | 0.978 | False | 14821 |
| px.unexpected_gap_hours | nan | nan | False | 14823 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.4355 | 120 |
| prefix_before_2022-01-01 | 0.8187 | 95 |

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
| 122e582d32638f9169366cdc0120be3dd2c612d87f3c516b49779e3a9cdd4e03 | WilliamsR, px.close_loc_12 |
| 29d954825e9ed90deceb670defc985d588c101b32069981703ae0be053522677 | cal.london_hod_cos, cal.utc_hod_cos, hod_cos |
| 54212b8582e133eb06faa82477cdbdb8e92e7392b6b48840f9df2276f14ecc36 | BB_position, CCI |
| 61e87e9784ffdd3bb0a617cf82ace3cf3248019c70ce7c1b2456f00f794e0e81 | BB_Width, rolling_std_24 |
| 6cb88e1a90e50c00164de7d483789acdc0959bdce7465834ec199e87582843ec | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| 6eef014805791429f4df99b8b9ea96620b88e05e3c2d483993558aead7e14026 | px.range_4, px.range_6 |
| 7859f70ed70d9312586f191b36bff6af93771f2be0b4e741a4e8d6846c8b8928 | ATR, ATR_ratio, px.atr_14 |
| 8ee58ff2da6bcdd610dd0dd4c142e47830111de36658c01c5e0683e800957ef3 | MACD, MACD_Signal |
| 8eeb3a6c5e056ee19f0d4b9d939fb9fbdc0885a1b7dc3d6c316e903a01e253f8 | cal.utc_dow_cos, dow_cos, px.hours_since_session_open |
| a393149ff88e0d0d1e619cfc22a1173cc09592e9d42731de781bc38be395c84e | cal.utc_dow_sin, dow_sin |
| a4655146edad9b66a885042a2f4ce703f6d4a36d60705363c2abb2d682748396 | cal.london_hod_sin, cal.utc_hod_sin, hod_sin |
| a59170e5ba506745ad771f515edee4625f8ca522b7d90c7bdedf7e86a232ea1a | px.close_loc_4, px.close_loc_6 |
| ac21b2d4de04cb129a540a0ac287fbc95ea5efd08375dd18d44a223a6b390b16 | RSI, price_minus_ema, px.close_loc_24 |
| b87ef943197b8a880c66ef90b1ba124c999da77487649845ec62399a693954e3 | px.rv_24, px.rv_48 |
| b92206498d2501a6e0e4d2061c67ca228ebcf8c34e337d13ff7a4b2dcee13b81 | Stochastic_D, Stochastic_K |
| e0e72e70ad74e4ce5396d41783c6fc8b224ae6396d0a996c3d8088ffefa9297d | CLOSE, HIGH, LOW, OPEN |

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
