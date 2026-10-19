# TIMING_PROXY_SELECTION tables (EURUSD_H1_TIMING, identity timing-selection:47d1e3fca7d01d2a)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `b4da1012c53fd21003ec36f1d0c9f0e79354db1661444013168f86592fc2cced`; TRAIN rows 18688; features 89 (30 inventory + 47 price_pool + 12 fractal = 89).

## Targets and support ledger

Semantics `timing_targets.v2`; pivots 7038, confirmed 7037, never confirmed 1; finality: all rows declared final unchanged under truncation and future perturbation = True.

| period | target | origins | LABELLED | CENSORED_UNCONFIRMED | CENSORED_END_OF_DATA | PURGED_SUPPORT_CROSSES_CUT | NO_LEG |
|---|---|---|---|---|---|---|---|
| INNER_A_2023H1 | t1 | 2445 | 2440 | 0 | 0 | 5 | 0 |
| INNER_A_2023H1 | t2 | 2445 | 2442 | 0 | 0 | 3 | 0 |
| INNER_A_2023H1 | t3 | 2445 | 2442 | 0 | 0 | 3 | 0 |
| INNER_B_2023H2 | t1 | 2964 | 2952 | 0 | 0 | 12 | 0 |
| INNER_B_2023H2 | t2 | 2964 | 2954 | 0 | 0 | 10 | 0 |
| INNER_B_2023H2 | t3 | 2964 | 2954 | 0 | 0 | 10 | 0 |
| REOBSERVED_2024 | t1 | 6243 | 6236 | 7 | 0 | 0 | 0 |
| REOBSERVED_2024 | t2 | 6243 | 6238 | 5 | 0 | 0 | 0 |
| REOBSERVED_2024 | t3 | 6243 | 6238 | 2 | 0 | 0 | 3 |
| TRAIN_2020_2022 | t1 | 18688 | 18659 | 0 | 0 | 29 | 0 |
| TRAIN_2020_2022 | t2 | 18688 | 18665 | 0 | 0 | 23 | 0 |
| TRAIN_2020_2022 | t3 | 18688 | 18665 | 0 | 0 | 23 | 0 |

## Selection recipe (primary T1 task only)

| step | features |
|---|---|
| MRMR K12 union over timing_pivot_k2, timing_valley_k2, timing_peak_k2 | 16: RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, RAW_LOG_RANGE, box_counting_dimension_w48, cal.dow_cos, cal.hod_cos, hurst_rs_w200, hurst_rs_w48, hurst_spectrum_width_w200, hurst_spectrum_width_w96, px.atr_ratio_14_120, px.close_loc_2, px.er_1, px.er_12, px.er_4, variogram_hurst_w96 |
| dropped by the T1 FDR screen | RAW_LOG_RANGE, box_counting_dimension_w48, cal.dow_cos, hurst_rs_w200, hurst_rs_w48, hurst_spectrum_width_w200, hurst_spectrum_width_w96, px.er_1, px.er_12, px.er_4 |
| pruned as redundant | - |
| SELECTED | 6: RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, cal.hod_cos, px.atr_ratio_14_120, px.close_loc_2, variogram_hurst_w96 |

Comparator subsets (reported, not used): timing_action_confirmed: RAW_CLOSE_LOG_RETURN, RAW_ELAPSED_HOURS, Stochastic_D, box_counting_dimension_w48, cal.dow_cos, cal.session_asia, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w96, px.er_1, px.er_4, px.logret_2; timing_expert_leg_remaining_pips: HIGH, RAW_ELAPSED_HOURS, RAW_LOG_RANGE, box_counting_dimension_w48, box_counting_dimension_w96, cal.session_asia, cal.session_us, hurst_rs_w48, px.close_loc_2, px.er_1, px.er_6, px.range_24

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| RAW_BAR_LOG_RETURN | 2174 | 0 | True | 18659 |
| RAW_CLOSE_LOG_RETURN | 2157 | 0 | True | 18659 |
| px.bar_body | 2174 | 0 | True | 18659 |
| px.close_loc_1 | 2423 | 0 | True | 18659 |
| px.close_loc_2 | 3078 | 0 | True | 18659 |
| px.close_loc_4 | 2034 | 0 | True | 18659 |
| px.close_loc_6 | 1437 | 0 | True | 18659 |
| px.logret_1 | 2157 | 0 | True | 18659 |
| px.logret_2 | 1823 | 0 | True | 18659 |
| px.close_loc_12 | 797.2 | 7.629e-174 | True | 18659 |
| WilliamsR | 683 | 4.988e-149 | True | 18659 |
| px.logret_4 | 624.5 | 2.484e-136 | True | 18659 |
| BB_position | 491.1 | 2.299e-107 | True | 18659 |
| px.close_loc_24 | 398 | 3.763e-87 | True | 18659 |
| px.logret_6 | 384.3 | 3.549e-84 | True | 18659 |
| RSI | 314.1 | 6.224e-69 | True | 18659 |
| CCI | 231.8 | 4.582e-51 | True | 18659 |
| px.close_loc_48 | 215.3 | 1.789e-47 | True | 18659 |
| ROC | 213.7 | 3.926e-47 | True | 18659 |
| px.logret_12 | 172.3 | 3.9e-38 | True | 18659 |
| price_minus_ema | 146.5 | 1.579e-32 | True | 18659 |
| Stochastic_K | 89.2 | 4.269e-20 | True | 18659 |
| px.logret_24 | 78.14 | 1.077e-17 | True | 18659 |
| MACD_Histogram | 72.79 | 1.564e-16 | True | 18659 |
| DI_plus | 70.43 | 5.086e-16 | True | 18659 |
| DI_minus | 69.26 | 9.137e-16 | True | 18659 |
| px.atr_ratio_14_120 | 23.43 | 8.183e-06 | True | 18659 |
| cal.hod_cos | 19.58 | 5.603e-05 | True | 18659 |
| hod_cos | 19.58 | 5.603e-05 | True | 18659 |
| cal.hod_sin | 15.49 | 0.0004337 | True | 18659 |
| hod_sin | 15.49 | 0.0004337 | True | 18659 |
| px.range_4 | 11.76 | 0.002791 | True | 18659 |
| px.range_ratio_1_24 | 11.59 | 0.003042 | True | 18659 |
| px.atr_14 | 11.4 | 0.003344 | True | 18659 |
| MACD | 9.93 | 0.006978 | True | 18659 |
| px.range_12 | 9.631 | 0.008104 | True | 18659 |
| px.er_2 | 8.971 | 0.01127 | True | 18657 |
| cal.session_asia | 8.116 | 0.01728 | True | 18659 |
| Stochastic_D | 7.489 | 0.02365 | True | 18659 |
| cal.session_us | 7.182 | 0.02758 | True | 18659 |
| ADX | 6.231 | 0.04436 | True | 18659 |
| RAW_ELAPSED_HOURS | 5.796 | 0.05514 | True | 18659 |
| px.elapsed_hours | 5.796 | 0.05514 | True | 18659 |
| CLOSE | 5.597 | 0.06092 | False | 18659 |
| OPEN | 5.456 | 0.06535 | False | 18659 |
| LOW | 5.454 | 0.06543 | False | 18659 |
| ATR | 5.335 | 0.06942 | False | 18659 |
| ATR_ratio | 5.335 | 0.06942 | False | 18659 |
| HIGH | 5.274 | 0.07158 | False | 18659 |
| RAW_LOG_RANGE | 5.23 | 0.07318 | False | 18659 |
| hurst_spectrum_width_w200 | 3.977 | 0.1369 | False | 18659 |
| cal.session_europe | 3.734 | 0.1546 | False | 18659 |
| hurst_rs_w200 | 2.69 | 0.2605 | False | 18659 |
| hurst_rs_w96 | 2.272 | 0.321 | False | 18659 |
| MACD_Signal | 2.222 | 0.3292 | False | 18659 |
| box_counting_dimension_w48 | 1.98 | 0.3715 | False | 18659 |
| hurst_spectrum_width_w96 | 1.657 | 0.4366 | False | 18659 |
| hurst_spectrum_width_w48 | 1.589 | 0.4518 | False | 18659 |
| cal.dow_cos | 1.067 | 0.5865 | False | 18659 |
| dow_cos | 1.067 | 0.5865 | False | 18659 |
| cal.dow_sin | 0.6869 | 0.7093 | False | 18659 |
| dow_sin | 0.6869 | 0.7093 | False | 18659 |
| BB_Width | 0.4015 | 0.8181 | False | 18659 |
| box_counting_dimension_w200 | 0.1296 | 0.9372 | False | 18659 |
| hurst_rs_w48 | 0.1129 | 0.9451 | False | 18659 |
| box_counting_dimension_w96 | 0.09723 | 0.9525 | False | 18659 |
| px.er_1 | nan | nan | False | 18535 |
| px.logret_48 | 44.93 | 1.756e-10 | True | 18659 |
| px.range_2 | 18.58 | 9.227e-05 | True | 18659 |
| variogram_hurst_w48 | 9.064 | 0.01076 | True | 18659 |
| px.rv_1 | 8.984 | 0.0112 | True | 18659 |
| variogram_hurst_w96 | 7.897 | 0.01928 | True | 18659 |
| px.rv_12 | 7.069 | 0.02918 | True | 18659 |
| px.range_6 | 5.527 | 0.06306 | False | 18659 |
| px.range_1 | 5.23 | 0.07318 | False | 18659 |
| variogram_hurst_w200 | 3.972 | 0.1372 | False | 18659 |
| px.er_24 | 3.284 | 0.1936 | False | 18659 |
| px.er_48 | 2.47 | 0.2908 | False | 18659 |
| px.rv_4 | 2.47 | 0.2908 | False | 18659 |
| px.er_4 | 1.964 | 0.3746 | False | 18659 |
| px.er_6 | 1.929 | 0.3812 | False | 18659 |
| rolling_std_24 | 1.781 | 0.4105 | False | 18659 |
| px.er_12 | 1.698 | 0.4278 | False | 18659 |
| px.rv_2 | 1.636 | 0.4413 | False | 18659 |
| px.rv_6 | 1.454 | 0.4833 | False | 18659 |
| px.range_48 | 1.127 | 0.5691 | False | 18659 |
| px.rv_48 | 0.8021 | 0.6696 | False | 18659 |
| px.rv_24 | 0.3978 | 0.8196 | False | 18659 |
| px.range_24 | 0.2273 | 0.8926 | False | 18659 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.8051 | 34 |
| prefix_before_2022-01-01 | 0.9245 | 44 |

Selected: FDR pass count over (folds + TRAIN) = RAW_BAR_LOG_RETURN: 3, RAW_ELAPSED_HOURS: 1, cal.hod_cos: 3, px.atr_ratio_14_120: 3, px.close_loc_2: 3, variogram_hurst_w96: 3

## Within-day variance

Constant within a day: cal.dow_cos, cal.dow_sin, dow_cos, dow_sin.

| selected feature | within-day variance ratio |
|---|---|
| RAW_BAR_LOG_RETURN | 0.9345 |
| RAW_ELAPSED_HOURS | 2.452 |
| cal.hod_cos | 0.899 |
| px.atr_ratio_14_120 | 0.5917 |
| px.close_loc_2 | 0.9097 |
| variogram_hurst_w96 | 0.0606 |

## Redundancy

| selected feature | VIF |
|---|---|
| RAW_BAR_LOG_RETURN | 1.768 |
| RAW_ELAPSED_HOURS | 1.004 |
| cal.hod_cos | 1.462 |
| px.atr_ratio_14_120 | 1.465 |
| px.close_loc_2 | 1.767 |
| variogram_hurst_w96 | 1 |

TRAIN Spearman clusters (|rho| >= 0.9, average linkage): 65 clusters, 12 with > 1 member.

| cluster | members |
|---|---|
| 02626b0fac834b098361cf2a77b6c32fd75812c6434271cbc7dca48fb6bd7cae | px.close_loc_4, px.close_loc_6 |
| 05839451dbf443ed9a24b28cd0bb8459fc595c9db42cb39e026608a1d095a4b6 | RSI, price_minus_ema, px.close_loc_24 |
| 227d085d376d6ae85753198a8ce20637d835772c2431db5e523c2fcd4a758979 | ATR, ATR_ratio, px.atr_14 |
| 2f34504634e55e2c2fbca4a5bb2868c071650898419d240c1059c4af1dd0fbf6 | BB_Width, rolling_std_24 |
| 345db8f506133e67adf19dffcf4013ac75c32d45bf8ccbf2d8e58dbf226e6616 | px.rv_24, px.rv_48 |
| 564e6ad6382005a5d7e3283eead4414291752f428cdee4c137888e5f21c3f373 | BB_position, CCI |
| 597aeef73b1370b7f390ae5c6fc5ff4c8bfd9e30c02f6b818f4b6d232fd99b09 | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| 76be372c3c9e0e86c24c38c2fd1d1d335c2deed48b323895a1841e49933307a3 | CLOSE, HIGH, LOW, OPEN |
| 76d5225562dc62dc24d07fb6d3e7c2149f48e86999d4077fa7ea4d1c108d9ead | px.range_4, px.range_6 |
| 7e76dcf684475b33cb59bbe2ef99d34c6f4f2966a77b8fe6dfef43750337bd27 | MACD, MACD_Signal |
| 98da1a2730f82c80761402583e35744ee3ac0ceb7cba72515af35a57321780f3 | WilliamsR, px.close_loc_12 |
| eec03d67b7f3a09bc5ae1783c12a6141f37541bf114a019b860fba374b306f05 | Stochastic_D, Stochastic_K |

Fractal columns vs the realized-vol family (Spearman, TRAIN):

| fractal | ATR | px.atr_14 | px.rv_1 | px.rv_12 | px.rv_2 | px.rv_24 | px.rv_4 | px.rv_48 | px.rv_6 | rolling_std_24 |
|---|---|---|---|---|---|---|---|---|---|---|
| box_counting_dimension_w200 | 0.06232 | 0.05376 | 0.04264 | 0.05915 | 0.05648 | 0.05395 | 0.0689 | 0.05578 | 0.07038 | -0.03782 |
| box_counting_dimension_w48 | -0.02795 | -0.02509 | 0.02341 | -0.03971 | -0.004481 | -0.03041 | -0.02694 | -0.03385 | -0.03841 | 0.03328 |
| box_counting_dimension_w96 | 0.01794 | -0.00407 | 0.06344 | 0.0025 | 0.07758 | 0.002418 | 0.07333 | -0.003884 | 0.05115 | -0.106 |
| hurst_rs_w200 | -0.1232 | -0.09986 | -0.04424 | -0.09455 | -0.05954 | -0.1228 | -0.07195 | -0.1474 | -0.0767 | -0.006038 |
| hurst_rs_w48 | -0.02325 | -0.02304 | -0.01915 | -0.03916 | -0.02304 | -0.03075 | -0.02724 | -0.04277 | -0.03142 | 0.01114 |
| hurst_rs_w96 | -0.09077 | -0.08076 | -0.04467 | -0.07733 | -0.0598 | -0.08369 | -0.06597 | -0.09407 | -0.07068 | -0.07749 |
| hurst_spectrum_width_w200 | 0.179 | 0.148 | 0.03698 | 0.1263 | 0.04789 | 0.1881 | 0.06438 | 0.2321 | 0.07886 | 0.1362 |
| hurst_spectrum_width_w48 | 0.09576 | 0.06821 | 0.006789 | 0.07693 | 0.01194 | 0.189 | 0.02179 | 0.2201 | 0.03227 | 0.2229 |
| hurst_spectrum_width_w96 | 0.1339 | 0.09913 | 0.02147 | 0.08847 | 0.0302 | 0.1633 | 0.04113 | 0.2245 | 0.05313 | 0.1317 |
| variogram_hurst_w200 | -0.1106 | -0.101 | -0.04401 | -0.1231 | -0.06482 | -0.1363 | -0.08938 | -0.1376 | -0.1055 | -0.02155 |
| variogram_hurst_w48 | 0.00319 | 0.02357 | -0.04188 | -0.01936 | -0.05523 | -0.01545 | -0.05673 | -0.02548 | -0.05057 | 0.299 |
| variogram_hurst_w96 | -0.05468 | -0.04818 | -0.04348 | -0.09064 | -0.05928 | -0.1091 | -0.07079 | -0.08409 | -0.07894 | 0.0516 |

## Causal stage: CONDITIONAL_PREDICTIVE_ASSOCIATION (tigramite 5.2.7.0 RegressionCI, PCMCI+, tau_max 2)

History H = ['px.logret_24', 'px.rv_24']. lag 0 = the candidate's value on the COMPLETED bar t (available at bar end) against the label of the window (t, t+k]; the label's support starts at bar t+1, so a lag-0 link is time-ordered x -> y. Controls: {'NEGATIVE_CONTROL_CIRCULAR_SHIFT': 'NO_LINK', 'NEGATIVE_CONTROL_NOISE': 'NO_LINK', 'POSITIVE_CONTROL_LABEL_LEAK_LAG1': 'LAGGED_PARENT'} (behave: True).

| feature | verdict | lag | edge | p | verdict H1 | verdict H2 |
|---|---|---|---|---|---|---|
| px.close_loc_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 8.022e-223 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_2 | LAGGED_PARENT | 0 | <-- | 2.855e-191 | LAGGED_PARENT | LAGGED_PARENT |
| px.close_loc_4 | LAGGED_PARENT | 0 | o-o | 1.063e-177 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| px.close_loc_6 | LAGGED_PARENT | 0 | --> | 8.766e-135 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RSI | LAGGED_PARENT | 0 | --> | 1.15e-98 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_12 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.51e-95 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_4 | LAG0_TIME_ORDERED_LINK | 0 | --> | 3.93e-91 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 5.597e-90 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| WilliamsR | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.372e-86 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| Stochastic_D | LAG0_TIME_ORDERED_LINK | 0 | --> | 9.07e-79 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RAW_BAR_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 3.039e-76 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| px.bar_body | LAG0_TIME_ORDERED_LINK | 0 | o-o | 3.039e-76 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| BB_position | LAG0_TIME_ORDERED_LINK | 0 | --> | 8.332e-75 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| RAW_CLOSE_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.08e-69 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| px.logret_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.08e-69 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| px.logret_2 | LAGGED_PARENT | 0 | <-- | 2.456e-67 | LAGGED_PARENT | LAGGED_PARENT |
| Stochastic_K | LAG0_TIME_ORDERED_LINK | 0 | o-o | 6.159e-63 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_6 | LAGGED_PARENT | 0 | x-x | 2.099e-59 | LAGGED_PARENT | LAG0_TIME_ORDERED_LINK |
| MACD | LAG0_TIME_ORDERED_LINK | 0 | o-o | 7.516e-55 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_48 | LAGGED_PARENT | 0 | --> | 3.154e-53 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| MACD_Histogram | CONFLICTING_OR_REVERSED | 0 | <-- | 2.113e-38 | CONFLICTING_OR_REVERSED | CONFLICTING_OR_REVERSED |
| px.logret_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 3.162e-33 |  |  |
| ROC | LAG0_TIME_ORDERED_LINK | 0 | --> | 2.114e-27 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_12 | LAG0_TIME_ORDERED_LINK | 0 | --> | 4.344e-14 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| price_minus_ema | CONFLICTING_OR_REVERSED | 0 | <-- | 1.077e-12 | NO_LINK | CONFLICTING_OR_REVERSED |
| px.logret_48 | LAG0_TIME_ORDERED_LINK | 0 | --> | 2.511e-12 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.range_4 | CONFLICTING_OR_REVERSED | 0 | <-- | 3.497e-10 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.range_2 | CONFLICTING_OR_REVERSED | 0 | <-- | 3.871e-08 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.range_ratio_1_24 | LAGGED_PARENT | 1 | --> | 2.971e-06 | LAGGED_PARENT | LAGGED_PARENT |
| CCI | LAG0_TIME_ORDERED_LINK | 0 | x-x | 8.752e-06 | CONFLICTING_OR_REVERSED | CONFLICTING_OR_REVERSED |
| ADX | LAG0_TIME_ORDERED_LINK | 0 | x-x | 9.344e-06 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.range_12 | CONFLICTING_OR_REVERSED | 0 | <-- | 1.022e-05 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.rv_1 | LAGGED_PARENT | 1 | --> | 2.04e-05 | NO_LINK | LAGGED_PARENT |
| px.er_2 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 0.0001098 | CONFLICTING_OR_REVERSED | NO_LINK |
| DI_plus | LAG0_TIME_ORDERED_LINK | 0 | o-o | 0.005474 | NO_LINK | NO_LINK |
| RAW_ELAPSED_HOURS | CONFLICTING_OR_REVERSED | 0 | <-- | 0.02046 | NO_LINK | NO_LINK |
| px.elapsed_hours | CONFLICTING_OR_REVERSED | 0 | <-- | 0.02046 | NO_LINK | NO_LINK |
| px.rv_24 | LAG0_TIME_ORDERED_LINK | 0 | x-x | 0.02771 |  |  |
| DI_minus | NO_LINK |  |  |  | NO_LINK | CONFLICTING_OR_REVERSED |
| cal.hod_cos | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| cal.hod_sin | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| cal.session_asia | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| cal.session_us | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| hod_cos | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| hod_sin | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.atr_14 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.atr_ratio_14_120 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.rv_12 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| variogram_hurst_w48 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| variogram_hurst_w96 | NO_LINK |  |  |  | NO_LINK | CONFLICTING_OR_REVERSED |

HAC lag screen (association comparator), timing_peak_k2: parents at lags 1..2 = 37: BB_position, CCI, DI_minus, DI_plus, MACD, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_D, Stochastic_K, WilliamsR, cal.hod_sin, cal.session_us, hod_sin, price_minus_ema, px.atr_14, px.atr_ratio_14_120, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_2, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.range_2, px.range_ratio_1_24, px.rv_1, px.rv_12

HAC lag screen (association comparator), timing_valley_k2: parents at lags 1..2 = 38: ADX, BB_position, CCI, DI_minus, DI_plus, MACD, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_D, Stochastic_K, WilliamsR, cal.hod_cos, cal.session_us, hod_cos, price_minus_ema, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_2, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.range_12, px.range_2, px.range_ratio_1_24, px.rv_1, px.rv_12, variogram_hurst_w96

### Matched-control contrast (association comparator; outcome = pivot within k)

| feature | events | excess | ci lo | ci hi | significant |
|---|---|---|---|---|---|
| variogram_hurst_w96 | 115 | 0.07609 | -0.02266 | 0.1722 | False |
| px.close_loc_2 | 2559 | 0.07349 | 0.0538 | 0.09426 | True |
| px.logret_48 | 379 | 0.05369 | -0.0006596 | 0.1039 | False |
| RSI | 711 | -0.05152 | -0.08926 | -0.01142 | True |
| DI_plus | 662 | -0.04703 | -0.08503 | -0.008114 | True |
| px.rv_12 | 324 | 0.04691 | -0.01224 | 0.1029 | False |
| px.range_2 | 1237 | -0.04558 | -0.07799 | -0.01846 | True |
| px.close_loc_48 | 627 | -0.04447 | -0.08646 | -0.001416 | True |
| MACD | 218 | -0.04128 | -0.1056 | 0.02573 | False |
| RAW_CLOSE_LOG_RETURN | 2797 | 0.04115 | 0.01907 | 0.05976 | True |
| px.logret_1 | 2797 | 0.04115 | 0.01907 | 0.05976 | True |
| RAW_BAR_LOG_RETURN | 2797 | 0.03818 | 0.01885 | 0.05563 | True |
| px.bar_body | 2797 | 0.03818 | 0.01885 | 0.05563 | True |
| px.range_ratio_1_24 | 1846 | -0.03732 | -0.06209 | -0.0128 | True |
| px.range_12 | 346 | -0.03468 | -0.09595 | 0.02486 | False |
| px.close_loc_1 | 3033 | 0.03239 | 0.01617 | 0.05047 | True |
| px.close_loc_24 | 892 | -0.02806 | -0.06846 | 0.007026 | False |
| ADX | 183 | -0.02732 | -0.1049 | 0.04809 | False |
| CCI | 848 | -0.02557 | -0.06197 | 0.01092 | False |
| MACD_Histogram | 519 | 0.0254 | -0.02451 | 0.07012 | False |
| DI_minus | 646 | -0.02508 | -0.06394 | 0.01162 | False |
| px.close_loc_4 | 2061 | 0.02358 | 0.001661 | 0.0446 | True |
| px.logret_6 | 1129 | -0.02257 | -0.05536 | 0.01041 | False |
| px.range_4 | 802 | -0.02251 | -0.05617 | 0.01671 | False |
| ROC | 887 | -0.02217 | -0.05785 | 0.01004 | False |
| px.logret_4 | 1414 | -0.01951 | -0.04967 | 0.009286 | False |
| variogram_hurst_w48 | 203 | 0.01946 | -0.06011 | 0.08741 | False |
| Stochastic_D | 539 | -0.01929 | -0.06104 | 0.02433 | False |
| px.close_loc_6 | 1744 | 0.01713 | -0.008669 | 0.04224 | False |
| px.logret_12 | 799 | -0.01569 | -0.04967 | 0.02162 | False |

## Predictive validation (proxy; identical rows)

| set | arm | cell | tau | period | role | PR-AUC valley | PR-AUC peak | base valley | log-loss | prec long | prec short | Sharpe | net USD | MDD % | entries | win % | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3268 | 0.3012 | 0.2012 | 1.032 | 0.3513 | 0.3057 | -1.58 | -2855 | 60.63 | 301.7 | 0.4663 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3627 | 0.3498 | 0.2131 | 1.011 | 0.4164 | 0.3741 | -2.561 | -5190 | 61.82 | 337.7 | 0.4828 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3595 | 0.353 | 0.2154 | 1.013 | 0.4266 | 0.398 | -1.339 | -5063 | 68.07 | 634.7 | 0.4731 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3197 | 0.2975 | 0.2012 | 1.045 | 0.3874 | 0.3338 | -1.478 | -4471 | 56.14 | 261 | 0.4974 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3793 | 0.3713 | 0.2131 | 1.012 | 0.4353 | 0.4439 | -1.963 | -4550 | 52.23 | 297 | 0.5094 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.366 | 0.3483 | 0.2154 | 1.023 | 0.4437 | 0.4063 | -0.9334 | -4166 | 65.14 | 519.7 | 0.48 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3384 | 0.3089 | 0.2012 | 1.01 | 0.4342 | 0.3636 | 0.01529 | -1236 | 38.85 | 117 | 0.4872 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.37 | 0.3672 | 0.2131 | 0.9814 | 0.3429 | 0.4217 | -0.2154 | -1213 | 32.18 | 120 | 0.4833 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3724 | 0.3472 | 0.2154 | 0.9944 | 0.439 | 0.4295 | 0.4354 | 1002 | 49.17 | 223 | 0.4753 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3572 | 0.3317 | 0.2012 | 1.022 | 0.37 | 0.3845 | -3.012 | -6573 | 72.32 | 383 | 0.4481 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3853 | 0.3941 | 0.2131 | 0.9975 | 0.4334 | 0.446 | -1.198 | -3492 | 45.74 | 436.3 | 0.4684 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3787 | 0.3935 | 0.2154 | 1 | 0.4313 | 0.445 | -1.408 | -5133 | 62.32 | 875.3 | 0.4632 | 1 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3683 | 0.3399 | 0.2012 | 1.016 | 0.3487 | 0.3096 | -3.814 | -5554 | 60.81 | 484.7 | 0.4236 | 1 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | INNER_B_2023H2 | FROZEN_SCORE | 0.4031 | 0.3887 | 0.2131 | 1.001 | 0.3774 | 0.3809 | -1.89 | -2943 | 39.1 | 503.7 | 0.446 | 1 |
| timing_selected | m2_gru | dropout0_weight_decay0.0001 | 0.4 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.388 | 0.3796 | 0.2154 | 1.007 | 0.3835 | 0.3736 | -3.972 | -7406 | 79.36 | 1037 | 0.4037 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3481 | 0.3366 | 0.2012 | 1.01 | 0.4096 | 0.4091 | -1.406 | -4359 | 54.11 | 127 | 0.4882 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3923 | 0.4007 | 0.2131 | 0.9822 | 0.4891 | 0.5392 | 0.8389 | 1705 | 29.02 | 153 | 0.5163 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.386 | 0.383 | 0.2154 | 0.9903 | 0.459 | 0.4592 | -1.733 | -6114 | 74.44 | 309 | 0.4693 | 1 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_A_2023H1 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_B_2023H2 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | REOBSERVED_2024 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
