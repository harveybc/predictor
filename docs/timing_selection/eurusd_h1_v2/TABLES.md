# TIMING_PROXY_SELECTION tables (EURUSD_H1_TIMING, identity timing-selection:2c6689c8f93232d7)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `4fd6710e5a1fa746d6b1303ca871dbbe3c81fc516638e04eb352300c6bcbf2d0`; TRAIN rows 18688; features 89 (30 inventory + 47 price_pool + 12 fractal = 89).

## Targets and support ledger

Semantics `timing_targets.v2`; pivots 6371, confirmed 6191, never confirmed 180; finality: all rows declared final unchanged under truncation and future perturbation = True.

| period | target | origins | LABELLED | CENSORED_UNCONFIRMED | CENSORED_END_OF_DATA | PURGED_SUPPORT_CROSSES_CUT | NO_LEG |
|---|---|---|---|---|---|---|---|
| INNER_A_2023H1 | t1 | 2445 | 2381 | 54 | 0 | 10 | 0 |
| INNER_A_2023H1 | t2 | 2445 | 2383 | 54 | 0 | 8 | 0 |
| INNER_A_2023H1 | t3 | 2445 | 2383 | 54 | 0 | 8 | 0 |
| INNER_B_2023H2 | t1 | 2964 | 2874 | 75 | 0 | 15 | 0 |
| INNER_B_2023H2 | t2 | 2964 | 2876 | 75 | 0 | 13 | 0 |
| INNER_B_2023H2 | t3 | 2964 | 2876 | 75 | 0 | 13 | 0 |
| REOBSERVED_2024 | t1 | 6243 | 6089 | 154 | 0 | 0 | 0 |
| REOBSERVED_2024 | t2 | 6243 | 6091 | 152 | 0 | 0 | 0 |
| REOBSERVED_2024 | t3 | 6243 | 6091 | 149 | 0 | 0 | 3 |
| TRAIN_2020_2022 | t1 | 18688 | 18070 | 579 | 0 | 39 | 0 |
| TRAIN_2020_2022 | t2 | 18688 | 18076 | 579 | 0 | 33 | 0 |
| TRAIN_2020_2022 | t3 | 18688 | 18076 | 579 | 0 | 33 | 0 |

## Selection recipe (primary T1 task only)

| step | features |
|---|---|
| MRMR K12 union over timing_pivot_k2, timing_valley_k2, timing_peak_k2 | 19: CLOSE, RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, box_counting_dimension_w48, box_counting_dimension_w96, cal.dow_cos, cal.hod_sin, cal.session_europe, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w200, px.atr_ratio_14_120, px.close_loc_2, px.er_1, px.er_12, px.er_4, px.er_6, px.logret_24, variogram_hurst_w96 |
| dropped by the T1 FDR screen | CLOSE, RAW_ELAPSED_HOURS, box_counting_dimension_w48, box_counting_dimension_w96, cal.dow_cos, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w200, px.er_1, px.er_12, px.er_4, px.er_6 |
| pruned as redundant | - |
| SELECTED | 7: RAW_BAR_LOG_RETURN, cal.hod_sin, cal.session_europe, px.atr_ratio_14_120, px.close_loc_2, px.logret_24, variogram_hurst_w96 |

Comparator subsets (reported, not used): timing_action_confirmed: RAW_ELAPSED_HOURS, Stochastic_D, box_counting_dimension_w200, box_counting_dimension_w48, cal.dow_cos, cal.session_asia, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w48, px.er_1, px.er_6, px.logret_2; timing_expert_leg_remaining_pips: RAW_ELAPSED_HOURS, RAW_LOG_RANGE, cal.dow_cos, cal.session_asia, cal.session_us, hurst_rs_w48, hurst_rs_w96, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.rv_48

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| RAW_BAR_LOG_RETURN | 1907 | 0 | True | 18070 |
| RAW_CLOSE_LOG_RETURN | 1892 | 0 | True | 18070 |
| px.bar_body | 1907 | 0 | True | 18070 |
| px.close_loc_1 | 1965 | 0 | True | 18070 |
| px.close_loc_2 | 2575 | 0 | True | 18070 |
| px.close_loc_4 | 1911 | 0 | True | 18070 |
| px.logret_1 | 1892 | 0 | True | 18070 |
| px.logret_2 | 1665 | 0 | True | 18070 |
| px.close_loc_6 | 1402 | 4.539e-305 | True | 18070 |
| px.close_loc_12 | 819.7 | 1.013e-178 | True | 18070 |
| WilliamsR | 721.5 | 2.091e-157 | True | 18070 |
| px.logret_4 | 666.5 | 1.876e-145 | True | 18070 |
| BB_position | 522.5 | 3.557e-114 | True | 18070 |
| px.logret_6 | 433.1 | 8.985e-95 | True | 18070 |
| px.close_loc_24 | 397.9 | 3.964e-87 | True | 18070 |
| RSI | 331.8 | 9.091e-73 | True | 18070 |
| CCI | 272.5 | 6.569e-60 | True | 18070 |
| ROC | 234.1 | 1.456e-51 | True | 18070 |
| px.close_loc_48 | 224.5 | 1.809e-49 | True | 18070 |
| px.logret_12 | 203.7 | 5.86e-45 | True | 18070 |
| price_minus_ema | 173.6 | 2.033e-38 | True | 18070 |
| Stochastic_K | 128.8 | 1.087e-28 | True | 18070 |
| MACD_Histogram | 100.7 | 1.329e-22 | True | 18070 |
| DI_plus | 91.08 | 1.672e-20 | True | 18070 |
| px.logret_24 | 87.01 | 1.279e-19 | True | 18070 |
| DI_minus | 86.42 | 1.714e-19 | True | 18070 |
| px.logret_48 | 52.62 | 3.742e-12 | True | 18070 |
| cal.hod_sin | 22.51 | 1.297e-05 | True | 18070 |
| hod_sin | 22.51 | 1.297e-05 | True | 18070 |
| cal.session_europe | 21.72 | 1.917e-05 | True | 18070 |
| Stochastic_D | 17.07 | 0.0001964 | True | 18070 |
| cal.session_asia | 15.32 | 0.0004719 | True | 18070 |
| px.atr_ratio_14_120 | 15.23 | 0.0004941 | True | 18070 |
| ADX | 15.17 | 0.0005084 | True | 18070 |
| MACD | 14.24 | 0.0008094 | True | 18070 |
| px.rv_1 | 14.14 | 0.000852 | True | 18070 |
| variogram_hurst_w48 | 12.75 | 0.001702 | True | 18070 |
| px.er_2 | 10.97 | 0.004138 | True | 18068 |
| px.er_24 | 8.874 | 0.01183 | True | 18070 |
| px.range_2 | 7.867 | 0.01958 | True | 18070 |
| px.er_48 | 6.848 | 0.03258 | True | 18070 |
| CLOSE | 5.19 | 0.07465 | False | 18070 |
| HIGH | 4.639 | 0.0983 | False | 18070 |
| LOW | 4.633 | 0.09864 | False | 18070 |
| OPEN | 4.41 | 0.1103 | False | 18070 |
| px.atr_14 | 4.408 | 0.1104 | False | 18070 |
| hurst_spectrum_width_w48 | 3.813 | 0.1486 | False | 18070 |
| hurst_spectrum_width_w200 | 3.334 | 0.1888 | False | 18070 |
| cal.hod_cos | 3.302 | 0.1918 | False | 18070 |
| hod_cos | 3.302 | 0.1918 | False | 18070 |
| hurst_rs_w200 | 2.548 | 0.2797 | False | 18070 |
| MACD_Signal | 2.547 | 0.2798 | False | 18070 |
| RAW_ELAPSED_HOURS | 2.439 | 0.2954 | False | 18070 |
| px.elapsed_hours | 2.439 | 0.2954 | False | 18070 |
| cal.dow_sin | 2.411 | 0.2995 | False | 18070 |
| dow_sin | 2.411 | 0.2995 | False | 18070 |
| hurst_rs_w96 | 2.313 | 0.3146 | False | 18070 |
| box_counting_dimension_w200 | 2.042 | 0.3603 | False | 18070 |
| RAW_LOG_RANGE | 1.448 | 0.4849 | False | 18070 |
| box_counting_dimension_w48 | 1.243 | 0.5372 | False | 18070 |
| box_counting_dimension_w96 | 1.034 | 0.5964 | False | 18070 |
| cal.dow_cos | 0.8041 | 0.6689 | False | 18070 |
| dow_cos | 0.8041 | 0.6689 | False | 18070 |
| ATR_ratio | 0.5471 | 0.7607 | False | 18070 |
| ATR | 0.5471 | 0.7607 | False | 18070 |
| BB_Width | 0.3219 | 0.8513 | False | 18070 |
| hurst_spectrum_width_w96 | 0.2997 | 0.8608 | False | 18070 |
| cal.session_us | 0.2024 | 0.9037 | False | 18070 |
| hurst_rs_w48 | 0.07684 | 0.9623 | False | 18070 |
| px.er_1 | nan | nan | False | 17949 |
| px.range_4 | 8.878 | 0.01181 | True | 18070 |
| variogram_hurst_w96 | 6.154 | 0.0461 | True | 18070 |
| rolling_std_24 | 3.724 | 0.1553 | False | 18070 |
| variogram_hurst_w200 | 3.214 | 0.2005 | False | 18070 |
| px.range_ratio_1_24 | 3.061 | 0.2164 | False | 18070 |
| px.rv_6 | 2.739 | 0.2542 | False | 18070 |
| px.range_12 | 2.391 | 0.3026 | False | 18070 |
| px.range_24 | 2.257 | 0.3234 | False | 18070 |
| px.range_48 | 2.222 | 0.3292 | False | 18070 |
| px.rv_2 | 2.163 | 0.3391 | False | 18070 |
| px.er_4 | 1.501 | 0.472 | False | 18070 |
| px.range_1 | 1.448 | 0.4849 | False | 18070 |
| px.rv_4 | 1.106 | 0.5753 | False | 18070 |
| px.er_6 | 1.054 | 0.5905 | False | 18070 |
| px.rv_12 | 1.011 | 0.6031 | False | 18070 |
| px.range_6 | 0.7909 | 0.6734 | False | 18070 |
| px.er_12 | 0.4149 | 0.8127 | False | 18070 |
| px.rv_24 | 0.2822 | 0.8684 | False | 18070 |
| px.rv_48 | 0.05814 | 0.9713 | False | 18070 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.8094 | 41 |
| prefix_before_2022-01-01 | 0.9454 | 47 |

Selected: FDR pass count over (folds + TRAIN) = RAW_BAR_LOG_RETURN: 3, cal.hod_sin: 2, cal.session_europe: 2, px.atr_ratio_14_120: 3, px.close_loc_2: 3, px.logret_24: 3, variogram_hurst_w96: 2

## Within-day variance

Constant within a day: cal.dow_cos, cal.dow_sin, dow_cos, dow_sin.

| selected feature | within-day variance ratio |
|---|---|
| RAW_BAR_LOG_RETURN | 0.9345 |
| cal.hod_sin | 0.8009 |
| cal.session_europe | 0.8665 |
| px.atr_ratio_14_120 | 0.5917 |
| px.close_loc_2 | 0.9097 |
| px.logret_24 | 0.3068 |
| variogram_hurst_w96 | 0.0606 |

## Redundancy

| selected feature | VIF |
|---|---|
| RAW_BAR_LOG_RETURN | 1.793 |
| cal.hod_sin | 1.623 |
| cal.session_europe | 1.49 |
| px.atr_ratio_14_120 | 1.342 |
| px.close_loc_2 | 1.772 |
| px.logret_24 | 1.05 |
| variogram_hurst_w96 | 1.003 |

TRAIN Spearman clusters (|rho| >= 0.9, average linkage): 65 clusters, 12 with > 1 member.

| cluster | members |
|---|---|
| 127945fb8de73626834bf643e60b49fd6216b70dce9d5dc7584069b4272899ac | CLOSE, HIGH, LOW, OPEN |
| 4522b18c7ef67a36af77a2cbfa80bc67486167ad7affcdaf4829efeb1c19e2dd | WilliamsR, px.close_loc_12 |
| 4a5dba65dc7e320246c26a2a6355e7172018fa29fde1d9fa91d688def7f6dd34 | Stochastic_D, Stochastic_K |
| 53f12d8a032ccaa86b8a1087a7fc6d0b9cc2667b394397ea6d11a8893b0dd385 | ATR, ATR_ratio, px.atr_14 |
| 88fea16e5d4e6db31b68b10482a5589f7167afe06dee0c88ea33572e3af6b271 | BB_Width, rolling_std_24 |
| b712451d711ae7b462122f7533c822c98e348e8871bfa6371dd28f2cbad7c900 | px.range_4, px.range_6 |
| c3a202c3a77bea797268aa238b1d376551e5aefd104673ab216844f1e1d572ac | RSI, price_minus_ema, px.close_loc_24 |
| c5068587a5986684021c49ef595c20d40ab31de985294efca4f41f8386363495 | px.rv_24, px.rv_48 |
| d7b22458c6453d1386aad18030a9593c2709b6290e5764bc625f3248071c2750 | px.close_loc_4, px.close_loc_6 |
| e544fdb30c3cca67410731826d72cc99f22598204cefb4453526bde6749fc1a2 | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| e97806c3083242ac1ce7d3109743dd6c51e0ccd7dd1ac84884a67bb20e44dc55 | BB_position, CCI |
| ea35ba526440656da62a19bc9d7143b65f1dd0d4d9b0e1a1339054ae0a4d9f9e | MACD, MACD_Signal |

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
| px.close_loc_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.419e-150 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_4 | LAG0_TIME_ORDERED_LINK | 0 | --> | 9.949e-137 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_2 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 1.828e-109 | LAGGED_PARENT | LAG0_TIME_ORDERED_LINK |
| px.close_loc_6 | LAG0_TIME_ORDERED_LINK | 0 | --> | 2.409e-109 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_12 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.092e-81 | LAGGED_PARENT | LAG0_TIME_ORDERED_LINK |
| WilliamsR | LAG0_TIME_ORDERED_LINK | 0 | --> | 9.148e-76 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RSI | LAG0_TIME_ORDERED_LINK | 0 | --> | 8.589e-72 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_4 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.971e-65 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| Stochastic_D | LAG0_TIME_ORDERED_LINK | 0 | x-x | 1.395e-62 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| Stochastic_K | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.013e-54 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| BB_position | LAG0_TIME_ORDERED_LINK | 0 | --> | 3.58e-53 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RAW_BAR_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 1.947e-51 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.bar_body | LAG0_TIME_ORDERED_LINK | 0 | o-o | 1.947e-51 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_6 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.059e-48 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RAW_CLOSE_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 9.462e-46 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 9.462e-46 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.235e-44 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| MACD | LAG0_TIME_ORDERED_LINK | 0 | o-o | 8.77e-42 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_2 | LAG0_TIME_ORDERED_LINK | 0 | --> | 3.041e-38 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_48 | LAG0_TIME_ORDERED_LINK | 0 | --> | 3.171e-38 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| MACD_Histogram | LAG0_TIME_ORDERED_LINK | 0 | o-o | 7.28e-28 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_24 | CONFLICTING_OR_REVERSED | 0 | <-- | 1.469e-25 |  |  |
| ROC | LAG0_TIME_ORDERED_LINK | 0 | x-x | 5.167e-21 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.logret_12 | LAG0_TIME_ORDERED_LINK | 0 | --> | 4.576e-14 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_48 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.171e-09 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.range_4 | CONFLICTING_OR_REVERSED | 0 | <-- | 3.644e-09 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.rv_1 | LAGGED_PARENT | 1 | --> | 1.858e-05 | LAGGED_PARENT | NO_LINK |
| ADX | LAG0_TIME_ORDERED_LINK | 0 | x-x | 2.938e-05 | LAG0_TIME_ORDERED_LINK | NO_LINK |
| px.range_2 | CONFLICTING_OR_REVERSED | 0 | <-- | 0.0001938 | CONFLICTING_OR_REVERSED | CONFLICTING_OR_REVERSED |
| price_minus_ema | CONFLICTING_OR_REVERSED | 0 | <-- | 0.0001961 | NO_LINK | CONFLICTING_OR_REVERSED |
| DI_plus | LAG0_TIME_ORDERED_LINK | 0 | o-o | 0.004568 | NO_LINK | NO_LINK |
| CCI | NO_LINK |  |  |  | NO_LINK | LAG0_TIME_ORDERED_LINK |
| DI_minus | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| cal.hod_sin | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| cal.session_asia | NO_LINK |  |  |  | NO_LINK | LAGGED_PARENT |
| cal.session_europe | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| hod_sin | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.atr_ratio_14_120 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.er_2 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.er_24 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.er_48 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.rv_24 | NO_LINK |  |  |  |  |  |
| variogram_hurst_w48 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| variogram_hurst_w96 | NO_LINK |  |  |  | NO_LINK | NO_LINK |

HAC lag screen (association comparator), timing_peak_k2: parents at lags 1..2 = 29: BB_position, CCI, DI_minus, DI_plus, MACD, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_D, Stochastic_K, WilliamsR, cal.session_asia, price_minus_ema, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.rv_1

HAC lag screen (association comparator), timing_valley_k2: parents at lags 1..2 = 36: BB_position, CCI, DI_minus, DI_plus, MACD, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_D, Stochastic_K, WilliamsR, cal.hod_sin, cal.session_asia, hod_sin, price_minus_ema, px.atr_ratio_14_120, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.er_2, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.range_2, px.range_4, px.rv_1, variogram_hurst_w96

### Matched-control contrast (association comparator; outcome = pivot within k)

| feature | events | excess | ci lo | ci hi | significant |
|---|---|---|---|---|---|
| px.close_loc_2 | 2458 | 0.08403 | 0.06409 | 0.1051 | True |
| variogram_hurst_w96 | 112 | 0.06384 | -0.02778 | 0.158 | False |
| ADX | 179 | -0.05084 | -0.1322 | 0.03716 | False |
| px.close_loc_1 | 2914 | 0.04726 | 0.02868 | 0.0651 | True |
| px.range_2 | 1186 | -0.04387 | -0.07476 | -0.01345 | True |
| MACD | 204 | -0.04216 | -0.1157 | 0.02651 | False |
| variogram_hurst_w48 | 202 | 0.04175 | -0.03612 | 0.1097 | False |
| DI_plus | 623 | -0.03991 | -0.07847 | -0.0008253 | True |
| px.logret_48 | 364 | 0.03901 | -0.01423 | 0.09167 | False |
| px.range_4 | 771 | -0.03796 | -0.07951 | 0.002107 | False |
| RAW_CLOSE_LOG_RETURN | 2683 | 0.03439 | 0.01677 | 0.05565 | True |
| px.logret_1 | 2683 | 0.03439 | 0.01677 | 0.05565 | True |
| px.close_loc_4 | 1961 | 0.03292 | 0.009422 | 0.05547 | True |
| RAW_BAR_LOG_RETURN | 2684 | 0.03287 | 0.01451 | 0.05375 | True |
| px.bar_body | 2684 | 0.03287 | 0.01451 | 0.05375 | True |
| px.close_loc_6 | 1649 | 0.0306 | 0.003162 | 0.05597 | True |
| MACD_Histogram | 495 | 0.02997 | -0.01391 | 0.07735 | False |
| px.logret_6 | 1063 | -0.02581 | -0.05929 | 0.009708 | False |
| px.logret_4 | 1335 | -0.02541 | -0.05611 | 0.003374 | False |
| DI_minus | 632 | -0.02334 | -0.06628 | 0.02182 | False |
| px.er_48 | 495 | 0.02138 | -0.02559 | 0.06817 | False |
| Stochastic_D | 517 | -0.02083 | -0.07192 | 0.01971 | False |
| px.close_loc_48 | 592 | -0.02005 | -0.06512 | 0.02537 | False |
| Stochastic_K | 658 | -0.01791 | -0.05272 | 0.02435 | False |
| px.er_2 | 2680 | 0.01728 | -0.003697 | 0.03649 | False |
| px.rv_1 | 2297 | 0.01637 | -0.006611 | 0.04023 | False |
| CCI | 801 | -0.01533 | -0.05161 | 0.02196 | False |
| RSI | 672 | -0.01431 | -0.0554 | 0.02701 | False |
| px.close_loc_12 | 1264 | 0.01267 | -0.01405 | 0.04101 | False |
| px.close_loc_24 | 839 | -0.01186 | -0.04561 | 0.0268 | False |

## Predictive validation (proxy; identical rows)

| set | arm | cell | tau | period | role | PR-AUC valley | PR-AUC peak | base valley | log-loss | prec long | prec short | Sharpe | net USD | MDD % | entries | win % | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.2868 | 0.2688 | 0.1865 | 1.034 | 0.3291 | 0.332 | -1.086 | -2830 | 53.4 | 248 | 0.4797 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.312 | 0.3306 | 0.1903 | 1.005 | 0.3588 | 0.3439 | -1.489 | -3932 | 46.38 | 300.3 | 0.4551 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3299 | 0.3189 | 0.1966 | 1.002 | 0.4169 | 0.3831 | -0.9577 | -3678 | 55.15 | 575.3 | 0.4595 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3013 | 0.2683 | 0.1865 | 1.031 | 0.3404 | 0.2995 | -2.164 | -4767 | 56.26 | 291.3 | 0.4709 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3245 | 0.3498 | 0.1903 | 0.9984 | 0.381 | 0.3956 | -2.353 | -4767 | 54.7 | 333.7 | 0.4736 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.1_weight_decay0.001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3388 | 0.3195 | 0.1966 | 1.003 | 0.4421 | 0.3842 | -1.573 | -5124 | 61.56 | 638 | 0.4659 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.2973 | 0.2905 | 0.1865 | 1.01 | 0.34 | 0.4407 | -0.1448 | -1687 | 51.13 | 89 | 0.4719 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3358 | 0.3555 | 0.1903 | 0.9703 | 0.4909 | 0.4366 | -0.5492 | -1790 | 39.29 | 108 | 0.3981 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization0 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3421 | 0.3281 | 0.1966 | 0.9778 | 0.4274 | 0.4516 | -0.1176 | -1482 | 49.48 | 208 | 0.4615 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3183 | 0.2799 | 0.1865 | 1.017 | 0.3183 | 0.3051 | -2.377 | -5007 | 60.76 | 357 | 0.4414 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3268 | 0.3556 | 0.1903 | 0.9961 | 0.3597 | 0.3688 | -2.671 | -5015 | 55.01 | 404.7 | 0.4399 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3483 | 0.3473 | 0.1966 | 0.9905 | 0.4049 | 0.402 | -1.51 | -4203 | 56.8 | 803.3 | 0.4621 | 1 |
| timing_selected | m2_gru | dropout0.2_weight_decay0.0001 | 0.5 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3192 | 0.2815 | 0.1865 | 1.028 | 0.3216 | 0.319 | -2.384 | -5131 | 58.27 | 350 | 0.464 | 1 |
| timing_selected | m2_gru | dropout0.2_weight_decay0.0001 | 0.5 | INNER_B_2023H2 | FROZEN_SCORE | 0.3542 | 0.3825 | 0.1903 | 1.003 | 0.4027 | 0.4169 | -1.455 | -3448 | 50.99 | 401.7 | 0.471 | 1 |
| timing_selected | m2_gru | dropout0.2_weight_decay0.0001 | 0.5 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3663 | 0.3627 | 0.1966 | 1.001 | 0.4212 | 0.4241 | -2.083 | -5916 | 66.67 | 783.3 | 0.4697 | 1 |
| timing_selected | m3_hgb | l2_regularization0 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3253 | 0.2878 | 0.1865 | 1.01 | 0.4722 | 0.2976 | -2.283 | -5393 | 67.16 | 138 | 0.4493 | 1 |
| timing_selected | m3_hgb | l2_regularization0 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3308 | 0.3745 | 0.1903 | 0.9798 | 0.3919 | 0.4756 | -0.5698 | -1985 | 27.24 | 135 | 0.4667 | 1 |
| timing_selected | m3_hgb | l2_regularization0 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3526 | 0.3442 | 0.1966 | 0.9828 | 0.4357 | 0.3972 | -0.592 | -2928 | 49.51 | 245 | 0.4898 | 1 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_A_2023H1 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_B_2023H2 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | REOBSERVED_2024 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
