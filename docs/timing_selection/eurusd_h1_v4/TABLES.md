# TIMING_PROXY_SELECTION tables (EURUSD_H1_TIMING_ADMITTED, identity timing-selection:20b92b59e7c9d43f)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `ca5d75c28b6da80e2c832546445aca989e44b6f37e9b57b91276b54b2e7e6163`; TRAIN rows 18698; features 93 (30 inventory + 51 price_pool + 12 fractal = 93).

## Targets and support ledger

Semantics `timing_targets.v2`; pivots 7094, confirmed 7093, never confirmed 1; finality: all rows declared final unchanged under truncation and future perturbation = True.

| period | target | origins | LABELLED | CENSORED_UNCONFIRMED | CENSORED_END_OF_DATA | PURGED_SUPPORT_CROSSES_CUT | NO_LEG |
|---|---|---|---|---|---|---|---|
| INNER_A_2023H1 | t1 | 3111 | 3102 | 0 | 0 | 9 | 0 |
| INNER_A_2023H1 | t2 | 3111 | 3104 | 0 | 0 | 7 | 0 |
| INNER_A_2023H1 | t3 | 3111 | 3104 | 0 | 0 | 7 | 0 |
| INNER_B_2023H2 | t1 | 3086 | 3075 | 0 | 0 | 11 | 0 |
| INNER_B_2023H2 | t2 | 3086 | 3077 | 0 | 0 | 9 | 0 |
| INNER_B_2023H2 | t3 | 3086 | 3077 | 0 | 0 | 9 | 0 |
| REOBSERVED_2024 | t1 | 6221 | 6215 | 6 | 0 | 0 | 0 |
| REOBSERVED_2024 | t2 | 6221 | 6217 | 4 | 0 | 0 | 0 |
| REOBSERVED_2024 | t3 | 6221 | 6217 | 2 | 0 | 0 | 2 |
| TRAIN_2020_2022 | t1 | 18698 | 18669 | 0 | 0 | 29 | 0 |
| TRAIN_2020_2022 | t2 | 18698 | 18675 | 0 | 0 | 23 | 0 |
| TRAIN_2020_2022 | t3 | 18698 | 18675 | 0 | 0 | 23 | 0 |

## Selection recipe (primary T1 task only)

| step | features |
|---|---|
| MRMR K12 union over timing_pivot_k2, timing_valley_k2, timing_peak_k2 | 19: ATR, RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, RAW_LOG_RANGE, box_counting_dimension_w48, cal.london_hod_cos, cal.utc_session_us, dow_cos, hod_cos, hurst_rs_w200, hurst_rs_w48, hurst_spectrum_width_w200, hurst_spectrum_width_w96, px.close_loc_2, px.er_1, px.er_12, px.er_4, px.unexpected_gap_hours, variogram_hurst_w96 |
| dropped by the T1 FDR screen | ATR, RAW_ELAPSED_HOURS, RAW_LOG_RANGE, box_counting_dimension_w48, cal.london_hod_cos, dow_cos, hod_cos, hurst_rs_w200, hurst_rs_w48, hurst_spectrum_width_w96, px.er_1, px.er_12, px.er_4, px.unexpected_gap_hours, variogram_hurst_w96 |
| pruned as redundant | - |
| SELECTED | 4: RAW_BAR_LOG_RETURN, cal.utc_session_us, hurst_spectrum_width_w200, px.close_loc_2 |

Comparator subsets (reported, not used): timing_action_confirmed: RAW_BAR_LOG_RETURN, RAW_ELAPSED_HOURS, Stochastic_D, box_counting_dimension_w48, cal.utc_dow_cos, cal.utc_hod_sin, hurst_rs_w48, hurst_rs_w96, px.er_1, px.er_4, px.logret_2, px.unexpected_gap_hours; timing_expert_leg_remaining_pips: RAW_ELAPSED_HOURS, RAW_LOG_RANGE, cal.utc_hod_sin, cal.utc_session_europe, dow_cos, hurst_rs_w48, px.close_loc_2, px.er_1, px.er_6, px.rv_48, px.unexpected_gap_hours, variogram_hurst_w96

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| RAW_BAR_LOG_RETURN | 1779 | 0 | True | 14773 |
| RAW_CLOSE_LOG_RETURN | 1775 | 0 | True | 14773 |
| WilliamsR | 557 | 1.098e-121 | True | 14771 |
| BB_position | 402.1 | 4.929e-88 | True | 14765 |
| RSI | 246.6 | 2.858e-54 | True | 14770 |
| CCI | 193.8 | 8.384e-43 | True | 14765 |
| ROC | 174.7 | 1.153e-38 | True | 14773 |
| price_minus_ema | 117 | 3.985e-26 | True | 14773 |
| Stochastic_K | 72.13 | 2.17e-16 | True | 14769 |
| MACD_Histogram | 62.98 | 2.114e-14 | True | 14752 |
| DI_minus | 57.26 | 3.681e-13 | True | 14770 |
| DI_plus | 52.37 | 4.247e-12 | True | 14770 |
| cal.utc_hod_sin | 23.03 | 9.988e-06 | True | 14773 |
| cal.utc_session_us | 22.17 | 1.534e-05 | True | 14773 |
| cal.london_hod_sin | 18.62 | 9.035e-05 | True | 14773 |
| hod_sin | 18.18 | 0.000113 | True | 14773 |
| cal.london_hod_cos | 11.68 | 0.002915 | True | 14773 |
| hod_cos | 10.59 | 0.005007 | True | 14773 |
| cal.utc_hod_cos | 9.059 | 0.01079 | True | 14773 |
| cal.utc_session_asia | 8.331 | 0.01552 | True | 14773 |
| hurst_spectrum_width_w200 | 7.636 | 0.02197 | True | 14635 |
| dow_cos | 7.277 | 0.02629 | True | 14773 |
| MACD | 6.854 | 0.03248 | True | 14759 |
| Stochastic_D | 6.771 | 0.03386 | True | 14767 |
| cal.utc_dow_cos | 4.526 | 0.104 | False | 14773 |
| ADX | 4.507 | 0.1051 | False | 14757 |
| hurst_rs_w200 | 3.884 | 0.1434 | False | 14635 |
| hurst_spectrum_width_w96 | 3.274 | 0.1946 | False | 14714 |
| hurst_rs_w96 | 2.673 | 0.2627 | False | 14714 |
| MACD_Signal | 2.196 | 0.3335 | False | 14752 |
| hurst_spectrum_width_w48 | 2.123 | 0.3459 | False | 14752 |
| box_counting_dimension_w48 | 1.812 | 0.4041 | False | 14752 |
| cal.utc_session_europe | 1.289 | 0.525 | False | 14773 |
| CLOSE | 1.23 | 0.5407 | False | 14773 |
| dow_sin | 1.229 | 0.5409 | False | 14773 |
| OPEN | 1.065 | 0.5871 | False | 14773 |
| LOW | 0.9891 | 0.6098 | False | 14773 |
| HIGH | 0.9827 | 0.6118 | False | 14773 |
| ATR | 0.8745 | 0.6458 | False | 14770 |
| ATR_ratio | 0.8745 | 0.6458 | False | 14770 |
| RAW_LOG_RANGE | 0.4837 | 0.7852 | False | 14773 |
| BB_Width | 0.3742 | 0.8294 | False | 14765 |
| RAW_ELAPSED_HOURS | nan | nan | False | 14773 |
| px.bar_body | 1779 | 0 | True | 14773 |
| px.close_loc_1 | 1931 | 0 | True | 14773 |
| px.close_loc_2 | 2484 | 0 | True | 14773 |
| px.close_loc_4 | 1652 | 0 | True | 14773 |
| px.logret_1 | 1775 | 0 | True | 14773 |
| px.logret_2 | 1526 | 0 | True | 14773 |
| px.close_loc_6 | 1176 | 3.761e-256 | True | 14773 |
| px.close_loc_12 | 647.9 | 2.026e-141 | True | 14773 |
| px.logret_4 | 509.8 | 2.028e-111 | True | 14773 |
| px.close_loc_24 | 322.6 | 8.931e-71 | True | 14761 |
| px.logret_6 | 320.5 | 2.519e-70 | True | 14773 |
| px.close_loc_48 | 171.6 | 5.421e-38 | True | 14752 |
| px.logret_12 | 145.8 | 2.167e-32 | True | 14772 |
| px.logret_24 | 63.21 | 1.876e-14 | True | 14760 |
| px.atr_ratio_14_120 | 14.84 | 0.0005981 | True | 14690 |
| px.atr_14 | 4.148 | 0.1257 | False | 14771 |
| px.elapsed_hours | nan | nan | False | 14773 |
| px.er_1 | nan | nan | False | 14670 |
| px.logret_48 | 32.68 | 7.995e-08 | True | 14752 |
| px.rv_1 | 17.48 | 0.0001604 | True | 14773 |
| px.er_2 | 11 | 0.004089 | True | 14771 |
| variogram_hurst_w48 | 8.432 | 0.01476 | True | 14752 |
| variogram_hurst_w96 | 6.339 | 0.04203 | True | 14714 |
| px.range_2 | 5.05 | 0.08004 | False | 14773 |
| px.er_4 | 4.676 | 0.09651 | False | 14773 |
| px.er_24 | 3.734 | 0.1546 | False | 14760 |
| px.er_12 | 3.579 | 0.167 | False | 14772 |
| px.er_48 | 3.315 | 0.1906 | False | 14752 |
| px.rv_6 | 3.188 | 0.2031 | False | 14773 |
| px.rv_2 | 3.185 | 0.2034 | False | 14773 |
| px.range_4 | 3.098 | 0.2125 | False | 14773 |
| px.range_ratio_1_24 | 3.085 | 0.2138 | False | 14761 |
| px.range_6 | 2.889 | 0.2358 | False | 14773 |
| rolling_std_24 | 2.663 | 0.2641 | False | 14761 |
| px.hours_since_session_open | 2.552 | 0.2792 | False | 14773 |
| px.range_48 | 2.488 | 0.2882 | False | 14752 |
| px.range_12 | 2.389 | 0.3029 | False | 14773 |
| px.er_6 | 1.948 | 0.3775 | False | 14773 |
| px.rv_12 | 1.88 | 0.3906 | False | 14772 |
| variogram_hurst_w200 | 1.569 | 0.4564 | False | 14635 |
| px.range_24 | 1.371 | 0.5038 | False | 14761 |
| px.rv_4 | 1.143 | 0.5648 | False | 14773 |
| px.range_1 | 0.4837 | 0.7852 | False | 14773 |
| box_counting_dimension_w200 | 0.4619 | 0.7938 | False | 14635 |
| cal.utc_dow_sin | 0.2709 | 0.8733 | False | 14773 |
| hurst_rs_w48 | 0.2238 | 0.8941 | False | 14752 |
| px.rv_48 | 0.1857 | 0.9113 | False | 14752 |
| box_counting_dimension_w96 | 0.1412 | 0.9319 | False | 14714 |
| px.rv_24 | 0.08461 | 0.9586 | False | 14760 |
| px.unexpected_gap_hours | nan | nan | False | 14773 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.8308 | 37 |
| prefix_before_2022-01-01 | 0.9376 | 47 |

Selected: FDR pass count over (folds + TRAIN) = RAW_BAR_LOG_RETURN: 3, cal.utc_session_us: 3, hurst_spectrum_width_w200: 1, px.close_loc_2: 3

## Within-day variance

Constant within a day: cal.utc_dow_cos, cal.utc_dow_sin.

| selected feature | within-day variance ratio |
|---|---|
| RAW_BAR_LOG_RETURN | 0.8908 |
| cal.utc_session_us | 0.8432 |
| hurst_spectrum_width_w200 | 0.04371 |
| px.close_loc_2 | 0.8582 |

## Redundancy

| selected feature | VIF |
|---|---|
| RAW_BAR_LOG_RETURN | 1.768 |
| cal.utc_session_us | 1 |
| hurst_spectrum_width_w200 | 1 |
| px.close_loc_2 | 1.769 |

TRAIN Spearman clusters (|rho| >= 0.9, average linkage): 66 clusters, 16 with > 1 member.

| cluster | members |
|---|---|
| 1c6dbadc337812dda4375ded803e9506bde02e6e21ff07b49f82e243379e2573 | MACD, MACD_Signal |
| 2201e3376fbbfc83f969f04a0e6f95e80a9bd3ec5878ba25b70c077d61139cd9 | WilliamsR, px.close_loc_12 |
| 2eb8cb7d552321a4fa756d194279a6dcc1e8181c272c2c5ec417512fb0d7a569 | px.range_4, px.range_6 |
| 34fbd927cd3359a0d71dd31e3873a909d72b8a3e632881ade4499c2e98daf500 | cal.utc_dow_cos, dow_cos, px.hours_since_session_open |
| 3f676114ec19d5de50daa74878c19261d513483d47f3ab3b55f3cf4ad89f74da | cal.london_hod_sin, cal.utc_hod_sin, hod_sin |
| 52a3015adfc6104ea3ebdd51c279a9b815e69fadaebee1687ac9045afcb4188f | px.close_loc_4, px.close_loc_6 |
| 6871229174f5590dc9d61d8ec91903a5af9fa4a360899f6188672f2e9d961565 | BB_position, CCI |
| 6d6dd2720ddb24f81411a9751f6276a254b2bee3bf0072a3519b6c7a562a824b | BB_Width, rolling_std_24 |
| 89a1d68d819d2389706d103b03180a39c6e497becb72e214b8a5d1587d29ccd0 | px.rv_24, px.rv_48 |
| 9725614c6a8d22af606c2e8c4e2bea28997598e88f9fb8350ca0b87d9ed009a9 | cal.london_hod_cos, cal.utc_hod_cos, hod_cos |
| b1f88a3dbee13460d9c34fd87e6f0cdfb5b689d5d56e789fefecf1d5890c88c8 | RSI, price_minus_ema, px.close_loc_24 |
| bd01f380d0387954cf7100aaa01e4a5b11fad732368c812d63eee6885009cdad | ATR, ATR_ratio, px.atr_14 |
| ed76e24325973fc3851d167d59f915f7204dbc560f8622b86588d2ecce3595db | Stochastic_D, Stochastic_K |
| ee64426086e42128fc75b428c30bcee306d07e4a0a656d0c6bbbe59fa945bc05 | cal.utc_dow_sin, dow_sin |
| f757a8536f8a9f3eddeb22b9bb2f7d5b361a6caa1b4e7121a3fad5854a67818e | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| feebd585f3a350ed7ad5ebfe03636ad800883fcfefc723a7608cb43dab14e44c | CLOSE, HIGH, LOW, OPEN |

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
| px.close_loc_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.519e-173 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_4 | LAGGED_PARENT | 0 | o-o | 1.47e-145 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| px.close_loc_2 | LAGGED_PARENT | 0 | o-o | 3.981e-145 | LAGGED_PARENT | LAG0_TIME_ORDERED_LINK |
| px.close_loc_6 | LAG0_TIME_ORDERED_LINK | 0 | --> | 2.628e-114 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RSI | LAGGED_PARENT | 0 | --> | 1.668e-84 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_12 | LAG0_TIME_ORDERED_LINK | 0 | --> | 7.295e-83 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.181e-79 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_4 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.074e-77 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| WilliamsR | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.456e-75 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| BB_position | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.897e-65 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| RAW_BAR_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.383e-54 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.bar_body | LAG0_TIME_ORDERED_LINK | 0 | o-o | 2.383e-54 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| RAW_CLOSE_LOG_RETURN | LAG0_TIME_ORDERED_LINK | 0 | o-o | 6.429e-54 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_1 | LAG0_TIME_ORDERED_LINK | 0 | o-o | 6.429e-54 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| Stochastic_K | LAG0_TIME_ORDERED_LINK | 0 | --> | 4.767e-51 | CONFLICTING_OR_REVERSED | LAG0_TIME_ORDERED_LINK |
| px.logret_6 | LAGGED_PARENT | 0 | x-x | 1.021e-49 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.close_loc_48 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.728e-47 | LAG0_TIME_ORDERED_LINK | CONFLICTING_OR_REVERSED |
| px.logret_2 | LAGGED_PARENT | 0 | --> | 7.717e-46 | LAG0_TIME_ORDERED_LINK | LAGGED_PARENT |
| MACD_Histogram | CONFLICTING_OR_REVERSED | 0 | <-- | 2.002e-35 | CONFLICTING_OR_REVERSED | CONFLICTING_OR_REVERSED |
| px.logret_24 | LAG0_TIME_ORDERED_LINK | 0 | --> | 2.162e-29 |  |  |
| ROC | CONFLICTING_OR_REVERSED | 0 | <-- | 3.136e-23 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_12 | LAG0_TIME_ORDERED_LINK | 0 | x-x | 1.018e-12 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| px.logret_48 | LAG0_TIME_ORDERED_LINK | 0 | --> | 1.463e-12 | LAG0_TIME_ORDERED_LINK | LAG0_TIME_ORDERED_LINK |
| CCI | CONFLICTING_OR_REVERSED | 0 | <-- | 2.595e-06 | NO_LINK | CONFLICTING_OR_REVERSED |
| cal.utc_session_us | LAGGED_PARENT | 0 | o-o | 8.156e-05 | LAGGED_PARENT | NO_LINK |
| DI_plus | LAG0_TIME_ORDERED_LINK | 0 | o-o | 0.0009936 | LAG0_TIME_ORDERED_LINK | NO_LINK |
| px.rv_1 | LAG0_TIME_ORDERED_LINK | 0 | --> | 0.003305 | CONFLICTING_OR_REVERSED | NO_LINK |
| px.er_2 | CONFLICTING_OR_REVERSED | 0 | <-- | 0.005335 | LAG0_TIME_ORDERED_LINK | NO_LINK |
| price_minus_ema | CONFLICTING_OR_REVERSED | 0 | <-- | 0.02752 | NO_LINK | CONFLICTING_OR_REVERSED |
| DI_minus | NO_LINK |  |  |  | NO_LINK | CONFLICTING_OR_REVERSED |
| cal.utc_hod_sin | NO_LINK |  |  |  | CONFLICTING_OR_REVERSED | NO_LINK |
| hod_sin | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| hurst_spectrum_width_w200 | NO_LINK |  |  |  | NO_LINK | NO_LINK |
| px.rv_24 | NO_LINK |  |  |  |  |  |
| variogram_hurst_w48 | NO_LINK |  |  |  | NO_LINK | NO_LINK |

HAC lag screen (association comparator), timing_peak_k2: parents at lags 1..2 = 28: BB_position, CCI, DI_minus, DI_plus, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_K, WilliamsR, cal.utc_session_us, price_minus_ema, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_2, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.rv_1

HAC lag screen (association comparator), timing_valley_k2: parents at lags 1..2 = 29: BB_position, CCI, DI_minus, DI_plus, MACD_Histogram, RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN, ROC, RSI, Stochastic_K, WilliamsR, cal.utc_hod_sin, cal.utc_session_us, hod_sin, price_minus_ema, px.bar_body, px.close_loc_1, px.close_loc_12, px.close_loc_24, px.close_loc_4, px.close_loc_48, px.close_loc_6, px.logret_1, px.logret_12, px.logret_2, px.logret_4, px.logret_48, px.logret_6, px.rv_1

### Matched-control contrast (association comparator; outcome = pivot within k)

| feature | events | excess | ci lo | ci hi | significant |
|---|---|---|---|---|---|
| px.close_loc_2 | 2020 | 0.06709 | 0.04382 | 0.09018 | True |
| hurst_spectrum_width_w200 | 79 | -0.06582 | -0.1747 | 0.06842 | False |
| variogram_hurst_w48 | 162 | 0.03992 | -0.03501 | 0.1135 | False |
| px.logret_48 | 318 | 0.03936 | -0.01175 | 0.09738 | False |
| RAW_CLOSE_LOG_RETURN | 2169 | 0.03264 | 0.01105 | 0.05504 | True |
| px.logret_1 | 2169 | 0.03264 | 0.01105 | 0.05504 | True |
| RAW_BAR_LOG_RETURN | 2178 | 0.03216 | 0.01094 | 0.05433 | True |
| px.bar_body | 2178 | 0.03216 | 0.01094 | 0.05433 | True |
| DI_plus | 532 | -0.03033 | -0.07412 | 0.01505 | False |
| px.close_loc_1 | 2398 | 0.03004 | 0.009451 | 0.0516 | True |
| px.logret_6 | 899 | -0.0279 | -0.06234 | 0.01071 | False |
| MACD_Histogram | 405 | 0.0279 | -0.02803 | 0.07781 | False |
| px.rv_1 | 1845 | 0.02654 | 0.000584 | 0.05461 | True |
| px.logret_12 | 657 | -0.02397 | -0.06416 | 0.01961 | False |
| CCI | 648 | -0.02395 | -0.06797 | 0.01833 | False |
| px.close_loc_6 | 1386 | 0.02292 | -0.003585 | 0.0536 | False |
| RSI | 566 | -0.02256 | -0.06709 | 0.02536 | False |
| px.close_loc_48 | 500 | -0.01863 | -0.06541 | 0.02697 | False |
| px.close_loc_4 | 1613 | 0.01708 | -0.0071 | 0.04635 | False |
| px.close_loc_12 | 1047 | 0.01608 | -0.01255 | 0.04894 | False |
| price_minus_ema | 465 | 0.0152 | -0.0306 | 0.06388 | False |
| ROC | 711 | -0.01346 | -0.05204 | 0.02419 | False |
| px.logret_4 | 1100 | -0.01283 | -0.04219 | 0.01781 | False |
| BB_position | 783 | 0.0126 | -0.02395 | 0.0543 | False |
| px.close_loc_24 | 698 | -0.01005 | -0.04994 | 0.03102 | False |
| px.er_2 | 2169 | 0.00982 | -0.01318 | 0.03386 | False |
| DI_minus | 490 | -0.007381 | -0.05507 | 0.03432 | False |
| WilliamsR | 972 | 0.006327 | -0.02525 | 0.03956 | False |
| Stochastic_K | 547 | -0.005058 | -0.04508 | 0.03993 | False |
| px.logret_2 | 1533 | -0.001903 | -0.02824 | 0.02493 | False |

## Predictive validation (proxy; identical rows)

| set | arm | cell | tau | period | role | PR-AUC valley | PR-AUC peak | base valley | log-loss | prec long | prec short | Sharpe | net USD | MDD % | entries | win % | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3467 | 0.341 | 0.204 | 1.011 | 0.4258 | 0.4089 | -1.81 | -4550 | 54.68 | 137 | 0.4375 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3585 | 0.3606 | 0.2151 | 1.005 | 0.424 | 0.4721 | -0.6154 | -1838 | 46.17 | 126.3 | 0.5209 | 1 |
| admitted_core_15_553fc047 | m1_dmo_class | dropout0_weight_decay0.001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3661 | 0.3414 | 0.2166 | 1.01 | 0.4598 | 0.4256 | 0.0212 | -690 | 49.06 | 237.7 | 0.4865 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.0001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3579 | 0.3253 | 0.204 | 1.017 | 0.477 | 0.3666 | -1.411 | -3271 | 54.06 | 103.3 | 0.4755 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.0001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3692 | 0.3677 | 0.2151 | 1.007 | 0.4641 | 0.4687 | 0.6188 | 1296 | 31.82 | 101.7 | 0.5378 | 1 |
| admitted_core_15_553fc047 | m2_gru | dropout0.2_weight_decay0.0001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.362 | 0.3354 | 0.2166 | 1.019 | 0.4662 | 0.4424 | -0.5456 | -2420 | 51.14 | 169.7 | 0.4868 | 0.6667 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3654 | 0.3535 | 0.204 | 0.9877 | 0.4725 | 0.4135 | -0.5528 | -2858 | 56.53 | 195 | 0.4308 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3806 | 0.3693 | 0.2151 | 0.9737 | 0.4405 | 0.4271 | -3.244 | -5758 | 59.48 | 173 | 0.4162 | 1 |
| admitted_core_15_553fc047 | m3_hgb | l2_regularization10 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3829 | 0.3336 | 0.2166 | 0.9856 | 0.4844 | 0.4 | -0.1139 | -1570 | 55.82 | 277 | 0.491 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3558 | 0.355 | 0.204 | 1.005 | 0.4403 | 0.4673 | -1.091 | -2545 | 45.19 | 175.3 | 0.4807 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3532 | 0.3935 | 0.2151 | 0.9989 | 0.3882 | 0.4856 | -0.6199 | -1981 | 43.06 | 168.3 | 0.4868 | 1 |
| timing_selected | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3654 | 0.3644 | 0.2166 | 1.003 | 0.4305 | 0.4608 | -0.1814 | -1379 | 45.65 | 307.3 | 0.4952 | 1 |
| timing_selected | m2_gru | dropout0.1_weight_decay0.0001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3894 | 0.3517 | 0.204 | 0.9939 | 0.5115 | 0.3802 | -0.3374 | -2035 | 47.88 | 204.3 | 0.4829 | 1 |
| timing_selected | m2_gru | dropout0.1_weight_decay0.0001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3866 | 0.3983 | 0.2151 | 0.9954 | 0.4522 | 0.5142 | -0.5175 | -1661 | 38.43 | 195.3 | 0.4778 | 1 |
| timing_selected | m2_gru | dropout0.1_weight_decay0.0001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3808 | 0.3588 | 0.2166 | 1.006 | 0.4568 | 0.4505 | -1.482 | -5602 | 63.59 | 358.3 | 0.4575 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.376 | 0.3459 | 0.204 | 0.99 | 0.453 | 0.3628 | -1.136 | -3957 | 45.87 | 225 | 0.5244 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.4005 | 0.3931 | 0.2151 | 0.9751 | 0.544 | 0.4513 | -1.563 | -4293 | 60.85 | 238 | 0.4706 | 1 |
| timing_selected | m3_hgb | l2_regularization10 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3979 | 0.377 | 0.2166 | 0.9817 | 0.5085 | 0.5024 | 0.738 | 2818 | 57.58 | 435 | 0.492 | 1 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3788 | 0.3444 | 0.204 | 1.012 | 0.489 | 0.4198 | -1.104 | -3580 | 53.84 | 154 | 0.4872 | 1 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3736 | 0.3695 | 0.2151 | 1.007 | 0.4218 | 0.4401 | -0.2547 | -903.4 | 45.74 | 157 | 0.4777 | 1 |
| v3_six_on_admitted_clock | m1_dmo_class | dropout0_weight_decay0.0001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3767 | 0.3686 | 0.2166 | 1.009 | 0.4703 | 0.4748 | -1.478 | -5291 | 59.66 | 276.7 | 0.4725 | 1 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3897 | 0.36 | 0.204 | 1.003 | 0.5078 | 0.4637 | -1.379 | -4234 | 54.32 | 184.7 | 0.487 | 1 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.3957 | 0.3791 | 0.2151 | 1.001 | 0.4837 | 0.472 | 0.5038 | 644.3 | 33.62 | 174.7 | 0.496 | 1 |
| v3_six_on_admitted_clock | m2_gru | dropout0_weight_decay0.001 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3757 | 0.3592 | 0.2166 | 1.007 | 0.4721 | 0.3939 | -0.8041 | -3355 | 56.95 | 304.3 | 0.4949 | 1 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.6 | INNER_A_2023H1 | CHOICE_BLOCK | 0.3623 | 0.3504 | 0.204 | 0.9797 | 0.4234 | 0.42 | -1.463 | -4706 | 61.28 | 276 | 0.5036 | 1 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.6 | INNER_B_2023H2 | FROZEN_SCORE | 0.4101 | 0.3916 | 0.2151 | 0.9708 | 0.4748 | 0.5194 | -1.126 | -3539 | 55.56 | 262 | 0.4962 | 1 |
| v3_six_on_admitted_clock | m3_hgb | l2_regularization10 | 0.6 | REOBSERVED_2024 | PREVIOUSLY_OBSERVED_RECONFIRMATION | 0.3923 | 0.3826 | 0.2166 | 0.9813 | 0.4778 | 0.4476 | -0.08226 | -1383 | 59.55 | 505 | 0.4792 | 1 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_A_2023H1 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_B_2023H2 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | REOBSERVED_2024 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
