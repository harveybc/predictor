# TIMING_PROXY_SELECTION tables (EURUSD_H1_PROFIT_ADMITTED, identity timing-selection:e2c14dffff74f2a9)

Status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION`; manifest sha256 `84196d4e80a408d6705fa8f0bd5e5954017ee072e7ddb0333553f20d42c1d3b2`; TRAIN rows 18698; features 93 (30 inventory + 51 price_pool + 12 fractal = 93).

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
| MRMR K12 union over timing_profit_h72, timing_profit_long_h72, timing_profit_short_h72 | 16: CLOSE, RAW_ELAPSED_HOURS, Stochastic_D, dow_cos, hod_cos, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.hours_since_session_open, px.unexpected_gap_hours, variogram_hurst_w200 |
| dropped by the T1 FDR screen | CLOSE, RAW_ELAPSED_HOURS, Stochastic_D, hod_cos, hurst_rs_w48, hurst_rs_w96, hurst_spectrum_width_w48, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.er_48, px.unexpected_gap_hours, variogram_hurst_w200 |
| pruned as redundant | dow_cos |
| SELECTED | 1: px.hours_since_session_open |

Comparator subsets (reported, not used): timing_best_order_pnl_sigma_h72: CLOSE, RAW_ELAPSED_HOURS, cal.utc_dow_cos, hurst_rs_w48, hurst_spectrum_width_w96, px.close_loc_1, px.er_1, px.er_24, px.er_4, px.hours_since_session_open, px.unexpected_gap_hours, variogram_hurst_w200; timing_long_order_wins_h72: CLOSE, RAW_ELAPSED_HOURS, Stochastic_D, dow_cos, hod_cos, hurst_rs_w48, px.close_loc_1, px.er_1, px.er_4, px.er_48, px.unexpected_gap_hours, variogram_hurst_w200

## Univariate screen vs T1 (Kruskal-Wallis, BH q 0.1)

| feature | H | p | pass_fdr | n |
|---|---|---|---|---|
| dow_cos | 305.4 | 4.815e-67 | True | 14823 |
| cal.utc_dow_cos | 301.4 | 3.516e-66 | True | 14823 |
| dow_sin | 98.36 | 4.389e-22 | True | 14823 |
| cal.utc_dow_sin | 78.43 | 9.295e-18 | True | 14823 |
| hurst_rs_w200 | 64.72 | 8.837e-15 | True | 14790 |
| ATR | 52.75 | 3.509e-12 | True | 14823 |
| ATR_ratio | 52.75 | 3.509e-12 | True | 14823 |
| hurst_spectrum_width_w200 | 52.26 | 4.491e-12 | True | 14790 |
| hurst_spectrum_width_w48 | 48.99 | 2.3e-11 | True | 14823 |
| RAW_LOG_RANGE | 44.69 | 1.972e-10 | True | 14823 |
| ADX | 40.24 | 1.826e-09 | True | 14823 |
| cal.utc_hod_cos | 30.51 | 2.374e-07 | True | 14823 |
| cal.london_hod_cos | 27.94 | 8.567e-07 | True | 14823 |
| hod_cos | 23.99 | 6.173e-06 | True | 14823 |
| cal.utc_session_asia | 22.69 | 1.181e-05 | True | 14823 |
| hurst_spectrum_width_w96 | 20.86 | 2.954e-05 | True | 14823 |
| hod_sin | 17.55 | 0.0001548 | True | 14823 |
| BB_Width | 15.02 | 0.000548 | True | 14823 |
| cal.london_hod_sin | 14.69 | 0.0006453 | True | 14823 |
| MACD | 12.19 | 0.002259 | True | 14823 |
| MACD_Signal | 11.77 | 0.002784 | True | 14823 |
| cal.utc_session_europe | 11.73 | 0.002844 | True | 14823 |
| box_counting_dimension_w48 | 11.5 | 0.003189 | True | 14823 |
| RSI | 10.96 | 0.004174 | True | 14823 |
| DI_minus | 10.91 | 0.00427 | True | 14823 |
| hurst_rs_w96 | 8.825 | 0.01212 | True | 14823 |
| DI_plus | 8.802 | 0.01227 | True | 14823 |
| cal.utc_hod_sin | 8.041 | 0.01794 | True | 14823 |
| price_minus_ema | 7.854 | 0.0197 | True | 14823 |
| cal.utc_session_us | 6.252 | 0.0439 | True | 14823 |
| ROC | 5.685 | 0.05829 | True | 14823 |
| hurst_rs_w48 | 5.398 | 0.06727 | True | 14823 |
| MACD_Histogram | 4.386 | 0.1116 | False | 14823 |
| box_counting_dimension_w200 | 4.151 | 0.1255 | False | 14790 |
| LOW | 3.316 | 0.1905 | False | 14823 |
| OPEN | 3.211 | 0.2008 | False | 14823 |
| CLOSE | 3.178 | 0.2041 | False | 14823 |
| HIGH | 3.027 | 0.2201 | False | 14823 |
| Stochastic_D | 2.549 | 0.2796 | False | 14823 |
| Stochastic_K | 1.863 | 0.394 | False | 14823 |
| box_counting_dimension_w96 | 1.745 | 0.4178 | False | 14823 |
| CCI | 1.116 | 0.5723 | False | 14823 |
| WilliamsR | 1.074 | 0.5844 | False | 14823 |
| BB_position | 0.9 | 0.6376 | False | 14823 |
| RAW_CLOSE_LOG_RETURN | 0.1834 | 0.9124 | False | 14823 |
| RAW_BAR_LOG_RETURN | 0.1573 | 0.9244 | False | 14823 |
| RAW_ELAPSED_HOURS | nan | nan | False | 14823 |
| px.hours_since_session_open | 323.5 | 5.767e-71 | True | 14823 |
| px.range_6 | 70.75 | 4.324e-16 | True | 14823 |
| px.range_4 | 64.97 | 7.796e-15 | True | 14823 |
| px.range_12 | 61.88 | 3.649e-14 | True | 14823 |
| px.range_2 | 60.14 | 8.716e-14 | True | 14823 |
| px.rv_6 | 56.17 | 6.361e-13 | True | 14823 |
| px.rv_4 | 54.56 | 1.421e-12 | True | 14823 |
| px.er_24 | 48.8 | 2.532e-11 | True | 14823 |
| px.range_1 | 44.69 | 1.972e-10 | True | 14823 |
| px.atr_14 | 43.02 | 4.543e-10 | True | 14823 |
| px.rv_2 | 40.97 | 1.271e-09 | True | 14823 |
| px.rv_12 | 37.85 | 6.041e-09 | True | 14823 |
| px.range_24 | 31.14 | 1.731e-07 | True | 14823 |
| px.rv_1 | 23.8 | 6.803e-06 | True | 14823 |
| px.rv_24 | 23.43 | 8.151e-06 | True | 14823 |
| px.rv_48 | 21.9 | 1.759e-05 | True | 14823 |
| variogram_hurst_w48 | 21.28 | 2.39e-05 | True | 14823 |
| px.er_48 | 19.17 | 6.875e-05 | True | 14823 |
| px.er_12 | 16.28 | 0.0002916 | True | 14823 |
| px.range_48 | 14.46 | 0.0007244 | True | 14823 |
| px.range_ratio_1_24 | 12.25 | 0.002184 | True | 14823 |
| rolling_std_24 | 12.19 | 0.002249 | True | 14823 |
| px.logret_48 | 10.51 | 0.00521 | True | 14823 |
| px.logret_24 | 9.074 | 0.0107 | True | 14823 |
| px.close_loc_48 | 7.502 | 0.02349 | True | 14823 |
| px.logret_6 | 7.321 | 0.02572 | True | 14823 |
| px.close_loc_24 | 6.179 | 0.04552 | True | 14823 |
| variogram_hurst_w200 | 5.925 | 0.05169 | True | 14790 |
| px.er_6 | 5.395 | 0.06736 | True | 14823 |
| px.atr_ratio_14_120 | 4.172 | 0.1242 | False | 14823 |
| px.logret_4 | 4.015 | 0.1343 | False | 14823 |
| px.logret_12 | 3.48 | 0.1755 | False | 14823 |
| variogram_hurst_w96 | 1.898 | 0.3871 | False | 14823 |
| px.close_loc_12 | 1.55 | 0.4607 | False | 14823 |
| px.er_4 | 0.9941 | 0.6083 | False | 14823 |
| px.close_loc_1 | 0.9524 | 0.6211 | False | 14823 |
| px.close_loc_6 | 0.7999 | 0.6704 | False | 14823 |
| px.logret_2 | 0.4712 | 0.7901 | False | 14823 |
| px.close_loc_2 | 0.2649 | 0.876 | False | 14823 |
| px.logret_1 | 0.1834 | 0.9124 | False | 14823 |
| px.close_loc_4 | 0.178 | 0.9148 | False | 14823 |
| px.bar_body | 0.1573 | 0.9244 | False | 14823 |
| px.elapsed_hours | nan | nan | False | 14823 |
| px.er_1 | nan | nan | False | 14721 |
| px.er_2 | 0.04442 | 0.978 | False | 14821 |
| px.unexpected_gap_hours | nan | nan | False | 14823 |

## Stability across prefix folds

| fold | rank Spearman vs TRAIN | passed |
|---|---|---|
| prefix_before_2021-01-01 | 0.3305 | 78 |
| prefix_before_2022-01-01 | 0.8694 | 58 |

Selected: FDR pass count over (folds + TRAIN) = px.hours_since_session_open: 3

## Within-day variance

Constant within a day: cal.utc_dow_cos, cal.utc_dow_sin.

| selected feature | within-day variance ratio |
|---|---|
| px.hours_since_session_open | 0.03203 |

## Redundancy

| selected feature | VIF |
|---|---|
| px.hours_since_session_open | nan |

TRAIN Spearman clusters (|rho| >= 0.9, average linkage): 66 clusters, 16 with > 1 member.

| cluster | members |
|---|---|
| 052b6874307139a770c1d156e626d9011e1b6ae27c147f4eb1f703f4cee9d55c | MACD, MACD_Signal |
| 1ebe158171e7701021c7650f99774bacc52644f8ab8c20b695274244c5d2303e | cal.london_hod_sin, cal.utc_hod_sin, hod_sin |
| 21cdbe047e354745db942cb4fc48740359452b0852a87c8bb6c93e71ab14acf4 | BB_Width, rolling_std_24 |
| 2fbab747839160563ef6d4d574282108b1422c644d8942e1d34d4d632d200b7b | cal.london_hod_cos, cal.utc_hod_cos, hod_cos |
| 40676a60f00b7aeaaf86c05449c8dbe2378a6b378050879e6aec103262ac7bf6 | px.rv_24, px.rv_48 |
| 4092419e29db3ae46da2ff7ae7771e3e78c1e941fad0642af610db2fca202f14 | px.close_loc_4, px.close_loc_6 |
| 53fdf3c9ec1f03e7097632204c409f18e81f2f95087741d5e34ad53d3d48765c | RAW_BAR_LOG_RETURN, RAW_CLOSE_LOG_RETURN |
| 593493a05c1cdaa590be62c7ce08656f2476fc6a7dc8cd5cb916aff5c9f5b3e5 | ATR, ATR_ratio, px.atr_14 |
| 8d2b2c935ace7286bed8a4bcf1d1028c755441b004bf11f414476294d1b8788e | CLOSE, HIGH, LOW, OPEN |
| 91f935c5c55130d61f6037813b10fa4ef718bd15be3c17c47d732ad3a48b4dd7 | cal.utc_dow_cos, dow_cos, px.hours_since_session_open |
| 9be0784d7cf09828b1223bca2cfcedbd77ec378a74318af55861de8d192f066b | BB_position, CCI |
| 9c9bfe7270889f5331bb4f239f289c1fbad31a74012b3ca1543465d7b40a0589 | RSI, price_minus_ema, px.close_loc_24 |
| b966b77290a0e3aac25e3b7683dbbd2f1bfc56a4eba8563510ccbdef42ae2433 | px.range_4, px.range_6 |
| e4d21c81cb83c3ecca4bef63c052c1d9f06a9a923229bf363facd0fb02e3d98f | WilliamsR, px.close_loc_12 |
| f377485e32a96e114dfd3e47a8e55dcfd872079683dc3a83bcf7b8243aac0ea4 | cal.utc_dow_sin, dow_sin |
| f64d415ad831cb8b2c47ba5f493de7868be9fb1add48fdc8f9fe428d06067eb7 | Stochastic_D, Stochastic_K |

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
| NO_TRADE_COMPARATOR | - | - | nan | INNER_A_2023H1 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | INNER_B_2023H2 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
| NO_TRADE_COMPARATOR | - | - | nan | REOBSERVED_2024 | COMPARATOR | nan | nan | nan | nan | nan | nan | 0 | 0 | 0 | 0 | nan | 0 |
