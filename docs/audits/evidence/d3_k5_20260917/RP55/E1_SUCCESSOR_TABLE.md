# E1 household DEV pilot — results (run satoshi-e1-successor-20260920)

Task W60_h60: context 3540 s, horizon 3600 s, model reach 3600 s. DEV rows [1412361, 1462761]. Common evaluation set 10020 origins. MASE denominator (persistence h, train) 0.6163 kW.

| unit | regime | seed | MASE | MAE (kW) | updates | epochs | stop | best epoch | censoring | fit s | child CPU s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_s1 | R0 | 1 | 0.8965 | 0.5525 | 3762 | 6 | EARLY_STOPPING | 3 | STOPPED_ON_VALIDATION | 391.16 | 406.681 | 915 |
| R1_s1 | R1 | 1 | 0.9087 | 0.5600 | 3762 | 6 | EARLY_STOPPING | 3 | STOPPED_ON_VALIDATION | 144.71 | 158.559 | 889 |
| R2_s1 | R2 | 1 | 0.8799 | 0.5423 | 3135 | 5 | EARLY_STOPPING | 2 | STOPPED_ON_VALIDATION | 257.303 | 270.118 | 915 |
| R0_s2 | R0 | 2 | 0.8929 | 0.5503 | 4000 | 7 | UPDATE_BUDGET | 5 | CENSORED_BY_BUDGET | 351.537 | 366.897 | 919 |
| R1_s2 | R1 | 2 | 0.8839 | 0.5447 | 4000 | 7 | UPDATE_BUDGET | 5 | CENSORED_BY_BUDGET | 160.48 | 171.35 | 892 |
| R2_s2 | R2 | 2 | 0.9237 | 0.5692 | 4000 | 7 | UPDATE_BUDGET | 6 | CENSORED_BY_BUDGET | 326.868 | 336.189 | 914 |
| R0_s3 | R0 | 3 | 0.8730 | 0.5380 | 2508 | 4 | EARLY_STOPPING | 1 | STOPPED_ON_VALIDATION | 209.488 | 218.847 | 913 |
| R1_s3 | R1 | 3 | 0.9176 | 0.5655 | 4000 | 7 | UPDATE_BUDGET | 4 | CENSORED_BY_BUDGET | 165.31 | 176.091 | 891 |
| R2_s3 | R2 | 3 | 0.8868 | 0.5465 | 2508 | 4 | EARLY_STOPPING | 1 | STOPPED_ON_VALIDATION | 210.373 | 223.092 | 911 |

| control | MASE | MAE (kW) |
|---|---|---|
| persistence | 1.0018 | 0.6174 |
| seasonal_naive_daily | 1.1873 | 0.7317 |
| linear_ridge | 0.8852 | 0.5455 |
| linear_reach | 0.8852 | 0.5455 |

| regime | n | MASE mean | MASE sd |
|---|---|---|---|
| R0 | 3 | 0.8875 | 0.0127 |
| R1 | 3 | 0.9034 | 0.0175 |
| R2 | 3 | 0.8968 | 0.0236 |

| paired difference (MASE) | n | mean | min | max | per seed |
|---|---|---|---|---|---|
| R1_minus_R0 | 3 | 0.0159 | -0.009 | 0.0446 | {"1": 0.0121, "2": -0.009, "3": 0.0446} |
| R2_minus_R0 | 3 | 0.0093 | -0.0166 | 0.0308 | {"1": -0.0166, "2": 0.0308, "3": 0.0137} |
| R2_minus_R1 | 3 | -0.0066 | -0.0309 | 0.0398 | {"1": -0.0287, "2": 0.0398, "3": -0.0309} |

| seed | AE updates | AE stop | masked val MSE | AE child CPU s |
|---|---|---|---|---|
| 1 | 1500 | UPDATE_BUDGET | 0.1969 | 46.852 |
| 2 | 1500 | UPDATE_BUDGET | 0.1932 | 43.236 |
| 3 | 1500 | UPDATE_BUDGET | 0.1919 | 47.068 |

Identity proofs per seed: {"1": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}, "2": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}, "3": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}}

Costs: {"ae_cpu": 137.156, "fit_cpu_by_regime": {"R0": 992.425, "R1": 506.0, "R2": 829.399}, "pilot_cpu": 65.673, "controls_cpu": 4.851, "spent_cpu_seconds_root": 2535.5039999999995, "reading": "same task budget per regime (fit CPU); total cost = fit + the AE of the seed for R1/R2 (reported apart)"}

Governance: LOCAL_TERMINALS_PUBLIC_PANEL_NOT_A_LAKE_RESOURCE — one lake registration of public_panels_c126_v2 in data-gov (config + service restart: an owner action)
