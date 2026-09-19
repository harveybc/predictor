# E1 household DEV pilot — results (run satoshi-e1-household-dev-20260919)

Task W60_h60: context 3540 s, horizon 3600 s, model reach 420 s. DEV rows [1412361, 1462761]. Common evaluation set 10020 origins. MASE denominator (persistence h, train) 0.6163 kW.

| unit | regime | seed | MASE | MAE (kW) | updates | epochs | stop | best epoch | truncated | fit s | child CPU s | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_s1 | R0 | 1 | 0.8831 | 0.5442 | 4000 | 7 | UPDATE_BUDGET | 6 | False | 245.792 | 258.165 | 888 |
| R1_s1 | R1 | 1 | 0.8865 | 0.5463 | 3135 | 5 | EARLY_STOPPING | 2 | False | 55.095 | 64.428 | 851 |
| R2_s1 | R2 | 1 | 0.8887 | 0.5477 | 3135 | 5 | EARLY_STOPPING | 2 | False | 192.488 | 201.081 | 876 |
| R0_s2 | R0 | 2 | 0.8981 | 0.5535 | 4000 | 7 | UPDATE_BUDGET | 6 | False | 244.99 | 253.354 | 889 |
| R1_s2 | R1 | 2 | 0.8955 | 0.5519 | 4000 | 7 | UPDATE_BUDGET | 6 | False | 75.028 | 84.687 | 851 |
| R2_s2 | R2 | 2 | 0.8912 | 0.5492 | 4000 | 7 | UPDATE_BUDGET | 6 | False | 243.946 | 252.662 | 890 |
| R0_s3 | R0 | 3 | 0.9149 | 0.5638 | 3762 | 6 | EARLY_STOPPING | 3 | False | 236.418 | 246.114 | 886 |
| R1_s3 | R1 | 3 | 0.9046 | 0.5575 | 3762 | 6 | EARLY_STOPPING | 3 | False | 68.566 | 78.29 | 858 |
| R2_s3 | R2 | 3 | 0.9206 | 0.5673 | 3762 | 6 | EARLY_STOPPING | 3 | False | 246.828 | 257.446 | 884 |

| control | MASE | MAE (kW) |
|---|---|---|
| persistence | 1.0018 | 0.6174 |
| seasonal_naive_daily | 1.1873 | 0.7317 |
| linear_ridge | 0.8852 | 0.5455 |

| regime | n | MASE mean | MASE sd |
|---|---|---|---|
| R0 | 3 | 0.8987 | 0.0159 |
| R1 | 3 | 0.8956 | 0.0091 |
| R2 | 3 | 0.9002 | 0.0177 |

| paired difference (MASE) | n | mean | min | max | per seed |
|---|---|---|---|---|---|
| R1_minus_R0 | 3 | -0.0032 | -0.0103 | 0.0033 | {"1": 0.0033, "2": -0.0026, "3": -0.0103} |
| R2_minus_R0 | 3 | 0.0014 | -0.0069 | 0.0056 | {"1": 0.0055, "2": -0.0069, "3": 0.0056} |
| R2_minus_R1 | 3 | 0.0046 | -0.0043 | 0.0159 | {"1": 0.0022, "2": -0.0043, "3": 0.0159} |

| seed | AE updates | AE stop | masked val MSE | AE child CPU s |
|---|---|---|---|---|
| 1 | 1500 | UPDATE_BUDGET | 0.1463 | 44.982 |
| 2 | 1500 | UPDATE_BUDGET | 0.1464 | 45.194 |
| 3 | 1500 | UPDATE_BUDGET | 0.1464 | 41.563 |

Identity proofs per seed: {"1": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}, "2": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}, "3": {"shared_initial_checkpoint": true, "R1_R2_same_imported_detector": true, "R0_detector_is_the_random_initial": true, "R1_detector_frozen": true, "R0_R2_detector_learns": true}}

Costs: {"ae_cpu": 131.739, "fit_cpu_by_regime": {"R0": 757.633, "R1": 227.405, "R2": 711.1890000000001}, "pilot_cpu": 64.819, "controls_cpu": 4.644, "spent_cpu_seconds_root": 1897.429, "reading": "same task budget per regime (fit CPU); total cost = fit + the AE of the seed for R1/R2 (reported apart)"}

Governance: LOCAL_TERMINALS_PUBLIC_PANEL_NOT_A_LAKE_RESOURCE — one lake registration of public_panels_c126_v2 in data-gov (config + service restart: an owner action)
