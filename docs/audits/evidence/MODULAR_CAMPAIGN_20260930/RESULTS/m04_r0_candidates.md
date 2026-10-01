# M04 batch-1 v3: verified R0 candidates (generated)

Campaign `m04_ecl_l24_h24_batch1_v3_deterministic`, CAMPAIGN sha `53bce359…`. Objective: MAE on validation, z_train (lower is better). Counts: {'blocked': 20, 'verified': 16}. Comparability: **NOT_COMPARABLE**: ECL L24 -> H1..24, all 321 channels, z_train MAE on the validation split; the published TimeFilter/ECL rows are L96 -> H96 on the test split, so no literature value applies.

| label | cfg | seed | objective | skill MAE | verify | exact | train host | peak GiB | s/update | updates | epoch | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| draw1_huber | cc4235d7 | 2021 | 0.403170 | 0.5265 | VERIFIED | True | worker_a | 3.88 | 0.0117 | 2880 | 12 | patience |
| draw1_huber | cc4235d7 | 2022 | 0.391178 | 0.5406 | VERIFIED | True | worker_a | 3.89 | 0.0112 | 3744 | 18 | patience |
| draw1_mae | 331188a4 | 2021 | 0.386604 | 0.5459 | VERIFIED | True | worker_b | 3.84 | 0.0191 | 4320 | 29 | max_epochs |
| draw1_mae | 331188a4 | 2022 | 0.421962 | 0.5044 | VERIFIED | True | worker_a | 4.31 | 0.0128 | 1584 | 3 | patience |
| draw2_huber | 8fa7d94d | 2021 | 0.408565 | 0.5201 | VERIFIED | True | worker_b | 3.81 | 0.0647 | 2592 | 15 | patience |
| draw2_huber | 8fa7d94d | 2022 | 0.393390 | 0.5380 | VERIFIED | True | worker_b | 3.84 | 0.0648 | 2448 | 14 | patience |
| draw2_mae | 4c14e945 | 2021 | 0.406981 | 0.5220 | VERIFIED | True | worker_b | 3.81 | 0.0647 | 2448 | 14 | patience |
| draw2_mae | 4c14e945 | 2022 | 0.391440 | 0.5402 | VERIFIED | True | worker_b | 3.85 | 0.0647 | 2448 | 14 | patience |
| draw3_huber | 468b6e1b | 2021 | 0.409110 | 0.5195 | VERIFIED | True | worker_a | 3.88 | 0.0109 | 1584 | 8 | patience |
| draw3_huber | 468b6e1b | 2022 | 0.439963 | 0.4833 | VERIFIED | True | worker_a | 3.88 | 0.0114 | 1008 | 4 | patience |
| draw3_mae | 4ebc89be | 2021 | 0.422334 | 0.5040 | VERIFIED | True | worker_a | 3.88 | 0.0125 | 720 | 2 | patience |
| draw3_mae | 4ebc89be | 2022 | 0.394921 | 0.5362 | VERIFIED | True | worker_a | 3.88 | 0.0104 | 1440 | 7 | patience |
| grouped32_R0_huber | fe256dab | 2021 | 0.422055 | 0.5043 | VERIFIED | True | worker_b | 3.81 | 0.0279 | 2870 | 5 | patience |
| grouped32_R0_huber | fe256dab | 2022 | 0.441446 | 0.4815 | VERIFIED | True | worker_a | 3.91 | 0.0165 | 2296 | 3 | patience |
| grouped32_R0_mae | e323775b | 2021 | 0.403343 | 0.5263 | VERIFIED | True | worker_a | 3.87 | 0.0145 | 2296 | 3 | patience |
| grouped32_R0_mae | e323775b | 2022 | 0.401216 | 0.5288 | VERIFIED | True | worker_a | 3.89 | 0.0141 | 2296 | 3 | patience |
