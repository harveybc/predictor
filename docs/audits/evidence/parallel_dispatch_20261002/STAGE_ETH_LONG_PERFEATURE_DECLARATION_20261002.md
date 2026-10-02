# Declaration: ETH 4h long-horizon per_feature huber/adam, seeds 2021 and 2022 (2026-10-02)

Declared before running. Arm: `per_feature_huber_adam`, seeds 2021 (root on the RTX 5090) and 2022 (root on the RTX 5070 Ti), worker_a, predictor pin bbb6fac6, tools/eth_forecast_campaign.py, data eth4h_long_l24_h6to36_v1 (train rows [0,13699) minus purge, validation 2160 rows; no test rows, 2025 rows [15895,18085) untouched).

Reason: grouped32 huber/adam (2.3708) has essentially zero skill over the intercept-only naive (2.37115); no further grouped32 seeds are spent.

Cost pilot (one cell, 5090, crispdm-run, probe cap 4000M, admitted): cgroup peak 2380156928 B, process VmHWM (peak RSS) 2872811520 B, 173926 parameters. Campaign cap amended to 1.25 x peak RSS = 3591014400 B = 3425M (derived; no cap lowered, existing 2470M raised for both roots).

Success reading: objective = mean validation MAE (z_train) over horizons 6..36, compared with the same-row intercept-only (2.37115) and train-mean (2.37263) naives from stage_sres_naive/naive_table_validation.json; skill reported with its sign, no claim from two seeds.
