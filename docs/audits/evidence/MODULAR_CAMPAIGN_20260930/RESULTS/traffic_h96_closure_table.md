# Closure table: traffic L96 -> h96 (generated, not typed)

- Scale: z_train (the training-scaler normalized target space; --inverse False); mean over all scored window x forecast-step x target-channel elements, not an unweighted average of batch means; float32, the author's own reduction; an independent float64 reduction is carried beside it; 282,432,576 elements (3,413 windows x 96 steps x 862 channels).
- Naive: persistence: the window's last observed value repeated over the horizon, every channel, same test windows, normalized space
- Pairing proof: population sha 717cb5b34a6fde26… equals the naive target sha (equal=True).
- Literature: Hu, Y., Zhang, G., Liu, P., Lan, D., Li, N., Cheng, D., Dai, T., Xia, S.-T., Pan, S. TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting. ICML 2025; arXiv:2501.13041. Author code github.com/TROUBADOUR000/TimeFilter, Table 8 (L = 96, full results) (read arXiv HTML v2, Tables 6, 7, 8, 9; read 2026-09-28); margin from Table 7 (Traffic 0.407+-0.008 / 0.268+-0.004).
- Comparability class: MATCHED_PUBLISHED_RECIPE_EXECUTED.

| seed | MSE | MAE | naive MSE | naive MAE | skill MSE | skill MAE | published MSE / MAE | diff MSE / MAE | record sha |
|---|---|---|---|---|---|---|---|---|---|
| 2021 | 0.3756999969 | 0.2513667941 | 2.7144524181 | 1.0772232192 | 0.8616 | 0.7667 | 0.375 / 0.251 | +0.00070 / +0.00037 | bdcf6e523e17… |
| 2022 | 0.3753611147 | 0.2512390912 | 2.7144524181 | 1.0772232192 | 0.8617 | 0.7668 | 0.375 / 0.251 | +0.00036 / +0.00024 | b1a1865dd352… |
| 2023 | 0.3745366931 | 0.2508221567 | 2.7144524181 | 1.0772232192 | 0.8620 | 0.7672 | 0.375 / 0.251 | -0.00046 / -0.00018 | 0fff7a744496… |
| **mean** | 0.3751992683 (sd 0.00060) | 0.2511426806 (sd 0.00028) | 2.7144524181 | 1.0772232192 | 0.8618 | 0.7669 | 0.375 / 0.251 | +0.00020 / +0.00014 | — |

Class on the seed mean: MSE **OPERATIONAL_AGREEMENT** (tolerance 0.0165), MAE **OPERATIONAL_AGREEMENT** (tolerance 0.0085). Rule: per horizon and for the four-horizon average (formed WITHIN each seed first), the seed mean of the replicated metric is in OPERATIONAL_AGREEMENT with the published value when |mean - published| <= 2 x std_paper + 0.0005; OPERATIONAL_PARTIAL when <= 3 x std_paper + 0.0005; OUTSIDE_OPERATIONAL_MARGIN otherwise. std_paper is THIS dataset's Table 7 standard deviation of the four-horizon average, borrowed per horizon as a predeclared operational margin - it is not a published per-horizon error bar and not statistical equivalence; our own seed dispersion is reported beside it and never replaces the criterion
