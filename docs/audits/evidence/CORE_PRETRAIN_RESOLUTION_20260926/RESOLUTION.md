# The measured resolution of the E1 household protocol

Generated 2026-09-26T18:07:18Z from retained artifacts. **No training was performed.**

## The resolution

On the household minute-level task (window 60, horizon 60), with this protocol, **3 seeds per arm** and **10020 evaluation rows**, a difference in mean MAE smaller than **0.031106 kW** (95% CI 0.020045 - 0.068498 kW; 0.050476 in persistence-scaled units) is indistinguishable from the protocol's own seed-to-seed variation at alpha=0.05, power=0.8.

sigma = 0.01136346 kW on 6 df (95% CI 0.00732254 - 0.02502309), pooled within-arm over R0, R1, R2. Estimator: (t_{1-alpha/2,df_test} + t_{power,df_test}) * sqrt(2/n) * sigma. Across the three defensible estimators the resolution spans 0.031106 - 0.049762 kW; the smallest is quoted, so the ruling cannot be blamed on a pessimistic test.

## Every retained contrast against that resolution

| contrast | effect (kW) | 95% CI | p | resolution (kW) | state | seeds for this effect |
|---|---:|---|---:|---:|---|---:|
| R1-R0 | +0.009805 | [-0.031540, +0.051150] | 0.4149 | 0.031106 | BELOW_THE_RESOLUTION | 23 |
| R2-R0 | +0.005731 | [-0.031007, +0.042470] | 0.5712 | 0.031106 | BELOW_THE_RESOLUTION | 63 |
| R2-R1 | -0.004074 | [-0.065611, +0.057464] | 0.8026 | 0.031106 | BELOW_THE_RESOLUTION | 124 |
| tcn_mse-core_mse | -0.010982 | [-0.038616, +0.016652] | 0.2294 | 0.031106 | BELOW_THE_RESOLUTION | 18 |
| core_mae-core_mse | -0.049160 | [-0.068156, -0.030164] | 0.0080 | 0.031106 | AT_THE_RESOLUTION_BOUNDARY | 3 |

## The dynamic range the effects live inside

- untrained, before any update: **1.597428 kW**
- fitted on 40080 **scrambled** train labels: **0.601542 kW**
- persistence naive on the same rows: **0.617372 kW**
- the best retained arm (core_mae): **0.497770 kW**
- the whole label-information span: **0.103772 kW**

## Ruling

- effect the module must resolve: **R1-R0 = 0.009805 kW**
- resolution at n=3: **0.031106 kW**
- the resolution is **3.17x** the effect
- seeds required per arm for that effect: **23** (69 fits for the three-arm design)
- seeds required per arm for a flat 0.01 kW effect: **22**
- **UNANSWERABLE_BY_THIS_PROTOCOL_AT_THIS_SEED_COUNT**

problems: none
