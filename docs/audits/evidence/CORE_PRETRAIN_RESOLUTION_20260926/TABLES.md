### Closure table — every arm mean in this document

| run | arm | seeds | model error (kW) | model error (z) | naive, same rows (kW), n | h | skill vs naive | literature value & source | comparability | custody | verified |
|---|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| `e1_phase1_v1b` | `core_mae` | 3 | 0.497770213 | 0.545492 | 0.617372056, n=10020 | 60 | +0.193727 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_v1b` | `core_mse` | 3 | 0.546930216 | 0.599365 | 0.617372056, n=10020 | 60 | +0.114099 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_v1b` | `tcn_mse` | 3 | 0.535948087 | 0.587330 | 0.617372056, n=10020 | 60 | +0.131888 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R0` | 3 | 0.546929167 | 0.599364 | 0.617372056, n=10020 | 60 | +0.114101 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R1` | 3 | 0.556734156 | 0.610109 | 0.617372056, n=10020 | 60 | +0.098219 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R2` | 3 | 0.552660340 | 0.605644 | 0.617372056, n=10020 | 60 | +0.104818 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_matched_v1` | `core_mae` | 3 | 0.497770213 | 0.545492 | 0.617372056, n=10020 | 60 | +0.193727 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |
| `e1_phase1_matched_v1` | `core_mse` | 3 | 0.539266392 | 0.590966 | 0.617372056, n=10020 | 60 | +0.126513 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |
| `e1_phase1_matched_v1` | `tcn_mse` | 3 | 0.522917583 | 0.573050 | 0.617372056, n=10020 | 60 | +0.152994 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |

**Literature note, identical for every row above** (the registry decides it from identity fields, never from a score): source — Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060. Published values — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53. Status **NOT_COMPARABLE**, comparator state `NONE`; `placed_in_comparison_column: false`. Reason: unknown identity fields cannot match as proof: ['target_transform']. Planned matched comparison: read the primary source (or its code) and fill the field; a placeholder is not a protocol.

### The retained contrasts against the measured resolution

| contrast | effect (kW) | 95% CI (kW) | p | state against the resolution | seeds for this effect |
|---|--:|---|--:|---|--:|
| `R1-R0` | +0.009805 | [-0.031540, +0.051150] | 0.4149 | BELOW_THE_RESOLUTION | 23 |
| `R2-R0` | +0.005731 | [-0.031007, +0.042470] | 0.5712 | BELOW_THE_RESOLUTION | 63 |
| `R2-R1` | -0.004074 | [-0.065611, +0.057464] | 0.8026 | BELOW_THE_RESOLUTION | 124 |
| `tcn_mse-core_mse` | -0.010982 | [-0.038616, +0.016652] | 0.2294 | BELOW_THE_RESOLUTION | 18 |
| `core_mae-core_mse` | -0.049160 | [-0.068156, -0.030164] | 0.0080 | AT_THE_RESOLUTION_BOUNDARY | 3 |

### The matched re-contrast against the unmatched one

| contrast | unmatched effect (kW) | matched effect (kW) | change (kW) | matched 95% CI | matched p | state | erased? |
|---|--:|--:|--:|---|--:|---|---|
| `core_mae-core_mse` | -0.049160 | -0.041496 | +0.007664 | [-0.049390, -0.033602] | 0.0019 | AT_THE_RESOLUTION_BOUNDARY | **NO** |
| `tcn_mse-core_mse` | -0.010982 | -0.016349 | -0.005367 | [-0.037549, +0.004852] | 0.0801 | BELOW_THE_RESOLUTION | **NO** |
| `core_mae-tcn_mse` | — | -0.025147 | — | [-0.038471, -0.011824] | 0.0148 | BELOW_THE_RESOLUTION | n/a |

### Per-arm means and seed spread, unmatched against matched

| arm | mean unmatched (kW) | mean matched (kW) | change (kW) | sd unmatched (kW) | sd matched (kW) | updates unmatched | updates matched |
|---|--:|--:|--:|--:|--:|--:|--:|
| `core_mae` | 0.497770213 | 0.497770213 | +0.000000000 | 0.000170593 | 0.000170593 | 11762 | 12000 |
| `core_mse` | 0.546930216 | 0.539266392 | -0.007663824 | 0.007804003 | 0.003189395 | 10270 | 12000 |
| `tcn_mse` | 0.535948087 | 0.522917583 | -0.013030504 | 0.005011532 | 0.005344970 | 11762 | 12000 |

### Like for like: did the repair sharpen the instrument?

Only `core_mse`, `tcn_mse` were run under BOTH protocols, so only those two answer it.

| | sigma (kW) | df | resolution at n=3 (kW) | seeds for a flat 0.01 kW |
|---|--:|--:|--:|--:|
| unmatched (RP63) | 0.006558121 | 4 | 0.019906 | 8 |
| matched (this round) | 0.004401190 | 4 | 0.013359 | 5 |
| ratio | 0.671105 | | | |
