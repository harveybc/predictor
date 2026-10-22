# FEATURE_CATALOG summary (EURUSD H1 candidate pool)

Rows: 370. Stores: layer1_pool = 197 columns, file sha256 5baea2e25f40...; l2 = 79 columns, fit end 2024-01-01, file sha256 4a89dfea2a64...; l2_fit20230101 = 14 columns, fit end 2023-01-01, file sha256 5fcccbce31ad...; cross_asset_r4 = 79 columns.

## Counts per family x generation_status

| family | GENERATED | PENDING | SUPERSEDED_BY_R4 | total |
|---|---|---|---|---|
| activity | 1 | 0 | 0 | 1 |
| calendar | 13 | 0 | 0 | 13 |
| candle | 11 | 0 | 0 | 11 |
| cross_asset | 79 | 0 | 79 | 158 |
| event | 6 | 0 | 0 | 6 |
| event_calendar | 0 | 1 | 0 | 1 |
| fracdiff | 3 | 0 | 0 | 3 |
| fractal | 12 | 0 | 0 | 12 |
| gmm | 8 | 0 | 0 | 8 |
| hilbert | 8 | 0 | 0 | 8 |
| hmm | 10 | 0 | 0 | 10 |
| inventory | 22 | 0 | 0 | 22 |
| kalman | 8 | 0 | 0 | 8 |
| multitaper | 8 | 0 | 0 | 8 |
| price_pool | 42 | 0 | 0 | 42 |
| raw_price | 8 | 0 | 0 | 8 |
| ssa | 3 | 0 | 0 | 3 |
| strength | 27 | 0 | 0 | 27 |
| wavelet | 21 | 0 | 0 | 21 |

| status | rows |
|---|---|
| GENERATED | 290 |
| SUPERSEDED_BY_R4 | 79 |
| PENDING | 1 |

## Preprocessing decisions (generated rows)

| decision | rows |
|---|---|
| NONE | 256 |
| DIFF | 12 |
| ASINH | 7 |
| DERIVED_DIFF_OF_OPEN | 1 |
| DERIVED_DIFF_OF_HIGH | 1 |
| DERIVED_DIFF_OF_LOW | 1 |
| DERIVED_DIFF_OF_CLOSE | 1 |
| DERIVED_ASINH_OF_cdl.upper_wick_pips | 1 |
| DERIVED_ASINH_OF_cdl.lower_wick_pips | 1 |
| DERIVED_ASINH_OF_cdl.range_pips | 1 |
| DERIVED_ASINH_OF_cdl.close_minus_low_pips | 1 |
| DERIVED_ASINH_OF_cdl.high_minus_close_pips | 1 |
| DERIVED_DIFF_OF_wv.lc.A4 | 1 |
| DERIVED_DIFF_OF_wv.lc.energy_D1 | 1 |
| DERIVED_DIFF_OF_wv.r1.energy_D1 | 1 |
| DERIVED_ASINH_OF_mt.r1.dominant_period_bars | 1 |
| DERIVED_DIFF_OF_wv.eur.energy_D1 | 1 |
| DERIVED_DIFF_OF_fd.lc.d0.3 | 1 |

## PENDING / REPAIR rows

- `cross_asset` (SUPERSEDED_BY_R4): r4 store exists: these layer-1 values still come from the r3 (HistData-gap) build; the selection lane should consume r4:<column> instead
- `event_calendar` (PENDING): FXMacroData history has no consensus; Investing.com export is research-only; candidate: Trading Economics/FXStreet acquisition — owner decision

## Fitted parameters (layer-2 learned families)

- `l2` / `kalman` (fit 2020-01-01 .. 2024-01-01 (exclusive)): {"boundary_hits":["q_slope"],"boundary_note":"a parameter below 1e-6 x max(q_level, r) is at the zero boundary of the likelihood: with q_slope ~ 0 the local linear trend degenerates to a local level with a drift that converges to a constant, so the slope output carries little information","converged":true,"negloglik":-135052.7692204022,"q_level":1.045273050660741e-06,"q_slope":2.1556426484968197e-24,"r":2.1311434638144307e-08}
- `l2` / `gmm` (fit 2020-01-01 .. 2024-01-01 (exclusive)): {"converged":true,"state_rv24_mean_raw":[0.003570944470654022,0.00522582919986994,0.008114972265576094],"transmat":null,"weights":[0.5116451562900889,0.4069863629459991,0.08136848076391189]}
- `l2` / `hmm` (fit 2020-01-01 .. 2024-01-01 (exclusive)): {"converged":true,"state_rv24_mean_raw":[0.0029004137016789546,0.004711096432720094,0.007581510368651949],"transmat":[[0.9749872241245531,0.0024880438566493085,0.022524732018797653],[0.018488170229735243,0.9417783224045293,0.03973350736573555],[0.013326479068729637,0.09768179333364661,0.8889917275976238]],"weights":null}
- `l2` / `fracdiff` (fit 2020-01-01 .. 2024-01-01 (exclusive)): {"adf":[[0.1,-1.794,"3.83e-01"],[0.2,-2.432,"1.33e-01"],[0.3,-3.449,"9.40e-03"],[0.4,-5.088,"1.48e-05"],[0.5,-7.723,"1.18e-11"],[0.6,-11.777,"1.06e-21"],[0.7,-17.227,"6.23e-30"],[0.8,-22.928,"0.00e+00"],[0.9,-27.526,"0.00e+00"]],"d":0.3,"window":256}
- `l2_fit20230101` / `kalman` (fit 2020-01-01 .. 2023-01-01 (exclusive)): {"boundary_hits":["q_slope"],"boundary_note":"a parameter below 1e-6 x max(q_level, r) is at the zero boundary of the likelihood: with q_slope ~ 0 the local linear trend degenerates to a local level with a drift that converges to a constant, so the slope output carries little information","converged":true,"negloglik":-100857.07559809003,"q_level":1.0808626497355067e-06,"q_slope":9.270177619681418e-15,"r":3.012928225721459e-08}
- `l2_fit20230101` / `gmm` (fit 2020-01-01 .. 2023-01-01 (exclusive)): {"converged":true,"state_rv24_mean_raw":[0.003419868931349476,0.0053061676285864464,0.008431087872897116],"transmat":null,"weights":[0.4863929893888266,0.42194856751035614,0.09165844310081724]}
- `l2_fit20230101` / `hmm` (fit 2020-01-01 .. 2023-01-01 (exclusive)): {"converged":true,"state_rv24_mean_raw":[0.002847200436301118,0.004775627660346453,0.007943642629600402],"transmat":[[0.9789265163823033,0.005176202944996141,0.015897280672700548],[0.018030570151512746,0.9471424699176021,0.03482695993088508],[0.009314078847071612,0.07914044184020251,0.9115454793127259]],"weights":null}
- `l2_fit20230101` / `fracdiff` (fit 2020-01-01 .. 2023-01-01 (exclusive)): {"adf":[[0.1,-1.333,"6.14e-01"],[0.2,-1.857,"3.52e-01"],[0.3,-2.673,"7.88e-02"],[0.4,-3.975,"1.55e-03"],[0.5,-6.076,"1.12e-07"],[0.6,-9.377,"7.13e-16"],[0.7,-13.999,"3.90e-26"],[0.8,-19.099,"0.00e+00"],[0.9,-23.368,"0.00e+00"]],"d":0.4,"window":256}

Outlier share = share of finite TRAIN (2020-2022) values beyond median +- 6 x 1.4826 x MAD (NaN when MAD = 0). selection_decision and consumed_by are PENDING (selection lane). Rows >= 2025-01-01 (TEST) are never read.
