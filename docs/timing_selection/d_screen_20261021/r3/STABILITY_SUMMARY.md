# D-screen stability (years as blocks) and 2023 confirmation

Survivors 149; stable in 2020, 2021 and 2022: **33**; confirmed on 2023: **14**.

| label | stratum | feature | lag | SMD | z2020 | z2021 | z2022 | 2023 stratum | z2023 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| hit | pre_tokyo_fix_2h | px.rv_2 | 24 | +0.471 | +3.32 | +3.14 | +1.83 | pre_tokyo_fix_2h | +0.70 | False |
| hit | pre_tokyo_fix_2h | px.range_2 | 24 | +0.478 | +2.83 | +2.40 | +2.20 | pre_tokyo_fix_2h | +0.88 | False |
| hit | pre_tokyo_fix_2h | px.elapsed_hours | 24 | +0.161 | +2.42 | +2.51 | +3.29 | pre_tokyo_fix_2h | +0.17 | False |
| hit | pre_tokyo_fix_2h | RAW_ELAPSED_HOURS | 24 | +0.161 | +2.42 | +2.51 | +3.29 | pre_tokyo_fix_2h | +0.17 | False |
| up | pre_tokyo_fix_2h | px.rv_1 | 24 | +0.560 | +1.82 | +3.09 | +2.99 | all | +0.91 | False |
| hit | pre_tokyo_fix_2h | px.rv_1 | 24 | +0.334 | +2.78 | +2.50 | +1.95 | pre_tokyo_fix_2h | +1.14 | False |
| hit | pre_tokyo_fix_2h | px.range_1 | 24 | +0.391 | +2.78 | +1.29 | +2.04 | pre_tokyo_fix_2h | +0.39 | False |
| hit | pre_tokyo_fix_2h | RAW_LOG_RANGE | 24 | +0.391 | +2.78 | +1.29 | +2.04 | pre_tokyo_fix_2h | +0.39 | False |
| hit | all | ev.tokyo_fix.hours_since | 8 | -0.124 | -4.11 | -3.45 | -3.42 | all | -3.91 | True |
| hit | all | ev.tokyo_fix.hours_to | 8 | +0.124 | +4.11 | +3.45 | +3.42 | all | +3.91 | True |
| hit | pre_tokyo_fix_2h | px.range_ratio_1_24 | 24 | +0.428 | +3.18 | +1.68 | +2.48 | pre_tokyo_fix_2h | +1.93 | False |
| hit | pre_tokyo_fix_2h | px.rv_4 | 24 | +0.357 | +2.44 | +2.16 | +1.45 | pre_tokyo_fix_2h | -0.09 | False |
| hit | pre_tokyo_fix_2h | cal.utc_dow_sin | 24 | -1.001 | -2.41 | -2.25 | -2.31 | pre_tokyo_fix_2h | -2.59 | True |
| hit | pre_tokyo_fix_2h | cal.utc_dow_cos | 0 | +1.002 | +2.41 | +2.25 | +2.30 | pre_tokyo_fix_2h | +2.60 | True |
| hit | lowvol_europe | fx.usdchf.ret_4 | 8 | +0.359 | +1.26 | +3.52 | +1.83 | lowvol_europe | -0.73 | False |
| hit | pre_tokyo_fix_2h | dow_cos | 0 | +0.823 | +2.31 | +2.20 | +2.25 | pre_tokyo_fix_2h | +2.47 | True |
| hit | all | px.range_ratio_1_24 | 1 | +0.156 | +3.41 | +2.72 | +2.37 | all | +1.64 | False |
| hit | post_london_fix_2h | dly.dax.ret_5d | 48 | -0.373 | -2.89 | -1.16 | -2.34 | post_london_fix_2h | -2.06 | True |
| hit | pre_tokyo_fix_2h | cal.utc_dow_cos | 120 | +0.972 | +2.32 | +2.27 | +2.27 | pre_tokyo_fix_2h | +2.51 | True |
| hit | pre_tokyo_fix_2h | cal.utc_dow_cos | 1 | +0.932 | +2.31 | +2.17 | +2.20 | pre_tokyo_fix_2h | +2.50 | True |
| hit | pre_tokyo_fix_2h | dow_cos | 1 | +0.932 | +2.31 | +2.17 | +2.20 | pre_tokyo_fix_2h | +2.50 | True |
| hit | pre_tokyo_fix_2h | dow_cos | 120 | +0.804 | +2.27 | +2.25 | +2.22 | pre_tokyo_fix_2h | +2.36 | True |
| hit | all | px.atr_ratio_14_120 | 48 | +0.221 | +3.53 | +1.22 | +1.52 | all | +0.67 | False |
| hit | post_london_fix_2h | dly.spx.ret_1d | 48 | -0.291 | -2.26 | -1.21 | -2.40 | post_london_fix_2h | -2.07 | True |
| hit | pre_tokyo_fix_2h | px.hours_since_session_open | 2 | -0.666 | -2.28 | -2.10 | -2.13 | pre_tokyo_fix_2h | -2.21 | True |
| hit | all | px.elapsed_hours | 24 | +0.038 | +2.27 | +2.17 | +2.30 | all | +1.63 | False |
| hit | all | RAW_ELAPSED_HOURS | 24 | +0.038 | +2.27 | +2.17 | +2.30 | all | +1.63 | False |
| hit | lowvol_europe | cal.london_hod_sin | 4 | +0.195 | +1.42 | +4.54 | +1.15 | lowvol_europe | +1.42 | False |
| hit | all | px.range_ratio_1_24 | 0 | +0.153 | +3.37 | +2.70 | +2.56 | all | +1.59 | False |
| hit | post_tokyo_fix_2h | dow_sin | 48 | -0.976 | -2.21 | -2.09 | -2.13 | post_tokyo_fix_2h | -2.24 | True |
| hit | midvol_us | fx.gbpusd.ret_24 | 4 | +0.395 | +2.25 | +2.07 | +1.92 | midvol_us | +1.22 | False |
| hit | post_tokyo_fix_2h | cal.utc_dow_sin | 48 | -0.967 | -2.20 | -2.07 | -2.13 | post_tokyo_fix_2h | -2.26 | True |
| hit | post_tokyo_fix_2h | px.rv_4 | 24 | +0.346 | +2.65 | +1.19 | +1.41 | post_tokyo_fix_2h | -0.37 | False |
