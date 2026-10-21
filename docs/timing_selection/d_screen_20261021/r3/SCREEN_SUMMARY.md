# D-screen (lag x stratum, split labels), TRAIN 2020-2022 with INNER_A confirmation

Tests (coverage >= 0.5): 47328; BH q = 0.1: **149 pass**; confirmed on INNER_A: **10**.

Input counts: {"TRAIN": {"rows": 18698, "hit_rows": 14823, "hit_rate": 0.31316197800715107, "up_rows": 4642, "up_rate": 0.5122791900043084}, "INNER_A": {"rows": 3111, "hit_rows": 2487, "hit_rate": 0.31363088057901084, "up_rows": 780, "up_rate": 0.6012820512820513}}

## Survivors (top 60 by p)

| label | stratum | feature | lag | SMD | z | p | confirmed 2023 S1 |
|---|---|---|---|---|---|---|---|
| hit | highvol_asia | dly.dax.ret_1d | 48 | -0.683 | -5.12 | 3.1e-07 | False |
| hit | highvol_europe | ev.tokyo_fix.hours_to | 8 | +0.175 | +5.00 | 5.6e-07 | False |
| hit | highvol_europe | ev.tokyo_fix.hours_since | 8 | -0.175 | -5.00 | 5.6e-07 | False |
| hit | post_tokyo_fix_2h | dly.dax.ret_1d | 48 | -0.450 | -4.97 | 6.7e-07 | False |
| hit | pre_tokyo_fix_2h | px.rv_2 | 24 | +0.471 | +4.91 | 9.2e-07 | False |
| hit | highvol_late | dly.dax.ret_1d | 48 | -0.652 | -4.68 | 2.9e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 24 | -0.440 | -4.56 | 5.0e-06 | False |
| hit | highvol_europe | cal.utc_session_europe | 1 | +0.151 | +4.55 | 5.2e-06 | False |
| hit | highvol_europe | cal.utc_session_asia | 1 | -0.151 | -4.55 | 5.2e-06 | False |
| hit | highvol_europe | cal.utc_session_asia | 8 | +0.151 | +4.55 | 5.2e-06 | False |
| hit | pre_tokyo_fix_2h | px.range_2 | 24 | +0.478 | +4.55 | 5.5e-06 | False |
| hit | pre_tokyo_fix_2h | px.elapsed_hours | 24 | +0.161 | +4.54 | 5.6e-06 | False |
| hit | pre_tokyo_fix_2h | RAW_ELAPSED_HOURS | 24 | +0.161 | +4.54 | 5.6e-06 | False |
| hit | all | dly.dax.ret_1d | 48 | -0.338 | -4.50 | 6.9e-06 | False |
| up | pre_tokyo_fix_2h | px.rv_1 | 24 | +0.560 | +4.49 | 7.1e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 8 | -0.410 | -4.42 | 9.8e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 0 | -0.410 | -4.42 | 9.8e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 4 | -0.410 | -4.42 | 9.8e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 2 | -0.410 | -4.42 | 9.8e-06 | False |
| hit | post_london_fix_2h | dly.spx.ret_5d | 1 | -0.410 | -4.42 | 9.8e-06 | False |
| hit | pre_tokyo_fix_2h | dly.dax.ret_1d | 48 | -0.391 | -4.40 | 1.1e-05 | False |
| hit | highvol_asia | fac.daily_f1 | 48 | -0.574 | -4.37 | 1.3e-05 | False |
| hit | highvol_europe | hod_sin | 4 | +0.174 | +4.28 | 1.8e-05 | False |
| hit | highvol_asia | dly.spx.ret_1d | 48 | -0.535 | -4.26 | 2.0e-05 | False |
| hit | pre_tokyo_fix_2h | px.rv_1 | 24 | +0.334 | +4.26 | 2.0e-05 | False |
| hit | post_tokyo_fix_2h | fac.daily_f1 | 48 | -0.378 | -4.25 | 2.1e-05 | False |
| hit | highvol_europe | px.atr_ratio_14_120 | 24 | +0.571 | +4.21 | 2.6e-05 | False |
| hit | highvol_late | dly.spx.ret_1d | 48 | -0.472 | -4.17 | 3.0e-05 | False |
| up | lowvol_us | px.rv_6 | 120 | +0.752 | +4.16 | 3.2e-05 | False |
| hit | highvol_asia | px.range_4 | 24 | +0.535 | +4.15 | 3.4e-05 | False |
| hit | post_london_fix_2h | dly.spx.ret_1d | 24 | -0.316 | -4.14 | 3.5e-05 | False |
| up | post_tokyo_fix_2h | fx.eurgbp.ret_4 | 48 | -0.491 | -4.12 | 3.8e-05 | False |
| hit | all | dly.spx.ret_1d | 48 | -0.271 | -4.12 | 3.9e-05 | False |
| hit | all | dly.spx.ret_5d | 24 | -0.360 | -4.05 | 5.2e-05 | False |
| hit | pre_tokyo_fix_2h | px.range_1 | 24 | +0.391 | +4.04 | 5.3e-05 | False |
| hit | pre_tokyo_fix_2h | RAW_LOG_RANGE | 24 | +0.391 | +4.04 | 5.3e-05 | False |
| hit | highvol_asia | dly.spx.ret_5d | 48 | -0.602 | -4.02 | 5.9e-05 | False |
| hit | highvol_us | dly.dax.ret_1d | 48 | -0.463 | -4.02 | 5.9e-05 | True |
| hit | all | ev.tokyo_fix.hours_since | 8 | -0.124 | -4.01 | 6.0e-05 | True |
| hit | all | ev.tokyo_fix.hours_to | 8 | +0.124 | +4.01 | 6.0e-05 | True |
| hit | highvol_us | dly.spx.ret_5d | 8 | -0.601 | -4.01 | 6.1e-05 | True |
| hit | highvol_us | dly.spx.ret_5d | 1 | -0.601 | -4.01 | 6.1e-05 | True |
| hit | highvol_us | dly.spx.ret_5d | 2 | -0.601 | -4.01 | 6.1e-05 | True |
| hit | highvol_us | dly.spx.ret_5d | 4 | -0.601 | -4.01 | 6.1e-05 | True |
| hit | highvol_us | dly.spx.ret_5d | 0 | -0.601 | -4.01 | 6.1e-05 | True |
| hit | pre_tokyo_fix_2h | px.range_ratio_1_24 | 24 | +0.428 | +3.99 | 6.6e-05 | False |
| up | lowvol_us | px.rv_12 | 120 | +0.834 | +3.98 | 6.8e-05 | False |
| hit | highvol_us | dly.spx.ret_5d | 24 | -0.593 | -3.98 | 7.0e-05 | False |
| up | lowvol_asia | fx.usdchf.rv_24 | 120 | +0.937 | +3.97 | 7.2e-05 | False |
| hit | all | dly.spx.ret_5d | 8 | -0.345 | -3.96 | 7.5e-05 | False |
| hit | highvol_asia | px.range_2 | 24 | +0.441 | +3.96 | 7.6e-05 | False |
| hit | highvol_asia | dly.spx.ret_5d | 24 | -0.611 | -3.95 | 7.7e-05 | False |
| up | lowvol_late | px.rv_12 | 120 | +0.879 | +3.94 | 8.3e-05 | False |
| hit | highvol_asia | dly.spx.ret_5d | 8 | -0.598 | -3.93 | 8.5e-05 | False |
| hit | highvol_europe | px.range_2 | 1 | +0.378 | +3.92 | 8.8e-05 | False |
| hit | all | dly.spx.ret_5d | 4 | -0.339 | -3.91 | 9.1e-05 | False |
| hit | all | dly.spx.ret_5d | 2 | -0.338 | -3.91 | 9.2e-05 | False |
| hit | pre_tokyo_fix_2h | px.rv_4 | 24 | +0.357 | +3.89 | 9.9e-05 | False |
| hit | all | dly.spx.ret_5d | 1 | -0.336 | -3.89 | 1.0e-04 | False |
| hit | all | dly.spx.ret_5d | 0 | -0.335 | -3.88 | 1.0e-04 | False |
