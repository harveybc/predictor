# D-screen (lag x stratum, split labels), TRAIN 2020-2022 with INNER_A confirmation

Tests (coverage >= 0.5): 42704; BH q = 0.1: **4 pass**; confirmed on INNER_A: **0**.

Input counts: {"TRAIN": {"rows": 18698, "hit_rows": 14823, "hit_rate": 0.31316197800715107, "up_rows": 4642, "up_rate": 0.5122791900043084}, "INNER_A": {"rows": 3111, "hit_rows": 2487, "hit_rate": 0.31363088057901084, "up_rows": 780, "up_rate": 0.6012820512820513}}

## Survivors (top 60 by p)

| label | stratum | feature | lag | SMD | z | p | confirmed 2023 S1 |
|---|---|---|---|---|---|---|---|
| hit | pre_tokyo_fix_2h | px.rv_2 | 24 | +0.471 | +4.64 | 3.5e-06 | False |
| hit | highvol_europe | ev.tokyo_fix.hours_to | 8 | +0.175 | +4.62 | 3.9e-06 | False |
| hit | highvol_europe | ev.tokyo_fix.hours_since | 8 | -0.175 | -4.62 | 3.9e-06 | False |
| up | pre_tokyo_fix_2h | px.rv_1 | 24 | +0.560 | +4.47 | 7.8e-06 | False |
