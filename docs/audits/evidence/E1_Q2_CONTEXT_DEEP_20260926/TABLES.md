### N1. The block as sealed

* design `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318`, schema `df_e1_block_design.v1`, block `Q2_CONTEXT_DEEP`, state at seal `SEALED_NOT_EXECUTED`, phase `DEVELOPMENT`
* tier: TIER1 CADENCE AND PATIENCE, BUDGET-MATCHED CEILING: RP66-RP73 blocks: patience 3 events (600 non-improving updates); ceiling 600 observed updates for every arm of the block, see recipe.budget_declaration
* prepared data `2b50b777dab1cf5ba9443b9b08533c6c819150a22b1d515c99914b44aeb71ff9`, panel rows 1410981..1462761 (pad 1380), common evaluation **10020 origins** (panel rows 41700..51719), sigma_evaluation 0.9125164391265214 kW
* train population `COMMON_INTERSECTION`: **38700 origins, identical for every arm**; subset of the source run's train origins: True; the 28 d baseline enumeration reproduces the source's train origins: True; the common evaluation equals the source's: True
* recipe: mae loss, adam, lr 0.003, batch 64, ceiling **600 updates**, validation every 200 observed updates, patience 3 events, restore_best True, min_delta 0.0
* scaler rule: COMMON: the source run's train-only scaler (28 d, W60 windows) for every arm and tier; calendar channels mean 0 / sd 1; the lag channel takes the target's scaler; one evaluation sigma = the target's train sd
* common evaluation rule: the intersection over the block's arms of admissible validation origins, with a finite label and a finite daily lookup, derived at prepare BEFORE any score; every arm scores on it

| arm | raw window | crop | usable context samples | core depth (dilated blocks) | core receptive field | features | role | parameters | per-arm train admissible before the intersection |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| `modular_w60` | 60 | — | 60 | 5 | 63 | base | ARM | 8127 | 40080 |
| `daily_lag` | 60 | — | 60 | 5 | 63 | daily_lag | ARM | 8208 | 40020 |
| `long_window_crop60` | 1440 | 60 | 60 | 5 | 63 | base | EXACT_INFORMATION_NULL: the raw input is cropped to its last 60 rows before the extractor | 8127 | 38700 |
| `short_window_deep_core` | 60 | — | 60 | 10 | 2047 | base | ARM | 12047 | 40080 |
| `long_window_local_support_67` | 1440 | — | 63 | 5 | 63 | base | EXTRA_CONTEXT_67_SAMPLES (measured): NOT a null; the clamped core still reaches branch 5 + core 63 - 1 = 67 raw samples | 8127 | 38700 |
| `long_window_own_depth` | 1440 | — | 1440 | 10 | 2047 | base | ARM | 12047 | 38700 |

**Reading rules, verbatim from the sealed design:** *three seeds on one task are development evidence* · *a fit that reached the update ceiling is CENSORED wherever its best checkpoint fell* · *no cell is removed after its score is seen* · *a published number under another protocol never enters the comparison column*

**The block's own question, as sealed:** context beyond the hour, SEPARATED from receiver depth and from train volume, at a MATCHED update budget. Volume is held fixed BY CONSTRUCTION: train_population COMMON_INTERSECTION, so every arm trains on the SAME origins. The crossing is raw input window {60, 1440} x causal core depth {5 dilated blocks, 10 dilated blocks}: (60,5) modular_w60, (60,10) short_window_deep_core, (1440,5) long_window_local_support_67, (1440,10) long_window_own_depth, with long_window_crop60 as the exact-information null and daily_lag as the causal daily-lag channel. WHAT IT CANNOT SEPARATE, declared before any score: in this architecture family the receptive field is 1 + 2*sum(dilations), so a raw window longer than 67 samples is only USED when depth grows; the (1440,5) cell carries the long window's padding and 67 samples of reach, NOT 1440 samples of information, and context beyond 67 samples is confounded with depth BY CONSTRUCTION of the family. Only the depth-10 row can carry long context, so the context contrast at matched depth and matched volume is long_window_own_depth - short_window_deep_core, and the depth contrast at matched context is short_window_deep_core - modular_w60 and long_window_own_depth - long_window_local_support_67. SECOND DECLARED LIMIT: every cell is fitted to a ceiling of 600 observed updates, so every cell is CENSORED_BY_BUDGET wherever its best checkpoint fell; an arm whose validation MAE_z improved by more than 0.005 over its last 200 updates in 2 or more of its 3 seeds is UNDERTRAINED_AT_CEILING and its contrast is NOT read as a context or depth effect

**Why this block exists and where its ceiling comes from, verbatim from the sealed design:** SEALED BEFORE ANY SCORE OF ANY CELL OF THIS BLOCK. It extends Q2_CONTEXT_BOUNDED (design 47a270eec01f203cdde2812deb1458db525e86d762c2b17f2b79a2ba571e17ea, twelve fits, four arms) with the two W1440 FULL-DEPTH arms that block declared it could not hold, and it refits ALL six arms under one budget so no arm is compared across budgets. The ceiling is 600 observed updates instead of 4 000, and that number comes from the RETAINED cost pilots' own measured rates on this same host (<worker-host>) and from nothing else: long_window_own_depth 4.165 CPU s per train update and 8 458 399 744 B peak RSS, long_window_local_support_67 4.4427 CPU s per update and 10 279 276 544 B peak RSS, modular_w60 0.0390, daily_lag 0.0392, long_window_crop60 0.0410 and short_window_deep_core 0.0710 CPU s per update with peak RSS under 1 GiB. At the 4 000-update ceiling the two deep arms would cost 17 507 s and 18 406 s of CPU PER CELL (about 30 CPU hours and roughly 19 h of wall for their six cells) and hold 7.9 and 9.6 GiB resident; at 600 updates they cost about 2 630 s and 2 760 s per cell. The reduction is a RESOURCE declaration made when no cell of this block had a score, never a removal or a re-budgeting after a score was seen (reading rule 3). Its cost is stated in the question field: every cell of every arm is CENSORED_BY_BUDGET and the block carries the UNDERTRAINED_AT_CEILING rule to say so per arm. No arm, seed, recipe field other than max_updates, scaler, row, cadence, monitor, patience or metric of Q2_CONTEXT_BOUNDED is otherwise changed

**The budget declaration, verbatim from the sealed recipe:** 600 observed updates, not the 4 000 of Q2_CONTEXT_BOUNDED, chosen BEFORE any cell of this block had a score from the retained cost pilots' own measured rates on this host: the two W1440 full-depth arms cost 4.165 and 4.4427 CPU s per train update, i.e. 17 507 s and 18 406 s of CPU per cell at a 4 000-update ceiling and roughly 19 h of wall for their six cells, against about 2 630 s and 2 760 s per cell at 600. Consequence, declared here and not discovered later: with validation every 200 updates there are 3 checkpoint opportunities and patience 3 can never expire, so EVERY cell of EVERY arm reaches the ceiling and is CENSORED_BY_BUDGET wherever its best checkpoint fell. Nothing in this block claims convergence, and the UNDERTRAINED_AT_CEILING rule in the block's question field refuses to read a contrast as an effect for an arm still improving at the ceiling.

### N2. The cost basis, and the projection that could not be produced

**There is no `REPORT.pilot.json` for this block, and the reason is part of the result.** `tools/df_e1_block_ungoverned.py pilot-report` builds its projection from EVERY pilot the design registers, and two of the six were never recorded: `pilot_long_window_own_depth` was terminated by systemd-oomd for user-session memory pressure after 1 292.106 CPU s, and `pilot_long_window_local_support_67` never launched. Under the stand-down neither was retried, so no projection over six arms exists and none is invented here. What follows is each pilot that DID land, read straight from its own cell record, beside the retained Q2_CONTEXT v1 measurements for the two arms that did not.

**Every peak figure in this section and in N3 is `MAIN_PROCESS_RSS_ONLY`** — `resource.getrusage(RUSAGE_SELF).ru_maxrss` as `df_e1_block.run_cell` records it. It is NOT a process tree or cgroup peak, and a placement decision needs the latter. See [`FOOTPRINT_BASIS.json`](FOOTPRINT_BASIS.json), which lists every figure with what measured it and the cgroup peaks systemd recorded for the same scopes.

| arm | pilot | CPU s per train update | peak RSS GiB (MAIN_PROCESS_RSS_ONLY) | projected CPU s per cell at this block's ceiling | source |
|---|---|---:|---:|---:|---|
| `modular_w60` | landed | 0.0414 | 0.87 | 24.8 | this block's own pilot |
| `daily_lag` | landed | 0.0407 | 0.87 | 24.4 | this block's own pilot |
| `long_window_crop60` | landed | 0.0424 | 0.87 | 25.4 | this block's own pilot |
| `short_window_deep_core` | landed | 0.0727 | 0.90 | 43.6 | this block's own pilot |
| `long_window_local_support_67` | **NOT RECORDED** | 4.4427 | 9.57 | 2665.6 | RETAINED Q2_CONTEXT v1 pilot, same host, 2026-09-21 |
| `long_window_own_depth` | **NOT RECORDED** | 4.1650 | 7.88 | 2499.0 | RETAINED Q2_CONTEXT v1 pilot, same host, 2026-09-21 |

* campaign ceiling 32400 CPU s, closure reserve 2000 CPU s, per-child CPU ceiling 5400 s, per-child wall ceiling 7200 s, parallel_children 1

### N3. Every fit, as it landed

Errors recomputed here from each cell's retained `arrays.npz`, each cross-checked against the value the cell record stored (a disagreement above 1e-12 refuses the whole table), and every arm verified to have scored the IDENTICAL origin array.

| cell | arm | seed | MAE_z | MAE kW | naive MAE kW, same rows | skill vs naive | worse than naive | stop | censoring | updates | best update | val MAE_z improvement over the last 200 updates | CPU s | peak RSS GiB (MAIN_PROCESS_RSS_ONLY) | reload max err | fresh-process replay |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `modular_w60_s1` | `modular_w60` | 1 | 0.559310 | 0.510380 | 0.617372 | 0.173303 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019321 | 27.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s1` | `daily_lag` | 1 | 0.550661 | 0.502487 | 0.617372 | 0.186087 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.016519 | 28.2 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s1` | `long_window_crop60` | 1 | 0.559310 | 0.510380 | 0.617372 | 0.173303 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019321 | 31.8 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s1` | `short_window_deep_core` | 1 | 0.570734 | 0.520804 | 0.617372 | 0.156417 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.005287 | 51.9 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s1` | `long_window_local_support_67` | 1 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s1` | `long_window_own_depth` | 1 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `modular_w60_s2` | `modular_w60` | 2 | 0.566499 | 0.516940 | 0.617372 | 0.162677 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.040595 | 27.9 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s2` | `daily_lag` | 2 | 0.560715 | 0.511662 | 0.617372 | 0.171227 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.023692 | 28.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s2` | `long_window_crop60` | 2 | 0.566499 | 0.516940 | 0.617372 | 0.162677 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.040595 | 32.4 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s2` | `short_window_deep_core` | 2 | 0.568177 | 0.518471 | 0.617372 | 0.160196 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 200 | +0.008548 | 57.5 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s2` | `long_window_local_support_67` | 2 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s2` | `long_window_own_depth` | 2 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `modular_w60_s3` | `modular_w60` | 3 | 0.565577 | 0.516098 | 0.617372 | 0.164040 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.021223 | 28.9 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `daily_lag_s3` | `daily_lag` | 3 | 0.563220 | 0.513947 | 0.617372 | 0.167524 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 400 | -0.011196 | 30.2 | 0.87 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_crop60_s3` | `long_window_crop60` | 3 | 0.565577 | 0.516098 | 0.617372 | 0.164040 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.021223 | 34.3 | 0.88 | 0.00e+00 | allclose(1e-6) PASS |
| `short_window_deep_core_s3` | `short_window_deep_core` | 3 | 0.556763 | 0.508056 | 0.617372 | 0.177067 | no | UPDATE_BUDGET | CENSORED_BY_BUDGET | 600 | 600 | +0.019160 | 58.2 | 0.91 | 0.00e+00 | allclose(1e-6) PASS |
| `long_window_local_support_67_s3` | `long_window_local_support_67` | 3 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |
| `long_window_own_depth_s3` | `long_window_own_depth` | 3 | **NOT FITTED** | — | — | — | — | NOT_STARTED | NOT_STARTED | — | — | — | — | — | — | — |

Censoring across the 12 fits: `CENSORED_BY_BUDGET`.

**6 of the 18 registered cells were NEVER FITTED and are named here, not dropped: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3`.** The design still registers all 18; N13 carries every memory reading that refused to start them. Arms with all 3 seeds: `modular_w60`, `daily_lag`, `long_window_crop60`, `short_window_deep_core`. No contrast is taken across an incomplete arm, and every contrast the missing cells block is printed below as `NOT_MEASURED` with the cells that are missing.

Initial-weight digests, per seed: seed 1: `modular_w60` b9caefebdc3c…, `daily_lag` 0f6b46da2c89…, `long_window_crop60` b9caefebdc3c…, `short_window_deep_core` 9d7edff6f248… · seed 2: `modular_w60` 84f67bc65cb5…, `daily_lag` f9c474649f56…, `long_window_crop60` 84f67bc65cb5…, `short_window_deep_core` 1a6308265c88… · seed 3: `modular_w60` e0e2765cded8…, `daily_lag` 7fc47852dea4…, `long_window_crop60` e0e2765cded8…, `short_window_deep_core` 59e492f0e56f…

### N4. Per arm, and the paired difference against the baseline arm

| arm | mean MAE_z | sd (ddof 1) | mean MAE kW | mean skill vs naive | paired Δ MAE_z vs `modular_w60`, per seed | mean Δ | signs (+ / −) |
|---|---:|---:|---:|---:|---|---:|---|
| `modular_w60` | 0.563796 | 0.003912 | 0.514473 | 0.166673 | — (this is the baseline arm) | — | — |
| `daily_lag` | 0.558198 | 0.006647 | 0.509365 | 0.174946 | -0.008649 · -0.005785 · -0.002357 | -0.005597 | 0 / 3 |
| `long_window_crop60` | 0.563796 | 0.003912 | 0.514473 | 0.166673 | +0.000000 · +0.000000 · +0.000000 | 0.000000 | 0 / 0 |
| `short_window_deep_core` | 0.565225 | 0.007439 | 0.515777 | 0.164560 | +0.011424 · +0.001678 · -0.008814 | 0.001429 | 2 / 1 |

Negative Δ = smaller error than the baseline arm. Three paired seeds on one task, one previously inspected DEV validation week: **development evidence**, as the sealed reading rule says. Both signs are printed and **no interval is claimed from n = 3**. A difference between two forecast errors is not a verified causal effect: no causal claim here is verified against a retained-row error, because a causal claim does not predict a retained row.

### N5. The crossing: input context x core depth, with volume fixed by construction

Volume is not a factor here: every arm trained on the SAME 38700 origins and scored the SAME 10020, both checked as arrays above. The two factors that remain are the raw input window and the depth of the causal core.

| | depth 5 blocks | depth 10 blocks |
|---|---|---|
| **raw window 60** | `modular_w60`<br>mean MAE_z **0.563796**<br>8127 parameters, usable context 60 samples | `short_window_deep_core`<br>mean MAE_z **0.565225**<br>12047 parameters, usable context 60 samples |
| **raw window 1440** | `long_window_local_support_67`<br>**NOT FITTED** (no seed)<br>8127 parameters, usable context 63 samples | `long_window_own_depth`<br>**NOT FITTED** (no seed)<br>12047 parameters, usable context 1440 samples |

**The contrasts, exactly as the seal declared them before any fit.** Each is taken within a seed.

| declared contrast | what it isolates | per-seed Δ MAE_z | mean Δ | signs (+ / −) |
|---|---|---|---:|---|
| `causal_channel`<br>`daily_lag - modular_w60` | a causal daily-lag channel y(t+h-1440) added to the W60 receiver | -0.008649 · -0.005785 · -0.002357 | -0.005597 | 0 / 3 |
| `context_at_matched_depth_10`<br>`long_window_own_depth - short_window_deep_core` | input context 1440 vs 60 at depth 10, with capacity identical (12 047 parameters both), depth identical and volume identical: **the context contrast** | **NOT_MEASURED** — these cells were never fitted: `long_window_own_depth_s1`, `long_window_own_depth_s2`, `long_window_own_depth_s3` | — | — |
| `context_at_matched_depth_5`<br>`long_window_local_support_67 - modular_w60` | input context at depth 5, where the core reaches only 63 samples: a few extra samples plus the long window's padding, not 1440 samples of information | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_local_support_67_s2`, `long_window_local_support_67_s3` | — | — |
| `depth_at_matched_context_1440`<br>`long_window_own_depth - long_window_local_support_67` | core depth 10 vs 5 with a 1440-row raw window: depth AND the context depth unlocks, jointly, so it identifies neither alone | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3` | — | — |
| `depth_at_matched_context_60`<br>`short_window_deep_core - modular_w60` | core depth 10 vs 5 with the usable context pinned at 60 samples: **the depth contrast** | +0.011424 · +0.001678 · -0.008814 | +0.001429 | 2 / 1 |
| `exact_information_null`<br>`long_window_crop60 - modular_w60` | the W1440 input cropped to its last 60 rows: the same computation as the baseline (RP87), measured here rather than assumed | +0.000000 · +0.000000 · +0.000000 | +0.000000 | 0 / 0 |
| `interaction`<br>`(long_window_own_depth - short_window_deep_core) - (long_window_local_support_67 - modular_w60)` | does the context difference itself depend on depth | **NOT_MEASURED** — these cells were never fitted: `long_window_local_support_67_s1`, `long_window_own_depth_s1`, `long_window_local_support_67_s2`, `long_window_own_depth_s2`, `long_window_local_support_67_s3`, `long_window_own_depth_s3` | — | — |

Negative Δ = the first-named arm has the smaller error. **Three seeds is three seeds:** a sign count of 3 / 0 on n = 3 is a direction, not an effect, and no interval is claimed for any row above.

### N6. The UNDERTRAINED_AT_CEILING verdict, by the rule sealed before any fit

* statistic: improvement in validation MAE_z over the last 200 observed updates, i.e. val_mae(second-to-last event) - val_mae(last event), from each cell's own retained events
* threshold: **0.005** MAE_z
* verdict: an arm whose improvement exceeds the threshold in 2 or more of its 3 seeds is UNDERTRAINED_AT_CEILING, and no contrast involving it is read as a context or depth effect
* declared: before any cell of this block was fitted

| arm | improvement over the last 200 updates, per seed | seeds above the threshold | verdict |
|---|---|---:|---|
| `modular_w60` | +0.019321 · +0.040595 · +0.021223 | 3 | **UNDERTRAINED_AT_CEILING** |
| `daily_lag` | -0.016519 · +0.023692 · -0.011196 | 1 | **NOT_UNDERTRAINED_BY_THIS_RULE** |
| `long_window_crop60` | +0.019321 · +0.040595 · +0.021223 | 3 | **UNDERTRAINED_AT_CEILING** |
| `short_window_deep_core` | -0.005287 · +0.008548 · +0.019160 | 2 | **UNDERTRAINED_AT_CEILING** |
| `long_window_local_support_67` | — · — · — | — | **NOT_FITTED_NO_VERDICT** |
| `long_window_own_depth` | — · — · — | — | **NOT_FITTED_NO_VERDICT** |

**3 arm(s) are UNDERTRAINED_AT_CEILING: `modular_w60`, `long_window_crop60`, `short_window_deep_core`.** By the rule sealed before any fit, no contrast involving them is read as a context or depth effect. The numbers stay published exactly as they landed.

### N7. The three declared references, on the same rows

Computed by `df_e1_block.baselines` on the block's 10020 common evaluation origins. The block's closure suppresses these when it fails, so they are published separately.

| reference | definition | MAE kW | MAE_z | skill vs persistence |
|---|---|---:|---:|---:|
| `persistence` | y(t) | 0.617372 | 0.676560 | 0.000000 |
| `daily_seasonal` | y(t+h-1440) | 0.731659 | 0.801804 | -0.185119 |
| `train_constant` | mean of the train labels of the 28 d tier; computed on the common evaluation set at closure, no fit, no terminal | 0.709950 | 0.778013 | -0.149955 |

Published exactly as they landed: `daily_seasonal`, `train_constant` are **worse** than persistence on these rows.

### N8. The owner closure table as it landed

`owner_closure_table.v2` from `tools/df_closure_table.py`, generated 2026-09-26T19:32:27Z: **18 rows, 0 verified, 0 preserved with a qualified scope**; custody classes {"UNCHECKED": 18}; preparation classes {"PREPARATION_LOCAL_ONLY": 18}.

**This is the load-bearing fact about this run, and it is printed before any number of mine.** The verifier's policy is that a score with **no accepted terminal receipt is not reported as a model error at all** — not as a qualified one. This host holds no data-gov service key, so no terminal was ever accepted, so every error column below is `null` and every custody is `UNCHECKED`. Published exactly as it landed; the verifier's custody policy was not touched to make a number appear.

| unit | task / horizon / split | metric and scale | model error | naive error | skill | literature value + source | placed in the comparison column | comparability | custody | binding |
|---|---|---|---:|---:|---:|---|---|---|---|---|
| `modular_w60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s1` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s2` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `modular_w60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `daily_lag_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_crop60_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `short_window_deep_core_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_local_support_67_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |
| `long_window_own_depth_s3` | uci_235.W60_h60.DEV_28d_7d | h=60 steps (3600 s) | DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | null | null | UNDEFINED | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (PUBLISHED (not reproduced here)) | False | **NOT_COMPARABLE** | UNCHECKED | NOT_BOUND |

**Why NOT_COMPARABLE, verbatim:** unknown identity fields cannot match as proof: ['target_transform']

**Planned matched comparison, verbatim:** read the primary source (or its code) and fill the field; a placeholder is not a protocol

**Every problem the table recorded, in full:**

* Q2_CONTEXT_DEEP_20260926/daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_local_support_67_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/long_window_own_depth_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* Q2_CONTEXT_DEEP_20260926/short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt

### N9. The unanchored measurement table

`unanchored_measurement_table.v1`. It supplies the five columns the owner's rule names, for a run the owner closure table can only print as `null`. Each error is recomputed from the cell's own retained arrays and cross-checked against the record's stored score; the contract columns are taken from the landed owner table above. **It is not an `owner_closure_table.v2` row, it is never verified, and its custody is `UNANCHORED_NO_ACCEPTED_TERMINAL` in every row.** Nothing may promote, select or rank on it.

| unit | metric and scale | model error | naive error, SAME rows | skill | rows (model / naive) | horizon (model / naive) | scale (model / naive) | literature value + source | comparability | custody |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `modular_w60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.510380 | 0.617372 | 0.173303 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.502487 | 0.617372 | 0.186087 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.510380 | 0.617372 | 0.173303 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s1` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.520804 | 0.617372 | 0.156417 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516940 | 0.617372 | 0.162677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.511662 | 0.617372 | 0.171227 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516940 | 0.617372 | 0.162677 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s2` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.518471 | 0.617372 | 0.160196 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `modular_w60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516098 | 0.617372 | 0.164040 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `daily_lag_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.513947 | 0.617372 | 0.167524 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `long_window_crop60_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.516098 | 0.617372 | 0.164040 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |
| `short_window_deep_core_s3` | MAE, kW (minute-averaged active power); MAE_z = MAE/sd_train in its own column | 0.508056 | 0.617372 | 0.177067 | 10020 / 10020 | 60 / 60 | kW / kW | NOT_COMPARABLE: Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060 | **NOT_COMPARABLE** | UNANCHORED_NO_ACCEPTED_TERMINAL |

**Fits that landed worse than their naive reference: 0 of 12.** Every fit's skill is printed above whatever its sign.

### N10. What the matched 600-update ceiling costs, measured on the arms both blocks share

The block `Q2_CONTEXT_BOUNDED` (design `47a270eec01f203c…`) fitted 4 of these arms on the SAME rows, the SAME scaler, the SAME seeds and the SAME cadence, but to a **4000-update** ceiling with patience 3 allowed to expire. Its numbers are a direct measurement of what this block's budget costs a cheap arm — and the only honest way to say how far an arm might still have had to travel.

| arm | mean MAE_z at 600 updates (this block) | mean MAE_z at 4000 updates (that block) | Δ (this − that) | best update there, per seed |
|---|---:|---:|---:|---|
| `modular_w60` | 0.563796 | 0.546949 | 0.016846 | 1200, 2400, 1200 |
| `daily_lag` | 0.558198 | 0.541800 | 0.016398 | 1000, 1400, 1200 |
| `long_window_crop60` | 0.563796 | 0.546949 | 0.016846 | 1200, 2400, 1200 |
| `short_window_deep_core` | 0.565225 | 0.550352 | 0.014873 | 800, 1200, 1200 |

This section compares two BLOCKS, not two arms: it measures the budget, and nothing in N5 is adjusted by it.

### N11. The closure as it landed

* `df_e1_block_report.v3`, design `7d0bf92152809a1e59973448f61d14cb7ffe98988944dac05a90d718be18a318`, block `Q2_CONTEXT_DEEP`
* **`verified`: False**
* common evaluation rows 10020; sigma_evaluation 0.9125164391265214; spent CPU 522.8 s
* closure code drift: `"none: closed under the sealed code"`
* scope: DEVELOPMENT; paired seeds within host blocks; one previously inspected DEV validation week; no test rows read
* disposition: `{"disposition": "HISTORICAL_DEV_ONLY", "task_id": "uci_235.W60_h60.DEV_28d_7d", "policy": "docs/tres_temas_entrevista/program_v3/SOTA_FIRST_2026_09_21.md", "why": "previous exploratory pilot (household task / adapted models): preserved with its receipts and failures, excluded from active selection, ranking and recommendations"}`; active_selection: `null`
* `summary`, `paired` and `baselines` are all `None`: a failed closure emits no verified comparator, no paired contrast and no selected arm. The per-arm means in N4, the crossing in N5 and the references in N7 are therefore published OUTSIDE the closure, recomputed from the arrays.

**Every problem the closure recorded, in full:**

* daily_lag_s1: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s2: a registered forecast unit has NO accepted terminal receipt
* daily_lag_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_crop60_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_local_support_67_s3: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s1: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s2: a registered forecast unit has NO accepted terminal receipt
* long_window_own_depth_s3: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s1: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s2: a registered forecast unit has NO accepted terminal receipt
* modular_w60_s3: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s1: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s2: a registered forecast unit has NO accepted terminal receipt
* short_window_deep_core_s3: a registered forecast unit has NO accepted terminal receipt
* closure without a warehouse read: no accepted custody, nothing is verified

**Fresh-process replays: 12 of 12 pass `allclose(1e-6, 1e-6)`; the maximum absolute difference over every replayed cell is 0.000e+00 kW.**

| cell | fresh-process replay | max abs difference (kW) |
|---|---|---:|
| `modular_w60_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s1` | allclose(1e-6) PASS | 0.000e+00 |
| `modular_w60_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s2` | allclose(1e-6) PASS | 0.000e+00 |
| `modular_w60_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `daily_lag_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `long_window_crop60_s3` | allclose(1e-6) PASS | 0.000e+00 |
| `short_window_deep_core_s3` | allclose(1e-6) PASS | 0.000e+00 |

### N12. Governance, stated as it is

* classification **NON_GOVERNING**
* data_gov_acquisition: **ABSENT**
* accepted_terminal: **ABSENT**
* receipt: **ABSENT**
* warehouse_read: **ABSENT**
* why: **no data-gov service key is held on this host; the governed runner refuses before opening data, so this driver ran instead and its results promote nothing**
* every closure row's custody is UNCHECKED: no accepted payload anchors the score
* the preparation's custody is PREPARATION_LOCAL_ONLY
* the closure's `verified` flag is False for that reason and for no other unless it names one
* these numbers are DEVELOPMENT measurements and cannot select, rank or promote anything
* data custody `BYTES_IDENTITY_ONLY`, panel sha256 `b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db`, 10890295 bytes — the file's sha256 equals the design's source_run.panel_sha256, which a previous GOVERNED acquisition recorded as VERIFIED_TRANSFER; this driver re-verified the BYTES, not the transfer, and holds no delivery id, no availability contract and no acceptance
* interpreter: Python 3.12.13 (anaconda env `trading-stack`)

### N13. Placement: every memory reading taken before a launch

* `MEMORY_GATE.jsonl`: 81 readings — 13 launches, 55 holds, 13 finished jobs, 0 units NOT started because memory never allowed it
* MemAvailable at launch: min 11.24 GiB, max 13.06 GiB
* MemAvailable while held: min 5.89 GiB, max 13.14 GiB; longest single wait 1440 s

| label | verdict | MemAvailable GiB | measured pilot peak GiB | required (peak + margin) GiB | cap |
|---|---|---:|---:|---:|---|
| `pilot:pilot_long_window_own_depth` | HELD_WAITING_FOR_MEMORY | 10.20 | 7.88 | 8.88 | 9G |
| `pilot:pilot_long_window_own_depth` | LAUNCH | 13.06 | 7.88 | 8.88 | 9G |
| `pilot:pilot_long_window_own_depth` | FINISHED exit -9 | 5.50 (after) | — | — | — |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 5.89 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 13.13 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 13.14 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.38 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.59 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.58 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.58 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.56 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.53 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.52 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.45 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.38 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.35 | 9.57 | 10.57 | 11G |
| `pilot:pilot_long_window_local_support_67` | HELD_WAITING_FOR_MEMORY | 12.40 | 9.57 | 10.57 | 11G |
| `modular_w60_s1#attempt1` | LAUNCH | 12.16 | 0.94 | 1.94 | 4G |
| `modular_w60_s1#attempt1` | FINISHED exit 0 | 12.11 (after) | — | — | — |
| `daily_lag_s1#attempt1` | LAUNCH | 12.12 | 0.94 | 1.94 | 4G |
| `daily_lag_s1#attempt1` | FINISHED exit 0 | 11.93 (after) | — | — | — |
| `long_window_crop60_s1#attempt1` | LAUNCH | 11.93 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s1#attempt1` | FINISHED exit 0 | 11.99 (after) | — | — | — |
| `short_window_deep_core_s1#attempt1` | LAUNCH | 11.99 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s1#attempt1` | FINISHED exit 0 | 11.99 (after) | — | — | — |
| `modular_w60_s2#attempt1` | LAUNCH | 11.99 | 0.94 | 1.94 | 4G |
| `modular_w60_s2#attempt1` | FINISHED exit 0 | 12.01 (after) | — | — | — |
| `daily_lag_s2#attempt1` | LAUNCH | 12.02 | 0.94 | 1.94 | 4G |
| `daily_lag_s2#attempt1` | FINISHED exit 0 | 11.82 (after) | — | — | — |
| `long_window_crop60_s2#attempt1` | LAUNCH | 11.83 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s2#attempt1` | FINISHED exit 0 | 11.82 (after) | — | — | — |
| `short_window_deep_core_s2#attempt1` | LAUNCH | 11.82 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s2#attempt1` | FINISHED exit 0 | 11.71 (after) | — | — | — |
| `modular_w60_s3#attempt1` | LAUNCH | 11.72 | 0.94 | 1.94 | 4G |
| `modular_w60_s3#attempt1` | FINISHED exit 0 | 11.24 (after) | — | — | — |
| `daily_lag_s3#attempt1` | LAUNCH | 11.24 | 0.94 | 1.94 | 4G |
| `daily_lag_s3#attempt1` | FINISHED exit 0 | 11.37 (after) | — | — | — |
| `long_window_crop60_s3#attempt1` | LAUNCH | 11.37 | 0.94 | 1.94 | 4G |
| `long_window_crop60_s3#attempt1` | FINISHED exit 0 | 11.52 (after) | — | — | — |
| `short_window_deep_core_s3#attempt1` | LAUNCH | 11.52 | 0.94 | 1.94 | 4G |
| `short_window_deep_core_s3#attempt1` | FINISHED exit 0 | 12.13 (after) | — | — | — |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.13 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.28 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.32 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt2` | HELD_WAITING_FOR_MEMORY | 12.26 | 7.88 | 8.88 | 11G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.14 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.61 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.62 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.64 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.59 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.60 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.31 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.39 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.71 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.37 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.39 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.37 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.45 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.36 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.20 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.92 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.23 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.80 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.82 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 10.83 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.76 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.86 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.81 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.80 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.88 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.18 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.16 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.06 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.17 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.17 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.27 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 11.90 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt1` | HELD_WAITING_FOR_MEMORY | 12.16 | 7.88 | 8.88 | 10G |
| `long_window_own_depth_s1#attempt2` | HELD_WAITING_FOR_MEMORY | 12.21 | 7.88 | 8.88 | 10G |

