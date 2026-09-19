# 13B — Next real steps (RP6, corrected under RP15): E1 public bank, forecasting/RL demand, E3 protocol, lanes

> **Amendment (RP32).** Two corrections to what this document's E1/E3 lanes may claim. (1) The learning regimes
> R0 / R1 / R2 are the detector's regimes as [13E v2](13E_v2_E1_TASK_SHEET_AND_REGIMES_2026_09_19.md) defines them
> (random trainable / pre-trained frozen / pre-trained adjustable), not grouping arms. (2) The E3 lane's weekly
> cycle is covered, as software, by `tools/e3_weekly_controller.py` and its tests; data availability, late
> releases and real weekly operation stay UNVERIFIED, and E3 keeps every dependency the plan gives it.


Nothing here is executed; every row has an identifier, source, dependency and deliverable. This
version replaces the earlier table that merely listed JSON files (review F6): the demand is now
traced effective config → entry point → plugin → data, and each configuration is classified as
HISTORICAL_CANDIDATE (its data file exists), RESOLVABLE (every named plugin has a registered entry
point and an existing module) or EXECUTABLE_VERIFIED_BY_INSPECTION (data present, plugins resolve,
no missing key). Nothing is executed and no historical default becomes a business decision.

## RL demand traced (agent-multi + gym-fx, read-only)

Entry point `agent-multi = app.main:main` (agent-multi `setup.py:9`). The environment plugin
`gym_fx_env = env_plugins.gym_fx_env:Plugin` (`setup.py:14`) does not implement an environment: its
`make_env()` (`env_plugins/gym_fx_env.py:87`) imports `gym_fx.build_environment` and resolves six
gym-fx plugins by entry point (`data_feed`, `broker`, `strategy`, `preprocessor`, `reward`, `metrics`;
`_load_bundle_plugin`, line 80). The real environment is gym-fx `app/env.py` (`gym_fx/env.py` re-exports
`app.env.GymFxEnv`).

Common chain of every traced config: strategy `direct_atr_sltp` (gym-fx
`strategy_plugins/direct_atr_sltp.py`, defaults at line 57: `atr_period 14, k_sl 2.0, k_tp 3.0,
position_size 1.0, leverage 1.0, size_mode fx_units`); reward `pnl_reward`
(`reward_plugins/pnl_reward.py:29`: `(equity − prev_equity) / initial_cash × reward_scale`,
`initial_cash 10000`); broker defaults `commission 0.0, slippage_perc 0.0, leverage 1.0`
(`broker_plugins/default_broker.py:21-24`; legacy key `slippage` accepted at lines 38-45), so **costs are
zero unless the config sets them**; observation window 32 from `preprocessor_plugins/default_preprocessor.py:22`
(price window + equity/cash + unrealised PnL/cash); episodes: `len(df) ≥ window_size + 2`
(`app/env.py:174-178`), termination on data exhaustion / bankruptcy or `equity ≤ min_equity` under
`solvency_mode normal_realistic` (`app/env.py:474-479`), `truncated` always False — **no fixed episode
length and no weekly boundary exist in the environment**; timesteps come from `total_timesteps`
(`pipeline_plugins/rl_pipeline.py:36`, default 10 000) or `epoch_timesteps × max_epochs` in the validation
pipeline (`rl_pipeline_with_validation.py:1327-1329`).

| config (`agent-multi/examples/config/`) | agent → module | data (exists? size, span) | features | costs / sizing | class |
|---|---|---|---|---|---|
| `dqn_btc_1h_twelve_atr.json` | `dqn_agent` → `agent_plugins/dqn_agent.py` | `./data/project2/btcusdt_1h/d4.csv` — yes, 2.66 MB, 8 713 lines, 2019-01-01 20:00 → 2019-12-31 23:00 | `twelve` | commission 0.001, slippage 0, position_size 1.0, cash 10 000, 500k steps | HISTORICAL_CANDIDATE, RESOLVABLE, EXECUTABLE_VERIFIED_BY_INSPECTION (needs `gymnasium` + editable gym-fx) |
| `p4_ppo_btc_1h.json` | `ppo_agent` | same BTC d4.csv | `twelve` | 0.0002, position_size 0.01, rel_volume 0.05, `size_mode notional`, 2M steps | same |
| `sac_btc_1h_twelve_atr.json` | `sac_agent` | same BTC d4.csv | `twelve` | 0.0002, continuous actions, threshold 0.10, 500k steps | same |
| `p4_ppo_eurusd_1h.json` | `ppo_agent` | `./data/project2/eurusd_1h/d4.csv` — yes, 32.2 MB, 93 187 lines, 2005-01-03 21:00 → 2019-12-31 16:00 | `twelve_macro` | 0.0005, position_size 1000, leverage 10, `fx_units`, min/max 1 000 / 100 000 | same |
| `project3_ethusdt_4h_sac_train_val_test_v3.json` | `project3_sac_actor_critic_agent` | predictor `examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv` — yes, 17.6 MB, 18 086 lines, 2017-09-28 04:00 → 2025-12-31 20:00 | `tech_stat` + 84 explicit `feature_columns` | 0.0002, position_size 0.01, notional; pipeline `rl_pipeline_with_validation`, preprocessor `feature_window_preprocessor` | same; the only config with a session filter (`entry_hour_start 12`, `force_close_dow 4`, `force_close_hour 20`) |
| `ppo_gymfx_default.json` | `ppo_agent` | `../gym-fx/examples/data/eurusd_sample.csv` — yes, 28.8 KB, 501 lines, one day, OHLCV only | none | 0 / 0, `default_strategy`, `env_mode inference`, 10k steps | HISTORICAL_CANDIDATE, RESOLVABLE; a smoke config, not a trading experiment |

Facts that matter for the contract: `features_preset` is metadata only at runtime (copied into
summaries: `pipeline_plugins/rl_pipeline.py:432`, `agent_plugins/_progress_callback.py:100`,
`app/canonical_config.py:40`); the columns come from the offline `tools/prepare_project2_data.py`
(`_FEATURE_COLS_12` at line 68: returns, momentum_5/20, volatility_5/20, atr_norm, bb_pos, rsi_14, macd,
macd_signal, macd_hist, volume_ratio; `twelve_macro` merges SPY_close, VIX, US_10Y_Yield, DXY_Broad,
Fed_Funds_Rate, CPI, Unemployment with a backward `merge_asof` and a 7-day forward-fill cap, line 178;
`DXY_Broad` is empty in the first EURUSD row). `tech_stat` has no preset definition in code: its columns
are the config's explicit list and the CSV header (86 columns). Import check: agent, pipeline and
optimizer modules import; `env_plugins.gym_fx_env` and `gym_fx` fail in the current interpreter with
`ModuleNotFoundError: gymnasium` (an environment gap: gym-fx is not installed into site-packages).
Splits are year-count based (`rl_pipeline_with_validation.py:537-552`: train 4 / val 1 / test 1,
`split_anchor start`); the `d4/d5/d6` naming is fixed by `prepare_project2_data._split` from
`config_manifest.json`, not by any RL config.

What no config or code expresses (gaps, owner Satoshi + owner): a weekly training cutoff and release
instant (`week` appears only in `pipeline_plugins/_weekly_metrics.py:50-62`, a post-hoc slicing of a
continuous episode by `W-SUN`, and in `lexicographic_weekly_v1` selection), decision/execution hours
(one config only), funding/swap/financing (absent in both repositories), latency (absent; costs are
commission fraction + slippage fraction; an execution-cost curriculum wrapper exists,
`env_plugins/execution_cost_curriculum.py`, unused by these configs), capital beyond `initial_cash
10 000` (`initial_balance` does not exist), risk budget or portfolio constraints. These are written as
the E3 experiment contract in 13C, not adopted as production defaults. No asset is chosen by having a
CSV: the universe these configs run is BTC/ETH 1h, EURUSD 1h and ETH 4h, pending the owner's business
decision (12E row 1); that decision is the only item escalated.

## Forecasting demand traced (predictor)

Entry point `predictor = app.main:main` (`setup.py:10`).

* `examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json`: `predictor_plugin ann` →
  `predictor_plugins/predictor_plugin_ann.py`; `pipeline_plugin stl_pipeline` →
  `pipeline_plugins/stl_pipeline.py`; `preprocessor_plugin stl_preprocessor`; `optimizer_plugin
  default_optimizer`; no `target_plugin` key (the registered default applies). Data
  `examples/data_downsampled/phase_1/normalized_d{4,5,6}.csv` present (d4: 6 299 lines, header
  `DATE_TIME,typical_price`, 2012-10-16 20:00 → 2017-09-20 00:00 — hourly stamps despite the "daily"
  name). `window_size 33`, `predicted_horizons [9,12,15,18,21,24]`, `target_column typical_price`,
  `stl_period 24`, `use_stl false`. EXECUTABLE_VERIFIED_BY_INSPECTION (and executed in AGENTS.md).
* `examples/config/phase_1c_direction/inference/phase_1c_direction_lstm_direction_long_1d_inference_config.json`:
  `direction_lstm` → `predictor_plugins/direction/predictor_plugin_direction_lstm.py`; `direction_pipeline`;
  `direction_target`; `stl_preprocessor`. Data `examples/data_downsampled/phase_1_c/normalized_d{4,5,6}.csv`
  present (d4: 7 644 hourly lines, 2012-10-26 13:00 → 2017-09-19 21:00; 26 columns incl. ATR, RSI, MACD,
  ADX, stochastics, BB, hod/dow encodings, `direction_long_label`, `direction_short_label`,
  `bars_to_friday`). `window_size 72`, `predicted_horizons [1]`, `target_column ATR` (inconsistent with
  the direction labels in the data — recorded, not fixed here). RESOLVABLE; EXECUTABLE_VERIFIED_BY_INSPECTION
  with that inconsistency flagged.
* Absent: every `examples/config/phase_1/*_1h_config.json` points at `examples/data/phase_1/normalized_d4.csv`,
  which does not exist (only `base_d*.csv`); of the 137 configs with all inputs present, none has `1h`
  in its name.

## E1 public multivariate bank (selected from the existing census, no re-census; corrected under RP23)

**Correction (RP23, dictum F4).** `E1_FAMILIES.json` (v1, frozen as PRE) declared eligibility from date ranges with
W = 96 / h = 1 / ≥ 200 blocks: withdrawn as a task judgement. The task contracts are now built from the governed bytes
of the two DEV families in [`E1_TASKS.json`](E1_TASKS.json) (`tools/df_e1_tasks.py`): entities, units, timestamps,
DST facts, structural zeros vs measurements, missing values, column roles, splits by time with purge derived from
(W, h), usable windows per split AFTER masks (per target column, "any" and "all"), contexts in physical units with
their measured periodicities (train-only spectrum bands: daily / half-day / weekly / slower-than-five-weeks), and
the explicit note that neither non-overlapping blocks nor client columns are independent replicates. Eligibility is
separated: catalogue (licence) / task (contract) / reserve (not judged). Prior exposure of Beijing and appliances
(D1/D2 descriptors in the census) is declared; they stay closed. UCI is a repository, not a single physical source.


Deliverable produced: [`E1_FAMILIES.json`](E1_FAMILIES.json) by `tools/df_e1_families.py` over the
warehouse census (`df_dim_dataset`: 715 datasets = 198 FINANCIAL `INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE`,
513 SYNTHETIC generated, 4 PUBLIC `OPEN_ATTRIBUTION` CC-BY-4.0, all UCI) and the coverage ledger
(`df_fact_coverage_v2`). Eligible public families (p ≥ 6, ≥ 2 candidate targets, licence open, sampling
and range known, ≥ 200 context + horizon blocks): electricity load diagrams 2011-2014 (370 clients, 15 min,
4 years) and household electric power consumption (7, 1 min, 4 years) proposed as DEV; Beijing multi-site
air quality (144, hourly, 4 years) and appliances energy prediction (28, 10 min, 4.5 months) as RESERVE
candidates, not opened, not generated, not profiled here. No universal "20 cycles": the manifest reports
the cycles of a daily period declared from the sampling (to be confirmed by the profile spectrum) AND the
number of non-overlapping context + horizon blocks, with the rule that an aperiodic process is judged by
blocks plus the D1 stationarity/dependence diagnostic. **Source deficit recorded**: four public families
cannot supply independent development and reserve SOURCES beyond themselves; the remedy is engineering
(licence evidence for financial sources through the governed census, or ingestion of further public
multivariate datasets), not a selection trick.

## E1 design references (from the proposal)

References fixed before development results: naive, seasonal naive, a pertinent multivariate statistical
model (VAR), DLinear, PatchTST, iTransformer, DUET, and one of MTST/Pathformer/TimeMixer chosen before
development results. R0/R1/R2 compared in development; context and groups selected on validation only;
margin ε and the precision plan fixed before any reserve; K = F + 4 Bonferroni intervals with
hierarchical bootstrap. The extractor regime enters E1 from MOD-ARCH-COMPARE (RP14); H-CORE stays after
E1 and the verified frozen prefix (MOD-FROZEN-PREFIX), not before.

## E3 forecasting + RL (mandatory, not conditional on H1 or H3)

Weekly cycle train → inner validation → untouched next week; raw vs modular with the same information,
same agent (from the traced configs), same reward/costs/execution; deliverables: net return with explicit
capital and costs, drawdown, Sharpe with its periodicity, turnover, exposure, uncertainty across
weeks/regimes, observed steps and retraining time; trivial policies and a competent non-modular control;
forecasting and RL compared, RL never replaced by MASE. Contract in
[13C](13C_E3_WEEKLY_CYCLE_CONTRACT_2026_09_18.md). Depends on MOD-E1 and BUSINESS-CONTRACT. No live orders.

## Lanes (identifier, source, dependency, deliverable)

| id | source | depends on | next task | deliverable |
|---|---|---|---|---|
| PRE-NOISE | P-PRE proposal | D2 evidence by regime | reconcile D2 candidates by regime; design the uncovered receiver/task contrasts with causal noise controls | H1 degradation, H2 conditional denoising, H3 residual preservation |
| TRN-TRANSFER | P-TRN matrix | T0–T2, D3 | per-variable vs global transfer with adequate receivers; abstention curves | conditional utility, transfer/cost, abstention, DOIN parity |
| CAP-CALIBRATION | P-CAP | M3/M4 mapping | random-label capacity and N_min protocol on the declared MLP/GRU | C_mem, log-loss bits, N_min with censoring |
| L2-PUBLIC-RL | P-L2 | B4/selector evidence | map to the POPGym/CARL protocol before new RL code | cost/regret, risk-coverage, transfer |
| INC-SIMULATION | P-INC | existing simulator | HPO-B pilot and a second training domain | allocation, learning, manipulation utility |
