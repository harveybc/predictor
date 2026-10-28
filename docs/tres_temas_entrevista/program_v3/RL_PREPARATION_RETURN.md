# RL Preparation Return (Lane R)

Order: `SATOSHI_DATA_AND_RL_ORDER_2026_10_18.md` section R, under
`RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md`. Date: 2026-10-18. Author role:
Satoshi (technical lead). Scope: code discovery, bounded literature, gap
matrix, synthetic test specification, brainstorming hypotheses, PDT
applicability and a finite campaign manifest. No fit, no GPU, no deploy, no
DOIN edit, no service/config change, no restored result, no monitoring agent.

Repositories read (never edited): `gym-fx` at `e287ea2` (clean checkout),
`agent-multi` at `e4d01fa0` (41 modified/untracked files, all docs, one
example config, one social tool and two untracked run artifacts; none of
the runtime files cited below is dirty), `predictor` at `09ba0c7a` (base of
this branch). Line numbers refer to those revisions. Discovery used the
codebase-memory graph (`search_graph`, both repositories already indexed at
those revisions) and was confirmed by reading every cited file.

Bounded CPU checks actually run (`CUDA_VISIBLE_DEVICES=""`, `crispdm-run -q
-m 2G -t 120s -n rl-prep`, pinned `trading-stack` environment, no cache or
bytecode written, both checkouts byte-identical afterwards):

| Check | Result |
|---|---|
| gym-fx `test_execution_cost_context`, `test_continuous_action_threshold`, `test_trade_reconciliation_stream`, `test_session_exposure` | 67 passed, 1.9 s |
| gym-fx `test_session_exposure_env`, `test_shared_execution_envelope`, `test_env_foundations`, `test_fractional_position_sizing` | 162 passed, 8.4 s |
| agent-multi `test_stopping_contract`, `test_weekly_promotion`, `test_rl_pipeline_with_validation_splits`, `test_validation_pipeline_test_firewall`, `test_rolling_origin_adaptation`, `test_experiment_transition_queue` | 101 passed, 1.9 s |
| pinned stack | `stable_baselines3 2.9.0`, `gymnasium 1.3.0`, `torch 2.13.0`; `sb3_contrib` NOT installed |

These prove that the cited tests exist and pass at the stated revisions.
They prove nothing about trading performance.

## R1. Source symbols and call paths

Runtime path (agent-multi -> gym-fx), as executed, not as documented:

`pipeline_plugins/rl_pipeline_with_validation.PipelinePlugin.run_pipeline`
(:1191) -> `_make_split_env` (:895) -> `env_plugins/gym_fx_env.Plugin.make_env`
(:88, loads the six gym-fx bundle plugins by entry point and calls
`gym_fx.build_environment`) -> `app/env.GymFxEnv.__init__` (:122) ->
`GymFxEnv.reset` (:388) builds `BTBridge` (:38) and `build_cerebro` (:892) and
starts the cerebro thread -> each `GymFxEnv.step` (:430) hands the action to
`BTBridgeStrategy.next` (:562) and waits for `_publish_obs` (:839).

### R1.1 Symbol map

| Concern | Symbol (file:line) | Evidence that the runtime uses it |
|---|---|---|
| Observation space declaration | `gym-fx app/env.py:41 build_base_observation_space`; session/calendar/cost blocks appended in `GymFxEnv.__init__` (:122-386) | called from `__init__`; `tests/test_env_foundations.py::test_declared_space_contains_every_emitted_observation` |
| Observation assembly (per step) | `app/env.py:1736 GymFxEnv._make_observation` -> `preprocessor_plugins/feature_window_preprocessor.py:194 Plugin.make_observation` (feature window rows `[step-window, step)`, causal z-score over `[step-scaling_window, step)`) or `default_preprocessor.py:34` (raw price/returns window); session block `:1637 _session_observation` | `step` returns `_make_observation()`; `test_feature_window_preprocessor.py::test_no_future_leakage_in_scaling` |
| Observation contract (agent-multi) | `pipeline_plugins/_observation_contract.py apply_observation_contract`, `validate_observation_contract` (called at `rl_pipeline_with_validation.py:1205-1208`) | `tests/unit/test_observation_contract_failclosed.py` |
| Observation rebuild outside an episode | `gym-fx app/observation_builder.py:602 build_observation` | `tests/test_observation_builder.py` (32 tests) |
| Action decoding | `app/env.py:579 GymFxEnv._coerce_action` (continuous `Box(-1,1)` -> {0 hold, 1 long, 2 short, 3 close} by `continuous_action_threshold`/`continuous_exit_threshold`, contracts `legacy_directional_v1` / `target_exposure_hysteresis_v2`); discrete `Discrete(3)`; raw value kept in `bridge.raw_action_slot` | `step` :434-440; `tests/test_continuous_action_threshold.py` |
| Position sizing (continuous magnitude) | `app/bt_bridge.py:709 BTBridgeStrategy._apply_action` with `fractional_position_sizing` (size = position_size x min(1,abs(raw)), same-direction rebalance); envelope sizing `strategy_plugins/shared_execution_envelope.py:303 Plugin.apply_action` (units = equity x min(leverage_cap, abs(raw)) x (1-headroom) / close) | `tests/test_fractional_position_sizing.py`; `test_shared_execution_envelope.py::test_portfolio_fraction_sizing_and_scale_invariance`, `::test_leverage_cap_binds` |
| Execution | `app/bt_bridge.py:709 _apply_action` (default buy/sell/close; flip = close + open, two fills, two commissions), delegating to `strategy_plugin.apply_action` when present: `shared_execution_envelope.py:303` (atomic bracket, stop-first collision rule, entry-bar synthetic check, direct settlement `:230 _settle_position`), `direct_fixed_sltp.py:52`, `direct_atr_sltp.py`. Orders submitted in `next()` at bar t fill at bar t+1 open (backtrader default; no cheat-on-open/close anywhere in the repository) | `test_session_exposure_env.py::test_the_fill_is_delayed_by_one_bar`; `test_shared_execution_envelope.py::test_post_entry_bars_fill_at_level_not_open`, `::test_gap_through_stop_fills_at_open_not_stop` |
| Reward | `app/env.py:430 step` lines 452-466: `reward_plugin.compute_reward(prev_equity, new_equity)` on ECONOMIC equity (equity minus recapitalisation debt) minus `_force_close_reward_penalty` (:1940, Stage-B diagnostic, off by default). Plugins: `reward_plugins/pnl_reward.py:26` (delta equity / initial cash), `sharpe_reward.py:34` (rolling window, annualisation 252 by default), `dd_penalized_reward.py:30` | `tests/test_solvency_modes.py::test_conservation_and_no_recap_gain`; `tests/test_force_close_reward_penalty.py` |
| Commission | `broker_plugins/default_broker.py:35 Plugin.build_bt_broker` -> `BackBroker.setcommission(commission, leverage)` (fraction of notional per fill); accumulated in `bt_bridge.py:416 notify_order` (`bridge.commission_paid`); direct settlement charges `comminfo.getcommission` in `shared_execution_envelope.py:230` | `test_shared_execution_envelope.py::test_no_double_close_and_exact_cash_accounting` (hand formula: loss = units x 5 + commission x units x (100+95)), `::test_full_exposure_entry_fills_with_commission` |
| Spread | Not a bid/ask book. `app/env.py:522 set_execution_cost_context` folds `full_spread_rate/2 + slippage_bps/1e4` into one adverse displacement written to `config["slippage_perc"]`; `default_broker.py:48` -> `BackBroker.set_slippage_perc(slip_open, slip_limit, slip_match)`. Contract: `simulation_engines/contracts.py:45 ExecutionCostProfile.quote_adverse_rate_per_side` | `tests/test_execution_cost_context.py` asserts the CONFIG keys only; no test asserts the displaced fill price on the executing backtrader path |
| Slippage | same mechanism as spread (per-side bps) | same |
| Financing / swap | `ExecutionCostProfile.financing_enabled` (`contracts.py:33`) is validated and stored (`env.py:551`) but NO code on the backtrader path debits financing; only the optional Nautilus engine applies `FXRolloverInterestConfig` (`simulation_engines/nautilus_adapter.py:784`) and requires `financing_rate_data_file` (`nautilus_gym.py:336`). The committed `project3_pessimistic_v1` profile sets `financing_enabled: true` and is silently cost-free on the default engine | `simulation_engines/bakeoff.py:179 build_financing_fixture` (Nautilus bakeoff only) |
| Forced-close cost | forced close is the ordinary close path (`_apply_action` action 3 or `force_flat_request`, `bt_bridge.py:715-729`; envelope `apply_action` action 3) so commission and displacement apply; `app/env.py:1968 GymFxEnv.flatten_step` for operator/handover liquidation | `test_session_exposure_env.py::test_forced_close_submits_the_close_command`, `::test_the_final_state_is_exactly_flat_with_one_close_event` |
| Margin / liquidation | `default_broker.py` `leverage` -> backtrader margin; ruin = `bt_bridge.py:888 _is_broke` (equity <= `min_equity`, default 1 % of initial cash) -> `termination_cause="min_equity"` or train-only recapitalisation `:664 _continue_after_would_margin_call`. `margin_closeout_percent` observation (`env.py:1916`) reads a bridge attribute nothing sets (always 0.0). `latency_ms`, `enforce_margin_preflight`, `intrabar_collision_policy`, `limit_fill_policy` from the profile are not referenced by `app/` or `strategy_plugins/` | `tests/test_solvency_modes.py` |
| Forced flatten / reopen / entry cutoff | `app/session_exposure.py:163 validate_policy` (`wind_down_hours`, `forced_flatten_hours`, `reopen_min_hours`, `reopen_min_closed_bars`, `stability_consecutive_checks`, `release_probation_factor`...; all required, none defaulted); `:599 session_state` (five states, precedence FORCED_FLATTEN > WIND_DOWN > REOPEN_BLACKOUT); `:762 overlay_action` (masks risk-increasing commands in WIND_DOWN/REOPEN_BLACKOUT, CLOSE in FORCED_FLATTEN, cancels pending entries); env side `app/env.py:1126 _apply_session_exposure_overlay` runs BEFORE the bridge sees the action for EVERY agent; cancellation executed in `bt_bridge.py:615 _process_cancel_requests`; confirmation only post-fill `env.py:1513 _session_post_fill_reconciliation`; durable obligation `app/flatten_custody.py` | `tests/test_session_exposure.py` (41), `tests/test_session_exposure_env.py` (104), `tests/test_flatten_custody.py` |
| Calendar / DST / holidays | session calendar = explicit UTC closure intervals bound to venue/account/symbol/digest (`session_exposure.py:258 SessionCalendar`), `holiday_policy` only `same_as_weekly` (holidays must appear as intervals). A tradable bar inside a declared closure raises `SessionDataContradictionError` (`env.py:1178`). The separate `app/oanda_calendar.py` (NY zoneinfo, Friday 14:00/15:00/15:45 windows) is OBSERVATION/INFO ONLY (`env.py:1872 _oanda_calendar_features`); it never blocks an order | `test_session_exposure_env.py::test_a_fabricated_bar_inside_a_closure_is_refused`; `test_env_foundations.py::test_dst_correctness_survives_the_index_layout`; `test_oanda_calendar.py` |
| Weekly warm start (weights) | `agent_plugins/sac_agent.py:373 Plugin.load_for_training` (policy tensors only, fresh optimiser, empty replay, tensor-hash evidence `:41 _policy_tensor_hash`), `:468 load_with_observation_expansion`; consumed at `rl_pipeline_with_validation.py:1258-1305` (`warm_start_model` + sha256) and `rl_pipeline_with_solvency_curriculum.py:960-985` | `tests/unit/test_sac_normal_finetune.py`, `tests/unit/test_sac_warm_start_expansion.py`, `test_rl_pipeline_with_validation_splits.py::test_warm_start_artifact_hash_fails_closed` |
| Weekly prospective loop | (a) `app/weekly_promotion.py:94 build_week_windows` / `:194 build_week_config` / `:275 weekly_result_from_pipeline` driven by `examples/scripts/run_phase_1_weekly_promotion.py`: each week RETRAINS from `train_start` (no `warm_start_model` is set in `build_week_config`) and scores the following Monday-aligned week; (b) `tools/rolling_origin_adaptation.py:154 score_interval`, `:486 execute_handover`, `:590 run`: prequential test-then-train with per-origin immutable checkpoints, unscored warm-up, cadence in bars, account continuity within a block. (b) is hard-wired to the ETH 4h dataset (module constants `DATA_FILE`, `BARS_PER_DAY = 6`, `ALLOWED_CADENCES`, an absolute data path) | (a) `tests/unit/test_weekly_promotion.py`; (b) `tests/test_rolling_origin_adaptation.py` (31) |
| Checkpoint monitoring (L1) | `rl_pipeline_with_validation.py:1525` epoch loop: `_early_stop_composite` (:446, train_tail + inner validation, trade gate, gap penalty), `_checkpoint_is_eligible` (:304, after `learning_starts`), `_update_l1_checkpoint_state` (:284), activity stop (:322), best checkpoint saved (:1664) and RELOADED before final evaluation (:1985), terminal checkpoint saved separately (:1980). Declared-vs-effective terminators: `app/stopping_contract.py:80 effective_stopping_rules`, `:178 assert_declared_stopping_contract` | `tests/unit/test_rl_pipeline_with_validation_splits.py`, `tests/unit/test_stopping_contract.py` |
| Test firewall | `_final_eval` (:2037) evaluates `paths["test"]` only when `evaluate_test_split` (plugin default `True`, :586); nested manifests seal `sealed_test` in every development mode (`pipeline_plugins/_nested_splits.py`, `tests/test_nested_splits.py::TestSealedTest`); return traces refuse rows >= 2025-01-01 without dual authorisation (`_return_trace.py:33 HELDOUT_BOUNDARY`) | `tests/unit/test_validation_pipeline_test_firewall.py`, `tests/unit/test_return_trace.py::test_heldout_rows_rejected_without_stage_c_authorization` |
| Scoring rollout | `rl_pipeline_with_validation.py:1056 _rollout`: deterministic `predict`, refuses replay-buffer growth during evaluation, excludes causal context prefix from the score, emits trace rows via `_return_trace.py:305 build_trace_row`; fresh env per evaluation (`:922 _eval_on_split`) | `tests/unit/test_nested_context_prefix_selection.py::TestRolloutRefusals::test_replay_writes_during_evaluation_are_refused` |
| Score aggregation | `pipeline_plugins/_weekly_metrics.py:21 canonical_weekly_metrics_from_trace` (W-SUN weeks, max drawdown on the equity path, `annual_* = 52 x weekly mean`); `app/weekly_promotion.py:369 aggregate_weekly_results` (compounded weekly returns, `:321 _annual_drawdown_from_traces` on concatenated traces); gym-fx `metrics_plugins/default_metrics.py:22` (backtrader analyzers: Sharpe with `timeframe=Days`, drawdown, won/lost) and `trading_metrics.py:23` (RAP = return - lambda x drawdown; `_finite_or_zero` coerces undefined values to 0.0) | `tests/unit/test_weekly_promotion.py::test_aggregate_weekly_results_uses_concatenated_equity_traces`; gym-fx `tests/test_trading_metrics.py` |
| Outbox / warehouse publication | predictor: `olap/outbox.py` + `tools/olap_loader.py:40 drain_once` (OLAP outbox -> PostgreSQL cube, idempotent, receipts); governed path `tools/governed_run.py` (data-gov `TerminalOutbox`); warehouse contracts for feature selection `tools/fs_phase23_warehouse.py open_warehouse`, `olap/store/src/predictor_olap_store/*`. agent-multi: `tools/governed_run.py` delegates to the same data-gov `consumer_main`; `app/weekly_promotion.py:405 init_weekly_olap` (local sqlite). No RL metric table/contract exists in the warehouse package | predictor `tests/test_governed_outbox_disposition.py`, `test_governed_outbox_race.py`; agent-multi `tests/test_governed_outbox_recovery.py` (skips without a data-gov checkout) |
| Campaign successors | `tools/experiment_transition_queue.py:312 record_terminal_result`, `:346 approve_successor`, `:502 claim_dispatch` (leased, fail-closed, one chain), `:901 reconstruct_transitions`; `tools/quarantine_inner_curriculum_successor.py:327 inspect_consumers`; DOIN job advance `app/campaign_supervisor.py:2966 CampaignSupervisor._advance_job`, `:2627 _archive_champion` | `tests/test_experiment_transition_queue.py`, `tests/test_successor_quarantine.py`, `tests/integration/test_campaign_supervisor_swarm.py` |
| Baseline agents | `agent_plugins/no_trade_agent.py` (always 0), `random_agent.py` (uniform over the env action space), `buy_hold_agent.py` (always 1), `momentum_agent.py`, `reversal_agent.py`; all pass through the same env overlays | `rl_pipeline_with_validation.py:1312-1326` deterministic-baseline branch |
| Algorithms | `agent_plugins/sac_agent.py:190 build` (SB3 `SAC`, `MlpPolicy`, env flattened by `:552 _flatten_if_dict`, requires `Box`), `dqn_agent.py:67` (SB3 `DQN`, `MlpPolicy`, flattened), `ppo_agent.py:76` (SB3 `PPO`, `MultiInputPolicy` on the Dict space). No `features_extractor_class`, no `sb3_contrib` import anywhere in agent-multi | `tests/unit/test_sac_*`, `tests/unit/test_policy_observation_and_action_contract.py` |

### R1.2 Business requirements of RL_FIRST section 3, status

Status legend: IMPLEMENTED_AND_TESTED (cites the test), IMPLEMENTED_UNTESTED,
MISSING, UNKNOWN (not readable from the files in scope).

| # | Requirement (section 3) | Status | Basis |
|---|---|---|---|
| B1 | EURUSD first; ETH under a separate 24x7 contract | MISSING (configuration) | No EURUSD H1 RL config exists in agent-multi; committed contracts and `rolling_origin_adaptation.py` are ETH 4h. The env is instrument-agnostic (`session_symbol`, calendar intervals are config). |
| B2 | One concurrent position; explicit BUY/SELL/FLAT | IMPLEMENTED_AND_TESTED | `DISCRETE_COMMANDS` (`session_exposure.py:44`), `_apply_action` flip = close+open, envelope `reversal_close`; `test_protected_order_execution.py::test_opposite_signal_cancels_protection_and_closes_before_reversal`, `test_shared_execution_envelope.py::test_reversal_closes_then_reenters_with_new_envelope`. Caveat: a flip keeps two orders in flight for one bar. |
| B3a | Versioned source/broker calendar, timezone, DST, holidays | IMPLEMENTED_AND_TESTED (mechanism) / MISSING (EURUSD artifact) | Calendar = bound UTC intervals with digest identity (`SessionCalendar`); holidays only as explicit intervals; DST is the generator's responsibility. No EURUSD 2020-2025 interval file exists in either checkout. `test_session_exposure.py::test_intervals_must_be_ordered_utc_nonoverlapping`, `::test_holiday_adjacency_uses_same_machine`. |
| B3b | Force flat 12 h before close; no reopen until 12 h after open; configured cutoff | IMPLEMENTED_AND_TESTED (mechanism) / MISSING (values) | `forced_flatten_hours`, `wind_down_hours` (entry cutoff), `reopen_min_hours` are required policy fields; nothing in code derives them from a weekday name. No committed policy sets 12/12 (tests use 36/4/8 and 8/4/8). `test_session_exposure_env.py::test_masked_entry_from_flat_submits_hold`, `::test_forced_close_submits_the_close_command`, `::test_blackout_exits_exactly_when_evidence_accumulates`. Note `forced_flatten_hours < wind_down_hours` is enforced, so "flat 12 h before close" implies a wind-down strictly longer than 12 h. |
| B4 | Prior-week history allowed; closed hours add no invented observations; open-session gaps not adjacent ticks | IMPLEMENTED_AND_TESTED (closures) / MISSING (mid-session gaps) | Fabricated bars inside a closure refuse (`test_a_fabricated_bar_inside_a_closure_is_refused`); the feature window reads whatever rows the CSV has, so a missing open-session hour becomes adjacent rows with no refusal. `_session_stability_check` detects `quote_discontinuity` only during reopen checks. |
| B5 | Env enforces rules for every agent incl. random controls | IMPLEMENTED_AND_TESTED (session overlay) / MISSING (continuous enlargement) | Overlay runs in `step` before the bridge for any caller. Gap: `_session_exposure_facts` (`env.py:949`) hard-codes `action_mapping="discrete_command_v1"`, so with `fractional_position_sizing` or envelope sizing a same-direction size INCREASE during WIND_DOWN classifies as `hold` (not risk-increasing) and reaches the broker. Documented by `test_session_exposure_env.py::test_enlargement_and_reduction_are_not_expressible`. |
| B6a | Input for decision t available by t incl. latest completed bar | IMPLEMENTED_UNTESTED (depends on lane D) | Window `[step-window, step)` with `step = len(data)` includes the bar whose close the strategy sees in `next()`; the fill is at the next bar's open. Whether the CSV timestamp labels bar open or bar close is exactly lane D's unresolved clock question; until it is resolved the causal claim cannot be certified. |
| B6b | Execution at an explicitly available quote with declared latency | IMPLEMENTED_UNTESTED | Next-bar open fill is implicit one-bar latency; `latency_ms` from the profile is unused on the backtrader path; a declared-latency test does not exist. |
| B6c | No intrabar TP/SL priority without an ambiguity rule | IMPLEMENTED_AND_TESTED (envelope) / UNKNOWN (direct_fixed_sltp, direct_atr_sltp use backtrader bracket semantics without a declared rule) | `collision_rule=stop_first_pessimistic`; `test_same_bar_sl_and_tp_collision_resolves_to_stop`, `test_entry_bar_touches_both_resolves_to_sl`. |
| B7a | Equity, margin, notional sizing, leverage, liquidation | IMPLEMENTED_UNTESTED (leverage, min-equity ruin) | `leverage` reaches backtrader; ruin at `min_equity`; no margin-closeout model (observation field constant 0.0). `test_solvency_modes.py` covers ruin semantics, not margin arithmetic. |
| B7b | Commission | IMPLEMENTED_AND_TESTED | see R1.1 commission row. |
| B7c | Spread, slippage | IMPLEMENTED_UNTESTED | Symmetric adverse displacement, labelled as such (`quote_adverse_rate_per_side`, "Synthetic quote displacement from mid used for OHLC-only inputs"). No fill-price assertion on the executing path; no measured bid/ask. `slippage_paid` is read by `build_trace_row` but never emitted by the env, so the cost breakdown cannot separate displacement from price movement. |
| B7d | Financing | MISSING (backtrader path) | flag accepted, never debited; see R1.1. |
| B7e | Forced-close costs | IMPLEMENTED_AND_TESTED | same close path; see R1.1. |
| B7f | Assumed costs labelled and sensitivity-tested | IMPLEMENTED_AND_TESTED (curriculum) | `env_plugins/execution_cost_curriculum.ExecutionCostCurriculumWrapper` + `pipeline_plugins/_execution_curriculum.py`; `tests/unit/test_execution_cost_curriculum_wrapper.py`, `tests/unit/test_execution_curriculum*.py`. |
| B8 | Initial capital USD 10,000 explicit | IMPLEMENTED_AND_TESTED | `initial_cash` default 10000.0 in `GymFxEnv.__init__` and `gym_fx_env.Plugin.plugin_params`; explicit key, overridable. |
| B9 | WEEKLY warm-start updates as business default; other cadences separate experiments | MISSING (EURUSD weekly) | Weight transfer exists and is tested; the weekly promotion loop retrains from scratch per week; the prequential runner is ETH-4h-specific. A weekly EURUSD H1 prequential runner that reuses `load_for_training` + `score_interval` semantics is not implemented. |
| B10 | TRAIN 2020-2023, VALIDATION 2024, TEST 2025 declared; 2025 not pristine | IMPLEMENTED_UNTESTED (declaration only) | `_return_trace.HELDOUT_BOUNDARY = "2025-01-01"` and nested `sealed_test` match the ranges; no EURUSD nested manifest exists; the "already examined" status of 2025 must be written into the RL contract explicitly (nothing in code records it). |
| B11 | PDT state in private configuration | UNKNOWN | see R6; `oanda_calendar.py` cites a FINRA/OANDA policy memo and hard-codes NY windows; no account-rule state object exists in gym-fx or agent-multi. |

Counts over the 17 rows above (a row with a split status is counted once
under each status that applies): IMPLEMENTED_AND_TESTED 10 (B2, B3a, B3b,
B4, B5, B6c, B7b, B7e, B7f, B8), IMPLEMENTED_UNTESTED 5 (B6a, B6b, B7a, B7c,
B10), MISSING 7 (B1, B3a artifact, B3b values, B4 mid-session gaps, B5
enlargement, B7d, B9), UNKNOWN 2 (B6c non-envelope plugins, B11).

### R1.3 Defects found while reading (no fix applied)

1. `financing_enabled: true` in `project3_pessimistic_v1.json` has no effect
   on the default engine (silent zero financing).
2. `trading_metrics._finite_or_zero` and `_early_stop_composite`
   (`NaN -> 0.0`) violate "undefined ratios stay undefined".
3. Session overlay cannot see enlargements under continuous sizing (B5).
4. `GymFxEnv.step` computes the reward from `bridge.prev_equity/equity`
   published by `_publish_obs`, which runs AFTER `_apply_action` but BEFORE
   the submitted order fills (next bar); the reward of step t therefore
   reflects the fill of the order submitted at t-1. Consistent, but every
   per-step attribution (traces, `trade_cost`) is one bar behind the action.
5. `rl_pipeline_with_validation.plugin_params["evaluate_test_split"] = True`:
   a legacy (non-nested) config opens the test split at final evaluation
   unless it sets the key to `False`.
6. `tools/rolling_origin_adaptation.py` hard-codes an absolute data path and
   ETH constants; not reusable for EURUSD without a parameterised successor.

## R2. Bounded literature (12 papers + official docs), retrieved 2026-10-18

Official implementation docs (SB3 2.9.0 is the pinned version; SB3-Contrib
is NOT installed in the pinned environment, so any contrib algorithm needs
an environment change that this lane did not make):

- SB3 RL tips, https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html — continuous SOTA listed as SAC, TD3, CrossQ, TQC; discrete: DQN family; rescale actions to [-1,1]; separate evaluation env, `deterministic=True`, `EvalCallback`; `check_env`; provide observation history for delayed observations (no recurrent advice).
- SB3-Contrib index and module pages, https://sb3-contrib.readthedocs.io/en/master/ (`modules/tqc.html`, `crossq.html`, `qrdqn.html`, `ppo_mask.html`, `ppo_recurrent.html`) — TQC: Box only, `top_quantiles_to_drop_per_net`, `n_quantiles` via `policy_kwargs`; CrossQ: Box only, based on SB3 SAC, no recurrent/CNN, default critic `[1024,1024]`; QR-DQN: Discrete only, `n_quantiles` via `policy_kwargs`; MaskablePPO: Discrete/MultiDiscrete/MultiBinary, masks via `action_masks()` or `ActionMasker`, `MaskableEvalCallback`, `check_env` ignores masks, no recurrence; RecurrentPPO: Box and Discrete, `MlpLstmPolicy`/`MultiInputLstmPolicy`, `lstm_states` + `episode_start` at predict, `load()` rebuilds.

| # | Source (URL) | What it is | Relevance to hourly EURUSD business | Mismatch stated |
|---|---|---|---|---|
| 1 | Haarnoja et al. 2018, SAC, https://arxiv.org/abs/1801.01290 | off-policy max-entropy actor-critic, continuous actions | the existing comparator | continuous-control benchmarks; no finance, no costs, no nonstationarity |
| 2 | Schulman et al. 2017, PPO, https://arxiv.org/abs/1707.06347 | on-policy clipped surrogate, discrete and continuous | basis of MaskablePPO/RecurrentPPO | locomotion/Atari; no finance |
| 3 | Mnih et al. 2013, DQN, https://arxiv.org/abs/1312.5602 | Q-learning with replay, discrete actions | the discrete comparator | Atari; no finance |
| 4 | Dabney et al. 2017, QR-DQN, https://arxiv.org/abs/1710.10044 | distributional Q via quantile regression | discrete alternative with return-distribution output (risk-aware selection possible) | Atari; no finance |
| 5 | Kuznetsov et al. 2020, TQC, https://arxiv.org/abs/2005.04269 | truncated quantile critics, overestimation control | continuous challenger; drop-in for SAC configs (Box) | continuous-control suite; trading relevance undemonstrated |
| 6 | Bhatt et al., CrossQ, ICLR 2024, https://arxiv.org/abs/1902.05605 | batch-norm critic, no target nets, UTD 1 | cheaper continuous challenger; Box only | continuous control; no finance |
| 7 | Huang and Ontanon 2020, invalid action masking, https://arxiv.org/abs/2006.14171 | theory and evidence for masking in policy gradients | justifies masking hold/open/close by session state instead of post-hoc overlay for the discrete arm | games; masks must come from the env calendar, which our overlay already computes |
| 8 | Ishikawa and Nakata 2021, online FX RL with costs, https://arxiv.org/abs/2106.03035 (HTML full text read) | DQN+LSTM, USD/JPY 1-minute 2020, actions {-1,0,1}, constant cost c per position change (0.01-0.1 %), two networks with online copy, baselines LSTM/LightGBM | directly about FX, transaction costs and online adaptation; positive return only at <= 0.05 % cost | one pair, one half-year, minute bars, no walk-forward, no code/data; table inconsistency (245 vs 145 trades) |
| 9 | Zhang, Zohren, Roberts 2019, DRL for trading, https://arxiv.org/abs/1911.10107 (PDF text read) | DQN/PG/A2C, discrete {-1,0,1} and continuous [-1,1], 50 continuous futures 2005-2019, retrain every 5 years, cost `bp` x price x abs(position change), volatility targeting | gives the cost term form and discrete-vs-continuous action comparison we need | DAILY futures, not hourly FX; 5-year refit, no weekly adaptation; data proprietary (Pinnacle), no code |
| 10 | Theate and Ernst 2020/2021, TDQN, https://arxiv.org/abs/2004.06627 (PDF text read) | DQN variant, daily OHLCV, 30 stocks, actions {Long, Short}, 0.1 % cost, train 2012-2017 / test 2018-2019, baselines B&H, TF, MR; code public (GitHub, link in paper) | rigorous benchmark set and cost handling; a published open implementation of a discrete trading DQN | DAILY equities, long/short only (no flat), no session calendar; not our business |
| 11 | Jiang, Xu, Liang 2017, portfolio DRL, https://arxiv.org/abs/1706.10059 | EIIE, 30-minute crypto, 0.25 % commission, online stochastic batch learning | one of the few intraday online-learning designs with costs | crypto 24x7 portfolio weights; no closures; results period short |
| 12 | Padakandla 2020/2021, RL in dynamically varying environments, https://arxiv.org/abs/2005.10619 | survey of nonstationary RL (change detection, context, continual) | frames the weekly-refit and regime hypotheses as a known problem class | no finance domain discussed |

Explicit searches: "transaction costs" (8, 9, 10, 11 model them; 1-7, 12 do
not), "nonstationarity / online adaptation" (8, 11, 12), "causal execution"
(none of the papers states a decision-to-fill latency rule; 9 fills at next
close, 8 at next minute; our next-open rule is stricter than both). No paper
in this set uses an order book. Benchmark superiority in 1-7 is not a trading
claim; profits in 8-11 are not evidence for our configuration.

## R3. Candidate table

Business question for all rows: does a policy that trades at most one EURUSD
position on hourly bars, under the weekly flat rule and declared costs, beat
NO_TRADE and the passive restricted BUY/SELL controls on net PnL, drawdown and
risk-adjusted return over prospective weeks? Data/time requirements for all
rows: admitted EURUSD H1 2020-2025 with bound calendar intervals, feature
columns from accepted selection, TRAIN 2020-2023 / VAL 2024 / TEST 2025
(declared non-pristine). Evaluation protocol for all rows: RL1 controls on
the same support -> L1 checkpoint selection on train_tail + 2024 -> weekly
prospective loop (predict, seal, mature, score, then update) -> cost
perturbation -> DEMO/PAPER.

| Candidate | Action type | Policy representation | Refit / update semantics | Costs | Public code / data | Exact difference from our business | Minimum useful experiment |
|---|---|---|---|---|---|---|---|
| SAC compact control (existing `sac_agent`) — FIRST COMPARATOR | continuous `Box(-1,1)`; genuine control = bounded target exposure with `fractional_position_sizing` or envelope `portfolio_fraction`, one net position | SB3 `MlpPolicy` on the flattened Dict (feature window x columns + agent state) | weekly `load_for_training` (tensors only, fresh optimiser, empty replay) | env commission + displacement; financing MISSING | SB3 (code), our data | SB3 paper is MuJoCo; our sizing contract must close the B5 enlargement gap first | 1 seed x 1 cost profile x RL1 controls, 2024 L1 selection, 8 prospective weeks of 2024 as development weeks; report net PnL/DD/trades vs NO_TRADE |
| SAC + modular temporal representation (`predictor_plugins/modular_temporal`: causal_conv1d / causal_dense_sequence branches, transformer_conv core) | same as above | requires an SB3 `features_extractor_class` adapter wrapping the modular bundle (Keras) or an exported PyTorch/ONNX equivalent — NOT implemented; must not be replaced by an MLP silently | same weekly scheme; extractor weights either frozen (pre-fit on TRAIN under fold-causal rules) or trained jointly | same | our code only | representation reuse across frameworks (Keras -> SB3/torch) is new engineering | same manifest as the compact control, same seeds, same weeks, reporting the pair side by side; a budget/latency measurement of the extractor on CPU first |
| DQN (existing `dqn_agent`) | separately declared `Discrete(4)` {hold, long, short, close} (today the env exposes `Discrete(3)` without explicit close; close is only reachable through overlays) | SB3 `MlpPolicy`, flattened | weekly warm start via `DQN.load` (no tensor-only loader exists; must be added like SAC's) | same | SB3; Theate and Ernst code for reference | fixed size only (no exposure magnitude); differs in expressivity from SAC, must be stated | after SAC compact: same support, same weeks, `Discrete(4)` env change + test |
| QR-DQN (sb3-contrib) | same discrete space | quantile head; CVaR-style selection possible | as DQN | same | sb3-contrib (not installed) | Atari paper | only after DQN result and after contrib install is approved |
| MaskablePPO (sb3-contrib) | discrete with masks from `session_state` (WIND_DOWN/BLACKOUT mask entries; FORCED_FLATTEN masks everything but close) | `MlpPolicy`/`MultiInputPolicy` | on-policy; weekly continuation of `learn` from saved weights | same | sb3-contrib (not installed) | masking moves the rule into the policy's support; the env overlay must still enforce (defence in depth) | after the discrete interface exists; compare against DQN under identical masks |
| RecurrentPPO (sb3-contrib) | Box or Discrete | `MultiInputLstmPolicy` (history inside the policy instead of the window) | on-policy; LSTM state reset per episode (weekly episodes) | same | sb3-contrib | partial-observability framing; `load()` rebuilds | only if a window-based policy shows regime sensitivity in RL4 |
| TQC or CrossQ (sb3-contrib) | Box (same as SAC) | TQC: quantile critics; CrossQ: BN critic, no target | same as SAC (`load_for_training` needs a sibling loader per class) | same | sb3-contrib (not installed; compatibility with SB3 2.9.0 unverified here) | continuous-control papers | one of the two, only after the first SAC result; same manifest, same seeds |
| Rule controls (existing `no_trade`, `buy_hold` restricted by the overlay, `random`, `momentum`, `reversal`) | discrete | none | none | same | ours | — | RL1: run first; they define the floor and the random negative control |

## R4. Smallest synthetic CPU tests for RL00-RL10

Principle: episodic policy fitting and checkpoint selection (L1 loop on
TRAIN/train_tail/2024) are one population; scored weekly prospective
interaction is another. No test may stop training on a TEST quantity, and no
test may feed a week's rewards into a decision already scored for that week.
All tests run on synthetic CSVs under `tmp_path`, CPU only, seconds each.
"Helper" names existing fixtures to reuse.

| ID | Test (file to create) | Synthetic input | Assertion | Existing helper to reuse | Status today |
|---|---|---|---|---|---|
| RL00 | `tests/test_rl00_admitted_inventory_manifest.py` (predictor) | fake admitted inventory JSON + monthly support table + selection manifest with sha256 | a manifest whose input hash differs from the admitted inventory is refused; monthly support denominators include missing months | predictor `tools/fs_phase23_warehouse.contract` identity idiom; agent-multi `_nested_splits` manifest verification (`tests/test_nested_splits.py::TestManifestFailClosed`) | MISSING (no EURUSD RL manifest) |
| RL01 | gym-fx `tests/test_rl01_future_perturbation.py` | 48 hourly bars, one feature column; run the env twice with identical actions, second run with all rows > t multiplied by 3 | observations, coerced actions, fills and rewards up to and including the bar that fills the t-submitted order are byte-identical; the fill price equals the t+1 open displaced by the declared adverse rate | `test_session_exposure_env.py::test_no_future_bar_influences_a_check` (pattern), `test_shared_execution_envelope._env` | MISSING for the full env path; partial for the stability check |
| RL02 | gym-fx `tests/test_rl02_round_trip_accounting.py` | constant-price series with one long round trip, one short round trip, one flat stretch; commission 0.001, displacement 0.0002, leverage 1 | hand-computed end equity = initial - 2 x commission x notional - 2 x displacement x notional per round trip; `commission_paid`, `trade_cost` and `closed_trade_stream` costs sum to the same number; flat stretch changes nothing; financing debit asserted (will FAIL until financing exists, encode as `xfail(strict=True)` with the defect id) | `test_no_double_close_and_exact_cash_accounting`, `test_trade_reconciliation_stream.drive` | PARTIAL (long round trip only; no displacement, no financing) |
| RL03 | gym-fx `tests/test_rl03_weekly_rule_all_agents.py` | two synthetic weeks with a real timestamp gap for the closure, one DST transition (interval generated in UTC from an America/New_York local time), one holiday interval; policy 12/>12/12 | for each driver in {random, always-long, always-short, enlarging continuous} the position is flat >= 12 h before every close, no entry < 12 h after every reopen, no fill carries a weekend timestamp; the enlarging driver's wind-down size increase is masked (will FAIL until B5 is closed; `xfail(strict=True)`) | `test_session_exposure_env._env`, `_drive`; `test_env_foundations.py::test_dst_correctness_survives_the_index_layout` | PARTIAL (long driver only; no 12/12 policy; enlargement gap) |
| RL04 | gym-fx `tests/test_rl04_ambiguity_gaps_resets.py` | a bar touching SL and TP, a gap through the stop, a missing open-session hour, a terminal bar with an open position, two resets | collision -> stop; gap -> fill at open; the missing hour raises a typed refusal (will FAIL today; `xfail(strict=True)`); terminal open position recorded as data_end liquidation, not silently dropped; second reset has new episode identity and an empty replay-equivalent state | `test_same_bar_sl_and_tp_collision_resolves_to_stop`, `test_gap_through_stop_fills_at_open_not_stop`, `test_final_bar_open_position_is_data_end_case`, `test_episode_reset_advances_identity_scope` | PARTIAL |
| RL05 | agent-multi `tests/unit/test_rl05_frozen_weights_and_order.py` | FakeAgent with hashable weights, 3 synthetic weeks | per week: policy tensor hash before scoring == hash after scoring; the checkpoint used for week k is written before week k's rewards exist; updating after week k changes the hash used for week k+1 only; rewriting week k+1 data leaves week k's checkpoint and score bytes identical | `tests/test_rolling_origin_adaptation.py` (immutable per-origin checkpoints, `score_interval`), `_rollout` replay-growth refusal, `sac_agent._policy_tensor_hash` | PARTIAL (ETH runner only) |
| RL06 | agent-multi `tests/unit/test_rl06_baseline_parity.py` | one synthetic month; run no_trade, restricted buy, restricted sell, random | all four runs share identical data hash, initial cash, calendar digest, cost profile fingerprint and exposure cap; flat periods appear in the equity series; Sharpe of a zero-variance series is reported as undefined (None), not 0.0 (will FAIL on `_finite_or_zero`; `xfail(strict=True)`) | `rl_pipeline_with_validation` deterministic-baseline branch; `_weekly_metrics.canonical_weekly_metrics_from_trace` | PARTIAL |
| RL07 | agent-multi `tests/unit/test_rl07_l1_l2_stopping.py` | FakeModel whose composite sequence is scripted | best checkpoint restored (hash equals the epoch-of-best hash), stop reason and step count recorded, `assert_declared_stopping_contract` refuses an undeclared terminator; an L2 search with fewer than the declared candidates is refused | `tests/unit/test_stopping_contract.py`, `test_rl_pipeline_with_validation_splits.py::test_l1_warmup_epoch_cannot_become_best_or_consume_patience` | MOSTLY PRESENT (L2 "silent tiny search" refusal missing) |
| RL08 | agent-multi `tests/unit/test_rl08_durable_dag.py` | transition records in `tmp_path`; simulate a crash between checkpoint write and pointer commit; RAM/VRAM/disk caps in config | reconstruction from records alone; idempotent restart replays nothing twice; typed retry with cooldown; a job over its declared memory cap is refused at admission | `tests/test_experiment_transition_queue.py`, `tests/test_rolling_origin_adaptation.py::TestSingleTransactionState`, `tests/test_successor_quarantine.py` | MOSTLY PRESENT (memory cap admission not in these tests) |
| RL09 | predictor `tests/test_rl09_rl_outbox_readback.py` | synthetic terminal with metrics, genome, seed, identities | outbox row -> loader -> readback equals the submitted canonical bytes; physical mirror hash equals; final policy artifact sha256 and tensor hash recorded and re-derivable | `tests/test_governed_outbox_disposition.py`, `tools/olap_loader.drain_once`, `fs_phase23_warehouse` readback idiom | MISSING (no RL metric/genome table contract) |
| RL10 | `lts` (not read in this lane) | — | paper endpoint verified; a live/real flag is refused by construction; a policy hash different from the validated one is refused at activation | gym-fx `oanda_broker` refuses without `GYMFX_ENABLE_LIVE=1`, but `oanda_practice` is a plain config flag — not a structural real-capital refusal | UNKNOWN (lts owns) |

## R5. Hypotheses for brainstorming with the owner (no implementation)

- H-R1 regime-conditioned experts: a small set of entry/exit experts
  (trend-follow, mean-revert, abstain) selected by a causal regime signal
  (realised-vol ratio, spread ratio — the same statistics the reopen check
  already computes) may beat one monolithic policy; the selector, not the
  experts, is the RL object. Counter-hypothesis: the selector overfits regime
  labels that are only identifiable in hindsight.
- H-R2 bounded online updates: cap each weekly update to a trust region
  (parameter-space norm or KL on held-out states) so a bad week cannot
  rewrite the policy; compare NONE / WEEKLY / WEEKLY-bounded on identical
  weeks. Prediction: bounded updates lower variance, not mean.
- H-R3 short-horizon heuristics as masks, not rewards: use the hourly
  calendar and spread evidence to mask entries (MaskablePPO style) and keep
  the reward purely economic; compare against penalty shaping. Prediction:
  masks reduce constraint violations to zero at no return cost; shaping does
  not.
- H-R4 cadence x history interaction (owner's HYP-R1): refit cadence
  {NONE, MONTHLY, WEEKLY, DAILY} crossed with history length {4, 13, 52
  weeks}; expectation of non-monotonicity; the measurement is prospective
  weekly net PnL dispersion, never fitted-week score.
- H-R5 exposure as the control variable: SAC's continuous output should be
  target exposure in [0,1] of a declared cap, with hysteresis; test whether
  the learned magnitude carries information (compare with its sign-only
  projection on the same weeks).
- H-R6 distributional selection: with QR-DQN/TQC, select checkpoints on a
  lower quantile of the weekly return distribution rather than the mean;
  hypothesis: better drawdown at equal mean.

## R6. PDT applicability (FINRA Regulatory Notice 26-10, read 2026-10-18)

Source: https://www.finra.org/rules-guidance/notices/26-10. The notice
announces amendments to FINRA Rule 4210 that replace the former day-trading
margin requirements in their entirety: the day-trade count test for the
"pattern day trader" designation and the USD 25,000 minimum equity are
removed and replaced by an intraday margin framework. Effective date
2026-06-04; members needing more time may phase in through 2027-10-20.
Scope: customer securities margin accounts of FINRA member firms (portfolio
margin accounts handled separately; good-faith accounts excluded from the
intraday deficit determination). Regular maintenance margin is unchanged.
The notice does not mention retail forex or FX products.

Applicability to this business, stated without inference beyond the text:

- The rule governs securities margin accounts at FINRA members. Spot FX
  with a retail forex dealer is not a securities margin account under this
  notice; whether any day-trade restriction applies to a given EURUSD account
  is a property of that broker, that account type and that regulator, not of
  this notice.
- Therefore neither a four-trades-per-five-days constraint nor its removal
  may be hard-coded. `app/oanda_calendar.py` currently hard-codes a NY
  session policy citing a project memo; it is observation-only and must not
  be read as a PDT rule.
- Where to record it: the existing private configuration layer that already
  holds account fingerprints and venue identities (agent-multi
  `~/.config/agent-multi/` operator authority, never in git; gym-fx
  `session_account_fingerprint` / `session_venue` config keys bound into the
  calendar digest). Recommended shape: a per-account `trade_frequency_rule`
  block {rule_id, source_document, effective_from, effective_to,
  max_round_trips_per_window or null, verified_on, verified_by_role} whose
  digest joins the session calendar identity, so a run records WHICH rule it
  ran under without placing the account identifier in any public file.
- A broker's adoption during the phase-in window (2026-06-04 to
  2027-10-20) must be read from the broker's own documentation at run time
  and recorded in that block; this document asserts nothing about any
  specific broker.

## Proposed finite campaign manifest (RL0 -> RL4), no ETA

| Stage | Runs (finite) | Reuses accepted code | Requires new tests / data acceptance |
|---|---|---|---|
| RL0 env certification | RL01-RL04 synthetic tests (R4) on CPU; one EURUSD calendar-interval generator run producing a digest-pinned 2020-2025 interval file; one 12/>12/12 policy JSON | gym-fx env, session overlay, envelope, custody | B4 gap refusal, B5 enlargement classification, financing debit or an explicit "financing assumed zero" label, `Discrete(4)` space; lane D clock resolution before B6a is certifiable |
| RL1 controls | 4 controls (NO_TRADE, restricted BUY, restricted SELL, RANDOM x 3 seeds) + 2 rules (momentum, reversal) on TRAIN, VAL and development weeks; 2 cost profiles | baseline agents, pipeline deterministic branch, weekly metrics | RL06 parity test; admitted EURUSD H1 and selection manifest (RL00) |
| RL2 SAC | compact control: 3 seeds x 2 cost profiles, L1 on train_tail+2024, then WEEKLY prospective loop over 2024 development weeks; modular representation: same 3 x 2 after the extractor adapter exists | `sac_agent`, `rl_pipeline_with_validation`, `load_for_training`, `score_interval` semantics | EURUSD prequential runner (parameterised successor of the ETH tool), RL05 test, extractor adapter + budget test, RL09 warehouse contract |
| RL3 contrasts | DQN 3 seeds x 2 profiles on `Discrete(4)`; then one of {MaskablePPO, TQC/CrossQ} 3 x 2 only if the SAC and DQN results make the comparison meaningful | `dqn_agent`, `ppo_agent` | DQN tensor-only loader, sb3-contrib install approval and compatibility check, action-mask function from `session_state` |
| RL4 cadence/history | cadence {NONE, MONTHLY, WEEKLY, DAILY} x history {13, 52 weeks} for the best RL2 policy, 3 seeds; cost perturbation +/- 50 %; then DEMO/PAPER under the no-real-money rule | prequential runner, cost curriculum | RL08 admission caps, RL10 (lts) endpoint verification |

Total scientific runs if every branch executes: 2 + 18 + 12 + 12 + 24 + 12
(perturbation) = 80 bounded runs plus baselines; none is authorised by this
document. 2025 remains declared non-pristine; forward paper weeks are the only
untouched confirmation population.

## Limits

- No fit, no GPU, no deploy, no DOIN edit, no service change, no data
  admission; nothing here is a performance claim.
- Line numbers are valid only at the revisions named at the top.
- Two papers (9, 10) were read from the extracted PDF text; three (8, 9, 10)
  were read in full, the others from abstracts plus official docs.
- `lts` (RL10) and `data-gov` internals were not read; gym-fx Nautilus
  engine code was read only for the financing path.
- The dirty agent-multi checkout was not stashed or cleaned; its dirty files
  do not include the runtime modules cited.
