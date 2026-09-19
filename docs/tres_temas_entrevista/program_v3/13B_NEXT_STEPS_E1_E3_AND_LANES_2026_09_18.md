# 13B — Next real steps (RP6): E1 public bank, forecasting/RL demand, E3 protocol, lanes

Nothing here is executed; every row has an identifier, source, dependency and deliverable.

## Forecasting / RL demand read from executable configurations (agent-multi, read-only)

| config | agent | env | asset / data | window | features | strategy | commission | slippage | reward | timesteps |
|---|---|---|---|---|---|---|---|---|---|---|
| `dqn_btc_1h_twelve_atr.json` | dqn_agent | gym_fx_env | btcusdt_1h | 32 | twelve | direct_atr_sltp | 0.001 | 0.0 | pnl_reward | 500000 |
| `dqn_eth_1h_twelve_atr.json` | dqn_agent | gym_fx_env | ethusdt_1h | 32 | twelve | direct_atr_sltp | 0.001 | 0.0 | pnl_reward | 500000 |
| `p4_ppo_btc_1h.json` | ppo_agent | gym_fx_env | btcusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 2000000 |
| `p4_ppo_eth_1h.json` | ppo_agent | gym_fx_env | ethusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 2000000 |
| `p4_ppo_eth_1h_iter8.json` | ppo_agent | gym_fx_env | ethusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 2000000 |
| `p4_ppo_eurusd_1h.json` | ppo_agent | gym_fx_env | eurusd_1h | 32 | twelve_macro | direct_atr_sltp | 0.0005 | 0.0 | pnl_reward | 2000000 |
| `p4_ppo_eurusd_1h_iter9.json` | ppo_agent | gym_fx_env | eurusd_1h | 32 | twelve_macro | direct_atr_sltp | 0.0005 | 0.0 | pnl_reward | 2000000 |
| `ppo_btc_1h_twelve_atr.json` | ppo_agent | gym_fx_env | ./data/project2/btcusdt_1h/d4.csv | 32 | None | direct_atr_sltp | 0.001 | 0.0 | pnl_reward | 50000 |
| `ppo_eth_1h_twelve_atr.json` | ppo_agent | gym_fx_env | ./data/project2/ethusdt_1h/d4.csv | 32 | None | direct_atr_sltp | 0.001 | 0.0 | pnl_reward | 50000 |
| `ppo_eurusd_1h_twelve_macro_atr.json` | ppo_agent | gym_fx_env | ./data/project2/eurusd_1h/d4.csv | 32 | None | direct_atr_sltp | 5e-05 | 0.0 | pnl_reward | 50000 |
| `ppo_gymfx_default.json` | ppo_agent | gym_fx_env | ../gym-fx/examples/data/eurusd_sample.csv | 32 | None | default_strategy | 0.0 | 0.0 | pnl_reward | 10000 |
| `project3_ethusdt_4h_sac_actor_critic.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 25000 |
| `project3_ethusdt_4h_sac_actor_critic_feature_aware.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 25000 |
| `project3_ethusdt_4h_sac_synth_anti_mem_v1.json` | sac_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 25000 |
| `project3_ethusdt_4h_sac_synth_bootstrap_v1.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | None |
| `project3_ethusdt_4h_sac_train_val_test.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | None |
| `project3_ethusdt_4h_sac_train_val_test_v2.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | None |
| `project3_ethusdt_4h_sac_train_val_test_v3.json` | project3_sac_actor_critic_agent | gym_fx_env | ethusdt_4h | 32 | tech_stat | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | None |
| `sac_btc_1h_twelve_atr.json` | sac_agent | gym_fx_env | btcusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 500000 |
| `sac_btc_1h_twelve_atr_iter8.json` | sac_agent | gym_fx_env | btcusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 500000 |
| `sac_eth_1h_twelve_atr.json` | sac_agent | gym_fx_env | ethusdt_1h | 32 | twelve | direct_atr_sltp | 0.0002 | 0.0 | pnl_reward | 500000 |
| `sac_eurusd_1h_twelve_macro_atr.json` | sac_agent | gym_fx_env | eurusd_1h | 32 | twelve_macro | direct_atr_sltp | 0.0005 | 0.0 | pnl_reward | 500000 |

What the configs fix: hourly bars (`*_1h`), an observation window of 32 bars, the `twelve` feature
preset, an ATR stop/take-profit execution strategy, commission 0.001 and zero slippage, PnL reward,
DQN/PPO agents. What they do **not** fix (gaps, owner Satoshi + owner): weekly training cutoff and
release time, decision/execution hours, the untouched next-week evaluation, funding, per-feature
availability at decision time, capital and position sizing beyond `position_size 1.0`. No asset is
chosen here by having a CSV; the universe is the one these configs run (BTC/ETH/EURUSD 1h) pending
the owner's business decision (12E row 1).

## E1 public multivariate bank (design, no re-census)

Reuse the existing census and profiles (D0–D2: 715 datasets profiled) — select, do not
recharacterise. Criteria for E1 families: multivariate (p ≥ 6), regular sampling, ≥ 20 cycles of the
dominant period in training, a licence allowing research use, documented availability, ≥ 2 targets.
Development and reserve families come from distinct sources (never the same dataset split by rows).
Deliverable: `E1_FAMILIES.json` with licence, availability, variables/targets, geometry and cost per
family, DEV vs RESERVE, produced by a query over the census — task MOD-E1, depends on MOD-E0-DEV.

## E1 design references (from the proposal)

References: naive, seasonal naive, a pertinent multivariate statistical model, DLinear, PatchTST,
iTransformer, DUET, and one of MTST/Pathformer/TimeMixer chosen before development results.
R0/R1/R2 compared in development; context and groups selected on validation only; margin ε and the
precision plan fixed before any reserve; K = F + 4 Bonferroni intervals with hierarchical bootstrap.

## E3 forecasting + RL (mandatory, not conditional on H1)

Weekly cycle train → inner validation → untouched next week; raw vs modular with the same
information, same agent (from the configs above), same reward/costs/execution; deliverables: net
return, drawdown, Sharpe with assumptions, turnover, exposure, uncertainty across weeks/regimes,
observed steps and retraining time; trivial policies and a competent non-modular control.
Depends on MOD-E1 and BUSINESS-CONTRACT. No live orders.

## Lanes (identifier, source, dependency, deliverable)

| id | source | depends on | next task | deliverable |
|---|---|---|---|---|
| PRE-NOISE | P-PRE proposal | D2 evidence by regime | reconcile D2 candidates by regime; design the uncovered receiver/task contrasts with causal noise controls | H1 degradation, H2 conditional denoising, H3 residual preservation |
| TRN-TRANSFER | P-TRN matrix | T0–T2, D3 | per-variable vs global transfer with adequate receivers; abstention curves | conditional utility, transfer/cost, abstention, DOIN parity |
| CAP-CALIBRATION | P-CAP | M3/M4 mapping | random-label capacity and N_min protocol on the declared MLP/GRU | C_mem, log-loss bits, N_min with censoring |
| L2-PUBLIC-RL | P-L2 | B4/selector evidence | map to the POPGym/CARL protocol before new RL code | cost/regret, risk-coverage, transfer |
| INC-SIMULATION | P-INC | existing simulator | HPO-B pilot and a second training domain | allocation, learning, manipulation utility |
