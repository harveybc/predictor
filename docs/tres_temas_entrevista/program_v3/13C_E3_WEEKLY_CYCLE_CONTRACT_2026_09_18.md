# 13C — E3 weekly cycle: contract of the experiment (RP15, amended under RP23 by 13D)

**Amendment (2026-09-19, RP23).** Musashi's [13D](13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md) resolves every
OWNER_DECISION item below for DEVELOPMENT/SIMULATION and prevails over them: universe = the traced
BTC/ETH/EURUSD candidates with the cash-spot BTC/ETH scenario first (conditional on data contracts; FX and
derivatives are distinct strata); capital = initial equity 1, long/flat actions, no credit or leverage, sum
of long notionals ≤ equity; retraining weekly with the UTC week [Monday 00:00, next Monday 00:00), cutoff and
release derived from measured availability and duration (never "the last bar before the release"); funding
= zero ONLY by definition of the cash-spot scenario (derivatives: per-instrument data or NO_EVALUABLE).
Clocks fixed before scoring: acquisition window, last permitted information, fit start/end, publication,
first decision, first execution, with cutoff ≤ fit_start < fit_end ≤ release ≤ first_decision and every
feature's available_time ≤ decision_time; the inner validation window precedes the cutoff; a late model
uses the declared fallback (last valid model or flat) and records the miss, never a shifted release; state
(equity, positions, pending orders, commissions, financing) persists across weeks; purge/embargo derive
from the pipeline. Real operating facts (venue, monetary capital, lot minimums, SLA, derivative funding)
remain UNKNOWN and are requested together only before a test that depends on them. The deterministic
environment tests of this contract are `tests/test_e3_weekly_env.py` (software tests, no financial return): they
exercise what the ENVIRONMENT does when given actions, with the policy, the release check and the fallback written
inside the test. **They validate neither real data availability, nor a late release, nor weekly operation.** The
component that decides the available model, the fallback and the scenario's action is `tools/e3_weekly_controller.py`,
tested by `tests/test_e3_weekly_controller.py` (RP31: early / late / absent model, fallback executed, week by
timestamps, continuity, close-to-flat without a short, sizing against a varying equity, gap between close[t] and
open[t+1] with positive latency, and mutants without a release check or turning flat into short that must fail).
Offline software: no financial operation and no RL training were run for it, and it is not a completed E3.


This is the contract E3 will be judged against. It is written as an EXPERIMENT contract: every value
below is either read from an executable configuration (13B, with its file and line), declared here as
an experimental choice, or marked OWNER_DECISION where it is a business fact nobody can deduce from the
code or the data. Nothing here is a production default and nothing orders a real operation.

## Cycle

| item | value | origin |
|---|---|---|
| bar | 1 h (BTC/ETH/EURUSD `d4.csv`, 13B) or 4 h (ETH `tech_stat`) | traced configs; the bar is a factor, not a choice made here |
| week | Monday 00:00 UTC → Sunday 23:00 UTC of the bar timestamps; the `W-SUN` grouping already used by `_weekly_metrics.py:50-62` | declared: aligns the evaluation week with the only weekly notion in the code |
| training cutoff | the last bar strictly before the release instant; the training window ends there and never sees the evaluation week | declared |
| release instant | Sunday 12:00 UTC (12 h before the first decision bar) | declared: an experimental latency budget, to be replaced by the owner's operating schedule (OWNER_DECISION: when can a model realistically be retrained and deployed?) |
| decision bars | every bar of the evaluation week for 1 h; `entry_hour_start 12` … `force_close_dow 4 / force_close_hour 20` where the config declares a session (only `project3_…_v3.json`) | traced; a session filter is a factor to declare, not to assume |
| evaluation week | the untouched next week; nothing of it enters training, validation, feature scaling or selection | declared (the rule the whole programme enforces) |
| inner validation | the last complete week before the cutoff (selection of checkpoints/hyper-parameters); train = the years before it (`train_years / val_years` of `rl_pipeline_with_validation.py:537-552` become weeks here) | declared |
| rolling | the cycle repeats over every week of the held period; weeks are the unit of uncertainty (never bars) | declared |

## Information available at decision time

| item | value | origin |
|---|---|---|
| observations | the preprocessor window (32 bars, `default_preprocessor.py:22`, or the config's `window_size`) of the feature columns of the CSV; nothing beyond the current bar | traced |
| feature availability | `twelve` features are functions of the same bar's OHLCV (computable at bar close); `twelve_macro` merges macro series with a backward `merge_asof` and a 7-day forward-fill cap (`prepare_project2_data.py:178`): the availability delay of each macro series must be DECLARED before use (publication ≠ reception, cf. the column-role contract) | traced + declared |
| funding / swap | not modelled anywhere (13B): E3 declares them as ZERO with that statement, or the owner supplies the instrument's funding schedule (OWNER_DECISION) | gap |
| latency | order at the close of the decision bar, filled at the next bar's open with `slippage_perc` (`default_broker.py:52`); no queue or network latency is modelled | traced + declared |

## Costs and capital

| item | value | origin |
|---|---|---|
| commission | 0.001 (DQN BTC), 0.0002 (PPO/SAC BTC/ETH), 0.0005 (PPO EURUSD) as fractions of notional; a cost is a FACTOR of the config, never a default (the broker's default is 0.0) | traced |
| slippage | 0.0 in every traced config; E3 adds a declared non-zero level as a sensitivity arm | traced + declared |
| capital | `initial_cash 10 000` (every config); `position_size` 1.0 / 0.01 / 1000 with `size_mode` `fx_units` or `notional` by asset class | traced; the real capital and the maximum exposure per position are OWNER_DECISION |
| leverage | 1.0 except EURUSD PPO (10) | traced |
| solvency | `min_equity` under `solvency_mode normal_realistic` ends the episode; the weekly cycle restarts from the previous week's ending equity, never from a reset | traced + declared |

## Sizing of the experiment

| item | value | origin |
|---|---|---|
| assets | the three the traced configs run (BTC 1h, ETH 1h/4h, EURUSD 1h); adding or removing an asset is OWNER_DECISION | traced |
| weeks held | every complete week of the data span after the first training window (BTC 2019: ~40; EURUSD 2005–2019: ~700; ETH 4h 2017–2025: ~400) | traced spans |
| arms | raw vs modular features with the SAME information, agent and costs; forecasting (MASE/MAE per horizon) and RL (net return, drawdown, Sharpe, turnover, exposure) on the same weeks; trivial policies (flat, buy-and-hold, random with the same turnover) and a competent non-modular control | declared (13B, E3) |
| budget | CPU only, agents' `total_timesteps` as declared per config, one retraining per week; the wall and CPU per week are measured in a cost pilot before the full cycle | declared |
| deliverables | per week and asset: net return with capital and costs explicit, drawdown, Sharpe (weekly periodicity, stated), turnover, exposure, observed steps, retraining time; uncertainty across weeks and regimes | declared |

## What is escalated

One decision only, because it is a business fact: the asset universe, the real capital / maximum
exposure per position, the operating retraining-and-release schedule and the instrument's funding
model. Options with effect: (a) keep the traced universe and sizes as the experiment's declared
choices (E3 then measures a hypothetical operator with 10 000 of capital and the traced sizes); (b) the
owner supplies the real values and E3 measures those. Everything else above proceeds under the
declared values.
