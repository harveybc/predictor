# TIMING_PROXY_SELECTION v5 (target family `timing_profit`): result

Status: DEVELOPMENT, nothing tradable. Owner decision 2026-10-20: improve the signal before the architecture.
Code: predictor `satoshi/timing-selection-20261019` @618dae72 (tools/timing_profit_targets.py, family dispatch in
tools/timing_selection.py, v4 path byte-identical under test), evaluator heuristic-strategy
`satoshi/timing-selection-eval-20261019` @177debc. Run on the 4090 host, `chain_v5.sh`, about 75 min end to end.

## Target

Profitable direction under the lane's order rule: fill at open(t+1); TP and SL at 2 sigma of the causal 24 h
volatility; SL first when one bar touches both; closed at the session's forced flat (12 h before close) or after
72 bars at that close, net of 0.1 sigma. Class 1 = the long wins and the short does not; 2 = the reverse; 0 = neither.
Support ledger, censoring and purges at cuts as v4; finality verified under truncation and future perturbation.
Labelled origins 2020–2024: class 1 11,127, class 2 11,620, class 0 1,950 (symmetric 2 sigma barriers make the label
"which barrier is touched first").

## Selection (TRAIN 2020–2022 only)

- Pool: the 93 v4 candidates (reused through the DAG).
- Dependence-aware screen (blocked circular shift, 4,999 shifts, BH q 0.1): **3 of 93 pass**, all calendar:
  `cal.utc_dow_cos`, `dow_cos`, `px.hours_since_session_open`. No price, volatility, indicator or fractal feature passes.
- MRMR K=12 recipe: **one input selected**, `px.hours_since_session_open`.
- Causal stage (2-bar lag window): controls behave (planted leak detected, both negative controls NO_LINK).

## Predictive validation on identical rows (3 sets x 3 models, 3 seeds; choices on 2023 S1 only)

Calibrated log-loss minus the constant TRAIN prior (negative = better than the prior): **positive for every set,
model and period** (+0.10 to +0.18), i.e. every model is worse than predicting the class frequencies. PR-AUC sits at
the base rate (for example 2024: 0.43–0.53 against 0.45–0.46).

Strategy at the frozen thresholds (lane rule, executable fill, 4/day cap), Sharpe, seed means:

| Set | Model | VAL B 2023 S2 | 2024 (single read) | Orders/week | Valid (>= 3/week) |
|---|---|---|---|---|---|
| v5 selected (1 calendar input) | m1 conv / m2 GRU / m3 HGB | −1.77 / −1.02 / −1.53 | −1.16 / −0.79 / −1.22 | 0.7–1.0 | no |
| Musashi's 15 | m1 / m2 / m3 | −0.26 / 0 (no trades) / −1.66 | −0.06 / 0 / −0.55 | 0–1.1 | no |
| v3 six | m1 / m2 / m3 | −0.11 / −1.21 / −1.91 | +0.21 / −0.22 / −0.69 | 0.7–1.3 | no |

No combination is frequency-valid and none is consistently positive.

## Reading

1. With the order-outcome label, the 93-candidate pool carries **no detectable dependence with the profitable
   direction** beyond weekday/session timing; the selected set is a single calendar clock.
2. All models, on all three sets, are worse than the constant prior in probability; trading is negative and invalid.
3. This agrees with the whole lane (pivot labels, capacity ladder, modular specialists): the hourly EURUSD direction
   behaves as a martingale with respect to these inputs. The next gain will not come from architecture or from
   re-selecting this pool; it needs inputs of a different nature (order flow / positioning, calendar surprises with
   point-in-time consensus, cross-asset) or a different objective (volatility-sized risk, where skill exists).
