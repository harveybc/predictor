# Actual Selected Inputs And SAC Contrast

## Selection Closed

The repaired/admitted EURUSD core produced30 reproducible candidates: eight
RAW-derived and22 existing technical/seasonal variables. Actual TRAIN-only
profiles,435 pairs, two prefix folds, three targets and nine filter methods
completed. The mRMR k12 union is pruned by existing TRAIN Spearman clusters,
leaving15 variables.37198 native warehouse rows have exact stored-value
readback; no pending/mismatched rows.39 scientific/input/database files and
two deployment metadata files have matching physical copies. This is a scoped
filter selection, not completed admission of the external catalogue, causal
effect identification, an optimized trading subset or a claim of profit.

Ordered selected variables: `HIGH`, `RAW_LOG_RANGE`, `RAW_CLOSE_LOG_RETURN`,
`RAW_ELAPSED_HOURS`, `ATR`, `RSI`, `MACD_Histogram`, `ADX`, `DI_minus`,
`Stochastic_D`, `WilliamsR`, `rolling_std_24`, `hod_sin`, `hod_cos`, `dow_cos`.
All30 candidates have explicit decisions/reasons. Unknown causal support is
not interpreted as evidence of uselessness. Retired scores/weights are unused.

Selection identity:
`553fc04707ee3d1d2d552a21af073bf942e3a1b880150f2b34245764446c09e2`.
Owner documentation:
[feature-selector](https://github.com/harveybc/feature-selector/blob/codex/curation-reset-20261018/docs/ADMITTED_CORE_SELECTION.md).
Global `feature_selection_complete:false` continues to describe the FULL
external inventory; this specific selection is complete
`WITHIN_DECLARED_ADMITTED_POPULATION`. Never conflate these scopes.

## Comparison Running

Leverage50 replacement started2026-10-19T10:50:19Z (05:50 Bogota). At startup verification,
SAC direct was training with a fresh heartbeat on the5090. The same
finite queue then runs validation, SAC temporal, its validation and four
mechanical controls across both splits:12 tasks. No agent watches logs.
Persistent worker, publication/readback timer and closed-artifact collector
continue across restarts. One heavy process on this host, never5070Ti parallel.

| Setting | Both SAC arms |
|---|---|
| Selected information | Same ordered15 variables, data, availability and selection pins |
| TRAIN / VALIDATION | 2020-2023 / 2024;24862 /6221 input rows |
| Source warmup |33 leading indicator rows only; no fabricated/interpolated rows |
| Input history |13 observations, stride2 observed bars, includes current completed bar |
| Scaling | Causal rolling z-score256, past-only |
| Learning | SAC, seed7,100000 transitions, no refit; static validation |
| Anti-overfitting ablations | Not active: no ES/dropout/BN/explicit weight decay in this fixed control |
| Capital / activity | USD10000, one position and at most one entry per NY business day |
| Mandatory order budget | Fraction chosen by action times0.1 times balance; one-base-unit illustrative minimum and downward step |
| Protection / leverage | Finite positive SL/TP required before entry; ATR multipliers2/4; pilot leverage1:50 |
| Closures | Existing12h force-flat/reopen rules and unchanged entry wind-down |
| Costs | Existing nonzero analytical scenario, not broker-verified costs |
| Direct arm | Existing compact SAC using the selected temporal window directly |
| Temporal arm | Per-variable causal Conv1D filters4, aligned fusion, core filters8; trainable with SAC |
| Final heads | Same32/32 actor/critic heads; no retired pretrained encoder weights |

Contract:
`37e0619224109416dc35a726831183975fd41b03823a30685630dc2896d38029`.
Scientific runtime agent-multi `ab265fad`, gym-fx `df1b602`.
Preparation code `41f5af170`; selection operation `9d4d06a`.
Verification: six prep tests,55 selector tests,135 focused RL/input/collector
tests and110 focused gym protection/budget/session tests pass. Counts are
independent suites, not a universal quality certificate.

The initial incomplete contract41c6acf3 was stopped and disabled after the
owner clarified the mandatory balance allocation. It used portfolio-fraction
sizing without rel_volume and is not reused. New request v2 changes only the
order-allocation contract and engineering guard, not selected data, window,
architecture, seed or SAC budget. Both new arms start fresh under this identity.

The owner then prioritized leverage1:50, not1:1. Incomplete contract143051b1
was stopped and not reused. Request v3 binds leverage50 into the broker and
both envelope leverage fields. The15 inputs,13 observations at stride2,
models, seed, rel_volume0.1, costs, protection and budgets remain unchanged.
Leverage1:100 is configurable; a1:1 contrast is deferred, not running in parallel.

Balance allocation is a margin ceiling in account currency, not risk-to-SL:
`min(abs(action),1) * rel_volume * balance(t)`. The existing0.002 cost headroom
reduces allocation; configured leverage converts to notional and price to base
units. Values below the broker minimum abstain, never round upward. Minimum
and step are examples, not broker-certified constraints. The scalar action
chooses a fraction; it is not a calibrated probability of low/high market risk.

No new policy results or measured full-campaign ETA existed at this start cut.
Generated status is authoritative. The RAW control remains separately closed;
its eight inputs are NOT the inputs of this comparison. No TEST tuning,
supervised predictor, AE pretraining, DOIN changes or real-capital execution.

## Next

Close all12 tasks, exact OLAP readback and two physical policy/metric copies;
report frozen-policy TRAIN and VALIDATION separately. Then controlled,
prospectively specified ES/dropout/weight-decay/BatchNorm ablations. The ES
candidate uses the same frozen checkpoint's comparable net daily TRAIN and
VALIDATION performance, not mixed-policy exploration returns or raw4yr/1yr
profit. It is a separate design, not a mid-run change or an immunity guarantee.
Broader source admission and causal identification remain explicit unfinished
work; this first selected contrast does not certify the entire catalogue.
