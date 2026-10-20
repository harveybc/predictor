# Immediate Minimal Temporal SAC

Superseded2026-10-20 by [fixed-budget execution](RL_FIXED_BUDGET_TEMPORAL_2026_10_20.md).
The frequent-monitor run described below was stopped with five complete pairs
retained and physically mirrored. Its scope is PARTIAL,not scientific closure.
The new identity preserves architecture/data/business but changes monitoring
and selects only the final100000 policy. Historical deployment facts follow.

Owner2026-10-20 releases the temporal pause and prioritizes an immediate
Conv1D contrast over the running flat capacity sweep. This is the current RL2
task. The primary market representation must retain the time-series layout
until temporal extraction; flat models remain explicitly labeled controls.
Flattening raw windows does not erase values or their fixed lag positions,
but it is not the temporal processing architecture now required by the owner.

## Running Recipe

15 selected features x13 observations,stride2 observed hourly open-market
bars,including the current completed bar. Existing causalConv1D branch per
feature:2 filters,kernel3; concatenate branches along filters at every time
position; one causalConv1D core:4 filters,kernel3. All13 positions remain,
then flatten52 learned temporal values and append unchanged business state.
SAC actor and twin critics have hidden[8]; independently trainable actor and
critic extractors use native SB3 behavior. No pretrained donor,dropout,
BatchNorm,L2,pooling,right padding,new clock,feature change or target.
Two convolution stages have local receptive field5 sampled positions; the
small output heads combine all13 positions. No arbitrary claim that this
configuration must beat Dense or fix generalization.

TRAIN2020-23,VALIDATION2024,seed7,costs,leverage50,mandatorySL/TP,volume budget,
one entry/day and session mask all preserved. No refit,TEST or DOIN. Monitor
every5000 transitions,patience10,absolute1e-8,cap100000,unchanged
M=-0.5*(full TRAIN mean daily log growth + full VALIDATION mean daily log growth).

## Evidence And Automation

`agent-multi@96e19edb`,contract9cae5a6fb9f0.166 local tests pass,including
six real-engine monitor variants;29 tests pass in the actual worker environment,
including temporal parent RNG/replay invariance,child encoder/arm identity and
native best-policy/optimizer restoration. Prefix/origin perturbations pass.
The first worker check had a baseline timeout under excessive memory.high
reclamation (130698 events,no OOM); the isolated engineering cap was raised,
and the final acceptance took11.968s,1.4GiB peak. Scientific6GiB cap unchanged.

Real5090 GPU smoke passed16 updates and exact reload on attempt1. The
SAC_PER_FEATURE_CONV1D queue arm is RUNNING,service restart count0. This is
verified startup,not a completed financial result or a measured full-run ETA.
Worker,publisher/readback/close and second-machine closed-model collector are
enabled on timers. No further owner approval or agent log polling required.

Real progress reached5000 transitions in109.0488s before its first paired
inference. The actual saved checkpoint has1062 actor parameters and1638
critic parameters (2700 trainable actor+critic elements plus entropy scalar);
target critic has1638 copied elements. The temporal feature vector is52
values plus17 business-state values,not a raw195-value Dense-only input.
These are actual policy counts,not synthetic-smoke counts; matching hidden[8]
does not make the total parameter budget identical to flat[8].

The flat[8] worker and all flat successors were stopped for this priority.
Its6 completed pairs are retained as partial evidence,final publication pumped,
and artifacts copied with identical second-machine checksums. There is no
completed flat[8] scientific result. Do not reuse its weights or combine its
partial scope with a full-run winner to claim a controlled architecture gain.
The full flat[32,32] reference has different capacity. Compare identical-step
flat[8] checkpoints descriptively when available,with the interrupted scope.

The numerical output will retain separate full TRAIN/VALIDATION net profit,
return,Sharpe(daily,nonannualized),drawdown,closed trades,win rate and profit
factor,plus checkpoint step,stop reason,support and identities in the OLAP.
The existing15-feature selection is unchanged; no Satoshi timing6 integration.

Implementation/runbook:
`agent-multi/docs/work_plan/50_MINIMAL_TEMPORAL_SAC.md`.
