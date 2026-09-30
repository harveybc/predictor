# Current execution spine (2026-09-30)

Snapshot at 22:12 UTC. This is the operational index for concurrent work. The
master plan retains the scientific questions and historical decisions; older
dated queues are evidence of their own time, not the current machine state.
An active process, a completed software test, and an accepted scientific result
are different states.

## Measured results already available

| Lane | Evidence | Limit |
|---|---|---|
| ECL TimeFilter L96 | 12 cells; mean normalized MSE/MAE 0.161962/0.259662 versus paper 0.158250/0.255750 | Operational agreement, not exact published reproduction or a modular contrast. |
| ECL TimeFilter L512 | 12 verified cells; 0.150307/0.246397 | Paper Table 9 lookback mapping is unresolved. |
| ECL modular R0/R1/R2 | DEV MAE 0.371174/0.374584/0.368596 on 1,802 support-disjoint validation windows | No external-test result; no same-population comparison with TimeFilter. |
| Weather TimeFilter | 12 cells and technical replay evidence | Scientific custody/comparability needs separate closure. |
| Branch Conv1D AE | One train-only donor pilot; reconstruction MAE train/validation 0.066813/0.066229 | Not a forecasting gain or multi-branch ablation. |
| Fused-core AE | 128 train/64 validation rows, four updates; R0/R1 MASE 10.5920/10.7044 | Plumbing/cost check only, not efficacy. |
| News classification | Real Laya wrapper matched the native SDK on 12/12 inputs; 13 owner-labeled smoke rows macro-F1 0.3333 | Not a financial-quality or held-out benchmark. |

No DOIN-optimized modular forecasting model, full branch-pretraining comparison,
validated financial forecasting winner, or trading return from such a model has
been measured. Do not infer one from the table above.

## Active and next, in parallel

| Lane | Actual state / owner | Next experiment or deliverable | Dependency local to this lane |
|---|---|---|---|
| Reference Traffic | Dragon RTX 4090 trains h96 seed 2022; seed 2023 queued. Omega RTX 4070 trains h96 seed 2021 with protective clock cap. | Finish exact author-space scores, same-row naive and receipts; then bounded full-population h192/h336/h720 cells. | Bounded evaluator integration and per-horizon probe before long-horizon scoring. No other lane waits. |
| Matched ECL architecture | Same-population scorer exists; nine selected modular weights are absent. Subagent `01a0f45f-221b-7263-867c-520ab0638008` owns recovery/cost path. | Train or recover all nine R0/R1/R2 weights under the matched 321-channel design, then compare on identical ECL test targets with TimeFilter. | A 14 GiB pilot was refused at 16.64 GiB projected use; no test score is permitted before weights exist. |
| Representations | Branch AE donor and fused-core AE feasibility code exist in isolated worktrees. | Wire branch donors and fused-core donor into matched R0/R1/R2, measure forecast metrics and ablations; retain temporal axes. | Small reconstruction/plumbing pilots are not substitutes for full fits. |
| DOIN | Subagent `01a0f45f-614f-7943-83ec-2e14bdd566a5` owns typed modular config/objective integration. | Run a dry-run round trip, then a measured optimization campaign only after the matched model path is runnable. | Existing external archive prototype is not a live optimized branching result. |
| Paper execution | Subagent `01a0f45f-847a-7d22-bcf7-eeb39b0ae101` owns read-only candidate handoff. Alpaca Paper runner active; MT5 Paper inactive. | Validate artifact, metrics, population and risk contract for a *future* winner before paper-only smoke. | No model promotion or broker order from this development lane. |
| M5PHET / news / calendar / RL | Product and provider lanes are separate from the matched doctoral score. | Keep usable classification/forecasting paths and causal calendar/RL work independent; benchmark each with its own task. | No classification smoke result authorizes a trading signal. |

The three newly dispatched agents are local subagents, **not** Hermes GPU jobs.
Their software tasks are disjoint from the active immutable Traffic snapshots.
Do not describe them as three GPU experiments.

## Capacity and ETA

At 22:08 UTC, omega 4070 and dragon 4090 were training Traffic. Gamma's
external 5090 and internal 5070 Ti were idle: its host reported 1.9 GiB
available RAM and 5.7 GiB unreclaimable slab. A GPU with free VRAM but no
admissible host RAM is not a runnable training slot. Do not restart services or
weaken memory admission to fill an activity graph. Recheck before each new job.

Dragon's last observed log reached epoch 28/30 at 22:12 UTC; first Traffic h96
score is estimated 22:20–22:35 UTC, subject to final evaluation. Omega's log is
buffered; its capped-clock run is estimated 23:00–23:30 UTC. The queued third
h96 seed follows dragon's first cell. Remaining Traffic horizons: 1–3 October
only if bounded-scoring admission and disk remain within measured limits.
Matched ECL training/score: earliest 3–7 October, conditional on a full-size
feasible cost path. Representation ablations and DOIN optimization: planning
window 7–14 October. A validated doctoral model in paper trading: planning
window 14–28 October; this is not a guarantee of superior accuracy or profit.
Real-money execution has no ETA and is outside this paper-only spine.

## Next update rule

For each completed cell: report model and same-row naive in identical metric
space, published value where truly comparable, full population, training cost,
artifact identity and score status. For each machine: observed job and memory/
thermal state. If an ETA changes, revise this snapshot and the visual timeline;
do not leave a stale optimistic date in one of them.
