# Fixed-Budget Temporal SAC Exploration

Owner approved immediate execution2026-10-20. This stage reduces expensive
monitoring while investigating generalization; it does not establish that
overtraining has been excluded. The earlier temporal run was deliberately
stopped with five full pairs retained and checksum-mirrored as partial evidence.
No historical result is deleted,reclassified as complete or reused as a donor.

## Exact Current Recipe

The same15 selected features,13 observations,stride2 observed hourly market
bars including t. Per-feature causalConv1D2 filters,kernel3; fused causal
coreConv1D4 filters,kernel3; hidden[8] native SAC actor and critics. Same
seed7,TRAIN2020-2023,full VALIDATION2024,business mask,one entry/day,
mandatorySL/TP,relative balance budget,leverage50 and declared nonzero costs.
No dropout,BatchNorm,L2,refit,pretraining,expert labels,DOIN or TEST.

Train100000 transitions. Full frozen-policy TRAIN/VALIDATION only at
25000,50000,75000,100000; no patience,no metric-driven stop,no intermediate
best-checkpoint restoration. Final100000 is the result even if an earlier
point looks better. The last observation occurs after the final SAC gradient
update,not in SB3's pre-update step callback. Native terminal reporting still
adds one full TRAIN/VALIDATION pair; do not omit that cost from the ETA.

Plots generated automatically after each complete pair show both daily mean
log-growth curves,gap,validation return/drawdown and trades. TRAIN improving
while VALIDATION deteriorates is descriptive evidence compatible with
overtraining,not proof or an automatic new selection rule. Preserve validation
dates and compare daily units,not four-year versus one-year cumulative profit.
The old combined M remains diagnostic only,not an optimization target here.

## Verified Execution

`agent-multi@ec4855ce4337fc4dbe1cc9d72f05256b0999923b`,contract
`fc9c0c532d43d541df4023ab2eeeda8fa63396ea80b938476c84da53da62d68f`.
144 local tests and25 actual-worker tests passed. GPU admission16 updates with
exact reload passed on attempt1. New native training started07:26:43UTC and
reached10000 steps in225.7434s on the5090. One heavy process group,zero
restarts; no parallel5070Ti fit. Effective config delta versus prior temporal
contract is solely `paired_early_stopping`; arms and selected input binding match.

Worker,publisher/readback/curves/close and second-machine collector run on
timers. Sparse monitoring means zero expected reports before the first full
pair,not zero progress. A failed overlay route was diagnosed as key expiry;
existing-host-key-authenticated LAN restored access and warehouse forwarding
(`/healthz`200),without changing scientific settings or authentication policy.
The overlay key itself still requires normal renewal separately.

Four pairs cost about25min instead of126min for20 at the measured375-381s
per pair. Provisional overall60-80min from startup,including training and final
reporting; refine from measured progress. This is not a completion guarantee.
Trading results must retain TRAIN/VALIDATION profit USD/%,daily nonannualized
Sharpe,max drawdown,closed trades,win rate and profit factor,with identities.

Implementation,traceability and status command:
`agent-multi/docs/work_plan/51_FIXED_BUDGET_TEMPORAL_SAC.md`.
No agent log-polling or repeated owner approvals are needed for this run.
