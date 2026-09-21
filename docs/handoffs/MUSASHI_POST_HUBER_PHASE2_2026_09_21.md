# Satoshi: phase-2 preparation and mandatory financial loss/optimizer design

Updated 2026-09-21 after the owner's final clarification. This is the current
handoff, replacing the earlier version that had not yet been delivered.
Read and incorporate [FIN-LOSS-OPT](../tres_temas_entrevista/program_v3/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md).
No further old OLAP/branch search and no repeat of heuristic-strategy/noise sweeps
as a prerequisite. The household result does NOT select a trading loss.

Read [the measured comparison](../audits/work_plan/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md)
and its full table first. Musashi executed this comparison; do not repeat it.
RP57-RP64 stays preserved. The task below is preparation and verification, not
a licence to run an unreviewed factorial or to change the doctoral hypotheses.

## Deliver one complete preparation packet

1. Incorporate the measured four-arm comparison into the work plan and correct
   the two overstatements: RP63 jointly changed loss and monitor; shuffled-label
   performance does not bound all learnable advantage. Keep historical labels
   and results; publish corrections beside them rather than rewriting evidence.
2. Freeze the phase-2 questions separately: calendar information, daily context,
   then training volume. Keep architecture, training recipe, forecast task and
   validation population constant wherever the question requires it. Use
   MAE+Adam only as the household continuity reference; no trading winner is established.
   Retain Huber+AdamW's measured RMSE tradeoff in the candidate registry.
3. Calendar: specify civil time, timezone/DST handling, sampling and publication
   assumptions; hour/weekday sin/cos and any other feature must be known at the
   decision. Test the real loader and transformation against future changes,
   missing timestamps and schema/role mistakes. No target or post-outcome phase.
4. Daily context: inspect `core_dilations(window)`. A longer window currently
   changes core depth/dilations, hence information and model simultaneously.
   Design an information-only control with matched architecture/capacity and
   measured reach, or declare and factorially separate both changes. State exact
   past sample identities/availability for daily lags and delayed observations;
   do not assume a 24-hour period without train-side evidence. Additional input
   channels can also change parameter count: show that control explicitly.
5. Volume: count unique raw rows, distinct windows, labels and repeated exposure
   separately. Grow history backwards while fixing evaluation where possible.
   If there is not enough prior history, declare a successor task and common
   evaluation before any score. State how train-only scalers and a fixed Huber
   delta in standardized units would change with sample size; do not hide a
   changed raw-unit objective inside a purported volume-only comparison.
6. Training adequacy: seven current fits hit 4,000 updates. Specify learning
   curves, validation cadence, patience and restore checks, compute ceilings and
   what will happen if the best checkpoint is near the ceiling. More windows
   under a fixed update budget are not automatically more training. Keep
   finite-budget comparisons separate from claims of convergence.
7. Define acceptance tests first, then implement the necessary runner changes.
   Prefix invariance must pass through production feature paths, and deliberately
   leaky controls must fail. No wavelet/STL full-series convenience path. Preserve
   governance before consumption, observed update counts, per-cell artifacts,
   outbox handling and exact warehouse content verification.
8. Produce the executable design, exact population, pilot-cost plan and resource
   assignment. If multiple hosts will train, place complete paired arms within
   host/seed blocks so optimizer or feature choice is not confounded with CPU.
   Do not repeat cells merely to occupy every machine. Only cheap acceptance
   fixtures are allowed now, not the phase-2 scientific training campaign.
9. In parallel, design FIN-LOSS-OPT for the actual financial forecasting task:
   MAE/Huber x Adam/AdamW, paired initializations and rows, explicit versioned
   defaults plus equally budgeted DEV tuning. Justify Huber delta from train-side
   residual scales and LR/decay candidates from architecture, data and training
   duration; scaling literature is a hypothesis, not an optimal-value formula.
   Include compact and business-relevant larger receivers, not a universal
   million-parameter requirement. Freeze short/long horizons and weekly folds
   from the task; the recalled 6h/~3d and four training years are not exact
   recovered configs. No final recipe from one electricity task or one delta.
10. Prepare acceptance FL01-FL08 before code. Verify the same MAE_z evaluation
    scale for every arm, naive on identical rows/horizon, independent float64
    metric recomputation and round-trip preservation of 1e-5/1e-6 differences
    through arrays, terminal and warehouse. Training/inference dtype and measured
    numerical variation remain explicit. Check early-stopping min_delta does not
    hide the intended resolution. Include temporal-block uncertainty and tuning
    multiplicity; no automatic economic-irrelevance threshold for tiny effects.
11. Deliver one combined packet: updated master/state, financial design and
    test traceability, phase-2 design, exact resource/precision plans and next
    execution conditions. FIN-LOSS-OPT stays NOT_STARTED until its campaign runs;
    designing it is not evidence of a winner. Before future financial R0/R1/R2
    contrasts, freeze the chosen recipe and use it equally in method/controls.
    Do not repeat the completed household factorial; do not infer profits from
    a tiny error improvement alone. No heuristic-strategy rerun is required here.

Report all points together, without requesting renewed permission for reading,
designing or testing. Stop only for a concrete unresolved data/causality or
resource constraint; identify the missing fact rather than assigning an
engineering task to the owner. R0/R1/R2, reserved confirmation and live trading
remain outside this preparation block.
