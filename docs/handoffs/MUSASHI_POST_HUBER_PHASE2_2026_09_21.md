# Satoshi: phase-2 design after the Huber/AdamW comparison

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
   MAE+Adam as the continuity reference; no new optimizer winner is established.
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

Report all points together, without requesting renewed permission for reading,
designing or testing. Stop only for a concrete unresolved data/causality or
resource constraint; identify the missing fact rather than assigning an
engineering task to the owner. R0/R1/R2, reserved confirmation and live trading
remain outside this preparation block.
