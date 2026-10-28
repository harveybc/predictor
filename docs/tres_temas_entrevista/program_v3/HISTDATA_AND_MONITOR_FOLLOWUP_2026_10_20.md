# HistData alert resolved in scope; generalization monitor contrast deployed

Owner accepted the short/medium/long-term plan and supplied the other lane's
HistData clock/price warning. Read-only audit used exact admitted c80847bb and
candidate HistData c6628375, under the observed London-wall-minus-five clock.

2023H1:3111 admitted bars;2436 shared;675 absent from HistData. All four OHLC
fields match on every shared bar (zero mismatch cells). The previous clock
diagnostic differenced the two series separately before joining. After an
absent candidate observation that compares a multihour return to a one-hour
return. Feb-Jul2023 legacy correlation.82757 becomes1.0 on1579 genuinely
matched hourly pairs (703 other pairs had unequal/nonhourly intervals).
This refutes the stated widespread 2023H1 price disagreement on common rows;
it does not change the accepted source, target, selection or past results.

Across2020-2024:30251 common bars,865 candidate absences,16 isolated OHLC
mismatch cells outside2023H1, all retained by timestamp/source/field for scoped
review. Whole-sample matched-hour return correlation.99999838. No source is
declared universally perfect because another feed agrees. The correction is
to the interval diagnostic; no interpolation or invented bars were introduced.
Public handoff: heuristic-strategy codex/monitor-generalization-20261020,
docs/short_horizon/HISTDATA_INTERVAL_FINDING_FOR_SATOSHI.md. The other lane's
running151-feature selection was not modified or stopped.

Authenticated curve audit:20/20 final FULL modules chose checkpoints later than
minimum2023H1 loss. The mean monitor can improve through TRAIN while internal
validation worsens. This is measured objective disagreement, not proof that a
particular replacement improves2024 trading. No historical weight was reselected.

New authorized finite study: ten FULL fits, five common seeds, reference
M0=.5*(TRAIN_loss+INNER_A_loss), treatment
M1=M0+.25*max(0,INNER_A_loss-TRAIN_loss). All other model, business, cost,
window13/stride2 and BC+3DAgger parameters held fixed. Reference and treatment
fit fresh under the same code; zero penalty reproduces predecessor training
bitwise on a CPU native-fit witness. Patience10,min_delta1e-8,max1000epochs.
Internal selector2023H1, diagnostic2023H2, complete2024 development; TEST closed.

Runtime d41d588ed1989e43a43c5d62895e8564ca14080b; spec
30bfbe92010959b77a1048bcf4d74d55ddec1a109eed8dbdca0955686178b6ae.
Source gate pinned and checked before dispatch.52 focused CPU regressions pass,
17 subsequent parity/gate/interval tests pass locally and on the worker, plus
one authenticated-curve test. Actual GPU synthetic penalty fit/bitwise reload
passed. First reference cell is running in the preferred GPU; sequential
worker, outbox/readback and independent-copy timers enabled. No LLM polling.

Short-term remaining: this finite contrast and module/cost/quarter diagnosis.
Medium-term: limited window/resolution contrasts with receptive-field coverage;
native SAC specialization and critic/replay/exploration diagnostics. Long-term:
consume useful new input families from the other lane, point-in-time regimes
and adaptation cadence, frozen evaluation then paper trading. Those later
experiments are separately designed after the current outcome, not assumed to
have started merely because they are listed in the plan.
