# Any-Horizon Patience Contrast

Owner change 2026-10-16: reset the global ten-epoch patience counter whenever
ANY horizon improves its own best significant monitor by more than 1e-8.
Keep the 200-epoch ceiling. For each h, monitor
M_h = (FIT normalized masked MAE_h + INNER normalized masked MAE_h)/2.
This preserves the historical FIT/INNER protocol; there is no averaging across
horizons for patience. Compare against that horizon's significant reference,
not the previous epoch: oscillation around a worse score must not reset it.
Invalid or missing per-horizon support cannot reset patience.

AP01: retain the completed 12-head mean-stop fit as control, without rerunning
it. One new identity and one new R0 fit on the SAME secondary GPU, same seed,
12-head architecture, data/scales/masks, optimizer and gradient loss.
AP02: reset when any M_h < reference_h - 1e-8; update improved references only;
stop after ten consecutive epochs without any such improvement or at 200.
AP03: preserve lowest-average checkpoint restoration to isolate the stopping
change. Do not combine horizon weights from different epochs: shared-core
weights belong to one coherent checkpoint. Record every per-horizon monitor,
reference and reset, plus the selected checkpoint and actual stopping reason.
The average still defines gradient loss and checkpoint selection, NOT patience.
AP04: preserve all original data and actual TF2.21/Keras3.13.2. The completed
campaign and the 5090 continuation stay immutable. TEST stays closed.
AP05: existing automatic launch, outbox/readback, mirrors and closure apply.
Do not start an agent polling loop or repeat a scientifically started fit.

Tests defined before implementation: one head improves despite worse mean;
all heads worsen; subthreshold progress accumulates against the reference;
oscillation does not reset; nonfinite/unsupported vector rejected; ceiling
applies despite ongoing improvements; best-average checkpoint remains coherent;
default legacy behavior unchanged; real-history replay agrees through the old
run's first 25 epochs; actual data hashes remain unchanged.

## Launch Evidence

Runtime `b33a5933`: 48 focused tests passed in 140.66 s, including a real
two-epoch synthetic CPU fit through the vector stopper, monitor self-check
and the legacy trainer regressions. PRE had ten failures because the new
stopper/factory did not exist. This is not a full-repository suite claim.

Campaign `i6dmo-any-head-patience-20261016` is armed with autonomous GPU and
publication timers. Start check observed epoch 3 / 195 updates on the secondary
4090, explicit per-horizon references and reset indices, stale=0. Preparation
verified the model and all three data files byte-identical to the completed
12-head control. Population seal:
`5a39fc444d7382fc40c3d83a854fafa466758dfc620c0543384d704ddd986832`.

The effective new field `training.patience_reset=ANY_HORIZON` selects the new
rule. Original source-descriptive strings, including `training.patience_rule`,
are retained for source comparison and do not override that explicit selector;
the resulting early_stop identity is `i6dmo.any_horizon_patience.v1` and records
the effective vector rule. The mean monitor still selects the coherent best
checkpoint and drives the gradient objective. Neither is silently changed.

Independent arithmetic replay of retained histories (not extra training):
the short-only run stopped at 13 with mean patience, but any-head stale would
be 4 there. The twelve-head mean-stop control stopped at 25; its h3 monitor
improved at epoch 25, so any-head stale would be zero. These are evidence
that the new rule permits more training, not evidence of improved prediction.

Historical short-only fit: 13 epochs, best 3, max 200, patience 10, stop reason
`patience`. No DEAP/DOIN search or genetic generations generated that fit.
