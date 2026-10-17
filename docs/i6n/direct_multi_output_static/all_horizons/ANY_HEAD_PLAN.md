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
