# Weekly TEST: Predict Then Update

The binding owner specification is
[PROCEDIMIENTO_CANONICO.md](PROCEDIMIENTO_CANONICO.md). Read it before execution.

1. Load each family's last usable VALIDATION checkpoint and its scalers.
2. Infer TEST week n and seal its predictions before reading its outcomes.
3. After observing week n, score its predictions against its own naive per head.
4. Warm-start the four-year refit ending at week n's start. The original mean
   TRAIN/observed-week stopping rule selects a checkpoint for week n+1 ONLY.
5. Repeat chronologically with unchanged architecture, downsampling, windows,
   masks, seed and optimizer/stopping parameters. Do not randomly reinitialize.

The short model retains seven heads (2,3,4,5,6,12,18 h), and the long model
retains six (18,24,36,48,60,72 h). Nineteen selected features remain unchanged;
STLFSI4 remains excluded. Every horizon uses its own paired original-price naive
and error (MAE_h-naive_h)/naive_h, negative being better. Keep per-family
conservative support, scale fitting on TRAIN only, the original 13-slot/2-hour
open-market grid including t, and the +12/-12-hour business boundaries.

Use `tools.i6dmo.prequential_test`; operational commands are in
[PREQUENTIAL_EXECUTION.md](PREQUENTIAL_EXECUTION.md). The autonomous finite
timers retain weights/.keras/scalers/configuration/predictions, publish through
the durable outbox with readback, verify physical mirrors and close themselves.
One heavy process uses the preferred GPU; no concurrent use of the other GPU
sharing its RAM. The independent secondary-machine campaign is unaffected.

Acceptance is PS01-PS10 in the canonical procedure. Source reconstruction and
exact prior array/scale replay remain valid input evidence. The former
same-week checkpoint-tuning campaign is superseded and stopped; its measurements
stay in [EXECUTION.md](EXECUTION.md) as historical, not clean TEST evidence.
Inherited current-vintage/publication-clock and assumed-calendar limitations
remain disclosed. No independent source-clock certification is claimed for 2025.
