# Training audit: early stops and available data

Read-only findings on 2026-10-18. Sources: authenticated monitor-campaign
RESULT/TRAINING/PERIOD packets; original INPUT and continuation exclusions.
No change to the active campaign's source or scientific identities.

The stored epoch histories reconstruct exactly as the configured 50/50 mean,
the significant-improvement reference minus 1e-8, patience counter, and minimum
checkpoint selection. This checks implementation arithmetic; it does not prove
patience 10 is appropriate or that the selected checkpoint generalizes.

Representative non-BN short update, week 2024-01-28: 16409 gradient examples,
65 batches/epoch. At epoch 1, FIT monitor .9436054839, known-validation monitor
.9827066119, mean .9631560479. At epoch 11, FIT .9406918592, validation
1.0486096686, mean .9946507639 and stale=10. It restores epoch 1 after 715
gradient updates. These are normalized losses using the FIT naive denominator,
not the owner's full TRAIN_REPLAY performance. Its initial and restored tensor
hashes differ. The model inherited prior trained weights, so 11 refers only to
this adjustment, not its entire training history.

Across all 50 complete legacy short updates, 8 stopped at epoch 11 and the
range was 11-97 epochs; across all 50 complete legacy long updates, 2 stopped
at 11 and the range was 11-60. In 37/50 short and 38/50 long cases the FIT
monitor was lower at stopping than at the chosen checkpoint, while calibration
was higher in 49/50 in both families. This is compatible with overfitting to
FIT relative to the calibration week, but is not an annual TRAIN_REPLAY gap.

Every prepared four-year FIT was checked for row counts: short 16285-16530,
long 16414-16614. The known weekly monitor has per-head medians short
[94,93,92,91,90,84,78], long [78,72,60,48,36,24]; long h72 has as few as 6
valid pairs in a boundary week. A weekly monitor can therefore be variable.

The early-2024 FIT has no retained training origins in March-July 2023. The
INPUT source has 1851 timestamps in those months, with 395 nonfinite
fx.gbpusd.logret_1h values. Spacing-2 window admission gives 666
INPUT_VALUE_NOT_FINITE, 1175 INPUT_SLOT_NOT_OBSERVED, only 10 INPUT_OK before
combining label support and other resolution requirements. The common-support
policy checks longer windows/resolutions too. This is available coverage inside
a four-calendar-year span, not four complete years of usable examples. It also
means the one-year monitor has fewer available rows than a complete year.
This source limitation is shared with the original winning experiments and
does not alone explain the difference between calibration and forward scores.

BatchNorm is not a necessary cause of the failure: every completed non-BN arm
already loses to its own naive on forward VALIDATION. Some BN comparisons
improve individual heads and others worsen them. Neither early-stopping budget
nor BN alone has been demonstrated as the root cause of the forecasting gap.

## Required next diagnostic

`STOPPING_DIAGNOSTIC_20261018.md` defines six matched probes with patience 50,
max_epochs 2000, no BN and unchanged donors/data/monitors. Compare retained
epoch prefixes and the next week's performance; do not infer benefit from more
epochs alone. They run automatically after their parent GPU family completes.
Source coverage correction needs an explicit new data identity and must not
silently fill closed-market or absent open-market bars. Full TRAIN_REPLAY is
still pending; this audit does not supply it.
