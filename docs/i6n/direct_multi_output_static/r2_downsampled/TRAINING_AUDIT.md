# Training Procedure Audit And First Bridge Proposal

Owner clarification, 2026-10-16: do not reopen feature selection or vendor-data
auditing as a prerequisite to this development contrast. Audit the actual
training procedure and propose carrying the short input recipe to long heads.
Existing source qualifications remain recorded; no independent TEST claim.

## Verified Training

`TRAINING_AUDIT.json` pins the four authenticated TRAINING.json files. A bounded
CPU pass independently replayed each epoch's per-head sums/counts, mean FIT/INNER
monitor, best epoch, significant-improvement reference, stale counter, stopping
condition, update count and restored checkpoint hash. All four pass. This is
retained-history verification, not a new forward replay or fresh training.

All four settings agree exactly: AdamW lr .001, weight decay .0001, batch256,
max200, patience10, min_delta1e-8; end-of-epoch mean FIT/INNER at equal weights;
best checkpoint restoration; no DEVELOPMENT checkpoint selection. No wavelet,
Hilbert, STL or other decomposition is introduced by this training path:
inputs are retained RAW feature values with FIT-only z-scoring and physical
time encoding. This does not assert how every upstream feature was produced.

R0 short and long have the identical initial weight hash:
`cf808dcb2c455ff8a33d7f9c7fbaf6f6f86c218821f8cca5fc96f0983f91c7b1`.

| Fit | Epochs | Restored epoch | Updates |
|---|---:|---:|---:|
| Short R0 | 12 | 2 | 780 |
| Short R2 | 12 | 2 | 780 |
| Long R0 | 22 | 12 | 1100 |
| Long R2 | 32 | 22 | 1600 |

The retained effective model differences are the input grid/context and output
horizons; fitting also has different supported rows, FIT scale values and
target distributions. The gap is real in the recorded metrics, not evidence
that the training code itself is defective or that short forecasts are already
useful trading signals.

## First Proposed Experiment: Short Inputs, Long Targets

Keep R0, 19 selected RAW features, 13 observed slots at 2 open-market-hour
spacing spanning 24 open-market hours, t included, actual UTC offsets/closure
flags, branch Conv1D4, core Conv1D8->4 and six Flatten->Dense8->Dense1 heads.
Predict the retained LONG horizons 18/24/36/48/60/72 elapsed hours, still issued
hourly. Keep static training, seed6102026 and the verified optimizer/stopping.
Only the targets/head labels change relative to the short recipe; their
business masks and FIT-only target scales must change consistently as well.
Do not import a short head under a long-horizon identity.

For comparison with the old long recipe, freeze common origins/head masks,
labels and one shared FIT-only input scaler before fitting. Input sequence and
its physical-time encoding are the treatment; no smoothing, denoising, feature
reselections, R2, capacity search, DOIN or cadence change. Reuse an existing
control only if its full identity/support/scaler contract remains admissible;
otherwise a fresh matched reference needs an explicit pre-fit justification.
This proposal is not a runnable seal and no fit has been launched.

If this small-input recipe helps, distinguish context length from spacing in
the next controlled comparison (24h/2h, 144h/2h, 144h/12h), then consider 4h/8h.
Do not open that wider sweep before the first result. Quarterly cadence stays
later. TEST and strategy promotion retain their existing gates.
