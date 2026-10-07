# Encoder arm v2: origin-covering phase (specification and cost; NOT implemented, owner decides)

Status: proposal only. The phase-4 runner (feature-extractor @deffa53) and every terminal it produced are untouched.

## Why
The runner's strided causal encoder 24 -> 12 -> 6 is trained at the EVEN phase: output j of a stride-2 layer reads rows
2j-2..2j. Its last latent step therefore reads rows up to origin-3 (3 hourly rows; see `docs/fs4/STAGE2_RULE.md` and
`tools/fs4_encoder_alignment.py`). That is sound (no future read, weights used as trained) but lagged: RAW sees the origin row,
encoder arms do not. The ODD phase (outputs at t = 1, 3, ..., 23) covers the origin, but applying the EVEN-trained weights there is an
untrained use and is not certified. The only way to an origin-covering encoder with trained weights is to retrain.

## Specification (one change to the architecture, same budget and identities)
- Arm names `TRAINED_ENCODER_V2` and `RANDOM_ENCODER_V2` (new task kinds, separate warehouse rows; v1 rows are never overwritten).
- In `build_models`, replace each strided layer `Conv1D(F, k, padding="causal", strides=2)` by
  `ZeroPadding1D((1, 0))` followed by `Conv1D(F, k, padding="valid", strides=2)`: output j then reads original rows 2j-1..2j+1, the
  last output covers the origin (rows 21..23 at the first level), the layer stays strictly causal (no row after the origin).
  The decoder (6 -> 12 -> 24 upsampling) is unchanged; the hidden-point scoring is unchanged; early stopping, mask derivation,
  rows, seeds and digests are unchanged except for a new `architecture.id` (`fs4_causal_conv_24_12_6_oc_v2`).
- Acceptance: the empirical support of the last latent step is exactly rows 15..23 (lag 0) by the same perturbation test; a replay
  reproduces each v2 terminal; the certificate rule is the same structural rule with alignment `ODD_PHASE_ORIGIN_COVERING` trained
  as such.

## Cost (measured on the 13-14 completed v1 TRAINED terminals, dragon RTX 4090, one stream)
| | tasks | mean wall per task | total |
|---|---|---|---|
| TRAINED_ENCODER_V2 EURUSD | 355 series x 5 folds = 1,775 | 125.6 s (median 133.9 s), 165 cpu-s, 0.34 GB VRAM | about 62 GPU-hours |
| TRAINED_ENCODER_V2 ETH | 78 series x 5 folds = 390 | 52.4 s (median 45.9 s), 76 cpu-s | about 5.7 GPU-hours |
| RANDOM_ENCODER_V2 (control, no optimizer) | 2,165 | seconds each | under 2 hours |
Total about 68 GPU-hours single stream; the VRAM footprint allows several concurrent streams on one 4090, so roughly 12-25 hours of wall
time depending on concurrency, plus a runner change that must be reviewed and pinned before any task starts. The weekly stage 2
for the v2 arms would then need its own certificate and a fresh frontier-bound plan (the extractibility closure digest changes).

## Cheaper option for the owner (also not implemented)
A `RAW_LAG3` control: the weekly RAW predictor fed windows that end at origin-3. It isolates the encoder's effect from the 3-row lag
without any retraining; cost is one more stage-2 input mode on the same list (about the RAW stage-2 cost, no encoder weights).
