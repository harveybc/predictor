# FS4 elapsed-hour support: integration order

Published implementation: `tools/fs4_hourly_support.py`, predictor and weekly
wrapper integration, and `tests/test_fs4_hourly_support.py`. This is TRAIN-safe
code on `codex/fs4-hourly-grid-20261007`, not a deployed weekly campaign.

## Observed defect

The runner encodes 24 positions on a 3600-second grid. The weekly predictor
builds RAW windows from 24 dataset rows. For four-hour ETH bars these cover
about 96 hours, not 24 hours. Moving the RAW window end to the last row at or
before `origin - 3 hours` does not repair the difference in receptive field.

## Required integration before VALIDATION

1. Merge this branch into the weekly branch after preserving its live edits.
   The active weekly branch already has uncommitted `RAW_LAG3` changes in the
   predictor and wrapper; do not overwrite either side. Route both RAW and
   RAW_LAG3 through `hourly_windows` using the
   sorted decision timestamps. RAW ends at the origin; RAW_LAG3 ends at
   `origin - 3 * 3600`. The weekly branch's current `window_end` uses time for
   the endpoint but still takes 24 preceding rows: remove that row-count
   windowing. Neither mode counts bars to construct its window.
2. Bind the TRAIN-fitted mean and scale, the 3600-second grid, 24 elapsed-hour
   steps and interleaved value/observed-mask channels into the predictor's
   architecture and input identities. The grouped branch has two input
   channels per feature. Do not silently apply the old one-channel weights.
3. For fit windows, pass the authenticated rolling fit start as
   `min_timestamp`; no 24-hour input may cross that boundary. Validation
   predictions may use earlier observed history, but never a row unavailable
   at their origin. Keep the same scored origin rows and naive for all arms.
4. Add a wrapper-level test on four-hour rows: RAW and encoder both receive
   24 hourly positions; RAW_LAG3's newest observed ETH bar is at origin-4h,
   while RAW sees origin. Test a weekend gap, a missing value, train-only
   standardization, future perturbation and a too-early fit origin.
5. Version the changed input/model identities. Re-run the TRAIN-only cost
   pilot for the new RAW input shape and caps. The previous pilot remains a
   historical cost measurement for the previous version, not an admission
   estimate for this one. Do not open VALIDATION until the integration and
   cost pilot pass. FS4 extractibility workers continue unchanged.

The 24-row contract is useful for old predictor tests and may remain as a
separate legacy helper; it must not be used by FS4's hourly comparison.

Verification on this branch: 38 CPU tests pass; a Keras forward smoke and
two determinism/permutation training tests passed in dragon's TensorFlow
environment with GPU disabled and governed memory caps. No VALIDATION or TEST
data was opened for this correction.
