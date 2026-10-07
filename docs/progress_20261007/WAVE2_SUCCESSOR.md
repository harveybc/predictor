# FS4 scoped successor, 2026-10-07

## Why this exists

The first-wave FS4 manifest admitted 126 features and 1,890 feature/fold/arm
tasks. Its retained readback receipts authenticate 1,647 measured terminals;
243 cells (81 paired triples) have no TRAIN observations. This is not the
parent 6,495-task `EXTRACTIBILITY_COMPLETE` closure and does not select any
feature for trading.

Only four EURUSD candidate sets had all their members measured on all five
TRAIN folds, leaving most of its 14 targets without a weekly pair. The
successor uses Phase-3 TRAIN rankings and the scoped closure to choose two
distinct sets of at most 12 members for each of the 14 EURUSD and six ETH
targets. Sets with a member already known to lack TRAIN observations are
ineligible. The rule is sealed in `tools/fs4_target_coverage.py`; it added
eight EURUSD features and no ETH features. All other features remain deferred.

## Evidence and automation

`tools/fs4_preliminary_close.py` checks exact selected task population,
five folds, all three arms, terminal validity, paired identities and retained
warehouse readback receipts. Its `EXTRACTIBILITY_PARTIAL_COMPLETE` document
names the parent plan, pinned wave manifest, denominator and terminal digest;
it never impersonates the full-plan closure. The original first-wave closure
is retained at
`~/.local/state/canonical_20261003/fs4/preliminary/EXTRACTIBILITY_PARTIAL_COMPLETE.json`.

The second-wave manifest and report are retained as `WAVE2_V2_MANIFEST.json`
and `WAVE2_V2_REPORT.json` in that same state directory. The successor timer
`fs4-wave-successor.timer` invokes `tools/fs4_wave_successor.py` every minute.
`STATUS.json` under `preliminary/wave2_successor/` reports one of:

- `WAITING_FOR_WAVE`: a selected cell remains pending or leased.
- `COVERAGE_GAP`: a chosen set still lacks paired extractibility; no weekly
  queue is created. A new TRAIN-only wave must be sealed.
- `WEEKLY_RAW_READY`: all chosen sets were checked and the versioned weekly
  RAW queue was created. This status **does not** say workers are deployed.

The weekly frontier rule is in `tools/fs4_wave_frontier.py`. It requires all
three arms on all five TRAIN folds for every member of each pinned target set;
other consolidated sets keep an explicit deferred disposition. The weekly
controller accepts this partial closure only with the matching wave-seal
schema, intact digest and zero VALIDATION reads. The full-closure path and
its prior rule remain available and separately validated.

An isolated predictor checkout at commit `cf9901a6` is retained under
`fs4_weekly/code/PREDICTOR_WAVE2_cf9901a6` on all three hosts. Dedicated
worker SSH keys are forced to the weekly controller and queue; other commands
are rejected. Four RAW slot configurations are prepared, two per worker.
The successor loads the alignment certificate from this same checkout; the
older retained copy under `fs4_weekly/alignment/` has a different Stage-2
rule digest and must not be used for this campaign.
`fs4-weekly-activation.timer` on each worker waits for the weekly plan, then
runs the existing weekly host installer; it retries admission without reducing
caps. EURUSD small slots use the measured 1,700 MiB cap and ETH slots use
1,200 MiB. GPU FS4 work and weekly RAW work have separate code paths and
queues. Weekly activation is not evidence of a completed weekly fit.

## Next steps

1. Let the second-wave workers finish. Check
   `~/.local/state/canonical_20261003/fs4/preliminary/wave2_successor/STATUS.json`.
2. On `WEEKLY_RAW_READY`, inspect the worker admission receipts and the first
   real paired weekly terminals. The 2024 feature files on both workers were
   independently checked by timestamp: EURUSD begins in January 2024 and ETH
   spans January through December 2024. A file named `features_train.parquet`
   under `validation_2024` is a 2024 input; its name alone is not proof of
   its split.
3. Run 52 consecutive validation weeks per target with the declared rolling
   retraining window. Compare every set with target persistence on identical
   scored rows. Keep TEST sealed. Only after the RAW stage closes may the
   stage-2 rule pick encoder comparisons, with the origin-lag control named.

No reconstruction MAE, feature persistence baseline or causal
`NOT_IDENTIFIED` state is a stand-alone feature-selection decision.

## Verification

```bash
python3 -m pytest -q \
  tests/test_fs4_preliminary_close.py tests/test_fs4_target_coverage.py \
  tests/test_fs4_wave_frontier.py tests/test_fs4_wave_successor.py \
  tests/test_fs4_weekly_campaign.py tests/test_fs4_frontier.py
systemctl --user is-active fs4-wave-successor.timer
```
