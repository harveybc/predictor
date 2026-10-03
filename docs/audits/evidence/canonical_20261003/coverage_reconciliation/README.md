# Canonical Coverage Reconciliation

This directory reconciles the three PS2 batches, their PS3-C joins and dossiers,
and the Lane E extractibility queue in `canonical_20261003`. It is a coverage
audit, not a final feature decision and not a global inventory statement.

## Regenerate

From the canonical predictor worktree root:

```bash
python docs/audits/evidence/canonical_20261003/coverage_reconciliation/reconcile_coverage.py \
  --repo-root . \
  --output-dir docs/audits/evidence/canonical_20261003/coverage_reconciliation
python -m unittest discover \
  -s docs/audits/evidence/canonical_20261003/coverage_reconciliation -v
```

The script uses only Python's standard `csv`, `json`, `gzip`, `hashlib`, and
`unittest` modules. It reads retained manifests, tables, joins, dossiers and
queue artifacts; it does not invoke model code, metrics, data services, or GPU
work. Regeneration fails closed if source identities, denominators, state
cross-checks, or omission reasons do not reconcile.

## Outputs

- `coverage_by_batch_feature.csv`: one row per admissible `(batch, feature_id)`.
  The three `rung*_state_counts_json` columns report dossier states across the
  14 target/horizon cells. `outside_join_reason` and
  `extractibility_omission_reason` make every omission explicit.
- `coverage_reconciliation.json`: the same feature rows plus population and
  rung summaries, interpretation notes, and SHA-256 digests for every input
  file read by the generator.
- `reconcile_coverage.py`: deterministic standard-library generator.
- `test_reconcile_coverage.py`: invariant and fail-closed tests.

## Interpretation and limits

The 87 admissible features outside the PS3-C join are all
`PROVISIONAL_LOW_PRIORITY` in all 14 PS2 cells. Their PS3-C dossiers exist, but
they are absent from the candidate payload; the source artifacts record the
triage, not a future-work queue. The 142 joined features outside Lane E split
into 132 tier-3-only features deferred by the queue rule and 10 calendar
conditioning variables not treated as extractor series. Neither state is a
permanent rejection.

`ASSOCIATION_REPORTED` is associational evidence. An identified rung-2 effect is
conditional on its declared assumptions and support; rung-3 output is
conditional on its declared structural causal model. These labels do not imply
unconditional causal truth. The separate event-episode dossiers are reported
apart from feature rows. The table preserves raw dossier states. The JSON also
shows the batch-report normalization: estimated states become `ESTIMATED`,
calendar `NOT_EVALUATED` states become `NOT_APPLICABLE`, and other
`NOT_EVALUATED` states become `NOT_IDENTIFIED`. For example,
`px.hours_since_prev_bar` has 14 raw `NOT_EVALUATED` states in batch 001, while
the normalized report places those 14 in `NOT_IDENTIFIED`.

All input SHA-256 values are embedded in `coverage_reconciliation.json`. The
CSV and JSON outputs are sorted by batch and feature identifier and contain no
generation timestamp, so identical inputs and code produce deterministic
outputs.
