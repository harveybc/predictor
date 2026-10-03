# PS3-C Fail-Closed Reanalysis

Observed: 2026-10-03. Review implementation: `causal-inference@48ae17c`.

## Scope

This is a read-only review of the retained PS3-C summaries joined to their
retained PS2 candidate populations. It did not rerun feature construction,
estimate causal effects, refit nuisance models, or read a test split. The
historical batch files remain unchanged. Each new join records the PS3-C input
digest, PS2 identities, and reviewer revision.

## Result

| Batch | Candidate features joined | Historical rung-2 `ESTIMATED` family | Batch-local BH q <= 0.05 | Nonlinear confirmations run | Current identification survivors |
|---|---:|---:|---:|---:|---:|
| batch_001 | 46 | 450 | 32 | 0 | 0 |
| batch_002 | 214 | 375 | 46 | 0 | 0 |
| batch_003 | 19 | 251 | 31 | 0 | 0 |
| **Total** | **279** | **1,076** | **109** | **0** | **0** |

`BATCH_LOCAL` is the only multiplicity scope computed by this review. The 109
rows are not causal findings: none passed the current identification gate, so
none was sent to nonlinear confirmation. The gate requires a supported common
population and estimand, balance within the frozen bound, overlap without
trimming, and declared assumptions with evidence references. The retained
summaries do not carry the required identity/evidence; the gate therefore
returns `NOT_IDENTIFIED` rather than borrowing status from an older report.
Rung-1 associations remain associations. `NOT_IDENTIFIED` is not a feature
rejection.

An evidence reference is a traceability requirement, not automatic proof that
an identification assumption is true; domain review and sensitivity analysis
remain necessary.

The three batches cover 279 of 366 admissible features. The remaining 87 were
`PROVISIONAL_LOW_PRIORITY` in the PS2 artifacts; the coverage ledger retains
them as pending rather than silently treating them as rejected.

## Outputs

- `batch_001/ps3c_join.json` SHA-256: `0b4f713fa28a94a1eef65b89981c9d9948e496d0beb31970548158e5f3ec4891`
- `batch_002/ps3c_join.json` SHA-256: `8cf03ab71ce6982feefacb9ca3a17c697b37006ed0ae4db4ed05575f1288aec1`
- `batch_003/ps3c_join.json` SHA-256: `ed6cfd0701147afad497f58a8a05b0beb74f3acb146c1fdd4066d1f600f822eb`

## Reproduce the review

From the causal-inference checkout containing commit `48ae17c`, set
`PREDICTOR_EVIDENCE` to the predictor evidence root and run once per batch:

```bash
PYTHONPATH=provider/src python -m causal_inference_provider.ps3c_review \
  --ps3c-batch "$PREDICTOR_EVIDENCE/laneC/batch_001" \
  --ps2-batch "$PREDICTOR_EVIDENCE/laneB/batch_001" \
  --lane-a-batch "$PREDICTOR_EVIDENCE/laneA/batch_001" \
  --out-join "$PREDICTOR_EVIDENCE/laneC/reanalysis_48ae17c/batch_001/ps3c_join.json" \
  --revision 48ae17c
```

For batches 002 and 003, pass the corresponding batch directories and add
`--base-batch "$PREDICTOR_EVIDENCE/laneA/batch_001"` as used for the retained
extension batches. The outputs are additive; do not overwrite the historical
`laneC/batch_00N/` reports.
