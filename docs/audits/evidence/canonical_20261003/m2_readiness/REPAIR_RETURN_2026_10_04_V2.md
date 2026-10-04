# M2 PS3-R population authentication repair

Base: `e33d1e82e579614a4beeac9bcd8610d117b56143`.

## Finding reproduced

The ingestor previously accepted re-sealed `probe` and `probe_delta` rows whose
target or horizon was outside the terminal contract. It also inferred metric
coverage from the submitted rows.

PRE: 8 failed, 21 passed. The failures cover forged targets, horizon 999,
forged delta loss, missing required regression/classification metrics, and an
absent probe contract.

## Repair

`ps3r_ingest_config.json` now declares the trusted target domains:

- `Y_s`: horizons 0..5, metrics MAE/MSE, delta metric MAE.
- `Y_l`: horizons 0..5, metrics MAE/MSE, delta metric MAE.
- `Y_b`: horizons 0..1, metrics log-loss/Brier, delta metric log-loss.

The ingestor rejects an absent or malformed contract, verifies its arithmetic
against the declared row-kind counts, and requires exact equality with the
expected fold x family x target x horizon x metric populations. Missing,
extra, duplicate, non-finite, wrong-target, wrong-horizon and wrong-metric
cells reject before utility is computed.

## Verification

POST: 38 passed in the complete M2 readiness and PS3-R ingestor suites.

The three authentic alternative terminals remain adopted with their original
digests and `identity`, `random`, `past_to_current_siamese` families. Generated
readiness evidence remains unchanged: 279 `NOT_IDENTIFIED`, 87
`OUTSIDE_JOIN_PENDING`, and 366 `NOT_READY_EVIDENCE_INCOMPLETE`. No selection
decision was emitted and no GPU process or other worktree was touched.
