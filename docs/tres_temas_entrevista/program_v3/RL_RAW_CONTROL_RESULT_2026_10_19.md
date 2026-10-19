# RAW control: closed, not selected-input comparison

Scope: EURUSD admitted core prices only; eight deterministic RAW inputs, not
selected winners. TRAIN2020-2023 and VALIDATION2024, seed7, USD10000 initial
capital. Both SAC arms used 100000 transitions, static inference, no refit,
early stopping, dropout, BatchNorm or explicit weight decay. Thirteen observed
bars at stride2 include the current completed bar. Costs are an analytical
scenario, not broker-verified costs. This is one seed, not an algorithm ranking.

## Measured Results

| Arm | TRAIN net USD | VALIDATION net USD | VALIDATION max drawdown % | Closed trades TRAIN / VALIDATION |
|---|---:|---:|---:|---:|
| SAC direct RAW | +699.66 | -366.67 | 3.8542 | 829 / 204 |
| SAC per-feature Conv1D / fusion / core RAW | +2275.59 | -496.18 | 5.0597 | 829 / 204 |
| No trade | 0.00 | 0.00 | 0.0000 | 0 / 0 |
| Business-constrained passive long | -1226.09 | -794.05 | 8.5694 | 801 / 197 |
| Business-constrained passive short | -2095.30 | -284.61 | 4.7283 | 805 / 195 |
| Random actions | -846.11 | -180.16 | 2.0023 | 829 / 204 |

All twelve full rollouts finish flat, with at most one filled entry per
business day; evaluated transition counts are 24869 TRAIN and6196 VALIDATION.
The SAC validation policy tensors are unchanged before/after scoring. Sharpe
in the retained summaries is the native Backtrader daily, nonannualized ratio;
undefined ratios remain null. Neither SAC is profitable in validation, and
both lose more than the random control in this single-seed scenario. The
TRAIN/VALIDATION gap motivates generalization tests; it does not isolate
overfitting from regime changes or prove temporal representations ineffective.

## Closed Publication And Copies

All12 reports delivered with exact warehouse readback, zero pending and zero
mismatches. The closed SQLite backup passes `PRAGMA quick_check` and the
second-host collector verified all82 copied artifact hashes. The collection
has `COLLECTION_COMPLETE`; the campaign has
`COMPLETED_AND_WAREHOUSE_READBACK_VERIFIED`.

- Contract: `b195273eb9f91cff23a735d3c72ee8ab95754dbf85c1c663fe46dc42550a30b7`
- Active agent code: `75f24688`; simulator: `461b594`
- Close receipt: `70c6a0a2dae705a175b737d7ff85a2849f9c7ae688655a81039c1ea601f27857`
- Physical metrics backup: `bc8626578f43af5a777f468a0cc5c6cd2c3fd5e8df8061b62b9d2a0d52767288`

Private deployment configuration owns storage locations. No data, credentials
or metric database are published in Git.

## Selection Remains Required

Financial selection on repaired candidate data is NOT complete. No new
selected-variable count or list is claimed. The next scientific dependency is
candidate admission and the prepared selector bundle, recomputed individual,
pairwise/redundancy and declared causal evidence, then an explicit selected
manifest and warehouse readback. A lack of causal identification is not proof
of irrelevance. Price-regression skill is not a mandatory RL feature gate.

Only then compare the SAME selected information directly in SAC versus
per-feature extractors, fusion and temporal core. No additional RAW campaign
is dispatched. Early stopping and regularizer studies remain separate,
prospectively specified experiments, not changes to these completed fits.
