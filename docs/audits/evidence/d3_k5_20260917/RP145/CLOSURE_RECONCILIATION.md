# Protocol B closed: twelve of twelve cells verified

Closure run 2026-09-24 on WORKER_B's external RTX 5090 by UUID, from the repaired tool at `07140d03`, against the campaign
root the native continuation service filled. Cost: 1,422.01 s user + 734.26 s system = **2,156 CPU seconds**, 35:40 wall,
5.87 GB peak resident. With the 14,011 CPU seconds the fits consumed, the continuation allocation stands at 16,167 of 24,000.

Every cell verified with no problems, and **every replay is bit-exact**: `max_abs_prediction_difference` 0.0 on all twelve.

| T | mean MSE (SD over 3 seeds) | mean MAE (SD) | matched persistence z-MAE, same rows | seasonal-24 z-MAE | skill vs persistence | state |
|---|---|---|---|---|---|---|
| 96 | 0.125925 (0.000417) | 0.220958 (0.000517) | 0.9455 | 0.3258 | 0.7663 | VERIFIED |
| 192 | 0.143650 (0.000606) | 0.237970 (0.000343) | 0.9507 | 0.3237 | 0.7497 | VERIFIED |
| 336 | 0.152696 (0.000706) | 0.251345 (0.000641) | 0.9613 | 0.3427 | 0.7385 | VERIFIED |
| 720 | 0.178957 (0.001175) | 0.275313 (0.000917) | 0.9754 | 0.3733 | 0.7177 | VERIFIED |

Unweighted four-horizon mean: **MSE 0.150307, MAE 0.246397**.

## Comparability, stated before anyone reads the numbers as a match

The table's per-horizon status is OPERATIONAL_AGREEMENT under the frozen predeclared band, against the values sealed in this
design's `lock.published`, which are Table 9's. **Table 9's per-horizon searched lookback is unresolved.** Executing the
released L512 script does not resolve it, so this is the released recipe reproduced and verified, and it is NOT a claim of
exact Table 9 identity. The H96 closeness (0.125925 against a published 0.126) is not evidence that the lookback matched.

## A against B

Protocol A's twelve-cell mean is 0.161962 / 0.259662; protocol B's is 0.150307 / 0.246397. The difference is a
**multi-parameter RECIPE comparison** between the L96 Table 8 recipe and the released L512 script, which differ in lookback,
patch length, dropout, batch size, learning rate and epoch allowance together. It is not an isolated context effect and this
document does not attribute it to the lookback.

Device attribution: the campaign's children asserted the external 5090's UUID in-child, so B's training device is MEASURED,
unlike protocol A's twelve cells. Its replays ran on that same measured device.
