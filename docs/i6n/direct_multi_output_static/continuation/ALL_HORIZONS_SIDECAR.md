# Exact Historical Short Recipe, Twelve Outputs

Owner request 2026-10-16: use the idle secondary 4090 for one independent
diagnostic while the unchanged 407-cell continuation stays on the 5090.

Authority is the plan on branch `codex/i6dmo-short-all-horizons-20261016`,
`docs/i6n/direct_multi_output_static/all_horizons/PLAN.md`, runtime `1601e943`.
That branch starts directly from the original short runtime `24f5a6cc`.
No model-building or trainer function is rewritten. The model registry admits
the twelve-head tuple and otherwise builds exactly the historical network.

Retain all 19 RAW inputs, 13 MG2 observations including t, branches Conv1D(4),
core Conv1D(8)->Conv1D(4), original physical-time encoder, Dense(8)->Dense(1)
per head, original AdamW settings and seed 6102026. Original FIT/INNER stopping
is deliberately retained for this historical-recipe experiment only. It does
not replace the current continuation's TRAIN/VALIDATION stopping contract.
Parameters change 4,882->7,480 only from the six added heads.

The original supported short rows, labels, masks, input statistics and short
normalization scales must be identical. Append exact physical-hour targets
18/24/36/48/60/72, using the SAME calendar and business eligibility predicate.
There are not two business rules: eligibility differs per output because t+h
differs. Never intersect all horizons or drop short rows to satisfy 72h.
Each MAE uses its own horizon naive on exactly that head's valid rows.
Report an additional 12h/18h common-origin comparison to distinguish support
effects from comparing different operational cohorts.

Same initial shared and short tensors and same initial short forward are
tested. Adding objectives still changes gradients and early stopping: the
historical short scores need not remain identical after joint training.
The secondary GPU class is the owner-requested hardware difference; no
cross-device bitwise scientific result is promised. The historical runtime
seal mismatch remains disclosed, and no old result is resealed.

Automate preparation, one admitted fit, publication/readback and two mirrors.
No TEST, pretraining, DOIN or feature-selection changes. Artifacts stay on
the configured storage volume, not GitHub. Status is a single bounded read:

```bash
python -B -m tools.i6dmo.all_horizons status --spec <private-run-spec>
```

Exact source branch invocation: `launch --spec <private-launch-config>`.
The private spec and credential EnvironmentFile remain outside Git. Timers
own progression; agents do not periodically monitor it. Any interruption
after STARTED is a failed terminal, not an automatic retraining attempt.

## Closed Result

The single fit completed on the secondary: 25 epochs, best 15, 128.3243 s,
2,817,531,904 B cgroup peak. One report/145 metrics read back, 13 files verified
in each physical mirror, both sidecar timers disabled. Relative error is
+3.874222% at 6h, +6.176801% at 12h and +10.478196% at 18h; none of the twelve
heads beats its own naive. Historical short improvements did not survive the
added objectives. This is not attributed uniquely to negative transfer because
the requested hardware class also changed. The original short predictions,
input bytes/scales and naive supports were not altered.

An independent calendar-arithmetic audit, without the C1 evaluator, reproduced
all 150,680 FIT, 6,466 INNER and 43,444 DEVELOPMENT valid pairs and input windows.
46 model/extension tests plus a separate nine market-grid regressions passed.
Full per-head model/naive table and limits are in the source branch's
`all_horizons/RETURN.md`, public documentation commit `9a2cf387`. Terminal digest
`b70b5ee3f6023b571f6fd515cb30de611ae528f1c956631453a00722b7ee29c9`.
The preferred GPU continuation remains independent and running; do not repeat
the sidecar fit or open TEST/strategy on these ineligible predictions.
