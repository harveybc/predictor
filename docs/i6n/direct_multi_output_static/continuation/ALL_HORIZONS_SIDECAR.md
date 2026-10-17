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
