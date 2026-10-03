# Retsu: M1 weekly forecast scoring and naive gate

This is an independent CPU lane alongside the current PS3-R/PS4 jobs. It does
not authorize a model training run or any read of the external test year. Keep
the active Gamma jobs untouched.

## Authority and start checks

Fetch `origin/satoshi/canonical-exec-20261003` and create a fresh worktree from
its tip. Read, in order:

1. `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md`;
2. `docs/tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md`;
3. `docs/tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_TRACEABILITY.json`;
4. this order.

The gap is specific: `tools/business_weekly_training.py` produces model
releases, while `tools/business_objective_firewall.py` accepts weekly metric
observations; neither currently connects predictions to targets and a paired
naive inside the weekly traversal. `anti_naive_lock` is not that integration.

## Deliverable

Implement and integrate one fail-closed scoring boundary for a forecast release
on one `WeekSpec`. It must accept explicit horizon, ordered origin identities,
target, prediction, naive prediction, unit, scale, model digest, and dataset
identity. Reuse existing forecast metric helpers where their semantics match;
do not duplicate a different definition of MAE/MSE.

For each horizon it must:

1. require unique, strictly increasing origin identities;
2. require prediction, target, and naive arrays to have the exact same length
   and bind all three to the same origin digest;
3. reject missing, reordered, duplicated, non-finite, or mismatched rows before
   computing any metric;
4. report model MAE/MSE, naive MAE/MSE, sample count, horizon, scale, week,
   release digest, and paired per-origin absolute-error difference;
5. derive skill from those values and set `strategy_eligible=True` only when
   the contract's primary error metric is strictly lower than the same-row
   naive at every required horizon;
6. never invoke the strategy callback on a failing, missing, or incomplete
   forecast family; retain the reason and the weekly disposition;
7. aggregate weekly results only after every expected week has an explicit
   terminal disposition. Do not treat overlapping hourly origins as
   independent replicates.

Use the contract's configured primary metric. If the existing business task
does not yet name one, emit `NOT_CONFIGURED` and no strategy eligibility; do
not silently choose MAE or MSE. Short and long forecast families remain
separately identified and both are required when the declared strategy consumes
both.

The test firewall must still be closed during validation selection and open
only for the sealed test traversal. This lane may exercise the firewall with
synthetic arrays, but must not open or read the real test data.

## Test-first acceptance matrix

Add focused tests before implementation. Each test must call the real scoring
boundary; do not replace the function being tested with a mock.

| Case | Required result |
|---|---|
| Same finite rows, model better than naive at all configured horizons | exact counts and metrics; eligible |
| Model ties or loses naive on one horizon | ineligible; callback invocation count remains zero |
| One origin missing, duplicated, reordered, or changed in one input | reject before metric output |
| Prediction, target, naive lengths differ | reject |
| NaN or infinity in any array | reject and name field/origin |
| Different unit, scale, horizon, model digest, or week identity | reject before aggregation |
| One of the two required short/long families absent | incomplete weekly disposition; ineligible |
| A week fails or is excluded | preserve its disposition; annual close refuses a reduced denominator |
| Test payload presented during validation | firewall rejects before rows are accessed |
| Serialize, restart, resume | completed week is not rescored and retains identical digest |

## Scope and execution

Write only the new scorer, its focused tests, and minimal integration in the
weekly business modules. Update `BUSINESS_WEEKLY_TRACEABILITY.json` only for
requirements whose full behavior is now demonstrated; software tests do not
turn a model or business result into scientific evidence.

Run the focused tests under one CPU-only governed scope on omega:

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 900s -n m1-weekly-score-gate -- \
  python -m pytest -q <the-new-focused-test-module>
```

If 2 GiB admission is denied, stop and report the denial. Do not use the 4070,
do not retry with a smaller evidence population, and do not interrupt the
desktop. No GPU is needed for this software lane.

## Return

Commit and push from the fresh branch. Report the exact commit, files, tests,
which BW traceability IDs changed and why, and explicit confirmation that no
model training ran, no external test data was read, and Gamma PS3-R/PS4 jobs were
not touched. If the primary metric is still unspecified, leave the gate closed
and name that exact remaining business-contract field.
