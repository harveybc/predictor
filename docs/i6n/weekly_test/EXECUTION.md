# Weekly TEST Execution

The source extension retained all 35,398 prior observations unchanged and added
6,225 bars from 2025. Prices replay exactly; selected feature replay agrees at
float32 input precision, with float64 transcendental-rounding differences up to
8.89e-16, explicitly recorded rather than substituted into retained history.
Original producer commit: 42272ad7f288d39d5b9f84c63bb219ef0b9a7b19.

Data-only regression replay matches every array and scale exactly on three
retained short weeks, two supported long weeks, and the original empty last
long week. No 2024 fit was repeated. Preparation produced 106 logical units:
51 supported weekly fits and two explicit no-label weeks for each family.

The source changes in continuation.py only label the evaluation role, progress
and output files as TEST; the training call is AST-identical. train.py, model.py
and continuation_models.py remain byte-identical to the retained recipe. The
market-grid extension requires an explicit authorized end; its default still
refuses TEST. Historical campaign source-pin tests intentionally reject this
new code identity and were not relaxed to admit it under an old campaign.

## Unattended Operation

Configure private paths using the existing deployment and warehouse profiles;
never commit that file or its EnvironmentFile. Then:

```bash
python -m tools.i6dmo.weekly_test_source --spec <private-source-spec.json>
python -m tools.i6dmo.weekly_test prepare --spec <private-campaign-spec.json>
python -m tools.i6dmo.weekly_test launch --spec <private-campaign-spec.json>
python -m tools.i6dmo.weekly_test status --spec <campaign-root>/RUN.private.json
```

Launch freezes code/data identities and enables bounded user timers. Each GPU
invocation executes at most one weekly fit. Short and long alternate, a second
heavy GPU process is refused, and no agent needs to poll. The status command is
read-only; generated status includes completed fits, failures, the active epoch,
publication readback and ETA measured from completed fits. At annual closure the
coordinator pools paired error sums, publishes metrics, verifies both mirrors
and stops only this campaign's timers. Interrupted/failed units remain explicit,
never silently counted as complete. No new hyperparameter search is authorized.

Results are TEST_2025_CHECKPOINT_TUNING_NOT_INDEPENDENT_OOS because the evaluated
week participates in early stopping, exactly as in the owner-requested recipe.

## Start Check

Deployment e0af8f23 is armed. The first real short fit reached epoch 2 after
matching the retained initial tensor digest. The two boundary no-label units
were automatically skipped and delivered with exact warehouse readback. Both
campaign timers are active. The unrelated secondary-GPU campaign continues;
the second GPU sharing RAM with the preferred device is intentionally unused.
The first deployment attempt stopped before installing timers because its
remote private directory was absent; the launcher was corrected, that attempt
was retained, and no fit ran under the earlier deployment commit.
