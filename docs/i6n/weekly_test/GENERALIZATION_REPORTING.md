# Generalization first: mandatory reporting

Owner directive, 2026-10-18. Forecasting priority 1 is reducing overfit and
demonstrating useful generalization under the business protocol. Priority 2 is
improving predictive performance. Small train/evaluation gaps alone do not confer
eligibility: a constant/naive model may have no gap and no predictive skill.

Every result/status containing model performance must show, per horizon:
TRAIN MAE and its own naive-relative error; VALIDATION MAE and its own
naive-relative error; explicit calibration-versus-forward role; the gap;
sample counts, checkpoint identity and data periods. TEST is a separate column,
never silently renamed VALIDATION. Unsupported cells are null, never zero.

## TRAIN performance: owner clarification

The reported TRAIN performance is NOT the static MAE of the examples used for
gradient updates. The required sequence is:

1. Perform the initial training with the configured TRAIN/known-VALIDATION mean
   early stopping. Save the selected weights, scalers and complete configuration
   as an immutable initial checkpoint.
2. Load that checkpoint into a separate TRAIN-evaluation trajectory. Traverse
   the declared TRAIN period chronologically, week by week: predict and seal,
   score, then warm-start the configured weekly update for the following week.
   Keep each week's predictions; never replace them with post-update predictions.
3. Pool those scored TRAIN predictions over the entire declared period to
   obtain MAE_TRAIN_REPLAY and its own naive per horizon. It is not the result
   of one week and not the in-fit training loss.
4. Report the separate VALIDATION weekly trajectory and its own per-horizon
   metrics alongside TRAIN_REPLAY. TEST, when evaluated, is separate again.

For a controlled comparison, freeze before execution whether VALIDATION starts
from a separate copy of the same initial checkpoint or from the final TRAIN
trajectory checkpoint. This choice must be explicit in the campaign identity,
not inferred after seeing results. Do not silently change the already executed
VALIDATION/TEST runs to another initialization or relabel them as this new run.

TRAIN_REPLAY starts from weights already trained on TRAIN. Its scores are
retrospective adaptive diagnostics, not independent out-of-sample performance,
even when later weekly predictions are sealed before each subsequent update.
Preserve that qualification without prohibiting the owner's diagnostic.

## Temporal roles

- FIT: rows/loss used for gradient updates inside an individual fit.
- TRAIN_REPLAY: the owner's reported TRAIN performance from the complete
  post-training weekly replay described above; never substitute FIT for it.
- VALIDATION_CALIBRATION: known rows used to select an epoch/recipe; performance
  on these rows is not an independent forecast of the next unknown week.
- VALIDATION_FORWARD: the following unknown week in the 2024 development stream,
  predicted and sealed before that week's labels guide any update.
- TEST_FORWARD: the same chronological procedure on 2025; already-observed
  weeks may guide later updates, never replace their own sealed forecasts.

Within a weekly trajectory, predictions, masks and scalers must be bound to the
actual inference donor. A prequential RESULT combines TEST/VALIDATION forecasts
from that donor with FIT/OBSERVED metrics from the later update: these are not
the same weights and FIT is not the required TRAIN_REPLAY measurement.
Across TRAIN_REPLAY and VALIDATION trajectories, compare the frozen learning
procedure, initialization policy and cadence. Do not demand equal weights after
their different observed weeks have caused different updates.

## Metrics

For each set S and horizon h, with identical model/naive masks and price units:

```
R_S,h = MAE_S,h / naive_S,h - 1
raw_error_growth = MAE_eval,h / MAE_train,h - 1
normalized_gap_pp = 100 * (R_eval,h - R_train,h)
normalized_error_growth = (1 + R_eval,h) / (1 + R_train,h) - 1
```

Do not divide R_eval by R_train: their zero/sign changes make that ratio
misleading. Zero training MAE gives an undefined relative growth, retained as
null without epsilon. Positive gaps indicate deterioration, not proof that
overfitting is its sole cause; changes in volatility/regime, support and scaling
also matter. Retain the raw and naive-normalized views together.

In these formulas, train means TRAIN_REPLAY, not the internal FIT metric.
Pool absolute-error sums/counts per horizon across the full declared periods.
Overlapping historical windows used for weekly gradient updates must not be
pooled and presented as if they were TRAIN_REPLAY predictions. Report the
TRAIN_REPLAY and VALIDATION counts separately. Never average raw errors across
horizons or promote from a negative multi-head average alone.

## No recursive early-stopping evaluation

The initial fit and each later weekly update retain their configured stopping
rule. Its per-epoch FIT and known-validation losses are inference evaluations
at the current weights, not full nested weekly-retraining campaigns. The owner's
TRAIN_REPLAY assessment runs AFTER the initial selected weights are saved.
Keep loss_FIT_for_stopping and MAE_TRAIN_REPLAY_for_reporting distinct.
The historical lookback remains four years unless a separately configured
one-year contrast is approved. Do not silently change it for efficiency.

## Current findings and permitted changes

The original four-year weekly fits did use VALIDATION and their stored metrics
are reproducible. They selected and scored the same known week. Those results
must stay labelled calibration/development. Chronological forward assessment is
separate, even when performed on the same calendar year's data.

The current stopping loss normalizes both sets by each TRAIN horizon's naive;
the displayed R uses each set's own naive. Record this difference explicitly.
The mean TRAIN/known-validation stopping rule remains unchanged until a labelled,
prospectively fixed contrast is authorized. No hidden metric/architecture changes.
One-year warm-start updates, BatchNormalization, dropout and L2/weight-decay
contrasts remain separate experimental factors, not explanations already proven.
The new complete TRAIN_REPLAY campaign has not yet been executed; existing
stored FIT metrics and the two-week pilot must not be renamed as that result.
