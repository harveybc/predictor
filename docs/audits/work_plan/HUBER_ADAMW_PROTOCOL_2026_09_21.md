# Huber / AdamW diagnostic, before advancing E1

Status: 11 acceptance tests pass; run pending. Initial red collection showed
the runner was absent, not a reproduced historical defect. Fresh-child restore
parity and warehouse comparisons are mandatory runtime acceptance checks.

User requirement: compare Huber and AdamW now, using MAE as a reported metric.
This is DEVELOPMENT on the already inspected household validation slice, not
confirmation, a reserved test, or a trading evaluation.

## Frozen comparison

- Four arms: MAE+Adam, Huber+Adam, MAE+AdamW, Huber+AdamW; seeds 1, 2, 3.
- Same original modular core, no pretraining; same prepared data by digest,
  input/target roles, train-only z-score, windows, horizon and split as RP63.
- Learning rate .003, batch 64, ceiling 4000 observed optimizer updates;
  early stopping patience 3 epochs, restore best validation MAE in every arm.
- Huber delta 1 in standardized target units. AdamW weight decay .004;
  beta1=.9, beta2=.999, epsilon=1e-7 for both optimizers. No tuning this round.
- Delta and decay are explicit initial configurations, not recovered historical
  settings or asserted optimal values. Huber and MAE have different gradient
  scales; equal learning rate does not establish each optimizer's best result.
- Report raw-kW MAE/RMSE, standardized MAE, naive MAE on identical origins,
  skill versus naive, paired differences, seed dispersion, updates, cost,
  stopping and restore checks. No inference from MASE to percentage error.
- Do not change target transform to log1p, architecture, context, or train volume
  during this factorial. Those are separate possible successor experiments.
- Maximum 14400 aggregate child CPU seconds; each child <=1800 CPU seconds.
  CPU only, bounded concurrency, no new model selection from reserved data.

## Requirements and evidence

| Requirement | Acceptance before fitting | Run evidence |
|---|---|---|
| Independent loss/optimizer factors | four actual recipes; analytical Huber values; actual decay on zero gradient | resolved Keras configs |
| Common selection | val_mae in every recipe | curves, best epoch, restored prediction MAE |
| Identical task | source digests, delivery before data load | origins, targets, scaler, initial weights hashes |
| Honest comparison | empty/partial/nonfinite arrays refuse; numeric metric oracle | array recomputation and paired table |
| Reproducibility | fresh child per cell, retained weights | reload predictions agree |
| Governed results | existing HTTP campaign/delivery/outbox path | accepted terminals and warehouse content check |

The existing phase-1 comparison changed both training loss and monitor. Its
result is a recipe contrast, not an isolated loss effect. This factorial fixes
the monitor to remove that ambiguity. All historical evidence stays intact.

References: [Keras Huber](https://keras.io/api/losses/regression_losses/),
[Keras AdamW](https://keras.io/api/optimizers/adamw/),
[decoupled weight decay paper](https://arxiv.org/abs/1711.05101).

## Execution amendment, no scientific change

First dispatch at `8357548` exposed a concurrent dynamic-import defect: the
second thread obtained `governed_run` before its body finished loading.
Only MAE+Adam seed 1 ran and closed (MAE 0.4970333449520751 kW). Its root
`huber_adamw_v1` is retained. No second campaign was registered by that thread.
The parent now finishes module loading before dispatch. The complete factorial
will use fresh `huber_adamw_v2` campaigns; the first result is operational
evidence and will not be pooled as an extra replicate. Loss, optimizer, seeds,
data, selection and budgets are unchanged. Redundant inherited graph prose
about MSE/Adam is corrected to reference the per-arm recipes.
The measured first-child peak was below 0.6 GiB at cgroup level. The successor
allows four children under the same joint 8 GiB memory limit, with six CPU cores
maximum. This scheduling change does not change each child's two-thread numeric
environment or any learning parameter.
