# 13E v2 — E1 task sheet, corrected learning regimes (RP25 successor of 13E v1)

Successor of [13E v1](13E_E1_TASK_SHEET_2026_09_19.md), which stays in the repository unchanged. Version 1 is
superseded in ONE place: what R0 / R1 / R2 name. Everything else of v1 (eligibility, family contracts, usable
windows per split, physical contexts, comparators, splits and the pipeline controls) remains in force and is
not restated here except where a number changed with the corrected spectra and DST dispositions of RP28.

## The defect of v1 and its correction

v1 used R0 / R1 / R2 for *grouping* arms (raw variables, profile grouping, grouping + fusion). The proposal's
R0 / R1 / R2 are **learning regimes of the detector**: what the modular pre-training question is about. Using
the same names for grouping made the pilot unable to answer that question and would have presented an arm that
freezes the whole extractor (E0's H3) as the proposal's R1. Corrected definition, binding from here on:

| regime | detector weights at the start | detector weights during training | what it isolates |
|---|---|---|---|
| **R0** | random, from the replicate's shared initial checkpoint | trainable (gradients reach it, it moves) | no pre-training: the control |
| **R1** | imported from the masked auto-encoder pre-trained on this task's TRAIN windows | frozen (no gradient variable, byte-identical after the fit) | the value of pre-trained features used as a fixed representation |
| **R2** | the SAME imported weights as R1 | adjustable (gradients reach it, it moves) | the value of pre-training as an initialisation |

R1 and R2 import the same bytes: their difference is exactly one factor, whether the detector is optimised.
R0 and R2 differ in exactly one factor, where the detector starts. A regime is proven per unit, not declared:
`tools/df_e1_regimes.py` records the digest of the model right after build (identical across R0/R1/R2 of a
seed), the detector digest after the import, the gradient report on one batch and the detector digest after the
fit. `tests/test_df_e1_regimes.py` fails if any of those four facts disagrees with the regime.

**Held constant across the regimes** (they are other factors, each with its own question and none varied in the
regime comparison): grouping, fusion, readout, architecture, preprocessing, window, horizon, split, scaler,
update ceiling, stopping rule and the evaluation set. **ARCH-0** (no learned extractor) is an *architecture*
control and never the definition of R0. The **E0 H3 arm that freezes the whole extractor** (detector +
integrator + adapter, trained on the synthetic generator) keeps its own scope: it is not presented, now or
retrospectively, as the proposal's R1.

## Pre-training, declared

Objective: masked reconstruction of the preprocessed input (a fraction of time × channel positions zeroed; the
loss counts masked positions only, the mask travelling in the target tensor). Data: the DEV **train** windows
only; internal validation is the inputs of the DEV validation windows, with no label — no score of the task is
read during pre-training. The decoder is a separate per-branch 1×1 convolutional stack, saved apart and never
connected at inference: the deployed model has no decoder layer. Reconstruction error is a diagnostic, never a
result of the task.

Pairing: one auto-encoder per seed; R1 and R2 of a seed consume that seed's detector file (digest recorded per
unit); R0 of the seed shares the same initial checkpoint. Three seeds show optimisation variability; they are
not a power calculation.

## Costs, read separately

Auto-encoder, fit, inference, metrics and reuse are timed per phase and reported apart. Regimes are compared at
the **same task budget** (the same update ceiling for the fit) and, as a second and explicitly separate reading,
at total cost including the auto-encoder that R1/R2 consume. A truncated curve or a `BUDGET_LIMITED` verdict is
a statement about the budget, never evidence that pre-training does not help.

## What v2 authorises

The household DEV pilot of RP30: one task and horizon, ARCH-A, R0/R1/R2 × 3 seeds, controls persistence,
seasonal naive (daily) and linear ridge on the same target, horizon and evaluation set, on a declared DEV
sub-partition with the final test unscored. Electricity may be prepared (contract, enumerator, spectra) but is
not launched here. This pilot does not confirm H1, does not select a universal model and does not replace the
benchmark of the proposal, whose families and comparators stay pending in the matrix.
