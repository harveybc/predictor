# Corrections to the RP57–RP64 return — published beside it, not written over it

Point 1 of [MUSASHI_POST_HUBER_PHASE2](../../handoffs/MUSASHI_POST_HUBER_PHASE2_2026_09_21.md), after
reading Musashi's measured four-arm comparison
([MUSASHI_HUBER_ADAMW_RESULTS](MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md)). The original return
([SATOSHI_PROGRAM_RP57_RP64_RETURN](SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md)) keeps its labels
and its numbers; what changes is how three of its sentences must be read.

## Correction 1 — RP63 was a recipe contrast, not an isolated loss effect

The return said: *"Training on the measure the question is judged by is worth more than anything
else tried … MAE loss 0.4978 ± 0.0002 against the run's own recipe 0.5469 ± 0.0078."*

What RP63 actually changed between `core_mse` and `core_mae` was **two things at once**: the training
loss (MSE → MAE) **and** the early-stopping monitor (validation MSE → validation MAE). The 0.049 kW
difference is therefore the effect of the *recipe*, and nothing in RP63 attributes it to the loss
alone. Musashi's factorial fixed the monitor to validation MAE in **every** arm and measured, on the
same rows and paired seeds:

| recipe (monitor = val MAE in all) | MAE kW, mean ± sd of 3 seeds | vs naive |
|---|---:|---:|
| MAE + Adam | 0.496382 ± 0.004487 | 19.598 % |
| MAE + AdamW | 0.495073 ± 0.004303 | 19.810 % |
| Huber + Adam | 0.520662 ± 0.001674 | 15.665 % |
| Huber + AdamW | 0.522479 ± 0.005178 | 15.371 % |

With the monitor held fixed, the MAE-loss arms reproduce phase 1's `core_mae` (0.4978), so the
monitor was not what carried the gain — but that is *Musashi's* measurement, not RP63's. The
correct sentence is: **aligning the training recipe (loss and monitor) with the measure is worth
0.049 kW on this task; with the monitor fixed, the loss family accounts for about 0.024 kW between
MAE and Huber (delta = 1), and the optimiser for less than 0.002 kW with inconsistent sign across
seeds.** No loss is selected for trading by any of this; MAE + Adam is the household *continuity*
reference only.

## Correction 2 — a shuffled-label run is a negative control, not a bound

The return said: *"Most of what the model 'achieves' is not learned from the labels … The entire
label-derived advantage is 0.049 kW."*

One scrambled-label fit, one seed, one selection rule, measures what **that** protocol recovers
without the input→label mapping. It does not bound what any protocol could learn from the labels,
and it does not bound "all attainable improvements": the very next measurement (the MAE recipe)
moved the error by that same 0.049 kW *with* the labels. The correct sentence is: **with all 40 080
train labels scrambled, the same protocol reached 0.6015 kW against 0.5525 with true labels (one
seed); the R0/R1/R2 differences we had compared (0.01 kW) are small next to that gap, which is the
only quantitative reading this control supports.** "Labels buy little" is withdrawn as a general
claim.

## Correction 3 — the RP61 matrix said a delta head had "not been tried"

`RP61/LITERATURE_TO_CODE_MATRIX.md` listed, among the tutorial's ideas not yet tried, *"a delta
head, zero-initialised, so that an untrained model starts exactly at persistence."* That is wrong for
the core this programme runs: `tools/df_e1_pilot.py::_model_for_target` (core `tcn_w`) already adds
the last observation to the head's output (`persistence_skip`), so the model predicts a **change**
over the last value. What it does not do is zero-initialise that head — RP60 measured an untrained
model at MAE 1.57–2.45 kW against persistence's 0.617 for exactly that reason. The correct
statement is: **the persistence skip exists; the zero-initialisation does not, and its effect is a
candidate for a later phase, not an untried architecture.**

## What is unchanged

Every number in the return, every artifact under `RP57`–`RP63`, the legacy-table dispositions
(`CAUSALITY_UNVERIFIED`, both lineages `UNBOUND`), the `stl_norm` repair and the leak-signature
calibration stand as published. The phase-3 financial result remains OWNER_REPORTED / UNBOUND; the
household experiments neither reproduce nor invalidate it.
