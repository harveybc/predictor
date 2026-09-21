# E1 successor — numeric ML sheet

Design `143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7`, run
`satoshi-e1-successor-20260920`, host omega, panel
`uci_235_individual_household_power/panel.parquet` (`b3192c0b…`, 10 890 295 bytes) delivered by
data-gov from the `public_panels` lake.

**When these quantities were fixed.** Every number in §1–§5 is read from the sealed design and from
`DATA.json`, both of which existed before the first fit: the design is sealed and self-digested, and
`DATA.npz` records the panel digest it was prepared from. This *document* was written after the run;
nothing in it was chosen after seeing a result, and no cell was retouched. §6 reports what the run
then observed.

## 1. Task and windows

| | |
|---|---|
| task id | `W60_h60` |
| window | 60 steps = 3 540 physical seconds of context |
| horizon | 60 steps = 3 600 physical seconds ahead |
| sampling | one row per minute (the archive's own grid) |
| purge between splits | 120 rows |
| model reach | 60 positions, **measured** by perturbation and gradient per input row (`tools/df_e1_receiver.py`), not asserted |
| target | `Global_active_power` |
| inputs | `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1..3`, `Global_active_power` (7 channels) |

Origins are enumerated at every admissible row, so consecutive windows overlap by 59 of 60 steps.
The count of *unique* rows covered is therefore the slice, not the origin count: 50 400 rows
(1 412 361–1 462 761) = 35 days, of which 28 days are DEV train and 7 days DEV validation, taken
from the family's train split and the first days of its validation split. **Test rows are never
read.**

## 2. Volume per split, missing and exclusions

| split | origins | admissible | withdrawn (non-finite inputs) | targets valid | support |
|---|---|---|---|---|---|
| DEV train | 40 141 | 40 081 | 60 | 40 080 | 2 404 860 s |
| DEV validation | 10 020 | 10 020 | 0 | 10 020 | 601 200 s |

No origin was withdrawn for grid reasons and no input was masked. The evaluation set is the
intersection of admissible validation origins with a finite label and a finite daily lookup:
**10 020 windows**, identical to the admissible count.

## 3. Model, parameters and activations

ARCH-A, sequence fusion, causal TCN core: dilations [1, 2, 4, 8, 16], kernel 3, 16 filters,
residual. Detector per branch: 2 residual causal Conv1D blocks (16 filters, k=3, ELU) with a 1×1
projection skip. Integrator: identity. Adapter: TimeDistributed Dense(8), linear. Readout and fusion
are held constant across regimes; the physical grouping (power | voltage | sub-metering) is held
constant and is not a factor here.

| regime | trainable parameters |
|---|---|
| R0 (no pretraining, detector learns) | 8 127 |
| R1 (imported detector, **frozen**) | 5 231 |
| R2 (imported detector, fine-tuned) | 8 127 |

R0/R1/R2 of a seed share the initial checkpoint; R1 and R2 of a seed share the same auto-encoder.
The closure recomputed those digests from the saved weight files: shared initial checkpoint, same
imported detector for R1/R2, R1's detector unchanged after the fit, R0's and R2's moved — for all
three seeds.

## 4. Target, baseline and denominator

MAE is the mean absolute error on the 10 020 validation windows. The ratio this programme published
as "MASE" divides it by the **train-only** denominator **0.6162615768463073**, which is the mean
absolute change over the HORIZON, `mean |Y[t+60] − Y[t]|`, on the DEV train origins.

(Corrected in RP57/RP60. An earlier wording of this sheet called that denominator the mean absolute
**one-step** change. It is not: the one-step value on the same train slice is **0.0851237797**
over 40 257 finite pairs, with 2 non-finite pairs excluded and no gap closed. The quantity is
therefore a **persistence-scaled error at the horizon**, not a conventional MASE, whose seasonal
period m is declared on its own grounds. `tools/forecast_comparison.py` reports it under its correct
name together with conventional MASE at m = 1 and m = 1440.)

The scaler is fitted on train windows only — that grain is part of its identity, and RP59 measured
the difference against the row grain at 0.0017 in scaled units. Controls are fitted on the same DEV
train and evaluated on the same windows.

## 5. Optimisation and early stopping

Batch 64, learning rate 0.003, MSE loss, ceiling **4 000 updates** per fit (the same task budget for
every regime; the auto-encoder's cost is reported apart). Early stopping monitors validation MSE
with patience 3 epochs and restores the best checkpoint. The AE budget is 1 500 updates.

**Censoring criterion, declared before the run**: a fit is adequate for comparison when it stopped
by EARLY_STOPPING with a non-improving slope over the last third; a fit that stopped at the update
ceiling is **CENSORED**: its budget ran out before its validation curve settled, so what it would
reach with more budget is **unknown**. (Corrected in RP60: an earlier wording called it a lower bound
on the reachable error. There is no such guarantee — more budget can lower the error, leave it where
it is, or raise it through overfitting. The censoring says what was explored, not what is reachable.)

## 6. What the run observed

| unit | MASE | MAE | updates / ceiling | epochs | best epoch | stop | verdict |
|---|---|---|---|---|---|---|---|
| R0_s1 | 0.8965 | 0.5525 | 3 762 / 4 000 | 6 | 3 | EARLY_STOPPING | STOPPED_ON_VALIDATION |
| R1_s1 | 0.9087 | 0.5600 | 3 762 / 4 000 | 6 | 3 | EARLY_STOPPING | STOPPED_ON_VALIDATION |
| R2_s1 | 0.8799 | 0.5423 | 3 135 / 4 000 | 5 | 2 | EARLY_STOPPING | STOPPED_ON_VALIDATION |
| R0_s2 | 0.8929 | 0.5503 | 4 000 / 4 000 | 7 | 5 | UPDATE_BUDGET | **CENSORED_BY_BUDGET** |
| R1_s2 | 0.8839 | 0.5447 | 4 000 / 4 000 | 7 | 5 | UPDATE_BUDGET | **CENSORED_BY_BUDGET** |
| R2_s2 | 0.9237 | 0.5692 | 4 000 / 4 000 | 7 | 6 | UPDATE_BUDGET | **CENSORED_BY_BUDGET** |
| R0_s3 | 0.8730 | 0.5380 | 2 508 / 4 000 | 4 | 1 | EARLY_STOPPING | STOPPED_ON_VALIDATION |
| R1_s3 | 0.9176 | 0.5655 | 4 000 / 4 000 | 7 | 4 | UPDATE_BUDGET | **CENSORED_BY_BUDGET** |
| R2_s3 | 0.8868 | 0.5465 | 2 508 / 4 000 | 4 | 1 | EARLY_STOPPING | STOPPED_ON_VALIDATION |

**Convergence is NOT declared.** Four of the nine fits reached the update ceiling, and in each of
them the best checkpoint is at or near the last epoch, so their budget ran out while the validation
curve was still moving. What they would reach with more budget is unknown, in either direction. Two seeds are therefore mixed-censoring across regimes, and the
means below are read with that limitation stated, not hidden.

### Means and paired differences (validation MASE)

| regime | n | mean | sd |
|---|---|---|---|
| R0 | 3 | 0.8875 | 0.0127 |
| R1 | 3 | 0.9034 | 0.0175 |
| R2 | 3 | 0.8968 | 0.0236 |

| paired difference | mean | per seed (1, 2, 3) |
|---|---|---|
| R1 − R0 | **+0.0159** | +0.0122, −0.0090, +0.0446 |
| R2 − R0 | **+0.0093** | −0.0166, +0.0308, +0.0138 |
| R2 − R1 | −0.0066 | −0.0288, +0.0398, −0.0308 |

### Controls (same windows, same denominator)

| control | MASE | MAE |
|---|---|---|
| linear ridge (train-only fit) | **0.8852** | 0.5455 |
| linear at the model's reach | 0.8852 | 0.5455 |
| persistence | 1.0018 | 0.6174 |
| seasonal naive (daily) | 1.1873 | 0.7317 |

### Cost

| item | CPU seconds |
|---|---|
| auto-encoders (3 seeds, reported apart) | 137.156 |
| fits R0 | 992.425 |
| fits R1 | 506.000 |
| fits R2 | 829.399 |
| cost pilot | 65.673 |
| controls | 4.851 |
| **run total (root)** | **2 535.504** |

Amortised AE cost: 137.156 s over the six fits that consume it (R1 and R2 of three seeds) =
**22.86 s per consuming fit**; per seed the AE is 45.7 s and is charged once to the two regimes that
share it. Total cost of an R1 or R2 arm = its fit CPU + the AE of its seed.

## 7. What this is, and what it is not

Three seeds on one task in one 35-day slice of one household are **development evidence**. They do
not confirm H1, do not establish equivalence, and are not a confirmatory reserve. The reading they
support is narrow and negative: on this task, with this budget, modular pretraining did not help —
R1 and R2 sit above R0 on average — and a competent linear control (0.8852) is better than all three
regime means. Four censored fits make that reading weaker still for those arms: their budget ran out
while the validation curve was still moving, so their error is what was explored, not what is
reachable.

Every unit of this run closed GOVERNED: campaign and verified delivery before the work, accepted
terminal after it, campaign reconciled (`E1_SUCCESSOR_CLOSE.json`, 15/15 `VERIFIED_AND_GOVERNED`).
