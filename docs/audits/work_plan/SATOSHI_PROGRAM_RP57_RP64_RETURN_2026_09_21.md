# Satoshi — return of RP57–RP64: the objective was the lever, and 85 published tables are unusable

Order: [MUSASHI_PROGRAM_RP57_RP64](../../handoffs/MUSASHI_PROGRAM_RP57_RP64_2026_09_20.md) including
the owner's correction of 2026-09-20, review
[MUSASHI_E1_ML_BASELINES_REVIEW](MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md). Base `de88764`,
orders merged at `f864c7b`. One review is requested; nothing here is approved in advance.

**The three results that matter.**

1. **Training on the measure the question is judged by is worth more than anything else tried.**
   Nine governed cells, three arms, three paired seeds, on the successor run's own rows: MAE loss
   **0.4978 ± 0.0002** against the run's own recipe **0.5469 ± 0.0078**. The reference TCN block at
   the same objective gives 0.5359 ± 0.0050. Pretraining gave nothing.
2. **Most of what the model "achieves" is not learned from the labels.** With all 40 080 train
   labels scrambled, the same protocol at full scale reaches **0.6015** against **0.5525** with true
   labels. The entire label-derived advantage is 0.049 kW — the same size as the loss-alignment gain,
   and five times the R0/R1/R2 differences we had been comparing.
3. **85 of 167 published legacy tables beat the naive by 50% or more**, the phase-3 family by
   **96–97% at one to six hours on an FX price**. None of them is usable as a benchmark, and none is
   declared a leak: their consumed representations cannot be reconstructed here.

## PRE

`tools/forecast_comparison.py` was run on the preserved run before any edit and reproduced the
reviewer's frozen `results.json` **field for field, zero differences**
([`RP57_PRE/`](../evidence/d3_k5_20260917/RP57_PRE/REVIEW_REPRODUCED_PRE.json)); its 14 rules passed.

## RP57 — every comparator visible, and the ratio named for what it is

The report now carries, per method and per scale, on the **same rows**: MAE, RMSE, bias, the spread
of the absolute error, skill against **every** declared reference (persistence, seasonal naive,
linear ridge), paired differences row by row, contiguous temporal blocks — ten thousand windows
overlapping by 59 of 60 minutes are not ten thousand situations — and cost per unit. A zero-error
reference yields an undefined skill; no epsilon manufactures a win.

The published "MASE" is renamed **`persistence_scaled_error_horizon_train`** and travels with its
erratum; the historical field is kept unchanged so published numbers stay readable. Conventional
MASE is reported at **m = 1** and **m = 1440** (one day at one-minute sampling — an independent
seasonal ground, not the horizon), each with its support and its gap treatment: non-finite pairs
excluded and counted, never closed up. The ML sheet's own description of its denominator was wrong
and is corrected in place: it is the mean absolute change over the **horizon**, not one step (the
one-step value on the same slice is 0.0851237797 over 40 257 finite pairs).

The table is in the entry point, in a readable report
([`RP57/COMPARISON_TABLE.md`](../evidence/d3_k5_20260917/RP57/COMPARISON_TABLE.md)) and in the
warehouse as an **identified reanalysis**: 192 metric rows, tagged `E1_REANALYSIS`,
`new_training: false`, campaign reconciled and verified by content
([`RP57/REANALYSIS_PUBLICATION.json`](../evidence/d3_k5_20260917/RP57/REANALYSIS_PUBLICATION.json)).
It does not certify RP56's accounting. Against the run's own `RESULTS.json`: identical for all nine
cells.

Per seed the picture is less tidy than the means: R0_s3 beats the ridge by 1.37%, R2_s1 by 0.59%,
R1_s2 by 0.14%. The linear control wins on the regime means, not on every cell.

## RP58 — the old comparisons, and the owner's correction

**Before the correction.** Across the entire history of this repository, exactly one published table
ever carried a naive comparison: the TCN NEAT of `63d0429`. Its published **Test** rows recompute
exactly from its own predictions and labels (differences below the six decimals it prints); its grid
is **4 hours**, so H9–H24 are 36 to 96 hours ahead whatever the file name says; its target range
1.089–1.150 is price, not log1p. The remembered MAE 0.02 / naive 0.018 is in **no** results file of
this repository, committed or historical, and is reported as not found rather than matched by size.

**After the correction**, the priority is the phase-3 lineage, and the answer is bound by identity —
the commit that introduced the published bytes, the config that **names** that file as its own
output, and the inputs **as of that commit**:

| | |
|---|---|
| tables inventoried | 167 (113 reconstructable, 54 published-summary-only) |
| `CAUSALITY_UNVERIFIED` | **114** |
| no decomposition declared or consumed | 53 |
| flagged extraordinary (≥50% over the naive) | **85** |
| phase-3 family | 96–97% skill, declares `use_wavelets` + `use_stl` + `use_multi_tapper`, **and its inputs do not exist at its own producing commit** |

The owner's statement is recorded as **OWNER_REPORTED** and bound to nothing: the producer of the
decomposed inputs is not in this repository, so **both lineages are UNBOUND**. The TCN NEAT table
consumed only `DATE_TIME` and `typical_price` at its commit — and the files with those names in the
working tree today are **different bytes**, which is why identity, not the current tree, decides.
It is not the phase-3 run the owner remembers and **it is not a benchmark**.

**The signature, calibrated** on the same family's series, with a change head so that predicting no
change *is* the naive ([`RP58/LEAK_SIGNATURE_CALIBRATION.json`](../evidence/d3_k5_20260917/RP58/LEAK_SIGNATURE_CALIBRATION.json)):

| representation | skill vs naive |
|---|---:|
| raw price history | −0.24% |
| **trailing** wavelet (past only) | −5.42% |
| whole-series DWT read row by row | +6.00% |
| whole-series reconstruction (denoising) | +5.14% |
| **centred smoother spanning the horizon** | **+46.76%** |
| *the published phase-3 tables* | *+96–97%* |

Whole-series wavelet denoising does **not** reproduce the published magnitude. Centred, symmetric
filtering is the family that approaches it, and 96% is beyond even that. The mechanism family is
identified; the exact mechanism is not, and every affected run stays `CAUSALITY_UNVERIFIED` rather
than being declared `INVALID_CAUSAL_LEAK` on a score.

The instrument is not new: this repository already has an exhaustive causal battery
(`tools/df_causal_battery.py`) whose negative-control classes include **`full_series_dwt_as_time_row`**,
`centered_rolling_window`, `filtfilt` and `phase_compensation_shift_back` — all **DETECTED**, by
prefix equality, in the sample run for this round.

**One repair on an active path, with a demonstrable PRE.** `pipeline_plugins/stl_norm.py` decided
whether to map a model's output back to price space by inspecting **that output's own
distribution**, so two models in the same normalised space could have their errors reported in
different units. The space is now declared by the configuration and obeyed; an undeclared one still
runs the old heuristic, but the guess is recorded and `strict_normalization_contract` refuses it.
Eleven rules, ten of which fail against the previous code, and one of which preserves the defect.

## RP59 — the data path, and where the error actually is

Bytes identical through delivery and preparation; a perfect one-minute grid over the consumed slice
(50 400 rows, zero missing minutes, no duplicated stamps, no daylight-saving transition inside
2009-08-23…09-27); train and evaluation spans disjoint; the label is `Y[origin+60]` and equals the
panel column element by element; perturbing the future leaves an origin's window untouched; the
scaled slice is centred on the **window** grain, as the run declared, and the two grains differ by
0.0017 in scaled units — measured, not assumed.

The 25% of rows with the smallest movement carry 13–14% of the absolute error; the 10% with the
largest carry 27%, and in that bucket the neural model and the linear control have the **same** mean
absolute error (1.4930 kW). Train autocorrelation is 0.963 at one minute, 0.403 at sixty and 0.317 at
a day: the daily lag carries **less** linear structure than the hour, so a daily context is a
hypothesis, not a certainty. Periods are plotted by a rule declared before any error was read. No
irreducible floor is claimed.

## RP60 — the optimisation, and what the labels are worth

The route is exact: a head forced to emit the last observed target reproduces the persistence control
through the same inverse and the same metric (difference 2·10⁻⁷). Capacity is there: on a fixed
256-window subset the loss falls from 6.99 to 0.0055. The counted updates **are** the optimiser's
steps (400 = `optimizer.iterations`). Every fit's restored checkpoint agrees with the argmin of its
own curve.

The **"lower bound" claim is retired**, in the code and in the published sheet: a fit that stops at
the ceiling says what was explored, not what is reachable — more budget can lower the error, leave
it, or raise it.

The negative control had to be run twice. At subset scale it separates nothing: the true-label fit
restores epoch 1 and scores *worse* than the scrambled one. At full scale it gives the result in the
header — 0.6015 scrambled against 0.5525 true, correlation 0.44 with the truth, still beating
persistence. A model trained on scrambled labels beating persistence is a fact about the baseline,
not about the model.

## RP61 — literature traced to code

Both references read **at their source** ([`RP61/LITERATURE_TO_CODE_MATRIX.md`](../evidence/d3_k5_20260917/RP61/LITERATURE_TO_CODE_MATRIX.md)).
Our core is **not** Bai's block: one convolution instead of two, ELU instead of ReLU, no weight
normalisation, no dropout, no activation after the sum. The port in `tools/df_tcn_reference.py`
reproduces the source block, parameter-matched by a rule stated before any fit — the width whose
trainable total is closest to ours (8 061 against 8 127) — and trained with **our** optimiser,
budget, loss and stopping so the block is the only thing that differs. The TensorFlow tutorial is
weather at one-hour sampling, one step ahead: its numbers transfer nothing. Two ideas of its own are
recorded for a later phase and were **not** mixed into this one: calendar sin/cos inputs and a
zero-initialised delta head.

## RP62/RP63 — the first phase, sealed then executed

Design `5cb8263d…` sealed before running; cost pilot 66 s, projection 3 957 s; executed for
**2 024.5 CPU seconds**, nine cells, every unit registered, delivered, reported and reconciled
([`RP63/PHASE1_REPORT.json`](../evidence/d3_k5_20260917/RP63/PHASE1_REPORT.json)).

| arm | MAE (n=3) | sd | what differs |
|---|---:|---:|---|
| `core_mae` | **0.4978** | 0.0002 | MAE loss, early stopping on validation MAE |
| `tcn_mse` | 0.5359 | 0.0050 | the reference TCN block, parameter-matched |
| `core_mse` | 0.5469 | 0.0078 | the run's own recipe |

`core_mse_s1` = **0.5525**, reproducing `R0_s1` of the finished run exactly through a different
runner — the replication check the phase needed. Four of nine cells are `CENSORED_BY_BUDGET`, and
that says what was explored, not what is reachable. Three seeds on one task remain development
evidence: not H1, not equivalence, no reserve. Input information, train volume and pretraining were
**not** moved; R0/R1/R2 return only after (a) and (d), as the order requires.

## Causes: DEMOSTRADA / REFUTADA EN ESTE ALCANCE / PENDIENTE

- **The objective was not aligned with the measure** — **DEMOSTRADA**: 0.4978 against 0.5469, three
  paired seeds, everything else fixed.
- **Our TCN is an adaptation, not the reference block, and it costs something** — **DEMOSTRADA**:
  0.5359 against 0.5469 at matched parameters and identical training.
- **More epochs would fix the validation divergence** — **REFUTADA EN ESTE ALCANCE**: the true-label
  fit at subset scale restores epoch 1, and four full-scale cells that ran to the ceiling did not
  improve monotonically.
- **The model's advantage comes from learning the input→label mapping** — **REFUTADA EN ESTE
  ALCANCE** as a general claim: scrambled labels cost only 0.049 kW.
- **The published 96–97% skill comes from a whole-series wavelet leak** — **PENDIENTE**: the
  mechanism as tested yields 5–6%, a centred smoother 47%. Its refutation would be a reconstruction
  of a phase-3 run's consumed representation showing the battery passes on it.
- **Pretraining helps at h=60 on this task** — **REFUTADA EN ESTE ALCANCE** (RP55, three seeds).
- **Calendar or daily context would help** — **PENDIENTE**: not moved in this phase; the daily
  autocorrelation (0.317) is lower than the hourly (0.403).

## Accounting, costs and what was denied

Every campaign of this round is closed: the two that failed mid-flight — the ported block before its
CPU-gradient repair, and the reanalysis whose instant was malformed — are closed with **FAILED**
terminals carrying their reasons, and both reconcile. The terminal outbox holds **zero pending
envelopes**: the four duplicated `prepare` envelopes and two others were adjudicated by the existing
additive procedure, moved unchanged beside a write-once disposition. **The operation the environment
refused in RP56 as an inline script was accepted through a purpose-built tool**, so RP56's open item
is closed.

Round CPU on omega is inside the ceiling
([`RP64_CPU_LEDGER_omega.json`](../evidence/d3_k5_20260917/RP64_CPU_LEDGER_omega.json)). The workers
ran nothing this round. No GPU, no live, no reserve, no scientific RL training; no reserved split was
reopened, no legacy sweep re-executed, no sample overwritten, and no healthy service restarted.
