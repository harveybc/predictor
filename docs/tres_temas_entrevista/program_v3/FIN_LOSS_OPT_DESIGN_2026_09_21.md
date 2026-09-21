# FIN-LOSS-OPT — executable design (DESIGNED, NOT_STARTED)

Policy: [FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21](FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md).
Sealed design: `docs/audits/evidence/d3_k5_20260917/RP65/FIN_LOSS_OPT_DESIGN_SEALED.json`
(`0df0ca76…`), produced by `tools/df_fin_loss_opt_design.py`, which also computes the parts a run
must take from data. Acceptance: `tests/test_fin_loss_opt_acceptance.py` (FL01–FL08). Designing this
is not evidence of a winner; the household factorial selects nothing financial.

## 1. The task, frozen from the governed resource

| | |
|---|---|
| resource | `financial_files : market_data/forex/g10/eurusd/1h.parquet` — 129 873 hourly bars, 2005‑01‑03 01:00 … 2025‑12‑31 16:00, time column `datetime` (coverage read from the lake on 2026‑09‑21) |
| reserve | `deny_from 2025-01-01` in the lake policy: never read, never queried for selection |
| target | hourly bar `close`; evaluation in **MAE_z = mean\|ŷ−y\| / σ_train** per fold, shared by every arm; original units secondary |
| horizons | **short = 6 hourly steps (6 h)**, **long = 72 steps (72 h)** — declared before any score. The owner's recalled 6 h / ~3 days are antecedents, not recovered configs |
| folds | weekly walk‑forward (13C week: Monday 00:00 UTC → Sunday 23:00 UTC), weekly retraining; history‑window candidates 52 / 104 / 208 weeks judged on DEV folds by regime coverage and learning curves; DEV = the last 26 weeks before 2025‑01‑01 |
| missing | non‑finite bars withdrawn; weekend gaps kept as gaps |
| naive | persistence at each horizon on identical rows; skill = 1 − MAE_z_model / MAE_z_naive; zero naive → UNDEFINED |
| benchmark contract | `fx.eurusd.1h.FIN-LOSS-OPT`, comparability **NOT_COMPARABLE** until a reference method is re‑executed under it |

## 2. Comparators and pairing

Full factorial **MAE + Adam, MAE + AdamW, Huber + Adam, Huber + AdamW**, paired initialisations
per seed, identical rows/targets, common monitor (validation MAE_z) and restore policy. All four
recipes of a (seed, horizon) run on one host — the optimiser is never confounded with CPU. Two
receivers, each running every recipe and never compared across: the compact modular receiver
(W=60, `tcn_w`, 5 037 parameters at 5 channels, reach 60) and a larger receiver motivated by the
business configs (W=144, width by a declared rule at pilot; parameters measured, not required to
reach a million).

## 3. Hyper‑parameters: explicit defaults plus an equal, bounded search

- **Defaults arm**: the installed version's explicit defaults (Adam/AdamW lr 1e‑3, β 0.9/0.999,
  ε 1e‑7; AdamW decay 0.004; Huber δ_z = 1) — recorded, not claimed to reproduce the owner's old optimum.
- **Search**: 12 DEV fits per loss family per horizon — the same number for MAE and Huber; learning
  rates {5e‑4, 1e‑3, 3e‑3}; AdamW decays {0, 1e‑3, 4e‑3, 1e‑2}; decay on kernels only, bias and
  normalisation excluded. The cumulative product ∏(1 − lr_t·λ) describes shrinkage due to decay alone
  and only orients candidates; it is not an optimal‑value formula, and μP rules do not transfer.
- **Huber δ**: `delta_candidates()` derives δ_z ∈ {¼, ½, 1, 2} × a residual scale computed **causally**
  inside train — the rolling median absolute h‑step change over train origins — plus δ_z = 1; each
  candidate reports its value in original units and the fraction of train residuals below it. FL06
  proves a change after the last train origin cannot move a candidate. δ is re‑derived per fold
  because a changed target scale changes the physical threshold.

## 4. Adequacy, precision and inference (what every cell records)

Train/validation curves per epoch; monitor and `min_delta` in declared units, chosen so the sought
1e‑5/1e‑6 resolution stays visible; best and restored checkpoint with reload parity; stop reason
and observed `optimizer.iterations`. Predictions and targets saved without rounding; metrics
accumulated in float64 and cross‑checked by an independent calculator; training/inference dtype
(float32 today) and the measured numeric floor recorded before any 1e‑6 is read. Paired analysis
by week/origin and seed with intervals that respect temporal blocks; multiplicity of the search
declared; small uncertain differences stay MEASURED with their interval.

## 5. Acceptance FL01–FL08 — state today

| ID | requirement | state |
|---|---|---|
| FL01 | real factorial | GREEN: four distinct recipes; decoupled decay verified on a zero gradient |
| FL02 | fair comparison | GREEN on the household factorial (paired seeds, one population); financial pairing pending |
| FL03 | no leak | GREEN: prefix invariance through the production calendar path, a leaky control fails; fold‑change rule **xfail** |
| FL04 | marginal precision | GREEN: analytic 1e‑5 and 1e‑6 MAE_z differences survive arrays → JSON → float64 with sign; terminal/warehouse legs **xfail** |
| FL05 | observed training | GREEN on the household factorial (updates, stop, best epoch, reload 0.0); min_delta/floor **xfail** |
| FL06 | hyper‑parameters | GREEN: causal δ candidates; declared grid |
| FL07 | statistical inference | GREEN: both signs kept, no best‑test selection; temporal‑block intervals **xfail** |
| FL08 | complete closure | GREEN on the household factorial (verified, no warehouse problems); financial delivery **xfail** |

The five `xfail(strict)` rules are the runner's obligations; they turn red the day it exists.

## 6. Cost and hosts

Per‑cell cost is unknown until a pilot on the governed bars (200 updates per recipe per receiver);
the household factorial measured 4 279 s for 12 cells at W=60 on omega, which is a reference for
the compact receiver only. Hosts: (seed, horizon) blocks; workers only after they run inside the
same accounted wrapper (RP56 debt: their CPU is UNMEASURED today).

## 7. Execution conditions

The campaign runs only when: financial data and causality gates are passed for the delivered bars
(RP58's full‑configuration tests reused, including the precomputed `wavelet.parquet` features in the
lake, which are NOT consumed until they pass the battery); this design is reviewed; the pilot's
projection fits the round's ceiling with closure reserved. Before any financial R0/R1/R2 contrast,
the chosen recipe is frozen and used equally in method and controls.
