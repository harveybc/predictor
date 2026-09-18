# 12D — Context, model, target and data-volume adequacy (S1–S4, corrected under T1–T4)

> **Corrections (T1–T4, 2026-09-18) beside the historical claims below — the S1–S3 text is kept as history:**
> (1) the S1 "observed-increment oracle" (clean recurrence increment) was **not** the conditional noise floor: it
> included the noise already observable in the current observation; the corrected oracle is
> `2cos(2π/P)·clean[t] − clean[t−1] − observed[t]`, whose residual is the next noise sample only. The old
> oracle is kept in the arrays as `oracle_old`; the S1 adequacy criterion "MAE ≤ 1.1 × oracle" is **not reused**.
> (2) S1 boundaries shifted training/validation dates with W (W4 train 790–1558, W256 train 286–1054), so a
> W comparison changed recency, phase and realisation as well as context, and the CNN depth changed with W
> (2 → 8 layers): the corrected design v2 uses the **same validation and test rows for every W, model and L**
> (purge = max W + h), nested training histories at one cutoff (1054), and **one fixed CNN graph** (RF 511)
> at every W; the variable-depth CNN is kept only as separate evidence. Ridge's parameter count still
> varies with W (declared limitation). (3) The S3 cost pilot **scored the test rows** (1664–2046) of three
> cells: those rows are a disclosed DEVELOPMENT diagnostic, not an untouched confirmation; new cost pilots
> have no test access (the accessor fails). (4) The S3 verification recorded ridge as FITTED, not all three
> pilots UNDERFIT (corrected wording); "exact need" was an extrapolation, now called a projection.

Orders: `MUSASHI_CONTEXT_MODEL_ADEQUACY_AND_S1_S4_2026_09_18.md` and the mandatory addendum
`MUSASHI_ML_REVIEW_REQUIRED_BEFORE_CAMPAIGNS_2026_09_18.md`. Stage: **S3 executed to its budget
limit; S4 delivered as design and verification**. `utildev-v1` (18 non-advances, 18 inconclusive)
and `utilinst-v1` keep their original diagnostic scope; nothing is recomputed or upgraded.

## Questions separated

| question | where | status |
|---|---|---|
| (a) can the learner predict the raw task? | this pilot (S1 design superseded by `T1_ADEQUACY_DESIGN_V2.json` `ad14cc17…`; S1: `S1_ADEQUACY_DESIGN.json`, `8a277881…`) | design frozen, tests green, **measurement budget-limited** |
| (b) how much history does it need? | this pilot (contexts W = 4/8/128/256, learning curves over L) | as above |
| (c) does a representation improve it? | utildev-v1 (relative, ridge/W = 4) | preserved at its scope; not re-run |
| (d) does it transfer to prediction/RL and weekly trading? | 12E protocol | design only; NOT_TESTED |

## What is frozen (before any training)

* **Units**: bank sinusoids seed 12 (P = 41.44, A = 1.47, noise sd 0.33) and seed 13 (P = 78.15,
  A = 0.87, noise sd 0.19), n = 2 048, clean and observed arrays bound by digest.
* **Tasks, each with its own baseline and truth-derived oracle**: `clean_next_level` (persistence
  baseline; oracle = recurrence `2cos(2π/P)·x[t] − x[t−1]`), `clean_increment` (zero-change
  baseline; recurrence increment), `observed_increment` (zero-change baseline; the clean
  recurrence increment against the observed label = the irreducible floor; zero error impossible).
* **Models**: ridge λ = 1 (closed form), causal Conv1D/TCN (16 filters, kernel 3, dilations
  1,2,4,… until the receptive field covers W; last-position head; RF recorded), LSTM (32 units,
  `RESET_PER_WINDOW`). Adam 1e-3, batch 64, ≤ 200 epochs, ≤ 3 000 updates, early stopping on the
  inner validation (patience 10, best restored), **no tuning allowance**.
* **Contexts and coverage** (consumed span (W−1)/P): seed 12 — W4 0.07, W8 0.17, W128 3.06, W256
  6.15; seed 13 — 0.04, 0.09, 1.63, 3.26. Two periods are covered by W = 128 on seed 12 and only by
  W = 256 on seed 13. Conv receptive fields: 7, 15, 255, 511.
* **Boundaries**: untouched test = rows 1 664–2 047 (384 rows), inner validation 96 rows, training
  L ∈ {256, 512, 768}, purge W + h between every pair of splits derived from the consumed support.
  L = 1 024 does not fit n = 2 048 with W = 256 and the purges: **longer series are a staged
  proposal**, not a silent drop. Train-only scaling. Seeds: 1.
* **Criteria**: skill = 1 − MAE_model/MAE_baseline; clean tasks ADEQUATE at skill ≥ 0.5 at some
  context, INADEQUATE if none reaches 0.1; observed task judged against the oracle floor
  (MAE ≤ 1.1 × MAE_oracle); data sufficiency SATURATED/UNSATURATED from the L-curve; every cell
  diagnosed FITTED / UNDERFIT / OVERFIT / OPTIMIZATION_FAILURE; fixed factorial, no re-seeding.

## Acceptance tests (before training; `tests/test_df_adequacy.py`, `tests/test_df_adequacy_run.py`, 12 rules)

Analytic clean-sine recurrence holds and **ridge with two lags solves the clean level task
exactly** (the special case Musashi noted); label/row/horizon identity; train-only scaling with
the test's mean provably not used; true forward held-out rows; future-perturbation invariance;
LSTM restart parity (chunked = whole); noise-only control (no large skill on white noise);
Conv receptive field from kernel/dilation; real optimizer updates observed with the weight change
they produced; diagnosis rules; reload parity; equal rows/labels across models; a cell through
the isolated child with losses recomputed from arrays, altered arrays refused on resume; a ceiling
hit is a complete `RESOURCE_EXCEEDED` with cost and no partial score; the governed runner freezes,
pilots, projects, registers before any cell, keeps incompletes.

## S3 — governed execution: budget-limited

`adequacy-v1` (`S3_ADEQUACY_V1_REPORT.json`): design frozen in the root; cost-pilot campaign
`adequacy-v1-adequacy-cost-pilot` registered before any child; three governed children at the
largest cell (W = 256, L = 768, 200-update ceiling): conv 18.4 s (168 updates, 0.109 s/update),
LSTM 12.6 s (192 updates, 0.065 s/update), ridge 0.14 s; all three UNDERFIT at 200 updates (as
expected under that ceiling; recorded, not interpreted). Projection of the full factorial (216
cells; every NN cell at its full update allowance, early stopping only lowers it): **10 538 s CPU
> 7 200 s**. As ordered, nothing was launched beyond the pilot; `PLAN.json` carries the exact need.
The three pilot cells were verified: losses recomputed from the arrays equal the records and the
parent, and the warehouse holds them equal (**live query**, `S3_ADEQUACY_V1_VERIFY.json`).
Spent: 31 s CPU.

**Staged successor options** (`S3_ADEQUACY_STAGED_OPTIONS.json`; proposals for a new sealed
design, none run): B ridge full + NN at L = 768 → 120 cells, 5 217 s; C + NN only at W ∈ {4, 256}
→ 96 cells, 3 258 s; D + NN on seed 12 only → 96 cells, 2 629 s; E + NN on the observed task
only → 88 cells, 1 766 s. Recommendation for review: **B** (keeps every model, task, unit and
context at the largest training length; the L-curve then comes from ridge everywhere and from the
NNs at one length, declared as such).

## Review table (status after S1–S3)

| requirement | executable check | evidence | status | owner | next action |
|---|---|---|---|---|---|
| question / estimand | four questions separated; decision supported: was ridge/W = 4 an adequate probe | design; 12C | PASS (declared) | Satoshi | measure (a),(b) |
| target and noise | label identity, recurrence oracle, noise-only control, oracle floor | tests green; pilot cells carry oracle MAE | PASS (tests) / NOT_TESTED (measurement) | Satoshi | run successor |
| context and support | W, (W−1)/P, RF, state policy per cell; equal rows across models | design; tests | PASS (tests) / NOT_TESTED (measurement) | Satoshi | run successor |
| data sufficiency | L-curves over {256, 512, 768}; two units; one regime | not measured (budget) | NOT_TESTED | Satoshi/Musashi | successor B; longer series staged |
| model adequacy | graph, updates, curves, diagnosis; capacity/context ablations | pilot: updates observed, curves recorded, UNDERFIT at 200 updates | PASS (mechanics) / NOT_TESTED (adequacy) | Satoshi | run successor |
| temporal validity | train-only scaling, purge, forward held-out, perturbation, restart | tests green | PASS (tests) | Satoshi | keep in every run |
| comparison fairness | same rows/labels per cell; spans explicit; no tuning | design | PASS (design) | Satoshi | — |
| statistics | block-wise skill dispersion; fixed factorial; no calibration transfer | design | NOT_TESTED | Satoshi | successor |
| business transfer | 12E protocol with gaps and owners | 12E | NOT_TESTED | Satoshi/owner | protocol review |
| independent evidence | losses recomputed from arrays; live warehouse query; negative tests | `S3_ADEQUACY_V1_VERIFY.json` (3 cells) | PASS (pilot) | Satoshi/Musashi | extend to successor |

Diagnosis of the pilot behaviour asked by the order (context/model/target noise): **not yet
measurable** — the budget stopped the factorial before any adequacy cell ran. What the frozen
design and the tests establish: the ridge/W = 4 probe consumed 0.07 (seed 12) and 0.04 (seed 13)
periods, far below two periods; the clean special case is solvable by two exact lags; the observed
target carries an irreducible floor that the oracle quantifies per cell. Whether the neural
learners need W ≥ 128 on these signals is the successor's measurement.

## T1–T4 (2026-09-18): corrected design v2, second cost pilot, still budget-limited

**Design v2** (`T1_ADEQUACY_DESIGN_V2.json`, `ad14cc17…`, sealed before any outcome): conditional
oracle; shared decision rows (validation 1311–1406, test 1664–2047 for every cell; cutoff 1054;
purge 257; training histories nested: L = 256/512/768 start at 798/542/286); fixed CNN graph
(dilations 1…128, RF 511, causal padding, effective support = W); exposure ledger; cost pilots
without test access; digest binding of the consumed arrays with the bank's scheme; diagnosis
with stop reason, restored checkpoint and reload parity; ceiling 14 400 s with 25 % headroom.
Tests: 19 rules (`tests/test_df_adequacy.py`, `tests/test_df_adequacy_run.py`), the T1/T2
counterexamples frozen first.

**Cost pilots `adequacy-v2`** (governed, 6 children at W = 4 and 256, L = 768, 200-update ceiling,
**no test access**, verified: arrays = record = parent = live warehouse, 6 units):

| cell | CPU s | fit s | overhead s | updates | s/update | stop | diagnosis (heuristic) |
|---|---:|---:|---:|---:|---:|---|---|
| conv W4 | 10.6 | 9.2 | 1.4 | 192 | 0.048 | UPDATE_BUDGET | UNDERFIT |
| conv W256 | 13.9 | 12.4 | 1.5 | 168 | 0.074 | EARLY_STOPPING | UNDERFIT |
| lstm W4 | 7.0 | 5.8 | 1.2 | 192 | 0.030 | UPDATE_BUDGET | UNDERFIT |
| lstm W256 | 13.2 | 11.8 | 1.4 | 192 | 0.061 | UPDATE_BUDGET | UNDERFIT |
| ridge W4 / W256 | 0.15 / 0.16 | 0 / 0.007 | 0.15 | 1 | — | CLOSED_FORM | FITTED |

Validation-only descriptive values (no test): skill 0.02–0.12; conditional oracle MAE 0.2135 on
the validation rows vs the expected Gaussian noise-only MAE 0.2619 (σ = 0.328) — the realised
value is below the expectation, which is why neither is a samplewise bound.

**Projection** (per-update cost interpolated in W, every NN cell at its full allowance, overhead
per child): 11 695 s; **with 25 % headroom 14 619 s vs 14 355 s remaining — does not fit by
264 s (1.8 %)**. As ordered, nothing was launched and no cell was removed (`T3_ADEQUACY_V2_PLAN.json`).

**Trade-off proposal** (`T3_ADEQUACY_V2_TRADEOFF.json`; no cell removed in any option):

| option | change | projected s | with headroom | fits 14 400 |
|---|---|---:|---:|---|
| A | none (as sealed) | 11 695 | 14 619 | no — needs a ceiling of 14 664 s (+264 s) |
| B | max_epochs 200 → 160 (training rule, needs review) | 9 401 | 11 751 | yes |
| D | max_epochs 200 → 150 | 8 828 | 11 035 | yes |

Recommendation for review: **A with a 14 700 s ceiling** (keeps the sealed training rule; the
overrun is 1.8 % of a conservative projection), otherwise B as a declared rule change re-sealed
before any outcome. Early stopping in the pilots already stopped one cell before its allowance.
