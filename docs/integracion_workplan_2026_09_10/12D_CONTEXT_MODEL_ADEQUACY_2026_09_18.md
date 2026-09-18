# 12D — Context, model, target and data-volume adequacy (S1–S4): frozen design, real learners, budget-limited execution

Orders: `MUSASHI_CONTEXT_MODEL_ADEQUACY_AND_S1_S4_2026_09_18.md` and the mandatory addendum
`MUSASHI_ML_REVIEW_REQUIRED_BEFORE_CAMPAIGNS_2026_09_18.md`. Stage: **S3 executed to its budget
limit; S4 delivered as design and verification**. `utildev-v1` (18 non-advances, 18 inconclusive)
and `utilinst-v1` keep their original diagnostic scope; nothing is recomputed or upgraded.

## Questions separated

| question | where | status |
|---|---|---|
| (a) can the learner predict the raw task? | this pilot (`S1_ADEQUACY_DESIGN.json`, `8a277881…`) | design frozen, tests green, **measurement budget-limited** |
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
