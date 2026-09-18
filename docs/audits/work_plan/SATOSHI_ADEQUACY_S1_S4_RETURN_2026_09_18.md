# Return: adequacy S1–S4 — frozen design, real learners, budget-limited governed pilot, business protocol

Orders: `MUSASHI_CONTEXT_MODEL_ADEQUACY_AND_S1_S4_2026_09_18.md`, addendum
`MUSASHI_ML_REVIEW_REQUIRED_BEFORE_CAMPAIGNS_2026_09_18.md`, requirements
`MUSASHI_UTILITY_STATUS_AND_WEEKLY_TRADING_REQUIREMENTS_2026_09_18.md`. Executed without pausing.
No GPU, financial data, RL training, reserve, broad sweep, service restart; nothing of
`utildev-v1`/`utilinst-v1` recomputed. **Stage: S3 executed to its budget limit; S4 delivered as
design, matrix and verification.** ML review table carried in 12D (statuses below).

**PRE**: the utility probe was ridge λ = 1 on 4 raw values (8 for raw_wide), consuming 0.07 (seed 12)
and 0.04 (seed 13) periods; target observed[t+1] − observed[t]; no adequacy of context, model,
target or volume had been demonstrated. **POST**: a sealed design separating the four questions
(`S1_ADEQUACY_DESIGN.json`, `8a277881…`), twelve acceptance tests green before any training, real
learners with recorded graphs and updates, a governed cost pilot verified to the live warehouse, an
honest projection that does not fit the ceiling, and the business/RL protocol with gaps and owners.

## S1 — tests and a bounded factorial before training

`tools/df_adequacy_design.py`, `tests/test_df_adequacy.py` (9) + `tests/test_df_adequacy_run.py` (3):
units seed 12/13 (P 41.44 / 78.15; noise sd 0.33 / 0.19); three DISTINCT tasks with their own
baseline and truth-derived oracle (clean level: persistence + recurrence; clean increment:
zero-change + recurrence increment; observed increment: zero-change + the clean recurrence
against the observed label as the irreducible floor — zero error never demanded); contexts
W = 4/8/128/256 with W/P and consumed span (W−1)/P (two periods only from W = 128 on seed 12 and
W = 256 on seed 13); training lengths 256/512/768 (1 024 does not fit n = 2 048 with W = 256 and
the purges: a staged proposal, not a silent drop); untouched test rows 1 664–2 047; inner
validation 96 rows; purge W + h between splits; train-only scaling; Adam 1e-3, batch 64,
≤ 200 epochs, ≤ 3 000 updates, early stopping patience 10, **no tuning allowance**; criteria
(skill ≥ 0.5 adequate / < 0.1 inadequate; observed vs oracle floor; saturation rule; diagnosis
classes; fixed factorial); budget 7 200 s CPU. The analytic recurrence
x[t+1] = 2cos(2π/P)x[t] − x[t−1] is verified on the clean arrays and **ridge with two lags solves
the clean level task to 1e-6** — the special case, not a universal requirement.

## S2 — real temporal learners

`tools/df_adequacy_models.py`: ridge (closed form), causal Conv1D/TCN (16 filters, kernel 3,
dilations 1,2,4,… until RF ≥ W; RF 7/15/255/511 computed and recorded; last-position head), LSTM
(32 units, `RESET_PER_WINDOW`: chunked = whole predictions). Every cell: layers, parameters, RF,
effective support, optimizer, observed updates and the weight change they produced,
train/validation curves, diagnosis (FITTED / UNDERFIT / OVERFIT / OPTIMIZATION_FAILURE by declared
rules), predictions, labels, row ids, baseline and oracle arrays, losses recomputed from the arrays,
reload parity, cost. Every model of a cell reads the same rows and labels (model-free preparation).
Deployed `predictor_plugin_cnn/lstm` are composite (bidirectional, Bayesian heads) and were **not**
labelled plain Conv1D/LSTM; they are listed for 12E row 4.

## S3 — governed execution: BUDGET_LIMITED

`tools/df_adequacy_run.py`, run `adequacy-v1` (`S3_ADEQUACY_V1_{REPORT,PLAN,VERIFY,DESIGN_FROZEN}.json`):
design frozen write-once; cost-pilot campaign registered before any child; three governed
children at W = 256, L = 768 with a 200-update ceiling — conv 18.4 s (168 updates), LSTM 12.6 s
(192 updates), ridge 0.14 s; all UNDERFIT at that ceiling (recorded, not interpreted);
reconciled `[]`; **verified**: losses recomputed from arrays = records = parent = warehouse
(**live query**). Projection of the 216-cell factorial at the full update allowance:
**10 538 s CPU > 7 200 s** (conv 6 514 s, LSTM 3 982 s, ridge 10 s). As ordered: stopped after the
pilot, no cell removed, exact need in `PLAN.json`; spent 31 s. Staged options for a new sealed
successor (`S3_ADEQUACY_STAGED_OPTIONS.json`): B ridge full + NN at L = 768 (120 cells, 5 217 s),
C (96, 3 258 s), D (96, 2 629 s), E (88, 1 766 s); recommendation B. No selection claim; no
calibration transferred.

## S4 — matrix, business protocol, RL counterpart

12C carries the model × context × task × preprocessing matrix with IMPLEMENTED / EXECUTED /
VERIFIED / NOT_TESTED apart and the shortcomings. 12E carries the weekly-business
requirement → test → evidence table (8 rows, all NOT_TESTED or PARTIAL, with owners and next
actions) and the RL counterpart design (agent-multi DQN configurations over gym-fx, comparable
arms, weekly cycle, what counts as evidence; no RL training here). 12D carries the ML review
table: question/estimand PASS (declared); target/noise, context/support, temporal validity
PASS (tests) — measurement NOT_TESTED; data sufficiency, statistics, business transfer
NOT_TESTED; model adequacy PASS (mechanics) / NOT_TESTED (adequacy); independent evidence PASS on
the pilot. **Diagnosis asked by the order** (does context/model/target noise explain the pilot
behaviour?): not yet measurable — no adequacy cell ran; what is established is that the ridge/W = 4
probe consumed < 0.1 periods, that the clean special case is exactly solvable with two lags, and
that the observed target carries an oracle-quantified floor per cell.

## Closure

**Suites** (trading-stack, `crispdm-run`, CPU; `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py olap/store/tests`): **1283 passed, 6 skipped**, 661 s. Targeted: adequacy 9 · adequacy run 3 (with the utility suites
unchanged). Skips: the store suite's own skip; `systemd-run`-gated children present here.
**Commits**: `91c1183` (merge) · `4ba4795` (S1–S3) · the closing commit that names this return.
Workers synced. Pending: the empty-envelope disposition (next needed window). CPU of this order:
31 s of 7 200 s; the successor is not launched without review.

Ending: **adequacy BUDGET_LIMITED** — frozen design, tests, real learners and verified cost pilot
delivered; measurement awaits the staged successor's review.
