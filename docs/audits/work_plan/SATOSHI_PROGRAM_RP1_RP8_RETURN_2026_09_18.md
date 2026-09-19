# Return: program restart RP1–RP8 — MOD-E0-DEV designed, tested, executed and verified (DEVELOPMENT)

Orders: `MUSASHI_PROGRAM_RESTART_RP1_RP8_2026_09_18.md` with the master plan v3, the metrics contract v1
and the persistent state. Executed without pausing; no reserve, no live, no financial scoring, no RL
training, no GPU, no service restart; the C1–C6 sinusoid campaign was not run (withdrawn). All prior
evidence preserved. This is a single review request at the end of the scope.

## RP1 — reconciliation and registration

* Master v3, metrics contract v1, state and orders integrated by merge `5718f06` (source of the
  historical correction: `4a53b49` on `musashi/program-plan-restart-20260918`); the merge kept the
  work branch's code and evidence and added the plan documents; conflicts only in `09_ADOPCION`
  (both blocks kept). Orders re-read from master by name.
* Reused evidence re-verified only where consumed: the isolated runner / early stopping / checkpoint
  / reload path (adequacy tests, T2), the governed registration/terminal/outbox path (utility runners),
  the metrics-by-content verification pattern. No D0–D2 profiles, D3 matrices or old calibrations were
  repeated. The plan checker passes documentary coverage (`check_plan.py`: PASS, scientific approval
  false). Persistent state updated at each stage (`PROJECT_METHOD_STATE.json`).

## RP2 — design before training (`13A_MOD_E0_DEV_DESIGN_SHEET_2026_09_18.md`, sealed `96f447e6…`)

Multivariate generator with two latent groups; H2 levels h = 0..3 widen persistence and period gaps
with sampling, p, N, amplitude and noise fixed; H3 pairs r = 1 (β-lagged partner, τ = 3) and r = 0
(phantom partner: same marginal process in distribution, no observable dependence); one-step direct
targets on all 8 variables; generator oracle (all latent states; unattainable floor) and an
observer-attainable linear window bar; W = 48 (2 cycles of P_A), branch reach 65, purge W + h, nested
chronological splits; 3 replicates × 3 predefined random assignments; shared seasonal-naive MASE
denominators with a zero policy; estimands e(h)/slope and d_r/γ with the replicate as the unit; no
margin; precision feeds the E0-CONF size rule. Every number carries a derivation, alternative and
sensitivity. Two sentences of the sealed derivation text predate the ML07 correction (architecture and
r = 0 wording); the executed graph and control are the ones in the code, tests and cell records —
corrected in the design module and 13A, and disclosed here rather than re-sealing history.

## RP3 — acceptance first (`tests/test_df_mod_e0.py` 11 rules, `tests/test_df_mod_e0_run.py` 5 rules)

ML01 generator equations/population/labels-never-features; ML02 future/test invariance and roles by
identity; ML03 random assignments change the partition, keep sizes and never relabel; ML04 marginals
preserved (variance, ACF) with the dependence present only under r = 1, shuffle rejected; ML05/ML06
identical frozen-extractor activations, temporal axis kept to fusion, only fusion/core/head weights
move, capacity within 15 %; ML07 positive learning with the real receiver against naive, generator
oracle and the linear bar, grouping recovers latent groups at h = 3 and not at h = 0; ML08 early
stopping restores the best validation weights, reload parity; ML09 MASE from arrays with shared
denominators and NO_APLICA; ML10 e(h)/slope/d/γ signs and units; ML11/ML12 ceiling before each child,
incompletes kept, a failed extractor leaves its terminal and makes its arms INCONCLUSIVE. Deliberate
alterations (unfrozen extractor, relabeling, shuffle, altered arrays) fail their tests.

**ML07 instrument diagnostic** (`RP3_ML07_RECEIVER_DIAGNOSTIC.json`): the first receiver (plain ReLU
conv stack, mae, lr 1e-3) stayed at the naive level; residual TCN blocks with ELU, mse optimisation,
Adam 3e-3, ≤ 6 000 updates, patience 30 reach 0.43–0.46 on 3/3 seeds (ReLU at 1e-2 collapsed on
3/3); sealed before any pilot cell. `build_branch` (Flatten) was not reused.

## RP4 — governed pilot (`mod-e0-dev-v2`, stage design `6173c5d1…`, successor of `96f447e6…`)

* Service identity and loader health were checked by advance (terminal outbox sent counter, live
  reconciliation), not by a 200 alone. Metrics contract v1 applied: MASE/MAE per split, naive/oracle/
  linear references, updates, trainable parameters and profile ARI as MEDIDO rows; NO_APLICA states
  in tags; nothing hidden behind labels.
* **Cost pilot** (`mod-e0-dev-v1`, `RP4_MOD_E0_V1_COST_PILOT_REPORT.json`): three governed children
  without test access at the sealed size (300-update ceiling): 0.041 s per update for a full modular
  model, 0.021 s for a frozen-extractor arm, ~7.7 s overhead per child. Projection of the sealed
  allowance (6 000 updates): **19 218 s with headroom > 14 400 s → not launched**. As ordered, a
  smaller DEV stage was designed BEFORE any score, keeping both contrasts and every condition,
  replicate and cell: update allowance 3 000 (the ML07 seed diagnostic stopped at 1 122–1 848 updates
  with patience 30, so 3 000 keeps > 60 % margin); sealed as a recorded successor
  (`RP4_MOD_E0_DESIGN_V2_STAGE.json`); projection 10 219 s with headroom, launched.
* **Execution**: 3 pilots + 66 cells, all `COMPLETED`; campaigns registered before any child
  (cost pilot, pilot arm, cells); before_run per cell; H3 arms only after their replicate's extractor;
  early stopping in 60 cells, the 3 000-update ceiling in 9 (flagged STOPPED_BY_BUDGET; 297–2 970
  updates observed); spent **3 191 s CPU** of 14 400 (v1 pilot's 54 s counted in the ledger);
  single host (16 cores, children at 2 threads; no worker dispatch needed: the projection fit with
  room), no GPU, no service touched. Envelope DEVELOPMENT `9fe6d516…`.
* **Defect found and repaired with PRE/POST** (`RP5_TERMINAL_REPAIR_PRE_POST.txt`): every terminal
  was refused by data-gov (`http 400 invalid metric schema`) because a `status` key had been added
  to the metric rows; the outbox kept all 72 pending, reconciliation showed 66 missing, the
  warehouse held nothing. Fix `bb213b3`: metric rows carry exactly the governed keys, states travel
  in tags; a repair mode superseded every refused terminal with a schema-valid generation-2
  successor (originals disposed SUPERSEDED; 69 + 3 superseded, 0 failed); all three campaigns
  reconciled with 0 missing. No cell was re-run.

## RP5 — reconciliation and the scientific question (`RP5_MOD_E0_V2_VERIFY.json`, `RP5_MOD_E0_V2_TABLES.md`)

**Verification**: every cell's MASE/MAE for model, naive and oracle recomputed from the persisted
arrays with the shared denominators; the consumed data regenerated from (level, r, seed) and
matched by digest; H2 arms checked to use / not use the profile partition; H3 arms checked frozen
(extractor weight change 0.0 in every arm); reload parity; parent report equal; **live warehouse
content equal for 69/69 units**; reconciliation 0 missing. ML07 per cell: **66/66 beat the naive**,
48/66 within 0.03 MASE of the linear window bar (the 18 outside are the H3 arms at r = 1, whose
frozen extractor was trained on the profile arm; largest gap 0.119, all summary arms). Grouping:
profiles recover the latent groups with ARI = 1.0 at h = 1, 2, 3 in every replicate; at h = 0 ARI
0.16–0.49 (no mechanism to recover).

**Effects (validation split; unit = replicate; descriptive)**:

| | | |
|---|---|---|
| H2 e(h) = MASE(profiles) − MASE(random) | h = 0: −0.009, h = 1: +0.006, h = 2: −0.000, h = 3: +0.006 (replicate SD ≈ 0.01) | slope **+0.0038**, bootstrap 95 % [+0.0000, +0.0064] |
| H3 d_r = MASE(sequence) − MASE(summary) | d_0 = **−0.023** (SD 0.006), d_1 = **−0.091** (SD 0.006) | γ = d_1 − d_0 = **−0.068**, bootstrap [−0.072, −0.064]; d_1 [−0.096, −0.085] |

Test-split effects (descriptive, not used for any selection): H2 slope +0.0006; H3 d_1 −0.088, γ −0.066.
MASE by condition (validation): H2 profiles/random 0.653/0.661 (h0), 0.554/0.548 (h1), 0.589/0.589
(h2), 0.444/0.439 (h3); H3 sequence/summary 0.659/0.682 (r0), 0.589/0.679 (r1); naive 0.79–0.50 and
generator oracle 0.56–0.37 across conditions; the extractor itself reaches 0.661/0.589.

**Answer to the pilot's question, at its scope**: (i) with the same branches, capacity, context,
information and training, assigning variables by profiles did **not** change the error relative
to random redistribution at any heterogeneity level (differences within ±0.01 MASE, well inside
the replicate dispersion); the H2 mechanism shows **no effect** here even though the profiles
recover the latent groups exactly from h ≥ 1 — under this generator every branch has the same
architecture and the fusion core can recombine the sequences regardless of which variables share a
branch. (ii) Keeping the sequences until fusion is **better than an early summary in every
replicate**, by 0.023 MASE without lagged dependence and by 0.091 with it: the advantage grows with
the lagged relation (γ = −0.068, consistent across replicates), which is the direction H3 predicts.
Both are descriptive development results with 3 replicates: no confirmatory support, no ADVANCES,
no financial or public eligibility. What is missing for confirmation: E1 development to fix margin,
precision and the procedure, then the reserved E0-CONF synthetic and E2 public families.

## RP6 — next steps (`13B_NEXT_STEPS_E1_E3_AND_LANES_2026_09_18.md`)

Forecasting/RL demand read from the executable agent-multi configurations (DQN/PPO over gym-fx, 1 h
bars, window 32, `twelve` features, ATR stop/take-profit, commission 0.001, PnL reward) with the gaps
(weekly cutoff/release, availability per feature, capital/sizing) and owners; E1 public bank by
selection over the existing census (criteria listed; no re-census); E1 references and R0/R1/R2 rules;
E3 forecasting + RL protocol mandatory and not conditional on H1; lanes P-PRE/P-TRN, P-CAP, P-L2, P-INC
with identifier, source, dependency and deliverable.

## RP7 — infrastructure reused

data-gov, outbox, DuckDB warehouse and the isolated runner reused; no new dashboard, graph library or
orchestrator; feature-eng lane left in the plan with its next task.

## RP8 — closure

**Suites** (trading-stack, `crispdm-run`, CPU; `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py olap/store/tests`): **1306 passed, 6 skipped**, 801 s at `2d59fc8`+docs; after the RP5 repair fix the targeted runner suite (6) and acceptance suite (11) re-run green; the plan checker PASS (documentary only). Disposition by block: design DONE (sealed v1 + stage v2); implemented DONE;
executed DONE (66/66 cells, 3 191 s CPU); verified DONE (files = parent = accounting = live warehouse); inference: DEVELOPMENT descriptive
only — no ADVANCES, no support to H2/H3, no eligibility. Commits: `5718f06` (merge) · `2d59fc8` (RP1–RP3) · `5306f84` (docs/state, runner fix) · `56f2b08` (stage v2) · `bb213b3` (RP5 repair) · `20c8e68` (closure) and the commit naming it. Workers synced.
Pending: the empty-envelope disposition (next needed window).

Ending: **MOD-E0-DEV pilot delivered for Musashi's review** (single review request).
