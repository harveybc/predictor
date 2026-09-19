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

## RP4 — governed pilot: RP4_EXEC_PLACEHOLDER

## RP5 — reconciliation and the scientific question: RP5_PLACEHOLDER

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

SUITES_PLACEHOLDER. Disposition by block: design DONE (sealed v1 + stage v2); implemented DONE;
executed RP4_STATUS_PLACEHOLDER; verified RP5_STATUS_PLACEHOLDER; inference: DEVELOPMENT descriptive
only — no ADVANCES, no support to H2/H3, no eligibility. Commits: COMMITS_PLACEHOLDER. Workers synced.
Pending: the empty-envelope disposition (next needed window).

Ending: **MOD-E0-DEV pilot delivered for Musashi's review** (single review request).
