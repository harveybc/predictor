# Timing-target feature selection: TIMING_PROXY_SELECTION (ZigZag expert as the label)

Purpose: select features for TIMING (pivot / entry-exit) decisions, using the optimized ZigZag hindsight
expert as the target. The expert's peaks and valleys are the close tick of one order and the open tick of the
next; this is NOT price prediction. This is expert-demonstration / hindsight-guided learning, not
human-preference RLHF. The output is a TIMING_PROXY_SELECTION: usefulness is established for the pivot
classification proxy and its trivial reversal strategy only, not for an RL policy; a fixed-budget SAC
comparison against the admitted-core 15 is a later, separately authorized integration.

Revision 2 (2026-10-19) incorporates Musashi's plan audit (REVISE_BEFORE_FINAL_SEAL, predictor
`codex/data-coverage-repair-20261018` @ed5b1acc). Everything computed under revision 1 (targets
`timing_targets.v1`) is superseded and was recomputed through the dependency DAG of section 5.
Everything here is ADDITIVE: no existing engine, method list, target pack, manifest or selection identity
is modified; the timing family is registered beside the existing ones and driven by a JSON config.

## 1. Map of the existing selection tooling (discovered before writing anything)

The public entry point of selection is the local `dataset_selection` checkout (README title
`feature_selector`, remote `harveybc/feature_selector`), which pins the engines in
`upstream.lock.json` and audits the extension points in `docs/PORTABILITY.md` (@94b1d88 / branch
`codex/curation-reset-20261018`). The scientific engines live in their owners; there is no
`feature-selector` repository with code. The role "feature-selector: selection" is therefore implemented
in predictor (phases 2-4) and causal-inference (phase 1), documented in dataset_selection.

| Stage | Repo / module (file@commit) | Entry point | Config / schema | What it selects | Target it uses | Split it decides on |
|---|---|---|---|---|---|---|
| Phase-1 profiles + causal ladder | causal-inference `provider/src/causal_inference_provider/feature_selection_worker.py`, `fs_causal.py`, `fs_causal_batch.py`, `fs_causal_discovery.py` @08f4474 | `run_column`, `run_stdio` (`stdio-json-v1`), `python -m causal_inference_provider.fs_causal_discovery pcmci` | `feature_selection_unit.v1`, `target_pack` {id in TARGET_PACKS, definitions[name,column,family,head,horizon_hours]} | per-feature profiles, rung-1/2/3 states (SUPPORTED / NOT_IDENTIFIED) | EURUSD pack: `Y_s_{1..6}h`, `Y_l_{24..144}h`, `Y_b_s6`, `Y_b_l144` (14 cells); ETH pack 6 | TRAIN prefix folds; BH at family level |
| Phase-1 inventory scheduler | predictor `tools/phase1_inventory_orchestrator.py` (fs4 branch @15893493) | JSON plan, `stdio-json-v1` | inventory CSV, hosts, warehouse endpoints | nothing (schedules) | as above | - |
| Population manifest | predictor `tools/fs_phase23_manifest.py` @15893493 | `build_manifest`, `write_manifest`, `verify_data`, `freeze` | `fs_phase23.population_manifest.v1` (`targets[{target_id,column,family,head,horizon_hours}]`, prefix folds) | freezes the population | any numeric target column | TRAIN only (refuses validation/test file names) |
| Phase 2 pairwise | predictor `tools/feature_pairwise_campaign.py`, `tools/feature_pairwise_worker.py` @15893493 | verbs `plan`, `run-worker`, `follow`, `close-phase2` | `DEFAULT_PARAMS` (lags {0,1,2,6,24,48,168}, MI bins 8, dcor cap, Spearman 0.9 average-linkage redundancy) | aliases, redundancy clusters, pair metrics/stability/gates | none (feature x feature) | TRAIN prefix folds |
| Phase 3 filters | predictor `tools/feature_filter_selection.py` @15893493 | `run_filter_methods`, chained by `follow` | `METHOD_NAMES` = 9 built-ins (SPEARMAN_CLUSTER, MRMR, JMI, *_CAUSAL, controls); K grid | candidate subsets per target (`CANDIDATES_FOR_VALIDATION.json`, `final_selection:false`) | every manifest target (MI with quantile-coded y) | TRAIN |
| Phase 4 extractibility (FS4) | predictor `tools/fs4_campaign.py`, `tools/fs4_worker.py`, `tools/run_feature_extractibility_queue.sh` (branch `codex/fs4-automation-20261006` @15893493; docs `docs/tres_temas_entrevista/`) | sqlite queue claim/complete/status | `fs4.extractibility.task.v1`, arms RAW / RANDOM_ENCODER / TRAINED_ENCODER | extractibility verdicts per feature | forecasting heads (`px.rv5` receipts) | inner_2023 |
| Predictive selectors (TRAIN-only) | predictor `tools/feature_selector_predictive.py`, `feature_selector_elastic_net.py`, `feature_selector_comparators.py` @15893493 | `ComparatorRunPlan` | sealed K grid | ExtraTrees / sequential / group gate / ChronoEpilogi rankings | fold target | chronological inner split inside TRAIN |
| Predictive validation (weekly) | predictor `tools/business_weekly_*.py`, `tools/fs_close_weekly.py`, `tools/fs_close_refit.py`, `tools/fs_close_manifest.py` @15893493 | closure follower | `feature_selection.manifest.v1` + `feature_selection.decision.v1`; gate `tools/selected_manifest_gate.py` | FINAL manifest (366 dispositions) | trading weekly heads | VALIDATION 2024 read once |
| Admitted-core selection (current RL baseline) | financial-data `_scripts/core_selection_bundle.py` @41f5af170 (candidates: 8 RAW + 22 technical from feature-eng `normalize_phase1b.py` @d081d0f); dataset_selection `scripts/complete_core_selection.py`; doc `docs/tres_temas_entrevista/program_v3/SELECTED_CORE_RL_2026_10_19.md` @f1210cd2 | `complete_core_selection.py complete` | identity `553fc047...`; recipe = mRMR k12 union over 3 targets pruned by Spearman 0.9 clusters | 15 ordered hourly variables | RETURN_4H, RETURN_12H, ABS_RETURN_12H | TRAIN 2020-2023, VALIDATION 2024 not used in selection |
| Causal event stage | causal-inference `provider/src/causal_inference_provider/event_impact.py` (branch `satoshi/event-impact-20261019` @a4cb9ea) | `estimate_family`, `select_events`, `match_controls` | calendar + hourly bars; refusals FORBIDDEN_TARGET_TRANSFORM / FUTURE_SHIFTED_EVENT / TEST_ROWS_FORBIDDEN | impact half-lives, matched-control contrasts | abs/log returns around releases | decision_start with purge |
| RPCMCI regime discovery | causal-inference `run_rpcmci_only.py`, `causal_regime_analysis_v2.py` @08f4474 | script | tigramite RPCMCI, 4h resample | per-regime parents of `fwd_ret_6` | fwd_ret_6 | none (exploratory) |
| Expert + pivot harness | heuristic-strategy `app/short_horizon/zigzag.py` (sha256 `b30788e6...`, branch `satoshi/short-horizon-strategy-20261018` @b9b4f9c; choice Depth 3 / Deviation 36.3 / Backstep 2 in `docs/short_horizon/results/zigzag/CHOICE.json`); `tools/short_horizon_pivot_detector.py`, `tools/short_horizon_pivot_detector_train.py` (branch `satoshi/pivot-detector-20261019` @cead861c) | `labels`, `evaluate`, `sanity` | k = 2 bars, thresholds {0.4..0.7}, owner's monitor | nothing (evaluates) | hindsight pivot-in-k | 2023 decision, 2024 once |
| Fractal candidates | agent-multi `tools/causal_fractal_features.py` (ported to feature-eng `app/plugins/causal_fractal.py`, branch `satoshi/causal-fractal-eurusd-20261019` @6c028aa); store manifest sha256 `906578e1...` | plugin | windows 200/96/48 | - | - | TRAIN 2020-2023 / VALIDATION 2024, no TEST |
| RL observation contract | gym-fx `app/observation_builder.py`, `preprocessor_plugins/feature_window_preprocessor.py` @e287ea2 | `build_observation` | `gym_fx.observation_contract.v1` (`feature_columns`, rolling z-score, agent-state contract) | - | - | - |

Plugin conventions: predictor model/pipeline/preprocessor/target components resolve through setuptools
entry points (`setup.py`); the selection engines are NOT entry-point plugins (filter methods are a
built-in list; phase-1 workers speak a JSON subprocess protocol; the causal worker validates a fixed set
of target packs). feature-eng plugins resolve through `feature_eng.plugins`; heuristic-strategy
strategies through its own entry points (`pip install -e .` required).

## 2. What this extension adds: files touched (additivity)

| Repo / file | Kind | Role | Proof existing behaviour is unchanged |
|---|---|---|---|
| predictor `tools/timing_pivot_targets.py` | ADDED | targets T1/T2/T3 (timing_targets.v2), support ledger, censoring, purges, finality replays | no existing file modified (`git diff --diff-filter=M` vs 15893493 is empty) |
| predictor `tools/timing_price_pool.py` | ADDED | multi-scale causal price pool, availability gate, quality profile, prefix/perturbation checker, within-day variance | as above |
| predictor `tools/timing_selection.py` | ADDED | JSON-config driver (verbs run, causal, attach-eval, tables, publish, snapshot); DAG; read ledger; gate-shaped manifest | as above |
| predictor `tools/test_timing_selection.py` | ADDED | 18 acceptance tests (A01-A05, A10, A12 and the screens) | existing suites `tools/test_fs_phase23_driver.py`, `tools/test_fs_rep_dispositions.py`, `tests/test_df_consumption_gate.py`, `tests/test_df_ecl_contrast_gates.py`, `tests/test_flow_v3_gate.py`: 93 passed / 17 failed / 1 skipped at base 15893493 AND on the branch, identical failure sets (12 failures are `duckdb` not installed); `tests/test_fs_close_manifest.py`, `tests/test_fs_close_weekly.py` (selected-manifest gate consumers): 25 passed at base and on the branch |
| predictor `examples/config/timing_selection/eurusd_h1_timing.json`, `docs/TIMING_SELECTION.md` | ADDED | config with placeholders; this document | - |
| causal-inference `provider/src/causal_inference_provider/fs_timing_pack.py` | ADDED | pack EURUSD_TIMING: definitions, validation, HAC lag screen, matched-control contrast, PCMCI+ wrappers, mixed-data RegressionCI stage with controls | - |
| causal-inference `provider/src/causal_inference_provider/feature_selection_worker.py` | EXTENDED (append-only) | `TARGET_PACKS` += `EURUSD_TIMING`; `TARGET_CELL_COUNTS["EURUSD_TIMING"] = 4` (two literal lines) | test TP01 asserts EURUSD/ETH entries, cell counts 14/6, denominators (14) and feature count (366) unchanged; existing suites `test_feature_selection_worker.py`, `test_fs_causal.py`, `test_fs_causal_batch.py`, `test_fs_causal_discovery.py`, `test_event_impact.py`: 59 passed / 1 skipped at base a4cb9ea AND on the branch |
| causal-inference `provider/tests/test_fs_timing_pack.py` | ADDED | TP01-TP09 | - |
| heuristic-strategy `tools/short_horizon_timing_eval.py` | ADDED | evaluator (develop / freeze / reconfirm / seal) | existing short-horizon suites (9 files): 92 passed / 1 skipped at base cead861c AND on the branch |
| heuristic-strategy `tests/unit_tests/test_short_horizon_timing_eval.py`, `test_short_horizon_timing_adapter_contract.py` | ADDED | TE01-TE08 (10 cases), AC01-AC09 | - |
| dataset_selection `docs/PORTABILITY.md` | EXTENDED (one table row + one pointer line) | extension table entry | documentation only |
| dataset_selection `docs/TIMING_SELECTION_MAP.md`, `examples/timing_pivot_eurusd.json` | ADDED | map and example config | - |

Engine stages that ran UNCHANGED on the timing population: `fs_phase23_manifest.build_manifest`,
`feature_pairwise_campaign` plan / run-worker / follow / close-phase2 / run-phase3 / close-phase3,
`feature_filter_selection.run_filter_methods` (nine methods; its quantile coding keeps the 3 classes
distinct), `fs_phase23_warehouse_adapter` (sqlite3 backend: duckdb is not installed here),
`feature_selection_profile.profile_series` (phase-1 profiler on the new families), `fs_causal.hac_partial_test`,
`event_impact.match_controls`, tigramite `PCMCI.run_pcmciplus` with `RegressionCI`, the strategy adapter
`Ctx.run` / `execution.run_backtest`, and the pivot-detector helpers (`temperature`, `calibrate`,
`signals_from`). New code was needed only for the targets and ledger, the price pool and checks, the screens
and reports, the causal pack wrappers and controls, the evaluator stages and the publication/snapshot verbs.

## 3. Runbook (CPU, memory-capped, DEVELOPMENT; every command checked with `--help`)

```bash
# tests first (predictor; the real-expert finality test needs the two env vars)
TIMING_ZIGZAG_MODULE=<HS>/app/short_horizon/zigzag.py TIMING_ZIGZAG_SHA256=<sha256> CUDA_VISIBLE_DEVICES="" \
  crispdm-run -q -m 3G -t 1200s -n fs-timing -- python -m pytest tools/test_timing_selection.py -q -p no:cacheprovider
# the selection (fill the placeholders of the example config in a private copy)
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 5400s -n fs-timing -- python tools/timing_selection.py run --config <CFG> --out <OUT>
# the causal stage with the mixed-data test (an interpreter with tigramite 5.2.7.0)
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 5400s -n fs-timing -- <VENV>/bin/python tools/timing_selection.py causal --config <CFG> --out <OUT>
# predictive validation (heuristic-strategy; trading-contracts on PYTHONPATH): one develop shard per set x arm,
# then freeze (INNER_A choices, INNER_B frozen score), reconfirm (the single 2024 read), seal
python tools/short_horizon_timing_eval.py develop --eval-input <OUT>/eval_input --out <OUT>/eval --set <SET> --arm <ARM>
python tools/short_horizon_timing_eval.py freeze --eval-input <OUT>/eval_input --data <LAKE>/1h.parquet --out <OUT>/eval
python tools/short_horizon_timing_eval.py reconfirm --eval-input <OUT>/eval_input --data <LAKE>/1h.parquet --out <OUT>/eval
python tools/short_horizon_timing_eval.py seal --eval-input <OUT>/eval_input --out <OUT>/eval
# seal, render, publish, snapshot (predictor)
python tools/timing_selection.py attach-eval --out <OUT> --eval <OUT>/eval
python tools/timing_selection.py tables --out <OUT>
python tools/timing_selection.py publish --out <OUT> --db <OLAP>.sqlite
python tools/timing_selection.py snapshot --out <OUT> --dest <COPY_1> --dest <COPY_2>
```

## 4. Plan-audit dispositions (F01-F09)

| Finding | Disposition | Artifact / test |
|---|---|---|
| F01 label support beyond k | Per-row ledger (origin, origin availability, event time, support end and label availability per target, expert_trade_id, label_group_id, status); support = the confirmation of the certifying pivot, verified by truncation AND future-perturbation replays (not a median lag); unconfirmed / beyond-data rows CENSORED (never NONE / zero); FIT rows whose support reaches any cut PURGED (cuts 2021-01-01, 2022-01-01, 2023-01-01, 2023-07-01, 2024-01-01, 2025-01-01); groups cannot straddle a cut because a purged row's whole certifying group is purged before it; uncertainty by ISO-week block bootstrap | `tools/timing_pivot_targets.py` (`label_arrays`, `build_targets`, `finality_check`, `status_table`); `LABEL_LEDGER.parquet`, `TARGETS.json`; tests `test_a01_*`, `test_a03_*` |
| F02 2024 already observed | 2023 = JOINT development (the expert was chosen with 2023); 2024 = PREVIOUSLY OBSERVED reconfirmation, read once by `reconfirm` after `FROZEN.json`; the subset is frozen (`FROZEN_SUBSET.json`) before any 2023 read | `READ_LEDGER.json`, `FROZEN_SUBSET.json`, evaluator `FROZEN.json`, `RECONFIRM.json`; test `test_te02_*` |
| F03 expert parity is not business parity | The adapter reused is heuristic-strategy `execution.py` + `Ctx.run` (pinned commit); its behaviour is pinned by tests: fill at the OPEN two bars after the signal bar (never the pivot extreme), stop-before-TP collision flagged, gap fills at the open, forced flat 12 h before close, <= 4 orders/day, integer units sized from the balance with margin refusal, close-and-reverse = two costed fills. The 12 h no-reopen rule is NOT applied by the `signal_blocks` path: the evaluator adds a declared reopen guard. Our >= 3/week / <= 4/day rule stays separate from the SAC <= 1/day contract and only marks validity | heuristic-strategy `tests/unit_tests/test_short_horizon_timing_adapter_contract.py` (AC01-AC09) |
| F04 candidate clock and phase-1 quality | Ordered inventory 30 + 47 + 12 = 89 with exact alias groups; quality profiles of every new column (own profile + the existing causal-inference `profile_series`); fractal store keyed by (EURUSD, timestamp, lake bar convention, lake bytes) with 0 close mismatches; prefix + future-perturbation checks for the inventory generator, the price pool, the fractal producer and the scaler; window units declared (observed market bars; elapsed hours only for `px.elapsed_hours`). The repaired admitted root is NOT available on this host: no hand clock shift is applied and calendar encodings are declared lake-clock features | `INVENTORY.json`, `QUALITY_PROFILES.json`, `PERTURBATION.json`, `POOL_META.json`; tests `test_a02_*`, `test_a05_*` |
| F05 T1/T2/T3 are different tasks | T1 = primary anticipatory target (earliest-event rule, tie refused, k in observed bars, (t, t+k], every hourly origin kept); T2 = reactive comparator reported separately; T3 = remaining gross distance for an agent holding the expert's leg (not trade P&L). The selection recipe uses the T1 task only; T2/T3 subsets are reported as comparators. Per-direction PR-AUC / precision / recall / support with NONE in the base rate. The admitted-core 15 is an unchanged scored baseline on identical rows and its members compete in the pool; rows lost to the longest warm-up are reported | `REPORT.json` (`recipe`, `comparator_subsets_T2_T3`), evaluator `per_direction`, `DEVELOP.json` (`rows_lost_to_union_warmup`) |
| F06 adaptive search | Bounded trial ledger (expert, k, targets, sets, arms, regularization cells, seeds, thresholds, choices); owner's mean TRAIN/INNER monitor kept, both terms in inference mode with the same weights at the same checkpoint, running loss logged apart; budget, best step, patience, stop cause logged; INNER_A (2023H1) for every choice, INNER_B (2023H2) scored frozen; ISO-week block bootstrap of paired differences; DSR as a supplement; declared regularization grid (section 6) | evaluator `SEAL.json` (`trial_ledger`, `deflated_sharpe_supplement`), `UNCERTAINTY.json`; tests `test_te03_*`..`test_te08_*` |
| F07 association is not intervention | Every causal-stage result labelled CONDITIONAL_PREDICTIVE_ASSOCIATION; PCMCI+ with tigramite RegressionCI (mixed data; T1 as ONE categorical variable; binary candidates typed discrete), positive control (label leak at lag 1) and negative controls (noise, circular shift), temporal halves; lag-0 mapping declared; NO_LINK never rejects a feature | causal-inference `fs_timing_pack.mixed_ci_stage`; `CAUSAL_STAGE.json`; tests TP07, TP08 |
| F08 target identity changes | Dependency DAG with content identities; an expert hash / semantics change recomputes every target-dependent stage and reuses only target-independent ones (pool, quality, perturbation, within-day variance); phase-2 pair metrics are target-independent in content but identity-bound in the pinned engine, so they are recomputed (declared) | `DAG.json`; tests `test_a10_*` |
| F09 proxy, not RL | Manifest named TIMING_PROXY_SELECTION, `proxy_only: true`, scope statement; no RL run | `SELECTION_MANIFEST.json` |

## 5. Dependency DAG (invalidation)

Target-independent (reusable with identical inputs): `pool_features` <- lake, inventory generator, bundle
script, fractal store, pool code, candidate config; `quality_profiles`, `perturbation_checks`,
`within_day_variance` <- pool. Target-dependent (recomputed on any change of expert bytes, expert
parameters, k, target semantics or splits): `targets_ledger` -> `finality_check`, `population_manifest` ->
`phase2_pairwise` (identity-bound) -> `phase3_filters` -> `recipe`; `univariate_screen` -> `fold_stability`;
`causal`; `eval_inputs` -> `evaluator` -> `selection_manifest`. `run --reuse-from PREV` applies the plan;
`DAG.json` records leaves, node identities and the plan.

## 6. Regularización

Owner decision 2026-10-19 (delegated), declared in the evaluator's trial ledger; never a silent monitor
change. Neural arms (m1 I6-DMO-class, m2 GRU): AdamW decoupled weight decay 1e-4 (unchanged recipe) plus
DROPOUT 0.1 after the fusion/core block and before the dense head (on the GRU output, never inside the
recurrence: `recurrent_dropout = 0`), with the owner's early stopping. No BatchNorm: BN combined with
dropout suffers a variance shift between training and inference (Li et al., CVPR 2019), and BN's
train/inference statistics are fragile under regime shift in financial series; Musashi's campaign also
excludes BN. LayerNorm is not added. Declared ablation grid on identical rows, seeds and budget: dropout
{0, 0.1, 0.2} x weight decay {1e-4, 1e-3} (6 cells x 3 seeds) for m1 and m2; primary cell (0.1, 1e-4).
Gradient boosting (m3): capacity is regularized, not the monitor: max_depth 4, min_samples_leaf 200,
l2_regularization {0, 1, 10} (primary 0), learning_rate 0.05, max_iter 400 with the same monitor every 10
iterations. Pre-declared cell rule, decided on INNER_A only: lowest mean INNER_A loss at the restored
checkpoint across seeds, ties by the smaller mean TRAIN-INNER gap; a non-primary winner is recorded as a
replacement. Reported per cell: per-epoch TRAIN and INNER losses and their gap, stop cause, best epoch,
downstream metrics.

References: Li, Chen, Hu, Yang, "Understanding the Disharmony between Dropout and Batch Normalization by
Variance Shift", CVPR 2019,
https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Understanding_the_Disharmony_Between_Dropout_and_Batch_Normalization_by_Variance_Shift_CVPR_2019_paper.html ;
Loshchilov, Hutter, "Decoupled Weight Decay Regularization", ICLR 2019, https://arxiv.org/abs/1711.05101 ;
Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014,
https://jmlr.org/papers/v15/srivastava14a.html ; scikit-learn histogram gradient boosting,
https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting ; Bailey, Lopez de
Prado, "The Deflated Sharpe Ratio", https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf .

## 7. Revision-2 results

The sealed evidence (manifest, targets document, inventory, DAG, read ledgers, checks, causal stage,
tables, the evaluator seal, OLAP readback, snapshot and gate check, with SHA256SUMS) is in
`docs/timing_selection/eurusd_h1_v2/`. The results audit request for the auditor role is
`docs/timing_selection/eurusd_h1_v2/AUDIT_REQUEST_TIMING_SELECTION_FOR_MUSASHI.md`. The large
artifacts (label ledger, feature matrices, models, predictions, warehouse files) live only in the two
verified snapshot copies, never in Git.

## 8. Revision-3 results (expert v2)

The expert changed (heuristic-strategy abbfaa6: ZigZag v2, Depth 3 / Deviation 40.7 pips / Backstep 0 at
cap 4/day, confirmation `confirmation_bars_fast`, executable fill path b+1). Per F08 the DAG reused the four
target-independent stages and recomputed every target-dependent one (`run --reuse-from`). The targets
module takes the confirmation definition from the config (`zigzag.confirmation`), and the evaluator takes
the expert and the strategy rule (`strategy.executable`, `max_orders_per_day`, `k_tp`) from
`FEATURE_SETS.json`. Sealed evidence, the clock and coverage witness against the admitted root, the
independent physical copy record and the v3 results audit request are in
`docs/timing_selection/eurusd_h1_v3/`. Revision 2 is kept and marked `SUPERSEDED.json`.

## 9. Revision-4 (admitted source, shared eligibility mask)

Musashi's v3 decision was REVISE_BEFORE_BUSINESS_USE (scope RETROSPECTIVE_LAKE_PROXY_RESEARCH_ONLY). v4 is a
new identity on the admitted UTC source, read through the clock lane's `bar_source`. It applies ONE
eligibility mask (`tools/timing_eligibility.py`) to labels, selection, causal stage, fitting, early
stopping, calibration, thresholds and scoring. It also adds:

- a dependence-aware blocked circular-shift screen gating the recipe;
- evaluator inputs bound by identity, refused before any model load on drift;
- TRAIN-prior and calibration baselines;
- full replay-payload snapshots on two machines.

Config: `examples/config/timing_selection/eurusd_h1_timing_v4_admitted.json`. Result, identities and the
finding → correction → test → result table: `docs/timing_selection/eurusd_h1_v4/RESULT.md`. The finite,
resumable chain (tests gate → selection → causal → develop shards → freeze → reconfirm → seal) ran on the
secondary worker, one capped job at a time.
