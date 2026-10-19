# Timing-target feature selection (ZigZag expert as the label)

Purpose: select features for TIMING (pivot / entry-exit) decisions of RL agents and heuristic strategies,
using the optimized ZigZag hindsight expert as the target. The expert's peaks and valleys are the close
tick of one order and the open tick of the next; this is NOT price prediction and claims no RL value.
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

## 2. What this extension adds (all new files unless marked EXTENDED)

| Repo / file | Kind | Role |
|---|---|---|
| predictor `tools/timing_pivot_targets.py` | ADDED | targets T1 (pivot within k, 3-class + 0/1 indicators), T2 (confirmed-timing action), T3 (expert leg P&L in pips); purges at cuts; the expert module loaded BY PATH with a sha256 pin |
| predictor `tools/timing_price_pool.py` | ADDED | causal multi-scale price pool (returns / ranges / close location / efficiency ratio / realized vol at 1,2,4,6,12,24,48 h; ATR and ratios; hour / weekday / session encodings), availability gate (refuses centred filters and future availability), within-day-variance report |
| predictor `tools/timing_selection.py` | ADDED | the JSON-config driver: pool assembly (inventory via the owners' generator by path, price pool, fractal store), manifest with the `timing_pivot` family, phases 2/3 through the pinned engines UNCHANGED, univariate FDR screen, fold stability, redundancy (clusters, VIF, fractal vs realized vol), causal stage through causal-inference, selection recipe, evaluator inputs, sealed manifest, `attach-eval` |
| predictor `tools/test_timing_selection.py` | ADDED | 11 acceptance tests incl. an end-to-end synthetic run through the real engines |
| predictor `examples/config/timing_selection/eurusd_h1_timing.json` | ADDED | the config (placeholders for private paths; every input pinned by sha256) |
| causal-inference `provider/src/causal_inference_provider/fs_timing_pack.py` | ADDED | EURUSD_TIMING pack: definitions, validation, HAC lag screen (lags 0..k, BH), PCMCI+ wrapper (NOT_RUN without tigramite), matched-control contrast (event_impact.match_controls) |
| causal-inference `feature_selection_worker.py` | EXTENDED (append-only) | `TARGET_PACKS` += `EURUSD_TIMING`; `TARGET_CELL_COUNTS["EURUSD_TIMING"] = 4`; test TP01 asserts the old entries unchanged |
| causal-inference `provider/tests/test_fs_timing_pack.py` | ADDED | TP01-TP06 |
| heuristic-strategy `tools/short_horizon_timing_eval.py` | ADDED | the predictive-validation evaluator: per-bar MLP with the owner's monitor, temperature on 2023, threshold on 2023 by Sharpe through the sealed adapter, 2024 once, identical rows for every set |
| heuristic-strategy `tests/unit_tests/test_short_horizon_timing_eval.py` | ADDED | TE01-TE04 |
| dataset_selection `docs/PORTABILITY.md`, `examples/timing_pivot_eurusd.json`, `docs/TIMING_SELECTION_MAP.md` | EXTENDED (new row) / ADDED | extension table entry, example config, the map |

Engine stages that ran UNCHANGED on the timing population: `fs_phase23_manifest.build_manifest`
(the `targets` list accepts the new family/heads as data), `feature_pairwise_campaign` plan /
run-worker / follow / close-phase2, `feature_filter_selection.run_filter_methods` (its quantile coding
keeps a 3-class target's classes distinct, test `test_engine_mi_codes_keep_three_classes_distinct`),
`fs_phase23_warehouse_adapter` (sqlite3 backend where duckdb is absent). Code was needed only for the
targets, the price pool, the screens/reports, the causal adapter and the evaluator.

## 3. Runbook (CPU, memory-capped, DEVELOPMENT)

```bash
# tests first
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n fs-timing -- python -m pytest tools/test_timing_selection.py -q -p no:cacheprovider
# the selection (fill the placeholders of the example config in a private copy)
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 3600s -n fs-timing -- python tools/timing_selection.py run --config <CFG> --out <OUT>
# predictive validation through the strategy's harness (needs trading-contracts on PYTHONPATH)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<TRADING_CONTRACTS>/src crispdm-run -q -m 3G -t 3600s -n fs-timing -- \
  python <HEURISTIC_STRATEGY>/tools/short_horizon_timing_eval.py --eval-input <OUT>/eval_input --data <LAKE>/1h.parquet --out <OUT>/eval
python tools/timing_selection.py attach-eval --out <OUT> --eval <OUT>/eval
```

Outputs under `<OUT>`: `MANIFEST.json` (population, `fs_phase23.population_manifest.v1`), `TARGETS.json`,
`campaign/` (PLAN, terminals, `PHASE_2_COMPLETE.json`, `PHASE_3_FILTER_COMPLETE.json`,
`CANDIDATES_FOR_VALIDATION.json`, `PHASE2_MATRICES.npz`), `metrics.duckdb` (sqlite3 or duckdb),
`REPORT.json` (screens, stability, within-day variance, redundancy, recipe), `CAUSAL_STAGE.json`,
`eval_input/` (features/targets for all loaded rows + `FEATURE_SETS.json`), `SELECTION_MANIFEST.json`
(`feature_selection.manifest.v1`, DRAFT until an independent decision record exists).

Rules enforced: TEST rows refused at every loader; TRAIN = 2020-2022 for the engines; INNER 2023 only in
the evaluator's monitor/threshold; CONFIRM 2024 read once by the evaluator; labels purged at every cut;
no host name, address or private path in any artifact (paths are redacted to basenames).
