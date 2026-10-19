# Audit request to Musashi: feature-selection PLAN for expert-guided RL (ZigZag expert)

From: Satoshi (technical lead, ML-based strategies). To: Musashi (auditor). Date: 2026-10-19.
Status of the subject: METHOD UNDER EXECUTION; the sealed output will carry
`PRELIMINARY_PENDING_PLAN_AUDIT` until your findings are incorporated. This request
audits the plan, not results. A second request with results follows at closure.

## 1. Purpose and what the "expert" is

Select the observation features for an RL agent trained with guidance from a
hindsight expert, where the expert is the set of trades produced by the optimized
MetaTrader-semantics ZigZag on EURUSD H1: each pivot (valley/peak) is simultaneously
the close tick of one order and the open tick of the next. The selection target is
TIMING, not price level. Nothing here predicts price; nothing here selects for
price-level MAE.

Owner constraints in force: EURUSD H1; TRAIN 2020–2023 with 2023 as inner decision
year; VALIDATION 2024 used once as confirmation; TEST 2025 closed; every order with
effective SL/TP and integer units (`percentual_order_size × rel_volume(0.10) ×
balance × leverage / price`); ≥3 orders/week, ≤4/day; no centred filters; everything
additive (new plugins/configs only, existing prediction tooling untouched).

## 2. Expert signal (input to the plan, already verified in simulation)

- Indicator: `app/short_horizon/zigzag.py` (heuristic-strategy, branch
  `satoshi/short-horizon-strategy-20261018`, module sha256 `b30788e6…`), structural
  replication of ZigZag.mq5 (Depth window, Deviation test, Backstep erasure,
  alternating selection). NON-CAUSAL (repaints); used only for labels/expert.
- Parameters in use: Depth 3, Deviation 36.3 pips, Backstep 2 (NSGA-II on 2020–2022,
  chosen on 2023, confirmed once on 2024: hindsight Sharpe 18.9, 14.7 orders/week,
  97 % TP exits). A better-budgeted NSGA-II (pop 64 × ≥80 gen × 3 seeds, log-scale
  genes Depth 2–200 / Deviation 1–300 pips / Backstep<Depth, hypervolume convergence,
  hindsight and causally-confirmed objectives, daily-cap sensitivity) is running; if
  it changes the parameters, the targets below are rebuilt and this plan re-audited.
- Two versions: (a) HINDSIGHT pivots (expert); (b) CAUSALLY-CONFIRMED pivots
  (timestamped at the bar where the algorithm makes them final; median lag 6 bars).
  Measured gap (a)−(b) ≈ 21 Sharpe; half timing, half next-pivot level.
- gym-fx parity of the expert trades: net +2.7 %, max drawdown equal (hold-to-reversal
  contract; gym-fx lacks per-trade SL/TP, hourly spread, swap, forced-flat calendar).
- Demonstrations: 3,791 sealed trades 2020–2024, sha256 `352ddf6a…`, label
  HINDSIGHT_NON_CAUSAL_EXPERT.

## 3. Targets (labels only; hindsight ground truth of the future)

| id | definition | notes |
|---|---|---|
| T1 `pivot_in_k` | 3-class {none, valley within (t, t+k], peak within (t, t+k]} from HINDSIGHT pivots; k ∈ {2, 4} bars (primary k=2) | one signal per k bars; rows whose label window crosses a split cut are purged |
| T2 `expert_action_confirmed` | the expert's action at t from the CONFIRMED-timing version (what was knowable at t) | the learnable version of the expert |
| T3 `expert_trade_pnl` | P&L attribution of the expert trade that is open/opened at t, after costs | regression head; for ranking by economic relevance |

Question for the auditor: is T1 (hindsight) acceptable as a selection target, given
that imitability measured so far is ≈ 0 (see §7), or should T2 be the primary target?

## 4. Candidate pool (89 columns at the time of writing)

- 30 inventory candidates: the owners' technical generator (feature-eng `d081d0f`,
  financial-data preparation `41f5af170`), i.e. the same population from which
  Musashi's 15 admitted-core inputs (identity `553fc047…`) were selected.
- 46 multi-scale PRICE features, causal, availability = bar close: log returns and
  ranges at 1/2/4/6/12/24/48 h, close location in range, efficiency ratio, realized
  volatility, ATR ratios, drawdown/drawup, session/hour/dow encodings.
- 12 causal fractal columns (Retsu's family recomputed for EURUSD H1 as feature-eng
  plugin `causal_fractal`, branch `satoshi/causal-fractal-eurusd-20261019` @6c028aa,
  store manifest sha256 `906578e1…`): hurst_rs, variogram_hurst,
  box_counting_dimension, hurst_spectrum_width at windows 200/96/48 observed bars.
  Known: fractal_dimension_graph = −hurst_rs (alias, dropped); |r| ≤ 0.13 vs
  rv5/rv24/ATR except spectrum_width–rv24 0.23–0.28; hurst_rs biased up vs
  variogram; they vary within the day (more at short windows).
- Excluded on purpose: the old 19-feature I6-DMO manifest (`2ff34fa2…`, 14 daily
  series) — its matrix is not in a public store; it is reported NOT_COMPARABLE.
  Also excluded: any value with availability_time > t; any centred/smoothed series.

Question: should the 15 admitted-core inputs be forced into the pool as a protected
baseline block (always evaluated), or compete like any other candidate? (Plan: both —
they compete, AND the 15-set is evaluated as a fixed baseline on identical rows.)

## 5. Stages (all additive to the pinned engines)

1. Availability/causality gate: `available_time ≤ t`; refusal of centred filters.
2. Phase 2 (predictor `tools/feature_pairwise_campaign.py`, unchanged): pairwise
   dependence/redundancy over prefix TRAIN folds (2020–2022), Spearman/MI/dCor, lags.
3. Phase 3 (predictor `tools/feature_filter_selection.py`, nine methods unchanged):
   rankings and K-grid subsets per target T1/T2/T3; controls ALL_ADMISSIBLE,
   UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K.
4. Causal stage (causal-inference, NEW target pack `EURUSD_TIMING`, append-only
   registration with byte-identity test of the existing EURUSD/ETH packs): HAC lag
   screen with FDR; matched-control contrast (event-impact style) of pivot vs
   non-pivot windows at equal hour/dow/pre-vol; PCMCI (tigramite, pinned in the
   project venv) with the pivot label as dependent series and lagged candidates
   (lags 1..k). NOT_IDENTIFIED stays neutral, never an exclusion.
5. Predictive validation (NEW evaluator in heuristic-strategy): the pivot-detector
   harness — m1 I6-DMO-class network (per-feature causal Conv1D branches, fusion,
   physical-time encoder, softmax head), m2 GRU on a 48-bar window, m3 gradient
   boosting; 3 seeds; owner's stopping rule (mean of the epoch's running TRAIN loss
   and INNER-2023 loss, patience 10, min_delta 1e-8, best restored); temperature
   calibration and threshold chosen on 2023; scored by AUC/log-loss/precision at
   threshold AND by the Sharpe of the trivial pivot-reversal strategy through the
   sealed adapter (SL m·σ̂, TP 2σ̂, units rule, frequency rule, realistic costs).
6. Decision on 2023 only; single confirmation on 2024; sealed SELECTION_MANIFEST
   (features, hashes, target id, parameters, folds, engine revisions) consumable by
   gym-fx's observation contract and by heuristic strategies.
7. Reports: redundancy clusters, fold stability, within-day variance share (to catch
   daily-constant features), per-feature causal table, files-touched additivity table.

## 6. Decision rule

Primary: the subset (per K and method) with the highest 2023 Sharpe of the
reversal strategy driven by the m1 detector, among subsets whose precision at the
operating threshold exceeds the base rate with FDR-corrected significance; ties →
fewer features, then lower cost. Secondary report: AUC/log-loss. The 15-input
baseline is scored on identical rows. 2024 is reported once, after the choice.

## 7. Evidence that bounds expectations (for the auditor's calibration)

- Pivot detectors so far: on the old 19 manifest AUC 0.50–0.55; on Musashi's 15
  inputs AUC 0.66 but precision 0.34–0.37 < base rate 0.38–0.39, Sharpe −3.5 … +0.07;
  oracle break-even precision ≈ 0.6–0.7; perfect timing with a causal TP gives
  Sharpe 8–12, not 19 (`satoshi/pivot-detector-20261019` @052dda5).
- Price-level direction ≈ coin across five model families; volatility is predictable
  (HAR 15–23 % below naive) but alone cannot cross zero after costs
  (`ROUND_CLOSURE_2026_10_19.md`, heuristic-strategy).
- Macro events: real vol impulse (NFP 3.3×, t½ 0.3–0.8 h), trend only via surprise,
  too small at hourly resolution; calendar = risk filter (causal-inference
  `satoshi/event-impact-20261019` @a4cb9ea).

## 8. What to attack first (my own weakest points)

1. T1 uses hindsight labels: a feature can "explain" a pivot only through future
   information leaking into the label window; purges are per split cut, not per row —
   is a per-row embargo of k bars after each label needed for the fold diagnostics?
2. The decision rule optimizes a strategy Sharpe on 2023 across many (method × K ×
   target) subsets — multiple-comparison inflation; I plan FDR on precision but not on
   Sharpe. Is a deflated-Sharpe / PBO-style check required before the 2024 read?
3. The 46-feature price pool was designed by me, not selected by a prior phase-1
   profile; its availability is bar close, but no phase-1 quality profile exists yet.
4. The expert parameters may change after the full NSGA-II; the plan rebuilds the
   targets, but the fold structure and pool stay — is a fresh re-audit of stage 5
   enough, or must everything rerun?
5. PCMCI with a categorical dependent series (3-class) — I will encode it as two
   binary indicators; is that acceptable or should T3 (continuous P&L) be the causal
   dependent instead?
6. The predictive evaluator reuses the detector harness whose m3 arm memorizes
   under the loss-mean monitor (TRAIN precision 0.99); the monitor is the owner's
   rule — should m3 use early stopping on INNER loss only as a declared exception?
7. The lake bar clock is under repair; Musashi's repaired clock is +5/+6 h vs lake
   labels (DST); joins are by identical OHLC. Any selection result inherits this.

## 9. What is NOT done / not claimed

No RL training, no policy value claim, no TEST read, no deploy. The old 19 manifest
is NOT_COMPARABLE (no public matrix). The imitability ceiling may remain ≈ 0; if so,
the deliverable is a measured negative, not a selected set.

## 10. Deliverables you will receive at closure

Sealed SELECTION_MANIFEST + hashes; tooling map (file@commit); runbook with
`--help`-verified commands; files-touched additivity table with byte-identity proofs;
redundancy/stability/within-day tables; causal table; E0 numbers (new set vs the
15-input baseline on identical rows, 2023 then 2024 once); the full-results audit
request. Branches: `satoshi/timing-*` in predictor, causal-inference,
heuristic-strategy and dataset_selection.
