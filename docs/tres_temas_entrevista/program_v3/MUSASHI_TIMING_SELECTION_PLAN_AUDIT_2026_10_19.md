# Timing Selection Plan Audit

Auditor: Musashi. Date: 2026-10-19. Verdict: REVISE_BEFORE_FINAL_SEAL.
This is a document/design audit, NOT an implementation or results certificate.
No expert simulation, feature matrix, causal test or detector was rerun here.
All numerical expert/detector claims below remain attributed to the request.

## Objects And Scope

Subject: predictor commit `5b2cd008`,
`docs/tres_temas_entrevista/program_v3/AUDIT_REQUEST_TIMING_SELECTION_PLAN_2026_10_19.md`.
Related design note at the same commit: `RL_FROM_HINDSIGHT_EXPERT_DESIGN_NOTE.md`.
Line references F01-F09 are to the subject, not to this report.
The active selected-input SAC campaign remains separate and unchanged. This
audit neither launches RL nor replaces its selected inputs or business contract.

## Findings, Ordered By Risk

### F01 - Label support extends beyond the nominal k-bar horizon

HIGH; design gap demonstrated by lines47-49 and128-130. A final hindsight
pivot inside (t,t+k] can depend on bars after t+k through repainting and
confirmation. T3 can depend on the exit of a long-running trade. Purging only
the k-bar label window at a split is not sufficient to establish independence.
This does not prove that the implementation currently leaks.

For every row retain origin_time, event_time, label_available_time,
label_support_end, expert_trade_id and target/expert digests. Compute the
support end from every bar actually used to determine pivot finality, exit,
SL/TP and cost, not a median confirmation delay. Unconfirmed tail events are
censored, not silently labelled NONE or zero P&L. Purge FIT rows whose label
support reaches the evaluation boundary. Protect event/trade groups from
cross-fold duplication; account for serial overlap when estimating uncertainty.
No arbitrary k-bar embargo after EVERY row is required: specify temporal folds,
actual overlapping label support and any additional dependence buffer explicitly.

Future-dependent labels are allowed for supervised learning; all features and
live actions must still use information available at origin t. TRAIN labels
must not derive from an expert fitted using later decision/confirmation data
when claiming an independent early fold. If expert selection used 2023, call
2023 joint development rather than independent generalization of that expert.

### F02 - 2024 is already observed, not a pristine confirmation

HIGH; documented contradiction, lines18/30/99/111 and115-118. Expert selection
already reports 2024 confirmation, and prior detectors are reported too. Audit
the read ledger: which periods, labels and metrics influenced each decision?
Do not erase those reads or rename 2024 an independent one-shot holdout.

Minimal resolution: retain the accepted 2020-2023 fitting/development plan and
report 2024 as previously observed development/reconfirmation. Freeze the next
candidate before its next read and forbid feedback-driven reranking disguised
as confirmation. TEST2025 remains closed under this order; no extra external
holdout or rerun of our selection is required to continue engineering.

### F03 - Expert parity is not yet business parity

HIGH; admitted limitation plus missing specification, lines12/17-21/38-41/97-98.
Hold-to-reversal parity without protections, costs and forced flattening does
not certify the proposed deployable strategy or its hindsight Sharpe.

Pin the EXACT gym-fx adapter/revision, calendar, bar completion timestamp and
execution convention. Our current adapter has protections and session rules;
the related note's blanket assertion that gym-fx lacks them is not a current
cross-repository fact. Inspect the revision Satoshi actually intends to reuse;
do not reimplement the environment from that assertion.

Retrospective pivot high/low may be a LABEL but is not a guaranteed executable
fill. Decide after the completed observation; fill at the next executable
quote under the pinned adapter. Specify SL/TP collision, gaps, spread, slippage,
financing, integer minimum/step and margin headroom. Test forced flat12h before
closure and no reopening until12h after reopening, plus the declared wind-down.
Count entries separately from closes: close-and-reverse is not a free action.
The lane's requested >=3/week and <=4/day must remain explicitly separate from
our current <=1/day SAC contract. A minimum-frequency objective cannot force
an illegal entry or override no-trade/insufficient-margin states.

### F04 - Candidate clock and phase-1 quality are not closed

HIGH; design omissions and count discrepancy, lines54-71/79/134-146.
30+46+12=88, not89; reconcile named ordered columns, aliases and hashes before
sealing. The new feature families require their own phase-1 quality profiles;
the existing core population's admission does not admit new columns by proxy.

Reuse the repaired admitted root where appropriate, with pinned data/clock
identity, not a hand-applied +5/+6h shift or OHLC-only join. Key by unique asset,
timestamp, bar convention and source; OHLC agreement is supporting evidence,
not a unique key. Profile coverage, duplicates, finite values, warm-up, extrema,
constants and availability. Preserve genuine extremes; no interpolation or
fabricated weekend bars. Verify prefix invariance and future perturbations for
every generator, scaling step and fractal window. Declare market-bar versus
elapsed-hour units. Causal trailing smoothing is not automatically look-ahead;
the current no-smoothing choice can nevertheless remain fixed as requested.

### F05 - T1, T2 and T3 are different tasks, not interchangeable targets

HIGH; ambiguous semantics, lines47-52/91-111. Keep T1 as the primary
ANTICIPATORY experiment if the intended business is anticipating a reversal.
Keep T2 as a separately reported reactive/confirmed-rule comparator. Merely
reproducing a causal deterministic indicator does not establish anticipation
or profitability. Negative detector experiments do not prove zero imitability.

For T1 freeze what happens if both valley and peak occur inside k, how k maps
across closed hours, endpoint inclusion, class weights and output origin/fill.
Prefer earliest event with deterministic tie handling, or an explicitly
multilabel design. Do not quietly thin hourly origins to one every k bars.
Report per-direction PR-AUC, precision, recall and support; identify whether
NONE is included in the base rate. T3 must specify remaining/net return versus
full-trade P&L and the agent position/action to which it applies. Full-trade
profit repeated at every row is not an independent action-value label.

Keep the 15-input set as an unchanged, scored baseline on identical rows; its
members also compete normally in the candidate pool. No mandatory inclusion
of all15 in each selected subset. Matching support must not conceal which
samples were lost to the longest new-feature warm-up.

### F06 - Selection significance does not survive adaptive search by FDR alone

HIGH; lines94-111/131-144. The same2023 interval chooses checkpoints,
calibration, thresholds and subsets. Precision FDR alone does not correct a
Sharpe winner selected across that whole adaptive search. Ordinary IID tests
are also unsuitable for overlapping labels/trades.

Freeze a bounded trial ledger including expert/config/K/target/model/seed and
all attempted thresholds. Retain the approved mean TRAIN/INNER stopping rule;
do NOT silently switch m3 to INNER-only. Measure TRAIN and INNER loss in the
same units, class weighting and inference mode at the same checkpoint; retain
running training loss separately if it includes changing weights/stochastic
training behavior. Log maximum budget, best iteration, patience and stop cause.

Use ordered, purged internal blocks for calibration/threshold screening and
score subsequent blocks without updating those choices on their outcomes.
If expert tuning already used all2023, those blocks do not restore independent
expert evaluation; disclose joint development as F01/F02 require. Report
selection-aware uncertainty with a declared dependence-preserving procedure
and trial family. DSR can supplement the ledger; it is not a magic correction
for arbitrary dependent trials. Full CSCV/PBO is not mandatory, particularly
if it would break temporal ordering. A memorizing m3 warrants a separate,
declared capacity/regularization ablation, not an unannounced monitor change.

### F07 - Causal association is not an intervention on a ZigZag event

MAJOR; lines85-90/139-141. Price-derived candidate indicators and a future
price-derived pivot share construction/parents. HAC/FDR or matched hour/dow/
volatility controls do not alone identify intervention or counterfactual
effects. Mark the result CONDITIONAL_PREDICTIVE_ASSOCIATION unless explicit
causal assumptions and estimands are justified; retain NOT_IDENTIFIED neutral.
Do not drop a useful feature solely because a causal hypothesis is unresolved.

Use a conditional-independence test valid for the actual mixed continuous/
categorical data and pinned Tigramite version, e.g. RegressionCI with its linear
assumptions or CMIknnMixed where supported. Two dummy indicators do not by
themselves validate a continuous/Gaussian ParCorr test. Retain type masks,
mutual-exclusivity handling, lags, test parameters, negative controls and
appropriate temporal resampling. Do not replace T1 with continuous T3 merely
to make a test convenient. Lag0 here can mean the known completed bar at t;
declare the timestamp mapping before excluding all current-bar information.

### F08 - Target identity changes invalidate more than detector stage5

MAJOR; lines29-34/136-138. If ZigZag parameters or label/fill/business semantics
change, recompute affected labels, per-target rankings/MI, causal tests,
purges, detector fits, calibration, strategy scores and final manifests.
Reuse only target-independent quality/pairwise objects with exact unchanged
inputs, sample masks and generation identities. Use a dependency DAG; do not
repeat independent expensive work. Keep provisional results marked as such.

### F09 - A detector-selected subset is not yet an RL-selected subset

MAJOR; lines91-111/148-152. Selection by m1 pivot classification and its
reversal strategy establishes usefulness for that proxy only, not for SAC.
Name the manifest TIMING_PROXY_SELECTION. Later, after owner acceptance,
compare fixed-budget SAC with15 versus the new subset using identical business
rules/seeds/support. That is a later integration, not authorized by this audit.
Separate BUY/SELL policies are an optional ablation, not a prerequisite: a
shared temporal trunk with direction-specific heads is also testable. Include
position, cash/margin, current SL/TP, holding time and session state; close and
open labels depend on the student's actual state, not merely the expert trade.

## Related Note Corrections, Not New Implementation Orders

This is expert-demonstration/hindsight-guided learning, not human-preference
RLHF unless humans actually supply preferences/rewards. Terminology does not
invalidate the experiment. A future oracle in training labels is legitimate;
it need not be perfectly realizable and its sampled detector score is not a
theoretical upper bound on all causal students. Do not describe future-aware
critic inputs or Q-filtering as automatically leakage-safe or sufficient to
make an unattainable expert attainable. State the learning objective and
validate each such ablation separately.

Potential shaping requires an actual boundary proof, not just mentioning Ng.
For a trajectory of length T, the shaping contribution telescopes to
`-Phi(s0) + gamma**T * Phi(sT)`. Freeze the potential definition and ensure the
terminal contribution cannot introduce a policy-dependent objective, including
forced close, episode truncation and randomized starts. Report unshaped net
returns. No privileged-critic/shaping/custom-SAC changes are part of our active
campaign or granted by this plan audit.

## Minimal Acceptance Evidence For Satoshi

| ID | Required executable evidence |
| --- | --- |
| A01 | Row support/availability ledger; split-crossing confirmation and long-trade negatives are purged |
| A02 | Repaint/tail perturbation changes permitted labels, never inputs at or before t; no unavailable label is used in FIT |
| A03 | Prefix-invariant confirmed expert; finality/censoring tested, not inferred from median lag |
| A04 | Simultaneous peak/valley and weekend-clock target cases; all hourly origins accounted for |
| A05 | Ordered88-or-corrected-count inventory, new profiles, repaired-root joins and warm-up support table |
| A06 | Effective environment revision; next-executable-fill, SL/TP collision, calendar, sizing and entry-cap cases |
| A07 | Immutable15 baseline; same rows, model budget and seed table versus candidate subsets |
| A08 | Mixed-data CI test with known negative/positive controls and declared statistical assumptions |
| A09 | Trial/read ledger, temporal calibration/threshold policy, dependence-aware uncertainty; no pristine2024 claim |
| A10 | Expert hash change invalidates every target-dependent stage; independent stages remain reusable |
| A11 | Per-role/per-direction metrics plus net PnL, drawdown, trades, win%, costs, Sharpe definition and no-trade comparator |
| A12 | Idempotent OLAP publication/readback, local snapshot and second physical copy; manifest-consumer negative tests |

## Bounded Return Order

Satoshi owns the minimum corrections on his branch. Do not merge this review
into an active scientific runtime. Continue independent inventory/profiling
and implementation; recompute only affected provisional stages. Before final
seal, return one table F01-F09 -> disposition -> exact artifact/test/commit,
the A01-A12 results, dependency invalidation and effective runtime pins.
No periodic agent log monitoring; finite scripts/timers own execution and
status. Never claim an unrun check passed. Do not require another owner
approval to correct these document/specification defects. New RL experiments,
new broker contracts and our campaign changes remain outside this lane.

## Primary References

- [Human preference RL](https://arxiv.org/abs/1706.03741): preference-based reward learning, distinct from these synthetic demonstrations.
- [Hindsight Learning for MDPs with Exogenous Inputs](https://proceedings.mlr.press/v202/sinclair23a.html): legitimate hindsight-learning framework, not a trading profitability guarantee.
- [Tigramite assumptions and mixed-data tests](https://github.com/jakobrunge/tigramite/blob/master/README.md): match the CI test and causal assumptions to the actual data.
- [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf): selection bias and non-normality; adaptation here requires its stated assumptions.
- [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf): conditions for potential-based shaping, not blanket permission for arbitrary hindsight rewards.
