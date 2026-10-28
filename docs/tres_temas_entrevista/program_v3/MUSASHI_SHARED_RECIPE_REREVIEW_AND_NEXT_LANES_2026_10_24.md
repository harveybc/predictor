# Shared Recipe Re-review And Next Lanes

## Decision

ACCEPT_SHARED_RECIPE_FOR_DEVELOPMENT. The measured v5 selection, T1 profiles,
T2 decision, T3 group decisions and combination C close this selection cycle.
They are the identified starting point for both research lanes, not a proof
that the inputs or representation are globally optimal or profitable.

No new selection or T3 rerun is required by this review. No policy campaign
was started here. Consumer-to-simulator integration and the new launch specs
are separate engineering work, not unfinished feature selection.

Reviewed Satoshi delivery: predictor 9a17e24c, heuristic-strategy 81f30d6.
Learned-producer implementation: predictor 583f5087. Scientific T3 producer:
dc8d4aac63ba973f67124c1e0968a123fb39bfa1. The old v4 recipe is not admissible.
Review fixes published: heuristic-strategy `338551a` on
`codex/recipe-v2-rereview-20261024`; SAC consumer `d88545b8` on
`codex/rl-static-temporal-20261018`.

## Fixed Baseline

| Object | Accepted value |
| --- | --- |
| Recipe | recipe_5a8f2155318ba7c7 |
| Recipe file SHA256 | 93ac66dc40de4b48a65a1455b0a391f15aad78cae43738a4e7d164d768cf118b |
| Store SHA256 | cd0eb8d20c0d0888f7814cda410557a79a927e5cc125e9de5c15dbe073d2f543 |
| Ordered columns | 121, strict v5, never an inferred directory member |
| Upstream encodings | 24 transformed columns recorded individually in recipe |
| Consumer encodings | Zero additional differences/transforms; apply frozen constants only |
| Scaler | Fixed TRAIN constants, fitted before 2023-07-01; no rolling or clipping |
| Window | 13 observed H1 slots, stride 2, includes the current completed bar |
| Elapsed window | Normally 24 hours; can span closures, measured maximum 105 hours |
| Group definition | 25 conceptual groups; four changed by T3 |
| Effective extractors | 31 branches: three pointwise and 28 causal convolution branches |
| Changed groups | inv.raw, inv.momentum, hilbert/log_close -> pointwise; px.close_loc -> seven per-column branches |
| Proxy regularization | Dropout 0.3, weight decay 0.01; measured for E3-M, not established optimal for SAC |
| Policy observations | Market features only; no agent feedback or expert labels |
| Main split | TRAIN 2020-2023, VALIDATION 2024, TEST 2025 closed |
| Internal split | Policy FIT 2020-2023H1; INNER_A 2023H2, explicitly within main TRAIN |

`rel24` is a measured encoding alternative, not an extra hidden operation.
The selected C's 121-column tensor does not contain an `enc.*` column.
Appending relative encodings is a separately identified future contrast; do
not claim that every encoding ever retained by a screen was consumed by C.
The recipe contains the full ordered list, per-column provenance and resolved
groups. T1 describes measured profiles; its expectations do not override T3.

Keep SL/TP on every order, sizing and leverage contract (baseline 1:50), one
entry/day, costs, next-open fills, no weekend position, force-flat 12h before
close and entry delay 12h after reopen. The two-orders/week rule is a period
average validity check, not forced trading every week. Missing closed-market
bars consume no observation slots; interior unavailable values are not filled.

## Re-review Evidence

Focused executed suites: 67 heuristic-strategy tests, seven upstream fit-scope
tests and 17 native SAC/input tests, all passing (91 total, not a full-suite
claim). No GPU training was used for this review.

| Finding | Verification | Disposition |
| --- | --- | --- |
| A1 upstream fitted state | Seven fit-scope regressions executed; real consumer rebuild-with-fit witness repeated with 3,086 perturbed held-out bars. Parameters and earlier values identical; contaminated positive control changes 270,509 earlier cells. Remote code SHA equals reviewed local code. | Corrected for the tested scope; peer retains the second real fold witness and ten scope reports |
| A2 recipe derivation | Real emitter rerun against C's complete receipt chain, selected list/store, groups and replay. Output SHA remains byte-identical to delivered recipe. Added five adversarial replay tests. | Corrected, with additional exporter fix below |
| A3 actual SAC consumption | Independent SAC loader reproduces 23x13x121 probe, window-valid masks, exact actor/twin-critic inputs, same-weight trunk forward and policy save/reload. No expert module generates its observations. | PASS for observation/model path; not a trading-engine acceptance test |
| A4 stage/clock semantics | Upstream vs consumer operations, ordered names, frozen constants, observed-row window and availability at bar-close are applied explicitly. | Closed for this recipe |

Additional defect found and fixed locally in the exporter: a `PASS` header with
12 checkpoints did not bind the replay unit to the current C result and did
not inspect individual module/rollout exactness. Five retained mutations
(stale, absent units, inexact module, inexact rollout, and misleading threshold)
all passed before the fix and are rejected after it. This does not invalidate
the authentic C: the original finalizer already made those checks, and the
strengthened exporter reproduces the exact same recipe bytes. Integrate the
`codex/recipe-v2-rereview-20261024` patch before future exports.

Evidence in `docs/audits/evidence/shared_recipe_rereview_20261024/`:
`SAC_PROBE.json`, `CHECK_consumer.json`. The reproducible SAC command is
`tools/verify_sac_shared_recipe.py --help`; give explicit recipe/reference
SHA256 pins and the admitted local paths. It constructs a native SB3 policy
on CPU, performs zero environment steps and zero scientific fits. The
window-valid counts (19,514 / 2,589 / 5,600) are NOT business-entry counts or
ZigZag-label eligibility counts; SAC must never borrow retrospective label
eligibility as an action/observation mask.

## Measured Result

| Representation | TRAIN Sharpe | INNER_A Sharpe | TRAIN-minus-INNER gap | VALIDATION 2024 Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Control | 2.9975 | 0.8362 | 2.1613 | -0.5722 |
| C | 2.5723 | 1.3920 | 1.1803 | -1.0112 |

Three-seed means. Frozen C rule is unchanged. 2024 is reported, not used by
that decision. All 45 units were authenticated; 540 original checkpoint probes
and 12 C probes are exact in their original CPU contexts. Control and C each
have nine exact full trading replays. No new profitability measurement here.

The list was selected using main TRAIN 2020-2023, including 2023H2. Therefore
INNER_A is not independent of list selection, although the consumer producer
fit and scaler exclude it. Keep that development limitation; do not redo
selection at July 2023 without a new explicit experimental question. 2024 has
also been inspected across this research program: it is not a pristine final
holdout. TEST is still closed.

## Exact Next Work

1. Satoshi: integrate the replay-binding fix; replace the stale v4 release and
   recipe references in the clean-round launcher/spec. The current captured
   `CLEAN_ROUND_SPEC_A1.json` still contains old release-state text and
   recipe_6f522eec4ffdacbc, and `e3m_cmd` does not add `--recipe`. Produce a fresh
   pinned command using the accepted recipe and C groups, and prove no scaler
   refit or implicit transform. This is launcher repair, not another T2/T3.
2. Satoshi: finish the expert-guided finite run spec on the assigned worker.
   Demonstrations and opportunity/both labels remain training-only treatments;
   do not relabel them as actual human feedback. Never feed hindsight into
   observations or real-time eligibility. Share exact spec and read-back.
3. Musashi: integrate the tested frozen observation adapter and grouped trunk
   into the actual gym-fx rollout path. Verify timestamp/action alignment,
   business masks and all execution costs against the current engine; retain
   engine position/SL/TP/accounting internally without policy feedback.
4. Musashi: pin the independent SAC finite comparison and its optimizer
   regularization explicitly. T2's Adam/weight-decay setting must not silently
   become a different optimizer in SAC. Use fixed budget/sparse diagnostics,
   no expensive per-checkpoint early-stopping loop or periodic retraining.
5. Launch each lane with its own sequential supervisor, disjoint assigned GPU
   resources, restart/receipts, model weights and warehouse read-back. Report
   TRAIN and VALIDATION profit, Sharpe, drawdown, orders, win rate, costs,
   activity and generalization gaps for every arm, including negative results.

The shared acceptance is not a fabricated `RELEASE.json` for the obsolete
launcher. No new owner decision about selection is needed for the above
integration work. Public/private path separation and immutable old receipts
remain intact. Warehouse delivery is not certified by the recipe or this
no-fit probe; verify it when wiring the campaign publisher.
