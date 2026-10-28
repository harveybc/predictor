# Expert-guided plan review and bounded follow-up

Date: 2026-10-25 UTC. Reviewer: Musashi. Scope: current plan, label producer,
weighted-loss wiring and threshold-search protocol. This is not a certification
of the entire campaign. No live code, sealed result, dataset or worker was changed.

Reviewed heuristic-strategy revisions:

- Plan: `12d8de32cc2773ac8b3f23a4c174c2532a33b6e1`.
- Expert implementation/spec: `9742ccdf827008c0ff70c6b833246a46aab6747c`.
- Relevant sources: `app/short_horizon/expert_guided/multires.py`, `learn.py`,
  `tools/expert_guided_e3m.py`, `tools/expert_guided_multires_deap.py`,
  `docs/short_horizon/clean_round/EXPERT_GUIDED_RUN_SPEC.md`.

## Disposition

Keep the current versioned comparisons and their results. Correct the weighted
NOP defect in a new identity, quantify its incidence, and rerun only affected
arms. The unweighted arm does not acquire this loss-weight defect. Do not silently
modify a live run. The shared feature-selection chain does not need to be repeated
for these label/loss issues. Our SAC remains a separate, expert-free experiment.

## 1. High priority: risk-rejected NOP can have zero loss weight

**Demonstrated producer-function counterexample.** In `multires.py:241`, rejection
for adverse excursion or insufficient volatility uses weight `clip(1-f, 0, 1)`.
For `f=1`, `mae=0.8`, `theta_mae=0.5`, the label is NOP but its weight is zero.
`expert_guided_e3m.py::row_weight` passes that zero unchanged; `learn.py::fit`
multiplies the per-row classification loss by it. Thus an unsafe candidate can
contribute no classification gradient precisely when the NOP example is needed.
This establishes a possible mechanism, not its frequency in the real dataset.

Required bounded follow-up:

1. Report counts and summed weights by NOP reason, period, side and seed.
2. Separate risk/conflict penalties from opportunity-size weighting. Risk rejection
   must retain positive classification weight; choose and record the rule before
   examining student performance. Unit weight for hard risk violations is a simple
   candidate, not a claim that it is optimal.
3. Add regression cases for f >= 1, high adverse excursion, low leg/sigma, conflicts,
   and an all-zero-weight batch. Test actual loss gradients, not only label values.
4. Repeat the affected weighted arm under a new identity; retain its unweighted
   comparator and the same evaluation population.

## 2. No pivot vote is not evidence of no economic opportunity

**Demonstrated semantics, interpretation risk.** `origin_label` returns hard NOP
before checking economic quantities when no resolution votes. The retained witness
holds f=1, mae=0.1 and leg/sigma=10 fixed: no vote gives `nop_hard`; one long vote
gives `open`. This is valid imitation of the teacher's recommendation, but it must
not be described as proof that no admissible profitable trade exists.

Keep the approved teacher-reference rule. Add a TRAIN-only diagnostic for nonpivot
origins: compare admissible open/hold/close outcomes in the actual execution engine,
including costs, SL/TP and weekly closure. Report recommendation absence separately
from negative economic opportunity. A later paired label contrast can test the
difference; do not make this diagnostic a new selection of the calendar or TEST.

## 3. Close-quality proxy and executable close are different

**Confirmed definition; effect not yet measured.** `leg_tables` uses current CLOSE
and the optimal leg's start for f_close, while the demonstrator executes at next
OPEN. `positioned_action` uses the side, not the actual entry or order protections,
to select that reference. This is an explicit optimum-relative proxy, not a measured
net value of closing the student's particular position. Future-derived training
labels are not themselves leakage into observations.

Add engine-level witnesses: identical current CLOSE with different next OPEN;
same side with early/late entry and different SL/TP; stop/target hit at the next
bar; and forced weekly closure. Compare close-now versus hold with identical
execution/cost ordering. Quantify disagreements before deciding whether an
execution-aligned label variant is useful. Do not alter the business engine.

## 4. Threshold selection and the student's INNER block overlap

**Confirmed protocol.** DEAP selects thresholds using all of 2023; student INNER_A
is 2023 H2. That block is therefore development/selection for both teacher and
student, not independent confirmation of teacher-threshold generalization.
Preserve and report those results as development; no convenient replacement year.

There is also a concrete cutoff mismatch to quantify: `replay` changes the DEAP
simulation interval to 2020-2022, but `Tables.row_end_ns` still uses consolidated
TRAIN's 2023-07-01 cutoff. Labels near the 2022 boundary can therefore be admitted
with a 2023 confirmation. The exported availability note already acknowledges the
row-period cutoff. Count affected labels, test perturbations after 2023-01-01, and
use the minimum of row-period and replay-period end for an isolated contrast when
claiming selection based only on 2020-2022. This review has not measured incidence.

## 5. Make comparisons complete and reuse identities strict

- Stage A is 4/7 completed, not a finished seven-arm comparison. The three
  memory-refused shorter-demo arms are not negative scientific results. Run them
  on an admitted worker, or use indexed/lazy windows with a tensor-parity test;
  do not silently shorten the dataset or change batch size.
- Current A and A-prime intentionally share recipe v2. Keep that control. The v3
  bridge needs both control and finalist on the new support; identical logits on
  old common rows do not certify trading on newly available rows.
- Report per-seed paired deltas, valid seeds, actual updates, unique origins,
  censoring/support hashes and effective sample weights. Multiple ZigZag votes
  are correlated proposals, not independent additional observations.
- `CACHE.json` is keyed only by rounded genes, and terminal seed checkpoints are
  reused by filename without a data/code/contract identity check in the DEAP
  driver. Bind reuse to the full effective experiment identity and reject mismatches.
  This is a code-level reuse risk, not evidence that this campaign reused bad data.
- Update the stale `describe()['sample_weights']` claim that weights are not
  consumed. Preserve the explicit fact that the early-stop monitor is unweighted
  while the weighted arm's classification objective is weighted.
- Consolidate current plan invariants: one entry/day, activity >=2/week as period
  average, strong regularization .3/.01, and 2024 as already-observed development.
  Historical sections may remain, clearly superseded; no extra approval needed.

## Reproduction and limits

Run from predictor with the producer commit available in a local repository:

```bash
python3 docs/tres_temas_entrevista/program_v3/probes/satoshi_multires_review_20261025.py \
  --repo <heuristic-strategy-checkout>
```

Executed successfully on 2026-10-25. It loads the exact pinned function ASTs and
uses synthetic scalar fixtures, without importing the training stack or reading
market data. Output: risk-rejected NOP loss weight 0; identical economic quantities
yield hard NOP without a vote and OPEN with one. These are isolated producer-function
witnesses, not GPU/end-to-end replays. No TEST, fit, deployment or service change.

## Parallel SAC status at review

Our TARGET control completed in about 38 minutes and reproduced the parent policy's
final tensor digest exactly. COMMANDS5 started automatically and reached 50,000 of
100,000 decisions at 03:38 UTC. This compares target-position semantics with explicit
HOLD/OPEN/CLOSE commands; it does not add agent-state inputs, rolling scaling or
expert labels. The independent collector had verified 15/15 reports at 03:34 UTC.
Live campaign receipts, not this snapshot, determine current status and final metrics.
