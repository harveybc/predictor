# Required ML review and continuation of S1-S4

Owner: Musashi (independent scientific review); implementation: Satoshi.
This addendum is part of the current S1-S4 order, not a replacement campaign.
Do not restart completed experiments or discard evidence. Current work is design and
implementation of the bounded adequacy pilot; no new results are claimed by this document.

## Correction of scope, not an invented invalidation

The missed requirement was demonstrating that context, model, target and data volume
were adequate before using relative utility results for broader conclusions. A short
window and ridge are not intrinsically invalid: they define a restricted experiment.
Their adequacy for noisy signals and weekly trading has not been demonstrated.

Preserve data profiling, operator mechanics, governance, numerical observations and
utildev-v1 as diagnostic history. Keep its 18 non-advances and 18 inconclusive outcomes
at their original scope; do not silently upgrade or erase them. No representation is
rejected for all models/domains on that basis. Instrument controls validate only their
tested conditions. Recompute old results only if a concrete defect changes their stated
estimand or evidence, with a documented affected population and successor, not by default.

## Before any new scientific campaign

Include one review table in the design: requirement, executable check, evidence, status,
owner and next action. Status is PASS, FAIL or NOT_TESTED; prose and green software suites
cannot substitute for measured ML adequacy. Required rows:

1. Question and estimand: absolute predictive skill, relative representation utility,
   denoising, forecasting and policy utility are distinct; specify the decision supported.
2. Target and noise: clean vs observed, exact row/horizon identity, irreducible future
   uncertainty, meaningful baseline and a known-truth control appropriate to the claim.
3. Context and support: time span, periods/time scales, feature lookback, emission time,
   model receptive field/state and equal information availability across comparison arms.
4. Data sufficiency: actual usable training/evaluation rows, independent series/weeks,
   regime coverage and learning curves. Overlapping windows are not independent replicates.
5. Model adequacy: actual graph, optimizer updates, train/validation curves, capacity and
   context ablations. Explicitly distinguish underfitting, overfitting and optimization failure.
6. Temporal validity: fitting and tuning train-only; nested chronological validation,
   label overlap separation, exact deployed-path prefix/future/missingness/restart tests.
7. Comparison fairness: raw skill first; match decision times and targets; account for
   information span as well as input width, model capacity, compute and tuning allowance.
8. Statistics: practical effect, uncertainty, multiplicity, independent replication,
   calibration scope, predefined stopping and handling of incomplete results. No repeated
   sample-size increases or seed searches until a favorable decision appears.
9. Business transfer: forecasting and RL tracked separately; weekly training cutoff,
   release time, costs and next-week evaluation. A transport/replay test is not policy evidence.
10. Independent evidence: losses recomputed from predictions and labels, rows linked to
    governed inputs, results reconciled by content, and a negative test that breaks each
    claimed guarantee. State what was freshly queried versus read from retained receipts.

An instrument/adequacy pilot is allowed to investigate NOT_TESTED rows. Its completion
does not authorize downstream selection when those requirements remain unmet. This is
a scientific review, not another service, credential or manual permission workflow.

## Execution order and ownership

| Stage | Owner | Deliverable and boundary |
|---|---|---|
| S1 | Satoshi | Frozen adequacy design and tests; integrate the table above before training |
| S2 | Satoshi | Actual ridge, causal Conv1D and LSTM paths; measured support, reload and update tests |
| S3 | Satoshi | Governed cost pilot and bounded adequacy measurements under the existing CPU/memory ceiling; all outcomes retained |
| S4 | Satoshi | Recomputed predictions/losses, learning curves, coverage matrix, RL counterpart and weekly-business design |
| Independent review | Musashi | Reproduce scoped checks, challenge data/context/model/statistical assumptions, then decide the next bounded experiment |

Keep S1-S4 moving without renewed owner permission. Independent eligible jobs can use workers
with their own governed identities; do not duplicate runs to fill machines. No larger sweep,
financial/RL scoring, reserve or GPU opening is implied. If the budget does not fit, preserve
the frozen design and measured projection as already ordered; do not weaken acceptance silently.

Musashi must review ML adequacy alongside implementation correctness at every return. The
user is not responsible for discovering these omissions by inspecting every test. Carry this
checklist and current stage in the work-plan return so context compaction cannot erase them.
