# Dated successor to 13E on the pre-training's internal validation

Status: successor, 2026-09-23 (RP142). It supersedes one paragraph of
[13E v2](13E_v2_E1_TASK_SHEET_AND_REGIMES_2026_09_19.md) and erases nothing. Everything else in 13E v2 stands.

## What is superseded

13E v2's pre-training paragraph describes the auto-encoder's internal validation as the inputs of the DEV validation windows,
with no label. The [next-intervention preparation](NEXT_INTERVENTION_PREPARATION_2026_09_23.md) repeated it. Both are wrong
about the rule and both are corrected here.

## The rule, as the code implements it

The auto-encoder's internal validation is a **chronological tail of the outer TRAIN origins**, purged from the AE's own
training origins by at least `window + horizon`. The outer validation split is never read during pre-training. This is what
`tools/df_e1_pilot.py` does when it builds `ae_tr` and `ae_va` from `train_origins` with `purge = W + h`, and what
`tests/test_df_e1_pretraining.py` asserts: the AE validation origins are inside TRAIN, disjoint from the evaluation origins,
disjoint from the AE training origins, and separated from them by the declared purge.

This is not a claim that any executed run leaked. The runner tested under RP36 already implements the purged inner tail; the
defect was in the prose that described it, which a successor design could have copied.

## What a successor design must prove before fitting

1. `ae_validation_origins` is a subset of the outer TRAIN origins.
2. It is disjoint from the outer validation origins and from the outer test origins.
3. It is disjoint from `ae_train_origins`, and `min(ae_validation) - max(ae_train) >= window + horizon`.
4. A perturbation of rows beyond a window's support leaves that window's inputs and targets unchanged.
5. The outer validation may select the downstream forecasting checkpoint, and nothing else; the outer test selects nothing.

Absence of a label is never accepted as evidence of TRAIN membership. Each of the five is a test over row identities, not a
statement in a document.
