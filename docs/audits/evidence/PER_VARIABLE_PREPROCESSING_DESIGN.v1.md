# C52: per-variable preprocessing successor — DESIGN ONLY

**Date:** 2026-09-12
**Status:** `SEALED_DESIGN_NO_SCORES_COMPUTED`
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C45_C52_AND_T2_R7_R10_ORDER_2026_09_12.md` §C52
**Authorizes:** nothing. No arm is run, no variable is selected, no
confirmation is touched, no GPU is opened.

---

## 1. What this is not

T2 screened ONE global transformation — EWMA with α=0.3 applied to
every series — and the answer was `DOES_NOT_ADVANCE`:
`electricity_weekly` at −0.0365 fell beyond the non-inferiority margin
and the primary estimand was −0.001048. **That result stands and this
design does not appeal it.**

A per-variable rule is not a repaired version of the global rule. It is
a **different hypothesis**: that the right transformation depends on the
variable, and that choosing per variable buys more than it costs in
selection error. A new hypothesis earns its own licence in development
before anything confirmatory is discussed. Nothing here inherits T2's
budget, its panels or its permission.

## 2. The unit of analysis

The statistical unit is the **conceptual variable within a panel**, and
panels are the outer cluster — the same hierarchy the T2 decision rule
fixed, for the same reason: rolling origins and seeds are nested
repeated measurements and never inflate the independent sample count.

A design that treated 1,965 variables as 1,965 independent tests would
manufacture significance out of arithmetic. The declared unit is the
panel effect; the per-variable choice is a *treatment*, not a sample.

## 3. Arms, with paired budget

For each panel, every arm sees the same data, the same origins, the
same seeds and the same wall-clock budget. Unequal budget is the
cheapest way to fake an improvement, so it is fixed before anything
runs.

| Arm | Content | Purpose |
|---|---|---|
| `A0_BASELINE` | `X`, no new transformation | what must be beaten |
| `A1_PER_VARIABLE` | `D_j(X_j)` — one candidate operator per variable | the hypothesis |
| `A2_AUGMENTED` | `[X_j, D_j(X_j), X_j − D_j(X_j)]` where the decomposition is defined | keeps the residual rather than discarding it |
| `A3_CAPACITY_CONTROL` | `X` widened to `A2`'s dimension with variables that carry no new information | separates "the transformation helped" from "more columns helped" |

`A2` is declared only where `X_j − D_j(X_j)` is meaningful — a smoother
has a residual, a rank transform does not. Where it is not defined, the
arm is `NOT_EVALUABLE` for that variable and is reported as such, never
silently dropped.

`A3` exists because `A2` triples the width. Without an equal-dimension
control, any gain is confounded with capacity, and the honest reading
of a positive result would be unavailable.

## 4. Causality

Every candidate operator must be **causal at the bar it is stamped
with**, verified against the C47 DAG and not against its name:

* the operator's window looks back only — no centring, no negative
  shift;
* `earliest_available_time = max(event_time of the inputs over the
  window) + the declared causal latency`;
* a variable whose producer is `UNRESOLVED_PRODUCER` in the C47 DAG is
  **ineligible** for this design. Its availability is unknown, so a
  causal claim about a transformation of it would be unfounded;
* a variable classed `NON_CAUSAL` is ineligible and is reported as a
  leakage finding, not as a candidate.

Eligibility is therefore derived, not declared: today that is the 52
columns the DAG resolves as `CAUSAL`, minus the 5 whose only producer
lives under a retired path, which are `REVIEW_REQUIRED` rather than
eligible.

## 5. Costs

Cost is part of the result, not an afterthought:

* wall seconds per arm per panel, measured, on CPU;
* the transformation's own cost separated from the fit's;
* a transformation that wins by a margin smaller than its cost premium
  is reported as **not worth it**, with both numbers shown.

The budget ceiling is fixed before execution and a run that exhausts it
ends `INCONCLUSIVE` — never "as far as it got".

## 6. Abstention

An arm may abstain per variable, and abstention is a declared outcome:

* `ABSTAIN_NO_CAUSAL_OPERATOR` — no candidate operator is causal for
  this variable;
* `ABSTAIN_INSUFFICIENT_SUPPORT` — fewer evaluable series than the
  per-panel minimum;
* `ABSTAIN_NOT_IDENTIFIABLE` — the C49 disposition for this variable is
  `NOT_IDENTIFIABLE` or `UNAVAILABLE`.

An abstention is never counted as a tie and never as a win. A panel
whose abstentions exceed a pre-fixed fraction is `NOT_EVALUABLE`.

## 7. The decision rule, fixed before any score

Advancement out of development requires **all** of:

1. broad-family consistency — a favourable grand average with
   concentrated harm in one family does not pass, exactly as in T2;
2. a confidence bound beyond a frozen practical margin, declared here
   and not adjusted afterwards;
3. no material harm on any panel beyond the non-inferiority margin;
4. complete cost accounting, including the transformation's own cost;
5. the capacity control `A3` NOT explaining the gain.

Multiplicity is controlled across all panels and all candidate
operators jointly under Holm, including the arms that abstain — an
abstention consumes its place in the family.

## 8. Withdrawal rule

The design is withdrawn, without a score being read, if any of these
becomes true before execution:

* the C47 DAG's `CAUSAL` set changes such that fewer than a pre-fixed
  minimum of variables remain eligible;
* the C49 coverage stops being complete for the panels in scope;
* the budget ceiling cannot be met with paired arms;
* a candidate operator is found to be non-causal after the population
  is fixed.

Withdrawal is a legitimate outcome and is recorded as one.

## 9. What must happen before this runs

1. external review of this design, sealed before any score;
2. external review of the C47 DAG, since eligibility depends on it;
3. an explicit development licence. This design does not carry one and
   does not ask the owner for one — Musashi opens the screen.

**No arm was executed. No variable was selected. No score exists.**
