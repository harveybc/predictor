# C64–C65: per-variable preprocessing successor — DRAFT CANDIDATE

**Date:** 2026-09-12
**Status:** `DRAFT_CANDIDATE_NO_SCORES_COMPUTED`
**Supersedes:** `PER_VARIABLE_PREPROCESSING_DESIGN.v1.md`, whose status
read `SEALED_DESIGN_NO_SCORES_COMPUTED`
**Machine-readable companion:** `PER_VARIABLE_PREPROCESSING_DESIGN.v2.json`
**Authorizes:** nothing. No arm is run, no variable is selected, no
confirmation is touched, no GPU is opened.

---

## 0. Why the status changed (C64)

v1 called itself **SEALED**. It was not sealed and could not have been.

A seal is an assertion that an external party fixed this document
before any score could influence it. No external party had seen it.
What actually happened is that I wrote a design and hashed it — which
establishes that *I* have not changed it since, and nothing more. An
author cannot seal his own design any more than he can review it: the
word imported an authority that did not exist, and it imported it into
the one artifact whose entire value is that nobody could have tuned it
after seeing a number.

`DRAFT_CANDIDATE_NO_SCORES_COMPUTED` claims exactly what is true: a
draft, offered as a candidate, with no scores computed. The second half
of the old label was accurate and is kept verbatim. v1 is **not
rewritten** — it stands as history with a supersession record beside
it, because a label that overclaimed is itself evidence.

This is the same defect class as the 137 "executable configs" and the
52 "CAUSAL" columns: a name asserting more than the measurement behind
it supports.

## 1. What this is not

T2 screened ONE global transformation — EWMA with α=0.3 applied to
every series — and the answer was `DOES_NOT_ADVANCE`:
`electricity_weekly` at −0.0365 fell beyond the non-inferiority margin
and the primary estimand was −0.001048443391358884, reproduced from the
single audited snapshot on 2026-09-12. **That result stands and this
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

## 4. Eligibility (C65 — bound to the v2 lineage)

A variable is eligible only if **all** hold, each checked against a
named artifact and digest rather than against a belief:

1. its C62 batch membership row exists and its batch is `EXACT`;
2. its v2 terminal's `outcome` is `MEASURED`;
3. its v2 terminal's `source.sha256` is present and matches the file
   the run would read;
4. its lineage class in `FEATURE_DAG.v2` is `CAUSAL_ACTIVE` — **not**
   `UNRESOLVED`, **not** `HISTORICAL_OR_RETIRED_PRODUCER`, **not**
   `NON_CAUSAL`;
5. its availability is not `UNAVAILABLE`.

Condition 5 is, on today's evidence, **unsatisfiable for every
variable**: no dataset declares what its timestamp means, so
`FEATURE_DAG.v2` gives every column `earliest_available_time:
UNAVAILABLE`. The eligible set is therefore **empty**, and this design
states that plainly rather than quietly dropping the condition. An
empty eligible set is why this is a draft candidate and not a runnable
screen.

## 5. Abstention states

* `ABSTAIN_NO_CAUSAL_OPERATOR` — no candidate operator is causal for
  this variable;
* `ABSTAIN_INSUFFICIENT_SUPPORT` — fewer evaluable series than the
  per-panel minimum;
* `ABSTAIN_NOT_IDENTIFIABLE` — the C49 disposition for this variable is
  `NOT_IDENTIFIABLE` or `UNAVAILABLE`;
* `ABSTAIN_LINEAGE_UNRESOLVED` — **new in v2**: the variable's
  transitive lineage does not resolve to declared leaves, so no claim
  about leakage can be made either way.

An abstention is never counted as a tie and never as a win. A panel
whose abstentions exceed a pre-fixed fraction is `NOT_EVALUABLE`.

## 6. The decision rule, fixed before any score

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

Sign tests, where used, take the **exact two-sided** form
`min(1, 2·min(P(X≤k), P(X≥k)))`. The T2 packet published 1.3125 for
3/6, which is not a probability; the corrected value is 1.0 and the
table for 0..6 successes is
`0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125`.

## 7. Withdrawal rule

The design is withdrawn, without a score being read, if any of these
becomes true before execution:

* the `FEATURE_DAG.v2` `CAUSAL_ACTIVE` set changes such that fewer than
  a pre-fixed minimum of variables remain eligible;
* the C49 coverage stops being complete for the panels in scope;
* the budget ceiling cannot be met with paired arms;
* a candidate operator is found to be non-causal after the population
  is fixed;
* any batch in scope stops being `EXACT` in
  `v_batch_membership_integrity`.

Withdrawal is a legitimate outcome and is recorded as one.

## 8. What must happen before this runs

1. **external review** of this design — the seal v1 claimed for itself
   is precisely this step, and it has not happened;
2. external review of `FEATURE_DAG.v2`, since eligibility depends on it;
3. dataset timestamp semantics and provider latency declared, without
   which the eligible set is empty by rule 5 above;
4. an explicit development licence. This design does not carry one and
   does not ask the owner for one — Musashi opens the screen.

**No arm was executed. No variable was selected. No score exists. The
eligible set is empty and is reported as empty.**
