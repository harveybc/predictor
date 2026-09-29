# Closure table — q2_household_governed_20260929

Generated from artifacts at 2026-09-29T21:09:27Z.  **NO_NEW_MEASUREMENT**: 0 scored rows.

| unit | estimand | model error (metric, scale) | paired naive, same rows | skill | reference (source) | comparability | scope peak | custody |
|---|---|---|---|---|---|---|---|---|
| pilot | MATCHED_BUDGET_DIFFERENCE | `null` | `null` | `null` | NOT_CARRIED (NOT_CARRIED) | NOT_COMPARABLE | 1463877632 B | ACCEPTED_TERMINAL_AND_WAREHOUSE_ROW |
| probe | MATCHED_BUDGET_DIFFERENCE | `null` | `null` | `null` | NOT_CARRIED (NOT_CARRIED) | NOT_COMPARABLE | 384962560 B | ACCEPTED_TERMINAL_AND_WAREHOUSE_ROW |

## Why each null is there

* **pilot** — model error: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: this unit fits no model, holds no checkpoint and scores no evaluation population  naive: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: there are no scored rows for a naive to be paired on  skill: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: skill is 1 - error_model/error_naive and both are absent  comparability: this unit reports a chain and a memory footprint, not an error; a published accuracy would be compared against nothing here, and the block's own estimand is a matched-budget difference which no converged-training publication shares
* **probe** — model error: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: this unit fits no model, holds no checkpoint and scores no evaluation population  naive: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: there are no scored rows for a naive to be paired on  skill: ABSENT_MEASUREMENT_NULL_IS_NEVER_A_NUMERIC_ZERO: skill is 1 - error_model/error_naive and both are absent  comparability: this unit reports a chain and a memory footprint, not an error; a published accuracy would be compared against nothing here, and the block's own estimand is a matched-budget difference which no converged-training publication shares

## What this round does not establish

* no successor cap for any W1440 cell: this round built no model, no gradients, no optimizer slots and no Keras graph, so its peak is a data-stage floor
* no custody for any of the twelve historical Q2 fits: they stay NOT_BOUND_TO_A_SEAL
* no promotion, selection or ranking of anything
