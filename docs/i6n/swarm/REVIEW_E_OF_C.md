# Lane E review of lane C (weekly units, 2023 folds, support, roll-up)

Reviewed, read-only (lane E wrote none of this code): predictor
`satoshi/i6n-swarm-C-weekly-20261011@9f055497`. Files:
- `tools/i6n_weekly_campaign.py`, `tools/i6n_weekly_rollup.py` and
  `tools/i6n_weekly_history_extension.py`;
- the HTTP 429 rule in `tools/i6n_external_clock_audit_v2_resume.py`;
- `C_RETURN.md`, `INTERFACE_C.json` (C.2) and
  `evidence_C/DECLARED_CLOCK_ASSUMPTION_2019.json`.

The review checked C against orders 2.C and plan sections 2 and 4.3-4.5.

## Method

- **C's suites**, run once in a scratch worktree (CPU,
  `crispdm-run -m 2G -t 120s`): `test_i6n_weekly_campaign`,
  `test_i6n_weekly_rollup`, `test_i6n_weekly_history_extension` and
  `test_i6n_external_clock_audit_v2_resume`. Result: 56 passed.
- **Reproducers** on C's tree, with invented rows only. No fit, no data, no
  network and no TEST rows.
  - Script: `evidence_E/review/review_e_of_c.py`.
  - Output: `evidence_E/review/review_e_of_c_result.json`.

## Findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| E-C1 | **FIX_REQUIRED** (integration) | `AnnualAccumulator` binds every weekly row to ONE `candidate_identity_sha256`. Under INTERFACE_A A.2, that field is the full evaluator identity: the cell config including `evaluation_week_id`, plus the context with the fold and the support. A finalist refit weekly therefore has a **different** `candidate_identity_sha256` every week. With real A receipts the accumulator accepts week one, refuses every other week with `IDENTITY_MISMATCH`, and the annual result stays INCOMPLETE forever. C's tests pass only because they reuse one identity for all weeks. Fix: key the accumulator on the arm, i.e. `genotype_canonical_sha256` or `effective_config_sha256` from the receipt header plus the finalist role. Record each week's `candidate_identity_sha256` per row, and keep the per-week `n` and `origins_sha256` binding. | C1: `W2024-01-07 ADDED`, `W2024-01-14 REFUSED IDENTITY_MISMATCH`, annual `INCOMPLETE` |
| E-C2 | low | The 429 rule is bounded per run (exactly one pending case, named by `--only-date`), but **not to one named date and not by an attempt count**. `transport_failure` treats a 429 on any holdout date as retryable, and the same single date can be re-authorized indefinitely. The DCA record states "no further retry", but the code does not enforce it. Fix: bind the rule to the date and chain of the declared assumption, or refuse once the DCA covering that date has been declared. | C2: 429 counts as a transport failure on 2019-10-28 and 2021-02-01; no attempt cap in the code |
| E-C3 | low | For DCA folds the cell config carries the extended input and source-clock pins, but not `assumption_id` / `assumption_sha256`. The assumption reaches A's identity only indirectly, through the clock file. The `_DCA` fold id is in the fold key, and therefore in B's `fit_id` and in every receipt row. Advisory: add the assumption pin to the cell so A's receipt header names it explicitly. | code read (`run_plan`, `cell_overlay`) |

## Checks that pass

- **Four-year TRAIN per cutoff:** `_years_before(cutoff, 4)`, with Feb 29
  mapped to Feb 28.
- **Purged 28-day inner split:** TRAIN covers [start, inner) with labels known
  by the inner start; INNER covers [inner, cutoff) with labels known by the
  cutoff. Membership requires both the origin and the target interval.
- **Refit timing:** each unit is refit at the previous Friday close and
  released at the next open.
- **53 weekly units** (W2023-12-31..W2024-12-29), none skipped or merged. 633
  (week, h) dispositions per arm. A missing span is typed WAIT_DEPENDENCY;
  history is never shortened.
- **Fold cutoffs:** the close of the last calendar week entirely inside each
  quarter, with UTC and calendar-zone agreement required.
- **Fold ids:** `2023Q1_DCA..2023Q3_DCA` and `2023Q4` carry the declared
  assumption in the fold id and in `unit_sha256`. This matches E's plan v1.2.0
  quarter-bound fold rule.
- **Support before any fit** comes from C1 `operational_eligibility` and equals
  `build_weekly_partitions` (tests). Each count satisfies
  `origins_enumerated == n + sum(exclusions)`.
- **No 2025 label opened:** targets from 2025-01-01 are typed
  `TARGET_IN_TEST_CLOSED` and never looked up. Splits that include TEST are
  refused. The extended source refuses TEST timestamps.
- **Annual roll-up:**
  - MAE is sum(error) / sum(n), with the naive pooled over identical rows.
  - `compare_annual` refuses a naive mismatch.
  - Inner-fold versus annual comparisons are refused.
  - A missing week means INCOMPLETE and no score.
  - Zero-support weeks are NO_SPLIT_SUPPORT.
  - These rules are correct once the identity key of E-C1 is fixed.
- **Cell configs** carry the plan's stopping rule, seed 6102026 and the
  endpoint base model, and pass `authenticated` (C's tests).
- **History extension tool:** it never downloads and never issues a
  certificate.

## Verdict

**FIX_REQUIRED** for E-C1 only, because it blocks the annual weekly stage with
real A receipts.

E-C2 and E-C3 are advisories. Everything else in scope is **ACCEPTED**: units,
cutoffs, TRAIN and inner rules, support, TEST closure and the DCA fold
identity. No lane C file was edited.
