# D2-R1 — support contract of the adjudicator (declared before the repair)

**Date:** 2026-09-14. **Order:** `MUSASHI_TO_SATOSHI_D2_SUPPORT_PORTABILITY_NEXT_ORDER_2026_09_13.md`, R1–R2.
**Frozen inputs:** `musashi_reproducer_frozen.out` (Musashi's reproducer on the published
`DECISIONS.jsonl`, sha256 `f4958c88f8caa6b78d67fb7ff00c2a6a697276aa9b19b6b73411b78d97622510`:
five favourable decisions with `n_seeds_valid=30` and 5–8 seeds without the primary
metrics; fixture outcomes `omitted_damage_metric = LAB_CALIBRATED`,
`inconclusive_* = LAB_CALIBRATED`).

## 1. What the five cases physically are

For every seed the reproducer lists as missing, the conserved rows exist with
`status = INCONCLUSIVE`, `reason = "undefined for this unit"` for
`snr_improvement_db`, `distortion_ratio`, `delay_samples`, `residual_signal_share`
and `extreme_retention_raw` (confirmation partition). `df_lab_evaluation.partition_metrics`
leaves them undefined when the clean signal has no variance on the window
(`var_c <= 1e-15`) or a power term is zero: the window carries no signal to compare
against. For the motif seeds this is the unit's own truth (no motif event inside the
confirmation window); for the step seeds the same geometry. So the category is
**absence of signal in the window → metric undefined by its estimator**, recorded by
the evaluator, and silently dropped by the adjudicator's `_seed_table` (only
`COMPLETED` rows entered `vars`), so `_agg` returned `None` and every check ignored the
seed while `n_seeds_valid` still counted it. Table per unit/variable/metric:
`SUPPORT_TABLE.jsonl` produced by `tools/df_d2_support.py` (see R2 evidence).

## 2. Support table (per arm × unit × variable × metric, confirmation partition)

| column | meaning |
|---|---|
| `applicable` | derived from the contract, never from a row's absence: improvement only in noisy regimes; distortion, delay, residual always; extremes when the raw counterpart could be measured (`extreme_retention_raw` not `INCONCLUSIVE`); an event metric when the unit's own event list has that kind inside the confirmation window (cross-checked against the evaluator's `<metric>__events` counts) |
| `status` | `OBSERVED` (COMPLETED row with a value) \| `INCONCLUSIVE` \| `UNAVAILABLE` (partition support) \| `REFUSED` (arm abstained by the missing-data rule) \| `FAILED` \| `MISSING_ROW` (never emitted) \| `NOT_APPLICABLE` |
| `reason` | the evaluator's reason (`undefined for this unit`, refusal reason) |
| `support` | `SUPPORTED` = applicable and observed; `UNSUPPORTED` = applicable and anything else; `NOT_APPLICABLE` |
| `component` | who is responsible: `df_lab_evaluation.partition_metrics` (undefined / support), `df_d2_unit_worker` missing-data rule (refusals), operator (failures), worker (missing rows) |

The synthetic truth (event list, clean signal) justifies applicability **in the
evaluator/adjudicator** only; it is never an input of any transformer.

## 3. Declared tests (`tests/test_df_d2_support.py`), PRE 10/10 failing, POST 10/10 passing

| # | rule | test |
|---|---|---|
| 1 | removing a metric that proves damage never turns rejection into a pass | `test_rule_1_…` |
| 2 | `INCONCLUSIVE` delay or residual does not satisfy its limit | `test_rule_2_…[delay_samples]`, `[residual_signal_share]` |
| 3 | an event that does not exist by contract is inapplicable, not measured as zero | `test_rule_3_…` |
| 4 | an event present with its metric missing does not become inapplicable | `test_rule_4_…` |
| 5 | a variable without support stays in the declared universe and denominator | `test_rule_5_…` |
| 6 | a seed without the primary contrast is not a complete seed | `test_rule_6_…` |
| 7 | fewer than two applicable non-inferiority pairs never give `passed=True` | `test_rule_7_…` |
| 8 | an SNR seed missing a required variable does not improve by dropping it | `test_rule_8_…` |
| control | the full fixture still calibrates | `test_control_full_fixture_still_calibrates` |

PRE/POST records: `PRE_declared_tests.out`, `POST_declared_tests.out` (adjudicator digests inside).

## 4. Repair (R2) — what changed and what did not

Changed (`tools/df_d2_adjudicate.py`): `_seed_table` keeps every row's status and the
evaluator's event counts; `_applicable()` and `_support()` derive applicability and
completeness; a valid seed is a **complete** seed; `n_seeds_valid` is the number of
complete seeds and the evidence publishes `seeds_planned / observed / complete /
abstained`, `unsupported_by_metric` (units) and `inapplicable_by_metric`; floors report
`applicable`, `applicable_seeds` and `unmeasured_seeds` (an unmeasured applicable floor
is `NOT_IDENTIFIABLE`, never a pass); non-inferiority with fewer than two complete pairs
of an applicable metric is `NOT_IDENTIFIABLE`, an inapplicable metric is skipped with
`passed = None`; `decide_snr` averages all required variables of a seed or declares the
seed incomplete.

Unchanged: margins, alpha and Bonferroni family, thresholds, seeds, partitions, metric
definitions, the sealed design and every published row. The semantics "a valid seed is
a complete seed" was not written in the sealed rules; it is published here as a
post-result clarification and resolved conservatively (the five cases lose their pass
and become `NOT_IDENTIFIABLE`; see the preview impact table). A different hypothesis
would need a future design, not this patch.

## 5. Governed re-adjudication (R3) — pending

The preview (`IMPACT_TABLE.json`, `DECISIONS_PREVIEW.jsonl`) is computed from the
conserved rows with the repaired adjudicator and is **not governing**: R3 registers it
as a Flow v3 review campaign that consumes the historical arrays/rows as evidence
resources with a current receipt, which needs the reconciled production micro-run
(GOV-N3, blocked on the operator).
