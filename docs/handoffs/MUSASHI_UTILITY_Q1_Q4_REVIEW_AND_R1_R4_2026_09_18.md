# Utility Q1-Q4 review and R1-R4 execution order

Date: 2026-09-18. Scope: utility development, not the older consumer R1-R6 order.
Reviewed checkout: predictor `0352328d554e70dd7aaed301cc4595f9ff2a7e3f`.
Owner authorization already covers this bounded continuation. Execute all blocks without
asking for renewed permission; a scientific failure is a result, not permission to tune it away.

## Independent review

1. **High: computation identity omits operator implementation.**
   `tools/df_utility_harness.py:212` binds the harness file, operator declaration and parameters,
   not the implementation in `df_d3_operators.py` and its executing local dependencies.
   An isolated behavioral mutation of the real operator leaves the computation key unchanged
   and returns the old calibration bytes as a cache hit. The reproducer changes the method
   in memory; add a fresh-process, source-file-change regression as well. This demonstrates
   an equivalence gap, not that the historical campaign used changed code.
2. **Medium: an incomplete cache cannot recover through its normal miss path.**
   `tools/df_utility_calibration_cache.py:92`: remove `record.json` while retaining META.
   Lookup correctly returns absent; store then raises `OSError errno 39` when renaming onto
   the existing nonempty directory. The real child calls this same store after recomputing.
   Current tests stop at the miss and never exercise recovery. Also test missing/broken META
   and differing measured records; elapsed cost alone is not a scientific disagreement.
3. **Scientific limit, not a software defect:** Q3 does not meet its positive-control
   criterion. H_T has 4/6 advances; H_A is uncalibrated. Positive deltas do not establish
   sufficient detection performance, and 4/6 alone does not estimate power precisely.
   Nor do these results show that the underlying data or representations are useless.

Independent focal command (CPU):

```sh
python -m pytest -q tests/test_df_utility_dev_close.py \
  tests/test_df_utility_calibration_cache.py tests/test_df_utility_controls.py \
  tests/test_df_utility_reverify.py
```

Result: **22 passed in 33.99 s** under trading-stack. This is not a rerun of the reported
1259-test suite. Independent read-only reclosure of real utildev-v1 returned TOTAL,
all four checks true, 18 paired rows and zero proposals. This re-reads scientific files
but consumes retained reconciliation/content receipts; no new live-cube query was performed.
Reproducer: `docs/handoffs/reproduce_utility_q_review_2026_09_18.py`, invoked
with `--repo <reviewed-checkout>`; writes temporary fixtures only. Observed:

```text
operator_behavior_changed=true, same_key=true, stale_cache_hit=true
MISSING_RECORD_LOOKUP (None, 'absent')
RECOVERY_EXCEPTION OSError errno 39
```

The existing partial closure policy is acceptable when every proposed pair's complete scope
passes and the artifact explicitly reports PARTIAL globally. Never present it as total closure.
Historic decisions and files remain preserved. Stored warehouse receipts are not a fresh live
query; label that distinction in every successor report.

## R1 - Bind reusable computations to executing code

Declare regression expectations before implementation. Build a finite, explicit identity of
the local scientific code actually used: operator implementation, inherited helpers, contract
and preprocessing/math helpers as applicable, plus harness and numeric environment. Keep
administrative labels outside equivalence; do not replace missing scientific identity with
an arbitrary whole-repository label. Record the numeric portability scope honestly; version
strings alone do not demonstrate cross-CPU bit identity.

Tests must cover changing transform/fit behavior without changing describe/params, a helper
change, unchanged code under different campaign labels, and fresh-process miss then hit.
Changing executing scientific code must prevent reuse of an old entry. Existing records retain
their historical status; legacy keys cannot acquire stronger reuse guarantees by relabeling.
Do not regenerate the 36 historical calibrations. Publish schema transition and explicit scope.

## R2 - Recover a cache miss end to end

Reproduce the incomplete-directory failure before editing. Implement a bounded, typed recovery
that preserves invalid/incomplete entries separately, validates before serving, and lets a valid
replacement become usable. Exercise the real isolated child, not lookup alone: absent record,
absent META, malformed record, interrupted publication, and concurrent independent producers.
The test must reach completed output and a subsequent verified hit, or a deliberately typed
non-success with preserved reason, never an unhandled exception or perpetual recomputation.

Separate scientific result equality from cost/provenance differences across producers. Conflicting
scientific results for one key require a recorded disposition, not silently treating both as
equivalent. Consumer accounting must be retry-idempotent by attempt identity; repeated reporting
of one hit must not multiply saved-work claims. Keep measured verification/lookup cost apart
from projected avoided compute. Carry one bounded recovery demonstration through governed
terminal -> accounting -> warehouse, including its failed attempt if applicable.

## R3 - Validate the instrument before extending selection

Implement the proposed instrument successor only after R1/R2 pass. Before new outcomes, freeze:
generators and effects, fresh seeds disjoint from Q3, n=2048, 12 replicates, four blocks, both
branch pairs, losses, margins, exact multiplicity and calibration plan, criteria and resources.
Use the campaign's actual confidence and effective alpha, not the 0.5-confidence fixture.
Derive required simulations from those values; 358 is not universal. Reuse only equivalent
verified calibrations. A failed calibration is INCONCLUSIVE, not negative utility.

Retain positive H_T/H_A, contrast-null, information-loss and future-leak controls. Fix every
criterion before running (positive criterion >=10/12; explicitly specify the others and justify
their finite-sample interpretation). Report detection fractions with uncertainty, complete
denominators, calibration support separately, and all incomplete runs. Twelve replicates do
not prove a universal power guarantee. The positive generator is deliberately aligned to this
operator and does not validate all six other representations.

Run once as governed DEVELOPMENT/instrument evidence, including full calibration costs. Check
budget before calibration as well as during controls, and enforce it through the isolated runner
(a check between replicates alone cannot bound a long calibration). Pilot cost first; maximum
**7200 aggregate CPU seconds for R1-R3 scientific rehearsals and instrument work**, counting
failed attempts and all machines. Existing memory/wall guards stay in force. If the projection
does not fit, deliver the sealed design and cost result, not a weakened criterion. Test suites
are reported separately. Independent bounded jobs may use the workers under their own governed
identities when scheduling saves time; serial dependency work need not occupy all machines.

Do not launch the six remaining operators in this order, even if the controls pass. Deliver
their executable, bounded next-stage design with operator-specific eligibility, resource and
sensitivity limits. If controls fail, diagnose using completed evidence and stop scientific
progression without retuning, reusing the same seeds as fresh, or opening a reserve.

## R4 - Close all evidence and update the plan

Reclose utildev-v1 from conserved bytes only; report any compatibility boundary rather than
retroactively requiring a new cache schema of old scientific results. No new historical scores.
Verify new results from files through parent, accounting and warehouse by content. Preserve
all attempts, negative and inconclusive results; distinguish live checks from retained receipts.
Update 09_ADOPCION and 12C with instrument status and what remains untested, including
feature-engineering, variable-specific denoising, learned representations and lake-wide selection.

Return exact commits, PRE/POST for both findings, full targeted suites with skips/environments,
cache recovery and accounting evidence, governed run identities, costs and instrument tables.
No GPU, financial data, live trading, reserve, broad sweep or service restart for housekeeping.
No owner action is required. The pending empty-envelope disposition waits for an already-needed
maintenance window. Keep unrelated index/Metabase/terms investigations separate.

Completion: `UTILITY_CODE_BOUND_REUSE_AND_INSTRUMENT_SUCCESSOR_REVIEW`, with instrument outcome
explicitly PASS / DOES_NOT_SEPARATE / INCONCLUSIVE / BUDGET_LIMITED. Completing the order does
not require a positive scientific result.
