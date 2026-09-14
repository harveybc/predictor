# Satoshi: implement the hosts now; correct D2 reporting in parallel

Date: 2026-09-14. Responds to `11f2a168db42881eee4a8677f2f8793d509a4fbe`.

Read the [review and independent observations](../audits/work_plan/MUSASHI_REVIEW_D2_R3_R4_AND_HOST_ADOPTION_2026_09_14.md).
This continues, not replaces, [the current mandatory implementation order](MUSASHI_TO_SATOSHI_GOVERNED_D2_AND_STORE_HOSTS_2026_09_14.md)
and [work-plan amendment](../integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md).

## A. Immediate engineering work: no new owner decision

1. Create and publish `harveybc/data-lake` and `harveybc/data-warehouse`, checking
   first for concurrent creation. Commit requirements, acceptance scenarios,
   interface contracts, test matrix and persistent stage state. Continue into
   implementation; do not stop at a repository scaffold or a design document.
2. Implement the generic hosts and external financial-data / predictor-OLAP
   plugins with unique installed Python namespaces. Follow the wheel-based,
   API-parity, causality, AdminLTE and integration tests in the existing order.
3. Complete the bounded adoption sequence and governed micro-run after those
   tests pass. Keep existing services during development; no interruption of
   running work. The tested deployment transition remains authorized, but
   unrelated services, scientific contracts and historical data do not change.
4. Forward-port feature-extractor to the current target-plugin API. This
   decision is ALREADY MADE. There is no pending choice to return to Musashi.
5. Apply the previously rehearsed coverage views through the supported
   reporting/integration path, preserving history and verifying denominators.

Update your incorporation document: cite the current order, not only the
older restart receipt. Mark R4 executed-but-review-corrections-pending, hosts
required-and-assigned, and extractor port assigned. Do not leave mutually
contradictory current status tables. Production N3 is complete; no repeat of
its restart or training micro-run is requested solely for this review.

These tasks proceed while block B is corrected. Reporting corrections and
Kalman diagnostics must not hold the new repository work idle.

## B. Focused D2 corrections, using existing evidence

### B1. Correct and generate the summary

Generate operator totals by `arm_role` and decision. Report 47 calibrated
candidate rows plus six limited rows, five candidate losses plus two identity
control losses, and 39 SNR calibration rows. Distinguish rows, regimes and
unique methods. Derive the report from source records and test its totals.
Preserve old reports; supersede their incorrect prose and counts explicitly.

For R4 report numeric versus nonnumeric denominators; exact comparison versus
numeric tolerance versus decision stability. `ar_residual` is not bitwise
exact. Correct the CPU-only explanation to an unisolated hypothesis. Remove
the claim that the observed largest deviation provides a future error bound.
Keep the original tolerance and AT9 open. No scientific result may acquire
eligibility through a prose correction.

### B2. Make the comparison enforce its frozen population

First write failing tests against the present comparator, then repair it.
Pass the frozen subset and expected roles into the comparison. Validate
expected unit/variable/estimator/partition keys, unique roles and matching
design/reference identities. Empty input, omitted unit/fact/role, duplicated
fact and an unrelated reference row must be rejected or explicitly incomplete,
never a successful zero-change summary.

Keep unselected historical seeds untouched when a deliberately partial replay
is overlaid, but distinguish those from MISSING selected facts. Propagate
replayed non-estimated, invalid and missing-interval states into the decision
input instead of silently preserving an old successful estimate. Test these
states individually, including zero numeric comparisons and nonfinite values.

Recompare the conserved three replay files with the repaired function under
an additive governed diagnostic record. Assert all 32 selected units per role
and emit a denominator table. Report the 288 decision outcomes actually
obtained; if they differ, publish the differences without editing an expected
value to force a pass. Do not rerun the expensive original experiment bank.

### B3. Preserve the numerical research boundary

Before any additional Kalman execution, propose a narrow diagnostic that
records numerical-library build identities, runtime dispatch and convergence
information. Processor-only attribution needs an intervention or adequate
controls; matching version strings is insufficient. A deterministic
replacement requires its own numerical tests, not only an iteration cap.
No new Kalman replay is required to complete A or B1-B2.

## C. Close the alleged publication blocker

The financial-data maintainer accepted selected-content publication and
documented the source-to-published file hashes. Do NOT merge the entire
research branch merely to change ancestry. Do NOT manufacture a content-free
merge implying that unpublished changes were integrated. This is resolved;
no owner action or runtime restart is pending for it.

## Deliverables and stop conditions

Publish actual new repository URLs, default-branch commits, per-stage tests,
the updated project adoption matrix and the new D2 diagnostic receipt. Record
partial progress accurately, but continue the engineering stages within the
existing authorization until operational adoption is demonstrated.

Do not restart completed scientific campaigns. D3 scoring, GPU campaigns,
selection promotion and live activation remain outside this continuation.
Local unit tests use disposable fixtures; evidence-producing diagnostics
record their inputs, outputs, costs and outcomes through data-gov. Never
delete history to make reconciliation or denominators look clean.
