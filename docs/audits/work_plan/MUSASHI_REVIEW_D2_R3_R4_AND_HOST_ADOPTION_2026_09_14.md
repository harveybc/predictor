# Review: governed D2 closure, numerical portability and host adoption

Reviewed return: predictor `11f2a168db42881eee4a8677f2f8793d509a4fbe`.
Date: 2026-09-14. Reviewer: Musashi.

Disposition: **R3 accounting and recorded successor accepted within scope;
R4 reporting/comparator corrections required; host implementation proceeds now.**
No scientific promotion, operator-wide acceptance or live activation follows.

## Findings

### 1. The newest execution order was not incorporated

Satoshi's incorporation document cites the older restart receipt and treats
the new hosts as a proposed next architecture. It still asks Musashi to choose
pin versus port for feature-extractor. Both decisions were already settled:
[the order at 2f52cd2](https://github.com/harveybc/predictor/blob/2f52cd2da4f514c73194d627540ce7879977188f/docs/handoffs/MUSASHI_TO_SATOSHI_GOVERNED_D2_AND_STORE_HOSTS_2026_09_14.md)
requires creation AND adoption of the hosts and a forward port of the extractor.
Repository lookup during this review did not resolve either new GitHub repo.
The previous D2 close-out is complete; the overall engineering order is not.

### 2. Candidate counts mix roles and contain an incorrect subtotal

I rehashed and recounted the actual 3,591 successor rows. The transition totals
reproduce: 138 changes, no new pass, seven lost passes across candidates AND
controls. There are five losses among candidate operators and two among
identity controls. The surviving operator rows are:

| Role | LAB_CALIBRATED | REGIME_LIMITED |
|---|---:|---:|
| CANDIDATE | **47** | 6 |
| IDENTITY_RAW_CONTROL | 26 | 0 |
| PREVIOUSLY_REJECTED_CONTROL | 2 | 0 |

Thus the candidate count is 58 -> 53, not 58 -> 54. All-role calibrated rows
total 75; that does not mean 75 candidate operators. Counts describe
operator/regime decision rows, not distinct methods or deployed approvals.
The 39 SNR calibration decisions are unchanged, with their existing scope.
Correct the summary from rows without rewriting the conserved evidence.

### 3. Four estimators are exactly equal on numeric cells; the fifth is close

The three committed replay files match their published digests. Recounting
the 1,716 comparison records gives 1,389 numeric comparisons and 327 without
a numeric difference. There are 1,223 exact numeric comparisons, 1,306 within
1e-9 dB (including exact ones), and 83 outside it.

`ar_residual` has **51 nonzero differences**, maximum
`1.9184653865522705e-13` dB. It satisfies the reported numeric tolerance on
this subset; it is not bitwise exact. The other four non-Kalman estimators
match exactly on their numeric comparisons. Undefined cases must remain
separate, especially wavelet_mad: 132 numeric and 264 nonnumeric comparisons.
Nothing establishes exactness on all tasks or all machines.

### 4. The comparator can turn missing evidence into a zero-change report

The actual `compare()` API accepts an empty replay and empty reference files,
returning units=0, compared=0, changed=0 instead of an incomplete-input outcome.
This was reproduced in a disposable fixture without running an estimator.
The expected frozen subset is not an argument of `compare()`; the CLI reads
the subset but does not pass it into this comparison.

Source inspection adds two risks requiring focused tests: missing replay facts
are silently replaced with historical rows, and an unestimated replacement
does not update the historical row's estimated state before re-adjudication.
Consequently, the reported 0/288 is a scoped observation of the present inputs,
not a validated guarantee that missing/failed replays cannot look stable.
Repair the comparator and re-evaluate existing replay bytes; do not regenerate
the 3,972 experimental units to fix this.

### 5. Processor causation and the prospective policy overreach the evidence

Package version labels are not proof of identical numerical binaries or
instruction dispatch. The report itself records different kernel releases.
The evidence establishes host-associated numerical differences; it does not
isolate the CPU as their sole cause or measure likelihood flatness.
Keep those explanations as hypotheses pending a controlled diagnostic.

An observed maximum deviation is not a proven future bound. Do not turn the
largest observed calibration/confirmation difference into a new permission
rule for subsequent decisions. Fixed iterations alone do not guarantee
cross-platform determinism either. Preserve AT9's original criterion and
scope Kalman as exploratory until independently validated. The other
estimators can proceed through their existing per-regime review separately.

### 6. No whole financial-data merge is needed

The three selected adapter files are byte-identical between published
`cc0f15e6e` and source `f00bc6c15`. The full trees differ in 80 paths. Selected
publication with explicit provenance is valid; it is not an unresolved owner
decision. See the [maintainer's file-level provenance disposition](https://github.com/harveybc/financial-data/blob/master/docs/ADAPTER_PUBLICATION_PROVENANCE_2026_09_14.md).
No fabricated ancestry or broad research merge is requested.

## What I independently verified

- Rehashed the successor file against the recorded digest and recounted rows.
- Recomputed 3,591 transition rows, 138 changes, seven losses and zero gains.
- Verified all three replay-file hashes and their 32-unit counts.
- Recounted exact, close, different and undefined comparison cells by estimator.
- Queried the production warehouse read-only: one COMPLETED review terminal,
  25 metrics, seven artifact rows; artifact hashes and sizes match the receipt.
- Observed the OLAP loader active with zero restarts.
- Exercised the empty-comparison case against the actual comparator function.
- Compared financial adapter files directly from the named Git revisions.

Evidence: [independent counts and fixture output](MUSASHI_D2_R3_R4_COUNTS_2026_09_14.json).
No service restart, scientific replay, training, GPU use or database mutation
was performed for this review. The original re-adjudication algorithm and
the full 288 decision recalculations were not independently rerun here.

One chronology correction is documentary: `evidence_spec()` hashes conserved
files before campaign registration, while the re-adjudication runs afterward.
Say that, rather than claiming no evidence bytes were read before registration.
The receipt correctly labels historical synthetic evidence and does not claim
a governed download at its original creation time.

Next: [execution continuation](../../handoffs/MUSASHI_TO_SATOSHI_HOSTS_NOW_AND_D2_REPORT_CORRECTIONS_2026_09_14.md).
