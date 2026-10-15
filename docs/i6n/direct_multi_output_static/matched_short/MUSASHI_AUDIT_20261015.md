# Musashi: matched short follow-up audit

Scope: launch `796ec8f4`, integrated source `62590fee`, and actual campaign
`i6dmo-6004b711281c`. The deployed runtime was not edited or redeployed.
No scientific fit was repeated, no TEST read, no DOIN or live trading change.

## Findings and disposition

1. **Prior runtime seal nonconformance, confirmed.** The CLOSED MG2/MG12
   recovery campaign's short design declares TF 2.18.0 / Keras 3.9.0; both
   retained trainer receipts record TensorFlow 2.21.0. These old receipts must
   retain their original bytes and actual observed versions. Their measured
   errors are not silently erased, but they cannot attest compliance with the
   declared runtime or serve as a runtime-matched donor/control. Add this
   qualification to future comparisons; no automatic rerun. The new matched
   designs declare TF 2.21.0 / Keras 3.13.2, and both current GPU receipts
   record TF 2.21.0. Resealing unused designs with a retained equality witness
   is distinct from retroactively resealing completed fits.
2. **R1 donor trust, confirmed and still open.** The independent retained
   witness adopts a hand-assembled donor with a fake scientific GPU receipt.
   `donor_fit_receipt_problem` checks labels but not enough origin/weight
   bindings. Current production has `prior_roots=[]`, so that witness does
   not demonstrate compromised current fits. Before R2/reuse, reject fake
   receipts and verify expected campaign/population/arm/weights/capacity plus
   trusted source-terminal provenance. Unkeyed self-consistent hashes alone
   do not authenticate a fully rewritten directory; state this limit.
3. **R2 pre-fit refusal starvation, reproduced and patched separately.** Two
   direct queue-selection regressions fail before the change. Matched dispatch
   now ranks candidates by persistent INFRA_PREFIT interruption counts, with
   stable plan order on ties. Other modes retain their original order and
   post-training failures are not reclassified as infrastructure refusals.
   Five focused checks pass. The additional governed 3-GiB integration request
   was refused before starting by SLICE_AGGREGATE_BUDGET; it was not rerun at
   a lower cap. After resources freed, the same 3-GiB admission and nine focused
   dispatch/worker checks passed in 4.09 s. Three older tests assumed repeated
   dispatch of the first arm; they now require the second arm to complete
   while the first remains in its legitimate wait. This patch is not deployed
   over the running identity.
4. **R3 scientific defaults, correction on follow-up branch.** Real production
   explicitly required scientific GPU receipts, but direct report/decision
   callers defaulted lenient. The follow-up requires strict behavior unless
   the caller supplies the literal boolean False for engineering. Record
   PRE: 51 failed / 18 passed (9.41 s). POST: 69 passed (4.07 s). The affected
   integration run passed 43 tests and failed the three obsolete fairness
   assertions subsequently corrected and checked above. This is focused
   verification, not a complete post-patch full-suite run. No old receipt is
   reclassified.
5. **Test scope must remain exact.** A previously retained full-suite file is
   from `4a77bea8` (506 passed), not the launch commit. Satoshi reports a later
   520/521 suite followed by a capacity fix and 85 matched checks. Do not
   combine those claims into a full-suite PASS at `796ec8f4`. A future release
   must report precisely which revision and tests it actually verified.
6. **Report admission was blocked by coordinator accounting.** The two fits
   completed, but real report launches were refused by the unchanged 14-GiB
   batch ceiling. Observed batch memory was approximately 10.5 GB, mostly file
   charges; no batch OOM or OOM kill was recorded. One delegated cgroup reclaim
   requested 3 GiB with swappiness=0; partial EAGAIN reclaimed 3,068,420,096 B,
   leaving limits unchanged and all processes/services untouched. Remaining
   tmpfs charges and active test reservations still count; no claim of unlimited
   headroom or a completed report follows from reclamation.
7. **New R4, report budget exhaustion after zero execution, confirmed.**
   `mg1.report` was marked ATTEMPTS_EXHAUSTED after three launcher admission
   exits 75; all stderr files explicitly say nothing started, stdout and out/
   are empty. Queue.claim excluded INFRA_PREFIT interruptions but not this
   literal launcher refusal. The future queue correction preserves the three
   rows/indices while not counting a proven admission refusal as execution.
   In the live queue only, an exclusive coordinator lock plus SQLite backup
   preceded bounded reclassification to the existing INFRA_PREFIT category
   and WAIT_RESOURCE. Original causes/hashes are retained in a new event and
   `gates/mg1.report.admission-recovery.json`. Code pin, model identities,
   memory caps and all stored terminals remained unchanged. No fit or failed
   scientific execution was retried; the original report had never run.
   Future queue regression PRE: 2 failed / 10 passed (0.15 s); POST: 12 passed
   (0.14 s), with TensorFlow absent. Actual crashes/timeouts still exhaust or
   terminate, and old attempt records/indices are preserved. This source
   correction is for the next release, not a changed live pin.
8. **CPU-only-synthetic wording is too broad.** The orchestration engineering
   mode is restricted to fixtures, but direct train_unit accepts device=cpu
   and marks that result non-scientific; it is not a blanket synthetic-data
   guard on every trainer call. No such real-data CPU fit was executed here.
   The scientific campaign's GPU rule and strict-default report/decision gate
   are the relevant protections. Do not describe them as a universal CPU ban.

## Current scientific evidence, not the stale launch snapshot

Both fresh GPU fits are COMPLETE, seed 6102026, static, no engineering override.
MG1: 38 epochs, best 28, 151.2136 s, cgroup peak 2791206912 B. MG2m: 12 epochs,
best 2, 54.5837 s, peak 2867683328 B. Early stopping is the declared FIT/INNER
mean, patience 10, minimum improvement 1e-8. Their eight trainer code-file
hashes each reproduce exactly from commit `796ec8f4`.

Both receipts bind common population
`ba01f87a14e1ceb39d1a5770d4d00f3803cd29748c64edf7254c1e6c57b249d1`.
Their scaler row digest, per-split masks/row digests and per-head counts agree.
MG1/MG2m measured parameter totals are 7186/4882; the 2304 difference is in
the heads, not branches/core. This remains a practical-recipe comparison,
not an isolated proof that more sampling information caused a change.

Independent saved-prediction replay verifies prediction hashes against trainer
receipts and recomputes the following in original-price units. Both official
reports subsequently completed with the same numbers. At 13:39 UTC, 110 reports
were delivered with readback, no mismatch or pending items; the comparison
unit was QUEUED. This establishes the two arm reports, not yet whole-campaign
publication/mirror closure. Equal-head relative scores: MG1 +0.04321014784883045,
MG2m -0.00007738037914937515; the frozen lower-score rule favors MG2m.
At 13:42 UTC all five units were COMPLETE; the real comparison executed in
5.29 s and returned MG2_SELECTED. Its COMPARISON.json digest is
`2ee85a125c3603e4069dbde42f1ccbf7ad32ac0b6c75ca798b7a8300cece13ee`.
The comparison's final publication/mirror/close still follows automatically.

Final focused source check after all three patches: 78 passed in 7.63 s
(strict defaults, dispatch and actual worker regressions). Separate pure queue
checks: 12 passed. This is not a full-suite certification of the next release.

| h | n | MG1 MAE | MG2m MAE | Naive for h | MG1 relative | MG2m relative |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 4724 | 0.000760919531 | 0.000746696886 | 0.000746663844 | +1.909251% | +0.004425% |
| 3 | 4671 | 0.000955645084 | 0.000928973457 | 0.000928891030 | +2.880215% | +0.008874% |
| 4 | 4618 | 0.001129517388 | 0.001079100759 | 0.001079278909 | +4.654819% | -0.016506% |
| 5 | 4565 | 0.001284581201 | 0.001225024135 | 0.001224979189 | +4.865553% | +0.003669% |
| 6 | 4512 | 0.001430557198 | 0.001361934024 | 0.001362063387 | +5.028680% | -0.009498% |
| 12 | 4196 | 0.002126244229 | 0.001994087262 | 0.001994833174 | +6.587571% | -0.037392% |

Relative = (MAE_h-naive_h)/naive_h. Same rows and naive per horizon in both
arms; never a naive shared across different horizons. These are DEVELOPMENT
2024 results; selected features used 2024. Neither independent holdout nor
statistical superiority nor full-pair trading eligibility is established.

## Resources and publication

Read-only host checks: both GPU workers respond, no restart indicated. At the
check preferred/secondary had about 9.7/21.6 GiB MemAvailable. After the fits
the training GPUs were idle while CPU reporting remained. Coordinator had
about 14.8 GiB MemAvailable, /home only 11.9 GiB free, /var/tmp 25.8 GiB,
and configured NTFS storage 710.5 GiB. Keep large artifacts off /home.

The incremental public diff was checked for private role aliases, home paths
and private addresses, with no matches. The historical producer_recovery.v1
object already exists in public base `0245c1f8` and is unchanged by these
commits; do not claim history was sanitized. A redacted public derivative can
coexist with the retained original hash, without falsifying old evidence.

Push of `satoshi/i6dmo-matched-short-20261015` at `62590fee` succeeded from
this session using its ordinary configured remote. No token was read/printed,
no permission configuration was changed and no owner action was required.
The denials reported by Satoshi's session did not recur here.
