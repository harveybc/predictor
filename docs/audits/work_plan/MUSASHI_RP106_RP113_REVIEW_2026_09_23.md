# RP106-RP113 independent review

Reviewed `e145303cbc5776ee3907d9997a3622e1eab272b8`.
Disposition: **CHANGES_REQUIRED; KEEP TRAINING AND REPORTED MEASUREMENTS;
EXACT-SCORER AND DELETION-ACCEPTANCE CLAIMS NOT CLOSED.**

## Executed findings

1. **High: resolving a report digest does not establish its acceptance.**
   `tools/df_sota_repro.py:2523` resolves the report named by the deletion marker,
   but both report and marker remain locally replaceable. Starting from a valid
   closure and deletion, I changed the report's MAE/MSE to zero, recomputed its
   digest and updated the marker. The retained record/checkpoint, vault and stub
   warehouse accepted artifacts were unchanged. The real verifier still returns
   historically verified with score zero and no problems. Resolution of bytes is
   now checked, but an independently accepted closure identity is still absent.

2. **High: regeneration has an independent false-success path.**
   In `tools/df_sota_repro.py:2634`, a local REGENERATION.json with matching
   copied prediction/target hashes and `author_metric={mae:0,mse:0}`, plus an
   ACCEPTANCE.json containing only `pass:true`, overrides the valid historical
   score. No regenerated arrays or actual inference were needed in the probe.
   The verifier returns historical verification, zero score, and no problems.
   These new evidence objects need their own accepted identity and content checks;
   copying the old inputs' digests cannot authenticate a new computation.

3. **High: deletion approval and backup are optional in the real API and CLI.**
   `deletion_preflight` (line 1698) enforces approval/backup only if arguments
   were supplied. `delete_predictions` (line 1771) and CLI flags (lines 3158-3159)
   default them to None. On a valid closed fixture, calling the deletion function
   without either deletes the only array and reports COMPLETE. The deletion gate
   also does not require CATALOG_ACCEPTANCE.json. A caller following the documented
   optional interface can therefore bypass the retention prerequisites. The
   separately supplied manifest checks local originals, not the continued existence
   and content of the backup destination; root flock alone does not establish
   exclusion of unrelated writers. The executed counterexample concerns missing
   approval/backup, not a claimed production race.

4. **High for exact reproduction: float32 mean denominator differs from NumPy.**
   `author_metric_exact` (line 351; final division at line 383) converts N to
   float32 before division. In the pinned worker environment, with **16,777,217
   float32 errors equal to one**, the real bounded route returns MAE **1.0**;
   NumPy's author operation `mean(abs(true-pred))` returns
   **0.9999999403953552**. Controls at N=8,193 and N=16,777,219 agree. Thus tests
   up to 6.3 million elements did not establish the claimed general bit parity.
   This is a 5.96e-8 counterexample, NOT evidence of large error inflation in ECL.
   Quantify actual-cell impact before changing any published value. Preserve
   NumPy's actual scalar promotion/division semantics and version the route.

5. **High for deletion: internal consistency is not independent estimator validation.**
   `accept_catalog` (line 2047) checks identities, flags and relationships among
   error means, but not independent expected values for the other estimator
   families. I injected a producer defect that emits residual SD=-123, entropy=999
   bits, MI=999 bits and ACF=42, while preserving actual MAE/MSE and their summaries.
   Running the real closure recomputes and reads back that catalog; the real
   acceptance then returns **pass=true, no refusals, every oracle zero**.
   This is an executed producer mutation, not manual corruption after acceptance.
   In particular 64x64 discrete MI cannot be 999 bits, and a standard deviation
   cannot be negative. Add independent numeric references and domain checks for
   every claimed family, not another comparison with the same producer.

All five are in [probe.py](../evidence/RP113_MUSASHI_REVIEW_2026_09_23/probe.py)
and [PROBE_RESULTS.json](../evidence/RP113_MUSASHI_REVIEW_2026_09_23/PROBE_RESULTS.json).
They use the actual-author tiny CPU fixture and actual closure/deletion paths.
Only disposable files were changed/deleted. Production measurements were NOT
modified by these probes. No new warehouse query or independent reduction of
the real ECL prediction arrays was performed in this review.

## Scope and real progress

The earlier single-record rewrite and unresolved-report probes are covered now,
as are conflicting-copy checks. The new failures extend those acceptance
boundaries; they do not establish that the real retained scores were forged.
Regeneration can recover omitted calculations without fitting again, but its
reported bit identity and acceptance must be checked independently rather than
being promoted from a local flag. Keep all actual results, old failures and
deletion receipts with their dates. Do not globally restart experiments.

The unchanged focal suite also ran independently on WORKER_B, CPU-only, bounded
to 3 GiB / 180 s with one numerical thread: **61 passed, 1 skipped in 55.03 s**,
five fork-after-thread warnings. The adversarial probe's host readings were
45.0/53.2 C before and 57.0/55.7 C after. External GPU was observed at 38 C and
0% utilization before the CPU checks; no GPU computation was launched by Musashi.
No service restart, production deletion, transfer of prediction arrays home,
compression, or use of held GPU devices occurred in this review.

## Reporting and protocol B

T=96 has measurements and independent historical metric checks. Its missing
property is an accepted replay under the applicable rule, not the existence of
measurements. Use MEASURED / REPLAY_UNVERIFIED and retain the cross-device failure;
do not replace it by a widened tolerance or quietly pool it as verified.
Legacy training-device attribution remains inferred even after a new measured
replay: 66 seconds of GPU time alone cannot retroactively measure the old device.
Document the evidence scope without retraining merely to manufacture a new UUID.

The protocol-B mapping correctly distinguishes the released L512 script from
the unknown per-horizon searched-L choice in Table 9. Its cost rationale needs
correction: the pinned model has `num_patches=(seq_len-patch_len)/patch_len+1`,
which is **4 for both 96/24 and 512/128**, not 5.3 times as many tokens. Input
samples and projection cost change, not that token count. The pinned custom
loader prepends seq_len rows at the test border and reports length
`len(data_x)-seq_len-pred_len+1`; consequently, for the same test split and
horizon, **test windows are num_test-pred_len+1**, not fewer solely because L
increases. These are code-derived corrections, not a measured B runtime.
The 6-16 hour and VRAM-fit claims are not admission evidence; measure a bounded
exact-recipe pilot before scheduling that campaign. No B campaign approved here.

**Erratum by Musashi, 2026-09-23 (RP121 review):** the paragraph above incorrectly
assumed A's patch length was 24. The pinned `scripts/ECL.sh` declares 32: A has
three patches per channel (963 tokens), B has four (1284 tokens). Satoshi's RP120
correction is accepted. The loader-population statement remains correct. This
is my documentary error, not a reason to invalidate or repeat measurements.

## Disposition

No new deletions until mandatory accepted-evidence and independently validated
catalog gates pass. The no-compression/no-permanent-array-archive policy remains;
this is its prerequisite, not a permanent retention reversal. Do not re-delete
or rewrite old receipts. The owner need not issue another engineering approval.
Thermal holds remain: only the external 5090 is GPU-eligible after admission;
WORKER_A and the internal GPU remain held until explicit cooling confirmation.

Next orders: [RP114-RP121](../../handoffs/MUSASHI_SOTA_RP114_RP121_2026_09_23.md).
