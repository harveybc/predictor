# RP98-RP105 independent review

Reviewed `55a339ce0cd3ac28b0495c5ad4bde57736ee9fa1`.
Disposition: **CHANGES_REQUIRED; TRAINING PRESERVED; TERMINAL BLOCKER REMOVED;
NO FURTHER PREDICTION DELETION UNTIL THE CORRECTED GATE PASSES.**

## Findings

1. **High: post-deletion closure accepts rewritten scientific results.**
   `tools/df_sota_repro.py:1909` trusts fields in PREDICTIONS_DELETED.json and
   reads metrics from the current, unbound cell.json. The historical branch
   skips the accepted record, checkpoint, vault and original closure checks.
   Executed through the real verifier: after a valid fixture closure/deletion,
   change both metrics to zero and remove vault/checkpoint. It still returns
   `historically_verified_units=[unit]`, no problems, and the table uses zero.
   Replacing the referenced report digest by 64 zeros and removing REPORT.json
   also passes. Historical verification must resolve preserved evidence, not
   trust a local success flag or just the existence of a digest string.

2. **High: deleting copies matches a folder name, not the verified bytes.**
   `tools/df_sota_repro.py:1544` selects extra-root copies by unit name. Its
   per-root gate does not compare each selected copy's digest with the accepted
   artifact. In the executed fixture, an unrelated file with different bytes
   under the same unit name was deleted alongside the verified arrays. The
   inventory had both different hashes, so this is not an unobservable race.
   Refuse conflicting identities before unlinking anything for that unit.
   This demonstrates a tool defect, not a claim that the real deleted copies
   were different or that benchmark measurements have been falsified.

3. **High for the exact-reproduction requirement: official scoring is incomplete.**
   `bounded_test` (`tools/df_sota_repro.py:752`) explicitly skips the author's
   float32 scorer when its estimate exceeds the memory allowance; `metric_of`
   (line 2159) substitutes float64. T=336/T=720 took that path. Labeling it is
   honest, but does not finish the owner's requested author-metric reproduction.
   Three T=336 arrays were deleted before that missing calculation was finished.
   Retained checkpoints permit inference-only regeneration, not retraining;
   regenerated arrays must be identified as such and compared with preserved
   prediction/target digests. Do not assume chunked float64 means reproduce the
   author's float32 reduction or widen the rule after observing the difference.

## Device and numerical scope

`trained_device_of` (line 514) prefers a recorded actual UUID, but its historical
fallback picks the GPU with memory growth, or simply the first listed GPU.
Global memory growth is not process attribution. This is inferred identity,
not a measured CUDA UUID, and must not certify same-device repeatability.
New directly recorded UUIDs are a different, stronger evidence class.

`route_trace` (line 1290) selects three discrepant windows and compares their
batches. No routing change in those batches supports a scoped observation, not
a global explanation of every CPU/GPU or cross-GPU difference. Preserve the
4090-to-5090 pointwise failures and their exact rule. Equal aggregate metrics
do not repair those failures. No tolerance amendment is approved here.

The DataLoader/allocator changes do not justify shrinking the model or changing
the recipe. Existing batch-parity tests are useful; add optimizer/RNG trajectory
checks over actual-author updates before claiming complete training equivalence.

## What was independently executed

[Probe and output](../evidence/RP105_MUSASHI_REVIEW_2026_09_22/PROBE_RESULTS.json)
use the real author's tiny test fixture and real closure/deletion functions in
an isolated worker CPU scope. Only disposable artifacts were altered/deleted.
The unchanged focal suite on the reviewed commit also passed: **40 passed,
1 skipped, 21.07 s**; two fork-after-thread warnings were emitted. Those passing
tests do not cover the adversarial failures above. No benchmark fit or GPU
replay was launched by this review, and no production predictions were deleted.

The six real post-deletion records and checkpoints match their independently
queried accepted warehouse artifacts. All six retained catalogs match the
preserved pre-deletion report `fbe2cd4c...` and the current worker copies; each
deletion marker resolves that actual report. A **14,542,485-byte / 37-file**
metadata-only backup is now retained separately, within the approved budget.
[RETAINED_CUSTODY.json](../evidence/RP105_MUSASHI_REVIEW_2026_09_22/RETAINED_CUSTODY.json)
records these comparisons. This is custody verification, NOT recomputation of
every metric from arrays that no longer exist. Thus there is no basis here to
declare the real retained scores corrupted, or the full extended catalog newly
independently certified.

## Operational unblock executed by Musashi

The running warehouse's artifact-size column was INTEGER. I verified installed
provider bytes against the reviewed source, stopped only that service, took a
held and content-verified snapshot, rehearsed migration on its copy, exercised
**4,198,064,038 bytes** with transaction rollback, and started the service.
The full operation took **7.70 s**. Live queries prove BIGINT and identical
counts/content digests across all eight governance tables before any resend.
One intentional stop/start; zero automatic restarts. No other service restarted.
Private backup retained, rehearsal copy removed; no rollback was necessary.
[WAREHOUSE_ADOPTION.json](../evidence/RP105_MUSASHI_REVIEW_2026_09_22/WAREHOUSE_ADOPTION.json).

I then resent the **three original pending T=720 envelopes**, without inventing
successors, changing timestamps or training again. All three are accepted and
reconciled, pending=0, failures=0; all 12 cell records/checkpoints match accepted
artifacts in the live warehouse. [LIVE_RECEIPTS.json](../evidence/RP105_MUSASHI_REVIEW_2026_09_22/LIVE_RECEIPTS.json).
After resend: 2,571 terminals, 144,264 metrics, 265 datasets, 373 artifacts.
Terminal acceptance is not a numerical replay or scorer certification.

## Current scientific disposition

Keep the reported H192/H336 figures as qualified measured evidence, not a new
independent metric recomputation by this reviewer. H336 retains its float64
label. H96 cross-device failure remains open; original-device GPU work stays
held until cooling is confirmed. H720's ingestion blocker is gone; its remaining
replay, catalog and official-scoring work is now Satoshi's next executable task.
Mixed-host training dispersion is not pure seed variability. Operational
agreement margins are not statistical equivalence or exact numerical equality
to a rounded paper table. No complete official-metric four-horizon result is
certified by this review.

Next orders: [RP106-RP113](../../handoffs/MUSASHI_SOTA_RP106_RP113_2026_09_22.md).
The only owner information still requested is physical cooling status for
WORKER_A. No new owner authorization is required for the independent blocks.
