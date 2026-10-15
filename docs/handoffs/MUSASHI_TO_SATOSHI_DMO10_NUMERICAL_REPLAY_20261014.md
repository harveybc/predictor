# Satoshi: DMO10 numerical replay clarification

Read while the bounded correction runs. Do not restart a scientific campaign,
change DOIN or launch another unrestricted debugging sweep for this message.
The previous business plan remains unchanged.

## What the retained evidence proves

GPU_REPLAY.json at predictor `8f8c4c1f` records a first-batch discrepancy under
monitor code `7c3bebb2`. It supports a reproduced monitor discrepancy, not a
demonstrated cuDNN/cuBLAS root cause. Autotune/trace/stream selection remains a
hypothesis until controlled isolation establishes it. Having an environment
variable set is not proof that the pinned TensorFlow build consumes it.

From its recorded/recomputed partial sums, using
`abs(recomputed-recorded)/abs(recorded)`, the relative differences are:

| Short head | Relative difference |
|---|---:|
| 2 h | 3.6415407435943697e-6 |
| 3 h | 1.2086535047264033e-5 |
| 4 h | 1.6731069808049074e-5 |
| 5 h | 8.331157347135476e-6 |
| 6 h | 1.8244613879748877e-5 |
| 12 h | 1.4897553155335735e-6 |

Three heads exceed 1e-5. Do not describe this retained witness as bounded by
1e-5, or increase a threshold until the existing failure passes.

`evaluate_monitored` computes these partial sums with CPU `math.fsum` over
float64 prediction/label arrays. Bit-identical predictions, labels, masks,
ordering and this same procedure must produce identical partial sums. A
GPU forward discrepancy is possible, but the current witness binds row IDs,
counts and evaluator source, not every consumed tensor byte.

## One bounded diagnostic, not repeated training campaigns

1. Freeze the first failing batch, checkpoint and preprocessing identity.
   Preserve exact X, Y, masks, row order, model/weight hashes, shape/dtype,
   scales, framework/build/device, inference trace and numerical flags.
   Keep arrays local; Git gets digests and the diagnostic program only.
2. Compare predictions element by element before reducing any metric. Record
   full tensor hashes and first differing element, absolute/relative/max/mean
   errors, then independently reproduce CPU float64 fsum sums. If predictions
   differ, do not claim cross-process prediction bitwise identity. If they
   agree but sums differ, investigate labels/masks/reduction/artifact identity.
3. Use supported pinned-runtime deterministic/algorithm settings established
   before TensorFlow initialization in the launcher and child. Log and verify
   effective settings, batch/layout/dtype, TF32/mixed precision, XLA/fusion and
   loaded software/device versions. Do not silently alter precision to pass.
   A controlled flags-on/off comparison can support the autotune hypothesis;
   a flag registry alone cannot. Reuse the retained checkpoint for fresh-process
   inference; no repeat production fits or seed averaging. If training-context
   reproduction is indispensable, use one bounded synthetic engineering case.
4. Record same-environment replay separately from cross-device replay. Do not
   require identical 5090/4090 arithmetic as if they were identical hardware,
   nor excuse a same-environment defect as an inevitable process property.

## Tolerance and acceptance

- Default DMO10 remains the sealed exact contract. Static algorithm selection
  is a reasonable attempted repair, not a guaranteed successful repair.
- A new numerical-equivalence rule is a separately versioned procedure,
  declared before its new acceptance replay, not predeclared for old witnesses.
  Its label is NUMERICALLY_EQUIVALENT, never BITWISE_IDENTICAL.
- If justified, restrict any tolerance to the diagnostic comparison of monitor
  replay. Declare absolute AND relative bounds and zero/near-zero handling;
  retain per-head, per-batch and aggregate differences. The old witness must
  retain its original FAIL if it exceeds the new bound.
- Do not change training min_delta=1e-8, patience or checkpoint-selection
  comparisons to accommodate replay. Show whether the measured uncertainty
  could change best_epoch, patience resets, stop_epoch, head eligibility or
  naive ordering. An all-last-epoch fixture is insufficient to establish that
  close checkpoint decisions are invariant; add a near-tie negative control.
- No tolerance propagates to DOIN champion comparisons, migration, consensus,
  chain identity, metrics/naive pairing or warehouse integrity.
- If elementwise predictions are genuinely bitwise equal across the contexts
  being claimed, prove it with their hashes; otherwise state the narrower
  guarantee honestly. No unconditional promise that the next round closes
  DMO10 as PASS. PASS, numeric equivalence and unresolved FAIL are distinct.

Return one concise artifact with the causal attribution status, tensor
comparison, flags, old/new protocol identities, decision-impact check and exact
acceptance label. Then let the established automation proceed only where its
actual gates are satisfied. No chat polling or new owner permission prompt.

## Primary references

- TensorFlow determinism requirements (same hardware/software and deterministic
  inputs): https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
- NVIDIA cuBLAS reproducibility guarantees and stream/workspace qualifications:
  https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Check supported settings against the installed toolkit/build, not just the
latest documentation version.

## Follow-up: proposed rtol 5e-5 / atol 1e-9

The revised bound is acceptable as a prospectively versioned
NUMERICALLY_EQUIVALENT monitor-replay protocol, conditional on independent
verification, not as retrospective BITWISE_IDENTICAL acceptance. Do not
inflate it again to fit a failing witness. State the observed discrepancy as
specific to the tested build/contexts; exclusion of eight hypotheses is not
proof of a universal or irreducible library property.

At inspected tip `0bcf7f6b`, the published source still uses rtol 1e-5 and no
new combined bound. Its tolerance activation at `train.py:508` checks only
TF_CUDNN_USE_AUTOTUNE==0. It does not enforce the claimed GPU/cross-process
scope. `_compare_witness` also tolerates monitor forward tensor differences,
not just sums. Inspect the NEW actual implementation, not the proposed prose.

Before release, the independent reviewer must verify these concrete points:

1. The new protocol's identity seals rtol/atol, reference direction and formula
   `abs(actual-reference) <= atol + rtol*abs(reference)`, scope, dtype, flags
   and runtime version. Old receipts cannot silently inherit the new rule.
2. CPU and same-process self-check paths remain strictly exact. A GPU
   cross-process diagnostic is explicitly identified and proven, rather than
   inferred from an environment flag. Tests cover CPU with GPU-style flags,
   same-process corruption, near-zero sums, boundaries and over-bound rejection.
3. Distinguish monitor_forward tensors from exported serving predictions.
   The retained witness has differing monitor tensors, so do not claim those
   are bitwise. Keep exported checkpoint reload/serving equality tests exact
   in the same explicitly named context. Altered inputs, masks, counts, graph,
   weights or metadata cannot be excused by a small metric delta.
4. The tolerance affects only replay verification, never the training stopper,
   min_delta, patience, best-weight restoration, naive eligibility, DOIN or
   OLAP. Test a near-tie early-stopping history. Report any inability to
   reproduce checkpoint selection across contexts rather than claiming full
   decision invariance merely because replay sums are close.

Start the independent review on the frozen candidate now; do not wait for an
author-declared green to begin reviewing. Use bounded synthetic/retained-artifact
tests, no new scientific fit and no new sequence of eight diagnosis rounds.
Proceed through the established automation once the revised, honestly labelled
acceptance passes. No additional owner approval is required for these steps.
