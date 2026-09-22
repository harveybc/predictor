# Travel thermal placement and prediction retention

Owner instruction, 2026-09-22. Applies immediately to the active SOTA orders.
This changes execution placement and storage, not the frozen ML recipe.

## Thermal placement

- The traveling coordinator performs light orchestration only. No new heavy
  CPU/GPU training, inference replay, decompression sweeps or maximum-compression
  benchmarks there. Existing desktop and governance services remain untouched.
- Prefer existing cooled remote workers, selected by actual available memory,
  measured thermal state and GPU compatibility. The external 5090 is eligible
  only after a device-specific check; absence of a compute PID does not prove idle.
- Select by physical device UUID in execution records, not assumed CUDA index.
  Record actual host-role/device, environment, start/end temperature and cost.
  Respect existing per-device thermal/admission policies. Do not invent a universal
  safe temperature from a single sensor sample. Do not change the benchmark model
  or numeric recipe to accommodate migration without declaring the difference.
- No local fallback if workers are unavailable. Finish independent lightweight
  work and report actual resource availability. Do not modify other users' jobs.
- A private local marker at ~/.config/crispdm/coordinator-travel-hold makes the
  installed crispdm-run refuse new batch jobs. This is a launcher guard, not a
  machine-wide GPU lock. Direct launches must obey the same rule. Removal requires
  owner confirmation that cooling is restored; services need no restart.

## What can be removed

User authorizes reclaiming storage that is no longer needed. Independent metric
verification alone does not make the sole prediction array disposable: summaries
cannot recover paired errors, tails, subgroup performance or prove a later metric
bug absent. Checkpoint replay can vary numerically; it is not the original output.

Remove working/local duplicates after verification and verified remote archival.
Keep one lossless authoritative prediction artifact for every retained scientific
cell, including failed/negative results needed for honest population accounting.
Keep a second verified copy on a separate failure domain for irreplaceable accepted
evidence when storage policy requires it; it need not be on the traveling laptop.
Never infer dispensability from a poor score. A later decision to destroy the last
copy must explicitly retire the corresponding replay/audit claim, not retain VERIFIED.

Before deleting a specific copy:

1. Inventory run/cell identity, path, size, original accepted digest, and every
   writer, reader, sync, outbox or closure that still depends on that location.
2. Complete independent metric and checkpoint verification on a remote worker.
   An unfinished campaign or transfer is not eligible for automatic pruning.
3. Produce the lossless archive atomically at a durable governed location, with
   codec/version, compressed digest, original digest and provenance manifest.
   Read-back restore in a separate directory must match the original accepted
   file digest, not just a new hash computed from an unanchored current copy.
4. Run the real closure/verifier on restored artifacts, including predictions,
   checkpoint and input identities. Test corrupted archive, wrong cell, missing
   chunk, unavailable archive and interrupted publication as refusals.
5. Publish an additive location/retention receipt and confirm the authoritative
   resolver can find it. Never rewrite the original accepted artifact digest.
6. Delete only the exact redundant paths from the verified inventory, record bytes
   actually reclaimed, and check no consumer still expects the removed path.
   Do not glob-delete a run or operate on an in-flight sync/compression input.

## Current implementation and compression experiment

At inspected predictor revision cd07830, df_sota_repro writes only `pred` into
np.savez_compressed(arrays.npz); targets are derived and hashed separately. It
already avoids storing a full repeated target tensor in that file. Its verifier
and replay still require arrays.npz at the original working path. There is NO
archive-aware resolver yet. Until one passes tests, restore the exact original
file in an isolated working root before verification. Do not turn a missing
arrays.npz into successful verification based on a cached metric receipt.

Benchmark gzip/bzip2/xz/zstd or other approved lossless codecs on representative
samples on a cooled worker. Choose by measured space saving, compression CPU,
decompression time, peak memory and exact restore, not by maximum level alone.
Avoid compressing every full artifact with every algorithm. Comparing raw float32
to an NPZ container must disclose container overhead and included fields. Do not
assume the sample ratio holds for all horizons or series.

Float16, decimal rounding and lossy residual coding are NOT archival compression:
they can erase the marginal improvements being studied. A redesigned chunked
format needs a reversible mapping, dtype/shape/order metadata and canonical
tensor verification, plus a successor artifact record. If existing custody binds
the ZIP/NPZ file bytes, numerical array equality alone cannot replace that identity.

Prefer computation close to artifacts and on-demand restoration. Retain small
design/config/env locks, training logs, checkpoint, data/scaler identities,
row/horizon mapping, verification reports and accepted accounting receipts.
Do not keep multiple full local prediction banks merely to display a table.

## Scope of this update

No artifacts deleted or re-encoded. No service restarted. No scientific run
launched. The first local sample showed CPU 61.4 C, GPU 54 C; after the user's
compression process finished, CPU was 57.5 C. These are observations, not a
guarantee about future thermals. Remote utilization must be rechecked at dispatch.
