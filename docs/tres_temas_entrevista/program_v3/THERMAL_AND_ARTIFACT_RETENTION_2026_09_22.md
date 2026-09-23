# Travel thermal placement and prediction retention

Owner instruction, 2026-09-22. Applies immediately to the active SOTA orders.
This changes execution placement and storage, not the frozen ML recipe.

## Thermal placement

- Standing dispatch preference, independent of the temporary travel holds:
  WORKER_B's external RTX 5090 is the FIRST choice for GPU tasks, especially
  individual jobs, even after other devices become eligible again. Do not
  distribute work evenly merely to occupy all GPUs. Prefer earliest feasible
  completion using measured runtime, queue, transfer and memory costs; the owner's
  reported speed advantages are motivation, not measured per-task projections.
  Other eligible GPUs are secondary for useful parallel work, incompatibility,
  external-device unavailability or an explicitly required original-device replay.
  This does not release current holds, migrate active jobs or authorize a restart.
- After an external-power interruption, host reachability does not establish
  that the eGPU returned correctly. Verify its physical UUID and driver state,
  then an admitted bounded CUDA health check before dispatch. If missing or
  unhealthy, stop admission to that device and diagnose; no silent internal-GPU
  fallback or automatic host reboot. Preserve active work before any separately
  coordinated recovery. Do not assume a future UPS is available infrastructure.
- The traveling coordinator performs light orchestration only. No new heavy
  CPU/GPU training, inference replay, decompression sweeps or maximum-compression
  benchmarks there. Existing desktop and governance services remain untouched.
- Owner clarification, 22-sep: ONLY WORKER_B's EXTERNAL RTX 5090 is eligible
  for new GPU work, subject to admission. Its independent cooler is the basis
  for that authorization, not a new temperature measurement by Musashi.
  WORKER_B's internal RTX 5070 Ti and WORKER_A's GPU remain held. The expected
  return tomorrow afternoon does not automatically release any hold.
  Verify physical UUID, connection, available memory, measured thermal state
  and compatibility; absence of a compute PID does not prove idle. Require the
  child process to use that physical device. No fallback to an internal GPU.
- The external cooler does not establish safe host CPU temperatures. Bound host
  CPU work, thread count and concurrency; monitor host and GPU temperatures.
  If either fails admission or its thermal limits, stop the work safely and
  continue only independent lightweight tasks. No unbounded CPU fallback.
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

## Owner supersession: analyze, retain metrics, delete predictions

The owner's later instruction on 2026-09-22 withdraws the earlier compression
and permanent-archive requirements. Satoshi is already implementing that order;
this amendment aligns Musashi's documents and creates no competing campaign.
No compression benchmarks, archive resolver or retained canonical prediction
copy are required. Deletion of the last prediction copy is expressly authorized
AFTER independent analysis and verified durable storage of the agreed metrics.

Before deletion, inventory run/cell identity, original accepted digest, exact
paths and copies on workers/coordinator, active readers/writers and transfers.
Verify the metric catalog independently, including its population, units,
reductions, finite/degenerate behavior, bin ranges, lags and approximations.
Publish metric artifacts and verify their persisted content. A file named
METRICS_VAULT.json or a test asserting keys exist does not establish correctness.
Close dependent comparisons before deleting their inputs. Do not delete files
that another active process is reading, writing or transferring.

Delete only inventoried prediction artifacts after those conditions pass, on
all known locations, including the final copy; retain per-path deletion receipts
and actual bytes reclaimed. Preserve configurations, code/environment identities,
data/scaler identities, checkpoints, metric artifacts, verification reports and
accounting. Do not delete an entire run with its supporting provenance.

Separate historical verification from current replayability. Original predictions
will no longer be available for direct re-audit. Their hash records their former
identity; it cannot reconstruct them. A later regeneration is a new artifact,
not automatically byte-identical. Missing arrays must not silently pass a current
array verifier using a cached metric receipt. Historical verification remains
dated and qualified by authorized deletion, not erased or relabeled as false.

## Scope of the metric catalog and future experiments

Use a finite, explicit catalog tied to the planned questions: official errors
and matched baselines; per-channel/horizon/time-block errors; residual moments,
tails and distributions; declared ACF/PACF or spectral/information diagnostics
where applicable; paired contrasts required by the current design. Each estimator
needs parameters, sample population, uncertainty/limitations and an independent
check. Undefined is not zero. Approximate quantiles or clipped histograms must
not be described as full exact distributions. Do not claim all possible future
metrics or subgroups are recoverable from these summaries.

STEP-12 already outlines adaptive routing and chronological OOS expert predictions
(source: STEP_12_ADAPTIVE_INFORMATION_QUALITY_ROUTING_LINK_ADAPTATION_FINAL.md,
sections 44-47 and 57). Its project financial protocol depends on upstream
eligible modes and is not a sealed, allocated current financial experiment.
META similarly remains deferred. Neither is justification for blanket retention
of present ECL or failed exploratory model predictions. When that experiment is
designed, generate the required dedicated OOS predictions for its frozen eligible
model bank and retain them only through its declared analysis lifecycle.

The current e98b269 implementation stores predictions uncompressed and adds a
metric vault. Its breadth and correctness require audit; this amendment does
not certify it. No new experiment is authorized solely to justify storage.

## Scope of this update

No artifacts deleted or re-encoded. No service restarted. No scientific run
launched. The first local sample showed CPU 61.4 C, GPU 54 C; after the user's
compression process finished, CPU was 57.5 C. These are observations, not a
guarantee about future thermals. Remote utilization must be rechecked at dispatch.
