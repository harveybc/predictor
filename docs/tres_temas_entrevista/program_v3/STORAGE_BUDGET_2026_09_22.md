# Storage budget using existing disks only

Measured 2026-09-22, approximately 06:58 UTC. Owner constraint: no NAS, new disks,
cloud storage or assumed future capacity. Keep the travel thermal hold. No
prediction compression. Delete inventoried predictions after independent analysis
and verified durable metric storage, as already authorized.

Evidence: [assessment](../../audits/evidence/STORAGE_CAPACITY_2026_09_22/ASSESSMENT.json),
[metadata snapshots](../../audits/evidence/STORAGE_CAPACITY_2026_09_22/SNAPSHOTS.json),
[read-only probe](../../audits/evidence/STORAGE_CAPACITY_2026_09_22/inventory.py).
No prediction values were loaded by this capacity assessment; it is not the
independent scientific verification required before deletion. Sizes can change
while the owner and Satoshi continue work. Recheck at dispatch and deletion.

## Available disk, not RAM or nominal drive size

GiB means 2^30 bytes. Only the mounted filesystem containing each home directory
is counted, once per physical filesystem. /tmp is tmpfs on all three machines
and contributes ZERO durable disk capacity. Existing datasets, environments,
databases and unrelated work are already deducted by filesystem availability.

| Existing role | Available now GiB | Identified prediction copies, conditional reclamation GiB | Available after that deletion GiB | Planning budget now, with 50 GiB operating reserve |
|---|---:|---:|---:|---:|
| COORDINATOR | 147.10 | 2.69 | 149.80 | 97.10 |
| WORKER_A | 673.11 | 3.76 | 676.87 | 623.11 |
| WORKER_B | 214.39 | 0.00 | 214.39 | 164.39 |

Total currently available: 1,034.60 GiB. Planning capacity after reserving 50 GiB
on EACH filesystem: 884.60 GiB. The 50 GiB floor is an explicit operating policy,
not a measured requirement or extra storage. The 6.45 GiB identified as possible
reclamation is NOT counted in the admission budget until actually freed. These
are SOTA prediction copies, not a claim to have found every disposable artifact.
Other data, checkpoints and historical database snapshots are not deletion targets.

## Exactly what remains after a cell is analyzed

1. One retained best checkpoint per measured cell, with its model configuration
   and code/environment identity. Do not retain every epoch's duplicate checkpoint.
   Do not remove accepted checkpoints while their audit is still open. Later
   model retirement is a separate explicit lifecycle decision, not this prediction
   deletion instruction. Keep a resumable state only while the job needs resuming.
2. A versioned metric bundle: official MAE/MSE and matched baseline results,
   sample counts, bias/RMSE and appropriate diagnostic scores, per-horizon and
   per-channel tables, time-block results and the paired contrasts actually
   required by the experiment. Store values and all estimator definitions,
   units, reductions, denominators and numerical precision, not just plots.
3. The bounded residual/information diagnostic catalog being audited: moments,
   quantiles and histogram counts with under/overflow; declared ACF/PACF or
   spectral diagnostics where applicable; correlations/information estimates
   with their binning, lag range, populations and approximation limits. Freeze
   the chosen grains and estimator parameters. No promise of every conceivable
   cross-product, future subgroup or information estimator. No silent truncation
   of necessary metrics merely to satisfy a byte allowance.
4. Training curves, stopping reason, actual updates, independent verification
   reports, campaign/terminal receipts, digests, failures and deletion inventory.
   Input/target provenance, scaler state and row/horizon identity rules remain.
5. Shared source data/code once per required version/location, referenced by
   digest across cells; no full dataset or materialized target-window copy per
   model. Preserve the provenance needed to derive targets independently.

Remove raw per-observation prediction copies on all inventoried locations AFTER
their required analyses close. No permanent prediction archive, no compression,
no per-observation residual clone disguised as a metric bundle. Temporary
predictions exist only during the cell/comparison lifecycle. Keep small metric
and receipt replicas on a second EXISTING host; no prediction replicas afterward.
Historical metric verification remains dated; current raw-output replay becomes
unavailable after deletion. Do not claim a cached summary verifies absent arrays.

## Current complete primary SOTA design: 12 cells

The sealed design contains 4 horizons x 3 seeds, L=96 and 321 channels.
Counts below are from DESIGN.json, not the inconsistent window-count transcription
in the prose lock. Formula: windows x horizon x 321 x 4 bytes (float32).

| Horizon | Test windows | Raw predictions per cell, bytes |
|---:|---:|---:|
| 96 | 5,165 | 636,658,560 |
| 192 | 5,069 | 1,249,650,432 |
| 336 | 4,925 | 2,124,763,200 |
| 720 | 4,541 | 4,198,063,680 |

Sum for all 12: **24,627,407,616 bytes = 24.63 GB = 22.94 GiB** of predictions
alone if retained, excluding NPZ container overhead. This is a design projection,
not 24 GB currently reclaimable. Only four distinct cells have measured retained
records in the snapshot; there are additional physical transport copies.

Observed checkpoint files: 27,783,541 bytes at H96 and 28,373,749 bytes at H192.
Four current METRICS_VAULT files together: 2,448,239 bytes; largest 877,981 bytes.
Their existence/size does not certify the metrics. The broader catalog is still
under review and can exceed the current vault. Original processed dataset:
95,581,762 bytes, shared across cells.

Explicit conservative PLANNING ALLOWANCES, not measured final sizes:

- Checkpoint: 64 MiB/cell (over twice the observed H96/H192 files).
- Expanded metrics, logs, receipts and indices: 128 MiB/cell.
- Shared input, source/configuration and campaign overhead: 1 GiB.
- Primary retained bundle: 12 x (64 + 128) MiB + 1 GiB = 3.25 GiB;
  round reservation up to **4 GiB** on WORKER_A.
- Second metric/receipt copy only: at most **1.5 GiB** on COORDINATOR.
- Temporary disk reservation: **32 GiB per active verification/fit slot** on
  the worker, including prediction/target working arrays and a verification copy.
  Use one slot initially. This is a disk reservation, NOT a RAM allowance or a
  measured bound on every author implementation; preflight must confirm it.

Thus the planned primary campaign allocation is 36 GiB on WORKER_A and 1.5 GiB
on COORDINATOR, before crediting any deletion. Both fit their measured budgets
by wide margins. Final long-horizon checkpoint/catalog sizes must be measured
before acceptance; exceeding an allowance triggers recalculation, not a smaller
model or silently omitted diagnostic. The optional L512 protocol is not included
in this 12-cell total; give it a separate manifest and allocation if activated.

## Complete research program: what can and cannot be certified

The master covers SIX scientific objects and multiple conditional stages. It does
not yet fix every dataset, receiver, architecture candidate, replication count,
training-volume tier, fold or retained-checkpoint size. Therefore a total lifetime
storage requirement for the full program is **NOT IDENTIFIABLE YET**. Claiming
that all future experiments fit would invent missing experimental populations.

One concrete warning: the deferred financial v4 blueprint has
26 candidates x 3 seeds x 26 folds x 2 horizons x 2 receivers = **8,112 cells**
for its primary history setting. Its receiver choices await the SOTA-first
successor, so this is not permission to run those old compact models. Blindly
assigning the SOTA planning allowance of 192 MiB/cell to that population alone
would require **1,521 GiB**, exceeding the 884.60 GiB current planning capacity.
Actual financial metrics/checkpoints may be much smaller, but must be measured;
do not reuse the electricity storage forecast as a full-program guarantee.

Operational solution WITH THE EXISTING DISKS: execute bounded campaign batches;
analyze and prune prediction workspaces before allocating the next batch; retain
shared data by identity and account for cumulative metrics/checkpoints/warehouse
growth. Large models are admitted by actual resource needs, not downgraded for
convenience. The remaining finite scientific designs need cell-count and storage
manifests when finalized, before dispatch. This is resource accounting for their
existing experiments, not a new scientific campaign or a request for new hardware.

Per-filesystem admission condition:

    available_now >= 50 GiB operating reserve
                     + reserved_growth_of_other_active_jobs
                     + new_job_peak_temporary_disk
                     + new_job_retained_artifact_growth

Count input transfers, working copies, metrics replicas, checkpoints, temporary
write/rename overlap, database growth and retained backups. Do not add capacities
as though three separate disks were one local filesystem. Refresh availability
before each allocation; concurrent allocations need one reservation ledger.
If it does not fit, serialize, reclaim only eligible disposable outputs, or report
the quantified deficit. No nonexistent NAS and no unbounded retention promise.

## Remaining implementation and non-disk constraints

The catalog audit, authorized all-copy deletion and reservation enforcement are
not certified implemented by this document. No predictions were deleted during
this assessment. Public metrics and the stored receipt contents must still be
verified before deletion; do not skip that requirement to claim freed space.

Satoshi's measured host-RAM deficit for the author's long-horizon evaluation is
separate. Free disk does not imply that the evaluator fits RAM or that swapping
to disk is a validated solution. No local GPU work is authorized on the traveling
coordinator. Placement of a GPU job requires both memory and disk admission.
