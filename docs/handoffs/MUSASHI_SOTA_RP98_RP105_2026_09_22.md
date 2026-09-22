# Satoshi: RP98-RP105, close evidence gaps and finish the faithful reproduction

Base: `fb6ec31e`. Read the
[independent review](../audits/work_plan/MUSASHI_RP90_RP97_REVIEW_2026_09_22.md)
and retained executable PRE first. This supersedes RP90-RP97 as the active order.
The owner has already authorized the scoped work. Finish all feasible blocks,
update persistent state incrementally, then return one consolidated audit request.
Do not ask the owner to approve each step. Do not bypass a runtime refusal.

Thermal/storage policies remain binding. No heavy coordinator work until the
owner confirms restored cooling; Sep 24 is not an automatic restart instruction.
Prefer the cooled WORKER_A GPU after fresh admission checks. Use WORKER_B only
with measured free RAM and an actually available compatible physical GPU. Do not
interrupt its eGPU workload or WORKER_A's VM. No compression, NAS, cloud, new
disks, larger unapproved memory limits or mass transfer of predictions home.

## RP98: freeze and reproduce the actual findings

Preserve Musashi's probe and JSON unchanged. Reproduce its successful fixture
baseline and the four false-success/undefined examples before changing code.
Add expected refusals to the actual closure and metric path, not source-text
tests or a substitute implementation. Preserve current benchmark cells and
original CPU replay failures. No re-fitting to repair a verification defect.

## RP99: make the metric catalog fit for authorized deletion

Bind each vault to exact prediction/target/preparation identities, shape, row
ordering, metric implementation, numeric space and estimator parameters. Refuse
short/extra/reordered loaders, invalid shapes, non-finites and wrong horizons.
Do not publish a population from array shape without proving it was consumed.
Verify reused vault contents against accepted evidence or recompute; a freshly
calculated hash of a changed local vault is not acceptance. Publish successors
atomically and preserve rejected candidates with a disposition.

Use an explicit finite catalog with DONE/UNDEFINED/APPROXIMATE/NOT_APPLICABLE
states and reasons. Follow the approved retention document for required errors,
matched baselines, per-horizon/channel/time-block summaries, paired current-design
contrasts and residual/information diagnostics. Complete planned time-block and
paired-seed summaries before deletion; do not call a list of global keys exhaustive.
Use independent numeric oracles for every estimator family, degenerate variance,
zero baseline/percentage denominator, short support, histogram overflow and ACF
population. Count excluded observations. Undefined correlation is not zero.
Rename test-relative MAE as such, not MASE. Percentage errors in centered z-space
are not physical percentage errors; label any retained diagnostic accordingly.
No unbounded sweep of all conceivable future metrics or retention workaround.

Recompute the corrected catalog on the four retained cells without fitting, using
Musashi's independent reduction as an external check of the overlapping metrics.
That reduction certifies only its stated scope, not every extended statistic.

## RP100: separate replay properties; authenticate the report

The real closure must reject altered cached replay results even when their input
identity is unchanged. Bind the accepted output to inputs, environment, actual
device identity and replay code. Validate finite values, shape/count, full pointwise
comparison and metric reductions. Never infer success from a supplied boolean;
never use aggregate error equality as a substitute for prediction equality.

Keep four separate fields: recipe fidelity, same-device repeatability, cross-device
portability, published-score agreement. Freeze a successor scope before new work:
same-device fresh-process full predictions use the existing atol/rtol 1e-4 rule
and original metric check; also report exact equality. Cross-device runs retain
that rule as a separate portability test. No enlarged tolerance inferred from
the observed maximum. Old CPU failures remain failures of the original rule;
a scoped successor must not relabel their history as passed.

Trace the actual routing on discrepant CPU/GPU rows before assigning a cause:
log route decisions and distances to thresholds, compare pre/post-route tensors,
repeat under identical batching, and distinguish numerical, stochastic and
environment effects. Bound this diagnostic; no training or changed top-p to make
it pass. Test that changed/permuted predictions with equal aggregate metrics fail.
H192's original-device replay is pending while that device is held: a worker
replay is cross-device evidence, not a claim of original-device equality.

## RP101: remove avoidable evaluation memory, preserve the computation

Profile training, author test(), target generation, scorer and independent
closure separately. Implement the narrowest operational disk-backed/bounded
buffer path needed for evaluation and verification. Keep author model, weights,
inputs, chronology, batch composition/order, precision and training untouched.
Do not accumulate all targets twice or retain unused full input arrays.

Before adoption, compare original and bounded paths on a small actual-author
fixture and existing H96 checkpoints: elementwise predictions/targets, complete
population, checkpoints untouched and official metrics. Preserve the author's
float32 reduction semantics or report/reject a demonstrated difference; chunked
means are not assumed bitwise equivalent. Include partial final batches, batches
of different sizes, interrupted writes, missing chunks and finalization failures.
Record the adapter diff and immutable source pin; do not silently edit the author
clone or claim an unmodified implementation if an adapter executes.

Use the approved WORKER_A disk envelope (32 GiB temporary slot, 4 GiB retained
primary and existing operating reserve), one cell at a time initially. Measure
cgroup RAM including file-backed pages, disk peak, GPU VRAM, thermals and elapsed
cost for H192/H336/H720 before admission. A memmap alone is not a memory proof.
Do not count prospective deletion as already free space. Keep the existing slice
limits and other workloads intact. If a real model/VRAM deficit remains, give
its minimum requirement and finish independent blocks, without substituting a
smaller model, window, dataset or test population.

## RP102: complete the finite author population when admitted

Continue protocol A's eight missing cells: H192 seeds2022/2023; H336 and H720
seeds2021/2022/2023. Reuse the four completed cells and version failed attempts.
Register before reading/fitting, close all units, account real device/CPU/GPU
costs and reconcile accepted contents with the live warehouse. No dataset, loss,
optimizer, early-stopping or precision tuning against public test scores.

Use the original author epochs and recipe, not the old 14,400-second pilot cap
as a reason to truncate. Allocate finite jobs with measured per-cell limits and
the complete projected ledger; do not launch an unlimited unattended sweep.
Check public-test monitoring for recipe fidelity; it never makes that benchmark
an untouched financial holdout. Audit the sealed source/receipts and replay all
new cells through the corrected path.

Protocol B is a distinct work item, not implied by A: map Table 9's searched
lookbacks to actual author invocations and freeze the supported cell population.
If the exact published per-cell choice cannot be established, report that gap
rather than asserting that a fixed L512 script reproduces the searched table.
Run only after its own recipe/admission checks and within existing resources.

## RP103: correct agreement and complete analysis, then delete predictions

Report published values, actual per-seed values/differences and measured seed
dispersion for each horizon. Keep the old borrowed-SD decision as a labeled
operational heuristic, not published per-horizon uncertainty. Do not widen it.
For the four-horizon average, average within each matched seed FIRST, then across
seeds. Missing, duplicated, foreign or unverified cells prevent a full average.
Test the aggregation against independent numeric examples with distinct horizon
means and identical seed results so between-horizon spread cannot masquerade as
seed spread. Numerical exact equality is observed, never guaranteed or optimized
on the test set.

For each cell whose dependent analyses and corrected metric catalog are finished,
publish the catalog, independent checks, code/data/checkpoint identities and
read back persisted contents. Preserve cross-device failure diagnostics needed
for the current review before deleting their inputs. Prove there are no active
readers/writers/transfers; enumerate exact prediction paths/copies and hashes.
The owner's deletion permission already exists: delete these inventoried arrays,
including the final copy, once the gate passes. No additional owner approval,
compression or permanent prediction archive is required. Delete only predictions,
not entire roots; preserve source inputs, checkpoints, configs, metrics and
receipts. Record actual reclaimed bytes by filesystem, including worker copies.
Never call missing arrays a successful current replay; retain dated historical
verification and authorized-deletion disposition. Test this lifecycle explicitly.

## RP104: verify the installed paths and environment scope

Run the small author/Torch suite in a fresh process, plus new adversarial rules,
bounded-path parity, catalog oracles and deletion lifecycle tests. Run other suites
only in compatible environments on workers. Name each remaining failure/skip and
whether it changes the exercised reproduction path; the reported 24 environment
failures are not automatically covered by the older three-failure/eight-error
legacy exception. Repair prerequisites actually required by this work, not an
unrelated repository-wide refactor. No heavyweight local suite on the coordinator.

## RP105: one return, persistent state, publication

Update statuses after each block. Return one table with official normalized MSE
and MAE, paper reference and protocol, persistence and seasonal naive on identical
rows, per-seed dispersion, population, recipe/replay/portability scopes, costs,
thermal placement and actual disk reclaimed. Separate observed values from
fully accepted reproduction, and distinguish published operational agreement
from statistical equivalence. Include PRE/POST, live reconciliation, scope of
independent verification and exact remaining deficits. Publish commits and sync
worker code through the established route; do not sync bulk prediction banks.

Historical household work remains excluded from active selection. Financial
Huber/MAE x Adam/AdamW, weekly forecasting/RL and doctoral modular/core-pretraining
hypotheses remain required downstream, not replaced by this benchmark. Do not
launch those fits while faithful-reference acceptance is still open.
