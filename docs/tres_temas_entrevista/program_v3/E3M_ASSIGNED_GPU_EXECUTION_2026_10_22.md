# Assigned E3-M GPU Package: Sequential Execution

Authority: owner2026-10-22 explicitly requests all12 assigned arms, unattended.
This is Satoshi's pinned E3-M diagnostic package, NOT native SAC and NOT a
change to Musashi's no-feedback SAC baseline. Full/none and regularization
treatments here are explicitly part of the assigned diagnostic factorial.

## Frozen Science

Package heuristic-strategy31d90f7, archive sha256
efa23475193865fc6594f7af4f6ee61a7d2cbb41d96756af89a8ed3cd04e202a.
The supplied command defines4 consolidated arms then8 dev3y diagnostics.
Preserve units2/head16/FULL/max-epochs1000/fixed_train/nonstationary-diff,
the pinned defaults, three seeds, labels and every business setting.
No TEST, no DOIN, no native SAC fits. Input transforms and prior fit-scope
findings remain declared; execution is not certification of the shared release.

## Operations And Test Design

Use only the large approved GPU, resolved by its observed name then pinned UUID,
not a guessed numeric index. Only one heavy child; retain the existing atomic
admission launcher and4GiB hard per-job request. Do not lower its cap, change
kernel settings, terminate another owner's process or use the second GPU.

| ID | Acceptance test |
| --- | --- |
| OP01 | Exactly12 jobs in supplied order, correct lists/splits/state/regularization |
| OP02 | GPU discovery rejects absent/ambiguous large GPU; stable UUID in child env |
| OP03 | Verify archive, code, data, lists and frozen config before launch/adoption |
| OP04 | Exit0 or a shell DONE marker alone is insufficient: validate all3 seeds,4 modules, metrics/scaler/probe |
| OP05 | Failed and interrupted attempts keep their artifacts; completed authenticated jobs never refit |
| OP06 | Resource refusals wait automatically; bounded entered retries, no cap/config alteration |
| OP07 | Persistent service+timer resumes after reboot; lock and live-child check prevent duplication |
| OP08 | A failed independent arm does not suppress later jobs; final closure distinguishes success/failures |
| OP09 | Per-attempt logs, heartbeat, command/input hashes and terminal output manifest |
| OP10 | No ranking/promotion: export native TRAIN/inner/2024 metrics and support signatures for peer collection |

Implement only the operations wrapper. Tests run on CPU synthetic artifacts,
then inspect the actual package and do one real start check. Do not wait in an
agent polling loop. Timers own continuation and artifact collection.

## Verification And Deployment

S14_RELEASE_OPERATION, RUNNING (not complete). PRE16 tests failed because the
worker did not exist. POST63 focused tests passed:30 worker/collector/recovery tests and
33 existing shared-queue regressions. They cover synthetic completion, missing
seeds/checkpoints/support, altered receipts, runtime drift, bounded failed arms,
whole-loop closure and adoption after an interrupted completed fit. There was
no physical reboot test and no claim of annual/model acceptance from these tests.

Archive and extracted scientific code agree;122 code/data/list/worker files
are pinned. Runtime package versions and commands are frozen in the private
configuration. Its identity is
4b19a3d12004f6035b4411d5166d08a5a61acb14592d084d3b363708b6b029f4.
Hardware discovery found the approved large GPU at numeric index1; execution
binds its private UUID, not index0 from the suggested shell command.

The supplied shell queue was NOT independently launched. The operations-only
wrapper drives the same scientific entry point and supplied arguments.
Service `musashi-e3m-assigned-20261022.service` is running and timer of the same
stem is enabled. User lingering was already enabled. The service resumes on
interruption/boot, waits behind another GPU compute process, and retains the
existing atomic admission launcher with4GiB per-job hard limit. Its own small
operations cgroup has a separate256MiB limit; the admitted science runs in the
launcher's4GiB scope, not under the256MiB operations limit.

The pre-start recovery helper handles a partial native E3M.json after an
interrupted final write: only an owned RUNNING attempt, no live child and no
accepted result receipt can be archived and made retryable. A valid result is
left for adoption; an accepted receipt is never repaired or overwritten. This
avoids an endless parse-error restart without changing the scientific writer.
The helper/service update was loaded without restarting the active fit.

Actual start check: cons_core15, effective fixed_train, consolidated contract,
FULL models, native FIT events for seed20261019 through DAgger round3. Exactly
one compute PID on the approved GPU, about0.8GiB GPU memory at the check; no
compute PID on the other GPU, no worker restarts. This is live training evidence,
not a completed E3M measurement. Host available RAM was about6.6GB after start;
worker disk had about75GB free before launch. No memory/kernel settings changed.

## Unattended Outputs And Handoff

Paths below are relative to the existing private package root:

- `ops/STATUS.json`, `JOBS.json`, `PLAN.json`, per-launch logs/receipts: progress,
  attempts, commands and causes; heartbeat is written by the worker itself.
- `out/<arm>/E3M.json`, `SCALER.json`, `PROBE.json`,12 native checkpoints per arm:
  scientific artifacts, unchanged formats for Satoshi's collector.
- `ops/<arm>.RESULT.json`: validated three-seed completion, exact requested and
  consumed columns, support hashes, native metrics and15 terminal file hashes.
- `CONSUMERS.json`: wrapper export without cross-arm ranking/promotion. The
  pinned scientific program's existing within-arm arbiter selection is unchanged.
- `ops/CLOSE.json`: exact done/failed/pending counts; failures cannot produce an
  all-success marker. Completed authenticated arms are adopted, not refitted.
  Partial entered attempts remain separate; retries never overwrite their files.

The independent collector timer `musashi-e3m-collector-20261022.timer` runs every
five minutes on the coordinator and after boot. Bulk copies go to the configured
external experiment-log volume, not home.
The collector refuses an absent/read-only volume, a destination escaping that
mount or less than1GiB free; it never creates a fallback tree on the root disk.
It checks every sealed terminal file
hash and the final consumer-export digest before writing MIRROR_CLOSE. Initial
collection succeeded, with zero sealed jobs yet. Both timers disable themselves
on their respective terminal closure. No agent polling loop is required.

The mirror is physical custody, NOT warehouse publication. Satoshi owns the
consumer table/audit integration and existing downstream publication; no new
warehouse destination or credential was invented. Split-support differences
must be reported rather than assumed equal because a queue was completed.

## Scope Limits

All12 assigned arms are development diagnostics on the pinned v3 release. Prior
upstream transform-fit findings stay declared and are not fixed or certified by
running this package. Negative/invalid trading policies still count as executed
when their native reports are complete. A real integrity drift blocks launching
changed science; an independent entered failure gets its disposition while the
remaining arms proceed. Native SAC's no-feedback baseline stays unlaunched.
