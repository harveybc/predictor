# Satoshi: RP136-RP139, cooling restored; recover and finish concurrent experiments

Owner explicitly confirmed all fans/cooling restored on 2026-09-23 after returning
home. No further routine authorization is required. This supersedes travel holds,
not thermal/memory/disk admission or immutable experimental designs. External
RTX 5090 remains first choice; other devices are eligible after fresh admission.

## Live ownership and actual progress

RP132-RP134 repair commits `103629bb` and `e473efbe` are pushed on
`satoshi/rp132-rp134-20260923`. The return, RP134 evidence and state edits in that
worktree are still uncommitted as inspected by Musashi. Do not redo that work.

RP135 now belongs to the native user systemd service
`crispdm-rp135-continuation-20260923.service`, under a real nonblocking flock.
Hermes task `t_28e5c655` is BLOCKED as operationally superseded: its runtime cap
counted from the morning start and repeatedly expired resumed workers. Do not
unblock it or dispatch another RP135 executor. At22:24Z the actual GPU child
PID856355 was training `L512_h96_s2023` on the external5090 (62C,60% use). The first
two L512 H96 cells finished training; ten remain untrained at this handoff.
Both now have accepted terminal receipts. Same-device replay and complete
scientific closure are separate and still required.

Recorded author normalized float32 MSE/MAE (not newly independently recomputed):
2021 = 0.125551 / 0.220453; 2022 = 0.125849 / 0.220934. Table 9 publishes
0.126 / 0.220 but its per-horizon searched lookback is unresolved: NOT an exact
matched comparator and NOT "exact agreement". No three-seed mean yet.

## RP136: deliver the existing repair round and preserve its limits

Finish your existing return and commit/push its currently uncommitted evidence;
refresh the experimental-status section from live receipts, not the morning's
process snapshot. Preserve the separate repair branch. Integrate these orders
without overwriting the concurrently updated master plan or immutable run tree.

The eight legacy regeneration records whose family parameters were not recorded
remain qualified. Matching field tolerances alone does not establish histogram
bins/ranges, quantile lists, MI bins, ACF lags or time-block populations. Establish
missing definitions from bound producer revision/configuration when possible;
otherwise retain the limitation. Do not regenerate all predictions or stop fits
merely to improve a retention certificate. No production prediction deletion in
this round, and no full-numerical claim for missing scientific definitions.

## RP137: finish the admitted L512 queue through its existing executor

The failed path was remote SSH forwarding, not failed data-gov processes.
Musashi installed enabled user-managed `crispdm-governance-tunnel@.service`
instances to the two workers, with reconnect and loopback-only forwarding.
The four existing store/governance services were not restarted. Observe real
authenticated responses, not just LISTEN or HTTP 200 from a public page.

The first resumed delivery exposed a SECOND dependency: the registered SOTA
benchmark lake service was inactive and disabled. Musashi enabled and started
that existing unit with its unchanged package/configuration/data. It is now
active with NRestarts=0. Preserve the failed delivery attempts; a public root-page
check alone did not establish that the complete governed download path worked.
Verify a single actual unit's delivery before submitting the remaining batch.

Cell 2 recovery used `df_sota_repro.py report`, not `execute`: without an accepted
receipt the latter would retrain an existing cell. Receipt and reconciliation
now record COMPLETED. Musashi's report invocation rebuilt timestamps and a
previous pending envelope returned generation conflict. The original terminal
was backed up before recovery. Preserve this incident: accepted metrics/artifact
identity are not evidence that the newly reported work-start timestamp is true.
Use the retained original terminal and run record to issue the existing additive
provenance correction/disposition; do not rewrite warehouse history or erase the
conflicting envelope. This bookkeeping does not block independent remaining fits.

Tell the executor to skip the two accepted cells, recheck its lease/live PIDs and
continue the ten untrained cells on the same external GPU and frozen design.
The native continuation already skipped both accepted cells; Satoshi supervises
that job, not a second copy. Logs are in the user journal for the service and the
existing campaign root. Do not edit the pinned training checkout.
Do not silently move an already sealed cell to another device or change its
model, training allocation, split, scaler, targets or metric reduction.

Continuation allocation, declared BEFORE new continuation fits: **24,000 CPU
seconds and 8 hours wall** for the remaining ten cells and their verification,
with per-child resource limits. This is a new explicitly recorded allocation,
not retroactive permission to exceed the preceding 14,400-second ceiling.
Snapshot prior costs; report cumulative campaign cost and continuation cost
separately. The executor's earlier statement that it could simply exceed a cap
was wrong; thermal/resource admission and budget limits must be enforced.

All twelve cells, including failures and censored/early-stopped training, remain
in the planned population. Verify replays on their actual training device,
recompute author metrics and matched naive, reconcile accepted content, and keep
arrays while the deletion hold remains. No tuning to the public test scores.

## RP138: original-device verification in parallel, no duplicate launch

Musashi's delegated verifier owns the initial bounded original-device checks:
WORKER_A H96 seeds 2021/2022/2023; COORDINATOR H192 seed 2021. Coordinate with
its process/receipt before any retry. These hosts may work while RP135 trains
on WORKER_B. Use their correct original environments and physical device UUIDs.

Update22:27Z: that delegated attempt has finished, no scopes remain. WORKER_A
spent692.70 CPU seconds; retained metrics match the records exactly for all
three cells, but `VAULT_CHANGED` stopped the CLI before GPU replay. Numerical
catalog fields match; the differing field is `metric_implementation_sha256`
(45a8a363... -> 44086f33...). Do not rewrite the bound catalog to make it pass.
Satoshi now owns resolving the implementation-identity transition: preserve
original authority and record a separately bound current verification, then run
the original-device replay without treating the old digest as the new one.
Avoid another eleven-minute catalog recomputation unless consumed evidence
actually changes. COORDINATOR admission refused a34% competing load; no replay
started there. Fresh admission is needed, not another owner cooling approval.
Full local evidence: `~/.local/state/crispdm-data-foundation/original-device-check-20260923/RESULT.json`.
Neither historical path has gained replay acceptance from this attempt.

The original roots retain predictions on these hosts. Use replay/close on those
originals; do not use a deleted-cell regeneration command or delete originals
to satisfy that command. Preserve current report metadata before replacing the
current report. Frozen pointwise tolerances remain; no promise of exact parity.

If replay succeeds, complete the catalog's actual numerical acceptance where
needed, then bind accepted evidence into the authoritative merged root by design,
cell, checkpoint, array and report identities. Never merge by filename alone.
If it fails, preserve measured discrepancies and scope; remaining training keeps
running. No verified four-horizon mean until all population and scoring gates
actually pass. A replay alone is not automatically a full catalog certification.

## RP139: liveness, consolidation and next experiment preparation

The experimental task was reclaimed three times and then spent hours blocked by
an unmonitored forwarding failure. Do not repeat that failure mode. Keep a real
durable supervisor, heartbeat at least every three minutes, single GPU dispatcher
and a lease tied to the actual process, not merely a JSON file saying OWNED.
On stale agent lease, first inspect the training child: attach if alive, do not
launch another copy or kill valid work. Record downtime and its cause.
Do not reuse the expired Hermes task lifetime as a fresh execution allocation.
The native continuation began22:20:06Z with an outer8h runtime and an inner
24000-second wall timeout, threads1 and6GiB host memory. These wall limits alone
do not prove the aggregate CPU allocation: measure the scope CPU ledger, include
verification cost, and refuse further children when the allocation cannot fit.

Forwarding reconnection is transport recovery, not permission to restart shared
stores. Check the restored authenticated path and resume the accepted outbox;
report conflicts with their preserved payload identities, without retraining.

While fits run, finish repairs and prepare the already planned matched doctoral
intervention against the strong reference, plus financial data/reference work.
Separate worktrees and measured host resources; no new filler experiment and no
unannounced change under a running process. Internal WORKER_B GPU shares host RAM
with the external GPU: use it only if combined admission fits and useful work is
ready, not simply because it is visible. No routine "continue" prompts.

One consolidated return leads with actual experimental results, normalized author
metrics, same-row naive, literature scope, completed/remaining cells, costs and
active jobs. Software tests and transport repairs are reported separately. Never
promote the existence of a terminal or a running executor to scientific acceptance.
