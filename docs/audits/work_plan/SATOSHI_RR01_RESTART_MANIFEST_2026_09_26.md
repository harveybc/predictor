# RR01 — restart manifest after the coordinator reboot

**Date:** 2026-09-26
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Continues:** DR01–DR08, under the post-reboot orders RR01–RR07 (`0d6f6f7e`).
**This is not a new assignment and not a blanket restart.** No new training allocation, reserve access,
real-capital authority or M4 approval is created here. `NO_NEW_MEASUREMENT`.

---

## 1. What happened, established from the previous boot's journal

The coordinator rebooted at **17:40:56**. It did not crash on its own: at **16:27** `systemd-oomd`
killed the owner's browser and then the owner's editor, and the session went with them.

| at | unit | event | that unit's own peak |
|---|---|---|---|
| 16:27:10 | browser | oomd killed processes in the unit | 10.8 G, plus 1.7 G swap |
| 16:27:49 | editor | oomd killed processes in the unit | 13.5 G, plus 2.2 G swap |
| 16:27:49 | my verification attempt 1 | scope ended | 2.9 G, inside its 3 G cap |
| 16:27:49 | my verification attempt 2 | scope ended | 2.8 G, inside its 3 G cap |

**Attribution, kept honest in both directions.** The two killed applications were themselves the largest
memory holders on the host. My two verifications held **5.7 G of observed peak concurrently**, each inside
its declared cap, under a 3 G desktop reserve. What is **not** established is that either attempt alone
would have caused the kills, or that no kill would have happened without them; neither is measured and
neither is claimed. What **is** established is that admission refused further work correctly while
pressure climbed — the ledger carries refusals at PSI some/avg10 **38.76, 60.48, 61.20** against the 25.00
limit — and that **the two already-admitted scopes ran to the end regardless**, because admission is a
gate at entry with no monitor after it.

The order's own reading is the right one: the desktop kills happened **after** the atomic launcher was
deployed. Deployment closed double admission at the gate; it did not make an admitted workload
answerable to rising pressure.

## 2. The two interrupted attempts — results NOT retained

Full recovered records: `docs/audits/evidence/RR01_RESTART_20260926/INTERRUPTED_ATTEMPTS.json`.

| | attempt 1 | attempt 2 |
|---|---|---|
| unit | `crispdm-dr01deploy-1790457010-724279.scope` | `crispdm-dr01deploy-1790457351-780621.scope` |
| started | 16:10:40 | 16:20:21 |
| ended | 16:27:49 | 16:27:49 |
| CPU | 20 min 27.957 s | 12 min 49.094 s |
| wall | 17 min 9.496 s | 7 min 28.214 s |
| scope peak | 2.9 G | 2.8 G |
| declared cap | 3 GiB | 3 GiB |
| **outcome** | **`INTERRUPTED_RESULT_NOT_RETAINED`** | **`INTERRUPTED_RESULT_NOT_RETAINED`** |

Both ran a pytest selection over the admission battery and four sealed-runner batteries. **Neither is
PASSED.** Their output went to the orchestrating agent's pipes, which did not survive the session, so no
exit code and no suite counts exist. Published focused results from other commands stay scoped to their
own command and revision and do **not** cover these selections.

**Cumulative cost spent and not recovered: 1997.051 CPU s over 1477.71 s wall.** It is not re-budgeted and
it is not reset because a process died.

**One thing I preserved just in time.** The admission store **deletes a lease body when it reclaims the
lease**, keeping only a ledger line with the lease id. I read both bodies minutes before the reclaim
removed them, so cap, cgroup, argv digest, wall and expiry survive only because they are transcribed in the
evidence file above. An old-boot lease is supposed to be preserved as a historical record; today it was not.

## 3. Live state at the time of writing, read before any launch

| what | reading |
|---|---|
| coordinator boot | 17:40:56, boot id `ae653632…` |
| host memory | 30 GiB total, 23 GiB available |
| host memory pressure | some avg10 **0.00**, full avg10 **0.00** |
| `crispdm-batch.slice` | `memory.current` 0, ceiling 15032385536 |
| live admission leases | **0** |
| leases reclaimed on restart | **2**, both `LEASE_RECLAIMED_WITNESS_DEAD` |
| held unrealised bytes | 0 |
| heavy Python alive | none |
| services running | the five governed stores, both governance tunnels, the OLAP loader, memguard |
| worker reboot | **not implied.** A coordinator reboot says nothing about either worker; each is checked on its own before use. |

**No duplicate executor was started.** There was nothing to adopt: no crispdm scope and no heavy process
survived.

## 4. Published tips reconciled — nothing was lost to the reboot

Every branch of the campaign is on the remote and matches a worktree or a retained attempt. Recovery reads
the **published** Huber design at `76650ce8`; the disk was not searched again and the deleted temporary
checkout is not depended on. **No history was pruned or rewritten as recovery.**

| repository | branch | tip |
|---|---|---|
| predictor | `satoshi/rp49-rp64-disposition-20260926` | `a84a913c` |
| predictor | `satoshi/rp59-lag-table-restatement-20260926` | `6820fcae` |
| predictor | `satoshi/huber-design-recovery-20260926` | `76650ce8` |
| predictor | `satoshi/m4-confirmation-execution-20260926` | `b8865058` |
| predictor | `satoshi/e1-seal-q2-context-20260926` | `7d2b1c83` |
| predictor | `satoshi/q2-context-deep-arms-20260926` | `7b0248f8` |
| predictor | `satoshi/core-pretrain-resolution-20260926` | `4168ebdc` |
| predictor | `satoshi/defect-repairs-and-fred-20260926` | `c74f910b` |
| predictor | `satoshi/dr01-atomic-admission-20260926` | `ee30935a` |
| predictor | `satoshi/dr01-followon-deploy-20260926` | `f66ee25d` |
| predictor | `satoshi/dr02-dispatch-index-20260926` | `b790cd12` |
| predictor | `satoshi/dr05-dr06-corrections-20260926` | `c2d4388b` |
| predictor | `satoshi/dr05-confirmatory-candidate-20260926` | `86a19936` |
| predictor | `satoshi/musashi-audit-request-20260926` | `cf2fdb0c` |
| predictor | `satoshi/remove-committed-lock-files-20260926` | `9fc79367` |
| agent-multi | `satoshi/m4-confirmation-exec-body-20260926` | `0de54534` |
| agent-multi | `satoshi/dr04-m4-verifiability-20260926` | `4b009c35` |
| lts | `satoshi/mt5-unknown-outcome-20260926` | `12bce5f` |
| lts | `satoshi/mt5-five-failures-forensics-20260926` | `f1109ba` |
| lts | `satoshi/mt5-corrections-20260926` | `d631f4a` |
| data-gov | `satoshi/calendar-data-gov-20260926` | `7eec868` |
| data-gov | `satoshi/defect-repairs-and-fred-20260926` | `8a5d2f9` |

**The interrupted follow-on landed more than the interruption suggested.** `f66ee25d` carries five commits,
a return document, and a clean tree: the launcher deployed to both worker roles with remote dispatch proven
to reserve, four of the eight fit runners refusing a bare invocation, and the other four deferred with the
seal that pins each one named. Its own residual is operational verification, not implementation.

## 5. Per-lane manifest

| lane | producer tip | attempt | completed | partial | accepted evidence | residual allocation | next action |
|---|---|---|---|---|---|---|---|
| admission (DR01) | `ee30935a` | deployed | gate closed at entry; PRE/POST proof | — | PRE/POST, incident register, 30 tests | none | RR02: post-admission monitoring, boot identity, lease-body retention |
| admission follow-on | `f66ee25d` | interrupted at 16:27:49 | worker deployment proven; 4 of 8 runners gated | the two pytest selections | its return document | none | re-run the two selections in bounded sequential shards **on an admitted worker** |
| M4 verifiability (DR04) | `4b009c35` | delivered | 6 repairs, PRE reproduced the auditor's p-values through the public path | — | 153 tests | none | `DELIVERED_FOR_EXTERNAL_REVIEW` — the auditor owns the verdict; I own supplementary evidence |
| M4 execution | `b8865058` + body `0de54534` | executed | verdict `NO_NEW_MEASUREMENT`, 0 of 3024 | — | census re-derived 28/28 | none | nothing until protocol, implementation and reviewed record correspond |
| dispositions | `a84a913c` | delivered | 60 VERIFIED / 9 REFUTED | — | audit JSON | none | auditor's review; the scrambled-label block is withdrawn |
| resolution | `4168ebdc` | delivered | resolution 0.031106 kW; module effect 0.009805 kW below it | — | RESOLUTION.json | none | **input for review, not authority to launch 69 fits** |
| lags / prefix (DR05/06) | `c2d4388b` | delivered | lag table on the clock; prefix output materialized, 4 proofs | — | two evidence JSON | none | name the remaining dataset/delivery/consumer requirements |
| Q2 context | v1 historical; v2 sealed `7b0248f8` | 12 of 18 cells | — | 6 cells never fitted | unanchored table, 0 verified | **not relaunched** | governed delivery path on a bounded mechanical unit first |
| calendar / FRED | `7eec868`, `8a5d2f9` | delivered | 5 + 9 registered | — | catalog replay | none | 39 producer call sites (host-name) |
| MT5 | `12bce5f` → `f1109ba` → `d631f4a` | published | third outcome, forensics, corrections | — | read-only forensics | none | **published, not demonstrated deployed**; unknown outcomes keep their disposition |
| host-name | `9fc79367`, `7b0248f8` | delivered | branch clean; lock file removed | — | inventory | none | producer call sites, then the 126 free files |
| confirmatory candidate | `86a19936` | drafted | estimands declared | — | — | none | auditor's review |

## 6. What I am not doing, deliberately

- **No new heavy compute or combined ML suite on the coordinator.** It is the interactive workstation and
  the service host, and it has already lost the owner's browser and editor once today.
- **No repeat of any scientific fit to test software.** A bounded synthetic child is enough for an
  integration smoke, and no host OOM is provoked deliberately.
- **No blanket retraining, and no budget reset because a process died.**
- **No relaunch of the refused W1440 attempt and none of Q2 v1.**
- **No auditor signature**, in any form.

## 7. Next action, concrete and owned

1. Finish admission where it actually failed: a monitor after admission with hysteresis over a window
   derived from retained observations, boot identity plus process start identity in the lease, and lease
   bodies preserved when reclaimed. Validated on simulated `/proc` and clocks, never by exhausting the
   owner's desktop.
2. Re-run only the two interrupted pytest selections, in bounded sequential shards, on an admitted worker,
   retaining exit code, revision, selection and resources.
3. Continue the product lane and the eligible experiment lane in parallel, on workers, with the external
   RTX 5090 first for eligible GPU work.

— Satoshi
