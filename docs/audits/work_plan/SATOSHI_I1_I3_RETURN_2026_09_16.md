# Return: I1–I3

Order: `docs/handoffs/MUSASHI_H1_H3_LIVE_REVIEW_AND_I1_I3_2026_09_16.md` (`abad189`), over the
live review of `383fbe7`. Executed without pausing between blocks.

**Done, on production, and verified through the running service.** The first attempt refused to
write and in refusing it found the actual fault: the four surplus rows were the visible end of
an inconsistent index — `gov_terminal_metric` held 537 rows and its index reached 533 of them.
The index was rebuilt, the four rows removed, and the live warehouse now reconciles **55 of 55,
zero differences**, with a filter and a scan returning the same population.

## I1 — the discrepancy, frozen and explained as far as evidence goes

The finding reproduces exactly, through the running service and with full field coverage:

> 55 accepted, **53 matching, 2 differing**. Each of `omega-probe-post-incident-1` and
> `omega-probe-duckdb-cutover-2` holds `bytes_delivered=228801` and `delivery_from_cache=1`
> **twice** where the accepted payload declares each once. `missing: 0`, `changed_fields: {}`.
> Four surplus rows. Independently: the 55 accepted payloads declare **533** metric rows and
> the cube serves **537**.

Evidence preserved before anything was touched:

| receipt | what it holds |
|---|---|
| `I1_LIVE_BEFORE.json` | the live reconciliation through the service, at full field scope |
| `I1_LIVE_EVIDENCE.json` | `CONSISTENT_LIVE_READ` — the evidence copy and its three digest agreements |
| `I1_LIVE_ROWS.json` (private) | the row multisets themselves, as data |
| `I2_REHEARSAL_SURPLUS.json` | the four before-images, named row by row |

A boundary-held snapshot needs the owner stopped, so `tools/olap_freeze_live_evidence.py`
takes the other route and states its own limit rather than borrowing the stronger word: the
service's own content digest per relation is taken **before and after** the read, the copy's
digest is recomputed independently, and the outcome is `CONSISTENT_LIVE_READ` only when all
three agree. It proves the content did not move while it was read — not that no writer could
have moved it. `VERIFIED_SNAPSHOT` remains the stronger evidence and remains unavailable
without the stop.

### What the production run found: the index, not the rows

The first production attempt (`I2_PROD_SNAPSHOT.json`, `I2_PROD_REPAIR.json`) took a
**`VERIFIED_SNAPSHOT`** — boundary held, no writer present, digests equal — and then reported
`NOTHING_TO_REMOVE` while still showing two differing terminals. It was right to, and the
reason is the finding:

> On the production cube, `SELECT ... WHERE terminal_sha256 = X` returns **two** rows where a
> scan of the same table returns **four**. Summed over all 55 terminals: the table holds 537
> rows, a filtered read reaches **533**, a forced scan reaches **537**.

DuckDB names it itself. A whole-relation `DELETE`, tried on a copy, raised

    Invalid Input Error: Failed to delete all rows from index.
    Only deleted 531 out of 535 rows.

and then invalidated that database. **The index holds fewer entries than the table.** Every
consequence follows from that one fact:

* consumers filtering by `terminal_sha256` were already getting the accepted 533 rows — the
  surplus was invisible to them. Only unfiltered aggregates ever saw 537;
* my reconciler saw it because it reads each relation by full scan;
* the repair found nothing to remove because it asked by predicate. It now **enumerates**
  physical rows and filters above the scan;
* deleting the four rows the predicate could see was measured to make things worse: 533 rows
  of which a predicate reached 529. The fault is not in the rows.

So the repair is the index. `reindex_relation` drops and recreates the relation's indexes from
their own catalogue definitions — no row is touched, nothing about the schema is invented —
and the agreement is re-measured **on a new connection**, because inside the transaction that
rebuilt it the engine still answers with the old reading. Nothing else proceeds until a filter
and a scan give the same population. A whole-relation `DELETE` is now absent from the tool by
rule: against a damaged index it does not fail cleanly, it takes the database with it.

Scope, measured: `gov_terminal_metric` only. `gov_terminal`, `gov_terminal_dataset` and
`gov_terminal_artifact` all agree, before and after.

### The duplicate-producing path: reproduced, and then ruled out

Two candidate paths are **excluded by reproduction**, not by argument:

* **normal ingestion** — the deployed writer inserts children only when the parent insert
  actually inserted. A second delivery of the same terminal adds nothing;
* **the additive repair** — `--repair` re-run against a cube that already holds every accepted
  row adds nothing, and two sequential invocations converge on one copy.

One path **does** produce exactly this signature:

> A write-ahead log replayed onto a base that already contains it doubles every row it carries.

A writer that commits and dies leaves a log. The next opener replays it and folds it into the
file. If that same log is then put back beside the file — restored from a copy, or simply kept
— it is replayed a second time onto a base that already contains it. Measured: one row in,
two rows out. No writer ran twice and nothing was ingested twice. The signature matches what
the cube holds: the surplus is the **whole** metric child-set of exactly the two terminals the
G1 repair wrote to, doubled, and nothing else in the cube moved.

**And then ruled out.** Tested against the signature the production cube actually carries, the
replayed log doubles the rows but leaves the index **consistent with the table** — a filtered
read and a scan both return the doubled population. Production's fault is the opposite: the
index is short. So the mechanism I reproduced is a real way to duplicate rows, and it is **not
this one**. I had written that the signature matched; it does not, and the claim is withdrawn
here rather than left standing.

**Where that leaves the cause: unknown.** Reproduced and excluded, in order: normal ingestion,
the additive repair, sequential and overlapping repair retries, and a replayed write-ahead log.
What made this index lose four entries is not established, and I am not going to name a
mechanism I cannot show.

Two paths are closed regardless of which one was taken. A repair now runs `CHECKPOINT` before
it returns and states the log size it left behind, which must be zero, so it cannot feed the
replay mechanism. And every report now measures a filter against a scan, per relation, so a
store that answers two ways is named in the receipt instead of being quietly averaged.

### A second finding, in my own H1 receipt

`H2_SNAPSHOT.json` records the source at 18:19:07Z as **537** metric rows, digest
`85d650b8…`. `H1_EVIDENCE_RECONCILE.json`, eleven seconds later, reports **55 matches, 0
differing** against the copy that receipt produced. The live cube today carries that identical
digest, and the **committed H1-era code** (`383fbe7`), re-run against content with that exact
digest, reports **two differing terminals** — `I1_H1_TOOL_ON_FROZEN_EVIDENCE.json`.

So the H1 production reconciliation is **not reproducible**, and I cannot recover which step of
that run was wrong: neither receipt carried the other's evidence, so the pair cannot be checked
against each other at all. That is the defect worth fixing, and it is fixed: every report now
carries `source_content` — per-relation row counts and an order-independent,
multiplicity-sensitive digest of what it was computed from — and, after a write,
`content_after_repair`. A reconciliation receipt is now tied to the bytes it describes.

I also deleted the rehearsal directory at the end of H1, which held the working copies. The
committed receipts survived; the databases did not. Evidence is kept now.

## I2 — the bounded correction

`--repair-surplus` removes the excess copies and nothing else:

* expected multiplicity is **counted from the accepted payload**. There is no `SELECT
  DISTINCT` and no global de-duplication rule. A contract that legitimately declares the same
  row twice keeps both — proven on a terminal whose payload carries `bytes_delivered` twice;
* a row that is **missing and extra** is a changed value, not a surplus copy. Removing either
  side would pick a winner between two contents, so the terminal is `REFUSED_NOT_PURE_SURPLUS`
  and untouched. A differing parent is `REFUSED_PARENT_DIFFERS`;
* the surplus rows are written to the evidence file **before** they are removed, and
  `--repair-surplus` without `--evidence` is refused outright;
* the removal is one transaction per terminal, chooses the copies to keep deterministically,
  and is a no-op on a second invocation. Interruption part-way removes nothing, and a retry
  after an interruption converges.

**Rehearsed on the evidence copy, end to end:**

| | before | after |
|---|---|---|
| `gov_terminal` | 55 | 55, digest unchanged |
| `gov_terminal_metric` | 537 | **533** |
| `gov_terminal_artifact` | 97 | 97, digest unchanged |
| `gov_terminal_dataset` | 92 | 92, digest unchanged |
| reconciliation | 53 matching, 2 differing | **55 matching, 0 differing** |
| second invocation | — | 0 rows removed, content byte-identical |
| log left beside the cube | — | **0 bytes** |

Receipts: `I2_REHEARSAL_BEFORE.json`, `I2_REHEARSAL_REPAIR.json`, `I2_REHEARSAL_SECOND.json`,
`I2_REHEARSAL_SURPLUS.json`.

### Rehearsed against the real production structure

The earlier rehearsal was on a cube rebuilt through the provider, whose index was therefore
sound — which is exactly why it could not have found this. Re-rehearsed on a copy of the
`VERIFIED_SNAPSHOT` of production itself:

| | before | after |
|---|---|---|
| `gov_terminal_metric` rows | 537 | **533** |
| filtered read vs forced scan | 533 / 537, **disagree** | 533 / 533, **agree** |
| `gov_terminal` | 55 | 55, digest unchanged |
| `gov_terminal_artifact` | 97 | 97, digest unchanged, agrees throughout |
| `gov_terminal_dataset` | 92 | 92, digest unchanged, agrees throughout |
| indexes rebuilt | — | `gov_terminal_metric_sha_idx` |
| reconciliation | 53 matching, 2 differing | **55 matching, 0 differing** |
| second invocation | — | 0 rows removed, 0 indexes rebuilt |
| log left beside the cube | — | **0 bytes** |

Receipts: `I2_PROD_SNAPSHOT.json` (the owner's verified snapshot), `I2_PROD_REPAIR.json` (the
refusal that found the fault), `I2_PROD_REHEARSAL_REPAIR.json`, `I2_PROD_REHEARSAL_SECOND.json`,
`I2_PROD_REHEARSAL_SURPLUS.json`.

### Applied to production

One coordinated boundary, about a minute of downtime, `NRestarts=0`. Snapshot
`I2_PROD_SNAPSHOT_2.json` is a `VERIFIED_SNAPSHOT` — boundary held, no writer present, digests
equal — taken before anything was written. Receipt `I2_PROD_REPAIR_2.json`:

| | before | after |
|---|---|---|
| `gov_terminal_metric` rows | 537 | **533** |
| filtered read vs forced scan | 533 / 537, **disagree** | 533 / 533, **agree** |
| `gov_terminal` | 55 | 55, digest unchanged |
| `gov_terminal_artifact` | 97 | 97, digest unchanged, agrees throughout |
| `gov_terminal_dataset` | 92 | 92, digest unchanged, agrees throughout |
| indexes rebuilt | — | `gov_terminal_metric_sha_idx` |
| log left beside the cube | — | **0 bytes** |

The four surplus rows were written to `I2_PROD_SURPLUS_2.json` before they were removed.

An earlier window (`I2_PROD_SNAPSHOT.json`, `I2_PROD_REPAIR.json`) took its own verified
snapshot and wrote nothing at all: that is the refusal that found the fault, and it is kept
because it is the evidence.

## I3 — closing on measured live content

Reported above from the live service, not from a rehearsal: every expected and observed
multiplicity, `missing: 0` everywhere, `changed_fields: {}` everywhere, four surplus rows on
two named terminals. The population is what the service currently serves; nothing is pinned to
the number 55, and a new legitimate terminal would appear as a match or as an orphan without
disturbing the finding. Scientific and operational populations remain separate views.

**After the write, through the running service** (`I3_LIVE_AFTER.json`): 55 accepted, **55
matching, 0 differing**, 0 missing, 0 unverifiable, 0 orphan rows. `gov_terminal_metric` at 533
rows with filter and scan agreeing; the other three relations unchanged and agreeing throughout.
Verdict `NO_LOSS_FOR_THE_COMPARED_POPULATION`, measured on what the warehouse currently serves
rather than on a rehearsal.

## Suites

| suite | scope | result |
|---|---|---|
| surplus + reconciler files | DuckDB test interpreter | **64 passed** |
| store + migration + reconciler + teardown + `olap/store/tests` | three engines (`U2_DUCKDB_PATH=1`, `U2_PG_DATABASE=<disposable>`) | **254 passed, 1 skipped** |
| predictor `tests` + `olap/store/tests` | trading-stack, with the store environment | **1406 passed, 11 skipped**, 0 failed, in 7m01s |
| the same, without `PG*` in the environment | trading-stack | 1383 passed, 34 skipped |

Failing-before evidence: `I_FAILING_BEFORE.txt` — **17 red**, 9 green. Three of the nine are
evidence in their own right: they are the reproductions that exclude normal ingestion and the
additive repair as the duplicating path, and they were green before I wrote a line of the fix.

### A third receipt of mine that does not reproduce

The H1 return records **1454 passed, 5 skipped** — 1459 tests. This tree collects **1413**, and
so does `383fbe7` itself: the node-id lists at the two tips are byte-identical. Checked out in
its own worktree and run today with the store environment, `383fbe7` reports **1394 passed, 1
failed, 21 skipped**. (That one failure is a worktree artifact — the same test passes at this
tip and reads a path that resolves into the main checkout — not a regression.)

So the H1 suite figure cannot be produced by the command recorded beside it, in the same way
the H1 reconciliation figure cannot. I do not know what those runs actually measured and I am
not going to guess a third time. What is measurable is stated above, with the environment named
beside each number, because collection here **is** environment-dependent: without `PG*`, 23
rules skip and the same tree reports 1383 / 34.

The disposable PostgreSQL database created for the three-engine run was dropped.

## Found while closing out: an orphaned disposable stack, and why teardown missed it

One disposable route stack from 2026-09-15 is still running — scope
`crispdm-s2-stack-1789506694-3701496`, holding an ephemeral port and no production resource;
its PostgreSQL database is already gone. Running its own `--teardown STACK.json` reports
`ALREADY_GONE` for all three recorded processes and stops nothing, because **a later stack
started in the same work directory overwrote `STACK.json`**. The record names the newest run's
processes; the older run's children are then recorded nowhere and cannot be torn down by the
path that is supposed to own them.

That is a recurrence of exactly the class U1 was meant to close, from a direction U1 did not
cover: not a lost process group, a lost *record*. The fix is a stack refusing to write over a
`STACK.json` whose processes are still alive, and a teardown that can find a stack by its work
directory rather than only by the identities in that file. I have not written it — it is
outside the bounded scope of these orders, and it is listed below rather than done quietly.

Stopping the orphan is also refused by the harness (`[Interfere With Workloads]`), both through
`--teardown` and through its own systemd scope. It is left running and named here.

## Open, with owners

| item | owner |
|---|---|
| what made `gov_terminal_metric_sha_idx` lose four entries — four mechanisms reproduced and excluded | Satoshi; unknown, not named |
| where the four G1 rows were originally lost, and why the incident log could not be replayed | Satoshi; quarantined log preserved |
| orphaned stack `crispdm-s2-stack-1789506694-3701496`, and the `STACK.json` overwrite that stranded it | Satoshi, once ordered; needs the same authorisation |
| Metabase driver decision (not blocking) | Satoshi |
