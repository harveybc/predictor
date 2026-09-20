# Satoshi — return of RP41–RP48: the real route repaired, the adoption still blocked

Order: [MUSASHI_PROGRAM_RP41_RP48](../../handoffs/MUSASHI_PROGRAM_RP41_RP48_2026_09_19.md), dictum
[MUSASHI_RP33_RP40_REVIEW](MUSASHI_RP33_RP40_REVIEW_2026_09_19.md) (F1–F5). Base `f588400`. Ending
state: `RP41_RP48_INTEGRATION_REPAIRED_ADOPTION_BLOCKED_READY_FOR_REVIEW`.

**The five findings are repaired and the sequence stops at one place.** RP41, RP42, RP43, RP44 and
RP45 are done, with tests that fail when the repair is removed. RP46's adoption — writing the data-gov
configuration and restarting that one service — was refused again by the execution environment's
permission layer. That refusal is not data-gov's, not the owner's and not scientific, and it does not
excuse any defect: every defect the dictum found was fixed before this return. Because the adoption did
not happen, **RP47 was not executed**: the successor pilot runs only through governance, and nothing was
trained on the real panel this round.

## PRE

The dictum's reproducer was run on the reviewed revision before any edit, and its report is field-for-field
identical to the reviewer's frozen outputs: the adoption that leaves a changed configuration with no
receipt, the CLI that returns 0 after a rolled-back adoption, the download-only route, the runner with one
dispatch and zero governance calls, two empty files turning history into GOVERNED, the broker executing
0.005 against 0.001 and 0.008 decided, and the distant-lag counterexample with R² = 1
([`RP41_PRE/`](../evidence/d3_k5_20260917/RP41_PRE/RP40_REVIEW_REPRODUCED_PRE.json)).

## RP41 — the adopter, repaired and provable

Every mutation now happens inside the recovery block; the receipt is written whatever happens; the
rollback restores the configuration, restarts the service, checks its health and the lake's absence, and
records a **failed** rollback as a failure; and the CLI exits non-zero whenever it did not adopt. An
adoption is refused unless a **rehearsal receipt binds to the bytes being adopted**: the deployed
configuration's content, the two panels' digests, the provider package and this checkout's identity.
`tests/test_public_lake_adopt.py` (11 rules) drives all of it on a COPY of the real configuration:
a restart timeout, a non-zero restart, an unhealthy service, a failed post-check, a route that raises,
a rollback that cannot write, a missing rehearsal, a failed rehearsal, a foreign rehearsal, the exit
codes, and the inspection modes that write nothing.

The route is no longer a download probe. It registers the campaign before reading, delivers, verifies,
**closes both of its units** — including a probe that deliberately fails, so no unit is left open —
reconciles the campaign, and reads the terminals back from the warehouse **by content** through its own
query endpoint. The rehearsal runs it on three disposable hosts
([`RP46/REHEARSAL.json`](../evidence/d3_k5_20260917/RP46/REHEARSAL.json)): route complete for both panels,
campaigns reconciled with nothing missing, and the two units present in the cube with the probe recorded
as FAILED.

**The external host, and the divergence I must report.** The order asks for the external data-lake host.
I ran it — the deployed `data_lake_service` with the `financial_data_store` provider — and it **refuses
these panels**: its `governed_download` parses `available_time_column` unconditionally, so the household
labels (`%d/%m/%Y %H:%M:%S`) come back as `unparseable time column`, and under a holdout it refuses the
whole resource as well. That refusal is measured in the rehearsal, not assumed. The correction belongs in
that provider (honour `untimed` in `governed_download`: deliver AS_IS with an UNDECLARED scope and refuse
every range), and the deployed build does not match any branch in this checkout, so I name it rather than
work around it silently. Until it is corrected, the bounded publication uses data-gov's own lake, which
does implement the UNDECLARED scope, and **parity with the external host is not claimed**.

The range refusal now follows the archive's own semantics: the holdout is the **archive's first day**
(2006-12-16, the household panel's own start), so every date-ranged request falls inside the withheld
span and is denied at campaign registration, while the whole resource stays deliverable AS_IS. It is not
a date chosen after the fact.

## RP42 — one governed path, in the runner itself

`tools/df_e1_pilot.run` now acquires **every unit's** campaign and delivery before anything is prepared
or computed, verifies that unit's delivery again at dispatch, and reports its terminal through the outbox
with the reconciliation attached; a terminal that is not accepted stops the run. A cached `DATA.npz` no
longer skips anything: the preparation re-checks that the bytes it was built from are the ones this run
was delivered. Local terminal documents and the `PROPOSAL_NOT_SUBMITTED` path exist only for a run that
is declared ungoverned. The probes run against the **real entry point** with the expensive child replaced
by a deterministic one (`tests/test_df_e1_governed_route.py`, 9 rules): every unit delivered, registered,
reported and reconciled; a unit without its own delivery refused; a receipt from another design refused;
cached data re-checked; a child that really fails still closing its unit; and the transfer/cache reuse
measured per unit.

## RP43 — closure by evidence

`GOVERNED` is now a set of facts read from the receipts' **contents**: this unit's campaign, design,
actor, resource, delivery and chronology, plus the accepted terminal's own payload and a reconciliation
that is complete and successful. Two empty files leave the units HISTORICAL_UNGOVERNED with the reason
recorded, and so do a malformed receipt, an empty unit list, a receipt for another unit or another design,
missing fields, an incomplete reconciliation, a terminal accepted before its delivery, and a terminal
belonging to another campaign (`tests/test_df_e1_close.py`, 28 rules). The divergence between the
verification used at dispatch, on resume and at closing is gone: `run_isolated` applies the closure's own
verdict, so a forged MAE is refused **before** any score is consumed. History was re-closed without a
single fit — 15/15 verified, all HISTORICAL_UNGOVERNED — and the 131 originals are byte-identical to
their freeze ([`RP43_HISTORY_RECLOSED.json`](../evidence/d3_k5_20260917/RP43_HISTORY_RECLOSED.json),
[originals](../evidence/d3_k5_20260917/RP43_ORIGINALS_COMPARED.json)).

## RP44 — information, capacity and optimisation (WORKER_A)

The previous claim is narrowed where it was wrong: position does not decide reachability, and the receiver
tool now says its distant-lag measurement is **empirical, on those series and that budget**. The
constructed claim lives in a new diagnostic whose generator is declared before it runs: the target is an
independent innovation entering the window 50 samples back, split **by realisation**, with a redundant
companion series that echoes the same innovation inside the short support. Run on WORKER_A
([`RP44/INNOVATION_RECOVERY_dragon.json`](../evidence/d3_k5_20260917/RP44/INNOVATION_RECOVERY_dragon.json)):

| receiver / control | recovery task (R²) | redundant task (R²) |
|---|---|---|
| short receiver, reach 7 | −0.12 | **0.82** |
| full-window receiver, reach 60 | 0.75 | 0.81 |
| linear on the whole window | 0.83 | — |
| linear on the last 7 samples | −0.01 | — |
| labels shuffled (control) | −0.13 (worse than the mean) | — |

The short receiver **can** use that innovation when the series carries it inside its support, so its
failure on the recovery task is information, not the optimiser. Budgets are reported as **observed
optimiser iterations** (1 518 for a request of 1 500). The historical errata is recorded and no fit was
repeated to change prose: with 4 000 windows and batch 64 a request of 400 ran 441 updates and one of
1 600 ran 1 638, and those runs saved no counter.

## RP45 — quantity, clock and executions (WORKER_B)

The decided quantity now reaches the broker through the environment's own extension point: a new
execution plugin places `strategy.buy(size=decided units)` and refuses a direction that arrives without a
size. Fills are read from the broker's completed-order events — order reference, executed price, executed
size, commission — and the fill's bar is its **execution instant** (`executed.dt`), not the bar the
notification was seen on. The clock is the data feed's own timestamp, checked strictly increasing and
equal to the row the bar names; an absent or constant counter is refused. The environment's **minimum
executable latency is 2 bars**, measured rather than assumed, so a contract of 1 is refused and a longer
one is produced by holding the order. Pending orders end by the broker's verdict, never by elapsed time.
Partial fills are declared out of scope because this simulator's broker raises on a residual size.
18 rules pass, plus five mutants of the production path; WORKER_B ran the battery on its own host
(26 passed, including the controller's 8).

## RP46 — blocked, exactly

[`RP46/ADOPTION_BLOCKED.json`](../evidence/d3_k5_20260917/RP46/ADOPTION_BLOCKED.json) records the command,
the two operations, the refusing layer and what is ready. Nothing was routed around: no production
configuration was written and no service was started, stopped or restarted by me.

## RP47 — not executed, and why

The successor pilot's prerequisite is that adoption. The runner will not read the panel without a
delivery, and I did not weaken it to proceed. The sealed design, its budget sheet and the governed path
are ready; the run is one approval away.

## RP48 — closure

Tests, budget, hosts and my own defects are in the table below. Plan and state updated at each stage.

## Request

One review of RP41–RP48: the repaired adopter and its recovery, the single governed path in the runner,
the evidence-based closure, the innovation diagnostic's scope, the RL execution contract — and a decision
on the one blocked operation.
