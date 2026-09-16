# The outage, accounted for (U1)

## The interval, from the service's own journal

    stopped   2026-09-15T16:13:15-05:00   (SIGTERM, from my own pkill; see the S1-S4 return)
    started   2026-09-15T19:11:24-05:00   (by Musashi)
    duration  2h 58m 09s

## What the outage actually touched: nothing

Eleven outbox directories on this host, 97 envelopes, all inspected
(`U1_OUTBOX_BEFORE.json`). **Zero** envelopes were written or last modified inside the outage
interval. Independently, `journalctl --user -u crispdm-data-gov.service` over the same window
contains exactly three entries, all `GET /healthz` — my own probes. No governed write was
attempted while the warehouse was down.

That is the measurement. It is deliberately not phrased as "nothing was lost because an outbox
exists", which is the claim the order forbids and which I had made.

## The one pending envelope — which predates the outage

| field | value |
|---|---|
| file | `.cache/data-gov-outbox/pending/f9cb1afe….json` |
| written | 2026-09-15T12:04:27Z — **nine hours before** the outage |
| campaign | `e199058b6353…` unit `omega-probe-1`, generation 1, COMPLETED |
| outbox's own class | `NOT_YET_SENT`, attempts 0 |

Its slot was **already closed in the cube**: `gov_terminal` held `omega-probe-1 | COMPLETED |
1` with `received_at 2026-09-15T12:04:27.337162Z`. So it was a stale marker, not a lost
terminal — the sender did not record its own success. The two are different and were reported
as different.

## Reconciled through the existing implementation

`tools/governed_exec.py --flush` against that outbox, unmodified:

    {"failures": {}, "pending": 0, "sent": 1}

Nothing was synthesised and nothing was dropped.

| check | before | after |
|---|---|---|
| pending envelopes, all roots | 1 | **0** |
| rows for that campaign in `gov_terminal` | 1 | **1** |
| that row's `received_at` | 12:04:27.337162Z | **12:04:27.337162Z** |
| terminals in the cube | 52 | **52** |

Duplicate protection held: the re-send created no second row and did not touch the original.
52 matches the count Musashi recorded independently.

## Eighteen envelopes that are not in the production cube, named rather than hidden

`U1_OUTBOX_BEFORE.json` reports 18 envelopes whose `(campaign, unit)` is absent from the
production cube. All 18 are in `sent` (14) or `adjudicated` (4) — **none is pending** — and all
come from `…/flow_v3_deploy_2026_09_13/p*_work/outbox`, the throwaway stacks of the 2026-09-13
deployment work, whose cubes were disposable SQLite files that no longer exist.

They are two days older than the outage and unrelated to it. I am **not** calling them
reconciled: their destination cubes are gone, so they cannot be re-verified at all. They are
recorded here as what they are — envelopes whose acceptance can no longer be checked — rather
than counted as either lost or fine.

## The teardown that caused it, and what replaces it

`pkill -f "data_warehouse_service.main"` matched production because the module name is shared
by design: the store hosts exist precisely so that one module serves any backend. A name can
therefore never be what teardown selects on.

`tools/disposable_route_stack.py --teardown STACK.json` now:

* signals the **process group** each service was started in (`start_new_session=True`), so the
  target set is one this stack owns and nobody else is in;
* refuses to signal any PID whose `/proc/<pid>/cmdline` does not contain that stack's own work
  directory — stale metadata after PID reuse yields `PID_REUSED_REFUSED` and the stranger
  lives;
* waits for and reaps its children, escalating SIGTERM → SIGKILL, and reports per-role
  outcomes;
* treats a **zombie** as gone. It had not: `/proc/<pid>` survives until the parent reaps, so
  teardown waited out the full grace and reported `STILL_RUNNING` for processes that had
  already exited.

Seven rules in `tests/test_disposable_stack_teardown.py`, including two stacks running the
**same module names** where tearing one down leaves the other serving and holding its data, a
stale-metadata refusal, partial setup, and a source check that no `pkill`, `killall` or
`pgrep` exists in the harness. The survivor is a second disposable stack; production is never
the survivor test.

Dogfooded: the U2 route stack was torn down with this mechanism, and all four production
services answered 200 afterwards.
