# The archive contract, resolvable after the producer is gone (S2)

Run on omega, 2026-09-15, on a **disposable PostgreSQL** database and a stack of candidate
packages on free ports. No production database was opened by the stack: the warehouse host is
the only process given `PG*`, and only for the throwaway database the run names. A guard in
`tools/disposable_route_stack.py` refuses a protected database name outright.

## What was missing, and what closes it

R4 proved that `UNKNOWN` reached the cube **by reference**: `gov_terminal_dataset` carried the
contract digest `62965feb…`, and nothing more. I reported the limit myself — reading the cube
alone, nobody could say the delivery was a retrospective archive without asking the producer.
A producer is exactly what may not exist when someone reads the cube later.

S2 adds a digest-keyed dimension, `gov_availability_contract`, and the view
`gov_delivery_availability`. The canonical bytes now travel the whole chain: the lake provider
publishes them, the lake host sends them as `X-Availability-Contract`, data-gov **verifies them
against the digest it is about to record** and retains them, and at terminal time it hands them
to the warehouse as a **separate call** — never inside the terminal body, because a terminal's
identity is the digest of its own body and putting contracts there would change the identity of
every receipt ever written, including the historical ones.

## Phase A — delivery, teardown, and a reader with only the database left

`S2_PHASE_A_PERSISTENCE.json`. No holdout declared.

| case | campaign | download | terminal | outcome |
|---|---|---|---|---|
| whole archive | 201 | **200** | 201 | delivered, closed, reconciliation empty |
| ranged archive | 201 | **422** | 201 | refused, naming the retrospective archive |
| point-in-time over the archive | 201 | **422** | 201 | same refusal |
| live tail over the archive | 201 | **422** | 201 | same refusal |
| plain synthetic resource, whole | 201 | **200** | 201 | regression: unaffected |

Every case is **closed**. A refusal is reported as a `REFUSED` terminal rather than an
abandoned campaign; six campaigns were opened and six were closed.

Then, before anything was read: the lake host was killed (confirmed dead — `/healthz` raised
`URLError`), and its configuration file was deleted (digest recorded first,
`config_present_after: false`).

A **fresh reader** — a new provider instance over the same database, with no data-gov, no
lake and no stack — then answered:

| delivery | resolution | use_class | completion_lag_max | digest re-verified |
|---|---|---|---|---|
| the archive | **RESOLVED** | `ARCHIVE_RETROSPECTIVE` | **`UNKNOWN`** | yes |
| the plain resource | **RESOLVED** | `OFFLINE_DAY_GRANULAR` | `4h` | yes |
| a delivery nobody recorded | `NO_SUCH_DELIVERY` | — | — | — |

"Digest re-verified" means the reader hashed the retained bytes and got back the digest the
delivery row references. It is not trusting the row; it is checking it.

The retained rows, from the database itself (`PG_RETAINED_CONTRACTS.txt`):

    62965feb…  ARCHIVE_RETROSPECTIVE  UNKNOWN  WINDOW_START  UNKNOWN            sha256  json.sort_keys.separators-comma-colon.ascii.v1
    41e67f99…  OFFLINE_DAY_GRANULAR   4h       WINDOW_START  PRODUCER_STATEMENT sha256  json.sort_keys.separators-comma-colon.ascii.v1

`62965feb…` is the same digest R4 recorded. It is now answerable.

## Phase B — the holdout, and a refusal I did not predict

`S2_PHASE_B_HOLDOUT.json`, holdout `2026-01-01`, declared on the lake and mirrored as
`deny_from` in the policy (data-gov refuses to start otherwise, which is itself a good sign).

| case | outcome |
|---|---|
| whole archive | **403 holdout** |
| plain range reaching the holdout | **403 holdout** |
| ranged / point-in-time archive | 422, the archive refusal, as in phase A |

The first row was not what I expected and it is the honest result: **a retrospective archive is
refused under any declared holdout.** Its publication time was never observed, so it cannot be
shown to lie before the boundary, and the governance declines rather than assume. That is the
"incompatible holdout" case, arrived at by measurement rather than by construction. It also
means the persistence proof needs a stack without a holdout, which is why there are two phases
instead of one — an earlier single-phase run had the archive refused at 403 and would have
reported a delivery that never happened.

## What stays unresolved on purpose

A delivery whose contract was never retained resolves as **UNRESOLVED** with `use_class` and
`completion_lag_max` **null** — not zero, not guessed. That is the state of every historical
row, and the migration does not backfill them: nothing retained supports it. The first run of
this probe, before the lake host published the contract, produced exactly that outcome for
both deliveries, which is how the missing seam was found.

## Migration

Additive and idempotent: one new table, one new view, no column, row, index or terminal digest
altered. Re-running the schema step over a populated database leaves existing rows untouched
(`test_the_migration_is_additive_and_repeatable`). **Not deployed.** The packaged provider
declares the divergence from the deployed revision in `PENDING_REVIEW` rather than moving its
pin, so nothing here can be mistaken for something running in production.
