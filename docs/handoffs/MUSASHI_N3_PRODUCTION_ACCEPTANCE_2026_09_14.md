# N3: production deployment and governed micro-run accepted

Date: 2026-09-14. Reviewer/operator: Musashi, acting on Harvey's explicit
authorization to update the three governance services.

Disposition: **N3_PRODUCTION_MICRORUN_RECONCILED**.
This supersedes the pending deployment status in Satoshi's return at
`182bf89fa0754e3b7b2cea208621531331af931e`; that historical packet is unchanged.
It does not approve D2 scientific conclusions, D3, live use or unreviewed data.

## Deployed identities

Each service was started from a detached checkout of the identity below,
with its existing production store configuration and private environment.

| Service | Port | Executed revision |
|---|---|---|
| data-gov | 5055 | `ff4503a4c7f1d950e47ab43894f176962025faa6` |
| financial-data lake | 5056 | `cc0f15e6ef9cf79c1fb47bd7d7731eb349831008` |
| predictor warehouse adapter | 5057 | `a7a86e906fd5d4e4057237f014f8f95d0cf6ea6a` |
| CPU micro-run consumer | not a service | `182bf89fa0754e3b7b2cea208621531331af931e` |

The first three revisions are published on their repositories' master
branches. The consumer remains the tested research revision; publishing the
warehouse did not merge all research changes into predictor master.

The restart was bounded to those three existing processes. Health checks
returned `200 / ok` for all three. PostgreSQL, Metabase, the OLAP loader,
training campaigns and trading services were not restarted.

## Before deployment

- The new operator interfaces and configuration tests passed in isolation.
- data-gov: **166 passed, 1 skipped**.
- Financial adapter: **60 passed, 1 skipped**.
- Warehouse: **24 passed**, including a real disposable PostgreSQL database.
- Disposable three-service end-to-end flow: passed with exact reconciliation.
- Browser verification: eight desktop/mobile views, no JavaScript errors or
  horizontal overflow; schema, query results and staged settings exercised.
- The accounting database and effective configurations were backed up privately.
  The new PostgreSQL dump was **419,859,109 bytes**.
- Table counts before and after service restart were identical.

Three preparatory attempts stopped before any service restart: two mistakes
in the deployment helper's store-ID field name and one omission of the
warehouse's documented PostgreSQL host defaults. The helper was corrected;
their private logs were retained. No claim of an uninterrupted first attempt
is made.

## Production result

Campaign identity:
`05d28bf90dc06d74a5a0cde8d16d82793e88d89364423ff56ed5436a71b6f17e`.

One fresh-output, CPU-only, two-epoch micro-run completed in **17.095 seconds**.
It was explicitly classified `ARCHIVAL_REPLAY_NON_AUTHORITATIVE`. This tests
transport, reproducibility records and accounting, not model quality.

| Observation | Result |
|---|---|
| Input deliveries verified and recorded | 6 |
| New terminal outcomes | 1, COMPLETED |
| New artifact records | 6 |
| New input-link records | 6 |
| New metric records | 90 |
| Subsequent outbox flushes | 2; each sent 0, pending 0, failures empty |
| Reconciliation | accounting-only 0, warehouse-only 0, missing units 0 |
| Historical non-governed table counts | unchanged |
| OLAP loader | active; same process; zero restarts |

All six inputs carried the deployed N2 scope:
`WINDOW_END`, completion bound `1h`, timezone evidence `UNKNOWN`, and
`OFFLINE_DAY_GRANULAR`. That bounded historical use is not live equivalence.
No missing timezone, availability fact or scientific license was invented.

Total governed tables now include both the prior micro-run and this run:
2 terminal outcomes, 12 artifacts, 12 input links and 180 metrics. The delta,
not the total, is the evidence for this deployment. Old records were retained.

Private evidence contains the deployment configuration hashes, process
identities, backup, run manifest, terminal, logs, before/after counts and
reconciliation. Credentials and local operational paths are not published.

## What opens next

Satoshi may execute the bounded governed D2 re-adjudication and the previously
frozen portability diagnostic under the accompanying order. N3 no longer
requires an action from the owner. The feature-extractor compatibility port
and reusable-host migration remain engineering work, not reasons to delay
the already supported experiment flow.

The operator UI saves pending configuration. It does not hot-apply edits or
establish automatic service recovery after a future host reboot. Those are
separate operational properties and are not claimed by this acceptance.
