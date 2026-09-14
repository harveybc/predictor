# Continuation: the four standing items, executed

Order: `docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.
This is the continuation after the return at predictor `7cc1546`; the activation and the four
production consumers are Musashi's, accepted, and are not repeated here.

Nothing below restarts a service or touches the cube's history. Every campaign is
**NON_GOVERNING**.

## 1. Offline DOIN replay, now in production

The disposable proof was already delivered; this closes it on the live services.

| | |
|---|---|
| campaign | `729c517f…`, unit `doin-offline-replay-prod-3` |
| status | **COMPLETED**, NON_GOVERNING |
| delivery | `synthetic_features_4h_train.csv`, VERIFIED_TRANSFER, contract `41e67f99…` |
| metrics recorded | `wall_seconds 7.11`, `total_timesteps 64` |
| reconciliation | nothing missing, nothing on one side only |
| cube | 33 → 35 terminals across this round; history untouched |

An earlier attempt (`doin-offline-replay-prod-1`) completed with **zero metrics**: the replay
wrote its numbers nested, and the collector reads flattened names, so the cost was reported
as "no metric" instead of being recorded. Both runs are kept — the first is what showed the
defect. The runner now writes the measured numbers where the collector reads them, and the
terminal carries the cost.

## 2. Column roles in the remaining two consumers

The contract predictor enforces is now enforced by feature-eng and preprocessor as well,
against their own loaders: declared columns only, in the declared order; an undeclared column
refused by name; a declared column the file lacks refused by name; a non-numeric feature
stopped before it becomes a number; a run with no contract refused unless it declares the
legacy migration by name; the plan and its digest recorded.

Eleven rules in each repository, passing: feature-eng `e7130f8`, preprocessor `89c83dd`
(branch `satoshi/column-roles-20260914` in both). Verified on the real feature-eng loader
with a production fixture: the contract is applied and an undeclared run is refused.

## 3. The harness gaps (P4)

* **Classification.** The rules rebuilt argparse inside the test and asserted on source text
  — which proves what the test wrote. `governed_run.build_parser()` now exposes the object
  the CLI uses, and the campaign assertion drives `main()` with the HTTP client replaced, so
  what is checked is what the tool submits. 8 rules passing.
* **Route.** The recorder no longer reports `kind/engine/transport` as null when a store is
  absent from the catalogue: a principal may hold `download` without `discover`, so the route
  is marked **not observable from there**, naming the receipts and the operator's active
  configuration as the evidence instead. Neither "local" nor "http proven" is inferred from a
  null.
* **Outbox per wrapper.** Still open: the real pending-recovery test covers the shared
  implementation; the wrappers that share it need their call site demonstrated, and
  agent-multi's new wrapper needs its own run of that test.

## 4. Financial temporal research — the producer's own code answered most of it

The acquisition script is in the repository, so this was answerable without asking anyone:

| question | answer, from `_scripts/workers/stage13_dragon_crypto_worker.py` |
|---|---|
| endpoint | `GET https://api.binance.com/api/v3/klines`, paginated, **historical REST**, not a stream |
| end bound | `END_MS = 2025-12-31T23:59:00Z`, fixed: no in-progress interval beyond it was requested |
| time zone | both time columns are the payload's epoch milliseconds converted with `utc=True` — the provider's encoding, not a column name |
| per-row publication or reception | **none recorded** |
| file-grain reception | the log shows the fetch at `2026-05-01T15:58:48.409Z` paging to 15:58:54, and the declaration records `acquired_at 15:58:56.166Z` |

So **reception is now `BOUNDED_AT_FILE_GRAIN`** with that measured bound — a real fact that
still supports no point-in-time claim — while **publication stays UNOBSERVED**. This does not
explain the 21 anomalous bars: they sit inside the series, not at the end bound, so their
construction remains undemonstrated and they stay excluded from any finality-requiring use.

Two questions remain, and only one is the owner's:

* **revisions**: measurable by re-fetching the same window and comparing. That is an outward
  network call, so it is proposed, not performed — and it would compare two snapshots, never
  establish a policy;
* **usage rights**: the one question this repository cannot answer. The exact action: obtain
  and record the terms applying to market data from `api.binance.com/api/v3/klines` for
  analysis, derived artefacts and publication.

Delivered in **financial-data PR #2** (`8ec5e16`).

## What is still open

| item | owner | next action |
|---|---|---|
| outbox proof per wrapper | Satoshi | run the pending-recovery test through agent-multi's wrapper and demonstrate the shared call site in the others |
| `ARCHIVE_RETROSPECTIVE` end to end | Satoshi | host → data-gov → receipt → cube on a disposable stack, checking UNKNOWN never becomes zero |
| revisions of the ETH resource | Satoshi, on approval | a bounded re-fetch and comparison — an outward call |
| usage rights | the owner | the terms named above |

Services: `:5055/:5056/:5057/:5058` healthy (re-checked at close), zero restarts, loader
advancing; the cube read 35 terminals after the last replay, with its history intact.

Stop: `DOIN_REPLAY_IN_PRODUCTION_COLUMN_ROLES_ACROSS_CONSUMERS_PROVENANCE_ANSWERED_FROM_THE_PRODUCER`.
