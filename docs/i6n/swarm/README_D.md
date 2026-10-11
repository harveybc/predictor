# Lane D: external archive → chain → OLAP (execution and status)

What lane D owns, how to run it, and how to read its state.

**Status:** code and tests are complete on disposable instances. **Nothing is
deployed.**

- No lake CAS service is running.
- The live warehouse does not have the `doin_0001` relations.
- No DOIN node ingests the reference spool yet.
- The previous `DisposableServiceAdapter` is still not a lake deployment and
  keeps `chain_verified=false`.
- The `external_payload` mode was never active before this lane, and it is
  still not active.

## Components and branches

All sibling repositories use the branch `satoshi/i6n-external-archive-20261011`.

| Repo | Base | What lane D added |
|---|---|---|
| doin-core | `a50a33e` | `doin_core/archive/external.py`: canonical body and manifest, `doin-cas://` locator, reference payload (`external_payload` / `full_payload`), byte verification, and the reference-tx check (the id must be bound to the payload) |
| doin-node | `4c23a6f` | `doin_node/archive/external.py`, which provides:<ul><li>`ExternalPayloadArchive.store_body` (PUT → exact GET → digest) and `resolve`</li><li>`HttpByteStore`</li><li>`build_reference_transaction`</li><li>`verify_chain`, `canonical_references` and `finality_of`</li><li>`ingest_reference_spool`</li></ul>`Chain._validate_block` also gets one *additional* check for reference txs. Nothing was removed. |
| data-lake | `e82b33e` | `FileByteStore`, a durable, immutable, fsynced CAS, under the entry point `cas_file_store`, plus `PUT/GET /api/v1/objects/<sha256>`. It is gated by `byte_upload` / `byte_read` and the bearer token. |
| data-warehouse | `868177d` (= the live wheel) | `POST/GET /api/v2/doin/projections` and `POST /api/v2/doin/projections/revoke`, plus their capabilities |
| predictor | `5c383d17` | <ul><li>`olap/store/.../doin_projection_store.py`: the additive migration `doin_0001` and both providers at 0.1.7</li><li>`tools/i6n_archive/`: records, outbox, publisher, ETL, sinks, snapshot and CLI</li></ul> |

## Data flow

1. **record.** B calls `ArchivePublisher.record_candidate(candidate, fold_receipts)`.
   D validates the documents against the A.2 and E.1 definitions, builds the
   canonical body bytes, and writes the outbox item. The item reaches
   `LOCAL_COMPLETE`.
2. **pump.** D PUTs the body and manifest to the lake, GETs both back, and
   compares the digests. The item reaches `OBJECT_DURABLE`. Only then is the
   reference transaction written to the spool. The node ingests it after it
   re-checks the transaction.
3. **chain.** The block's header commits to the transaction through its merkle
   root. The body digest, manifest digest and block hash are three distinct
   values.
4. **etl.** For each run, the ETL does the following in order:
   1. Verify the chain.
   2. Revoke projections whose blocks are no longer canonical.
   3. For each finalized reference, resolve the bytes and verify them.
   4. Derive the reports from the bytes only.
   5. Project them, then read them back exactly.

   The item reaches `CHAIN_FINALIZED` after step 3 and
   `OLAP_READBACK_VERIFIED` after step 5.
5. **warehouse.** The analytic content is an ordinary `gov_report` /
   `gov_metric` report. It carries no block, mode or locator, so full and
   external payloads produce one report. The typed identity and the DOIN
   metadata live in `gov_doin_projection` and the append-only
   `gov_doin_projection_event`. Read results through the views
   `gov_doin_metric_canonical` and `gov_doin_projection_current`.

## Commands

The test runner puts the worktree sources first on the path and uses one
`crispdm-run` job (2G/120s). Run it once per suite:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=<predictor-D>:<doin-core>/src:<doin-node>/src:<data-lake>:<data-warehouse> \
crispdm-run -m 2G -t 120s -- <venv-with-duckdb>/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_i6n_archive.py olap/store/tests
```

Status and operation run on the coordinator, with a private config that is
never committed. Its shape is in `INTERFACE_D.json` under
`publisher_api.private_config`.

```bash
python -m tools.i6n_archive status --config <private.json>   # read-only, no network
python -m tools.i6n_archive pump   --config <private.json>   # needs DATA_GOV_LAKE_TOKEN in env
python -m tools.i6n_archive etl    --config <private.json>   # needs WAREHOUSE_TOKEN + DATA_GOV_LAKE_TOKEN
python -m tools.i6n_archive.snapshot take --source cas=<dir> --source outbox=<file> --dest <dir>
python -m tools.i6n_archive.snapshot verify --dest <dir> [--restore-to <dir>]
```

Tokens come only from the environment of an existing service's
EnvironmentFile. No D tool reads or prints a token file.

## Reading status

`status` returns the following fields:

- `per_state`: a count for each of the four states.
- `green`: the number of items where all four states are present.
- `missing`: for each item that is not green, the states it lacks.
- `submitted_unfinalized`: references that were handed to the node but are not
  finalized yet.

A full-payload item gets `OBJECT_DURABLE` only when its transaction is
finalized; the evidence is `durable_in: FINALIZED_CHAIN_TRANSACTION`. A reorg
clears `CHAIN_FINALIZED` and `OLAP_READBACK_VERIFIED` on the affected items,
and the history table keeps the `REVOKED_REORG` event.

## Not done here

These are listed in `DEPLOY_STEPS_D.md`:

- Live lake CAS instance.
- Warehouse wheel upgrade.
- Node spool hook (B).
- Systemd timers.
- Two-host physical snapshot.

Each step needs either a service restart, which lane D may not perform, or the
DOIN runtime, which B owns.
