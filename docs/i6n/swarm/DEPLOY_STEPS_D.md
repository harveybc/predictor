# Lane D deploy steps (prepared, NOT executed)

Lane D did not restart, reconfigure, install or start any service, timer, venv
or node. Each step below is for the operator, or for B where marked. Lane C
reviews the code first.

Credentials are used only through existing EnvironmentFiles:

- the lake token: `DATA_GOV_LAKE_TOKEN` in the existing 5056 service env file;
- the warehouse token: `WAREHOUSE_TOKEN` in the existing follower secret env
  file used by `predictor-i6m-olap.service`.

Nobody, agent or script, reads or prints those files. A token reaches a
process only through systemd `EnvironmentFile=`: either the existing unit's, or
a transient `systemd-run --user -p EnvironmentFile=<existing env file>`. The
process then reads `os.environ` only. Every check below that needs a token is a
`tools.i6n_archive.probe` run in such a transient unit (review F3). If a step needs a
credential that no existing service env provides, stop and report it.

Placeholders follow the repository's public-host rules: `<coord>` is the
coordinator, `<worker-b>` is the secondary host, `<state>` is
`~/.local/state`, and `<lake-cas-port>` is a free local port. Ports 5055–5058
and 5060 were in use on 2026-10-11, and 5061 was free then.

## Pins

| Repo | Branch | Commit |
|---|---|---|
| doin-core | satoshi/i6n-external-archive-20261011 | `4a321cd` |
| doin-node | satoshi/i6n-external-archive-20261011 | `018ca2d` |
| data-lake | satoshi/i6n-external-archive-20261011 | `fbb6e58` (data-lake-service 0.1.1) |
| data-warehouse | satoshi/i6n-external-archive-20261011 | `463ad56` (data-warehouse-service 0.1.3) |
| predictor | satoshi/i6n-swarm-D-archive-20261011 | see D_RETURN.md (predictor-olap-store / predictor-duckdb-store 0.1.7) |

## 0. Build wheels once, from the pins

```bash
D=/var/tmp/i6n_D_predeploy; mkdir -p $D/src $D/wheels
for r in doin-core:4a321cd doin-node:018ca2d data-lake:fbb6e58 data-warehouse:463ad56 predictor:<pin>; do
  name=${r%%:*}; rev=${r##*:}; mkdir -p $D/src/$name
  git -C ~/Documents/GitHub/$name archive $rev | tar -x -C $D/src/$name
done
for p in doin-core doin-node data-lake data-warehouse predictor/olap/store predictor/olap/duckdb_store; do
  <python3.12> -m pip wheel --no-deps $D/src/$p -w $D/wheels
done
sha256sum $D/wheels/*.whl > $D/wheels/SHA256SUMS   # record in the deploy receipt
```

## 1. Rehearse the migration on a disposable instance first

This step is already done in git: the test `olap/store/tests/test_doin_projection_migration.py` passed. It builds a cube with the deployed provider code (`5c383d17`), then opens it with 0.1.7. The prior relations and rows stay unchanged, and exactly four relations are added.

The optional operator rehearsal on a copy of the production cube:

1. Take the copy during the step-3 window, while the service is stopped.
2. Install the wheels into a scratch venv.
3. Open the copy with `PredictorDuckdbStore` (`memory_limit=1GB`, `threads=1`).
4. Check that `gov_report`/`gov_metric` counts are unchanged and that
   `gov_doin_projection` holds 0 rows.
5. Delete the copy.

## 2. Lake CAS instance (new, separate venv; live lake services untouched)

1. `python3.12 -m venv ~/.venvs/doin-archive-lake && ~/.venvs/doin-archive-lake/bin/pip install wheels/data_lake_service-0.1.1-*.whl`
2. Write the private host config `<state>/crispdm-data-foundation/doin-archive-cas.host.json`:
   ```json
   {"kind": "lake", "store_id": "doin_archive_cas", "transport": "http",
    "web_host": "127.0.0.1", "web_port": <lake-cas-port>,
    "backend": {"distribution": "data-lake-service", "entry_point": "cas_file_store",
                "settings": {"root": "<persistent volume>/doin_archive_cas", "deployed": true}}}
   ```
   `deployed: true` is the operator's declaration. The store never infers it.
3. Create the unit `crispdm-data-lake-doin-archive.service` by copying
   `crispdm-data-lake-financial.service`, then make these changes:
   - `EnvironmentFile`: keep the same existing 5056 env file, so the same
     service account is used;
   - `ExecStart`: `~/.venvs/doin-archive-lake/bin/python -m data_lake_service.main --load_config <that json>`;
   - limits: `MemoryMax=1G`, `MemorySwapMax=0`.

   Then run `systemctl --user daemon-reload && systemctl --user enable --now crispdm-data-lake-doin-archive.service`.
4. Verify:
   - `curl -s 127.0.0.1:<lake-cas-port>/healthz` returns `ok`.
   - The capabilities and a 3-byte PUT/GET probe, with the token passed only
     through the existing env file:
     ```bash
     systemd-run --user --wait --pipe --collect -p EnvironmentFile=<existing 5056 env file> \
       -p WorkingDirectory=<pinned predictor checkout> -p Environment=PYTHONPATH=<pinned predictor checkout> \
       <python> -m tools.i6n_archive.probe lake --url http://127.0.0.1:<lake-cas-port> --token-env DATA_GOV_LAKE_TOKEN
     ```
     Expect `"ok": true`. The probe object stays, which is harmless because
     objects are immutable.

## 3. Warehouse upgrade (maintenance window; owner process only)

1. While the service is still running, record the baseline counts and capabilities:
   ```bash
   systemd-run --user --wait --pipe --collect -p EnvironmentFile=<existing follower secret env file> \
     -p WorkingDirectory=<pinned predictor checkout> -p Environment=PYTHONPATH=<pinned predictor checkout> \
     <python> -m tools.i6n_archive.probe warehouse --url http://127.0.0.1:5057 --token-env WAREHOUSE_TOKEN
   ```
   This yields `gov_report` and `gov_metric` counts; `gov_doin_projection`
   answers an HTTP error before the upgrade, which is expected.
2. Then `systemctl --user stop crispdm-data-warehouse-olap.service` and take the
   pre-upgrade snapshot (§6). The snapshot tool refuses a DuckDB file that
   still has a `.wal`.
3. Install `data_warehouse_service-0.1.3`, `predictor_olap_store-0.1.7` and
   `predictor_duckdb_store-0.1.7` into `~/.venvs/store-hosts-duckdb-prod` with
   `pip install --no-deps <wheels>`.
4. `systemctl --user start crispdm-data-warehouse-olap.service`. The start-up DDL
   creates the `doin_0001` relations with `CREATE ... IF NOT EXISTS` only.
5. Verify:
   - Run the same `probe warehouse` transient unit as in 3.1. It must report
     `doin_capabilities_ok: true`, the 3.1 counts unchanged, and
     `gov_doin_projection` = 0.
   - The loader and `predictor-i6m-olap` keep running. They use only the old
     routes.
6. Rollback: reinstall the previous wheels. Their SHA-256 values are the ones
   in the venv's `direct_url.json` (`100fb36f…` warehouse, `222e2751…` duckdb
   provider). Restart the service. The additive relations are inert, or
   restore the snapshot.

## 4. DOIN node (B)

1. Install doin-core `4a321cd` and doin-node `018ca2d` into the node environment of the swarm runtime.
2. In the runtime loop, call `ingest_reference_spool(<archive-root>/spool, node.consensus.record_transaction, known_tx_ids=node._seen_candidate_tx_ids)` on each tick.
3. Make sure blocks are persisted to `<node data dir>/chain.json`. The ETL reads only that file, and fork choice stays with the node.
4. Until this step is done, references wait in the spool and items stay at `OBJECT_DURABLE`. They are never green.

## 5. Archive timers (templates; install only after steps 2–4)

- `i6n-archive-pump.service` (oneshot):
  - `ExecStart=crispdm-run -m 1G -t 300s -- <python> -m tools.i6n_archive pump --config <private.json>`
  - `EnvironmentFile=<existing 5056 env>`
  - `WorkingDirectory=<pinned predictor checkout>`
  - `PYTHONPATH` = that checkout. doin-core and doin-node are installed in that python.
- `i6n-archive-etl.service` (oneshot): same layout with `etl`, plus an
  `EnvironmentFile` for each of the two existing env files: the lake token and
  the follower `WAREHOUSE_TOKEN`.
- Timers: pump every 2 min, ETL every 5 min. Both are idempotent, and an outage
  just leaves work in the outbox. The ETL is incremental: a `(tx, block)` that was
  already read back is skipped through the `etl_done` watermark, so the cost of
  a run grows with new references, not with history. Chain verification
  remains linear in chain length. A reorg clears the watermark of revoked
  blocks.
- Heartbeat check, run once: `python -m tools.i6n_archive status --config <private.json>`.
  This command needs no token.
  This proves the services run. It is not a claim of on-chain publication.
  Then run the `probe warehouse` transient unit from §3.1 once. Its
  `gov_doin_projection` count must be above 0 once `status` shows an item with
  OLAP_READBACK_VERIFIED.

## 6. Physical snapshot on two hosts

These steps run on `<coord>`. Stop the archive ETL timer first. Take the
warehouse cube only inside the step-3 window, while the owner process is
stopped.

```bash
python -m tools.i6n_archive.snapshot take \
  --source cas=<persistent volume>/doin_archive_cas \
  --source outbox=<archive-root>/outbox.sqlite \
  --source warehouse=<state>/crispdm-duckdb/prod/cube.duckdb \
  --dest /var/tmp/i6n_D_snapshot_<date>
python -m tools.i6n_archive.snapshot verify --dest /var/tmp/i6n_D_snapshot_<date>
rsync -a --protect-args /var/tmp/i6n_D_snapshot_<date>/ <worker-b>:<snapshot dir>/
ssh <worker-b> '<python> -m tools.i6n_archive.snapshot verify --dest <snapshot dir>'
```

The cube is about 10 GB, so check free space on both hosts first.

Commit only the following to git, under `docs/i6n/swarm/evidence_D/`:

- `MANIFEST.json` → its `files_sha256`, file count and bytes;
- the two `verify` outputs.

Never commit `.duckdb`, `.sqlite`, CAS objects, host names or paths.

## Known blockers (true at the time of writing)

- Steps 2, 3 and 5 restart or install services, which lane D may not do.
- Step 4 is B's runtime.
- No chain has finalized a reference outside the in-process test chain. Until
  it does, no claim of on-chain custody is made.
