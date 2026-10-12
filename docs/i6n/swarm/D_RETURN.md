# Lane D return: external archive → chain → OLAP

Date: 2026-10-11. Reviewer: lane C.

## Summary

The `external_payload` mode is implemented end to end. Full-payload
compatibility is kept.

**What is verified:** everything below was verified only on disposable
instances: a temporary file CAS directory, an in-process chain double, and
throwaway SQLite and DuckDB files.

**What is not deployed:**

- the lake CAS service;
- the warehouse wheels and the `doin_0001` relations;
- the node spool hook;
- the timers.

No live chain has finalized a reference, so **no on-chain custody is claimed**.
The earlier `DisposableServiceAdapter` is untouched, is still not a lake, and
keeps `chain_verified=false`.

## Pins

| Repo | Branch | Commits (oldest → newest) |
|---|---|---|
| predictor | satoshi/i6n-swarm-D-archive-20261011 | `3733b833` ACK, `4bedc36c` INTERFACE D.1, `5dd7d51c`, `8c8afd1b`, `790e5640`, `83c47b50`, plus this docs commit |
| doin-core | satoshi/i6n-external-archive-20261011 (from `a50a33e`) | `4a321cd` |
| doin-node | satoshi/i6n-external-archive-20261011 (from `4c23a6f`) | `323ff7b`, `018ca2d` |
| data-lake | satoshi/i6n-external-archive-20261011 (from `e82b33e`) | `cf672f5`, `fbb6e58` |
| data-warehouse | satoshi/i6n-external-archive-20261011 (from `868177d` = live wheel, file-for-file) | `159f7fc`, `463ad56` |

## Requirement → test → result

Every test below was written first. Each PRE run failed because the module or
route was absent; the cell says *collection error* or *3 failed* where that
applies.

| Requirement | Test | PRE | POST |
|---|---|---|---|
| Body, manifest and block hash are distinct and non-circular | doin-core `test_external_payload.py::test_body_manifest_and_block_hash_are_distinct_and_not_circular` | collection error | pass |
| full ≡ external analytic equality | predictor `test_i6n_archive.py::test_full_and_external_payload_project_identical_analytic_rows`; doin-core `test_full_payload_resolves_to_the_same_bytes_as_external` | collection error | pass |
| A failed PUT never commits | `test_failed_put_never_commits`; doin-node `test_failed_put_returns_no_reference`, `test_readback_mismatch_returns_no_reference` | collection error | pass |
| A crash at PUT → commit → ETL → ACK is recoverable | `test_crash_between_every_step_recovers_without_duplicates` (crash after PUT, after node submit, after projection before ACK; deterministic tx id) | collection error | pass |
| Wrong bytes, digest or locator are rejected | `test_wrong_bytes_digest_and_locator_are_rejected_before_projection`; doin-core `test_wrong_bytes_digest_and_locator_are_refused`; doin-node `test_unknown_locator_store_and_wrong_bytes_are_refused`, `test_chain_refuses_forged_reference_and_broken_links` | collection error | pass |
| A duplicate does not add metrics | `test_duplicate_publication_does_not_add_metrics`; store `test_replay_and_duplicate_block_do_not_add_metrics` (SQLite + DuckDB) | collection error | pass |
| A reorg revokes the canonical projection and preserves history | `test_reorg_revokes_canonical_projection_and_preserves_history`; store `test_reorg_revokes…`, `test_reinstating_a_block_after_a_reorg_back` | collection error | pass |
| Snapshot restore | `test_snapshot_restore_reproduces_the_canonical_projection`, `test_snapshot_cli_take_and_verify` | collection error | pass |
| A new peer can resolve the body | `test_new_peer_resolves_the_body_through_the_lake_http_routes` (real data-lake Flask app + HttpByteStore; real warehouse Flask app + HttpSink); doin-node `test_store_then_reference_order_and_new_peer_resolves` | collection error | pass |
| Separate states; a missing state is never green | `test_full_flow_states_are_separate_and_green_only_at_the_end` (CONFIRMING ≠ finalized) | collection error | pass |
| All candidates kept, no invented score, no TEST | `test_full_flow…` (FAILED candidate row, fitness only for scored); `test_invented_scores_and_test_rows_are_refused` | collection error | pass |
| Alignment with A.2, C.1 and E.1 | `test_interface_a2_roles_weekly_validation_rows_and_support_digest`; E formulas are verified in `records.validate_candidate` | n/a (added on relay) | pass |
| Additive migration, tested first on a disposable instance | `olap/store/tests/test_doin_projection_migration.py` (cube built by `5c383d17` code, reopened by 0.1.7); store `test_migration_is_additive_and_idempotent` | collection error | pass |
| Lake object routes | data-lake `tests/test_object_routes.py` (5) | collection error | pass |
| Warehouse projection routes | data-warehouse `tests/test_doin_projection_routes.py` (3) | 3 failed | pass |
| DOIN validations are kept | doin-node full suite (the chain check is additive) | — | 444 passed |
| Spool ingest validates and is idempotent | doin-node `test_reference_spool_ingest_validates_and_is_idempotent` | — | pass |

## POST suite totals

Each suite was run once.

| Suite | Result |
|---|---|
| doin-core | 304 passed |
| doin-node | 444 passed (with data-lake on the path; the 3 offchain-shadow tests need it, which was already true before this lane) |
| data-lake | 39 passed (`-m "not packaging"`) |
| data-warehouse | 89 passed |
| predictor `tests/test_i6n_archive.py` + `olap/store/tests` | 108 passed, 2 skipped |
| predictor `tests/test_i6n_archive.py` after the A.2 alignment | 14 passed |

## Commands

All commands ran under CPU and `crispdm-run`, one job at a time.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree srcs> \
  crispdm-run -m 2G -t 120s -- <python> -m pytest -q -p no:cacheprovider <files>
```

- The DuckDB tests used a scratch venv that has `duckdb==1.5.5` and
  `duckdb_engine==0.17.0`, matching production, layered over the anaconda
  site-packages.
- doin-core and doin-node used the trading-stack python.

## Risks and open items

- **Not deployed.** See DEPLOY_STEPS_D.md. Steps 2, 3 and 5 install or restart
  services, which this lane may not do. Step 4, the node spool hook, is B's
  runtime.
- **Fork choice.** The ETL trusts the block list that the node hands it as
  canonical. It re-verifies links, merkle roots, header hashes and references,
  but it does not choose forks.
- **Duplicate-commit view.** Two finalized blocks carrying the same reference
  produce one PROJECTED projection plus DUPLICATE history rows. If the
  PROJECTED block is reorged away while the duplicate block stays, the row is
  absent from `gov_doin_metric_canonical` until the next ETL run reinstates it.
  It is absent, never double-counted.
- **Fold-id check.** The TEST guard on fold ids is a string check: a fold id
  containing `2025` or `TEST` is refused, and rows with `split=TEST` are
  refused. The authoritative split/cutoff enforcement stays with A and C.
- **Mode switch.** A campaign outbox binds one archive mode. Switching mode
  needs a new outbox root.
- **Snapshot size.** The production cube is about 10 GB, so a two-host
  physical snapshot needs a maintenance window and free disk on both hosts.

## Next artifact

1. Lane C review of this return.
2. B wires `record_candidate` into `collect` and `ingest_reference_spool` into the node.
3. The operator executes DEPLOY_STEPS_D §2–§6.
4. Record a heartbeat: `status` JSON plus one `gov_doin_metric_canonical` readback, and
   `evidence_D/` snapshot manifests from both hosts.
