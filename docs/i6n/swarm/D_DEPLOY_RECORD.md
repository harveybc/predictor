# Lane D deploy record (evidence owner, task C of SATOSHI_I6N_EXECUTE_NOW_20261012)

Date: 2026-10-12 (UTC). Orders: `docs/handoffs/SATOSHI_I6N_EXECUTE_NOW_20261012.md`
(bc01dee9, sha256 b78ec898…), sections 1, 2.C, 3 and 4, relayed by the coordinator.

## Outcome: STOPPED_AT_PERMISSION_DENIAL

The live deployment did not happen. The permission classifier denied two
consecutive commands of the deploy sequence with the reason
`[Production Deploy]`. The orders say not to bypass a denial, so the
deployment stopped at the first one.

- **What the denied commands were.** Both were read-only:
  - an inspection of the built lake wheel's `backends.py` (to diagnose the
    failure below);
  - a check that no lane D unit exists and that the live warehouse, lake and
    `i6n-launch-cost-collect.timer` units were active.
- **What did not happen:**
  - No systemd unit was created, enabled, started, stopped or restarted.
  - The live warehouse (5057), the live lakes (5056/5058/5060), the loader,
    the publishers and `i6n-launch-cost-collect.timer` were not touched.
  - The production cube was not stopped, copied or migrated.
  - No token or env file was read.
- **Pilot publications.** The existing 72 reports / 558 metric rows are
  unaffected, because nothing wrote to the warehouse.
- **Rollback.** None was needed, because nothing live changed.

## Done before the denial (inert, nothing references it)

| Item | State |
|---|---|
| Code for the orders (predictor `satoshi/i6n-swarm-D-archive-20261011`) | `47bf8cf8` off-chain readback (`OFFCHAIN_VERIFIED`, `chain_finalized=false`, superseded never duplicated by later finality; visible and retryable failures without refit) and the lifecycle status (QUEUED/CLAIMED/UPDATING/COMPLETED per fold, NOT_RECORDED/LOCAL_COMPLETE/ARCHIVED/READBACK_VERIFIED per candidate, separate `chain_finalized`, ETA kind + basis_n, unknown without samples). `e90a75ee` snapshots a stopped DuckDB owner's file with its `.wal`. `1acfda59` merges integration `23dbc81c` so a single checkout holds archive + swarm code. |
| Tests (CPU, `crispdm-run -m 2G -t 120s`, one run each) | PRE: 6 off-chain regression tests failed. POST: 48 passed, 1 skipped (archive + projection store/migration/parity); 3 snapshot tests passed; after the merge, `tests/test_i6n_archive.py` + B's `test_i6n_swarm_campaign.py`/`test_i6n_swarm_review_e.py` gave 60 passed. |
| Wheels built from the pins | `/var/tmp/i6n_D_predeploy_20261012/wheels` with `SHA256SUMS`: data_lake_service 0.1.1 `4e3524c6…`, data_warehouse_service 0.1.3 `467e54a4…`, doin_core 0.1.0 `065724f7…`, doin_node 0.1.0 `f66ff50b…`, predictor_duckdb_store 0.1.7 `8341662f…`, predictor_olap_store 0.1.7 `2731e226…` |
| Pre-deploy verification | Every deployed `.py` file of predictor_olap_store 0.1.6, predictor_duckdb_store 0.1.6 and data_warehouse_service 0.1.2 is byte-identical to the lane D bases (`5c383d17`, `868177d`). The upgrade therefore adds only lane D's changes. The rollback wheels are present and their hashes match the venv's records (`100fb36f…`, `056563a9…`, `222e2751…`). |
| Lake CAS preparation | A new venv `~/.venvs/doin-archive-lake` with data_lake_service 0.1.1. The private host config `<state>/crispdm-data-foundation/doin-archive-cas.host.json` (store `doin_archive_cas`, 127.0.0.1:5061, root `<state>/doin_archive_cas`, `deployed: true`). The empty data root `<state>/doin_archive_cas`. Nothing serves it, and no unit exists. |

## Open defect found during preparation

`python -m data_lake_service.main --print-identity` failed in the new venv with
`a backend declares unknown capabilities: ['byte_read', 'byte_upload']`.
data-lake declares those capabilities in `backends.CAPABILITIES` at
`e82b33e`/`fbb6e58`, and the lane D tests pass from the worktree. The installed
wheel therefore did not carry the checkout's `backends.py`. The cause is not
established, because the inspection command was the one denied.

The likely candidate is the stale `build/` tree committed in the data-lake
repository. setuptools reuses `build/lib` when building from a source tree that
contains it, so the wheel may carry an old copy. Fix before deploying:

1. Build from a tree without `build/`, or remove `build/` from git.
2. Rebuild the wheel.
3. Check `unzip -p <wheel> data_lake_service/backends.py | grep byte_`.

## What remains (operator, or a session with deploy permission)

Run DEPLOY_STEPS_D.md (D.4) in order:

1. §0: rebuild the lake wheel (defect above).
2. §2: the lake CAS unit and the probe.
3. §3: the warehouse window. Record counts while the service runs, stop it,
   take a snapshot with `--owner-stopped`, upgrade, probe, and roll back
   immediately on any failed check.
4. §4: B's node spool hook.
5. §5: the archive timers. If the campaign root does not exist yet, install
   them disabled with a WAIT_DEPENDENCY note.
6. §6: the second-host copy.
7. Run the smoke record through the deployed path. Expected outcome: OBJECT_DURABLE,
   spool reference, OLAP_READBACK_VERIFIED with `chain_finalized=false`, and
   CHAIN_FINALIZED stays empty until a live chain finalizes.

None of these were executed, and no deployment state is claimed.
