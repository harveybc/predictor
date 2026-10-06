# fs_phase23/data: contract, warehouse schema, readback, snapshot (data lane)

Order: `docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md` sections B, D, F, G.
Code: `tools/fs_phase23_warehouse.py`, `olap/migrations/fs_phase23/0001_fs_phase23_additive.sql`,
`tests/test_fs_phase23_warehouse.py` (26 tests, `TEST_RESULTS.txt`).

| file | what |
|---|---|
| `CONTRACT.json` | frozen populations derived from phase-1 artifacts: EURUSD 366/14/5 folds/66,795 pairs, ETH 83/6/3 folds/3,403 pairs; phase-1 digests; CAUSAL_SUPPORTED candidates (12 + 3) |
| `MIGRATION_PLAN.md`, `MIGRATION_DRYRUN_phase1_snapshot.json` | additive migration, dry-run evidence, integration steps, open items |
| `READBACK_REPORT_FORMAT.md`, `EXAMPLE_*.json` | the reconciliation documents close-phase2/3 consume (examples are synthetic) |
| `SNAPSHOT_PROCEDURE.md` | `.duckdb.zst` + SHA-256 + release asset, with the owner-credential boundary |

Regenerate the contract (arguments are the phase-1 artifacts by role; no path is stored):
`python tools/fs_phase23_warehouse.py contract --eurusd-plan ... --eth-completion ... --out CONTRACT.json`
then `verify-contract CONTRACT.json`.
