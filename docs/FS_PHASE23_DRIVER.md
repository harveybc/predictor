# Feature-selection phases 2 and 3: driver runbook

Authority: `docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md`
and the order `docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md`.
Every path, identity and host id below is an argument; the repository holds no host
name, address, credential or private path.  Placeholders: `<DATA>` is the directory
holding the phase-1 TRAIN parquet files (`eurusd_features_train.parquet`,
`eurusd_targets_train.parquet`, `eth_*`), `<STATE>` a per-population state root on
each host, `<COORD>` the follower's state root on the coordinator, `<WH>` the warehouse
path/handle accepted by `tools/fs_phase23_warehouse.py` (the DATA agent's module) or,
until it exists, a throwaway database file for the adapter.

## Modules

| File | Role |
|---|---|
| `fs_phase23/manifest/EURUSD_MANIFEST.json`, `ETH_MANIFEST.json` | frozen populations (366/14/5 and 83/6/3) under `phase1-eurusd-final:94d20c038d55e152` and `phase1-final:ETH:29d2f745f5d9e87c`; TRAIN only, folds keep train ranges only; digests of every phase-1 artifact |
| `tools/fs_phase23_manifest.py` | `freeze` from phase-1 closure artifacts (adopts `fs_phase23/data/CONTRACT.json` when present), `verify` data digests |
| `tools/feature_pairwise_campaign.py` | verbs `plan`, `run-worker`, `follow`, `close-phase2`, `run-phase3`, `close-phase3`, `status` |
| `tools/feature_pairwise_worker.py` | bounded block compute: gate, Pearson, Spearman, Kendall tau-b, quantile-bin MI, distance correlation, lagged cross-correlation at elapsed hours {0,1,2,6,24,48,168}, fold stability, states |
| `tools/feature_filter_selection.py` | phase 3: Spearman clustering, mRMR, JMI, causal variants, controls, K={4,8,12,16,24,32} |
| `tools/feature_selection_phase23_status.py` | `STATUS.json` from evidence: expected/complete/failed/active/rate/ETA per population x method x host |
| `tools/fs_phase23_warehouse_adapter.py` | same-signature adapter over a throwaway DuckDB/sqlite, MARKED FOR REPLACEMENT by `tools/fs_phase23_warehouse.py` |
| `tools/fs_phase23_measure_cost.py` | memory/time per 1,000 pairs on a real-sized synthetic block |
| `tools/test_fs_phase23_driver.py` | 19 acceptance tests (subplan section 6) |

## Tests (CPU, memory-capped, never on `/tmp` of a worker)

```bash
crispdm-run -m 3G -t 30m -n fs23-tests -- python -m pytest tools/test_fs_phase23_driver.py -q -p no:cacheprovider \
  -o tmp_path_retention_policy=none --basetemp=<STATE_PARENT>/pytest
```

## Phase 2

```bash
# 1. plan once per population on the coordinator (shards are deterministic; host ids are arguments)
python tools/feature_pairwise_campaign.py plan --manifest fs_phase23/manifest/EURUSD_MANIFEST.json \
  --state-root <COORD>/eurusd --n-shards 256 --host <small-host-id>:small --host <large-host-id-1>:large --host <large-host-id-2>:large
python tools/feature_pairwise_campaign.py plan --manifest fs_phase23/manifest/ETH_MANIFEST.json \
  --state-root <COORD>/eth --n-shards 32 --host <small-host-id>:small --host <large-host-id-1>:large --host <large-host-id-2>:large
# copy <COORD>/<pop>/PLAN.json and MANIFEST.json to each host's <STATE>/<pop>/ (same commit everywhere)

# 2. worker on each host (restartable systemd --user unit or the durable supervisor); one process = one CPU
crispdm-run -m <CAP> -t 12h -n fs23-eurusd-<host-id> -- python tools/feature_pairwise_campaign.py run-worker \
  --plan <STATE>/eurusd/PLAN.json --state-root <STATE>/eurusd --data-root <DATA> --host-id <host-id> --threads 1 [--steal]
#   --steal lets a host take unclaimed shards assigned to others once its own are done; claims are exclusive and expire after 900 s

# 3. follower on the coordinator (adopt, quarantine, submit, readback, close, chain phase 3, STATUS.json every minute)
python tools/feature_pairwise_campaign.py follow --plan <COORD>/eurusd/PLAN.json --state-root <COORD>/eurusd \
  --terminals <mirror-of-host-1>/eurusd/terminals --terminals <mirror-of-host-2>/eurusd/terminals --terminals <mirror-of-host-3>/eurusd/terminals \
  --warehouse <WH> --data-root <DATA> --every 60 --phase3-workers 4
#   terminal mirrors are the INTEGRATION agent's transport (rsync/sshfs); the follower only reads them

# 4. manual closure verbs (the follower runs them automatically; they are idempotent)
python tools/feature_pairwise_campaign.py close-phase2 --plan <COORD>/eurusd/PLAN.json --state-root <COORD>/eurusd --warehouse <WH> --data-root <DATA>
python tools/feature_pairwise_campaign.py run-phase3  --plan <COORD>/eurusd/PLAN.json --state-root <COORD>/eurusd --warehouse <WH> --data-root <DATA> --workers 4
python tools/feature_pairwise_campaign.py close-phase3 --plan <COORD>/eurusd/PLAN.json --state-root <COORD>/eurusd --warehouse <WH>

# 5. status (any host; the follower also writes <COORD>/<pop>/STATUS.json each cycle)
python tools/feature_selection_phase23_status.py --plan <COORD>/eurusd/PLAN.json --plan <COORD>/eth/PLAN.json \
  --state-root <COORD>/eurusd --state-root <COORD>/eth --out <COORD>/STATUS.json --every 60
```

## Evidence produced

* `<STATE>/<pop>/terminals/<unit_id>.json.gz` + `.meta.json`: immutable, named by the digest of population, shard, method and parameters; `failures/`, `claims/`, `quarantine/`.
* `<COORD>/<pop>/adopted/`, `receipts/<unit_id>.json` (identity + digest + readback count per table), `PHASE2_MATRICES.npz`, `ALIAS_GROUPS.json`, `REDUNDANCY_CLUSTERS.json`, `PHASE_2_COMPLETE.json`.
* `<COORD>/<pop>/phase3/terminals/`, `phase3/receipts/`, `PHASE_3_FILTER_COMPLETE.json`, `CANDIDATES_FOR_VALIDATION.json` (no predictive winner, no test).
* Warehouse tables: `feature_pair_metrics`, `feature_pair_stability`, `feature_pair_gate`, `feature_alias_groups`, `feature_redundancy_clusters`, `feature_filter_rankings`, `feature_filter_subsets`; unique key `(run_id, row_key)`.

## Row denominators (per population)

Per pair and fold slot (TRAIN + each inner fold): 5 base metrics at lag 0, `xcorr` at lag 0, and two
lead directions for each positive lag (12 rows): 18 rows.  EURUSD: 66,795 pairs x 6 slots x 18 =
7,213,860 metric rows, 1,202,310 stability rows, 66,795 gate rows.  ETH: 3,403 x 4 x 18 = 245,016,
61,254 and 3,403.  Lags that do not fall on the bar grid (ETH 4h bars at 1, 2, 6 h) are explicit
`NOT_APPLICABLE` dispositions, never omissions.

## Warehouse interface expected from `tools/fs_phase23_warehouse.py`

`open_warehouse(path_or_handle)` returning an object with `submit_rows(run_id, table, rows) -> receipt`
(receipt carries `run_id`, `table`, `row_count`, `rows_sha256` = sha256 of the sorted per-row sha256 of
canonical JSON, `receipt_sha256`, `inserted`, `duplicates_ignored`), `read_run(run_id, table, unit_id=None)`
(the optional `unit_id` filter keeps per-unit readback cheap; without it the follower filters client-side)
and `reconcile(run_id) -> {"run_id", "tables": {table: {"count", "rows_sha256"}}}` with the same digest rule.
