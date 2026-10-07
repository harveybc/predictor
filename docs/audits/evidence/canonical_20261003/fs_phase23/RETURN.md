# RETURN — feature selection phases 2 and 3, automated (§G of MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05)

Observed 2026-10-07T00:19Z from artifacts under `~/.local/state/canonical_20261003/fs_phase23/`
(roles only: coordinator / worker_a / worker_b; no host name, address or token appears here).
Phase 1's causal results were inputs; nothing below is a final feature selection.

## 1. Commits and branches per repository touched

| repository | branch | tip | content |
|---|---|---|---|
| predictor | `satoshi/fs-phase23-driver-20261005` | fc026d04 (driver 06a8f153) | driver, worker, filters, status, manifests, 19 tests, runbook `docs/FS_PHASE23_DRIVER.md` |
| predictor | `satoshi/fs-phase23-data-20261005` | 18865d4c (first 6859b07a) | `tools/fs_phase23_warehouse.py`, migration `olap/migrations/fs_phase23/0001`, store packages 0.1.5 / 0.1.4, 27 tests |
| predictor | `satoshi/fs-phase23-integration-20261005` | 9e613bd3 | deployment kit Phase A (`tools/fs_phase23_deploy/`) |
| predictor | `satoshi/fs-phase23-integrated-20261006` | **this commit** (campaign deployed from f71d328e → 631be7f4 → 9e182505 → 7cdf4660 → abac6600) | merge of the three + fixes: owner fixes f5fbe4ad / abac6600 (live URL preserved through CLI), 631be7f4 (incident 1), 9e182505 (incident 2), 7cdf4660 (follower heartbeat); deploy receipts in `integration/` |
| predictor | `satoshi/canonical-exec-20261003` | pulled from the integrated branch (no rebase) | canonical line |
| data-warehouse | master | 5ca5937 (built on the deployed 2d4550d) | capability `write_fs_phase23_rows`, routes `/api/v2/fs-phase23/{rows,reconcile}`, store bumps; deployed and smoked (`warehouse/LIVE_SMOKE_20261006.json`, run `phase2-smoke:2026-10-06`) |
| host (coordinator) | — | — | owner changed `crispdm-data-warehouse-olap.service` MemoryMax 2 GiB → 6 GiB after incident 2 (191 OOM kills of the store under 26k-row POSTs); service active, 0 restarts since 2026-10-06T22:28Z local start |

## 2. Expected / complete / failed per asset × method × host role

Pairwise (`pairwise_v1`), from `STATUS.json` by_host at 00:14Z and the live reconcile:

| asset | role | expected (plan) | complete (authored) | stolen (authored shards of another role) | failed |
|---|---|---|---|---|---|
| EURUSD | coordinator | 85 | 85 | 0 | 0 |
| EURUSD | worker_a | 86 | 86 | 0 | 0 |
| EURUSD | worker_b | 85 | 85 | 0 | 0 |
| EURUSD | total | 256 units / 66,795 pairs | 256 / 66,795 | — | 0 |
| ETH | coordinator | 11 | 11 | 0 | 0 |
| ETH | worker_a | 11 | 8 | 0 | 0 |
| ETH | worker_b | 10 | 13 | 3 (took worker_a's remaining shards) | 0 |
| ETH | total | 32 units / 3,403 pairs | 32 / 3,403 | 3 | 0 |

STATUS.json also counts "stolen" claims that lost the race and were adopted from the home
role (EURUSD worker_a 5, worker_b 31; ETH worker_a 3, worker_b 11); the table above counts
the author recorded in each terminal's `.meta.json`, the only count that changes a row's
`host_role` in the store. Quarantined terminals 0, submit failures 0 at closure.

Filters (`filter_v1`, coordinator only): EURUSD 14 / 14 targets, ETH 6 / 6; 0 failed.

Live warehouse rows (run = phase-1 identity; `integration/closure/LIVE_RECONCILE_*.json`):

| table | EURUSD | ETH |
|---|---|---|
| feature_pair_gate | 66,795 | 3,403 |
| feature_pair_metrics | 7,213,860 | 245,016 |
| feature_pair_stability | 1,202,310 | 61,254 |
| feature_alias_groups | 1 | 5 |
| feature_redundancy_clusters | 277 | 38 |
| feature_filter_rankings | 45,990 | 4,212 |
| feature_filter_subsets | 686 | 294 |

Metric dispositions (EURUSD, from `PHASE_2_COMPLETE.json`): MEASURED 6,804,870,
INSUFFICIENT_SUPPORT 391,104, NOT_APPLICABLE 17,886 — zero silent omissions.
`uses_validation_or_test: false` in both closures.

## 3. Time, peak memory, cost

From the 288 terminal manifests (`wall_seconds`, `peak_rss_bytes`), the cost measurement
`fs_phase23_20261005/COST_MEASUREMENT_worker_b.json` (0.68 s/pair, 1.45 GiB peak incl. the
synthetic builder) and the deploy/incident receipts:

| asset | role | units | pairs | compute wall | peak RSS per process |
|---|---|---|---|---|---|
| EURUSD | coordinator (1 slot, 1 GiB cap) | 85 | 20,845 | 10,250 s | 0.71 GiB |
| EURUSD | worker_a (1 slot, 2 GiB) | 86 | 23,150 | 6,897 s | 0.73 GiB |
| EURUSD | worker_b (3 slots, 2 GiB) | 85 | 22,800 | 15,761 s | 0.76 GiB |
| EURUSD | total | 256 | 66,795 | 32,909 s = 9.14 CPU-h (0.49 s/pair) | — |
| ETH | all | 32 | 3,403 | 1,001 s (0.29 s/pair) | 0.37 GiB |

Cells: a pair × fold-slot × metric row. EURUSD 7,213,860 metric rows in 32,909 CPU-s =
**4,562 CPU-s per million cells** (1.27 CPU-h); ETH 245,016 rows in 1,001 s = 4,085 CPU-s per
million. Elapsed: campaign launched 2026-10-06T01:26Z; ETH phase 3 closed 04:30Z; EURUSD
phase 2 closed 23:54:40Z and phase 3 23:57:02Z (the EURUSD elapsed time is dominated by the
two follower incidents below, not by compute). Phase-3 filters: 9.5 s per EURUSD target,
0.7 s per ETH target. Store peak under 26k-row batches exceeded the 2 GiB unit cap (incident
2); the owner raised it to 6 GiB. Snapshot copy job on the coordinator: tree peak 4.27 GiB,
stopped by the admission monitor (incident 5).

## 4. Warehouse and snapshot

* Warehouse: the coordinator's governed DuckDB OLAP store, service URL `http://127.0.0.1:5057`
  (token from the environment only), handle `tools/fs_phase23_warehouse.open_warehouse(url)`.
* Readback query (any role with the token):
  `GET /api/v2/fs-phase23/rows?run_id=phase1-eurusd-final:94d20c038d55e152&table=feature_pair_gate&unit_id=<unit>`
  and `POST /api/v2/fs-phase23/reconcile {"run_id": ...}`; locally
  `python tools/fs_phase23_warehouse.py reconcile --service-url http://127.0.0.1:5057 --run-id <identity>`.
* Live reconcile 00:19Z: `complete: true` for both runs; `integration/closure/LIVE_RECONCILE_{eurusd,eth}.json`;
  all 14 table digests equal to the ones sealed in the closures (`DIGEST_PARITY_live_vs_closures.txt`).
  The `--receipts` path was refused by the service for the follower's per-unit wrapper files;
  receipt coverage is the closures' own reconcile (`receipts_cover_store: true`, receipted_inserted = count).
  `readback --service-url` is refused by the query route (`LIMIT too large`): a DATA-module
  follow-up; the digest comparison above covers the same evidence.
* Snapshot: store stopped 00:20:44Z–00:45:04Z (user unit), exclusive open + CHECKPOINT +
  copy by `tools/olap_duckdb_migrate.py snapshot --owner-stopped`; its measurement phase was
  stopped by the admission monitor (incident 5), so the migrate verdict is UNVERIFIED_COPY.
  The copy was verified instead on the phase-2/3 boundary: `tools/fs_phase23_warehouse.py`
  reconcile on the copy equals the live reconcile table by table (counts and digests, 14/14,
  `DIGEST_PARITY_snapshot_copy_vs_live.txt`, `SNAPSHOT_COPY_RECONCILE_*.json`).
  The compression itself was also stopped by the coordinator's admission monitor (incident 6), so the
  verified copy was moved to worker_b (digest equal on both ends) and the asset, its `.sha256`
  and `SNAPSHOT_MANIFEST.json` were built and verified there under its capped launcher.
  Asset `cube_phase2-3_20261007.duckdb.zst` (zstd level 12), SHA-256 **89d19f47b308bcf9f24853369ab9870c784cc22121ab7a300f0116609394e107 (3,562,568,823 bytes; decompressed file SHA-256 dd3ce4324fa769190e8087aec0225e6b07933ffbe3ec1f73d02a784a21c73333, 10,620,252,160 bytes)**,
  `SNAPSHOT_MANIFEST.json` and `.sha256` beside it; verify-snapshot: **`verified: true` on worker_b (asset digest, decompressed digest and every relation's row count and digest equal to the manifest; `integration/closure/VERIFY_SNAPSHOT_worker_b.json.txt`)**.
  Release: https://github.com/harveybc/predictor/releases/tag/phase2-3-feature-selection-20261007 —
  **publish pending the owner's hand**: the policy classifier denied the `gh release create/upload`
  from this session (data-exfiltration class), and it was not pursued another way. The asset
  (3,562,568,823 bytes), its `.sha256`, `SNAPSHOT_MANIFEST.json` and `RELEASE_NOTES.md` sit on the
  coordinator under `/var/tmp/fs23_snapshot_20261007/release/` (and on worker_b under the campaign
  state `snapshot/release/`); the exact commands are `publish_commands` in
  `integration/closure/SNAPSHOT_MANIFEST.json`. Manifest, digest file and notes are committed in
  `integration/closure/`; the binary is never committed.

## 5. Alias groups, clusters and K trajectories (produced, no winner)

* EURUSD alias groups: 1 — `px.logret_1h` ≡ `tv.wav_d1` (EXACT_AFFINE), representative
  `px.logret_1h`; admissible 365 of 366. Redundancy clusters (average linkage on
  1−|Spearman_TRAIN|, cut |ρ| ≥ 0.9): 277 (226 singletons, 31 pairs, 12 triples, 4 of size 4, 1 of 5,
  1 of 6, 2 of 7); largest: the seven US-index daily log-return mirrors (`yh.dia/dji/gspc/ivv/spy/voo/vti.logret_1d`,
  representative `yh.dia.logret_1d`), the same seven at 5d, six short-rate levels
  (`fred.rates.dff/dgs3mo/dprime/dtb3/fedfunds/tb3ms.level`), five oscillators (`ta.bb_pctb_20`,
  `ta.cci_20`, `ta.ema20_dev`, `ta.rsi_14`, `px.zclose_24`).
* ETH alias groups: 5 BYTE_IDENTICAL pairs (`log_return_1`≡`statistical__log_return_1`,
  `bb_middle`≡`sma_20`, `return_10/20/60`≡`roc_10/20/60`); admissible 78 of 83. Clusters 38
  (22 singletons … one of size 14: Bollinger bands + EMA 10/20/50/100/200 and price levels).
* Phase 3, per population × target × method: full rankings (45,990 + 4,212 rows) and ordered
  subsets for K = {4, 8, 12, 16, 24, 32} for SPEARMAN_CLUSTER, MRMR, JMI, MRMR_CAUSAL, JMI_CAUSAL,
  UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K, plus ALL_ADMISSIBLE (K = 365 / 78): 686 EURUSD
  candidates over 14 targets and 294 ETH over 6 (`CANDIDATES_FOR_VALIDATION.json`,
  `label: FILTER_CANDIDATE`, `is_final_selection: false`, `predictive_winner: null`,
  `uses_test_split: false`). Example, not a choice: MRMR K=8 for `Y_s_1h` = cal.hour_cos,
  fred.credit.bamlc0a0cm.level, fred.fx_indices.dtwexb.logret_5d, fred.rates.dprime.logret_1d,
  px.close_loc, px.hours_since_prev_bar, px.parkinson_24, px.rv5. No predictive winner was
  declared; the test split was never read.

## 6. Tests and parities

| suite | PRE | POST (tip a2b7352f, campaign venv, `crispdm-run -m 3G`) |
|---|---|---|
| driver `tools/test_fs_phase23_driver.py` | red by absence (PRE receipt `fs_phase23_20261005/PRE_RECEIPT.json`) | 19 + 7 regressions |
| warehouse `tests/test_fs_phase23_warehouse.py` | — | 27 |
| kit `tools/fs_phase23_deploy/test_*.py` | — | 26 (policy 10, claims 14, register 2) + 2 |
| total | | **81 passed** (`integration/TESTS_INTEGRATED_7f0d570b.txt` first merged run 70; +8 regressions from the incidents; final run 81) |

Restart parity: every restart of a worker slot or follower adopted existing valid terminals and
recomputed none (`adopted_existing` in run-worker summaries; follower `already` counts);
restarting the loop on a finished plan exits ALL_DONE without a new claim (loop exercise,
`integration/loop_exercise/`). Future perturbation and column-order invariance: driver tests
(subplan §6) pass. Readback: every receipt carries readback_count = inserted = row_count per table;
live reconcile and digest parity above.

## 7. Incidents (root cause → fix → receipt)

1. **PK phantom after SIGKILL** (EURUSD follower crash loop, 189 restarts): DuckDB 1.5.6 ART
   primary-key index kept a phantom `row_identity_sha256` after the follower was SIGKILLed
   mid-transaction; SELECT found no row, INSERT raised Duplicate key; the engine exception escaped.
   Fix 631be7f4: per-unit submit failures isolated, SIGTERM-safe loop, `rebuild_duckdb_pk_index.py`
   (count/digest parity). Receipts `integration/incident_20261006/INCIDENT_RECORD.json`,
   `PK_INDEX_REBUILD_{eurusd,eth}.json`, `DEPLOY_RECEIPT_631be7f44364.json`.
2. **int/str K digest** (ETH stuck in phase 3): phase-3 terminals digested with integer K keys that
   sort differently after the JSON round trip, so every terminal verified as corrupt. Fix 631be7f4
   (`canonical_bytes` digests the JSON form; subsets keyed by `str(K)`; corrupt phase-3 terminals
   quarantined and recomputed). Same receipts.
3. **Host OOM loop + memory fix** (EURUSD follower cycling, 122 units never landed): each 26k-row
   metrics POST drove the warehouse host over its 2 GiB MemoryMax (191 OOM kills); the client saw
   RemoteDisconnected without retry and recorded units as failed. Fix 9e182505 (bounded timeouts,
   retry budget after `/healthz`, batched submissions of 2,000 rows with in-store readback) plus the
   owner's unit change MemoryMax → 6 GiB. Receipt `DEPLOY_RECEIPT_9e182505bad9.json`, then
   7cdf4660 (STATUS heartbeat inside long passes, `DEPLOY_RECEIPT_7cdf46606d1f.json`).
4. **False PHASE_3_COMPLETE status**: the status module promoted a phase from any closure file in
   any state root (a throwaway-store closure from the redeploy side effect counted). Fix 9e182505:
   phase derived only from the population's digest-valid closure in its own state root; STALLED
   follower state when the heartbeat is older than max(10 min, 3 passes). Side effect recorded in
   `INCIDENT_RECORD.json` (runner.env rewritten by a redeploy without `FS23_WAREHOUSE_OVERRIDE`);
   owner fixes f5fbe4ad / abac6600 keep the live URL intact through CLI parsing.
5. **Snapshot copy stopped by the admission monitor** (this closing step): the migrate tool's
   measurement of the 10.6 GB boundary reached a 4.27 GiB tree peak under a 4 GiB cap and was
   stopped (`PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND`, admission incident
   `fs23-snapshot-copy-1791332447-2151252-39321e`); the post-CHECKPOINT copy was kept and verified
   on the phase-2/3 boundary instead (§4). Store downtime 24 min; it restarted clean (WAL folded,
   0 restarts).
6. **Compression stopped by the admission monitor** (same step): `zstd -12 -T1` over the 10.6 GB copy
   on the coordinator reached a 3.10 GiB tree peak under a 3 GiB cap (the cgroup charges the page
   cache of the file read) with host pressure above the respond line
   (`fs23-snapshot-zst-1791334029-2179796-262127`). Per the memory-aware placement rule the copy
   was moved to worker_b (574 GB free, 11 GiB admission headroom) and the asset built and verified
   there (`crispdm-run -m 6G`); no further heavy I/O was run on the owner's desktop.

## 8. Next exact object

**Phase 4 extractibility DESIGN** — raw / random-encoder / trained-encoder extractibility on the
phase-3 candidate subsets (per population × target × method × K) with matched controls
(ALL_ADMISSIBLE, UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K), frozen populations from
`CANDIDATES_FOR_VALIDATION.json`. Design only: not started, no training, no GPU, no test split.
