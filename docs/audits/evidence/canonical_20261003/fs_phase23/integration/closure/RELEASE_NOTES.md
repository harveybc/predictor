# PHASE_2_3 OLAP snapshot — feature selection phases 2 and 3 (closed 2026-10-06)

Checkpointed copy of the governed DuckDB OLAP cube taken 2026-10-07 with the store stopped
(exclusive open + CHECKPOINT), compressed with zstd level 12.

- `cube_phase2-3_20261007.duckdb.zst` — SHA-256 `89d19f47b308bcf9f24853369ab9870c784cc22121ab7a300f0116609394e107` (3,562,568,823 bytes)
- decompressed `cube_phase2-3_20261007.duckdb` — SHA-256 `dd3ce4324fa769190e8087aec0225e6b07933ffbe3ec1f73d02a784a21c73333` (10,620,252,160 bytes)
- `SNAPSHOT_MANIFEST.json` (schema `olap_snapshot_manifest.v2`) carries per-relation row counts and digests.

Runs inside: `phase1-eurusd-final:94d20c038d55e152` (EURUSD: 66,795 pairs; 7,213,860 metric rows,
1,202,310 stability rows, 1 alias group, 277 redundancy clusters, 45,990 filter ranking rows, 686
candidate subsets), `phase1-final:ETH:29d2f745f5d9e87c` (ETH: 3,403 pairs; 245,016 / 61,254 / 5 / 38 /
4,212 / 294) and the one-row smoke run `phase2-smoke:2026-10-06`. Live-cube reconcile for both runs
was `complete: true` and every table digest equals the one sealed in `PHASE_2_COMPLETE.json` /
`PHASE_3_FILTER_COMPLETE.json`; the copy's reconcile equals the live one table by table.

No predictive winner is declared and the test split was never read: phase-3 outputs are candidates
for wrapper validation. Evidence and the single return: predictor
`docs/audits/evidence/canonical_20261003/fs_phase23/RETURN.md`.

Restore: verify the asset SHA-256, decompress (`zstd -d`), verify the decompressed SHA-256, then open.
