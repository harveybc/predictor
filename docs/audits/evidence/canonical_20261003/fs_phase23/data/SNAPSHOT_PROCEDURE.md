# Physical snapshot procedure (`.duckdb.zst` + SHA-256 + release asset)

The repository keeps manifest, schema, SHA-256 and URL; the binary is a GitHub release
asset (master plan v3 section 15). The phase-1 precedent is
`docs/audits/evidence/PHASE1_FEATURE_SELECTION_20261005/SNAPSHOT_MANIFEST.json`
(release `phase1-feature-selection-20261005`, assets uploaded by the owner's account).

## 1. Take a verified copy (never compress the live file)

```bash
python tools/olap_duckdb_migrate.py snapshot \
  --source <live-cube.duckdb> --target <snapshots>/<name>.duckdb \
  --expect-terminals <count the service reports> --out <snapshots>/SNAPSHOT.json
```

It copies cube + WAL, `CHECKPOINT`s the copy and verifies counts. A verified boundary needs
`--owner-stopped`; that stop/start belongs to the integration agent.

## 2. Compress, digest, manifest

```bash
python tools/fs_phase23_warehouse.py snapshot \
  --source <snapshots>/<name>.duckdb --out-dir <snapshots>/<release-dir> \
  --tag phase2-feature-selection-<date> --phase FEATURE_SELECTION_PHASE_2 \
  --repo harveybc/predictor --level 19
```

Refuses a source with a non-empty `.wal` beside it (a live store). Writes
`<name>.duckdb.zst`, `<name>.duckdb.zst.sha256` and `SNAPSHOT_MANIFEST.json`
(`olap_snapshot_manifest.v2`: both sizes and SHA-256s, per-relation row counts and
content digests for the six fs_phase23 relations, release and asset URLs, the exact publish
commands). Copy the manifest into `docs/audits/evidence/.../fs_phase23/` and commit it.

## 3. Publish

```bash
gh release create phase2-feature-selection-<date> --repo harveybc/predictor \
  --title 'Feature Selection Phase 2 OLAP Snapshot' --notes-file RELEASE_NOTES.md
gh release upload phase2-feature-selection-<date> <name>.duckdb.zst <name>.duckdb.zst.sha256 \
  --repo harveybc/predictor --clobber
```

Checked read-only on 2026-10-05: `gh` is authenticated on the coordinator as the
repository owner's account (keyring token) and the phase-1 release is visible with its two
assets. Creating a release and uploading an asset writes under that owner credential; this
agent did not run those commands (there is no phase-2 snapshot yet) and will not run them
without the owner's say-so. `--publish` on the `snapshot` subcommand runs the two commands
above verbatim when that authorization exists.

## 4. Verify the published asset

```bash
gh release download phase2-feature-selection-<date> --repo harveybc/predictor -p '*.duckdb.zst' -D <work>
python tools/fs_phase23_warehouse.py verify-snapshot --asset <work>/<name>.duckdb.zst \
  --manifest SNAPSHOT_MANIFEST.json --work-dir <work>/restore
```

Asset SHA-256, decompressed SHA-256 and per-relation counts must all match the manifest
before the file is opened for any reading. A snapshot nobody read back is a belief.
