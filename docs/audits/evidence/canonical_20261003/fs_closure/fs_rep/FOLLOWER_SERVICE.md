# FS-REP follower service

Roles only; no host names. The follower runs on the coordinator because it is
pure-python aggregation (no TensorFlow import, measured peak 134 MB) over the
coordinator's read-only mirror of the PS3-R result trees that
`feature-selection-sync.service` refreshes every 60 s. It never touches a GPU
job, never reads targets, VALIDATION or TEST, and never writes outside
`fs_closure/fs_rep/` and its own state directory.

## Unit

Transient `systemd --user` unit `fs-rep-follower.service` inside
`crispdm-batch.slice`, `MemoryMax=256M`, `MemorySwapMax=0`,
`Restart=on-failure`, started from the canonical-exec worktree:

```
systemd-run --user --unit=fs-rep-follower --slice=crispdm-batch.slice \
  -p MemoryMax=256M -p MemorySwapMax=0 -p Restart=on-failure -p RestartSec=60 \
  --working-directory=<canonical-exec worktree> \
  <python> tools/fs_rep_dispositions.py \
  --config docs/audits/evidence/canonical_20261003/fs_closure/fs_rep/fs_rep_config.json
```

Poll: 120 s. Each cycle re-reads every planned cell (411 = 137 x 3 terminals),
caches each admitted terminal's reduction under its `results_sha256`, applies
`DECISION_RULE.md` (`fs_rep_rule.v1`) and rewrites:

| file | content |
|---|---|
| `representation_dispositions.csv` | 822 rows = 137 heavy candidates x 6 families; every metric cell MEASURED / FAILED / NOT_APPLICABLE / PENDING with the receipt digest |
| `candidate_decisions.csv` | one row per candidate: decision, families present, flags |
| `progress.json` | coverage over 137, families present per candidate, terminals by role, ETA per GPU queue and for full coverage, output digests, process peak RSS |
| `representation_metric_catalog.csv.gz` | long-format metric catalog slice for the grouping phase (committed at the first table and at full coverage; the uncompressed copy and its digest live in the state directory and `progress.json`) |

When the output directory differs from HEAD the follower commits only that
directory (`git commit -- fs_closure/fs_rep`), pulls with `--no-rebase` and
pushes; outcomes are appended to `<state>/git_log.jsonl`. Nothing else in the
worktree is staged or touched.

## Local, uncommitted inputs

`<state>/local_paths.json` maps the mirror role keys used by the config
(`mirror_alternatives`, `mirror_baseline_002`, `mirror_baseline_successor`) and
the queue STATUS keys to directories on the coordinator, so the repository
never carries worker directory names.

## Refit input (gate G4)

`refit_input` points at FS-CLOSE's `fs_closure/fs_close/refit_gain_export.csv`
(repo-relative; it fills as the 137 ALL_MINUS removal refits land on the
secondary worker). Rows without a finite `refit_gain` are skipped until they
land. The export is identity-only (trained-family refits are not materialised,
declared in the file), so per rule v2 the identity row of each candidate shows
`refit_gate_applied = true` and its `refit_gain` once landed, with flag
`RAW_REFIT_GAIN_POSITIVE` / `RAW_REFIT_GAIN_NONPOSITIVE`; trained rows keep
`G4_refit = NOT_APPLICABLE`, `refit_gate_applied = false`. The follower picks
the file up on its next cycle (it is pulled into the worktree with the rest of
the branch); no restart is needed.

## ETA source

`fs_closure/fs_gpu/ps3r_eta.json` when FS-GPU publishes it; until then the
three queue STATUS files (median / nearest-rank p90 of observed `wall_seconds`,
one worker per queue). The successor queue has no own sample until its first
cells close, so its durations are proxied by the baseline queue on the other
GPU and the proxy is declared in `progress.json`.

## Stop / restart

`systemctl --user stop fs-rep-follower` stops it; re-run the command above to
restart. A second instance exits 75 on the state lock.
