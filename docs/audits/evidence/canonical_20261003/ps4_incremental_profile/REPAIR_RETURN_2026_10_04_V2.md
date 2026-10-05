# PS4 incremental profile: second audit repair

## Scope

This repair closes the three independent audit findings against commit
`5481dcaa`. It does not run a profiler, retrain a model, read a target, or
change any retained metric row.

## PRE

The audit mutations were frozen before implementation. The focused suite
reported 4 failures and 16 passes:

- a coherent metric rewrite could recompute both internal hashes;
- the terminal manifest did not authenticate retained `results.jsonl` bytes;
- report aggregation did not require an external accepted-unit identity;
- one legacy assertion still expected the former synthetic terminal hash.

## POST

- `classify_unit()` can return `MEASURED` only when the complete unit file
  matches an externally supplied acceptance entry.
- `accepted_units.json` binds each accepted unit's complete-file digest,
  feature, fold, source digests, and terminal identity.
- the acceptance index itself is accepted only under a caller-supplied SHA-256;
  the report records that digest.
- `load_terminal_identity()` opens the retained `results.jsonl`, rejects links
  and absent bytes, hashes the file, and compares it with the terminal
  manifest.
- `count_units()` and `publish_report()` consume the authenticated index. A
  coherent rewrite of metrics or terminal provenance becomes `PENDING`.

The original 105 metric rows remain byte-for-byte equal at the scientific-row
level: their five `rows_sha256` values are unchanged. The report remains
`MEASURED=5`, `PENDING=1`, `FAILED=0`.

## Retained evidence

- PS3-R results: 441 rows, SHA-256
  `ec16a107f0ecc2e5c86002e15a2397e214e2d7c0a9df1145048f678e14546f8a`.
- accepted-unit index SHA-256:
  `42413a6d3f1757ee79eb7ef1d5e055e8fbc7f4a0a717a03fb41fb9a6ba8760a6`.
- focused suite: 20 passed.
- `git diff --check`: clean.

## Changed evidence surface

- `run_ps4_incremental_profiles.py`
- `test_run_ps4_incremental_profiles.py`
- `accepted_units.json`
- retained terminal `run_manifest.json` and `results.jsonl`
- five accepted unit envelopes, identity fields only
- `REPORT.json`

No GPU process or other worktree was touched.
