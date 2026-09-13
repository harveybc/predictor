# C166-C184 evidence tools

Scripts that produced custody evidence for the D2 reattestation. They run from the predictor checkout root
(they import `tools/` by path), under `crispdm-run`, and write only into write-once state roots. Hosts appear
by role only.

| Script | Order item | What it produced |
|---|---|---|
| `scan_short_fit_stretch.py` | C172 | read-only scan: every MCAR unit of the C137 bank has a complete TRAIN stretch under 50 rows |
| `reseal_d2_design_v2.py` | C171 | `d2_design_c171_v2`: the design resealed after the worker fix; refuses unless only the worker digest moved |
| `build_d2_shard_jobs.py` | C172, C174 | shard directories of unit symlinks and the dispatcher jobs file |
| `reconcile_c172_dispatch.py` | C174 | `d2_dispatch_c172_v1_reconciliation`: the first reanalysis dispatch reconciled against physical units |
| `reconcile_dispatch.py` | C174 | reconciliation of a stopped dispatch (generic form of the above) |
| `c172_compare.py` | C172 | split by family, compare with C137 in capped processes, merge: `C172_COMPARISON.json` |
| `c172_table_dir.py` | C172, C178 | the reanalysis and comparison rows as a loader table directory |
| `d2_fresh_adjudicate.py` | C174-C177 | collect the fresh shard roots into one root, split by family, decide, merge |
| `c179_submission.py` | C179 | the review submission binding design, code, tape, roots, decisions and load |
