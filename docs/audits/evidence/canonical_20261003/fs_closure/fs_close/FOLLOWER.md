# FS-CLOSE follower (order 2026-10-05 section 8)

Roles only: coordinator, worker_a (RTX 5090 host), worker_b (CPU refit host).

## What runs where

| Unit | Host role | What it does |
|---|---|---|
| `feature-selection-status-v2.service` (transient; FS-GPU's v2 of the coordinator status loop, which replaced v1 on 2026-10-05) | coordinator | the existing status loop, extended with one step at the end of its cycle: `tools/fs_close_manifest.py follow-once --worker <worker_b alias>` under `crispdm-run -m 1G`, every cycle (about 2 min); `--git-commit` is added at most every 30 min. The same block was first added to v1 (`selection_status_loop.sh`) and must be carried into any later rewrite of the loop (`grep FS-CLOSE` the live script) |
| `fs-close-refit-<epoch>.service` (transient) | worker_b | `crispdm-run -q -W 14400 -m <cap> -n <job>` around `tools/fs_close_refit.py`; sequential, CPU only, never `/tmp`, never a GPU job |

The loop script lives in the coordinator's state directory (outside the repository); it was
extended, not duplicated. A second status writer was not added. The block to carry over:

```bash
  # ---- FS-CLOSE closure follower (order 2026-10-05 section 8) -------------------------------
  FS_CLOSE_STAMP="$STATE/fs_close/last_commit_epoch"; mkdir -p "$STATE/fs_close"; FS_CLOSE_GIT=""
  if [ ! -f "$FS_CLOSE_STAMP" ] || [ $(( $(date +%s) - $(cat "$FS_CLOSE_STAMP") )) -ge 1800 ]; then
    FS_CLOSE_GIT="--git-commit"; date +%s > "$FS_CLOSE_STAMP"
  fi
  CRISPDM_PYTHON="$PYTHON" "$HOME/.local/bin/crispdm-run" -m 1G -t 30m -n fs_close_follow -- \
    "$PYTHON" "$REPO/tools/fs_close_manifest.py" follow-once --worker <worker_b ssh alias> $FS_CLOSE_GIT \
    > "$STATE/fs_close/last_cycle.json" 2> "$STATE/fs_close/last_cycle.err" || true
```

## First measurements (pilot, largest inner fold `inner_2023`, seed 0, identical rows per cell)

`pilot_receipt.json`: 56 cells (ALL_ADMISSIBLE and RANDOM_K:24, heads ridge and hgb), 273 s,
whole-process peak RSS 972.7 MB; cap for the full run = 1.25 x = 1216 MB (`refit_cap.json`).
Skill = 1 - loss / stricter same-row naive (zero return or fit mean for Y_s/Y_l; fit prior for
Y_b), mean over the fold's cells; every row carries its paired naive.

| set | head | Y_s (6 h) | Y_l (6 h) | Y_b h6 | Y_b h144 |
|---|---|---:|---:|---:|---:|
| ALL_ADMISSIBLE (366) | ridge | -0.062 | -0.366 | +0.162 | -0.798 |
| ALL_ADMISSIBLE (366) | hgb | -0.448 | -0.904 | +0.169 | -0.115 |
| RANDOM_K:24 | ridge | -0.0017 | -0.019 | +0.002 | +0.021 |
| RANDOM_K:24 | hgb | -0.0095 | -0.054 | +0.010 | -0.063 |

No raw-input head beats the same-row naive on any Y_s/Y_l cell of this fold; only the 6 h
barrier cell shows skill (log-loss 0.816 vs prior 0.974). This agrees with FS-REP's finding on
the PS3-R probes and is reported, not smoothed: a selection that cannot beat its naive on a
target is marked `strategy_eligible = false` in the manifest.

## One cycle of the follower

1. population: 366 model-input candidates from lane A (`role == feature`) minus the 37 selector
   episode sources of the role overlay; any other count stops the cycle (`POPULATION_MISMATCH`);
2. lane evidence: FS-PRED rankings/selector sets, FS-CAUSAL `causal_evidence.jsonl`, FS-REP
   `candidate_decisions.csv` (+ the 822-row controls table), FS-GEN `generative_evidence.csv`,
   PS2 status, PS3-R terminals (mirror), PS4 profiles, GPU ETAs (`fs_gpu/ps3r_eta.json`, else
   the live queue status);
3. refit plan: ALL_ADMISSIBLE, RANDOM_K control, every complete FS-PRED method at
   K in {8,16,24,32,48}, PRED_BEST / PLUS_CAUSAL / PLUS_REP arms when their inputs exist,
   KNOCKOFF only if calibrated (recorded as NOT_CALIBRATED/EMPTY otherwise, never dropped),
   and one removal refit per heavy candidate (feeds FS-REP gate G4 through
   `refit_gain_export.csv`);
4. worker dispatch: push plan + engine, pull results; a bounded pilot (largest fold, two
   heaviest sets, cap 1500M, own job name) measures the footprint first; the full run then
   uses 1.25 x the measured whole-process peak under a job name that carries that cap
   (the launcher refuses a lowered cap under one name, so the cap is monotone by construction);
5. dispositions: one automatic row per candidate (SELECTED / REJECTED / PENDING + reason codes
   + evidence digests); rejection needs positive evidence; NOT_IDENTIFIED is neutral;
6. fail-closed checks C1-C10 (+ C11 gate acceptance when FINAL);
7. artifacts: DRAFT manifest (schema `feature_selection.manifest.v1-DRAFT`, refused by the gate
   by construction) listing the missing objects, or FINAL manifest + separate decision record
   (`feature_selection.decision.v1`, decider role != producer role) that passes
   `tools/selected_manifest_gate.py` (byte-exact c0345f83, digest pinned);
8. `FEATURE_METRICS_CATALOG.parquet/.csv.gz` (rebuilt only when an input changed),
   `STATUS.json`, `PROGRESS.png`, `MASTER_MILESTONE_STATUS.json` (M2 regenerated; other
   milestones kept verbatim) and `MASTER_MILESTONE_PROGRESS.png`.

## Closure step (declared before any VALIDATION read)

`closure_rule.json`. The frozen K=24 sets are refit on all TRAIN rows and scored once on
EXTERNAL VALIDATION rows (2024). The winner is the highest mean skill over the 14 cells
against the stricter paired naive; ties go to parsimony then cost. The loader refuses any row
at or after 2025-01-01; TEST is never read. The step cannot run until VALIDATION features
for the 366 candidates and their targets are materialised by the PS1 producer (missing object
`VALIDATION_2024_FEATURES_AND_TARGETS`), so the manifest stays DRAFT until then.

## Memory

Coordinator aggregation peak RSS is written in `STATUS.json` (`aggregation_peak_rss_bytes`);
the PS2 reader is chunked so a cycle stays under the 256 MB coordinator budget when the
catalog is rebuilt. Worker footprint: `refit_receipt.json` and `refit_cap.json`.
