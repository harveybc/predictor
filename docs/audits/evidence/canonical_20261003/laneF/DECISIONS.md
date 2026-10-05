# Lane F (PS3-R, alternative extractor families): decisions log

Roles only; no host names, IPs or GPU identifiers. Times are UTC.

## D1. Probe rows are identical to lane E's (2026-10-03)
Lane E fixed `--max_fit_windows 16384 --max_val_windows 0 --max_ref_windows 256 --probe_lags 0,1,2,23`
(laneE ROWS_CONTRACT) because full-row folds exceed the shared 8 GiB batch slice: lane F's own
full-row probe at a 6 GiB cap was stopped by the admission monitor under sustained memory pressure
in fold 3 of 5 (tree peak 6.02e9 B). Lane F adopts the same arguments so every probe cell uses
the same rows as lane E. Same code for both lane F families, seed 0, window 168, D=8.

## D2. P2C trains its pretext on all consecutive anchors; probes keep the shared rows
At code 8987c57 the pilot refused any `--max_*_windows` for `past_to_current_siamese`, which made
it incomparable with lane E's rows and infeasible in the slice. Lane F commit
feature-extractor `satoshi/laneF-p2c-probe-rows-20261003` @ 31add12 (pilot only, +1 test; 50 tests
pass on CPU): when windows are subsampled, P2C trains on all consecutive fold anchors (pairs are
sampled per epoch) while encodings, probes and every metric use the shared subsampled rows. Per
fold `pretext_support` records the pretext training windows and `current_window_start_offset_steps`
= max_lag = 3*T = 504 h: P2C current windows start 504 hourly steps after the fold's first fit
anchor (its later data start, declared for the paired comparison). MTAE behaviour is unchanged;
all lane F runs use 31add12 so both families share one code identity.

## D3. Measurement cap request mis-chosen at 8 GiB; relaunched once at 7936M (coordinator approved)
- Old request: `laneF-measure-mtae`, `-m 8G`, queued with code HOST_HEADROOM; later found
  structurally inadmissible (SLICE_AGGREGATE_BUDGET): 8 GiB equals slice_memory_max and the slice
  carries ~53 MB of baseline charge, so it could never be admitted. A smaller cap under the same
  name was refused by the launcher (CAP_LOWERED_AFTER_REFUSAL).
- Why this is not lowering a measured need: no work had run at either cap; 8 GiB was a probe-cap
  mis-choice.
- Action: request cancelled through the admission `cancel` verb (no kill); relaunched ONCE as
  `laneF-measure-mtae-r2` at `-m 7936M` (highest admissible). Coordinator ruling: APPROVED with
  conditions: if r2 is pressure-stopped or OOM-killed, no lower retry (MTAE -> BLOCKED_BY_SLICE);
  production cap = 1.25 x measured cgroup peak only if it fits; otherwise reported, margin not shaved.
- Scheduling: lane E and lane F alternate job by job on the slice (each job needs ~7-8 GB of a
  14.6 GiB host); concurrent execution is impossible within 8 GiB. A larger slice is an
  owner/hardware decision surfaced by the coordinator.

## D4. Measured caps (one real feature, batch_001 px.close_loc, --max_epochs 1, rows as D1)
| family run (identity+random+family) | peak RSS (B) | cgroup peak (B) | cap = 1.25 x max | wall (s) |
|---|---|---|---|---|
| masked_temporal_ae (r2, admitted at 7936M) | 4,976,996,352 | 4,779,393,024 | 5940M | 91 |
| past_to_current_siamese (admitted at 7936M) | 5,591,740,416 | 5,401,935,872 | 6670M | 115 |
Both caps fit the slice; neither margin was shaved. The driver reads one cap per family and runs
ONE feature per process, yielding to a waiting lane E request at every boundary.
