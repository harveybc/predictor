# Satoshi — RP98–RP105 return: verifications corrected, memory-bounded evaluation without reducing the model, the finite population (12 cells executed, 6 verified and deleted, 6 retained), predictions deleted after verified metrics

Orders: [MUSASHI_SOTA_RP98_RP105_2026_09_22](../../handoffs/MUSASHI_SOTA_RP98_RP105_2026_09_22.md) (fff9575, review of fb6ec31e).
Commits after fff9575: 2d42784 … 279a2abd (45 commits). Compute: WORKER_A (dragon, RTX 4090 Laptop) for T = 96 (RP95) and the test
suites; WORKER_B (gamma, RTX 5090, its own cooler, authorized by the owner after gamma's reboot) for T = 192 s2022/s2023, T = 336 ×3,
T = 720 ×3 and the consolidated closure, one cell at a time under an 8 GiB request; omega under the travel hold (marker present:
text, git and one package install only — no compute, no tests, no service restart). Every governed unit registered before reading;
every failed or interrupted attempt closed FAILED and kept as a versioned folder.

## Owner-facing table — official normalized MSE / MAE first
Population: 12 cells (L = 96 × T ∈ {96, 192, 336, 720} × seeds {2021, 2022, 2023}), design 9b49010d…, official ECL (TSL,
electricity.csv sha256 7e45845d…, 26304 × 321), TimeFilter @ dffde87e, author epochs (15, patience 3, batch 16, lr 1e-3),
metrics in the normalized space, author reduction. Frozen operational margin: |mean − published| ≤ 2·σ_paper + 0.0005
(0.0105 MSE / 0.0125 MAE). Generated from the consolidated closure on WORKER_B
([SOTA_TABLE.gamma.md](../evidence/d3_k5_20260917/RP103/SOTA_TABLE.gamma.md), [REPORT](../evidence/d3_k5_20260917/RP103/REPORT.gamma.pre_deletion.json)).

| T | published MSE / MAE | executed per seed 2021 / 2022 / 2023 (MSE ; MAE) | verified mean (sd, n) | difference | matched persistence / seasonal-24 (z-MSE / z-MAE) | training | cost | replay in this closure (RTX 5090) | agreement |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.133 / 0.230 | 0.13332 / 0.14008 / 0.13309 ; 0.23055 / 0.23778 / 0.23032 (WORKER_A, RTX 4090 L; author float32) | **NO_MEASUREMENT in this closure** (n = 0 verified; shown apart, not pooled) | — (the RP97 three-seed mean 0.13550 / 0.23288 was +0.0025 / +0.0029) | 1.5878 / 0.9455 ; 0.3211 / 0.3258 (RP97 catalogs) | 15/15 epochs, best 15 ×3 | 3 × 34 min, 6.5 GiB RAM | cross-device (4090 → 5090) **FAILS the pointwise rule**: max|Δ| 1.9e-4 / 2.2e-4 / 2.7e-4 > atol/rtol 1e-4, exact fractions 0.114 / 0.118 / 0.109; replayed author metrics **bit-identical**; same-device history on WORKER_A bit-identical (HISTORICAL_RECORD_ONLY) | UNVERIFIED here: needs the 3 × ~22 s same-device replays on WORKER_A's GPU (owner: cooler) |
| 192 | 0.154 / 0.248 | 0.16086 / 0.15863 / 0.15345 ; 0.25599 / 0.25243 / 0.24807 (author float32) | **0.15765 (0.0038) / 0.25216 (0.0040), n = 3** | +0.0036 / +0.0042 | 1.5962 / 0.9507 ; 0.3043 / 0.3237 (5069 windows) | 15/15 each, best 14 / 15 / 12 | s2021 82 min omega (RTX 4070 L, 10.0 GiB); s2022/s2023 15.5 min each on the 5090, 7.1 / 7.0 GiB RAM (workers = 1) | s2021 cross-device (omega → 5090) PASS max|Δ| 4.8e-6, exact 0.256; s2022/s2023 same-device PASS max|Δ| 0.0, exact 1.0 | **OPERATIONAL_AGREEMENT** (0.0036 ≤ 0.0105; 0.0042 ≤ 0.0125) |
| 336 | 0.162 / 0.261 | 0.16259 / 0.16818 / 0.16267 ; 0.26067 / 0.26741 / 0.26087 (float64 basis: the author's float32 reduction needs 6.4 GB of temporaries, budget 4 GiB; float64 − float32 ≤ 2.1e-8 wherever both exist) | **0.16448 (0.0032) / 0.26298 (0.0038), n = 3** | +0.0025 / +0.0020 | 1.6178 / 0.9613 ; 0.3273 / 0.3427 (4925 windows) | 15/15 each, best 10 / 13 / 12 | 15–15.6 min each on the 5090, 5.3 GiB RAM (workers = 0) | same-device PASS ×3, max|Δ| 0.0, exact 1.0 | **OPERATIONAL_AGREEMENT** (0.0025 ≤ 0.0105; 0.0020 ≤ 0.0125) |
| 720 | 0.184 / 0.284 | 0.18714 / 0.19817 / 0.18537 ; 0.28752 / 0.29865 / 0.28568 (float64 basis, 12.6 GB needed) | **NO_MEASUREMENT in this closure** (terminals not yet accepted: custody UNCHECKED; shown apart) | — | — (no verified catalog yet) | 15/15 each, best 12 / 5 / 11 | 20 min each on the 5090, 2.7 GiB RAM (workers = 0, allocator patch) | not replayed (custody first) | PENDING: warehouse restart by the owner → `report` ×3 → closure |
| avg | 0.158 / 0.256 | — | NOT_COMPUTED (no seed has all four horizons verified) | — | — | — | — | — | — |

## What was done, per block
* **RP98** — PRE frozen: Musashi's probe rerun unchanged on dragon at fff9575 (20 leaves, 0 differences,
  [RP98_PRE](../evidence/d3_k5_20260917/RP98_PRE/PROBE_RESULTS_PRE.json)). POST on the repaired tree, his steps repeated by
  [reproduce_post.py](../evidence/d3_k5_20260917/RP98/reproduce_post.py) →
  [REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP98/REVIEW_REPRODUCED_POST.json): the altered vault is refused
  (VAULT_CHANGED, candidate preserved, successor recomputed from the accepted arrays and read back; it verifies at the next closure);
  the contradictory cached replay is never adopted (replays re-run every closure; max|Δ| 0.0 measured, exact fraction 1.0); the short
  loader is a typed refusal (INCOMPLETE POPULATION); constant correlation is None with an UNDEFINED catalog state; the four-horizon
  average is formed within each seed (SD 0.0, n = 3 on his oracle). Each became a rule of the actual closure and metric path
  (`tests/test_df_sota_repro.py`, 31 passed + 1 skipped in its own process on dragon). No re-fit; the four measured cells and the
  original CPU replay failures are preserved.
* **RP99** — catalog v3 (`df_sota_metrics_vault.v3`): the loader must yield exactly the predictions' windows, in order, with the
  cell's horizon and channel count, finite; the population is the count consumed; identity block (pred/true/arrays/checkpoint/record
  digests, scaler digest, metric-implementation digest, row order, numeric space); explicit catalog states DONE / APPROXIMATE /
  UNDEFINED / NOT_APPLICABLE with parameters, excluded counts and reasons; per-window series and weekly time blocks; paired seed
  and baseline contrasts computed at closure from the persisted series; `mae_relative_to_test_persistence` (not MASE);
  `mape_zspace`/`mspe_zspace` labelled as z-space ratios; undefined is None. Independent numeric oracles per family in the tests.
  Catalogs recomputed on every closure from the accepted arrays; a persisted candidate of an older schema is SUPERSEDED (kept),
  one of the same schema that differs is REJECTED (kept) and never verifies; Musashi's independent reduction agrees with the
  overlapping globals (his scope only).
* **RP100** — replays are re-run every closure (no cache as input), outputs validated (finite, shape, count, full pointwise
  comparison, exact-equality fraction, metric reductions, targets), bound to inputs, environment, device UUID and the replay code;
  four separate fields per cell: recipe fidelity, same-device repeatability, cross-device portability, published-score
  agreement. Route-level diagnostic on the three T = 96 cells
  ([ROUTE_TRACE.*](../evidence/d3_k5_20260917/RP98/)): 0 route flips out of 59,351,616 routes per gating block, 0 cut-index changes,
  gating-probability differences ≤ 8e-5, minimum distance to the top-p threshold ~8e-5, same-device repeat 0.0 → the CPU/GPU
  discrepancy is NUMERICAL_ONLY (kernel-level differences amplified through the network); the routing hypothesis is not supported.
  The old CPU failures remain failures of the original rule; permuted predictions with equal aggregates fail (tested).
* **RP101** — bounded evaluation adapter (`df_sota_bounded_eval.v1`): the author's forward pass, loader, batch composition,
  order and float32 are untouched; predictions and targets stream to memmaps, targets are hash-streamed, inputs are not retained,
  the author's float32 `metric()` runs on the memmaps only within a declared budget and the float64 reduction always; the
  adapter's source digest is recorded per cell and the author clone stays pinned and clean. Parity on the real tiny author
  fixture: identical predictions, targets, population, checkpoint bytes and author metric; partial final batches consumed;
  incomplete/extra loaders and a missing checkpoint refuse. Profiles on dragon under an 8 GiB request
  ([EVAL_PROFILE.*](../evidence/d3_k5_20260917/RP98/)): cgroup peak incl. file pages 3.77 / 6.30 / 6.45 / 7.73 GB for T = 96 / 192 /
  336 / 720, disk 1.3 / 2.6 / 4.3 / 8.5 GB, VRAM 0.9 GB, 16–29 s; the author's float32 reduction is NOT executed within a 4 GiB
  temporaries budget for T = 336 / 720 (needs 6.4 / 12.6 GB) → those cells carry the labelled float64 basis (difference vs the
  author reduction ≤ 2.1e-8 on every cell where both exist; never assumed bitwise).
* **RP102** — the eight missing cells of Protocol A were trained with the author's epochs on WORKER_B's RTX 5090 (its own
  cooler; the owner's authorization after gamma's reboot) under an 8 GiB request, one cell at a time with the thermal guard
  between cells (GPU 38 → 54–69 °C during training, no driver slowdown), `--bounded --author-metric-budget-gib 4
  --dataloader-workers 0`; every attempt registered before reading, every failed or interrupted attempt closed FAILED and kept as
  a versioned folder (`retire-attempt`): dragon's T = 192 s2022 stopped at the owner's cooler concern; a duplicate of T = 192
  s2021 on gamma (horizon filter) interrupted at 2 min; T = 336 s2021 killed twice by gamma's memory-pressure killer (7.1 GB
  scope: first the page cache of the memmapped evaluation, then a 24-min stall at epoch 2) and stopped once by me at 7.1 GB — the
  diagnosis was the DataLoader worker process (num_workers = 1 forks the dataset: 4.5 GB of private pages beside the 4.9 GB main
  process); the declared operational option `--dataloader-workers 0` (batch order and content unchanged, tested) brought the
  scope to 4.3–5.3 GB. A reporting defect surfaced on the first T = 336 cell: the terminal was built from the author's float32
  reduction, which is None when it is not executed within the budget (T = 336/720 need 6.4/12.6 GB of temporaries) — a TypeError
  after the arrays, record and checkpoint were written; fixed (terminal from the record's basis, basis tagged) and the two
  affected cells were reported from their records by the new `report` subcommand, without re-training. The T = 720 cells
  exposed a second host-memory limit: after epoch 1 (1100 iterations at 0.046 s each) the main process grew from 2.4 to 7.3 GB of
  anonymous memory within 15 s at the end-of-epoch validation and sat throttled at the slice ceiling (GPU idle) — the author's
  `vali()` moves every batch's outputs and targets to the host to score them (14.8 MB each, 404 batches, ~6 GB in seconds) and
  glibc's dynamic mmap threshold keeps those freed buffers resident; profile runs (scratch, nothing kept) showed the jump with the
  default allocator and a flat 2.4 GB with fixed malloc thresholds (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:
  glibc.malloc.trim_threshold=2097152`), the CPU loader alone being unaffected. The setting is a declared operational patch of
  cell and replay processes (host memory only, no arithmetic; recorded per cell and per replay; `--malloc-tunables none` turns it
  off; tested). Two T = 720 attempts (one after epoch 1, one at start) were closed FAILED and retired before the relaunch.
  All twelve cells are executed; nine terminals are accepted; the three T = 720 terminals
  are pending the owner's warehouse restart (below).
* **RP103** — the consolidated closure ran on WORKER_B (memory-bounded closure: predictions streamed chunk by chunk from the
  stored or compressed npz member, targets streamed by the author's loader to a file whose digest is checked against the record,
  the frozen replay rule evaluated per chunk in float64; a T = 720 closure would otherwise have held ~17 GB) after merging
  WORKER_A's T = 96 cells and omega's T = 192 s2021 (worker replay histories carried as HISTORICAL_RECORD_ONLY). Every replay is
  re-run on the RTX 5090 (the actual CUDA device UUID is recorded — nvidia-smi index 0 on gamma is the 5070 Ti, a defect of the
  first two closure runs, fixed at aceb756d): the five gamma-trained T = 192/336 cells are same-device bit-identical (max|Δ| 0.0,
  exact fraction 1.0); omega's T = 192 s2021 passes cross-device (max|Δ| 4.8e-6, exact fraction 0.26); WORKER_A's three T = 96
  cells FAIL the pointwise rule cross-device (RTX 4090 Laptop → RTX 5090: max|Δ| 1.9e-4 / 2.2e-4 / 2.7e-4 against atol/rtol
  1e-4, exact fractions 0.11–0.12) while their author metrics replay bit-identically (0.13331516087055206 / 0.23055443167686462
  etc.) and their same-device GPU history on WORKER_A is bit-identical — so T = 96 stays UNVERIFIED in this closure until the
  66 s of same-device replays on WORKER_A's GPU that the owner has not yet cleared (cooler). Agreement (frozen operational
  margin 2·σ_paper + 0.0005): **T = 192 mean 0.157645 / 0.252163 vs 0.154 / 0.248 → OPERATIONAL_AGREEMENT** (three seeds, one
  cross-device); **T = 336 mean 0.164480 / 0.262985 vs 0.162 / 0.261 → OPERATIONAL_AGREEMENT** (three seeds, float64 basis
  labelled); T = 96 and T = 720 NO_MEASUREMENT in the closure table (executed values shown apart, never pooled); the four-horizon
  within-seed average NOT_COMPUTED (no seed has all four horizons verified). Paired contrasts from the persisted per-window
  series: seed pairs at T = 192 differ by 0.0036–0.0079 MAE (SD 0.017–0.032 over 5069 windows), at T = 336 by −0.0067–0.0065
  (SD 0.011–0.019 over 4925 windows); the model beats seasonal-24 by 0.068–0.082 MAE and persistence by ~0.70 on every window
  set. **Deletion (RP103, owner's authorization):** after the pre-deletion closure
  ([REPORT.gamma.pre_deletion.json](../evidence/d3_k5_20260917/RP103/REPORT.gamma.pre_deletion.json), sha256 fbe2cd4c…) the gate
  passed for the six verified cells (custody accepted, catalog recomputed and read back with the reported digest, independent
  check, required errors and baselines DONE, route diagnostics preserved where a history holds a failure, no reader, arrays hashed
  to the record) and refused the six others (T = 96 unverified in this closure; T = 720 without accepted terminal); 7 arrays
  deleted on WORKER_B (root + staging copy of T = 192 s2021; 11,161,227,264 bytes reclaimed —
  [receipt](../evidence/d3_k5_20260917/RP103/DELETION_RECEIPT.gamma.json)) and 2 on WORKER_A (root + `_from_omega` copy of T = 192
  s2021; 2,287,599,616 bytes — [receipt](../evidence/d3_k5_20260917/RP103/DELETION_RECEIPT.dragon.json)), each attempt left with a
  `PREDICTIONS_DELETED.json` that names the deleted paths, the arrays' former digest and the closure report it was verified under
  ([example](../evidence/d3_k5_20260917/RP103/PREDICTIONS_DELETED.L96_h192_s2021.gamma.json)); the catalog of that cell is in the
  evidence ([METRICS_VAULT.L96_h192_s2021.json](../evidence/d3_k5_20260917/RP103/METRICS_VAULT.L96_h192_s2021.json), 5069 windows,
  seasonal-24 0.3043 / 0.3237 z-MSE / z-MAE, persistence 1.5962 / 0.9507). Records, checkpoints and catalogs are never deleted.
  Omega's copies (`_from_dragon` T = 96 ×3, 1.8 GB — retained anyway as unverified; root T = 192 s2021, 1.1 GB — deletable)
  wait for the end of the travel hold: nothing was run on omega. The post-deletion closure (run 5) reports the deleted cells as
  METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION with their dated verification. __POST_DELETION__
* **RP104** — suites in compatible environments, at aceb756d on WORKER_A (no test ran on omega during the hold):
  `tests/test_df_sota_repro.py` 40 passed + 1 skipped (the skip: `seal()` needs the benchmark store receipt, present on omega
  only) in its own process; closure table + benchmark contract + block + financial acceptance 117 passed, 2 deselected (FL08
  needs omega's governance stack); lake adopters 4 skipped (need omega's deployed data-gov configuration; they passed on omega
  before the hold); the warehouse store suite (`olap/store/tests`) 106 passed + 1 skipped on SQLite AND DuckDB in a throwaway
  venv (duckdb 1.5.5, SQLAlchemy 2.0.54), including the migration of a 32-bit artifact-size column and the reopen of the
  migrated file. The 24 dragon-environment failures of the earlier full run (operators causality: `crispdm-run` not on the
  non-login PATH; governed_run classification: sibling preprocessor worktree absent; acceptance c1_c16: incremental_census
  absent) do not touch the reproduction path and were not re-run.
* **RP104** — suites in compatible environments, at aceb756d on WORKER_A (no test ran on omega during the hold):
  `tests/test_df_sota_repro.py` 40 passed + 1 skipped (the skip: `seal()` needs the benchmark store receipt, present on omega
  only) in its own process; closure table + benchmark contract + block + financial acceptance 117 passed, 2 deselected (FL08
  needs omega's governance stack); lake adopters 4 skipped (need omega's deployed data-gov configuration; they passed on omega
  before the hold); the warehouse store suite (`olap/store/tests`) 106 passed + 1 skipped on SQLite AND DuckDB in a throwaway
  venv (duckdb 1.5.5, SQLAlchemy 2.0.54), including the migration of a 32-bit artifact-size column and the reopen of the
  migrated file. The 24 dragon-environment failures of the earlier full run (operators causality: `crispdm-run` not on the
  non-login PATH; governed_run classification: sibling preprocessor worktree absent; acceptance c1_c16: incremental_census
  absent) do not touch the reproduction path and were not re-run.


## Exact remaining deficits
1. **T = 720 terminals** (3 cells executed, arrays/records/checkpoints on WORKER_B): refused by the OLAP warehouse — its
   `gov_terminal_artifact.bytes` column is a 32-bit INTEGER and a T = 720 array is 4,198,064,038 bytes. Repaired in the repo
   (93a25d10: BIGINT + in-place widening + CHECKPOINT; store suite 106 passed on SQLite and DuckDB) and INSTALLED into the DuckDB
   host's environment on omega, but my restart of `crispdm-data-warehouse-olap.service` was refused by the execution environment
   and is not bypassed. **Owner action:** `systemctl --user restart crispdm-data-warehouse-olap.service` on omega (optionally copy
   `~/.local/state/crispdm-duckdb/prod/cube.duckdb*` aside first); then `report --unit L96_h720_s202X` ×3 on WORKER_B re-sends the
   COMPLETED terminals from their records and a closure verifies/replays them (same device, 5090) and evaluates the agreement.
2. **T = 96 verification**: the three WORKER_A cells fail the frozen pointwise rule cross-device on the 5090 (max|Δ| ≤ 2.7e-4)
   while their metrics replay bit-identically and their same-device GPU history is bit-identical; the frozen rule is not
   widened. Verification in the actual closure needs 3 × ~22 s of same-device replays on WORKER_A's RTX 4090 — pending the
   owner's word on its cooler (it reached 83–86 °C with driver slowdown during training). Until then T = 96 stays
   NO_MEASUREMENT in the table and its arrays are retained on WORKER_A, WORKER_B and omega.
3. **Four-horizon average**: NOT_COMPUTED until every horizon of at least one seed is verified (1 and 2).
4. **Omega copies**: T = 192 s2021 root arrays (1.1 GB) and `_from_dragon` (1.8 GB) untouched during the travel hold; deletion
   there after the hold (gate on omega with the consolidated report).
5. **Cross-GPU discrepancy (4090 → 5090)** characterized only at the prediction level (max|Δ|, exact fractions, metric
   identity); no route-level trace between the two GPUs was run (the RP100 trace covered CPU vs 4090).

## Review request
One consolidated audit request: review of (a) the memory-bounded closure path (StoredArray chunked reads incl. compressed members, streamed targets, chunked
replay rule) and the two declared operational patches (`--dataloader-workers 0`, `GLIBC_TUNABLES` malloc thresholds) as
host-memory-only changes; (b) the closure's device identity (actual CUDA UUID; older records resolved from occupied GPU memory)
and the same-device / cross-device labels; (c) the T = 96 cross-GPU finding under the frozen rule and the decision on WORKER_A's
same-device replays; (d) the deletion lifecycle as executed (gate, receipts, historical verification) and the warehouse schema
repair pending the owner's restart; (e) the T = 336 float64-basis agreement and the T = 192 mixed-host population.
