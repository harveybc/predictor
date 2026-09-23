# Satoshi — RP106–RP113 return: exact official scoring, bound retention, and the twelve cells

Orders: [MUSASHI_SOTA_RP106_RP113_2026_09_22](../../handoffs/MUSASHI_SOTA_RP106_RP113_2026_09_22.md) (c779711), on the
[RP98–RP105 review](MUSASHI_RP98_RP105_REVIEW_2026_09_22.md) of 55a339ce. Commits after c779711: 3f52aabb … 494f54c6.

**Placement.** Every new GPU execution in this round ran on WORKER_B's EXTERNAL RTX 5090 (UUID a9f35631…), admitted by physical
UUID before each dispatch and asserted inside the child; WORKER_B's internal RTX 5070 Ti, WORKER_A's GPU and the coordinator's
heavy work stayed held, and nothing fell back to them. Host CPU work ran single-threaded inside the governed slice and the host's
own temperature was measured beside the GPU's (the external cooler does not cool the laptop): the host stayed at 41–63 °C and the
5090 at 35–41 °C. One memory admission refusal (8 GiB requested, 8.0 GiB free above the host reserve) was respected and the work
re-requested at 6 GiB rather than bypassed. No VM was touched, no slice widened, nothing compressed, no prediction array moved
home, no recipe reduced.

## Owner-facing table — official normalized MSE / MAE first

Population: 12 cells (L = 96 × T ∈ {96, 192, 336, 720} × seeds {2021, 2022, 2023}), design 9b49010d…, official processed ECL
(TSL, electricity.csv sha256 7e45845d…), TimeFilter @ dffde87e, the author's recipe and epochs, metrics in the normalized space
through the author's own reduction. Frozen operational margin: |mean − published| ≤ 2·σ_paper + 0.0005 = 0.0105 MSE / 0.0125 MAE.
This margin is a predeclared operational band, **not** statistical equivalence and not exact equality to a rounded paper table.
Two closures are cited: the one that verified T = 720 while its predictions existed
([REPORT.gamma.json](../evidence/d3_k5_20260917/RP111/REPORT.gamma.json), sha256 bc9d3388…) and the final one after the
authorized deletion ([REPORT.gamma.final.json](../evidence/d3_k5_20260917/RP111/REPORT.gamma.final.json), sha256 b84a7767…),
where those three cells are historically bound to the first. Sample SD across seeds, per horizon: 0.00380 / 0.00396 (T = 192,
two hosts — mixed-host dispersion is not pure seed variability), 0.00320 / 0.00384 (T = 336, one host), 0.00694 / 0.00702
(T = 720, one host). Paired seed contrasts from the persisted per-window series are DONE for T = 192, 336 and 720 and
NOT_APPLICABLE for T = 96 ([PAIRED_CONTRASTS](../evidence/d3_k5_20260917/RP111/PAIRED_CONTRASTS.gamma.json)).

| T | published MSE / MAE | official author-float32 per seed 2021 / 2022 / 2023 | mean (sd, n) | difference | independently checked float64 (mean) | matched persistence / seasonal-24 (z-MSE / z-MAE) | training | custody | original scorer | same-device repeatability | cross-device portability | operational agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.133 / 0.230 | 0.13332 / 0.14008 / 0.13309 ; 0.23055 / 0.23778 / 0.23032 | **NO_MEASUREMENT in this closure** (n = 0 verified; values shown apart, never pooled) | — | 0.135496 / 0.232884 | 1.5878 / 0.9455 ; 0.3211 / 0.3258 | 15/15 epochs, best 15 ×3, WORKER_A RTX 4090 L | ACCEPTED_ARTIFACT_CHAIN | author float32 recomputed at closure (exact bounded route) | **not tested in this closure**; preserved WORKER_A evidence on the training device is bit-identical (max\|Δ\| 0.0) but is HISTORICAL_RECORD_ONLY | **FAIL** on the 5090: max\|Δ\| 1.9e-4 / 2.2e-4 / 2.7e-4 > atol/rtol 1e-4, exact fractions 0.114 / 0.118 / 0.109; replayed author metrics bit-identical | UNVERIFIED: the frozen pointwise rule is not widened |
| 192 | 0.154 / 0.248 | 0.16086 / 0.15863 / 0.15345 ; 0.25599 / 0.25243 / 0.24807 | **0.157645 (0.00376) / 0.252163 (0.00397), n = 3** | +0.00365 / +0.00416 | 0.157645 / 0.252163 | 1.5962 / 0.9507 ; 0.3043 / 0.3237 (5069 windows) | 15/15 each, best 14 / 15 / 12; s2021 on omega (RTX 4070 L), s2022/s2023 on the 5090 | ACCEPTED_ARTIFACT_CHAIN, predictions deleted 2026-09-22 under authorization | author float32 of the original closure, bound by digest to report fbe2cd4c… | historical: s2022/s2023 PASS on their training device (max\|Δ\| 0.0) before deletion | historical: s2021 PASS (omega → 5090, max\|Δ\| 4.8e-6) | **OPERATIONAL_AGREEMENT** (0.0036 ≤ 0.0105; 0.0042 ≤ 0.0125) |
| 336 | 0.162 / 0.261 | 0.16259 / 0.16817 / 0.16267 ; 0.26067 / 0.26741 / 0.26087 | **0.164480 (0.00318) / 0.262985 (0.00381), n = 3** | +0.00248 / +0.00198 | 0.164480 / 0.262985 | 1.6178 / 0.9613 ; 0.3273 / 0.3427 (4925 windows) | 15/15 each, best 10 / 13 / 12, on the 5090 | ACCEPTED_ARTIFACT_CHAIN, predictions deleted 2026-09-22 | **author float32 obtained after the deletion**: inference from the retained checkpoints reproduced the predictions **bit-identically** (pred and target digests equal the record's) and the exact route scored them | historical: PASS on the training device before deletion (max\|Δ\| 0.0) | not tested | **OPERATIONAL_AGREEMENT** (0.0025 ≤ 0.0105; 0.0020 ≤ 0.0125) |
| 720 | 0.184 / 0.284 | 0.18714 / 0.19817 / 0.18537 ; 0.28752 / 0.29865 / 0.28568 | **0.190226 (0.00686) / 0.290617 (0.00688), n = 3** | +0.00623 / +0.00662 | 0.190226 / 0.290617 | 1.6468 / 0.9754 ; 0.3670 / 0.3733 (4541 windows) | 15/15 each, best 12 / 5 / 11, on the 5090 | ACCEPTED_ARTIFACT_CHAIN (the three terminals Musashi resent) | author float32 recomputed at closure by the exact bounded route (the records carry the float64 reduction only) | **NOT_CERTIFIED**: the replay on the 5090 is bit-identical (max\|Δ\| 0.0, exact fraction 1.0) but the training device is INFERRED, not measured | UNDETERMINED_ATTRIBUTION (same UUID, inferred training side) | **OPERATIONAL_AGREEMENT** (0.0062 ≤ 0.0105; 0.0066 ≤ 0.0125) |
| avg | 0.158 / 0.256 | — | **NOT_COMPUTED** | — | — | — | — | — | — | — | — | T = 96 is unverified, so no seed has four verified horizons; the missing horizon stays in the denominator |

Cost, measured, on the admitted device (full ledger: [RUN_LEDGER.json](../evidence/d3_k5_20260917/RP111/RUN_LEDGER.json)):
15.3–15.6 min per T = 192/336 cell, 20.6–20.8 min per T = 720 cell, ≤ 5.2 GiB VRAM, 2.7–7.1 GiB peak host RSS; 19 attempts in
the ledger (12 with records, 7 retired or interrupted and carried with an explicit NONE_RECORDED cost), 5.37 h of measured wall
time in total, 210.3 GiB free disk measured after the cleanup.

## What was done, per block

* **RP106 — the PRE preserved and the real path made to fail correctly.** Musashi's probe is kept unchanged
  ([musashi_probe_unchanged.py](../evidence/d3_k5_20260917/RP106/musashi_probe_unchanged.py)) and its output at the reviewed
  commit is frozen as the PRE ([PROBE_RESULTS_PRE.json](../evidence/d3_k5_20260917/RP106/PROBE_RESULTS_PRE.json)): the
  different-identity copy deleted, the rewritten record verified with a zero score, the forged report accepted. On the corrected
  tree his probe's own later steps no longer apply, because its first scenario now refuses and deletes nothing, so the POST
  ([reproduce_post.py](../evidence/d3_k5_20260917/RP106/reproduce_post.py) →
  [REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP106/REVIEW_REPRODUCED_POST.json)) re-creates the three scenarios on
  top of a valid deletion and adds the two controls. Every scenario is also a frozen test that calls the real closure and deletion
  entry points, written before the implementation.
* **RP107 — historical evidence is bound, not self-declared.** A deleted cell's score is now the ORIGINAL pre-deletion closure's,
  resolved BY DIGEST from preserved reports (`REPORT.json`, the content-addressed `reports/REPORT.<sha256>.json` written at every
  closure, or a retained `REPORT*.json`), and bound to: the accepted terminal chain (the record and checkpoint on disk must hash to
  the accepted artifacts), the retained catalog (the digest that report recomputed), the targets the record scored, and the design.
  The marker is a pointer; `verified_at_deletion` is no longer read at all. Typed refusals: HISTORY_UNRESOLVED_REPORT,
  HISTORY_DESIGN_MISMATCH, HISTORY_ROW_UNVERIFIED, HISTORY_CUSTODY_*, HISTORY_RECORD_CHANGED, HISTORY_CHECKPOINT_CHANGED,
  HISTORY_ARRAYS_IDENTITY, HISTORY_VAULT_MISMATCH, HISTORY_TARGETS_MISMATCH. A refused row carries no score and enters no mean; the
  row also keeps what the current record claims, so a rewrite is visible rather than silently used. Historic reports and receipts
  are never rewritten: successor evidence is added beside them.
* **RP108 — deletion is content-specific and its approval is durable.** Under an exclusive lock on the root, the whole inventory of
  a unit is preflighted before any unlink: every candidate path must hash to the ACCEPTED predictions artifact, and a conflicting
  copy, a conflicting attempt (an extra root whose record is another record), a symlink or non-canonical alias, an active reader,
  an unverified catalog, an approval naming another report, or a backup manifest that does not already cover the record and the
  catalog refuses the WHOLE unit with nothing removed. Identity is re-checked at the instant of each unlink; an interrupted
  deletion leaves a PARTIAL marker with accurate per-path status and resumes later only under the same approval. Receipts carry
  per-path bytes, digests, filesystem deltas and the measured free space afterwards.
* **RP109 — the author's own scorer, completed within actual resources.** `author_metric_exact` reproduces
  `utils.metrics.metric` BIT FOR BIT with about 64 MB of temporaries: the element-wise float32 errors are formed chunk by chunk and
  numpy's pairwise summation tree is replicated over the flattened population (leaves of at least 128 elements reduced by numpy
  itself), with the mean taken as `float32(sum) / float32(N)`. Its equivalence is not assumed: it is proved against
  `np.add.reduce` for n = 7 … 6.3e6 and several leaf sizes, against the author's function on adverse ranges (offsets of 1e4 with
  small noise, partial final leaves, 3-D contiguous arrays and stored npz members), and on the real author fixture; wherever the
  author's own function still fits in memory it is executed beside the route and must agree bitwise or the cell is refused. A mean
  of chunk means is not this and is not used. The float64 reduction remains a separately named check and never appears under
  `author_metric_float32`.
* **RP110 — device attribution and the scope of the operational patches.** A record's training device now carries its evidence
  class: MEASURED (the process asked its own CUDA runtime), VISIBILITY_MASK (a UUID mask it was started with),
  INFERRED_GPU_MEMORY (the host GPU whose memory grew while it ran), UNKNOWN, or CPU. Only MEASURED on both sides (or CPU on both)
  certifies same-device repeatability; an equal UUID under an inferred attribution is reported as NOT_CERTIFIED with
  UNDETERMINED_ATTRIBUTION, never as same-device. Original records are preserved; the class is appended evidence. The DataLoader
  worker change was tested over an actual training trajectory on the real author fixture — four optimizer steps with the author's
  model, criterion and optimizer: batches, loss, weights, optimizer state and the torch/numpy RNG states after every step, plus the
  validation and test loader inputs, are identical with one worker and with none. The allocator option is shown to reach the child's
  environment while leaving the child's argument vector identical.

* **RP111 — the remaining numerical work, without duplicate training.** The three T = 720 terminals Musashi resent are
  reconciled by this closure's custody check (accepted artifact chain, digests equal on disk). Their original-rule replays ran on
  the admitted 5090 and are bit-identical (max\|Δ\| 0.0, exact fraction 1.0); their official scoring now comes from the exact
  route, their float64 reductions remain as a separate check, and their finite catalogs were recomputed and read back at closure.
  The bounded run ledger is frozen from measured costs — every current and retired attempt with its host, device-attribution
  class, wall and CPU seconds, peak RSS and VRAM, patches and regeneration — and contains no projection. For T = 96 the
  same-device requirement was checked against existing evidence FIRST, as ordered: WORKER_A's preserved replays on the device that
  trained those cells are bit-identical, they travel with the merge as HISTORICAL_RECORD_ONLY, and they are not upgraded to a
  current verification; no new execution was launched on that held GPU. The cross-GPU failures are preserved with their exact
  rule, and equal aggregate metrics are never offered as a repair.
* **RP112 — independent catalog acceptance, then authorized cleanup.** Every retained catalog was accepted independently of the
  process that produced it: identity against the retained record and the accepted chain, population and denominators
  (windows × steps × channels, loader fully consumed), the required estimator families with their DONE / APPROXIMATE /
  NOT_APPLICABLE states, and oracles between the catalog's own estimators — per-step, per-channel, per-window and
  window-weighted time-block means against the global, the skill and RMSE identities, and the per-step naive mean. Nine cells
  pass with **every oracle exactly 0.0**; the three T = 96 catalogs are refused for one reason only, that the current closure does
  not verify those cells. A metadata-only durable backup (84 files, 25.2 MB: records, catalogs, acceptances, terminals, receipts,
  reports, markers, ledger; no array, no checkpoint) was taken and its manifest bound into the deletion approval. Only then were
  the three T = 720 arrays deleted through the corrected per-content path: 12,594,204,672 bytes reclaimed, measured free disk
  210.3 GiB afterwards, one receipt line per path. The T = 96 arrays are retained because they are unverified. Nothing was
  deleted on the coordinator or on WORKER_A in this round: their copies stay held under the thermal policy, and no multi-GB
  hashing or replay was run locally to work around it.
* **Protocol B** — kept as a separate, explicit deliverable and NOT executed:
  [SOTA_PROTOCOL_B_MAPPING_2026_09_23](../../tres_temas_entrevista/program_v3/SOTA_PROTOCOL_B_MAPPING_2026_09_23.md) states the
  author's exact L = 512 recipe (patch 128, top_p 0.0, dropout 0.5), the fact that Table 9's per-column input length is not
  published and the script offers only L = 512, what may therefore be claimed from an L = 512 run, and a cost projection from
  measured protocol A cells. No protocol A number appears as a protocol B result.

## Exact remaining deficits

1. **T = 96 verification.** Its cells fail the frozen pointwise replay rule cross-device (WORKER_A's 4090 Laptop → 5090,
   max\|Δ\| ≤ 2.7e-4) while their metrics replay bit-identically, and the rule is not widened. Verification in the actual closure
   needs ~66 s of same-device replays on WORKER_A's GPU, which remains held with cooling explicitly not confirmed. Their arrays
   are retained on WORKER_B, WORKER_A and the coordinator until then, so no evidence is lost by waiting.
2. **The four-horizon average** stays NOT_COMPUTED while T = 96 is unverified; the horizon is not dropped from the denominator.
3. **Device attribution for the nine GPU-trained cells of T = 192/336/720** is INFERRED or VISIBILITY_MASK, never measured: their
   records predate the measured-UUID field. Same-device repeatability is therefore reported as NOT_CERTIFIED for T = 720 and as
   preserved history for the deleted cells. New cells record the measured UUID.
4. **Coordinator and WORKER_A copies** (T = 96 ×3 on both, plus the coordinator's staging copy of T = 192 s2021 already deleted)
   are inventoried but untouched under the thermal policy.
5. **The cross-GPU discrepancy** is characterized at the prediction level only (max\|Δ\|, exact fractions, metric identity); no
   route-level trace between the two GPUs was run, and the earlier CPU-vs-4090 trace is scoped to its three windows and that
   device pair.
6. **Protocol B** is specified and costed but unexecuted; its twelve cells are 6–16 h on the admitted device.

## Suites, with scope and accounting (WORKER_A, CPU, governed slices, at 494f54c6)

| suite | result | wall | peak RSS |
|---|---|---|---|
| `tests/test_df_sota_repro.py` (own process) | 61 passed, 1 skipped | 49.4 s | 1.35 GB |
| closure table + benchmark contract + block + financial acceptance | 117 passed, 2 deselected (FL08 needs the coordinator's stack) | 163.5 s | 2.96 GB |
| `olap/store/tests` on SQLite **and** DuckDB | 106 passed, 1 skipped | 3.6 s | 0.22 GB |
| lake adopters | 4 skipped (need the coordinator's deployed data-gov configuration) | — | — |

Interpreter 3.12.13 with numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cu130, scikit-learn 1.9.0; the store suite in a throwaway venv
with duckdb 1.5.5 and SQLAlchemy 2.0.54. No test ran on the coordinator.

## Review request

One consolidated audit request: (a) the bound historical verification and its typed refusals, against the frozen PRE/POST;
(b) the content-specific deletion path — preflight, exclusive boundary, identity at the instant of deletion, partial markers and
resumption, and the approval bound to the accepted report and the backup manifest; (c) the exact author-scorer route and its
bitwise equivalence claim, including the T = 336 regeneration that recovered the official metric after deletion; (d) the device
attribution classes and the refusal to certify same-device repeatability without a measured UUID; (e) the catalog acceptance
oracles and the T = 720 deletion they authorized; (f) the protocol B mapping as the separate deliverable it is.
