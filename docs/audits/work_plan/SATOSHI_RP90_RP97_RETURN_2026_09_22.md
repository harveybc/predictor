# Satoshi — RP90–RP97 return: SOTA-first reproduction of TimeFilter on the official ECL — PARTIAL by population, verified where executed

Orders: [MUSASHI_SOTA_FIRST_RP90_RP97_2026_09_21](../../handoffs/MUSASHI_SOTA_FIRST_RP90_RP97_2026_09_21.md) (832f31b, review of 3389a25).
Commits of this return after 832f31b: 38fb774 … e194094 plus the closing commit of this document (Musashi's 582eefa and ad4eedf arrived on the branch and were taken as they came) (see `git log 832f31b..`). Hosts: dragon (T = 96 cells, closure), omega
(prepare, T = 192 seed 2021; then the owner's travel thermal hold), gamma (none: measured memory deficit). Every governed step
registered its campaign before reading; every terminal reported and reconciled through the live data-gov/warehouse; failed and
interrupted attempts preserved and closed with FAILED terminals. No service was changed outside the tested adoption/rollback
procedure (one adoption: the benchmark lake). No financial data, no reserve, no live orders.

## Owner-facing table — published normalized metrics first (MSE / MAE in the paper's space and reduction)

Dataset/protocol: official processed ECL (Time-Series-Library `electricity.csv`, sha256 `7e45845d…`, 26 304 × 321, hourly),
chronological 7/1/2 by the author loader, StandardScaler fit on train rows, all 321 channels, L = 96, metrics = the author's
`utils.metrics.metric` over every test window × step × channel, no inversion. Model/revision: TimeFilter @ `dffde87e`
(ICML 2025), the author script's arguments and run.py defaults, Adam + MSE (+0.05 MoE loss), 15 epochs, patience 3.
Published values: Table 8 (L = 96), three runs; σ_paper of the four-horizon average = 0.005 / 0.006 (Table 7).
Agreement rule (frozen before any score): |mean − published| ≤ 2σ_paper + 0.0005 → NUMERICAL_AGREEMENT.

| T | published MSE / MAE | replicated per seed (2021 / 2022 / 2023) | mean (sd, n) | difference | matched naive (persistence) MSE / MAE | seasonal-24 naive | training | cost | replay | agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 0.133 / 0.230 | 0.13332 / 0.14008 / 0.13309 ; MAE 0.23055 / 0.23778 / 0.23032 | **0.13550 (0.0040) / 0.23288 (0.0042), n = 3** | +0.0025 / +0.0029 | 1.5878 / 0.9455 | 0.3211 / 0.3258 | 15/15 epochs each, best epoch 15, not early-stopped | 3 × 34 min on dragon (RTX 4090 L), 6.5 GiB RAM, 4.7 GiB VRAM | CPU fresh-process reload through the author's `test(test=1)`: replayed metrics equal the stored ones to ≤ 1.5e-8 (MSE) / ≤ 1e-9 (MAE); per-element max|Δ| 1.9e-3 / 4.0e-4 / 1.0e-2 — **fails the frozen prediction-level rule (atol/rtol 1e-4)**, which is not widened. Supplementary, same-device replay on dragon's GPU (the device that trained them): max|Δ| = **0.0** on all three, metrics bit-identical ([REPLAYS_GPU.json](../evidence/d3_k5_20260917/RP90/sota_timefilter_ecl_v1/REPLAYS_GPU.json)) | NUMERICAL_AGREEMENT on both metrics by the frozen rule (|Δ| 0.0025 ≤ 0.0105; 0.0029 ≤ 0.0125) — **reported as unverified by the closure** because the prediction-level replay rule failed (below) |
| 192 | 0.154 / 0.248 | 0.16086 / — / — ; MAE 0.25599 / — / — | 0.16086 / 0.25599, **n = 1** (seeds 2022 interrupted, 2023 not started) | +0.0069 / +0.0080 (single seed; the frozen rule needs the three-seed mean) | 1.5962 / 0.9507 | 0.3043 / 0.3237 | 15/15 epochs, best 14 | 82 min on omega (RTX 4070 L), 10.0 GiB RAM | REPLAY_PENDING (omega, from 09-24) | INCOMPLETE POPULATION |
| 336 | 0.162 / 0.261 | not executed | — | — | — | — | — | — | — | NOT EXECUTED: capacity deficit |
| 720 | 0.184 / 0.284 | not executed | — | — | — | — | — | — | — | NOT EXECUTED: capacity deficit |
| avg | 0.158 / 0.256 | — | NOT COMPUTED (two of four horizons) | — | — | — | — | — | — | — |

Reading. Under the author protocol, the three T = 96 seeds reproduce the published cell to within the paper's own
run-to-run dispersion (mean +0.0025 MSE / +0.0029 MAE; seed sd 0.0040 / 0.0042, of the same order as the paper's 0.005 /
0.006). Seed 2022 is the high one (0.140 / 0.238); seeds 2021 and 2023 sit on the published values. The one T = 192 seed is
+0.007 / +0.008 above the published values, inside the tolerance but not a three-seed result. Persistence loses badly here
(MAE 0.95 vs 0.23: skill +0.756) and the 24 h seasonal naive (MAE 0.326) is the informative baseline: the model beats it by
29 % MAE at T = 96. Each cell's exhaustive metrics vault (per-step, per-channel, residual moments/quantiles/histogram,
residual autocorrelation, MI/entropy, seasonal naive) is beside its record (`METRICS_VAULT.json`, ~0.5–0.9 MB), as the owner
ordered, so the raw prediction arrays can be deleted after Musashi's independent analysis.

Protocol fidelity (separate verdict): NOT_ESTABLISHED by the closure's own frozen rule. Everything else holds: sealed author revision and file digests unchanged on every host, the official file by digest, the preparation anchored by the accepted prepare terminal, the four cells' predictions/checkpoints/records in the accepted artifact chain with config digest and tags bound, targets re-derived hash-equal, the author's metric recomputed bitwise, float64 within 1e-6. What failed is the per-element CPU replay tolerance I froze at 1e-4: a checkpoint trained on a GPU and reloaded on CPU through the author's code reproduces the METRICS to 1e-8 but not every element to 1e-4 (max 1.0e-2 in normalized units; TimeFilter's top-p MoE routing is discontinuous, so a kernel-level difference can flip a few routes). The same checkpoints replayed on the GPU that trained them reproduce every prediction bit for bit (max|Δ| 0.0, three cells): the discrepancy is CPU-vs-GPU kernels, not the checkpoint or the code. The rule is not widened after the fact; the measured maxima are reported and a re-frozen rule (metric-level, or same-device replay) is proposed for the next round, not applied. Declared environment divergence: torch 2.13.0+cu130 / numpy 2.5.1 /
pandas 3.0.3 / scikit-learn 1.9.0 on RTX 4090 L and 4070 L vs the author's torch 2.3.1 / numpy 1.26.4 / pandas 2.2.3 /
scikit-learn 1.5.2 on an A100. Operational patches, none mathematical (tested): import shims for `sktime.datasets`/`patoolib`
(refuse any call), `np.Inf` alias, run.py `__main__` replicated to vary the seed, `metric` wrapped to capture the arrays,
`torch.load` mapped to CPU inside the CPU replay. Paper/code disagreements are listed in the dossier and the lock.

## What was done, per block

* **RP90** — HISTORICAL_DEV_ONLY implemented where selection happens: `df_benchmark_contract.disposition()` (recomputed
  from the task id; never a stored label) gates `reference_evidence`/`decide` (a household reference now returns
  HISTORICAL_DEV_ONLY with its verification underneath and opens no lane), `df_closure_table.build` emits `active_ranking`
  (ACTIVE task rows only) beside all rows with their disposition, block/financial closures carry `disposition` and
  `active_selection: None`. Tests: a zero-error household row verifies, is preserved, never ranks — from a rebuilt table,
  from a cached table without labels, from a cached table with a flipped label, from an inherited closure.
* **Musashi's RP89 findings** — (1) the block closure hashes the checkpoint bytes on disk and binds a full replay identity
  (checkpoint, arrays, design, replay code, cell) before any cached or adopted replay is reused; (2) the financial
  verification derives the fold population from the ACCEPTED preparation record (record artifact anchored; a rewritten
  FIN_DATA.json is PREPARATION_RECORD_CHANGED); (3) a cost-pilot refusal closes FAILED with a non-zero exit and a pending
  outbox refuses; (4) the HistData producer lineage is recorded unchanged — nothing invented. POST of his four probes:
  [RP90/REVIEW_REPRODUCED_POST.json](../evidence/d3_k5_20260917/RP90/REVIEW_REPRODUCED_POST.json).
* **RP91** — [reference dossier](../../tres_temas_entrevista/program_v3/SOTA_REFERENCE_DOSSIER_2026_09_21.md): PatchTST,
  iTransformer, TimeFilter, NPMixer read at source with per-horizon values; TimeFilter leads on both MSE and MAE under the
  fixed-L = 96 and the searched-L protocols; NPMixer's code NOT FOUND (unverified, not refuted); disagreements named.
* **RP92** — [protocol lock](../../tres_temas_entrevista/program_v3/SOTA_PROTOCOL_LOCK_2026_09_21.md) sealed in
  `DESIGN.json` (design `9b49010d…`) from the author's files by AST/script parsing; the official file adopted as a governed
  resource (lake `sota_benchmarks`, rehearsal + receipt in RP90/); tests fail on substituted predictions, altered checkpoint,
  changed scaler, wrong horizon/channels/window/target, another reduction, missing artifact, wrong tags.
* **RP93** — the author's parser, script, loader, model, loss, optimizer, early stopping and scorer are what runs; row-level
  controls tested on the author loader (scaler fit population, future perturbation, boundary count); fresh-process reload
  through the author's `test()`.
* **RP94** — [allocation](../../tres_temas_entrevista/program_v3/SOTA_ALLOCATION_RP94_2026_09_22.md): governed preflights
  on omega and dragon (300 / 132 s per epoch, 4.8 GiB VRAM), evaluation-path memory probes; **named deficit**: T = 336
  (~12 GiB) and T = 720 (~20 GiB) exceed the 14 GiB batch-slice ceiling on every host; gamma has 5 GiB.
* **RP95** — four cells executed under governance (above); `L96_h192_s2022` interrupted at the owner's order (omega GPU at
  87 °C while travelling) and closed FAILED; the owner then placed a total hold on omega until 2026-09-24.
* **RP96** — closure on dragon (CPU): accepted artifact chain for the four cells (predictions, checkpoint, record digests in
  the warehouse's accepted terminals; config digest and tags bound to the design cells), preparation anchored by the
  accepted prepare terminal, targets re-derived from the delivered file by the author loader (hash-equal), the author's
  metric recomputed bitwise from the arrays, float64 reduction within 1e-6, naive on identical windows, fresh-process CPU replays of the three T = 96 cells (metrics reproduced to ≤ 1.5e-8; the per-element 1e-4 rule failed at max|Δ| ≤ 1.0e-2 — reported, not widened); `L96_h192_s2021` replay pending (omega, from 09-24); exhaustive metrics vault v2 per cell with every estimator's parameters and limitations declared.
* **RP97** — this return; state and master plan updated; single review request below.

## What remains unexecuted (exactly)

* `L96_h192_s2022`, `L96_h192_s2023`: omega from 2026-09-24 (only host whose admissible batch memory covers the measured
  10.0 GiB); `L96_h192_s2021` replay: same.
* `L96_h336_*`, `L96_h720_*` (six cells): no host under the current ceilings; the recipe was not shrunk. Removing the deficit
  is an owner decision (raise omega's batch-slice ceiling through the tested procedure, stop dragon's VM, or a larger host).
* Protocol B (L = 512 script): not started.
* Test suites: `tests/test_df_sota_repro.py` 20 passed + 1 skipped on dragon (store receipt absent there); the RP90/RP89
  suites (contract 33+2, closure 35+2, block 19+1, fin 25+2, lake adopters 16) passed on omega before the hold; full suite on
  dragon (CPU, FL08 stack test deselected): 2302 passed, 24 failed, 171 skipped, 8 legacy collection errors (18.5 min; `tests/test_df_sota_repro.py` in its own process: 20 passed + 1 skipped). The 24 failures are dragon-environment failures in modules this round did not touch (`test_df_operators_causality` — `crispdm-run` not on the PATH of the non-login shell there, `test_governed_run_classification` — a sibling preprocessor worktree absent on dragon, `test_acceptance_c1_c16` — module `incremental_census` absent), none referencing the changed tools; on omega the same suite closed RP89 at 2469 passed with only the 3 legacy config failures and was not rerun there (hold).

## Evidence

Reproduction root records (scrubbed): [sota_timefilter_ecl_v1/](../evidence/d3_k5_20260917/RP90/sota_timefilter_ecl_v1/) —
`DESIGN.json` (lock + cells), `REPORT.json` (verification, table, replay patch), `SOTA_TABLE.md/.json`, `REPLAYS.json` (CPU),
`REPLAYS_GPU.json` (same-device), `TERMINAL_RECEIPTS.json`, `PREFLIGHT.*`, `PREFLIGHT_MEMORY.omega.json`, `EXECUTE.dragon.json`,
`MERGE.*`, and per cell `cell.json` + `METRICS_VAULT.json` (v2: every estimator with parameters and limitations). Lake adoption and
RP90 POST: [RP90/](../evidence/d3_k5_20260917/RP90/). Storage inventory and thermal/retention amendments: Musashi's commits
582eefa, ad4eedf on this branch (taken as they arrived).

## Review request

Single request: review this return and the four verified cells' custody, the RP90 exclusion, the RP89 repairs and the lock;
then delete the raw prediction arrays after your independent analysis (owner's order), keeping records, vaults, checkpoints
and receipts.
