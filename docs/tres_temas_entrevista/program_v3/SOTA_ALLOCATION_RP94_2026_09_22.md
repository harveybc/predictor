# RP94 — provisioning of the TimeFilter/ECL recipe: measured capacity and the allocation actually executed

Every number below is measured on the real delivered file with the author's own loop (bounded governed preflight units,
20 optimizer steps + a few validation forwards; no test score), or on the author's evaluation path with an untrained model
(memory only). Nothing was estimated from a smaller dataset. The recipe was not shrunk: 321 channels, L = 96, four horizons,
batch 16, 15 epochs with the author's early stopping, three seeds.

## Hardware and compute ceilings (2026-09-22)

| host | GPU (usable) | host RAM total / available | batch-slice MemoryMax (MemoryHigh = 0.9×) | s / train step | s / epoch (train + vali + test passes) | max h / cell (15 epochs) |
|---|---|---|---|---|---|---|
| omega | RTX 4070 Laptop 8 GB | 30 / 15 GiB | 14 GiB | 0.228–0.246 | 300–306 | 1.26–1.28 |
| dragon | RTX 4090 Laptop 16 GB | 30 / 12 GiB (a 5.5 GB VM is live) | 14 GiB | 0.100–0.106 | 131–132 | 0.55 |
| gamma | RTX 5070 Ti 12 GB (the 5090 is another workload's) | 15 / 5 GiB | 8 GiB | preflight scope killed at 1.8 GiB after 82 s | — | — |

GPU memory is not the constraint: peak allocation is 4.75–4.78 GiB for every horizon (`PREFLIGHT.<host>.json`).
Parameters: 1.82 M (T = 96) → 2.78 M (T = 720).

## The constraint: host memory of the author's evaluation path

`Exp.test()` accumulates predictions, targets and inputs in float32 lists, concatenates them, then `utils.metrics.metric`
builds full-size temporaries (MAE, MSE, RMSE, MAPE, MSPE). Measured/derived anonymous peaks (`PREFLIGHT_MEMORY.omega.json`):

| horizon | test windows | predictions | evaluation-path peak (anon) | verdict under a 14 GiB cap |
|---|---|---|---|---|
| 96 | 5 165 | 0.64 GB | ≈ 5.8 GiB (preflight RSS 2.0 GiB before evaluation) | executable |
| 192 | 5 069 | 1.25 GB | ≈ 8.4 GiB | executable on omega (−m 12G); dragon cannot request > 9G |
| 336 | 4 925 | 2.13 GB | ≥ 11.2 GiB measured, thrashing at ~12 GiB below MemoryHigh 12.6 GiB (probe stopped at 13.5 min) | **not admissible** |
| 720 | 4 541 | 4.20 GB | ≈ 20 GiB (probe stopped at 15 min, 10.9 GiB, thrashing before the metric stage) | **not admissible** |

## Allocation executed (and the owner's travel hold)

* dragon: `L96_h96_s2021/2022/2023` (−m 8G): all three completed, 15 epochs each, ~34–35 min, peak RSS 6.5 GiB, GPU 4.7 GiB.
* omega: `L96_h192_s2021` completed (−m 12G, 82 min, peak RSS 10.0 GiB); `L96_h192_s2022` was INTERRUPTED at the owner's
  order (omega's GPU at 87 °C while travelling without its cooler) and closed with a FAILED terminal, its partial attempt
  preserved; `L96_h192_s2023` not started. **No work of any kind runs on omega until 2026-09-24** (owner order; the
  execution wrapper enforces a travel thermal hold). Dragon cannot take T = 192: its admissible request is 9 GiB
  (12 GiB available, 3 GiB reserve) against a measured 10.0 GiB peak.
* gamma: nothing (measured host-memory deficit; live workload present).

Pending cells and where they can run: `L96_h192_s2022/s2023` on omega from 2026-09-24; `L96_h336_*` and `L96_h720_*`
nowhere under the current ceilings (below).

## Named capacity deficit (not a permission to adapt)

Six cells — `L96_h336_s*` and `L96_h720_s*` — cannot run under the batch-slice ceilings of the three hosts. Removing the
deficit needs ≥ 22 GiB admissible batch memory on one host (raising `crispdm-batch.slice` MemoryMax on omega is a service
change that goes only through the tested adoption/rollback procedure and was NOT done here) or a larger machine. Not done:
no reduction of channels, context, horizons, batch, metric population; no float16; no re-implementation of the author's
scorer in chunks. The reproduction returned in RP97 is therefore PARTIAL by population (two of four horizons), and says so.
