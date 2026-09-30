# M06: evidence and resources. Traffic cells, 5090 host, GPU-slot desk and machine-readable state

Satoshi, successor technical lead. 2026-09-30.
Order: `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` (§2, §6 ECL paragraph, §8), at `dc72170e`.
Evidence directory: `docs/audits/evidence/MODULAR_CAMPAIGN_20260930/`. It holds `RETURN.md`, `STATUS.json`, `RESULTS/` and `PROGRESS.png`.

Hosts are named by role: coordinator (the owner's desktop, RTX 4070 Laptop), worker_a (the 5090 host: external RTX 5090 plus RTX 5070 Ti Laptop), worker_b (RTX 4090 Laptop).

## 1. Adopted Traffic cells (TimeFilter published recipe, L96 → h96)

These are measured cells run with the author's code (dffde87e) under the sealed design (`6cba7e20…`). Every row below was produced by the sealing executor's own `comparison_row()` from the retained record. The executor rejects a record whose digest does not recompute.

Metric: official normalized MSE/MAE, z_train space, float32 author reduction over 282,432,576 elements (3,413 windows × 96 steps × 862 channels). Naive: persistence on the same rows (population sha `717cb5b3…`). Published: TimeFilter, ICML 2025, Table 8, L=96: 0.375/0.251.

| seed | device | wall s | MSE | MAE | naive MSE / MAE | MAE skill | Δ vs published (MSE / MAE) | checkpoint sha |
|---|---|---|---|---|---|---|---|---|
| 2022 | 4090, GPU-a8bd1b2c | 3,906 | 0.3753611147 | 0.2512390912 | 2.7144524181 / 1.0772232192 | 0.7668 | +0.00036 / +0.00024 | 59d61b47… |
| 2023 | 4090, GPU-a8bd1b2c | 3,982 | 0.3745366931 | 0.2508221567 | same rows, same values | 0.7672 | −0.00046 / −0.00018 | e5932881… |
| 2021 | 4070, GPU-612d1e0c | 8,292 | 0.3756999969 | 0.2513667941 | same rows, same values | 0.7667 | +0.00070 / +0.00037 | 4b434bc9… |

**Three-seed closure, from the sealing executor's `classify()`, against the published 0.375/0.251:**
- **MSE:** mean 0.3751992683 (sd 0.00060), Δ +0.00020, tolerance 0.0165: **OPERATIONAL_AGREEMENT**.
- **MAE:** mean 0.2511426806 (sd 0.00028), Δ +0.00014, tolerance 0.0085: **OPERATIONAL_AGREEMENT**.

The tolerance comes from the paper's Table 7 dispersion (Traffic 0.407±0.008 / 0.268±0.004), so the margin is wide relative to the observed differences. All three seeds are within ±0.0007 MSE of the published value, on both sides. The orchestrator independently checked s2023 on the preserved copy.

- All three cells hit the 30-epoch budget with the best validation at the last epoch (`EPOCH_BUDGET_CEILING_BEST_AT_LAST_EPOCH`).
- Comparison class: `MATCHED_PUBLISHED_RECIPE_EXECUTED`. The agreement class exists only on the three-seed mean.
- Evidence state: measured, NOT independently verified (one execution per cell, not replayed).
- Preserved read-only copies (checkpoint, record, author log) live under `~/.local/state/crispdm-data-foundation/preserved_m06_20260930/` on the host that ran each cell and on the coordinator. Each checkpoint's sha equals its record's `checkpoint_sha256`.
- Seed handling: the executor fixes seeds to the cell seed, so seed 2021 is the author's own hard-coded seed. The recorded `effective_args.seed = 2` is the argparse default and has no effect.

How the ETA was estimated: observed throughput. W is the mean train cost of the last five logged epochs plus the median eval overhead, where eval overhead is the excess of each epoch's first-100-iteration speed. Epochs completed = time since the first checkpoint directory ÷ W. Earliest ETA = early stop three epochs after the last checkpoint save. Latest ETA = remaining epochs at 1.10·W plus about 2W of scoring.
- At adoption: s2021 W=292 s, s2023 W=139 s.
- s2023 predicted 23:01–23:38Z; actual 23:22:42Z.
- s2021 predicted 23:10–23:56Z at adoption; actual 23:28:08Z, earlier than the last running estimate (23:41–23:52Z), which overcounted the eval overhead on the 4070. The clock-restore unit ran `nvidia-smi -rgc` at 23:28:17Z.

Cooling and observed limits (reported, not changed):
- **4070 (coordinator):** application clock locked at 1500 MHz of 3105, 74–76 °C, 55 W of a 114 W limit. No active thermal reason; cumulative SW thermal slowdown was 22.6 s since driver load. The `gpu_idle` bit reads set while the GPU is at 100% utilization, an artefact of the locked clock on this laptop part. The coordinator's GPU clock-restore unit (`codex-<coordinator>-gpu-clock-restore.service`) runs `sudo -n nvidia-smi -rgc` when s2021's unit ends; sudo allows nvidia-smi without a password.
- **4090 (worker_b):** SW THERMAL SLOWDOWN ACTIVE throughout s2023. 82–83 °C (target 87 °C), 120 W of 175 W, SM clock swinging 930–1470 MHz of 3105. Cumulative SW thermal slowdown 1,771 s since driver load; no HW thermal slowdown. The epoch cost matched s2022's (about 112 s), so this is the host's steady thermal ceiling, not a degradation. The laptop's cooling is saturated. Recommendation for the owner: check the intake and exhaust (stand, dust), or cap power if longer fits are planned there. No breach occurred: no HW slowdown and no stop condition.

## 2. The 5090 host (worker_a), read-only diagnosis

- **What happened.** An interactive operator (the owner) on that host ran `apt update`/`apt upgrade` and then `shutdown -r now` at 17:52–17:53 local. The new boot is at 17:53:58: kernel 7.0.0-31 → 7.0.0-34, driver 580.178.04 unchanged. The previous boot had been up 8 d 5 h, since 2026-09-22. Unreclaimable slab went from 5.48 GiB to 0.27–0.28 GiB, and MemAvailable to about 12.7 of 14.98 GiB. The slab growth has been **cleared, not diagnosed**.
- **Driver host-allocation failures in the previous boot.** There were 2,201 `NVRM … NV_ERR_NO_MEMORY` lines from `_memdescAllocInternal`/`system_mem.c`. By day: 09-24 29, 09-25 8, 09-26 2, 09-27 2, 09-28 35, 09-29 347, 09-30 1,778. Separately, 154 `_kgmmuClientShadowFaultBufferPagesAllocate: big page size` failures fell on 09-30. There were no kernel page-allocation failures and no OOM kills. In other words, the driver's own host allocations failed as the unreclaimable pool grew, which is why an "idle, cold" GPU was unusable.
- **Slab caches.** `/proc/slabinfo` and `slabtop` need root, which M06 does not have, so the cache that held the 5.48 GiB is **not named**. It cannot be recovered after the reboot.
- **Likely cause (hypothesis, UNVERIFIED).**
  - The GB202 (5090) re-initialised 594–1,400 times a day. Each `kbifInitLtr_GB202` line marks one RM init; the count was 991 on 09-23 and 1,300–1,400 on each later day.
  - `nvidia-persistenced` runs with `--no-persistence-mode`, so each NVML/nvidia-smi poll of an idle GPU brings it up and tears it down. Several pollers exist: the GPU temperature watchdog, crispdm admission monitors, the campaign supervisor, and this lane's own status writer (now throttled to one query per 10 min on that host).
  - 5.2 GiB over about 11,000 re-inits is roughly 0.5 MiB per init.
  - Post-reboot series (`worker_a_slab_series.jsonl`): +26 MiB over 0.43 h across 28 re-inits, about 0.9 MiB per re-init and 61 MiB/h. This is confounded by early-boot services and M04's own GPU probes on that host. Measurement continues every writer cycle.
- **Remedy (the owner's decision; none of it done by M06).**
  1. Enable persistence mode on that host (drop `--no-persistence-mode`, or `nvidia-smi -pm 1`). Then compare the SUnreclaim slope per hour against the current series.
  2. Add a root timer that snapshots the top of `/proc/slabinfo` hourly to a user-readable file, so the next growth names its cache.
  3. If growth continues with persistence on, test a newer 580-series or 590 driver on the 7.0.0-34 kernel.
  4. Until then, schedule a reboot before SUnreclaim passes about 3 GiB. Growth would reach it in about 2 days at the post-reboot slope.
- **TensorFlow eligibility.** After the reboot the host was memory-admissible, but TF 2.21.0 registered zero GPUs. The orchestrator traced this to cu12 and cu13 wheels mixed in the `tensorflow` env. The launch-time recipe is `LD_LIBRARY_PATH` = every `site-packages/nvidia/*/lib` except `cu13/`; it installs nothing.
  - The orchestrator's three GPU facts on worker_a: devices listed, a matmul on GPU:0, memory info.
  - M06 reproduced the same cause and the same fix on worker_b (probe `m06-tf-probe`, 2G/3m). Bare: "Cannot dlopen", 0 GPUs. With the recipe: 1 GPU (RTX 4090 Laptop), matmul on GPU:0, memory info current 263,680 / peak 787,968 bytes.

## 3. GPU-slot desk

| time (Z) | host / device | state | action |
|---|---|---|---|
| 22:53 | worker_a / GPU-a9f35631 (5090) | memory-admissible after the owner's reboot | M06 notified the orchestrator. The slot was held because TF registered 0 GPUs. |
| ~23:03 | worker_a / GPU-a9f35631 | TF-eligible with the recipe | Orchestrator handed the slot to M04 (a083979bc9a8fa1b8). |
| 23:22:42 | worker_b / GPU-a8bd1b2c (4090) | s2023 released | M06 sent the release fact to M01 and M02. m01-suite 4G and m02-suite-wb 3G were auto-admitted. Aggregate 8.32 GiB of 14.00, so a second 3G fits (11.32), and M03's 2G too (13.32). |
| 23:28:08 | coordinator / GPU-612d1e0c (4070) | s2021 released | s2021 sealed; clock restored to 3105 MHz max at 23:28:17Z; 12 GiB lease released; queued CPU suites admit. 4070 and 4090 now free GPU slots; 5090 reserved for M04. |

After s2021, the orchestrator handed the 4090 (GPU-a8bd1b2c) to M04 at about 23:40Z; its lease is expected there. The coordinator's 4070 is kept free under the standing no-heavy-compute rule. M04's pilot (6G) on the 5090 was **queued, not running**. The blocker is the admission defect below, not the GPU.

**Admission finding ADM-DEADCACHE-01.**
- Read by the orchestrator on worker_a: slice `memory.current` 2,385,801,216 = anon 0 + file 2,327,924,736 (shmem 16,564,224) + slab 57,008,080, with no scope, no process and no lease. This is the page cache of M04's finished NPZ build, recharged to the slice.
- The gate added it as live use: 2.386 + 6.442 = 8.83 > 8.59 GiB, so the request was queued, and nothing reclaims clean cache without pressure.
- M06's read-only check of the slice minus its live scopes:
  - worker_a: 2.223 GiB (file 2.169).
  - coordinator: 0.645 GiB (file 0.424, slab 0.217).
  - worker_b: 1.020 GiB (file 0.988, of which 0.751 is shmem/tmpfs, which cannot be reclaimed without swap).
- The fix is on branch `satoshi/m06-admission-dead-cache-20260930`, tip `0928dc06`, off the deployed `ddadf4a9`. It is **not deployed**; it waits for the orchestrator's review. 45/45 admission tests pass.
  - The gate charges `memory.current − clean file`, where clean = file − shmem − dirty − writeback − unevictable.
  - `crispdm-run` runs the command under `scope-exec`, which stays in the job's own scope and forwards TERM/INT/HUP. It preserves the exit status and re-raises a death by signal. When the command ends, it writes that scope's own clean file bytes to the scope's own `memory.reclaim`, guarded to `crispdm-*.scope` directly under the named slice.
  - The existing dead cache stays until the owner decides on a one-time reclaim.
- Launcher fact: `crispdm-run` is byte-identical on all three hosts (sha `499fdc18…`, options `m:t:n:W:L:E:P:S:qh`). It has no GPU option. `--require-gpu-uuid` belongs to the Traffic executor, and M06's first slot note misplaced it on the launcher. The correct pin is `CUDA_VISIBLE_DEVICES=<uuid>`, with the UUID asserted inside the child.

## 3b. Generated closure table

`RESULTS/traffic_h96_closure_table.md` and `.json` are generated by `tools/m06_closure_table.py` from the three records and the design lock; nothing in them is typed by hand. The generator is independent of the sealing executor and refuses forged records, foreign designs, missing, extra or duplicated seeds, population mismatches, unpaired naives, non-finite values and disagreeing classes. It has 9 tests (`tests/test_m06_closure_table.py`). The orchestrator recomputed the three-seed mean independently and got the same values.

## 4. Machine-readable state and heartbeat audit

- `STATUS.json` is written by `tools/m06_status_writer.py`, which uses the standard library only and runs no LLM. It runs as unit `m06-status-writer.service` through `crispdm-run -m 512M -t 12h` with `python -u`. It polls every 45 s and writes atomically (temp file, fsync, rename) every 120 s and on every job-state change.
- Contents:
  - per host and GPU: jobs, stage, ETA with basis and assumptions, temperature, clocks, decoded throttle reasons, MemAvailable, unreclaimable slab, PSI, and admission headroom and leases;
  - per lane: agent id, worktree, tip and dispatch block;
  - queued-for-admission jobs, found from launcher processes that hold no lease.
- `PROGRESS.png` is generated from `STATUS.json` alone. It shows no weighted percentage.
- Heartbeat audit (at most 60 s apart):
  - **Traffic s2021 and s2023 fail.** `author_stdout.log` is block-buffered, with gaps of 17 minutes or more, and `checkpoint.pth` is written only when validation improves. The fix for future runner snapshots is `PYTHONUNBUFFERED=1` or `-u` plus a per-epoch heartbeat JSON. The executing source was not edited.
  - m02-synthetic-pilot passes: unbuffered, with `heartbeat.jsonl`.
  - The m03 profiles have a 5 s lease heartbeat only; M06 did not verify a progress heartbeat.
  - The status writer passes (`HEARTBEAT.json` every ~45 s).

## 5. Independent review: what is measured, retained, synthetic

| item | class |
|---|---|
| Traffic h96 seeds 2021, 2022 and 2023 | new measurement (this session's cells), per-seed rows, NOT independently verified |
| Traffic three-seed class | MSE and MAE OPERATIONAL_AGREEMENT (three-seed mean, executor classify()) |
| M01/M02 component tests and M02 synthetic pilot | synthetic component checks, not forecasting results |
| TF GPU registration on both workers | environment facts, not model evidence |
| Slab-growth cause on worker_a | hypothesis, UNVERIFIED (no root; cache not named) |
| Admission fix | implementation plus simulated-host tests; not deployed, not exercised on a real cgroup write |

## 6. Report (M5PHET §6 shape)

```
M06 — evidence and resources
repo/branch/tip: predictor satoshi/m06-evidence-resources-20260930 @ 0e521881 (and later evidence commits) ; predictor satoshi/m06-admission-dead-cache-20260930 @ 0928dc06
files: tools/m06_status_writer.py, tools/m06_traffic_closure.py, tools/m06_progress_png.py,
       docs/audits/evidence/MODULAR_CAMPAIGN_20260930/{RETURN.md,STATUS.json,registry.json,RESULTS/,PROGRESS.png,HEARTBEAT.json,worker_a_slab_series.jsonl},
       tools/crispdm_admission.py, tools/crispdm-run, tests/test_crispdm_admission.py (admission branch)
suites: tests/test_crispdm_admission.py 45/45 (simulated host, run on worker_b under crispdm-run -m 1G)
acceptance: Traffic h96 rows via executor comparison_row (3 records, digests recomputed); MSE and MAE OPERATIONAL_AGREEMENT (three-seed mean, executor classify())
what is NOT done / refused / not measured: slab cache not named (needs root); slab cause unverified;
       admission fix not deployed (review); dead cache not reclaimed (owner); branch history af215de3 still
       contains a host-named default path (force-push denied; owner's call); M03 dispatch block missing;
       M04 cost pilot queued behind ADM-DEADCACHE-01
```

Signed: Satoshi, successor technical lead. 2026-09-30.

