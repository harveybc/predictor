# Modular campaign 2026-09-30: return packet (lane M06, evidence and resources)

Everything in this directory comes from the same evidence:

| File | What it is |
|---|---|
| `STATUS.json` | Fleet state in `modular.program.status.v1`. `tools/m06_status_writer.py` rewrites it atomically every 120 s and on every job-state change. The writer runs as a capped background unit (`crispdm-run -m 512M -t 12h`) with no LLM in the loop, and heartbeats to `HEARTBEAT.json` every ~45 s. |
| `registry.json` | Hand-kept inputs the writer merges into `STATUS.json`: agents, results, milestones, events, recipes and the heartbeat audit. |
| `RESULTS/traffic_h96_rows.json`, `RESULTS/traffic_h96_results.csv` | Per-cell closure rows produced by the sealing executor's own `comparison_row()`. The three-seed class comes from its `classify()` once all seeds are sealed. |
| `PROGRESS.png` | Rendered from `STATUS.json` by `tools/m06_progress_png.py`. It has no weighted completion percentage. |
| `worker_a_slab_series.jsonl` | One sample of the 5090 host's unreclaimable slab and GPU re-init count per writer cycle, taken after the reboot. |

Hosts are named by role only (coordinator, worker_a = the 5090 host, worker_b = the 4090 host) and GPUs by UUID.

The latest state and the full report are in
`docs/audits/work_plan/SATOSHI_M06_EVIDENCE_RESOURCES_2026_09_30.md`.

## Return, 2026-09-30 (as of 23:35Z)

### New numerical results

This is a new measurement from this session's cells, not a retained result.
- **Traffic h96, TimeFilter published recipe, L=96, seeds 2021, 2022 and 2023.**
- **Mean MSE 0.3751992683:** published 0.375, OPERATIONAL_AGREEMENT.
- **Mean MAE 0.2511426806:** published 0.251, OPERATIONAL_AGREEMENT.
- **Persistence on the same rows:** 2.7144524181 / 1.0772232192, so MAE skill is about 0.767.
- **Classification:** assigned by the sealing executor's `classify()`.
- **Per-seed rows:** in `RESULTS/traffic_h96_rows.json` and `RESULTS/traffic_h96_results.csv`.
- **Verification:** the result is measured, not replayed. Only s2023 was checked independently, by the orchestrator.

### What became executable

- The 5090 host is memory-admissible again after the owner's reboot, and TensorFlow can use it with the cu12-only `LD_LIBRARY_PATH` recipe. The same recipe also works on worker_b.
- `STATUS.json` is written by a capped background writer.
- `PROGRESS.png` is rendered from `STATUS.json`.
- The ADM-DEADCACHE-01 admission fix is reviewed and accepted, but not deployed. It is on branch `satoshi/m06-admission-dead-cache-20260930` at `0928dc06`, with 45/45 tests passing.

### What failed or is blocked

- **M04's 6G cost pilot:** it is queued on the 5090 host behind dead clean page cache that is charged as live use. Unblocking it is the owner's step: deploy the fix with `DEPLOY.md`, or run a one-time reclaim.
- **The Traffic runners:** they lack a heartbeat. Their logs are block-buffered.
- **Unreclaimable-slab growth on the 5090 host:** the cause is not diagnosed. Naming the slab cache needs root access.

### Remaining dependencies

- The owner's deployment or reclaim on the 5090 host.
- The owner's decisions on persistence mode and a root slabinfo snapshot.
- M03's dispatch block.

### Next

- M04 relaunches its cost pilot once admission clears.
- The free GPU slots right now are the coordinator's 4070 and worker_b's 4090.
- The writer keeps sampling the 5090 host's slab on every cycle.

## Corrections (2026-09-30, 23:55Z)

1. **Who rebooted the 5090 host.** My final chat report said the orchestrator ran the apt upgrade and the reboot on the preferred worker (22:52–22:53Z). That is **wrong**. The **owner** did both, at his own terminal; neither the orchestrator nor any other agent was involved. This directory's evidence files already said so. The error was only in the report text, and it is recorded here as a correction rather than silently edited.
2. **M03 declaration digest.** `declaration_sha256 c7e20f15` is now **VERIFIED** by the orchestrator. The digest is taken over canonical JSON: sorted keys, separators `(",", ":")`, default `ensure_ascii`, and the body without the digest field itself, as in `tools/declare_admissible_inputs.py`. It had been recorded as UNVERIFIED.
3. **Lane states.**
   - Returned: M02 (tip 310a836f; tested code 9844dafd, 71 of 71 tests), M05, and M06.
   - Dispatched at about 23:25Z as post-consolidation lanes: C07 and S07.
   - Still working: M01 (work in progress, untested) and M03 (profiling).
   - M04: its pilot is running on the 4090 and blocked on the 5090.
