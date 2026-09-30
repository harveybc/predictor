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
