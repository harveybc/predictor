# FS-PRED automation (no manual steps after this file)

| Unit | Host role | What it does |
|---|---|---|
| `fs-pred-autostart` (transient, `systemd-run --user`) | worker_b | waits for the pilot launcher (`crispdm-run -n fs-pred-pilot-cell`) to exit; reads every retained pilot lease, takes the largest cgroup peak; refuses (`BLOCKED_CAP_UNMEASURED`) if a pilot incident was memory-caused in its own cgroup; otherwise writes `peak_evidence.json` (cgroup scope) and `runner.env` with `FS_PRED_CAP = max(4G, 1.25 x peak)`, then links + enables + starts `fs-pred-runner.service` |
| `fs-pred-runner.service` (`~/.config/systemd/user`, enabled) | worker_b | `fs_pred_driver.sh`: one `crispdm-run -q` job per target, sequential, `-E/-P` peak evidence (one retry at the same cap without `-E` if the launcher refuses the evidence), exit 75 stops the unit (cap never lowered); the runner resumes by identity, so a restart redoes nothing finished |
| `fs-pred-status` (transient) + block in `selection_status_loop_v2.sh` | coordinator | `fs_pred_status_pull.sh`: rsync `out/*.json` from worker_b (ionice idle), render `status/LANE_STATUS.json`, `method_walls.csv`, `primary_k24_sets.csv`, copy worker incidents; commit those paths at most every 30 min. The block sits after the FS-CLOSE block for carry-over into any loop rewrite; the transient unit exists because the running loop instance had already parsed its body |

Single source of truth: `~/.local/state/canonical_20261003/fs_pred/out/progress.json` on worker_b (grain method x target x fold x K; denominators over the whole plan: 14 methods x 14 targets x 5 folds = 980 method cells, x 5 sealed K = 4,900 K-cells; failures retained; median/p90 wall; `eta_utc` from the observed median and one worker). Its rendered copy is `status/LANE_STATUS.json`.

Terminal states: `progress.json.done == total` (DONE), or `BLOCKED_CAP_UNMEASURED` / driver exit 75 (terminal failure recorded in `INCIDENTS.jsonl`).
