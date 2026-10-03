# G2 RL: four-arm first-seed screen, v1

Written 2026-10-03T04:20:00Z. Order: docs/handoffs/SATOSHI_CORRECTIVE_PARALLEL_ORDER_2026_10_03.md section 3 row G2, section 4 (RTX 4090), section 6. Host worker_b, RTX 4090 <worker_b-4090-uuid>, cap `crispdm-run -m 3G -t 5h`.

## Labels

Every number below is **DEVELOPMENT_NOT_CONFIRMATORY**. The feature manifest is FROZEN_DEVELOPMENT (83 features). Each cell is evaluated on the **validation** rows [13699,15895), the same episode its checkpoint was selected on, so these are not held-out numbers. The test split was not read. Sharpe is **per 4h bar, not annualized** (ddof=1), computed over the 2196 scored bars after the 280-bar forced-hold context. Costs: commission 0.001 per fill, slippage 0, financing off, initial cash 10,000. One seed per arm is not a comparison, so no ranking of arms is claimed.

## Arms

| Arm | Algorithm | Representation | seed 101 (first) | 202 | 303 | 404 |
|---|---|---|---|---|---|---|
| RL-S0 | SAC | native_flat (conventional) | DONE | DONE (extra) | DONE (extra) | HELD |
| RL-S1 | SAC | modular_temporal, all R0 random init | DONE | DONE (extra) | RUNNING | HELD |
| RL-D0 | DQN | native_flat (conventional) | DONE | DONE (extra) | DONE (extra) | HELD |
| RL-D1 | DQN | modular_temporal, all R0 random init | DONE | DONE (extra) | DONE (extra) | HELD |

The first-seed screen (seed 101 for all four arms) is **complete**, 4 of 4, and no first-seed arm is missing. Seeds 202 and 303 ran before the order and are kept as evidence. `eth_4h_frozen` refers to the frozen feature manifest. It does not mean a frozen encoder. The modular arms are R0 random init; they are not donor-initialized or differentiated (R1/R2).

## Completed cells

| Cell | Net return | Sharpe/4h bar | Max DD | Trades closed | Exposure | Turnover (units) | Commission paid | Wall s | Source |
|---|---|---|---|---|---|---|---|---|---|
| RL-D0_seed101 | 0.3832 | 0.0429 | 0.0995 | 156 | 0.9995 | 313 | 941.44 | 2608 | RESULT.json metrics (scored-bars view) |
| RL-D0_seed202 | 0.2824 | 0.0326 | 0.1301 | 122 | 0.9995 | 245 | 752.56 | 890 | RESULT.json metrics (scored-bars view) |
| RL-D0_seed303 | 0.0078 | 0.0029 | 0.1325 | 245 | 0.9982 | 491 | 1496.20 | 421 | RESULT.json metrics (scored-bars view) |
| RL-D1_seed101 | 0.0451 | 0.0070 | 0.1479 | 35 | 0.9995 | 71 | 234.06 | 12556 | DERIVED_FROM_TRACE (RESULT.json not mutated) |
| RL-D1_seed202 | 0.0596 | 0.0082 | 0.1604 | 118 | 0.9995 | 237 | 692.45 | 3449 | RESULT.json metrics (scored-bars view) |
| RL-D1_seed303 | 0.1786 | 0.0221 | 0.1425 | 6 | 0.9900 | 13 | 45.76 | 8576 | RESULT.json metrics (scored-bars view) |
| RL-S0_seed101 | 0.0707 | 0.0091 | 0.2118 | 102 | 0.9991 | 205 | 629.56 | 1719 | RESULT.json metrics (scored-bars view) |
| RL-S0_seed202 | 0.3579 | 0.0411 | 0.0869 | 16 | 0.9936 | 33 | 101.35 | 1092 | RESULT.json metrics (scored-bars view) |
| RL-S0_seed303 | 0.3541 | 0.0390 | 0.1245 | 73 | 0.9995 | 147 | 442.89 | 1703 | RESULT.json metrics (scored-bars view) |
| RL-S1_seed101 | -0.0027 | 0.0015 | 0.1664 | 0 | 0.7436 | 1 | 3.31 | 14689 | RESULT.json metrics (scored-bars view) |
| RL-S1_seed202 | 0.1062 | 0.0139 | 0.1563 | 0 | 0.9995 | 1 | 2.27 | 9237 | RESULT.json metrics (scored-bars view) |

The no-trade baseline is 0.0 for every cell. No heuristic or buy-and-hold baseline is reported. Most cells sit in a position about 99 to 100 percent of scored bars. Both completed RL-S1 cells closed 0 trades from one entry, which means each held a single position, so a positive return there may be ETH drift rather than policy skill.

RL-D1_seed101 as written (status RESULT, prefix bars included): net return 0.0451, Sharpe 0.0066, max DD 0.1479, exposure 0.8865. The table row is the scored-bars view, derived read-only from its trace.csv.

## Hold, successor and ETA

- Hold mechanism: the queue's own STOP file `worker_b:~/.local/state/scratch/g-rl/runs/gpu_4090/STOP`. The queue checks it before it starts each cell. The held list is in `worker_b:~/.local/state/scratch/g-rl/HELD_CELLS_G2_20261003.txt`: RL-D0/D1/S0/S1_seed404. The running cell was not touched.
- Successor queue: **none created**. There is no missing first-seed arm, so when RL-S1_seed303 exits the queue (pid 2769046) reads STOP and ends. The 4090 then has no G2 work.
- RL-S1_seed303 started 03:54:11Z. The RL-S1 family walls are 14689 s and 9237 s, which gives an ETA between 06:28Z and 07:59Z (mean 07:13Z). The crispdm hard timeout falls at 08:54Z.
- Release: remove STOP and launch a new `rl_temporal_queue.sh` with the held list. Cells that already have RESULT.json are skipped.

## Defects found

- RL-D1_seed101/RESULT.json was never relabelled: status RESULT and its Sharpe/exposure include the 280 forced-hold context prefix bars; the other 10 results use the scored-bars view. G2 derived the scored-bars values read-only from trace.csv; it did not rewrite the artifact.
- tools/rl_temporal_relabel_result.py: res['metrics_as_originally_written'] = res['metrics'] aliases the same dict, which is then mutated, so 'metrics_as_originally_written' records the NEW values (verified on RL-D0_seed101 against RESULT.pre_relabel.json). Only RL-D0_seed101 kept a pre_relabel backup. Fix: copy.deepcopy before mutation (agent-multi).
- The worker_b agent-multi worktree has a dangling gitdir; the running code has no recoverable git tip.
- No heuristic or buy-and-hold baseline is reported (baselines.heuristic UNAVAILABLE); only no-trade (0.0). Exposure is ~0.99-1.00 in most cells and both RL-S1 cells closed 0 trades with 1 entry, i.e. a single held position, so positive returns may be market drift rather than policy skill.

## Decisions

- D1: the four-arm first-seed screen is seed 101 for every arm (MATRIX order); it was already complete before this order, so G2's 'finish' requires no new cell.
- D2: seeds 202 and 303 were run before the order; they are kept as completed evidence, not discarded and not called the screen.
- D3: hold seed 404 for all four arms via the queue's own STOP file; no successor script, because none is needed and one would only add a dispatcher.
- D4: RL-S1_seed303 runs to completion untouched; it is an extra seed already in flight; killing it would waste its wall and section 6 keeps a retry on the same identity and seed anyway.
- D5: label every number DEVELOPMENT_NOT_CONFIRMATORY on validation with selection-on-validation; no comparison or arm ranking is claimed from one seed per arm.
- D6: report D1_seed101 with a read-only scored-bars derivation and keep its as-written values next to it.

## Code identity

- `note`: worker_b checkout ~/.local/state/scratch/g-rl/wt/agent-multi is a git worktree whose gitdir (agent-multi/.git/worktrees/agent-multi-g-rl-20261001) no longer exists; no git tip is readable, so file digests stand in
- `tools/run_rl_temporal_cell.py`: f1c0dd713bad1caa42360fff63e5af319513121bca5c0caf75cd394fc0a572ec
- `tools/rl_temporal_queue.sh`: f3a81d9d044270cda281bf6ac2fbe4f5ed2555c677f417e3d3d59b005205021c
- `examples/config/rl_temporal/eth_4h_frozen/MATRIX.json`: 5267c7c0035c80a75612b6a3d47de242ab29baf4ae152a130535542f4bb823b3
