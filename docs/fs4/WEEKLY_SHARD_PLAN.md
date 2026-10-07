# FS4 weekly campaign: cost, caps and shard/priority plan (measured 2026-10-07, TRAIN only)

Pilot: three December-2023 weeks (4, 11, 18 Dec; each scored week ends before the TRAIN cut), one small, one median and one
large set per population, each fit its own `crispdm-run` job through `fs4_weekly_wrapper.py run-task --pilot-train-only`
on dragon (CPU only, `CUDA_VISIBLE_DEVICES=""`). No VALIDATION byte was opened. Raw rows: `WEEKLY_COST_PILOT_*.json`;
derived inputs: `WEEKLY_COST_TABLE.json`. The pilot decides cost only.

| set size | peak (cgroup) | cap = 1.25 x peak | wall per fit | cpu-seconds per fit |
|---|---|---|---|---|
| EURUSD 4 / 12 features | 1.28 / 1.34 GB | 1.7 GB (small class) | 17-24 s / 17-20 s | 70-120 / 120-180 |
| EURUSD 365 (ALL_ADMISSIBLE) | 4.81 GB | 6.0 GB (large class) | 116 / 136 / 366 s | 2,600 / 3,040 / 8,650 |
| ETH 4 / 16 features | 0.54 / 0.71 GB | 1.7 GB class cap covers it | 8.5-10.7 s / 10-25 s | 22-35 / 58-224 |
| ETH 78 (ALL_ADMISSIBLE) | 0.93 GB | 1.2 GB | 25-31 s | 340-400 |

Stage 1 (RAW, sealed frontier x 52 weeks): EURUSD 350 sets = 18,200 fits, ETH 124 sets = 6,448 fits, 24,648 in all.
Sum of per-fit wall seconds: EURUSD 139-149 h, ETH 24-27 h (lower bound flat per size class, upper bound linear in F).
CPU: 1,710-2,070 core-hours (EURUSD 1,557-1,861, ETH 154-209). The 14 EURUSD ALL_ADMISSIBLE sets alone are 728 fits,
about 963 core-hours and 41.6 h of single-slot wall time (each such fit keeps about 24 cores busy).

Hosts that may take compute: dragon (32 cores, about 12 GiB available) and gamma (32 cores, about 7 GiB available); the
coordinator takes none. Memory (cap + 3 GiB desktop reserve) allows about 5 small slots on dragon, 2 on gamma and one large
slot on dragon. Core-bound estimate on 64 cores at 75 % utilisation: stage 1 is about 36-43 h, i.e. close to the 48 h line,
so the frontier is NOT reduced; the work is sharded by size class and ordered by priority instead.

## Slots (workers pull with `claim`; nothing is assigned by hand)

- SMALL class: `FS4_MAX_FEATURES=32`, `FS4_CAP=1700M` (EURUSD) - dragon x5, gamma x2. Pull order is week-major, RAW first.
- LARGE class: `FS4_MIN_FEATURES=33`, `FS4_CAP=6G` - one slot on dragon (EURUSD ALL_ADMISSIBLE). ETH large sets (78) fit the
  small cap's neighbour, so ETH slots use `FS4_CAP=1200M` with no feature filter.
- Priority: (1) stage 1 RAW week-major so every week closes early; (2) ALL_ADMISSIBLE large sets start at once on their own
  slot because they dominate wall time; (3) stage 2 (RANDOM_ENCODER, TRAINED_ENCODER) only after `stage2` computes the list.
- Stage 2 cost is NOT measured (it needs the runner's TRAINED_ENCODER terminals); upper bound per encoder mode is the RAW
  cost of the stage-2 list (its core is smaller on 6 latent steps, plus one encoder forward pass per feature).
