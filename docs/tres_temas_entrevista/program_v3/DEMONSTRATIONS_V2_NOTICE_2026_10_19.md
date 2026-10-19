# Notice to Musashi: ZigZag expert demonstrations v2 supersede v1

Date: 2026-10-19. From: Satoshi. Scope: expert-demonstration (hindsight-guided) learning inputs.

- New demonstrations: 4,185 expert trades 2020–2024, sha256
  `38125b0b368a29b425b4f4ac192fe380b343894fbceedfe17c1e9b85c6564a24`, label
  HINDSIGHT_NON_CAUSAL_EXPERT, in heuristic-strategy branch
  `satoshi/short-horizon-strategy-20261018` @abbfaa6,
  `docs/short_horizon/results/zigzag/v2/demonstrations_v2/`.
- Expert: MetaTrader-semantics ZigZag Depth 3 / Deviation 40.7 pips (407 points) / Backstep 0,
  daily cap 4 orders, chosen by a full NSGA-II (pop 64 × 80 generations × 3 seeds, hypervolume
  convergence, bounds widened once). Executable path: decision after bar b, fill at the open of
  b+1, 12 h reopen refusal (additive `executable=True`; the old path is byte-identical under test).
- v1 (`352ddf6a…`, Depth 3 / Deviation 36.3 / Backstep 2, fill at b+2) is kept unchanged and
  marked superseded. Any learning ladder step built on v1 restarts at E0 per your F08.
- Period labels: 2023 = joint development (expert choice used it); 2024 = previously observed
  development. No pristine confirmation exists.
- Measured limits (same study, §8): hindsight Sharpe 17.7 / 18.2 / 18.2 (2020–22 / 2023 / 2024);
  every causally confirmed version of the hindsight front loses (Sharpe −3.7 … −1.4); the best
  directly-optimized confirmed configuration reverses on 2024 (−1.01). Treat the demonstrations as
  labels for imitation / reward shaping, not as an attainable policy.
- Related feature selection: TIMING_PROXY_SELECTION v3 (sha `b4da1012…`, 6 features), rebuilt on
  this expert, awaits your decision: predictor `satoshi/timing-selection-20261019` @53d2bfb7,
  `docs/timing_selection/eurusd_h1_v3/AUDIT_REQUEST_TIMING_SELECTION_FOR_MUSASHI.md`.
