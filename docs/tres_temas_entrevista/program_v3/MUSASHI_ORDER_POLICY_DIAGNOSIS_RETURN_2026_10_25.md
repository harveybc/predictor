# SAC order-policy diagnosis: closure and next contrast

Finished 2026-10-25T06:55:14Z. Agent-multi producer `0998df03`, 31 focused
tests, ten exact TRAIN/INNER_A metric/order replays, 32 manifested files verified
on a second machine. No new fit, TEST scoring or modification of parent artifacts.
Report SHA256: `faf136ecabda9282a2fd39327c19235e6b11db7709c992a421ef6d0e5dcd639b`.

The order campaign itself finished at 03:57UTC, with 24/24 financial reports
published and read back. TARGET reproduced the parent model exactly. COMMANDS5
final has no trades on TRAIN, INNER_A or the already-observed 2024 development year.

## What was learned

- At 25k, every deterministic decision on TRAIN and INNER_A is OPEN_SHORT.
  Its positive internal-period result does not demonstrate learned entry timing.
- At 50k and 100k, every decision is CLOSE_LONG while flat; the engine correctly
  retains HOLD. At 75k, CLOSE_SHORT dominates, with a few executed entries.
- Final INNER_A stochastic samples also concentrate in CLOSE_LONG:8179/8192.
  This is not just an artifact of using the deterministic policy at evaluation.
- The final critic prefers the +0.99 CLOSE_LONG probe in256/256 sampled
  observations in both periods. This is a learned estimate, not realized profit.
- Final train-mode dropout probes change no command; temporal embeddings are
  nonconstant. Neither fact proves dropout or representation harmless during
  learning. No replay buffer was retained, so Bellman calibration is unresolved.

Detailed evidence and limitations:
[agent-multi diagnosis return](https://github.com/harveybc/agent-multi/blob/codex/rl-static-temporal-20261018/docs/work_plan/64_ORDER_POLICY_DIAGNOSIS_RETURN.md).

## Next, without mixing the expert lane

Define a paired critic-only dropout contrast before any larger sweep: actor,
inputs, weight decay, business rules, observations and budget unchanged. This
tests a learning-stability hypothesis; it is not a demonstrated fix. Inventory
feedback/routing, action masks, reward penalties and categorical/hybrid policies
remain separate explicit designs, never hidden changes. No successor fit has
been launched by the diagnostic. Satoshi's label-generation work is independent.
