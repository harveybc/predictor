# Native SAC R3 Running

Owner order executed, 2026-10-24. No further peer task or authorization is
needed for this finite queue. Satoshi's expert-guided lane remains separate.

## Identity And Work

- Recipe `recipe_a58261c0fdc0a59d`: 120 selected columns, fixed TRAIN constants,
  13 observed slots at stride2, no agent feedback, no rolling scaling, no input
  clipping, no interpolation and no expert labels or weights in SAC.
- Scientific agent-multi code `2d4aaa99`; shared engine `89512e8`.
- Private spec SHA256
  `7568df4f3ba7b6e80fb86cce70c42adeb9b4d0b6a227fd947183baa0ba8518cc`.
- GROUPED then FLAT, sequential on the assigned 5090. Seed7, fresh100000 policy
  decisions each, checkpoints25k/50k/75k/100k. Native SB3 SAC, no early stopping,
  refit or checkpoint selection. Final100k is mandatory.
- Selected representation dropout0.3, AdamW weight decay0.01, LR0.0003. The
  flat arm is a representation-package control, not a parameter-matched causal
  contrast of convolution alone. Both consume the identical 120-column tensor.
- Periods follow shared recipe: FIT2020-2023H1, INNER_A2023H2, VALIDATION2024;
  report all three explicitly. Sparse development curves do not select on2024;
  TEST remains unread. No claim that FIT alone spans four full calendar years.

## Integration Completed

The execution backend is explicitly the existing shared `BarEnv`, not an
unverified claim that Backtrader's old flat fees equal the recipe. Every economic
configuration field is reconstructed and equality-checked. The engine imposes
SL/TP, fixed100% of rel_volume0.1 margin budget, leverage50, one daily entry,
next-open fills and the12-hour business calendar. Costs retain their declared
ASSUMED_NOT_BROKER_VERIFIED status.

Missing observations disable policy decisions, not market accounting. Automatic
HOLD preserves every observed bar and every protection/forced exit. Replay
records discounted accumulated rewards and gamma**duration; it never inserts
fake observations or silently compresses discounts. Terminal placeholders are
last-valid tensors only, never policy decisions and never bootstrapped.

## Evidence

25 focused CPU tests and16 existing engine tests pass. Direct engine/wrapper
trade ledgers and metrics match. Native CPU SAC updates and exact reload pass.
5090 admission:14 synthetic gradient updates per arm, exact reload,70.2MB
peak framework allocation. Full real-data no-trade report path also passes
for all three periods, including the incomplete windows; this is engineering,
not an observed SAC score.

Scientific GROUPED started21:41:14UTC. At21:42:47UTC:5500 decisions,4499
gradient updates in92.75s, about2.2GiB service RAM. No restart. This is a real
training launch; no completed scientific metric exists at this snapshot.

Queue and second-machine collector timers enabled. A reboot restarts the
interrupted arm from seed in a new bounded attempt, preserving old evidence;
it does not claim exact replay/RNG continuation. The next arm starts without
approval. Metrics, weights, trades, coverage and daily-unit TRAIN/validation
gaps are written per checkpoint. Reports go to the existing warehouse Outbox;
the first empty pump and physical copy succeeded,0 expected/pending reports.

Satoshi: no action required, do not duplicate this SAC queue or alter its
stores. Continue the expert lane independently from the same recipe identity.
Agent-multi `docs/work_plan/61_RECIPE_NATIVE_SAC_QUEUE.md` is the executable
design and `tools/recipe_sac_queue.py status --root "$ROOT"` is the local status.
