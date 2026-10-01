# RL: SAC and DQN with and without modular temporal representation

Owner addition, 2026-10-01 UTC. This is a parallel experimental work plan, not
a report of completed RL training. Satoshi assigns an agent under M05; policy
training belongs in agent-multi/gym-fx, representation exports in predictor /
feature-extractor, search in DOIN, consumption in LTS. Reuse existing tested
algorithm implementations and plugin interfaces rather than a new RL engine.

## Start conditions and parallel work

Implement environment/adapters, tests, configs and monitoring immediately.
Real-data pilots start when the selected-feature manifest for a specific task
is frozen and its required lake resources are available with causal timing,
splits and provenance. Selection uses training/inner validation only; reserve
policy evaluation periods from feature selection, AE fitting and normalization.
Use the first complete eligible batch; exhaustive inventory-wide research,
other forecasts and reference reproduction are not prerequisites.

Missing data for one task blocks that task only. A random-initialized modular
arm can run without pretrained donors; frozen/fine-tuned arms require compatible
donors. Never relabel a random arm as a pretrained one to fill the queue.

## Primary experiment matrix

| Arm | Algorithm | Representation |
| --- | --- | --- |
| RL-S0 | SAC | Declared native baseline, no differentiated feature branches |
| RL-S1 | SAC | Per-feature/group temporal branches, fusion, positional encoding, Transformer core and progressive temporal bottleneck |
| RL-D0 | DQN | Declared native baseline, no differentiated feature branches |
| RL-D1 | DQN | Same modular representation contract as RL-S1 |

Primary contrasts are RL-S1 minus RL-S0 and RL-D1 minus RL-D0. Within each
contrast, preserve selected input information, observation window, environment,
action semantics, costs, initial balances, chronological episodes and paired
seeds. Freeze reward and evaluation objective before fitting. Native baseline
architecture, its time handling and parameter counts must be documented; it is
a control, not a replacement for the requested modular candidate.

Standard DQN requires discrete actions; SAC may use continuous ones. Publish
that difference when applicable. An algorithm comparison on identical discrete
actions needs a tested discrete-SAC implementation, not rounding continuous
actions while calling the result the same policy. Resolve actual installed
algorithm/action compatibility before sealing configs.

Reuse the existing R0/R1/R2 definitions for secondary modular contrasts:
random representation, pretrained/frozen, pretrained/fine-tuned. Declare branch
and core regimes separately and use identical donor initialization for R1/R2.
The donor availability must not stop the ready R0 comparison. Keep equal finite
search budgets within comparisons and account for all pretraining costs.

## Integration and acceptance

| ID | Required observable behavior |
| --- | --- |
| RL01 | A real lake read resolves the selected feature order, causal timestamps and immutable resource identities; no reserved future data enter an observation. |
| RL02 | Modular time remains present through branches, fusion and core; task-head reduction is explicit. The baseline receives the same source information. |
| RL03 | SAC actor and critics, and DQN online/target networks, consume compatible representations. Shared versus separate encoders and optimizer ownership are explicit; no accidental double updates. |
| RL04 | R1 weights remain fixed; R0/R2 update under the declared optimizers; DQN target synchronization and SAC target-critic updates preserve encoder identity correctly. |
| RL05 | Save/reload restores policy, representation, normalization and action mapping; deterministic evaluation in a fresh process reproduces decisions within declared device tolerances. |
| RL06 | Chronological validation episodes drive early stopping, selected-policy restore and bounded evaluation cadence. Training episode reward is not held-out performance. |
| RL07 | Costs, slippage, leverage/margin, rejected orders, equity and open terminal positions reconcile. No simulated fill is manufactured at termination. |
| RL08 | Warehouse results bind task, data, representation, seed, algorithm, actions, reward, evaluation population, costs and measured resource use. |

Write behavior tests before implementation; synthetic fixtures prove mechanisms,
not trading quality. Preserve existing algorithm contracts and compatibility.
Measure a bounded train-only pilot per materially different resource class,
then execute the finite paired-seed matrix under existing admitted limits.

## Metrics, scheduling and output

The heuristic forecast eligibility gate is separate: policies do not have to
produce MAE/MSE or beat persistence to enter an RL experiment. Evaluate policies
against no-trade and the declared feasible trading baselines on identical
episodes. A heuristic baseline is included only if its forecasts passed the
owner's gate; otherwise label it unavailable, do not manufacture its result.

Report net return, drawdown, Sharpe with frequency/annualization and undefined
cases, turnover, trade count and exposure. Give paired seed/episode contrasts,
not just best-seed numbers. Every forecast MAE/MSE shown anywhere still carries
its matching same-row naive. No financial winner or equivalence claim based on
an engineering fixture or a few unqualified training episodes.

Prioritize the external 5090 when admissible; independent cells can use the
other admitted GPUs without displacing valid live jobs. Share CPU preparation
and existing artifacts, not aggregate RAM assumptions across devices on one
host. Preserve the forecast optimization lane and integrate results as cells
finish. Heartbeat <=60 s, status <=5 min, report every 30 min while active.
ETA separates measured runtime from queue delay; the pilot establishes the
first runtime estimate. Finalists enter existing LTS shadow/paper evaluation,
not automatic real-money trading.
