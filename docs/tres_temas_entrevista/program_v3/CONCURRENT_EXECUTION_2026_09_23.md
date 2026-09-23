# Concurrent experimental execution

Authority: owner's explicit instruction, 2026-09-23. Effective now, not after
another review. The objective is continuous useful experimental throughput,
not GPU utilization produced by filler jobs or unsafe execution.

## Responsibilities and isolation

- Musashi owns scientific dependencies, independent audit and the next queue.
- Satoshi owns dispatch, measured capacity, campaign state and integration.
- Experiment worker runs pinned, governed designs in an immutable checkout and
  dedicated output root; it never pulls a branch while a cell is running.
- Repair worker uses a different worktree and bounded CPU. Its changes cannot
  silently alter a running experiment's code, environment, data or configuration.
- Analysis worker verifies completed cells and prepares the next design while
  other independent cells train. Serialize GPU inference with training unless
  measured admission proves concurrency fits; CPU/catalog work is also budgeted.

Use available Hermes/subagent instances for these disjoint roles, after checking
their existing assignments. No duplicate campaign launch, credential copy into
documents, restart of another agent, or new unbounded agent pool. If an agent is
unavailable, Satoshi can run the admitted experiment with the existing durable
runner and work on an isolated repair tree while it executes. A new scheduler
implementation is NOT a prerequisite for using the existing runners.

One dispatcher owns each physical GPU. Existing locking/admission must prevent
two agents reserving it independently. Follow existing memory, thermal and disk
guards. Repair tests cannot take all host RAM/CPU and starve the experiment.

## Queue and dispatch

The initial finite queue is
[EXPERIMENT_EXECUTION_QUEUE.json](EXPERIMENT_EXECUTION_QUEUE.json).
These entries are ordered, NOT a claim of execution or runtime enforcement.
Satoshi supplies the sealed design digest, immutable producer commit and live
unit/campaign identities before moving a cell to RUNNING.

1. Prepare and admit RP135 first, using its existing recipe and resource pilot.
2. Run twelve cells in fixed horizon-major, seed-minor order; no score-selected
   shortening or changes to the author's recipe. Refresh admission each time.
3. While a cell runs, perform independent repairs and design preparation with
   separate ownership and resource limits. No pause for an unrelated audit.
4. On completion, record observed cost, checkpoint, terminal and the next cell.
   Same-device replay is useful experimental verification, not an idle GPU.
5. If a cell is blocked, record the concrete dependency and continue eligible
   independent cells. Preserve failures. A code fix needing remeasurement gets
   a sealed successor, never an overwritten attempt or silent mid-run patch.
6. Before the finite queue empties, prepare the next experiment already in the
   master plan with its real reference, data and acceptance dependencies. Do not
   revive suspended household pilots or invent work merely to occupy hardware.

Updated after explicit owner cooling confirmation, 2026-09-23: all GPUs are
eligible after fresh admission; external5090 remains first choice. The owner
actually confirmed all fans functional, not merely an expected arrival time.
Distribute independent work by measured memory/runtime, not equal work quotas.
Original-device A replays now proceed on their own hosts and do not hold B's
queue. Do not migrate existing sealed B cells to newly available devices.

## Dependency-local holds

| Condition | Stops | Does not stop |
|---|---|---|
| Deletion certificate defect | Deleting retained artifacts | A new governed fit with enough disk to retain artifacts |
| Stale readiness summary | Claiming that readiness | Actual training/scoring paths already independently checked |
| Original-device cooling unconfirmed | Work on that physical device | Admitted external-5090 cells |
| Fault in consumed data/target/scoring | Affected units and their claims | Units proved not to consume the faulty dependency |
| Insufficient current resources | Admission of jobs that do not fit | Other useful jobs that fit without changing their scientific design |

Thermal/resource refusal must not be bypassed to satisfy a utilization target.
An empty eligible queue is an orchestration incident to diagnose, not a reason
to fabricate an experiment or metric. Record reason, checked alternatives,
next action and timestamps; distinguish admission/preparation/replay/repair/idle.

## Every audit and return

Lead with new experimental work since the previous audit: named campaign and
cells, scientific question, normalized author metrics, same-row naive and
literature value with comparability, evidence state, cost and remaining cells.
Then report defects and their bounded impact. Tests passed are not experimental
results. If no new measurement finished, explicitly say so, name the running
job or actual idle cause, and never relabel an old result as new.

Include per-device useful job time and measured idle time when observable;
unobserved periods are UNKNOWN, not zero. The user should not need to send
"continue" after each subtask. One consolidated completion return, with alerts
only for real failures requiring external facts or material safety concerns.

Do not promise that every future audit necessarily has a new finished score:
the requirement is active useful work and honest reporting, not manufactured
measurements. Receipt of orders, process launch and scientific acceptance are
three separate states and must be reported separately.
