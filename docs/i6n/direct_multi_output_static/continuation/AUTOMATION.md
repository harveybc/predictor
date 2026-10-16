# One Launch, Full Continuation

The private launch JSON has `root`, `source_deployment`, `input_dir` (absolute
paths outside git) and optional `strategy_source` pointing to a retained
prepared strategy-source manifest. It contains no token. Existing role Python,
CUDA library environment, warehouse EnvironmentFile, storage and SSH settings
come from the governed deployment. No new credential action is requested.

Once the exact code is committed and tests/data/runtime gates pass:

```bash
crispdm-run -m 4G -t 1800s -n predictive-continuation -- \
  python -B -m tools.i6dmo.resolution_batch continue-bootstrap --spec <private-launch.json>
```

That command freezes and replicates one population and ALL stages/periods,
pins the runtime on every role, and installs the owned sequential GPU timer
and CPU publication/status timer. It does not launch DOIN or an unrelated
agent. GPU admission remains 4 GiB provisional RAM (hard bounded), one 5090
process, no 5070 Ti concurrency. Cost and peak are measured per actual fit.

The generated `BATCH.private.json` lives in `root`. Query once, at any level:

```bash
python -B -m tools.i6dmo.resolution_batch status --spec <root>/BATCH.private.json
python -B -m tools.i6dmo.resolution_batch status --spec <root>/BATCH.private.json --stage joint
python -B -m tools.i6dmo.resolution_batch status --spec <root>/BATCH.private.json --cell dmo_short_ds2_conv
```

Summary is intentionally small; `--level full` explicitly requests every cell.
No agent should loop on these commands. The persisted timer automatically
chooses the next declared cell, adopts an exactly identical authenticated fit,
handles scientifically negative results, and skips gated cadence cells with a
typed reason. Completion of heads, single72, joint or transfer is NOT terminal
for the whole graph. All phases precede final eligibility/delivery/mirrors.

Master status reports exact stage totals, actual fits versus adopted cells,
skips, failures, latest epoch heartbeat, publication and measured/unknown ETA.
An estimate across different model classes is provisional, not a promised
duration. An interrupted scientific fit is retained and not silently repeated;
resource refusal is pre-execution waiting, never a reason to trim the cap.

All numeric metrics, original-scale per-horizon naive, genome/config identity,
seed, costs, exclusions/dispositions and annual cadence comparisons are
published to OLAP with exact readback. Large models/prediction arrays remain
in configured storage with hash-verified local and second-host copies; queues,
locks and SQLite/WAL remain ext4. `RESULTS.md`, `SELECTION.json` and
`PAIR_STRATEGY.json` retain compact navigable results. No GitHub binary upload.

2024 participates in checkpoint/recipe selection: strategy diagnostics there
must say RETROSPECTIVE_DEVELOPMENT_NOT_OOS. Final TEST stays closed by the
scientific gate, not a chat permission. Missing actual input artifacts are
reported by name, never replaced with invented prices or zero metrics.
