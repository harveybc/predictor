# Satoshi: literal parallel dispatch, 2026-10-02

## Purpose

Keep the two active scientific GPU lanes running, and use Hermes/OpenCode Go for
small, bounded implementation and inspection tasks. Do not use a long primary
agent session to rediscover paths, repeat diagnostics, or decide the experiment.

This order is operational. Execute the commands exactly; if a command fails,
record its stdout/stderr once and mark that task failed. Do not retry it with
variants unless this document explicitly says to do so.

## Non-negotiable rules

1. Do not stop, restart, duplicate, or modify either active training lane.
2. Never run more than one seed for a new exploratory cell. A second or third
   seed is allowed only after a declared paired comparison requires it.
3. A result table containing MAE or MSE must include the same-row persistence
   naive value and skill. Do not route a model to heuristic-strategy unless it
   beats its relevant naive under the declared gate.
4. Hermes tasks must be isolated worktrees and have at most 12 tool turns.
   Their prompts are below; do not extend them with open-ended research.
5. No `find` over `/home/harveybc/Documents/GitHub`, no `git worktree list`,
   no recursive repository scan, and no repeated `--help` calls in the primary
   agent. Every path required below is explicit.

## Active GPU lanes: observe only

### Gamma RTX 5090

Owner: F2 forecasting campaign. Current child:

```text
/home/harveybc/anaconda3/envs/tensorflow/bin/python -u tools/eth_cell_runner.py \
  --campaign-id f2_eth4h_long_h6to36_5090_v1 \
  --label grouped32_huber_adamw \
  --gpu-uuid GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8
```

Do not launch another process on this GPU. The campaign runner owns its queue.

### Dragon RTX 4090

Owner: RL temporal queue. Current child:

```text
/home/harveybc/anaconda3/envs/trading-stack/bin/python \
  tools/run_rl_temporal_cell.py \
  --cell examples/config/rl_temporal/eth_4h_frozen/RL-S0_seed202.json \
  --device cuda
```

Do not launch another process on this GPU. The queue already contains the
matched frozen/differentiated SAC/DQN cells.

## First command: one status snapshot only

Run once from omega. Save its output verbatim at
`docs/audits/evidence/parallel_dispatch_20261002/GPU_SNAPSHOT.txt` in the
dedicated status worktree; do not put it in a source worktree.

```bash
set -euo pipefail
ssh -o BatchMode=yes gamma \
  'nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader'
ssh -o BatchMode=yes dragon \
  'nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader'
nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

## Hermes tasks: launch exactly these four, once each

Use the installed Hermes command. Hermes is configured for `deepseek-v4-pro`
through OpenCode Go. Each task may edit only its own worktree. None may launch
a GPU process.

### H1 — feature-extractor compatibility adapter (highest priority)

```bash
cd /home/harveybc/Documents/GitHub/feature-extractor
/home/harveybc/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main chat -Q \
  --worktree --max-turns 12 --accept-hooks --source tool \
  -q 'Read AGENTS.md, app/autoencoder_manager.py, app/autoencoder_helper.py, app/plugins/encoder_plugin_vae_small.py, and input_config.json only. Implement a minimal typed adapter that accepts an already materialized `(samples, steps, channels)` float32 NPZ and emits a `.keras` encoder plus JSON manifest with: input SHA256, train-row ids SHA256, architecture id, seed, reconstruction MAE, reconstruction MSE, and weight SHA256. Add tests proving it rejects non-finite tensors and uses no rows outside the supplied train NPZ. Do not read validation/test, do not launch training, do not modify existing legacy entry points. Commit the patch and report the exact train command.'
```

### H2 — feature metric table materializer

```bash
cd /home/harveybc/Documents/GitHub/feature-eng
/home/harveybc/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main chat -Q \
  --worktree --max-turns 12 --accept-hooks --source tool \
  -q 'Read AGENTS.md and only the existing ETH 4h feature inventory/data-contract code. Implement a train-only metrics materializer for the existing 83 input features. For every feature write a typed row with availability, missingness, variance, ADF/KPSS status if the installed dependency supports it, autocorrelation at lags 1/6/24, spectral entropy, dominant trailing period estimate, and target association at horizons 6/12/18/24/30/36. Persist row ids, fit-scope, data digest, and parameter digest. Tests must prove that changing validation/test cannot change the output. Do not select features, make causal claims, or launch GPU work. Commit and report the exact materialization command.'
```

### H3 — causal ladder data feasibility adapter

```bash
cd /home/harveybc/Documents/GitHub/causal-inference
/home/harveybc/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main chat -Q \
  --worktree --max-turns 12 --accept-hooks --source tool \
  -q 'Read AGENTS.md, README, and existing point-in-time event/calendar ingestion code only. Implement a pure train-only feasibility report for the three-stage causal ladder: association candidates; observed historical intervention strata; counterfactual-support overlap. It must report support and diagnostics, never label an effect causal merely because a method runs. Tests must reject future timestamps, missing availability timestamps, and empty intervention strata. Do not train a model, download data, or touch live services. Commit and report the exact report command.'
```

### H4 — campaign status and progress evidence

```bash
cd /home/harveybc/Documents/GitHub/.worktrees/predictor-modular-neat-20261001
/home/harveybc/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main chat -Q \
  --worktree --max-turns 10 --accept-hooks --source tool \
  -q 'Read docs/audits/evidence/modular_architecture_20261001/SUMMARY.json and tools/render_modular_progress.py only. Update the summary so verified top64-MI is marked VERIFIED, R3 is marked measured but replay-outside-tolerance, and no future item is shown as complete. Regenerate PROGRESS.png, add a compact pending-jobs JSON with owner, GPU, dependency, and next artifact, and add tests for those status labels. Do not alter any metric values or source model code. Commit the patch.'
```

## GPU assignment after H1 commits

Only after H1 reports its exact command and the command's tests pass:

1. Dispatch one single-seed branch-extractor reconstruction pilot to Gamma's
   RTX 5070 Ti, capped at 30 minutes wall time and 8 GiB. It must use the
   materialized ETH TRAIN NPZ only. Record reconstruction MAE/MSE; this is a
   diagnostic, not a feature-selection result.
2. After H2 commits, materialize the metrics table on CPU while that 5070 Ti
   pilot runs. The table informs grouping and extractor family selection; it
   does not itself drop features.
3. Omega RTX 4070 stays available for a separate, small reconstruction pilot
   only after H1's contract is installed locally. Do not use it for a legacy
   VAE configuration that lacks the typed manifest.

## Reporting format

Every 30 minutes while any job is alive, write one JSON object, one line, to
`docs/audits/evidence/parallel_dispatch_20261002/STATUS.jsonl`:

```json
{"utc":"...","host":"...","gpu_uuid":"...","lane":"...","state":"RUNNING|QUEUED|DONE|FAILED","seed":2021,"config_digest":"...","elapsed_seconds":0,"eta_seconds":null,"artifact":"...","naive":null,"mae":null,"mse":null,"reason":null}
```

At completion report only: changed paths, commands actually executed, tests,
one table of new metrics with naive, and the next queued task. Do not narrate
failed speculative attempts.
