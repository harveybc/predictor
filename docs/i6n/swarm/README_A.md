# Lane A: trajectory evaluator, model and training (execution and status)

Owned code: `tools/i6n_historical_evaluator.py`, `tools/i6n_historical_cell.py`,
`tools/i6n_model.py`, `tools/i6n_training.py`,
`tools/i6n_contracts/temporal_ops.py`. Contract: `INTERFACE_A.json` (A.2).

## What it does

One subprocess evaluates one candidate genome on one frozen (horizon, fold):

1. **Freeze, once per (h, fold), no fit.** This seals the endpoint-h TRAIN/inner support:
   ```sh
   python -m tools.i6n_historical_evaluator freeze --config <cell.json> \
     --output <new dir> --experiment <id> --fold <2023Q1|W2024-01-07|...> [--score-outer]
   ```
   The cell config must carry plan stopping (`max_epochs=50, patience=5,
   min_delta=1e-6, seed=6102026`). Only weekly-refit units pass
   `--score-outer`, which seals the outer week's endpoint support before any fit.
2. **Evaluate in a subprocess:** run `BINDING.i6n_evaluator_command` with one
   JSON request on stdin, `{config, generation, context, support[, attempt]}`.
   It prints one receipt JSON. `config` is `tools.i6n_search.effective_config(genome)`.

Supervision:
- `endpoint`: one head h.
- `trajectory`: heads [dy, 2dy, ..., h], where dy and dx are independent whole hours.

Each head gets its own C1 mask (exact UTC elapsed hours), its own TRAIN-only
target center and scale, its own naive, and an equal loss weight. Invalid
labels are selected away before residuals, so they produce no NaN and no
gradient. A head without TRAIN support is excluded (`NO_TRAIN_SUPPORT`), not
scored as zero. The fitness is the endpoint h on the frozen fold support only.
Intermediate heads and the outer week are C2-scored diagnostics.

## Status files

`<output>/candidates/<identity>[.attempt<k>]/STATUS.json` holds
`RUNNING | COMPLETED | FAILED`, plus progress, `elapsed_seconds` and
`eta_seconds` (null before the first epoch). A RUNNING status whose `RUN.lock`
is free means INTERRUPTED. A COMPLETED attempt is always adopted without a
refit. Attempt k>1 must quote the exact STATUS digest of attempt k-1. There
are at most three physical attempts.

Do not run `evaluate` just to inspect status, because it can start a fit.
Read STATUS.json instead.

Artifacts:
- `model.keras` and `scores.npz`
- `{split}.input.json` and `{split}.c2.json` for the endpoint
- `{split}.h{o}.*` for intermediate heads
- `RESULT.json`, which binds the artifact digests

## Tests (CPU, crispdm-run)

```sh
cd <worktree>
E='CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1'
env $E crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider tests/test_i6n_trajectory_contract.py
env $E crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider tests/test_i6n_trajectory_model.py tests/test_i6n_temporal_ops.py
env $E crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider tests/test_i6n_trajectory_fit_h12.py
env $E crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider tests/test_i6n_trajectory_fit_h72.py
env $E crispdm-run -m 2G -t 120s -- python -B -m pytest -q -p no:cacheprovider tests/test_i6n_historical_evaluator.py tests/test_i6n_search_publish.py
```

Run one suite at a time. The h12 fit suite takes about 60 s.

## Current status (2026-10-11)

- Implemented and unit-tested on invented CPU fixtures.
- No historical fit, GPU run, deployment or TEST access was performed.
- Not yet integrated by B. The legacy local-controller test needs a one-line
  `freeze(..., experiment=, fold=)` update by its owner.
