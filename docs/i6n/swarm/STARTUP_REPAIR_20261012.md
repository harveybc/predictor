# Production startup repair

The owner started i6nw1. Its ledger and secondary agent started, but no model
training was confirmed. The preferred join timer could not import
`tools.i6n_swarm_deploy` from the exported predictor pin. The secondary agent
passed `--data-root` to an evaluator whose Python API supported it but whose
CLI did not; argument parsing exited with code 2 before preparation or fitting.

The faulty preparation agent was paused. Retain its original ledger and logs;
do not erase failed states, repeat completed cost pilots or relabel these
failures as model-quality results. A newly pinned successor may run the same
scientific plan because these failed preparation attempts trained no model.

## Repairs and evidence

- Forward `--data-root` and `--family` through the evaluator CLI to `freeze`.
  All three endpoint/short/long regression cases failed before and pass after.
- Check the actual exported deploy/campaign/evaluator entrypoints and CLI
  options before installing services. The deployment pin must include the
  deployer itself; checking only the caller's worktree is insufficient.
- Limit fixed preparation to four units per worker-agent invocation by default,
  then synchronize DOIN. This prevents the full preparation queue from
  withholding already-ready GPU work. A bounded-batch regression failed before
  and passes after. Existing library callers retain unlimited batch behavior
  unless they specify the bound.

CPU verification, GPU hidden, admitted at 2 GiB / 120 seconds:

```bash
python -m pytest -q tests/test_i6n_startup_entrypoints.py tests/test_i6n_swarm_deploy.py tests/test_i6n_swarm_integration_a4.py tests/test_i6n_multi_output_arm.py
python -m pytest -q tests/test_i6n_startup_entrypoints.py tests/test_i6n_swarm_agent.py tests/test_i6n_swarm_deploy.py tests/test_i6n_swarm_runner.py
```

First verification: 38 passed in 93.39 s before adding the bounded-batch test.
Second verification: 39 passed in 20.66 s after the batch correction. Test sets
overlap; do not add these counts as distinct tests. The multi-output test uses
invented CPU data, not a new scientific result.

No changes to horizon sets, features, calendars, eligible rows, naive, weekly
refitting, search budgets, model families or early stopping. S3 remains a
retained timeout and is not admitted or retried by this repair. S1/S2 placement
still requires fresh health/resource checks.
