# Lane B interface (B.1, published early)

Machine-readable contract: `INTERFACE_B.json`. Summary for A, C, D and E.

- **CLI:** `python -m tools.i6n_swarm_campaign {plan,preflight,install,start,status,collect}`.
  `status` reads the ledger and heartbeat files only (no TensorFlow import,
  no launch); `collect` never trains.
- **Identities:** `fit_id` = one physical fit = sha256 of
  `{experiment, fold, effective_config_sha256, context_sha256, support_sha256}`;
  plan-independent so verified terminals are reused by identity, never refit.
  `search_candidate_sha256` = sha256 of `{plan_sha256, horizon_hours,
  genotype_canonical_sha256}` (D's fold-independent key).
- **Ledger:** SQLite on the coordinator, reached by workers through a small
  authenticated HTTP service. Claims carry a globally monotonic fencing token
  that is passed to A's evaluator as `attempt.fencing_token`. At most 3
  physical attempts per `fit_id`, at most one linked RECOVERY attempt, OOM /
  timeout / evaluator refusal are terminal FAILED and never retried with a
  smaller cap.
- **Evaluation:** the DOIN node runs the shared wrapper
  (`doin_plugins.predictor.optimizer`) in `i6n_swarm` mode; the I6N DEAP
  plugin calls `tools.i6n_swarm.bridge:evaluate`, which fans out over the
  plan folds, claims each fold in the ledger, and runs A's evaluator once per
  fold inside `crispdm-run`. Fitness = unweighted mean over folds of
  `tools.i6n_search._score(fold receipt, fold support)`; any missing fold
  means no fitness.
- **Role:** `higher_is_better=false`, `shared_population_seed=6102026`,
  patience 3, improvement tolerance 1e-4, reproduction failure fail-closed.
- **What B assumes from C and E** until they publish: see `consumes` in the
  JSON. B reads them through one adapter (`tools/i6n_swarm/inputs.py`) so a
  changed field name is a one-file change.
- **D:** `collect` calls `ArchivePublisher.record_candidate(candidate,
  fold_receipts)` once per terminal candidate disposition.
