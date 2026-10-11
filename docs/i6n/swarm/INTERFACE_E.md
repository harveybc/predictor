# Lane E interface (E.3, plan v1.2.0)

Machine-readable: `INTERFACE_E.json`. Summary for A, B, C, D:

- **Genotype v2** (`tools.i6n_search_space`): exact plan domain; `canonicalize`
  raises `Infeasible` with every typed reason, never repairs; inactive genes
  (`transformer_blocks` for conv, `trajectory_k` for endpoint, `adapter_width`
  for none) and the endpoint alias (k=1 == endpoint) are canonicalized.
  Strides are the gene `reducer_strides` (free 1|2 per reducer, ruled
  2026-10-11); none/progressive/final/mixed is a derived label.
- **Single identity:** `genotype_canonical_sha256 = genotype_sha256(g)` equals
  `tools.i6n_search.effective_identity(to_evaluator_genome(g))`, so A's receipt
  field, B's ledger and the 48 legacy candidates share one hash.
  `tools/i6n_search.py` stays frozen.
- **A's request config:** `evaluator_config(g)`; `effective_config_sha256(g)` is
  the canonical-JSON sha256 of that config (A must emit this value).
- **B's fit_id:** `fit_id(g, {experiment, fold_id, context_sha256, support_sha256})`,
  B.1 formula verbatim. `search_candidate_sha256(g, plan_sha256)` likewise.
- **Plan JSON:** `docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json`, sealed
  (`plan_sha256 606a3555...a3c`, supersedes v1.0.0 `8d8865f1...` and v1.1.0 `e7de41f0...`), with domain, classes, promotion, folds, search, fitness,
  budgets, initial genotypes, finalists, rollout and prior mapping.
- **Shared plugin:** `optimizer_plugins.i6n_optimizer.Plugin` takes the v2 path
  when `opt_config["i6n_plan"]` is present (wire `i6n-shared-deap-v2`); the v1
  path is unchanged.
- **Folds:** the plan binds the four 2023 quarters; fold ids come from lane C
  (`2023Q1_DCA..2023Q3_DCA, 2023Q4` in C.2), one per quarter
  (`resolve_fold_ids`).
- **Fold fitness:** `score_folds(..., genotype_canonical_sha256=, horizon_hours=,
  origins_sha256=)` refuses (typed `RECEIPT_*`) any receipt not bound to the
  candidate, its fold, h and the frozen support (lane A review E-A1). B ranks
  only through it and passes C's origins digests.
