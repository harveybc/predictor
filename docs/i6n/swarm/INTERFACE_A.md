# Lane A interface (A.1, published early)

Machine-readable contract: `INTERFACE_A.json`. Summary for B, C, D and E:

- **Freeze per (horizon, fold):** `python -m tools.i6n_historical_evaluator freeze --config <cell.json> --output <new dir> --experiment <id> --fold <id>`.
  No fit. Seals the endpoint-h TRAIN/inner support of that fold and writes
  `BINDING.json`. The cell config must carry plan section 2 stopping
  (`max_epochs=50, patience=5, min_delta=1e-6, seed=6102026`).
- **Evaluate in a subprocess:** run `BINDING.i6n_evaluator_command` with one
  JSON request on stdin: `{config, generation, context, support[, attempt]}`.
  `config` is `tools.i6n_search.effective_config(genome)`; trajectory
  genomes (`output_mode=trajectory`, `delta_y`) are now trained, not rejected.
  `attempt` (optional) links a recovery attempt to the previous terminal
  status digest; a COMPLETED attempt of the same identity is always adopted.
- **Shared receipt:** unchanged keys for `tools.i6n_search._score`; fitness is
  endpoint-h only. `metadata.candidate_receipt` (schema
  `i6n.candidate_receipt.v1`) carries genotype/config digests, dx/dy/L/n,
  heads with offsets/status/TRAIN scales, per head x split rows of
  sums+counts and naive sums+counts (C2-scored), stopping, cost, attempt
  lineage and placement.
- **Row identity keys** (default shared keys): experiment,
  candidate_identity_sha256, fold, horizon_hours, head_offset_hours, role,
  population, split, metric, unit, schema_version. Only rows with
  `comparison=true` (endpoint) feed selection; intermediate rows are
  diagnostics with their own naive.
- Fold aggregation (mean over four 2023 folds) belongs to the caller (B/E);
  the receipt is per fold.
