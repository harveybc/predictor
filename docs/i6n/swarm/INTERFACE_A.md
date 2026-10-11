# Lane A interface (A.2, implemented at 298619e4)

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

## A.2 changes (implemented)

- **Role value change:** `rows.role` is C1's canonical `MANAGEMENT`/`ENTRY`;
  the C2 name moves to the additive `rows.c2_role`. A.1 rows were never
  produced, so no alias for old receipts exists or is needed.
- **Outer week:** `freeze --score-outer` (WEEKLY_REFIT units only) seals the
  outer week's endpoint S, count and naive before any fit. The receipt gains
  `split=VALIDATION` rows per head and `metadata.outer_metrics`; fitness,
  `error_sums` and early stopping stay TRAIN/inner. INNER_FOLD units must not
  pass `--score-outer`.
- **Additive:** `rows.support_sha256` (lane C origins digest).
- **Confirmed with E.1:** `effective_config_sha256` and
  `genotype_canonical_sha256` definitions are identical to lane E's.
- **Confirmed with C.1:** C's per-(unit, h) `historical_cell.v1` configs go
  straight to `freeze --config`; they must carry plan stopping (50/5/1e-6,
  seed 6102026) and an endpoint base model. See
  `cell_config_requirements_for_C` in the JSON.
