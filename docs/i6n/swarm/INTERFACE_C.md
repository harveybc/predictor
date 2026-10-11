# Lane C interface (C.1)

Machine-readable contract: `INTERFACE_C.json`. Summary for A, B, D and E:

- **Builder:** `python -m tools.i6n_weekly_campaign plan --config <private.json> --output <new dir>`
  reads only digest-pinned calendar/clock/scope/template files and the `ts`
  member of the retained input; writes a sealed `PLAN.json` (`plan_sha256`,
  per-unit `unit_sha256`), `VIEWS.*.json`, `SUPPORT.csv` and one
  `historical_cell.v1` config per READY (unit, h). `verify` rechecks digests;
  `export --to <dir>` writes the git-safe redacted manifest and counts.
- **Units:** `W<week_id>` weekly refits (53 for 2024) and `2023Q1..2023Q4`
  inner folds. Each has cutoff, refit window, release, TRAIN/INNER(/VALIDATION)
  split bounds, data state with exact missing objects, per role x head offset
  x split support (n, exclusions, `origins_sha256`) and per-h state
  `READY | NO_ELIGIBLE_ROWS | WAIT_DEPENDENCY`.
- **For B:** `folds[h]` and `weeks` views follow B.1's assumed shape with
  explicit differences: `binding` is null (B runs A's `freeze` on C's
  `cell_config` with `--fold <fold_id>`), weeks carry one cell config per h,
  multi-output arms are not C's. Role vocabulary differs from A rows
  (`MANAGEMENT` vs `management_origin`).
- **Annual:** `tools.i6n_weekly_rollup.AnnualAccumulator` pools VALIDATION
  rows by sums/counts against the frozen weekly support; `compare_annual`
  enforces identical naive; `assert_comparable` refuses inner-fold vs annual
  and cross-horizon comparisons.
- **Dependencies:** A.1 does not score the outer week (VALIDATION); 2023Q1-Q3
  wait on a clock scope certificate (see `C_RETURN.md`).
