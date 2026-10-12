# Lane C interface (C.3)

Machine-readable contract: `INTERFACE_C.json`. Summary for A, B, D and E:

- **Builder:** `python -m tools.i6n_weekly_campaign plan --config <private.json> --output <new dir>`
  reads only digest-pinned calendar/clock/scope/template files and the `ts`
  member of the retained input; writes a sealed `PLAN.json` (`plan_sha256`,
  per-unit `unit_sha256`), `VIEWS.*.json`, `SUPPORT.csv` and one
  `historical_cell.v1` config per READY (unit, h). `verify` rechecks digests;
  `export --to <dir>` writes the git-safe redacted manifest and counts.
- **Units:** `W<week_id>` weekly refits (53 for 2024) and the four 2023
  inner folds `2023Q1_DCA`, `2023Q2_DCA`, `2023Q3_DCA` (declared clock
  assumption, extended source; `quarter` keeps `2023Qn`) and `2023Q4`. Each has cutoff, refit window, release, TRAIN/INNER(/VALIDATION)
  split bounds, data state with exact missing objects, per role x head offset
  x split support (n, exclusions, `origins_sha256`) and per-h state
  `READY | READY_WITH_DECLARED_CLOCK_ASSUMPTION | NO_ELIGIBLE_ROWS | WAIT_DEPENDENCY`.
  `clock_assumption` and `data_source` are part of `unit_sha256`; the `_DCA`
  fold id reaches every receipt row through A's `fold` key.
- **For B:** `folds[h]` and `weeks` views follow B.1's assumed shape with
  explicit differences: `binding` is null (B runs A's `freeze` on C's
  `cell_config` with `--fold <fold_id>`), weeks carry one cell config per h,
  multi-output arms are not C's. Weeks are scored on the outer week
  (`freeze --score-outer`, A.2); folds never. Canonical roles are C1's
  `ENTRY`/`MANAGEMENT`. Cell templates must carry plan stopping and seed.
- **Annual:** `tools.i6n_weekly_rollup.AnnualAccumulator` pools VALIDATION
  rows by sums/counts against the frozen weekly support; `compare_annual`
  enforces identical naive; `assert_comparable` refuses inner-fold vs annual
  and cross-horizon comparisons.
- **Assumption:** 2023Q1-Q3 use the extended successor under declared clock
  assumption `DCA_2019_EU_DST_POLICY_2019-03-11_2019-12-01` (see `C_RETURN.md`).
- **C.3:** the annual roll-up is keyed by arm (`arm_sha256` = receipt-header
  genotype, plus `arm_role`). Weekly candidate identities are kept per row, and
  the lowest attempt is canonical. The legacy keyword is refused, so B's
  `collect` must adapt. The multi-output arms manifest
  (`tools.i6n_weekly_multi_output`) is a template of 212 arms; B passes only
  READY arms after finalist resolution.
