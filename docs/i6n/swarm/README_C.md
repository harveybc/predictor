# Lane C: weekly campaign builder, execution and status

Owned code: `tools/i6n_weekly_campaign.py`, `tools/i6n_weekly_rollup.py`,
`tools/i6n_weekly_multi_output.py`,
`tools/i6n_weekly_history_extension.py` and their tests. None of them trains,
reads prices for the plan, opens TEST rows, downloads, or starts a service.

## What it produces

| Artifact | Where | In git |
| --- | --- | --- |
| Sealed plan: 53 weekly refit units (2024) + four 2023 inner folds, support per unit x role x head offset x split | `<campaign root>/plan-v2/PLAN.json` | no (5 MB, private) |
| B views `folds[h]`, `weeks` with cell-config digests | `VIEWS.private.json` / `VIEWS.redacted.json` | redacted copy in `evidence_C/` |
| One `i6n.historical_cell.v1` config per READY (unit, h) | `cells/<unit>_h<h>.json` (645) | no |
| Redacted unit manifest (fit role, business heads) | `evidence_C/WEEKLY_UNITS.redacted.json` | yes |
| Real counts per unit/role/h/split | `evidence_C/SUPPORT_BUSINESS.csv` | yes |
| Observed bars per calendar week | `evidence_C/WEEK_COVERAGE.csv` | yes |
| 2019 extension feasibility | `evidence_C/HISTORY_EXTENSION_2019.json` | yes |
| Declared-assumption extended successor (INPUT.npz, CLOCK*.json) | `<campaign root>/successor-dca-v1/` | no; record in `evidence_C/DECLARED_CLOCK_ASSUMPTION_2019.json` |

## Commands

Run from the checkout root, CPU only, one job at a time.

```bash
# Plan (output dir must not exist); reads only the ts member of INPUT.npz
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 300s -- \
  python -m tools.i6n_weekly_campaign plan \
    --config <campaign root>/CONFIG.private.json --output <campaign root>/plan-vN
python -m tools.i6n_weekly_campaign verify --output <campaign root>/plan-vN   # exit 0 = VERIFIED
python -m tools.i6n_weekly_campaign export --output <campaign root>/plan-vN \
  --to docs/i6n/swarm/evidence_C

# 2019 extension feasibility (local files only; exit 4 = WAIT_DEPENDENCY)
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 180s -- \
  python -m tools.i6n_weekly_history_extension \
    --source <statistical72 INPUT.npz> --source-sha256 <sha> \
    --archives <annual verified archive dir> --clock-audit <v2-final REPORT.json> \
    --design <statistical72 DESIGN.json> --successor-input <successor INPUT.npz> \
    --successor-sha256 <sha> --start 2019-03-12T21:00:00+00:00 --output <new dir>

# Declared-assumption extended successor (local files only; new output dir)
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 300s -- \
  python -m tools.i6n_weekly_history_extension materialize \
    --source <statistical72 INPUT.npz> --source-sha256 <sha> \
    --archives <annual verified archive dir> --clock-audit <latest v2 REPORT.json> \
    --attempt-report <each 2019-10-28 attempt REPORT.json> ... \
    --design <statistical72 DESIGN.json> --successor <base successor dir> \
    --start 2019-03-11T00:00:00+00:00 --output <new dir>

# Tests
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q \
  tests/test_i6n_weekly_campaign.py
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q \
  tests/test_i6n_weekly_rollup.py tests/test_i6n_weekly_history_extension.py \
  tests/test_i6n_weekly_data.py tests/test_i6n_external_clock_audit*.py
```

The private config (`i6n.weekly_campaign.config.v1`) pins the nominal
calendar, the successor input, `CLOCK.json`, `CLOCK_SCOPE.json` and a cell
template by SHA-256; a changed byte refuses the plan. Plans are immutable:
write a new `plan-vN` instead of editing one.

## Status

- Current plan `plan-v4`, `plan_sha256 40d3f5a0...`: VERIFIED. It supersedes
  `plan-v3` (`be5750da...`); supports are unchanged, and only the 3 DCA fold
  split ids carry the assumption sha (E-C3).
- Multi-output template: `evidence_C/MULTI_OUTPUT_ARMS.template.json`, 212 arms,
  WAIT_DEPENDENCY until finalists exist (`python -m tools.i6n_weekly_multi_output
  build|resolve`).
- 53/53 weekly units READY: 633 (week, h) dispositions per arm, 1,266 for
  two arms (ceiling 1,272).
- 2023Q4 READY on the certified primary source. 2023Q1_DCA, 2023Q2_DCA and
  2023Q3_DCA are READY_WITH_DECLARED_CLOCK_ASSUMPTION on the extended
  successor; no fold waits.
- 681 cell configs, all accepted by `i6n_historical_cell.authenticated`.
- Calendar `FX_NY_17_ASSUMED_NOT_BROKER` / `i6n-v1`, nominal, not certified.
