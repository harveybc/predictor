# Lane C: weekly campaign builder, execution and status

Owned code: `tools/i6n_weekly_campaign.py`, `tools/i6n_weekly_rollup.py`,
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

# Tests
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q \
  tests/test_i6n_weekly_campaign.py tests/test_i6n_weekly_rollup.py \
  tests/test_i6n_weekly_history_extension.py tests/test_i6n_weekly_data.py
```

The private config (`i6n.weekly_campaign.config.v1`) pins the nominal
calendar, the successor input, `CLOCK.json`, `CLOCK_SCOPE.json` and a cell
template by SHA-256; a changed byte refuses the plan. Plans are immutable:
write a new `plan-vN` instead of editing one.

## Status

- `plan_sha256 dd34986a...`: VERIFIED. 53/53 weekly units READY; 633
  (week, h) dispositions per arm, 1,266 for two arms (ceiling 1,272).
- 2023Q4 READY. 2023Q1-Q3 WAIT_DEPENDENCY: retained successor rows start
  2019-12-01 and the clock scope certificate starts 2019-12-01. The 2019
  rows reconcile exactly with the local annual archive, but the planned
  independent clock case 2019-10-28 is INCONCLUSIVE (HTTP 429), so no
  certificate may be extended. See `C_RETURN.md`.
- Calendar `FX_NY_17_ASSUMED_NOT_BROKER` / `i6n-v1`, nominal, not certified.
