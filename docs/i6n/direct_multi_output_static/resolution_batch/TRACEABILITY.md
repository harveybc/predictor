# Requirement To Evidence

| Requirement | Planned executable evidence |
|---|---|
| Four DMO fits before h72 sweep | test_resolution_batch plan and next-cell tests |
| DS4 13 ticks, current origin included | test_resolution_data grid/alignment |
| h72 6/9/13/18/24, spacing selected automatically | test_resolution_batch selection |
| Shared support and FIT-only scales | test_resolution_data perturbations/scales |
| Minimal Conv1D and linear controls | test_resolution_models structure/affinity |
| Business masks, raw targets, no TEST | reused audited data/market_grid + population assertions |
| Epoch stopping mean FIT/INNER | reused train.fit_dmo + recorded history/replay |
| One fit at a time, bounded resources | owned GPU timer, admission and flock |
| No repeated completed fits | authenticated terminal adoption tests |
| OLAP exact readback and two copies | reused outbox + hash-verified mirrors |
| Status and ETA without agent | generated STATUS and per-cell progress |

State: test design frozen before implementation. Acceptance is conferred only
by retained test output and deployment receipts, not by this table.
