# Causal Kalman operator in DOIN's hierarchical space (design, DEVELOPMENT)

Owner: H1 (front H). Hand-off target: M04 (agent a083979bc9a8fa1b8). Status: DESIGN plus an executable declaration and
tests; nothing in this document is a measured effect. The pilot measurements it relies on are in `RESULTS_ETH.json`,
`RESULTS_EURUSD.json` and the summary table in `RETURN_H1.md` of this folder.

## What is declared

`tools/h_kalman_search_space.py` (tests: `tests/test_h_kalman_search_space.py`) declares six flat parameters. Every
parameter other than `kalman.active` exists only when the operator is active, and the two ratios exist only under the
declared-ratio source:

| flat name | grid | present when |
|---|---|---|
| `kalman.active` | {0, 1} | always |
| `kalman.group` | {level, slope, both} | active |
| `kalman.param_source` | {moments_train, declared_ratio} | active |
| `kalman.ratio_level` | {1e-4, 1e-3, 1e-2, 1e-1, 1.0} | active and declared_ratio |
| `kalman.ratio_slope` | {1e-6, 1e-4, 1e-2} | active, declared_ratio, group in {slope, both} |
| `kalman.mode` | {append, replace} | active (append = arm B, replace = arm C) |

The Cartesian product is 360 points. The conditional space is 77 points: 1 inactive plus, over group x source x mode,
6 (level) + 16 (slope) + 16 (both) per mode. The validation function refuses an inactive parameter that is present, an
active parameter that is missing, a value outside the grid and a value of the wrong type, before any fit, exactly like the
`train.huber_delta` rule of `tools/modular_search_space.py`.

## What DOIN does not decide

* Feature selection. The groups are the lane B declared groups (`KALMAN_CANDIDATE_FEATURES.v1.json`, rule written
  before measuring: lag-1 autocorrelation of the level and of its difference on TRAIN). `level` is the LOCAL_LEVEL group,
  `slope` the LEVEL_PLUS_SLOPE group, the two binary `ema_cross_*` flags are excluded because they are not Gaussian states.
* The fit role. Q/R are estimated only on TRAIN (closed form) or fixed by the declared ratios; the plan carries
  `fit_role: TRAIN`.
* The smoother. The backward smoother is not a grid value and cannot be made one: `validate_spec` refuses its kind.

## Phases

Phase 1 (best bounded variant first): the space is {inactive, PROMOTED_VARIANT}. PROMOTED_VARIANT is a single point
chosen from the lane H pilot by validation utility and stability (see `RETURN_H1.md` for the point and the numbers behind
it; if the pilot shows no utility the recommendation there is to not promote and keep the operator out of the search).
It is measured inside the differentiated model against the same model without it, paired seeds, same rows.
Phase 2 opens the 77-point conditional space only if the phase 1 effect exceeds the seed spread and the measured cost is
acceptable. Per-feature or per-group widening comes after, only when effect and cost justify it.

## How the outputs enter the modular engine (M01's contract)

Each Kalman output is a named column computed causally at every bar and carried in the window: `<f>__kf_level`,
`<f>__kf_slope`, `<f>__kf_innov`, `<f>__kf_zinnov`, `<f>__kf_logvar` (`state_var` enters as its log). The model input is
the `(B, 24, F)` tensor; the raw feature stays as its own branch so the Kalman arm is an addition that can be ablated
(`append`) or replaces only the declared features (`replace`). `logvar` is a model covariance under the declared
Gaussian model, not a probability of being right. Innovation channels are already centred: do not apply `window_mean`
input normalisation to them.

## Cost the optimizer must budget (measured, ETH 4h variant A, 15,895 rows, CPU, one thread)

See `RETURN_H1.md`: fit plus transform of the 29 declared features is under one CPU second; the per-tick latency of the
incremental path is hundreds of microseconds for a full row including artifact verification on every call. The
operator needs no GPU.
