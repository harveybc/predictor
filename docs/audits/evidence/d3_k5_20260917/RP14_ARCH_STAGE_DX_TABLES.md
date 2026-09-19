# ARCH comparison — tables (validation; population 4 cells, verified 4, closure TOTAL)

## Receiver adequacy (H2 profiles at the top level, r = 1)
| arch | cell | model MASE | naive | linear | oracle | beats naive |
|---|---|---|---|---|---|---|

## Effects per architecture (replicate = unit; SD with n replicates; descriptive)
| arch | interpretable | e(h) | slope | d_0 | d_1 | gamma | rho_1 (last − pooled) | donor delta | n rep |
|---|---|---|---|---|---|---|---|---|---|
| A | False |  | None | None | None | None | None | None | 0 |
| B | False |  | None | None | None | None | None | None | 0 |
| C | False |  | None | None | None | None | None | None | 0 |
| 0 | False |  | None | None | None | None | None | None | 0 |

## Cells (raw error and MASE, denominator, control delta, support, updates, stop, cost)
| cell | arch | arm | status | MAE val | MASE val | naive | linear | oracle | MASE test | denom mean | reach | updates | stop | cpu s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DX__trend_event__s1__0__profiles | 0 | profiles | VERIFIED | 0.5392 | 0.4154 | 0.4999 | 0.4124 | 0.3592 | 0.4101 | 1.3114 | 3 | 1100 | UPDATE_BUDGET | 10.5 |
| DX__trend_event__s1__A__profiles | A | profiles | VERIFIED | 0.5219 | 0.4038 | 0.4999 | 0.4124 | 0.3592 | 0.3972 | 1.3114 | 7 | 1100 | UPDATE_BUDGET | 32.1 |
| DX__trend_event__s1__B__profiles | B | profiles | VERIFIED | 0.5873 | 0.4510 | 0.4999 | 0.4124 | 0.3592 | 0.4313 | 1.3114 | 48 | 1100 | UPDATE_BUDGET | 46.0 |
| DX__trend_event__s1__C__profiles | C | profiles | VERIFIED | 0.5350 | 0.4131 | 0.4999 | 0.4124 | 0.3592 | 0.4042 | 1.3114 | 48 | 726 | EARLY_STOPPING | 50.6 |

## Costs per architecture
| arch | cells | cpu total s | cpu mean s | fit mean s | updates mean | stop reasons |
|---|---|---|---|---|---|---|
| A | 1 | 32 | 32.1 | 21.7 | 1100 | {'UPDATE_BUDGET': 1} |
| B | 1 | 46 | 46.0 | 34.8 | 1100 | {'UPDATE_BUDGET': 1} |
| C | 1 | 51 | 50.6 | 41.3 | 726 | {'EARLY_STOPPING': 1} |
| 0 | 1 | 11 | 10.5 | 5.9 | 1100 | {'UPDATE_BUDGET': 1} |

## Diagnostic trend/event (adequacy only)
| arch | cell | model | naive | linear | oracle | beats naive | within linear + 0.03 |
|---|---|---|---|---|---|---|---|
| A | DX__trend_event__s1__A__profiles | 0.4038 | 0.4999 | 0.4124 | 0.3592 | True | True |
| B | DX__trend_event__s1__B__profiles | 0.4510 | 0.4999 | 0.4124 | 0.3592 | True | False |
| C | DX__trend_event__s1__C__profiles | 0.4131 | 0.4999 | 0.4124 | 0.3592 | True | True |
| 0 | DX__trend_event__s1__0__profiles | 0.4154 | 0.4999 | 0.4124 | 0.3592 | True | True |

Bootstrap: percentile intervals from resampling 2 replicates with replacement: descriptive precision, not confirmation
- A: H2_slope —, H3_gamma —, H3_d1 —
- B: H2_slope —, H3_gamma —, H3_d1 —
- C: H2_slope —, H3_gamma —, H3_d1 —
- 0: H2_slope —, H3_gamma —, H3_d1 —
