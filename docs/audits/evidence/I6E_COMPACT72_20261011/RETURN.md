# I6-E compact72: verified neural closure and statistical continuation

The compact R0 neural contrast completed 156/156 cells (52 weekly refits for
each of 3/5/9 origin-anchored tokens), seed 6102026, 34,921 parameters each,
and 3,712 common 72-hour origins per arm. Original-price reconciliation
passed for every target. Price MAE in USD/EUR: 0.0045251248242256644,
0.004701839555138262 and 0.004598672941583923 versus paired persistence
0.004403521012931035. No arm passes the strategy gate; TEST remains closed.

The live warehouse accepted 156 reports / 2,184 metrics and exact readback
verified identity, value, unit, horizon and split. Physical report-bundle
SHA-256: `03e3edbe5ad78b024e03537c3428891ef1635316c1de11081c196c44d970902b`.
Original-price source SHA-256:
`c746f344ac743768e8c7f20e2e4ef3b7fb8222d96b3660f196c0a1d74039f327`.
Neural-closure SHA-256:
`ddf5cbdc49f5388ab60459a611a1e3b44b52c7a52ba4bd6dabf0e010cc86faf2`.
Local source lineage is not promoted to governed delivery by publication.

## Traceability

| Requirement | Test / artifact |
| --- | --- |
| Exact source clock and unchanged target | `test_i6e_price_metrics.py`, `PRICE_CLOSURE.json` |
| Metrics in price and return units separately | `test_i6e_publish_olap.py`, `olap/REPORTS.json` |
| Full identity/value warehouse readback | `test_i6e_publish_olap.py`, `olap/PUBLICATION.json` |
| Physical phase routing and calendar gaps | `test_i6e_statistical.py` |
| No future conditioning or future regressors | `test_i6e_statistical.py` perturbations |
| Fixed weekly parameters, no hourly refit | `test_i6e_statistical.py`, statistical phase records |
| Registered population cannot shrink itself | `test_i6e_statistical.py` design mutation |

The statistical continuation has separate identity and artifacts. It is not
complete merely because neural metrics are published. The two worker models
are ARIMA/ARIMAX (1,1,0), no drift, delayed exogenous values, statsmodels 0.14.5,
one TRAIN-only cost pilot followed by 156 weekly-resolution cells per model.
The controller collects artifacts and closes/publishes only complete evidence.
Nonconverged/interrupted cells are retained; no naive replacement or automatic
repeat confers a successful terminal. No real trading or TEST is launched.
