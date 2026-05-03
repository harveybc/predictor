# Project 3 ETHUSDT 4h SAC Tech-Stat Export

This folder contains a local export from Project 3 Stage A for external strategy experiments.

## Files

- `ethusdt_4h_tech_stat_full_model_ready.csv`
  - Main monolithic CSV.
  - Full Project 3 ETHUSDT 4h model-ready history after rolling-feature warmup.
  - Rows: 18,085.
  - Date range: 2017-09-28 04:00:00 through 2025-12-31 20:00:00.
  - Columns: `DATE_TIME`, `typical_price`, OHLCV, and all `tech_stat` input features used by the Stage A SAC run.

- `ethusdt_4h_tech_stat_full_with_warmup_nans.csv`
  - Same columns, but includes the initial warmup rows with NaNs from rolling features.
  - Rows: 18,337.
  - Date range: 2017-08-17 04:00:00 through 2025-12-31 20:00:00.

- `ethusdt_4h_sac_tech_stat_full_config.json`
  - Full config bundle for the best preliminary Stage A run.
  - Includes `config.json`, `config_out.json`, `summary.json`, and the Stage A input metadata.

- `ethusdt_4h_tech_stat_export_metadata.json`
  - Export provenance, feature list, row counts, date ranges, and key SAC parameters.

## Notes

ETHUSDT does not have 20 years of Binance spot 4h data in Project 3. The export is full available Project 3 history through 2025-12-31.

The source Stage A run was:

`ethusdt_4h_sac_tech_stat_direct_atr_sltp_s0_20260502T051413Z_project3_stage31_firstwave`

Stage A result:

- Total return: 0.1512164417774693
- Sharpe ratio: 0.011535593140210989
- Max drawdown: 11.113226854446845%
- Trades: 426

This is a promising screening signal, not a validated production strategy.
