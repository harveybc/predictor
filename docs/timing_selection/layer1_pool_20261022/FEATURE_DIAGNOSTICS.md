# Layer-1 pool: per-feature diagnostics (TRAIN 2020-2022) and preprocessing decisions

Features 197 ({'cross_asset': 75, 'price_pool': 42, 'inventory': 22, 'calendar': 13, 'fractal': 12, 'candle': 6, 'event': 6, 'raw_price': 4, 'ssa': 3, 'activity': 1}); decisions {'NONE': 171, 'DIFF': 7, 'ASINH': 6}; candle/SSA future-perturbation passed = True.

| feature | group | ADF p | KPSS p | skew | kurt | decision | reason |
|---|---|---|---|---|---|---|---|
| OPEN | raw_price | 0.763 | 0.010 | -0.508 | -0.744 | DIFF | non-stationary level (ADF p=0.763, KPSS p=0.010) |
| HIGH | raw_price | 0.755 | 0.010 | -0.507 | -0.745 | DIFF | non-stationary level (ADF p=0.755, KPSS p=0.010) |
| LOW | raw_price | 0.765 | 0.010 | -0.509 | -0.744 | DIFF | non-stationary level (ADF p=0.765, KPSS p=0.010) |
| CLOSE | raw_price | 0.764 | 0.010 | -0.508 | -0.744 | DIFF | non-stationary level (ADF p=0.764, KPSS p=0.010) |
| RAW_BAR_LOG_RETURN | inventory | 0.000 | 0.100 | 0.136 | 14.943 | NONE | stationary, not strongly skewed |
| RAW_LOG_RANGE | inventory | 0.000 | 0.010 | 3.091 | 20.132 | NONE | skewed (skew 3.09) but asinh does not halve it |
| RAW_CLOSE_LOG_RETURN | inventory | 0.000 | 0.100 | 0.065 | 14.913 | NONE | stationary, not strongly skewed |
| RAW_ELAPSED_HOURS | inventory | 0.000 | 0.100 | 10.903 | 118.024 | NONE | skewed (skew 10.90) but asinh does not halve it |
| ATR | inventory | 0.000 | 0.010 | 1.584 | 4.267 | NONE | stationary, not strongly skewed |
| RSI | inventory | 0.000 | 0.026 | -0.025 | -0.297 | NONE | bounded or calendar encoding |
| MACD | inventory | 0.000 | 0.070 | 0.021 | 1.434 | NONE | stationary, not strongly skewed |
| MACD_Histogram | inventory | 0.000 | 0.100 | -0.060 | 2.516 | NONE | stationary, not strongly skewed |
| MACD_Signal | inventory | 0.000 | 0.060 | 0.019 | 1.282 | NONE | stationary, not strongly skewed |
| ADX | inventory | 0.000 | 0.100 | 0.879 | 0.418 | NONE | stationary, not strongly skewed |
| DI_plus | inventory | 0.000 | 0.100 | 0.421 | 0.052 | NONE | stationary, not strongly skewed |
| DI_minus | inventory | 0.000 | 0.053 | 0.518 | 0.253 | NONE | stationary, not strongly skewed |
| Stochastic_K | inventory | 0.000 | 0.033 | 0.025 | -1.291 | NONE | bounded or calendar encoding |
| Stochastic_D | inventory | 0.000 | 0.032 | 0.027 | -1.273 | NONE | bounded or calendar encoding |
| BB_Width | inventory | 0.000 | 0.010 | 1.925 | 6.001 | NONE | stationary, not strongly skewed |
| CCI | inventory | 0.000 | 0.100 | -0.003 | -0.065 | NONE | stationary, not strongly skewed |
| WilliamsR | inventory | 0.000 | 0.035 | 0.025 | -1.302 | NONE | bounded or calendar encoding |
| ROC | inventory | 0.000 | 0.100 | 0.094 | 4.060 | NONE | stationary, not strongly skewed |
| ATR_ratio | inventory | 0.000 | 0.010 | 0.703 | 0.594 | NONE | stationary, not strongly skewed |
| BB_position | inventory | 0.000 | 0.100 | 0.011 | -0.617 | NONE | bounded or calendar encoding |
| rolling_std_24 | inventory | 0.000 | 0.010 | 1.788 | 5.039 | NONE | stationary, not strongly skewed |
| price_minus_ema | inventory | 0.000 | 0.095 | 0.022 | 2.334 | NONE | stationary, not strongly skewed |
| hod_sin | calendar | 0.000 | 0.100 | -0.001 | -1.500 | NONE | bounded or calendar encoding |
| hod_cos | calendar | 0.000 | 0.100 | 0.002 | -1.500 | NONE | bounded or calendar encoding |
| dow_sin | calendar | 0.000 | 0.100 | -0.357 | -1.218 | NONE | bounded or calendar encoding |
| dow_cos | calendar | 0.000 | 0.100 | 0.159 | -1.646 | NONE | bounded or calendar encoding |
| px.logret_1 | price_pool | 0.000 | 0.100 | 0.065 | 14.913 | NONE | stationary, not strongly skewed |
| px.range_1 | price_pool | 0.000 | 0.010 | 3.091 | 20.132 | NONE | skewed (skew 3.09) but asinh does not halve it |
| px.close_loc_1 | price_pool | 0.000 | 0.010 | 0.006 | -1.273 | NONE | stationary, not strongly skewed |
| px.er_1 | price_pool | — | — | — | — | NONE | stationary, not strongly skewed |
| px.rv_1 | price_pool | 0.000 | 0.010 | 3.892 | 31.679 | NONE | skewed (skew 3.89) but asinh does not halve it |
| px.logret_2 | price_pool | 0.000 | 0.100 | 0.146 | 10.651 | NONE | stationary, not strongly skewed |
| px.range_2 | price_pool | 0.000 | 0.010 | 2.616 | 12.571 | NONE | skewed (skew 2.62) but asinh does not halve it |
| px.close_loc_2 | price_pool | 0.000 | 0.010 | 0.001 | -1.256 | NONE | stationary, not strongly skewed |
| px.er_2 | price_pool | 0.000 | 0.100 | -0.706 | -1.069 | NONE | stationary, not strongly skewed |
| px.rv_2 | price_pool | 0.000 | 0.010 | 3.278 | 21.693 | NONE | skewed (skew 3.28) but asinh does not halve it |
| px.logret_4 | price_pool | 0.000 | 0.100 | 0.218 | 7.505 | NONE | stationary, not strongly skewed |
| px.range_4 | price_pool | 0.000 | 0.010 | 2.288 | 8.803 | NONE | skewed (skew 2.29) but asinh does not halve it |
| px.close_loc_4 | price_pool | 0.000 | 0.010 | 0.006 | -1.274 | NONE | stationary, not strongly skewed |
| px.er_4 | price_pool | 0.000 | 0.100 | 0.158 | -1.208 | NONE | stationary, not strongly skewed |
| px.rv_4 | price_pool | 0.000 | 0.010 | 2.733 | 14.210 | NONE | skewed (skew 2.73) but asinh does not halve it |
| px.logret_6 | price_pool | 0.000 | 0.100 | 0.186 | 5.491 | NONE | stationary, not strongly skewed |
| px.range_6 | price_pool | 0.000 | 0.010 | 2.105 | 7.095 | NONE | skewed (skew 2.10) but asinh does not halve it |
| px.close_loc_6 | price_pool | 0.000 | 0.010 | 0.015 | -1.290 | NONE | stationary, not strongly skewed |
| px.er_6 | price_pool | 0.000 | 0.100 | 0.396 | -0.855 | NONE | stationary, not strongly skewed |
| px.rv_6 | price_pool | 0.000 | 0.010 | 2.438 | 10.890 | NONE | skewed (skew 2.44) but asinh does not halve it |
| px.logret_12 | price_pool | 0.000 | 0.100 | 0.049 | 3.405 | NONE | stationary, not strongly skewed |
| px.range_12 | price_pool | 0.000 | 0.010 | 1.826 | 5.110 | NONE | stationary, not strongly skewed |
| px.close_loc_12 | price_pool | 0.000 | 0.024 | 0.016 | -1.303 | NONE | stationary, not strongly skewed |
| px.er_12 | price_pool | 0.000 | 0.100 | 0.633 | -0.297 | NONE | stationary, not strongly skewed |
| px.rv_12 | price_pool | 0.000 | 0.010 | 2.028 | 6.955 | NONE | skewed (skew 2.03) but asinh does not halve it |
| px.logret_24 | price_pool | 0.000 | 0.100 | 0.077 | 1.845 | NONE | stationary, not strongly skewed |
| px.range_24 | price_pool | 0.000 | 0.010 | 1.595 | 3.819 | NONE | stationary, not strongly skewed |
| px.close_loc_24 | price_pool | 0.000 | 0.072 | 0.047 | -1.301 | NONE | stationary, not strongly skewed |
| px.er_24 | price_pool | 0.000 | 0.100 | 0.747 | -0.011 | NONE | stationary, not strongly skewed |
| px.rv_24 | price_pool | 0.000 | 0.010 | 1.759 | 4.710 | NONE | stationary, not strongly skewed |
| px.logret_48 | price_pool | 0.000 | 0.052 | 0.006 | 1.480 | NONE | stationary, not strongly skewed |
| px.range_48 | price_pool | 0.000 | 0.010 | 1.532 | 3.432 | NONE | stationary, not strongly skewed |
| px.close_loc_48 | price_pool | 0.000 | 0.025 | 0.048 | -1.330 | NONE | stationary, not strongly skewed |
| px.er_48 | price_pool | 0.000 | 0.100 | 0.746 | 0.071 | NONE | stationary, not strongly skewed |
| px.rv_48 | price_pool | 0.000 | 0.010 | 1.447 | 2.624 | NONE | stationary, not strongly skewed |
| px.atr_14 | price_pool | 0.000 | 0.010 | 1.563 | 4.070 | NONE | stationary, not strongly skewed |
| px.atr_ratio_14_120 | price_pool | 0.000 | 0.100 | 0.653 | 0.667 | NONE | stationary, not strongly skewed |
| px.range_ratio_1_24 | price_pool | 0.000 | 0.100 | 1.690 | 4.176 | NONE | stationary, not strongly skewed |
| px.bar_body | price_pool | 0.000 | 0.100 | 0.136 | 14.943 | NONE | stationary, not strongly skewed |
| cal.utc_hod_sin | calendar | 0.000 | 0.100 | -0.001 | -1.500 | NONE | bounded or calendar encoding |
| cal.utc_hod_cos | calendar | 0.000 | 0.100 | 0.002 | -1.500 | NONE | bounded or calendar encoding |
| cal.utc_dow_sin | calendar | 0.000 | 0.100 | -0.384 | -1.162 | NONE | bounded or calendar encoding |
| cal.utc_dow_cos | calendar | 0.000 | 0.100 | 0.127 | -1.653 | NONE | bounded or calendar encoding |
| cal.utc_session_asia | calendar | 0.000 | 0.100 | 0.917 | -1.159 | NONE | bounded or calendar encoding |
| cal.utc_session_europe | calendar | 0.000 | 0.100 | 1.153 | -0.671 | NONE | bounded or calendar encoding |
| cal.utc_session_us | calendar | 0.000 | 0.100 | 0.707 | -1.500 | NONE | bounded or calendar encoding |
| cal.london_hod_sin | calendar | 0.000 | 0.100 | -0.001 | -1.500 | NONE | bounded or calendar encoding |
| cal.london_hod_cos | calendar | 0.000 | 0.100 | 0.002 | -1.500 | NONE | bounded or calendar encoding |
| px.elapsed_hours | price_pool | 0.000 | 0.100 | 10.903 | 118.024 | NONE | skewed (skew 10.90) but asinh does not halve it |
| px.hours_since_session_open | price_pool | 0.000 | 0.100 | 0.008 | -1.198 | NONE | stationary, not strongly skewed |
| px.unexpected_gap_hours | price_pool | — | — | — | — | NONE | stationary, not strongly skewed |
| hurst_rs_w48 | fractal | 0.000 | 0.100 | -0.065 | -0.310 | NONE | stationary, not strongly skewed |
| variogram_hurst_w48 | fractal | 0.000 | 0.010 | -0.095 | -0.668 | NONE | stationary, not strongly skewed |
| hurst_spectrum_width_w48 | fractal | 0.000 | 0.010 | 0.712 | 1.497 | NONE | stationary, not strongly skewed |
| box_counting_dimension_w48 | fractal | 0.000 | 0.010 | 0.658 | -0.112 | NONE | stationary, not strongly skewed |
| hurst_rs_w96 | fractal | 0.000 | 0.067 | -0.037 | -0.168 | NONE | stationary, not strongly skewed |
| variogram_hurst_w96 | fractal | 0.000 | 0.010 | -0.414 | -0.132 | NONE | stationary, not strongly skewed |
| hurst_spectrum_width_w96 | fractal | 0.000 | 0.010 | 0.726 | 0.960 | NONE | stationary, not strongly skewed |
| box_counting_dimension_w96 | fractal | 0.000 | 0.013 | 0.579 | 0.294 | NONE | stationary, not strongly skewed |
| hurst_rs_w200 | fractal | 0.000 | 0.010 | -0.155 | -0.153 | NONE | stationary, not strongly skewed |
| variogram_hurst_w200 | fractal | 0.000 | 0.010 | -0.524 | 0.541 | NONE | stationary, not strongly skewed |
| hurst_spectrum_width_w200 | fractal | 0.000 | 0.010 | 0.653 | 0.597 | NONE | stationary, not strongly skewed |
| box_counting_dimension_w200 | fractal | 0.000 | 0.013 | -0.163 | 0.487 | NONE | stationary, not strongly skewed |
| cdl.body_pips | candle | 0.000 | 0.100 | 0.081 | 12.403 | NONE | stationary, not strongly skewed |
| cdl.upper_wick_pips | candle | 0.000 | 0.010 | 3.477 | 26.444 | ASINH | positive and skewed (skew 3.48 -> 0.12) |
| cdl.lower_wick_pips | candle | 0.000 | 0.010 | 2.800 | 13.990 | ASINH | positive and skewed (skew 2.80 -> 0.12) |
| cdl.range_pips | candle | 0.000 | 0.010 | 2.923 | 17.695 | ASINH | positive and skewed (skew 2.92 -> 0.14) |
| cdl.close_minus_low_pips | candle | 0.000 | 0.010 | 3.543 | 26.934 | ASINH | positive and skewed (skew 3.54 -> -0.15) |
| cdl.high_minus_close_pips | candle | 0.000 | 0.010 | 3.265 | 20.541 | ASINH | positive and skewed (skew 3.26 -> -0.11) |
| fx.gbpusd.ret_1 | cross_asset | 0.000 | 0.100 | -0.990 | 43.035 | NONE | stationary, not strongly skewed |
| fx.gbpusd.ret_4 | cross_asset | 0.000 | 0.100 | -0.372 | 13.519 | NONE | stationary, not strongly skewed |
| fx.gbpusd.ret_24 | cross_asset | 0.000 | 0.100 | -0.631 | 8.695 | NONE | stationary, not strongly skewed |
| fx.gbpusd.rv_24 | cross_asset | 0.000 | 0.010 | 4.026 | 30.297 | NONE | skewed (skew 4.03) but asinh does not halve it |
| fx.usdchf.ret_1 | cross_asset | 0.000 | 0.100 | -0.276 | 16.279 | NONE | stationary, not strongly skewed |
| fx.usdchf.ret_4 | cross_asset | 0.000 | 0.100 | -0.407 | 7.805 | NONE | stationary, not strongly skewed |
| fx.usdchf.ret_24 | cross_asset | 0.000 | 0.100 | -0.532 | 3.559 | NONE | stationary, not strongly skewed |
| fx.usdchf.rv_24 | cross_asset | 0.000 | 0.010 | 2.492 | 10.281 | NONE | skewed (skew 2.49) but asinh does not halve it |
| fx.usdjpy.ret_1 | cross_asset | 0.000 | 0.100 | -2.745 | 71.753 | NONE | stationary, not strongly skewed |
| fx.usdjpy.ret_4 | cross_asset | 0.000 | 0.100 | -1.739 | 28.361 | NONE | stationary, not strongly skewed |
| fx.usdjpy.ret_24 | cross_asset | 0.000 | 0.100 | -0.722 | 7.400 | NONE | stationary, not strongly skewed |
| fx.usdjpy.rv_24 | cross_asset | 0.000 | 0.010 | 3.919 | 23.781 | NONE | skewed (skew 3.92) but asinh does not halve it |
| fx.eurgbp.ret_1 | cross_asset | 0.000 | 0.100 | 1.061 | 30.538 | NONE | stationary, not strongly skewed |
| fx.eurgbp.ret_4 | cross_asset | 0.000 | 0.100 | 0.710 | 13.314 | NONE | stationary, not strongly skewed |
| fx.eurgbp.ret_24 | cross_asset | 0.000 | 0.100 | 0.765 | 7.778 | NONE | stationary, not strongly skewed |
| fx.eurgbp.rv_24 | cross_asset | 0.000 | 0.010 | 4.026 | 28.814 | NONE | skewed (skew 4.03) but asinh does not halve it |
| fx.eurjpy.ret_1 | cross_asset | 0.000 | 0.100 | -1.698 | 38.422 | NONE | stationary, not strongly skewed |
| fx.eurjpy.ret_4 | cross_asset | 0.000 | 0.100 | -0.914 | 13.855 | NONE | stationary, not strongly skewed |
| fx.eurjpy.ret_24 | cross_asset | 0.000 | 0.100 | -0.404 | 3.616 | NONE | stationary, not strongly skewed |
| fx.eurjpy.rv_24 | cross_asset | 0.000 | 0.010 | 3.147 | 18.781 | NONE | skewed (skew 3.15) but asinh does not halve it |
| fx.audusd.ret_1 | cross_asset | 0.000 | 0.100 | -0.144 | 23.605 | NONE | stationary, not strongly skewed |
| fx.audusd.ret_4 | cross_asset | 0.000 | 0.100 | -0.152 | 12.583 | NONE | stationary, not strongly skewed |
| fx.audusd.ret_24 | cross_asset | 0.000 | 0.100 | -0.325 | 5.316 | NONE | stationary, not strongly skewed |
| fx.audusd.rv_24 | cross_asset | 0.000 | 0.010 | 4.143 | 31.093 | NONE | skewed (skew 4.14) but asinh does not halve it |
| fx.usd_strength.ret_1 | cross_asset | 0.000 | 0.100 | -0.239 | 21.252 | NONE | stationary, not strongly skewed |
| fx.usd_strength.ret_4 | cross_asset | 0.000 | 0.100 | -0.394 | 13.366 | NONE | stationary, not strongly skewed |
| fx.usd_strength.ret_24 | cross_asset | 0.000 | 0.100 | 0.014 | 4.960 | NONE | stationary, not strongly skewed |
| fx.eur_strength.ret_1 | cross_asset | 0.000 | 0.100 | -0.491 | 15.196 | NONE | stationary, not strongly skewed |
| fx.eur_strength.ret_4 | cross_asset | 0.000 | 0.100 | -0.109 | 7.131 | NONE | stationary, not strongly skewed |
| fx.eur_strength.ret_24 | cross_asset | 0.000 | 0.100 | 0.132 | 2.322 | NONE | stationary, not strongly skewed |
| btc.ret_1 | cross_asset | 0.000 | 0.024 | -1.060 | 49.917 | NONE | stationary, not strongly skewed |
| btc.ret_4 | cross_asset | 0.000 | 0.020 | -0.910 | 21.251 | NONE | stationary, not strongly skewed |
| btc.ret_24 | cross_asset | 0.000 | 0.010 | -0.923 | 9.559 | NONE | stationary, not strongly skewed |
| btc.rv_24 | cross_asset | 0.000 | 0.010 | 4.059 | 32.938 | NONE | skewed (skew 4.06) but asinh does not halve it |
| btc.taker_buy_share_24 | cross_asset | 0.000 | 0.010 | -0.515 | 0.529 | NONE | stationary, not strongly skewed |
| dly.us2y.level | cross_asset | 0.998 | 0.010 | 1.119 | -0.196 | DIFF | non-stationary level (ADF p=0.998, KPSS p=0.010) |
| dly.us2y.chg_1d | cross_asset | 0.000 | 0.010 | -0.020 | 7.124 | NONE | stationary, not strongly skewed |
| dly.us2y.chg_5d | cross_asset | 0.000 | 0.010 | 0.215 | 5.515 | NONE | stationary, not strongly skewed |
| dly.us10y.level | cross_asset | 0.983 | 0.010 | 0.846 | -0.360 | DIFF | non-stationary level (ADF p=0.983, KPSS p=0.010) |
| dly.us10y.chg_1d | cross_asset | 0.000 | 0.010 | -0.362 | 3.087 | NONE | stationary, not strongly skewed |
| dly.us10y.chg_5d | cross_asset | 0.000 | 0.010 | -0.196 | 1.611 | NONE | stationary, not strongly skewed |
| dly.us10y2y.level | cross_asset | 0.942 | 0.010 | -0.320 | -0.538 | DIFF | non-stationary level (ADF p=0.942, KPSS p=0.010) |
| dly.us10y2y.chg_1d | cross_asset | 0.000 | 0.010 | -0.079 | 0.906 | NONE | stationary, not strongly skewed |
| dly.us10y2y.chg_5d | cross_asset | 0.000 | 0.010 | 0.219 | 0.594 | NONE | stationary, not strongly skewed |
| dly.vix.level | cross_asset | 0.014 | 0.010 | 2.521 | 10.395 | ASINH | positive and skewed (skew 2.52 -> 0.75) |
| dly.vix.chg_1d | cross_asset | 0.000 | 0.100 | 2.313 | 23.110 | NONE | stationary, not strongly skewed |
| dly.spx.ret_1d | cross_asset | 0.000 | 0.100 | -0.756 | 11.078 | NONE | stationary, not strongly skewed |
| dly.spx.ret_5d | cross_asset | 0.000 | 0.010 | -0.961 | 5.546 | NONE | stationary, not strongly skewed |
| dly.dax.ret_1d | cross_asset | 0.000 | 0.100 | -0.646 | 10.923 | NONE | stationary, not strongly skewed |
| dly.dax.ret_5d | cross_asset | 0.000 | 0.096 | -1.403 | 9.834 | NONE | stationary, not strongly skewed |
| dly.gold.ret_1d | cross_asset | 0.000 | 0.100 | -0.208 | 4.032 | NONE | stationary, not strongly skewed |
| dly.gold.ret_5d | cross_asset | 0.000 | 0.011 | -0.157 | 2.579 | NONE | stationary, not strongly skewed |
| dly.oil.ret_1d | cross_asset | 0.000 | 0.100 | 0.015 | 18.416 | NONE | stationary, not strongly skewed |
| dly.oil.ret_5d | cross_asset | 0.000 | 0.026 | -0.405 | 14.058 | NONE | stationary, not strongly skewed |
| cot.eur.net_long | cross_asset | 0.000 | 0.010 | 1.220 | 0.665 | NONE | stationary, not strongly skewed |
| cot.eur.net_change | cross_asset | 0.000 | 0.100 | 0.040 | 2.836 | NONE | stationary, not strongly skewed |
| cot.eur.net_share | cross_asset | 0.000 | 0.010 | 0.131 | -0.140 | NONE | stationary, not strongly skewed |
| cot.eur.net_z156 | cross_asset | 0.000 | 0.010 | 1.194 | 0.827 | NONE | stationary, not strongly skewed |
| fac.fx_f1.ret_1 | cross_asset | 0.000 | 0.100 | -0.495 | 19.440 | NONE | stationary, not strongly skewed |
| fac.fx_f1.ret_4 | cross_asset | 0.000 | 0.100 | -0.255 | 7.640 | NONE | stationary, not strongly skewed |
| fac.fx_f1.ret_24 | cross_asset | 0.000 | 0.100 | -0.568 | 4.959 | NONE | stationary, not strongly skewed |
| fac.fx_f1.kalman_slope | cross_asset | 0.000 | 0.100 | -0.339 | 11.639 | NONE | stationary, not strongly skewed |
| fac.fx_f2.ret_1 | cross_asset | 0.000 | 0.100 | -1.721 | 43.106 | NONE | stationary, not strongly skewed |
| fac.fx_f2.ret_4 | cross_asset | 0.000 | 0.100 | -1.191 | 20.640 | NONE | stationary, not strongly skewed |
| fac.fx_f2.ret_24 | cross_asset | 0.000 | 0.100 | -0.577 | 5.777 | NONE | stationary, not strongly skewed |
| fac.fx_f2.kalman_slope | cross_asset | 0.000 | 0.100 | -1.622 | 30.032 | NONE | stationary, not strongly skewed |
| fac.fx_f3.ret_1 | cross_asset | 0.000 | 0.100 | -0.361 | 22.337 | NONE | stationary, not strongly skewed |
| fac.fx_f3.ret_4 | cross_asset | 0.000 | 0.100 | -0.062 | 9.780 | NONE | stationary, not strongly skewed |
| fac.fx_f3.ret_24 | cross_asset | 0.000 | 0.100 | 0.269 | 3.966 | NONE | stationary, not strongly skewed |
| fac.fx_f3.kalman_slope | cross_asset | 0.000 | 0.100 | -0.206 | 14.438 | NONE | stationary, not strongly skewed |
| fac.daily_f1 | cross_asset | 0.000 | 0.084 | -1.323 | 8.200 | NONE | stationary, not strongly skewed |
| fac.daily_f2 | cross_asset | 0.000 | 0.100 | 0.923 | 11.146 | NONE | stationary, not strongly skewed |
| kal.eurusd.slope_fast | cross_asset | 0.000 | 0.100 | 0.095 | 4.355 | NONE | stationary, not strongly skewed |
| kal.eurusd.slope_mid | cross_asset | 0.000 | 0.100 | 0.059 | 1.761 | NONE | stationary, not strongly skewed |
| kal.eurusd.slope_slow | cross_asset | 0.000 | 0.010 | -0.146 | 1.563 | NONE | stationary, not strongly skewed |
| ev.tokyo_fix.hours_since | event | 0.000 | 0.100 | 0.001 | -1.203 | NONE | bounded or calendar encoding |
| ev.tokyo_fix.hours_to | event | 0.000 | 0.100 | -0.001 | -1.203 | NONE | bounded or calendar encoding |
| ev.london_fix.hours_since | event | 0.000 | 0.100 | -0.001 | -1.204 | NONE | bounded or calendar encoding |
| ev.london_fix.hours_to | event | 0.000 | 0.100 | 0.001 | -1.204 | NONE | bounded or calendar encoding |
| ev.ny_cut.hours_since | event | 0.000 | 0.100 | -0.001 | -1.205 | NONE | bounded or calendar encoding |
| ev.ny_cut.hours_to | event | 0.000 | 0.100 | 0.001 | -1.205 | NONE | bounded or calendar encoding |
| act.p_hit | activity | 0.000 | 0.100 | 0.429 | -1.030 | NONE | stationary, not strongly skewed |
| ssa.trend_minus_close_pips | ssa | 0.000 | 0.100 | -0.141 | 3.074 | NONE | stationary, not strongly skewed |
| ssa.c2_pips | ssa | 0.000 | 0.100 | 0.249 | 5.086 | NONE | stationary, not strongly skewed |
| ssa.c3_pips | ssa | 0.000 | 0.100 | 0.107 | 4.412 | NONE | stationary, not strongly skewed |
