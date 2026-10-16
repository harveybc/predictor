# Long MG12m R0 Versus R2

**CLOSED: R0_RETAINED.** Campaign `i6dmo-319c029a316f`, executing source
`5d430951`. All 25 units completed, including 19 new long donors; 172/172
governed reports read back, 177 files verified per physical mirror. All
campaign timers and the outer continuation stopped automatically. No TEST,
trading or cadence contrast ran. Exact values and digests: `LONG_RESULT.json`.

## Measured Result

Selected RAW inputs: 19 features, 13 observations spanning 144 open-market
hours, including t, with actual elapsed offsets. Six direct physical-hour
heads, hourly issuance, static training and supported DEVELOPMENT-2024 rows.
Each horizon's own naive uses exactly its model's rows and original-price scale.

| Horizon | Naive for this horizon | R0 MAE | R2 MAE | R0 relative error | R2 relative error |
|---|---:|---:|---:|---:|---:|
| 18 h | 0.002415050 | 0.002806181 | 0.002895639 | +16.195553% | +19.899754% |
| 24 h | 0.002814782 | 0.003277624 | 0.003520163 | +16.443240% | +25.059879% |
| 36 h | 0.003476803 | 0.004134853 | 0.004318055 | +18.926873% | +24.196127% |
| 48 h | 0.004138977 | 0.005163759 | 0.005076711 | +24.759325% | +22.656183% |
| 60 h | 0.004426645 | 0.006089152 | 0.005804504 | +37.556806% | +31.126479% |
| 72 h | 0.004791474 | 0.006891529 | 0.006550347 | +43.829004% | +36.708384% |

Relative error is `(MAE_h - naive_h) / naive_h`, negative better. The frozen
equal-head mean is R0 **+26.285134%**, R2 **+26.607801%**. R2 improves
48/60/72 h versus R0 but worsens 18/24/36 h; neither arm beats its naive at
any long horizon. R0 is retained, with no eligible long head. This is a
one-seed recipe decision, not a rejection of pretraining in general.

R0 completed 22 epochs/restored epoch 12 in 74.62 fit seconds; R2 completed
32/restored 22 in 106.07 seconds. Host cgroup peaks: 2.41/2.44 GiB. New long
donors used the reviewed mean FIT/purged-INNER monitor, unlike the retained
short INNER-only donors. Reconstruction and probe scores did not select the
winner. The R0 per-head errors reproduce the retained RAW long recipe.

## Sequence Outcome

Both short and long retained R0. The short recipe has eligible 4/6/12 h
heads, but zero long eligible heads prevents a trading pair. The approved
static sequence is finished: 50/50 units and 288/288 report readbacks across
the two campaigns, no failed units. Source publication clocks, calendar/cost
assumptions and development-selection limitations remain; TEST stays closed.
The next planned contrast is quarterly versus static with a fixed recipe,
not a silent repeat or automatic promotion of this negative result.
