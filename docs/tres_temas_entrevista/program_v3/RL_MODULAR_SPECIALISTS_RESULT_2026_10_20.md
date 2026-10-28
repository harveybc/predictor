# Controlled modular specialists: completed result

All 15 authorized cells completed at first attempt, from 20:31:43 to
21:00:58 UTC on 2026-10-20 (29m15s). Runtime f6ddd53c18c34a12f809498d124874ba82ee908f;
spec fb808aa85830deac27cc838fad9cc326ac3d4bb57790b51e782f583250b946a6.
160/160 reports delivered with exact warehouse readback; zero pending,
mismatches or integrity blocks. All 15 terminal directories independently
copied and hash verified. No fit remains active in this finite sequence.

Five-seed means. TRAIN=2020-2022; checkpoint and threshold selection=2023H1;
diagnostic=2023H2; VALIDATION/development=full 2024; TEST unused. Capital USD10000,
leverage50, one entry/day, positive assumed execution costs. This controlled
BC/DAgger protocol differs from the earlier Satoshi four-entry/day/leverage100
run and from native SAC. Original-three and new-two seed summaries are retained.

| Architecture | TRAIN return % | 2024 net USD | 2024 return % | Sharpe annualized daily | Max DD % | Trades | Win % | PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JOINT16 + abstention | -5.8006 | -674.3773 | -6.7438 | -1.8258 | 8.8997 | 65.8 | 37.9697 | .5353 |
| JOINT matched + abstention | -9.9233 | -677.7131 | -6.7771 | -1.2527 | 7.5100 | 98.6 | 33.1873 | .6396 |
| FULL combined | +23.1510 | -740.6002 | -7.4060 | -1.0480 | 11.8625 | 66.8 | 27.7548 | .6518 |

FULL loses in every 2024 seed: -855.6212/-1000.7185/-541.0928/-762.4922/
-543.0760 USD. Specialization helps fitted-period profit here, but does not
improve validation net profit or drawdown against JOINT16. This is not evidence
that every specialization scheme fails or that the inputs contain no signal.

FULL fixed-exit control with the same threshold: TRAIN +46.3307%, 2024 -5.5080%,
Sharpe -.2722, DD15.9889%, trades50.4, win52.5810%, PF.8859. Learned closing
reduces drawdown but worsens net result relative to this control. The
fixed-exit threshold separately calibrated on 2023H1 gives 2024 -8.2560%,
Sharpe -.3975, DD18.0459%, confirming that threshold selection matters too.
No control has positive mean 2024 profit; selecting the best seed would mislead.

Canonical artifacts are the collector's RESULTS.csv, ANALYSIS.json and
READOUT.md, plus each cell's round weights, curves, CONFIG/FROZEN/RESULT/CLOSE
and per-period trade/equity ledgers. Comparisons of train/development gaps use
daily log growth rather than subtracting multi-year and annual cumulative returns.
Window/stride studies remain documented as subsequent designs; this completed
queue does not silently launch an unreviewed successor experiment.
