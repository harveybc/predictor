# Temporal Capacity Ladder Result

The autonomous ladder completed7/7 new trials,all on first attempt,between
2026-10-20T09:50:33UTC and18:07:38UTC,about8h17min. All used seed7,
100000 transitions,final100000,selected15 features,window13,stride2 and the
same business/cost/data protocol. No candidate improved full VALIDATION2024
daily growth or net return over baseline;each also had worse validation maxDD.
The stage incumbent therefore stayed baseline throughout. No promotion means
the four conditional shared-seed fits were correctly not dispatched.70/70
new warehouse reports have readback,zero pending/mismatch;the independent
collector verified8 closed runs including baseline. No failed trials or active
fit remains in this sequence. TEST was not consumed.

| Model change from baseline | TRAIN return2020-23 | VALIDATION return2024 | V maxDD |
| --- | --- | --- | --- |
| Baseline:branches2/core4/hidden8/depth1 | -14.5477% | -7.2783% | 10.7319% |
| Core8 | -44.9437% | -11.5466% | 15.8127% |
| Core16 | -18.2250% | -17.6482% | 22.5606% |
| Hidden16 | -26.8143% | -23.9880% | 25.6218% |
| Hidden32 | -22.9119% | -17.6642% | 20.1785% |
| Branches4 | -42.1498% | -26.3552% | 28.3153% |
| Branches8 | -11.6612% | -22.6025% | 23.8043% |
| Core depth2 | -20.2817% | -15.6083% | 22.4963% |

Since no stage promoted,each row differs from the baseline by exactly the
listed capacity parameter. No hidden combination of successful earlier
changes exists in these results. Actor and twin-Q hidden widths change
together in the hidden contrasts. These are one-seed development results,
not proof every larger architecture fails or that capacity never interacts
with optimization and observation context.

Baseline remains unprofitable:TRAIN net-1454.77USD,maxDD37.0186%,827 trades,
44.8609% wins,PF0.9340,daily nonannualized Sharpe-0.02002;VALIDATION
net-727.83USD,maxDD10.7319%,203 trades,43.3498% wins,PF0.8326,Sharpe-0.06103.
The branches8 case modestly improves TRAIN while worsening VALIDATION;its
daily-growth gap grows from0.00011630 to0.00072393. Reducing the TRAIN/VALIDATION
gap by damaging TRAIN is not success. Raw four-year and one-year accumulated
returns are not interchangeable generalization-gap measurements.

Automatic RESULTS.csv retains all16 role rows including baseline,with profit
USD,return fraction,maxDD percent,daily nonannualized Sharpe,trades,win percent,
profit factor,seed and architecture. Closed child artifacts retain effective
config,checkpoint/data/code pins and four checkpoint curves. Sweep runtime
ee35cb3b,specb7cca8ea;all decisions and the no-promotion disposition are in the
generated trial ledger. No unrequested successor experiment was launched.
