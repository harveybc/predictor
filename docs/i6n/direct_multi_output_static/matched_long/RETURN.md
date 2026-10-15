# RAW Long Matched Result

2026-10-15. Campaign `i6dmo-0f9e6c8d5d74`, executing commit `ac635820d37083a83c5b1c240543598b6476bf7d`.
**CLOSED / PAIR_NOT_ELIGIBLE**, automatic trigger; 6/6 units COMPLETE, zero failed.
This is full supported DEVELOPMENT-2024 evidence, not TEST performance.

## Experiment

19 retained RAW features, static training on 2020-2023 with purged INNER, seed 6102026.
Both arms span 144 open-market hours and include current t. MG12m uses 13 observations;
MG1l uses 145. Heads remain exact elapsed 18/24/36/48/60/72 h with hourly issuance.
Identical per-head common masks, FIT-only scales and original price units.
Parameters differ: 4882 versus 30226, including 25344 additional hourly readout parameters.
Thus the comparison is a practical recipe contrast, not an isolated sampling effect.
No smoothing, target denoising, new pretraining, short refit, DOIN or TEST.

## Measured Results

Each row has its own naive for the indicated physical horizon. Within that row,
both arms share exactly the same scored examples. Relative error is
`(MAE_h - naive_h) / naive_h`; positive is worse, negative is better.

| Horizon | Rows | Naive for this horizon | MAE MG12m | Relative MG12m | MAE MG1l | Relative MG1l |
|---|---:|---:|---:|---:|---:|---:|
| 18 h | 3410 | 0.002415050 | 0.002806181 | +16.196% | 0.003122208 | +29.281% |
| 24 h | 3145 | 0.002814782 | 0.003277624 | +16.443% | 0.003799042 | +34.968% |
| 36 h | 2615 | 0.003476803 | 0.004134853 | +18.927% | 0.005312871 | +52.809% |
| 48 h | 2091 | 0.004138977 | 0.005163759 | +24.759% | 0.006418434 | +55.073% |
| 60 h | 1565 | 0.004426645 | 0.006089152 | +37.557% | 0.006877049 | +55.356% |
| 72 h | 1038 | 0.004791474 | 0.006891529 | +43.829% | 0.007512073 | +56.780% |

The frozen equal-head relative means are +26.285134% MG12m and +47.377774% MG1l.
The comparison selected MG12m, but **zero long heads beat naive**. Retained short
MG2m has eligible 4/6/12 h; the frozen pair policy requires at least 1 short and 2 long.
Decision: `LONG_ELIGIBLE_HEADS_0_LT_2`; strategy abstained without reading prediction
arrays or trading bars on the negative path. No trading metrics were invented.

## Cost And Stopping

MG12m ran on 5090: 22 epochs, best 12, 74.4244 s fit, 2830843904-byte cgroup peak.
MG1l ran concurrently on 4090: 30 epochs, best 20, 123.2074 s fit, 3606646784-byte peak.
Both stopped on patience 10 using mean end-of-epoch FIT/INNER, min_delta 1e-8,
restored best weights. Maximum 200 epochs, one seed, no fit repetitions.
The 5070 Ti was never a parallel slot; coordinator did bounded CPU scoring.

## Publication And Custody

112/112 governed reports DELIVERED, zero pending or mismatched readbacks.
61 files verified in each of two physical mirrors, including authenticated CLOSE.
CLOSE SHA256: `81463a5eb1e37f6d9b53bd61ba34a04739cc2d47da7aab37ea7a0e461cdd6908`.
Population SHA256: `360f3bc1c7587e1ae17fe02d60714d233987c25c66b01c6bcce4d274ab3bbef2`.
Governed experiment set: `i6dmo:i6dmo-0f9e6c8d5d74`.
Full arrays, weights, DB backups and physical paths stay outside Git.
Numeric evidence and metrics pins: [RESULT.json](RESULT.json).
Master milestones and per-horizon result chart: [PROGRESS.png](PROGRESS.png).
Execution/status: [AUTOMATION.md](AUTOMATION.md).
Pre-fit failures retained: [PREFIT_INCIDENTS.md](PREFIT_INCIDENTS.md).

Campaign timers stopped automatically; the outer launcher timer then stopped
on its next CLOSED read. Four immutable closure-input snapshots and identical
CLOSE bytes in both mirrors were independently checked after closing. No
scientific task remains active. A later status-only presentation correction
exposes CLOSED / 6 of 6 and DONE ETAs at the top level while preserving the
original launcher journal. Its 18 focused checks and 14 independent checks
do not rerun or reseal any fit; original execution remains ac635820.

## Interpretation And Next Boundary

This closes the missing RAW long sampling/control and pair-decision block.
It does not prove that the target is unpredictable or that all model capacities,
cadences and representations fail. A single small static recipe is not an
optimized predictor. Sampling and readout capacity differ; devices also differ,
so no device-independent speedup or isolated sampling causality is established.
Selection reused DEVELOPMENT 2024: these are development findings, not independent
holdout claims. Source-age/FRED clocks, assumed nominal calendar and unverified
broker costs still prevent TEST promotion. The earlier closed-runtime qualification
remains intact. R2 and its retained donors remain deferred, not silently resumed.
Capacity/window and cadence experiments require their own frozen contrasts;
none was launched just to fill an idle GPU after this finite campaign closed.
