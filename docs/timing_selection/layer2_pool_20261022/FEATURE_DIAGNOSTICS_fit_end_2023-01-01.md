# Layer-2 pool_fit_end_2023-01-01: per-feature diagnostics (TRAIN 2020-01-01 .. 2023-01-01) and preprocessing decisions

Features 14 ({'kalman': 4, 'gmm': 4, 'hmm': 5, 'fracdiff': 1}); fit period for learned families 2020-01-01 .. 2023-01-01 (exclusive); decisions {'NONE': 14}.

Future perturbation (real-data prefix, rows after the cut replaced by noise + 500 appended; positive control = bar at the cut perturbed):

| family | columns | identical <= cut | max abs diff | cells changed after cut | positive control cols changed | passed |
|---|---|---|---|---|---|---|
| kalman | 4 | True | 0 | 7996 | 4 | True |
| gmm | 4 | True | 0 | 7987 | 4 | True |
| hmm | 5 | True | 0 | 9995 | 5 | True |
| fracdiff | 1 | True | 0 | 1999 | 1 | True |

| feature | family | ADF p | KPSS p | skew | kurt | decision | reason |
|---|---|---|---|---|---|---|---|
| kf.lc.level_minus_close | kalman | 0.000 | 0.100 | -0.074 | 14.926 | NONE | stationary, not strongly skewed |
| kf.lc.slope | kalman | 0.000 | 0.010 | -7.695 | 145.719 | NONE | stationary, not strongly skewed |
| kf.lc.innovation | kalman | 0.000 | 0.100 | 0.074 | 14.913 | NONE | stationary, not strongly skewed |
| kf.lc.innovation_std | kalman | 0.000 | 0.100 | 0.074 | 14.919 | NONE | stationary, not strongly skewed |
| rg.gmm.p0 | gmm | 0.000 | 0.010 | 0.010 | -1.868 | NONE | stationary, not strongly skewed |
| rg.gmm.p1 | gmm | 0.000 | 0.010 | 0.318 | -1.712 | NONE | stationary, not strongly skewed |
| rg.gmm.p2 | gmm | 0.000 | 0.010 | 2.931 | 7.037 | NONE | skewed (skew 2.93) but asinh does not halve it |
| rg.gmm.entropy | gmm | 0.000 | 0.010 | 0.929 | -0.504 | NONE | stationary, not strongly skewed |
| rg.hmm.p0 | hmm | 0.000 | 0.010 | 0.411 | -1.800 | NONE | stationary, not strongly skewed |
| rg.hmm.p1 | hmm | 0.000 | 0.010 | 0.522 | -1.650 | NONE | stationary, not strongly skewed |
| rg.hmm.p2 | hmm | 0.000 | 0.010 | 1.301 | -0.214 | NONE | stationary, not strongly skewed |
| rg.hmm.entropy | hmm | 0.000 | 0.011 | 2.372 | 4.853 | NONE | skewed (skew 2.37) but asinh does not halve it |
| rg.hmm.surprise | hmm | 0.000 | 0.010 | 8.293 | 265.401 | NONE | stationary, not strongly skewed |
| fd.lc.d0.4 | fracdiff | 0.002 | 0.010 | -0.689 | -0.130 | NONE | stationary, not strongly skewed |
