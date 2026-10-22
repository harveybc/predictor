# Layer-2 pool: per-feature diagnostics (TRAIN 2020-01-01 .. 2023-01-01) and preprocessing decisions

Features 79 ({'wavelet': 18, 'hilbert': 8, 'multitaper': 7, 'strength': 26, 'kalman': 4, 'gmm': 4, 'hmm': 5, 'fracdiff': 1}); fit period for learned families 2020-01-01 .. 2024-01-01 (exclusive); decisions {'NONE': 67, 'DIFF': 5, 'ASINH': 1}.

Future perturbation (real-data prefix, rows after the cut replaced by noise + 500 appended; positive control = bar at the cut perturbed):

| family | columns | identical <= cut | max abs diff | cells changed after cut | positive control cols changed | passed |
|---|---|---|---|---|---|---|
| wavelet | 18 | True | 0 | 35520 | 18 | True |
| hilbert | 8 | True | 0 | 15991 | 8 | True |
| multitaper | 7 | True | 0 | 13878 | 6 | True |
| strength | 26 | True | 0 | 51766 | 26 | True |
| kalman | 4 | True | 0 | 7995 | 4 | True |
| gmm | 4 | True | 0 | 7987 | 4 | True |
| hmm | 5 | True | 0 | 9994 | 5 | True |
| fracdiff | 1 | True | 0 | 1999 | 1 | True |

| feature | family | ADF p | KPSS p | skew | kurt | decision | reason |
|---|---|---|---|---|---|---|---|
| wv.lc.A4 | wavelet | 0.764 | 0.010 | -0.604 | -0.632 | DIFF | non-stationary level (ADF p=0.764, KPSS p=0.010) |
| wv.lc.D4 | wavelet | 0.000 | 0.100 | -0.085 | 3.310 | NONE | stationary, not strongly skewed |
| wv.lc.D3 | wavelet | 0.000 | 0.100 | -0.076 | 5.197 | NONE | stationary, not strongly skewed |
| wv.lc.D2 | wavelet | 0.000 | 0.100 | -0.085 | 7.249 | NONE | stationary, not strongly skewed |
| wv.lc.D1 | wavelet | 0.000 | 0.100 | -0.033 | 10.124 | NONE | stationary, not strongly skewed |
| wv.lc.energy_D4 | wavelet | 0.000 | 0.010 | 2.813 | 10.233 | NONE | skewed (skew 2.81) but asinh does not halve it |
| wv.lc.energy_D3 | wavelet | 0.007 | 0.010 | 2.548 | 9.310 | NONE | skewed (skew 2.55) but asinh does not halve it |
| wv.lc.energy_D2 | wavelet | 0.032 | 0.010 | 2.346 | 7.785 | NONE | skewed (skew 2.35) but asinh does not halve it |
| wv.lc.energy_D1 | wavelet | 0.118 | 0.010 | 1.821 | 3.479 | DIFF | non-stationary level (ADF p=0.118, KPSS p=0.010) |
| wv.r1.A4 | wavelet | 0.000 | 0.100 | 0.173 | 9.659 | NONE | stationary, not strongly skewed |
| wv.r1.D4 | wavelet | 0.000 | 0.100 | -0.044 | 3.893 | NONE | stationary, not strongly skewed |
| wv.r1.D3 | wavelet | 0.000 | 0.100 | -0.120 | 5.675 | NONE | stationary, not strongly skewed |
| wv.r1.D2 | wavelet | 0.000 | 0.100 | -0.018 | 6.968 | NONE | stationary, not strongly skewed |
| wv.r1.D1 | wavelet | 0.000 | 0.100 | -0.070 | 9.434 | NONE | stationary, not strongly skewed |
| wv.r1.energy_D4 | wavelet | 0.000 | 0.010 | 2.809 | 10.419 | NONE | skewed (skew 2.81) but asinh does not halve it |
| wv.r1.energy_D3 | wavelet | 0.007 | 0.010 | 2.788 | 12.467 | NONE | skewed (skew 2.79) but asinh does not halve it |
| wv.r1.energy_D2 | wavelet | 0.048 | 0.010 | 2.290 | 6.941 | NONE | skewed (skew 2.29) but asinh does not halve it |
| wv.r1.energy_D1 | wavelet | 0.114 | 0.010 | 1.732 | 2.965 | DIFF | non-stationary level (ADF p=0.114, KPSS p=0.010) |
| hb.r1.amplitude | hilbert | 0.000 | 0.010 | 2.899 | 17.161 | NONE | skewed (skew 2.90) but asinh does not halve it |
| hb.r1.phase_cos | hilbert | 0.000 | 0.100 | -0.008 | -1.421 | NONE | stationary, not strongly skewed |
| hb.r1.phase_sin | hilbert | 0.000 | 0.100 | 0.008 | -1.529 | NONE | stationary, not strongly skewed |
| hb.r1.inst_freq | hilbert | 0.000 | 0.100 | -1.158 | 0.612 | NONE | stationary, not strongly skewed |
| hb.lc_detr.amplitude | hilbert | 0.000 | 0.010 | 1.477 | 3.016 | NONE | stationary, not strongly skewed |
| hb.lc_detr.phase_cos | hilbert | 0.000 | 0.100 | -0.018 | -1.533 | NONE | stationary, not strongly skewed |
| hb.lc_detr.phase_sin | hilbert | 0.000 | 0.100 | 0.009 | -1.556 | NONE | stationary, not strongly skewed |
| hb.lc_detr.inst_freq | hilbert | 0.000 | 0.100 | -0.728 | 4.633 | NONE | stationary, not strongly skewed |
| mt.r1.power_gt48h | multitaper | 0.001 | 0.010 | 2.094 | 5.195 | NONE | skewed (skew 2.09) but asinh does not halve it |
| mt.r1.power_12_48h | multitaper | 0.000 | 0.010 | 2.325 | 6.720 | NONE | skewed (skew 2.33) but asinh does not halve it |
| mt.r1.power_4_12h | multitaper | 0.000 | 0.010 | 2.340 | 7.205 | NONE | skewed (skew 2.34) but asinh does not halve it |
| mt.r1.power_lt4h | multitaper | 0.000 | 0.010 | 1.763 | 3.026 | NONE | stationary, not strongly skewed |
| mt.r1.spectral_entropy | multitaper | 0.000 | 0.010 | -1.119 | 2.400 | NONE | stationary, not strongly skewed |
| mt.r1.dominant_period_bars | multitaper | 0.000 | 0.010 | 11.597 | 158.704 | ASINH | positive and skewed (skew 11.60 -> 1.65) |
| mt.r1.total_power | multitaper | 0.000 | 0.010 | 1.963 | 4.496 | NONE | stationary, not strongly skewed |
| wv.usd.A4 | strength | 0.000 | 0.100 | -0.412 | 16.056 | NONE | stationary, not strongly skewed |
| wv.usd.D4 | strength | 0.000 | 0.100 | 0.067 | 7.318 | NONE | stationary, not strongly skewed |
| wv.usd.D3 | strength | 0.000 | 0.100 | 0.163 | 9.386 | NONE | stationary, not strongly skewed |
| wv.usd.D2 | strength | 0.000 | 0.100 | -0.023 | 10.002 | NONE | stationary, not strongly skewed |
| wv.usd.D1 | strength | 0.000 | 0.100 | 0.131 | 14.158 | NONE | stationary, not strongly skewed |
| wv.usd.energy_D4 | strength | 0.000 | 0.010 | 4.047 | 21.517 | NONE | skewed (skew 4.05) but asinh does not halve it |
| wv.usd.energy_D3 | strength | 0.003 | 0.010 | 3.989 | 21.545 | NONE | skewed (skew 3.99) but asinh does not halve it |
| wv.usd.energy_D2 | strength | 0.040 | 0.010 | 3.234 | 13.301 | NONE | skewed (skew 3.23) but asinh does not halve it |
| wv.usd.energy_D1 | strength | 0.038 | 0.010 | 3.273 | 13.280 | NONE | skewed (skew 3.27) but asinh does not halve it |
| hb.usd.amplitude | strength | 0.000 | 0.010 | 3.673 | 27.456 | NONE | skewed (skew 3.67) but asinh does not halve it |
| hb.usd.phase_cos | strength | 0.000 | 0.100 | 0.012 | -1.413 | NONE | stationary, not strongly skewed |
| hb.usd.phase_sin | strength | 0.000 | 0.100 | 0.012 | -1.531 | NONE | stationary, not strongly skewed |
| hb.usd.inst_freq | strength | 0.000 | 0.100 | -1.162 | 0.653 | NONE | stationary, not strongly skewed |
| wv.eur.A4 | strength | 0.000 | 0.100 | -0.286 | 10.324 | NONE | stationary, not strongly skewed |
| wv.eur.D4 | strength | 0.000 | 0.100 | 0.128 | 4.289 | NONE | stationary, not strongly skewed |
| wv.eur.D3 | strength | 0.000 | 0.100 | 0.072 | 4.776 | NONE | stationary, not strongly skewed |
| wv.eur.D2 | strength | 0.000 | 0.100 | 0.051 | 7.324 | NONE | stationary, not strongly skewed |
| wv.eur.D1 | strength | 0.000 | 0.100 | -0.036 | 9.428 | NONE | stationary, not strongly skewed |
| wv.eur.energy_D4 | strength | 0.000 | 0.010 | 3.276 | 15.819 | NONE | skewed (skew 3.28) but asinh does not halve it |
| wv.eur.energy_D3 | strength | 0.009 | 0.010 | 2.652 | 10.658 | NONE | skewed (skew 2.65) but asinh does not halve it |
| wv.eur.energy_D2 | strength | 0.041 | 0.010 | 3.086 | 14.489 | NONE | skewed (skew 3.09) but asinh does not halve it |
| wv.eur.energy_D1 | strength | 0.096 | 0.010 | 2.637 | 9.373 | DIFF | non-stationary level (ADF p=0.096, KPSS p=0.010) |
| hb.eur.amplitude | strength | 0.000 | 0.010 | 2.935 | 20.105 | NONE | skewed (skew 2.93) but asinh does not halve it |
| hb.eur.phase_cos | strength | 0.000 | 0.100 | -0.017 | -1.419 | NONE | stationary, not strongly skewed |
| hb.eur.phase_sin | strength | 0.000 | 0.100 | -0.005 | -1.528 | NONE | stationary, not strongly skewed |
| hb.eur.inst_freq | strength | 0.000 | 0.100 | -1.172 | 0.663 | NONE | stationary, not strongly skewed |
| kf.lc.level_minus_close | kalman | 0.000 | 0.100 | -0.072 | 14.927 | NONE | stationary, not strongly skewed |
| kf.lc.slope | kalman | 0.000 | 0.010 | -8.339 | 159.499 | NONE | stationary, not strongly skewed |
| kf.lc.innovation | kalman | 0.000 | 0.100 | 0.072 | 14.914 | NONE | stationary, not strongly skewed |
| kf.lc.innovation_std | kalman | 0.000 | 0.100 | 0.072 | 14.920 | NONE | stationary, not strongly skewed |
| rg.gmm.p0 | gmm | 0.000 | 0.010 | -0.049 | -1.866 | NONE | stationary, not strongly skewed |
| rg.gmm.p1 | gmm | 0.000 | 0.010 | 0.375 | -1.670 | NONE | stationary, not strongly skewed |
| rg.gmm.p2 | gmm | 0.000 | 0.010 | 2.953 | 7.166 | NONE | skewed (skew 2.95) but asinh does not halve it |
| rg.gmm.entropy | gmm | 0.000 | 0.010 | 0.929 | -0.491 | NONE | stationary, not strongly skewed |
| rg.hmm.p0 | hmm | 0.000 | 0.010 | 0.406 | -1.805 | NONE | stationary, not strongly skewed |
| rg.hmm.p1 | hmm | 0.000 | 0.010 | 0.601 | -1.556 | NONE | stationary, not strongly skewed |
| rg.hmm.p2 | hmm | 0.000 | 0.010 | 1.192 | -0.486 | NONE | stationary, not strongly skewed |
| rg.hmm.entropy | hmm | 0.000 | 0.010 | 2.309 | 4.508 | NONE | skewed (skew 2.31) but asinh does not halve it |
| rg.hmm.surprise | hmm | 0.000 | 0.010 | 8.860 | 294.708 | NONE | stationary, not strongly skewed |
| fd.lc.d0.3 | fracdiff | 0.079 | 0.010 | -0.640 | -0.424 | DIFF | non-stationary level (ADF p=0.079, KPSS p=0.010) |
