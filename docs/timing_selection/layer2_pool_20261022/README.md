# Layer-2 pool for EURUSD H1 (2026-10-22) and the machine-made FEATURE_CATALOG

Producers (committed): `tools/layer2_pool_h1.py` (families below), `tools/feature_catalog_h1.py` (catalog),
`tools/lib/native_wavelet.py` (vendored verbatim from financial-data `_scripts/lib/native_wavelet.py` @307ded100),
`tools/lib/gaussian_hmm.py` (numpy Gaussian HMM, EM + causal forward filter; hmmlearn is not installed),
`tools/lib/kalman_llt.py` (local-linear-trend filter after feature-eng `cross_asset_h1.py` @c133ccd, with Q and R by ML).
Tests: `tests/test_layer2_pool_h1.py` (20) and `tests/test_feature_catalog_h1.py` (8). Stores live outside Git:
`/var/tmp/features-candidates/eurusd_h1_layer2_pool/` and `/var/tmp/features-candidates/eurusd_h1_catalog/`.

Grid: the admitted EURUSD H1 file (sha256 c80847bb…), rows < 2025-01-01 only (31,116 bars 2020-01-02 .. 2024-12-31),
UTC bar start, decision at bar start + 1 h. No TEST row is read by any producer (enforced, tested).

| family | columns | fitted on | notes |
|---|---|---|---|
| wavelet `wv.lc.*`, `wv.r1.*` | 18 | nothing | PyWavelets db4 wavedec, trailing window 256, level 4, right-edge coefficient of A4/D4..D1 + band energies, on log close and on 1-bar log returns |
| hilbert `hb.r1.*`, `hb.lc_detr.*` | 8 | nothing | scipy analytic signal on the trailing window of 128 (returns demeaned; log close OLS-detrended inside the window); amplitude, cos/sin phase, instantaneous frequency at the last sample; the FFT wrap-around edge effect is documented in the manifest |
| multitaper `mt.r1.*` | 7 | nothing | DPSS NW=3.5, K=5, window 256 of returns; band powers (>48 h, 12-48 h, 4-12 h, <4 h, cycles/bar), spectral entropy, dominant period, total power |
| strength `wv.usd.*`, `wv.eur.*`, `hb.usd.*`, `hb.eur.*` | 26 | nothing | the same wavelet/Hilbert producers on the cross-asset r4 `fx.usd_strength.ret_1` / `fx.eur_strength.ret_1` returns (finite share 99.98 % / 99.99 %), used as returns, never cumulated or filled |
| kalman `kf.lc.*` | 4 | 2020-01-01 .. 2024-01-01 | filter only; q_level 1.045e-6, q_slope 2.2e-24 (zero boundary: the slope degenerates to a constant drift, recorded as `boundary_hits`), r 2.13e-8 |
| gmm `rg.gmm.*` | 4 | 2020-01-01 .. 2024-01-01 | sklearn GaussianMixture k=3 diag on (logret_1, rv_24, log_range); row-wise posterior; states by rv_24 mean 0.00357 / 0.00523 / 0.00811 |
| hmm `rg.hmm.*` | 5 | 2020-01-01 .. 2024-01-01 | numpy Gaussian HMM k=3 diag, EM 17 iterations, causal forward filter; states by rv_24 mean 0.00290 / 0.00471 / 0.00758 |
| fracdiff `fd.lc.d0.3` | 1 | 2020-01-01 .. 2024-01-01 | fixed 256-weight window; d = 0.3 is the smallest grid value with ADF p < 0.05 on the fit period (table in the manifest) |

A second set for the RL three-year development arm (`*_fit_end_2023-01-01.*`, learned families only, fit
2020-01-01 .. 2023-01-01): Kalman q_level 1.081e-6, q_slope 9.3e-15 (boundary), r 3.01e-8; GMM states
0.00342 / 0.00531 / 0.00843; HMM states 0.00285 / 0.00478 / 0.00794; fracdiff d = 0.4.

Diagnostics follow layer-1 (TRAIN 2020-01-01 .. 2023-01-01; ADF/KPSS/skew/kurtosis; NONE/DIFF/ASINH only when the
metric justifies it): default set 67 NONE / 5 DIFF / 1 ASINH (companion columns `@diff`/`@asinh` stored beside the
raw value). Note the fracdiff selection uses the contract fit period (2020-2023, ADF p = 0.0094 at d = 0.3) while
the TRAIN-2020-2022 diagnostic gives ADF p = 0.079, so the layer-1 rule adds a `fd.lc.d0.3@diff` companion; on the
2020-2022 fit the rule picks d = 0.4.

Future perturbation on real data (prefix 6,000 bars, cut at 4,000, rows after the cut replaced by noise and 500
synthetic bars appended; positive control = the bar at the cut perturbed): every family bit-identical at <= cut,
positive control changes every column except one multitaper column (dominant period) — table in FEATURE_DIAGNOSTICS.md.

Catalog (370 rows, 29 columns exactly as agreed): 290 GENERATED, 79 SUPERSEDED_BY_R4 (the r3 cross-asset columns
still packaged inside layer-1; the r4 store exists, so the selection lane consumes `r4:<column>`), 1 PENDING
(economic-calendar consensus/surprise: owner decision on a point-in-time source). `selection_decision` and
`consumed_by` are PENDING for the selection lane.
