# FS-GEN: conditional generator and knockoffs as a calibrated control (lane report)

Order: `docs/handoffs/SATOSHI_FEATURE_SELECTION_CLOSURE_2026_10_05.md` §7 (secondary, non-blocking,
fail-closed). Contract: SYNTHETIC_OFFLINE (work plan §6.2). Roles only: coordinator, worker_b.

## What was built (feature-extractor, branch `satoshi/fs-gen-knockoffs-20261005` from 8987c57)

| Module | Role |
|---|---|
| `app/fs_gen/contracts.py` | conditions limited to `signal/observed_mask/delta_time/calendar`; target-like names and economic-calendar columns refused (reuses the lane D guards); rows after `train_end_ts` refused |
| `app/fs_gen/generator.py` | `ConditionalARGenerator`: heteroskedastic AR on the last 24 **observed** values, time encodings (sin/cos hour/dow/doy) and delta time; empirical innovation law; free-running synthesis preserves the real mask; `prefix_invariance_check` (FS01-style) |
| `app/fs_gen/calibration.py` | frozen gates (`CALIBRATION_RULE.json`): ACF lags 1-48, Welch PSD log-ratio, KS, q99 tail ratio, kurtosis ratio, regime coverage (rolling-std deciles), prefix invariance, minimum validation rows; knockoff exchangeability: per-feature second-moment gap, swap-classifier AUC, conditional-independence leak guard |
| `app/fs_gen/knockoffs.py` | second-order Gaussian **group** knockoffs (group equicorrelated S), Gaussian-copula marginal map, group Lasso-coefficient-difference statistic, knockoff+ threshold |
| `app/fs_gen/pipeline.py` | unattended batch driver over `ps2_batch.v1` batches: generator fit (fold fit range) -> diagnostics (inner validation range, inside TRAIN) -> knockoffs only for CALIBRATED features; `progress.json` after every feature; per-cell JSON (resumable); `generative_evidence*.csv`; `run_manifest.json` with digests and cost |
| `tests/test_fs_gen.py` | 16 tests, tiny synthetic, CPU; see `tests_receipt.json` |

Decisions recorded in `ACK.json`: generator = conditional AR (not CVAE) because it fits the CPU cap
(<= 3 GB, one seed), has an explicit conditional law (needed to argue exchangeability) and is
deterministic; knockpy is MIT (admissible) but not installed in the worker environment and installing
into a shared env on a host with a live GPU cell is an environment mutation, so the knockoff
construction is implemented in-repo (~150 lines).

Selection variable: the raw feature at t; its knockoff is `mu_t(real past, calendar, delta) +
sigma_t * copula(z~_t)` where `z~` is the second-order group knockoff of the standardized
innovations. Unobserved rows carry 0 in both X and X~ (mask-symmetric). Y_b is encoded TP=+1,
SL=-1, timeout=0, no support excluded. q = 0.10 (lane B `fdr_q`), knockoff+ offset 1, one seed (0).

## Data and scope

Inputs: the three PS2 batches on worker_b (`ps2/batch_001|002|003`, series/targets digests equal
to the lane B manifests: `3e4404d2…`, `1502e0f8…`, `293db083…`, targets `7dbb0b95…`), inner TRAIN
folds inner_2019..inner_2023, `train_end_ts` = 1703887200; TEST never read (asserted on the grid).
Denominator: 366 candidates from `coverage_reconciliation/coverage_by_batch_feature.csv`; 269
have a series in a batch, 10 `cal.*` are conditioning variables (NOT_APPLICABLE), 87
PROVISIONAL_LOW_PRIORITY features have no series in any batch (NOT_EVALUATED, knockoff
NOT_CALIBRATED).

Compute: worker_b CPU only, `crispdm-run -q -m 3G -t 20h`, sequential, 4 BLAS threads; GPU not
requested. The coordinator ran tests only (capped 1500M).

## Results

Filled from `generative_evidence.csv` / `run_manifest.json` when the run completes (see
`progress.json`). Until then: NO_NEW_MEASUREMENT.

## Regenerate

```bash
# on worker_b, from the deployed feature-extractor checkout
crispdm-run -q -m 3G -t 20h -n fsgen -- env CUDA_VISIBLE_DEVICES= python -m app.fs_gen.pipeline \
  --batch_dirs <ps2>/batch_003,<ps2>/batch_001,<ps2>/batch_002 --lane_b_dirs <same> \
  --out_dir <state>/fs_gen/runs/<run> --denominator_csv coverage_by_batch_feature.csv \
  --seed 0 --order 24 --max_fit_rows 20000 --fdr_q 0.10
```

## Limits

- The generative column is evidence and stability only; it never substitutes external utility,
  causal evidence or the paired refit (order §7).
- Exchangeability is tested operationally (moment gaps, swap AUC), not proven; the FDR guarantee
  is conditional on those gates and on the second-order approximation.
- Per-feature synthesis is marginal: FS09 applies, approving a marginal generator does not approve a
  joint dataset. Cross-dependence enters only through the innovation covariance used by the knockoffs.
