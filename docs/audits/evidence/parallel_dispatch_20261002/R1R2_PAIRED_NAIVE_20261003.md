# R1/R2 development contrast: paired same-row naive (2026-10-03)

Source of the owner's numbers: worker_a (gamma) `~/.local/state/scratch/m07/STATUS_r12_5090.json`, campaign `f2_eth4h_r1r2_5090_v1`
(root `~/.local/state/scratch/m07/campaign_eth_r12_5090`), 8/8 cells verified, seeds 2021 and 2022, split validation.
Dataset: ETHUSDT 4h, `eth4h_l24_h6_v1` (source commit b1f8a74f, 2190 validation rows, rows 13699-15895, train rows 0-13699, purge 6 bars),
target Y_h = z-scored cumulative log return, horizons 1..6, metric MAE in z_train units, objective = mean over the six horizons.
Naive computed with `stage_sres_naive_skill.py` + `tools/eth_forecast_naives.py` on validation rows only (intercepts fit on train rows only). No test rows read.
Full output: `R1R2_PAIRED_NAIVE_20261003.json`.

## Objective-equivalent naive MAE (mean over h=1..6, same 2190 rows)

| Naive | MAE |
|---|---|
| intercept-only, train median | 0.865066 (best) |
| intercept-only, train mean | 0.865085 |
| train mean (zero-return after scaling mu) | 0.865075 |
| zero-return | 0.865949 |
| persistence (last value) | 1.022438 |

Per horizon (best naive = train-mean/median constant): h1 0.4647, h2 0.6572, h3 0.8275, h4 0.9551, h5 1.0879, h6 1.1981.

## Paired table

| Config | model MAE_z | best naive (intercept median) | skill vs best naive | skill vs zero-return | skill vs persistence |
|---|---|---|---|---|---|
| per_feature + Huber + AdamW + R2 | 0.864938 | 0.865066 | +0.00015 | +0.00101 | +0.1538 |
| per_feature + Huber + AdamW + R1 | 0.865781 | 0.865066 | -0.00083 | +0.00019 | +0.1532 |
| per_feature + MAE + AdamW + R2 | 0.865862 | 0.865066 | -0.00092 | +0.00010 | +0.1531 |
| per_feature + MAE + AdamW + R1 | 0.866888 | 0.865066 | -0.00211 | -0.00108 | +0.1521 |

## Verdict

R2 (0.864938) beats the strongest naive only by 0.000128 z (skill +0.015%), which is a tie, not a result: the two seeds differ from each other by
4e-6 and the margin over a constant is of the same order as the constant-choice differences (mean vs median 2e-5). R1 (0.865781) does NOT beat the
intercept-only naive (skill -0.083%); it only edges the zero-return naive. Both models beat persistence by about 15%, but persistence is a weak,
non-minimum baseline for a z-scored cumulative return. Read plainly: at this development scale the contrast is a statement that the models have converged
to the unconditional constant; the R1 vs R2 ordering (0.00084) is real between configs but neither shows forecasting skill over a constant.
Not a licensing claim; development split only.
