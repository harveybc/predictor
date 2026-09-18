# Utility run `utilpilot-v2` — successor re-verification

Frozen commit `be1f170`, resumed `85246fa`; harness that applied `80c1ffa05502…`; failure policy `WORST_CASE_FAILED_COUNTED_AS_ADVANCES`.

## Calibrations (derived from every simulation)

| operator | stated advances/scored (+failed) | derived | stated bound | derived bound | decision bound | supports | why |
|---|---|---|---:|---:|---:|---|---|
| `cusum_causal` | 0/538 (+0) | 0/538 (+0) | 0.00555 | 0.00555 | 0.00555 | YES | — |
| `delta_run_length` | 0/538 (+0) | 0/538 (+0) | 0.00555 | 0.00555 | 0.00555 | YES | — |
| `mad_extremes_trailing` | 1/538 (+0) | 1/538 (+0) | 0.00879 | 0.00879 | 0.00879 | NO | derived upper bound 0.0088 exceeds alpha_adjusted 0.0056 (1/538 at 95%) |

## Contrasts

| unit | variable | operator | loss | raw | transformed | Δ (raw−transf.) | lower (1−α/m) | blocks | paired rows | train/val rows | cpu s | null scope | original | re-verified |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `bumps__white__snr10__none__n2048__v1__seed11` | v0 | `cusum_causal` | mae | 0.18053 | 0.20386 | -0.02333 | -0.05163 | 4 | 2044/2048 | 5604/1022 | 1.036 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `bumps__white__snr10__none__n2048__v1__seed11` | v0 | `delta_run_length` | mae | 0.18053 | 0.18337 | -0.00283 | -0.00810 | 4 | 2043/2048 | 5600/1022 | 0.639 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `bumps__white__snr10__none__n2048__v1__seed11` | v0 | `mad_extremes_trailing` | mae | 0.17999 | 0.20476 | -0.02477 | -0.05323 | 4 | 2029/2048 | 5561/1015 | 0.655 | white_null ×538 @n=2048, bound 0.00879 | INCONCLUSIVE_UNCALIBRATED | INCONCLUSIVE_UNCALIBRATED |
| `sinusoid__white__snr10__none__n2048__v1__seed11` | v0 | `cusum_causal` | mae | 0.34573 | 0.40464 | -0.05891 | -0.07944 | 4 | 2044/2048 | 5604/1022 | 1.069 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `sinusoid__white__snr10__none__n2048__v1__seed11` | v0 | `delta_run_length` | mae | 0.34573 | 0.34798 | -0.00225 | -0.00580 | 4 | 2043/2048 | 5600/1022 | 0.647 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `sinusoid__white__snr10__none__n2048__v1__seed11` | v0 | `mad_extremes_trailing` | mae | 0.34647 | 0.40581 | -0.05934 | -0.07504 | 4 | 2029/2048 | 5561/1015 | 0.647 | white_null ×538 @n=2048, bound 0.00879 | INCONCLUSIVE_UNCALIBRATED | INCONCLUSIVE_UNCALIBRATED |
| `steps__white__snr10__none__n2048__v1__seed11` | v0 | `cusum_causal` | mae | 0.45855 | 0.56744 | -0.10889 | -0.14785 | 4 | 2044/2048 | 5604/1022 | 1.043 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `steps__white__snr10__none__n2048__v1__seed11` | v0 | `delta_run_length` | mae | 0.45855 | 0.45310 | +0.00545 | -0.01003 | 4 | 2043/2048 | 5600/1022 | 0.653 | white_null ×538 @n=2048, bound 0.00555 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `steps__white__snr10__none__n2048__v1__seed11` | v0 | `mad_extremes_trailing` | mae | 0.46035 | 0.57065 | -0.11030 | -0.16021 | 4 | 2029/2048 | 5561/1015 | 0.659 | white_null ×538 @n=2048, bound 0.00879 | INCONCLUSIVE_UNCALIBRATED | INCONCLUSIVE_UNCALIBRATED |

Decision delta: 0 contrast(s) changed; all evidence verified: True.

DOES_NOT_ADVANCE is not equivalence and says nothing about other domains, horizons or models; INCONCLUSIVE_UNCALIBRATED is descriptive (its record's derived bound exceeds α/m).

Resumed under new code: files changed ['docs/audits/evidence/d3_k5_20260917/N1_UTILREH_V4_RECOVERY.json', 'docs/audits/evidence/d3_k5_20260917/N4_UTILREH_V7_CONTENT_CHECK.json', 'docs/audits/evidence/d3_k5_20260917/N4_UTILREH_V7_REPORT.json', 'docs/audits/work_plan/SATOSHI_D3_N1_N5_RETURN_2026_09_17.md', 'tests/test_df_utility_run_order.py', 'tools/df_utility_run.py']; scientific code unchanged: True; harness unchanged: True.
