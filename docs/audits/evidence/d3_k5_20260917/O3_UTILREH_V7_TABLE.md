# Utility run `utilreh-v7` — successor re-verification

Frozen commit `be1f170`, resumed `be1f170`; harness that applied `80c1ffa05502…`; failure policy `WORST_CASE_FAILED_COUNTED_AS_ADVANCES`.

## Calibrations (derived from every simulation)

| operator | stated advances/scored (+failed) | derived | stated bound | derived bound | decision bound | supports | why |
|---|---|---|---:|---:|---:|---|---|
| `cusum_causal` | 1/239 (+0) | 1/239 (+0) | 0.01969 | 0.01969 | 0.01969 | NO | derived upper bound 0.0197 exceeds alpha_adjusted 0.0125 (1/239 at 95%) |
| `delta_run_length` | 0/239 (+0) | 0/239 (+0) | 0.01246 | 0.01246 | 0.01246 | YES | — |
| `mad_extremes_trailing` | 0/239 (+0) | 0/239 (+0) | 0.01246 | 0.01246 | 0.01246 | YES | — |

## Contrasts

| unit | variable | operator | loss | raw | transformed | Δ (raw−transf.) | lower (1−α/m) | blocks | paired rows | train/val rows | cpu s | null scope | original | re-verified |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| `fab` | v0 | `cusum_causal` | mae | 0.87267 | 1.06149 | -0.18882 | -0.28650 | 4 | 2396/2400 | 6572/1198 | 1.227 | white_null ×239 @n=2400, bound 0.01969 | INCONCLUSIVE_UNCALIBRATED | INCONCLUSIVE_UNCALIBRATED |
| `fab` | v0 | `delta_run_length` | mae | 0.87261 | 0.85297 | +0.01964 | -0.00236 | 4 | 2395/2400 | 6568/1198 | 0.688 | white_null ×239 @n=2400, bound 0.01246 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |
| `fab` | v0 | `mad_extremes_trailing` | mae | 0.87248 | 1.06120 | -0.18871 | -0.29716 | 4 | 2381/2400 | 6529/1191 | 0.7 | white_null ×239 @n=2400, bound 0.01246 | DOES_NOT_ADVANCE | DOES_NOT_ADVANCE |

Decision delta: 0 contrast(s) changed; all evidence verified: True.

DOES_NOT_ADVANCE is not equivalence and says nothing about other domains, horizons or models; INCONCLUSIVE_UNCALIBRATED is descriptive (its record's derived bound exceeds α/m).
