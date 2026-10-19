# Full-Year Checkpoint Diagnostic

Cut: 2026-10-19 18:40 UTC. No fits, no period changes, no new checkpoint
selection. Full fixed TRAIN2020-23 and VALIDATION2024 support for every pair.
The diagnostic rehashed referenced artifacts, independently recomputed growth
from equity traces and replayed the declared mean objective and patience.
Each pair retains24837 TRAIN and6196 VALIDATION observations; timestamp support
and admitted input identity are identical across checkpoints within each role.

## Findings

| Arm | Complete pairs | Resets after first | Resets with V worse than selected incumbent | V-positive checkpoints | Official selected step | Selected TRAIN return | Selected V return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No regularizer, closed | 20 | 9 | 5 | 2 | 100000 | +361.075936% | +0.751879% |
| Critic dropout0.05, closed | 20 | 3 | 2 | 0 | 80000 | +85.408261% | -7.969994% |
| Critic L2 1e-6, partial through40000 | 8 | 2 | 2 | 0 | 30000 | +50.078323% | -19.585567% |

Returns here are cumulative over their own full periods; TRAIN covers four
years and V one year. The objective comparison uses mean log growth per
observed New York date, not the unequal-duration return percentages above.
The monitor includes observed partial-session dates in its day count; this
diagnostic preserves that exact definition, not a newly chosen weekday count.

Compare each improvement to the previously selected incumbent, not always the
previous evaluated checkpoint. Against the immediately previous evaluated
checkpoint, reset-with-V-worse counts are4,0,1 respectively. These differ because
intermediate checkpoints may have been rejected. Every comparison is retained.

The mean objective can improve through TRAIN while V deteriorates, as expected
mathematically and now observed. This does not prove a software defect.
In the baseline,100000 also has the best observed V return: stopping earlier
would not improve its observed V performance. Dropout has no profitable V
checkpoint at all. Thus selecting an earlier checkpoint alone does not produce
a profitable dropout policy. No conclusion is final for the active L2 arm.

## Complete Closed-Arm Validation Curves

Every cell evaluates the same whole validation year. These are diagnostic
observations, not a replacement selection criterion or a selected subset.

| Transitions | Baseline V return % | Dropout V return % |
| ---: | ---: | ---: |
| 5000 | -7.129511 | -2.360732 |
| 10000 | -11.548146 | -8.819618 |
| 15000 | -21.253862 | -4.744664 |
| 20000 | -17.575644 | -2.130278 |
| 25000 | -3.487669 | -9.362713 |
| 30000 | -4.336009 | -8.245315 |
| 35000 | -9.316371 | -11.911863 |
| 40000 | -12.521192 | -5.892644 |
| 45000 | -9.319530 | -10.890970 |
| 50000 | -4.200168 | -13.919468 |
| 55000 | -6.098592 | -10.976615 |
| 60000 | -6.200586 | -15.226593 |
| 65000 | -11.900711 | -18.480216 |
| 70000 | -10.298286 | -17.260912 |
| 75000 | -15.913299 | -11.958420 |
| 80000 | -16.915651 | -7.969994 |
| 85000 | -9.856728 | -13.880684 |
| 90000 | -10.784424 | -15.506072 |
| 95000 | +0.150892 | -9.059195 |
| 100000 | +0.751879 | -14.459813 |

## Evidence And Persistence

Analyzer: agent-multi `tools/paired_checkpoint_diagnostics.py`,17 focused
CPU tests. Same script ran read-only against real source artifacts in bounded
CPU services. No model was loaded for training and no learner was changed.
48/48 governed diagnostic reports delivered with readback;0 pending/mismatch.
Closed SQLite backups and report artifacts mirrored to a second physical
machine; recursive content-checksum comparison returned no differences.

| Arm | RESULT.json SHA256 |
| --- | --- |
| Baseline | af310e17de9e428f205728f988239027e68ab648e4911ea78743e83e0898f301 |
| Dropout | 13545876a3edbfbae040ce2be4e2275de327160957d32fd2e32e8e0120c0b880 |
| L2 snapshot | 0f5b85727a4f76a8281c8d1093981bf9c812fc6d5a788fb914c2d4d252f34a8a |

No causal diagnosis follows from these curves alone. This closes the narrow
checkpoint/mean-objective investigation in GD18 for the two completed arms;
the remaining GD hypotheses still need their declared evidence. Preserve the
current regularization queue and the paused status of new temporal fits.
