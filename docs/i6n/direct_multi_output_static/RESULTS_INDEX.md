# I6-DMO results

Reference identity and comparator clarification:
[short score reconciliation](SHORT_REFERENCE_RECONCILIATION.md). Original MG2
has 2/6 annual DEVELOPMENT naive wins; matched MG2m has 3/6. The 6/6 matched
downsampling wins are over the hourly learned control, not over naive. Exact
original-device reproduction retained all original scores without a new model.

| Stage | Result | Evidence |
|---|---|---|
| Exact elapsed DS2 short | Six heads above their own naive; 28 epochs, best 18, 94.3 s fit | [Release](RETURN_RELEASE.md) |
| Market-time support | MG2 validation 4772 origins; MG12 3804, versus zero for elapsed DS12 | [Support study](RETURN_SUPPORT_STUDY_20261015.md) |
| MG2 short fit and recovered report | 6/12 h improve MAE 0.049615%/0.093170%; 2/3/4/5 h remain above naive; 54 reports and 2006 metrics readback verified | [Recovery return](RETURN_REPORT_RECOVERY_20261015.md) |
| MG12 long | Six heads above naive by 24.08-41.08%; 37 epochs, best 27, 124.55 s | [Recovery return](RETURN_REPORT_RECOVERY_20261015.md#mg12-long-development-2024) |
| Pair and strategy | CLOSED / PAIR_NOT_ELIGIBLE; no trading or TEST; 110 reports / 3952 metrics delivered, 58 files per mirror | [Closure](RETURN_REPORT_RECOVERY_20261015.md#pair-disposition-publication-and-remaining-engineering) |
| Matched short MG1/MG2m | CLOSED, MG2_SELECTED. Mean relative MAE +4.321015% / -0.007738%; MG2m beats naive at 4/6/12 h | [Audit and per-horizon results](matched_short/MUSASHI_AUDIT_20261015.md) |
| Compatible branch R0/R2 | 19/19 donors, R0/probe complete; R2 FAILED before training: donor terminals absent on GPU host. No R2 prediction comparison; deferred behind RAW | [R2 audit](matched_short/MUSASHI_R2_AUDIT_20261015.md) |
| RAW long matched sampling/strategy | CLOSED / PAIR_NOT_ELIGIBLE. MG12m selected: mean relative MAE +26.285134% versus +47.377774% hourly MG1l; zero eligible long heads. Both GPU fits complete; 112 reports read back, 61 files per mirror; strategy correctly abstained | [Long result and per-horizon table](matched_long/RETURN.md) |

All reported 2024 results are development. TEST remains closed. Each stage
keeps its own support, model identity and cost; these are not controlled
ablations of sampling alone. Generated STATUS supersedes snapshot run states.
The closed MG2/MG12 pair has a [runtime-seal qualification](RETURN_REPORT_RECOVERY_20261015.md#runtime-qualification-added-2026-10-15):
actual TF 2.21.0 differed from declared TF 2.18.0; do not claim runtime compliance.
Machine-readable qualification `RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21`: [RUNTIME_QUALIFICATION_MG2_MG12.json](RUNTIME_QUALIFICATION_MG2_MG12.json).
