# D2 operator totals (derived from the decision records)

Grain: `subject_kind + subject + operator_params + regime`. Rows: published 3591, successor 3591.

| arm role | decision | rows | regimes | methods |
|---|---|---|---|---|
| (none) | SNR_CALIBRATED_FOR_REGIME | 39 | 28 | 4 |
| (none) | SNR_NOT_IDENTIFIABLE | 604 | 171 | 6 |
| (none) | SNR_REGIME_LIMITED | 167 | 81 | 5 |
| (none) | SNR_REJECTED | 216 | 107 | 5 |
| CANDIDATE | LAB_CALIBRATED | 47 | 13 | 7 |
| CANDIDATE | LAB_REJECTED | 825 | 116 | 7 |
| CANDIDATE | NOT_IDENTIFIABLE | 294 | 125 | 7 |
| CANDIDATE | REGIME_LIMITED | 6 | 6 | 2 |
| CANDIDATE | UNDERPOWERED | 880 | 106 | 7 |
| IDENTITY_RAW_CONTROL | LAB_CALIBRATED | 26 | 26 | 1 |
| IDENTITY_RAW_CONTROL | LAB_REJECTED | 40 | 40 | 1 |
| IDENTITY_RAW_CONTROL | NOT_IDENTIFIABLE | 105 | 105 | 1 |
| NON_CAUSAL_ORACLE_CONTROL | CONTROL_NOT_AN_ARM | 171 | 171 | 1 |
| PREVIOUSLY_REJECTED_CONTROL | LAB_CALIBRATED | 2 | 2 | 1 |
| PREVIOUSLY_REJECTED_CONTROL | LAB_REJECTED | 158 | 158 | 1 |
| PREVIOUSLY_REJECTED_CONTROL | NOT_IDENTIFIABLE | 11 | 11 | 1 |

## What moved

138 rows change decision; 7 lose a pass and 0 gain one.

| arm role | from | to | rows |
|---|---|---|---|
| (none) | SNR_REGIME_LIMITED | SNR_NOT_IDENTIFIABLE | 4 |
| (none) | SNR_REJECTED | SNR_NOT_IDENTIFIABLE | 13 |
| CANDIDATE | LAB_CALIBRATED | NOT_IDENTIFIABLE | 4 |
| CANDIDATE | LAB_REJECTED | NOT_IDENTIFIABLE | 1 |
| CANDIDATE | NOT_IDENTIFIABLE | LAB_REJECTED | 10 |
| CANDIDATE | NOT_IDENTIFIABLE | UNDERPOWERED | 9 |
| CANDIDATE | REGIME_LIMITED | NOT_IDENTIFIABLE | 1 |
| IDENTITY_RAW_CONTROL | LAB_CALIBRATED | NOT_IDENTIFIABLE | 2 |
| IDENTITY_RAW_CONTROL | LAB_REJECTED | NOT_IDENTIFIABLE | 94 |

## Superseded prose

the prose of the earlier reports: '48 LAB_CALIBRATED plus 6 REGIME_LIMITED' and 'seven lose a pass' are withdrawn; the records say 47 + 6 for the CANDIDATE arm, and five candidate losses plus two identity-control losses

