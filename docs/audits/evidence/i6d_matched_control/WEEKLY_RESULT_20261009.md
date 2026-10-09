# I6-D weekly BUSINESS result, 2026-10-09

Scope: EURUSD `Y_s_1h`, 20 selected features, 52 validation weeks in 2024,
6,196 scored rows, seed 0, and a fresh full retrain over the preceding four
calendar years before every week. TEST was not read. Both arms use the same
rows and same-row zero-return naive.

## Annual result

| Training coordinate | Arm | MAE | MSE | Naive MAE | MAE skill | Weeks beating naive | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw target | Conv temporal | 0.000946410 | 0.000001816 | 0.000492598 | -92.126% | 0/52 | 3,494.9 |
| Raw target | Dense branches | 0.001321842 | 0.000013028 | 0.000492598 | -168.341% | 0/52 | 4,097.8 |
| Fit-only robust coordinate | Conv temporal | 0.000493193 | 0.000000614 | 0.000492598 | -0.121% | 18/52 | 2,109.1 |
| Fit-only robust coordinate | Dense branches | 0.000502364 | 0.000000643 | 0.000492598 | -1.982% | 18/52 | 1,921.0 |

The robust coordinate estimates median and MAD from each week's fitting rows,
fits in that coordinate, and inverts predictions before scoring. It does not
denoise or replace the target. Median completed epochs fell from 30 for both
raw-target arms to 10 for Conv and 8 for Dense.

The corrected Conv arm is the better architecture in the annual comparison,
but neither arm passes the annual naive gate. Consequently these predictions
are not eligible for the heuristic trading strategy and TEST remains sealed.

## Evidence identities

- Raw closure: `bf6d0bd304ee56a17bd6fe70506e143c984002a6b1fa74360ec3670e73d3b2b9`
- Robust-coordinate closure: `46c84b3f256b993ba0b206ac0f90efe3d006c3589f3f422a7bae1ee6a4bacd67`
- Robust-coordinate design: `4954d28ea397abbde59468cec04904711ba764f20da4b590938d109412a4e4ba`

## TRAIN-only horizon diagnostic

This separate diagnostic is not an annual BUSINESS result. It is retained only
to characterize horizon behavior before another governed campaign.

| Horizon (h) | Conv skill | Dense skill |
|---:|---:|---:|
| 1 | -160.782% | -142.209% |
| 2 | -66.172% | -66.866% |
| 3 | -30.676% | -138.059% |
| 4 | -21.502% | -58.573% |
| 5 | -27.258% | -65.697% |
| 6 | -21.893% | -39.047% |
| 24 | -20.090% | -120.443% |
| 48 | -14.264% | -47.388% |
| 72 | -5.392% | -14.752% |
| 96 | -2.908% | -13.400% |
| 120 | +0.717% | -19.960% |
| 144 | -5.877% | -17.415% |

Only Conv at 120 hours beats the paired naive in this TRAIN-only diagnostic.
It does not authorize strategy evaluation or a claim about validation/test.
