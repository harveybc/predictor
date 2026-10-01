# M04 batch-1 v3: incumbent history with per-horizon skill (generated)

Comparability: **NOT_COMPARABLE**: ECL L24 -> H1..24, all 321 channels, z_train MAE on the validation split; the published TimeFilter/ECL rows are L96 -> H96 on the test split, so no literature value applies.

## Incumbent 1: `e323775b` grouped32_R0_mae, mean validation MAE 0.402279 over seeds [2021, 2022]
Reason: first paired-seed verified configuration.

| seed cid | MAE | persistence MAE | skill MAE | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 | h17 | h18 | h19 | h20 | h21 | h22 | h23 | h24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4c477413d1cf615a | 0.403343 | 0.851406 | 0.5263 | -0.52 | +0.08 | +0.31 | +0.44 | +0.52 | +0.57 | +0.61 | +0.63 | +0.64 | +0.66 | +0.66 | +0.66 | +0.66 | +0.66 | +0.65 | +0.63 | +0.61 | +0.58 | +0.53 | +0.45 | +0.34 | +0.18 | -0.13 | -0.64 |
| c52af1984bcc7389 | 0.401216 | 0.851406 | 0.5288 | -0.54 | +0.06 | +0.31 | +0.44 | +0.51 | +0.56 | +0.60 | +0.63 | +0.64 | +0.66 | +0.67 | +0.67 | +0.67 | +0.66 | +0.66 | +0.64 | +0.62 | +0.58 | +0.53 | +0.46 | +0.36 | +0.19 | -0.12 | -0.63 |

Horizons with NEGATIVE skill (persistence wins) in any seed: ['h1', 'h23', 'h24'].

## Incumbent 2: `cc4235d7` draw1_huber, mean validation MAE 0.397174 over seeds [2021, 2022]
Reason: lower mean validation objective across all paired seeds.

| seed cid | MAE | persistence MAE | skill MAE | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 | h9 | h10 | h11 | h12 | h13 | h14 | h15 | h16 | h17 | h18 | h19 | h20 | h21 | h22 | h23 | h24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 51cb6a1a07776228 | 0.403170 | 0.851406 | 0.5265 | -0.49 | +0.08 | +0.31 | +0.43 | +0.51 | +0.56 | +0.60 | +0.63 | +0.65 | +0.67 | +0.67 | +0.67 | +0.66 | +0.65 | +0.63 | +0.61 | +0.60 | +0.57 | +0.53 | +0.46 | +0.36 | +0.19 | -0.10 | -0.58 |
| 938871724b45a521 | 0.391178 | 0.851406 | 0.5406 | -0.44 | +0.13 | +0.35 | +0.45 | +0.53 | +0.58 | +0.61 | +0.64 | +0.66 | +0.67 | +0.68 | +0.67 | +0.67 | +0.65 | +0.64 | +0.63 | +0.61 | +0.58 | +0.54 | +0.48 | +0.38 | +0.22 | -0.07 | -0.53 |

Horizons with NEGATIVE skill (persistence wins) in any seed: ['h1', 'h23', 'h24'].

