# Retsu phase-1 master checkpoint

2026-11-22. Dragon `queue_done` at 2026-11-21T13:13:44Z. The cited sweep is the one whose Flipout posterior is a Keras weight. All nine jobs stopped at early-stop wait 80/80 with learning rate 4.8e-7. MAE is the seasonal component in price units. Reloading each champion reproduces the 2024 MAE with difference 0.

Task 5 stays blocked. Tasks 1–4 are the follow-up this checkpoint authorizes: encoder panel and seed-7 control embeddings on Dragon CPU, coverage and the 2024 semester split on Dragon CPU at the same time, then the control head on the RTX 4090. Omega's GPU stays Musashi's.

## Results

| # | Result | Sign | Numbers |
|---|---|---|---|
| 1 | Identidad del encoder 61 | Positivo | Ventanas estrictas 2016–2024: 38782 filas, 28991 ids. 2020–2024: 19197, 14360. 2024–2025: 4912, 3627. Salida de 4 ventanas: [4, 208], finita. |
| 2 | Encoder CPU contra la 4090 | Positivo | Diferencia máxima absoluta 0.001052. El umbral de aborto era 0.01. Los embeddings guardados son los de CPU. |
| 3 | STL causal en train | Positivo | Periodo 48, seasonal 13, robust, historia 192. Fuerza en orígenes 0.3203 (mínimo 0.30). ACF del residual en orígenes −0.07455. ACF del residual en todo el train −0.09045 (tope 0.10). |
| 4 | Soporte de horizontes | Negativo de 72 h en adelante | Corto usable: 1–6 h. Pool 24896, monitor 1506, 2024: 3303. Largo usable: 24 y 48 h. Pool 8946, monitor 527, 2024: 1188. 72, 96, 120 y 144 h no dejan 1575 de train y 200 de monitor dentro del mismo segmento. |
| 5 | Barrido sin estandarizar la seasonal | Negativo, inválido | 2024 cerca de 2.74–2.77 en el corto y 2.34–2.44 en el largo, contra naive 0.001391 y 0.004382. La cabeza estaba en unidades de orden 1. No es el resultado. |
| 6 | short_1575 | Negativo contra la persistencia | Época 423, campeón 343, 4 reducciones. Train 0.003863, mejor val 0.002570, 2024: 0.002032. Naive 2024: 0.001295. Naive monitor: 0.001391. |
| 7 | short_3150 | Negativo | Época 440, campeón 360, 4 reducciones. Train 0.003315, mejor val 0.002403, 2024: 0.001902. Mismos naive. |
| 8 | short_6300 | Negativo | Época 281, campeón 201, 4 reducciones. Train 0.003190, mejor val 0.002316, 2024: 0.001885. |
| 9 | short_12600 | Negativo, el mejor corto | Época 183, campeón 103, 4 reducciones. Train 0.002963, mejor val 0.002274, 2024: 0.001842. Por horizonte en 2024: 0.001776, 0.001807, 0.001830, 0.001863, 0.001869, 0.001905 contra naive 0.000693, 0.001006, 0.001245, 0.001446, 0.001617, 0.001766. |
| 10 | short_24896 | Negativo | Época 182, campeón 56, 5 reducciones. Train 0.002894, mejor val 0.002291, 2024: 0.001845. Más datos después de 12600 no bajan el 2024. |
| 11 | long_1575 | Positivo contra la persistencia seasonal | Época 263, campeón 183, 4 reducciones. Train 0.004215, mejor val 0.002871, 2024: 0.002255. Naive 2024: 0.003124. Naive monitor: 0.004382. |
| 12 | long_3150 | Positivo | Época 203, campeón 123, 4 reducciones. Train 0.004086, mejor val 0.002767, 2024: 0.002270. |
| 13 | long_6300 | Positivo | Época 204, campeón 65, 5 reducciones. Train 0.003612, mejor val 0.002789, 2024: 0.002242. |
| 14 | long_8946 | Positivo, el mejor largo | Época 181, campeón 60, 5 reducciones. Train 0.003623, mejor val 0.002732, 2024: 0.002235. Horizonte 24: 0.002195 contra 0.003157, R² 0.095. Horizonte 48: 0.002274 contra 0.003090, R² 0.050. Desde 6300 el 2024 ya no baja. |
| 15 | Posterior Flipout entrenado | Positivo como peso, sin mejorar el error | `loc` std salió de 0.053 y quedó entre 0.094 y 0.216. Escala media salió de 0.99 y quedó entre 0.759 y 0.935. El 2024 quedó a menos de 0.00003 del barrido en que ese posterior estaba congelado. |
| 16 | Precio reconstruido, short_12600, 2024, 3303 filas | Negativo | Tendencia del origen más seasonal predicha contra el CLOSE futuro. Horizontes 1–6: 0.001873, 0.001962, 0.002050, 0.002154, 0.002241, 0.002351. Persistencia del CLOSE: 0.000552, 0.000780, 0.000971, 0.001124, 0.001272, 0.001415. |
| 17 | Precio reconstruido, long_8946, 2024, 1188 filas | Negativo | 24 h: reconstrucción 0.004071 contra persistencia del CLOSE 0.003370. 48 h: 0.004919 contra 0.004399. |
| 18 | Techo con la seasonal verdadera más la tendencia del origen | Mixto | short_12600, horizontes 1–6: 0.000523, 0.000654, 0.000786, 0.000912, 0.001031, 0.001142, por debajo de la persistencia del CLOSE. long_8946, 24 h: 0.002774 contra 0.003370. 48 h: 0.004967 contra 0.004399, por encima. |

The unscaled sweep in row 5 stays archived and is not cited. The strategy plugin did not receive these values.

## Pending work at the time of this checkpoint

| # | Task | What is measured | Machine and GPU | Parallel |
|---|---|---|---|---|
| 1 | Frozen encoder panel | L2 norm and activation range by group, 1×1 fusion, `fixed_fusion_offset`, `post_fusion` attention and the TCN core, on a fixed window panel | Dragon CPU. The 4090 is not required | Yes, with rows 2 and 3, and with a 4090 training job |
| 2 | Posterior coverage | Empirical versus nominal coverage, and interval width, by horizon, on H2 2023 and 2024. Uses the nine saved champions | Dragon CPU | Yes, with rows 1 and 3 |
| 3 | 2024 semester cut | MAE, naive and R² by horizon on 2024 H1 and H2 | Dragon CPU | Yes, with rows 1 and 2 |
| 4 | Control without pretrained weights | Same Bayesian head, same rows, same protocol. 2024 seasonal MAE against persistence, to see what the 61 encoder adds | Dragon RTX 4090. Omega's GPU stays Musashi's | Not with another training job on that 4090. Yes with rows 1–3 on CPU |
| 5 | Horizons 72, 96, 120 and 144 | Nothing under the current segment mask: there are not 1575 train rows and 200 monitor rows | None, until that rule changes | Not applicable |

## Layout

- Pretrained champions: `/var/tmp/retsu-phase1/jobs`.
- Untracked-Flipout archive: `/var/tmp/retsu-phase1/jobs_flipout_untracked_20261121`.
- Follow-up outputs: `/var/tmp/retsu-phase1/followup`.
- Seed-7 control queue: `/var/tmp/retsu-phase1-control`.
- Tools in this repo: `tools/retsu_phase1`.
