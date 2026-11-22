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

## Follow-up already measured

Tasks 1–4 finished on 2026-11-22. The seed-7 control reached `queue_done` at 2026-11-22T18:27:08Z. Task 5 was not run.

| # | Task | Result |
|---|---|---|
| 1 | Encoder panel, 377 windows (256 train, 60 monitor, 61 holdout) | Frozen relative L2 is 0. The panel matches the saved CPU embeddings with max abs 2.38e-7. Parameter L2: branches 21.67, fusion 5.08, post_fusion 6.34, core 7.18. `fixed_fusion_offset` L2 3.04, absmax 0.729. Activation std: fusion 0.668, fusion plus offset 0.748, post_fusion 0.517, core and encoder output 0.353. All 22 branches and every activation are finite. |
| 2 | 90% predictive interval, 24 Flipout samples | short_12600 holdout coverage by horizon 0.739, 0.747, 0.741, 0.726, 0.726, 0.723. Mean width about 0.0055. long_8946 holdout coverage 0.757 at 24 h and 0.715 at 48 h. Width 0.00626 and 0.00660. Nominal is 0.90, so the intervals are short of calibrated. |
| 3 | 2024 semesters | short_12600 H1 MAE 0.001830 against naive 0.001237 (1684 rows). H2 MAE 0.001854 against naive 0.001356 (1619 rows). long_8946 H1 MAE 0.002170 against naive 0.003135 (600 rows). H2 MAE 0.002301 against naive 0.003113 (588 rows). The full-year sign holds in both halves. |
| 4 | Seed-7 encoder, model.pt not loaded | Embeddings differ from the pretrained encoder by max abs 11.41. All nine heads stopped at early-stop wait 80/80 and LR 4.8e-7. Every 2024 MAE is lower than the pretrained head. Best short is short_24896 at 0.001591 against pretrained 0.001845 and 2024 naive 0.001295. Best long is long_8946 at 0.002220 against pretrained 0.002235 and 2024 naive 0.003124. The pretrained recipe-61 weights do not improve this seasonal target. |
| 5 | Horizons 72–144 | Not run. The segment mask still does not admit them. |

2024 seasonal MAE of the seed-7 control minus the pretrained head. A negative delta means the control is lower. Monitor naive is 0.001391 short and 0.004382 long. 2024 naive is 0.001295 short and 0.003124 long.

| Job | Epoch | Champion | Train | Best val | Control 2024 | Pretrained 2024 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| short_1575 | 475 | 395 | 0.004125 | 0.002269 | 0.001803 | 0.002032 | −0.000229 |
| short_3150 | 430 | 350 | 0.003330 | 0.002123 | 0.001633 | 0.001902 | −0.000269 |
| short_6300 | 267 | 187 | 0.003157 | 0.002090 | 0.001611 | 0.001885 | −0.000275 |
| short_12600 | 188 | 96 | 0.002885 | 0.002099 | 0.001603 | 0.001842 | −0.000239 |
| short_24896 | 182 | 49 | 0.002818 | 0.002100 | 0.001591 | 0.001845 | −0.000255 |
| long_1575 | 226 | 146 | 0.004423 | 0.002798 | 0.002213 | 0.002255 | −0.000042 |
| long_3150 | 188 | 108 | 0.004279 | 0.002759 | 0.002232 | 0.002270 | −0.000038 |
| long_6300 | 183 | 103 | 0.003651 | 0.002744 | 0.002229 | 0.002242 | −0.000013 |
| long_8946 | 182 | 70 | 0.003747 | 0.002739 | 0.002220 | 0.002235 | −0.000014 |

short_24896 by horizon in 2024: 0.001485, 0.001535, 0.001571, 0.001619, 0.001651, 0.001684 against naive 0.000693, 0.001006, 0.001245, 0.001446, 0.001617, 0.001766. Horizon 6 is the only short horizon under its own persistence. long_8946: 0.002148 at 24 h against 0.003157, and 0.002293 at 48 h against 0.003090.

## Layout

- Pretrained champions: `/var/tmp/retsu-phase1/jobs`.
- Untracked-Flipout archive: `/var/tmp/retsu-phase1/jobs_flipout_untracked_20261121`.
- Follow-up outputs: `/var/tmp/retsu-phase1/followup`.
- Seed-7 control queue: `/var/tmp/retsu-phase1-control`.
- Tools in this repo: `tools/retsu_phase1`.
- CLOSE queues: `/var/tmp/retsu-phase1-close-seed7` and `/var/tmp/retsu-phase1-close-pretrained`.
- CLOSE summary: `/var/tmp/retsu-phase1/followup/close_results.json`.

## CLOSE target, 2026-11-22

Both queues finished `queue_done`. Seed-7 at 2026-11-22T19:24:19Z. Pretrained at 2026-11-22T19:28 UTC, log exit 2026-11-22T14:28:47-05:00. Every job stopped at early-stop wait 80/80 and LR 4.8e-7. The target is future CLOSE on the seasonal masks. The naive is CLOSE at the origin. 2024 persistence is 0.000552, 0.000780, 0.000971, 0.001124, 0.001272, 0.001415 for 1–6 h, and 0.003370, 0.004399 for 24 h and 48 h.

No horizon beats that persistence. The gates were 0.001415 at 6 h and 0.003370 at 24 h. The lowest 6 h MAE is 0.028627 (pretrained short_3150). The lowest 24 h MAE is 0.018976 (seed-7 long_1575). `Prediction_h_*` and `Prediction_d_*` were not written.

Several champions are epoch 1: the validation loss never improved on the CLOSE level after the first epoch.

| Encoder | Job | Epoch | Champion | Train | Best val | 2024 | 6 h or 24 h |
|---|---|---:|---:|---:|---:|---:|---:|
| seed-7 | short_3150 | 367 | 1 | 0.022680 | 0.031427 | 0.029349 | 6 h 0.029118 |
| seed-7 | long_1575 | 181 | 1 | 0.031648 | 0.022098 | 0.018952 | 24 h 0.018976 |
| pretrained | short_3150 | 181 | 1 | 0.020605 | 0.031143 | 0.028897 | 6 h 0.028627 |
| pretrained | long_1575 | 181 | 1 | 0.030552 | 0.022929 | 0.019464 | 24 h 0.019107 |

The other seven jobs on each encoder are higher. Full rows are in `close_results.json`.

## CLOSE change, 2026-11-22

The level run predicted the train mean of CLOSE (about 1.111, train MAE stuck at the 0.0208 standard deviation) and never approached the origin price. The next measurement is the change: future CLOSE minus CLOSE at the origin. Zero change is CLOSE persistence, so the 2024 gates stay 0.001415 at 6 h and 0.003370 at 24 h. Masks, horizons 1–6, 24 and 48, champion protocol, and both frozen embeddings are unchanged. The encoder stays frozen.

`build_dclose_targets.py` refused to train unless the recomputed 2024 persistence matched `followup/close_persistence.json`. It matched. The 1 h change standard deviation on the short pool was 0.001147, against a CLOSE-level standard deviation of 0.0544. Both queues finished `queue_done` with early-stop wait 80/80, five LR reductions, and LR 4.8e-7. Seed-7 exited 2026-11-22T16:08:46-05:00. Pretrained exited 2026-11-22T16:12:49-05:00. `Prediction_h_*` and `Prediction_d_*` were not written.

No 6 h forecast beat 0.001415362. The closest is seed-7 `short_12600` at 0.001415892. Seven of the eight 24 h forecasts sit under 0.003369959, by 2.6e-6 to 1.95e-5. The largest gap is seed-7 `long_8946` at 0.003350437. Seed-7 `long_6300` at 24 h is 0.003381522 and misses. The train mean of the 24 h change is about 1e-4, larger than every one of those gaps, and several champions are epoch 1. Validation MAE sits on the zero-change naive. Full rows are in `followup/dclose_results.json`.

| Encoder | Job | Epoch | Champion | Train | Best val | 2024 | Gate horizon |
|---|---|---:|---:|---:|---:|---:|---|
| seed-7 | short_1575 | 223 | 4 | 0.002866 | 0.001240 | 0.001021 | 6 h 0.001417293 |
| seed-7 | short_3150 | 184 | 3 | 0.002547 | 0.001239 | 0.001020 | 6 h 0.001416696 |
| seed-7 | short_6300 | 184 | 8 | 0.002217 | 0.001239 | 0.001020 | 6 h 0.001416139 |
| seed-7 | short_12600 | 181 | 13 | 0.001941 | 0.001237 | 0.001019 | 6 h 0.001415892 |
| seed-7 | short_24896 | 182 | 10 | 0.001814 | 0.001236 | 0.001019 | 6 h 0.001415904 |
| seed-7 | long_1575 | 194 | 1 | 0.007797 | 0.004091 | 0.003890 | 24 h 0.003366316 |
| seed-7 | long_3150 | 181 | 64 | 0.007015 | 0.004093 | 0.003890 | 24 h 0.003358936 |
| seed-7 | long_6300 | 186 | 7 | 0.006336 | 0.004111 | 0.003889 | 24 h 0.003381522 |
| seed-7 | long_8946 | 192 | 35 | 0.006460 | 0.004061 | 0.003871 | 24 h 0.003350437 |
| pretrained | short_1575 | 181 | 7 | 0.002664 | 0.001242 | 0.001026 | 6 h 0.001426734 |
| pretrained | short_3150 | 181 | 26 | 0.002287 | 0.001238 | 0.001024 | 6 h 0.001423725 |
| pretrained | short_6300 | 181 | 17 | 0.002064 | 0.001239 | 0.001024 | 6 h 0.001420456 |
| pretrained | short_12600 | 181 | 14 | 0.001831 | 0.001235 | 0.001023 | 6 h 0.001422806 |
| pretrained | short_24896 | 181 | 10 | 0.001730 | 0.001235 | 0.001024 | 6 h 0.001421997 |
| pretrained | long_1575 | 181 | 9 | 0.007414 | 0.004163 | 0.003882 | 24 h 0.003359879 |
| pretrained | long_3150 | 193 | 13 | 0.006852 | 0.004170 | 0.003880 | 24 h 0.003358567 |
| pretrained | long_6300 | 186 | 1 | 0.006235 | 0.004146 | 0.003881 | 24 h 0.003367314 |
| pretrained | long_8946 | 258 | 5 | 0.006303 | 0.004138 | 0.003882 | 24 h 0.003362254 |

## Account, 2026-11-22

The handoff sends the forecast and the naive through the same long/short plugin after the error tables. The price column is origin CLOSE plus the predicted change. The naive column is the origin CLOSE. Champions are seed-7 and pretrained `short_24896` with `long_8946`. The second predict reproduced the saved 2024 MAE within 1e-5. The individual is the plugin default `(5, 0.9, 2.0, 0.5, 2.0, 24)`, not fit on 2024. The feed is the 1188 holdout hours that have both horizons. Open, high and low are the CLOSE, because the STL series has no range.

Persistence takes no trade: the predicted price is the current price, and the entry asks for 5 pips. The change champions do trade, and the account falls.

| Arm | Profit USD | Trades | Win % | Sharpe | Max trade DD, pips |
|---|---:|---:|---:|---:|---:|
| seed-7 model | -220.23 | 145 | 46.2 | -9.04 | 306 |
| seed-7 persistence | 0.00 | 0 |  |  |  |
| pretrained model | -365.57 | 109 | 40.4 | -14.82 | 322 |
| pretrained persistence | 0.00 | 0 |  |  |  |

Start balance is 10000. Full trades are in `followup/account/account_results.json`. `Prediction_h_*` and `Prediction_d_*` were not written into the predictor pipeline.

## Curves, 2026-11-22

The handoff asks for a TRAIN/VAL figure per run, with the learning rate and each reduction, plus the hash of the restored champion. `plot_curves.py` read the 54 `epochs.jsonl` files of the six cited queues and did not fit or predict. The figures and `champion_sha256` values are in `/var/tmp/retsu-phase1/followup/curves/`. Nine jobs in each of seasonal pretrained, seasonal seed-7, CLOSE seed-7, CLOSE pretrained, change seed-7, and change pretrained.

## Encoder fine-tune, 2026-11-22

The frozen linear head does not beat CLOSE persistence, and the true seasonal plus the origin trend does. This contrast keeps that seasonal target, the same masks, the full pools, and the same champion rule. The only change is whether the recipe-61 encoder receives gradients. A fresh torch linear head is used in both arms so the head class is not confounded with Flipout. `dump_windows.py` rebuilt the 32618 windows of shape [13, 61] and matched the saved CPU embedding with max abs 0. Both queues finished at 2026-11-22T17:37:42-05:00. Every job stopped at early-stop wait 80/80, four LR reductions, and LR 4.8e-7.

| Arm | Job | Epoch | Champion | Train | Best val | 2024 seasonal |
|---|---|---:|---:|---:|---:|---:|
| frozen | short | 419 | 339 | 0.002318 | 0.002229 | 0.001808 |
| frozen | long | 205 | 125 | 0.002762 | 0.002739 | 0.002229 |
| tuned | short | 179 | 7 | 0.001398 | 0.001578 | 0.001372 |
| tuned | long | 179 | 4 | 0.001896 | 0.002708 | 0.002217 |

Tuned short by horizon: 0.001181, 0.001258, 0.001346, 0.001423, 0.001482, 0.001542. The 2024 seasonal naive is 0.000693, 0.001006, 0.001245, 0.001446, 0.001617, 0.001766. Horizons 4, 5 and 6 beat that naive. Horizons 1, 2 and 3 do not. The mean 0.001372 stays above the naive mean 0.001295. Tuned long is 0.002198 at 24 h and 0.002236 at 48 h, against seasonal naive 0.003157 and 0.003090. The frozen long arm is 0.002183 and 0.002274. Fine-tuning moves the short seasonal error and leaves the long arm where the frozen head already was.

Price reconstruction is trend at the origin plus the seasonal forecast, against future CLOSE. No horizon beats CLOSE persistence. Tuned short, horizons 1–6: 0.001067, 0.001211, 0.001356, 0.001505, 0.001640, 0.001775 against 0.000552, 0.000780, 0.000971, 0.001124, 0.001272, 0.001415. Tuned long: 0.004162 against 0.003370 at 24 h, and 0.004988 against 0.004399 at 48 h. The gates 0.001415 and 0.003370 stay closed. `Prediction_h_*` and `Prediction_d_*` were not written. Rows are in `followup/unfreeze_results.json`.
