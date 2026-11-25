# Mini work plan — fase 2 del canal estacional

Este archivo manda. Si un script, un default o una idea posterior disienten de aquí, se obedece este archivo. No se abre otro experimento. No se retoma el barrido viejo de 61 columnas ni se sustituye una entrada por su estacional de serie completa.

El worker marca la casilla del paso solo cuando el artefacto de ese paso existe, y no empieza el paso siguiente con la casilla anterior abierta. Actualizar la casilla es parte del paso. No se añaden pasos.

## Hecho que no se toca

- Corrida causal terminada: `/var/tmp/retsu-phase1-causal` en Dragon, identidad `recipe_08d65af029b62773`. No se reescribe, no se reentrena, no se borra.
- Target STL ya elegido (periodo 48, seasonal 13, robust true, history 192). No se recalcula. `y` guardado es el incremento estacional. `seasonal_origin` queda en cero.
- 2024 se predice una sola vez, al final de este plan, después de restaurar el campeón. Ninguna celda intermedia mira 2024.
- Identidades prohibidas: `recipe_b8fa85625fe025a7`, `recipe_b5fb7bcdfa1dbec8`.
- No se editan `plugin_loader`, `plugin_resolver`, `config_merger` ni `app/causal_stl_inputs.py`. Se importa `fit_history` y `endpoint_series`.
- No se escriben columnas `Prediction_h_*` ni `Prediction_d_*`.
- No se relaja la máscara de misma sesión. No hay horizonte de 72 h o más. El año de test no se mueve a 2025.
- No se usa la GPU de omega. Se entrena en Dragon (`harveybc@192.168.1.235`, puerto 62024, venv `/var/tmp/discrete-sac-probe-venv`). No se hace reset del master local de Dragon.
- Los commits de esta campaña salen del worktree detached `/tmp/predictor-master-ff` con `git push origin HEAD:master`. No se hace commit del checkout sucio de omega.
- `decide_denoising` no se aplica.

## Constantes

El builder lee solo este bloque. No duplica estos números en otro default.

```phase2-plan
{
  "schema": "retsu_phase2_mini_plan.v1",
  "decision_available_at_before": "2023-07-01T00:00:00Z",
  "bar_to_available": "1h",
  "frozen_fit_before": "2022-07-01T00:00:00Z",
  "frozen_es_from": "2022-07-01T00:00:00Z",
  "frozen_es_until": "2023-01-01T00:00:00Z",
  "mask_score_from": "2023-01-01T00:00:00Z",
  "mask_score_until": "2023-07-01T00:00:00Z",
  "monitor_from": "2023-07-01T00:00:00Z",
  "monitor_until": "2024-01-01T00:00:00Z",
  "test_from": "2024-01-01T00:00:00Z",
  "acf_lags_bars": [24, 48],
  "acf_min_rows": 2000,
  "acf_z": 1.96,
  "stl_seasonal": [7, 13],
  "stl_robust": [false, true],
  "stl_history_multiples": [4, 8],
  "stl_history_floor": 64,
  "stl_forecast_horizon_bars": 24,
  "stl_origins": 16,
  "close_period_default_bars": 48,
  "never_transform_prefixes": ["cal.", "ev.", "hod_", "dow_", "rg."],
  "never_transform_exact": ["px.elapsed_hours", "px.hours_since_session_open", "px.unexpected_gap_hours"],
  "parent_recipe": "recipe_08d65af029b62773",
  "forbidden_recipe": ["recipe_b8fa85625fe025a7", "recipe_b5fb7bcdfa1dbec8"],
  "finished_dir": "/var/tmp/retsu-phase1-causal",
  "out_dir": "/var/tmp/retsu-phase1-causal-p2",
  "hist_ohlc": "/var/tmp/sac399-series/history397-61-r2/OHLC.parquet",
  "bars_parquet": "/var/tmp/sac70-warm-v3-20261108/data/bars.parquet",
  "min_count": 2000,
  "probe_bars": 4200,
  "probe_cut": 3800,
  "prefix_dust": 1e-8,
  "tolerance": 0.0,
  "remove_on_tie_deletion_only": true,
  "min_features": 1,
  "seed": 7,
  "head_lr": 3e-4,
  "weight_decay": 0.004,
  "encoder_lrs": [3e-5, 3e-6],
  "reduce_lr_grace": 20,
  "reduce_lr_patience": 20,
  "reduce_lr_factor": 0.2,
  "lr_floor": 4.8e-7,
  "es_patience": 80,
  "es_start_epoch": 100,
  "min_delta": 1e-8,
  "max_epochs": 2000,
  "batch": 256,
  "trajectory_coefficients": 3,
  "short_horizons": [1, 2, 3, 4, 5, 6],
  "long_horizons": [24, 48],
  "short_h_ref": 6,
  "long_h_ref": 48
}
```

`available_at` es el timestamp de barra (inicio) más 1 hora. 24 h y 48 h son 24 y 48 barras de la grilla H1, el mismo reloj que el periodo 48 del target. Los huecos de cierre no se interpolan.

## Paso 1 — estacionariedad y sesgo

Entrada: las 111 columnas de `causal_features_raw.parquet` del directorio terminado, en su orden, sin el z expansivo. Esas columnas ya son salidas causales. No se reconstruye Kalman ni GMM.

Regla: `decide()` y `diagnostics()` de `tools/layer1_pool_h1.py`, sin reescribirlas. El voto es solo las filas con `available_at` anterior a 2023-07-01. H2 2023 y 2024 no votan. DIFF es la primera diferencia hacia atrás que devuelve `decide`. ASINH, LOG y ASINHS siguen el orden y el dominio congelado de esa función: una fila posterior fuera del dominio pasa a NaN y no cambia la receta.

No se transforma calendario, reloj de eventos ni probabilidades de régimen. Eso cubre los prefijos y los nombres exactos del bloque, más lo que `decide` ya trata como acotado (`Stochastic_D`, `BB_position`). `px.er_1` sigue relleno con 1.

La columna que ve el modelo es la serie ya transformada, con el mismo nombre. La serie previa no entra al modelo. El parquet nuevo vive en `out_dir`. El parquet terminado no se modifica.

## Paso 2 — estacional causal al lado, no en lugar

Se conserva la columna del paso 1. Al lado se agrega el endpoint estacional causal solo si el TRAIN de esa columna muestra un pico en 24 o en 48 barras. Siempre se agrega el estacional causal de CLOSE. No se descompone calendario ni reloj de eventos. No se reemplaza la columna por su estacional.

Pico, en la serie ya transformada y solo en filas de decisión: se quita el NaN inicial de calentamiento; si queda un NaN interior, esa columna no recibe estacional. Hacen falta al menos 2000 filas finitas. ACF con `nlags=50`. El lag L en {24, 48} es pico si `acf[L] > 0`, `acf[L] > 1.96/sqrt(n)`, `acf[L] > acf[L-1]` y `acf[L] >= acf[L+1]`. Si los dos califican, el periodo es el de mayor ACF.

Parametros solo en TRAIN, grilla cerrada del bloque: `seasonal` en {7, 13}, `robust` en {false, true}, `history = max(64, multiple * period)` con multiple en {4, 8}. El criterio es el de `choose_parameters`: MAE escalado del pronóstico a 24 barras en 16 orígenes cuyo horizonte sigue dentro del corte de decisión. Empate: menor history, luego menor seasonal, luego robust false. No se llama `choose_parameters` tal cual, porque su grilla puede elegir periodo 6, 12 o 120. Se llama `fit_history`.

CLOSE es incondicional. Si ningún lag califica, el periodo es 48 y el resto de la grilla se elige igual en TRAIN. Si CLOSE no tiene orígenes elegibles, el build falla. No se copian los parámetros del target como atajo y no se vuelve a ajustar el target.

El valor en t usa solo la historia hasta t: `endpoint_series` / `CausalSTLColumn`, último punto estacional. Ajustar STL en la serie completa y cortar en t es filtración. Cada estacional es su propio bloque `conv`, units 2, nombre `stl.close` o `stl.` más el nombre de la columna, después de los 24 grupos originales. Un bloque de senos y cosenos, o el bloque de probabilidades GMM, no se parte.

El z expansivo se recalcula después del paso 1 y del paso 2, con filas finitas estrictamente anteriores a t, ddof 0, varianza recortada a 0, desviación 0 sustituida por 1, NaN si el conteo pasado es menor que 2000. Los arrays de la receta siguen en identidad y no se reaplican.

Sonda, con la receta ya congelada (no se reajustan las decisiones dentro de la sonda):

- Prefijo de 4200 barras recomputado hasta el corte 3800: polvo por debajo de 1e-8, sin desacuerdo de NaN.
- Perturbar las filas desde el corte y recomputar con la misma receta: cero celdas distintas antes del corte.
- Control positivo: la fila del corte de `cdl.body_pips` se mueve.

Ventanas: offsets `[-24, -22, …, 0]`, 13 observaciones. Labels: el `dataset.npz` terminado, de solo lectura. `y` ya es el incremento; no se resta el origen otra vez. `ok` nuevo = `ok` guardado y ventana finita. La supervivencia de monitor y holdout no se relaja a propósito: si el estacional abre NaN, esa fila sale, y se anota el conteo.

Estado de la receta intermedia: columnas y transformaciones cerradas antes de 2023-07-01, selección todavía abierta. `selection_period` null. No es la receta del campeón.

## Paso 3 — two_pass_greedy

Algoritmo de Musashi: todos los bloques admitidos, en el orden del manifest. Pase 1: probar quitar cada bloque una vez contra el subconjunto aceptado; aceptar solo una mejora elegible mayor que la tolerancia, o un empate exacto porque `remove_on_tie` aplica solo al borrado. Pase 2: solo los bloques quitados, en el orden en que salieron; reinsertar cada uno una vez y conservarlo solo si el score mejora de verdad. Sin tercer pase, sin vecindario, sin reinicio. Como máximo `1 + 2B` evaluaciones. No se edita el paquete `feature_selector`. Si su flag de empate también afectara la reinserción, el pase 2 se ejecuta en este script con la regla estricta.

Score, por familia (corto y largo por separado): en el corte interior `[2023-01-01, 2023-07-01)` de `available_at`, media sobre los horizontes de `(naive_mae_h - mae_h)`. El naive es la persistencia del nivel estacional, o sea el incremento cero, medida en ese mismo corte. Mayor score es mejor. Tolerancia 0. No es retorno de cuenta, no exige más de 2 órdenes por semana, no exige 108 columnas, no usa 2024 como periodo de selección.

Pesos congelados durante la máscara. Como las columnas son nuevas, primero se entrena un solo modelo preparatorio, seed 7, encoder DoubleTCN sin actualizar, cabeza lineal, sobre filas con `available_at < 2022-07-01`, early stopping en `[2022-07-01, 2023-01-01)`. Esos pesos no se reentrenan en el pase. La máscara pone el canal en 0 (la media del z) y no recorta el tensor: recortarlo cambiaría la forma del encoder. H2 2023 no entra a este modelo.

Después de cerrar las columnas se sella una receta nueva por familia y se reentrena una vez: pool de train completo (`available_at < 2023-07-01`), early stopping en H2 2023, mismo protocolo (AdamW 3e-4, weight decay 0.004, ReduceLR gracia 20 paciencia 20 factor 0.2 piso 4.8e-7, EarlyStopping paciencia 80 desde la época 100, máximo 2000, batch 256, min_delta 1e-8). Encoder otra vez congelado en el seed 7 y cabeza lineal. Restaurar el campeón conserva el learning rate ya reducido. Este refit todavía no predice 2024.

## Paso 4 — grilla pequeña, ya con columnas selladas

No es un barrido de arquitectura. Por familia, sobre las columnas selladas, solo estas celdas:

- learning rate del encoder: 3e-5 y 3e-6 (una y dos órdenes bajo el 3e-4 de la cabeza; la cabeza sigue en 3e-4);
- descongelar solo `encoder.core[-1]` (el último ResidualTCN del core; ramas, fusión, atención y el core anterior siguen congelados) o descongelar el encoder actor completo;
- cabeza lineal, o cabeza de trayectoria suave: `Linear(features_dim, 3)` y `a + b (h/h_ref) + c (h/h_ref)^2`, con `h_ref` 6 en corto y 48 en largo, sin activación extra.

Son 8 celdas por familia. El early stopping de cada celda es H2 2023. Gana la menor MAE de validación del campeón (media de horizontes, la línea jsonl de la época campeona). Una celda sustituye al refit del paso 3 solo si su MAE de validación baja más que `min_delta`. Si no, el modelo final es el refit del paso 3. Empate interno de la grilla: menos parámetros descongelados, luego menor learning rate del encoder, luego cabeza lineal.

2024 se predice una vez por familia, después de restaurar ese campeón final. No se predice por celda.

## Tablas

Toda MAE dice que es la MAE del incremento estacional, en unidades de precio, el horizonte, el corte, el número de filas, y llama a la base «naive» (persistencia del nivel estacional = incremento cero). La MAE de train del log es el L1 en z por la desviación media del target, no es por horizonte. La MAE de validación por horizonte sale de la época campeona.

## Progreso

- [x] 0. Este plan queda escrito y entra a `origin/master` antes de construir columnas.
- [ ] 1. `decide()` sobre las 111, voto anterior a 2023-07-01, árbol nuevo.
- [ ] 2. Endpoints estacionales causales al lado, sonda en cero hacia el futuro, ventanas nuevas.
- [ ] 3. two_pass_greedy por familia, luego un refit con early stopping en H2 2023.
- [ ] 4. Grilla de 8 celdas por familia sobre las columnas selladas.
- [ ] 5. Una predicción 2024 por familia, después de restaurar el campeón final.
