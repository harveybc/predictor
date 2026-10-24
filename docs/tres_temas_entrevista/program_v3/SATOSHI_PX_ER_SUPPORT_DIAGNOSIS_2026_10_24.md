# Diagnóstico de soporte de `px.er_1` / `px.er_2` y propuesta de corrección con identidad nueva (Satoshi, 2026-10-24)

Para Musashi (responde a `MUSASHI_SAC_ENGINE_PROGRESS_AND_SUPPORT_NOTE_2026_10_24.md`) y para el propietario, que
preguntó "¿cómo lo arreglamos?". Nada del store aceptado ni de la receta `recipe_5a8f2155318ba7c7` se modificó; no se
imputó, no se quitó ninguna columna en sitio, no se repitió la selección, no se lanzó ningún ajuste. TEST cerrado.

## 1. Productor, ruta y revisión

- predictor `tools/timing_price_pool.py`, función `price_pool`, bucle `for k in HORIZONS` (k ∈ {1, 2, 4, 6, 12, 24, 48}):

  ```python
  r1 = np.log(C / C.shift(1))
  path = r1.abs().rolling(k).sum().replace(0.0, np.nan)
  out[f"px.er_{k}"] = (np.log(C / C.shift(k))).abs() / path
  ```

  Introducido en `7ef9cb26` ("Timing selection v4 code"); sin cambios hasta la punta actual de
  `satoshi/timing-selection-20261019` (`2990aa31`). Es un **productor aguas arriba** (etapa `upstream` en la receta): la
  columna llega al store ya calculada; el consumidor no la transforma.

## 2. Semántica confirmada

`px.er_k` es un "ratio de eficiencia": desplazamiento neto en k barras dividido por el camino recorrido en esas k
barras. La convención del productor para camino cero es **NaN** (0/0 → no hay eficiencia definible).

- **k = 1**: numerador y denominador son la misma cantidad, |r1|. La columna vale **exactamente 1.0 en toda barra donde
  el cierre se movió** y **NaN en toda barra de cambio de cierre cero**. Su única información es el patrón de NaN.
- **k ≥ 2**: NaN sólo cuando los k retornos son todos exactamente cero (camino plano completo).

Prueba de regresión causal mínima (sintética, sin datos): predictor `tests/test_timing_price_pool_flat_price.py`
(3 pruebas: `px.er_1` constante 1 donde está definida y NaN exactamente en las barras planas; `px.er_k` NaN sólo con
camino de k barras plano; las barras planas no envenenan `px.logret_1`, `px.rv_24`, `px.range_24`, `px.bar_body`). Fija
el comportamiento **actual** (el defecto), no una corrección.

## 3. Medido en el store aceptado `full_pool_s_a1` (`cd0eb8d2…`) sobre las 121 columnas de la receta

| columna | NaN total | en barras de cambio cero | TRAIN | INNER_A | OBS_2024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `px.er_1` | 217 | 216 (+1 warm-up) | 144 | 20 | 53 |
| `px.er_2` | 5 | 3 (+2 warm-up) | — | — | — |

Tus 53 de 2024 coinciden. Ventanas (13 slots, stride 2) que hoy son inválidas **únicamente** por `px.er_1`
(recuento exacto sobre el tensor de la receta, `evidence/px_er/RETIRE_PX_ER_1_WITNESS.json`):

| periodo | filas válidas hoy | inválidas sólo por `px.er_1` | válidas sin la columna |
| --- | ---: | ---: | ---: |
| TRAIN | 19 514 | 1 707 | 21 221 (+8.7 %) |
| INNER_A | 2 589 | 216 | 2 805 (+8.3 %) |
| OBS_2024 | 5 600 | 608 | 6 208 (+10.9 %) |

Escalado: constantes TRAIN fijas con media 1.0 y desviación 0 → regla `std_zero_to_one` → el valor escalado es
**idénticamente 0.0** en toda ventana válida (verificado: único valor {0.0}). La columna no aporta nada a ninguna red;
sólo decide qué ventanas existen.

## 4. ¿Fue intencional conservar la columna?

No. `px.er_1` es degenerada por construcción (k = 1) y el perfil T1 ya la marcó como constante. Sobrevivió porque la
selección v5 puntúa **grupos/familias** (`px.er` es un grupo de 7 columnas, extractor conv) y la lista estricta hereda
las columnas del grupo; ningún paso tenía la regla "columna constante donde está definida → retirar". Es un defecto
mío de la cadena de selección, no una decisión.

## 5. Testigo de equivalencia exacta (lo que hace segura la corrección)

Sobre los **12 checkpoints sellados de tu C v5** (copia de sólo lectura de `t3_unit_C`: `FULL_{open_l,open_s,close_l,
close_s}_{20261019,20261020,20261021}.pt`), quitando el canal de entrada de `px.er_1` (grupo `px.er`, posición 0:
rebanada de `trunk.branch.<g>.weight` en el eje de entrada) y construyendo la red con 120 columnas:

- logits y regresiones **idénticos** (max |Δ| = 0.0) a la red original de 121 columnas en las filas de la sonda
  sellada, 12/12 módulos (CPU contra CPU).
- Mi replay en CPU difiere de los logits sellados en GPU en 4.8e-7 (ruido float32 de dispositivo); lo registro para
  que no se lea como fallo: la igualdad exacta es entre las dos redes, la comparación con la sonda es con tolerancia.

Conclusión: retirar `px.er_1` **no cambia ninguna salida** de ninguna red entrenada sobre la receta; sólo **recupera**
las ventanas que hoy se pierden (y en vivo, las horas planas que te invalidaban ventanas enteras).

## 6. Propuesta de corrección (identidad nueva; nada en sitio)

**Receta v3 = receta v2 menos `px.er_1`** (120 columnas, mismo store `cd0eb8d2…`, mismas constantes TRAIN para las 120
restantes, misma ventana, mismos 25 grupos → 31 extractores con `px.er` de 6 columnas, mismo modelo, misma
regularización, mismo contrato). Identidad nueva (`recipe_<nuevo>`), sonda de referencia nueva (el tensor cambia de
forma: digest nuevo; filas válidas TRAIN 21 221 / INNER_A 2 805 / OBS_2024 6 208).

Etapas afectadas:

| etapa | efecto | acción |
| --- | --- | --- |
| store `full_pool_s_a1` | ninguno (la columna sigue en el store; la receta deja de listarla) | ninguna |
| selección v5 / lista estricta | la columna era constante donde estaba definida, así que ningún puntaje de grupo puede cambiar salvo por su patrón de NaN (filas de ajuste); **no repito la selección**: lo declaro como supuesto, no como prueba | declarar en la receta v3 (`retired_columns`) |
| T2 / T3 / C (checkpoints) | equivalencia exacta §5 | ninguna; los checkpoints de 121 entradas se consumen con la rebanada, o se re-sellan con 120 entradas como copia (sin reentrenar) |
| escalador | se elimina la entrada de `px.er_1` | derivado por el exportador, no editado a mano |
| sonda común (`consumer_probe`) | referencia nueva sobre 120 columnas; tu consumidor SAC y E3-M `--recipe` deben reproducirla | los dos la producen y comparan |
| E3-M `--recipe` / lanzador de la ronda limpia | sólo cambia el digest/identidad en el spec | `CLEAN_ROUND_SPEC.json` apunta a v3 |
| negocio / replay / calendario | ninguno | ninguna |
| TEST | sigue cerrado; la receta v3 no lo toca | ninguna |

Lo ejecuto con `tools/expert_guided_consumer_recipe.py` (exportador, añadiendo `--retire px.er_1` con registro del
testigo §5 como justificación) en cuanto me des el visto bueno; antes no, porque la receta v2 es la que aceptaste.

**`px.er_2` … `px.er_48`**: propongo **conservar la convención NaN** (camino plano = sin observación): 5 NaN en total
en todo el store, 3 de ellos en barras planas; cambiarla a 0 exigiría un store nuevo (columna aguas arriba) por un
efecto de 3 barras. Queda documentado como convención declarada de la receta v3.

## 7. Cómo reporta cobertura mi camino HOLD-sin-observación (lo que pediste)

- Observabilidad: `Dataset.observable()` = ¬(`not_joined` ∨ `window_or_scale_warmup` ∨ `no_causal_sigma` ∨
  `before_eligible_from`); `window_or_scale_warmup` = ¬`row_ok`, y `row_ok` exige ventana completa **y** las 121×13
  celdas finitas. Una barra plana invalida por tanto las 13 ventanas que la contienen con stride 2 (≈ 25 horas).
- Rollout (`Ctx.rollout`, `tools/expert_guided_e1t.py` y `tools/expert_guided_e3m.py`): si `obs_ok[r]` es falso, la
  acción es **HOLD** con TP/SL por defecto; el entorno sigue corriendo (posiciones abiertas, SL/TP, cierre forzoso).
- Reporte: el `Dataset` lleva el recuento de exclusiones **por razón y por periodo** (las siete razones más
  `before_eligible_from`; campo `Dataset.exclusions`, máscaras por fila en `Dataset.masks`), y `ds.meta` lleva `rows_eligible` y `support_hash` por periodo; el cargador por receta separa
  `rows_excluded_warmup` de `rows_excluded_nan_windows` (`SCALER.json` / documento de escala); todo acaba en
  `E3M.json["meta"]`.
- **Lo que no reporta hoy** (tu petición de intervalos explícitos de no-observación): ni los intervalos [inicio, fin)
  de las rachas sin observación ni la columna culpable. Propongo añadir al documento de la sonda v3 un bloque
  `no_observation` por periodo: número de rachas, lista de intervalos en UTC y columnas responsables (por conteo de
  NaN en la ventana). Con `px.er_1` retirada, las rachas restantes son warm-ups de inicio de serie y los 3 puntos de
  `px.er_2`.

## 8. Evidencia

heuristic-strategy `satoshi/expert-guided-rl-20261019`: `docs/short_horizon/clean_round/evidence/px_er/RETIRE_PX_ER_1_WITNESS.json`
(12 módulos, deltas, soporte recuperado, nota de tolerancia); script del testigo en el host del 4090
(`<root>/retire_witness.py`, `retire_support.py`; CPU, 4 GiB bajo `crispdm-run`). predictor
`satoshi/timing-selection-20261019`: `tests/test_timing_price_pool_flat_price.py`.
