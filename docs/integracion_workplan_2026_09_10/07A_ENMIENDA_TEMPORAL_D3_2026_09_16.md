# D3 — enmienda temporal superseding al diseño 07 (J1)

Fecha: 2026-09-16. Estado: **ENMIENDA SELLADA ANTES DE CANDIDATOS**. Autorizada explícitamente
por el revisor en `../handoffs/MUSASHI_R6_ACCEPTANCE_AND_D3_J1_J3_2026_09_16.md` §J1.

Diseño enmendado: `07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md`,
**bytes originales conservados sin edición**, sha256
`45959063e01895c6f9669a21f2784c38ffb60b6f7b4198129ca3f0fea22ffa86` (commit `3e2a18a`).
Este documento **prevalece** donde se cruza con el §1 y el §3 del original; no cambia hipótesis
científicas, márgenes de utilidad, particiones ni resultados aceptados de D2. La versión
legible por máquina, sellada por digest, es `tools/df_d3_design.py` → `D3_AMENDMENT_V1`.

## 0. Qué se corrige y por qué

Cuatro defectos reproducidos por el revisor sobre `df_d3_acceptance.py` antes de implementar
operador alguno:

1. `check_availability` sumaba `delay + lookback` y aprobaba con retardo 0 y lookback 7 frente a
   un recurso con lag 5: el pasado no hace que una entrada tardía esté disponible antes. Y no
   miraba marcas de emisión reales.
2. Rechazaba `completion_lag_max='0s'` por "no ser un conteo de muestras": el contrato del lago
   habla en duraciones; `int(lag)` no es el parser del productor y trunca fracciones.
3. `check_prefix` aceptaba el control no causal centrado: excluir las últimas `lookback` muestras
   esconde exactamente la frontera donde aparece la fuga.
4. Dos operadores del diseño no tienen gemelo no causal; el arnés devolvía `undecided`, lo que
   hace su aceptación inconclusa por construcción y no por medida.

## 1. Cuatro instantes distintos por salida

Para cada salida en el índice de evento `t` se declaran y miden **por separado**:

| instante | símbolo | origen |
|---|---|---|
| índice/tiempo del evento | `t`, `timestamp[t]` | contrato del recurso (`timestamp_meaning`) |
| disponibilidad de la entrada | `available_at[t]` | contrato del recurso: `label + completion_lag_max`; en el snapshot, `available_at` |
| tiempo de emisión de la salida | `emitted_at[t]` | **≥ max(available_at[i])** sobre las entradas `i` que la salida realmente consume, más el retardo declarado de emisión |
| retardo de respuesta de la señal | `response_delay` | medido con la sonda predeclarada del operador; no es una disponibilidad |

Reglas:

* `lookback`, `warm_up` y retardo de grupo/respuesta **no son medidas intercambiables** de
  disponibilidad. Ninguno de los tres puede sumarse para "hacer disponible" una entrada.
* Una salida **no puede emitirse antes** de la última disponibilidad de las entradas que
  consume. Ésa es la única regla de disponibilidad, y es por salida.
* `UNKNOWN` permanece `UNKNOWN`: un recurso que no declara cuándo está completo no certifica
  ningún tiempo de emisión. El resultado es `undecided` con motivo, nunca aceptado.

## 2. Duraciones, UTC y muestras

* La disponibilidad del recurso se lee con la **semántica del productor** (`pd.Timedelta`, la
  misma que usa `data-gov` en `lake_plugins/files_lake.py`): `'0s'`, `'4h'`, `'1h'` son válidos;
  `UNKNOWN`/`None` no son cero.
* Una duración se convierte a muestras **solo** con un contrato de muestreo declarado
  (`frequency` del recurso, o `frequency_nominal_seconds`). La conversión es exacta: si la
  duración no es múltiplo entero del periodo, se **rehúsa** (`FRACTIONAL_SAMPLE_OFFSET`); nunca
  se trunca ni se redondea.
* Sin contrato de muestreo, o con cadencia `IRREGULAR`/`UNKNOWN`, no hay conversión a muestras:
  la disponibilidad se evalúa en tiempo (`emitted_at` vs `available_at` en segundos UTC) o
  queda `undecided`. Los huecos **no** se tratan como observaciones equiespaciadas.
* Se prueban: duración fraccionaria, llegadas tardías (`available_at > timestamp`), marcas de
  tiempo ausentes y un contrato real `'0s'`.

## 3. Invariancia de prefijo, sin exención

* La prueba 1 compara **toda** salida declarada disponible en el corte: su valor, su máscara de
  disponibilidad y su `emitted_at`. **No hay exención por lookback pasado.** Ésta es la regla que
  ya aplicaba la batería D2 (`df_causal_battery.prefix_all_t`) y que el §3.1 original relajó.
* Una representación retrospectiva con retardo declarado se compara **solo después** de su
  tiempo de emisión declarado, sin retrodatarla.
* La población comparada debe ser **no vacía**; si el corte no deja salidas disponibles, el
  resultado es `INSUFFICIENT_TEST`, no `PASS`.

## 4. Cortes, longitudes, sondas y regímenes

* Los cortes son **predeterminados**: `2^j−1, 2^j, 2^j+1`, `warm_up−1, warm_up, warm_up+1`,
  múltiplos de ventana/periodo, `0`, `n−2`, `n−1` y cortes aleatorios con semilla declarada;
  se repiten en ≥ 2 longitudes y con y sin faltantes (`mcar`, bloques).
* Perturbación del futuro: el menú D2 (`zeros`, `large_constant`, `other_seed_noise`,
  `reversed`, `nan_blocks`, `impulse_at_t_plus_1`, `step`, `chirp`, `regime_change`).
* **Estado ajustado y reinicio se prueban por separado.** El ajuste congelado sobre el prefijo
  de train **no** incluye el futuro evaluado; cambiar el futuro de los datos de ajuste no
  cambia las salidas pasadas.
* Cada rama de prueba **instancia o recarga su propio estado**: una `transform` no puede
  contaminar la comparación siguiente.

## 5. Gemelos no causales

* Cada operador declara `non_causal_twin`: la especificación de su gemelo, **o**
  `NOT_APPLICABLE` con la razón de diseño. La ausencia **no** es aprobación: un operador sin
  gemelo y sin razón es `MECHANICALLY_REFUSED`.
* Para los códecs puntuales sin estado (cuantizador uniforme por deciles, delta+run-length) el
  gemelo se marca `NOT_APPLICABLE` con razón "no hay ventana que centrar"; sus pruebas
  temporales independientes (1, 2, 4, 5, 7) siguen siendo obligatorias.

## 6. Retardo: inicio de impulso ≠ retardo de grupo

* La primera salida que cambia ante un impulso mide el **inicio de respuesta**, no el retardo de
  grupo de un filtro ni la latencia de un detector no lineal.
* Cada operador declara su **sonda** (`impulse`, `step`, `level_shift`, `variance_shift`) y el
  inicio esperado. Un detector o cuantizador cuya respuesta un impulso no identifica usa su
  sonda predeclarada o deja el diagnóstico `UNIDENTIFIED` con motivo; **nunca un cero
  fabricado**.

## 7. Soporte de filtros y estado recursivo

* Wavelets: el soporte real se **deriva** de la biblioteca (`pywt.Wavelet(w).dec_len`), del modo
  de borde y del nivel: `soporte(L) = (dec_len−1)·(2^L−1)+1`. `2^L` solo describe a Haar.
* Filtros recursivos (Butterworth): el estado se representa como **dependencia recursiva**
  (`scipy.signal.lfilter` con `zi`), no como memoria finita de orden `p`.
* Nunca se reutiliza la fuga wavelet/STL antigua como una característica "mejor".

## 8. Estado de preparación, en tres palabras distintas

`ready_to_measure` significa **presencia de infraestructura** (R2, R6, N3). No significa
aceptación de la batería temporal, y ninguna de las dos significa utilidad científica. Los tres
estados se publican separados, en prosa y en JSON.

## 9. Lo que no cambia

Los nueve operadores del §2 original, sus controles donde existen, el banco (sintético D2 +
tres recursos toy contratados), la clasificación `NON_GOVERNING` de la mecánica, el techo de
2 GiB por proceso, el piloto de coste de un hilo y la abstención como salida válida.
