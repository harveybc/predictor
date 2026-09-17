# 07B — Enmienda sucesora de D3: sonda desde el ajuste, gemelos insuficientes, cobertura (K2–K3)

Estado: **sellada antes de medir**. Datos sellados en `tools/df_d3_design.py`
(`D3_PROBE_AMENDMENT_V1`, `design_sha256` propio; `D3_DESIGN_CURRENT` apunta a ella). Sucede a
la enmienda temporal `07A` (`d3_temporal_amendment.v1`), a la que cita por digest, y esta a su
vez al diseño 07 original, cuyos bytes siguen intactos. Contrato de operador: `d3_operator_spec.v3`.
Todo sigue `NON_GOVERNING`: nada aquí es utilidad científica ni selección.

## Por qué

La corrida `d3mech-v1` rechazó el cuantizador de deciles en 63 variables y SAX en 6 por la prueba
de respuesta. La sonda de la batería v2 era un escalón fijo (+25) sobre ruido N(0,1) **sin relación
con el dominio ajustado** del operador: sobre un train en miles no cruzaba ningún borde en el
instante de impacto y cruzaba uno una muestra después por azar del ruido. La revisión (Musashi,
2026-09-16) fijó dos límites a la corrección: una representación con ventana trailing no es
necesariamente sin memoria — que la perturbación no cruce un umbral en el primer punto no
identifica por sí solo un retraso — y la regla no puede convertir un retraso real en `UNIDENTIFIED`.

## §1 Construcción de la sonda: solo ajuste de entrenamiento y resolución declarada

| elemento | regla |
|---|---|
| línea base `b` | cuantil 0.10 de los valores finitos del prefijo de entrenamiento |
| escala `S` | cuantil 0.90 − `b`; `S = 0` (train constante) ⇒ `UNIDENTIFIED` |
| rama quieta | `b + σ·N(0,1)`, `σ = 0.01·S`, semilla fija, **el mismo ruido** en ambas ramas |
| amplitud | `operator.probe_resolution(state, baseline=b, scale=S, sigma=σ)`: la menor excitación que el operador **garantiza** que mueve la salida en la muestra de impacto, calculada **solo desde su ajuste**; o `UNIDENTIFIED` con razón |
| forma | `impulse`: `x[p] += A`; `step`/`level_shift`: `x[p:] += A`; `variance_shift`: `x[i] = b + (x[i]−b)·g` desde `p`, con `g` declarada |
| nunca | validación ni test; amplitudes buscadas hasta obtener un pase; umbrales o parámetros del operador cambiados para salvarlo |

Resoluciones declaradas (todas desde el ajuste; ninguna del fixture):

* `uniform_decile_quantizer`: primer borde por encima de la banda de ruido `b + 3σ`, más `3σ`.
  La base de un cuantizador ajustado al mismo train **es** un borde: sin la banda, el ruido ya lo
  cruza y la medida sería un artefacto. Sin borde por encima ⇒ `UNIDENTIFIED` (saturación).
* `sax_paa_trailing`: la ventana de impacto tiene `s−1` muestras quietas y una excitada; su PAA sube
  `A/s`; primer breakpoint por encima de la banda `z` de la ventana quieta (ruido `σ/√s`).
* `cusum_causal`: alcanzar `mean + k` desde `b` (`S+` se mueve con certeza), más margen.
* `variance_regime_trailing`: ganancia declarada `8`.
* lineales/sin cota (`delta_run_length`, `stft`, `wavelet`, `butterworth`, `mad_extremes`): `S`.

Casos definidos y probados: train constante; rangos extremos (`1e−9`, `1e9`, desplazamientos de
`−2·10⁶`); saturación; cuantizador de dominio conocido a cinco escalas; SAX a dos; STFT con inicio
declarado 1; un operador deliberadamente retardado; el control desplazado de J2.

## §2 Tres hechos, registrados aparte

1. **Excitación identificable** — declaración del operador (`identifiable`, con la excitación y su
   razón). `False` es `UNIDENTIFIED`.
2. **Primer cambio observado** — la primera salida disponible tras el impacto cuyo valor difiere
   entre rama quieta y rama excitada (`first_change_observed`).
3. **Coincidencia con la respuesta declarada** — `first_change_observed == expected_onset_samples`
   (`matches_declared`).

Política:

* declarada identificable y **nada se mueve** ⇒ `FAILED` (la declaración queda contradicha; no
  hay abstención);
* primer cambio ≠ declarado ⇒ `FAILED` (un retraso real incompatible con el contrato sigue
  fallando: el control desplazado de J2 mide 2 y falla; un cuantizador retardado mide 1 y falla);
* `UNIDENTIFIED` solo por declaración ⇒ indecidido: el veredicto es `INCONCLUSIVE`, **nunca**
  `MECHANICALLY_ACCEPTED`. La abstención universal no existe: cuesta el veredicto.

## §3 Gemelos: detección demostrada, insuficiencia propagada

* Una infracción **demostrada** del gemelo (prefijo o perturbación futura `FAILED`) cuenta como
  detección ⇒ `non_causal_twin: PASSED`.
* Sin ninguna comparación observable ⇒ `INSUFFICIENT_TEST` (indecidido ⇒ `INCONCLUSIVE`). La
  batería v2 lo marcaba fallido con la explicación "el gemelo pasó las pruebas de causalidad":
  ausencia de evidencia no es evidencia.
* Comparado muchas veces y nunca fallido ⇒ `FAILED` (o no es el gemelo que dice ser, o las pruebas
  no miden causalidad).
* Para cada control se registran `twin_emissions` y `twin_comparisons`.
* Se conserva el control centrado con datos completos que la batería **debe** detectar.

## §4 Cobertura de emisión, aparte de causalidad y de inaplicabilidad

El informe lleva `coverage = {n, emitted, inputs_available}`. Bajo MCAR 10 % un soporte de 50 emite
`≈0.9⁵⁰` de las muestras: eso es cobertura, no causalidad ni utilidad. No se interpola el futuro y
el soporte 50 no cambia. Un NaN aislado apaga exactamente un soporte; los bloques, su longitud más el
soporte; en la frontera de soporte se emite exactamente una salida.

## Sin cambio

Las doce pruebas obligatorias; los nueve operadores y sus parámetros por defecto; menú de cortes,
longitudes y regímenes de faltantes; clasificación `NON_GOVERNING`. Las filas de una corrida bajo
esta enmienda llevan su `design_sha256`; las de `d3mech-v1` conservan el de 07A.
