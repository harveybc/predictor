# 12 — Protocolo del experimento de utilidad de representaciones por variable (diseño y arnés listos para revisión; NO ejecutado)

Estado: **protocolo concreto y arnés probado con verdad fabricada; ninguna medición sobre datos
del proyecto; ninguna reserva puntuada; ninguna autorización de ejecución.** Sucede al borrador
del 2026‑09‑17 (esta misma fecha, primera versión) tras la revisión L5. Arnés:
`tools/df_utility_harness.py`; reglas: `tests/test_df_utility_harness.py` (11).

## 0. Qué se mide y cómo se nombra

Se mide la **pérdida predictiva de un modelo de prueba pequeño y fijo** — `log-loss` con un
logístico para un target de dirección, `MAE` con un ridge para un target de retorno — alimentado
por una rama de features, y la **diferencia pareada** de esa pérdida entre ramas sobre las
**mismas filas emitibles**. Esa diferencia es una diferencia de pérdidas. No es información en
bits, no es información mutua, no es utilidad de trading; el arnés lo escribe en cada resultado
(`note`).

## 1. Población y elegibilidad, exactas por variable y régimen

* Unidad de elegibilidad: **(dataset, variable, régimen de faltantes, representación)**. Una
  representación entra para esa celda solo si su veredicto en la **matriz verificada** del
  replay vigente es `MECHANICALLY_ACCEPTED` en esa celda exacta (no por promedio del dataset ni
  por familia). `INCONCLUSIVE` o `MECHANICALLY_REFUSED` ⇒ la celda no se mide; no se re‑mide
  para que entre. El arnés **rechaza** (`REFUSED`) cualquier representación cuyo registro no
  declare `accepted`.
* Tres poblaciones separadas y con reglas propias:
  1. **Desarrollo**: banco sintético (por familia y régimen) y toys gobernados. Aquí se hace toda
     la selección y todo el ajuste del protocolo.
  2. **Confirmación pública**: datasets públicos con contrato de disponibilidad declarado; solo
     el protocolo ya sellado, sin cambios.
  3. **Revalidación financiera**: recursos financieros solo con `use_class` compatible y
     contrato de disponibilidad; nunca `OFFLINE_DAY_GRANULAR` como si fuera live. Su reserva es
     de uso único (§6).
* Rama cruda obligatoria por celda: `v` sin transformar, con el **mismo** tratamiento de faltantes
  (ninguna imputación; ninguna interpolación futura).

## 2. Ventanas por identidad de observación, decisión, disponibilidad y horizonte

* **Fila `t`** = decisión en el instante de emisión de las salidas disponibles en `t`. Sus
  features son los `w` rezagos de la rama (salida en `t`, `t−1`, …), y la fila es emitible solo
  si **todos** los rezagos están disponibles (`lag_matrix`).
* **Etiqueta de `t`** consume muestras hasta `t + h` (`label`): dirección `sign(x[t+h] − x[t])`
  o retorno `x[t+h] − x[t]`. Filas sin etiqueta no se puntúan.
* **Purga** entre bloque de entrenamiento y bloque de validación: `h + reach_right(R) + w`
  (horizonte de la etiqueta **más** el alcance declarado de la representación **más** la
  ventana). La purga no depende solo de lookback+delay: cubre las etiquetas futuras del
  entrenamiento (`test_the_purge_covers_the_label_horizon_the_reach_and_the_window`).
* Normalización (`_standardise` con estadísticos del bloque de entrenamiento), selección y
  todo ajuste ocurren **dentro del train** del bloque.

## 3. Comparación pareada y lo que se reporta aparte

* Ambas ramas se puntúan sobre la **intersección** de filas emitibles con etiqueta
  (`rows_paired`); nunca se gana eliminando casos difíciles: la cobertura por rama, las filas
  pareadas y los faltantes de entrada se reportan aparte (`coverage`).
* **Presupuesto agotado** (`BUDGET_EXHAUSTED`) y **filas insuficientes** (`INSUFFICIENT_ROWS`)
  son desenlaces registrados por contraste, no descartes invisibles.
* Coste (CPU por contraste) e intentos fallidos se registran con el desenlace.

## 4. Ramas y control de capacidad

* `raw`: `w` rezagos de `v`. `transformed`: `w` rezagos de la salida declarada de `R(v)`
  (**un canal de salida, el que el operador declara como principal en su spec**; no "la primera
  componente" por conveniencia — si un operador declara varios canales, cada canal es su propia
  representación y su propio contraste). `augmented`: `raw ⊕ transformed` (2·`w` columnas), un
  contraste **separado** que responde "¿añadir `R` a lo crudo ayuda?", no "¿`R` sustituye a lo
  crudo?".
* Control de capacidad predeclarado: `raw` y `transformed` reciben exactamente `w` columnas y el
  mismo modelo con la misma regularización; `augmented` se declara con su capacidad doble y se
  interpreta solo contra `raw`.
* Qué recibe cada modelo y qué identifica cada contraste:
  * `raw` vs `transformed`: si la representación, con la misma capacidad, reduce la pérdida del
    modelo pobre respecto a los rezagos crudos.
  * `raw` vs `augmented`: si la representación aporta algo que los rezagos crudos no tienen.

## 5. Estadística declarada antes de medir

* **Unidad estadística**: el bloque de validación walk‑forward (no solapado, con purga). `Δ_k =
  L_raw,k − L_R,k`. Estimando: media de `Δ_k` sobre bloques con intervalo t.
* **Familia de comparaciones**: el número de contrastes (celda × representación × rama)
  predeclarado en el freeze (`comparisons`); corrección de Bonferroni sobre `alpha`.
* **Decisión** por contraste: `ADVANCES` sii el límite inferior del intervalo corregido supera
  el `margin` predeclarado; si no, `DOES_NOT_ADVANCE`. Abstenciones: `INSUFFICIENT_ROWS`
  (menos de `n_blocks` bloques con ≥ `min_rows_per_block` filas en train y validación),
  `BUDGET_EXHAUSTED`, `NOT_COMPARABLE` (cobertura por rama incompatible), `REFUSED`.
* **Dependencia entre variables/series**: variables del mismo dataset comparten bloques y
  semilla; la corrección múltiple cubre la familia completa; nada se agrega entre datasets. Una
  variable de una unidad multivariable es su propia celda.
* Target, horizonte, modelo, `w`, `n_blocks`, margen, semillas y `comparisons` viven en el
  `Protocol` sellado (`protocol_sha256`) antes de cualquier puntuación.

## 6. Selección en desarrollo, reserva de uso único

* Toda selección (qué representaciones, qué `w`, qué margen) ocurre en **desarrollo**.
* La **reserva** (confirmación pública y revalidación financiera) se adjudica **una sola vez**
  por protocolo: `adjudicate_holdout` escribe un marcador write‑once antes de puntuar y rechaza
  un segundo uso ("a second look").

## 7. Gobernanza del experimento (cuando se ordene)

Freeze sellado (población por celda, representaciones elegibles con su `spec_sha256` y la
matriz verificada citada por digest, `Protocol`, familia de comparaciones) antes de medir; una
campaña data‑gov por dataset (SYNTHETIC para el banco, DATASETS por recurso), terminales por
unidad con identidad de métrica única, sobre `campaign_envelope` con `result_class`
**DEVELOPMENT** y `data_consumed` con `{id, digest, eligibility_state}`; salida por contraste
(`Δ`, intervalo, `outcome`, cobertura, coste), nunca un ranking global.

## 8. Lo que el arnés prueba hoy (verdad fabricada, sin datos del proyecto)

Control positivo (el target depende del promedio móvil ⇒ `transformed` avanza); control negativo
(target ruido ⇒ no avanza); representación no aceptada ⇒ rechazada antes de puntuar; filas
pareadas; purga = `h + reach + w` y bloques que la respetan; presupuesto agotado registrado;
filas insuficientes; rama aumentada como contraste aparte; reserva de uso único; `log-loss`
con logístico para dirección.

## 9. Lo que este documento no hace

No ejecuta, no reserva máquinas, no cambia operadores ni umbrales, no promete que alguna
representación avance, no abre el experimento.
