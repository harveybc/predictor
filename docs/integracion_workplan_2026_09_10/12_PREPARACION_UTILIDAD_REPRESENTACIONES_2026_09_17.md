# 12 — Protocolo del experimento de utilidad de representaciones por variable (v3 tras N1–N4; arnés causal, calibración verificable, gobernanza antes del trabajo; piloto descriptivo autorizado)

Estado: **protocolo v2 (`df_utility_protocol.v2`) y arnés causal por construcción, probado con
verdad fabricada y ensayado de punta a punta bajo data‑gov (`utilreh-v4`); ninguna medición sobre
datos del proyecto; ninguna reserva puntuada; ninguna autorización de ejecución.** Sucede a la v1
del 2026‑09‑17 tras la revisión M2–M4 (fuga de etiqueta futura, tiempo de emisión ignorado,
intervalo casero, presupuesto autodeclarado). Arnés: `tools/df_utility_harness.py`; entrada
gobernada: `tools/df_utility_run.py`; reglas: `tests/test_df_utility_harness.py` (20, R1–R10).

## 0a. Lo que cambió en v3 (N1–N4) y por qué

* **El score es el archivo verificado**, nunca el resumen del proceso: el padre relee el archivo
  que el hijo nombró y el runner re‑hasheó, coteja bytes, esquema, identidad de contraste y de
  protocolo y valores finitos; ausente/alterado/discordante ⇒ `SCORE_UNVERIFIED`, sin ceros
  fabricados. Un intento completado se reanuda desde su `outcome.json`, nunca se re‑ejecuta.
  (`utilreh-v4` reportó `metrics: []` por leer el resumen; sus tres mediciones se recuperaron
  desde los archivos como terminales de **generación 2** ligados a los originales — `RECOVERY.json`.)
* **Registro de calibración completo y verificable** (`df_utility_calibration.v1`): generador
  (**nulo de no efecto**; `ar1_null` es diagnóstico estructurado y se rechaza como nulo), plan
  sellado (`calibration_plan` en el protocolo: generador, `n_sims`, `n`, confianza de la cota),
  intentos/válidas/fallidas/avances, tasa, **cota superior Clopper–Pearson** a la confianza
  predeclarada, `alpha_adjusted`, longitud, operador (kind/spec/params), identidad base del
  protocolo, familia, margen/bloques/ventana, cada simulación con semilla y desenlace bajo digest,
  digest del arnés, coste. El consumidor verifica identidad y alcance (operador, protocolo,
  longitud, familia, multiplicidad) y decide solo si `upper_bound ≤ alpha_adjusted`; si no, el
  resultado es descriptivo (`INCONCLUSIVE_UNCALIBRATED`). Cero simulaciones, tasas no finitas,
  generador desconocido, denominador incompleto o digest roto ⇒ el protocolo no se sella.
  `sims_required_for_zero(alpha, c)` da el tamaño mínimo (0.0125 a 0.95 ⇒ 239).
* **Nulo con dependencia relevante**: `ar1_features_independent_target` (features AR(1),
  etiqueta de una serie independiente) además del blanco intercambiable; controles positivos
  aparte.
* **Gobernanza antes del trabajo**: campaña de calibración registrada antes de cualquier hijo;
  `before_run` antes de cada hijo; calibración y mecánica preparatoria en hijos aislados con los
  mismos techos y con terminales propios (coste e instantes reales); registro persistido
  (`CAMPAIGNS.json`); una raíz congelada bajo otro `code_identity` no se reanuda.
* **Piloto descriptivo** (N4): hasta tres unidades sintéticas de desarrollo ya expuestas, con
  elegibilidad desde las celdas verificadas de `d3mech-v3`, protocolo sellado, sin reserva; sirve
  para medir coste y flujo, no para elegir representaciones.

## 0bis. Lo que cambió en v2 y por qué

* **La representación la construye el operador real** dentro del arnés, por bloque: `fit` solo con
  el prefijo de entrenamiento de ese bloque, `transform` con estado propio, salida validada por el
  contrato (`emitted_at`, `available`). No se aceptan arrays precalculados ni un booleano `accepted`.
* **Cada feature se consume solo si fue emitida en o antes de la decisión de su fila**; una salida
  tardía se alinea a la fila posterior donde ya está disponible — no se cura con purga.
* **Comprobación de prefijo**: se re‑transforma la serie cortada en filas de decisión muestreadas y
  se exige la misma salida; una representación cuyo prefijo cambia con la cola se rechaza como no
  causal *antes de puntuar* (el control de etiqueta futura del revisor cae aquí aun con un registro
  de elegibilidad falsificado).
* **Elegibilidad por registro de celdas verificado** `(unidad, variable, operador, spec_sha256)`
  exportado por `verify()` (`MATRIX.verified.*.cells.json`); un control fuera del registro no se
  puntúa, por espectacular que sea su pérdida.
* **Identidad**: ids estrictamente crecientes, tiempos no decrecientes, disponibilidad ≥ marca;
  discordancias rechazan; los huecos se conservan.
* **Inferencia**: `scipy.stats.t.ppf` con `alpha/|familia|`; bloques *todo‑o‑insuficiente*;
  `ADVANCES` solo bajo un **registro de calibración** sellado en el protocolo (tasa de falsos
  avances medida bajo el nulo intercambiable de incrementos independientes, semilla diagnóstica);
  sin él, o con tasa > `alpha_adjusted`, el resultado es descriptivo (`INCONCLUSIVE_UNCALIBRATED`).
  Diagnóstico registrado (`utility_rehearsal_v3/CALIBRATION_DIAGNOSTIC.v2.json`, 200 sims):
  nulo blanco 0/200 (margen 0 y 0.05); AR(1) φ=0.6 33 % con margen 0 y 1 % con 0.05 — ese
  generador tiene estructura real que `delta_run_length` capta: es un control positivo, no un nulo.
* **Protocolo validado** antes de puntuar: dominios finitos, par target/modelo, `n_blocks ≥ 3`,
  ramas declaradas (una rama no declarada se rechaza), familia sellada de contrastes
  (`comparisons = |familia|`), política de bloques, esquema de calibración.
* **Presupuestos observados**: cada contraste corre en un hijo aislado bajo `df_isolated_runner`
  (techos de muro/CPU/memoria impuestos durante el trabajo); un modelo lento controlado termina
  `RESOURCE_EXCEEDED` con coste medido y **sin puntuación parcial**; nada se promueve.
* **Reserva ligada a identidad**: una adjudicación por `(reserva: campaign_sha256 o digests de
  datasets, protocolo)`, marcador write‑once en el directorio de estado, nunca un directorio nuevo.

## 0ter. Ensayo gobernado (fabricado, `NON_GOVERNING`, DEVELOPMENT/REHEARSAL)

`df_utility_run.py`: la batería D3 real sobre la serie fabricada produce la elegibilidad; calibración
(200 sims, nulo blanco, 0 falsos avances) sellada; freeze write‑once (protocolo, familia con un
control lento declarado, digest de datos y de elegibilidad, presupuestos); un hijo aislado por
contraste; campaña data‑gov SYNTHETIC `utilreh-v4-utility-rehearsal`, un terminal por contraste
(deltas como métricas; `RESOURCE_EXCEEDED` con coste para el control lento), conciliada
(`missing_units: []`); sobre DEVELOPMENT `da341cc5…`. Tres contrastes `DOES_NOT_ADVANCE`
(descriptivo de esa serie; no es una afirmación sobre representaciones).


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
