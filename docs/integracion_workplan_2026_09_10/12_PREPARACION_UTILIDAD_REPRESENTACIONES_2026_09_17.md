# 12 — Preparación del experimento de utilidad de representaciones por variable (sin ejecutar)

Estado: **preparado, no ejecutado, no autorizado a ejecutar**. Pasar la mecánica de D3 (07A/07B) no
significa mejorar predicción, información ni trading; este documento fija cómo se mediría eso
cuando el owner lo ordene, para que la pregunta no se improvise después.

## 1. Qué se pregunta y qué no

* **Pregunta**: para una variable de entrada `v` de un dataset `D`, ¿una representación
  `R(v)` de la lista mecánicamente aceptada aporta, respecto a la **rama cruda** `v` misma, una
  mejora medible en un objetivo declarado de antemano, bajo el mismo presupuesto y el mismo
  ajuste train-only?
* **No** se pregunta: si el modelo final mejora (eso es un experimento posterior con su propio
  diseño), si conviene al trading (utilidad financiera, `SCIENTIFIC_UTILITY` sigue sin reclamar),
  ni qué representación es "la mejor" en general.

## 2. Población y elegibilidad

* Representaciones candidatas: solo las `MECHANICALLY_ACCEPTED` en la matriz **verificada**
  (`MATRIX.verified.json`) de la corrida sucesora, por (operador, familia/dataset) exactamente
  donde fueron aceptadas. Un `INCONCLUSIVE` o `MECHANICALLY_REFUSED` no entra, y no se re-mide
  para que entre.
* Variables: las del banco sintético (por familia) y las de los toys gobernados; para datos
  financieros reales, solo con contrato de disponibilidad declarado y `use_class` compatible
  (nada `OFFLINE_DAY_GRANULAR` como si fuera live).
* Rama cruda obligatoria por variable: `v` sin transformar, con el **mismo** tratamiento de
  faltantes que la representación (ninguna interpolación futura).

## 3. Estimando y objetivo declarados antes de medir

* Objetivo por variable, uno solo, declarado en el freeze: ganancia de información predictiva a
  horizonte `h` sobre un target ya existente del proyecto (p. ej. retorno a `h` pasos), medida
  como diferencia de una pérdida propia (log-loss para clasificación de dirección, MAE para
  regresión) entre un **mismo** modelo pobre de capacidad fija (regresión lineal/logística con
  ventana `w`) alimentado con `R(v)` y alimentado con `v`.
* Estimando: `Δ = L(raw) − L(R)` en **validación** por bloques temporales (walk-forward, sin
  solapamiento, con purga igual al mayor lookback + delay de `R`), con intervalo por bloques;
  `Δ ≤ margen` predeclarado ⇒ "no avanza". El margen se fija en el freeze, no después.
* El test **nunca** se toca hasta la adjudicación final, y solo una vez.

## 4. Presupuestos comparables y ajuste train-only

* El ajuste de `R` (deciles, breakpoints, `zi`, `k/h`) se hace **solo** en el prefijo de
  entrenamiento del bloque (la regla `fit_scope` de 07A), igual que la normalización de la rama
  cruda.
* El modelo pobre recibe el mismo número de columnas (una representación puede emitir varias:
  se fija por adelantado cuáles, o se usa la primera componente declarada) y el mismo `w`.
* Presupuesto por unidad: CPU y memoria iguales para ambas ramas; se registra coste por 1 000
  muestras y se refuse cualquier unidad que exceda su presupuesto en una rama y no en la otra
  (no comparable).
* Semillas fijas; corridas por CPU bajo `crispdm-run`; sin GPU, sin live.

## 5. Gobernanza del experimento

* Freeze sellado (población, variables, representaciones elegibles con sus `spec_sha256`,
  objetivo, margen, bloques, semillas, presupuestos) antes de medir; `design_sha256` propio que
  cita 07B y la matriz verificada por digest.
* Una campaña data-gov por dataset (SYNTHETIC para el banco; DATASETS por recurso para toys y
  financieros), terminales por unidad con identidad de métrica única
  `(metric, split, horizon, unit)`, sobre `campaign_envelope` con `result_class` **DEVELOPMENT**
  (no MECHANICAL: aquí sí hay un estimando), `data_consumed` con `{id, digest,
  eligibility_state}`.
* Salida: tabla por (dataset, variable, representación) con `Δ`, intervalo, coste, cobertura de
  emisión y veredicto "avanza / no avanza / no comparable"; nunca un ranking global.

## 6. Lo que este documento no hace

No ejecuta nada, no reserva máquinas, no cambia operadores ni umbrales, no promete que alguna
representación avance. Si el owner ordena ejecutar, el primer paso es el freeze del §5.
