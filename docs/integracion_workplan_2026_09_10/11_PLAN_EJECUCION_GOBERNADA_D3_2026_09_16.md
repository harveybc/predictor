# D3 — plan de ejecución gobernada (preparación, sin ejecutar)

Fecha: 2026-09-16. Estado: **SUPERSEDIDO EN PARTE POR J1–J3** — la enmienda temporal
(`07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md`) prevalece sobre el §1 y el §3 del diseño 07 y sobre
lo que este plan decía de la batería v1; los nueve operadores están implementados y la
mecánica gobernada corre como `d3mech-v1` (ver el retorno J1–J3). Sigue sin haber selección por
utilidad ni entrenamiento.

Orden: bloque 3 de `../handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`
— *"Inventory the next unfinished preprocessing/feature-engineering step from committed
orders and identify its actual prerequisites. Prepare its data contracts, tests and governed
execution plan."*

Diseño vigente y **sellado**: `07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md`.
Este documento lo transcribe donde hace falta y **no lo modifica**.

## 1. Cuál es el siguiente paso, y por qué es este

La secuencia vinculante de `06_ESTADO_REAL_PREPROCESAMIENTO_Y_SECUENCIA_2026_09_12.md` §3 es

```
D0 -> D1 -> D2 -> D3 -> D4 -> D5 -> I5 -> I6 -> I7-I9 -> I10
```

D2 tiene evidencia candidata y espera revisión externa; la reja D2 sigue rehusando. El
siguiente paso **no ejecutado** es **D3**: pasos STEP 04 (cuantización), 05 (entropía y
compresión), 06 (tiempo/frecuencia) y 07 (detectores), los cuatro en estado
`Escrito` / `No común` / sin evidencia en la tabla §2 de ese documento.

Que D2 espere revisión no adelanta D3 ni lo bloquea: D3 se mide contra el banco sintético y
los tres recursos toy contratados, no contra los resultados de D2.

## 2. Prerrequisitos reales, medidos y no leídos de una tabla

`tools/df_d3_prerequisites.py`, ejecutado contra el almacén vivo el 2026-09-16
(`docs/audits/evidence/duckdb_corrections_20260916/D3_PREREQUISITES.json`):

| prerrequisito (§4 del diseño) | estado medido | evidencia |
|---|---|---|
| **R2** adjudicador reparado | `PRESENT` | `tools/df_d2_adjudicate.py` en el checkout |
| **N3** micro-run productivo reconciliado | `PRESENT` | acta `MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md` |
| **R6** vista de cobertura vigente | `NOT_APPLIED` al sondear → **`PRESENT` tras aplicarlo** | ver §2.1 |

El bloqueo era concreto, no formal. `df_fact_coverage` tiene **440.694 filas**, y contiene
**una** `run_id` con **dos** digests de código; `df_fact_coverage_v2` tiene **633.189**. Sin
vista de selección, una cifra de cobertura no podía decir qué ejecución y qué digest contó, y
el diseño exige cobertura vigente *para toda medición que fundamente decisiones*. Medir D3 sin
R6 habría producido números que nadie puede atribuir.

### 2.1 R6, aplicado el 2026-09-16

Se aplicó por la ruta de adopción —ensayo sobre copia, ventana coordinada, recibo— y **no** por
edición directa del cubo. `df_coverage_current` 633.189 = la matriz seleccionada, con **una**
ejecución y **un** digest; `df_coverage_history` 1.073.883 = 440.694 + 633.189; las tablas de
hechos **sin cambiar una fila**; denominador 715 datasets; WAL en 0.

Reprobado después: **`ready_to_measure: true`, nada bloqueando.** D3 ya no tiene prerrequisito
pendiente. Lo que falta antes de ejecutarlo es la revisión del contrato y la batería, y luego
implementar los nueve operadores.

## 3. Contratos de datos

### 3.1 Contrato del operador — `tools/df_d3_contract.py`

Los campos obligatorios del §1 del diseño, transcritos y **exigibles**: `kind`, `params`,
`bytes_state`, `fit_scope`, `lookback_samples`, `output_availability`, `warm_up_samples`,
`delay_samples`, `cost_cpu_seconds_per_1000`, `applicability`, `chunk_restart`. Reglas que
rehúsan en vez de suponer:

* falta cualquier campo → rechazo nombrando el campo; no hay valores por defecto;
* `fit_scope` fuera de `NONE`/`TRAIN_PREFIX_ONLY` → rechazo: calibración y confirmación no
  son ámbitos de ajuste;
* `output_availability` y `delay_samples` son el mismo hecho dicho dos veces; si discrepan,
  rechazo. Ésa es exactamente la forma de reclamar retardo cero en silencio;
* `fit_scope: NONE` con estado ajustado → rechazo;
* coste no medido (cero o no finito) → rechazo;
* `applicability` vacía → rechazo: un operador aplicable a nada no se puede medir;
* un campo que el contrato no define → rechazo (no se cuela un `score`);
* una salida marcada disponible dentro del warm-up declarado → rechazo: un warm-up escrito
  como `0.0` es una observación fabricada.

La salida lleva siempre tres ramas: `values`, `available` y **`raw`**. La rama cruda es parte
del contrato, no una convención, porque una regla que solo vive en prosa no la puede fallar
nadie. Identidades: `spec_sha256` y `state_sha256`.

### 3.2 Contrato del recurso

La disponibilidad la manda el contrato del lago (`gov_availability_contract`, GOV-N2):
`label`, `completion_lag_max`, `timezone_evidence`, `use_class`. **`UNKNOWN` no es cero**: un
archivo que no puede decir cuándo una barra está completa no ha dicho que lo esté de
inmediato. La prueba 7 devuelve `undecided` en ese caso y el veredicto global pasa a
`INCONCLUSIVE` — no a aceptado.

## 4. Pruebas — `tools/df_d3_acceptance.py`

Las diez pruebas del §3 del diseño, ejecutables contra cualquier operador que cumpla el
contrato, con veredicto `MECHANICALLY_ACCEPTED` / `MECHANICALLY_REFUSED` / `INCONCLUSIVE`.
Son **mecánicas**: dicen que lo que el operador declara de sí mismo es cierto. No dicen que
sea útil y no seleccionan nada.

`tests/test_d3_contract_and_acceptance.py` — **40 reglas** — demuestra que la batería puede
fallar: un operador causal declarado con honestidad pasa; su gemelo de ventana centrada —el
control no causal del diseño— falla; y hay un fixture por prueba que miente en un solo campo
y hace fallar solo esa prueba.

### 4.1 Un límite medido de la prueba 1, para el revisor

El §3.1 compara `transform(X[:n])` y `transform(X)` solo hasta `n - lookback - delay`. Un
operador que mira **4** muestras hacia adelante mientras declara **7** de lookback es por eso
invisible a la prueba 1: todos los índices que compara son índices que ambas llamadas pueden
calcular. Medido, con el control centrado: prueba 1 **pasa**, prueba 2 **falla**.

La prueba 2 sí lo atrapa, y el par cumple el propósito. Se registra como observación, **no se
corrige**: ampliar la prueba 1 a `n - delay` cerraría la brecha y es una decisión de diseño,
que no me corresponde tomar sobre un diseño sellado.

## 5. Ejecución gobernada — cómo correría, cuando se ordene

| | |
|---|---|
| entradas | banco sintético D2 (pasos, impulsos, motivos, régimen de media/varianza, tendencia) + los tres recursos toy contratados. El banco financiero **queda cerrado** hasta que sus contratos tengan evidencia |
| ruta | Flow v3 por `tools/governed_run.py`: campaña con clave exacta, `governed_download` → `X-Delivery-ID` → confirmación, terminal `governed_terminal.v1`, conciliación por `/reconcile` |
| salida | un terminal por operador × régimen, con `spec_sha256`, `state_sha256` y el informe de la batería como artefacto; sin métricas de utilidad |
| clasificación | `NON_GOVERNING` mientras la batería sea mecánica: no es evidencia científica y no debe aparecer en `gov_scientific_evidence` |
| cómputo | CPU, un hilo, `crispdm-run -m 2G -t <wall> -n d3-<operador>`; techo de memoria 2 GiB por proceso, del propio diseño |
| coste | el piloto de coste (prueba 8) fija el presupuesto por operador **antes** del barrido; un operador que no cabe no se corre |
| abstención | un diagnóstico completo con abstención es salida válida. Que los nueve operadores sean rechazados no permite saltar a D4 con entradas no caracterizadas |

## 6. Lo que este bloque no hizo, a propósito

No implementó ninguno de los nueve operadores; no ejecutó D3; no cambió el diseño sellado; no
inventó aceptación científica; no lanzó campaña de entrenamiento; y no promovió nada. R6 sí se
aplicó (§2.1), por ser el bloqueo medido y por la ruta de adopción. Los operadores de `tests/test_d3_contract_and_acceptance.py` son fixtures
del arnés, no candidatos.

## 7. Siguiente paso, y de quién es

1. ~~R6 por la ruta de adopción~~ — **hecho el 2026-09-16**; ver §2.1.
2. Revisión de este contrato y esta batería antes de implementar operador alguno: si la
   batería cambia después de que existan candidatos, los resultados no son comparables.
3. Con R6 aplicado y la batería aceptada, implementar los nueve operadores del §2 del diseño
   y correr la batería, operador por operador, bajo la ruta gobernada del §5.
