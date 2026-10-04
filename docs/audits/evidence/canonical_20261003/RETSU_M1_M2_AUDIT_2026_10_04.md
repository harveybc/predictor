# Auditoria M1/M2 de Retsu

Observado: 2026-10-04 04:32 UTC. Alcance: `retsu/m1-weekly-score-gate-20261003@39a45e88`, `retsu/m2-readiness-ledger-20261003@5da79f34` y artefactos vivos PS3-R de Gamma/Dragon. Este dictamen no detiene ni invalida entrenamientos terminales autenticos.

## Veredicto

Las dos ramas contienen avance util, pero ninguna es integrable todavia. M1 tiene tres defectos de identidad/custodia estadistica. M2 mezcla evidencia causal historica superada con el reanalisis autoritativo y adopta resultados PS3-R mediante una lista escrita a mano. No existe aun manifiesto final de seleccion.

## Hallazgos, por severidad

### F1 - CRITICO: M2 publica identificacion causal contradicha por la evidencia vigente

`m2_readiness/REPORT.json` declara 65 features `IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS` y 13 mixtas. La autoridad vigente es `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 estimaciones historicas, 109 asociaciones con BH local, cero confirmaciones no lineales y **cero supervivientes de identificacion**. `NOT_IDENTIFIED` sigue siendo neutral, no rechazo.

La causa es reproducible: `generate_m2_readiness.py::ps3c_status` lee los conteos `rung2_state_counts`/`rung3_state_counts` de `coverage_reconciliation.json`, no los tres `ps3c_join.json` emitidos por `causal-inference@48ae17c`. Por tanto, los conteos causales de M2 no se aceptan y PS5 no puede usar sus brazos causales.

### F2 - ALTO: M2 adopta resultados PS3-R mediante constantes

El generador agrega manualmente cada feature terminal a `completed`. Esto obliga a modificar codigo por cada celda, deja fuera las celdas alternativas `tv.*` de lane F y no puede representar correctamente una celda viva/no terminal. La evidencia debe descubrirse desde raices permitidas y aceptarse solo tras verificar manifiesto, feature, familia, seed, codigo, input y hash de resultados.

### F3 - ALTO: el scorer semanal pierde evidencia pareada al reiniciar

La serializacion de `HorizonScore` no conserva `paired_differences`; `WeeklyScoreLedger.from_json()` la reconstruye como tupla vacia. El digest actual sigue validando porque tampoco cubre esos valores. Un cierre recargado ya no contiene la evidencia por origen que dice custodiar.

### F4 - ALTO: una semana exige por error una sola identidad de modelo

La identidad usada por `score_week` incorpora `model_digest` y luego exige igualdad entre todos los horizontes. Eso rechaza el caso legitimo del negocio: un modelo corto y otro largo, cada uno con su propio digest, sobre la misma semana y poblacion. Deben compartir identidad de tarea/poblacion/cutoff, no necesariamente pesos.

### F5 - MEDIO: semanas contiguas se cuentan como replicas independientes

`AnnualClose.independent_replicates = denominator` convierte semanas consecutivas de una misma trayectoria en replicas independientes. Deben reportarse como unidades longitudinales de evaluacion; la independencia no esta establecida.

### F6 - MEDIO: PS5 parece preparado por feature aunque falte evidencia

Las 366 filas llevan `PREPARED_INPUT_NOT_TRAINED`, incluso cuando carecen de PS3-R o PS4. Solo el diseno global de PS5 esta preparado. Cada candidata incompleta debe decir `NOT_READY_EVIDENCE_INCOMPLETE` con su faltante.

## Trabajo autentico conservado

- M1: 13 pruebas focalizadas pasan; el calendario y la reduccion semanal son reutilizables tras reparar F3-F5.
- PS4: diez transformaciones, cinco folds, 1,050 filas, repeticion byte-identica y pico de 133,099,520 bytes.
- Dragon: `fred.stress.vixcls.logret_5d` terminal autentica, hash `ec16a107...`, utilidad mixta, sin seleccion.
- Gamma lane F: `tv.hilbert_amp`, `tv.kalman_dev` y `tv.stl_dev` tienen 296 filas cada una, seed 0, codigo `31add127` y hashes de resultados iguales a sus manifiestos. Son mediciones de extractor, no decisiones de feature.
- A las 04:31 UTC la 5090 ejecutaba secuencialmente `tv.stl_seasonal`; Dragon retenia la celda `fx.eurjpy.logret_1h`. No se ordena detenerlas.

## Estado cientifico

Denominador: 366 candidatas. Seleccion final: `NOT_ISSUED`. Causalidad aceptada bajo el metodo vigente: 0/279 identificadas; esto no elimina ninguna feature. PS3-R y PS4 siguen progresando. La comparacion common-K PS5 continua bloqueada por evidencia incompleta, no por autorizacion.
