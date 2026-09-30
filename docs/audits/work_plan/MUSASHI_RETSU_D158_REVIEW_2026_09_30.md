# Revision acotada de d158a787

Musashi, 2026-09-30. CHANGES_REQUIRED antes de B0 o adopcion del archivo.
Acepto el microensayo como evidencia sintetica de rutas de decision, no de
rentabilidad ni de importancia relativa de corto y largo. No hubo entrenamiento,
inferencia GPU, codigo remoto ni acceso a bases desplegadas en esta revision.

## F1. Alta: el ETL acepta contenido que contradice su manifiesto

doin-node 212c263, `src/doin_node/archive/warehouse.py`, project_metrics toma
envelope.sections directamente. No exige lectura verificada desde el archivo
ni recomputa su enlace al manifiesto. Con dataclasses.replace conserve cuerpo,
digests y manifiesto originales y cambie performance de una seccion a 999.
Se inserto 999 junto al manifest_digest original. La serializacion inicial
correcta no protege a este consumidor posterior.

La sonda solo uso SQLite en memoria y objetos sinteticos. No demuestra que
exista corrupcion en una cadena o warehouse desplegado. Remedio: el proyector
consume bytes recuperados/verificados por referencia; valida inventario,
secciones, tipos y vinculos antes de comenzar una transaccion analitica.

## F2. Alta antes de barridos: apalancamiento y costos no gobiernan la caja

heuristic-strategy 781022a, `app/elapsed_hour_harness.py`, run_elapsed_hour_harness
entrega leverage a la estrategia pero usa setcommission(margin=None, mult=1)
sin configurar el apalancamiento del broker. El fixture sube el capital a
2,000,000 para llenar tamanos que no llena con 10,000. Esto sirve para activar
ramas del plugin, pero NO verifica la semantica requerida por el usuario de
fraccion del balance destinada a margen multiplicada por apalancamiento.

Ademas, `app/plugins/plugin_long_short_predictions.py:455` resta swap a la
variable reportada profit_usd, no a la caja. Reconteo independiente del MICRO:

| Brazo | Caja final - inicial - PnL reportado | Swap reportado |
| --- | ---: | ---: |
| ideal/ideal | 33.33333333369 | 33.33333333333 |
| persistencia/ideal | 37.50000000039 | 37.5 |

Ambos terminan planos. La diferencia queda explicada por ese swap, pero los
costos no reducen los recursos usados al dimensionar nuevas ordenes. Es un
defecto heredado que el ensayo expone, no una alteracion de estrategia de Retsu.
No corregirlo reescribiendo sus resultados: sucesor de contabilidad separado.
Auditar tambien las unidades de comision: una tasa stock-like no equivale sin
mas a un importe fijo por lote y round-trip. No afirmo una cuantia corregida.

## F3. Media: proyeccion incompleta y deduplicacion que omite identidad

project_metrics no proyecta item['metrics']; el doble descarta parameters y
metric_schema. Un candidato con metrics={'MAE': 0.1} pierde MAE en el resultado
analitico. No sirve aun para reconstruir el OLAP solicitado.

DisposableWarehouse.record_round reutiliza round_id comparando solo domain_id
y detail_metrics. Probe el mismo id con otro experiment_id, round_number y
performance: devuelve el id anterior sin insertar ni reportar contradiccion.
Cambiar el sujeto o el valor no es un reintento identico. Se requiere identidad
de registro completa, conflicto explicito e insercion/proyeccion atomica.

Limite adicional, no declarado como corrupcion: dos manifiestos con candidatos
distintos para el mismo bloque tienen igual body_digest y distinto manifest_digest;
el indice por body_digest rechaza el segundo con CONFLICT. Definir si la primera
instantanea es inmutable o si se admiten sucesores por manifest_digest. No
confundir cuerpo de bloque con identidad del paquete completo de evaluaciones.

## F4. Media: la solicitud de clasificacion sigue sin bootstrap completo

predictor e1d04ad, `RETSU_BANKING77_EXECUTABLE_REQUEST_2026_09_30.md`, declara
pesos y corpus NOT_PRESENT. El comando exige el snapshot local, no lo descarga,
y activa HF_DATASETS_OFFLINE sin preparar el corpus Banking77. Una decision
positiva del dueno aun no deja un camino completo ejecutable desde este estado.
La prueba de snapshot sintetico no ejecuta el cargador real ni MTEB.

Preparar una fase de adquisicion separada y pinada de pesos y dataset, seguida
de verificacion y ejecucion offline; no abrir la red al codigo remoto. Distinguir
flags offline de una restriccion de red impuesta al proceso. Esto sigue sujeto
a aprobacion expresa del dueno para ejecutar ese codigo.

## Evidencia aceptada y limites

- Reejecute 28 tests de estrategia: 28 passed, 1.44 s, una advertencia Pydantic,
  CPU bajo crispdm-run 2 GiB/120 s. Dependencia trading_contracts via src local;
  no certifica instalacion limpia.
- Corto ideal cambia una salida concreta con largo ideal. Largo persistente no
  dispara entradas en este fixture. Eso NO responde cual familia es mas valiosa
  en mercado ni prueba una curva exponencial respecto de MAE.
- close() en stop() sin fill queda bien distinguido de liquidacion; conservar.
- DOIN esta correctamente etiquetado archivo local desechable, sin lago, cadena,
  ahorro medido ni AT01-AT10 aceptados. No trasladar este rechazo al resto del plan.
- No recertifique en esta ronda las 11 pruebas de clasificacion ni todo el
  lanzador GPU. Sus resultados son los de Retsu; la reparacion de proceso nuevo
  esta en el diff, pero el diagnostico real sigue sin ejecutarse.

Sonda independiente: `docs/audits/retsu_shadow_probe_20260930.py`.
Salida generada: `docs/audits/evidence/RETSU_D158_REVIEW_20260930/RESULTS.json`.
Se ejecuto bajo 2 GiB/120 s, sin modificar repos revisados ni stores reales.
Fuentes: retorno d158a787, estrategia 781022a, core dae92be, node 212c263,
GPU c5aa8dd4 y solicitud de clasificacion e1d04ad. No se reentreno nada.
