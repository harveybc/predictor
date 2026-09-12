# Auditoria Musashi: recuperacion R1-R6 y CRISP-DM C31-C44

**Fecha:** 2026-09-12
**Retorno auditado:** `predictor@e4f5cde`
**Integracion limpia auditada:** `predictor@9c84f99`
**Repos relacionados:** `financial-data@205e0ef53`, `lts@20057d2`,
`agent-multi@e343b4f2`, B4 `agent-multi@01947d0c`, T2
`agent-multi@b8a1720e`

## 1. Veredicto

**`REVISE`**.

La recuperacion operacional es mayormente correcta. Sin embargo, el cierre B4
reproduce sus conteos pero no queda aceptado porque verifica y consume
artefactos mediante aperturas de ruta separadas. C41-C44 tampoco cumplen el contrato cientifico
de la orden: se confundio un registro de datasets de investigacion con demanda
de consumidores activos, no se construyo el linaje causal por caracteristica y
solo se caracterizaron 107 de las 1.965 variables conceptuales comprometidas.

Ademas, el cierre T2 mezcla codigo revisado con un reconstructor posterior no
incluido en la identidad revisada y vuelve a abrir por ruta los records despues
de verificarlos. El resultado negativo no se contradice, pero su autoridad
queda pendiente de una re-adjudicacion corregida.

| Bloque | Disposicion |
|---|---|
| R1, cierre B4 | **REVISE**: conteos candidatos 2/1/9 reproducidos; custodia descriptor-first no implementada |
| R2, cierre T2 | **REVISE**: resultado candidato negativo, custodia de re-adjudicacion incompleta |
| R3, P1LR historico | **ACCEPT** |
| R4, recuperacion Alpaca | **ACCEPT con observacion P2** |
| R5-R6, OLAP y matriz | **ACCEPT** |
| C31-C40 | **ACCEPT con residual menor en duplicados de entry points** |
| C41 | **REVISE**: cobertura 107/1.965, no 1.965/1.965 |
| C42-C44 | **REVISE**: demanda, DAG y candidato no representan consumidores activos |

## 2. Verificacion independiente

1. La bateria focal de identidad, terminal, OLAP y caracterizacion produjo
   **98/98** en la rama de retorno y **97 passed / 1 skipped** en la rama de
   integracion limpia. La bateria de linaje produjo **16/16**.
2. La bateria existente de cierre B4 produjo **21/21** y la reconstruccion no emisora
   reprodujo `COMPLETED_VERIFIED=2`, `QUARANTINED_PARTIAL=1`,
   `NOT_STARTED=9` y `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`. Esas pruebas no
   ejercitan reemplazos de ruta entre lectura, hash, parseo y conteo.
3. La reconstruccion profunda no emisora de T2 termino en 823,03 s: verifico
   242/242 unidades y reprodujo `DOES_NOT_ADVANCE` con estimando
   `-0.001048443391358884`, sin descarga, escritura ni reentrenamiento. La misma
   salida publico un valor p imposible de `1.3125`, incluido abajo.
4. La matriz de recuperacion produjo 17 componentes `OK` y una atencion
   tipada: GPU no enumerable.
5. PostgreSQL y Metabase responden; el loader OLAP esta `active/running`,
   `NRestarts=0`, heartbeat de 30 s, backlog 0, dos dead-letters adjudicados y
   `healthy=true`.
6. Supervisor, sesiones Dragon/Gamma, runner Alpaca y timers esperados estan
   activos y habilitados. El runner Alpaca conserva una orden pendiente, cero
   posiciones y el estado `model_close_deferred_market_closed`.

## 3. Hallazgos

### P0-1. El cierre B4 no consume los bytes que dice verificar

El paquete afirma que las dos celdas completas fueron verificadas por
descriptor, pero `b4_campaign_closure.py` usa aperturas de ruta independientes:
lee el terminal, lo hashea despues, abre de nuevo el ledger por barra para
contar filas, y repite el patron con claim, intentos y sellos. El inventario de
la celda parcial tambien combina `rglob()`, `is_file()`, `stat()` y hash por
ruta antes de volver a leer objetos de estado. La clasificacion `NOT_STARTED`
se apoya en comprobaciones de existencia igualmente separadas.

Un reemplazo entre esas operaciones puede hacer que se validen unos bytes y se
consuman otros. Por tanto, `2/1/9` y
`SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT` son una reconstruccion candidata
plausible, no un cierre con autoridad. La reparacion no requiere entrenamiento:
debe leer, hashear, validar y consumir cada objeto desde un unico descriptor y
repetir la adjudicacion sobre una copia de la raiz preservada.

### P0-2. C42 deriva demanda de un inventario, no del consumidor real

`demand_columns()` llama "ACTIVE consumer" a cada header de los dos datasets
incluidos en `crispdm_dataset_inventory.v1.json`. No encuentra un config que
consuma esos archivos: una busqueda exacta de sus rutas solo devuelve el
registro, el inventario y el indice del banco. El propio registro los describe
como material de desarrollo/inventario, no como ejecuciones activas, y no
declara targets.

Por tanto:

- la demanda supervisada activa no esta demostrada;
- la demanda RL activa es correctamente cero;
- las 93 columnas son nombres unicos de dos tablas registradas, mientras 97
  son pares `(dataset, columna)`; no son la misma cardinalidad;
- `rl_is_subset_of_supervised=true` es una verdad vacua y debe ser
  `NOT_APPLICABLE`;
- `FINANCIAL_AVAILABILITY_CANDIDATE.v1.json` no puede autorizar variables
  consumidas. Puede conservarse, supersedido, como inventario de candidatos
  del banco de investigacion.

La correccion no consiste en borrar esas 97 variables: consiste en separar
**demanda ejecutante**, **universo de investigacion** y **targets**.

### P0-3. C43 no materializa un DAG causal por caracteristica

`derive_feature_lineage.py` asigna a las 84 features derivadas la misma frase:
"computed inside this view's producer from the OHLCV of the same bar". El
programa no inspecciona `feature-eng`, `feature-extractor` ni el productor real
de cada columna, y no registra funcion, archivo, commit, entradas directas,
ventana, desplazamiento, alineacion o digest del transformador.

Eso no demuestra la afirmacion `max(input event times) + latency`. Una media
rodante, un retorno desplazado, una descomposicion centrada y una variable
publicada externamente no comparten necesariamente el mismo grafo ni la misma
disponibilidad. Si el productor no puede localizarse, el estado correcto es
`UNRESOLVED_PRODUCER`, no una procedencia generica.

### P0-4. El cierre T2 usa una identidad mixta y reabre evidencia

El record externo fija siete archivos del ejecutor en `7bcd3f0d`, pero no fija
`t2_completion_reconstruction.py` ni `t2_campaign_closure.py`, creados despues.
`load_reviewed_modules()` importa confirmatorio/ejecutor del checkout revisado
y el reconstructor desde el tip posterior. La salida no fue producida
enteramente por la identidad que el cierre denomina revisada.

Hay ademas una reapertura concreta: `final_adjudication()` verifica las
unidades, y despues `reconstruct()` vuelve a leer cada `RECORD_*.json` por ruta
para construir la lista entregada a `adjudicate_screen()`. Un reemplazo entre
ambos pasos permite verificar un record y puntuar otro. No existe una bateria
dedicada a `t2_campaign_closure.py`.

El resultado `DOES_NOT_ADVANCE`, estimando aproximado `-0.001048`, queda como
**candidato reproducido**, no como autoridad final, hasta cerrar este punto.
No se reentrena nada para corregirlo.

### P0-5. T2 publica un valor p fuera de su dominio

Con tres efectos positivos entre seis paneles, `adjudicate_screen()` calcula el
doble de una sola cola binomial y publica `sign_test_exact_p_two_sided=1.3125`.
Un valor p debe pertenecer a `[0,1]`; la formula tampoco es simetrica para cero,
uno o dos signos positivos. La condicion de avance exige 6/6 y el gate de dano
ya determina el resultado negativo, por lo que este defecto no lo invierte,
pero una adjudicacion cientifica no puede conservar una inferencia imposible.

Debe usarse un test binomial bilateral exacto predeclarado y probarse la tabla
completa de 0 a 6 exitos: `0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875,
0.03125`.

### P1-6. C41 ejecuto 107/1.965 variables

El ledger v2 declara exactamente 107 variables conceptuales y seis apariciones
fisicas intentadas: 97 financieras, siete publicas y tres sinteticas. La orden
exigia que cada una de las 1.965 variables conceptuales del lago terminara en
`MEASURED`, `NOT_IDENTIFIABLE`, `UNAVAILABLE` o `FAILED`.

Quedan **1.858 variables conceptuales sin disposicion de caracterizacion**. El
censo completo de 1.965 no sustituye su medicion. Las 420 filas piloto se
conservaron correctamente y las 2.568 filas nuevas estan ligadas a bytes.

### P1-7. Identidades y conteos necesitan nombres no ambiguos

- El archivo `AVAILABILITY_BRIDGE.v2.json` publica schema `...bridge.v1`.
- Deben coexistir campos distintos para `unique_column_names`,
  `dataset_column_subjects`, `active_x_subjects`, `active_y_subjects` y
  `research_bank_candidates`.
- Ningun conjunto vacio debe producir una afirmacion positiva de subconjunto,
  suficiencia o cobertura.

### P2-8. Duplicados identicos de entry point quedan por orden de instalacion

El resolvedor aplica su politica solo cuando hay mas de un **valor** distinto.
Dos distribuciones que publiquen el mismo `(group, name, value)` dejan dos
matches y se consume `matches[0]` sin registrar ambas distribuciones. Como el
modulo resultante suele ser el mismo, no invalida esta ronda, pero la politica
debe cubrir tambien este caso o rehusarlo.

### P2-9. El runner Alpaca esta protegido, pero emite ruido repetitivo

El servicio esta vivo y reporta `protection_preserved=true`; no hay posicion y
el mercado esta cerrado. Sin embargo, cada minuto registra
`terminal due-bar decision cannot be revised` antes de volver al estado
`pending_entry_target_changed`. No es una caida, pero conviene convertir la
repeticion esperada en un no-op idempotente tipado, sin tocar la orden abierta.

## 4. Estado de GPU tras el reinicio

El reinicio ocurrio a las 22:53. Una actualizacion automatica de NVIDIA se
instalo despues, entre 23:49 y 23:51. Por eso el kernel conserva el modulo
`580.173.02`, mientras DKMS, `modinfo` y las bibliotecas ya estan en
`580.178.04`. `nvidia-smi` falla con mismatch de ABI.

La reparacion minima es **un segundo reinicio del equipo**. Todos los servicios
revisados estan habilitados para volver. Despues se debe ejecutar solamente la
matriz read-only y comprobar `nvidia-smi`; este dictamen no autoriza una nueva
campana GPU.

## 5. Siguiente paso

Ejecutar la orden complementaria
`MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C45_C52_AND_T2_R7_R10_ORDER_2026_09_12.md`.
Su primer bloque corrige el cierre B4 sin tocar la campana; en paralelo se
corrige T2 y se trabaja en CPU sobre verdad de demanda, DAG causal por feature
y caracterizacion completa del lago. No se abre seleccion, confirmacion, live,
DOIN ni una nueva campana GPU.
