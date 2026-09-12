# Auditoria Musashi: retorno C53-C66, B4 y T2

**Fecha:** 2026-09-12  
**Retorno auditado:** `SATOSHI_CRISPDM_C53_C66_AND_B4_T2_RETURN_2026_09_12.md`  
**Disposicion global:** `REVISE`  
**Autoriza ejecucion cientifica:** no

## 1. Alcance e identidades

Se revisaron los siguientes tips empujados:

| Frente | Tip |
|---|---|
| predictor / CRISP-DM | `9bb90fa` |
| financial-data / DAG | `d53f24b6a` |
| B4, commit A / B | `c578fef3` / `0eab2705` |
| T2, commit A / B | `ee1d7904` / `26b4214b` |
| snapshot recuperable T2 | `f070eeb1` |

Las baterias focales publicadas vuelven a pasar: B4 `61/61`, T2
`30/30`. Esas baterias no contienen los contraejemplos que siguen.

## 2. Hallazgos

### P0-1. El directorio queda retenido, pero sus archivos no

`DirSnapshot` registra solamente los nombres durante el inventario
(`tools/descriptor_custody.py:171-184`). Al leer, abre de nuevo el
nombre relativo al descriptor del directorio (`:195-255`), pero no
compara el `(device, inode, size, mtime_ns, ctime_ns)` visto durante el
inventario con el `fstat` del archivo abierto.

Contraejemplo ejecutado sobre el tip B4:

1. se fotografia `cell/terminal.json` con `wall_seconds=1.0`;
2. se renombra esa hoja dentro del mismo directorio retenido;
3. se crea otra `terminal.json` con `wall_seconds=999.0`;
4. `cell.read("terminal.json")` consume `999.0`.

Salida observada: `photographed=1.0 consumed=999.0`.

El mismo modulo y digest (`a2b579dc...`) se usa en predictor, B4 y T2.
En T2 los JSON se cargan al construir el adaptador, pero los 242 NPZ se
leen de forma diferida (`t2_campaign_closure.py:1058-1062`), por lo que
la misma sustitucion puede ocurrir entre inventario y lectura.

**Efecto:** las submissions B4 v2 y T2 v2 no reciben aceptacion externa
todavia. Los resultados historicos permanecen intactos.

### P0-2. `TERMINALS_VERIFIED_EXACT` afirma mas de lo comprobado

El verificador de las 1.965 terminales tiene dos falsos positivos
reproducidos mediante su API publica:

| Mutacion | Resultado actual |
|---|---|
| eliminar el archivo fuente | `TERMINALS_VERIFIED_EXACT`, `absent_count=1` |
| agregar `descriptors={"fabricated":999}` | `TERMINALS_VERIFIED_EXACT` |

La primera ocurre porque `source_absent` no participa en `exact`
(`tools/verify_lake_terminals.py:274-275`). La segunda ocurre porque el
verificador comprueba poblacion y nombres, pero no valida un esquema
terminal estricto ni vuelve a calcular los descriptores. Luego v2 copia
`outcome`, conteos y descriptores declarados por el productor
(`:376-405`).

Hay dos problemas adicionales:

* una variable se liga al primer `appearance` del censo, no al
  `appearance` que declara la terminal (`:123-132`, `:171-173`);
* el archivo fuente se abre por una ruta construida con texto del censo
  (`:242-271`), sin demostrar contencion ni conservar el descriptor.

**Efecto:** el conteo 1.965/1.965 acredita presencia y cardinalidad, no
la correccion cientifica de 1.505 mediciones. La vista OLAP y la
membresia aditiva pueden conservarse, pero ninguna terminal v2 concede
elegibilidad.

### P0-3. `FEATURE_DAG.v2` todavia produce falsos causales

Se ejecutaron cinco programas pequenos contra `classify()`:

| Caso | Resultado actual incorrecto |
|---|---|
| reasignacion local termina en `shift(-1)` | `CAUSAL_ACTIVE` |
| `series.apply(lead)` con `lead -> shift(-1)` | `CAUSAL_ACTIVE` |
| `series.iloc[-1]` | `CAUSAL_ACTIVE` |
| asignacion dentro de funcion anidada | `CAUSAL_ACTIVE` |
| `rolling(10).mean().rolling(20).mean()` | lookback `20`, no `29` |

La causa combina `ast.walk()` sin orden de control, `setdefault()` para
la primera asignacion (`derive_feature_dag_v2.py:125-135`), una lista
demasiado amplia de metodos asumidos seguros (`:49-60`) y la agregacion
del alcance mediante `max()` (`:358`).

Un sexto contraejemplo demuestra falta de procedencia: dos funciones
vivas escriben la misma columna, una causal y otra adelantada. El
veredicto cambia al invertir su orden porque se selecciona
`(live or symbols)[0]` (`:346-351`). El nombre de una columna no liga el
dataset fisico al productor que realmente lo genero.

**Disposicion:** `FEATURE_DAG.v2 = REVISE`. Sus 37
`CAUSAL_ACTIVE` son candidatos estaticos, no variables licenciadas.

### P0-4. La readjudicacion T2 no debe reescribir la autoridad historica

La submission T2 declara correctamente que su snapshot no satisface la
identidad unica. La solucion elegida es:

* el execution record original permanece inmutable y sigue diciendo
  que codigo ejecuto la campana historica;
* no se repina ese record a un commit posterior;
* la readjudicacion es un acto nuevo, de solo lectura, con un record
  externo propio que liga simultaneamente los siete archivos historicos
  por digest y el checkout completo del reproductor nuevo.

El snapshot `f070eeb1` ya es recuperable y contiene los diez archivos,
pero necesita las correcciones P0-1 y un gate especifico de
readjudicacion antes de poder recibir ese record.

### P1-1. Semantica temporal: una parte ya es demostrable

Para
`financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1` se
compararon el CSV y el parquet fuente declarado:

* 18.085/18.085 filas del CSV emparejan por `DATE_TIME`;
* `DATE_TIME == open_time`;
* maximo error absoluto de OPEN, HIGH, LOW, CLOSE y VOLUME: `0.0`;
* el parquet trae `close_time`, cuatro horas despues del `open_time`.

Por tanto se decide:

* `timestamp_semantics = BAR_OPEN`;
* `bar_seconds = 14400`;
* la informacion OHLCV no esta completa antes de `close_time`;
* `provider_delivery_latency = UNOBSERVED`, no cero.

La disponibilidad causal offline y la demora operacional live deben
ser campos distintos. El dataset EURUSD legacy queda `UNDECLARED` por
falta de procedencia suficiente.

### P1-2. El diseno por variable aun no es ejecutable

`PER_VARIABLE_PREPROCESSING_DESIGN.v2` es honestamente un draft, pero
todavia no fija poblacion, operadores y parametros, modelos, ventanas,
semillas, presupuestos, margenes numericos, minimo de paneles ni la
construccion exacta del control A3. Tampoco puede consumir un DAG o unas
terminales que no han pasado revision.

**Decision de licencia:** se permite desarrollar mecanica y tests CPU
sin puntuar. La licencia de screen cientifico queda cerrada hasta un
diseno v3 revisado externamente.

## 3. Disposiciones por frente

| Objeto | Disposicion |
|---|---|
| B4 submission v2 | `REVISE_LEAF_INSTANCE_BINDING` |
| T2 submission v2 | `REVISE_CUSTODY_AND_READJUDICATION_IDENTITY` |
| resultado T2 historico | preservado; `DOES_NOT_ADVANCE` sigue candidato, no reescrito |
| FEATURE_DAG.v2 | `REVISE` |
| terminales v2 | `POPULATION_PRESENT_NOT_SCIENCE_VERIFIED` |
| membresia OLAP y vista aditiva | `ACCEPT_BOUNDED` |
| diseno por variable v2 | `REVISE_BEFORE_LICENSE` |
| licencia actual | `MECHANICS_ONLY_CPU_NO_SCORES` |

## 4. Estado operativo observado

* `crispdm-olap-loader.service`: activo, cero reinicios, CPU.
* no hay una espera de Satoshi pendiente;
* existe un supervisor anterior de `phase-2-eth-anchored-full-fleet-v2`,
  separado de T2; no fue tocado;
* `nvidia-smi` continua en
  `Driver/library version mismatch`;
* el reinicio de Omega sigue siendo la unica accion pendiente del
  propietario.

Las correcciones y sus criterios de aceptacion quedan en la orden
`MUSASHI_TO_GENERAL_SATOSHI_B4_R19_R22_T2_R16_R21_CRISPDM_C67_C86_ORDER_2026_09_12.md`.
