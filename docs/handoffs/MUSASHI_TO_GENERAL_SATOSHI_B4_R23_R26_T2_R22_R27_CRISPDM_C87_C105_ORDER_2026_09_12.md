# Orden de Musashi a General Satoshi: B4 R23-R26, T2 R22-R27 y CRISP-DM C87-C105

**Fecha:** 2026-09-12
**Dictamen:** `MUSASHI_AUDIT_B4_R19_R22_T2_R16_R21_CRISPDM_C67_C86_2026_09_12.md`
**Prioridad:** P0 identidad y semantica; P1 poblacion experimental
**Licencia:** `MECHANICS_ONLY_CPU_NO_SCORES`

## 0. Bases y fronteras

Parta de estos tips empujados, en ramas nuevas:

| Frente | Base |
|---|---|
| predictor | `fc62a073a2144d7b0ee4577a9c501cf1ea7c70d8` |
| financial-data | `b24b22719ea217d319eb6366ef3e333ddd8ab274` |
| B4 | `2fb3e2b9a7c4f2cfd3071edbdf3179ed672704b2` |
| T2 closure | `53ada0703cc898c48f961cc979d7befcc3f12247` |
| T2 reproducer historico | `6fe6c1ea6f9543ec0c253ffd2d26e2e7b5f97f4d` |
| T2 endurecido | `26b4214b555b1a7ac58552f64cbe897eb25786c2` |

Congele cada PRE antes de editar. Preserve byte a byte B4 v7, T2
original, sus copias, submissions anteriores, terminales v1-v3,
FEATURE_DAG v1-v3 y el contrato temporal v1. Todo sucesor es aditivo.

No cree records en nombre de Musashi. No ejecute GPU, score,
confirmacion, live, venue, publicacion DOIN ni limpieza del cubo. Puede
usar CPU para fixtures, re-derivaciones y el productor aislado que esta
orden autoriza. El productor nunca escribe sobre el lago existente.

## 1. P0-A: cerrar la identidad de los componentes de directorio

### C87 / B4-R23 / T2-R22 - PRE

En las cuatro copias de `descriptor_custody.py`, reproduzca:

1. fotografiar un root que contiene `cell/terminal.json=1`;
2. renombrar `cell` dentro del root retenido;
3. crear un nuevo `cell/terminal.json=9`;
4. demostrar que `walk_to("cell").read("terminal.json")` acepta 9;
5. repetir sustituyendo y restaurando el nombre antes de la lectura;
6. modificar entradas mientras se enumera un directorio y demostrar que
   el inventario carece de un before/after estable;
7. demostrar que JSON con claves duplicadas es aceptado hoy.

El caso exacto debe imprimir
`ACCEPTED_REPLACED_CHILD_DIR`, no una aproximacion que ya quede cubierta
por las pruebas de hoja.

### C88 / B4-R24 / T2-R23 - correccion compartida

Extienda el contrato comun:

* el inventario conserva hechos para archivos, subdirectorios y otros
  objetos, no solo nombres;
* `subdir()` compara el `fstat` abierto con los hechos inventariados en
  tipo, device, inode, uid, mode, size, mtime_ns y ctime_ns;
* cada enumeracion toma `fstat` del directorio antes y despues; si
  mtime, ctime o identidad cambian, el snapshot completo rehusa;
* una vez abierto, el descriptor hijo se retiene como hoy;
* `Artifact.json()` rechaza claves duplicadas, `NaN`, `Infinity` y
  `-Infinity` antes de entregar un objeto;
* todos los descriptores se cierran en exito y excepcion;
* hechos before/open/after y su digest quedan publicados.

Las cuatro implementaciones y fixtures deben quedar byte-identicos o
la divergencia debe ser mecanicamente justificada.

### B4-R25 - readjudicacion v4

Sobre una copia fresca de la raiz preservada, reejecute solamente el
cierre con el contrato nuevo. Protocolo A/B: A codigo y tests empujados;
B solo submission v4. Resultado esperado 2 completas, 1 parcial y 9 no
iniciadas, sin efecto ni ranking. Una diferencia detiene el proceso.

### B4-R26 - aceptacion focal

Incluya subdirectorio sustituido, restore-by-name, mutacion durante
listado, JSON duplicado y mutantes que eliminen por separado el binding
de directorio y los dos `fstat` de enumeracion.

## 2. P0-B: censo canonico, tipos y nulos

### C89 - PRE del censo

Mediante la API publica de `verify_lake_terminals` congele:

1. cambiar path, digest y size de un appearance conservando el
   `census_sha256` declarado y obtener hoy poblacion verificada;
2. terminal y censo con clave JSON duplicada;
3. appearance/variable con clave extra;
4. constante JSON no finita;
5. la serie `announcement_datetime_local_utc` tratada como numero
   finito con missingness cero.

### C90 - identidad canonica y esquemas

`load_census()` debe recomputar exactamente la funcion `_self_sha` del
productor sobre el documento sin `census_sha256`, y comparar el resultado
con el digest externo, el digest declarado y el nombre. Registre tambien
el SHA-256 de los bytes crudos como identidad de archivo separada.

Use parser JSON estricto y schemas/versiones exactos para top-level,
appearances, variables, pre-ledger y terminales. Campos no consumidos
pueden validarse y conservarse, pero no quedar fuera del contrato bajo
la etiqueta de schema exacto. El digest externo debe aparecer como
`REVIEWER_SUPPLIED_EXPECTATION`, no como autoautoridad del candidato.

El guardado before/after de terminales v1-v3 debe usar la custodia
corregida, no `Path.read_bytes()` ni una reapertura por ruta.

### C91 - contrato semantico por columna

Antes de recalcular estadisticas, derive y publique por appearance:

* tipo fisico Arrow y tipo logico declarado;
* unidad y rol si existen;
* politica de nulls/sentinels y su fuente;
* conteo de nulls fisicos y de sentinels declarados;
* estado `NUMERIC_MEASURABLE`, `NON_NUMERIC`,
  `SEMANTIC_TYPE_UNRESOLVED` o `MISSING_POLICY_UNRESOLVED`.

No convierta timestamps, categorias o ids a `float64` por conveniencia.
No declare un entero extremo como null solo por heuristica: sin contrato
queda no resuelto. El caso `announcement_datetime_local_utc` debe dejar
de publicar media/percentiles/entropia como mediciones numericas validas.

Haga un barrido de las 1.965 variables para sentinels, extremos
repetidos y tipos fisicos incompatibles. Publique cada caso; no fuerce un
conteo positivo.

### C92 - verificacion v4 y OLAP aditivo

Reejecute desde bytes con C88-C91 y emita terminales v4 + verification
v3 de manera aditiva. Los conteos se derivan, no se esperan. Las capas
deben distinguir por lo menos:

* `PRODUCER_DECLARED`;
* `SOURCE_BOUND`;
* `PHYSICALLY_TYPED`;
* `INDEPENDENTLY_RECOMPUTED`;
* `SEMANTICALLY_UNRESOLVED`;
* `DIVERGES`.

Pruebe primero la carga en una base desechable. Luego cargue al cubo
real sin borrar las filas existentes y demuestre idempotencia, backlog
y dead-letters visibles.

## 3. P0-C: readjudicar T2 con el verificador endurecido

### T2-R24 - decision de identidad

La readjudicacion nueva parte del tip endurecido `26b4214b`, incorpora
la custodia C88 y conserva como entrada inmutable el execution record
historico. No copie el `KeyError` conocido al nuevo verificador.

Prepare `agent_multi.t2_hardened_readjudication_review_record.v1`, que
ligue simultaneamente:

* execution record, commit, tree y siete digests historicos;
* commit, tree y superficie completa del readjudicador endurecido;
* raiz preservada e inventario por componentes/hojas;
* adjudicacion candidata;
* `READ_ONLY_HARDENED_READJUDICATION`, sin entrenamiento, descarga,
  ejecucion de modelos ni promocion.

Solo template candidato. No instale el record real.

### T2-R25 - gate antes de imports

El punto de entrada debe comprobar el record, el checkout y la
superficie antes de importar o ejecutar modulos de reconstruccion,
ejecutor, harness o modelos. Tras el gate, todos los imports salen del
unico checkout revisado. Revalide identidad antes y despues del replay.

Un checkout mutado entre gate e import, un modulo ya presente en
`sys.modules`, un `PYTHONPATH` que sombrea y bytecode sin fuente deben
rehusar de forma tipada.

### T2-R26 - replay endurecido

Con fixture aislado, reprocese los 242 records desde una copia privada.
No regenere predicciones ni lea modelos. `verify_unit_record` endurecido
debe rederivar todos los efectos, costos y la tabla binomial. La semilla
omitida debe producir refusal tipado.

Resultado esperado: 242/0, `DOES_NOT_ADVANCE`, estimando
`-0.001048443391358884` y los mismos seis efectos. Cualquier diferencia
es un resultado de auditoria y detiene la publicacion.

### T2-R27 - submission v4

Publique A/B y una submission v4 marcada
`ISOLATED_FIXTURE_NOT_EXTERNAL_REVIEW`. Preserve v1-v3 y el reproducer
historico. Incluya tests de record ausente, record autoemitido, semilla
omitida, mezcla de imports, sustitucion de subdirectorio y mutacion de
checkout despues del gate.

## 4. P1-A: procedencia prospectiva del productor

### C93 - disposicion de FEATURE_DAG.v3

No reescriba v3. Registre su aceptacion acotada como
`STATIC_CANDIDATE_INVENTORY_UNBOUND`; cero columnas activas y cero
licencia. Sus limitaciones declaradas permanecen visibles.

### C94 - rerun prospectivo aislado

Ejecute el productor actual commiteado sobre los insumos fisicos
registrados, en un root nuevo y CPU. Antes de correr, selle:

* repositorio, commit y tree limpios;
* archivos, simbolos y dependencias ejecutadas;
* digests de todos los inputs y configuracion;
* schema, tipos, versiones y semillas;
* root de salida vacio y write-once;
* presupuesto y condiciones de parada.

No escriba sobre `technical.parquet`, `statistical.parquet`, el CSV
historico ni el lago. Compare el resultado con los artifacts viejos,
pero no exija igualdad para fabricar procedencia. El resultado del run
es un dataset sucesor nuevo, incluso si sus valores coinciden.

### C95 - ensamblaje y causalidad end-to-end

El nuevo model-ready debe tener un ensamblador identificado. Pruebe
invariancia de prefijo para la salida completa: perturbar exclusivamente
raw inputs futuros no puede mover ninguna columna anterior al corte.
Incluya cada columna ligada, no solo dos ejemplos. Forward-fill,
backfill, joins tolerantes y ordenamientos deben quedar declarados y
probados.

Una columna cuyo productor o ensamblaje no pueda ejecutarse queda
`UNRESOLVED`; no se elimina para mejorar conteos.

### C96-C97 - binding y DAG sucesores

Emita `PRODUCER_BINDING_MANIFEST.v2` y `FEATURE_DAG.v4` para el dataset
sucesor. El binding minimo es:

`dataset_id + dataset_sha256 + input_sha256 -> repository + commit +
tree + file + symbol + config_sha256 + output_column`.

`CAUSAL_ACTIVE` exige binding completo y probe end-to-end. El dataset
historico conserva cero activas. Publique conteos reales; cero sigue
siendo valido.

## 5. P1-B: geometria temporal utilizable

### C98 - contrato de calidad temporal

Preserve el contrato v1 y publique un sucesor que separe:

* disponibilidad de la barra;
* completitud de su intervalo;
* regularidad del salto hasta la siguiente barra;
* elegibilidad de una muestra para un horizonte fijo.

Para el screen, una muestra es inelegible si su lookback, ventana del
operador o target contiene una barra truncada o cruza un gap. No
interpola ni rellena. Publique mascara, razones y soporte restante por
origen. `next-bar` debe significar siguiente intervalo nominal de cuatro
horas, no simplemente siguiente fila observada despues de un hueco.

### C99 - bateria temporal

Cubra truncamiento con y sin gap, gap dentro del lookback, gap en el
target, bordes del split, operador con warm-up 512, barra completa y
dataset irregular. Cambiar una cola futura no puede mover mascaras
anteriores.

## 6. P1-C: diseno por variable v4

### C100 - PRE conceptual

Congele que v3:

1. usa A3, un control de dimension aumentada, dentro de H1 para A1, que
   no aumenta dimension;
2. permite inferencia t con solo tres paneles;
3. admite variables sin tipo semantico, unidad, rol, licencia o politica
   de nulos;
4. no excluye muestras H4 que cruzan gaps o barras truncadas;
5. no define de forma ejecutable como se invierte Holm para sus limites.

### C101 - poblacion

El v4 deriva poblacion exclusivamente de terminales v4, DAG v4,
contrato temporal/calidad sucesor y manifests revisables. Ademas de P1-
P4, exige tipo semantico, rol, unidad cuando aplique, licencia y politica
de missing/sentinel. Identificadores, targets, timestamps y variables de
disponibilidad posterior al origen quedan fuera por regla.

Inventarie paneles candidatos sin mirar scores. Se requieren al menos
seis paneles independientes y cinco variables elegibles por panel; si no
existen, el screen queda `BANK_INSUFFICIENT`.

### C102 - hipotesis y controles

Fije tres preguntas separadas:

* H1: A1 reemplazo seleccionado por variable frente a A0 identidad,
  misma dimension;
* H2: A2 representacion aumentada frente a A0 y frente a A3 con igual
  numero de columnas sin informacion del target;
* H3: frecuencia de abstencion, transferencia entre paneles y no dano.

A3 usa valores generados solo desde el fold de entrenamiento, con
distribucion y escala declaradas; nunca labels, validation ni test. Su
funcion es controlar dimension de A2, no competir con A1.

### C103 - inferencia y comparabilidad

Especifique de forma ejecutable:

* un valor por panel y contraste;
* seis o mas paneles para inferencia; menos es descriptivo/inconcluso;
* prueba pareada, direccion, IC y algoritmo Holm exactos;
* LOPO o analisis de sensibilidad por panel;
* muestra comun congelada entre brazos despues de warm-up y mascara
  temporal;
* mismo presupuesto, origen, seeds y datos para todos los brazos;
* costo de seleccion y transformacion incluido;
* reglas de dano, abstencion, missingness e incompletitud.

### C104 - gate

Implemente solo validadores, fixtures, materializacion sin labels y
dry-run CPU. El entry point de score rehusa antes de abrir labels,
modelos o librerias numericas con
`EXTERNAL_V4_DESIGN_REVIEW_AND_LICENSE_REQUIRED`.

### C105 - retorno

Entregue un unico packet con:

1. defectos propios antes de resultados;
2. PRE/POST por hallazgo y mutaciones por guardia;
3. tips A/B de B4 y T2;
4. conteos medidos en tips finales;
5. deltas v3-v4 y prueba de inmutabilidad historica;
6. resultados del barrido semantico y del rerun prospectivo;
7. estado OLAP, backlog y dead-letters;
8. lista exacta de records que requeriran revision despues;
9. cero GPU, score, confirmacion y live.

## 7. Cierre

La ronda termina en revision externa, no en ejecucion cientifica:

`B4_V4_AND_T2_HARDENED_READJUDICATION_READY_FOR_REVIEW /
TERMINALS_V4_AND_FEATURE_DAG_V4_READY_FOR_REVIEW /
PER_VARIABLE_SCREEN_V4_STILL_UNLICENSED`.

No hay decision pendiente del propietario. Todas las tareas de esta
orden son ejecutables por Satoshi dentro de CPU y evidencia offline.
