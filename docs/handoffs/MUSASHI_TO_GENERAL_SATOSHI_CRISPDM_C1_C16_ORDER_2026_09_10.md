# Orden a Satoshi: C1-C16, cerrar inventario, reja y OLAP activo

**Fecha:** 2026-09-10

**Dictamen gobernante:**
`docs/audits/MUSASHI_AUDIT_CRISPDM_P0_P6_2026_09_10.md`

**Base de retorno:** `predictor@ac53433`, `financial-data@983c4e8f4`,
`preprocessor@d6643f9`, `agent-multi@4d310029`,
`doin-domains@be6495a`, `doin-plugins@60af316`.

## 0. Disposicion

Ejecutar C1-C16 de punta a punta. CPU solamente. No abrir GPU, live, venue,
confirmacion, seleccion cientifica ni optimizacion DOIN. No reabrir los
resultados de T2, M4 o B4: esta orden corrige su representacion y transporte,
no su ciencia.

El riesgo de esta ronda es **validez de evidencia y orden de ejecucion**. No
expandirla a modelos de amenaza de sistema operativo, permisos, servicios de
trading o ceremonias de claves. No borrar filas OLAP existentes ni crear un
cubo nuevo.

## 1. PRE obligatorio

Congelar antes de editar, mediante APIs publicas o el punto de llamada real:

1. `use_optimizer=true` alcanza `optimizer_plugin.optimize()` antes de la reja.
2. manifest valido + `subject_ids=[]` devuelve `ELIGIBILITY_GATED`.
3. manifest y digest autoemitidos por el llamador conceden una decision
   positiva.
4. los sujetos configurados pueden diferir de las columnas fisicamente
   consumidas.
5. sustituir data/partitions manteniendo el id no se compara en
   `gate_subjects()`.
6. claves JSON duplicadas, `NaN`, digest no canonico y fecha futura alcanzan el
   parser actual.
7. un primer censo con entradas nuevas produce cero digests; una mutacion de
   igual longitud no obliga re-hash.
8. una instancia real indexada por `feature_family` no alcanza una variable
   consultada por `entity`.
9. el indice comun contiene 214 filas y cero filas de variable conceptual.
10. dos sobres con igual `campaign_key` y distinta identidad comparten la
    dimension.
11. un string de 64 caracteres concede el gate de respaldo aunque el dump no
    exista.
12. una particion MTM mas corta que `window_size+1` devuelve un tipo distinto
    del contrato del llamador.

Cada PRE queda como regresion permanente. No aceptar un fallo accidental como
kill: la prueba debe nombrar la razon correcta.

## 2. P0: reja ejecutable de variables y operadores

### C1. Orden real

Mover la reja antes de **cualquier** optimizacion, construccion de ventanas,
fit, `fit_transform` o materializacion de operador. Probar el camino normal,
`use_optimizer=true`, `load_model=true` y cada pipeline registrado. La carga de
clases puede ocurrir antes; ningun metodo que consuma datos puede hacerlo.

### C2. Universo derivado, no lista opcional

Construir un `resolve_consumed_subjects()` unico que derive ids, versiones y
columnas desde los archivos y contratos que el run va a consumir. Leer
encabezados/esquema sin construir ventanas ni ajustar nada. Exigir:

- conjunto no vacio cuando el contrato declare entradas;
- igualdad exacta entre train/validation/test y el mapping de variables;
- ausencia de duplicados, columnas extra o columnas sin id;
- digest de los bytes/esquema, particiones y codigo en el ultimo punto de uso.

`eligibility_subjects` puede funcionar como asercion adicional, nunca como la
fuente de verdad. Una lista vacia no concede nada.

### C3. Autoridad separada

El manifest producido por la campana es una **submission** y no concede
elegibilidad. La decision positiva requiere un record de revision separado que
ligue exactamente manifest, censo fisico, codigo, particiones, alcance y fecha.
El path y digest no pueden ser elegidos por el mismo config que solicita el
permiso. Satoshi entrega template y submission; **no** emite el record real.

### C4. Parser estricto

Aplicar a manifest, submission y record:

- esquema exacto en todos los niveles;
- tipos exactos, sin `bool` como numero;
- rechazo de claves duplicadas y constantes no finitas;
- SHA-256 canonico de 64 hex minusculas;
- fechas RFC3339 UTC canonicas, no futuras y orden temporal coherente;
- archivo completo, sin ignorar bytes despues del limite;
- ids y alcances no vacios y unicos.

### C5. Reproduccion historica expresa

Sin record de revision, un experimento nuevo refusa. El modo heredado solo se
abre mediante `execution_purpose=ARCHIVAL_REPLAY_NON_AUTHORITATIVE` y queda
estampado en config, resultados y OLAP. No inferirlo por la ausencia del
manifest. Las configuraciones antiguas siguen reproducibles al declarar ese
modo.

Replicar C1-C5 en los consumidores reales de `preprocessor` y `agent-multi`.
Los adaptadores sin punto de llamada en DOIN no se presentan como integrados.

## 3. P1: censo fisicamente ligado

### C6. Digest incremental correcto

En el primer censo, toda aparicion presente es `ADDED` y se digiere. En censos
posteriores:

- una entrada nueva se digiere;
- cambios de tamanio, `mtime_ns`, `ctime_ns` o identidad fisica obligan re-hash;
- una entrada realmente sin cambio conserva el digest anterior y lo etiqueta
  `REUSED_FROM_PREVIOUS_VERIFIED_CENSUS`;
- seleccion explicita siempre re-hash-ea;
- una mutacion de igual longitud muerde una regresion.

Ejecutar el censo completo real de 14,4 GB. Reportar bytes leidos, segundos y
cobertura. No presentar `stat` como digest.

El artefacto content-addressed se escribe `O_EXCL`; si ya existe, verificar
igualdad completa y no truncarlo.

### C7. Disponibilidad por alcance minimo

Derivar desde manifests y encabezados la lista exacta de familias consumidas
por el contrato de observacion v2 ratificado y por las tareas financieras
supervisadas activas. Instanciar contratos solo cuando fuente, worker y politica
de publicacion demuestren los ocho campos. Todo lo demas conserva
`UNAVAILABLE`.

Corregir el join `feature_family -> variables` con un mapping explicito y
ejecutable; no usar coincidencia accidental con `entity`. Probar dos entidades
de una familia, una entidad de dos familias cuando el contrato lo permita y
familia ausente.

### C8. Perfiles externos y metadata

Validar el self-digest del inventario externo, el esquema exacto, cada digest
de dataset y los bytes fisicos que dice perfilar. Un resumen autoconsistente no
se llama `FULL_BOUND`. Resolver procedencia por DAG/worker/recibo; conservar
`AMBIGUOUS` cuando falte evidencia. Los 330 stubs siguen no semanticos; solo se
completa el alcance minimo.

## 4. P2 y P4: indice comun y borde MTM

### C9. Indice comun consumible

Emitir filas tipadas y ligadas para:

- dataset y serie del banco publico;
- generador/unidad del banco sintetico;
- dataset, aparicion fisica y las 1.965 variables conceptuales financieras;
- operadores de transformacion conocidos.

Cada fila porta su autoridad, fuente, identidad y digest. Recalcular conteos
desde las filas. El join no puede transformar `CALIBRATION_ONLY` o
`FINANCIAL_DEVELOPMENT` en evidencia publica.

### C10. MTM train-only total

Hacer estable el tipo de retorno en particiones cortas. Si train no produce un
scaler valido, validation/test no pueden ajustarlo: la unidad queda
`NOT_EVALUABLE` con razon. Probar train constante/corto, validation con varianza,
reinicio y cola futura alterada.

## 5. P5: diseno de seleccion superseding

### C11. Diseno v2 despues del universo

Solo despues de C3 y C9, emitir un diseno que ligue el manifest/review record y
el indice fisico. La validacion debe ser anidada: seleccion y ajuste dentro del
training interior; outer validation/test una sola vez para comparacion. Congelar
unidad estadistica, comparadores, presupuestos, multiplicidad, extremos,
abstencion y costos. Un self-digest del candidato prueba cronologia, no revision
externa.

Ejecutar solo preflight mecanico CPU sin score. Entregar submission a Musashi;
no emitir record de autorizacion.

## 6. P6: OLAP veraz y activo

### C12. Sobres ligados a productores

Cada adaptador T2/M4/B4 debe consumir el artefacto original por un parser
especifico, verificar su digest fisico y ejecutar su verificador/rederivacion
existente antes de crear el sobre. El sobre conserva el digest del archivo
fuente y la identidad de su verificador. Un JSON fabricado con campos parecidos
refusa.

Los 60 hechos actuales se conservan y se marcan aditivamente
`TRANSLATED_SUMMARY_NON_AUTHORITATIVE`, con enlace al sobre que los supersede.
No borrar ni actualizar silenciosamente historia.

### C13. Identidad relacional e idempotencia

Al conflicto por `campaign_key`, comparar productor, clase, design, codigo y
run. Diferencia = refusal antes de hechos. Preferir una clave de generacion o
envelope como identidad primaria y mantener `campaign_key` como id logico.
Esquemas exactos y tipos estrictos tambien para sobres y unidades.

El gate de respaldo recibe path y digest: abre el dump, recomputa SHA-256 y solo
entonces permite migracion. Un digest bien formado sin archivo refusa.

### C14. Ingesta continua por outbox

Implementar un outbox local durable y un loader CPU idempotente:

1. cada run/candidato/celda terminal escribe un sobre o evento al outbox;
2. el productor cientifico termina aunque PostgreSQL este temporalmente caido;
3. el loader reintenta, carga una sola vez y conserva recibo;
4. fallos e inconclusos se cargan, no solo positivos;
5. un heartbeat publica ultima ingesta, pendientes, fallidos y retraso;
6. integrar primero `predictor` y los productores T2/M4/B4 ya existentes.

Probar caida/reinicio de PostgreSQL con una base desechable, no con el cubo
real. Tras las pruebas, respaldar el cubo real, aplicar solo migracion aditiva,
backfill corregido y arrancar el loader CPU sin reiniciar PostgreSQL ni
Metabase. Demostrar segunda corrida sin cambios y consultar los conteos.

### C15. Censo e indice dentro del cubo

Cargar aditivamente apariciones, variables conceptuales, series publicas,
generadores y decisiones de elegibilidad. Conservar separados `event_time` y
`available_time`, autoridad y estado de metadata. No mezclar estas tablas con
los 99 perfiles historicos ya presentes.

## 7. DOIN y aceptacion

### C16. Resolver el supuesto punto de llamada

Inventariar refs locales/remotas y superficies de `doin-domains` y
`doin-plugins`. Si existe una ruta real de genes L2 trading, documentarla y
proponer el cableado sin ejecutarlo. Si no existe, publicar
`TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED`. Un `__pycache__` no es modulo y
un helper sin caller no es integracion.

## 8. Bateria minima de aceptacion

Ademas de los 12 PRE muertos por su razon exacta:

1. optimizador, pipeline y operador no alcanzan datos antes de la reja;
2. sujeto vacio, extra, omitido o renombrado refusa;
3. manifest autoemitido no concede revision;
4. sustituir data/codigo/particiones/evidencia refusa en punto de uso;
5. primer censo digiere 1.680/1.680; segundo conserva 1.680 digests; mutacion
   de igual longitud se re-digiere;
6. contratos de disponibilidad alcanzan exactamente sus variables;
7. indice contiene filas de las 1.965 variables y su cardinalidad se rederiva;
8. MTM corto/constante no ajusta fuera de train;
9. colision de campana, adjudicacion fabricada y respaldo ausente refusan;
10. outbox sobrevive DB caida, restart y duplicado sin perder ni duplicar;
11. OLAP conserva 39 experimentos y 1.404 hechos historicos;
12. ninguna fila anterior se borra; las 60 traducidas quedan explicitamente
    no autoritativas y supersedidas.

Ejecutar suites focales y completas pertinentes en tips finales. Las cuentas se
toman del terminal despues del ultimo commit; no se estiman.

## 9. Retorno

Entregar un solo packet con:

- tips PRE/POST en los seis repositorios;
- tabla C1-C16 con evidencia y tests;
- censo fisico, cobertura, costo y delta;
- lista exacta de familias de disponibilidad instanciadas y excluidas;
- cardinalidades del indice por tipo y autoridad;
- esquema/migracion OLAP, conteos antes/despues y estado del loader;
- disposicion de las 60 filas historicas traducidas;
- gaps DOIN reales;
- faltas propias y efectos no ejecutados.

**Stop final:** no GPU, live, venue, confirmacion, seleccion cientifica ni genes
DOIN. La proxima puerta la abre la auditoria de Musashi sobre este retorno, no
una etiqueta del candidato.
