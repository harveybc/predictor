# Orden en cola: data-gov beta C185-C204

**De:** Musashi  
**Para:** General Satoshi  
**Fecha:** 2026-09-13  
**Estado:** `QUEUED_NOT_EXECUTABLE_UNTIL_CURRENT_RETURN_AND_REVIEW`  
**Prioridad:** P0 gobernanza reproducible; CPU y base desechable.

## 0. Condicion de inicio

No interrumpa ni mezcle esta orden con su trabajo C166-C184, la reserva fresca
D2 o el ensayo E2E que ya estan en curso. Primero entregue ese retorno con tips,
conteos y efectos reales. Musashi auditara la identidad final y rebasara esta
orden si su implementacion ya resolvio alguno de los puntos.

La especificacion revisora es
`predictor/docs/integracion_workplan_2026_09_10/08_GOBERNANZA_TRANSVERSAL_DATA_GOV_2026_09_13.md`.
El contrato candidato esta en `data-gov/docs/04_FLOW_V2.md`. La propuesta
candidata esta en
`data-gov/docs/05_PROPUESTA_MUSASHI_BETA_2026_09_13.md`.

## 1. Objetivo

Entregar una beta sencilla que gobierne cualquier ejecucion capaz de cambiar
una decision sin introducir llamadas remotas en el ciclo de aprendizaje:

```text
campaign submission -> verified data receipt -> local run
 -> durable terminal -> batched report -> configured OLAP -> reconciliation
```

Un lote registra su campana una vez. Las pruebas mecanicas pueden ser
`NON_GOVERNING`; jamas conceden elegibilidad.

## 2. Reglas que no se negocian

1. No borre, reescriba ni migre destructivamente el OLAP real.
2. Toda prueba de esquema o recuperacion usa PostgreSQL desechable y puertos
   alternos antes de tocar 5055-5057.
3. No active GPU, live, venue, B4, I5 ni entrenamiento cientifico.
4. No entregue credenciales de PostgreSQL al experimento. El adaptador del
   servidor escribe en el cubo local o remoto.
5. No hay llamadas data-gov por observacion, batch, paso de entorno ni
   gradiente.
6. Todos los terminales se conservan: exito, fallo, inconcluso, rechazo y
   cuarentena.
7. Un resultado no gobernante no se promueve por una etiqueta escrita por el
   productor.
8. La autorizacion del propietario para reiniciar 5055, 5056 y 5057 ya existe,
   pero solo despues de que el E2E desechable y esta bateria cierren. No
   autoriza reiniciar PostgreSQL, Metabase ni ningun servicio adicional.

## 3. Descenso de diseno antes de editar codigo

### C185 - Estado persistente y trazabilidad

Crear en `data-gov/docs/` un `PROJECT_METHOD_STATE.json` y una matriz
requisito-caso-prueba-componente-evidencia. Incorporar como requisitos los diez
criterios de salida de la especificacion Musashi. Registrar estado y accion
siguiente despues de cada bloque.

### C186 - Casos de uso

Definir, antes de implementar:

- campana de cien unidades que comparte tres datasets;
- reutilizacion de cache en un segundo proceso;
- unidad completada, fallida, inconclusa, rehusada y en cuarentena;
- data-gov o cubo ausente despues del computo;
- dataset que cambia, corte concurrente y contrato temporal ambiguo;
- productor DOIN que reporta un lote desde su registro;
- prueba mecanica `NON_GOVERNING` que intenta conceder elegibilidad.

### C187 - Pruebas alfa, sistema, integracion y unidad

Congelar primero las pruebas y su resultado PRE. Deben incluir los
contraejemplos C188-C197. No escriba prosa de correccion hasta haber leido las
mutaciones en terminal.

## 4. Contraejemplos y correcciones P0

### C188 - Registro real de campana

**PRE:** un actor elige libremente `experiment_key` o `experiment_set_key`,
descarga y publica sin manifiesto previo.

**POST:** submission direccionada por contenido con actor, proyecto, clase
`GOVERNING|NON_GOVERNING`, identidad de codigo/configuracion, unidades,
solicitudes de datos, destino de terminales y reglas temporales. La politica
autoriza automaticamente la submission; no hay aprobacion humana por corrida.
Una unidad ajena o un set inventado rehusa.

### C189 - Recepcion verificada y cache honesta

**PRE:** cerrar la respuesta despues de los headers o usar un cache hit deja un
`download allow` capaz de verificar linaje.

**POST:** `delivery_authorized` no concede linaje. El cliente confirma un
`delivery_id` solo despues de rehashear los bytes recibidos o la copia en
cache; nace `delivery_verified`. Cache y transferencia son eventos distintos.
Un cuerpo incompleto no puede respaldar un terminal.

### C190 - Snapshot fisico y corte write-once

Congelar ataques de misma ruta/tamano/`mtime_ns`, sustitucion entre inspeccion
y lectura, y dos procesos materializando el mismo corte. Consumir un descriptor
retenido; ligar dispositivo, inodo, tamano, `mtime_ns`, `ctime_ns` y digest.
Publicar el corte con eleccion exclusiva. Si ya existe, verificar digest
completo; una diferencia rehusa. Eliminar `exists -> replace` como autoridad.

### C191 - Contrato temporal por recurso

Retirar autodeteccion como autoridad gobernante. El manifiesto del recurso
declara `event_time`, `available_time`, zona, unidad de epoca, frecuencia,
semantica de limites y politica de ausencias. El holdout usa
`available_time`. Probar offsets, DST, epoch mal tipado, barras truncadas,
publicacion retardada y columnas intercambiadas. Un recurso ambiguo queda
`UNAVAILABLE`, no se adivina.

### C192 - Terminal universal

Sustituir el reporte solo-de-metricas por un terminal exacto por unidad:
estado, razon, inicio/fin, costos, identidad de campana/codigo/config, recibos
de datos, metricas opcionales y digest propio. Estados admitidos:
`COMPLETED`, `FAILED`, `INCONCLUSIVE`, `REFUSED`, `QUARANTINED`. Duplicados
identicos son idempotentes; dos terminales distintos para la misma generacion
rehusan y quedan para adjudicacion.

### C193 - Outbox durable

El terminal se escribe localmente de forma atomica antes de reportar. Caida
del servicio, del adaptador OLAP o del proceso deja `PENDING`; el loader retoma
sin duplicar. Reutilice el patron de outbox ya validado en CRISP-DM en lugar de
crear un segundo protocolo incompatible. La ciencia termina aunque la carga
quede pendiente.

### C194 - Identidad canonica de metricas y codigo

Definir una clave completa y unica de metrica; rechazar duplicados, bool como
numero, NaN/inf, schemas extra y orden ambiguo. La serializacion debe ser
invariante a permutacion. Una corrida `GOVERNING` exige commit limpio o
manifiesto exacto por archivo; `-dirty` solo puede producir
`NON_GOVERNING`.

### C195 - Reconciliacion cubo-accounting

Implementar `orphan_report` antes de declarar uso obligatorio: comparar
terminales del cubo, submissions, receipts y accounting. No borra ni repara en
silencio; clasifica faltantes y permite reenvio idempotente desde outbox.

## 5. Integraciones P1

### C196 - Predictor

Corregir `tools/governed_run.py` para que:

- consuma una submission de campana, no claves libres;
- derive inputs de un manifiesto explicito, conservando el adaptador actual de
  seis campos solo como compatibilidad tipada;
- confirme entregas/cache;
- escriba terminal incluso cuando predictor falle o no produzca metricas;
- use el outbox comun y nunca sobrescriba resultados commiteados.

### C197 - Productores de datos, preprocesamiento y features

Definir el contrato de artefacto derivado: padres, operador y digest de codigo,
parametros, estado ajustado, particion, evento/disponibilidad, output digest y
licencia. `financial-data` conserva la propiedad de datos y cortes.
`preprocessor` y `feature-eng` publican artefactos; no copian CSV como
autoridad. No implemente features F1-F5: el estado DGPD aun no ha llegado a
S8.

### C198 - DOIN y estrategia

Probar adaptadores que produzcan el mismo terminal. DOIN lee su registro por
lotes y conserva identidad de bloque/transaccion/candidato; el destino OLAP se
configura del lado data-gov. `heuristic-strategy` usa el cliente comun. Si un
punto de llamada no existe, entregue contrato y prueba pendiente; no lo
fabrique en una ruta muerta.

## 6. Ensayo integral y operacion

### C199 - E2E desechable

En puertos 5065-5068 y PostgreSQL desechable ejecutar:

1. una campana de 100 unidades y tres datasets;
2. descarga, cache en proceso fresco y un miembro no declarado;
3. los cinco estados terminales;
4. caida y recuperacion del servicio y del cubo;
5. reenvio repetido y proceso muerto entre escritura OLAP y accounting;
6. corte concurrente con dos procesos;
7. contratos temporales validos e invalidos;
8. predictor, fixture DOIN y fixture de estrategia.

Resultado requerido: poblacion exacta, cero duplicados, cero terminales
perdidos y reconciliacion completa.

### C200 - Costo

Medir llamadas, bytes, latencia, disco y memoria por fase. Probar
estructuralmente cero llamadas remotas en loops. Proponer y sellar el umbral de
overhead de beta antes de la primera campana cientifica; no maquillarlo despues
de medir.

### C201 - Despliegue acotado

Solo tras C199-C200 verdes: verificar respaldos y salud, reiniciar 5055-5057
con los mismos `serve.sh`, comprobar `/healthz`, schemas aditivos y version de
codigo. Si sus reglas le impiden reiniciar, emitir runbook exacto para el
operador. No tocar PostgreSQL ni Metabase.

### C202 - Alpha real pequeña

Ejecutar una corrida CPU corta de predictor por el camino gobernado. Comparar
metricas con el camino anterior, verificar el terminal en OLAP y confirmar que
un segundo envio no agrega filas. No usar el resultado cientificamente.

### C203 - Puerta de obligatoriedad

Emitir un veredicto beta. Solo `BETA_ACCEPTED` permite aplicar la regla a nuevas
campanas D3-F5. Si falla, el trabajo cientifico puede continuar con identidades
ya selladas, pero ninguna campana nueva puede llamarse gobernada por data-gov.

### C204 - Retorno

Paquete unico con PRE/POST, mapa requisito-prueba-evidencia, commits por repo,
conteos del tip final, costo medido, filas OLAP antes/despues, estado de
servicios, faltas propias y bloqueadores por objeto/poseedor/resolucion minima.
No emita records en nombre de Musashi.

## 7. Aceptacion minima

- 10/10 criterios de salida de la especificacion Musashi.
- 100/100 unidades terminales en el E2E.
- 5/5 estados visibles en el cubo.
- cero resultados sin submission y `delivery_verified`.
- cero diferencias no explicadas tras recuperacion y reenvio.
- schemas aditivos; tablas historicas byte/logicamente intactas.
- cero llamadas de gobernanza dentro del aprendizaje.
- servicios reales solo reiniciados despues de la reja y con salud comprobada.
