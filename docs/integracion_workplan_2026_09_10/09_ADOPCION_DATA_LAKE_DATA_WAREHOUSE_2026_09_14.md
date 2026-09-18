# Plan obligatorio: crear y adoptar data-lake y data-warehouse

Fecha: 2026-09-14. Decisor: Harvey. Ejecutor: Satoshi. Revision: Musashi.
Estado actual: repositorios publicos y hosts desplegados; catalogo sintetico
ampliado activo y cuatro consumidores probados en produccion. Siguen las
pruebas causales del procesamiento, el replay DOIN y la semantica financiera.

Actualizacion de Musashi tras el retorno `329af81`: ver
[acta productiva](../handoffs/MUSASHI_STORE_HOSTS_PRODUCTION_ACCEPTANCE_2026_09_14.md).
La orden anterior de publicacion queda satisfecha en ese alcance. Continuacion:
[contratos y adopcion](../handoffs/MUSASHI_TO_SATOSHI_CONTRACTS_AND_CONSUMER_ADOPTION_2026_09_14.md).

## Encaje en el plan vigente

Esta es una enmienda operativa al [plan maestro CRISP-DM](https://github.com/harveybc/predictor/blob/182bf89fa0754e3b7b2cea208621531331af931e/docs/integracion_workplan_2026_09_10/00_PLAN_MAESTRO_CRISP_DM_DATA_CENTRIC.md)
y al [plan de integracion Flow v3](https://github.com/harveybc/predictor/blob/182bf89fa0754e3b7b2cea208621531331af931e/docs/integracion_workplan_2026_09_10/08_PLAN_INTEGRACION_FLOW_V3_AGENT_MULTI_DOIN_LIVE_2026_09_14.md).
No sustituye sus requisitos cientificos ni sus resultados historicos.

La creacion de los hosts y su puesta en uso son una etapa inicial obligatoria
de infraestructura. Se ejecutan ahora, en paralelo al cierre de resultados
anteriores. Antes de iniciar nuevas campanas cientificas, los consumidores
deben usar los hosts nuevos mediante data-gov. Los cierres y diagnosticos ya
ordenados pueden terminar en la infraestructura vigente; no se interrumpen
trabajos en curso ni se reescriben sus resultados.

## Responsabilidades

| Componente | Responsabilidad |
|---|---|
| data-gov | Un unico sistema de politicas, entregas, recibos y contabilidad |
| data-lake, nuevo repositorio | Servicio configurable, inventario, consola AdminLTE e interfaz de plugins para lagos |
| financial-data | Primer plugin externo del lago financiero y sus datos/contratos de origen |
| data-warehouse, nuevo repositorio | Servicio configurable, inventario de tablas/vistas, consola AdminLTE e interfaz de plugins para warehouses |
| predictor | Primer plugin externo del cubo OLAP existente, sin duplicar la base de datos |
| Satoshi | Crear los repositorios, implementar, probar, desplegar y registrar adopcion |
| Musashi | Revisar evidencia y corregir problemas de integracion; no imponer una nueva decision del owner ya concedida |

La configuracion selecciona plugins instalados mediante entry points de
Python. Los paquetes usan namespaces distintos. Una diferencia representable
por configuracion no justifica otro plugin. El [diseno de interfaces](https://github.com/harveybc/data-gov/blob/ff4503a4c7f1d950e47ab43894f176962025faa6/docs/STORE_PACKAGES_DESIGN.md)
detalla la separacion y las pruebas.

## Etapas y evidencia de cierre

| Etapa | Entregable y criterio de terminacion | Estado (2026-09-14, Satoshi) |
|---|---|---|
| 1. Crear | Ambos repositorios publicados con URL real, README, AGENTS.md, requisitos, pruebas previstas y estado persistente | Completado: publicos; proveedor OLAP integrado en master por PR 44 |
| 2. Implementar | Hosts instalables y plugins externos desde financial-data/predictor, probados como wheels en entornos limpios | Completado en las revisiones instaladas del acta; pruebas independientes 33+28+4 |
| 3. Integrar | Paridad con las APIs actuales: inventario, bytes, disponibilidad temporal, particiones, recibos, resultados y reintentos | Paridad previa 8/8 y 11/11; E2E desechable re-ejecutado por Musashi |
| 4. Probar interfaz | Configuracion e inventario AdminLTE, esquema de recursos, escritorio y movil; pruebas de sistema y aceptacion | Pruebas y capturas publicadas por Satoshi; suites de consola re-ejecutadas por Musashi |
| 5. Poner en uso | Transicion controlada con los mismos datos, IDs y cubo; microexperimento gobernado por ambos hosts y conciliacion exacta | Completado en infraestructura y transporte sintetico; 5.275 recursos conservados, terminal y cuatro metricas en cubo real |
| 6. Adoptar | Configuraciones de consumidores actualizadas y cobertura documentada por proyecto; nuevos experimentos usan esta ruta | Mecanica productiva probada: preprocessor, feature-eng, feature-extractor y predictor. Pendientes: causalidad de procesamiento y replay offline agent-multi/DOIN |

Satoshi actualiza cada etapa al terminarla, con commits, pruebas y evidencia.
Los resultados esperados no cuentan como evidencia. La metodologia sigue
requisitos y pruebas de arriba hacia abajo, e implementacion y verificacion
de abajo hacia arriba. El estado y el siguiente paso se guardan en archivos,
no solo en la memoria de la conversacion.

## Adopcion de experimentos

El primer uso operativo debe probar todo el recorrido:
experimento -> data-gov -> data-lake/plugin -> datos identificados ->
resultado -> data-gov -> data-warehouse/plugin -> cubo y conciliacion.

Cada consumidor registra datos o generador sintetico, semilla, hashes de
artefactos, version de codigo, configuracion, costos y desenlace. Un test
unitario local no necesita levantar los servicios; un resultado que se use
como evidencia experimental si necesita trazabilidad y registro reproducible.

La adopcion cubre preprocessor, feature-eng, feature-extractor, predictor y
los experimentos offline de agent-multi/DOIN. Se reporta por consumidor:
implementado, probado, desplegado y usado realmente. Ninguna integracion se
declara completa por tener solo un wrapper o un README.

Para live se preparan configuracion, pruebas offline y compatibilidad. Esta
enmienda no activa trading ni agrega llamadas remotas al bucle de ejecucion.
Las condiciones de activacion operativa y validacion causal siguen vigentes.

## Continuidad y limites

- No mover ni duplicar datasets para adaptarse a los repositorios nuevos.
- No recrear el cubo ni borrar el historial o los pendientes del outbox.
- Preservar IDs de recursos, contratos temporales y semantica de APIs.
- Mantener los servicios actuales hasta que los sustitutos pasen las pruebas.
- La transicion y el micro-run forman parte del trabajo encargado, no de una
  recomendacion futura. Si fallan, corregir y documentar; no dar por adoptado.
- Este cambio de infraestructura no demuestra utilidad de preprocesamiento,
  ausencia universal de fuga temporal ni rentabilidad de un modelo.

Orden ejecutiva: [trabajos de Satoshi](../handoffs/MUSASHI_TO_SATOSHI_GOVERNED_D2_AND_STORE_HOSTS_2026_09_14.md).


## Matriz de adopción por consumidor (2026-09-14)

Estados: `IMPLEMENTADO` (existe el envoltorio gobernado), `PROBADO` (corrió bajo gobernanza
con recibo), `DESPLEGADO` (su ruta apunta a los hosts nuevos), `USADO` (una campaña real lo
usó). Ningún consumidor cambia de URL: todos hablan con data-gov en :5055, y la sustitución
ocurre por detrás.

| consumidor | implementado | probado bajo gobernanza | desplegado en la ruta nueva | usado realmente |
|---|---|---|---|---|
| predictor | sí (`tools/governed_run.py`) | sí — microexperimento de producción de Musashi (`n3-mechanics-…`) y D2-R3/R4 (campañas `733736b7…`, `aef32e87…`) | pendiente de la etapa 5 | sí, en el cubo real |
| feature-eng | sí (`tools/governed_run.py`) | sí — `p5_feature_eng_throwaway.out`, stack desechable | pendiente de la etapa 5 | no en producción |
| feature-extractor | sí (`tools/governed_run.py` + port de la API) | sí — `p6_feature_extractor_throwaway.out`, stack desechable | pendiente de la etapa 5 | no en producción |
| preprocessor | sí | sí — `p1_preprocessor_throwaway.out` | pendiente de la etapa 5 | no en producción |
| agent-multi / DOIN (offline) | no — falta el contrato `doin_governed_result.v1` y su fixture | no | no | no |

Lo que falta por consumidor, declarado: feature-eng, feature-extractor y preprocessor no
tienen todavía una campaña **de producción** (solo desechable); agent-multi/DOIN no tiene
contrato de resultado gobernado. Para live únicamente configuración y replay offline: esta
enmienda no activa operaciones reales.

## Bloqueos declarados (no son decisiones pendientes del owner)

1. **Visibilidad pública** de los dos repositorios: la orden la decidió; el entorno de
   ejecución de este agente rechaza crear o convertir superficie pública
   (`gh repo create --public`, `gh repo edit --visibility public`). Los repositorios existen
   como privados con todo el contenido; un solo comando del owner los hace públicos.
2. **Transición productiva** (etapa 5): detener y arrancar servicios está rechazado por el
   mismo mecanismo. Todo lo previo está hecho y verificado; el procedimiento exacto, con su
   reversión, está en el packet de retorno.
3. **`predictor-olap-store` en la rama por defecto**: el empuje directo a `master` fue
   rechazado; va como PR #44, y la revisión fijada queda declarada como origen instalable
   mientras se integra.


## Actualizacion 2026-09-14 (Satoshi, tras el acta productiva de Musashi)

Etapas 1-5: **cerradas por Musashi** (repos publicos, PR 44 integrado, hosts sirviendo con
servicios persistentes, micro-run productivo conciliado). Verifique la identidad que sirve
realmente en `/api/v1/host` de cada puerto, el inventario 5.275 y que el cargador **avanza**
(latido publicado en el instante de la comprobacion, `healthy: true`, pendientes 0), no solo
que el proceso este activo. Dos detalles registrados: `governance_smoke` no aparece en
`/api/v1/lakes` porque su politica concede `download` y no `discover`, y el almacen agrega
un campo aditivo `transport` en `describe` que el adaptador anterior no enviaba.

### Etapa 6 — adopcion por consumidor (estado real)

| consumidor | entrega gobernada | exito | salida rancia | fallo con costo | reintento sin duplicado | alcance |
|---|---|---|---|---|---|---|
| preprocessor | VERIFIED_TRANSFER | **COMPLETED** | REFUSED | FAILED | si | **completo** |
| predictor | 6 entradas verificadas | **COMPLETED** | REFUSED | FAILED | si | **completo** |
| feature-eng | VERIFIED | su pipeline rechaza las columnas de muestra | REFUSED | FAILED | si | solo transporte |
| feature-extractor | 6 entradas verificadas | idem | REFUSED | FAILED | si | solo transporte |
| agent-multi / DOIN | — | — | — | — | — | sin empezar |

Recibos: `docs/audits/evidence/repro_runs/adoption_20260914/`. Todas las campanas
NON_GOVERNING; 12 terminales aditivos, conciliacion exacta, sin filas duplicadas.

Falta por consumidor, declarado: feature-eng y feature-extractor necesitan que sus fixtures
propios se publiquen como recurso gobernado con contrato derivado; agent-multi/DOIN necesita
el contrato `doin_governed_result.v1` y su fixture, en sus repositorios.

### Contratos de disponibilidad financiera

Primer candidato entregado como **financial-data PR #1**
(`market_data/crypto/spot_top50/ethusdt/4h.parquet`, `contract_sha256` `998e3f80…`), con la
evidencia del productor, once pruebas sobre bytes y tres desconocidos declarados
(latencia de entrega UNOBSERVED, politica de revisiones UNKNOWN, derechos de uso UNKNOWN).
**No instalado**: instalarlo cambia lo que una campana puede consumir y pasa por revision.


## Actualizacion 2026-09-14 (Satoshi, tras el dictamen A1-A5)

**Semantica temporal corregida.** El contrato candidato con `close_time` como columna de
disponibilidad y lag cero queda **retirado**: el cierre de una ventana no es disponibilidad.
El recurso ETH 4h se caracteriza como **archivo retrospectivo** (publicacion y recepcion
UNOBSERVED, finalizacion NO DEMOSTRADA, revisiones UNKNOWN), con las 21 barras anomalas
clasificadas desde los bytes (1 marcador vacio, 12 intervalos nominales mas cortos, 8
agregados parciales) y ninguna declarada final. Extension minima propuesta para poder decirlo
sin inventar un numero: `use_class: ARCHIVE_RETROSPECTIVE` con `completion_lag_max: "UNKNOWN"`,
entrega entera o nada, rango rechazado. financial-data **PR #2**; nada instalado.

**Etapa 6 — ruta real probada por consumidor.** Con fixtures deterministas
(`tools/make_consumer_fixtures.py`, semilla 20260914, siete archivos con los esquemas que los
consumidores realmente leen) servidos por **data-lake + proveedor externo por http**, los
cuatro consumidores producen salida real en stack desechable:

| consumidor | entrega gobernada | salida del pipeline | rancio | fallo con costo | reintento |
|---|---|---|---|---|---|
| preprocessor | VERIFIED_TRANSFER | producida | REFUSED | FAILED | sin duplicado |
| feature-eng | VERIFIED_TRANSFER | producida | REFUSED | FAILED | sin duplicado |
| feature-extractor | VERIFIED_TRANSFER | producida | REFUSED | FAILED | sin duplicado |
| predictor | VERIFIED (transfer + cache) | producida | REFUSED | FAILED | sin duplicado |

Control negativo: con el host de entrada detenido, la misma corrida falla en la entrega con
`503 lake unreachable`. Reintento real (N5): terminal varado por caida del destino durante la
corrida, recuperado por el outbox del wrapper, **una sola fila** en el cubo, segundo vaciado
sin envio y recibo de cache en la segunda corrida.

**Pendiente de activacion:** el catalogo sintetico ampliado esta escrito como configuracion
**pendiente** (`5058.pending.json`); la activa sigue sirviendo un solo recurso. Activar =
promover ese archivo y reiniciar solo `crispdm-data-lake-synthetic`; este agente no puede
controlar procesos de servicios vivos. Tras la activacion, repetir los cuatro consumidores
contra produccion es una sola ejecucion del mismo arnes.


## Actualizacion 2026-09-14 (Satoshi, ordenes vigentes P1-P7, continuacion)

**Activacion resuelta por el owner**, no por este agente: catalogo sintetico activo con 8
recursos y los cuatro consumidores comprobados en produccion. No se repite ninguna de esas
corridas ni el reinicio; el estado se toma del acta de aceptacion y se verifica contra el cubo.

**Roles de columnas (P1) — ahora en los tres consumidores.** El mismo contrato explicito
(columnas declaradas y en el orden declarado; columna no declarada rechazada por nombre;
columna declarada ausente rechazada por nombre; feature no numerica detenida antes del tensor;
corrida sin contrato rechazada salvo `column_roles_migration` declarado; plan y digest
registrados) rige tambien en **feature-eng** (`satoshi/column-roles-20260914`, `e7130f8`) y
**preprocessor** (`satoshi/column-roles-20260914`, `89c83dd`), once reglas verdes en cada uno.

**Replay offline DOIN (P5) — ejecutado en produccion.** `doin-offline-replay-prod-3`
COMPLETED, NON_GOVERNING, entrega `synthetic_features_4h_train.csv` VERIFIED_TRANSFER
(contrato `41e67f99…`), metricas `wall_seconds 7.11` y `total_timesteps 64` en el terminal,
conciliacion vacia en las tres listas. El intento previo (`prod-1`) completo **sin metricas**:
el replay escribia los numeros anidados y el colector lee nombres aplanados; ambos recibos se
conservan porque el primero es lo que mostro el defecto. Recibo:
`docs/audits/evidence/repro_runs/doin_offline_20260914/GOVERNED_OFFLINE_REPLAY_PRODUCTION.json`.
Con esto la fila `agent-multi / DOIN` de la Etapa 6 deja de estar "sin empezar".

**Arnes (P4).** El parser real y el punto de entrada real quedan ejercitados en las pruebas de
clasificacion (`build_parser()` + `main()` con el cliente HTTP sustituido, 8 reglas); el
registrador ya no infiere ruta desde un nulo: marca **no observable** con su razon. Queda
abierto: la prueba de recuperacion del outbox por cada wrapper que no comparte implementacion.

**Investigacion temporal financiera (P6).** Respondida desde el codigo del propio productor
(`_scripts/workers/stage13_dragon_crypto_worker.py`): REST historico paginado contra
`api.binance.com/api/v3/klines`, cota fija `END_MS = 2025-12-31T23:59:00Z`, ambas columnas de
tiempo son epoch-ms convertidos con `utc=True`, sin hora de llegada por fila. La recepcion pasa
a `BOUNDED_AT_FILE_GRAIN` (cota `2026-05-01T15:58:56.166Z`); la publicacion sigue UNOBSERVED y
las 21 barras anomalas siguen sin explicacion de construccion. financial-data `8ec5e16`.
Quedan dos preguntas: revisiones (medibles con un re-fetch acotado, llamada saliente,
**propuesta no ejecutada**) y derechos de uso (unica accion del owner).

**Cubo medido tras el ultimo replay: 35 terminales**, historia intacta, servicios `:5055/:5056/
:5057/:5058` sanos y sin reinicios.


## Actualizacion 2026-09-15 (Satoshi, ordenes R1-R6 ejecutadas)

Retorno completo en `docs/audits/work_plan/SATOSHI_R1_R6_RETURN_2026_09_15.md`. Resumen para
este plan, con lo que cambia en la adopcion:

**Las tres maquinas consumen por data-gov.** Entrega gobernada probada desde omega, gamma y
dragon contra los hosts reales (tunel inverso desde omega, porque los servicios escuchan en
`127.0.0.1`), verificada contra `X-Content-SHA256`. La segunda campana de gamma **no
transfirio nada**: confirmo contra la cache por contenido. Es decir, "una descarga por grupo de
experimentos" ya es conducta desplegada. Ninguna maquina necesita copias locales de datasets.

**Etapa 6, fila agent-multi/DOIN corregida.** `doin-offline-replay-prod-11` COMPLETED,
NON_GOVERNING, conciliacion vacia, 15.7 s. Su recibo declara `requested_timesteps 64` y **no**
declara `observed_timesteps`: la aplicacion no escribio resumen y eso se dice, en vez de
presentar el presupuesto como trabajo hecho. El recibo anterior (`prod-3`) si lo presentaba.

**Adquisicion publica registrada.** La comparacion del archivo ETH contra una segunda toma
publica quedo en la contabilidad como campana `d3cf00b6ea06…`, terminal `2a4b455d7e9a…`,
metricas `bars_compared 3000` y `bars_materially_differing 0`.

**Contratos de disponibilidad.** `ARCHIVE_RETROSPECTIVE` tiene nueve reglas que prueban que
UNKNOWN no se convierte en cero y que un rango se rechaza por su razon declarada; **sigue sin
instalarse**: el proveedor que el host del lake importa hoy no contiene la clase, y una regla
falla el dia que la contenga.

**Pendiente de operador:** los principales por maquina (`satoshi-gamma`, `satoshi-dragon`) ya
estan escritos en la configuracion de ejecucion pero **no cargados**; cargarlos es
`systemctl --user restart crispdm-data-gov.service`, que esta sesion no puede ejecutar.
## Revision del retorno A1-A5

Musashi verifico los doce desenlaces en el cubo y la salud de los servicios.
Preprocessor y predictor completaron sus pruebas mecanicas con el adaptador
local predictor_examples; esto no completa aun la adopcion del host de
entrada nuevo. Feature-eng y feature-extractor probaron transporte, no exito
de sus pipelines. PR 1 financiero conserva valor descriptivo, pero no se
instala: cierre de barra no demuestra disponibilidad.

La investigacion financiera y los fixtures sinteticos avanzan en paralelo.
No se requiere una nueva decision del owner para empezar la continuacion.
Ver [dictamen](../audits/work_plan/MUSASHI_REVIEW_A1_A5_CONTRACT_AND_ADOPTION_2026_09_14.md).

## Activacion completada tras el retorno N1-N8

Musashi activo solo el lago sintetico y ejecuto los cuatro consumidores
contra produccion: doce terminales conciliados, 119 metricas, datos y
artefactos ligados. La activacion NO esta pendiente del owner.
Ver [acta](../handoffs/MUSASHI_SYNTHETIC_CATALOG_AND_FOUR_CONSUMERS_ACCEPTANCE_2026_09_14.md).
El exito mecanico no demuestra todavia fit solo en train ni ausencia de fuga
en transformaciones; esos trabajos y el replay DOIN siguen ahora en paralelo
a la investigacion financiera.

Orden ejecutiva vigente: [causalidad del pipeline y replay offline DOIN](../handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md).

## Revision independiente de la continuacion

Sobre predictor 90d4fcf, Musashi verifico 35 terminales en el cubo y los cinco
servicios activos. El replay agent-multi tiene entrega y metricas persistidas;
su identidad dirty y el origen de los contadores limitan su reproducibilidad.
No demuestra ejecucion distribuida DOIN. No se repite la activacion anterior.

La bateria causal usa operadores de prueba, no los plugins reales. Dos
contraejemplos permiten target implicito y metadata numerica como features.
La validacion causal productiva queda pendiente, no completada por esos tests.

Secuencia ahora: roles reales -> procesamiento e indicadores reales -> replay
reproducible y recuperacion de outbox -> archivo retrospectivo end-to-end.
Investigacion de revisiones y derechos avanza en paralelo sin bloquear lo
sintetico. No se abre seleccion cientifica por completar transporte.

Orden ejecutiva vigente para esta continuacion:
[R1-R6: plugins reales y replay reproducible](../handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md).
Ver [dictamen y pruebas](../audits/work_plan/MUSASHI_REVIEW_STANDING_ORDERS_2026_09_14.md).

## Activacion de trabajadores, 2026-09-15

Musashi completo la carga de identidades y la instalacion privada de llaves
propias en ambos trabajadores. Autenticacion HTTP comprobada desde ambos;
solo data-gov reiniciado, los otros cuatro servicios conservados. La activacion
ya no depende del owner. Falta completar entrega y terminal con la identidad
propia, contadores observados del replay y archivo retrospectivo end-to-end.
El retorno 7c6df0e no sustituye esos entregables con pruebas parciales.

Orden vigente: [activacion y cierre efectivo de R1-R6](../handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md).

## Auditoria de cierre 2574890

Las cuatro entregas y terminales de trabajadores fueron contrastadas con SQLite
y PostgreSQL por Musashi: aceptadas como atribucion en data-gov, no identidad
independiente en el store. Los tokens por entrega siguen diferidos.
La aritmetica de PPO explica 256 pasos, pero _n_updates mide epocas en PPO,
no llamadas al optimizador. El archivo retrospectivo necesita resolucion
persistente del contrato, y el PDF oficial de terminos ya fue localizado.

Orden de continuacion [S1-S4](../handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md):
contrato de computo exacto, semantica temporal recuperable y fuentes de derechos.
P1LR queda fuera de esta aceptacion; no bloquea estos trabajos ni se lanza.

## Actualizacion 2026-09-15 (Satoshi, ordenes S1-S4)

Retorno completo en `../audits/work_plan/SATOSHI_S1_S4_RETURN_2026_09_15.md`.

**Incidente propio, abierto.** Un `pkill` demasiado amplio al desmontar el stack desechable
detuvo el host de almacen de **produccion** `crispdm-data-warehouse-olap.service` (:5057). El
entorno me niega arrancarlo; el comando es
`systemctl --user start crispdm-data-warehouse-olap.service`. Los otros tres servicios
respondieron 200 todo el tiempo y los terminales quedan retenidos por el outbox durable.

**S1 — contrato de computo.** `_n_updates` NO son llamadas al optimizador: en PPO cuenta
**epocas**. Medido sobre SB3 2.9.0: cuatro rollouts por dos epocas son 8 epocas y **16**
llamadas. Las llamadas se cuentan instrumentando el optimizador real. Un presupuesto y un tope
son campos distintos, y una configuracion cuyo rollout minimo no cabe en el tope se **rechaza
antes** de dar un paso. Un modelo reanudado reinicia su contador (64 -> 64), asi que el delta
ingenuo reporta cero por un entrenamiento real: se **detecta**, no se supone. Corrida sucesora
`doin-offline-replay-successor-1` con identidad **limpia**: 64 pedidas, 64 observadas, 1
rollout, 1 epoca, **1** llamada al optimizador, 384 transiciones de evaluacion, tope respetado,
**21 contadores persistidos en el cubo de produccion**. `prod-12` y `prod-13` intactos.

**S2 — contrato de disponibilidad recuperable.** El hueco que yo mismo reporte en R4 queda
cerrado: `gov_availability_contract` retiene los **bytes canonicos** por su propio digest
(recomputado al escribir) y `gov_delivery_availability` resuelve una entrega o dice
`UNRESOLVED`. Los contratos viajan **al lado** del terminal, nunca dentro: la identidad de un
terminal es el digest de su cuerpo. Probado sobre **PostgreSQL desechable**: archivo entregado
y cerrado; rango, punto-en-el-tiempo y cola viva rechazados 422 con la razon declarada y
cerrados como REFUSED; recurso normal entregado como regresion. Despues se **mato** el host del
lago y se **borro** su configuracion, y solo entonces un lector fresco respondio
`ARCHIVE_RETROSPECTIVE` / `UNKNOWN` y **reverifico el digest**. Hallazgo no previsto: con
holdout declarado, un archivo retrospectivo se rechaza siempre — su publicacion nunca se
observo. **No desplegado**: el paquete declara su divergencia en `PENDING_REVIEW` en vez de
mover su pin.

**S3 — terminos.** El documento es el **ADGM Binance Global Terms of Use**, vigente **21 de
julio de 2026**: rige el uso de una **cuenta** Binance, es **posterior** a la adquisicion del
1 de mayo, y su clausula 14.1.2(b) remite a **terminos de API separados** que no contiene.
Se retira la inferencia de que el uso interno sin ingresos queda fuera de las restricciones.
Verificado en vez de repetido: la afirmacion de que no hay filas de mercado publicas es
**falsa** — `financial-data` es PUBLICO y tiene **47.233** barras en seis CSV. Lo que si es
cierto y es lo unico que se afirma: el recurso gobernado del lago **no** esta publicado.

**S4 — P1LR.** La herramienta **no escribe** archivo de veredicto, asi que la ruta es una
convencion del operador y las **dos** declaradas estan mal; la canonica es la raiz de replica
del contrato. En dragon el temporizador de guardia esta **activo** cada 15 minutos y ha fallado
**130 veces en 24 horas**: disposicion **dormante**, comandos nombrados, nada aplicado. Plan
costeado: **640.000 transiciones** exactas; reloj y memoria **no** estimados, se miden con una
celda piloto.
## Recuperacion del warehouse tras S1-S4

Musashi arranco el warehouse productivo tras la terminacion accidental reportada
por Satoshi. Salud HTTP confirmada, 52 terminales observados y sucesor con 21
metricas; cargador sin reinicio. Falta conciliacion explicita de los outboxes del
intervalo, no se presume ausencia de perdidas.

P1LR queda efectivamente dormante: temporizador y arranque de decision
deshabilitados, fallos y diarios conservados. La dimension temporal no esta
desplegada: 10 tests focales pasan, pero el lector aun no revalida el digest
que presenta como RESOLVED. Corregir antes de aprobar la migracion.

Orden vigente: [U1-U4: recuperacion, lector y procedencia publica](../handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md).

## Actualizacion 2026-09-15 (Satoshi, ordenes U1-U4)

Retorno en `../audits/work_plan/SATOSHI_U1_U4_RETURN_2026_09_15.md`.

**Almacen restaurado.** Musashi arranco `crispdm-data-warehouse-olap.service`; :5057 responde
200 y los cinco servicios estan activos sin reinicios automaticos. La caida fue mia y su
contabilidad esta cerrada: **cero** envelopes escritos en el intervalo, sobre 11 raices de
outbox y 97 envelopes, corroborado por el journal de data-gov. El unico pendiente era
**anterior** a la caida y su ranura ya estaba cerrada en el cubo — marcador rancio, no
perdida. Vaciado con la implementacion existente: pendientes 1 -> 0, **una sola fila**,
`received_at` intacto, **52 terminales**. Dieciocho envelopes de stacks desechables del 13-sep
quedan declarados **no verificables**, no reconciliados: sus cubos ya no existen.

**Desmontaje corregido.** El arnes ya no permite seleccionar por nombre de modulo: senala al
**grupo de procesos** propio y rechaza cualquier PID cuya linea de comando no lleve el marcador
del stack (`PID_REUSED_REFUSED`). Un zombi ya no se lee como proceso vivo. Siete reglas, entre
ellas dos stacks con **los mismos nombres de modulo** donde desmontar uno deja al otro
sirviendo y con sus datos.

**Lector de archivo corregido (defecto real de Musashi).** Alterar los bytes retenidos dejando
la clave y las columnas cacheadas intactas hacia que el lector respondiera `RESOLVED` con lag
`UNKNOWN` sobre bytes que decian `0s`. El lector devolvia la fila de la vista, y un `LEFT JOIN`
demuestra que una clave coincide, no que algo se haya verificado. Ahora verifica algoritmo,
canonicalizacion y **bytes**, y deriva de ellos lo que muestra: siete rechazos distintos y solo
`VERIFIED` lleva semantica. La vista dice `STORED`/`ABSENT` y sus columnas son `stored_*`.
**40 pruebas verdes sobre SQLite y PostgreSQL desechable.** Candidato de produccion con
revisiones exactas, migracion aditiva, respaldo, comprobaciones de no-cambio-de-historia y
reversion: `SATOSHI_S2_PRODUCTION_CANDIDATE.md`. **Nada desplegado**; el pin sigue nombrando la
revision desplegada.

**Filas publicas.** 47.233 confirmadas por medicion independiente, pero la union de dias
calendario distintos es **4.125**: cuatro de los seis archivos son la misma serie BTC a tres
resoluciones mas una tabla derivada. Procedencia establecida por **codigo publicado**:
`fetch_binance.py` contra `/api/v3/klines` y `consolidate_data.py` leyendo su salida para los
cuatro `btcusd_*`; los dos diarios de `feature_store` son de **Yahoo** y empiezan tres anos
antes de que existiera BTCUSDT, asi que quedan fuera de la pregunta. Ninguno es el recurso
gobernado del lago. Retirada mi inferencia de que unos terminos de cuenta no alcanzan una
lectura sin credencial: la misma frase alcanza "any other Binance Services".
## DuckDB para el warehouse OLAP - 2026-09-16

El owner autoriza reemplazar PostgreSQL exclusivamente como motor del cubo OLAP,
sin afectar otras bases o servicios. La implementacion y el despliegue completos
se asignan a Satoshi: no requieren permisos incrementales por cada bloque.
Estado: ORDENADO, no implementado ni desplegado por este cambio documental.

Orden vigente: [D0-D6: migracion autorizada a DuckDB](../handoffs/MUSASHI_DUCKDB_WAREHOUSE_MIGRATION_ORDER_2026_09_16.md).
Extiende [V1-V4](../handoffs/MUSASHI_U1_U4_REVIEW_AND_V1_V4_2026_09_15.md),
conserva sus correcciones semanticas y reemplaza la espera de despliegue de V3
por aceptacion verificable seguida de despliegue autorizado. La evidencia actual
se selecciona por procedencia de campana, no por resultados favorables; el
historico se archiva sin borrarlo y queda fuera de las vistas cientificas por
defecto. Se requieren scripts reproducibles, paridad de contenido, recuperacion,
consumidores reales, analitica utilizable y conciliacion posterior al cambio.

## Actualizacion 2026-09-16 (Satoshi, V1-V4 y D0-D6: DuckDB es el motor OLAP)

Retornos: `../audits/work_plan/SATOSHI_V1_D6_RETURN_2026_09_16.md`. Evidencia:
`../audits/evidence/temporal_semantics_20260916/` y `../audits/evidence/duckdb_migration_20260916/`.

**V1 — semantica temporal.** Musashi tenia razon y el defecto era mio: yo verificaba que los
bytes cuadraran con su digest y trataba eso como validez. Un `ARCHIVE_RETROSPECTIVE` que declara
`0s`, `-1` o `not-a-duration` cuadra perfectamente y volvia como VERIFIED. Reproductor congelado
ANTES de tocar nada; ahora rigen **las reglas del productor**, aplicadas al escribir y otra vez
al leer, mas forma canonica y claves duplicadas. Once reglas de mutacion prueban que cada
comprobacion es la que rechaza. Cero desplegado sin revision: el pin sigue nombrando la revision
desplegada.

**D0-D6 — DuckDB en produccion.** El cubo OLAP corre sobre **DuckDB 1.5.5** en
`~/.local/state/crispdm-duckdb/prod/cube.duckdb`, servido por el host existente con el proveedor
externo `predictor-duckdb-store` por el mismo entry point. **Solo** cambia el almacen OLAP:
Metabase, `fxpg` y cualquier otro uso de PostgreSQL quedan intactos, y la base previa se
conserva como origen de reversion. Archivo historico **segregado** en `archive.duckdb`
(6.860.035 filas, 64/64 verificadas por contenido) y fuera de toda vista cientifica por defecto.

**Tres defectos que solo aparecieron al hacerlo:** `LIMIT/OFFSET` sin orden no es paginacion
(nueve relaciones con el conteo correcto y las filas equivocadas); `CREATE TABLE AS SELECT` no
conserva constraints, asi que `gov_terminal` quedo sin clave primaria y el cubo **respondia
consultas y rechazaba todo terminal gobernado**; y copiar el fichero sin su WAL no es respaldo
—la copia omitio dos terminales recien aceptados—. Los tres corregidos y con regla.

**Aceptacion en produccion:** entrega gobernada y terminal **201**, conciliacion vacia, **54
terminales** en el cubo. Reversion ensayada sobre base desechable: identifica los 2 terminales
de la era DuckDB que se perderian y que `gov_availability_contract` **no existe** en PostgreSQL.

**Brechas declaradas con dueno:** Metabase v0.56.3 no trae driver DuckDB (verificado contra su
propia API), asi que la analitica hoy es la consola del host; y `tools/olap_loader.py`, unico
otro escritor OLAP, queda **detenido y deshabilitado** con su cola en cero.
### Relectura del retorno 68f4e39

Warehouse activo con entorno DuckDB; cargador legado detenido, verificados por
systemd. Cierre integral NO aceptado: seleccion por tablas en vez de linaje de
campana, rollback solo diagnostico, snapshot no coordinado, importacion parcial
sin resume y ruta df_* pendiente. No se constata perdida de datos por esta
revision; se conservan fuente y archivo. Orden de cierre:
[E1-E6](../handoffs/MUSASHI_DUCKDB_CLOSEOUT_CORRECTIONS_2026_09_16.md).

### Retorno d35bb21: cierre de recuperacion pendiente

Warehouse y cargador sucesor activos, verificados por systemd. Dos pruebas
independientes en bases temporales reproducen importacion parcial que duplica
clave y replay que deja hijos ausentes cuando ya existe su padre. El rollback
implementado apunta a DuckDB, no al PostgreSQL anterior; catchup CLI sigue sin
cierre de hijos. WAL productivo intacto durante la auditoria. Continuacion
autorizada: [F1-F5](../handoffs/MUSASHI_E1_E6_REVIEW_AND_F1_F5_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, ordenes F1-F5)

Retorno: `../audits/work_plan/SATOSHI_F1_F5_RETURN_2026_09_16.md`.

**Los cinco hallazgos de Musashi eran correctos.** Los reprodujo llamando a las funciones, no
leyendo mis recibos, y los dos defectos que aisló quedan congelados como pruebas antes de
cualquier arreglo: reanudar una tabla con clave reinsertaba el prefijo ya escrito (y sin clave
lo duplicaba en silencio), y un padre ya presente ocultaba hijos ausentes.

**F1/F2.** La copia reanuda desde la marca del destino; una relacion sin clave con filas
dentro se **rechaza** en vez de anexar. El replay reconcilia padres compartidos, compara hijos
por contenido y **rechaza** una identidad compartida con contenido distinto. La reversion real
es a **PostgreSQL** y quedo ensayada sobre base desechable: el proveedor crea la tabla de
contratos que falta, 2 terminales y 3 filas hijas restauradas, 54 terminales en destino y una
segunda corrida sin efecto. `copy-cube` es ahora la copia DuckDB→DuckDB, con nombre propio.
`catchup` pasa por el mismo cierre, asi que los hijos sin marca de tiempo viajan con su padre.
Igualdad de conteo ya no declara completo, y el trabajo rechazado **sale con codigo distinto de
cero**.

**F3 — correccion importante.** Mi causa raiz del incidente **no se reproduce**: cuatro arreglos
en bases desechables reabren todos. Pasa a **hipotesis**, y el checkpoint a **mitigacion**. Lo
que si queda demostrado es la no-perdida, contra el registro independiente de data-gov: **55
aceptados, 55 presentes, 0 ausentes, 0 discrepancias de estado**; los 10 sin hijos son
exactamente los REFUSED. Los dos WAL en cuarentena son identicos byte a byte y nunca se abrieron.

**F4.** Pertenencia y admisibilidad son dos preguntas y ahora dos columnas, con vistas
ejecutables en el cubo: **73 miembros, 48 operativos, 24 cientificos actuales**. B4, T2 y M4
vuelven a ser **miembros** con su propio alcance, citando la linea del plan maestro que lo
decide; excluirlos por estar revisados en otra reja fue un error mio.

**F5.** Cargador sucesor verificado tras las correcciones con un envelope real: cargado,
consultado por el servicio, reintento idempotente (`skipped_existing`) y outbox conciliado.
### Retorno e99c6ba: progreso probado y alcance del incidente

Las dos reproducciones anteriores ya pasan independientemente. Warehouse y
cargador activos. La conciliacion del incidente todavia acepta metricas alteradas
y todos los hijos ausentes: identidad/estado no equivalen a contenido preservado.
No se afirma perdida productiva. Resume con huecos y snapshot con frontera solo
declarada siguen pendientes. Continuacion acotada:
[G1-G3](../handoffs/MUSASHI_F1_F5_REVIEW_AND_G1_G3_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, ordenes G1-G3)

Retorno: `../audits/work_plan/SATOSHI_G1_G3_RETURN_2026_09_16.md`.

**Correccion mayor: SI hubo perdida.** Mi verificador hasheaba el cubo recuperado y lo comparaba
**consigo mismo**; Musashi lo demostro ejecutandolo tres veces —fixture original, metrica
cambiada a 999999 y tablas hijas vaciadas— con salida 0 en las tres. Reconstruido contra el
payload canonico que retiene la contabilidad de data-gov, encontro **dos terminales a los que
les faltaban cuatro filas de metricas**, ambos escritos en la era DuckDB y con esas filas en el
WAL que yo puse en cuarentena. Reparados desde ese registro independiente —solo se anaden filas
ausentes, bajo un padre cuyos campos ya coinciden, en transaccion— y el cubo concilia ahora
**55/55 sin diferencias**, verificado por el servicio. La afirmacion anterior de "nada
gobernado se perdio" queda **corregida con fecha** al lado del texto original, no borrada.

**G1.** Identidad, acuerdo de estado, preservacion de contenido y replicabilidad quedan
separados. Los hijos se comparan como **multiconjuntos**: una ausencia, una adicion y un numero
cambiado son diferencias. Sin payload retenido el resultado es `CONTENT_UNVERIFIABLE`, nunca
"preservado". Un outbox solo cuenta para la **generacion exacta** con payload recuperable.

**G2.** La reanudacion **demuestra su prefijo** en vez de confiar en `max(key)`: el caso
[1,2,3] sobre [1,3] repara el hueco, y una fila extra o modificada se rechaza. El snapshot
**mide** su frontera tomando el lock exclusivo, y reporta `VERIFIED_SNAPSHOT` o
`UNVERIFIED_COPY`; su limite queda escrito: DuckDB bloquea por proceso.

**187 reglas verdes sobre tres motores.** La causa del fallo del WAL **sigue sin conocerse** y
eso, por si solo, no justifica detener un almacen sano.
### Retorno 6b80923: cuatro metricas restauradas y verificadas

Musashi comparo nombres/valores de las cuatro metricas por API viva con los
payloads independientes de data-gov: coinciden. 23 pruebas focales independientes
verdes. No equivale a preservacion de todos los campos: faltan costos, identidad
de codigo y vinculos de disponibilidad en la conciliacion. El snapshot libera
el bloqueo antes de copiar. Continuacion limitada:
[H1-H3](../handoffs/MUSASHI_G1_G3_REVIEW_AND_H1_H3_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, ordenes H1-H3)

Retorno: `../audits/work_plan/SATOSHI_H1_H3_RETURN_2026_09_16.md`.

**H1.** El conjunto comparado se **deriva del contrato del propio almacen**, no de una lista
elegida a mano: 17 campos del padre —incluidos costes, identidad de codigo, tiempos y tags— y
los 13 del dataset, con el enlace al contrato de disponibilidad. Las dos sondas del revisor
—`costs_json` a 999999 y el digest del contrato a 64 ceros— son ahora reglas y fallan. Cada
informe lleva `field_coverage` y **ninguna columna almacenada queda sin representar**. La
comparacion es tipada y las diferencias **nombran el campo**. Un payload retenido que no cuadre
con su digest es `CONTENT_UNVERIFIABLE`. Produccion conciliada en solo lectura al alcance nuevo:
**55/55, cero diferencias**.

**H2.** La conexion exclusiva se **mantiene** durante checkpoint, medicion del origen, copia y
comparacion; antes se cerraba antes de copiar. Un segundo proceso que intenta escribir durante
la copia es **rechazado**, y la copia se compara contra el origen medido.

**H3 — retractacion.** Dije que las cuatro filas "estaban en el WAL que puse en cuarentena".
**No se sigue.** Que el payload se reproduzca bien en un cubo desechable descarta la logica del
proveedor, no dice donde se perdieron ni excluye al escritor desplegado. Queda: **establecido**
que faltaban y que estan restauradas; **no probado** donde se perdieron ni por que el WAL no
pudo reproducirse.

**206 reglas verdes sobre tres motores.**

### Retorno 383fbe7: codigo verificado, discrepancia viva nueva

60 pruebas focales independientes pasan; cobertura H1 y bloqueo durante copia H2
corregidos al alcance probado. Consulta viva independiente: 55 terminales, 537
metricas, 53 terminales coincidentes; los dos reparados contienen cuatro metricas
sobrantes por duplicacion exacta. No se atribuye causa ni momento sin evidencia.
Orden de conciliacion y reparacion acotada autorizada:
[I1-I3](../handoffs/MUSASHI_H1_H3_LIVE_REVIEW_AND_I1_I3_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, ordenes I1-I3)

Retorno: `../audits/work_plan/SATOSHI_I1_I3_RETURN_2026_09_16.md`.

**I1.** La discrepancia se reproduce por el servicio vivo y a cobertura completa: 55 aceptados,
**53 coinciden, 2 difieren**; cada uno de los dos terminales guarda `bytes_delivered` y
`delivery_from_cache` **dos veces** donde el payload aceptado declara una. `missing: 0`,
ningun campo cambiado. Los 55 payloads declaran **533** filas de metrica; el cubo sirve **537**.
Evidencia congelada antes de tocar nada, por una via que declara su propio limite:
`CONSISTENT_LIVE_READ` —el digest del servicio antes y despues de la lectura y el de la copia,
tres acuerdos— que es **mas debil** que `VERIFIED_SNAPSHOT` y lo dice.

**Camino reproducido.** La ingesta normal y la reparacion aditiva quedan **excluidas por
reproduccion**: ninguna puede anadir un hijo ya presente. Si lo hace un registro de escritura
adelantada reproducido sobre una base que ya lo contiene: una fila entra, dos salen. La firma
coincide con lo que guarda el cubo. **No esta establecido** que eso fuera lo ocurrido: no hay
recibo de una restauracion del registro, y no se infiere. El camino queda cerrado igualmente:
toda reparacion consolida su registro antes de volver y el recibo declara los bytes que dejo.

**Hallazgo propio.** El recibo H1 de produccion **no es reproducible**: el snapshot de H2 midio
537 filas y once segundos despues H1 declaro 55/55 sobre esa copia; el mismo codigo de entonces,
sobre contenido con ese mismo digest, reporta hoy dos terminales distintos. No se recupera que
paso fallo, porque ningun recibo llevaba la evidencia del otro. Corregido: cada informe lleva
ahora `source_content` —conteos y digest por relacion— y, tras una escritura,
`content_after_repair`.

**I2.** `--repair-surplus` quita solo las copias sobrantes. La multiplicidad esperada se
**cuenta del payload aceptado**; no hay `SELECT DISTINCT` ni regla global, y un duplicado que el
contrato declara se conserva. Un valor en conflicto es `REFUSED_NOT_PURE_SURPLUS`. Las filas se
preservan **antes** de quitarlas y sin `--evidence` la operacion se rechaza. Ensayado de punta a
punta sobre la copia de evidencia: 537 → **533**, las otras tres relaciones con digest intacto,
55/55, segunda invocacion sin efecto, registro de 0 bytes.

**Pendiente: la escritura en produccion.** Requiere parar el servicio dueno y el arnes rechaza
ese comando (`[Interfere With Workloads]`). La frontera esta ensayada paso a paso en el retorno.
El almacen vivo quedo corriendo y sin modificar.

**Cifras.** 56 reglas focales; **1406 pasan, 11 se saltan, 0 fallan** en trading-stack con el
entorno del almacen; 241+1 sobre tres motores. Tercer recibo propio que no reproduce: la cifra
de suite de H1 (1454/5) no se obtiene con el comando anotado a su lado — el arbol recoge 1413 y
`383fbe7` medido hoy da 1394/1/21. Se registra lo medido, con el entorno nombrado al lado.

**Hallazgo al cerrar.** Una pila desechable del 2026-09-15 sigue viva
(`crispdm-s2-stack-1789506694-3701496`, solo un puerto efimero; su base ya no existe). Su propio
`--teardown` no la para: una pila posterior en el mismo directorio **sobrescribio `STACK.json`**,
de modo que los hijos de la anterior no quedan registrados en ninguna parte. Es la recurrencia de
la clase U1 por otro flanco —no un grupo de procesos perdido, sino un **registro** perdido—. No se
corrige aqui por estar fuera del alcance acotado; queda listado con su reproduccion.

### Correccion 2026-09-16 (tras el intento en produccion): la falla es el indice

El owner abrio la frontera y la secuencia ensayada **no escribio nada**: tomo un
`VERIFIED_SNAPSHOT` y reporto `NOTHING_TO_REMOVE`. Al negarse encontro la falla real.

**El indice de `gov_terminal_metric` tiene menos entradas que la tabla.** 537 filas; una lectura
filtrada alcanza **533**, un barrido forzado **537**. El motor lo nombra solo, al intentar un
borrado completo sobre una copia: *"Failed to delete all rows from index. Only deleted 531 out of
535 rows"*, y acto seguido invalida esa base. De ahi se sigue todo: la reparacion preguntaba por
predicado y no veia nada; borrar las filas que el predicado si veia **empeoraba** el estado
(533 filas de las que el predicado alcanzaba 529). Los consumidores que filtran por terminal
**ya reciben las 533 correctas**; solo los agregados sin filtro cuentan cuatro de mas.

El remedio es el indice: `reindex_relation` lo reconstruye desde su propia definicion del
catalogo —ninguna fila se toca— y el acuerdo se vuelve a medir **en una conexion nueva**, porque
dentro de la transaccion que lo reconstruye el motor sigue respondiendo con la lectura vieja.
Nada avanza hasta que un filtro y un barrido den la misma poblacion. El borrado de relacion
completa queda **prohibido por regla** en la herramienta.

**Retractacion.** El WAL reproducido duplica filas pero deja el indice **consistente**, asi que
no reproduce esta firma. Cuatro mecanismos reproducidos y descartados; la causa del indice corto
**no esta establecida**.

Reensayado sobre copia del snapshot real: indice reconstruido, 537 → **533**, filtro y barrido de
acuerdo, 55/55, segunda invocacion sin efecto, WAL de 0 bytes. Falta una sola frontera mas para
escribirlo en produccion. 64 reglas focales; 254+1 sobre tres motores; 1406/11/0 en trading-stack.

### Cierre 2026-09-16: aplicado en produccion y verificado por el servicio

Una frontera coordinada, ~1 minuto de indisponibilidad, `NRestarts=0`. Snapshot
`VERIFIED_SNAPSHOT` antes de escribir. Indice `gov_terminal_metric_sha_idx` reconstruido,
cuatro filas sobrantes retiradas tras preservarlas, WAL de 0 bytes.

`gov_terminal_metric` 537 → **533**, y filtro/barrido de **533/537 en desacuerdo** a
**533/533 de acuerdo**. Las otras tres relaciones con digest intacto y de acuerdo en todo
momento. Conciliado despues de la escritura **por el servicio activo**: 55 aceptados,
**55 coinciden, 0 difieren**, cero huerfanas. I1-I3 cerrado.

### Retorno 18ffa4e aceptado al alcance operacional medido

Musashi ejecuto conciliacion por API viva: 55/55, cero diferencias, 533 metricas,
filtro/barrido concordante en metricas, datasets y artefactos. 64 pruebas focales
independientes pasan. Pila desechable huerfana retirada por su scope exacto;
produccion intacta y activa. No repetir reparacion ni retirada. Siguiente trabajo:
prevenir sobreescritura de STACK.json, vigilancia acotada y preparar el siguiente
paso cientifico ya previsto, sin nuevas autorizaciones operativas por etapa.
[Acta y continuacion](../handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, bloques de seguimiento)

Retorno: `../audits/work_plan/SATOSHI_STACK_WATCH_D3_RETURN_2026_09_16.md`.

**1. Sobrescritura de `STACK.json`, cerrada.** Cada invocacion toma una identidad inmutable
(`run_id`, 32 hex) y su **propio directorio hijo**; el marcador que autoriza una senal es ese
directorio, no el que nombro el llamante —que dos ejecuciones comparten por definicion—. El
registro se escribe en `work/runs/<run_id>.json` **antes de que exista hijo alguno** y se
actualiza segun arrancan, de modo que un lanzamiento que falla a medias deja constancia exacta
de lo que quedo corriendo. `--teardown <WORKDIR>` recorre todas las ejecuciones abiertas alli.
`STACK.json` se conserva como puntero, ya no como registro. **19 reglas** (18 rojas antes).

**2. Vigilancia acotada, desplegada.** `tools/olap_consistency_watch.py` observa multiplicidad
sobrante y desacuerdo filtro-contra-barrido. **De solo lectura por construccion**: un unico
`urlopen`, GET, sin metodo de escritura, y una regla que exige que el codigo —sin prosa— no
contenga `DELETE`, `INSERT`, `CREATE INDEX`, `--repair` ni `reindex_relation`. Un hallazgo es
evidencia, nunca disparador. **La cadencia sale del coste medido**: 16 consultas, 0,055 s
contra el cubo vivo; al 1 % de ciclo de trabajo gobierna el suelo de 300 s. El numero de
consultas **no crece con los terminales**. Timer `crispdm-olap-consistency-watch.timer` activo;
las observaciones se anexan y no se reemplazan. **14 reglas**.

**3. D3 preparado, no ejecutado.** El siguiente paso sin ejecutar de la secuencia vinculante es
**D3**. Prerrequisito real, **probado y no leido de una tabla**: R2 y N3 `PRESENT`, y **R6
`NOT_APPLIED` — bloquea**: `df_coverage_current`/`df_coverage_history` no existen en el cubo,
mientras `df_fact_coverage` tiene 440.694 filas con **una** ejecucion y **dos** digests de
codigo. Sin vista de seleccion, ninguna cifra de cobertura es atribuible. Preparados el
contrato del operador (`df_d3_contract.py`), las diez pruebas de aceptacion del diseno
(`df_d3_acceptance.py`, **40 reglas**, con control no causal que falla) y el plan de ejecucion
gobernada (`11_PLAN_EJECUCION_GOBERNADA_D3_2026_09_16.md`). Sin implementar operadores, sin
ejecutar D3, sin tocar el diseno sellado y sin aplicar R6.

**Limite medido del diseno sellado.** La prueba 1 compara solo hasta `n - lookback - delay`, de
modo que un adelanto de 4 muestras con 7 de lookback declarado le resulta invisible: medido,
prueba 1 **pasa** y prueba 2 **falla**. Se registra para el revisor; no se corrige un diseno
sellado.

**327 pasan, 1 se salta** sobre tres motores.

### R6 aplicado 2026-09-16: la cobertura ya es atribuible

Era el unico bloqueo medido de D3, asi que se quito en vez de reportarse.
`tools/df_r6_apply.py` corre el SQL de R6 **verbatim**, traduce las dos formas de PostgreSQL que
el motor del cubo no comparte y **nombra ambas en el recibo** —una traduccion silenciosa es un
cambio que nadie reviso—, ensaya sobre una copia del propio archivo y **rehusa tocar el original**
si el ensayo no cuadra por contenido.

Defecto real del SQL, encontrado al ejecutarlo: el texto de la seleccion dice
`(9 states + applicability); supersedes v1 c140`, y una division ingenua por `;` partia la
sentencia en dos. El separador respeta comillas, y eso —y un `--` entre comillas— son reglas.

Ventana coordinada de bastante menos de un minuto, `NRestarts=0`:

| relacion | filas |
|---|---|
| `df_fact_coverage` | 440.694 — **sin cambios** |
| `df_fact_coverage_v2` | 633.189 — **sin cambios** |
| `df_coverage_current` | **633.189** — exactamente la matriz seleccionada |
| `df_coverage_history` | **1.073.883** = 440.694 + 633.189 |
| `df_coverage_version_selection` | 1, con razon explicita |
| `df_coverage_current_denominator` | 715 datasets |

Leido por el servicio activo: la vista actual lleva **una** ejecucion y **un** digest de codigo.
Nada borrado ni deduplicado; WAL en 0. Reprobados los prerrequisitos: **`ready_to_measure: true`**.
Conciliado tras la escritura por el servicio: 55/55, filtro y barrido de acuerdo. **17 reglas**
nuevas; **344 pasan, 1 se salta** sobre tres motores.

### Retorno 54f2df0: R6 medido, siguiente paso D3

R6 consultado independientemente: 633189 filas actuales con una ejecucion/un
digest; historico 1073883; denominador 715. 57 pruebas focales pasan. D3 tiene
infraestructura, pero su prueba de disponibilidad confunde lookback con latencia
y rechaza '0s'; prefijo omite una frontera causal relevante. Se autoriza enmienda
pre-candidatos y correccion de bateria, seguida condicionalmente por implementacion
y mecanica CPU gobernada, sin otra pausa de autorizacion:
[J1-J3](../handoffs/MUSASHI_R6_ACCEPTANCE_AND_D3_J1_J3_2026_09_16.md).

## Actualizacion 2026-09-16 (Satoshi, ordenes J1-J3: D3)

Retorno: `../audits/work_plan/SATOSHI_D3_J1_J3_RETURN_2026_09_16.md`.

**J1.** Enmienda superseding al diseno 07 (`07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md`, sellada
como datos en `tools/df_d3_design.py`), con los bytes originales intactos y citados por digest.
Cuatro instantes separados por salida; emision >= ultima disponibilidad consumida; duraciones
con el parser del productor y a muestras solo por division exacta; prefijo sin exencion por
lookback; gemelo obligatorio o `NOT_APPLICABLE` con razon; inicio de impulso != retardo de
grupo; soporte wavelet derivado de la biblioteca; filtros recursivos con estado.

**J2.** Contrato v2 y bateria v2 probados contra los objetos reales: los cuatro hallazgos del
revisor como negativos, dos controles positivos, duraciones fraccionarias, llegadas tardias,
marcas ausentes, `'0s'` real, ventanas, normalizacion, reinicio, futuro cambiado. **40 reglas**.

**J3.** Los nueve operadores del §2 con sus siete gemelos, sobre numpy/scipy/pywt con versiones
registradas; los nueve *review-ready*; cada gemelo falla solo. Declaraciones **medidas**: la STFT
de Hann no ve `x[t]` y declara inicio 1. **37 reglas.** Mecanica gobernada: trabajador de
unidad con el protocolo hijo de D2, toys entregados por gobernanza (`VERIFIED_TRANSFER`),
congelamiento sellado de poblacion y presupuestos con piloto de coste, despacho `NON_GOVERNING`
a WORKER_A/B (coordinador excluido por su proceso de GPU), recoleccion con verificacion de
digest, dos campanas data-gov (SYNTHETIC + DATASETS) y un sobre `MECHANICAL`. **10 reglas.**
Dos defectos del despacho encontrados y corregidos en el primer intento (raiz nueva bajo reja;
rutas sin `~/`), ambos congelados como reglas.

**Corrida `d3mech-v1` (2026-09-17).** 504 unidades del banco + 7 toys, 43 shards en dos
workers: **511/511 verificadas**, 0 discrepancias, 83,070 filas, 2.91 h de wall. Tres unidades
toy de 26 variables murieron por `WALL_TIME_LIMIT` en el primer intento (presupuesto por variable
entregado plano al hijo: defecto mio); corregido, congelado y reintentadas como intento 2
versionado (133 s cada una). Matriz (`tools/df_d3_matrix.py`): seis operadores
`MECHANICALLY_ACCEPTED` en las 710 variables; `stft` declara y mide inicio 1; el cuantizador de
deciles rechazado en 63 variables y SAX en 6 por una sonda escalon **sin escalar al dominio
ajustado** (artefacto del fixture; propuesta: amplitud relativa a la resolucion del operador y
`UNIDENTIFIED` cuando el escalon no mueve nada en *p*); `wavelet` (soporte 50, cualquier NaN
→ no disponible) inaplicable bajo MCAR 10 % (14 de 15 variables) y un hueco de la bateria:
un gemelo que no emite nada contaba como "paso" (propuesta: `INSUFFICIENT_TEST`). Nada se
reajusto tras la corrida. Gobernanza: 504 terminales SYNTHETIC (444 gen. 1 + 60 gen. 2 tras
rechazo por identidad de metrica duplicada, corregida y superseded por el outbox), 7 campanas
DATASETS de un recurso y una unidad (la campana de siete recursos no puede completarse por
regla del servidor: forma mia, servidor correcto; 14 sobres cerrados `INVALID_ENVELOPE`), un
sobre `MECHANICAL` cargado en DuckDB (6,390 unidades, 551 consumos). Dos hallazgos para el
dueno del store: un documento ilegible responde 503 en vez de 400; el loader descarta el cuerpo
de la respuesta reintentable. Todo `NON_GOVERNING`; sin seleccion por utilidad.

### Retorno 8364714: mecanica presentada, matriz aun no verificada

93 pruebas D3 focales ejecutadas independientemente pasan. Dos reproducciones
aisladas prueban que el reductor acepta un veredicto sin las doce pruebas y omite
un archivo ausente sin rechazo. No implica alteracion de la campana real.
Corregir verificacion de matriz, sonda escalada por entrenamiento, insuficiencia
del gemelo y diagnosticos de ingesta; despues mecanica sucesora CPU gobernada.
Conservar v1, intentos y rechazos. Sin utilidad cientifica ni promocion financiera.
Orden completa, sin pausas de autorizacion entre bloques:
[K1-K5](../handoffs/MUSASHI_D3_J1_J3_REVIEW_AND_K1_K5_2026_09_16.md).


## Actualizacion 2026-09-17 (Satoshi, ordenes K1-K5: D3 sucesora)

Retorno: `../audits/work_plan/SATOSHI_D3_K1_K5_RETURN_2026_09_17.md`. Ending
`D3_SUCCESSOR_MECHANICS_AND_MATRIX_READY_FOR_EXTERNAL_REVIEW`.

**K1.** Matriz verificada separada del resumen: poblacion esperada desde el manifiesto congelado,
intentos desde el ledger, digests re-leidos contra sus terminales, filas ligadas a diseno/spec/
codigo/contrato, cobertura exacta por unidad x variable x operador x test, veredicto recalculado.
15 reglas (los dos contraejemplos de Musashi incluidos). v1 re-agregada desde sus bytes:
VERIFIED, 0 rechazos, sin diferencia.

**K2-K3.** Enmienda 07B sellada antes de medir: sonda desde el ajuste de entrenamiento y la
resolucion declarada por cada operador; tres hechos aparte; abstencion solo por declaracion y
cuesta el veredicto; el control retardado sigue fallando; gemelo sin comparaciones =
INSUFFICIENT_TEST; cobertura aparte. 26 reglas; parametros sin tocar.

**K4.** Validacion en frontera (400 tipado), sidecar de reintento con estado/clase/razon,
atencion tras 3x5xx; 10 reglas con stand-in y con el warehouse real efimero. Paquete
`predictor-olap-store` 0.1.1 **adoptado** en produccion por el procedimiento (respaldo, ensayo,
rueda, reinicio, comprobacion); el sobre malformado responde 400 en produccion. Error propio
registrado: la primera post-comprobacion cargo un sobre DEVELOPMENT (`475b93fb…`) de la era
Postgres en el cubo; sonda corregida; nada borrado; disposicion para Musashi.

**K5.** Corrida `d3mech-v2` completa (511/511 verificadas, 2.88 h): intento 1 fallido por no
sincronizar las entradas a los workers (defecto propio, conservado); intento 2 completo. Matriz
sucesora VERIFIED; delta v1→v2 todo atribuido: los 63+6 rechazos de cuantizador/SAX desaparecen
(inicio 0 en su dominio ajustado); 13 variables de train constante quedan INCONCLUSIVE por
declaracion; wavelet bajo MCAR: 3 gemelos INSUFFICIENT y 5 FAILED por comparaciones **no
sensibles** (hallazgo para la siguiente enmienda: solo cuenta una comparacion cuyo soporte cruza
el corte). 511 terminales gen. 1 conciliados, sobre cargado, conciliacion por contenido contra la
contabilidad: NO_LOSS (1,077 terminales). 232 pruebas. Experimento de utilidad preparado sin
ejecutar (doc 12). Defecto de nombre: el sobre v2 lleva `campaign_key d3-mechanics-v1`;
corregido para futuras corridas.

### Retorno a37d495: relectura D3 v2 y cierre acotado pendiente

Musashi ejecuto 53 pruebas focales y releyo v2: 511 unidades completas, 710
variables totales, 83070 filas; el SHA canonico del freeze real coincide.
Tres fixtures muestran que verify aun acepta contrato ausente, digest de freeze
incorrecto e intento duplicado. No se afirma alteracion de la evidencia real.
Cerrar esas omisiones sin repetir la campana, medir sensibilidad del gemelo
solo al alcance necesario, conservar el sobre DEVELOPMENT accidental con
disposicion no cientifica, corregir errores del host y concretar el borrador de
utilidad con pruebas del arnes. No se abre la reserva ni utilidad cientifica.
[Orden completa L1-L6](../handoffs/MUSASHI_D3_K1_K5_REVIEW_AND_L1_L6_2026_09_17.md).


## Actualizacion 2026-09-17 (Satoshi, ordenes L1-L6: cierre del verificador y diseno de utilidad)

Retorno: `../audits/work_plan/SATOSHI_D3_L1_L6_RETURN_2026_09_17.md`. Ending
`D3_MECHANICS_VERIFIER_CLOSED_AND_UTILITY_DESIGN_READY_FOR_REVIEW`.

**L1.** Las tres omisiones del revisor congeladas y cerradas: contrato y toys obligatorios,
freeze recalculado + esquema + cardinalidades + cotejo con el registro de campana conservado,
asignacion unidad/shard/rol contra el despacho, copia de transporte identica vs intento
contradictorio por regla explicita, estados desconocidos e ids discordantes rechazados. 28
reglas. v1 y v2 reverificadas desde sus bytes: identicas, sin copias de transporte.

**L2.** Hipotesis demostrada por unidad (diagnostico): los 5 FAILED de wavelet bajo MCAR tenian
0 comparaciones sensibles. Enmienda 07C sellada antes de medir: alcance declarado por gemelo,
comparacion sensible = alcance cruza el corte (necesario, no suficiente), cualquier infraccion
observada es deteccion, 0 sensibles = INSUFFICIENT_TEST, sensibles sin efecto = declaracion
fallida; pruebas causales del candidato intactas; controles por mascara y por emision. Replay
compuesto `d3mech-v3`: solo `non_causal_twin` en los 7 operadores con gemelo sobre toda la
poblacion; las 11 pruebas restantes heredadas de v2 por digest y verificadas por celda
(4 reglas). Resultado: VERIFIED; delta v2->v3 todo atribuido: 5 FAILED -> INSUFFICIENT_TEST;
511 terminales gen 1 conciliados; sobre cargado; NO_LOSS sobre 1,588 terminales.

**L3.** Disposiciones publicadas por el procedimiento (stop/publish/start): sobre accidental
`475b93fb` = ACCIDENTAL_OPERATIONAL_INGESTION, no admisible cientificamente, digest rederivado,
hijos verificados, fuera de `gov_scientific_evidence`; relacion correctiva para el sobre v2 bajo
`d3-mechanics-v1`. Seleccion por identidad (`df_d3_cube_select`) y sonda de idempotencia que no
introduce runs, ambas probadas con el warehouse real efimero.

**L4.** En `data-warehouse`: 400/422 entrada tipada, 503 solo indisponibilidad real, 500 defecto
interno nombrado y JSON con clase; SQL invalido ya no es HTML 500; gancho apagado por defecto
para inducir defectos en stacks desechables; 8 reglas. Adoptado por el procedimiento: el primer
intento puso el warehouse en bucle de reinicio (build/ rastreado en git empaqueto un backends.py
rancio), restaurado al instante desde el respaldo; el adoptador ahora exporta el commit limpio
y exige paridad byte a byte rueda/fuente; segunda y tercera adopcion verdes.

**L5.** Doc 12 reescrito como protocolo (perdida del modelo de prueba, nunca informacion;
elegibilidad por celda; ventanas por identidad de observacion; purga = horizonte + alcance +
ventana; pareado sobre las mismas filas; capacidad declarada; unidad estadistica = bloque;
Bonferroni sobre la familia predeclarada; reserva de uso unico) y arnes con 11 reglas de verdad
fabricada. No ejecutado; no abre el experimento.

### Retorno ea555f5: composicion D3 y arnes de utilidad revisados

Relectura independiente v3: 511 completas, 710 variables, 9940 filas medidas,
cero rechazos. Bateria 72p/1s y repeticion de diagnosticos con candidato 14p,
sin sumar conteos solapados. Se reprodujeron herencia aceptada tras cambiar un
operador no remedido, etiqueta futura aceptada como feature y tiempos de emision
ignorados por el arnes. La aproximacion t difiere de scipy en df bajo. No se ha
medido utilidad real con ese arnes: cerrarlo antes de abrir el piloto. Limites
observados y limpieza del build rastreado incluidos, sin pedir permiso por bloque.
[Orden M1-M6](../handoffs/MUSASHI_D3_L1_L6_REVIEW_AND_M1_M6_2026_09_17.md).


## Actualizacion 2026-09-17 (Satoshi, ordenes M1-M6: composicion cerrada, arnes causal)

Retorno: `../audits/work_plan/SATOSHI_D3_M1_M6_RETURN_2026_09_17.md`. Ending
`D3_COMPOSITION_CLOSED_UTILITY_CAUSAL_HARNESS_READY_FOR_REVIEW`.

**M1.** Herencia por equivalencia (params + spec_sha256, diff permitido vacio en 07C; misma
poblacion y contratos; particion disjunta y completa; fila heredada ligada al freeze fuente);
el caso del revisor y cinco mas congelados en rojo; 34 reglas. v2 y v3 reverificadas desde
bytes: identicas; ninguna medicion invalidada. **M2.** Arnes reescrito: operador real ajustado
por bloque, features solo si emitidas antes de la decision, ids/tiempos validados, comprobacion
de prefijo, elegibilidad por registro de celdas verificado; la etiqueta futura del revisor cae por
registro y por prefijo. **M3.** scipy t; bloques todo-o-insuficiente; calibracion obligatoria
sellada (nulo intercambiable 0/200; AR(1) es control positivo, no nulo); protocolo validado y
familia sellada. **M4.** Presupuestos observados en hijo aislado (control lento RESOURCE_EXCEEDED
sin puntuacion parcial); entrada gobernada `df_utility_run.py` ensayada bajo data-gov
(`utilreh-v4`, conciliada, sobre DEVELOPMENT); reserva por identidad. **M5.** build/ y
__pycache__ fuera del indice en data-warehouse (43edc44), build limpio desde commit publicado
probado con entry point real, sin reinicio de produccion; cronologia de los 22 reinicios
conservada; seleccion mecanica estricta por identidad (el sobre accidental queda fuera; la vista
`gov_mechanical_evidence` se documenta como historial operativo amplio). 287 pruebas.

### Retorno b6facb8: recuperar metricas y acotar calibracion

54 pruebas focales independientes pasan. En la contabilidad real los tres
COMPLETED de utilreh-v4 tienen metrics vacio; sus perdidas existen localmente.
El padre devuelve result.json en vez de contrast.json. La calibracion admite
cero simulaciones y NaN; no liga el alcance del operador/protocolo/longitud.
Registro de campana posterior al trabajo detectado en el entry point.
Recuperacion aditiva, calibracion verificable, registro previo y ensayo completo;
despues piloto descriptivo de hasta tres unidades sinteticas ya expuestas,
sin reservas, seleccion ni confirmacion. No repetir D3 mecanico.
[Orden N1-N5](../handoffs/MUSASHI_D3_M1_M6_REVIEW_AND_N1_N5_2026_09_17.md).


## Actualizacion 2026-09-17 (Satoshi, ordenes N1-N5: resultados recuperados, calibracion con alcance, piloto descriptivo)

Retorno: `../audits/work_plan/SATOSHI_D3_N1_N5_RETURN_2026_09_17.md`. Ending
`UTILITY_RESULTS_RECOVERED_CALIBRATION_SCOPED_DESCRIPTIVE_PILOT_REVIEW`.

**N1.** El score es el archivo verificado (bytes, esquema, identidad, finitud), nunca el resumen
del proceso; reanudacion sin re-ejecutar; `UNAVAILABLE` en vez de cero. Las tres mediciones de
`utilreh-v4` recuperadas desde sus archivos como terminales de generacion 2 ligados a los
originales; contenido del cubo igual a los archivos valor a valor; sobre correctivo. Dos errores
propios registrados (sobre vacio cargado -> disposicion pendiente; duplicados gen 2 por marcas
de tiempo -> dispuestos). **N2.** Registro de calibracion completo (generador nulo de no efecto,
plan sellado, intentos/validas/fallos/avances, tasa y cota Clopper-Pearson, longitud, operador,
protocolo base, familia, cada simulacion bajo digest); el consumidor verifica alcance e identidad
y decide solo con `upper_bound <= alpha_adjusted`; nulo con dependencia (AR(1) + target
independiente); AR(1) puro es control positivo. **N3.** Registro de campanas antes de cualquier
hijo, `before_run` por hijo, calibracion y mecanica en hijos aislados con terminales propios
(instantes y costos reales), registro persistido, reanudacion sin duplicados; probado con gov de
prueba (orden por callbacks) y en produccion (v5: 409 antes de cualquier hijo; v6: instantes
`+00:00` rechazados, normalizados a `Z`). **N4.** Ensayo `utilreh-v7`: calibracion por operador
(239 sims, cotas 0.01246/0.01246/0.01969), contrastes con metricas verificadas hasta DuckDB
(contenido igual), control lento `RESOURCE_EXCEEDED`, conciliado. Piloto descriptivo
`utilpilot-v2` sobre tres unidades sinteticas de desarrollo: calibracion por operador (538 sims; cotas 0.00555/0.00555/0.00879 frente a alfa/9 = 0.00556), 9 contrastes (6 DOES_NOT_ADVANCE, 3 descriptivos), contenido igual en el cubo (12 unidades), sobre DEVELOPMENT cargado; ~1 s CPU por contraste. Descriptivo: nada afirma mejora ni selecciona. 301 pruebas.
### Retorno 9880f45: recuperacion confirmada, cota y resume pendientes

68 pruebas focales independientes pasan. Contabilidad leida: las tres recuperaciones
gen-2 contienen doce metricas, con deltas iguales a los archivos. Dos casos nuevos:
poner cota MAD a cero concede apoyo sin cambiar per_sim; resume con score ausente
conserva ADVANCES junto a SCORE_UNVERIFIED. Corregir, reverificar las simulaciones
conservadas y cerrar descriptivamente el piloto sin repetir campanas. Preparar
el siguiente diseno por variable, sin ejecutar reserva o ampliar poblacion ahora.
[Orden O1-O4](../handoffs/MUSASHI_D3_N1_N5_REVIEW_AND_O1_O4_2026_09_17.md).

**2026-09-17 · O1–O4 (Satoshi, sobre la revisión de N1–N5).** O1: conteos, tasa y cota de cada calibración se rederivan de las simulaciones (índices, semillas, etiqueta coherente con delta y margen, finitud, denominador completo) y se comparan con los resúmenes; la cota del registro nunca decide; política de fallos declarada (`WORST_CASE_FAILED_COUNTED_AS_ADVANCES`); plan, confianza, alfa, margen y código aplicable verificados contra el protocolo. O2: un intento reanudado tiene un solo desenlace veraz (`SCORE_UNVERIFIED` si la evidencia no verifica o el job difiere del registrado; el resumen viejo queda como historia; el sobre nunca emite COMPLETE desde un resumen); diff limitado de la reanudación del piloto comprobado (código científico y arnés sin cambio). Reverificación sucesora de `utilreh-v7` y `utilpilot-v2` desde resultados conservados (`tools/df_utility_reverify.py`): toda la evidencia verificada, 0 cambios de decisión, contenido del cubo igual. O3: tabla legible por unidad (pérdida cruda/transformada, delta e intervalo, filas pareadas, cobertura, costo, alcance del nulo) y diseño de desarrollo por variable sellado (12A: `raw`/`transformed`/`augmented` con control de capacidad `raw_wide`, umbrales heredados, replicación independiente, etapas separadas; sin ejecutar). O4: el sobre vacío `2e68209e…` está fuera de `gov_scientific_evidence` y `gov_mechanical_evidence` (evidencia O4), manifiesto ligado al sucesor `57e5f959…`, publicación en la próxima ventana necesaria. Suite 1231 passed / 6 skipped. Salida `UTILITY_PILOT_REVALIDATED_AND_NEXT_DEVELOPMENT_DESIGN_READY`.
### Retorno d402532: 12A requiere coherencia de ambos contrastes

53 pruebas focales pasan. La calibracion actual solo llama raw/transformed;
H_A requiere raw_wide/augmented. El validador acepta margen interno cambiado
con digest exterior reparado. La purga debe incluir ocho lags de raw_wide.
Corregir y sellar sucesor, sin repetir D3 ni el piloto; autorizacion condicional
para 36 contrastes de desarrollo y sus calibraciones/replicas, techo cuatro
horas CPU agregadas y limites existentes, sin reserva/finanzas/GPU.
[Orden P1-P4](../handoffs/MUSASHI_UTILITY_O1_O4_REVIEW_AND_P1_P4_2026_09_17.md).

**2026-09-18 · P1–P4 (Satoshi, sobre la revisión de O1–O4).** P1: cada hipótesis se calibra con su par exacto por el mismo callable del experimento; el registro v2 liga par, anchuras y política de filas y nunca transfiere entre pares. P2: validación recursiva de cada protocolo de familia (campos heredados, digests, miembros, pares, plan, longitud y elegibilidad desde los recursos ligados), mapa explícito de replicación al mismo régimen (bumps→bumps, sinusoid→sinusoid, steps→steps; semilla y datos distintos), frontera train/validación derivada del soporte consumido y registrada por identidades; diseño 12B sellado (`55ef8321…`), 12A preservado como historia. P3: campaña gobernada `utildev-v1` — piloto de costo, proyección 9 232 s ≤ 14 400 s, gasto 5 852 s CPU, 6 familias, 36 contratos de calibración, 36 contrastes, 6 sobres DEVELOPMENT cargados. P4: archivos → padre → contabilidad → warehouse verificados en las seis familias; ninguna pareja propuesta (18 DOES_NOT_ADVANCE bajo contrato favorable, 18 descriptivos); observación: las semillas de calibración se repiten entre familias (mismos sorteos nulos ×6). Retorno `SATOSHI_UTILITY_P1_P4_RETURN_2026_09_18.md`. Salida `PER_VARIABLE_DEVELOPMENT_SELECTION_AND_REPLICATION_READY_FOR_REVIEW`.
### Retorno fb1833c: sin propuestas en utildev-v1

17 pruebas focales pasan. Relectura del cierre: cuatro checks true, 18 sin avance,
18 inconclusos y ninguna propuesta; se consumieron recibos conservados, no una
consulta nueva al cubo. Dos contraejemplos del cierre: diseno vacio todo true;
propuestas emitidas aun con files_verified=false. Corregir sin repetir campana.
Reconteo de calibraciones: seis lotes per_sim, cada uno repetido seis veces.
Cache cientificamente equivalente autorizado en ensayo; cambiar semillas no
ahorra CPU. Controles de utilidad conocida y mapa de areas pendientes antes de
otro barrido. Sin nuevas reservas, GPU ni datos financieros.
[Orden Q1-Q4](../handoffs/MUSASHI_UTILITY_P1_P4_REVIEW_AND_Q1_Q4_2026_09_18.md).

**2026-09-18 · Q1–Q4 (Satoshi, sobre la revisión de P1–P4).** Q1: cierre ligado a la población registrada (identidad del diseño, familias, miembros, contratos, pares y mapa derivados; rechazos tipados; recibos ligados al run y a la población completa; propuestas sólo con ambas familias del par verificadas; política de cierre parcial); cierre sucesor de `utildev-v1` sin remedir: 0 cambios, 0 propuestas. Q2: clave canónica de la computación de calibración (sin etiquetas), registros v3, caché verificado (miss→hit con cero simulaciones y bytes idénticos en ensayo gobernado; replay de los 36 registros = 6 computaciones ×6, 4 801 s evitables como proyección). Q3: controles del instrumento (nulo, pérdida de información y fuga separados 6/6; positivos por debajo del criterio predeclarado: H_T 4/6, H_A indecidible bajo la calibración de fixture) y mapa de cobertura 12C con propuesta acotada del siguiente experimento (no lanzado). Retorno `SATOSHI_UTILITY_Q1_Q4_RETURN_2026_09_18.md`. Salida `DEVELOPMENT_CLOSURE_VERIFIED_CALIBRATION_REUSE_TESTED`.
