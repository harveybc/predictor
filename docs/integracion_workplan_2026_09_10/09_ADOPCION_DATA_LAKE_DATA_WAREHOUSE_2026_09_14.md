# Plan obligatorio: crear y adoptar data-lake y data-warehouse

Fecha: 2026-09-14. Decisor: Harvey. Ejecutor: Satoshi. Revision: Musashi.
Estado actual: repositorios locales creados; publicacion y adopcion productiva pendientes.

Actualizacion tras el retorno `939fb41`: ambos repositorios git locales
existen; Satoshi reporta pruebas desechables de hosts y proveedores. La
publicacion en GitHub y el despliegue productivo siguen pendientes. Visibilidad
decidida: **publica para ambos repositorios**. La
[orden de publicacion y despliegue](../handoffs/MUSASHI_TO_SATOSHI_PUBLICAR_Y_DESPLEGAR_HOSTS_2026_09_14.md)
es la continuacion vigente y no exige otra decision del owner para esos pasos.

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
| 1. Crear | Ambos repositorios publicados con URL real, README, AGENTS.md, requisitos, pruebas previstas y estado persistente | **PUBLICADO con visibilidad PRIVADA**: https://github.com/harveybc/data-lake (`1ba23ed`) y https://github.com/harveybc/data-warehouse (`6f16565`), rama por defecto `master`, historia local conservada, README + AGENTS.md + ejemplos + `docs/IMPLEMENTATION_STATE.md`. **La visibilidad pública quedó bloqueada por el entorno de ejecución** (ver §Bloqueos); es un cambio de un comando que el owner puede aplicar |
| 2. Implementar | Hosts instalables y plugins externos desde financial-data/predictor, probados como wheels en entornos limpios | **PROBADO**: entorno virtual limpio (sin `--system-site-packages`, ningún checkout en `sys.path`) instalando las cuatro distribuciones **desde las URL de GitHub**; los cuatro entry points resuelven; 23 + 18 pruebas contra el código instalado. Proveedores: `financial-data-store` en `financial-data@d9be1b368` (rama por defecto) y `predictor-olap-store` en `predictor@6d5c9ed` (PR #44 hacia master; revisión pública fijada como origen instalable declarado) |
| 3. Integrar | Paridad con las APIs actuales: inventario, bytes, disponibilidad temporal, particiones, recibos, resultados y reintentos | **PROBADO sobre la revisión a desplegar**: lago 8/8 rutas idénticas en estado, bytes y cabeceras; almacén 11/11 idénticas en estado y cuerpo, incluidas las idempotencias de reporte y terminal; campaña Flow v3 completa por ambos hosts con la gobernanza intacta. Dos defectos reales hallados aquí y corregidos: el proveedor OLAP empaquetado no era el que corre en producción (contaba filas en `discover`), y los `include_globs` del lago vivían en la aplicación legada (16.346 recursos contra 5.275) |
| 4. Probar interfaz | Configuracion e inventario AdminLTE, esquema de recursos, escritorio y movil; pruebas de sistema y aceptacion | **PROBADO**: consola AdminLTE en ambos hosts (inventario, metadatos de recurso / esquema de relación, consulta acotada con su tabla de resultado escapada, configuración **pendiente** que no mueve la activa); 10 + 10 pruebas; aceptación en navegador real a 1440×900 y 390×844 — 6 y 8 páginas, todos los activos servidos por el propio host, cero desbordamiento horizontal — con recibo y PNG en `docs/console/` de cada repositorio |
| 5. Poner en uso | Transicion controlada con los mismos datos, IDs y cubo; microexperimento gobernado por ambos hosts y conciliacion exacta | **BLOQUEADO por el entorno de ejecución**, no por evidencia: preparado y validado en seco (configuraciones equivalentes verificadas contra las vivas, inventario idéntico 5.275/5.275 en un puerto libre, almacén candidato idéntico al vivo salvo un campo `transport` aditivo, ventana sin entregas en vuelo, respaldos de contabilidad y tablas `gov_*`, límites de memoria y procedimiento de reversión escritos). Detener :5056/:5057 y arrancar los reemplazos fue rechazado por el clasificador del entorno |
| 6. Adoptar | Configuraciones de consumidores actualizadas y cobertura documentada por proyecto; nuevos experimentos usan esta ruta | **MATRIZ PUBLICADA, ejecución pendiente de la etapa 5**: ver §Matriz de adopción. Nada se declara adoptado por tener un envoltorio o un README |

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
