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
