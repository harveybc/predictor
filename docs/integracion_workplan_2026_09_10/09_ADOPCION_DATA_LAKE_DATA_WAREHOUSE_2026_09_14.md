# Plan obligatorio: crear y adoptar data-lake y data-warehouse

Fecha: 2026-09-14. Decisor: Harvey. Ejecutor: Satoshi. Revision: Musashi.
Estado actual: repositorios publicos y hosts desplegados; prueba productiva
sintetica conciliada. Contratos financieros y adopcion completa por consumidor
siguen pendientes.

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

| Etapa | Entregable y criterio de terminacion | Estado |
|---|---|---|
| 1. Crear | Ambos repositorios publicados con URL real, README, AGENTS.md, requisitos, pruebas previstas y estado persistente | Completado: publicos; proveedor OLAP integrado en master por PR 44 |
| 2. Implementar | Hosts instalables y plugins externos desde financial-data/predictor, probados como wheels en entornos limpios | Completado en las revisiones instaladas del acta; pruebas independientes 33+28+4 |
| 3. Integrar | Paridad con las APIs actuales: inventario, bytes, disponibilidad temporal, particiones, recibos, resultados y reintentos | Paridad previa 8/8 y 11/11; E2E desechable re-ejecutado por Musashi |
| 4. Probar interfaz | Configuracion e inventario AdminLTE, esquema de recursos, escritorio y movil; pruebas de sistema y aceptacion | Pruebas y capturas publicadas por Satoshi; suites de consola re-ejecutadas por Musashi |
| 5. Poner en uso | Transicion controlada con los mismos datos, IDs y cubo; microexperimento gobernado por ambos hosts y conciliacion exacta | Completado en infraestructura y transporte sintetico; 5.275 recursos conservados, terminal y cuatro metricas en cubo real |
| 6. Adoptar | Configuraciones de consumidores actualizadas y cobertura documentada por proyecto; nuevos experimentos usan esta ruta | Pendiente, Satoshi |

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

Orden ejecutiva vigente: [contratos y adopcion de consumidores](../handoffs/MUSASHI_TO_SATOSHI_CONTRACTS_AND_CONSUMER_ADOPTION_2026_09_14.md).
