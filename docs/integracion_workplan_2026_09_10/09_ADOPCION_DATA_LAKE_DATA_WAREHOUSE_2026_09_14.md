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

| Etapa | Entregable y criterio de terminacion | Estado |
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

### Retorno e99c6ba: progreso probado y alcance del incidente

Las dos reproducciones anteriores ya pasan independientemente. Warehouse y
cargador activos. La conciliacion del incidente todavia acepta metricas alteradas
y todos los hijos ausentes: identidad/estado no equivalen a contenido preservado.
No se afirma perdida productiva. Resume con huecos y snapshot con frontera solo
declarada siguen pendientes. Continuacion acotada:
[G1-G3](../handoffs/MUSASHI_F1_F5_REVIEW_AND_G1_G3_2026_09_16.md).

### Retorno 6b80923: cuatro metricas restauradas y verificadas

Musashi comparo nombres/valores de las cuatro metricas por API viva con los
payloads independientes de data-gov: coinciden. 23 pruebas focales independientes
verdes. No equivale a preservacion de todos los campos: faltan costos, identidad
de codigo y vinculos de disponibilidad en la conciliacion. El snapshot libera
el bloqueo antes de copiar. Continuacion limitada:
[H1-H3](../handoffs/MUSASHI_G1_G3_REVIEW_AND_H1_H3_2026_09_16.md).

### Retorno 383fbe7: codigo verificado, discrepancia viva nueva

60 pruebas focales independientes pasan; cobertura H1 y bloqueo durante copia H2
corregidos al alcance probado. Consulta viva independiente: 55 terminales, 537
metricas, 53 terminales coincidentes; los dos reparados contienen cuatro metricas
sobrantes por duplicacion exacta. No se atribuye causa ni momento sin evidencia.
Orden de conciliacion y reparacion acotada autorizada:
[I1-I3](../handoffs/MUSASHI_H1_H3_LIVE_REVIEW_AND_I1_I3_2026_09_16.md).

### Retorno 18ffa4e aceptado al alcance operacional medido

Musashi ejecuto conciliacion por API viva: 55/55, cero diferencias, 533 metricas,
filtro/barrido concordante en metricas, datasets y artefactos. 64 pruebas focales
independientes pasan. Pila desechable huerfana retirada por su scope exacto;
produccion intacta y activa. No repetir reparacion ni retirada. Siguiente trabajo:
prevenir sobreescritura de STACK.json, vigilancia acotada y preparar el siguiente
paso cientifico ya previsto, sin nuevas autorizaciones operativas por etapa.
[Acta y continuacion](../handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md).

### Retorno 54f2df0: R6 medido, siguiente paso D3

R6 consultado independientemente: 633189 filas actuales con una ejecucion/un
digest; historico 1073883; denominador 715. 57 pruebas focales pasan. D3 tiene
infraestructura, pero su prueba de disponibilidad confunde lookback con latencia
y rechaza '0s'; prefijo omite una frontera causal relevante. Se autoriza enmienda
pre-candidatos y correccion de bateria, seguida condicionalmente por implementacion
y mecanica CPU gobernada, sin otra pausa de autorizacion:
[J1-J3](../handoffs/MUSASHI_R6_ACCEPTANCE_AND_D3_J1_J3_2026_09_16.md).
