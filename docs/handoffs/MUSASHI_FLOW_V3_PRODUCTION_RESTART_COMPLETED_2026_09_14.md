# Flow v3: reinicio y micro-run productivo completados

Fecha UTC: 2026-09-14. Ejecutor: Musashi.
Complementa la orden GOV-N1..N8 del 2026-09-13. No sustituye sus tareas de
correccion, adopcion ni evaluacion cientifica.

## Resultado

El bloqueo operativo de reinicio queda resuelto. El owner no necesita ejecutar
comandos ni conceder otra autorizacion para ese paso. No reiniciar de nuevo
los tres servicios como si siguieran pendientes.

Se reiniciaron solamente los servicios 5057, 5056 y 5055, en ese orden,
mediante parada normal y arranque. No hubo KILL forzado. PostgreSQL, Metabase,
el loader y las campanas no fueron reiniciados por esta intervencion.

| Servicio | Codigo al arranque | Estado comprobado |
|---|---|---|
| Warehouse 5057 | predictor `39b1d2f7e25cca2d5d2f1329f86b39928d514781` | health OK; terminal real escrito y conciliado |
| Lake financiero 5056 | financial-data `13e6b1f4741974c655370b60f90b9cd1a5562077` | health OK; ruta v2 responde validacion, no 404 |
| Data-gov 5055 | `2470e4b42a678fb60dfef4ff71487fcd53e165b2` | health OK; campana, entregas, terminal y conciliacion completos |

Data-gov y lake financiero se ejecutan desde worktrees desprendidos del codigo
probado porque sus checkouts de desarrollo ya tenian cambios concurrentes de
Satoshi. Los datos y el registro de uso NO se copiaron a nuevos lagos: las
configuraciones de runtime apuntan a sus ubicaciones existentes.

Ubicaciones relativas al directorio comun de repositorios:

- `.worktrees/data-gov-runtime-20260913`.
- `.worktrees/financial-data-runtime-20260913/lake`.
- El warehouse conserva `predictor/olap/lake`.

No modificar ni eliminar esos worktrees mientras sirvan trafico. Los cambios
posteriores de N2/N4 no estan desplegados por este reinicio: necesitan su propia
prueba y actualizacion deliberada. No declarar que editar master actualiza el
servicio desprendido. Se conservaron el modelo de proceso desacoplado y sus
logs; esta intervencion no instala autoarranque tras un futuro reinicio del host.

## Prueba productiva

Consumidor limpio: `predictor@eec10c8`, en worktree separado.
Clase: `ARCHIVAL_REPLAY_NON_AUTHORITATIVE`. Dos epocas CPU, limites de 300
pasos de datos train/test y dos muestras MC. No uso de GPU ni validacion de
estrategia, rentabilidad o causalidad del preprocesamiento historico.

Unidad: `flow-v3-production-mechanics-20260914T045557Z`.
Campana: `ca2c0f1f088a3f46e616878e0eb9222813a04ee5852b72324d157d292bdd4d1b`.
Tiempo de la invocacion gobernada: 16.49 segundos; pipeline: 11.47 segundos.

| Evidencia | Resultado |
|---|---|
| Terminal | COMPLETED, uno en `gov_terminal` |
| Entradas | seis entregas registradas en `gov_terminal_dataset` |
| Metricas | 90 en `gov_terminal_metric` |
| Artefactos | seis en `gov_terminal_artifact` |
| Conciliacion | missing_units, accounting_only, lake_only: listas vacias |
| Dos flush posteriores | ambos sent=0, pending=0, failures={} |
| Idempotencia | conteos `gov_*` iguales antes y despues de ambos flush |
| Tipos publicados | financial_files=lake, olap_cube=warehouse, predictor_examples=lake |

El loader conserva el mismo PID, ActiveState=active y NRestarts=0 entre ambas
observaciones. Se observo +1 fila en `fact_campaign_unit` mientras estaba
activo; no afirmar que absolutamente todas las tablas ajenas a `gov_*`
permanecieron con el mismo conteo. El log del micro-run tambien muestra la
emision de un terminal al outbox CRISP-DM; conciliar esa representacion en N7
sin contar dos experimentos. No se borro ni trunco ninguna tabla.

## Verificacion y limites

Antes de actuar: respaldo SQLite y pg_dump nuevo de 419844744 bytes, configs
y entorno privados preservados, identidad actual de procesos comprobada.
Pruebas independientes focales: data-gov 21 passed, lake financiero 1 passed,
warehouse 4 passed. No se reclama una suite completa nueva.

Mi comprobador tuvo dos supuestos incorrectos: esperaba JSON en healthz
(el servicio responde texto `ok`) y esperaba 200 en un listado sin sus
parametros requeridos. Se detuvo, preserve sus registros, corregi la
comprobacion y comprobe el recorrido real. No fueron fallos de los servicios
ni se requirio volver a ejecutar el micro-run exitoso.

Evidencia privada bajo el identificador de despliegue
`musashi-flow-v3-deploy-20260914T045557Z` en el estado local de la campana.
Incluye deployment.json, production-probe.json, logs, respaldos y entornos;
no publicar estos ultimos. SHA-256 del reporte de prueba observado:
`733b60f8b5ea22672bdefb89fcb54a5e884a54e9538aec25e1a6434ead2d6027`.

## Relevo a Satoshi

1. Actualizar N3 a `PROVEN_PRODUCTION` para este alcance y enlazar esta acta.
2. No repetir el reinicio. Continuar N2/N4/N5 en desarrollo y sus pruebas.
3. El prerrequisito operativo de R3/R4 de D2 ya existe. Continuar una vez
   cumplidas sus propias dependencias de soporte, diseno y registro de campana;
   este ensayo no acepta decisiones cientificas ni habilita D3.
4. Usar los servicios activos para el flujo autorizado. No afirmar adopcion
   de feature-eng, feature-extractor, DOIN o live sin sus ensayos pendientes.
5. Mantener visible que datos financieros UNKNOWN siguen cerrados. Este
   despliegue no cambia contratos temporales ni licencias por declaracion.
