# Orden actualizada: desplegar Flow v3 y completar su adopcion

Fecha: 2026-09-13. Emisor: Musashi. Destinatario: Satoshi.
IDs de esta orden: GOV-N1..N8. No renumerar las ordenes D2-R1..R8.

## 1. Dictamen y punto de partida

Lei el retorno de Flow v3 en `predictor@eec10c8`, incluida su matriz de
adopcion y el procedimiento de despliegue. Esta revision documental NO es
una nueva ejecucion independiente de sus suites ni una aceptacion productiva.
Sus pruebas desechables son evidencia de integracion; no prueban despliegue.
Los defectos que encontro en mi implementacion deben conservar sus pruebas:
fallo del servidor con pandas/Arrow, fuga de slots, nombres de metricas y
diagnostico del outbox, y extension de cache. No restaurar versiones previas.

Situacion al retorno: los tres servicios aun usan codigo anterior; no hay
tablas `gov_*` en el cubo productivo. No declarar gobernanza adoptada en todos
los experimentos. No repetir la orden anterior desde cero.

Insumos obligatorios, accesibles tambien desde Web:

- [Retorno de Flow v3](https://github.com/harveybc/predictor/blob/eec10c8/docs/audits/work_plan/SATOSHI_FLOW_V3_ADOPTION_RETURN_PACKET_2026_09_13.md).
- [Cambio lake/warehouse y pruebas de Musashi](https://github.com/harveybc/data-gov/blob/f83f676/docs/STORE_KINDS_CHANGE.md), rama `musashi/store-kinds-20260913`.
- [README corregido](https://github.com/harveybc/data-gov/blob/f83f676/README.md).
- [Auditoria cientifica D2](https://github.com/harveybc/predictor/blob/7e3a649/docs/audits/work_plan/MUSASHI_REVIEW_C166_C184_AND_DATA_GOV_2026_09_13.md).
- [Orden D2-R1..R8 vigente](https://github.com/harveybc/predictor/blob/7e3a649/docs/handoffs/MUSASHI_TO_SATOSHI_D2_SUPPORT_PORTABILITY_NEXT_ORDER_2026_09_13.md).

## 2. Secuencia y limites

Primero N1 y N2. Despues N3 si el entorno permite el despliegue; mientras
tanto avanzar N4, N5, R1-R2 de D2 y los disenos N6-N7 en ramas separadas.
No esperar sin trabajo a que un servicio sea reiniciado.

CPU solamente. Para pruebas ordinarias, un trabajador por host, un hilo de
algebra lineal y limite duro de 2 GiB por proceso; cambiarlo solo con piloto
medido y reserva disponible. No lanzar pruebas de OOM deliberado. No ocupar
las tres maquinas si una prueba local basta. Para D2 rigen sus limites propios.
No GPU, entrenamiento cientifico, D3, seleccion, publicacion DOIN ni live.
No modificar las propuestas doctorales ni resultados historicos.

Aplicar el metodo ya adoptado: requisitos y pruebas de aceptacion, componentes
y pruebas unitarias, implementacion, integracion y prueba de usuario. Mantener
el estado de etapa y la matriz requisito-prueba-evidencia en archivos. No
crear otro sistema de aprobaciones para estos pasos de implementacion.

## N1 - Integrar lake/warehouse sin perder las correcciones nuevas

Integrar `data-gov@f83f676` con el master que contiene `8cd45f5` y sus
antecesores. Resolver por contenido, especialmente `files_lake.py`: conservar
tanto los metadatos de tipo como la correccion del servidor con pandas.
Preservar cambios ajenos sin comprometerlos como propios.

Un kernel y las mismas politicas. `kind` distingue `lake` de `warehouse`;
el motor y el transporte son campos distintos. `financial-data` es lake;
el cubo es warehouse; HTTP no es un tipo de almacen. No partir repositorios,
crear un tercer tipo ni renombrar `datagov.lake` o sus rutas existentes.

Pruebas: configuracion mixta, compatibilidad de configuraciones anteriores,
aislamiento de metadatos entre plugins, API y etiquetas del tablero. Ejecutar
las regresiones de servidores HTTP reales, no solo Flask test_client.
Repetir E2E con PostgreSQL desechable sobre el tip combinado y registrar las
identidades reales de los tres servicios. Actualizar README si la integracion
cambia algun comando; no publicar promesas de despliegue como hechos.

## N2 - Contrato temporal: probar el alcance que realmente permite

El retorno instala `DATE_TIME` en ambos campos, pero deriva una cota de
finalizacion `t+1h`. Un corte diario correcto no prueba disponibilidad dentro
del dia ni valida automaticamente cada particion de aprendizaje.

Verificar que el uso offline por dia sea una restriccion ejecutable y no
solo una nota. Probar extremos de intervalo, dias incompletos, uso intrabar y
fronteras train/cal/test, con una observacion futura alterada. Publicar por
separado etiqueta, cota de disponibilidad y zona horaria desconocida. Si la
API no puede representar esa distincion, corregir el contrato y consumidor
antes de presentar esos recursos como aptos para decisiones temporales.

El micro-run es `ARCHIVAL_REPLAY_NON_AUTHORITATIVE`: prueba transporte y
contabilidad, NO causalidad del preprocesamiento historico ni calidad del
modelo. No convertir por inferencia los otros recursos UNKNOWN en aptos.
No reconstruir todo el lago para probar tres recursos del ensayo.

## N3 - Despliegue acotado y prueba productiva

El owner ya aprobo la adopcion. El pendiente comunicado es una restriccion
del entorno ejecutor, no otra decision cientifica del owner. Si la politica
de Satoshi sigue impidiendo reiniciar, registrar ese limite una vez y pasar
este bloque a Musashi en un entorno permitido o al operador. No buscar otra
via para eludir la negativa ni detener los bloques independientes.

El ejecutor permitido debe actualizar la identidad actual de cada servicio;
los PID del apendice A son una observacion antigua, no instrucciones vigentes.
Revisar supervisor, entorno, configuracion efectiva, respaldo recuperable y
rollback antes de actuar. No ejecutar literalmente ese script: carece de
parada estricta ante algunos fallos y contiene escalado automatico a KILL.
Si no se libera un servicio, detener ese paso y diagnosticarlo.

Alcance: solo adaptador warehouse 5057, lake financiero 5056 y data-gov 5055,
en ese orden, comprobando cada dependencia antes de continuar. No reiniciar
PostgreSQL, Metabase, loader, maquinas ni campanas. Mantener configuraciones
locales necesarias; no sustituirlas ciegamente por defaults del repositorio.

Aceptacion: comprobar version/commit efectivo, rutas v2, contratos y tipos;
despues ejecutar un unico micro-run CPU con destino nuevo y presupuesto
acotado. Verificar entregas, hashes, terminal, metricas y reconciliacion.
Probar el segundo flush idempotente sin duplicar resultados. Los escenarios
de caida permanecen en el stack desechable: no apagar produccion para probarlos.

Comparar conteos antes/despues por identidad de run; las escrituras concurrentes
del loader no son cambios propios. Conservar tablas y filas historicas. Marcar
`DEPLOYED` y `GOVERNED_RUN_PROVEN_PRODUCTION` solo con esa evidencia.

## N4 - Un rechazo permanente no puede bloquear todo indefinidamente

Reproducir el pendiente permanente observado en el retorno. Distinguir errores
temporales, errores corregibles de configuracion y un sobre realmente invalido;
un 4xx por si solo no basta para decidir esta ultima categoria.

Implementar disposicion explicita y trazable: preservar sobre y motivo,
registrar decision, y cuando corresponda enlazar un terminal corregido como
sucesor. Nunca borrar evidencia, convertir FAILED en COMPLETED ni reescribir
el mismo ID con contenido distinto. La recuperacion debe seguir el protocolo
de idempotencia y unicidad de desenlace existente.

Pruebas con base desechable: caida y recuperacion, rechazo permanente,
correccion valida, replay, aislamiento de unidades no afectadas y conciliacion
sin huecos ocultos. El estado de salud distingue pendientes recuperables,
casos adjudicados y fallos sin resolver. Un error no se oculta para mostrar verde.

## N5 - Completar los consumidores que hoy son solo wrappers

Preparar datos pequenos con procedencia y contratos explicitos para pruebas
de integracion. Sintetico es valido para mecanica, no demuestra utilidad publica
ni financiera. Los resultados de estas pruebas tambien se registran, con su
clase no cientifica, en el destino desechable correspondiente.

`feature-eng`: ejecutar el plugin real con configuracion reproducible, entradas
gobernadas y salida fresca. Inventariar todos los CSV producidos, parametros,
columnas, filas y hashes; no solo el archivo principal. Mantener entradas
crudas y salidas derivadas identificadas por separado.

`feature-extractor`: resolver la dependencia real del plugin de preprocesamiento
en un entorno reproducible por aplicacion; no depender accidentalmente del
namespace de otro checkout. Probar nombres de metricas desde una ejecucion
pequena real y capturar tambien los plots con sufijo. Esta es una prueba CPU
de mecanica, no una nueva campana de entrenamiento o seleccion.

En ambos: cambio de input cambia linaje, salida vieja rehusa, fallo conserva
terminal y costo, reintento no duplica. No llamadas a data-gov en el bucle de
fit/transform. La matriz de adopcion debe enlazar pruebas ejecutadas, no solo
existencia de un wrapper. No modernizar todos los indicadores en esta orden.

## N6 - Retomar D2 sin repetir la confirmacion

Ejecutar D2-R1..R8 de la orden enlazada, sin volver a realizar su descubrimiento.
R1-R2 pueden arrancar antes del despliegue; R3 y la sensibilidad R4 requieren
el micro-run productivo reconciliado y el registro de su campana.

El problema principal es el soporte: una metrica ausente o inconclusa no
satisface un umbral. Re-adjudicar los registros existentes y explicar cambios
en las 3.591 decisiones, incluidos los cinco casos observados. Conservar
historico; no regenerar las 3.972 unidades por cambiar la gobernanza.

La diferencia entre CPU del estimador Kalman sigue abierta bajo su tolerancia
original. No adoptar 0.006 dB despues de observarla. Aplicar el diagnostico
acotado de R4 y no generalizarlo a todos los estimadores.

Registrar evidencia historica con recibo actual y fecha de produccion original.
Para sinteticos, enlazar generador, config, semillas y arrays materializados;
clean/noise pertenecen al evaluador, no al transformador. D3 sigue en diseno.

## N7 - Programar las integraciones restantes, sin activar campañas

Actualizar el plan vigente con tareas, componentes y pruebas para agent-multi,
DOIN y live. Cada unidad cientifica necesita procedencia y terminal, sin hacer
una consulta remota por paso. Mantener el OLAP configurable como warehouse
remoto; no entregar credenciales directas del cubo a cada experimento.

DOIN: definir como el resultado referencia el terminal gobernado y como su ETL
conserva esa identidad hasta el cubo. No duplicar la metrica desde blockchain
y terminal como si fueran dos experimentos. Contrato y fixture extremo a
extremo antes de modificar publicadores reales.

Live: separar llamadas operativas ya existentes de llamadas de gobernanza.
El hallazgo de una API en `next()` no autoriza retirarla de un sistema vivo:
evaluar el replay offline y disenar la adopcion sin tocar el servicio.

Seleccion y procesamiento por variable siguen despues de caracterizacion y
pruebas de utilidad; un recibo de data-gov prueba procedencia, no ausencia de
fuga, licencia del proveedor, SNR verdadero o eficacia del operador.

## N8 - Entrega y bloqueadores

Un packet con commits completos leidos del repositorio, pruebas y estados:
`IMPLEMENTED`, `PROVEN_DISPOSABLE`, `DEPLOYED`, `PROVEN_PRODUCTION`, o pendiente
con objeto faltante, responsable y accion minima. No llamar completo a un
bloque cuyo ensayo real no corrio. Publicar resultados fallidos e inconclusos.

| Pendiente | Responsable | Accion minima |
|---|---|---|
| Integracion lake/warehouse y E2E combinado | Satoshi | N1 y N2 |
| Restriccion de reinicio en su entorno | Musashi con ejecucion permitida u operador | N3; no otra ratificacion cientifica |
| Terminales permanentemente pendientes | Satoshi | N4, probado en desechable |
| Wrappers sin ejecucion demostrada | Satoshi | N5 |
| Soporte y portabilidad D2 | Satoshi, revision posterior Musashi | N6 y orden R1..R8 |
| Contratos financieros desconocidos | Productor/integrador de datos | Evidencia de procedencia/tiempo antes de uso cientifico; no afecta ensayos sinteticos |
| DOIN y live | Satoshi disena, Musashi revisa | N7, sin despliegue en esta orden |

Adjuntar la matriz de cobertura por proyecto y evidencia del loader/outbox,
sin afirmar observacion actual de servicios desde un reporte antiguo. El
objetivo inmediato es un flujo productivo pequeno y verificable, seguido de
la re-adjudicacion D2; no una campana mas grande sobre datos no caracterizados.
