# Orden: publicar ambos hosts y completar su adopcion operativa

Fecha: 2026-09-14. Ejecutor: Satoshi. Coordinador: Musashi.
Responde al retorno predictor `939fb413378b8e4c8a62432de3a5005bbf44ea4a`.

## 1. Decision explicita: PUBLICOS

Publicar los repositorios **harveybc/data-lake** y
**harveybc/data-warehouse** como **publicos**, conservando la historia local.
Crear sus remotos y subirlos forma parte de esta orden. No volver a detener
este trabajo para preguntar si se permite crear esos dos repositorios o que
visibilidad usar. Esta decision no cambia la visibilidad de otros repositorios.

Los dos repositorios locales existen, estan en master y estaban limpios al
emitir esta orden. Verificar otra vez su estado y los remotos antes de actuar;
no sobrescribir trabajo concurrente ni crear un repositorio duplicado.

## 2. Publicacion inmediata

1. Verificar que ambos paquetes contienen codigo, pruebas, README, AGENTS.md,
   configuracion de ejemplo y estado del proceso de implementacion. Las
   instrucciones de instalacion y los ejemplos deben usar plugins externos
   realmente disponibles, no paquetes que solo existen en un entorno local.
2. Publicar ambos repositorios, con master como rama inicial predeterminada.
   No cambiar la historia local para fingir una cronologia distinta. Publicar
   codigo, pruebas y ejemplos; excluir datos restringidos, configuraciones
   operativas privadas y artefactos experimentales ajenos a los ejemplos.
3. Publicar los proveedores `financial-data-store` y `predictor-olap-store`
   desde sus repositorios correspondientes. Integrar solamente sus paquetes,
   pruebas y documentacion necesarios en las ramas predeterminadas. No fusionar
   indiscriminadamente las ramas cientificas. Mientras se integra, una revision
   publica fijada puede servir como origen instalable y debe declararse asi.
4. Desde un directorio y entorno nuevos, clonar por las URLs de GitHub e
   instalar los hosts y proveedores publicados. Repetir las pruebas de
   descubrimiento y la paridad 8/8 y 11/11. Las instalaciones no deben depender
   de un checkout hermano mediante `sys.path`, enlaces locales o un editable
   no declarado. Leer los commits finales de Git, no escribirlos de memoria.

Informar las dos URLs y commits en cuanto esten publicados y continuar al
despliegue. La publicacion es un hito, no el final de esta orden.

## 3. Validacion antes de sustituir servicios

El retorno declara pruebas desechables, no despliegue productivo. Revisar sus
resultados contra las revisiones que se van a instalar y repetir lo que cambie.
No se solicita regenerar los experimentos D2 ni repetir su banco cientifico.

Antes del cambio deben pasar:

- Contrato completo data-gov -> nuevo data-lake/proveedor -> nuevo
  data-warehouse/proveedor, con hashes y conciliacion exacta.
- Disponibilidad temporal, holdout y cortes con el mismo comportamiento que
  los adaptadores actuales; conservar el test de HoldoutError 403 frente a 500.
- Resultados COMPLETED, FAILED y REFUSED con registro e idempotencia; un
  rechazo no puede ocultar una excepcion de implementacion.
- Inventario, metadatos de recursos, tablas/vistas y configuracion AdminLTE,
  con prueba de escritorio y movil. Un cambio pendiente no se presenta como
  ya aplicado.
- Arranque reproducible desde los paquetes publicados y configuracion efectiva
  registrada, con limites de recursos y procedimiento de rollback.

Si aparece una regresion, corregirla con una prueba focal y continuar. No
pedir otra autorizacion del owner para esa correccion dentro de este alcance.

## 4. Transicion productiva autorizada

La sustitucion controlada de los adaptadores actuales por ambos hosts nuevos
esta autorizada como continuacion de la adopcion ya ordenada. No hace falta
otra decision sobre si desplegar una vez satisfechas las pruebas anteriores.

1. Comprobar trafico y trabajos en curso. Respaldar configuraciones,
   contabilidad y estado necesario para recuperar el servicio. Conservar los
   comandos, versiones y configuraciones anteriores para rollback.
2. Usar los **mismos datos y la misma base OLAP**. Conservar store/resource IDs,
   contratos, recibos, historial y outboxes; no copiar datasets para acomodar
   la estructura de repositorios ni recrear tablas historicas.
3. Mantener los servicios actuales mientras se preparan y prueban los nuevos
   procesos. Sustituir solamente los adaptadores del lago y del warehouse en
   una ventana sin entregas activas. Si no es posible en ese momento, terminar
   todo lo independiente y registrar la condicion operativa concreta.
4. Mantener data-gov en 5055 sin reinicio si su configuracion no cambia. Los
   endpoints del lago y warehouse deben conservar la configuracion consumida
   por data-gov. No reiniciar PostgreSQL, Metabase, loader OLAP ni servicios
   de trading. No tocar GPUs, terminales de mercado ni campanas cientificas.
5. Si la sustitucion falla, restaurar los adaptadores anteriores con sus
   datos intactos, registrar el fallo y corregir la causa. No declarar adopcion
   por tener procesos escuchando en los puertos.

## 5. Prueba productiva y adopcion por consumidores

Ejecutar un microexperimento CPU pequeno, expresamente no cientifico, con
salidas nuevas, que atraviese **ambos hosts nuevos** mediante data-gov.
Usar un solo proceso, sin GPU, entradas acotadas y un presupuesto declarado
antes del arranque. No reutilizar una salida previa como prueba del nuevo
despliegue.

Registrar y verificar:

- Datos efectivamente entregados, hashes, contrato temporal y version de cada
  host y proveedor que participo.
- Codigo/configuracion del consumidor, parametros, costos, artefactos y
  desenlace en el warehouse real.
- Conciliacion sin unidades ni registros sin correspondencia; repetir el envio
  del mismo desenlace no agrega duplicados.
- Deltas de filas atribuibles al microexperimento y preservacion del historial.
  No usar un conteo viejo como estado inicial ni atribuirse escrituras de
  otros procesos que hayan ocurrido simultaneamente.
- Salud posterior y continuidad del loader sin reinicios.

Actualizar configuraciones y matriz de adopcion para preprocessor, feature-eng,
feature-extractor, predictor y experimentos offline de agent-multi/DOIN.
La ruta nueva debe quedar como predeterminada para las siguientes campanas;
no basta con mencionarla en los README. Validar cada consumidor por separado
y declarar los que aun no se han probado. Para live, solo configuracion y
replay offline: esta orden no autoriza activar operaciones reales.

## 6. Trabajo que no debe volver a ponerse como decision pendiente

- Visibilidad de los nuevos repositorios: PUBLICA, decidida en esta orden.
- feature-extractor: forward port ya elegido y reportado; verificar su revision
  publicada y adopcion, no volver a preguntar pin versus port.
- financial-data: procedencia por archivo aceptada; no se pide fusionar toda
  la rama de investigacion para alterar la ascendencia de Git.
- Kalman: la tolerancia original y sus limites siguen vigentes; resolver esa
  investigacion no es prerrequisito para publicar o adoptar los hosts.

## 7. Cierre requerido

Actualizar el [work plan obligatorio](../integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md)
y el estado de trabajo despues de cada hito. Distinguir PUBLICADO,
PROBADO_EN_DESECHABLE, DESPLEGADO y PROBADO_EN_PRODUCCION.

Entregar URLs reales, commits, pruebas del codigo final, servicios/versiones
desplegados, recibo del microexperimento, conciliacion y matriz de consumidores.
No cerrar con "creados y en uso" si solo se usaron en una base desechable.
La orden termina con adopcion operativa demostrada y los faltantes por
consumidor identificados, no con una nueva pregunta de publicacion.
