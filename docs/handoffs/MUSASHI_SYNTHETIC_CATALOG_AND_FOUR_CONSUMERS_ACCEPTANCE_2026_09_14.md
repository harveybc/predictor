# Catalogo sintetico activado y cuatro consumidores comprobados

Fecha: 2026-09-14. Ejecucion y revision: Musashi.
Autorizacion: el owner ratifico activar el catalogo y continuar los trabajos.
Retorno revisado: predictor `0723fd16ed2ff84f3d5da15a110026c2ebb4dc9d`.

## La activacion esta completada

No queda un reinicio pendiente del owner. Musashi activo la configuracion
preparada por Satoshi y reinicio exclusivamente `crispdm-data-lake-synthetic`.
Se verificaron cero entregas pendientes antes del cambio y se conservaron
la configuracion anterior y la candidata para reversion.

La diferencia de configuracion contiene solo include_globs y los siete
contratos nuevos. La raiz, el proveedor instalado, los endpoints y el
contrato de panel.csv permanecieron iguales. El archivo activo sigue separado
del archivo pendiente. Los otros tres servicios y el loader conservaron sus
procesos; PostgreSQL y Metabase no se reiniciaron.

Antes de activar, Musashi regenero los siete archivos desde el generador y
la semilla declarados en un directorio temporal. Coincidieron los bytes de
todos los archivos, el manifiesto completo y los contratos. El catalogo
sintetico paso de 1 a 8 recursos. panel.csv conserva su digest previo.

Generador: `1ca00f81a282b96236b5566142919bba3f842562a1c9763669a5f82b5b586474`.
Configuracion activa aceptada:
`88a35021f19dc06b23179e06fa407f060f2c6a94106bfd93bf7d10ad36b8d2f1`.

## Prueba ejecutada contra produccion

Se ejecuto el arnes de Satoshi, sus wrappers y sus configuraciones acotadas,
secuencialmente, con CUDA deshabilitado y limite de memoria de 6 GiB para
el grupo de procesos. No se uso un pipeline simulado. Alcance NON_GOVERNING:
comprobar transporte y mecanica, no utilidad cientifica.

| Consumidor | Salida real | Salida antigua | Fallo intencional |
|---|---|---|---|
| feature-eng | COMPLETED | REFUSED | FAILED con costo |
| feature-extractor | COMPLETED | REFUSED | FAILED con costo |
| preprocessor | COMPLETED | REFUSED | FAILED con costo |
| predictor | COMPLETED | REFUSED | FAILED con costo |

El cubo paso de 20 a 32 terminales. Delta de esta ejecucion: 12 terminales,
119 metricas, 28 recibos de datos y 39 artefactos. Musashi consulto los doce
desenlaces, verifico sus entradas y contratos contra el manifiesto regenerado,
y comprobo la conciliacion de cada campana: sin faltantes ni registros
presentes solo en una parte. No se emitieron borrados ni actualizaciones de
resultados anteriores.

[Recibo publico con campanas y conteos](../audits/evidence/repro_runs/musashi_four_consumers_production_20260914.json).
Los logs, respaldos, configuraciones operativas y recibos completos estan
conservados en el estado privado de la operacion.

## Distinciones que se conservan

- La ruta real esta respaldada por la configuracion activa http_lake hacia
  el host sintetico y los recibos verificados. El catalogo visible para el
  principal no ofrece discover sobre ese almacen: el arnes escribe null en
  esa descripcion aunque la descarga funcione. No se fabrico un resultado de
  descubrimiento ni se amplio la politica solo para poner verde el reporte.
- Los segundos vaciados de outbox no enviaron nada. La recuperacion de un
  terminal varado que Satoshi publico fue una prueba desechable sobre
  preprocessor; no se repitio una caida del warehouse productivo ni se afirma
  haber probado ese caso independientemente para los otros tres wrappers.
- Las particiones sinteticas tienen semillas distintas, pero comparten el
  mismo intervalo de calendario. Sirven para esta mecanica; no demuestran
  particiones cronologicas disjuntas ni ausencia de fuga en fit/transform.
- Las columnas sinteticas con nombres de indicadores prueban esquema y
  transporte, no reproducen necesariamente la semantica de esos indicadores.
- Se re-ejecutaron las ocho pruebas de clasificacion de predictor y las 19
  de la caracterizacion financiera de PR 2. Pasan, pero algunos tests de
  clasificacion reconstruyen el parser o inspeccionan texto, y no sustituyen
  pruebas conductuales del entry point real.

## Semantica financiera

Se acepta retirar close_time como disponibilidad demostrada. La distincion
de archivo retrospectivo permite describir lo conocido sin inventar latencia.
Esto no constituye una instalacion de PR 2 ni una aprobacion de uso financiero
point-in-time. No se cambio ningun contrato financiero.

Correccion todavia requerida en el instrumento propuesto: la prueba que
inyecta received_time lo declara publication MEASURED y mantiene reception
UNOBSERVED. Un tiempo de recepcion no demuestra la publicacion del proveedor;
hay que registrar el tipo de reloj y su evidencia por separado.
Las categorias de barras derivadas solo de duracion deben seguir declaradas
como clasificacion geometrica, no explicacion confirmada de su productor.

No se autoriza implicitamente una revision no probada de la API de archivos.
Prepararla y comprobarla es una tarea del equipo, no una aprobacion pendiente
del owner. La investigacion financiera no bloquea las pruebas sinteticas.

## Continuacion

La activacion y la mecanica de los cuatro consumidores quedan cerradas en
el alcance anterior. No repetirlas como si fueran trabajo pendiente.
Continuar con [pruebas causales, roles de columnas y replay DOIN](MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md).
