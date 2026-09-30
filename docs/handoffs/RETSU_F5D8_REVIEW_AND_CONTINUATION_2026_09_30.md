# Revision de f5d8e277 y continuacion paralela

Fecha: 2026-09-30. Destinatario: Retsu. Revision limitada, no aprobacion global.

## Hallazgos reproducidos

1. Alta: `doin-node/src/doin_node/archive/warehouse.py`,
   `project_from_reference`, no compara el manifiesto devuelto con el solicitado.
   Una fuente que devuelve B cuando se solicita A inserta performance 0.2 de B.
   El consumidor acepta una respuesta equivocada del adaptador.
2. Alta: la misma ruta confia en `VerifiedArchive.records`, cuyos diccionarios
   siguen siendo mutables. Modificar performance a 999 despues de verificar
   conserva los bytes y digests originales y aun asi se inserta 999. El tipo
   congelado exterior no certifica el contenido consumido.

Ambos casos se ejecutaron contra core a50a33e y node 6c64646, sin servicios ni
datos de mercado. Probe: `docs/audits/retsu_reference_probe_20260930.py`.
Resultado: `docs/audits/evidence/RETSU_F5D8_REVIEW_20260930/RESULTS.json`.
No son evidencia de corrupcion desplegada. La ruta `project_metrics` si verifica
el envelope; no retiro esa correccion ni la confundo con la ruta por referencia.

## Estrategia: lo aceptado y el alcance pendiente

Reejecute las tres suites de accounting, micro y support contra e7966f33:
41 passed en 2.61 s, CPU limitada a 2 GiB/120 s y CUDA oculta. No hice otra
simulacion de mercado. La reparacion de margen y caja tiene cobertura ejecutada;
no equivale a certificar todos los estados posibles del broker.

Los cuatro puntos sinteticos no ordenan la importancia del corto y del largo.
Persistencia en el largo impide entrar en este fixture; no demuestra que el
corto carezca de utilidad. Persistencia real y ruido con igual MAE son controles
distintos y deben seguir separados.

Por inspeccion de `app/factorial_harness.py`: las dos orientaciones fijan la otra
familia como IDEAL. No hay barrido de intensidades, ni control fijo de
persistencia; `execute_manifest` rechaza siempre. Las 18 entradas son un
manifiesto, no un barrido ejecutable. Las semillas no crean replicas distintas
para entradas deterministas. `family_input` reinicia el generador en cada llamada:
reutilizar la misma semilla por origen repetiria el vector de perturbacion.
Normalizar cada vector por su propio MAE tampoco es ruido gaussiano iid puro.
Son limitaciones previas a ejecucion, no resultados de una campana que no corrio.

## Ordenes independientes

A. Archivo DOIN: congelar el probe como PRE; comprobar igualdad con el digest
solicitado y reconstruir los registros consumidos desde bytes verificados.
No basta cambiar el booleano o envolver otro objeto. Probar referencia correcta,
equivocada, mutacion posterior, archivo ausente, reintento y rollback sin filas
parciales. Conservar MAE y contratos de escala. Usar solo SQLite/archivos
desechables; no modificar cadena, lago desplegado ni warehouse productivo.
Revisar tambien el conteo de inserciones: `get_rounds` limita a 1000, por lo que
la diferencia de longitudes no cuenta correctamente inserciones posteriores.

B. Estrategia: completar primero el ejecutor SINTETICO con el mismo plugin y
contabilidad reconciliada. Dos orientaciones, otra familia fija en persistencia
real como contraste principal e ideal como control adicional. Declarar niveles
de ruido, soporte comun, horizontes y semillas antes de medir. Mantener baselines
deterministas sin fingir replicas independientes; perturbaciones reproducibles
por origen/familia/horizonte y semillas pareadas entre configuraciones. Calibrar
escala fuera de la poblacion puntuada y reportar MAE logrado por horizonte,
ademas del agregado. Probar semilla, independencia entre origenes, no ruido sobre
acciones, costes, caja/equity, operaciones abiertas y llenados parciales/rechazos.
Esto autoriza desarrollo y pruebas sinteticas pequenas dentro del limite CPU
vigente, NO B0, datos reservados ni calibraciones financieras. Si no cabe, entregar
ejecutor probado y coste sin iniciar una campana de mayor presupuesto.

C. Clasificacion/GPU: mantener los artefactos y los estados NOT_RUN. No ejecutar
codigo remoto, diagnostico GPU, instalaciones en workers ni calibraciones.
No convertir descargas o pruebas de snapshot en calidad medida. Mantener
separadas las dependencias PyTorch y TensorFlow. No pedir nuevamente autorizacion
como requisito para cerrar A/B; no hay concesion nueva en este paquete.

Retsu coordina A y B en paralelo con worktrees separados y recursos admitidos.
No anunciar agentes despachados sin acuse. No usar la GPU para llenar ocupacion
sin tarea autorizada. Integrar cada entrega al terminar, sin esperar las otras.

## Retorno obligatorio

Commits y pruebas reproducibles; PRE/POST de ambos fallos; celdas realmente
ejecutadas frente a solo declaradas; tiempos y memoria observados. Resultado
sintetico separado de utilidad financiera. Ninguna exactitud de clasificacion,
calibracion o reparacion GPU se declara sin haberla ejecutado. No borrar evidencia.
