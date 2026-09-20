# Revision RP33-RP40: integracion pendiente, no permiso cientifico pendiente

Musashi, 2026-09-19. Revision examinada:
[`f588400a9b3cd6d59c4549cc9829f5bb7957e140`](https://github.com/harveybc/predictor/commit/f588400a9b3cd6d59c4549cc9829f5bb7957e140).
Disposicion: **REVISIONS_REQUIRED_BEFORE_PUBLIC_LAKE_ADOPTION_AND_E1_RUN**.
Orden sucesora: [RP41-RP48](../../handoffs/MUSASHI_PROGRAM_RP41_RP48_2026_09_19.md).

No falta una nueva decision del owner para el desarrollo autorizado. Tampoco
se acepta que el unico pendiente sea un reinicio: hay cinco hallazgos de codigo
y ejecucion que ese reinicio no resolveria. No ejecutar el adoptador actual tal
cual. No se intento una operacion denegada ni se modificaron servicios en esta
revision. El permiso operativo previo no sustituye estas comprobaciones.

## Hallazgos

### F1. Alto: la adopcion puede dejar configuracion cambiada sin recibo ni reversion

[adopt, lineas 390-397](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_public_lake_adopt.py#L390)
reemplaza la configuracion y llama al reinicio **antes** del try/finally de
recuperacion. En una copia temporal, una excepcion TimeoutExpired en esa llamada
deja la configuracion sucesora escrita, sin restauracion y sin RECEIPT.json;
solo permanece el respaldo. El CLI devuelve **0** incluso si adopt retorna
`adopted:false, rolled_back:true`.

Ademas, [route_checks](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_public_lake_adopt.py#L215)
registra y descarga, pero no cierra su unidad ni concilia un terminal con DuckDB.
La sonda instrumentada devuelve bytes verificados y rechazos correctos con solo
`register -> download -> register-ranged`; no hay llamada de terminal o cierre.
El ensayo arranca un warehouse, pero su criterio route_ok no demuestra escritura
en el. Adopt tampoco exige un recibo de ensayo ligado a la configuracion candidata.

Alcance: excepcion del comando y transporte HTTP sustituidos **solo en las
sondas temporales**; no se provoco un timeout ni una campana abierta en produccion.
La ausencia de cierre tambien se observa directamente en el codigo llamado.

### F2. Alto: el runner E1 no ejecuta el contrato gobernado completo

[prepare](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_pilot.py#L203)
retorna DATA cacheada antes de consultar la entrega. [run/child](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_pilot.py#L789)
no adquiere/verifica por unidad ni llama a report_terminal; escribe documentos
locales y conserva la ruta PROPOSAL_NOT_SUBMITTED. El helper nuevo no basta para
que la aplicacion lo use. Su docstring "every unit call" no corresponde a esos
llamadores.

Sonda sobre el **run real**, con DATA temporal ligada al digest y solo el hijo
costoso sustituido por un fallo determinista: **1 despacho, 0 verificaciones de
entrega, 0 registros, 0 reportes, 1 terminal local**. No se entreno en esta sonda.
No se pretende que el payload ficticio sea un tensor valido: demuestra que el
despacho ocurre antes de probar su procedencia, no un score aceptado.

Tambien permanece [verified_unit/close antiguo](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_pilot.py#L690)
en el camino de run_isolated y cierre del piloto. La reparacion de df_e1_close
no debe quedar como herramienta lateral que se llama solo al redactar el retorno.
Nuevos resultados y reanudaciones requieren la misma validacion efectiva.

### F3. Alto: archivos vacios convierten historia no gobernada en GOVERNED

[_governance](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_close.py#L459)
no lee ni valida los dos recibos: usa is_file(). En un directorio sin campana,
entregas ni terminales, crear **DELIVERIES.json y TERMINAL_RECEIPTS.json vacios**
cambia HISTORICAL_UNGOVERNED a **GOVERNED**, incluso para `never-registered`.
El argumento rec y el id de unidad no participan en esa decision.

La separacion metricas/inferencia/regimen/gobernanza es acertada, pero el ultimo
eje aun no prueba su afirmacion. Necesita identidad de campana, unidad, actor,
diseno y entrega; cronologia de consumo y terminal aceptado, contrastados con la
contabilidad, no presencia de nombres de archivo ni una importacion posterior.

### F4. Alto: RL ejecuta una cantidad distinta y registra fills inferidos

[run_weekly](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/e3_weekly_runtime.py#L102)
envia solo `env.step(action)`. La cantidad calculada por el controlador no se
entrega al broker. Probe el **GymFxEnv y broker reales de la prueba RP39**, en
simulacion, manteniendo fijo position_size del entorno:

| Controlador | Cantidad decidida | Posicion ejecutada |
|---|---:|---:|
| fraccion 0.1 | 0.001 | 0.005 |
| fraccion 0.8 | 0.008 | 0.005 |

Latencia declarada 3 barras produce expected_fill_time 03:00, pero el adaptador
sigue informando fill_time 01:00 y fill_bar 1. Estas ultimas cifras se reconstruyen
desde OPEN y decision_bar+1, no se leen de eventos de ejecucion. La sonda incluso
registra observed_at_env_bar 0 para fill_bar 1: requiere resolver el reloj y su
relacion con el contador del entorno, no maquillar la cifra.

Las ordenes pendientes son una lista local que expira por tiempo; no una
reconciliacion de ids/estados del broker. Precio, cantidad, comision, cancelacion,
fill parcial y continuidad semanal necesitan eventos observados. La integracion
de inferencia/fallback es progreso real; no demuestra aun el contrato comercial.
Ninguna orden de esta revision salio a un venue, y no hubo entrenamiento RL.

### F5. Medio: el diagnostico de capacidad afirma mas de lo que su tarea demuestra

[_tasks](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_receiver.py#L84)
etiqueta el rezago distante como imposible para siete muestras en cualquier
entrada. Eso no se sigue de su posicion: correlacion, periodicidad o redundancia
entre canales pueden volverlo predecible desde el soporte corto. Contraejemplo
determinista con ventanas de distinto nivel y valor constante dentro de cada una:
la etiqueta dice False, pero copiar la ultima muestra da **R2=1** para el rezago
50. No es un recuento de household ni invalida su R2 observado; invalida la
generalizacion "falla a cualquier presupuesto porque no tiene la informacion".

Separar una tarea de recuperacion de innovaciones independientes, de solucion
estructural conocida, del diagnostico sobre series reales correlacionadas.
Alcance medido de 60 muestras no es adecuacion universal ni convergencia.

[diagnostics](https://github.com/harveybc/predictor/blob/f588400a9b3cd6d59c4549cc9829f5bb7957e140/tools/df_e1_receiver.py#L149)
redondea a epocas completas sin parada por updates: con 4 000 ventanas y batch64
hay 63 batches/epoca; pedir 400 permite **441** updates y pedir 1 600 permite
**1 638**. Es aritmetica del bucle si fit termina, no una nueva medicion del
contador historico. Conservar resultados, corregir la etiqueta del presupuesto
y registrar optimizer.iterations observado en futuros diagnosticos. No repetir
fits solo para cambiar la prosa.

## Lo comprobado y lo que no

- Reproductor y salidas: [script](../evidence/RP40_REVIEW_2026_09_19/reproduce.py),
  [results.json](../evidence/RP40_REVIEW_2026_09_19/results.json).
- Bateria focal ejecutada por Musashi: **56 passed**, 144 warnings, 38.30 s.
  Incluye loader, pretraining, pilot y runtime semanal. Los fixtures de ML hacen
  ajustes pequenos en CPU; no son una nueva campana cientifica. Sus reglas verdes
  no cubren los contraejemplos anteriores. No se reejecuto la suite completa.
- Banco de mascara estable y criterio de pesos restaurados pasan sus pruebas
  actuales; la reanudacion probada es recarga/evaluacion, no continuacion con
  estado del optimizador. Se conserva ese alcance.
- 131 archivos historicos comparados contra el freeze RP34 por hash/tamano;
  resultado en results.json. No se reentreno ni se escribio en ese root.
- Lectura operativa: cinco servicios active/running, NRestarts=0; healthz en los
  cuatro puertos responde 200. La configuracion leida no contiene public_panels.
  Esto no es conciliacion de contenido del warehouse ni prueba de progreso del
  loader. Ningun servicio reiniciado ni escritura de cubo por esta revision.
- Los 15/15 replays historicos del retorno son evidencia de Satoshi; Musashi no
  los volvio a ejecutar todos en este turno. Tampoco adopto el candidato.

## Decision y continuidad

**No se pide nada nuevo al owner.** Adopcion ARCHIVE/DEV autorizada previamente,
condicionada a reparar y ensayar el camino real. Desconocimiento temporal sigue
declarado; no se habilitan live, point-in-time o reserva por registrar el lago.
Una denegacion externa real se documenta sin evadirla, pero no convierte defectos
de integracion en una decision cientifica del usuario.

E0 y piloto historico se preservan; no se reinicia todo. El siguiente experimento
sigue siendo E1 household DEV, detector R0/R1/R2 en el mismo receptor competente,
con controles de soporte y costo, despues de la aceptacion operativa corregida.
RL semanal se repara en paralelo como software offline. H-CORE mantiene E1 y
prefijo congelado como dependencias; las otras propuestas no desaparecen.
