# Revision D3 K1-K5 y continuacion L1-L6

## Dictamen y evidencia independiente

Revision: `a37d4952668ac4638826dbcb1bcc11d8a6049cb4`.
No he repetido entrenamiento ni la campana distribuida. He leido el retorno,
el verificador, la bateria de gemelos y el borrador de utilidad, y ejecutado:

```bash
python -m pytest -q tests/test_d3_matrix_verify.py tests/test_d3_probe_amendment.py tests/test_d3_twin_coverage.py tests/test_olap_ingest_diagnostics.py
python -m pytest -q tests/test_d3_matrix_delta.py
```

En trading-stack: **50 passed (17.09 s) + 3 passed (0.11 s)**, sin skips.
Una primera invocacion uso por error `test_d3_delta.py`, inexistente: no corrio
ninguna prueba. Los conteos anteriores corresponden a las invocaciones corregidas.
La bateria de diagnosticos incluye su servicio efimero, no una escritura mia
en produccion. No he recontado los 1077 terminales contra la contabilidad viva.

Reejecute `verify()` en lectura sobre el run v2: 511 esperadas/completas, cero
faltantes, 710 variables totales, nueve operadores, doce pruebas, 83070 filas,
`verified=true`, cero rechazos. Comprobe aparte el SHA canonico de FREEZE.json:
coincide; declara 504 unidades sinteticas y siete toys. Son 710 variables en
total, NO 511 multiplicado por 710 variables.

Los contraejemplos K1 anteriores quedaron corregidos. No obstante, hay tres
omisiones reproducidas con el callable productivo sobre fixtures desechables:

1. **Contrato ausente aceptado.** `expected_population()` tolera UNIT.json
   ausente y convierte su binding en None; `_check_rows()` entonces omite esa
   comprobacion. `verify()` devuelve verdadero sin rechazos.
2. **Freeze no verificado.** Cambiar su digest a 64 ceros no impide aceptar.
   El verificador lee el campo pero no recalcula su identidad. El freeze REAL
   si coincide; esto es una deficiencia del verificador, no evidencia de que
   el run real haya sido alterado.
3. **Intento duplicado aceptado.** Copiar el terminal y sus filas bajo otro
   worker/shard para la misma unidad e intento no produce rechazo. La ordenacion
   escoge uno sin establecer una unica asignacion del intento.

Reproductor adjunto: `reproduce_d3_k_review_2026_09_17.py`.
Por eso acepto las correcciones al alcance medido, pero la matriz todavia no
tiene aceptacion externa final. No se declara fuga causal observada ni se
invalidan arbitrariamente las mediciones conservadas.

## L1. Cerrar el verificador, sin otra campana

Congelar primero los tres casos anteriores y corregirlos. Todo objeto requerido
para derivar identidad y poblacion debe existir y validarse; ausencia no es
permiso para omitir un binding. Incluir TOYS.json y TOY.json: no aceptar poblacion
o nombres de variables indeterminados cuando el freeze declara toys.
Recalcular el freeze canonico, validar esquema y cardinalidades, y cotejar su
identidad contra el registro de campana ya conservado. Un self-digest por si solo
no acredita que se este leyendo el diseno que rigio la ejecucion.

Validar unicidad de unidad/intento y asignacion worker/shard contra el despacho.
Si hay copias de transporte legitimas, distinguirlas de intentos nuevos con una
regla explicita; resultados contradictorios jamas se resuelven por orden de ruta.
No aceptar silenciosamente estados terminales desconocidos, ids discordantes o
registros fuera de la poblacion. Pruebas de contratos ausentes, toys ausentes,
freeze cambiado, duplicado identico y duplicado contradictorio antes del arreglo.

Reverificar v1 y v2 desde los bytes preservados, con resultados sucesores del
verificador y delta explicado. No modificar sus matrices originales, ni repetir
mediciones por un cambio exclusivamente del verificador.

## L2. Sensibilidad del control wavelet, no relajacion causal

La interpretacion de las cinco comparaciones no sensibles del retorno es una
hipotesis que debe demostrarse por unidad: cortes, indices, emisiones, soporte
y posiciones realmente perturbadas. Cruzar geometricamente el corte es necesario
para ciertos controles, pero no demuestra sensibilidad por si solo (coeficiente
cero, saturacion o falta de datos pueden impedirla).

Congelar enmienda del diagnostico antes de medir. Cero comparaciones sensibles
es INSUFFICIENT_TEST; una infraccion causal observada nunca se descarta por una
declaracion de soporte del mismo operador. No restringir las pruebas causales
del candidato a la mascara de sensibilidad del gemelo.

Probar centro conocido, impulso futuro justo tras el corte, pesos extremos cero,
missingness, bordes y restart. Exigir controles deliberadamente no causales que
fallen por valor, mascara de disponibilidad y tiempo de emision. Mantener el
control con datos completos y los tres rechazos warm-up separados. No cambiar
soporte 50, imputacion, umbrales o parametros para hacer pasar los cinco casos.

Si solo cambia la interpretacion y hay hechos suficientes, readjudicar sin
reentrenar. Si requiere medicion nueva, piloto y replay gobernado de TODO el
alcance afectado predeclarado (tambien casos antes aceptados), versionado y CPU.
No repetir todos los operadores por comodidad: justificar dependencias y alcance.

## L3. Disposicion del sobre DEVELOPMENT y nombres de campana

Decision: conservar el sobre `475b93fb...` y su historia; no borrar, no ascender
a evidencia cientifica y no cambiar retroactivamente su procedencia. Registrar
una disposicion aditiva de ingestion operativa accidental, ligada al digest
completo rederivado, con elegibilidad cientifica falsa. Verificar sus hijos y
el efecto en vistas de ciencia/desarrollo/operacion. DEVELOPMENT por si solo
no significa que todas sus filas sean admisibles para el estudio actual.

La sonda de idempotencia debe elegir por identidad un sobre YA presente y
comparar contenido, no solo escoger el ultimo archivo. Si ninguno existe, no
escribir uno antiguo como sustituto. Probar que una sonda no introduce runs.

El sobre v2 bajo `campaign_key d3-mechanics-v1` se conserva. Registrar relacion
correctiva consultable y probar que seleccionar v1, v2 o el actual no mezcla
poblaciones. Usar run/diseno/digest, no prefijos de nombre. No reemitir las
6390 unidades para arreglar una etiqueta.

## L4. Completar el tratamiento de errores en data-warehouse

Esto pertenece al trabajo, no es un bloqueo que deba resolver el owner.
En el repo propietario distinguir entrada invalida tipada (400/422), fallo
interno (500) e indisponibilidad realmente transitoria (503). No convertir
todas las excepciones de base de datos o de programacion en transitorias.
Mantener diagnostico acotado y sin secretos en el outbox, sin perder el sobre.

Pruebas con el host y proveedor reales en stack desechable: documento invalido,
defecto interno inducido, caida transitoria, recuperacion y segundo drenaje sin
duplicados. Tras pasar, adoptar por el procedimiento existente y comprobar
contenido antes/despues. No tocar otros servicios ni eliminar WAL conservados.

## L5. Corregir el diseno de utilidad antes de congelarlo

El documento 12 es un BORRADOR, no un protocolo listo para correr. Concretar:

- MAE/log-loss miden perdida predictiva del modelo de prueba. No denominar esa
  diferencia cantidad de informacion en bits ni informacion mutua.
- Poblacion y elegibilidad exactas por variable/regimen, no por el promedio
  del dataset. Separar desarrollo sintetico/toys de confirmacion publica y
  revalidacion financiera. Sin nueva exigencia de permiso entre tareas de diseno.
- Ventanas por identidad de observacion, decision, disponibilidad y horizonte
  del target. La purga no puede depender solo de lookback+delay ignorando etiquetas
  futuras. Normalizacion, seleccion y todos los ajustes dentro del train.
- Comparacion pareada sobre las mismas filas emitibles y reporte separado de
  cobertura, missingness, costo e intentos fallidos; no ganar eliminando casos
  dificiles. Presupuesto agotado es un desenlace registrado, no descarte invisible.
- Ramas cruda, transformada y aumento si la hipotesis lo exige; control de capacidad
  predeclarado. No escoger 'la primera componente' arbitrariamente para igualar
  dimensiones. Explicar que informacion recibe cada modelo y que contraste identifica.
- Target, horizonte, modelo, margen, semillas, bloques, unidad estadistica,
  dependencia entre variables/series, comparaciones multiples y reglas de
  abstencion concretas. Seleccion en desarrollo, test reservado sin uso repetido.

Implementar pruebas del arnes con datos fabricados de verdad conocida y controles
negativos, sin puntuar la reserva ni reclamar resultados cientificos. Entregar
diseno y arnes listos para revision. Esta orden NO abre el experimento de utilidad.

## L6. Cierre completo y operacion

L1-L5 autorizadas sin pedir continuacion entre bloques. Distribuir las mediciones
CPU necesarias por memoria disponible con identidades propias; no lanzar GPU ni
interferir con trabajos del owner. Todo resultado con terminal, costo, procedencia
y conciliacion por contenido. Conservar fallos y generaciones originales.

Actualizar work plan, trazabilidad y packet con PRE/POST, pruebas exactas,
exclusiones, digests leidos y commits publicados. Resolver autonomamente los
defectos del alcance; continuar bloques independientes si uno se atasca.
Salida: `D3_MECHANICS_VERIFIER_CLOSED_AND_UTILITY_DESIGN_READY_FOR_REVIEW` o
estado parcial con evidencia precisa. Indice, Metabase y terminos siguen aparte.
No declarar su cierre ni usarlos para detener la mecanica o el diseno CPU.
