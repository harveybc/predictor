# Revision ML y adopcion de RP25-RP32

Fecha 2026-09-19. Base revisada: [60daac965143d9e9b6c1a357420b54c74f83afaf](https://github.com/harveybc/predictor/tree/60daac965143d9e9b6c1a357420b54c74f83afaf).
Disposicion: **conservar las mediciones; piloto no gobernado, cierre no aceptado;
corregir validacion ML y adopcion antes del siguiente entrenamiento**.
No es necesario reiniciar todo el programa. No hay decision del owner pendiente
para registrar un recurso publico y ejecutar el desarrollo ya autorizado.

## F1. Alto: se entreno fuera de la gobernanza exigida

[seal/run](https://github.com/harveybc/predictor/blob/60daac9/tools/df_e1_pilot.py#L169)
declara de antemano terminales locales y continua hasta entrenar. La orden RP30
no autorizaba esa sustitucion. Configuracion de arranque revisada: cuatro lagos,
ninguno de paneles publicos. Consulta de SOLO LECTURA a la contabilidad activa:
**0 registros** de `satoshi-e1-household-dev-20260919-e1-household-dev-pilot` y
0 campanas cuyo nombre contenga household. Los cuatro servicios consultados
estan active/NRestarts=0; no se tocaron.

Un documento local con schema de terminal no es un terminal aceptado ni prueba
de entrega. Registrar el recurso despues no convierte la lectura pasada en
gobernada. Conservar los 15 documentos con sus fechas/digests, registrar su
importacion retrospectiva como tal y mantener fuera de la evidencia gobernada
prospectiva. Investigar/reparar la falta del catalogo es tarea tecnica, no decision
cientifica ni de cuenta financiera del owner. No inventar disponibilidad en vivo
para estos archivos retrospectivos.

## F2. Alto: el cierre E1 reproduce el cociente, no la tarea que dice verificar

[`verified_unit`](https://github.com/harveybc/predictor/blob/60daac9/tools/df_e1_pilot.py#L488)
solo recomputa MASE en arrays suministrados por el mismo resultado. No liga sus
origenes, etiquetas, denominador y pesos al job/diseno/datos aceptados, ni verifica
MAE y las demas metricas reportadas. `abs(a - nan) > tolerance` es falso.

Reproductor sobre copias con digests locales coherentes:

| Caso | Resultado |
|---|---|
| MAE alterado a 999 | Aceptado |
| MASE del registro = NaN | Aceptado |
| diseno/datos/cell_id ajenos | Aceptado |
| origenes desplazados 123 filas | Aceptado |
| sin pesos ni job, con cell y arrays | Aceptado |

En el **cierre completo**, cambiar DESIGN a digest de ceros y horizonte 120
publica ese nuevo horizonte con las mismas medias, todas las unidades presentes
marcadas verificadas. No es solo una funcion auxiliar mal llamada. El nuevo
verificador debe derivar poblacion y tarea del registro independiente, revalidar
todos los campos cientificos, y distinguir metricas de arrays de replay de pesos.

## F3. Alto: la validacion del autoencoder cambia durante el stopping

[`WindowBatches`](https://github.com/harveybc/predictor/blob/60daac9/tools/df_e1_pilot.py#L253)
usa `(seed, epoch, batch, 7)` para la mascara, tambien con shuffle=False, y siempre
incrementa epoch. La validacion de AE y su evaluate final usan este dataset.
Con pesos del AE s1 intactos, mismas 16 ventanas reales y mismos valores limpios:
**2785 posiciones de mascara cambian; MSE 0.259784 -> 0.326498** al avanzar epoch.
No se entreno un solo paso en esta comprobacion.

El criterio mezcla cambio del modelo y del estimulo de validacion. Muestrear
mascaras puede estimar una esperanza, pero aqui no hay banco fijo ni incertidumbre
de esa estimacion; restaurar el minimo no demuestra restaurar rendimiento sobre
un criterio comparable. Usar mascara/banco fijo por identidad de ventana para
validacion, distinto del ruido de entrenamiento; evaluar checkpoint restaurado
sobre ese mismo criterio. Probar con el runner AE real y sus callbacks, no solo
`masked_pretrain`, que es otra ruta.

Keras permite modificar PyDataset mediante on_epoch_end y evalua validation_data
durante fit: [PyDataset](https://keras.io/api/utils/preprocessing_utils/),
[training API](https://keras.io/api/models/model_training_apis/). El problema
medido es de nuestro dataset, no un defecto supuesto de Keras.

Corregir ademas el relato: el runner usa INPUTS de validacion DEV, no una
validacion interna separada dentro de train. Es desarrollo con seleccion sobre
validacion; no una estimacion independiente de generalizacion ni uso del test.

## F4. Alto para interpretar utilidad: soporte desigual y adecuacion no demostrada

Recargue los pesos R0 s1 en el grafo productivo. Alterar en +100 las primeras
**53/60 filas no cambia nada**; alterar las siete finales si cambia la salida
(maximo 0.4700 en escala del modelo). Replay de las primeras 64 predicciones
contra arrays guardados: diferencia maxima **8.94e-8 kW**, alcance acotado.

La red tiene siete muestras de alcance, separadas por **360 s** entre primera
y ultima (420 s seria otra convencion de soporte de intervalos), para predecir
a 3600 s. Ridge usa 60 muestras; seasonal-naive consulta t-1380, fuera del
contexto neuronal. Tener el mismo target historico como canal no iguala esos
soportes. Los controles siguen siendo utiles, pero no prueban inferioridad de
las redes a informacion igual ni adecuacion del receptor para esa tarea.

No se exige una ventana diaria por axioma. Demostrar soporte util en train,
comparar controles con igual soporte y separar al estacional de mayor historia.
Conservar detector local; decidir el nucleo temporal por la tarea y medir su
alcance antes de otra comparacion de R0/R1/R2. No cambiar arquitectura por buscar
un ganador despues de mirar este piloto.

Cuatro ajustes principales paran por presupuesto, ademas del piloto de costo;
el mejor checkpoint no estar al final NO demuestra ausencia de truncamiento o
convergencia. Retirar esa frase del retorno. Diferencias menores que SD de semillas
tampoco son prueba de equivalencia; mostrar diferencias pareadas e incertidumbre
al alcance de tres inicializaciones sobre una sola tarea seleccionada por DEV.

## F5. Alto: controlador aislado, no el ciclo de ejecucion solicitado

[test_e3_weekly_controller.py:48](https://github.com/harveybc/predictor/blob/60daac9/tests/test_e3_weekly_controller.py#L48)
implementa su propio simulador `_run`: no usa gym-fx/broker, comisiones ni ordenes
pendientes. `expected_fill_time` no gobierna el fill, que el test fija en i+1.
El valor que llama equity mientras hay posicion es cash + PnL, no cash + valor
de la posicion; la asercion final tampoco prueba continuidad por evento.

El controlador recibe una propuesta externa sin identidad del modelo que la
produjo: puede etiquetarla con el ultimo modelo liberado sin haberlo ejecutado.
En el callable real, latency_bars=0 se acepta y precio NaN emite LONG. El parametro
fallback se guarda pero no decide la conducta. Los tests existentes pasan porque
construyen el escenario permitido, no porque el runtime lo imponga.

Reusar entorno/broker reales, ligar inferencia al modelo seleccionado y probar
el adaptador con costos, pendientes, estado continuo y tiempos efectivos. Es
software de simulacion, no autorizacion para live ni un experimento RL completo.

## F6. Medio: faltan dominios ejecutables y alcance honesto de mascaras

TaskContract acepta h=0 con historia del target: el target es literalmente el
ultimo valor de entrada (125 ventanas en el fixture); h=-1 termina IndexError,
no rechazo de contrato. El piloto ejecutado h=60 no tiene este defecto; falta
proteger el siguiente diseno de forecasting. Reconstruccion h=0, si se ofrece,
debe ser otra tarea, no un forecaster validado por este contrato.

`mask_ffill` promete canal de mascara, pero build_tensors solo devuelve valores
rellenos y mask_y; no agrega mascara de entrada ni la entrega al modelo. El
piloto usa withdraw y no se invalida por ese modo no usado. Implementarlo de
verdad o rechazarlo hasta tenerlo probado, sin afirmar cobertura general.

## F7. Medio: la equivalencia del donante aun omite entrenamiento

La union ahora rechaza los cinco casos anteriores, pero
[`_record_equivalent`](https://github.com/harveybc/predictor/blob/60daac9/tools/df_mod_e0_arch_verify.py#L409)
compara una lista que omite updates, pesos/codigo y varios campos de tarea.
Cambiar updates del donante heredado **1100 -> 0** preservando su MASE produce
union aceptada con contradictions vacio. Validar todos los hechos que la propia
equivalencia promete; no solicitar nuevos fits para reparar esta verificacion.

## Resultados que se conservan y alcance de revision

Recalcule MAE/MASE de **10 ajustes (9 principales + costo)**: finitos, diferencias
exactamente cero respecto de arrays, mismos origenes/labels que DATA preparado.
Las medias 0.8987/0.8956/0.9002 son descriptivamente correctas. No recree DATA
desde el panel crudo en esta auditoria ni replaye todos los pesos; no confundir
este alcance con validacion completa de fuente a resultado. 131 archivos
originales preservados por digest.

El piloto sugiere una diferencia de costo, no una equivalencia demostrada:
R1 ajuste 227.405 CPU s, mas AE 131.739 = 359.144, frente a R0 757.633.
Es costo observado de esta tarea y estos techos, no ganancia general de negocio.
R1/R2 no muestran una ventaja de error clara y consistente entre semillas;
este estudio no confirma ni refuta H1, ni valida toda ARCH-A.

Tests existentes ejecutados **44 passed**: efectos E0, loader, espectro,
controlador semanal. No reejecute la suite completa ni entrenamientos. Los
verdes coexisten con las reproducciones nuevas. Mi primera sonda h=-1 termino
con el IndexError antes de completar el reporte: ahora lo captura y distingue
del caso h=0; la corrida completa se repitio. No se modificaron originales.

[Script](../evidence/RP32_REVIEW_2026_09_19/reproduce.py),
[resultados](../evidence/RP32_REVIEW_2026_09_19/results.json),
[lectura operativa](../evidence/RP32_REVIEW_2026_09_19/operational_read.json).
Ejecutar con Python trading-stack, CUDA_VISIBLE_DEVICES vacio, OMP/BLAS/TF
threads=1: script --repo <checkout-60daac9> --run-root <e1-household-pilot-v1>
--out <reporte>. El script solo escribe fixtures temporales y su reporte.

[Orden RP33-RP40](../../handoffs/MUSASHI_PROGRAM_RP33_RP40_2026_09_19.md).

Plan/estado/README actualizados. Dos regresiones documentales nuevas fallaron
antes del arreglo (prerrequisito de gobernanza ausente o sustituido por registro
posterior); despues **21/21** pasan y check_plan indica cobertura documental
PASS, scientific_approval false. No reemplazan los tests del flujo desplegado.
