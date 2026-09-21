# RP57-RP64: diagnostico ML, referencias y reparacion de la comparacion

Musashi, 2026-09-20. Base `de88764`, despues del sucesor E1 ya ejecutado.
[Lectura obligatoria](../audits/work_plan/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md).
No falta decision del owner sobre mas epocas u otra tarea: esta es la secuencia.
No pasar a otra tarea buscando un resultado favorable, ni ampliar epocas a ciegas.

## Regla de ejecucion

Completar los ocho bloques aplicables sin confirmaciones intermedias. PRE y
criterios top-down antes de cambios, codigo bottom-up, integracion y aceptacion
en los llamadores reales. Preservar historial y propuesta doctoral. No nuevo
rediseno de gobernanza. La evidencia nueva se registra antes de medir, resultados
al warehouse por outbox y reconciliacion de contenido; importaciones historicas
se identifican como tales, nunca retroactivamente prospectivas.

14 400 CPU s agregados para esta ronda, incluyendo fallos/tests/cierre, bajo
memoria acotada. Medir todos los hosts mediante el wrapper existente, corrigiendo
la deuda de costos UNMEASURED de RP56. Reservar cierre antes del piloto de costo.
Sin GPU, live, reserva ni entrenamiento cientifico RL en este bloque.
Coordinador RP57/58/61/cierre; WORKER_A RP59/60; WORKER_B RP62 y pruebas.
Distribuir fits independientes despues de la aceptacion. No alterar trabajos
ajenos ni reiniciar servicios sanos para estos diagnosticos.

## RP57. Comparadores visibles, identicos y comprobados

Reproducir el reporte de Musashi desde arrays. Incluir por unidad y por tarea:
MAE/RMSE/bias originales, MAE z-score, MAE log1p cuando su dominio lo permite,
naive h, naive estacional declarado y control lineal, mejora de MAE vs cada
referencia en las MISMAS filas, costos, n, dispersion y diferencias pareadas.
No ocultar fallos ni mejorar promedios eliminando filas propias de un metodo.
Denominador cero = indefinido; ninguna epsilon que fabrique una victoria.

Renombrar en sucesor el actual cociente como error escalado por persistencia
horizon-train; conservar el campo historico con errata/version. MASE convencional
lleva periodo m, formula, origen, soporte train y tratamiento de huecos. No escoger
m=h salvo justificacion estacional independiente. Mantener H1 de la propuesta,
sin cambiar criterio a la escala en que gane nuestro metodo. Logs punto a punto,
NO log del MAE; no recortar predicciones invalidas sin declararlo.

Integrar la tabla en entry point, reporte legible y OLAP configurado. Esta
revision por si sola no certifica la contabilidad de RP56. Reusar cierre existente,
comparar al reporte local y conservar la diferencia exacta si no coincide.

## RP58. Recuperar las comparaciones financieras antiguas sin adivinar

Inventariar configs efectivas, commits productores, CSV/arrays, transformaciones
de x/y y su orden, inverse, baseline, horizon en tiempo fisico, entrenamiento,
validacion y test. Primero el TCN NEAT localizado y luego las corridas con MAE
cerca de 0.02 y naive cerca de 0.018. No atribuirlas al log1p por su magnitud:
el champion localizado declara log1p FEATURES, no prueba ese target historico.

Congelar la tabla de resultados ya publicada antes de elegir filas. Reconstruir
metricas cuando hay predicciones/labels/ids; si faltan, PUBLISHED_SUMMARY_ONLY
con el objeto faltante. No reabrir test reservado para ajustar modelos, no
ejecutar sweeps antiguos ni sobrescribir muestras. No generalizar una config
adyacente al job que produjo un CSV. Auditar tambien heuristicas de denormalize
en stl_norm: estado de transformacion debe provenir del contrato, no de adivinar
la distribucion de cada prediccion. Corregir solo rutas activas con PRE demostrable.

## RP59. Datos, target y preprocesamiento, antes de arquitecturas

Para household, recorrer bytes -> roles -> tiempos -> split -> scaler -> ventana
-> label -> inverse -> loss/metricas. Probar identidad de y(t+h), h minutos
fisicos, periodicidad/missing/DST, autocorrelacion train, unidades, ceros y picos,
colas/distribucion por split, grano del scaler (filas vs ventanas repetidas).
Perturbacion futura no cambia train ni features pasadas; no cerrar huecos.

Graficar periodos seleccionados por regla previa, no solo los mejores:
verdad/naive/lineal/modelo en kW y log, error por hora/dia, nivel y magnitud
del cambio, tasas de observaciones/variables faltantes. Cuantificar que parte del
error viene de transiciones abruptas frente a consumo estable. No afirmar un
suelo irreducible empirico por ver un modelo fallar.

Poner calendario disponible y contexto diario/semanal como hipotesis de entradas,
no cambiar aun el target a una media mas facil. Cualquier agregado es otra tarea
con referencia y costos propios. No smoothing centrado ni wavelet sobre futuro.

## RP60. Optimizar el objetivo declarado, sin confundir presupuesto con convergencia

Recomputar curvas y checkpoint realmente restaurado. Retirar "cota inferior"
para errores de fits al techo: es presupuesto limitado, sin garantia sobre error
futuro. Ni todos necesitan mas epocas ni early stopping demuestra ausencia de
sobreajuste. Probar que updates son optimizer.iterations efectivos.

Diseñar contraste acotado loss/monitor antes de entrenar: MSE z-score actual
frente a MAE alineado a la medida principal; log1p como objetivo distinto y
explicitamente rotulado, con inverse y dominio. Primero sanity de sobreajuste
de un subconjunto train fijo (capacidad de ajustar, no generalizacion), cero-head
igual al naive por la ruta real y etiquetas independientes como negativo.
Si falla, reparar antes de barrer hiperparametros.

Examinar LR, gradientes, activaciones, regularizacion, train/val gap, overfit de
AE, y objetivo de reconstruccion frente a h60. Formular causa -> prueba ->
resultado que la refutaria. No decir "faltan datos" sin curva por volumen ni
"mal optimizador" sin aislarlo. Usar el mismo conjunto y selector para comparadores.

## RP61. Configuraciones de literatura, trazadas hasta codigo

Matriz articulo/implementacion/commit -> tarea/datos/split/unidades -> grafo,
activaciones, canales, normalizacion/dropout, LR, loss, batch, stopping y metricas.
No comparar cifras de clima, clasificacion o agregacion diaria con consumo
minutal h60. No hay configuracion universal transferible por nombre de modelo.

Referencias iniciales obligatorias: TensorFlow forecasting (lineal/CNN/LSTM
residual) y Bai/locuslab TCN. Nuestro bloque una conv ELU no es su bloque dos
convs normalizadas con dropout. Declarar adaptaciones, comprobar alcance real
y no eliminar componentes de la referencia para que quepa en presupuesto.
TS2Vec sirve para distinguir pretraining contrastivo de AE, no para prometer
que el nuestro gana. Seleccionar UN comparador neuronal independiente ademas
del lineal para el diagnostico inicial; TCN oficial es primera referencia por
la discrepancia encontrada. LSTM/CNN quedan enumeradas para contraste posterior,
no lanzar todas las familias indiscriminadamente.

## RP62. Diseno de diagnostico secuencial, no factorial descontrolado

En DEV, fijar contrastes que separen: (a) entrenamiento/loss, (b) informacion de
entradas y contexto, (c) volumen train, (d) arquitectura, (e) pretraining.
No mover todos a la vez y luego atribuir el efecto al extractor.

Primera fase: R0 actual vs referencia TCN independiente con datos/filas/objetivo
comunes, y contraste de loss/monitor; usar tres semillas pareadas si cabe. Si no,
cost pilot y fase menor sellada explicitamente como diagnostico, sin eliminar
celdas tras ver score. Solo ampliar volumen dentro de desarrollo permitido:
28 dias no se convierten en anos contando ventanas; respetar reservas.
Reentrenamiento semanal determina evaluacion posterior, no se infiere de este
piloto electrico. Proyectar todos los hijos y reservar verificacion/cierre.

Cambios de calendario/contexto se admiten despues del chequeo temporal y
disponibilidad, con misma informacion para brazos comparados y ventana comun
de evaluacion. Predeclarar origenes adicionales DEV para comprobar estabilidad;
validacion ya inspeccionada no se convierte en confirmacion por cambiar semilla.

## RP63. Ejecutar la primera fase y luego R0/R1/R2 si procede

Pasados RP57-RP62, ejecutar la fase dentro del presupuesto, por los consumidores
reales y con gobierno. Mantener tambien la referencia original inmutable.
Rechazar corruptos/labels desplazados/inverse equivoco antes de interpretar.

Si se demuestra un defecto, repararlo con test rojo/verde y emitir sucesor;
invalidar solo mediciones afectadas. Si hay mejora de desarrollo por opcion
metodologica, replicarla con el par y costos declarados. Volver a R0/R1/R2
solo con receptor/entrenamiento adecuados comunes, mismos pesos iniciales y AE
compartido en R1/R2. No aprender una opcion diferente por brazo que destruya H1.
Si no cabe esa segunda fase, dejarla sellada con costo medido, no inventar
resultado; completar diagnostico y cierre sin otra pregunta entre bloques.

## RP64. Retorno util para decidir

Tabla naive/lineal/neuronal por unidades y escalas, curvas, graficos, diferencias
pareadas por bloques temporales y semillas, costo completo, grado de convergencia
y distribucion del error. No tratar 10 mil ventanas solapadas como replicas iid.
Una frase por causa: DEMOSTRADA / REFUTADA EN ESTE ALCANCE / PENDIENTE, con prueba.
Nada de concluir solo "negativo" sin inspeccionar las causas que ordena este bloque.

Conciliar cubo por contenido, publicar artefactos/adopcion, actualizar master y
estado; un retorno final con commits y comandos. Conservar RL/H-CORE y los otros
frentes. La disposicion de los cuatro prepare duplicados usa el procedimiento
aditivo existente sin borrado, si disponible; no bloquea ML. Ninguna autorizacion
nueva del owner requerida, ni trabajo vivo permitido por omision.
