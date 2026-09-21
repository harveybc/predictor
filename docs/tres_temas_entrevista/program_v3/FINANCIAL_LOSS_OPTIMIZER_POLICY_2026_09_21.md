# FIN-LOSS-OPT: precision marginal en forecasting financiero

Estado: REQUERIMIENTO APROBADO; diseno de aceptacion pendiente de Satoshi.
Fecha: 2026-09-21. Responsable de ejecucion: Satoshi; revision: Musashi.

## 1. Decision del owner y alcance

Comparar MAE/Huber y Adam/AdamW es OBLIGATORIO en los experimentos de forecasting
financiero. El piloto electrico no elige ni excluye una receta para trading.
La propuesta anterior de MAE+AdamW como referencia general queda supersedida:
no hay ganador financiero. MAE+Adam permanece solo como control de continuidad
del diagnostico electrico, sin cambiar corridas ya selladas.

El owner reporta una respuesta muy pronunciada del beneficio y Sharpe cerca
del error del ingenuo, con predicciones corta/larga y un ensayo de ruido sobre
predicciones ideales. Ese antecedente motiva estudiar mejoras de MAE de 1e-5
y 1e-6; no demuestra una ley exponencial universal ni una ganancia garantizada
para cualquier estructura de error. No buscar mas el OLAP/ramas historicos ni
exigir reconstruirlo. No repetir heuristic-strategy ni el barrido de ruido como
prerrequisito de esta tarea. E3/RL mantiene su evaluacion propia del sistema;
la eleccion de loss del actor o critico RL no se deduce de este forecasting.

Ubicacion: disenar en paralelo a la preparacion E1, ejecutar al disponer de
datos financieros gobernados y causalmente admisibles, ANTES de fijar la receta
de las comparaciones financieras de representacion/preentrenamiento. No esperar
a un supuesto ganador electrico ni sustituir esta tarea por otro piloto publico.
Esta orden prepara diseno y tests; no lanza entrenamiento cientifico sin el
protocolo, poblacion y presupuesto previamente fijados y revisados.

## 2. Pregunta y comparadores

Pregunta: con informacion identica y entrenamiento adecuado, que loss/optimizer
reduce el error financiero fuera de muestra, incluso marginalmente, y con que
estabilidad, costo y alcance por arquitectura, horizonte y periodo?

- Factorial inicial completo: MAE+Adam, MAE+AdamW, Huber+Adam, Huber+AdamW.
  Una diferencia entre las dos recetas diagonales no identifica sus dos causas.
- Misma poblacion de filas/targets, inicializacion pareada por semilla, calendario
  de validacion, monitor comun y politica de restauracion. Emparejar los cuatro
  brazos dentro del bloque host/semilla; no confundir optimizer con hardware.
- Declarar los horizontes corto/largo desde la tarea financiera. Las referencias
  del owner a 6 horas y aproximadamente 3 dias son antecedentes, no un contrato
  exacto recuperado. Justificar los valores finales antes de puntuar.
- Reentrenamiento semanal, ventanas de historia justificadas por cobertura de
  regimenes y curvas de aprendizaje. Cuatro anos es una configuracion candidata
  motivada por el antecedente, no un minimo universal ni permiso para leer futuro.
- Evaluar receptores relevantes: modular compacto y modelo de mayor capacidad
  motivado por las configs del negocio, con campos receptivos y capacidad medidos.
  No exigir artificialmente un millon de parametros ni extrapolar desde 8127.
- Primero comparar recetas con informacion/arquitectura fijas. Solo despues,
  congelar la receta por familia de tareas para los contrastes R0/R1/R2 y de
  representaciones, aplicandola igual al metodo y sus controles.

## 3. Escala comun y reporte obligatorio

MAE_z = mean(abs(yhat-y)) / sigma_train, con media/desviacion del target original
ajustadas en train por fold y compartidas por TODOS los brazos. Declarar poblacion
de ajuste sin ponderar accidentalmente por repeticion de ventanas. Si sigma es
cero o degenerada, tipar el caso; no fabricar un denominador para hacerlo pasar.
Las transformaciones usadas para entrenar pueden variar en experimentos propios;
las predicciones se llevan al mismo target y al mismo z-score de EVALUACION.
No comparar directamente un MAE en log1p contra otro en z-score.

Por horizonte, fold, semilla y brazo conservar MAE_z, MSE_z/RMSE_z, los mismos
errores del ingenuo, y skill = 1 - MAE_z_model / MAE_z_naive. Skill positivo
significa MENOR error; no confundir aumento de MAE con mejora. Un ingenuo con
error cero requiere estado explicito, no una division silenciosa. Registrar
tambien diferencias pareadas absolutas y relativas, cuantiles de error y costos.
Reportar unidades originales secundariamente, sin mezclar escalas en rankings.

La normalizacion permite comparacion dimensional, no igualdad de dificultad
entre horizontes/dominios. No cambiar la metrica confirmatoria de la propuesta.
Para reproducir literatura: misma tarea, target, resolucion, horizonte, splits,
transformacion y metrica del protocolo publicado; MAE_z es adicional si difiere.
Una correccion causal de ese protocolo se identifica como experimento separado.

## 4. Ajuste fino sin usar la reserva

1. Conservar un brazo de defaults EXPLICITOS de la version instalada; no afirmar
   que reproduce los defaults historicos desconocidos del owner. Registrar
   delta, learning_rate, decay, betas, epsilon, clipping y politica de precision.
2. Disenar una busqueda acotada por familia de loss, con presupuesto comparable
   y declarado. Ajustar LR/decay para ambas perdidas, delta para Huber, usando
   solo train y validacion temporal interna. No dar mas busqueda oculta a Huber.
3. Justificar candidatos delta por una escala de residuos obtenida causalmente
   dentro de train (origenes rodantes), no por residuos del test. Registrar
   conversion a escala original y fraccion de residuos por debajo de delta.
   Un unico delta=1 no descarta Huber. Cambiar la escala del target cambia el
   umbral fisico: controlarlo al variar volumen, transformacion o horizonte.
4. AdamW: declarar grupos de pesos regularizados y exclusion de bias/norm si
   aplica; registrar LR schedule y numero observado de updates. El producto
   acumulado de factores (1 - lr_t * decay) describe la contraccion debida SOLO
   al decay, no los pesos finales ni el decay optimo. Puede orientar candidatos
   de regularizacion junto con anchura, datos y duracion, no decidir el ganador.
   Reglas de escalamiento bajo muP no se trasladan a redes sin esa parametrizacion.
5. Adecuacion: curvas train/val, paciencia y min_delta en unidades declaradas,
   checkpoint mejor/restaurado, motivo de parada y actualizaciones reales.
   No elegir una tolerancia de stopping que haga invisible la mejora buscada;
   no exigir que cada epoch mejore 1e-6. Un techo alcanzado no prueba convergencia.
6. Validacion externa walk-forward y reserva sin consultas de seleccion; busqueda
   anidada y costo total incluidos. No reutilizar como intacto un test historico
   que ya influyo en decisiones. No descartar colas/extremos reales como ruido
   para mejorar el score. Preprocesamiento y disponibilidad se prueban antes.

## 5. Precision y evidencia de diferencias pequenas

No se impone un umbral de irrelevancia economica que descarte 1e-5 o 1e-6.
Tampoco se confunde un decimal positivo con una mejora estable. Los valores son
resoluciones de interes expresadas en la escala declarada, no tolerancias
universales de aceptacion ni promesas de potencia estadistica.

Guardar predicciones y targets sin redondeo de presentacion; acumular metricas
en float64 y contrastar un calculador independiente. Esto no recupera precision
perdida en inferencia float32: registrar dtype real, recarga, ejecuciones repetidas
y variacion por backend/host. Medir el suelo numerico antes de interpretar 1e-6.
El cubo conserva deltas y valores completos; mostrar suficientes decimales.

Analisis pareado por semana/origen y semilla, con intervalos que respeten bloques
temporales; no tratar ventanas solapadas como replicas independientes. Declarar
busqueda/multiplicidad. Una diferencia pequena incierta queda MEDIDA con su
intervalo, no eliminada ni promovida como ganancia. Estimar precision/costo desde
desarrollo antes de fijar replicas; no aumentar muestras hasta obtener un pase.

## 6. Aceptacion top-down y cierre bottom-up

Disenar estas reglas antes de implementar; su presencia aqui NO significa que
ya existan o pasen. Ejecutarlas por los entry points y lectores reales.

| ID | Requisito | Prueba exigida / evidencia |
|---|---|---|
| FL01 | factorial real | cuatro recetas y configs resueltas; cambiar loss no cambia optimizer/monitor inadvertidamente; verificar decaimiento real |
| FL02 | comparacion justa | identidades de filas/targets, scalers e inicializaciones pareadas; faltante/duplicado/cambio de horizonte rechaza |
| FL03 | no fuga | prefijos/futuro perturbado en preproceso y features; cambiar test no cambia scaler, delta ni seleccion; control con fuga falla |
| FL04 | precision marginal | fixture con delta analitico 1e-5/1e-6 sobre MAE_z no se redondea ni colapsa en archivo, terminal o warehouse; signo verificado |
| FL05 | entrenamiento observado | monitor, min_delta, updates, parada, pesos restaurados y errores desde arrays; recarga y variacion numerica medidas |
| FL06 | hiperparametros | LR/decay/delta, escala residual, version y grupos de pesos ligados al registro; candidatos dependen solo de train/DEV |
| FL07 | inferencia estadistica | bloques/semillas/periodos exactos; no pseudo-replicacion; cambios negativos y positivos incluidos, sin seleccion del mejor test |
| FL08 | cierre completo | gobierno antes de consumir, todos los desenlaces, predicciones/deltas -> padre -> contabilidad -> warehouse por contenido |

Entregable de Satoshi ahora: protocolo ejecutable, matriz requisito/test, reglas
rojas/verdes de cambios necesarios, poblacion, costeo y asignacion de hosts.
Entregable al ejecutar FIN-LOSS-OPT: factorial y tuning gobernados, tabla por
horizonte/modelo/periodo, curvas, incertidumbre, recursos, receta seleccionada o
limites explicitos. No marcar la tarea hecha solo por entregar su diseno.

## 7. Fundamento y limites

- [Huber, definicion ejecutable](https://keras.io/api/losses/regression_losses/):
  cuadratica cerca de cero y lineal fuera del umbral; suavidad no garantiza
  menor MAE fuera de muestra.
- [Adaptive Huber Regression](https://arxiv.org/abs/1706.06991) y
  [extension a datos dependientes](https://arxiv.org/abs/1904.09027): justifican
  estudiar escala, tamano, dimension y dependencia; sus garantias de regresion
  bajo supuestos no son una formula de delta optimo para nuestra red neuronal.
- [AdamW](https://arxiv.org/abs/1711.05101) y
  [escalamiento de decay](https://arxiv.org/abs/2405.13698): motivan separar y
  ajustar regularizacion; no prueban que una red mayor obligatoriamente gane.
- [Resultados electricos preservados](../../audits/work_plan/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md):
  12 fits, una tarea, siete en techo; no constituyen seleccion financiera.

## 8. Verificacion de esta incorporacion

Tres pruebas nuevas del checker documental fallaron antes del cambio y pasan
despues: eliminar FIN-LOSS-OPT, perder su documento o quitar BUSINESS-CONTRACT
como dependencia ya se detecta. Suite documental completa: 24/24; check_plan.py:
PASS con scientific_approval=false. No se ejecutaron FL01-FL08, entrenamientos,
simulaciones financieras, cambios en servicios ni escrituras al cubo en esta
incorporacion. La suite ML completa no se repitio: el cambio es documental y
del validador del plan, no del runner/modelo.
