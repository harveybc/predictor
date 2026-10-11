# I6-N: contexto, resolucion y capacidad conjuntos

Fecha: 2026-10-11. Estado: ANTECEDENTE_AUDITADO_NO_ORDEN_DE_EJECUCION.
La ejecucion autonoma esta autorizada en I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md.
Ese subplan sustituye las etapas N1/N2/N3 separadas, sus presupuestos y las
autorizaciones pendientes de este antecedente. No se debe ejecutar la antigua
escalera como seleccion irreversible antes de buscar ventana y capacidad
conjuntamente. Este documento conserva el diseno sometido a auditoria, no una
segunda cola. El lugar en el master sigue siendo I6-N, antes de R1/R2, H-CORE
y TEST. La cola sellada I6-E-M no se modifica ni se reanuda.

## 1. Campana conservada y pausa

I6-E-M compara resoluciones con contexto 2*h y nucleo fijo. Sirve como control,
no demuestra ventana/capacidad optimas. El despachador fue detenido y
deshabilitado por solicitud de replanteamiento; las asignaciones terminaron
y se recogieron sin nuevos claims. A las 05:25 America/Bogota: NEURAL 138
COMPLETED; ARIMA 82 COMPLETED y 46 FAILED; 3,762 QUEUED; cero RUNNING.
Las 266 disposiciones tienen publicacion y lectura exacta OLAP verificada.
Follower y observador siguen disponibles. No borrar ni relabelar terminales.

## 2. Contrato invariable

- EURUSD primero, 20 features RAW retenidas; no llamarlas optimas para cada h.
- Corto 2,3,4,5,6,12 h; largos propuestos 18,24,36,48,60,72 h: seis por familia.
  El propietario autorizo ajustar largos al calendario operativo. La version
  anterior 24..144 h se conserva en historia; 120/144 no caben antes de la
  liquidacion semanal con offsets 12 h. No modificar grillas historicas.
- Rige BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md: refit durante cierre,
  plano cierre-12 h, entrada apertura+12 h, corte de entradas jueves propuesto.
  Targets TRAIN/score dentro de la misma semana operativa, mascara por h;
  inputs conservan historia pasada disponible durante horas sin operar.
- Estrategia de negocio: sin posicion, extremos de las seis predicciones
  largas determinan candidatos compra/venta y TP/SL; aplicar umbrales
  configurados de beneficio/riesgo y elegir la direccion de mejor resultado.
  Con posicion, ordenar ambas familias por horizonte y cerrar temprano si
  anticipan SL antes de TP. Una posicion concurrente por activo. Verificar
  el plugin efectivo: min/max sin orden temporal no certifica esta regla.
- Emision horaria; objetivo t+h exacto; ventana incluye t y se ancla en el.
  n muestras abarcan (n-1)*delta horas. Sin suavizado/interpolacion.
- Muestreo de entrada y objetivos de salida independientes. Una salida
  escalar t+h no cambia por llamarla h/delta ticks; eso no genera otro fit.
- TRAIN movil de cuatro anos calendario antes del cutoff de cierre; ajustes
  semanales durante cierre; inner cronologica
  purgada por h; escaladores TRAIN-only. Target nunca limpiado.
- VALIDATION 2024 completa, 53 segmentos. TEST 2025 reservado.
- MAE y naive para el horizonte de X horas: precio USD/EUR, mismos origenes.
  Skill=(MAE_h-naive_h)/naive_h; negativo mejora. Naives intermedios separados.
- Seed 6102026 inicialmente, sin repetir unidades identicas. Nunca diez seeds.

## 3. N0: contratos, pruebas y coste

Primero resolver calendario y soporte operativo por h, con WMC01-WMC10.
Excluir objetivos fuera de actividad tanto al entrenar como al puntuar;
no descartar datos pasados observables de las ventanas. Resolver contratos
de disponibilidad parcial de largos para la estrategia antes de entrenar.
Usar E(t,h,rol) conocida en t, con t+h<B en la misma semana, y M historica
que agrega existencia exacta de label, disponibilidad y split. Una sola
funcion versionada gobierna dataset, adaptador y scoring. No consultar
existencia futura de barras para decidir en vivo ni recortar el horizonte.
Probar R=73/72/48 horas restantes para h=72, preservacion de cortos validos,
labels NaN excluidas antes de residuos y denominadores por salida.
Publicar por h dos poblaciones identificadas: oportunidades evaluables S_h
y predicciones efectivamente consultadas U_h, registradas antes del desenlace.
Cada MAE tiene su naive sobre exactamente esas filas, unidades y targets;
no reutilizar naives sin mascara ni elegir filas por acierto o trade ganador.
Registrar tambien cobertura y fallos sobre la poblacion elegible completa.
Entregables N0: tabla de soporte por h/semana/rol, contrato de mascara y
consumo, pruebas WMC01-WMC10, digests y esquema OLAP para ambas poblaciones.
No iniciar N1 con esas pruebas pendientes ni atribuirlas a fits anteriores.
Sellar soporte comun por h para la mayor historia candidata y recalcular sus
conteos/naives. No copiar cifras anteriores cuando cambie la poblacion.
Probar inclusion de t, t+h, muestreos independientes, perturbacion futura,
purga, escalas, restauracion del mejor peso, restart, idempotencia y OLAP.
Medir TRAIN por clase y reservar 1.25*pico, nunca recortar el margen.

## 4. N1: escalera inicial de capacidad

Representantes: h=4 y 72. Anclaje inicial L=2*h, delta_x=h/2, n=5,
salida directa t+h. Este anclaje no se declara optimo.
Ramas constantes: Conv1D causal filters=16, kernel=3, stride=1; fusion por
canales y posicion fisica inmediatamente despues de fusion.

No exigir Transformer. Nueve semillas de configuracion, no nueve repeticiones:

| Familia | Configuraciones iniciales | Reduccion de caracteristicas |
| --- | --- | --- |
| Conv1D solamente | filters iniciales 8,16,32; dos Conv1D causales en total | segunda Conv1D filters=max(4,filters_iniciales/2) |
| Transformer + Conv1D | (d_model,bloques)=(8,1),(16,1),(32,1),(16,2),(32,2) | dos Conv1D, filters=max(4,d_model/2) y max(4,d_model/4) |
| Control vigente | arquitectura sellada exacta | la del control, no una reinterpretacion |

En Transformer, Conv1D 1x1 proyecta a d_model; dos cabezas,
key_dim=d_model/2; FFN 2*d_model->d_model por posicion y Add/LayerNorm en
ambos residuales. Conv1D kernel=3, stride=1 inicialmente. Todas las rutas
conservan el eje temporal dentro del nucleo. Cabezal ultimo estado->Dense(1).
La reduccion de filters NO reduce por si misma el numero de ticks.

El nucleo tiene tres componentes configurables con responsabilidad separada:
entrada (Conv1D, Transformer o combinacion), reductor principalmente Conv1D,
y adaptador final opcional antes de los cabezales. Sus contratos describen
formas, soporte temporal y pesos importables; sin Flatten dentro del nucleo.
La posicion fisica permanece inmediatamente despues de fusion, antes de
cualquier Transformer. LSTM final usara return_sequences=True: el cabezal
decide como leer la secuencia, no la LSTM destruyendo el eje en secreto.

Cuatro semanas internas de 2023, predefinidas por trimestre, cuatro anos
anteriores por ajuste: 9*2*4=72 fits maximos. Sellar calendario antes de correr.
Es screening, no resultado anual. Conservar hasta tres alternativas de error
y coste, incluida una pequena. Son puntos iniciales de la busqueda DEAP/DOIN,
no una seleccion irreversible de familia o ventana. No excluir para siempre
capacidades grandes: pueden rescatarse cuando cambie la ventana/resolucion.

## 5. N2: interacciones DEAP/DOIN con presupuesto

DEAP propone, DOIN distribuye; no NEAT ni producto cartesiano exhaustivo.
Hasta 24 configuraciones NUEVAS por representante x cuatro folds: 192 fits.

- Historia L={1,2,4,6}*h; entrada horaria o delta_x=h/k, k={1,2,3,4,8,9}.
  Solo horas enteras; n=1+L/delta_x. No periodicidades fabricadas.
  Ejemplo h=72, L=432, delta_x=8: 55 muestras frente a 433 horarias,
  ambas con t y 432 horas de historia. Ese contraste debe ser admisible.
- Entrada del nucleo: Conv1D, Transformer o Conv1D+Transformer. Bloques
  Transformer={0,1,2}; cero es valido para familia Conv1D. d_model={8,16,32,64}
  solo en rutas con Transformer; filters de entrada={8,16,32,64} en Conv1D.
- Reductor Conv1D: 1,2,3 etapas; filters_i configurables individualmente en
  {4,8,16,32,64}, no crecientes dentro del reductor. La mitad progresiva es
  una inicializacion barata, no una regla obligatoria ni garantia informativa.
- Adaptador final antes del cabezal: ninguno, Conv1D causal (filters=4,8,16,32)
  o LSTM (units=8,16,32, return_sequences=True). El LSTM no entra en N1;
  empieza en N2 tras medir su coste. No exigir una capa costosa a toda ruta.
- Reduccion temporal: ninguna, progresiva o solo al final. Toda ruta declara
  soporte y cubre t, nunca reproduce el rezago origin-3 del encoder historico.
  Reduccion progresiva factor 2 solo si quedan al menos dos posiciones.

Genes de filters y de stride temporal son independientes; cambios de familia,
profundidad, ancho, historia y resolucion se proponen conjuntamente. Mutacion
y cruce deben respetar el esquema jerarquico y excluir combinaciones inactivas
de la identidad, para no evaluar dos genotipos del mismo modelo efectivo.
No hay una capa que garantice reduccion sin perdida: comprobar beneficio por
pronostico pareado y coste, y reconstruccion solo como diagnostico separado.

Ranking por error agregado de los folds y coste medido, no por la mejor semana.
La parada tras configuraciones mayores sin mejora describe una meseta explorada,
no demuestra optimo global. Se sella la tolerancia y regla de parada antes del
fit; early stopping usa promedio TRAIN/inner, patience=5, best-weight restore.
Batch=256, max_epochs=50, AdamW/Huber segun contrato heredado inicialmente.

## 6. N3: representacion de entrada y objetivos de salida

Hasta cuatro configuraciones NUEVAS por representante x cuatro folds: 32 fits.
Contrastes emparejados sobre hasta dos alternativas; no cambiar simultaneamente
transformacion y supervision sin un control que permita separar los efectos.

Entradas: RAW escalado frente a cambios/valores relativos al origen, con reglas
tipadas por feature. Precio actual conservado como referencia al reconstruir.
No quitarlo para obligar artificialmente a alejarse del naive.
Salidas: escalar t+h frente a trayectoria con delta_y=h/2 cuando sea entero,
independiente de delta_x. Misma metrica final t+h; purga por endpoint mas lejano.
No generar inicialmente toda la trayectoria horaria larga. ARIMA sobre una
serie muestreada pronostica h/delta pasos: no fingir independencia inexistente.
Guardar dispersion, correlacion, direccion, salida constante y error por semana
como diagnosticos, sin cambiar despues la metrica primaria.

## 7. N4: transferencia y ano completo

Transferir hasta dos recetas TRAIN a los 12 h, con ventanas/formas explicitas;
no llamarlas optimas en horizontes no estudiados. Maximo 2*12*53=1,272
disposiciones neuronales en VALIDATION. ARIMA: una resolucion elegida en TRAIN
por h, hasta 12*53=636 disposiciones CPU. Fallos tipados no reciben MAE anual.
MAE anual por suma de errores/conteos de todo el ano, no media sin pesos de
semanas. VALIDATION escoge; TEST confirma el procedimiento congelado.

N4-M: antes de TEST, contraste de produccion direct multi-output separado:
un modelo corto (2,3,4,5,6,12 h) y otro largo (18,24,36,48,60,72 h), cada uno
con ramas/fusion/nucleo compartidos dentro de su familia y cabezales por h.
No imponer una unica ventana optima a ambas familias. El contexto y el nucleo
compartidos de cada familia son otra configuracion a medir, no la union
automatica de los modelos independientes. Conservar metricas y naive por h,
normalizar perdidas por escalas TRAIN por h y declarar pesos de las tareas.
Soporte elegible y conteos por cabezal explicitos, sin borrar origenes cortos
porque falte una etiqueta larga. Presupuesto separado: hasta dos recetas por
familia x dos familias x 53 semanas = 212 disposiciones; medir coste antes.

## 8. Continuacion del master

Despues, R0/R1/R2 con donantes compatibles con ventana, resolucion, escalado
y TRAIN; preentrenamiento del nucleo como contraste separado. No reutilizar
ciegamente encoders de 24 muestras. Costear esa etapa por separado.
Confirmacion TEST anual con refits semanales antes de afirmar habilidad fuera
de desarrollo. Estrategia existente solo tras su naive; RL conserva su lugar.
NEAT permanece al final, no reemplaza a DEAP/DOIN.

## 9. Automatizacion y limites

Un comando init/run/status/stop-drain, niveles etapa/h/config/fold, cola durable,
claims exclusivos y registro de reuse por identidad completa. Aprobar el diseno
habilita continuacion automatica: no nuevas visitas entre etapas ni agentes
leyendo logs. No se reutilizan cifras si cambian inputs, soporte, init o modelo.

5090 preferente, un hijo en su host; 4090 cuando cabe. No 5070 Ti concurrente
por RAM compartida. Desktop: coordinacion/OLAP/controles CPU pequenos.
Paralelizar candidatos independientes; no saltar dependencias cientificas.
Todo resultado/decision/seed/coste/digest al OLAP con readback; copia fisica
local. Git solo scripts, configs, reglas, resumen y manifiestos, no binarios.
STATUS generado muestra etapa, counts, latidos, siguiente unidad y ETA medida.

Presupuesto N1-N3: hasta 296 fits internos; N4 hasta 1,272 neuronales + 636
ARIMA; N4-M hasta 212 neuronales adicionales. No incluye R1/R2 o H-CORE.
Folds semanales son protocolo del negocio,
no repeticiones de seed. ETA completa despues de pilotos por clase; no
extrapolar el modelo pequeno a ventanas grandes ni prometer un optimo global.
