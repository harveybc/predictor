# Plan maestro v3: de información a conocimiento y operación

Actualizado: 2026-10-19. Este documento contiene solo el plan vigente.
Los resultados afectados por datos defectuosos se eliminan, no se conservan
como evidencia historica ni gobiernan ningun despacho.

## 1. Autoridad vigente

Avance real2026-10-19: cerrado un subconjunto de15/30 variables derivadas del
nucleo EURUSD reparado y admitido, con37198 filas OLAP verificadas y dos copias.
Ya corre SAC directo frente a la representacion temporal con ESE MISMO
manifiesto; no otro RAW. Ver [seleccion y contraste](program_v3/SELECTED_CORE_RL_2026_10_19.md).
Este cierre acotado no declara admitido/seleccionado el catalogo externo ni
efectos causales identificados. Mantener ambos alcances separados.

Regla inmutable de cada entrada: SL y TP obligatorios, mas un presupuesto
maximo en moneda de cuenta igual a fraccion_elegida * rel_volume * balance(t).
rel_volume es configurable (inicial0.1); el apalancamiento se aplica aparte
para convertir margen a nominal. El minimo del broker nunca fuerza superar
el presupuesto: si no alcanza, no abrir. Mantener esta regla en ambos SAC y
controles; la comparacion actual usa apalancamiento1:1 y minimo/paso de ejemplo.

Precision del propietario 2026-10-19: la seleccion de caracteristicas es una
dependencia cientifica obligatoria del contraste RL solicitado, no decoracion.
El ensayo RAW cerrado (12/12, OLAP y dos copias verificados) es solo un control independiente; no completa ni
sustituye RL2 con seleccion. No abrir mas campanas RAW ni busquedas de agentes
antes de cerrar el inventario admitido y la seleccion recalculada. El siguiente
contraste comparte exactamente el manifiesto seleccionado: SAC directo frente
a SAC con extractores por variable, fusionador y nucleo temporal. El control
RAW se conserva unicamente para medir, por separado, el aporte de seleccionar.
Los controles de sobreajuste se contrastan individualmente, sin cambiar el
ensayo ya cerrado. Detalle y limitaciones en el subplan RL, seccion 8, y
[resultado del control](program_v3/RL_RAW_CONTROL_RESULT_2026_10_19.md).

Continuacion operativa 2026-10-19: EURUSD tiene admision CORE_PRICES_ONLY
completa (37309 horas, 72 meses, sin horas operativas requeridas ausentes ni
interpolacion). Con esa poblacion se prepara el control bruto SAC estatico y
sus controles mecanicos, independiente del cierre del inventario restante.
Su politica RAW_BASELINE no se presenta como seleccion final. No entrenar con
datos pendientes: la admision del nucleo y la aceptacion real de hora/fill/
sesiones preceden ese control. La comparacion con caracteristicas seleccionadas
espera su propia seleccion recalculada. Costes iniciales son supuestos
analiticos no nulos, etiquetados; no hechos verificados del broker.

Cambio de prioridad 2026-10-18: terminar datos y seleccion; despues RL como
primer consumidor, reutilizando componentes validos. Los experimentos de
predictores corto/largo, su estrategia y NEAT pasan al final. No se exige vencer
el naive de precios para ensayar una politica RL. El detalle vinculante es
[RL primero tras reparar datos](program_v3/RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md).
Preparacion de contratos y pruebas CPU sinteticas puede avanzar en paralelo;
ningun fit financiero se adelanta a la admision real.

Precision aprobada 2026-10-18: el primer contraste RL es ESTATICO, sin reajuste
durante evaluacion. SAC con datos directos frente a SAC con ramas Conv1D pequenas
y nucleo temporal, inicialmente 13 observaciones cada dos horas de mercado.
Maximo una apertura por dia de negocio; salidas y cierre forzoso no se bloquean.
Reajuste y otros algoritmos se contrastan despues. Implementacion en
agent-multi y gym-fx, sin modificar DOIN ni retomar predictores.

Plan operativo activo y trazabilidad de reparaciones:
[RL: ejecucion y entrega de datos](program_v3/RL_EXECUTION_REPAIR_PLAN_2026_10_18.md).
La aceptacion sintetica del simulador no sustituye la admision del lake ni la
seleccion recalculada. No se transfieren automaticamente resultados anteriores.

Directiva 2026-10-18: inventariar, recuperar y curar los datos antes de cualquier
nuevo entrenamiento o seleccion. Eliminar los resultados, pesos, configuraciones,
rankings, registros, metricas y copias derivados de los inputs afectados; no
archivarlos ni reutilizarlos como referencias. Conservar codigo generico y
trabajos independientes, no antiguas afirmaciones de performance.

TRAIN2020-2023, VALIDATION2024 y TEST2025: cobertura efectiva mes a mes, no solo
extremos de fecha. Recuperar cotizaciones observadas con fuentes pagadas y
gratuitas; contrastar valores y relojes. Nada de interpolacion, relleno de precios,
ticks sinteticos de mercado cerrado ni limpieza del target. Outliers: verificar
y distinguir corrupcion de extremos reales, no borrarlos automaticamente.

El lake canonico debe contener solamente recursos admitidos. Despues se
recalculan preprocesamiento y todas las fases de seleccion con nueva identidad;
los datos descartados/sustituidos se eliminan tras la recuperacion aplicable.
No declarar admision completa, seleccion terminada o campanas activas sin
verificacion real. Solo el control bruto con su poblacion ya admitida puede
avanzar en paralelo; la poblacion ampliada no hereda esa admision.

Procedimiento completo: financial-data/work_plan/17_CANONICAL_INVENTORY_AND_CURATION.md.
Politica: financial-data/configs/curation_policy.v1.json.
Controles: [estandar de calidad](../i6n/weekly_test/DATA_QUALITY_AND_SELECTION_STANDARD.md).

## 2. Objetivo

Construir y comprobar un sistema de aprendizaje para trading algorítmico que:

- use sólo datos disponibles en el instante de decisión;
- seleccione características por utilidad predictiva y de negocio, evidencia
  causal, calidad de representación, complementariedad y coste;
- preserve la dimensión temporal desde cada entrada hasta el cabezal;
- mantenga una ANN densa ramificada por característica como control explícito
  no temporal, aunque contradiga nuestra hipótesis arquitectónica;
- permita reutilizar extractores por rama y el nucleo sin exigir agotar sus
  barridos ni la prediccion supervisada antes del primer ensayo RL;
- compare pronóstico supervisado, estrategia heurística y políticas RL;
- optimice configuraciones con DEAP/DOIN sin confundir búsqueda con el modelo;
- opere primero en shadow, MT5 demo y Alpaca paper bajo el control de riesgo
  existente; capital real queda fuera de este plan.

El objetivo científico no es producir muchas corridas. Es identificar qué
propiedades de datos, transformaciones, arquitectura y régimen de entrenamiento
explican mejoras reproducibles frente a una referencia fuerte y frente al naive
en la misma población.

## 3. Reglas no negociables

1. **Tiempo y causalidad.** Todo dato, transformación, label, split y replay
   declara tiempo de evento, tiempo de disponibilidad y soporte. Scalers,
   imputacion y checkpoint selection usan fit/inner; VALIDATION puede elegir
   hiperparametros y features, declarando su uso como desarrollo. TEST no
   participa en ninguna seleccion. No se permite fuga por bordes, revisiones
   ni disponibilidad; seleccionar con 2024 no convierte folds 2023 en OOS.
2. **Naive pareado.** Cada MAE/MSE comunicado incluye el naive de las mismas
   filas, escala y horizonte. Una predicción que no supera estrictamente el
   naive no entra a la estrategia heurística ni genera artefactos de trading.
3. **Literatura fiel primero.** Cada dominio usa datasets, horizontes, métricas,
   modelo y receta comparables a una referencia publicada. Nuestras
   intervenciones se ejecutan después y se etiquetan como tales.
4. **Test externo una vez.** Desarrollo y selección usan TRAIN e inner/outer
   validation temporal. El test congelado sólo confirma el finalista sellado.
5. **Semillas austeras.** Una semilla por defecto; hasta tres sólo cuando la
   variabilidad o la comparación publicada lo exijan. Nunca más de tres.
6. **No repetir.** Reutilizar solo artefactos de datos admitidos, nunca los
   retirados por esta reparacion. Solo se repite por perdida,
   corrupción, comparación necesaria o cambio científico explícito.
7. **Predicciones efímeras.** Se derivan todas las métricas declaradas y se
   verifica el catálogo antes de borrarlas. Se conservan config, semilla,
   checkpoints elegibles, métricas, identidades y recibos.
8. **Evidencia distinta de producto.** Tests de software prueban mecanismos;
   métricas científicas prueban hipótesis; shadow/paper prueba integración.
9. **Paralelismo útil.** Un bloqueo local no detiene perfiles, referencias,
   preparación ni entrenamientos independientes y elegibles.
10. **Espacio de salida.** En omega, las salidas grandes nuevas usan
    [`EXPERIMENT_STORAGE_ROOT`](../EXPERIMENT_STORAGE_POLICY.md) después de
    comprobar montaje escribible y espacio disponible. Bases vivas, SQLite,
    bloqueos, secretos, repositorios y worktrees siguen en ext4. No mover
    recursos ajenos ni rutas de servicios activos sin verificar dependencias.
11. **El negocio gobierna la evaluacion financiera.** Calendario, disponibilidad,
   mascaras por salida, cierre forzoso y costes son obligatorios en todos los
   modos. La frecuencia de reajuste es explicita: NONE, QUARTERLY, MONTHLY o
   WEEKLY, con identidad propia. WEEKLY es el modo de negocio por defecto;
   controles estaticos solo en disenos posteriores explicitos. Se exigen
   cuatro anos TRAIN, ano completo VALIDATION y ano completo TEST. Reajustes
   usan cuatro anos moviles salvo contraste aprobado. Todos los modos
   reportan cada semana y el total anual; no se sustituyen por una semana.
   `LITERATURE_STATIC` sigue reservado a reproducciones fieles de papers.

## 4. Checklist de control de alto nivel

Este checklist es la vista que debe consultar el orquestador antes de cada
despacho. Un punto sólo cambia de estado mediante evidencia enlazada.

- [ ] I0. Contrato de negocio: targets corto/largo, barrera, RL, costes y riesgo.
- [ ] I0-W. Cadencia configurable: semanal por defecto, controles de cadencia con identidad propia; anos externos completos y modo literatura separado.
- [ ] I1. Inventario: fuente, columna, unidad, frecuencia, licencia y disponibilidad.
- [ ] I2. Perfil básico completo por celda de métrica, no una marca global.
- [ ] I3. Recalcular perfil individual y escalera causal por feature-target sobre datos admitidos.
- [ ] I4-C. Dependencia y redundancia entre características; alias y clusters.
- [ ] I4-R. Rankings por clustering, mRMR y JMI; trayectorias K reproducibles.
- [ ] I5. Comparación conjunta de conjuntos y manifiesto final.
- [ ] I5-P. Ablación RAW/reconstrucción/latente sobre los rasgos seleccionados;
  el target y el naive permanecen intactos.
- [ ] I9-R/0-4. PRIORIDAD tras I5: contrato de entorno y contabilidad ->
  baselines de negocio -> SAC compacto/modular -> DQN y contrastes justificados
  -> cadencia y DEMO/PAPER. Ver subplan RL; no esperar I6-DMO ni NEAT.
- [ ] I6-A. Agrupación, campos receptivos y controles ARCH emparejados.
- [ ] I6-D. ANN histórica por característica y contraste Dense de ramas,
  después de I5; resultado emparejado, sin atribuir tiempo a neuronas latentes.
- [ ] I6-B. Preentrenamiento de extractores de ramas seleccionadas.
- [ ] I6-DMO. Direct multi-output corto/largo: solo despues de I5 y sobre la
  nueva poblacion admitida. Registrar TRAIN y VALIDATION por horizonte,
  checkpoints, parametros y semillas. Corto: 2,3,4,5,6,12 h; largo:
  18,24,36,48,60,72 h, sujetos a soporte de negocio por salida.
  Ningun modelo anterior es donante.
- [ ] I6-E. Viabilidad de doble pronostico con calendario y mascaras de negocio.
  **Naive para el horizonte de X horas** sobre exactamente las mismas filas y
  escala que su MAE. Error relativo `(MAE_h-naive_h)/naive_h`, negativo mejora.
  Resolucion, ventana y arquitectura se prueban sin suavizar ni limpiar targets;
  los experimentos anteriores no constituyen evidencia reutilizable.
- [ ] I7. E1: R0 frente a R1 frente a R2, mismo diseno y poblacion admitida.
- [ ] I7-H. Fijar prefijo ganador y probar H-CORE por separado.
- [ ] I8. Optimizacion DEAP distribuida por DOIN: conservar su protocolo y
  codigo, no cadenas ni candidatos de datos retirados. Nueva identidad y
  configuraciones colaborativas verificadas solo tras datos y seleccion.
- [ ] I9-N. Representación final congelada: Dense control frente a NEAT.
- [ ] I10. LTS semanal, shadow, MT5 demo y Alpaca paper.
- [ ] I11. Extensión final opcional: calendario económico causal como entrada.
- [ ] I12. Posprograma: configurar `feature-selector` desde el chat de M5PHET.

Prioridad de la poblacion ampliada: inventario y recuperacion -> admision ->
preparacion verificada -> seleccion recalculada -> RL con seleccion. El control
bruto usa exclusivamente el nucleo admitido, sin esperar esa poblacion.
Predictores y NEAT quedan diferidos
al final por directiva del usuario, aunque mantengan sus identificadores I6/I9.
No hay permiso para adelantar fits,
extractores o trading con inputs pendientes. El paralelismo se usa dentro de
la recuperacion y verificaciones independientes, con memoria y disco admitidos.

## 5. Datos y selección progresiva

### Decision de negocio sobre los predictores

I6-E queda DEFERRED_AFTER_RL. Cuando se retome debe producir una decision explicita sobre continuar la estrategia de
pronosticos cortos/largos. Informar habilidad por horizonte y por funcion:
cierre temprano frente a apertura/TP/SL. La compuerta es mejora estricta del
MAE anual sobre el naive en las mismas filas, no mejora obligatoria cada semana.
No exigir que todos los horizontes ganen: conservar los elegibles y comprobar
si bastan para ambas funciones mediante la interfaz real de heuristic-strategy.
Un subconjunto distinto exige su propia configuracion e identidad, no sustituir
silenciosamente horizontes que el plugin requiere. No ensayar configuraciones
inelegibles en trading. Ganar al naive no demuestra rentabilidad: despues medir
costes, riesgo, Sharpe y utilidad pareada con el plugin existente.

Si el ensayo finito no aporta senales elegibles para ambas funciones, registrar
que la formulacion probada no justifica mas optimizacion, no una imposibilidad
matematica universal. Priorizar entonces los carriles existentes de politicas
SAC/DQN y definir por separado objetivos de eventos, direccion, barreras o
parametros de orden. No lanzar automaticamente una busqueda ilimitada de
predictores ni convertir el resultado negativo en una afirmacion sobre RL.

### Secuencia de datos

La primera seleccion financiera de negocio es **EURUSD**. Los targets vigentes
de la estrategia son corto 2/3/4/5/6/12 h y largo 18/24/36/48/60/72 h; perfiles
anteriores 1-6 h y 24-144 h quedan como evidencia, no calendario de despacho.
Tambien se conservan barreras de orden y objetivo de politica. FXMacroData aporta
calendario, consenso, actual y revisiones con vintage; Yahoo Finance y Alpaca
aportan covariables donde tengan cobertura point-in-time. ETH puede probar
mecánica y controles en un manifiesto secundario, pero no reemplaza EURUSD por
conveniencia ni se mezcla en su denominador.

La selección no empieza entrenando cientos de ramas. La secuencia vigente es:

1. Fase 1, por recalcular para la poblacion afectada: perfil individual y
   escalera causal sobre datos admitidos; ETH independiente no se invalida sin trazabilidad;
2. Fase 2: matrices feature-feature, alias, redundancia, complementariedad y
   estabilidad temporal;
3. Fase 3: clustering por correlación, mRMR, JMI y controles, con rankings y
   trayectorias K por target/horizonte;
4. Puerta preliminar entre las fases 3 y 4: congelar una primera ola de
   candidatos por target/horizonte a partir de los rankings TRAIN y la
   evidencia causal. Separar los controles `ALL_ADMISSIBLE` y `RANDOM_K` de
   la union de rasgos que requiere extractores GPU. Conservar por identidad
   los rasgos diferidos: ausencia de identificacion causal no prueba ausencia
   de efecto. Ninguna decision usa VALIDATION ni TEST. La cola GPU debe leer
   el manifiesto de esta puerta, no la union del control amplio.
5. Fase 4: extractibilidad raw/random/trained sobre la primera ola y controles
   de costo acotado; expansion a rasgos diferidos solo si el contraste semanal
   justifica su valor incremental;
6. validación wrapper bajo walk-forward semanal y manifiesto final;
7. primer consumidor prioritario: entorno RL validado, baselines y politicas;
8. ablacion I5-P y grupos/ramas/fusion reutilizables cuando aporten al contraste
   RL; barridos completos y forecasting quedan para despues.

Despues de los hitos principales, estudiar el error de reconstruccion de
extractores como posible detector de anomalias y de transiciones de regimen.
No usar esa idea para retrasar la preseleccion, la validacion semanal ni la
representacion modular.

No se reutilizan conteos de candidatos EURUSD de las metricas retiradas como
seleccion vigente. Las fases 2 y 3
se rigen por su [subplan específico](program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md).

La escalera causal tiene alta prioridad, pero no es una prueba binaria universal.
Se implementa enteramente con episodios ya observados:

1. La unidad es una ventana histórica anclada en una decisión t. A, el
   tratamiento, es un estado o cambio definido antes de mirar outcomes: por
   ejemplo tipo y sorpresa de un evento, transición de régimen o cruce de un
   indicador. Y es Y_s, Y_l o Y_b después de t. H contiene únicamente historia
   y calendario disponibles antes del evento.
2. El peldaño observacional mide asociación condicional y ganancia predictiva
   out-of-fold de A/X sobre Y dados H, con placebos temporales.
3. El peldaño intervencional **busca en el dataset** episodios tratados donde A=a
   y episodios de control donde A=a' con evento, calendario, régimen e historia H
   comparables. Matching/propensity/overlap y estimadores doubly robust o
   g-computation aproximan do(A=a) sólo dentro de soporte observado. Sin pares y
   balance suficientes, el resultado es NOT_IDENTIFIED.
4. El peldaño contrafactual toma un episodio observado, abduce de un SCM temporal
   sus perturbaciones, sustituye sólo A por un nivel alternativo con soporte
   histórico y propaga sus descendientes. Historia, no descendientes y shocks
   restantes permanecen fijos. Se exige reconstrucción factual, pares históricos
   análogos, placebos y sensibilidad; no se inventa un simulador externo ni se
   declara observable el outcome contrafactual.

`NOT_IDENTIFIED` conserva incertidumbre y no equivale automáticamente a rechazo.
Sí impide afirmar causalidad.

La primera escalera completa se ejecuta durante la **selección** sobre EURUSD.
Cada candidata priorizada recibe los tres intentos, no sólo asociación:

- si es evento, decisión o fuente exógena, A es su realización histórica;
- si es un estado endógeno o indicador derivado, se busca una transición o shock
  histórico bien definido y su mecanismo upstream; no se finge que hacer
  do(RSI=70) equivale a una acción económica físicamente interpretable;
- se buscan tratados, controles y niveles alternativos en períodos históricos
  con soporte; se ejecutan peldaños 2 y 3 o se conserva una negativa explícita
  de identificación, nunca una excusa para detenerse en correlación.

Usar el calendario para localizar tratamientos/controles del selector **no**
significa introducir el calendario como feature del predictor. Esa integración
es I11 y queda aplazada hasta terminar representación, NEAT, RL y la línea
operativa primaria. En ECL, Weather, Traffic u otro benchmark, la receta de
literatura nunca se modifica para satisfacer nuestra selección.

## 6. Extractibilidad por característica

Se añadirá a `feature-extractor` un plugin univariado temporal con interfaz
común, no un modelo declarado ganador de antemano. La unidad de entrada es:

- la serie causal de una característica en una ventana;
- calendario conocido en cada instante: sin/cos de hora, día de semana, día del
  año y, cuando aplique, sesión/feriado publicado;
- máscara de observado/faltante y delta temporal;
- opcionalmente el target **sólo como señal de entrenamiento/probe o condición
  generativa offline**. El target futuro nunca entra al encoder operacional.

El encoder conserva el eje temporal. Debe poder exportarse sin decoder y aceptar
R0/R1/R2. Se compararán, bajo igual presupuesto:

- identidad/raw y encoder aleatorio;
- AE Conv1D causal y denoising AE como controles;
- masked temporal autoencoder;
- un candidato de dependencia temporal tipo siamés/past-to-current;
- CVAE condicionado como variante generativa, no como requisito.

TimeSiam y los masked autoencoders son referencias candidatas, no una adopción
automática. La implementación elegida debe tener código/licencia admisibles y
reproducir su benchmark antes de llamarse referencia.

“Extractible” no significa sólo reconstruible. El informe por característica
incluye reconstrucción en escala original y normalizada, correlación/DTW/espectro,
estabilidad entre folds, preservación de probes hacia targets corto/largo/barrera,
ganancia incremental y ablación con reajuste, coste y sensibilidad al contexto
estacional. Buena reconstrucción sin utilidad no selecciona; mala reconstrucción
no rechaza sin el resto de evidencias. La capacidad generativa se evalúa aparte.

La reconstrucción como *entrada* operativa se comprueba después del
manifiesto final de I5 y antes de comparar arquitecturas en I6-A. Su
[subplan](program_v3/SELECTED_INPUT_DENOISING_ABLATION_2026_10_08.md) separa
entrada RAW, reconstrucción y latente, exige probar el soporte temporal real
del decoder y conserva **sin cambios** el target y el naive. El encoder actual
tiene un rezago informativo certificado de tres horas en su último estado:
emitir una muestra etiquetada en t no demuestra retraso cero.

## 7. Representación temporal modular

El modelo acepta un número **F configurado** de entradas o grupos; cuatro ramas
en un diagrama son ilustrativas, no un límite.

1. Cada rama recibe `(batch, time, channels)` y usa un extractor configurable.
   El default es Conv1D causal con ventana física mínima de 24 h y salida temporal.
2. Las ramas temporales se alinean causalmente a una rejilla común y se
   concatenan sólo en canales. En esta arquitectura ninguna Dense/Flatten/
   Pooling colapsa tiempo antes del cabezal. I6-D es un control no temporal
   separado; igualar tamaños o aplicar `Reshape` no crea pasos cronológicos.
3. La codificación posicional se aplica inmediatamente después de la fusión.
4. El núcleo default proyecta por instante, usa dos bloques Transformer completos
   y tres etapas Conv1D residuales que llevan tiempo 24→12→6→6 y canales
   32→16→8.
   La identidad inicial I6-DMO es una excepcion pequena y explicita: branches
   Conv1D de 4 filtros y core Conv1D 8->4, todos stride 1; no cambia ese default
   ni el diseno de plugins. Colapsa tiempo exclusivamente dentro de cada head.
5. El cabezal recibe la representación temporal `(batch, 6, 8)` o la forma
   configurada.

Cada rama puede cargar un donante propio. Early stopping, mejor checkpoint
restaurado, save/load y conteo de actualizaciones son obligatorios.

## 8. Orden arquitectónico y regímenes

La secuencia arquitectonica siguiente es un subprograma, no una dependencia
que bloquee RL. Puede reutilizarse el nucleo/extractor implementado, si sus
pesos y datos son elegibles; no se restaura ningun donante retirado.
El subprograma despues del manifiesto final es:

1. definir grupos, campos receptivos y controles ARCH con información emparejada;
2. preentrenar extractores por rama;
3. ejecutar E1 con **R0/R1/R2**;
4. escoger arquitectura y régimen mediante validación;
5. fijar un prefijo de ramas idéntico;
6. materializar secuencias fusionadas y preentrenar H-CORE;
7. ejecutar un experimento separado de transferencia del núcleo;
8. congelar/materializar la representación final para cabezales tardíos.

I6-D arranca tras I5 y la definición de controles I6-A, en paralelo con I6-B.
Primero se identifica el código, configuración y población de la ANN histórica:
ventana por característica -> Flatten -> capas Dense propias -> concatenación
vectorial -> su fusión/cabezal original. Una réplica usa sus entradas y
preprocesamiento originales; un reajuste con los rasgos seleccionados se
etiqueta como control nuevo. Si esa identidad no se recupera, se declara
`NOT_REPRODUCIBLE` y no se atribuyen resultados nuevos a la tesis. El segundo
brazo es una **nueva** rama Dense causal de ventana local con salida
anclada a cada instante, compatible con la misma fusión, núcleo y cabezal
temporales que Conv1D. No se presenta como la ANN histórica. Ambos controles
usan los rasgos finalmente seleccionados, idénticas filas, targets, horizontes,
naive y protocolo semanal para los controles nuevos; la réplica conserva su
protocolo original, con presupuestos y parámetros reportados. Una semilla
inicial; hasta tres sólo si la variabilidad obliga. El resultado puede favorecer
Dense sin ser descartado por no encajar en la explicación temporal. I6-D no
reabre I5 ni autoriza consultar TEST para elegir arquitectura.

- **R0:** extractores aleatorios y entrenables.
- **R1:** mismos donantes preentrenados, congelados.
- **R2:** mismos donantes que R1, ajustables.

No existe R3 canónico en este plan. Cualquier uso histórico de esa etiqueta tiene
otro alcance y no define un régimen nuevo. H-CORE no se ejecuta antes de elegir
arquitectura/régimen; tampoco se confunde “prefijo fijado durante H-CORE” con
“todo el branch congelado en R1”.

## 9. DEAP, DOIN y NEAT

- **DEAP** propone configuraciones e hiperparámetros.
- **DOIN** distribuye y registra evaluaciones de candidatos.
- **NEAT no optimiza esos parámetros.** Es un cabezal evolutivo tardío.

NEAT queda al final, despues del programa RL prioritario y de existir una representación final
congelada/materializada. Recibe esa representación, nunca las series crudas, y se
compara con un cabezal Dense simple bajo las mismas filas, targets y presupuesto.
Su contrato debe fijar interfaz latente, gramática, fitness, early stopping,
semilla y límites antes del primer ajuste. El ajuste fino de la representación
pertenece a R2 con cabezales diferenciables.

## 10. Benchmarks y métricas

Forecasting conserva Electricity/ECL, Weather y Traffic con la receta publicada.
Clasificación usa Banking77 u otro benchmark reciente sólo tras licencia, entorno
y código remoto admitidos. Finanzas conserva su contrato propio; un score
eléctrico no autoriza una decisión de trading.

Cada tabla incluye dataset/split/periodo, target/horizonte, población,
modelo/configuración, semilla, MAE/MSE u otras métricas en su escala, naive
pareado, skill, referencia/comparabilidad, dispersión cuando aplique, coste,
censura/early stop e identidad del artefacto.

| Evidencia consolidada | Resultado | Alcance |
|---|---|---|
| TimeFilter ECL L96 | MSE/MAE 0.161962/0.259662; publicado 0.158250/0.255750 | acuerdo operacional, 12 celdas |
| TimeFilter ECL L512 | MSE/MAE 0.150307/0.246397 | mapeo Tabla 9 no resuelto |
| Modular ECL previo | R0/R1/R2 MAE 0.371174/0.374584/0.368596; naive 0.868283 | desarrollo, arquitectura anterior |
| Weather | 12 celdas replicadas bit a bit según retorno | atribución en su expediente |
| Traffic H96 | MSE/MAE 0.375199/0.251143; publicado 0.375/0.251 | acuerdo operacional reportado |
| Laya wrapper | 12/12 salidas iguales al SDK; smoke macro-F1 0.3333/13 | fidelidad, no calidad financiera |

Todavía no existe un ganador modular producido por la selección completa, un
modelo modular optimizado por DOIN con test externo ni utilidad financiera
demostrada a partir de ese ganador.

## 11. Trading heurístico, RL y operación

RL es ahora el siguiente consumidor despues de datos y seleccion. Su
[subplan](program_v3/RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md) fija negocio,
baselines, metricas, pruebas, aprendizaje online y orden de ejecucion. La
seleccion no debe descartar utilidad para politica/riesgo por el solo fracaso
de un probe de MAE. No se afirma que SAC garantice ganancias.

La estrategia heuristica queda diferida. Usa familias de predicciones cortas y largas según su
plugin existente. Los barridos válidos mantienen una familia en persistencia
pareada mientras varían la otra, y después hacen el factorial. Sólo entran
pronósticos que superen su naive por horizonte.

RL compara `SAC raw` frente a `SAC modular` y `DQN raw` frente a
`DQN modular`. No se llama “cabezal” al actor/crítico ni se exige que SAC y DQN
compartan red. Reward, costes, episodios, early stopping y mejor checkpoint son
identidades separadas.

LTS consume sólo artefactos instalados y verificados. La regla vigente es
simulacion con profit neto positivo y costes configurados -> MT5 demo/Alpaca
paper automaticamente; no TEST ni otra aprobacion para demo. Comprobar cuenta
y endpoint paper/demo antes de servir; rechazar dinero real. Shadow queda
como comprobacion tecnica opcional, no como una autorizacion discrecional.

## 12. Extensión final: calendario causal como entrada

Integrar calendario económico, efectos compuestos o dossiers causales como
entradas del predictor/política tiene alto riesgo metodológico y queda
**DEFERRED_FINAL_OPTIONAL**. Sólo puede diseñarse cuando estén cerrados:

1. selección causal/extractiva y manifiesto final;
2. extractores por rama;
3. E1 R0/R1/R2 y prefijo elegido;
4. H-CORE y transferencia;
5. representación final, Dense frente a NEAT;
6. SAC/DQN raw frente a modular;
7. baseline shadow/paper del modelo sin esa extensión.

I11 tendrá su propio control sin calendario, as-of joins, disponibilidad,
ablación y test de aporte incremental. No bloquea ninguno de los puntos 1-7 ni
autoriza ahora plugins de entrada, adapters de consumidor o ajuste de modelos.

## 13. M5PHET

M5PHET ofrece una interfaz tipo Laya: datos/contexto y lenguaje natural entran a
un contrato tipado; un proveedor especializado compatible ejecuta; una salida
tipada se valida y persiste. Sus familias son clasificación, forecasting,
representación/no supervisado, causalidad y RL. DOIN es transversal.

Data-gov/data-lake/warehouse son adaptadores opcionales pero recomendados en las
apps de ejemplo. La ruta local funciona sin desplegar el stack. RAG puede ayudar
a interpretar documentación o contexto point-in-time por encima del proveedor;
no elige la respuesta ni altera capacidades declaradas.

## 14. Ejecución paralela y recursos

La RTX 5090 externa es primera opción para ajustes GPU largos; 4090, 5070 Ti y
4070 reciben trabajos independientes cuando memoria, temperatura, compatibilidad
y servicios lo permitan. La afinidad de replay al dispositivo original
prevalece. Las dos GPU de un host comparten RAM.

La VM MT5 de dragon no se reserva ni se ejecuta todavía. Hasta que extractores y
H-CORE preentrenados estén listos para inferencia sin entrenamiento, toda la RAM
y GPU admisibles de dragon pueden utilizarse para experimentación.

Siempre deben coexistir, cuando sean elegibles:

- GPU: extractibilidad de supervivientes, mejor candidato modular/DOIN,
  referencia fiel, pretraining o RL;
- CPU: inventario/perfiles, causalidad, métricas, warehouse y verificaciones;
- ingeniería: integración y pruebas en worktrees separados;
- operación: shadow/paper sin interferir con entrenamiento.

No se solicita autorización rutinaria. Sólo se eleva una acción humana
irreducible: credenciales, aceptación de licencia/código remoto, broker, hardware
físico o capital real.

## 15. Evidencia y almacenamiento

Data-gov gobierna disponibilidad; data-lake conserva datasets y artefactos por
contenido; warehouse recibe **todas** las métricas, disposiciones, rankings,
costes e identidades de cada fase. Ningún cierre científico depende sólo de un
JSON local. Cada fase exige lectura de vuelta del OLAP y copia fisica verificada
en almacenamiento propio, con un segundo espejo. Git conserva codigo, esquema,
configuracion, hashes e instrucciones de regeneracion, no bases grandes ni
releases de metricas como requisito de cierre. Los datos y resultados retirados
no se conservan como snapshots. DOIN puede usar cadena liviana:
persiste el cuerpo completo fuera de cadena y encadena manifiesto, locator y
hashes. Lectura y ETL verifican hash, esquema, autoridad y continuidad.

El cierre guarda configuración efectiva, semilla, versiones, identidad de
datos/modelo, métricas, naive, coste y recibos. Los arrays grandes se borran sólo
después de recomputación independiente y recibo por contenido.

## 16. Estado actual

DATA_INTEGRITY_HOLD. Recuperacion real de cotizaciones y limpieza en curso.
No hay receta predictiva promovida ni modelo entrenandose. RL_FIRST_AFTER_DATA_REPAIR
es la prioridad siguiente; predictores y NEAT diferidos. Rehacer las
metricas y la seleccion sobre el inventario admitido; no reutilizar cierres
ni porcentajes de los resultados eliminados. La existencia de nuevas descargas
o de este plan no equivale a haber completado la curacion del lake.

## 17. Posprograma: feature-selector desde M5PHET

I12 queda **DEFERRED_POST_PROGRAM**: empieza después de cerrar I0-I11 y los
trabajos futuros ya comprometidos, nunca como requisito de I5 ni como desvío
de la selección actual. `feature-selector` será el punto de entrada reusable;
M5PHET sólo interpretará la solicitud y coordinará proveedores declarados.
No se implementa en la campaña vigente.

La conversación podrá nombrar un data lake, uno o varios datasets (o todos los
admisibles) y un data warehouse para las métricas. Si el usuario no indica
warehouse, se usará sólo un perfil previamente configurado y autorizado: el
chat no inventa credenciales, destinos, targets ni permisos. El intérprete
producirá primero un plan tipado y revisable: población y columnas,
disponibilidad point-in-time, target/horizonte, split y calendario de
reentrenamiento, técnicas de perfil/causalidad/redundancia/extractibilidad,
presupuesto y destino analítico. Data-gov resolverá autoridad; data-lake
entregará bytes con identidad; `feature-selector` ejecutará los motores
versionados; warehouse recibirá métricas, disposiciones y recibos con lectura
de vuelta. Ningún texto recuperado por RAG sustituirá identidad o evidencia.

La interfaz ofrecerá estado, ETA medido, reanudación idempotente y resultados
en lenguaje natural enlazados a métricas y manifiestos tipados. Rechazará un
dataset sin acceso, un target ambiguo, una combinación proveedor/salida no
declarada y cualquier intento de llamar causal a `NOT_IDENTIFIED`. Antes de
ejecutar, mostrará el alcance y costo estimado; trabajos costosos requerirán
confirmación en el producto. Un ejemplo local sin stack desplegado seguirá
siendo posible. La aceptación exige pruebas de extremo a extremo con dataset
propio y ajeno, reinicio, aislamiento de credenciales y paridad con la CLI de
`feature-selector` sobre idénticos bytes y configuración. El chat no adelanta
ni altera el manifiesto financiero final de I5.

Las antiguas ordenes de lanzamiento, donantes y contrastes de modelos no
siguen vigentes. El procedimiento de prediccion antes del reajuste se conserva
como contrato de negocio para nuevos datos admitidos, no como autorizacion
para restaurar un checkpoint retirado ni repetir su experimento.
