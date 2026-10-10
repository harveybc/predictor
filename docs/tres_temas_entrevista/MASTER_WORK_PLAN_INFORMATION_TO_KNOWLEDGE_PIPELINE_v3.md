# Plan maestro v3: de información a conocimiento y operación

Actualizado: 2026-10-10. Este documento contiene sólo el plan vigente. Los
retornos RP, snapshots de máquinas, restricciones ya levantadas y dictámenes
superados son evidencia histórica en `docs/audits/` y `docs/handoffs/`; no
gobiernan la ejecución.

## 1. Autoridad vigente

El orden de autoridad es:

1. este plan maestro;
2. [contrato semanal del negocio](program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md);
3. [selección progresiva de características y representaciones](program_v3/FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md);
4. [fases 2 y 3 de selección](program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md);
5. [fase 4 de extractibilidad y validación](program_v3/FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md);
6. [ablación de reconstrucción causal de entradas](program_v3/SELECTED_INPUT_DENOISING_ABLATION_2026_10_08.md);
7. [arquitectura temporal modular](program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md);
8. [estado metodológico](program_v3/PROJECT_METHOD_STATE.json);
9. [cola actual](program_v3/EXPERIMENT_EXECUTION_QUEUE.json) y
   [estado de ejecución](program_v3/CURRENT_EXECUTION.md);
10. [checklist legible por máquina](program_v3/MASTER_CHECKLIST.json), proyección
   del apartado 4 y no un plan alterno.

La vista ejecutiva obligatoria es
[`MASTER_MILESTONE_PROGRESS.png`](../audits/evidence/canonical_20261003/MASTER_MILESTONE_PROGRESS.png),
generada desde
[`MASTER_MILESTONE_STATUS.json`](../audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json).
Todo retorno de orquestación debe mantener visibles M1-M8, actualizar evidencia,
próxima compuerta y ETA, y distinguir porcentaje de ingeniería de confianza
científica. Ningún esquema local de carriles reemplaza estos hitos.

Un documento fechado anterior sigue siendo auditable, pero no puede abrir una
campaña, cambiar dependencias ni redefinir un componente. Cuando contradiga esta
lista, prevalece esta lista. Una medición retenida conserva su significado y su
identidad; limpiar el plan no reescribe resultados.

## 2. Objetivo

Construir y comprobar un sistema de aprendizaje para trading algorítmico que:

- use sólo datos disponibles en el instante de decisión;
- seleccione características por utilidad predictiva y de negocio, evidencia
  causal, calidad de representación, complementariedad y coste;
- preserve la dimensión temporal desde cada entrada hasta el cabezal;
- mantenga una ANN densa ramificada por característica como control explícito
  no temporal, aunque contradiga nuestra hipótesis arquitectónica;
- permita preentrenar extractores por rama y, después, el núcleo fusionado;
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
   declara tiempo de evento, tiempo de disponibilidad y soporte. No se ajusta
   nada con validación/test ni se permite fuga por bordes, relleno o revisiones.
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
6. **No repetir.** Reutilizar artefactos autenticados. Sólo se repite por pérdida,
   corrupción, comparación necesaria o cambio científico explícito.
7. **Predicciones efímeras.** Se derivan todas las métricas declaradas y se
   verifica el catálogo antes de borrarlas. Se conservan config, semilla,
   checkpoints elegibles, métricas, identidades y recibos.
8. **Evidencia distinta de producto.** Tests de software prueban mecanismos;
   métricas científicas prueban hipótesis; shadow/paper prueba integración.
9. **Paralelismo útil.** Un bloqueo local no detiene perfiles, referencias,
   preparación ni entrenamientos independientes y elegibles.
10. **El negocio gobierna la evaluación financiera.** EURUSD validation y test
   se recorren por semanas consecutivas. Antes de cada semana se actualiza el
   modelo con una ventana móvil de cuatro años terminada en el cutoff. Se congela
   el procedimiento antes del test, no un único juego de pesos para todo el año.
   `LITERATURE_STATIC` permanece como modo separado para reproducciones fieles.

## 4. Checklist de control de alto nivel

Este checklist es la vista que debe consultar el orquestador antes de cada
despacho. Un punto sólo cambia de estado mediante evidencia enlazada.

- [ ] I0. Contrato de negocio: targets corto/largo, barrera, RL, costes y riesgo.
- [ ] I0-W. Walk-forward: cuatro años móviles, actualización semanal, años completos de validation/test y modo literatura separado.
- [ ] I1. Inventario: fuente, columna, unidad, frecuencia, licencia y disponibilidad.
- [ ] I2. Perfil básico completo por celda de métrica, no una marca global.
- [x] I3. Perfil individual y escalera causal por feature-target.
- [ ] I4-C. Dependencia y redundancia entre características; alias y clusters.
- [ ] I4-R. Rankings por clustering, mRMR y JMI; trayectorias K reproducibles.
- [ ] I5. Comparación conjunta de conjuntos y manifiesto final.
- [ ] I5-P. Ablación RAW/reconstrucción/latente sobre los rasgos seleccionados;
  el target y el naive permanecen intactos.
- [ ] I6-A. Agrupación, campos receptivos y controles ARCH emparejados.
- [ ] I6-D. ANN histórica por característica y contraste Dense de ramas,
  después de I5; resultado emparejado, sin atribuir tiempo a neuronas latentes.
- [ ] I6-B. Preentrenamiento de extractores de ramas seleccionadas.
- [ ] I7. E1: R0 frente a R1 frente a R2, mismo diseño y población.
- [ ] I6-E. Viabilidad de doble pronostico: resoluciones h, h/2 y h/4,
  ventanas ancladas al origen horario, sin suavizado, naive anual pareado.
  [Subplan de ejecucion](program_v3/I6E_ORIGIN_ANCHORED_RESOLUTION_WORK_PLAN.md).
  Preparacion CPU independiente; ajustes GPU despues del contraste I6-B/I7 activo.
- [ ] I7-H. Fijar prefijo ganador y probar H-CORE por separado.
- [ ] I8. Optimización DEAP distribuida por DOIN del modelo elegible.
- [ ] I9-N. Representación final congelada: Dense control frente a NEAT.
- [ ] I9-R. SAC raw/modular y DQN raw/modular.
- [ ] I10. LTS semanal, shadow, MT5 demo y Alpaca paper.
- [ ] I11. Extensión final opcional: calendario económico causal como entrada.
- [ ] I12. Posprograma: configurar `feature-selector` desde el chat de M5PHET.

Las casillas no implican ejecución serial. I2 de un lote puede coexistir con I4
de otro; referencias públicas, ingeniería, M5PHET y paper trading pueden avanzar
si no consumen artefactos que todavía no existen.

## 5. Datos y selección progresiva

### Decision de negocio sobre los predictores

I6-E debe producir una decision explicita sobre continuar la estrategia de
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

La primera selección financiera de negocio es **EURUSD**, con targets de retorno
1-6 h, 24-144 h, barreras de orden y objetivo de política. FXMacroData aporta
calendario, consenso, actual y revisiones con vintage; Yahoo Finance y Alpaca
aportan covariables donde tengan cobertura point-in-time. ETH puede probar
mecánica y controles en un manifiesto secundario, pero no reemplaza EURUSD por
conveniencia ni se mezcla en su denominador.

La selección no empieza entrenando cientos de ramas. La secuencia vigente es:

1. Fase 1, completa: perfil individual y escalera causal sobre EURUSD y ETH;
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
7. ablación I5-P de preprocesamiento sobre rasgos ya seleccionados;
8. grupos/ramas, fusión y downstream.

Despues de los hitos principales, estudiar el error de reconstruccion de
extractores como posible detector de anomalias y de transiciones de regimen.
No usar esa idea para retrasar la preseleccion, la validacion semanal ni la
representacion modular.

Los 34 pares EURUSD y tres pares ETH favorecidos por la regla causal de fase 1
son **candidatos con respaldo causal**, no la selección final. Las fases 2 y 3
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
5. El cabezal recibe la representación temporal `(batch, 6, 8)` o la forma
   configurada.

Cada rama puede cargar un donante propio. Early stopping, mejor checkpoint
restaurado, save/load y conteo de actualizaciones son obligatorios.

## 8. Orden arquitectónico y regímenes

La secuencia canónica después del manifiesto final es:

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

NEAT sólo se ejecuta después de existir una representación final
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

La estrategia heurística usa familias de predicciones cortas y largas según su
plugin existente. Los barridos válidos mantienen una familia en persistencia
pareada mientras varían la otra, y después hacen el factorial. Sólo entran
pronósticos que superen su naive por horizonte.

RL compara `SAC raw` frente a `SAC modular` y `DQN raw` frente a
`DQN modular`. No se llama “cabezal” al actor/crítico ni se exige que SAC y DQN
compartan red. Reward, costes, episodios, early stopping y mejor checkpoint son
identidades separadas.

LTS consume sólo artefactos instalados y verificados. Las promociones son offline
→ shadow → MT5 demo/Alpaca paper. No hay autorización de capital real.

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
JSON local. Cada fase exige lectura de vuelta del OLAP, snapshot físico DuckDB
local y copia pública versionada. El repositorio conserva manifiesto, esquema,
SHA-256 y URL; el binario comprimido se publica como asset de release para no
inflar el historial Git. DOIN puede usar cadena liviana:
persiste el cuerpo completo fuera de cadena y encadena manifiesto, locator y
hashes. Lectura y ETL verifican hash, esquema, autoridad y continuidad.

El cierre guarda configuración efectiva, semilla, versiones, identidad de
datos/modelo, métricas, naive, coste y recibos. Los arrays grandes se borran sólo
después de recomputación independiente y recibo por contenido.

## 16. Estado actual

El estado observado y la cola se mantienen únicamente en
`CURRENT_EXECUTION.md` y `EXPERIMENT_EXECUTION_QUEUE.json`. Las fases 1, 2 y 3
están cerradas (2026-10-07: `PHASE_2_COMPLETE.json` y
`PHASE_3_FILTER_COMPLETE.json` para EURUSD y ETH, cubo vivo reconciliado,
snapshot `phase2-3-feature-selection-20261007`; sin ganador predictivo). El
objeto siguiente es integrar el runner real de fase 4, medir su costo y ejecutar
la cola automatica. El snapshot binario se conserva en dos maquinas, con
manifiesto y SHA-256 en Git; la subida a GitHub fue cancelada. El calendario se usa
ahora sólo como evidencia histórica del selector; su integración como entrada
permanece en I11.

NEAT queda fuera de la cola hasta que exista representación final congelada.
Toda orden anterior que lo ejecute sobre entradas crudas o lo use como optimizador
de hiperparámetros está superada.

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
