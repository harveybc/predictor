# Plan maestro v3: de información a conocimiento y operación

Actualizado: 2026-10-05. Este documento contiene sólo el plan vigente. Los
retornos RP, snapshots de máquinas, restricciones ya levantadas y dictámenes
superados son evidencia histórica en `docs/audits/` y `docs/handoffs/`; no
gobiernan la ejecución.

## 1. Autoridad vigente

El orden de autoridad es:

1. este plan maestro;
2. [contrato semanal del negocio](program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md);
3. [selección progresiva de características y representaciones](program_v3/FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md);
4. [fases 2 y 3 de selección](program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md);
5. [arquitectura temporal modular](program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md);
6. [estado metodológico](program_v3/PROJECT_METHOD_STATE.json);
7. [cola actual](program_v3/EXPERIMENT_EXECUTION_QUEUE.json) y
   [estado de ejecución](program_v3/CURRENT_EXECUTION.md);
8. [checklist legible por máquina](program_v3/MASTER_CHECKLIST.json), proyección
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
- [ ] I6-A. Agrupación, campos receptivos y controles ARCH emparejados.
- [ ] I6-B. Preentrenamiento de extractores de ramas seleccionadas.
- [ ] I7. E1: R0 frente a R1 frente a R2, mismo diseño y población.
- [ ] I7-H. Fijar prefijo ganador y probar H-CORE por separado.
- [ ] I8. Optimización DEAP distribuida por DOIN del modelo elegible.
- [ ] I9-N. Representación final congelada: Dense control frente a NEAT.
- [ ] I9-R. SAC raw/modular y DQN raw/modular.
- [ ] I10. LTS semanal, shadow, MT5 demo y Alpaca paper.
- [ ] I11. Extensión final opcional: calendario económico causal como entrada.

Las casillas no implican ejecución serial. I2 de un lote puede coexistir con I4
de otro; referencias públicas, ingeniería, M5PHET y paper trading pueden avanzar
si no consumen artefactos que todavía no existen.

## 5. Datos y selección progresiva

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
4. Fase 4: extractibilidad raw/random/trained sobre candidatos y controles;
5. validación wrapper bajo walk-forward semanal y manifiesto final;
6. grupos/ramas, fusión y downstream.

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

## 7. Representación temporal modular

El modelo acepta un número **F configurado** de entradas o grupos; cuatro ramas
en un diagrama son ilustrativas, no un límite.

1. Cada rama recibe `(batch, time, channels)` y usa un extractor configurable.
   El default es Conv1D causal con ventana física mínima de 24 h y salida temporal.
2. Las ramas se alinean causalmente a una rejilla común y se concatenan sólo en
   canales. Ninguna Dense/Flatten/Pooling colapsa tiempo antes del cabezal.
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
`CURRENT_EXECUTION.md` y `EXPERIMENT_EXECUTION_QUEUE.json`. La fase 1 está
cerrada. La prioridad crítica única es terminar fase 2 y fase 3:
dependencia/redundancia entre features y filtros reproducibles. Fase 4,
extractibilidad, permanece sin iniciar hasta ese cierre. El calendario se usa
ahora sólo como evidencia histórica del selector; su integración como entrada
permanece en I11.

NEAT queda fuera de la cola hasta que exista representación final congelada.
Toda orden anterior que lo ejecute sobre entradas crudas o lo use como optimizador
de hiperparámetros está superada.
