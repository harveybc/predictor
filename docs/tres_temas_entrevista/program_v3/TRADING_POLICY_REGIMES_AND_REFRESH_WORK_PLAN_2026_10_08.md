# Baselines, politicas por regimen y frecuencia de actualizacion

Fecha: 2026-10-08. Estado: `DESIGNED / NOT_EXECUTED`.
Autoridad: [master v3](../MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md)
y [contrato semanal](BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md).

## 1. Objetivo y posicion en el programa

Elegir un procedimiento de trading que combine utilidad neta, riesgo y coste de
actualizacion bajo nuestro calendario real. No maximizar profit y minimizar
riesgo como si siempre existiera una solucion que optimizara ambos a la vez.
Conservar la frontera retorno/riesgo/coste y aplicar el mandato de riesgo vigente.

| Trabajo | Hito padre | Orden y resultado exigido |
|---|---|---|
| B0: contrato de baselines y riesgo | I0 / I0-W | Antes de comparar politicas; especificacion ahora, pruebas al implementar |
| B1: controles economicos ejecutables | I9-R / I10 | Cash, pasivo semanal, referencia continua; misma contabilidad |
| S1: MPC con costes | Banco de estrategias de I9-R | Mismos forecasts elegibles que la heuristica; no cambiar arquitectura a la vez |
| S2: contexto de regimen y mapa de utilidad | I9-R | Politicas simples y expertos ya disponibles antes de entrenar especialistas |
| S3: salida anticipada | I9-R | Mantener entradas, tamano y SL/TP; cambiar solo salida |
| C1: historia/cadencia/presupuesto | I8 aplicado a I9-R | DEAP/DOIN acotado; semanal, mensual y solo despues piloto diario |
| O1: promocion | I10 | Procedimiento sellado, shadow y demo/paper; no capital real |

No cambia la cola I6-A, no exige nuevas features para terminar I6/I7, no adelanta
NEAT ni I11. El factorial SAC raw/modular y DQN raw/modular mantiene las
dependencias actuales. Los controles simples pueden prepararse en CPU sin
esperar a los modelos, pero un resultado modular requiere sus artefactos reales.
Estas filas son trabajo planificado, no agentes ni experimentos despachados.

## 2. Estrategia baseline frente a metrica

Buy-and-hold es una politica. Retorno, drawdown, Sharpe y Sortino son medidas
para compararla con RL y otras politicas. No son sustitutos unos de otros.

| Identidad propuesta | Regla | Uso |
|---|---|---|
| `CASH` | No abrir riesgo; remuneracion solo si el contrato realmente la proporciona | Control operativo obligatorio |
| `PASSIVE_WEEKLY_LONG` | Abrir exposicion larga al comienzo elegible, mantener y cerrar antes de la pausa; reabrir la semana siguiente | Baseline pasivo compatible con el negocio |
| `BUY_HOLD_CONTINUOUS` | Comprar una vez y mantener, con financiacion y costes del instrumento declarado | Referencia de literatura, no control equivalente si cruza fines de semana |
| `SIMPLE_TREND` | Regla de tendencia causal, con parametros fijados en desarrollo | Control economico sencillo |
| `SIMPLE_REVERSION` | Regla de reversion causal, con parametros fijados en desarrollo | Control economico sencillo |
| `CURRENT_HEURISTIC` | Plugin existente con pronosticos elegibles cortos/largos | Control de nuestro sistema |

Registrar capital, moneda de cuenta, unidades, precio ejecutable, exposicion,
margen, apalancamiento, fees, spread, slippage, swap/funding y calendario.
Un EURUSD spot y un CFD no tienen el mismo contrato economico; tampoco ETH spot
y un perpetuo. No llamar buy-and-hold a una operacion financiada ignorando sus
costes. Definir en configuracion la exposicion inicial y el reajuste: mantener
unidades no es rebalancear cada barra a una fraccion de equity.

Publicar el pasivo sin apalancamiento como referencia transparente cuando sea
representable en el instrumento; comparar ademas con un control de presupuesto
de riesgo equivalente, sin escalar usando la volatilidad futura realizada.
La comparacion principal mantiene el mandato de riesgo del negocio. Una
politica con mas apalancamiento no gana por presentar solo mayor PnL bruto.
Para CASH con volatilidad cero, Sharpe/Sortino pueden ser indefinidos: guardar
`NOT_APPLICABLE` y la razon, no infinito ni una cifra inventada.

No crear un unico cociente de "skill RL" equivalente al MAE: retorno relativo
y riesgo tienen varias dimensiones. El naive de pronostico sigue pareado por
filas y horizonte, agregado en todo el periodo declarado; un filtro semanal
retrospectivo no sustituye esa puerta. La elegibilidad de un predictor para
TEST se decide con desarrollo, no con el naive futuro de ese TEST. Una politica
RL sin salida predictiva se evalua economicamente, sin MAE ficticio.

## 3. Metricas y contrato de decision

Catalogo minimo por politica, instrumento y periodo: retorno neto/bruto, PnL
realizado y no realizado, equity mark-to-market, CAGR cuando sea valido,
volatilidad, Sharpe, Sortino, maximo drawdown, Calmar y expected shortfall al
95%; operaciones, rotacion, exposicion, tenencia y costes desglosados. Declarar
frecuencia, anualizacion, tasa de referencia, unidades y tratamiento de periodos
sin mercado. El reward de entrenamiento no reemplaza estas metricas externas.

El drawdown se calcula sobre la equity neta cronologica completa, incluidos
costes y posiciones abiertas. No promediar drawdowns semanales para llamar al
resultado drawdown anual. Se informa PnL por regimen y tambien riesgo de la
trayectoria completa; concatenar periodos discontinuos del mismo regimen no
produce un drawdown operativo real. Mantener las semanas sin operaciones.

Comparar resultados sobre mismos instantes y capital, reportando diferencias
netas y riesgo. La eleccion usa un objetivo primario y limites de riesgo
predeclarados, mas una frontera de alternativas; no cambiar el objetivo de
profit a Sharpe despues de ver quien gana. Los limites vienen del mandato
operativo existente y quedan ligados a la configuracion del experimento.

## 4. Regimenes que permiten decisiones utiles

Un cluster no es rentable por definicion. Primero debe ser inferible con datos
disponibles y suficientemente estable; despues debe demostrar que condiciona
la utilidad de alguna politica frente a no usar ese estado.

1. Fijar observables causales: retorno/tendencia, volatilidad, liquidez o spread
   cuando existan, y contexto temporal conocido. Usar datos admitidos y no
   ampliar automaticamente el inventario congelado de la campaña actual.
2. Comparar un estado unico con pocos estados propuestos por HMM, jump model o
   detector online ya reutilizable. Buscar numero de estados, persistencia y
   parametros dentro de TRAIN/inner-validation temporal.
3. Emitir estados o probabilidades filtradas con cutoff. Rechazar etiquetas
   obtenidas por smoothing sobre toda la futura validacion. Guardar version y
   mapeo de estados para que un cambio de etiqueta no parezca cambio economico.
4. Construir una tabla `regimen x politica x parametros x bloque` con retorno,
   riesgo, rotacion, coste y soporte. Examinar pasivo, tendencia, reversion,
   heuristica, MPC y RL solo cuando cada componente sea elegible.
5. Elegir el experto y sus parametros usando bloques internos anteriores.
   Evaluar su enrutamiento en el bloque siguiente; reportar tambien un unico
   experto sin regimen y una mezcla fija como controles.
6. Entrenar especialistas solo si el contexto ofrece valor y hay soporte.
   Partir de representacion compartida y cabezales o ponderaciones por estado;
   una red separada por cada cluster no es el punto de partida obligatorio.

El mapeo "tendencia -> momentum; rango -> reversion; crisis -> cash" es una
hipotesis, no una asignacion aceptada. Los objetivos de TP/SL se evalúan junto a
volatilidad, spread y frecuencia de falsas salidas: un TP/SL estrecho puede ser
mas sensible al ruido, no mas tolerante. Las transiciones tambien se puntuan.

Cada celda informa barras, episodios independientes aproximados, operaciones,
ocupacion, incertidumbre y efecto frente al control. Los minimos de soporte se
declaran antes de comparar y pueden concluir `INSUFFICIENT_SUPPORT`; eso lleva
al experto compartido o CASH segun la regla fijada, no a eliminar esas fechas.
La maxima ganancia retrospectiva escogiendo el mejor experto en cada periodo
no es una estrategia ejecutable ni un baseline principal.

## 5. Construccion de datasets por estado

Para cada cutoff, ajustar detectores y preprocesadores con el corpus historico
permitido. Las tablas usadas para seleccionar expertos se forman con replay
temporal fuera del bloque de ajuste del detector; conservan el estado que
habria podido emitirse entonces. Nunca segmentar el año futuro completo y
despues entrenar sobre sus supuestos regimenes.

Cada episodio conserva activo, tiempos, contexto previo, estado, accion,
target/reward, disponibilidad y version de la politica que lo genero. Ventanas
de contextos pasados pueden solaparse, pero el soporte de labels maduras no
cruza el cutoff. La purga usa horizontes y tenencia, no solo fecha de origen.

No pegar meses discontinuos de un mismo regimen para que RL imagine una
transicion entre ellos. Mantener limites de episodio, warm-up temporal y
transiciones reales. Un muestreo ponderado para entrenar es otro tratamiento,
documentado; la evaluacion conserva frecuencia natural, transiciones y costes.
Retener los casos ambiguos, raros y nuevos con fallback explicito.

## 6. Tres relojes y una busqueda acotada

Separar frecuencia de observacion/inferencia del estado, frecuencia de decision
y frecuencia de actualizacion de pesos. Un detector puede responder cada hora
sin que se reajuste el encoder, el nucleo y SAC cada hora.

Referencia inalterada: cuatro años calendario de desarrollo moviles, actualizacion
semanal y años completos de validacion y test recorridos cronologicamente.
Los cuatro años son la historia de ajuste, no cuatro años nuevos de evaluacion
por cada candidato. Mensual es una aproximacion/sensibilidad con identidad
propia. Diario se estudia solo despues, no se supone superior por ser reciente.

Busqueda propuesta por etapas, sin producto cartesiano masivo:

1. Costear semanal/cuatro años para cada familia y modo FULL/WARM elegible,
   incluidos seleccion, pretraining, encoder/nucleo y checkpoint final.
2. Comparar mensual/cuatro años con la misma arquitectura y año de validacion.
3. Explorar historia en el conjunto inicial `{1, 2, 4}` años calendario,
   con la misma cadencia, antes de cruzar candidatos finalistas. La historia
   corta puede perder estados raros; se exige cobertura de regimenes.
4. Si hay utilidad y presupuesto, piloto diario sobre finalistas: ajuste ligero
   de cabezal o politica, o FULL de un modelo que quepa. Despues, solo los
   supervivientes recorren el año completo. Un piloto no es aceptacion anual.
5. DEAP propone y DOIN evalua candidatos sobre bloques internos temporales;
   se fija el calendario y las reglas antes del periodo externo. Registrar
   tambien intentos descartados y coste de busqueda.

Cada candidato fija cadencia, años, FULL/WARM, componentes entrenables,
presupuesto, early stopping, replay RL, parentesco de checkpoints y fallback
por release tardio. WARM puede retener memoria anterior a la ventana; no se
presenta como olvido estricto. Cambiar scaler exige mantener compatible la
representacion de entrada del checkpoint o reentrenar segun contrato.

El early stopping consume solo train e inner-validation del corpus anterior,
nunca la semana puntuada. Se conserva la regla compartida de paciencia basada
en la media train/inner-validation cuando corresponda al runner, con metrica,
direccion, escalas y restauracion declaradas. En RL son evaluaciones comparables
de politica, no una media arbitraria de reward y perdida del critico.

Estimacion de coste: enumerar releases reales segun mercado/calendario y sumar
los costes medidos por componente y modo. Para ETH, diario y pausa semanal
requieren convencion propia. No usar 252 o 365 sin enumerar los dias de ese
contrato. La disponibilidad antes de la apertura es parte de la factibilidad.

## 7. Evaluacion, automatizacion y limites de paralelismo

Primero desarrollo interno, despues validacion externa para seleccionar el
procedimiento y finalmente un recorrido TEST con reglas congeladas. Durante
TEST puede actualizarse con semanas pasadas y labels maduras conforme a esas
reglas; sus resultados no redisenan parametros, universo o cadencia. Una
politica adaptativa predeclarada no equivale a seleccionar manualmente sobre TEST.

Requisitos del futuro ejecutor, no comandos implementados por este documento:

- Declaracion JSON -> manifiesto de tareas por candidato/cutoff -> cola durable
  con claims exclusivos -> cierre -> OLAP/readback -> resumen por subfase.
- Estado consultable por campaña/candidato/semana: completas/esperadas,
  disposiciones, unidad activa, ultimo latido, RAM/VRAM medida y ETA por tiempos
  observados. ETA desconocida se nombra hasta el primer piloto representativo.
- Reanudar desde artefactos autenticados; una semilla por defecto y nunca mas
  de tres. No requerir un agente leyendo logs para despachar la siguiente unidad.
- Paralelizar candidatos independientes y controles CPU. WARM respeta la
  cadena de checkpoints; la equity y el estado operativo se recorren en orden.
  FULL puede ajustar ventanas independientes solo si sus entradas/selecciones
  no dependen de resultados externos ni de un checkpoint anterior.
- La 5090 es preferente para ajustes elegibles, con admision por memoria medida;
  otras GPUs reciben unidades independientes. No ocupar GPU con contabilidad
  que solo necesita CPU ni desalojar I6-A por esta ampliacion documental.
- Todas las metricas y fallos se guardan en OLAP; snapshot fisico local y segunda
  copia, manifiestos y scripts en Git. No subir bases binarias grandes a GitHub.

## 8. Aceptacion y entrega

| ID | Prueba requerida | Estado inicial |
|---|---|---|
| PR01 / BW19 | Pasivo semanal cierra/reabre con costes; continuo queda separado por contrato | PLANNED |
| PR02 / BW20 | Cambio de checkpoint no reinicia equity; cierre cancelado/fallido no fabrica fill | PLANNED |
| PR03 | Costes, exposicion y calendario pareados; Sharpe indefinido no se convierte en infinito | PLANNED |
| PR04 / BW21 | Alterar futuro no cambia estado, mapeo de experto ni parametro anterior | PLANNED |
| PR05 | Regimen raro conserva fechas y fallback; episodios desconectados no crean transiciones RL | PLANNED |
| PR06 / BW22 | Cambio de historia/cadencia modifica identidad y releases; no reutiliza score de otro protocolo | PLANNED |
| PR07 | WARM respeta padres y replay; FULL declara inicializacion y poblacion | PLANNED |
| PR08 | Busqueda DOIN no lee TEST; experto por estado no se elige con PnL futuro | PLANNED |
| PR09 | MDD anual sale de equity cronologica, no del promedio por semana/regimen | PLANNED |
| PR10 | Reinicio sin duplicados; cierre solo con poblacion completa y readback OLAP | PLANNED |

Publicar una matriz de politicas por regimen, curva de equity neta completa,
frontera retorno/riesgo/coste, diferencias pareadas y un parrafo con valores por
subfase. Aceptar una politica para shadow/paper requiere cumplir los limites
predeclarados y evidencia suficiente frente a sus baselines; no basta un Sharpe
positivo ni una semana favorable. Esta ampliacion no concede capital real.

## 9. Base bibliografica

- Shu, Yu y Mulvey, 2024: comparan regimenes inferidos online con HMM/jump model,
  buy-and-hold y costes. Apoya evaluar utilidad operativa del estado, no presume
  una estrategia ganadora en nuestro mercado.
  [Articulo](https://arxiv.org/html/2402.05272v3).
- Boyd et al., 2017: control multi-periodo con retorno, riesgo y costes;
  reutilizable para los horizontes corto/largo.
  [Articulo y codigo](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
- [Revision de estrategias](research/TRADING_STRATEGY_REVIEW_2026_10_08.md):
  alternativas y alcance de cada evidencia. El orden aprobado aqui concreta
  esa revision sin convertir resultados ajenos en mediciones propias.

La cadencia semanal y los cuatro años son decisiones de nuestro negocio.
No se presentan como optimos demostrados por la literatura. La busqueda de
cadencias e historias debe justificar cualquier alternativa con nuestros datos.
