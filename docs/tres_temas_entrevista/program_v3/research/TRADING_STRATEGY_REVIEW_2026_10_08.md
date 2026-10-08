# Estrategias de trading: literatura, alternativas y prioridades

Fecha de consulta: 2026-10-08. Estado: `RESEARCH_ONLY / NO_NEW_MEASUREMENT`.

Esta revision responde a una solicitud de explorar estrategias mientras avanza
I6-A. No modifica el master plan, la seleccion congelada, las colas de computo,
los modelos en ejecucion ni las reglas de acceso a TEST. Ninguna estrategia
descrita aqui ha sido implementada o medida por esta revision.

Lectura rapida: [oportunidades](#2-mapa-de-oportunidades),
[estrategias cercanas](#3-familias-con-mejor-encaje),
[extensiones](#4-otras-familias-que-merecen-conservarse),
[codigo reutilizable](#5-reutilizar-codigo-antes-de-implementarlo),
[comparacion propuesta](#7-comparacion-que-produciria-conocimiento-util).

## 1. Alcance y lectura

Revision amplia y dirigida, no metaanalisis ni afirmacion de haber agotado toda
la literatura. Incluye antecedentes necesarios y trabajos recientes de
2024-2026. Se priorizaron articulos, versiones de autor, editoriales y codigo
oficial. Los resultados publicados no se trataron como reproducciones nuestras.
Se inspeccionaron metodos y costes en los textos completos de CPD, jump models,
arbitraje estadistico profundo, Momentum Transformer, Network Momentum,
AlphaGen y AlphaForge. En otras referencias se verificaron la ficha primaria
y el alcance del resumen; no se extraen de ellas resultados numericos de tablas.

Una estrategia necesita cinco decisiones, no solo un predictor: cuando entrar,
en que direccion, cuanto comprar o vender, cuando salir y cuando no operar.
Un detector de regimen, una representacion, un factor y un estimador causal son
componentes posibles; por separado no completan esas decisiones.

**Recomendacion:** conservar el camino actual y, cuando corresponda al hito de
estrategias, contrastar primero control multi-horizonte con costes, seleccion
de politica por regimen y salidas por barreras. Arbitraje relativo y lead-lag
son las siguientes familias. Calendario causal, descubrimiento masivo de alphas,
agentes de lenguaje y NEAT siguen siendo extensiones, no prerrequisitos nuevos.

## 2. Mapa de oportunidades

La prioridad es juicio de diseno nuestro, no un ranking publicado de rentabilidad.
CPU/GPU indica la necesidad probable del prototipo, no un coste medido.

| Idea | Decision que produce | Fuente de ventaja propuesta | Encaje / coste inicial |
|---|---|---|---|
| Control multi-horizonte con costes | Posicion objetivo y trayectoria de reduccion/aumento | Aprovechar persistencia de la senal sin pagar rotacion innecesaria | Alto; CPU, reutiliza pronosticos |
| Expertos por regimen | Tendencia, reversion o efectivo, con pesos | No exigir la misma politica en todos los mercados | Alto; detector CPU y expertos existentes |
| Primera barrera / salida anticipada | Abrir, mantener, reducir o cerrar | Predecir eventos de utilidad directa y su orden | Alto; etiquetas/cabezal de eventos, GPU acotada |
| Tamano por riesgo e incertidumbre | Exposicion y abstencion | Evitar apuestas grandes con edge debil o riesgo alto | Alto; CPU, complemento |
| Arbitraje relativo | Largo una cesta, corto otra | Convergencia de desviaciones respecto a factores comunes | Medio; datos de varios activos y dos patas |
| Red lead-lag | Operar seguidores tras movimientos de lideres | Propagacion temporal de informacion | Medio; CPU primero |
| Politica diferenciable | Posicion directamente desde la representacion | Optimizar utilidad neta en vez de solo MAE | Medio; GPU, comparador de SAC/DQN |
| Selector adaptativo de estrategias | Distribuir capital entre politicas y efectivo | Adaptarse sin reentrenar todas las politicas a cada tick | Medio; CPU, exige politicas ya utiles |
| Eventos macro con efectos dinamicos | Exposicion condicionada al shock y su horizonte | Respuesta heterogenea o incompleta al evento | Final; calendario point-in-time |
| Alphas simbolicos | Formula y combinacion de factores | Descubrir interacciones interpretables | Final; busqueda costosa y muchas pruebas |
| Carry / basis | Posiciones compensadas spot-futuro | Prima de financiacion o desequilibrio de base | Otro negocio; faltan contratos y costes verificados |
| Market making adaptativo | Precios bid/ask y control de inventario | Captura de spread compensando seleccion adversa | Muy posterior; libro, colas y latencia |
| Agente multimodal | Extraer eventos y apoyar una politica | Informacion de texto no presente en precios | Posterior; coste de inferencia y riesgo temporal |

## 3. Familias con mejor encaje

### 3.1 Control multi-horizonte: usar mejor el corto y el largo

Garleanu y Pedersen distinguen senales por su velocidad de desaparicion y
obtienen una politica de ajuste parcial hacia una posicion objetivo. Su ejemplo
empirico es **in-sample**: sirve para estudiar el control con costes, no para
demostrar capacidad predictiva fuera de muestra. [Articulo de autor, 2013](https://pages.stern.nyu.edu/~lpederse/papers/DynamicTrading.pdf).

Boyd et al. formalizan un controlador que planifica varias operaciones,
ejecuta solo la primera y vuelve a resolver con informacion nueva. Combina
retorno esperado, riesgo, costes de transaccion y de tenencia; la calidad del
pronostico queda fuera del alcance de ese trabajo. [Articulo y software, 2017](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).

**Propuesta nuestra:** nuestros horizontes cortos y largos alimentan un control
predictivo (MPC). No comprar solo porque el extremo previsto sea positivo:
comparar la ventaja esperada de abrir, mantener, reducir y cerrar, descontando
spread, comision, slippage y financiacion. Incluir una zona de no operacion.

Los retornos acumulados a 1, 2 y 24 horas no se suman como rendimientos
independientes. Primero hay que construir una trayectoria coherente de retornos
incrementales y su incertidumbre. La restriccion de cerrar antes del fin de
semana entra como condicion terminal. El control explicito es una comparacion
util contra la heuristica original y contra SAC, no un sustituto impuesto.

### 3.2 Regimenes: elegir una politica, no reiniciar modelos sin parar

Shu, Yu y Mulvey usan jump models con penalizacion por cambios de estado.
En su tabla 4, S&P 500, 1990-2023, el Sharpe pasa de 0,48 en buy-and-hold a
0,68; el maximo drawdown pasa de -55,2% a -26,6%. Incluyen 10 puntos basicos
por lado y retraso de ejecucion. Es evidencia en indices diarios, no en EURUSD
horario. El detector online tambien llega tarde a algunos giros.
[Texto y tabla 4, 2024](https://arxiv.org/html/2402.05272v3).

**Propuesta nuestra:** comenzar con pocos estados y un experto compartido.
El estado condiciona pesos entre tendencia, reversion y efectivo. Usar
probabilidades o puntuaciones de estado cuando esten disponibles; mantener
una politica comun cuando el estado sea ambiguo o tenga pocas muestras.
Actualizar los parametros en el calendario semanal acordado, mientras la
inferencia del estado puede actualizarse a cada barra. Un cambio detectado
no exige entrenar desde cero inmediatamente.

Pruebas separadas: modelo unico sin regimen; mismo modelo con contexto de
regimen; expertos con compuerta. Asi distinguimos si ayuda el contexto o la
especializacion. Los estados usados en backtest deben ser los que podian
inferirse entonces, nunca una segmentacion retrospectiva de todo VALIDATION.
La representacion temporal modular puede ser compartida por los expertos.

### 3.3 Tendencia lenta y reversion rapida

Wood, Roberts y Zohren agregan deteccion online de cambios a una Deep Momentum
Network. Informan una mejora de Sharpe de aproximadamente un tercio en
1995-2020, pero las tablas principales son antes de costes. El apendice C
muestra deterioro rapido por encima de unos 2 puntos basicos en su configuracion,
asociado a la rotacion de la reversion rapida. [JFDS 2022, version completa](https://arxiv.org/html/2105.13727v3).

El Momentum Transformer posterior estudia una arquitectura attention-LSTM,
50 futuros y retornos netos de costes. Usa ventanas expansivas y no nuestro
reentrenamiento semanal. [Version de autor, 2022](https://arxiv.org/html/2112.08534v3).

**Propuesta nuestra:** probar una regla explicita que decida si una contradiccion
entre corto y largo significa salir, reducir temporalmente o invertir la posicion.
Compararla con nuestra regla actual usando exactamente las mismas predicciones.
La inversion rapida no tiene por que ser la mejor despues de pagar dos cambios
de posicion. No hace falta comenzar con otro Transformer.

### 3.4 Barreras y tiempo de salida

**Hipotesis de diseno nuestra, no resultado publicado trasladado:** predecir
`TP_primero`, `SL_primero`, `ninguno_antes_del_cierre_semanal`, junto con tiempo
hasta el evento y excursion favorable/adversa. Puede haber modelos de riesgos
competitivos o clasificacion calibrada, condicionados a la posicion actual.

Esto responde directamente a la logica del plugin existente: no basta conocer
el precio al final del horizonte si el stop se toca antes. El control inicial
mas limpio mantiene entradas, tamanos y SL/TP actuales y cambia **solo la salida
anticipada**. Despues se investiga la entrada. No atribuir una mejora a la salida
si tambien se cambiaron capital, margen o reglas de entrada.

La literatura de optimal stopping muestra herramientas para decisiones de
continuar o detener, pero los resultados de opciones simuladas no demuestran
alpha en trading direccional. [Becker et al., 2019](https://arxiv.org/abs/1908.01602).

Si una barra toca SL y TP, OHLC no revela necesariamente el orden. Se necesita
mayor resolucion o una regla conservadora predeclarada. Calibrar probabilidades
en bloques temporales internos y registrar censura; no etiquetar como perdida
un evento cuyo horizonte no pudo observarse.

### 3.5 Riesgo, incertidumbre y abstencion

Moreira y Muir documentan estrategias que reducen riesgo cuando aumenta la
volatilidad, en varios factores y currency carry. Es una politica de exposicion,
no una demostracion de que cualquier predictor de volatilidad produzca alpha.
[Volatility-Managed Portfolios, JF 2017](https://www.nber.org/papers/w22208).

**Propuesta nuestra:** separar direccion de tamano. Comparar la misma senal con
exposicion fija, volatilidad objetivo y tamano dependiente de ventaja neta e
incertidumbre calibrada. Respetar maximos de margen, apalancamiento y perdida.
La confianza de una red no se interpreta automaticamente como probabilidad
de ganar. Abstenerse debe competir contra controles de exposicion comparables:
reducir operaciones por si solo reduce algunos riesgos.

### 3.6 Arbitraje relativo y residuos

Pairs trading tiene antecedentes sistematicos; Gatev, Goetzmann y Rouwenhorst
estudian una regla de valor relativo, no un predictor aislado.
[Version NBER](https://www.nber.org/papers/w7032).
Avellaneda y Lee estudian residuos de factores/PCA y reversion a la media en
acciones estadounidenses. [Texto de autores](https://math.nyu.edu/inmemoriam/avellaneda/AvellanedaLeeStatArb20090616.pdf).

Guijarro-Ordonez, Pelger y Zanotti combinan residuos de factores condicionales,
CNN-Transformer y politica. En la tabla IX, IPCA con tres factores y objetivo
Sharpe obtiene Sharpe OOS 1,24, con 5 puntos basicos de transaccion y 1 punto
basico de coste diario por posicion corta. No modelan impacto de mercado.
[Deep Learning Statistical Arbitrage, seccion III.J](https://arxiv.org/html/2106.04028v2).

**Propuesta nuestra:** una cesta relacionada con el activo, hedge ratio estimado
solo con pasado y politica sobre el residuo, con control lineal/OU antes de un
extractor profundo. Un filtro de Kalman puede actualizar ese hedge ratio; no
confundir filtrado con suavizado retrospectivo. Alta correlacion no garantiza
cointegracion ni convergencia futura. EURUSD y ETH no se presuponen un par.
Cotizar ambas patas, disponibilidad real de cortos, financiacion y riesgo de
ruptura. No simular una cobertura que el instrumento contratado no permite.

### 3.7 Redes lead-lag

Li y Ferreira combinan momentum propio y propagacion entre futuros mediante
una red. El texto describe costes por mercado y una convencion explicita de
retraso; su evidencia incluye bootstrap de series de precios. Se consulta como
preprint, no como reproduccion nuestra. [Network Momentum, 2025](https://arxiv.org/html/2501.07135v1).

Cartea, Cucuringu y Jin construyen redes dirigidas con firmas de orden dos.
El resumen reporta resultados en acciones estadounidenses; aqui no se adopta
su cifra de Sharpe porque no se audito su tabla completa de costes. La pagina
institucional registra publicacion en Journal of Empirical Finance en 2026.
[Trabajo](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4599565),
[ficha institucional](https://www.maths.ox.ac.uk/people/alvaro.cartea).

**Propuesta nuestra:** investigar si movimientos de activos relacionados
anticipan retorno, volatilidad o cambio de politica de EURUSD/ETH a determinados
retardos. Seleccionar retrasos dentro de TRAIN, controlar multiplicidad y medir
estabilidad fuera del bloque de ajuste. Emparejar informacion por disponibilidad
real: cierres asincronos y precios estancados pueden fabricar lideres aparentes.
Un enlace predictivo dirigido no se etiqueta automaticamente como causal.

## 4. Otras familias que merecen conservarse

### 4.1 Politica diferenciable como tercer comparador

Deep Momentum Networks aprende direccion y tamano optimizando Sharpe e incorpora
penalizacion por rotacion. [Lim, Zohren y Roberts, JFDS 2019](https://arxiv.org/abs/1904.04912).
Zhang, Zohren y Roberts tambien plantean optimizacion directa de carteras.
[Deep Learning for Portfolio Optimization, 2020](https://arxiv.org/abs/2005.13665).

Propuesta: cabezal que produce posicion desde nuestra representacion temporal,
entrenado con utilidad neta diferenciable. Comparar con heuristica y SAC/DQN.
El simulador completo sigue siendo el juez: una perdida diferenciable puede
omitir fills discretos, stops o restricciones. SAC continuo y DQN discreto
requieren registrar espacios de accion distintos, no llamarlos identicos.

### 4.2 Aprendizaje online para asignar capital entre estrategias

Lin, Wang y Zhou estudian bandits contextuales con criterio media-varianza y
Thompson sampling, con una aplicacion de cartera. El resultado teorico depende
de sus supuestos, no garantiza rendimiento en mercados no estacionarios.
[Trabajo primario, 2022](https://arxiv.org/abs/2206.12463).

Propuesta: elegir entre politicas previamente aceptadas y efectivo, con coste
de cambio. Si podemos simular el resultado de todos los expertos con el mismo
historial, usar aprendizaje con informacion completa como control; no inventar
un problema bandit donde no existe feedback parcial. Las recompensas retrasadas
y operaciones todavia abiertas deben tratarse explicitamente.

### 4.3 Eventos, efectos causales y respuestas con retraso

Andersen et al. encuentran saltos asociados a sorpresas macro, definidos como
diferencia entre publicacion y expectativa, con efectos de signo y momento.
No demuestra que pueda capturarse ese salto despues de recibirlo.
[Micro Effects of Macro Announcements, AER 2003](https://www.nber.org/papers/w8959).

Lewis y Syrgkanis desarrollan estimacion de efectos dinamicos de tratamientos
con controles de alta dimension. No es un articulo de rentabilidad financiera.
[NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bf65417dcecc7f2b0006e1f5793b7143-Abstract.html).
Runge et al. explican identificacion causal en series y limites por supuestos.
[Nature Communications, 2019](https://www.nature.com/articles/s41467-019-10105-3).

Propuesta para las tres preguntas, con datos historicos:

1. Asociacion: curva de respuesta por horizonte y estado previo al evento.
2. Intervencion: comparar episodios historicos tratados y controles comparables,
   ajustando confusores anteriores al tratamiento; documentar identificacion,
   solapamiento y sensibilidad. No controlar mediadores posteriores como si
   fueran covariables previas.
3. Contrafactual: para un episodio, estimar una trayectoria alternativa bajo un
   modelo estructural o controles historicos validos, con incertidumbre. Esa
   trayectoria es estimada, no un segundo desenlace observado del mismo episodio.

Encontrar casos similares ayuda, pero no vuelve verdadero `do(x)` por si solo.
Separar efecto del evento, efecto ya incorporado al precio y efecto residual
aprovechable tras latencia/costes. La ausencia de identificacion no prueba efecto
cero. Eventos simultaneos necesitan control conjunto o exclusion predeclarada.
Esta extension de calendario sigue **al final del plan**, como se ordeno.

### 4.4 Descubrimiento de alphas interpretables

AlphaGen busca conjuntos de formulas que funcionen conjuntamente, mediante RL;
es un trabajo aceptado en KDD 2023. [Articulo](https://arxiv.org/abs/2306.12964).
AlphaForge combina generacion y pesos adaptativos; publicado en AAAI 2025.
[Articulo](https://ojs.aaai.org/index.php/AAAI/article/view/33365).
AlphaQCM usa RL distribucional para explorar formulas; ICML 2025.
[Articulo](https://proceedings.mlr.press/v267/zhu25ag.html).

Propuesta: una gramatica limitada de factores a partir de entradas admisibles,
con limites de complejidad, rotacion y correlacion con estrategias existentes.
DEAP, distribuido mediante DOIN, seria un control natural para busqueda de
formulas. Esto no convierte NEAT en optimizador de hiperparametros ni adelanta
su lugar en el plan. La eficacia de alphas transversales sobre cientos de
acciones no se traslada automaticamente a una unica serie EURUSD.

### 4.5 Carry, liquidez y texto: diferentes requisitos

**Carry/basis:** Schmeling, Schrimpf y Todorov estudian la diferencia futuro-spot
y fricciones de arbitraje en cripto; Management Science, 2026.
[Crypto Carry](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.05069).
Comprar spot y vender futuro puede reducir riesgo direccional, no eliminar
liquidacion, margen, custodia, riesgo de base ni fallo de contraparte. Futuros
con vencimiento y perpetuos con funding son contratos diferentes. No asumir
que nuestras suscripciones permiten ejecutar o reconstruir ambos.

**Market making:** RL puede ajustar aversion al riesgo y cotizaciones de un
controlador Avellaneda-Stoikov, en vez de inventar precios sin estructura.
[Estudio PLOS ONE, 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277042).
Exige libro, llegada de ordenes, posicion en cola, latencia y fills creibles.
No es una prueba honesta con nuestras barras horarias solamente.

**Texto/multimodal:** FinAgent combina noticias, precios y herramientas.
[Version de autor, 2024](https://arxiv.org/abs/2402.18485).
Propuesta: primero extraccion tipada de eventos y novedad, no un LLM con control
directo del broker. Hay que descartar noticias revisadas, memoria futura y
contaminacion temporal del modelo; contabilizar coste de inferencia. No se
promueve una cifra de rentabilidad del resumen como si fuera nuestra evidencia.

## 5. Reutilizar codigo antes de implementarlo

Enlaces verificados o enlazados explicitamente desde el articulo. No se
instalo ni ejecuto ninguno. Fijar revision, auditar licencia/dependencias y
reproducir la configuracion original antes de una adaptacion.

| Componente | Codigo primario | Limite concreto |
|---|---|---|
| MPC / carteras | [cvxportfolio](https://github.com/cvxgrp/cvxportfolio) | GPL-3.0; necesita forecasts y un contrato de costes coherente |
| Regimenes persistentes | [jump-models](https://github.com/Yizhan-Oliver-Shu/jump-models) | Enlazado por el autor; verificar inferencia online, no solo ajuste retrospectivo |
| Momentum Transformer | [trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformer) | Enlazado por el articulo; datos originales de futuros licenciados |
| Arbitraje profundo | [dlsa-public](https://github.com/gregzanotti/dlsa-public) | README exige `use_residual_weights=True`; datos originales no redistribuidos por licencia |
| AlphaGen | [ICT-FinD-Lab/alphagen](https://github.com/ICT-FinD-Lab/alphagen) | El enlace antiguo RL-MLDM redirige; entorno y mercados distintos |
| AlphaForge | [repositorio enlazado en el texto](https://github.com/Anonymous240816/AlphaForge) | Identidad anonima heredada de revision; revisar licencia antes de integrar |
| AlphaQCM | [ZhuZhouFan/AlphaQCM](https://github.com/ZhuZhouFan/AlphaQCM) | MIT; instrucciones con Python 3.8 / Torch 1.13.1, no asumir compatibilidad local |
| Causalidad temporal | [Tigramite](https://github.com/jakobrunge/tigramite) | GPL-3.0; descubrimiento y estimacion sujetos a supuestos, no motor de PnL |

Codigo disponible no equivale a replica posible: pueden faltar datos originales,
vintages, costes, universo point-in-time o versiones. Se anota por componente,
sin convertir una adaptacion a otros datos en reproduccion exacta.

## 6. Datos y fidelidad al negocio

Las suscripciones a Yahoo Finance, Alpaca y FXMacroData son oportunidades de
adquisicion, no evidencia de que cada campo historico ya exista en el lago.
No se verificaron credenciales, permisos comerciales ni inventarios en esta
revision. Antes de usar un campo: cobertura, frecuencia, zona horaria,
publicacion/recepcion, revisiones, licencia y digest de los bytes.

| Familia | Datos minimos adicionales | Riesgo que hay que resolver |
|---|---|---|
| Regimen/MPC/salidas | Precios y costes ejecutables, pronosticos as-of | Orden intrabar, spread, hora de decision y fill |
| Cestas/lead-lag | Activos sincronizados, ajustes y universo historico | Sesgo de supervivencia, desfases, shorts y dos patas |
| Macro | Actual, consenso previo, publicacion, recepcion y vintage | Revision o consenso recibido despues del anuncio |
| Carry | Spot y contrato futuro/perpetuo, margen y financiacion | Datos de funding no equivalen a financiacion garantizada |
| Market making | Quotes/trades y libro con timestamps | Posicion en cola y seleccion adversa no observables en OHLC |

Mantener el negocio ya definido: cuatro anos moviles para ajustar antes de cada
semana evaluada, validacion/test anuales recorridos cronologicamente, variante
mensual solo como aproximacion declarada, cierre obligatorio antes del fin de
semana. ETH necesita calendario de cierre propio porque cotiza continuamente.
Normalizadores, seleccion adaptativa, calibradores y detectores se ajustan sin
ver la semana futura. Solo etiquetas ya maduras pueden entrar al entrenamiento.
En una evaluacion prequential, una semana de TEST pasada puede incorporarse al
ajuste posterior si esa regla estaba congelada; nunca para redisenar la politica
con lo que acaba de revelar el TEST.

Los papers diarios, mensuales o con ventanas expansivas no se comparan de forma
exacta con ese protocolo comercial. Conservar dos resultados: replica del paper
y adaptacion a nuestro negocio, cada uno con su contrato.

## 7. Comparacion que produciria conocimiento util

No lanzar todas las ideas. La primera ronda propuesta reutiliza los mismos
datos y pronosticos ya aceptados y cambia un componente cada vez:

1. Heuristica corta/larga original, con contabilidad auditada.
2. Misma heuristica mas exposicion por riesgo; entradas y salidas sin cambios.
3. Mismos pronosticos con controlador MPC y costes.
4. Misma familia de politicas con/sin contexto de regimen.
5. Mismas entradas, SL/TP y tamano, cambiando solo la salida anticipada.

Estas son propuestas para el hito correspondiente, no jobs despachados hoy.
Para estudiar si el corto aporta mas que el largo, conservar primero las mismas
oportunidades de entrada: si un brazo no abre ninguna orden, su PnL cero no mide
la calidad de su politica de salida. Evaluar tanto la estrategia completa como
el efecto condicionado a entradas emparejadas, sin mezclarlos en una sola cifra.
Despues, comparar la representacion modular bajo el mismo cabezal/politica y
el mismo soporte de informacion. No atribuir a branching una mejora introducida
por otro universo de activos o un simulador diferente. Mantener R0/R1/R2 como
contrastes de inicializacion/congelacion, y NEAT en su etapa posterior.

**Baselines:** no operar/efectivo; exposicion direccional simple donde tenga
sentido; regla de tendencia o reversion; heuristica actual; controles de igual
volatilidad/exposicion. Para pronostico se conserva el naive global del periodo
sobre las mismas filas, con desglose por horizonte. No se cambia aqui la puerta
vigente frente al naive. Para una politica directa, MAE puede no estar definido:
requiere su propio contrato de utilidad, no un MAE ficticio.

**Metricas OLAP propuestas:** PnL neto y bruto, equity, retorno, Sharpe con
convencion temporal declarada, drawdown, expected shortfall, exposicion,
rotacion, operaciones, tenencia, costes por tipo, participacion de cada regimen,
calibracion de eventos, latencia de decision, rendimiento por semana y
complementariedad con las otras estrategias. Medir valor marginal en una
cartera, no solo el Sharpe individual.

Usar bloques temporales para incertidumbre; no tratar miles de ventanas
solapadas como miles de observaciones independientes. Guardar todas las variantes
intentadas, no solo ganadores. El Deflated Sharpe Ratio trata seleccion multiple
y no normalidad, pero no corrige fugas ni todos los efectos de dependencia.
[Bailey y Lopez de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

Una semilla por defecto; hasta tres solo con justificacion. Bootstrap sobre
retornos retenidos no exige reentrenar redes, pero debe respetar dependencia.
Coste de ajuste/inferencia tambien cuenta. Registrar la diferencia entre
`LITERATURE_REPORTED`, `LOCAL_REPLICATION`, `LOCAL_ADAPTATION` y `HYPOTHESIS`;
ningun resultado bibliografico entra como rendimiento medido de nuestros modelos.

## 8. Decisiones pendientes de la discusion

- Elegir una sola extension inicial cuando cierre el hito actual. Recomendacion:
  MPC con costes por su reutilizacion de los pronosticos; regimen como siguiente
  dimension, y experimento de salida separado para responder la hipotesis del
  corto plazo. Si no hay pronosticos elegibles, no se fuerza esa promocion.
- Elegir primero un instrumento y frecuencia con datos/ejecucion suficientes;
  no suponer que un hallazgo de acciones diarias resuelve FX horario.
- Definir la ventaja economica minima frente al coste y la politica de exposicion
  antes de buscar umbrales. No escogerlos mirando el mejor resultado de TEST.
- Mantener calendario causal compuesto, generacion masiva de alphas, market
  making, agentes LLM y nuevas fuentes fuera del camino critico actual.

Conclusion: hay alternativas reales a aumentar el tamano del predictor.
Las mas cercanas a nuestro sistema cambian la **decision**: gestion de costes,
seleccion de politica, tamano, abstencion y salida. Las mas independientes
cambian la fuente de ventaja: valor relativo, propagacion entre activos, carry
o informacion de eventos. No hay evidencia aqui de que una de ellas sea ya
rentable en nuestros instrumentos ni de que SAC/DQN/NEAT sean ganadores.
