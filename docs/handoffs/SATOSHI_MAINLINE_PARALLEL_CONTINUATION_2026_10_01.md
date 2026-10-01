# Satoshi: continuacion paralela sin desviacion del plan principal

Fecha: 2026-10-01. Emisor: Musashi, por instruccion del Maestro.
Destinatario: Satoshi y agentes delegados. Ejecucion inmediata.

Esta orden es incremental a
`SATOSHI_AUTONOMOUS_PARALLEL_EXECUTION_2026_10_01.md`. No crea otro programa,
no reinicia trabajos vivos y no sustituye los planes de cada frente. Su objeto
es mantener una sola ruta critica de negocio mientras todos los subplanes
independientes avanzan en paralelo.

## 1. Estado adoptado, no reiniciar

Adoptar por identidad el estado verificado a las 09:50Z:

- M04/D2 tiene runner durable activo y la 5090 ya tomo
  `m04-train-4ea6cbed`; la celda ECL v2 fue admitida tras liberarse el lease.
- El DQN semilla 202 termino y libero su lease. La retirada de la cola G en
  `worker_a` se conserva; no relanzar esa celda ni duplicar su resultado.
- M07 y lane G siguen vivos en `worker_b`; sus relevos durables ya estan armados.
- La 4090 ejecuta M07 y un DQN modular bajo vigilancia agregada de RAM.
- La 5070 Ti esta libre. Su siguiente trabajo declarado es una celda per-feature
  de M07 cuando cierre el piloto de la 4090. No reducir silenciosamente buffer o
  cambiar configuracion RL para hacerla caber: cualquier sucesor asi es un nuevo
  diseno y la GPU debe tomar entretanto otro trabajo util admisible.
- El handover v1 que se reconocia a si mismo esta retirado. Solo v2 puede
  gobernar relevos y cada STOP debe nombrar exactamente el runner objetivo.
- Kalman H1, causalidad C2, PS5/R3, seleccion y evidencia siguen como carriles
  activos de CPU. No esperar un retorno global para integrar sus entregas.

Antes de cualquier despacho, releer procesos, leases, unidades y heartbeats.
Este corte es una identidad inicial, no una autorizacion para duplicar trabajo.

## 2. Ruta critica unica

El resultado de negocio buscado es este, en este orden causal:

1. dataset financiero point-in-time con targets de corto y largo plazo;
2. seleccion progresiva y representacion temporal diferenciada consumible;
3. entrenamiento y optimizacion DOIN de R0/R1/R2 con arquitectura modular;
4. puerta de naive por horizonte y semilla;
5. replay de estrategia heuristica y contraste RL SAC/DQN;
6. MT5 demo y Alpaca paper con el mejor candidato elegible;
7. sustitucion progresiva del modelo desplegado solo por evidencia mejor.

Todo frente debe declarar a cual de esos siete pasos aporta. Si no aporta a uno,
es soporte o investigacion secundaria y no desplaza recursos de la ruta critica.

ECL, Traffic, Weather y clasificacion conservan sus funciones: comparabilidad
con literatura, pruebas del motor y controles metodologicos. No son el producto
final y no pueden consumir indefinidamente la GPU preferida mientras exista una
celda financiera ejecutable.

## 3. Prioridad y uso de recursos

### 3.1 GPU

1. **RTX 5090 externa:** terminar las cuatro celdas activas de ECL v2 ya
   comprometidas. Inmediatamente despues, prioridad al mejor experimento
   financiero modular ejecutable y a su optimizacion DOIN. Las 16 celdas ECL
   retenidas solo entran cuando tengan donantes/pilotos y no desplacen una celda
   financiera lista.
2. **RTX 4090:** M07 financiero primero; RL en paralelo solo mientras el consumo
   agregado siga dentro de la admision medida. Al cerrar cada hijo, el relevo
   toma la siguiente celda financiera o RL ya preparada.
3. **RTX 5070 Ti:** termina el DQN vivo. Despues queda libre para permitir la
   admision inmediata de la 5090. Solo recibe otro trabajo cuando la suma de RAM
   del anfitrion no retrase a la 5090.
4. **RTX 4070:** margen del escritorio. Solo inferencia, verificacion o pilotos
   pequenos que no comprometan la sesion interactiva.

Una GPU libre debe tener `next_task` y hora de proxima admision. No ejecutar
trabajo artificial para ocultar inactividad. Si una celda no cabe, recolocarla
o ejecutar otra celda util; la negativa tecnica de una no detiene las demas.

### 3.2 CPU y agentes

Satoshi mantiene integracion y decisiones de arquitectura. Puede delegar en
Hermes/OpenCode tareas acotadas con write sets disjuntos: pruebas, ETL, perfiles,
tablas, documentacion, fixtures y adaptadores. Un agente delegado no decide la
ciencia, no cambia denominadores y no integra su propio trabajo sin revision de
Satoshi.

Asignacion continua sugerida:

- A: PS5/R3, motor modular, carga de donantes y paridad R0/R1/R2;
- B: inventario financiero, disponibilidad, metricas y seleccion progresiva;
- C: escalera causal, PS3-C/PS3-R y calibracion de controles;
- D: runners DOIN y contraste supervisado ECL/finanzas;
- E: prediction provider, LTS, estrategia y paper trading;
- F: warehouse, STATUS, ETA y evidencia generada;
- G: SAC/DQN con y sin representacion diferenciada;
- H: Kalman causal y controles;
- I: M5PHET usable, sin quitar recursos a A-H.

Cuando un agente termina, recibe el siguiente item independiente del mismo
frente o del backlog. No mantener agentes esperando un merge global.

## 4. Entregables por frente

### A. Arquitectura modular y preentrenamiento

- Finalizar PS5/R3 y publicar el primer resultado medido, no solo tests.
- Mantener ramas configurables por rasgo o grupo, dimension temporal alineada,
  fusion por canales, codificacion posicional, Transformer residual y reduccion
  temporal progresiva sin `Flatten` o `Dense` que colapse el tiempo antes del
  cabezal.
- Verificar extractores y nucleo preentrenables por separado, early stopping,
  restauracion del mejor checkpoint y tres regimenes R0/R1/R2.
- R1 debe demostrar pesos congelados bit a bit; R2 parte de los mismos donantes
  y demuestra actualizacion. Reportar costo de donantes separado del ajuste.
- Conservar compatibilidad con configuraciones y entry points anteriores.

### B. Datos y seleccion progresiva

- Producir primero un dataset financiero consumible, no esperar el inventario
  completo. Targets de corto y largo plazo, splits temporales y disponibilidad
  point-in-time deben quedar ligados por digest.
- Inventariar Yahoo Finance, Alpaca, FXMacroData y fuentes existentes sin omitir
  campos utilizables de los planes contratados. Incluir features originales,
  indicadores tecnicos, regimen, calendario y transformaciones causales
  candidatas: retornos, wavelet, multitaper, Hilbert, STL y Kalman.
- Controlar explosion de variables en etapas: filtro semantico/disponibilidad,
  metricas baratas, redundancia/clustering, utilidad predictiva, escalera causal
  y capacidad de extraccion. No materializar el producto cartesiano completo.
- Entregar por iteracion: `eligible`, `rejected` y `pending`, cada uno con razon,
  target, horizonte, filas y coste. La reconstruccion de un autoencoder es un
  criterio auxiliar, nunca seleccion automatica.

### C. Causalidad y extraccion

- Cerrar la calibracion actual y convertirla en una regla aplicable al primer
  dataset financiero. Separar asociacion, intervenciones naturales observadas y
  contrafactuales emparejados; explicitar supuestos y soporte.
- No llamar seleccion terminada al piloto PS3-R. Ampliar solo sobre candidatos
  que sobrevivan B y comparar original, transformada y control destruido.
- Toda salida causal conserva event time, availability time y poblacion.

### D. Entrenamiento supervisado y DOIN

- Adoptar y cerrar las cuatro celdas ECL v2 activas. Reportar el incumbente
  provisional contra naive estacional sobre las mismas filas.
- Lanzar despues el primer factorial financiero modular completo: base fuerte,
  diferenciado R0, R1 y R2; MAE/Huber por Adam/AdamW bajo presupuesto comparable.
- DOIN mantiene optimizacion del mejor candidato financiero elegible mientras
  CPU ejecuta literatura y controles. No usar MLP plano como sustituto del
  modelo diferenciado; solo como control declarado.
- Cada resultado incluye MAE/MSE normalizado, naive por horizonte en las mismas
  filas, skill, semillas, coste, parametros y referencia comparable si existe.

### E. Trading y producto

- Construir y probar ahora las interfaces, contratos, replay, sizing, costes,
  reanudacion y observabilidad de MT5 demo y Alpaca paper; esto no necesita
  esperar al ganador.
- No enviar a estrategia un predictor que no supere el naive pareado en todos
  los horizontes que esa estrategia consume. ETH R0 actual, 0/12, no es elegible.
- En cuanto exista candidato elegible, ejecutar estrategia heuristica con corto
  y largo, luego ablaciones corto solo, largo solo y horizontes individuales.
- Operacion real usa exclusivamente el mandato de riesgo ya existente. No
  aumentar exposicion ni sustituir paper por dinero real por esta orden.

### F. Evidencia, warehouse y seguimiento

- Mantener runners durables, pero el mantenimiento no cuenta como resultado
  cientifico ni puede ocupar el frente principal una vez estable.
- Reparar `PROJECT_METHOD_STATE.json` y el encabezado del master plan: hoy aun
  describen RP140-RP143. El estado vigente debe enlazar esta orden, los subplanes
  y los tips activos sin reescribir la historia.
- Actualizar `STATUS.json`, `RESULTS`, `PROGRESS.png` y warehouse desde artefactos,
  no desde texto narrativo. Separar implementado, ejecutado y verificado.

### G. RL

- Terminar las 16 celdas actuales y ejecutar los cuatro brazos emparejados:
  DQN y SAC, cada uno con representacion convencional y diferenciada.
- Usar la misma poblacion, costes, acciones y seeds por contraste. Reportar
  retorno, Sharpe con periodicidad, drawdown, trades, exposicion, turnover,
  costes y baselines. Una celda aislada no elige politica.
- Integrar la seleccion B y los pesos de A cuando esten disponibles, sin detener
  los brazos que ya pueden ejecutarse limpiamente.

### H. Kalman

- Continuar el sucesor determinista ya publicado. Entregar paridad batch/tick,
  replay tras reinicio, controles EWMA y smoother no causal rechazado.
- Ejecutar los tres brazos acordados sobre el primer subconjunto financiero:
  original, original+Kalman y reemplazo declarado. Solo despues incorporar sus
  parametros condicionales al espacio DOIN.

### I. M5PHET

- Mantener el producto usable y avanzar un adaptador real por turno de CPU
  disponible. No afirmar un area por contrato solamente: debe tener proveedor
  real, entrada natural/estructurada, salida tipada, persistencia y prueba E2E.
- No desplazar entrenamiento financiero, seleccion o trading para ampliar UI.

## 5. Reglas contra desviacion

- Ningun frente nuevo se crea sin mapearlo a A-I y al paso 1-7 de la ruta
  critica. Una idea nueva entra al backlog; no interrumpe procesos sanos.
- Reparaciones operativas tienen presupuesto acotado y un criterio de salida.
  Tras estabilizar el runner, el agente vuelve a ciencia o producto.
- No reconstruir resultados existentes si sus artefactos verificables bastan.
- No usar datos insuficientes para declarar ganadores. Un piloto puede decidir
  coste o siguiente medicion, no confirmar utilidad.
- No cambiar targets, splits, horizonte, escala, metricas o presupuesto dentro
  de un contraste. Un sucesor distinto recibe identidad nueva.
- No esperar autorizaciones adicionales para los trabajos aqui ordenados.

## 6. Reporte requerido

Emitir un parte cada 30 minutos y ante cada cierre/fallo. Formato obligatorio:

1. **Resultado nuevo primero:** cifra, escala, poblacion, naive pareado,
   referencia y clase de evidencia.
2. **Ruta critica 1-7:** completado/total por paso y siguiente objeto concreto.
3. **Frentes A-I:** estado, agente, tip, trabajo actual y siguiente tarea.
4. **Recursos:** cada GPU y host con UUID, proceso, celda, memoria, temperatura,
   progreso observado, ETA y sucesor preparado.
5. **Colas:** running/queued/held/failed/verified con denominador versionado.
6. **ETA:** intervalo basado en celdas comparables; si no existe muestra,
   indicar hora de la primera medicion que permitira estimarlo.
7. **Desviaciones:** trabajo retirado, duplicado evitado y toda correccion propia.

El proximo retorno consolidado debe incluir:

- `RETURN.md` con decisiones y resultados;
- `STATUS.json` generado;
- `RESULTS.csv` o JSON con naives y referencias;
- `PROGRESS.png` con ruta critica, A-I, recursos y ETA;
- tabla de elegibilidad para estrategia y despliegue;
- peticiones al Maestro solo cuando exista una accion fisica o credencial que
  no pueda realizar Satoshi. Una decision tecnica ordinaria se resuelve y se
  registra, no se devuelve como permiso pendiente.

## 7. Accion inmediata

1. Verificar progreso real de `m04-train-4ea6cbed` en la 5090 y conservar su
   relevo durable para las tres celdas D2 restantes.
2. Admitir en la 5070 Ti la primera celda util que quepa sin retrasar la 5090;
   preferir M07 per-feature cuando su piloto cierre y preparar ahora su sucesora.
3. Mantener M07 y RL en la 4090 sin superar la admision agregada.
4. Asignar agentes CPU a A, B, C y H; usar Hermes para tareas acotadas.
5. Preparar antes del cierre ECL la primera celda financiera modular que tomara
   la 5090 despues de D2.
6. Reconciliar el master plan y `PROJECT_METHOD_STATE.json` con esta realidad.
7. Reportar el primer resultado nuevo y la siguiente asignacion; no esperar un
   cierre global.
