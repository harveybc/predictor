# Satoshi: ejecucion autonoma y paralela del programa modular

Fecha: 2026-10-01. Emisor: Musashi, por instruccion expresa del Maestro.
Destinatario: Satoshi y sus agentes. Prioridad: ejecucion inmediata.

Revision 1, 2026-10-01: incorpora el frente Kalman aprobado por el Maestro y
refuerza la reasignacion continua de agentes y recursos.

## 1. Mandato vigente

El Maestro autoriza la ejecucion autonoma de los trabajos del plan. Satoshi
planifica, asigna recursos, implementa, prueba, integra y ejecuta sin esperar
una firma, revision o permiso adicional de Musashi ni del Maestro. Esta orden
sustituye las esperas de autorizacion introducidas por ordenes anteriores para
esos trabajos. Musashi audita en paralelo; su revision no es una dependencia
de despacho. No cerrar la jornada dejando trabajo ejecutable sin responsable.

Resolver localmente las decisiones tecnicas: entornos aislados, dependencias,
pilotos, presupuestos medidos, contratos derivados de evidencia, sucesores de
disenos y colocacion de cargas. Registrar decisiones y continuar. Un dato
faltante exige obtenerlo o preparar otro trabajo independiente; no una espera
general. La autorizacion no convierte datos ausentes en conocidos ni resultados
de desarrollo en resultados confirmatorios.

## 2. Primer despacho

1. Inspeccionar procesos, reservas, colas y GPUs en vivo; adoptar trabajos
   existentes por identidad para evitar duplicados. El estado de las 07:15Z
   es una referencia historica, no una orden de reiniciar procesos.
2. Reunir los tips integrados de motor, preentrenamiento, seleccion, DOIN y
   LTS. Actualizar el master plan con un responsable y siguiente accion por
   frente. Usar worktrees separados por agente y propiedad de archivos.
3. Despachar los frentes de la seccion 3 inmediatamente. Preparar la siguiente
   celda antes de terminar la actual. Una dependencia retiene solo las tareas
   que realmente la necesitan.
4. Priorizar la 5090 externa para entrenamiento y optimizacion. Usar las otras
   GPUs con trabajos independientes cuando memoria y temperatura lo permitan.
   Las dos GPUs de un anfitrion comparten RAM: admitir su consumo agregado.
5. No terminar con un informe de implementacion si queda un experimento listo
   para correr. Lanzarlo, comprobar su primer avance y reportar su identidad.
6. Mantener un tablero de capacidad con todos los agentes y recursos. Un agente
   que termina recibe otro trabajo independiente en el mismo ciclo de control.
   Una GPU que libera una celda adopta la siguiente celda admisible ya preparada.
   Preparar siempre al menos una sucesora por GPU antes del cierre de la actual.

## 3. Frentes simultaneos

| Frente | Trabajo y primera entrega | Dependencia real |
| --- | --- | --- |
| A: motor | Integrar motor modular y donantes; verificar carga de ramas y nucleo, R0/R1/R2, early stopping, pesos seleccionados y reanudacion. Entregar configuracion ejecutable financiera. | Versiones y contratos compatibles |
| B: datos y seleccion | Ejecutar seleccion progresiva sobre fuentes disponibles, con metricas por caracteristica, transformaciones causales y utilidad respecto al target. Entregar un primer dataset financiero trazable utilizable y ampliar cobertura en paralelo. | Disponibilidad temporal demostrable de cada entrada |
| C: causalidad y extraccion | Ejecutar PS3-C/PS3-R sobre candidatos de B; evaluar aporte predictivo, estabilidad y reconstruccion. Estimar intervenciones y contrafactuales con supuestos identificadores explicitos. | Poblacion y target definidos por B |
| D: DOIN y entrenamiento | Preparar y lanzar el contraste financiero del modelo diferenciado, y mantener optimizacion del mejor candidato elegible. Contrastar R0, R1 y R2 con controles comparables. | Primer dataset utilizable de B y motor de A; no requiere terminar todo el inventario |
| E: trading y producto | Integrar modelos elegibles con prediction_provider/LTS y la estrategia heuristica existente; ejecutar replay y demo/paper, con costes y dimensionamiento vigentes. Mantener operativo M5PHET. | Modelo financiero que pase el naive pareado en los horizontes consumidos |
| F: evidencia y recursos | Mantener colas, telemetria, warehouse, comparaciones y PNG de progreso. Corregir campos obsoletos del estado. | Observar cada frente; no bloquearlos por el cierre documental |
| G: RL | Preparar y ejecutar SAC y DQN, con y sin representacion temporal diferenciada, respetando sus espacios de accion. | Dataset financiero valido, entorno probado y piloto de coste; no depende de un ganador supervisado |
| H: Kalman causal | Implementar y evaluar filtros de estado como transformaciones por caracteristica y como entradas adicionales. Ejecutar en CPU tres brazos pareados y entregar coste, causalidad, paridad batch/incremental y efecto predictivo. | Primer subconjunto financiero trazable de B; no depende de cerrar toda la seleccion |

Satoshi orquesta a sus agentes con tareas y archivos disjuntos. Acusar cada
despacho con el identificador real del agente. Reasignar agentes al cerrar una
tarea y resolver las integraciones continuamente. No esperar un retorno global
para integrar o ejecutar una entrega independiente.

Asignacion inicial sugerida, ajustable por Satoshi segun coste medido:

- agente 1: A, integracion y configuracion financiera ejecutable;
- agente 2: B, primer dataset financiero y seleccion progresiva;
- agente 3: H, operador Kalman causal, pruebas y piloto CPU;
- agente 4: D, manifiesto y runner del contraste financiero;
- siguientes agentes disponibles: C, G, E y F en ese orden de dependencia;
- 5090 externa: D y optimizacion DOIN del mejor candidato vigente;
- 4090 y 5070 Ti: R0/R1/R2, preentrenamientos o RL independientes que quepan;
- 4070: verificacion, inferencia o piloto pequeno, conservando margen para el
  escritorio. CPU: inventario, seleccion, causalidad, Kalman, ETL y cierres.

Esta asignacion no obliga a esperar cuatro cierres. En cuanto A o B entregue el
primer artefacto consumible, D prepara y lanza su primera celda. Las ampliaciones
de cobertura siguen en paralelo. Si una tarea no cabe en su recurso, el agente
prepara la sucesora o toma otro frente mientras el orquestador la recoloca.

## 4. Metodo diferenciado y campana financiera

Conservar la arquitectura solicitada: ramas configurables por caracteristica
o grupo, salida temporal alineada, concatenacion por canales, codificacion
posicional, bloques Transformer y reduccion progresiva del nucleo. Mantener
plugins jerarquicos y compatibilidad de configuraciones previas. No sustituir
esta arquitectura por un MLP ni colapsar el tiempo antes de fusionar.

Preentrenar extractores y nucleo con early stopping propio. Entrenar el nucleo
sobre representaciones producidas por las ramas y fusionador vinculados a sus
pesos. R0 inicia sin donante; R1 carga y congela los componentes declarados;
R2 carga los mismos donantes y permite ajustarlos. Registrar costes del
preentrenamiento y comprobar realmente que los pesos congelados no cambian.

Congelar antes de medir el primer contraste financiero: fuente, target,
horizontes, filas, particiones, transformaciones ajustadas en train, modelos,
semillas, presupuesto y regla de seleccion. Satoshi realiza este paso y lanza
sin esperar firma externa. Una reserva previamente consultada es desarrollo;
usar una reserva limpia o prospectiva para confirmacion.

Usar datos ya disponibles de los servicios contratados y del lago cuando sean
adecuados. Incorporar Yahoo Finance, Alpaca y FXMacroData al inventario real,
incluyendo lo disponible por sus planes; no prometer cobertura no descargada.
Si falta metadata, derivarla de evidencia de la fuente cuando sea posible;
excluir solo la entrada cuyo tiempo de disponibilidad no pueda establecerse.

En el contraste, comparar arquitectura base y diferenciada con el mismo
target, filas y presupuesto declarado. Mantener el carril de replicas de la
literatura separado y emparejado con sus protocolos publicados. Probar Huber
y MAE con Adam/AdamW segun el subplan financiero, con busqueda comparable.

## 4.1 Kalman causal aprobado

Implementar Kalman como familia de transformaciones configurables, no como un
reemplazo obligatorio de la entrada ni como una afirmacion de mejora. Reutilizar
el contrato de operadores causal existente: `fit`, `transform`, continuacion
incremental, estado durable, warm-up tipado, coste, latencia y cero acceso futuro.

Primera familia acotada:

- modelo de nivel local;
- modelo de nivel y pendiente;
- opcion multivariada solo despues de que los dos anteriores tengan piloto y
  coste, empezando por grupos pequenos justificados por seleccion;
- parametros Q/R estimados exclusivamente en train o suministrados por una
  configuracion predeclarada; estado inicial y covarianza registrados;
- filtro forward causal como camino admisible. El smoother retrospectivo queda
  como control no causal y jamas produce una entrada elegible.

Por cada caracteristica conservar, segun aplique: observacion original, estado
filtrado, pendiente, innovacion, innovacion estandarizada y covarianza o
incertidumbre del estado. Registrar unidades, escala, timestamp de disponibilidad,
warm-up y digesto del estado ajustado. No llamar probabilidad de acierto a la
covarianza del filtro.

Ejecutar tres brazos sobre las mismas filas, targets, semillas y presupuesto:

1. entrada original;
2. original mas salidas Kalman;
3. salidas Kalman reemplazando solo las caracteristicas declaradas.

Incluir controles de identidad y de destruccion de senal: original sin cambio,
EWMA causal comparable, smoother deliberadamente no causal como rechazo, ruido
o permutacion con capacidad equivalente, y escenarios sinteticos con estado
conocido. Medir MAE/MSE normalizado, skill frente al naive pareado por horizonte,
preservacion de extremos, fase/retardo, estabilidad de la innovacion, coste y
memoria. La seleccion se decide por utilidad predictiva fuera de train y
estabilidad; suavidad o reconstruccion por si solas no seleccionan una entrada.

El `local_level_kalman` historico ajustado por MLE no es portable entre todos los
CPU observados. No usarlo para una decision gobernante. Crear un sucesor con
algoritmo determinista: presupuesto de iteraciones y tolerancia fijos, o estimador
cerrado cuando exista; float64, un hilo BLAS y entorno registrado. Probar paridad
batch contra actualizacion tick a tick, reinicio desde estado durable y replay en
los roles disponibles. Si no reproduce, queda exploratorio y los otros frentes
continuan.

Integrar los parametros Kalman en el espacio jerarquico de DOIN solo despues del
piloto. Condicionar sus parametros a que el operador este activo para evitar un
producto cartesiano inutil. Promover primero la mejor variante acotada y medirla
en el modelo diferenciado; ampliar por caracteristica o grupo solo cuando el
efecto y el coste lo justifiquen.

## 5. Recursos y continuidad

La autorizacion de ejecucion ya esta concedida. Los limites de memoria y
temperatura son comprobaciones tecnicas automaticas, no solicitudes de permiso.
Resolver una negativa de admision mediante colocacion, liberacion comprobada
de recursos propios terminados, reduccion de concurrencia o un sucesor medido;
no modificar el experimento silenciosamente ni provocar otro OOM del escritorio.

Reservar RAM y CPU para el escritorio del coordinador. Usar preferentemente
los obreros para cargas pesadas. No ocupar una GPU con trabajo artificial para
mejorar el indicador de utilizacion. Si esta libre, despachar el siguiente
trabajo util que quepa; si ninguno cabe, asignar inmediatamente la reparacion
concreta y reportar dependencia, responsable y siguiente comprobacion.

No duplicar entrenamientos para cerrar informes. Reutilizar checkpoints y
artefactos verificables. Operacion de brokers bajo el mandato de riesgo ya
existente; no inventar limites de riesgo ni ampliar exposicion por esta orden.

## 6. Resultados y reportes obligatorios

- Primer parte tras el despacho: trabajos realmente iniciados, agente,
  anfitrion, UUID de GPU, pid/servicio, dataset y etapa observada.
- Heartbeat por trabajo al menos cada dos minutos: celda, epoca/actualizacion,
  tiempo, memoria y ultimo avance. Un PID vivo sin progreso no basta.
- Parte cada 30 minutos y en cada cierre o fallo: resultado nuevo, colas,
  recursos libres, dependencia concreta y accion tomada autonomamente.
- El parte de recursos muestra todos los agentes, CPUs y GPUs: trabajo actual,
  siguiente trabajo preparado y causa tecnica medible cuando un recurso este
  libre. `IDLE` sin sucesor asignado es un defecto de orquestacion a corregir en
  ese mismo ciclo, salvo margen reservado explicitamente para el escritorio.
- ETA en intervalo, calculado con tiempos observados de celdas comparables.
  Antes del primer piloto, indicar la hora de su proxima medicion. No inventar
  porcentajes: mostrar completadas/planificadas por frente y alcance versionado.
- Cada MAE/MSE debe incluir escala, poblacion, horizonte y naive calculado sobre
  las mismas filas. Mostrar referencia publicada solo cuando sea comparable.
  No ejecutar la estrategia con predictores que incumplan la puerta naive
  establecida para los horizontes que consume.
- Entregar RETURN.md, STATUS.json, RESULTS.csv/JSON y PROGRESS.png, enlazados
  al master plan y generados desde evidencia. El PNG debe mostrar hitos,
  porcentajes con denominador, trabajo actual, siguientes dependencias y ETA.
- Publicar los commits de entrega con tips verificables sin incorporar cambios
  ajenos. El informe distingue implementado, ejecutado y verificado.

## 7. Punto de partida y correcciones del estado

El estado retenido de M06 a las 07:15Z informa campana ECL cerrada: 32 celdas
verificadas y 4 rechazadas por el motor. No reutilizar mensajes obsoletos que
todavia anuncian celdas pendientes de esa campana.

Sobre validacion ECL, el estado informa MAE_z de R0 0.383565 y R1 0.375091,
persistencia 0.851406 y estacional 24 h 0.247966. Estos valores no son evidencia
financiera ni comparacion con un protocolo publicado distinto. R1 mejora el
promedio R0; ambos pierden contra la referencia estacional. Conservar todos
los resultados y sus limites sin declarar un ganador de trading.

La matriz de perfiles no equivale a seleccion completada. Verificar el ledger
actual de evaluaciones y seleccion y actualizar sus denominadores. Dar prioridad
a cerrar el primer recorrido financiero completo, ampliando despues la cobertura
del inventario mediante el subplan progresivo ya solicitado.

Kalman pasa a formar parte de ese recorrido progresivo como candidato causal de
CPU. No reabrir toda la confirmacion historica D2 ni detener A-G para completarla;
usar la politica de portabilidad existente y producir el sucesor determinista
descrito en 4.1.

No responder a esta orden solicitando autorizacion para ejecutarla. Despachar,
medir, integrar y reportar; resolver los problemas tecnicos dentro del frente
afectado mientras los demas siguen trabajando.
