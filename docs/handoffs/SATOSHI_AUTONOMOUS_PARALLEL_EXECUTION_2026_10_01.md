# Satoshi: ejecucion autonoma y paralela del programa modular

Fecha: 2026-10-01. Emisor: Musashi, por instruccion expresa del Maestro.
Destinatario: Satoshi y sus agentes. Prioridad: ejecucion inmediata.

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

Satoshi orquesta a sus agentes con tareas y archivos disjuntos. Acusar cada
despacho con el identificador real del agente. Reasignar agentes al cerrar una
tarea y resolver las integraciones continuamente. No esperar un retorno global
para integrar o ejecutar una entrega independiente.

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

No responder a esta orden solicitando autorizacion para ejecutarla. Despachar,
medir, integrar y reportar; resolver los problemas tecnicos dentro del frente
afectado mientras los demas siguen trabajando.
