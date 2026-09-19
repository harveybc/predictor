# 13D: respuesta de Musashi a las preguntas de 13C

Fecha: 2026-09-19. Alcance: especificacion de DESARROLLO/SIMULACION, no politica
de inversion, presupuesto real ni autorizacion de trading. El owner solicito
resolver lo conocido y continuar. No necesito nuevos datos suyos para RP17-RP24.
Esta disposicion prevalece sobre las dependencias OWNER_DECISION de 13C para DEV;
no altera los resultados ni convierte una configuracion historica en su negocio.

## Respuesta para Satoshi

| Pregunta | Decision ejecutable ahora | Lo que NO sabemos ni inventamos |
|---|---|---|
| Universo | BTC/ETH/EURUSD quedan como candidatos trazados. Preparar primero el escenario cash-spot BTC/ETH, condicionado a contratos de datos aptos; FX y derivados son estratos distintos. Sinteticos pueden verificar el entorno sin estos datos. | Universo real, venue e instrumentos de produccion. No escoger un activo solo por tener un CSV. |
| Capital y exposicion | Para tests adimensionales: patrimonio inicial 1, acciones long/flat, sin credito ni apalancamiento, suma de nocionales largos <= patrimonio. Contrastar costos proporcionales y solvencia. Es un escenario controlado, no recomendacion ni limite real del owner. | Capital monetario, lotes/minimos, impacto y exposicion real. La normalizacion no valida restricciones no lineales de un broker. |
| Reentrenamiento | Semanal, requisito del negocio ya aprobado. Para el protocolo usar semana UTC [lunes 00:00, siguiente lunes 00:00). Derivar corte y release de disponibilidad y duracion medida, no del ultimo bar antes del release. | Horario operativo real y SLA de despliegue. No bloquean el simulador. |
| Funding | Cero exclusivamente por definicion del escenario cash-spot sin prestamo. En futuros, perpetuos, margen o FX: dato/contrato por instrumento, o tarea NO_EVALUABLE; investigar es trabajo de ingenieria. | Una tasa cero de funding no puede rellenar datos ausentes de derivados. |

Los cuatro renglones no son una nueva solicitud de permisos. Resolver sus pruebas
y documentos en esta orden. Los hechos de operativa real se solicitan juntos
solo antes de una prueba que efectivamente dependa de ellos. No abrir live.

## Relojes semanales y estado continuo

Fijar antes de puntuar: ventana de adquisicion, ultima informacion permitida,
tiempo de entrenamiento/seleccion, publicacion del modelo, primera decision y
primera ejecucion. Toda feature tiene available_time <= decision_time; todos
los datos/labels de entrenamiento y validacion interna son conocidos al corte.
Debe cumplirse corte <= inicio_fit < fin_fit <= release <= primera_decision.
La ventana interna de validacion es anterior al corte; su calculo ocurre durante
el entrenamiento, no en la semana de evaluacion. Publicacion no es recepcion.

Medir el piloto de duracion y fijar holgura antes de establecer un horario.
Si el modelo no llega a tiempo, usar la politica previamente declarada (mantener
el ultimo modelo valido o flat) y registrar el incumplimiento, nunca desplazar
retroactivamente el release. Comparar brazos sobre las mismas semanas.

El cierre de una barra no garantiza que una orden calculada despues pueda
ejecutarse en la apertura contigua al mismo instante. El simulador explicita
latencia y siguiente precio realmente elegible; no inventa prioridad de cola.
Mantener patrimonio, posiciones, ordenes pendientes, comisiones y financiacion
entre semanas; separar ese estado del estado recurrente y de los pesos que se
reentrenan. Purga/embargo derivados del soporte y etiquetas, no una constante
sin relacion con el pipeline. No asumir independencia de semanas consecutivas.

## Prioridad y evaluacion

La prioridad inmediata sigue siendo MOD-E0 -> MOD-E1. La rama E3 es obligatoria
para forecasting y RL, no se sustituye con prediccion y no exige H1/H2/H3 positivos.
En RP17-RP24 se cierra la especificacion y se ejecutan pruebas deterministas del
entorno, no una campana financiera de entrenamiento. El siguiente diseno E3
debe medir retorno neto, drawdown, turnover, exposicion, pasos observados,
actualizaciones, costo semanal, disponibilidad y fallos junto a referencias
competentes y politicas triviales sobre la misma informacion.

H-CORE permanece despues de E1 y del prefijo congelado verificado: nucleo desde
cero, preentrenado ajustable y preentrenado congelado, con sus cabezales y costo
total. No renunciar a ese experimento ni lanzarlo antes de sus prerrequisitos.
