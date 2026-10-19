# Ejecucion vigente

Actualizado2026-10-19. Autoridad: master v3 y RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md.
Ninguna orden ni donante de las campanas predictivas retiradas puede despachar
trabajo. El propietario autoriza continuar automaticamente con datos admitidos.

## Continuacion del 19 de octubre

Actualizacion por orden del propietario: nuevos ensayos del SAC temporal
PAUSADOS. Su TRAIN y VALIDATION de referencia ya terminaron; no se detuvo
ningun ajuste vivo ni se modifico la campana. Los controles siguen corriendo
y el sucesor de early stopping del SAC directo sigue armado. No se han
ejecutado pruebas cientificas de BatchNorm, dropout ni weight decay.

Siguiente trabajo aprobado, ya desplegado: [early stopping del SAC directo](RL_PAIRED_EARLY_STOP_2026_10_19.md),
contratoac792153, esperando cierre verificado del contraste37e06192.
Arranque en5090, evaluacion y publicacion por checkpoint, cierre y segunda copia
estan automatizados. No usa BatchNorm ni dropout. Esto NO es un fit ya iniciado;
el estado actual es WAIT_PREDECESSOR. El directo de referencia ya completo
TRAIN y VALIDATION: +351.12% frente a-0.19%, respectivamente; ver subplan.

Corte mas reciente: [15 variables seleccionadas y contraste SAC activo](SELECTED_CORE_RL_2026_10_19.md).
La seleccion acotada del nucleo EURUSD esta cerrada:30 candidatas,435 pares,
nueve filtros,15 variables,37198 filas nativas OLAP y dos copias verificadas.
El catalogo externo y la identificacion causal completa siguen pendientes.
La cola seleccionada con apalancamiento1:50 arranco10:50:19Z,
SAC directo en5090. Temporal sigue automaticamente con identicas15 entradas.
Contrato37e06192, seleccion553fc047. Cada entrada exige SL/TP y presupuesto
fraccion_elegida * rel_volume * balance(t), rel_volume0.1; apalancamiento1:50.
El intento1:1 incompleto143051b1 se detuvo sin reutilizar pesos al recibir
la nueva prioridad del propietario.1:100 configurable;1:1 queda para despues.
El intento incompleto41c6acf3 se detuvo sin reutilizar pesos porque usaba
fraccion de cartera en lugar del presupuesto relativo aclarado por el usuario.
No hay resultados nuevos de politicas aun.
Los parrafos siguientes sobre seleccion pendiente describen el catalogo
completo y el corte anterior, no niegan este nuevo subconjunto cerrado.

El propietario reitera que la seleccion debe alimentar el contraste RL final.
No existe aun manifiesto financiero seleccionado sobre toda la poblacion
reparada. El ensayo RAW en curso no satisface ese hito. No abrir nuevas
campanas RAW antes de cerrar la seleccion; ver subplan RL seccion8.
Control RAW cerrado: 12/12 tareas, 12/12 informes con readback exacto, cero
pendientes/discrepancias y copia fisica en dos maquinas verificada por hashes.
SAC plano: TRAIN2020-2023 +699.66 USD, VALIDATION2024 -366.67 USD;
SAC temporal: TRAIN +2275.59 USD, VALIDATION -496.18 USD. Pesos congelados en
VALIDATION, maximo una entrada por dia y todos los rollouts terminan sin posicion.
Resultados y limites: [cierre RAW](RL_RAW_CONTROL_RESULT_2026_10_19.md).
No queda ningun fit de esta campana pendiente; la seleccion financiera SI.
No atribuir toda diferencia a sobreajuste sin separar cambio de regimen.
El control actual NO incorpora dropout ni early stopping: presupuesto fijo.

El intento `c6d31d3a...` fallo tras 6000 pasos por ordenes protectoras
Submitted cuya cancelacion nativa era ignorada. No produjo modelo aceptado.
Corregidos cancelacion, hermanos OCO y limpieza reentrante de brackets;
la prohibicion de cruzar cierres con exposicion no cambia. Revision
independiente aceptada y regresiones retenidas. La evidencia de reapertura
calcula los mismos conteos/racha final sin recorrer todo el historial en cada
barra: replay de observaciones/recompensa/equity/posicion/ordenes bit-identico.
297 pruebas del simulador, 65 de integracion y 72 de cola/publicacion/copia
pasan sin omisiones. Replay aleatorio completo: TRAIN 24894 barras en 264 s;
VALIDATION 6221 barras, sin fallo. No son metricas de una politica entrenada.

Relanzado 2026-10-19T08:10:33Z con contrato `b195273e...`, 12 tareas,
ambos SAC de 100000 pasos, modelo y datos sin cambios. Primeras 1000
transiciones reales verificadas; la cola mantiene una sola carga en la 5090.
Servicios persistentes y publicador con timer instalados; lingering habilitado.
Sin aprobacion pendiente. Runtime agent-multi `75f24688`, gym-fx `461b594`.
El orden corregido pone ambos SAC antes de los controles CPU. El primer
control parcial se cancelo por programacion, sin modelos SAC entrenados ni
terminales publicados; no se presenta como resultado. Datos, modelos, seed,
costes y presupuestos se compararon y son identicos. Se corrigio ademas una
colision operativa: el EnvironmentFile del warehouse ocultaba CUDA; el servicio
fija ahora la 5090 al ejecutar el proceso sin cambiar los secretos ni el modelo.
El estado generado, no este corte, determina el progreso actual.

- EURUSD: admision CORE_PRICES_ONLY completa, 37309 horas observadas, los
  72 meses 2020-2025 y cero horas requeridas ausentes bajo el calendario
  operativo conservador declarado. Sin interpolacion. 37306 barras tienen
  linaje exacto de dos barras M30; tres son reemplazos nativos verificados.
  El calendario no se presenta como certificacion universal del broker.
- HistData: reloj de los seis archivos 2023-2025 contrastado contra FXCM y
  testigos independientes; 32326 horas completas normalizadas. Consolidacion
  terminada. Cuatro informes nuevos publicados con readback exacto y copia
  fisica coincidente en dos maquinas; cero pendientes/discrepancias.
- El control bruto EURUSD puede avanzar independientemente del inventario
  completo: SAC plano y Conv1D compacto, sin reajuste, TRAIN2020-2023 y
  VALIDATION2024, mas controles sin operar/largo/corto/aleatorio. La politica
  RAW_BASELINE se identifica expresamente: no es una seleccion final simulada.
- Antes del fit: aceptar la ruta real de hora de disponibilidad/fill y sus
  cierres de sesion. La cola es secuencial, con estado durable y continuacion
  automatica; una sola carga pesada en la 5090. Costes no nulos declarados como
  escenario, no falsamente verificados con el broker. TEST no se usa para ajustar.
- Ruta real aceptada: reloj de disponibilidad/fill/cuota y fronteras de cierre,
  incluidas reproducciones reales de semana/mes. Copia fisica del nucleo y de
  sus metricas publicada; el nuevo informe de admision tambien tiene readback
  exacto y copia coincidente en dos maquinas. El cierre RL exige sus propios
  12 terminales, readback y backup, no hereda el cierre del nucleo.
- El resto del inventario, la seleccion financiera recalculada y su evidencia
  causal siguen pendientes. GBPUSD no esta admitido por esta entrega.

Las cifras de las secciones siguientes son cortes anteriores, no el estado
vigente del nucleo EURUSD descrito arriba.

## Ahora

- Recuperacion concreta: descargadas 40 horas nativas faltantes por par del
  archivo FXCM 2019/53, correspondientes a enero de 2020. La preparacion del
  parche termino: 40 filas por par, hashes y precios conservados, 25 pruebas.
  Es independiente del lago y no declara admision general.
- Cuatro ZIP anuales nuevos de HistData descargados sin fallos: EURUSD y GBPUSD
  2024/2025, 6243/6225 horas candidatas por par. Ya existen localmente; falta
  consolidar la regla de reloj y verificar los reemplazos antes de incorporarlos.
- HistData 2023: replay de los ZIP originales, 1539 comparaciones diarias
  emparejadas. El conflicto horario sigue las fechas DST europeas. Evidencia
  publicada para 2023-2025: 6 informes, 22890 metricas, cero pendientes/discrepancias y copias
  fisicas coincidentes en dos maquinas. No se ha elegido un reloj a escondidas.
- Doce consultas acotadas a archivos de minutos FXCM de las semanas faltantes
  2024-2025 devolvieron 404; recibos guardados, sin bucle de reintentos inutiles.
- Selector @2e90ee2: adaptador secuencial/reanudable probado con los motores
  existentes de perfiles, pares y nueve filtros. Publicacion local DuckDB y
  readback verificados sobre datos sinteticos; no es seleccion financiera final,
  no calcula aun evidencia causal y falta conectar publicacion remota.
- Datos: recuperacion nativa2020-2025 y candidatos alternativos. Forexite declara
  CET/CEST en su pagina oficial: reloj verificado, 86 archivos/206858 filas
  normalizados sin interpolacion. Faltan quote side, etiqueta temporal de minuto,
  calendario real y admision de todo el inventario requerido.
- Nueva calidad:4 informes,1122 metricas y lectura de vuelta; sin discrepancias
  ni pendientes en la lectura de cierre de esta sesion. Copias fisicas y espejo
  verificados. Esto no completa seleccion ni certifica los datos.
- Evidencia nueva de reloj: 1 informe adicional, 6 metricas, lectura de vuelta
  sin pendientes/discrepancias y copias fisicas verificadas en dos maquinas.
- Detalle y status vivo: procedimiento17 de financial-data y su CLI
  `_scripts/native_recovery_status.py`; las cifras de este texto son un corte,
  no sustituyen el estado generado.
- Retornos anteriores de Satoshi integrados y revisados en
  SATOSHI_RETURN_REVIEW_2026_10_18.md. El propietario informa trabajo adicional
  de Satoshi en datos; no se presume completado ni se sobrescribe.
- Musashi ejecuta RL_EXECUTION_REPAIR_PLAN_2026_10_18.md: corregir primero la
  transicion real accion/fill/recompensa, luego costos y reglas del negocio.
- Corte de ingenieria: transicion next_open_v2, comisiones y capital de cierre,
  financiacion analitica explicita y seleccion sin NaN->0 corregidos; 256 pruebas
  de gym-fx, 65 de agent-multi y 7 del verificador pasan. No son fits financieros.
  Corregido tambien el ciclo nativo Trade tras un cierre directo: ya no pierde
  el registro de la siguiente operacion. Nueve regresiones nuevas pasan.
  Pendientes: admision/seleccion, costos reales y aceptacion completa del entorno.

## Siguiente

Datos admitidos -> preprocesamiento verificado -> seleccion recalculada ->
contrato/entorno RL -> baselines de negocio -> SAC compacto/modular ->
comparaciones RL -> DEMO/PAPER. Predictores y NEAT al final.

El insight de posible utilidad del reajuste queda como hipotesis cualitativa,
no como un resultado numerico conservado. Aumentar la frecuencia no garantiza
mejora. Separar inferencia/decision, puntuacion y actualizacion posterior.

No usar el naive de precios como compuerta RL: evaluar PnL neto, drawdown,
Sharpe, trades, exposicion, costes y reglas de negocio frente a controles
emparejados. No se afirma que RL vaya a solucionar por si solo la rentabilidad.
