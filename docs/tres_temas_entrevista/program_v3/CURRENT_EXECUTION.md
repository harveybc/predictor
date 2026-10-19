# Ejecucion vigente

Actualizado2026-10-19. Autoridad: master v3 y RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md.
Ninguna orden ni donante de las campanas predictivas retiradas puede despachar
trabajo. El propietario autoriza continuar automaticamente con datos admitidos.

## Continuacion del 19 de octubre

Ejecucion real iniciada 2026-10-19T07:20Z: contrato local `829e6959...`,
12 tareas secuenciales, primer control NO_TRADE recorriendo TRAIN.
Servicios persistentes y publicador con timer instalados; lingering habilitado.
Sin aprobacion pendiente. Runtime agent-multi `b211624a`, gym-fx `e8710fe`.
Pruebas: 295 de simulador y 85 de integracion/cola, sin omisiones.
SAC tiene la 5090 asignada para cuando llegue su turno; no se afirma que ya
este entrenando mientras se ejecutan los controles CPU. El estado generado,
no este corte, determina la tarea actual.

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
