# Estado de ejecución vigente

Observado: 2026-10-11. Este archivo sustituye estados operativos anteriores.

Orden vigente posterior: I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md. Satoshi recibe
cinco carriles de implementacion paralelos (trayectorias, swarm, semanas/datos,
archivo/OLAP, busqueda/verificacion); despues los scripts continuan etapas sin
agentes monitoreando. El paquete esta preparado, no hay acuse de despacho aun.
No afirmar que el swarm o el archivo externo estan activos por tener la orden.
La opcion de usar 5070 Ti simultanea requiere admision agregada medida y
reserva para 5090; no se fuerza concurrencia ni se espera un OOM para decidir.

Actualizacion posterior al corte, 21:54 UTC: las busquedas endpoint h12/h72
terminaron antes del corte, 24/24 candidatos cada una, cero fallos. No hay
entrenamientos activos ni ajustes interrumpidos pendientes en estas colas.
Lectura nueva del OLAP: 51 terminales, 396 informes y 3,015 filas de metricas
verificados contra los archivos retenidos. El candidato elegido por fitness
TRAIN/inner tiene error relativo inner +0.4506788% en h12 y -6.8473431% en h72;
son 310 y 70 observaciones internas, no evaluacion anual independiente.
El preferido y el coordinador reiniciaron; el secundario no (casi 13 dias).
Se recomienda reiniciar el secundario por errores NVIDIA NV_ERR_NO_MEMORY
recientes y memoria kernel no recuperable elevada, pese a pasar una operacion
GPU pequena. No se recomienda otro reinicio en los otros dos hosts.
Ver `docs/i6n/POST_POWER_RETURN_20261011.md` en la rama de integracion.
Pendientes: trayectorias, swarm DOIN y relevo durable, evaluacion semanal
completa. La actualizacion de las 18:01 UTC abajo queda como historial.

Actualizacion operativa 18:01 UTC: I6-N ejecuta busqueda conjunta endpoint h12
en la GPU preferida y h72 en el obrero secundario, codigo `6f36f16e`.
H12 tiene dos candidatos terminados sin fallos; h72 registra actualizaciones
del primer ajuste. Maximo 24 candidatos distintos por busqueda, una semilla;
el observador y la entrega OLAP funcionan por timer sin agentes. Primera
entrega nueva verificada a las 18:00:53 UTC, sin errores. Los tres pilotos
historicos h2/h3/h72 ya terminaron, con 12 informes OLAP y lectura de vuelta.
Ninguno supera su naive en esa unica semana; no es resultado anual.
La ETA h12 tras dos candidatos era 44 minutos, provisional; h72 aun sin ETA
medida. Ver `docs/i6n/SEARCH_EXECUTION_RETURN_20261011.md` en integracion.
El puente historico ya existe. Siguen pendientes supervision de trayectoria,
swarm DOIN completo y evaluacion semanal completa. La pausa mencionada abajo
corresponde a I6-E-M; no a estas dos busquedas sucesoras activas.

Precision posterior del negocio: rige BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md.
Reajuste durante cierre; plano cierre-12 h, entradas desde apertura+12 h;
targets TRAIN/score dentro de la misma semana operativa. Largos propuestos
18,24,36,48,60,72 h. Campana pausada; no hay afirmacion de implementacion o
revalidacion de los resultados anteriores bajo estas nuevas mascaras.
Aceptacion del propietario: elegibilidad dinamica t+h<B por horizonte,
mascara por salida en TRAIN y parejas MAE/naive con filas identicas, tanto
para oportunidades como para consumo real de la estrategia.

Actualizacion 2026-10-11: el propietario autoriza implementacion y ejecucion
autonomas; no se espera otra aprobacion para los pasos de ingenieria. C1/C2
aceptados como componentes y ahora integrados en `codex/i6n-integration-20261011`,
codigo base `3a9c575e`, sucesor historico `ecc94a49`. Ver `docs/i6n/INTEGRATION_EXECUTION.md` y
`docs/i6n/WEEKLY_INTEGRATION.md` en esa rama.
Verificacion automatizada: 721 pruebas CPU, 10 etapas, 71.33 segundos.
Incluye puente C1/C2, reduccion anclada en t, entrenador enmascarado,
particiones semanales con cobertura estricta, fabrica configurable y busqueda
DEAP local conjunta. El ensayo completo calendario->fit->C2 es SINTETICO.
El primer entrenamiento historico I6-N termino y fue publicado en OLAP;
DOIN distribuido no esta operativo todavia.
La auditoria del plan no se convierte en aprobacion de un runner inexistente.
El adaptador historico semanal y el plugin compartido DEAP ya estan escritos;
el segundo aun necesita el puente de evaluacion historica y despliegue del swarm.
Se recuperaron los archivos anuales 2019-2024: 36,553 cierres horarios coinciden
exactamente con los precios retenidos. La hipotesis EST fijo fue refutada por
cotizaciones UTC independientes; el candidato de reloj europeo desplaza 2,448
filas una hora hacia adelante y ninguna hacia atras. No sustituir un cierre de
auditoria INCONCLUSIVE por VERIFIED: el informe independiente es la autoridad.
El calendario NY nominal queda identificado como supuesto, no como calendario
verificado de broker. Las limitaciones de vintages y de seleccion retrospectiva
2024 se conservan. Piloto historico h2, semana 2024-01-07: 18,346 TRAIN,
350 inner, 94 pares de evaluacion; MAE 0.0008196087246374458 frente al naive
de 2 horas 0.000818191489361712, skill +0.1732156% (peor). Un ajuste,
23 epocas, 295.599 s en GPU preferida. Cuatro informes OLAP con lectura de
vuelta. No es una evaluacion anual ni habilita trading. Ver
`docs/i6n/HISTORICAL_PILOT_RETURN.md` en la rama de integracion.
TEST cerrado. La cola antigua permanece pausada; no sustituye al sucesor.
Se retiraron solo los ocho dispatchers FS4 terminados del obrero secundario y
su activador. La presion de RAM de ese host se debe principalmente a slab
no atribuida; no se promete recuperarla al retirar procesos pequenos.

## Cierre de fase 1

`PHASE_1_COMPLETE` está verificado para EURUSD 366/366 y ETH 83/83, con
perfiles, evidencia causal, recibos y lectura de vuelta del warehouse. Los 34
pares EURUSD (12 features distintas) y tres pares ETH son candidatos con
respaldo causal; no constituyen selección final.

Evidencia EURUSD:
`/home/harveybc/.local/state/predictor/phase1/eurusd-profile-4229a8a/PHASE_1_COMPLETE.json`.
Snapshot publicado:
`https://github.com/harveybc/predictor/releases/tag/phase1-feature-selection-20261005`.

## Cierre de fases 2 y 3 (2026-10-06, verificado desde artefactos)

Orden `docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md`,
plan `FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md`. Retorno único:
`docs/audits/evidence/canonical_20261003/fs_phase23/RETURN.md`.

- Fase 2 cerrada: EURUSD `PHASE_2_COMPLETE.json` 2026-10-06T23:54:40Z (66,795 pares,
  256 unidades, 0 fallidas; 7,213,860 filas de métricas, 1,202,310 de estabilidad,
  66,795 de compuerta; 1 grupo alias, 277 clusters de redundancia) y ETH
  2026-10-06T04:30:18Z (3,403 pares, 32 unidades, 0 fallidas; 245,016 / 61,254 /
  3,403; 5 alias, 38 clusters). Cubo vivo reconciliado `complete: true` para ambos
  runs; los 14 digests por tabla del cubo vivo son iguales a los sellados en los cierres.
- Fase 3 cerrada: EURUSD `PHASE_3_FILTER_COMPLETE.json` 23:57:02Z (14 targets, 45,990
  filas de ranking, 686 subconjuntos) y ETH 04:30:59Z (6 targets, 4,212 / 294). Nueve
  métodos (SPEARMAN_CLUSTER, MRMR, JMI, MRMR_CAUSAL, JMI_CAUSAL y controles
  ALL_ADMISSIBLE, UNIVARIATE_MI, CAUSAL_SUPPORTED, RANDOM_K), K={4,8,12,16,24,32}.
  `CANDIDATES_FOR_VALIDATION.json`: `predictive_winner: null`, `uses_test_split: false`.
- Snapshot fisico verificado en coordinator y worker_b; manifiesto y SHA-256
  retenidos en Git. La publicacion del binario en GitHub fue cancelada por
  decision del propietario; el enlace de release del retorno historico no
  identifica un asset publicado.
- Recursos usados: CPU en tres roles (coordinator 1 slot/1 GiB, worker_a 1/2 GiB,
  worker_b 3/2 GiB), sin GPU; tiempo de cómputo de pares 32,909 s EURUSD y 1,001 s ETH
  (0.49 y 0.29 s/par), RSS pico por proceso 0.76 GiB.

## Ultimo frente ejecutado y siguiente trabajo

I6-E compacto a 72 h: 156/156 celdas neuronales, 52 semanas por brazo,
cuatro anos anteriores para cada ajuste, seed 6102026, 20 rasgos RAW heredados
del diseno semanal retenido. Las 3,712 filas son identicas entre los tres brazos.
No se afirma seleccion optima para 72 h. Target sin limpieza y TEST cerrado.

Precio original reconciliado: MAE USD/EUR de 0.004525125 (3 tokens),
0.004701840 (5), 0.004598673 (9), contra naive pareado 0.004403521.
Ningun brazo habilita trading. OLAP: 156 informes / 2,184 metricas con lectura
exacta verificada; copia fisica `i6e/weekly_compact72_v1/olap/REPORTS.json`.

El control ARIMA cerro 156/156 celdas sin fallos. Con separacion de 36 h y
18 h, su MAE anual en precio mejora 0.464463% y 0.193648% frente al mismo
naive para el horizonte de 72 horas; la separacion de 72 h empeora 0.491589%. Son pronosticos a 72 h sobre
3,712 filas, no rentabilidad. Tres informes agregados / 18 metricas adicionales
estan publicados y verificados por lectura de vuelta. ARIMAX tiene sus 156
disposiciones: 121 exitosas y 35 fallos numericos. El recibo de publicacion
registra 312 informes / 2,910 metricas con readback. No atribuir promedio
cientifico a un brazo ARIMAX incompleto. TEST sigue cerrado.

El frente I6-E-M esta PAUSADO por solicitud de replanteamiento. Plan y grilla:
`I6E_MULTI_HORIZON_RESOLUTION_WORK_PLAN.md` y `I6E_MULTI_HORIZON_PLAN.json`.
Horizontes 2,3,4,5,6,24,48,72,96,120 h; 28 brazos downsampled primero (k=1..4
cuando h/k sea entero) y 10 horarios despues, contexto 2h, de menor a mayor
ventana. Las tres configuraciones equivalentes se ejecutan una sola vez.
Cada MAE en precio se compara con el naive para SU horizonte en precio.

Ejecutor publicado en `48c55ac0`, 38 pruebas de contratos/modelo real y dos del
observador. Cola durable: 2014 disposiciones NEURAL y 2014 ARIMA. Campana
privada vigente: `i6e/multihorizon_2024_v2`; su `STATUS.json` y
`MONITOR_STATUS.json` son la autoridad LIVE, no los conteos de este documento.
Diseno `f57b9b92d4b2a2188ab49b9acaf10df655bde18f10c7b92e8bb26b63258859c1`.
El primer despacho tuvo dos defectos de escritura de recibos: quedaron
conservados sus FAILED. El sucesor adopto los dos fits y los pronosticos
retenidos autenticados sin volver a ajustar. La 5090 y 4090 ya producen
semanas nuevas; el control ARIMA CPU continua tras sus fallos de convergencia,
sin esconderlos, y esos brazos incompletos no reciben MAE anual.
Primeros cuatro recibos: lectura exacta OLAP verificada; el follower sigue
entregando cada terminal. No hay afirmacion anual ni TEST abierto.
El barrido R2 anterior de 24 muestras queda conservado como configuracion fija,
no como prueba de ventanas adaptadas. No borrar ni reescribir sus artefactos.

La cola antigua de fase 4 no es el frente vivo y sus instrucciones de arranque
estan superadas. No saltar a NEAT, core pretraining, TEST o trading.

El contrato nuevo exige transicion D->H sin visita del usuario, continuacion
de unidades independientes ante fallos tipados, y entrega OLAP reintentable
sin reentrenar. La memoria se mide por hijo+envoltorio antes de produccion;
el host preferido solo usa su GPU preferida, un job a la vez.

## Almacenamiento obligatorio

Cada métrica y disposición se carga al OLAP y se verifica por lectura de vuelta.
Cada cierre produce snapshot DuckDB fisico verificado en dos maquinas. Git
conserva el procedimiento, esquema, manifiesto y SHA-256; los binarios
analiticos grandes quedan fuera de GitHub.

## Siguiente compuerta

Los servicios persistentes de coordinacion/OLAP estan enabled; el timer de
observacion refresca cada minuto sin agentes. Dos slots GPU y un control CPU
de 1 GiB; preferido solo 5090, secundario ventanas <=9, largas al preferido.
Primera clase n=3 medida: pico 1.73 GB; reserva siguiente con margen 1.25.
Para nuevas clases, medicion inicial conservadora 6 GiB en preferido y 4 GiB
en secundario; el lanzador exige margen vivo y vigila memoria durante el hijo.
La ETA de clases nuevas no se extrapola del score-only de fits adoptados:
consultar `tools.horizon_campaign_monitor --root <campana> --live`.
Actualizacion 2026-10-11 05:25 America/Bogota: despachador detenido y
deshabilitado; asignaciones ya hechas terminadas y recogidas sin nuevos claims.
NEURAL: 138 COMPLETED. ARIMA: 82 COMPLETED, 46 FAILED. Cero RUNNING,
3,762 QUEUED; 266/266 disposiciones publicadas con readback. Follower y
observador siguen disponibles. Las afirmaciones de ejecucion y transicion
automatica anteriores describen la campana antes de esta pausa explicita.
Siguiente accion autorizada: completar el sucesor I6-N mediante la secuencia
de `docs/i6n/INTEGRATION_EXECUTION.md`; no se necesita otra aprobacion del
propietario para implementacion, pilotos admitidos o etapas ya especificadas.
No reanudar I6-E-M como sustituto, promover semanas parciales a resultado anual
ni modificar ejecutores/checkpoints sellados.
