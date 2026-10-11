# Estado de ejecución vigente

Observado: 2026-10-11. Este archivo sustituye estados operativos anteriores.

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

El frente EN EJECUCION es I6-E-M. Plan y grilla:
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
Siguiente compuerta: cierre autonomo D->H y readback de ambas etapas. Ningun
agente ni nueva visita del usuario es requisito. No promover semanas parciales.
