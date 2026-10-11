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

## Frente activo

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
naive; la separacion de 72 h empeora 0.491589%. Son pronosticos a 72 h sobre
3,712 filas, no rentabilidad. Tres informes agregados / 18 metricas adicionales
estan publicados y verificados por lectura de vuelta. ARIMAX sigue en dos
workers, semanas pares/impares sin duplicacion, con fallos numericos retenidos.
La coleccion, el PNG, ETA y publicacion de disposiciones se actualizan con timers.

Sigue `i6e/statistical72_v1`: controles ARIMA y ARIMAX (1,1,0), sin drift,
remuestreo al origen, parametros semanales fijos y actualizacion de estado
horaria por fase. El ejecutor mide TRAIN primero y continua automaticamente
con 312 celdas semanales-resolucion, conservando fallos y evitando repeticiones.
Estado consultable con `python -m tools.i6e_statistical_campaign status
--output <statistical-directory>`. La cola antigua de fase 4 no es el frente
vivo y sus instrucciones de arranque estan superadas.

Despues del cierre estadistico sigue la extension multihorizonte descrita en
`I6E_ORIGIN_ANCHORED_RESOLUTION_WORK_PLAN.md`, con seleccion de configuracion
solo en TRAIN. No saltar a NEAT ni atribuir utilidad financiera a un MAE.

## Almacenamiento obligatorio

Cada métrica y disposición se carga al OLAP y se verifica por lectura de vuelta.
Cada cierre produce snapshot DuckDB fisico verificado en dos maquinas. Git
conserva el procedimiento, esquema, manifiesto y SHA-256; los binarios
analiticos grandes quedan fuera de GitHub.

## Siguiente compuerta

Cerrar los controles estadisticos en la misma poblacion y publicar con lectura
de vuelta. Si faltan celdas, informar evidencia incompleta sin promediar las
sobrevivientes ni reemplazar un fallo por naive. La extension por horizonte
conserva su propio diseno, costo y compuerta antes de cualquier ensayo de trading.
