# Estado de ejecución vigente

Observado: 2026-10-07T01:44Z. Este archivo sustituye estados operativos anteriores.

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

Fase 4 de extractibilidad y seleccion predictiva, definida en
`FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md`. El controlador de tareas
y sus pruebas ya existen; el piloto de coste `px.rv5` esta corriendo en la
4090 bajo `fs4-cost-pilot-20261007.service` (4 GiB, 30 min). El runner
cientifico integrado y los recibos de la cola siguen pendientes. No hay
ganador final. La
validacion del wrapper usa `BUSINESS_WEEKLY_WALK_FORWARD`.

Arquitectura modular, DOIN, NEAT, RL, referencias, M5PHET y nuevas corridas siguen
pausadas hasta que ese diseño quede congelado y aprobado.

## Almacenamiento obligatorio

Cada métrica y disposición se carga al OLAP y se verifica por lectura de vuelta.
Cada cierre produce snapshot DuckDB fisico verificado en dos maquinas. Git
conserva el procedimiento, esquema, manifiesto y SHA-256; los binarios
analiticos grandes quedan fuera de GitHub.

## Siguiente compuerta

Integrar el runner real de extractibilidad, medir una celda piloto y
despachar la cola automatica; despues, wrapper semanal y manifiesto final.
