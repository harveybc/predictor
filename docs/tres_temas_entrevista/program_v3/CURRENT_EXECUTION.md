# Estado de ejecución vigente

Observado: 2026-10-05. Este archivo sustituye estados operativos anteriores.

## Cierre de fase 1

`PHASE_1_COMPLETE` está verificado para EURUSD 366/366 y ETH 83/83, con
perfiles, evidencia causal, recibos y lectura de vuelta del warehouse. Los 34
pares EURUSD (12 features distintas) y tres pares ETH son candidatos con
respaldo causal; no constituyen selección final.

Evidencia EURUSD:
`/home/harveybc/.local/state/predictor/phase1/eurusd-profile-4229a8a/PHASE_1_COMPLETE.json`.
Snapshot publicado:
`https://github.com/harveybc/predictor/releases/tag/phase1-feature-selection-20261005`.

## Único frente activo

Fases 2 y 3 de selección, bajo
`FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md` y la orden
`docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md`.

- Fase 2: 66,795 pares EURUSD y 3,403 ETH; alias, Pearson, Spearman, Kendall,
  información mutua, distance correlation, lags y estabilidad temporal.
- Fase 3: clustering-Spearman, mRMR y JMI, con controles y trayectorias
  `K={4,8,12,16,24,32}` por target/horizonte.
- Recursos: CPU distribuida entre omega, gamma y dragon. Sin GPU.
- Operación: driver durable, shards exclusivos, terminales atómicos, follower
  de warehouse, `STATUS.json` y continuidad automática entre fases.

Fase 4 (extractibilidad/AE/DAE), arquitectura modular, DOIN, NEAT, RL,
referencias, M5PHET y nuevas corridas quedan pausadas durante este cierre.

## Almacenamiento obligatorio

Cada métrica y disposición se carga al OLAP y se verifica por lectura de vuelta.
Cada cierre produce snapshot DuckDB físico local y asset de release comprimido.
Git conserva manifiesto, esquema, SHA-256 y URL, no una copia binaria duplicada
dentro del historial.

## Siguiente compuerta

`PHASE_2_COMPLETE.json`, seguido automáticamente por
`PHASE_3_FILTER_COMPLETE.json`. El siguiente trabajo permitido después es
congelar el diseño de fase 4; no empezarlo dentro de esta orden.
