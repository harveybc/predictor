# Estado de ejecución vigente

Observado: 2026-10-03 16:22 UTC. La autoridad `codex/workplan-consolidation-20261003@02434903` está integrada en esta rama. Este corte sustituye las observaciones anteriores; no modifica resultados históricos.

## Secuencia vigente

Prioridad crítica: completar el inventario/admisibilidad EURUSD (PS0/PS1), terminar PS2-PS5 sin descartar silenciosamente poblaciones y reanalizar PS3-C bajo su método reparado. El fix PS3-C quedó publicado en `causal-inference@48ae17c`; los expedientes previos siguen preservados y diagnósticos, no son selección. `NOT_IDENTIFIED` no significa rechazado. Ninguna feature está seleccionada.

No ejecutar NEAT ahora. NEAT es un cabezal tardío, después de selección, ARCH, preentrenamiento por ramas, E1 R0/R1/R2, H-CORE, transferencia y representación congelada. DEAP busca configuraciones; DOIN distribuye/evalúa candidatos. NEAT no sustituye a ninguno de los dos.

## Recursos observados

| Host/GPU | Corte observado | Trabajo / disponibilidad |
|---|---|---|
| Omega RTX 4070 Laptop | 38%, 1,185 MiB, 50 C | Uso de escritorio/servicios; sin job científico batch. Se protege el escritorio. |
| Gamma RTX 5090 | 41%, 30,426 MiB, 49 C | Lane E, `rg.di_spread`, activo en batch 003. |
| Gamma RTX 5070 Ti Laptop | 0%, 14 MiB, 38 C | Lane F terminó `ta.roc_10` (resultado y manifiesto retenidos); `ta.rsi_14` espera. E/F comparten el slice de 8 GiB. |
| Dragon RTX 4090 Laptop | 7%, 14 MiB, 29 C | Sin trabajo científico; `lts-mt5-paper` RUNNING y host con 13 GiB disponibles. No hay hoy ruta gobernada completa: faltan contratos de recursos, archivos de reconstrucción exacta y pin de código. No se copió ni lanzó nada. |

La cola F no equivale a trabajo usando la 5070 Ti mientras E ocupa el slice. El driver E/F alterna al terminar una celda; no iniciar un duplicado ni forzar concurrencia.

## Estado por etapa

| Etapa | Estado comprobado | Alcance y siguiente paso |
|---|---|---|
| PS0/PS1 inventario y perfil | PARCIAL | Lotes 001-003: 366 candidatas, 37 fuentes de episodios y 4,444 celdas de perfil (4,432 medidas, 10 NA, 2 fallidas). El inventario de lote 001 tiene 388 filas de fuente, no 388 features aceptadas. Resolver fuentes/transformaciones pendientes y preservar razones de no admisión. |
| Fuentes con suscripción | PARCIAL | Yahoo Finance figura con 78 fuentes inventariadas y cierres diarios; las 2 entradas FXMacroData no cubren TRAIN de esta tarea (una solo VALIDATION/TEST, otra es calendario futuro); Alpaca está identificado pero no ingerido por data-gov. No declarar cubiertas las suscripciones por su sola presencia en el inventario. |
| Variantes de señal | PARCIAL | MODWT-Haar causal, multitaper trailing, Hilbert trailing, STL trailing y Kalman filter son variantes identificadas/admisibles; perfiles PS1/PS4 siguen pendientes. DWT global, Hilbert global, STL global y RTS smoother fallan el probe de prefijo por usar futuro. |
| PS2 | PARCIAL | 366 admisibles: 279 pasaron a la shortlist de PS3-C; 87 quedaron con prioridad baja en las 14 celdas. Los conteos de tiers se solapan y no deben sumarse. La cola no expresa aún cómo reconsiderar esas 87; PS5 debe dar estado a cada una. |
| PS3-C causal | FIX Y REVISIÓN PUBLICADOS | `causal-inference@48ae17c`; 45 pruebas focales pasan. Revisión read-only de los tres lotes guardada en `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 filas históricas; 109 con BH local, 0 identificación aceptada, 0 confirmaciones no lineales. La suite general del proveedor deja 6 fallos de entorno (EconML y paquete no instalado en subproceso), 149 pasan y 27 skip. |
| PS3-R extractibilidad | EN EJECUCIÓN | E: batch 001 tier-1 33/33; batch 003 midió `rg.atr_pct`, `rg.atr_ratio`, `rg.bb_width_pct`; `rg.di_spread` activo. F: 28 resultados retenidos (`ta.roc_10` ya cerrado); `ta.rsi_14` espera. De la shortlist: 137/279 en la matriz E; quedan 142 (132 tier-3, 10 calendario) y 87 low-priority con disposición pendiente, no rechazadas. |
| Cobertura de features | RECONCILIACIÓN PARCIAL | `coverage_reconciliation/`: 366/366 filas, 279 en join PS3-C + 87 fuera, 137 en cola E, 142 fuera de E = 132 tier-3 + 10 calendario. Ocho pruebas pasan. Proveedores pagados y transformaciones siguen requiriendo su propia reconciliación; no se declara exhaustivo. |
| PS4/PS5 manifiesto conjunto | NO INICIADO | Espera reanálisis causal, cobertura de extractibilidad y disposiciones explícitas. Ninguna feature está finalizada como selección de negocio. |
| ARCH/E1/H-CORE/NEAT/RL | AÚN NO ELEGIBLES | Ejecutar en orden canónico solo tras manifiesto final; H-CORE después de E1 y NEAT al final sobre representación congelada. |
| Literatura | COLA NO VIVA | No hay runner científico observado en Dragon; el STATUS anterior describía una cola Traffic, no un proceso vivo en este corte. |

## Trabajo en paralelo y orden inmediato

1. **Gamma E/F:** E corre `rg.di_spread` en 5090; F `ta.rsi_14` espera en 5070 Ti. El pico previo de E (≈6.5 GiB) + el consumo de F (≈3.6 GiB) exceden 8 GiB; alternan sin solaparse.
2. **CPU, causal-inference:** revisión read-only de los tres lotes completada bajo `48ae17c`; preservar originales y mantener los 0 casos identificados como resultado del gate, no como rechazo de features. La siguiente acción es cerrar fuentes/transformaciones y reabrir análisis cuando haya evidencia de assumptions verificable.
3. **CPU, cobertura de selección:** la partición de 366 features ya se reconcilió; continuar las disposiciones de 87 low-priority, 142 fuera de E, transformaciones PS4 y fuentes pagadas/no disponibles. Sin omisiones implícitas.
4. **Dragon:** auditoría dio NO-GO reproducible para `px.ewma_vol_168`: faltan contratos de disponibilidad, archivos exactos gobernados, pins de feature-eng/feature-extractor y bundle PS2. La 4090 sigue disponible pero no es una colocación válida hoy; no desviar ni copiar bytes por fuera del lago. `lts-mt5-paper` permanece activa.
5. **Omega:** mantener solo verificaciones CPU pequeñas y reportes; no ocupar la 4070 de escritorio para un job largo.

## Resultado/ETA

No hay resultado nuevo de exactitud financiera ni feature seleccionada. La reconciliación reduce incertidumbre de cobertura, pero no decide selección. ETA total no estimable hasta medir duración completa de E/F y saber qué parte de las 229 entradas pendientes requiere cómputo frente a una disposición documentada más barata.

Progreso visual del corte: [PROGRESS_20261003T1622Z.png](../../audits/evidence/canonical_20261003/PROGRESS_20261003T1622Z.png). Revisión causal: [REPORT](../../audits/evidence/canonical_20261003/laneC/reanalysis_48ae17c/REPORT.md). Cobertura reproducible: [README](../../audits/evidence/canonical_20261003/coverage_reconciliation/README.md).
