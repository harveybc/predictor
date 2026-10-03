# Estado de ejecución vigente

Observado: 2026-10-03 18:04 UTC. La autoridad `codex/workplan-consolidation-20261003@02434903` está integrada en esta rama. Este corte sustituye las observaciones anteriores; no modifica resultados históricos. El `STATUS.json` de las 10:02 UTC es una instantánea anterior, no el estado vivo.

## Secuencia vigente

Prioridad crítica: completar el inventario/admisibilidad EURUSD (PS0/PS1), terminar PS2-PS5 sin descartar silenciosamente poblaciones y reanalizar PS3-C bajo su método reparado. El fix PS3-C quedó publicado en `causal-inference@48ae17c`; los expedientes previos siguen preservados y diagnósticos, no son selección. `NOT_IDENTIFIED` no significa rechazado. Ninguna feature está seleccionada.

No ejecutar NEAT ahora. NEAT es un cabezal tardío, después de selección, ARCH, preentrenamiento por ramas, E1 R0/R1/R2, H-CORE, transferencia y representación congelada. DEAP busca configuraciones; DOIN distribuye/evalúa candidatos. NEAT no sustituye a ninguno de los dos.

## Recursos observados

| Host/GPU | Corte observado | Trabajo / disponibilidad |
|---|---|---|
| Omega RTX 4070 Laptop | 38%, 1,185 MiB, 50 C | Uso de escritorio/servicios; sin job científico batch. Se protege el escritorio. |
| Gamma RTX 5090 | 50%, 30,428 MiB, 53 C | Lane E, `tv.kalman_dev`, activo en batch 003; 43 manifiestos de rasgo E retenidos. |
| Gamma RTX 5070 Ti Laptop | 0%, 14 MiB, 39 C | Lane F tiene 37 manifiestos de rasgo retenidos y espera turno; E/F comparten el slice de 8 GiB. |
| Dragon RTX 4090 Laptop | 7%, 14 MiB, 29 C | Sin trabajo científico; `lts-mt5-paper` RUNNING y host con 13 GiB disponibles. No hay hoy ruta gobernada completa: faltan contratos de recursos, archivos de reconstrucción exacta y pin de código. No se copió ni lanzó nada. |

La cola F no equivale a trabajo usando la 5070 Ti mientras E ocupa el slice. El driver E/F alterna al terminar una celda; no iniciar un duplicado ni forzar concurrencia.

## Estado por etapa

| Etapa | Estado comprobado | Alcance y siguiente paso |
|---|---|---|
| PS0/PS1 inventario y perfil | PARCIAL | Lotes 001-003: 366 candidatas, 37 fuentes de episodios y 4,444 celdas de perfil (4,432 medidas, 10 NA, 2 fallidas). Ledger de 388 filas de fuente: 141 `DECLARED_ONLY`, 119 `UNKNOWN`, 123 `NOT_APPLICABLE`, 3 `NOT_AVAILABLE_FOR_TRAIN`, 1 `NOT_INGESTED`, 1 `MEASURED`. No son 388 features aceptadas. |
| Fuentes con suscripción | PARCIAL | Yahoo Finance figura con 78 fuentes inventariadas y cierres diarios; las 2 entradas FXMacroData no cubren TRAIN de esta tarea (una solo VALIDATION/TEST, otra es calendario futuro); Alpaca está identificado pero no ingerido por data-gov. No declarar cubiertas las suscripciones por su sola presencia en el inventario. |
| Variantes de señal | PARCIAL | Cinco variantes trailing/filter pasan solo la prueba de computabilidad causal; sus perfiles PS1/PS4 siguen `PENDING_PROFILE`. Cuatro variantes globales/smoother fallan el probe de prefijo por usar futuro. Ninguna es seleccionada por esta prueba. |
| PS2 | PARCIAL | 366 admisibles: 279 pasaron a la shortlist de PS3-C; 87 quedaron con prioridad baja en las 14 celdas. Los conteos de tiers se solapan y no deben sumarse. La cola no expresa aún cómo reconsiderar esas 87; PS5 debe dar estado a cada una. |
| PS3-C causal | FIX Y REVISIÓN PUBLICADOS | `causal-inference@48ae17c`; 45 pruebas focales pasan. Revisión read-only de los tres lotes guardada en `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 filas históricas; 109 con BH local, 0 identificación aceptada, 0 confirmaciones no lineales. La suite general del proveedor deja 6 fallos de entorno (EconML y paquete no instalado en subproceso), 149 pasan y 27 skip. |
| PS3-R extractibilidad | EN EJECUCIÓN | E: 43 manifiestos de rasgo retenidos de 137 en la cola; F: 37 de 274 trabajos planeados, alternando bajo el slice compartido. De la shortlist: 137/279 en la matriz E; quedan 142 (132 tier-3, 10 calendario) y 87 low-priority con disposición pendiente, no rechazadas. Reconstrucción no equivale a utilidad predictiva. |
| Cobertura de features | RECONCILIACIÓN PARCIAL | `coverage_reconciliation/`: 366/366 filas, 279 en join PS3-C + 87 fuera, 137 en cola E, 142 fuera de E = 132 tier-3 + 10 calendario. Ocho pruebas pasan. Proveedores pagados y transformaciones siguen requiriendo su propia reconciliación; no se declara exhaustivo. |
| PS4/PS5 manifiesto conjunto | PILOTO PARCIAL | PS4: nueve variantes sin perfil ampliado. PS5: un contraste EURUSD Y_l@24h `inner_2019`, una semilla, cuatro brazos, 4,944 orígenes pareados y 429 actualizaciones por brazo. Pareja MAE 0.002228, mejor brazo individual 0.002252, naive pareado 0.002199: reingreso relativo, pero falla el naive. Solo lote 001 presente en el piloto local; no se libera selección ni se prueba en estrategia. Ejecutor y prueba de compuerta en `d5d8d077`. |
| ARCH/E1/H-CORE/NEAT/RL | AÚN NO ELEGIBLES | Ejecutar en orden canónico solo tras manifiesto final; H-CORE después de E1 y NEAT al final sobre representación congelada. |
| Literatura | COLA NO VIVA | No hay runner científico observado en Dragon; el STATUS anterior describía una cola Traffic, no un proceso vivo en este corte. |

## Trabajo en paralelo y orden inmediato

1. **Gamma E/F:** E corre `tv.kalman_dev` en 5090; F espera en 5070 Ti. El pico previo de E (≈6.5 GiB) + el consumo de F (≈3.6 GiB) exceden 8 GiB; alternan sin solaparse.
2. **CPU, causal-inference:** revisión read-only de los tres lotes completada bajo `48ae17c`; preservar originales y mantener los 0 casos identificados como resultado del gate, no como rechazo de features. La siguiente acción es cerrar fuentes/transformaciones y reabrir análisis cuando haya evidencia de assumptions verificable.
3. **CPU, cobertura de selección:** ledger de fuentes/transformaciones incorporado desde Retsu `749ba6a8` como `292e13cc`, sin diferencia de árbol entre esos dos commits. Continuar las disposiciones de 87 low-priority, 142 fuera de E, transformaciones PS4 y fuentes pagadas/no disponibles. Sin omisiones implícitas.
4. **Dragon:** auditoría dio NO-GO reproducible para `px.ewma_vol_168`: faltan contratos de disponibilidad, archivos exactos gobernados, pins de feature-eng/feature-extractor y bundle PS2. La 4090 sigue disponible pero no es una colocación válida hoy; no desviar ni copiar bytes por fuera del lago. `lts-mt5-paper` permanece activa.
5. **Omega:** mantener solo verificaciones CPU pequeñas y reportes; no ocupar la 4070 de escritorio para un job largo.

## Resultado/ETA

Hay un piloto predictivo PS5 de desarrollo, no exactitud financiera ni feature seleccionada: su mejor brazo no supera al naive pareado. La reconciliación reduce incertidumbre de cobertura, pero no decide selección. Al ritmo observado, las colas E/F pendientes representan aproximadamente 80 horas de GPU compartida; no es una ETA de PS5 completo ni de selección final.

Progreso visual anterior: [PROGRESS_20261003T1634Z.png](../../audits/evidence/canonical_20261003/PROGRESS_20261003T1634Z.png). Revisión causal: [REPORT](../../audits/evidence/canonical_20261003/laneC/reanalysis_48ae17c/REPORT.md). Cobertura de features: [README](../../audits/evidence/canonical_20261003/coverage_reconciliation/README.md). Fuentes y transformaciones: [REPORT.json](../../audits/evidence/canonical_20261003/source_transform_coverage/REPORT.json).
