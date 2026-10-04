# Estado de ejecución vigente

Observado: 2026-10-04 04:32 UTC. La autoridad `codex/workplan-consolidation-20261003@02434903` está integrada en esta rama. Este corte sustituye las observaciones anteriores; no modifica resultados históricos. Los `STATUS.json` anteriores son instantáneas, no el estado vivo.

## Secuencia vigente

Prioridad crítica: completar el inventario/admisibilidad EURUSD (PS0/PS1), terminar PS2-PS5 sin descartar silenciosamente poblaciones y reanalizar PS3-C bajo su método reparado. El fix PS3-C quedó publicado en `causal-inference@48ae17c`; los expedientes previos siguen preservados y diagnósticos, no son selección. `NOT_IDENTIFIED` no significa rechazado. Ninguna feature está seleccionada.

Prioridad de negocio co-rectora: implementar
`BUSINESS_WEEKLY_WALK_FORWARD` antes de emitir cualquier veredicto financiero
final. Validation y test recorren todas sus semanas consecutivas; antes de cada
semana se ajusta con exactamente los cuatro años calendario anteriores y datos
disponibles al cutoff. `LITERATURE_STATIC` conserva su uso para réplicas, pero no
puede etiquetarse como negocio. El procedimiento se congela antes de test; los
pesos pueden cambiar por semana bajo el update mode declarado.

No ejecutar NEAT ahora. NEAT es un cabezal tardío, después de selección, ARCH, preentrenamiento por ramas, E1 R0/R1/R2, H-CORE, transferencia y representación congelada. DEAP busca configuraciones; DOIN distribuye/evalúa candidatos. NEAT no sustituye a ninguno de los dos.

## Recursos observados

| Host/GPU | Corte observado | Trabajo / disponibilidad |
|---|---|---|
| Omega RTX 4070 Laptop | escritorio, sin job batch observado | Contratos y tests CPU pequeños solamente. |
| Gamma RTX 5090 | activa, 19,518 MiB, 50 C | Lane F secuencial; despues de cerrar `tv.hilbert_amp`, `tv.kalman_dev` y `tv.stl_dev`, ejecutaba `tv.stl_seasonal` al corte. |
| Gamma RTX 5070 Ti Laptop | 0%, 14 MiB, 26 C | Ociosa. No concurre con la 5090 porque ambas comparten 14 GiB de RAM del host. |
| Dragon RTX 4090 Laptop | hijo vivo, 14,434 MiB, 40 C | Ejecuta `fx.eurjpy.logret_1h` de la cola baseline PS3-R, una celda a 8,000 MiB. |

La cola F no equivale a trabajo usando la 5070 Ti mientras E ocupa el slice. El driver E/F alterna al terminar una celda; no iniciar un duplicado ni forzar concurrencia.

## Estado por etapa

| Etapa | Estado comprobado | Alcance y siguiente paso |
|---|---|---|
| PS0/PS1 inventario y perfil | PARCIAL | Lotes 001-003: 366 candidatas, 37 fuentes de episodios y 4,444 celdas de perfil (4,432 medidas, 10 NA, 2 fallidas). Ledger de 388 filas de fuente: 141 `DECLARED_ONLY`, 119 `UNKNOWN`, 123 `NOT_APPLICABLE`, 3 `NOT_AVAILABLE_FOR_TRAIN`, 1 `NOT_INGESTED`, 1 `MEASURED`. No son 388 features aceptadas. |
| Fuentes con suscripción | PARCIAL | Yahoo Finance figura con 78 fuentes inventariadas y cierres diarios; las 2 entradas FXMacroData no cubren TRAIN de esta tarea (una solo VALIDATION/TEST, otra es calendario futuro); Alpaca está identificado pero no ingerido por data-gov. No declarar cubiertas las suscripciones por su sola presencia en el inventario. |
| Variantes de señal | PERFIL PS4 PARCIAL ACEPTADO | Diez salidas de cinco variantes causales tienen 50 unidades feature-fold y 1,050 métricas PS4 completas. Cuatro variantes globales/smoother siguen rechazadas por usar futuro. Ninguna queda seleccionada por perfilado. |
| PS2 | PARCIAL | 366 admisibles: 279 pasaron a la shortlist de PS3-C; 87 quedaron con prioridad baja en las 14 celdas. Los conteos de tiers se solapan y no deben sumarse. La cola no expresa aún cómo reconsiderar esas 87; PS5 debe dar estado a cada una. |
| PS3-C causal | FIX Y REVISIÓN PUBLICADOS | `causal-inference@48ae17c`; 45 pruebas focales pasan. Revisión read-only de los tres lotes guardada en `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 filas históricas; 109 con BH local, 0 identificación aceptada, 0 confirmaciones no lineales. La suite general del proveedor deja 6 fallos de entorno (EconML y paquete no instalado en subproceso), 149 pasan y 27 skip. |
| PS3-R extractibilidad | EN EJECUCIÓN EN DOS GPU | Dragon continua baseline. Gamma lane F cerro tres celdas past-to-current con hash de resultados igual al manifiesto (`tv.hilbert_amp`, `tv.kalman_dev`, `tv.stl_dev`) y continua secuencialmente. Son mediciones de extractor; reconstruccion/utilidad no equivalen a seleccion. |
| Cobertura de features | RECONCILIACIÓN PARCIAL | `coverage_reconciliation/`: 366/366 filas, 279 en join PS3-C + 87 fuera, 137 en cola E, 142 fuera de E = 132 tier-3 + 10 calendario. Ocho pruebas pasan. Proveedores pagados y transformaciones siguen requiriendo su propia reconciliación; no se declara exhaustivo. |
| PS4/PS5 manifiesto conjunto | LEDGER DE RETSU RECHAZADO, PILOTO PARCIAL | PS4 acepta diez transformaciones: 50/50 unidades y 1,050 filas. El ledger `5da79f34` usa estados causales historicos (65 identificadas + 13 mixtas) que contradicen `48ae17c` (0 identificadas), y adopta celdas con constantes. Debe repararse antes de common-K; no se libera seleccion ni estrategia. |
| ARCH/E1/H-CORE/NEAT/RL | AÚN NO ELEGIBLES | Ejecutar en orden canónico solo tras manifiesto final; H-CORE después de E1 y NEAT al final sobre representación congelada. |
| Walk-forward semanal | IMPLEMENTACIÓN PARCIAL, SCORER NO ACEPTADO | Contrato BW01-BW18 y nucleo de calendario/modos implementados. El scorer `39a45e88` pierde diferencias pareadas al recargar, exige indebidamente el mismo modelo corto/largo y cuenta semanas como replicas independientes. Debe repararse antes del piloto semanal. |
| Literatura | COLA NO VIVA | No hay runner científico observado en Dragon; el STATUS anterior describía una cola Traffic, no un proceso vivo en este corte. |

## Trabajo en paralelo y orden inmediato

1. **Gamma:** no detener lane F; una celda en la 5090, sin concurrencia con la 5070 Ti mientras el host no admita ambas.
2. **Dragon:** no detener la baseline PS3-R; una celda terminal por vez y sin duplicados.
3. **CPU causal/ledger:** reconstruir M2 exclusivamente desde `reanalysis_48ae17c`; mantener 0 identificadas como abstencion, no rechazo.
4. **CPU evidencia:** reemplazar constantes por adopcion automatica de manifiestos terminales verificados, separando baseline y familias alternativas.
5. **CPU PS4:** perfilar incrementalmente cada feature terminal autentica sin target/test, sin esperar las 366.
6. **CPU negocio:** reparar persistencia pareada, identidad separada corto/largo y dependencia longitudinal del scorer; fijar MAE primario antes del piloto semanal.

## Resultado/ETA

Hay un piloto predictivo PS5 de desarrollo, no exactitud financiera ni feature seleccionada: su mejor brazo no supera al naive pareado. La reconciliación reduce incertidumbre de cobertura, pero no decide selección. Al ritmo observado, las colas E/F pendientes representan aproximadamente 80 horas de GPU compartida; no es una ETA de PS5 completo ni de selección final.

Progreso visual anterior: [PROGRESS_20261003T1634Z.png](../../audits/evidence/canonical_20261003/PROGRESS_20261003T1634Z.png). Revisión causal: [REPORT](../../audits/evidence/canonical_20261003/laneC/reanalysis_48ae17c/REPORT.md). Cobertura de features: [README](../../audits/evidence/canonical_20261003/coverage_reconciliation/README.md). Fuentes y transformaciones: [REPORT.json](../../audits/evidence/canonical_20261003/source_transform_coverage/REPORT.json).
