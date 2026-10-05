# Estado de ejecución vigente

> **Actualizacion vinculante 2026-10-05:** por orden del propietario, el unico
> frente experimental activo es el cierre de **seleccion de caracteristicas,
> fase 1**. La orden ejecutable es
> `docs/handoffs/MUSASHI_TO_SATOSHI_FEATURE_SELECTION_PHASE1_ONLY_2026_10_05.md`.
> Se pausan PS3-R/extractibilidad, autoencoders, CVAE, arquitectura modular,
> NEAT, RL y nuevas replicas hasta que el warehouse confirme 366/366 perfiles,
> la escalera causal EURUSD/ETH finalizada globalmente y
> `PHASE_1_COMPLETE`. La evidencia ya obtenida se conserva y no se interpreta
> como seleccion final.

Observado: 2026-10-05 03:06 UTC. La autoridad `codex/workplan-consolidation-20261003@02434903` esta integrada en esta rama. La orden operativa vigente de cierre es `docs/handoffs/SATOSHI_FEATURE_SELECTION_CLOSURE_2026_10_05.md`. Este corte sustituye las observaciones anteriores; no modifica resultados historicos. Los `STATUS.json` anteriores son instantaneas, no el estado vivo.

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
| Gamma RTX 5090 | activa, 19,520 MiB, 43 C | Alternativas PS3-R: 244/274 completas, 0 fallos; ejecuta `yh.lqd.logret_5d` con `past_to_current_siamese`. Luego toma 51 baselines declaradas y trabajo pendiente sin duplicar. |
| Gamma RTX 5070 Ti Laptop | 0%, 14 MiB, 26 C | Ociosa. No concurre con la 5090 porque ambas comparten 14 GiB de RAM del host. |
| Dragon RTX 4090 Laptop | activa, 14,430 MiB, 40 C | Baseline PS3-R: 31/86 completas, 0 fallos; ejecuta `yh.slv.logret_1d`, una celda a 8,000 MiB. |

La cola F no equivale a trabajo usando la 5070 Ti mientras E ocupa el slice. El driver E/F alterna al terminar una celda; no iniciar un duplicado ni forzar concurrencia.

## Estado por etapa

| Etapa | Estado comprobado | Alcance y siguiente paso |
|---|---|---|
| PS0/PS1 inventario y perfil | PARCIAL | Lotes 001-003: 366 candidatas, 37 fuentes de episodios y 4,444 celdas de perfil (4,432 medidas, 10 NA, 2 fallidas). Ledger de 388 filas de fuente: 141 `DECLARED_ONLY`, 119 `UNKNOWN`, 123 `NOT_APPLICABLE`, 3 `NOT_AVAILABLE_FOR_TRAIN`, 1 `NOT_INGESTED`, 1 `MEASURED`. No son 388 features aceptadas. |
| Fuentes con suscripción | PARCIAL | Yahoo Finance figura con 78 fuentes inventariadas y cierres diarios; las 2 entradas FXMacroData no cubren TRAIN de esta tarea (una solo VALIDATION/TEST, otra es calendario futuro); Alpaca está identificado pero no ingerido por data-gov. No declarar cubiertas las suscripciones por su sola presencia en el inventario. |
| Variantes de señal | PERFIL PS4 PARCIAL ACEPTADO | Diez salidas de cinco variantes causales tienen 50 unidades feature-fold y 1,050 métricas PS4 completas. Cuatro variantes globales/smoother siguen rechazadas por usar futuro. Ninguna queda seleccionada por perfilado. |
| PS2 | PARCIAL | 366 admisibles: 279 pasaron a la shortlist de PS3-C; 87 quedaron con prioridad baja en las 14 celdas. Los conteos de tiers se solapan y no deben sumarse. La cola no expresa aún cómo reconsiderar esas 87; PS5 debe dar estado a cada una. |
| PS3-C causal | FIX Y REVISIÓN PUBLICADOS | `causal-inference@48ae17c`; 45 pruebas focales pasan. Revisión read-only de los tres lotes guardada en `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 filas históricas; 109 con BH local, 0 identificación aceptada, 0 confirmaciones no lineales. La suite general del proveedor deja 6 fallos de entorno (EconML y paquete no instalado en subproceso), 149 pasan y 27 skip. |
| PS3-R extractibilidad | EN EJECUCION EN DOS GPU | Alternativas 5090: 244/274; baseline Dragon: 31/86; baseline sucesora 5090: 0/51. No hay fallos terminales. PS4 ha aceptado y perfilado las 31 terminales baseline descubiertas. Son mediciones de extractor; reconstruccion/utilidad no equivalen a seleccion. |
| Cobertura de features | RECONCILIACIÓN PARCIAL | `coverage_reconciliation/`: 366/366 filas, 279 en join PS3-C + 87 fuera, 137 en cola E, 142 fuera de E = 132 tier-3 + 10 calendario. Ocho pruebas pasan. Proveedores pagados y transformaciones siguen requiriendo su propia reconciliación; no se declara exhaustivo. |
| PS4/PS5 manifiesto conjunto | LEDGER Y PRIMER PERFIL INCREMENTAL ACEPTADOS | El ledger autentica 14 terminales y su producto pliegue × familia × target × horizonte × metrica; conserva 279 `NOT_IDENTIFIED` + 87 `OUTSIDE_JOIN_PENDING` y 366 `NOT_READY`. El primer perfil incremental conserva 5 unidades VIX `MEASURED`, 1 `PENDING`, 105 metricas y procedencia PS3-R externa. No se libera seleccion ni estrategia. |
| ARCH/E1/H-CORE/NEAT/RL | AÚN NO ELEGIBLES | Ejecutar en orden canónico solo tras manifiesto final; H-CORE después de E1 y NEAT al final sobre representación congelada. |
| Walk-forward semanal | NUCLEO Y SCORER DURABLE IMPLEMENTADOS | Contrato BW01-BW18, calendario/modos y scorer pareado pasan 72 pruebas focalizadas. MAE financiero, identidad separada corto/largo, contrato y `WeekSpec` completos, diferencias por origen y semantica de entrega `AT_LEAST_ONCE_IDEMPOTENT_SCORE_DIGEST` quedan retenidos. Falta el piloto anual semanal completo. |
| Literatura | COLA NO VIVA | No hay runner científico observado en Dragon; el STATUS anterior describía una cola Traffic, no un proceso vivo en este corte. |

## Trabajo en paralelo y orden inmediato

1. **Gamma:** cerrar las 30 alternativas, encadenar las 51 baselines y despues ayudar a Dragon con claims exclusivos.
2. **Dragon:** no detener la baseline PS3-R; una celda terminal por vez y sin duplicados.
3. **CPU selectores:** ejecutar common-K predictivo/redundancia, ChronoEpilogi y controles sobre folds TRAIN identicos.
4. **CPU causal:** completar la escalera causal con abstencion honesta y comparadores lagged; `NOT_IDENTIFIED` permanece neutral.
5. **CPU evidencia/PS4:** ingerir cada terminal y decidir representacion raw/random/trained sin confundir reconstruccion con seleccion.
6. **CPU cierre:** preparar refits emparejados, manifiesto 366/366 y naive de las mismas filas. El piloto semanal sigue independiente, sin retrasar seleccion.

## Resultado/ETA

Hay un piloto predictivo PS5 de desarrollo, no exactitud financiera ni feature seleccionada: su mejor brazo no supera al naive pareado. La reconciliacion reduce incertidumbre de cobertura, pero no decide seleccion. Con las tasas autenticadas, las 30 alternativas restantes tienen ETA 1.6-2.1 h y las 55 baselines actuales de Dragon 19.7-25.0 h; la sucesora de 51 celdas medira su ETA tras tres terminales. El cierre common-K se construye en paralelo y no espera ociosamente esas colas.

Progreso visual anterior: [PROGRESS_20261003T1634Z.png](../../audits/evidence/canonical_20261003/PROGRESS_20261003T1634Z.png). Revisión causal: [REPORT](../../audits/evidence/canonical_20261003/laneC/reanalysis_48ae17c/REPORT.md). Cobertura de features: [README](../../audits/evidence/canonical_20261003/coverage_reconciliation/README.md). Fuentes y transformaciones: [REPORT.json](../../audits/evidence/canonical_20261003/source_transform_coverage/REPORT.json).
