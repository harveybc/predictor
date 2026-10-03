# Estado de ejecución vigente

Observado: 2026-10-03 23:35 UTC. La autoridad `codex/workplan-consolidation-20261003@02434903` está integrada en esta rama. Este corte sustituye las observaciones anteriores; no modifica resultados históricos. Los `STATUS.json` anteriores son instantáneas, no el estado vivo.

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
| Gamma RTX 5090 | 0%, 48 MiB, 36 C | Ociosa. `dprime` está en espera de admisión a 8,000 MiB; no existe hijo de modelo. La nueva orden mueve esa carga a dragon y asigna lane F, que sí cabe, a la 5090. |
| Gamma RTX 5070 Ti Laptop | 0%, 14 MiB, 26 C | Ociosa. No concurre con la 5090 porque ambas comparten 14 GiB de RAM del host. |
| Dragon RTX 4090 Laptop | 7%, 14 MiB, 30 C | Ociosa tras completar `fred.stress.vixcls.logret_5d`; 11 GiB disponibles y batch 002 autenticado presente. Recibe la cola PS3-R de 8,000 MiB. |

La cola F no equivale a trabajo usando la 5070 Ti mientras E ocupa el slice. El driver E/F alterna al terminar una celda; no iniciar un duplicado ni forzar concurrencia.

## Estado por etapa

| Etapa | Estado comprobado | Alcance y siguiente paso |
|---|---|---|
| PS0/PS1 inventario y perfil | PARCIAL | Lotes 001-003: 366 candidatas, 37 fuentes de episodios y 4,444 celdas de perfil (4,432 medidas, 10 NA, 2 fallidas). Ledger de 388 filas de fuente: 141 `DECLARED_ONLY`, 119 `UNKNOWN`, 123 `NOT_APPLICABLE`, 3 `NOT_AVAILABLE_FOR_TRAIN`, 1 `NOT_INGESTED`, 1 `MEASURED`. No son 388 features aceptadas. |
| Fuentes con suscripción | PARCIAL | Yahoo Finance figura con 78 fuentes inventariadas y cierres diarios; las 2 entradas FXMacroData no cubren TRAIN de esta tarea (una solo VALIDATION/TEST, otra es calendario futuro); Alpaca está identificado pero no ingerido por data-gov. No declarar cubiertas las suscripciones por su sola presencia en el inventario. |
| Variantes de señal | PERFIL PS4 PARCIAL ACEPTADO | Diez salidas de cinco variantes causales tienen 50 unidades feature-fold y 1,050 métricas PS4 completas. Cuatro variantes globales/smoother siguen rechazadas por usar futuro. Ninguna queda seleccionada por perfilado. |
| PS2 | PARCIAL | 366 admisibles: 279 pasaron a la shortlist de PS3-C; 87 quedaron con prioridad baja en las 14 celdas. Los conteos de tiers se solapan y no deben sumarse. La cola no expresa aún cómo reconsiderar esas 87; PS5 debe dar estado a cada una. |
| PS3-C causal | FIX Y REVISIÓN PUBLICADOS | `causal-inference@48ae17c`; 45 pruebas focales pasan. Revisión read-only de los tres lotes guardada en `laneC/reanalysis_48ae17c/`: 279 candidatas, 1,076 filas históricas; 109 con BH local, 0 identificación aceptada, 0 confirmaciones no lineales. La suite general del proveedor deja 6 fallos de entorno (EconML y paquete no instalado en subproceso), 149 pasan y 27 skip. |
| PS3-R extractibilidad | EN EJECUCIÓN, COLOCACIÓN CORREGIDA | Dragon completó `fred.stress.vixcls.logret_5d` en 1,134.3 s: 441 filas y pico cgroup 7.315 GB. AE/DAE reconstruyen bien, pero su utilidad contra random es mixta y ambos pierden Y_b en 10/10 celdas. Las celdas E de 8,000 MiB pasan a dragon; gamma 5090 ejecuta lane F a sus topes medidos. Reconstrucción no equivale a selección. |
| Cobertura de features | RECONCILIACIÓN PARCIAL | `coverage_reconciliation/`: 366/366 filas, 279 en join PS3-C + 87 fuera, 137 en cola E, 142 fuera de E = 132 tier-3 + 10 calendario. Ocho pruebas pasan. Proveedores pagados y transformaciones siguen requiriendo su propia reconciliación; no se declara exhaustivo. |
| PS4/PS5 manifiesto conjunto | PILOTO PARCIAL | PS4 acepta solo la subpoblación de diez transformaciones: 50/50 unidades y 1,050 filas. PS5 conserva un contraste EURUSD Y_l@24h que falla el naive pareado (0.002228 frente a 0.002199). Falta ledger de preparación de 366 filas, evidencia PS3-R/PS4 requerida y comparación common-K; no se libera selección ni estrategia. |
| ARCH/E1/H-CORE/NEAT/RL | AÚN NO ELEGIBLES | Ejecutar en orden canónico solo tras manifiesto final; H-CORE después de E1 y NEAT al final sobre representación congelada. |
| Walk-forward semanal | IMPLEMENTACIÓN PARCIAL | Contrato BW01-BW18 y núcleo de calendario/modos implementados; faltan resolver población as-of, firewall test completo, adapters de entrenamiento y unión al runtime semanal. Los folds estáticos retenidos no cambian de clase. |
| Literatura | COLA NO VIVA | No hay runner científico observado en Dragon; el STATUS anterior describía una cola Traffic, no un proceso vivo en este corte. |

## Trabajo en paralelo y orden inmediato

1. **Gamma:** cancelar únicamente el waiter E sin hijo; ejecutar lane F en la 5090 a su tope medido, una celda por vez. No reintentar `dgs30`/`dprime` con un tope reducido.
2. **CPU, causal-inference:** revisión read-only de los tres lotes completada bajo `48ae17c`; preservar originales y mantener los 0 casos identificados como resultado del gate, no como rechazo de features. La siguiente acción es cerrar fuentes/transformaciones y reabrir análisis cuando haya evidencia de assumptions verificable.
3. **CPU, cobertura de selección:** ledger de fuentes/transformaciones incorporado desde Retsu `749ba6a8` como `292e13cc`, sin diferencia de árbol entre esos dos commits. Continuar las disposiciones de 87 low-priority, 142 fuera de E, transformaciones PS4 y fuentes pagadas/no disponibles. Sin omisiones implícitas.
4. **Dragon:** batch 002 autenticado está presente y una celda real terminó. Continuar allí la cola pesada E de 8,000 MiB con claim global y sin duplicar celdas de Gamma.
5. **Omega:** mantener solo verificaciones CPU pequeñas y reportes; no ocupar la 4070 de escritorio para un job largo.
6. **CPU negocio:** completar protocolo semanal, población as-of y firewall de
   validation/test; luego adaptar forecasting, heurística y RL a una identidad
   común de semana/cutoff sin detener PS3-R.

## Resultado/ETA

Hay un piloto predictivo PS5 de desarrollo, no exactitud financiera ni feature seleccionada: su mejor brazo no supera al naive pareado. La reconciliación reduce incertidumbre de cobertura, pero no decide selección. Al ritmo observado, las colas E/F pendientes representan aproximadamente 80 horas de GPU compartida; no es una ETA de PS5 completo ni de selección final.

Progreso visual anterior: [PROGRESS_20261003T1634Z.png](../../audits/evidence/canonical_20261003/PROGRESS_20261003T1634Z.png). Revisión causal: [REPORT](../../audits/evidence/canonical_20261003/laneC/reanalysis_48ae17c/REPORT.md). Cobertura de features: [README](../../audits/evidence/canonical_20261003/coverage_reconciliation/README.md). Fuentes y transformaciones: [REPORT.json](../../audits/evidence/canonical_20261003/source_transform_coverage/REPORT.json).
