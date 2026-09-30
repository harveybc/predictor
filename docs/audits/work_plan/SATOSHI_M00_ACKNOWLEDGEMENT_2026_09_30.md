# Acuse de despacho — optimización modular concurrente

**Fecha:** 2026-09-30T22:50:30Z
**De:** Satoshi III (Mujuro Utsutsu), líder técnico sucesor
**Órdenes:** `SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` (master `dc72170e`) y
`SATOSHI_POST_CONSOLIDATION_2026_09_30.md`, ambas en vigor.
**Publicado dentro de la ventana de quince minutos exigida por §8.**

## 1. Estado observado antes de despachar (leído, no supuesto)

| anfitrión (rol) | GPU (UUID) | estado | RAM libre | losa no recl. | reservas / ámbitos |
|---|---|---|---|---|---|
| coordinador, escritorio del dueño | 4070 `GPU-612d1e0c…` | **Traffic h96 s2021 entrenando**, 98 %, 75 °C, **reloj 1500 / 3105 MHz** (tope tras ralentización; `codex-omega-gpu-clock-restore.service` activo esperando el fin de la celda) | 15 / 30 GiB | 0.60 GiB | 1 / 1 |
| obrero secundario | 4090 `GPU-a8bd1b2c…` | **Traffic h96 s2023 entrenando**, 98 %, **82 °C, 1305 MHz, razón de estrangulamiento 0x20 = ralentización térmica SW** | 15 / 30 GiB | 2.37 GiB | 1 / 1 |
| preferido (5090) | 5090 `GPU-a9f35631…` + 5070 Ti | ambas **inactivas y frías** (36 / 28 °C) | **2 / 14 GiB** | **5.48 GiB, creciendo desde el 2026-09-22** (4.38 → 4.93 → 5.48) | 0 / 0 |

**Adoptadas, sin tocar:** las dos celdas de Traffic. Seed 2022 ya terminó (3 906 s; MSE/MAE normalizados
0.3753611147 / 0.2512390912 contra 0.375 / 0.251 publicados; ingenuo pareado 1.0772232192) — **una celda, no
el resultado de tres semillas**. Nada se reinicia, nada se duplica, ningún código se tira debajo de un hijo.

**Código heredado localizado y adoptado:** `codex/modular-stack-20260930` `556c5f3e` contiene motor
`1b76981a`, evaluador `1b9daf6c` y preentrenamiento `621570e1` (verificados como ancestros); motor en
worktree `predictor-modular-temporal` `caa0b8bb`; feature-eng `codex/feature-metrics-audit-20260930`
`dce037b`; doin-node `7411e5bf` en `feat/predictor-candidate-bridge-20260930`; AE del núcleo
`b8fde409`; ECL igualado `9c3321d9`; preflight LTS `99184e2`. Los cuatro agentes heredados ya
retornaron; ninguno se reemplaza.

## 2. Asignaciones reales — seis carriles, despachados y con manejador de tarea vivo

| carril | rama (worktree propio) | base | entregable | acuse |
|---|---|---|---|---|
| **M01** ensamblaje del modelo | `satoshi/m01-model-assembly-20260930` | `556c5f3e` | plugin modular opt-in, facade `predictor.plugins` intacta, MS01–MS08 con pruebas de comportamiento en entorno instalado | lanzado; bloque de despacho pendiente en su primera respuesta |
| **M02** preentrenamiento | `satoshi/m02-pretraining-20260930` | `556c5f3e` | AE de rama → representación fusionada acotada → AE del núcleo → regímenes; parada temprana en cada etapa (MS09–MS11) | ídem |
| **M03** inventario de características | feature-eng `satoshi/m03-feature-inventory-20260930` | `dce037b` | índice de cobertura con denominador, perfiles sólo-entrenamiento en lotes acotados, protocolo de selección (MS14) | ídem |
| **M04** ejecución DOIN — **prioridad** | doin-node `satoshi/m04-doin-execution-20260930` | `7411e5bf` | contrato con el API de optimizador existente, cola persistente finita e incumbente, piloto de coste real, puntuación independiente por checkpoint (MS13) | ídem |
| **M05** adaptador papel | lts `satoshi/m05-paper-adapter-20260930` | `main` | adaptador inferencia/características/acción para las rutas LTS, replay de entradas grabadas, humo papel sólo lectura (MS15) | ídem |
| **M06** evidencia y recursos | `satoshi/m06-evidence-resources-20260930` | `dc72170e` | celdas adoptadas, diagnóstico del anfitrión 5090, mesa de turnos GPU, `STATUS.json` cada 5 min, `PROGRESS.png`, `RETURN.md`, `RESULTS/` | ídem |

Cada carril lleva un manejador de tarea del arnés vivo y notificará su bloque de despacho (worktree, tip de
origen, primer comando, prueba de aceptación, presupuesto, bloqueo) como primera respuesta; esos bloques se
anexarán aquí tal cual lleguen. **Ningún carril se anuncia como activo sin ese acuse.**
Los dos carriles de la orden de consolidación (revisión sin ejecutar del código remoto de clasificación;
correcciones de soporte de la estrategia) se lanzan **al retornar los dos primeros de estos seis**, para
respetar el techo inicial de seis y no matar de hambre ni experimentos ni escritorio.

## 3. Cola inmediata de candidatos

1. **Primer turno GPU admisible** → M04, piloto de coste del candidato modular completo (R0 pedido), y a
   continuación el lote finito de candidatos DOIN persistido. El turno lo asigna M06 cuando termine la
   primera celda de Traffic; la 5090 sigue **inadmisible** hasta que su anfitrión se diagnostique.
2. Mientras: M04 construye y prueba el contrato y el piloto de coste en CPU (0.887 GiB medidos; el rechazo de
   16.64 GiB fue presión de reservas agregadas, no huella del modelo).
3. Donantes R1/R2 de M02 → candidatos pareados en la cola de M04 cuando estén listos.
4. Las dos celdas de Traffic restantes se **terminan**, no se relanzan; sus recibos entran por M06.

## 4. Cadencia de reporte

- `STATUS.json` atómico cada ≤5 min (escritor acotado de M06, sin LLM en el bucle).
- Resumen cada 30 min y al completar resultados (bucle propio del orquestador).
- Latido ≤60 s en cada hijo experimental vivo; M06 audita y nombra al que no lo cumpla.
- Entrega: `RETURN.md`, `STATUS.json`, `RESULTS/`, `PROGRESS.png` bajo
  `docs/audits/evidence/MODULAR_CAMPAIGN_20260930/`, todos ligados a la misma evidencia.

## 5. Lo que este acuse no afirma

Ninguna medición de exactitud nueva. Las 24 + 30 + 6 pruebas heredadas son **comprobaciones sintéticas de
componentes**, no resultados de pronóstico. Ningún modelo queda promovido a papel ni a dinero real. Los
pesos retenidos del ECL igualado siguen faltando y ese carril es aparte.

— Satoshi
