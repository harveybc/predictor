# Retsu → Musashi — STEP 12: elegir modo, no inventar uno

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento router, MoE ni RL. No abro STEP_13. No toco GPU. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** `STEP_12_ADAPTIVE_INFORMATION_QUALITY_ROUTING_LINK_ADAPTATION_FINAL.md` (2996 líneas; SHA-256 `171deeda6dc61f334397179b1da027cc3827affe2884dfea318d531e27305649`).

Harvey: esto es muy interesante. De acuerdo. Es el primer paso que *usa* los medidores de 03–11 en vez de añadir otra transformada. Por eso no se implementa primero.

---

## 0. Una línea

Si la calidad de la información cambia, ¿un router causal elige entre **modos ya validados**, o un campeón estático basta?

Homólogo: AMC / Goldsmith–Chua. No es MQAM. Es \(\pi(q_t)\in\mathcal A\) con \(\mathcal A\) finito y congelado.

La cantidad que decide el paso:

\[
G_{\mathrm{oracle}}=P_{\mathrm{oracle}}-P_{\mathrm{static}}.
\]

Si \(G_{\mathrm{oracle}}\approx 0\), un modo estático basta y 12 cierra. Eso es tesis, no fracaso.

El router **no inventa** denoiser, extractor, detector ni predictor. Solo elige paquetes que ya pasaron su compuerta. Hoy esa banca está vacía. 12A no arranca hasta que existan **al menos dos** modos (p. ej. crudo vs denoise de 03, si H2 no es nulo).

---

## 1. Qué no recortaría

1. **12 ≠ 13.** 12 = *cuál* modo. 13 = *cuánto* presupuesto a cada rama activa.  
2. **12 ≠ TTA (08).** Aquí \(\theta_a\) está congelado; cambia \(a_t\).  
3. **12 ≠ MoE.** Time-MoE / Moirai-MoE / Pathformer son comparadores. El primero es umbral sobre IQS.  
4. **IQS causal.** \(q_t=f(X_{\le t})\). No \(Y_{t+h}\). Labels del router pueden usar pérdidas históricas; el input online no.  
5. **Oráculo primero.** Si el oráculo no gana, no hay especialización ruteable.  
6. **Escalera:** estático → oráculo → regla → meta chico → gate → RL tarde.  
7. **Abstenerse** como modo seguro (no-trade / no señal). Analogía con Chow/Geifman de *L2*, **otro objeto**. No se mete en el PDF de La Sabana.  
8. **`predict_with_uncertainty`** existe (p. ej. `predictor_plugins/common/base.py`). Reusar MC; no un modelo de incertidumbre nuevo. `state_regime_entropy` de agent-multi: consumir *si* esa tubería está validada; no duplicar; el router no actúa.

No usar el precio crudo como atajo de IQS.

---

## 2. Recorte de hipótesis: doce es mucho

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H12.1 | El oráculo cambia de modo con \(q_t\) | Un modo gana casi siempre |
| H12.2 | Un router causal gana al campeón estático, misma banca | No hay mejora OOS estable |
| H12.4 | Un umbral captura fracción útil del oráculo | Las reglas no valen |
| H12.5 | Lo aprendido solo si gana a la regla, mismo costo | El gate no aporta |
| H12.11 | RL innecesario si el reward es miope | (default: no RL) |
| H12.12 | No hay un \(q\) universal | Un solo medidor rutea todo |

H12.6–H12.8 (missingness, OOD, sync) esperan que 10/11 existan como *modos*. Chatter/histéresis: diagnóstico, no tesis.

TimeRouter / FreqMoE 2026: papel. No se reimplementan.

---

## 3. Compuertas

| Gate | Qué | Si falla |
|---|---|---|
| **12A** | \(\ge 2\) modos con heterogenidad real (oráculo no trivial) | 12 no existe todavía |
| **12B** | \(q_t\) predice qué experto gana, más que azar | Especialización no ruteable |
| **12C** | Regla ≥ estático | Se queda el campeón |
| **12D** | Aprendido > regla | Se queda la regla |
| **12E** | Fallback OOD no empeora | Sin modo agresivo en OOD |
| **12F** | Histéresis / dwell; no chatter | Se endurece el switch |
| **12G** | Test una vez | — |

Si 03–11 son todos nulos, \(\mathcal A=\{\mathrm{crudo}\}\) y 12A ni abre. Eso es coherente: primero el banco CPU de ruido.

---

## 4. Dónde vive, si algún día

No en el extractor, no en un plugin de denoise. En el borde que ya sabe qué modo está disponible (orquestación / pipeline de predictor). Tú eliges el repo cuando haya dos modos. Ahora: nada.

CPU. No campañas. No SAC como router.

---

## 5. Entrevista

El correo sigue siendo el tema 1. Si tiran a “adaptar el enlace”: *solo entre modos ya medidos; si el oráculo no gana, un modo basta.* Abstención de L2 \(\neq\) este fallback.

STEP_13 cuando caiga en Downloads.

— Retsu
