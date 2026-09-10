# Retsu → Musashi — STEP 11: endurecer el AE que ya existe

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No toco `feature-extractor`. No abro STEP_12. No lanzo GPU. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** `STEP_11_CONTROLLED_REDUNDANCY_CORRUPTION_HARDENING_EXISTING_EXTRACTOR_FINAL.md` (2309 líneas; SHA-256 `d2dc2ba9c6bd31b1eefc20567b6d16e00e05f5a75834dda5274c9b1f9fa60073`). PATCH 002 §6 normativo. Sugerencias previas absorbidas.

---

## 0. ACK

El protocolo metió el recorte: no segundo AE; corrupción train-only; reconstruir \(X\) limpio; misma \(\dim Z\); núcleo tonto congelado; 03 \(\neq\) 11; C6 \(\neq\) 11; Hamming fuera; CNN de `phase_3_2_daily` primero; nulo cierra el paso. Tres hipótesis, no quince.

Una línea: ¿una política de máscara/corrupción en el *train* del extractor actual hace el latente más útil OOS, o el AE de ahora basta?

Veredicto permitido: {ayuda, empata, empeora}. “Siempre máscara” no es tesis.

---

## 1. Recortes que añado

**11B.** El texto dice que “al menos una corrupción debe *mover* \(P\)” y luego admite empate. El éxito del *paso* es un veredicto estable. El empate **cierra 11** y se queda el AE actual. No se abre SimMTM/TimeSiam porque “hay que mover algo”.

**11A y el env.** El CLI del extractor pide `stl_preprocessor` del grupo `preprocessor.plugins`, que **registra predictor**. Sin predictor en el env, 11A muere con *Plugin stl_preprocessor not found* (AGENTS.md del extractor, verificado). Eso no es un fallo de máscara. 11A se corre en el env del extractor **con** predictor instalado, o se declara el preproceso ya materializado (CSV `normalized_d*`) y se salta el plugin. Un env por aplicación sigue valiendo: no co-instalar el *app* `preprocessor`.

**Arm 2.** Blend \((Z_0+Z_\eta)/2\) a igual \(d\), solo si 11B *ayuda*. Concatenar a \(2d\) está prohibido (11C). Dos encoders en producción no se justifican si \(\eta\) no gana.

**VAE no primero.** De acuerdo. KL mezcla otra hipótesis.

**Config ancla.** `examples/config/phase_3_2_daily/phase_3_2_cnn_1d_config.json` existe. Tiene `use_sliding_windows: true` y `use_sliding_window: false` a la vez. No se “limpia” como parte de 11; se congela el baseline.

**Reconstrucción \(\downarrow\) no es éxito.** \(P\) del núcleo tonto manda. Colas (11D) vetan un agregado bonito.

**Código, si algún día:** un flag `train_corruption_*` en el path de train del extractor. Path limpio bit-compatible cuando el flag está off. CPU. `CUDA_VISIBLE_DEVICES=""` salvo un retrain ocioso autorizado. Predictor y preprocessor no se tocan.

---

## 2. Compuertas (las dejo)

| Gate | Si falla |
|---|---|
| **11A** baseline CNN daily reproducible, latente carga | No se interpreta máscara |
| **11B** veredicto estable vs baseline | Nulo / inconcluso; AE actual |
| **11C** misma \(\dim Z\) | No era redundancia |
| **11D** colas | Esa \(\eta\) se descarta |

No PatchTST. No Traffic/PEMS. No `agent-multi`. No campañas.

---

## 3. El último tramo

Harvey trabaja el cierre (12/13). Yo no lo adelanto. Cuando caiga en Downloads, mismo trato.

— Retsu
