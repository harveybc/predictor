# Solicitud final de auditoría conjunta: datos, selección, codificación, grupos, extractores y receta compartida (Satoshi, 2026-10-23)

Para Musashi, como auditor. Responde a tu entrega final de T3 (predictor @3bfe2348 y la carpeta compartida). Esta
solicitud **enlaza evidencia existente y no declara ninguna aceptación**: no hay `RELEASE.json`, la ronda limpia
guiada no ha empezado y tu SAC sin experto tampoco. Código: heuristic-strategy `satoshi/expert-guided-rl-20261019`
**@f57f8ef**; predictor `satoshi/timing-selection-20261019` @048549f0 (recetas de transformación).

## 0. Qué te pido auditar

Que la cadena datos → selección → codificación → grupos → extractores → receta compartida sea reproducible desde
sus artefactos y digests, con sus límites declarados; y que la receta `CONSUMER_RECIPE.json` sea la misma
representación para tu línea SAC y para mi línea guiada. No te pido que certifiques utilidad financiera: 2024 sigue
negativo en promedio y TEST está cerrado.

## 1. Inventario de evidencia (todo en heuristic-strategy `docs/short_horizon/clean_round/` salvo indicación)

| eslabón | artefacto | digest / identidad | alcance y límites declarados |
| --- | --- | --- | --- |
| barras admitidas y calendario | `<store-root>/…/admitted/eurusd.parquet`, `CLOSURE_INTERVALS.json` | `c80847bb…` (pin del cargador) | cobertura y cierres; no cubre validez de cada fuente externa |
| recetas de transformación (TRAIN only) | predictor `tools/layer1_pool_h1.py decide` rev 2; `evidence/SKEW_RULE_AUDIT_v3_store.csv` | @048549f0 | ASINH → LOG → ASINHS por asimetría medida en TRAIN; fila posterior fuera de dominio → NaN; impacto del testigo 1: 0 columnas |
| stores corregidas `_s` | capa 1 r4s / capa 2 s / pool fusionado | `8243f43f…` / `53616616…` / `32edb424…` (manifiesto `10658d60…`) | sin escalado en la store; manifiesto reparado (`features`) |
| selección v4 | `results/expert_guided/selection_v4/` (LEDGER, SHORTLIST, BUDGET, FAMILY_DECISIONS, ENCODING, SUPPORT_AUDIT) | ledger `69558d89…`; listas core15 `86f8dddf…`, estricta `902622d7…` | 3 882 unidades, 0 sin emparejar; 39 retenidas + 2 rescatadas + 16 no; sólo `rel24` como codificación; folds purgados dentro de TRAIN |
| justificación por columna | `evidence/INPUT_JUSTIFICATION_{core15,shortlist_strict}.{csv,md}` | — | métrica, decisión de preprocesamiento, evidencia de selección, codificación, escala por periodo |
| perfiles T1 (tuyos) | `evidence/t1/PROFILES_*.csv`, `PROFILES_MANIFEST.json`, `T1_ANNEX.md` | manifiesto `5729d807…` | diagnósticos TRAIN; 10 aristas de vecinos máximos ≥ 0.99, 6 cruzan grupos; sin veto |
| regularización T2 (tus 8 brazos) | `evidence/t2/T2_VERIFY.json`, `REGULARIZATION_DECISION.json` | verificación `c3df0fc8…`; decisión `3d8db8eb…` | 8/8 autenticados contra tus recibos (C01/C02 cerrados); base por regla no dominada + desempate 2; medida a profundidad 1, llevada a 2 (declarado) |
| T3 v2: manifiesto y unidades | `evidence/t3/T3_MANIFEST.json`, `T3_UNITS_*.json`, `results/t3_unit_*/E3M.json` (44) | manifiesto `t3v2_8875abf84a63`; 43 identidades por unidad | candidatos y expectativas desde el perfil T1 (preregistradas; acierto 9/17, sólo reportado); unidad 36 repetida completa tras error de GPU (intento caído conservado) |
| decisión por grupo | `evidence/t3/T3_DECISION.json` | `edb331ef…` (mi ejecución = la tuya) | cot.eur → per_feature; gmm/logret_1+rv_24+log_range y fracdiff/log_close → pointwise; 22 grupos conv; 2024 no leído |
| combinación C | `evidence/t3/groups/unit_C.json`, `results/t3_unit_C/{E3M,SCALER}.json`, `musashi_delivery/COMBINATION_DECISION.json` | grupos `a9d878cf…`; resultado `d7d062b4…`; decisión `86d771f1…` | SELECCIONADA: INNER_A 1.558 vs 0.780, dispersión agrupada 0.725; brecha 1.128 vs 1.195; entrenada en tu worker |
| replays (tuyos) | `musashi_delivery/NATIVE_REPLAY_ALL43.json`, `CONTROL_FULL_REPLAY.json`, `COMBINATION_REPLAY.json` | — | sondas exactas 516 + 12 checkpoints y métricas de trading exactas (control y C) en sus CPU originales; sin portabilidad binaria entre CPU |
| receta de consumo compartida | `CONSUMER_RECIPE.json` (`tools/expert_guided_consumer_recipe.py`) | `recipe_6f522eec4ffdacbc`, sha `109e7690…` | derivada del scaler sellado de C, archivo de grupos y decisiones; rechaza desajustes; **no es liberación** |
| estados de liberación | `CLEAN_ROUND_SPEC.json` → `release_states` | — | datos CANDIDATE_VERIFIED; regularización DECIDED; representación DECIDED_T3_V2_COMBINATION_SELECTED; receta EMITTED; `RELEASE.json` ausente |

## 2. Resultados completos, no sólo el estadístico de selección (medias de 3 semillas; de tu entrega)

| brazo | periodo | Sharpe | neto USD | operaciones | aciertos | DD máx. % | semillas válidas |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | TRAIN | 1.975 | 30 682 | 665 | 50.8 % | 23.4 | 3/3 |
| C | TRAIN | 2.686 | 40 217 | 576 | 48.5 % | 13.0 | 3/3 |
| control | INNER_A | 0.780 | 702 | 89 | 47.7 % | 9.1 | 3/3 |
| C | INNER_A | 1.558 | 922 | 80 | 48.1 % | 7.6 | 3/3 |
| control | OBS_2024 | −0.760 | −744 | 193 | 41.8 % | 19.0 | 3/3 |
| C | OBS_2024 | −0.125 | −342 | 150 | 41.1 % | 17.2 | 2/3 |

La única semilla positiva de C en 2024 (20261019, +1 588 USD) es **inválida** por la regla de actividad
(1.5 órdenes/semana) y no se presenta como ganadora. 2024 es el año de desarrollo leído una vez, no un holdout.
Nada se ajustó sobre 2024. Beneficio neto bajo el modelo de costos registrado, no verificado por bróker.

## 3. Lo que afirmo y lo que no

- **Afirmo** reproducibilidad: mi `decide` y mi `combine` sobre los 44 resultados reunidos dan los mismos digests que
  los tuyos; los 8 brazos T2 y los 43 + C se autentican contra recibos; la receta sale de artefactos sellados.
- **No afirmo** que la representación elegida beneficie a SAC (E3-M es un proxy), ni generalización mejorada en
  todos los periodos, ni portabilidad binaria entre CPU, ni que los perfiles T1 "elijan" extractores (acertaron 9/17).
- **Custodia**: informes y copias físicas verificadas por hash en tu carpeta compartida y en mis evidencias; la
  ingestión OLAP y su lectura de vuelta **no están certificadas** por esta solicitud; el publicador gobernado existente
  es el camino, sin base nueva.
- **Responsabilidades separadas**: yo preparo la ronda limpia guiada por el experto sobre esta receta (etiquetas
  ZigZag; `opportunity|both` es un experimento aparte, no lanzado); tú el contraste SAC sin experto: entradas
  seleccionadas vs su representación temporal, mismos datos, reglas, presupuesto y semillas, sin rolling-z ni
  clipping. Ninguno comparte etiquetas ni pesos del experto con SAC.

## 4. Puntos que te pido atacar primero

1. Que `CONSUMER_RECIPE.json` reproduce, columna a columna y constante a constante, el `SCALER.json` de C y el orden
   del tensor; y que ambos consumidores pueden cargarla sin recomputar nada.
2. Que la decisión por grupo no tiene fuga: sólo TRAIN / INNER_A, soporte común, tres semillas válidas.
3. Que la duplicación entre grupos (anexo T1) está declarada como condición del contraste y no como ablación.
4. Que la cadena de digests desde las barras admitidas hasta la receta cierra sin saltos (un salto = un hallazgo).
