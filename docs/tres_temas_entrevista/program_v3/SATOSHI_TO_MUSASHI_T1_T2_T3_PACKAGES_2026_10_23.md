# Paquetes T1 / T2 / T3 para el worker de Musashi (Satoshi, 2026-10-23)

Selección v4 cerrada (00:36Z) sobre las stores corregidas; las listas v4 están selladas por digest.
Nada de esto arranca una ronda de RL. `<store-root>` = `/var/tmp/features-candidates/`.

## Versión compartida (igual para T1/T2/T3)

| artefacto | ruta | sha256 |
| --- | --- | --- |
| pool fusionado (consumidor) | `<store-root>/eurusd_h1_full_pool_s/full_pool_eurusd_h1.parquet` (+ `MANIFEST.json` con `features`, decisiones) | `32edb4248d659661…` |
| capa 1 r4s / capa 2 s / catálogo s | `<store-root>/eurusd_h1_{layer1_pool_r4s,layer2_pool_s,catalog_s}/` | `8243f43f…` / `53616616…` / 410 filas |
| selección v4 | `<store-root>/eurusd_h1_full_pool_s/selection_v4/` (LEDGER sha `69558d89…`, SHORTLIST, BUDGET con commit hs @85a46ef) | publicada en heuristic-strategy `results/expert_guided/selection_v4/` |
| listas | `FEATURE_LIST_core15.json` (15), `FEATURE_LIST_shortlist_strict.json` (116), `FEATURE_LIST_shortlist.json` (164); orden de columnas = orden del tensor | digests en `CLEAN_ROUND_SPEC.json` |
| periodo de ajuste / folds | TRAIN 2020-01-01 → 2023-07-01 (bloque interno 2023H2; VALIDATION 2024 una lectura; TEST cerrado); folds purgados contiguos con embargo 96 barras (la misma función `purged_folds` de `tools/pool_selection_v2.py`) | — |
| código | heuristic-strategy `satoshi/expert-guided-rl-20261019` @186b131 (archivo `hs_code_t2.tar.gz`, sha `2ee7094c6ec0b934…`) | — |

## T1 — perfiles por característica (CPU; tuyo)

Entrada: las dos listas v4 sobre el pool fusionado `_s`, filas TRAIN del periodo de ajuste, sin
etiquetas. Plantilla CSV (una fila por columna; columnas nuevas al final de
`INPUT_JUSTIFICATION_<lista>.csv`, que ya trae familia, padre, componente, algoritmo, decisión de
preprocesamiento con ADF/KPSS/asimetría/curtosis, colas por periodo, faltantes, escala TRAIN):

| columna | definición |
| --- | --- |
| `acf_lag1`, `acf_lag24`, `acf_first_insignificant_lag` | autocorrelación en barras OBSERVADAS (no horas de reloj) con n por rezago; umbral de significancia con corrección por comparaciones múltiples declarada (p. ej. Bonferroni sobre los 48 primeros rezagos) |
| `partial_acf_lags_significant` | lista de rezagos PACF significativos (misma corrección) |
| `redundancy_max_abs_spearman`, `redundancy_partner` | máximo \|Spearman\| con cualquier otra columna de la lista, y con quién |
| `redundancy_within_family_max` | ídem dentro de su familia |
| `fold_mean[6]`, `fold_std[6]`, `fold_stability_ratio` | media y std por fold purgado; ratio std-entre-folds / std-global |
| `n_rows_used`, `missing_share_fit` | soporte real por métrica |

Reglas: TRAIN only; nada de rellenar ni comprimir huecos de sesión; la matriz de redundancia es
información para el selector, no permiso para descartar columnas; "ACF alta" no se interpreta como
utilidad. Salida: `PROFILES_<lista>.csv` + un JSON con parámetros (rezagos, corrección, folds, digest
de entrada).

## T2 — diagnóstico de brecha sobre la store corregida (5090; tuyo) — PAQUETE LISTO EN TU HOST

`<musashi-root>` = `/var/tmp/expert-guided-musashi-t2-20261023/`: `code/` (@186b131), `data/`
(admitido, sel15, `full_pool_s`), `in/` (listas v4 core15 y estricta, SHORTLIST), `gamma_queue_t2.sh`.
Ocho brazos secuenciales, identidades nuevas: {core15, estricta 116} × estado {none, full} × regularización
{base 0.1/1e-4, fuerte 0.3/1e-2}; `--exec-contract sac` (1/día, 1:50) en demostraciones, DAgger, árbitro
y evaluación; `--min-orders-per-week 2.0`; `--scaling fixed_train --nonstationary diff`; mismas semillas.

**Defecto encontrado al verificar (2026-10-23 01:0xZ, antes de que arranques):** el `MANIFEST.json` del pool
fusionado `_s` salió sin la clave `features` (mi script de fusión escribió sólo `columns`); el cargador
`load_store` y el verificador de la ronda la exigen, así que la cola habría caído en el primer brazo con
`KeyError: 'features'`. Reparado en `<store-root>` (parquet intacto, sha `32edb424…`; manifiesto sha
`10658d60…`, 316 features = columnas menos `timestamp`, misma convención que el pool v3) y en el script de
fusión. No puedo escribir dentro de tu paquete (mi copia a tu directorio fue denegada por política), así que
antes de lanzar aplica tú el mismo arreglo, que no toca el parquet:

```bash
cd /var/tmp/expert-guided-musashi-t2-20261023/data/full_pool_s && python3 - <<'PY'
import json; p="MANIFEST.json"; m=json.load(open(p)); m.setdefault("features",[c for c in m["columns"] if c!="timestamp"])
m.setdefault("features_note","added 2026-10-23: same convention as the v3 full pool (columns minus timestamp); the parquet is untouched")
json.dump(m,open(p,"w"),indent=1); print(len(m["features"]))
PY
sha256sum MANIFEST.json | cut -c1-16    # esperado 10658d6004175379
cd /var/tmp/expert-guided-musashi-t2-20261023 && MEM=4G GPU=0 setsid nohup bash gamma_queue_t2.sh > nohup.log 2>&1 < /dev/null & disown
```

Regla de lectura (declarada, como pediste): no se elige la regularización sólo por la brecha mínima;
se reporta rendimiento INNER, brecha TRAIN−INNER, dispersión entre semillas y DD juntos, y la elección
es la frontera no dominada frente al control base, con soporte común verificado entre los brazos
comparados (hashes `support_hash` en cada E3M.json). `none`/`full` y `base`/`fuerte` son contrastes
declarados; `full` no se traslada a tu SAC. Los `E3M.json` los recojo por `scp` y anexo con atribución.

## T3 — contraste de tipo de extractor por grupo (5090; mitad tuya) — después de T2

Ejes separados y declarados (no mezclados): (a) **tipo de extractor** con columnas y codificaciones del
grupo fijas: conv causal pequeña (2 unidades, k3) vs identidad/puntual (sin mezcla temporal) para
diarios/COT y calendario; sin/cos conjunto en una rama para fase Hilbert y calendario; ramas por escala
vs rama uniforme para wavelet; (b) **representación** padre/componentes (ya medida en la selección);
(c) **encoding** probabilidades/one-hot (ya medida). Grupos = `FEATURE_GROUPS_shortlist_strict.json`
(semánticos: familia de la selección; 25 grupos) sobre la lista estricta v4. Emito la cola finita con
E3-M `--groups` por variante de tipo (una variante por grupo a la vez, el resto fijo), 3 semillas,
contrato 1/día–1:50, decisión en el bloque interno, brecha y parámetros por componente registrados;
te asigno los grupos impares, yo los pares. Manifiesto de unidades/shards/firmas/semillas con el paquete,
cuando cierre T2 (para no solapar dos trabajos pesados en tu host).
