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
(admitido, sel15, `full_pool_s`), `in/` (listas v4 core15 y estricta, SHORTLIST), `queue_t2.sh` (el script de la cola del paquete).
Ocho brazos secuenciales, identidades nuevas: {core15, estricta 116} × estado {none, full} × regularización
{base 0.1/1e-4, fuerte 0.3/1e-2}; `--exec-contract sac` (1/día, 1:50) en demostraciones, DAgger, árbitro
y evaluación; `--min-orders-per-week 2.0`; `--scaling fixed_train --nonstationary diff`; mismas semillas.

**Manifiesto `_s` sin `features`:** defecto mío (el script de fusión escribió sólo `columns`), reparado en
`<store-root>` (manifiesto `10658d60…`, parquet intacto `32edb424…`) y **ya corregido por Musashi dentro de su
paquete** (copia previa en `ops/preflight/`); no hay nada más que copiar.

**Split explícito (tu punto):** `queue_t2.sh` (el script de la cola del paquete) no pasaba `--split-contract` y el parser por defecto usa
`dev3y`. El comando efectivo de cada unidad lleva ahora `--split-contract consolidated` (TRAIN 2020-01 → 2023-07
de ajuste, bloque interno 2023H2 = `INNER_A`, 2024 una lectura, TEST cerrado); el resultado **no** es comparable
con el diagnóstico v3 (dev3y). El listado tipado de las 8 unidades (orden, marcador, comando efectivo, identidad =
sha256 del comando, archivo y campos esperados) está en `T2_UNITS_2026_10_23.json` junto a este documento: un
`T2_DONE` en el LOG no prueba nada; cada `E3M.json` se verifica por sus campos y por la identidad de su comando.
Tú lanzas con tu wrapper local (GPU por UUID; el `GPU=0` del ejemplo era sólo un ejemplo); yo no corro copia
alguna de T2 ni toco los pins una vez iniciada.

Regla de lectura, ahora **ejecutable** (tu punto 3): `tools/expert_guided_clean_round.py regularize --spec
CLEAN_ROUND_SPEC.json --results <musashi-root>/out --out REGULARIZATION_DECISION.json [--apply]`. Lee sólo los
cuatro brazos `agent_state=none` (los `full` son tu contraste declarado, nunca fuente); por lista exige tres
semillas válidas y el mismo `support_hash` TRAIN/INNER_A entre los dos ajustes; vector por ajuste = (Sharpe
INNER_A ↑, brecha TRAIN−INNER_A ↓, desviación entre semillas ↓, DD máximo INNER_A ↓); gana el único no dominado;
si quedan dos, desempate declarado antes de leer nada: (1) mayor Sharpe medio si la diferencia supera la
desviación agrupada de semillas, (2) menor brecha media, (3) base; sin lista válida común → `DEFAULT_BY_VALIDITY`,
que nunca se aplica. `OBS_2024` no se lee. El spec lleva `candidates` {base 0.1/1e-4, fuerte 0.3/1e-2} y la
observación v3 queda marcada como histórica, no fuente. Los `E3M.json` los recojo por `scp` y anexo con atribución.

## T3 — contraste de tipo de extractor por grupo (5090; mitad tuya) — manifiesto EMITIDO, fits tras T2

Ejes separados y declarados (no mezclados): (a) **tipo de extractor** con columnas y codificaciones del
grupo fijas: conv causal pequeña (2 unidades, k3) vs identidad/puntual (sin mezcla temporal) para
diarios/COT y calendario; sin/cos conjunto en una rama para fase Hilbert y calendario; ramas por escala
vs rama uniforme para wavelet; (b) **representación** padre/componentes (ya medida en la selección);
(c) **encoding** probabilidades/one-hot (ya medida). Grupos = `FEATURE_GROUPS_shortlist_strict.json`
(semánticos: familia de la selección; 25 grupos) sobre la lista estricta v4. Emito la cola finita con
E3-M `--groups` por variante de tipo (una variante por grupo a la vez, el resto fijo), 3 semillas,
contrato 1/día–1:50, decisión en el bloque interno, brecha y parámetros por componente registrados;
te asigno los grupos impares, yo los pares. **Manifiesto ya emitido** (`tools/expert_guided_t3_manifest.py emit`;
heuristic-strategy `docs/short_horizon/clean_round/evidence/t3/T3_MANIFEST.json`, identidad `t3_7a056b35acd8`):
26 unidades = control (todo `conv`, id 0, mía) + 25 grupos de la lista estricta v4 con ids estables por el orden del
archivo de grupos (digest pinneado); impares tuyos (13), pares + control míos (13); tipo de contraste por grupo:
`pointwise` (kernel 1, sin mezcla temporal dentro del extractor) para calendario, eventos, COT, diarios, factores,
fuerza y grupos de una columna; `per_feature` (un extractor por columna: ramas por escala en wavelet, por
componente en Hilbert) para el resto; columnas y codificaciones del grupo fijas; ventana [24,1] fija;
regularización = **dependencia explícita** del `REGULARIZATION_DECISION.json` de T2 (`queue --decision …` se niega
a emitir sin él, así que no hay que relanzar la cola por una decisión manual). Cada unidad lleva su identidad y
los campos esperados; un resultado negativo también cierra la unidad; nada se declara óptimo para SAC desde este
proxy. Las colas por worker se emiten cuando exista la decisión T2.
