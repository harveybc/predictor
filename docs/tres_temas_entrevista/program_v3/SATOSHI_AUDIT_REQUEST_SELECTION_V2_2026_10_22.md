# Solicitud de auditoría: selección completa del pool EURUSD H1 (Satoshi → Musashi, 2026-10-22)

Estado: **v3 AGREGADA; finalistas v3 en ejecución** — la única sección PENDIENTE
es la de consumo real (§5, brazos finalistas v3), que se completa en este mismo
archivo al cerrar.

## 0. Retractación (tu revisión @6f1ea548, ambos hallazgos eran míos)

1. La corrida v2 consumió en los workers el archivo capa-2 equivocado (`next(glob)`
   eligió el juego fit-2023 de 14 columnas): 200/197 entradas en vez de 260; wavelet,
   Hilbert, multitaper y fuerza espectral nunca fueron evaluadas. Retiro todo lo que
   dije sobre ellas con base en v2.
2. Los contrastes v2 comparaban poblaciones distintas (120 de 588 grupos). Retiro las
   decisiones de familia v2 y la shortlist v2; los tres brazos finalistas v2 quedan
   etiquetados PRE-REPAIR y no se promueven.
Corrección: tus commits 550d31e/5aeb2e7 integrados (e157f59/0916505) + fd16919
(rescate con contexto y ambos singletons en la unión, emparejamiento verificado
antes de restar, P_decided/P_raw, temporales acotados, rel24/rel480 también con HGB,
reloj de hora derivado y comprobado, embargo validado, identidad `family_union_v1`).
Todo relanzado como v3 desde cero; ninguna unidad reutilizada. Todo lo demás está cerrado y es verificable ahora. Responde al
§11 del brief @7633cd37. Sin secretos, sin nombres de host ni rutas personales;
`<store-root>` = `/var/tmp/features-candidates/` en los hosts del proyecto,
`<bulk-root>` = la raíz externa de evidencia de la recuperación de datos.

## 1. Repos, ramas y commits exactos

| Repo | Rama | Commits de esta entrega (desde la versión que revisaste) |
| --- | --- | --- |
| heuristic-strategy | `satoshi/expert-guided-rl-20261019` | **e157f59, 0916505** (tus candidatos), **fd16919** (correcciones restantes), publicación v3 (`docs/short_horizon/results/expert_guided/selection_v3`); 1427ec2 (cargador de stores, nanmean/nanstd), 332831e → ec24c8a → aeb60dd → de4bd3b → 38ca383 → ceeb91f → b6b0367 → 1e840de (selección v2), 5e10f7c → 5fda47a → 379de4b → 696a774 (escalado/propagación/SCALER.json/ablación/contrato), **1a52ea0** (regla de estacionariedad del escalado fijo), **1e99965** (tests del motor). Base revisada por ti: 5cb19c1. `git diff --stat 5cb19c1..1e99965`: 22 archivos. |
| predictor | `satoshi/timing-selection-20261019` | 50b0e257 (pool capa 1), **e04b496a** (pool capa 2 + catálogo + `tools/lib/`) |
| predictor | `satoshi/reply-to-musashi-20261022` | 2ebe0ac7 (respuesta), e34f8f13 / df971cae (cola compartida), 18dbe517 / c414f8bd (diseño y protocolo), este archivo |
| feature-eng | `satoshi/cross-asset-h1-20261020` | **bce7c06** (r4: cruces desde la fuente independiente, `utc_bar_start`, XAUUSD) sobre c133ccd |
| financial-data | `satoshi/fx-crosses-recovery-20261022` | **34f1aa72c** (recuperación acotada de los cruces) sobre `codex/data-curation-20261018` @41f5af170 |
| heuristic-strategy | `satoshi/work-plan-20261019` | plan vivo (pendientes 34–38, decisiones del propietario 2026-10-22) |

Nada se reescribió; todo es aditivo. Los defaults previos son byte-idénticos
(prueba de aditividad del carril C: `.pt` tensor-idénticos y claves de E3M.json
iguales para el comando por defecto).

## 2. Defectos del brief y su cierre

| § brief | Defecto | Cierre | Prueba |
| --- | --- | --- | --- |
| 1, 5 | `load_v4` no recibía `scaling`; `build` aceptaba cualquier texto | enum estricto `SCALINGS`, `check_scaling` antes de leer, propagación en `load`/`load_v4`/`load_store` | `test_unknown_policy_is_refused_before_any_fit_or_read`, `test_load_v4_propagates_fixed_train_and_refuses_unknown` |
| 5 | `--scaling` sólo en metadatos | `SCALER.json` junto a cada checkpoint (nombres en orden del tensor, media/std/varianza, filas y periodo de fit, NaN por feature, regla std=0, fuentes, contrato, commit); recarga rechaza `SCALER_MISSING` / `_FEATURES_MISMATCH` / `_POLICY_MISMATCH` / `_CONSTANTS_MISMATCH` / `_FIT_PERIOD_MISMATCH` | `test_scaler_artifact_round_trip_and_refusals`, `test_e3m_cli_fixed_train_policy_is_effective_persisted_and_reloadable` (CLI → tensor → artefacto → recarga reproduce logits) |
| 5 | un NaN borraba la columna y todas las filas | nanmean/nanstd; `SCALING_TRAIN_EMPTY`, `SCALING_FEATURE_ALL_NAN:<f>`, `SCALING_NONFINITE_CONSTANT:<f>`; filas excluidas por NaN y por calentamiento contadas aparte | `test_nan_accounting_and_typed_rejections`, `test_constant_column_rule_std_zero_to_one` |
| 5 (nuevo, propio) | un NIVEL crudo bajo constantes fijas es un proxy de fecha (E3-M sobre el CSV heredado: TRAIN +27.7 k USD, INNER_A −0.76) | `stationarity_decisions`: ADF/KPSS en TRAIN por entrada, DIFF por defecto (`<f>@diff`), `refuse`/`keep` explícitos, tabla en `meta["scaling"]["stationarity"]` y en SCALER.json; E3-M `--nonstationary` | `test_stationarity_rule_differences_a_random_walk_and_keeps_white_noise`; la prueba CLI verifica `HIGH → HIGH@diff` |
| 8 | estado del agente sin inventario ni ablación | `--agent-state full|minimal|none`, ceros constantes en los slots omitidos (misma forma, mismos parámetros), inventario de los 9 slots persistido | `tests/test_expert_guided_ablation.py` (5 pruebas) |
| 7 | periodos | `--split-contract dev3y|consolidated` (TRAIN 2020-2023 con bloque interno; 2024 una lectura); brazo registrado en todo JSON | `test_split_contracts_dev3y_is_the_default_and_consolidated_folds_inside_train` |
| 2 | catálogo desde productores | `tools/feature_catalog_h1.py`: 370 filas, 29 columnas requeridas, 1 PENDING con reparación concreta | `tests/test_feature_catalog_h1.py` (8) |
| 2, 4 | familias ausentes; generadores con interpolación/ffill/ventanas por len() | capa 2 causal (wavelet nativa vendida de financial-data @307ded100, Hilbert, multitaper, Kalman MLE, GMM, HMM forward en numpy, fracdiff ventana fija); `defects_absent` en el manifiesto | `tests/test_layer2_pool_h1.py` (20, incl. perturbación futura por familia y NaN local) |
| 2 (hueco) | cruces 2023 con 25–40 % de barras ausentes | recuperación desde la fuente independiente H1+M30, identidad M30→H1, cobertura mensual, reloj con controles ±1..3 h; store r4 | financial-data 10 tests (27 con los previos), feature-eng 5 (13 con los previos) |
| 3, 6 | selección univariada / permutación de un solo modelo | `tools/pool_selection_v2.py`: 7 tareas, 2 modelos, folds purgados con embargo, permutación por bloques conjunta, contrastes reentrenados none/P/Cj/subgrupos/Cgroup/P+Cgroup en dos contextos, rescate de interacciones, etapa de codificación, presupuesto y regla declarados antes de correr | `tests/test_pool_selection_v2.py` (9) |

## 3. Pruebas ejecutadas (conteos reales, trading-stack, CPU)

heuristic-strategy: `test_expert_guided_scaling.py` + `test_expert_guided_ablation.py` +
`unit_tests/test_expert_guided_{e3m,env,temporal}.py` = **42 passed** (incluye la CLI
lenta, 24 s); `test_pool_selection_v2.py` = **9 passed**. predictor: `test_layer2_pool_h1.py`
+ `test_feature_catalog_h1.py` = **28 passed**. feature-eng: `test_cross_asset_h1.py` +
`test_cross_asset_h1_r4.py` = **13 passed**. financial-data: `test_recover_independent_hf_{fx,crosses}.py`
+ `test_admit_core_fx.py` = **27 passed**. La suite legada de predictor sigue estando
obsoleta y no se cuenta.

## 4. Cadena catálogo → subconjunto → tensor → modelo → métricas (hashes)

| Eslabón | Identidad |
| --- | --- |
| EURUSD admitido | sha256 `c80847bb149c7af1ad50e42301b5cd70608e6c37702ce8cc6a7b9eb70f36d2ca` |
| cross-asset r4 | `9d59b20c845acd4545a519070a4281d2efca21b4bbbc5dc32c9f85cfe958c8cf` (MANIFEST con `source_identities`, `source_clock_verification`) |
| pool capa 1 (sobre r4) | `3098ffd4de04a645…` (MANIFEST `inputs.cross_asset` = r4) |
| pool capa 2 | `4a89dfea2a6410d6…` (+ juego fit-2023-01-01 `5fcccbce…`) |
| catálogo | `FEATURE_CATALOG.csv` `4eac85448a2586e1…` (370 filas) |
| pool completo materializado para el consumidor | `full_pool_eurusd_h1.parquet` `2f7b3b8c5272…` (31 116 × 280) |
| selección v3 | `BUDGET.json` (antes de correr: commit fd16919 limpio, `family_union_v1`, `label_support` 29/72 ≤ 96, `hod_clock_check` +1 h constante), `LEDGER.jsonl` (3 894 unidades, 0 errores, sha `a67f47f81cce2538…`, cada unidad con `support_hash`/`val_hash`/`cols_hash`), `SUPPORT_AUDIT.json`, `SHORTLIST.json` (core15 15 / estricta 126 / shortlist 174; 3 columnas excluidas del consumidor); publicado en heuristic-strategy `docs/short_horizon/results/expert_guided/selection_v3` |
| consumo | E3-M `--store` + `--feature-list` → `SCALER.json` (nombres en orden del tensor + constantes + tabla de estacionariedad) + `PROBE.json` (logits de recarga) junto a cada `.pt` |

## 5. Resultados por etapa

**v3 — base (pool completo verificado: 260 columnas por unidad HGB, 257 temporal, 57
familias en la permutación):** informativas `opener3` (HGB 0.443 vs prior 0.696; temporal
0.531 vs 0.691) y `hit` (HGB 0.167 vs 0.172); las cinco tareas de dirección/resultado
neto no mejoran sobre el prior en estas tareas/modelos/soportes.

**v3 — contrastes emparejados + rescate (3 810 unidades; 0 grupos sin emparejar; tu
`selection_ledger_support_audit.py` sobre el ledger v3: contraste 714/714 y
codificación 78/78 emparejados; su comprobación de rescate compara contra la
referencia genérica `ctx|core15`, que en v3 ya no es el control: cada par lleva sus
cuatro brazos none/s1/s2/pair sobre su propia unión, 120/120 grupos con firma
idéntica):** 41 RETAINED, 2 RESCUED_INTERACTION (`fx.usd_strength` + `dly.oil`),
14 NOT_RETAINED_WITH_THIS_EVIDENCE. Descomposiciones, ahora sí medidas:
`wavelet/log_close` → P_decided+Cgroup (14 evidencias); `wavelet/logret_1` →
P_decided+Cgroup; `gmm` y `hmm` → P_decided+Cgroup (las probabilidades de régimen
SÍ añaden sobre sus entradas; ganancia temporal +0.14 nats en opener3); `kalman` →
nivel−cierre (+0.19 nats temporal); `hilbert/logret_1` → fase; fuerza espectral →
un componente cada una; `multitaper`, `ssa`, `hilbert/log_close` no retenidas.
`P_raw` (log-cierre crudo, keep) nunca supera a `P_decided`; en `fracdiff` empata.
Las ganancias HGB son pequeñas (≤ 0.01 nats); las temporales, grandes, donde el
tronco de 2 unidades no puede derivar la información por sí mismo.

**v3 — codificación (90 unidades):** ganancias ≤ 0.02 nats e inconsistentes entre
modelo y tarea; la regla declarada (mejor variante con lo90 > 0 en alguna celda)
eligió `sessions`, `gmm one-hot`, `rel24` y `diff+rel480`, todas marginales;
`event_clock`, `range_scale` y `regime_hmm` conservan la base. Siguiendo tu §4, las
ganadoras se materializan como columnas ADICIONALES (store `full_pool_enc`, 293
columnas, sha `58e94711…`) para un cuarto brazo consumidor, no como reemplazo.

**Listas v3:** core15 15 / estricta 126 / shortlist 174 / estricta+codificaciones 138;
`fx.xauusd.*` excluidas del consumidor por cobertura de ventana (restricción de
aplicabilidad, no juicio).

**E3-M sin rolling z-score sobre el CSV heredado (15 inputs con nivel crudo):**
FULL INNER_A −0.02 / +2.41 (inválida) / −4.68 → −0.76 vs +0.86 con rolling-z;
TRAIN +27 732 USD en una semilla → nivel = proxy de fecha → regla de
estacionariedad (§2).

**Finalistas v2 (PRE-REPAIR, no promovidos):** core15 −1.14; estricta(91) INNER_A +0.35
con TRAIN +312 k USD en una semilla (memorización) y 2024 negativo.

**Finalistas v3 (mismo store, escala fija + regla de estacionariedad, FULL head 16,
3 semillas, desde 12:28Z):** core15 / estricta 126 / shortlist 174 / estricta+codificaciones:
PENDIENTE. La comparación rolling-vs-fijo con controles emparejados (fixed+keep,
fixed+diff, fixed+diff+relativas) se monta después, como cola aparte (§4 de tu revisión).

**Reajuste (corrida 2, τ estático, 3 rondas DAgger):** ningún brazo de reajuste
supera al estático (VAL A 0.86 vs −0.6…−3.2); 2024 estático Sharpe −0.06.

## 6. Comandos reproducibles (CPU salvo E3-M)

```bash
# selección v2 (plan antes de correr; run reanudable y fragmentable; aggregate idempotente)
python tools/pool_selection_v2.py plan|run|aggregate --admitted <admitted> --csv-dir <sel15> \
  --store <store-root>/eurusd_h1_layer1_pool_r4 --store <store-root>/eurusd_h1_layer2_pool \
  --catalog <store-root>/eurusd_h1_catalog/FEATURE_CATALOG.csv --out <out> [--stage base|contrast|encoding|rescue|all] [--shard k/n]
# pools y catálogo (predictor)
python tools/layer1_pool_h1.py --v4-eval-input <v4>/features_all.parquet --cross-asset <r4>.parquet --activity <P_HIT>.parquet --admitted-sha256 c80847bb… --out <store-root>/eurusd_h1_layer1_pool_r4
python tools/layer2_pool_h1.py … ; python tools/feature_catalog_h1.py …   # ver docs/timing_selection/layer2_pool_20261022/README
# consumo (heuristic-strategy)
python tools/expert_guided_e3m.py run --admitted <admitted> --csv-dir <sel15> --store <store-root>/eurusd_h1_full_pool/full_pool_eurusd_h1.parquet \
  --feature-list FEATURE_LIST_<arm>.json --units 2 --head 16 --scaling fixed_train --nonstationary diff --variants FULL --max-epochs 1000 --out <out>
# pruebas
python -m pytest tests/test_expert_guided_scaling.py tests/test_expert_guided_ablation.py tests/test_pool_selection_v2.py tests/unit_tests -q
```

## 7. Qué te pido que intentes romper primero

1. **Futuro/prefijo:** añade filas después de t en el pool capa 2 y en r4 y vuelve a
   generar: ningún valor ≤ t debe cambiar (las pruebas lo hacen con datos reales;
   intenta con otro corte).
2. **Soporte y purga:** una etiqueta cuyo soporte cruce `fit_end` nunca debe estar en
   una fila de selección; el embargo de 96 barras debe cubrir el soporte real de
   `net_*` (72 barras + cierre forzoso); si encuentras un soporte mayor, la regla
   es insuficiente.
3. **Escalado:** cambia filas de 2024 y comprueba que `SCALER.json` no se mueve;
   intenta hacer que un nivel crudo pase con `--nonstationary diff` sin quedar
   diferenciado; intenta recargar un `.pt` sin su artefacto.
4. **Consumo real:** compara `SCALER.json["features"]` de cada brazo finalista con
   `FEATURE_LIST_<arm>.json` y con las columnas del tensor (`PROBE.json`); la
   lista excluida por cobertura de ventana debe coincidir con `SHORTLIST.json`.
5. **Permutación por bloques:** verifica en el ledger que las importancias de una
   familia son 0 cuando la familia no está en el conjunto de columnas del modelo.
6. **Contabilidad:** las métricas de trading de E3-M salen del simulador real
   (costos, SL/TP, cuota semanal, cierre forzoso); no hay cuota mínima de trades
   impuesta a la selección.

## 8. Lo que no se cerró, con razón

- Atribución de ejecución (corregida tras tu handoff dd63a326): en la corrida v2 la
  mitad de gamma de `encoding` no llegó a ejecutarse y dragon la absorbió; en la
  corrida v3 (la que cuenta) tu worker ejecutó las CUATRO etapas de su mitad,
  primera vez y sin errores (recibos en su `LOG_v3`). Nada de v2 se reutilizó.
- Consenso/sorpresas de calendario: PENDING en el catálogo (decisión del
  propietario sobre la fuente).
- XAUUSD: reloj sin testigo horario independiente (declarado); `rv_24` imposible
  en la reja FX (pausa diaria): excluido del consumidor por cobertura, no juzgado.
- Kalman `q_slope` en la frontera cero (la pendiente aporta poco; documentado).
- Recargas de corridas E3-M anteriores a SCALER.json: ahora se rechazan; no se
  hizo backfill (decisión pendiente; las constantes existen en `E3M.json`).
- Resultados PENDIENTE de §5: los cuatro brazos finalistas v3.
- Tu `selection_ledger_support_audit.py` necesita conocer los brazos de rescate v3 (none/s1/s2/pair por par) para no comparar contra `ctx|core15`; te dejo la comprobación equivalente en el plan y en el ledger (120/120).
