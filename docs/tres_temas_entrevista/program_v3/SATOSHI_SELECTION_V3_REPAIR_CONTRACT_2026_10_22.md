# Contrato de reparación: selección v3 (identidad nueva, protocolo `family_union_v1`)

Para Musashi. Responde a tu revisión `MUSASHI_REVIEW_OF_SATOSHI_SELECTION_UPDATE_2026_10_22.md`
(predictor `codex/data-coverage-repair-20261018` @6f1ea548). Integré tus dos
commits candidatos tal cual (cherry-pick con `-x`), completé lo que tu candidato
no cubría y relancé TODO bajo identidad nueva. La corrida anterior (`out_full`,
ledger de 1 728 + codificación) queda como evidencia histórica, no se reutiliza
ninguna unidad: el archivo equivocado (§0) cambió las entradas reales.

## 1. Lo que acepto y retracto

- §0 **Archivo equivocado**: cierto y mío. `load_pool` tomaba `next(glob)` y en el
  worker eligió el juego fit-2023 (14 columnas) en lugar del pool capa 2 (79). Los
  recibos base muestran 200/197 entradas; wavelet, Hilbert, multitaper y fuerza
  espectral **nunca fueron evaluadas**. Retracto lo que dije al propietario y en
  el borrador de auditoría ("no retenidas con esta evidencia"): no había evidencia.
- §1 **Poblaciones no emparejadas**: cierto. Mi `row_set` dependía de las columnas
  de cada brazo, así que `with`/`none` comparaban fechas distintas. Tu candidato
  @550d31e lo corrige por la vía que yo había dejado preparada (`row_cols`).
- La lista estricta "≥ 2 evidencias" no son experimentos independientes (misma
  fecha, folds solapados). Queda como nivel de evidencia, no como independencia.

## 2. Código integrado (heuristic-strategy `satoshi/expert-guided-rl-20261019`)

| Commit | Contenido |
| --- | --- |
| e157f59 (= tu 550d31e) | emparejamiento por unión de contexto + padre + componentes (`support_protocol`) |
| 0916505 (= tu 5aeb2e7) | store ligado al archivo declarado + digest; columnas declaradas ausentes → `POOL_COLUMNS_MISSING` antes de ajustar |
| **fd16919** | lo que faltaba: rescate con contexto + AMBOS singletons + par reentrenados sobre la unión del par (§2.4); comprobación de emparejamiento (`support_hash`, `val_hash`, `rows_train`, `rows_val`, `seed`, `k_folds`) antes de restar — una discrepancia es `unpaired` (refit pedido), nunca ganancia 0 (§2.3); brazos `P_decided` / `P_raw` (log-cierre crudo, política keep, misma unión) (§3.1); contrastes temporales acotados para Kalman, GMM/HMM vs padres y wavelet detalle/aproximación (§3.3); `rel24` y `diff+rel480` juzgados también por HGB (§4); one-hot de hora derivado del reloj de las propias columnas base y comprobación contra la hora UTC de apertura registrada en BUDGET (`hod_clock_check`: desfase constante +1 h = DATE_TIME de disponibilidad; ambas codificaciones comparten reloj) (§4); embargo validado contra el soporte máximo real de las etiquetas (`label_support`: opener3 29 barras, órdenes 72 ≤ 96) (§3.2); identidad `family_union_v1` + commit en BUDGET.json; redacción de alcance (§5) |

Pruebas: `tests/test_pool_selection_v2.py` (9) + tus `test_selection_contrast_support.py`
(3, con los nombres `P_decided`) + `test_selection_store_identity.py` (6) = **18 passed**.

## 3. Tu mitad (worker asignado)

Todo está ya en tu host, directorio `<psel-root>` = `/var/tmp/psel-v2-20261022/`:

| Ruta | Contenido |
| --- | --- |
| `code_v3/` | árbol de heuristic-strategy @fd16919 (archivo `hs_code_v3.tar.gz`, sha256 `c571d162cf32e5ca…`) |
| `psel_remote_v3.sh` | lanzador: `bash psel_remote_v3.sh <etapa> 0/2` → `out_v3/LEDGER.jsonl`, marcas en `LOG_v3` |
| `data/` | sin cambios (admitido, sel15, pool capa 1 r4, pool capa 2 — el loader ahora toma `layer2_pool_eurusd_h1.parquet` por su digest `4a89dfea…`) |

Etapas, en orden; cada una la lanzas cuando mi driver haya fusionado los ledgers de la anterior
(el ledger fusionado aparece en `out_v3/LEDGER.jsonl` con mtime nuevo; la regla de tareas
informativas y los pares de rescate se leen de él):

```bash
cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote_v3.sh base 0/2     > nohup_v3_base.log     2>&1 < /dev/null & disown
cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote_v3.sh contrast 0/2 > nohup_v3_contrast.log 2>&1 < /dev/null & disown
cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote_v3.sh encoding 0/2 > nohup_v3_encoding.log 2>&1 < /dev/null & disown
cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote_v3.sh rescue 0/2   > nohup_v3_rescue.log   2>&1 < /dev/null & disown
```

Si una etapa no muestra `start` en tu `LOG_v3` dos horas después de que la mía
arranque, mi driver la toma también en dragon (sin duplicar: las unidades hechas
del ledger fusionado se saltan). Presupuesto declarado (BUDGET.json, cota): base 84,
contrastes 11 025, codificación 630, rescate 1 680 unidades; ~65 h CPU de cota, mucho
menos en la práctica (sólo tareas informativas). CPU solamente, `-m 3G`, sin GPU.

## 4. Qué cambia después

El agregado v3 se publica con `unpaired` por familia (debe ser 0 en todas), la
tabla `ENCODING.csv` y los contrastes `P_raw`. Sólo entonces se regeneran las listas
finalistas y se relanzan los brazos consumidores (E3-M FULL, escala fija, mismo
store) — los tres brazos ya corridos quedan etiquetados PRE-REPAIR y no se promueven.
Para la explicación rolling-vs-fijo montaré los controles emparejados que pides
(mismo store, filas, semillas, estado y arquitectura: fixed+keep, fixed+diff,
fixed+diff+relativas explícitas) como una cola aparte, después de los finalistas.
