# 05 — Datos, precondiciones y cubo

## 1. Cadena de repos (no mezclar oficios)

Fuente: `docs/tres_temas_entrevista/UBICACION_REPOS.md`.

```text
CSV crudo
  → preprocessor          (CSV → particiones + normalización; operadores T0/T1; I2 datos)
  → feature-eng           (features de dominio, si se usan)
  → feature-extractor     (AE; I3 pesos del detector; STEP 11 train-only)
  → predictor             (ventanas, STL *en train*, cabezales; I3/I4; ETL)
  → prediction_provider / LTS / gym-fx / doin-plugins
```

Hay **dos** preprocessors. DOIN carga `predictor/preprocessor_plugins` por entry points. El app `preprocessor` es el sitio de operadores CSV. Co-instalar predictor + gym-fx + app preprocessor **mezcla** el grupo. Un entorno por aplicación.

## 2. Precondiciones por paquete

| Paquete | Datos | Código previo | Hardware | Autorización |
|---|---|---|---|---|
| I0 contrato | ninguno | este directorio | — | Harvey ya ordenó el paquete; Musashi agenda |
| I1 esquema OLAP | **base throwaway**, no el cubo poblado | `olap/init_db.py`, `olap/etl_migrate_v2.py` | CPU + Postgres de laboratorio | no `reset_olap.py` contra resultados reales |
| I2 `D.*`/`Y.*` | un CSV público chico + un sintético T1 | plugin diagnóstico en `preprocessor` | CPU | no campañas |
| I3 `M.*`/`G.*` | el mismo CSV | callback en `predictor` / extractor | CPU; `CUDA_VISIBLE_DEVICES=""` | no GPU |
| I4 H-ES piloto | ≥30 runs ANN/DLinear CPU (E0 o daily chico) | I1–I3 verdes | CPU | no sustituir patience |
| I5 banco ruido STEP 03 | series públicas + ruido plantado post-split | T0 hecho; T1 lab | CPU | H2/H3 de P-PRE |
| I6 STEP 04–05 | alfabeto provisional; **no** espera 4B para H0 | I2 | CPU | 5A sintético antes de interpretar finanzas |
| I7 T2 | geometría dos orígenes; C31–C36 cerrados | T1 | CPU primero | no sellar sin I2 en el mismo run |
| I8 E0 P-MOD | generadores sintéticos de escalas conocidas | perfiles + TCN causal | CPU | no E2 |
| I9 P-CAP lab | booleanas / MLP; señales temporales chicas | I3 | CPU | no producción |
| I10 T3–T5 | solo operadores `PUBLICLY_ELIGIBLE` | T2 sellado | T4 CPU; T5 GPU solo con sello | no warm-start L2 retenido |
| I11 P-L2 | banco RL público | espacio de codificadores **ya** construido por I8/I10 | según presupuesto L1 de `tesis_sac/03` | no finanzas como evidencia principal |
| I12 STEP 08–13 | CSI / \([X,C,U]\) / \(\tau\) según cada STEP | medidores 03–05 vivos | CPU sintético primero | 08 ≠ StandardScaler |
| I13 P-INC / F5 | dominios clase 1 | WP1 inmutable | sin GPU de trading | Hanzo audita; Retsu no despacha |
| I14 frentes | — | ledger F1–F5 | — | F1 live sigue prohibido |

## 3. Cubo actual (hueco)

ETL vigente: schema `public` (`olap/init_db.py` + `etl_migrate_v2.py`).

Grano de `fact_performance`: experimento × fase × split × horizonte × métrica → avg/std/min/max.

Métricas que el predictor **ya** emite en CSV: MAE, R2, SNR, Uncertainty, Naive_MAE; clasificación: AUC_ROC.

**No existen hoy** (verificado en `init_db.py` / convención del CSV):

- nada por **epoch**;
- nada de `D.*` / `Y.*` / `M.*` / `G.*`;
- nada de longitud comprimida;
- nada de grafo de pesos.

Hay un schema `olap` en `.sql` **no desplegado**. No se usa. No se espera `olap.v_experiment_config_kv`.

Bug conocido: `metric_value` falta tras `init_db.py`; el ETL aborta. En throwaway: o se añade la columna, o se envuelve el backfill en savepoint. No se “arregla” el cubo real en este paquete salvo orden aparte.

## 4. Extensión propuesta del cubo (I1, laboratorio)

No se aplica al Postgres de campañas. Diseño para throwaway:

### 4.1 Hechos nuevos

`fact_run_epoch` (grano: experiment_key × seed × epoch):

- `split_key` no aplica al snapshot de pesos; `val_loss`, `train_loss` sí.
- `M.n_params`, `M.bytes_raw`, `M.L_zstd`, `M.L_lzma`, `M.L_ratio`.
- `G.density`, `G.weight_entropy`, `G.spec_radius`, `G.eff_rank` (nullable).
- `stopped` bool, `stop_reason` (`patience` / `max_epoch` / `h_es_pilot` / `manual`).

`fact_split_information` (grano: experiment_key × split_key × series_key):

- `D.bytes`, `D.L_zstd`, `D.L_lzma`, `D.H0`, `D.H_ctx` (nullable), `D.SNR_hat` (nullable), `D.I_free_synth` (nullable).
- filas `series_key='__TARGET__'` para `Y.*`.

`dim_series` (nombre de variable, frecuencia, unidad).

`dim_metric` se amplía con `metric_type` ∈ {forecast, information, graph, capacity} y `higher_is_better`.

### 4.2 Contrato de escritura

- Un run sin `fact_split_information` **no entra** al cubo de laboratorio a partir de I1-gate.
- Compresores y niveles se fijan en el manifiesto del experimento (mismo zstd level, mismo lzma preset).
- Serialización de pesos: numpy `save` de arrays ordenados por `model.weights` name, float32, C-order. No `.keras` zip (mezcla grafo y pesos).

## 5. Datasets por lane (sin inventar paths privados)

| Uso | Qué | Qué no |
|---|---|---|
| I2/I5/I6 | series públicas ya usadas en T1/TFB cuando la licencia lo permita; sintético de T1 | CSV de broker; series cuya SNR se “conoce” por un filtro |
| I8 E0 | generadores factoriales de P-MOD (escalas y desfases conocidos) | reetiquetar ruido de mercado como \(N\) |
| I9 | booleanas / MLP; temporales cortas de P-CAP | ImageNet, LLM |
| E3 P-MOD | inventario `financial-data` **histórico** | live, paper con capital |
| P-L2 | POPGym / CARL / lo que el contrato L2 ya fijó | sustituir el banco por ETH |

De 224 configs de `predictor`, 87 no tienen CSV en el checkout. I2 no depende de esas.

## 6. Flags y CLI

`predictor`: solo flags largos (`--epochs`, no `-e`).  
Verificación: `CUDA_VISIBLE_DEVICES=""` salvo orden explícita de GPU.  
No lanzar sweeps del host. No parar Postgres/Metabase/GPU ajenos.

## 7. Definition of done por medidor

Un medidor está **vivo** cuando:

1. existe test CPU que reproduce el número en un fixture fijo (hash);
2. el ETL escribe la fila o falla cerrado (no swallow);
3. el README del experimento declara el manifiesto de compresor/umbral;
4. una consulta Metabase de laboratorio devuelve la fila.

Sin (1)–(4) no se usa para H-ES ni para el PDF.
