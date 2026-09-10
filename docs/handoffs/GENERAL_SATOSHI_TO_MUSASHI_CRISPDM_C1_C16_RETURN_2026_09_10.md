# Retorno C1-C16 — inventario ligado, reja ejecutable y OLAP activo

**Fecha:** 2026-09-10
**Orden:** `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C1_C16_ORDER_2026_09_10.md`
**Dictamen gobernante:** `docs/audits/MUSASHI_AUDIT_CRISPDM_P0_P6_2026_09_10.md`

**Stop respetado:** ninguna GPU, live, venue, confirmación, selección científica
ni optimización DOIN. Nada de T2, M4 o B4 fue reabierto: esta ronda corrigió su
representación y su transporte, no su ciencia. Ninguna fila OLAP fue borrada y
no se creó un cubo nuevo.

---

## 1. Tips PRE y POST

| Repositorio | PRE | POST | Commits |
|---|---|---|---:|
| `predictor` | `0dd982c` | `c1e6588` | 4 |
| `financial-data` | `983c4e8f4` | `7e895b4c0` | 1 |
| `preprocessor` | `d6643f9` | `05fa1f4` | 1 |
| `agent-multi` | `4d310029` | `35aeb226` | 1 |
| `doin-domains` | `be6495a` | `79b8d2a` | 1 |
| `doin-plugins` | `60af316` | `43064fb` | 1 |

Rama en los seis: `satoshi/crispdm-c1-c16-20260910` en `predictor`;
`satoshi/crispdm-census-gate-20260910` en los otros cinco.

---

## 2. PRE: los doce puntos, y los diecinueve que reproduje

Congelado en `predictor@5ef9785` antes de tocar una línea:
`docs/audits/evidence/repro_runs/crispdm_c1_c16_pre_2026_09_10.{py,out}`.

**Los doce puntos del §1 de la orden reprodujeron, y encontré siete más
mientras los reproducía — diecinueve en total, cada uno por su razón exacta.**

Lo más importante que confirmé de mí mismo: **mi afirmación de "punto único"
era falsa.** `optimizer_plugin.optimize()` estaba en `app/main.py:268` y mi
reja en la 291, y el test que escribí solo comparaba la reja contra el
*pipeline*, nunca contra el optimizador. Verifiqué lo que no era.

---

## 3. Tabla C1-C16

| # | Corrección | Evidencia ejecutable |
|---|---|---|
| **C1** | La reja precede al optimizador (271 vs 280 vs 295) | `test_gate_c1_c5.py` — un test enumera **cada** llamada que consume datos y prueba que ninguna la precede |
| **C2** | `resolve_consumed_subjects()` deriva ids, columnas y digests de los encabezados que el contrato nombra | esquemas distintos entre particiones, duplicados, columnas sin nombre, archivo ausente y contrato vacío rehúsan por su razón |
| **C3** | Autoridad separada: submission ≠ decisión | `review_record_path()` **no recibe argumentos**, así que ningún config lo alcanza; el record liga submission+manifest+censo+código+particiones+alcance+cronología |
| **C4** | Parser estricto | claves duplicadas, `NaN`, digest no canónico, `bool` como número y fecha futura rehúsan; archivo completo por un descriptor |
| **C5** | Reproducción histórica EXPRESA | `execution_purpose=ARCHIVAL_REPLAY_NON_AUTHORITATIVE`, estampada; un experimento nuevo sin manifest **rehúsa** |
| **C6** | Censo físicamente ligado | **1.680/1.680 apariciones digeridas, 14.436.534.039 bytes leídos, cobertura 1.0, 17,9 s**; segundo censo: 0 bytes, 1.680 reusadas, 3,5 s; mutación de igual longitud re-digerida |
| **C7** | Join de disponibilidad explícito | familia→entidades por prefijo `__`, familias sin match reportadas, entidad ambigua sin resolver; alcance mínimo derivado |
| **C8** | Perfiles externos re-derivados y re-hasheados | `FULL_PROFILE_PHYSICALLY_VERIFIED` vs `DECLARED_ONLY_NOT_VERIFIED` con razón |
| **C9** | Índice consumible | **8.513 filas** (antes 214): 1.965 variables, 1.680 apariciones, 4.650 series, 202 generadores, 10 datasets, 4 operadores; cardinalidades recontadas desde las filas |
| **C10** | Borde MTM total | tipo de retorno estable en toda longitud; sin scaler de train → `NOT_EVALUABLE`, nunca un ajuste tardío |
| **C11** | Diseño v2 `51639954` supersede `aa846d6d` | liga índice, censo físico y cardinalidad; validación anidada declarada; autoridad dice que el self-digest prueba cronología, no revisión |
| **C12** | Sobres ligados a productores | parser de esquema exacto + self-digest re-derivado + digest del archivo fuente + identidad del verificador; JSON fabricado rehúsa |
| **C13** | Identidad relacional | colisión de `campaign_key` compara identidad y **rehúsa antes de todo hecho**; gate de respaldo **abre y re-hashea** el dump |
| **C14** | Outbox durable + loader idempotente | PostgreSQL caído deja pendientes (no falla la ciencia), recupera al volver, no duplica, carga fallos e inconclusos, heartbeat con pendientes/fallidos/retraso |
| **C15** | Censo e índice en el cubo | 5 tablas aditivas separadas de los 99 perfiles históricos |
| **C16** | Veredicto DOIN | **24 refs inventariadas**, `TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED`, cero llamadas vivas |

---

## 4. Censo físico: cobertura, costo y delta

| | Primer censo | Segundo censo |
|---|---:|---:|
| Apariciones digeridas | **1.680** | 0 |
| Digests reusados | 0 | **1.680** |
| Bytes leídos | **14.436.534.039** | **0** |
| Cobertura | **1.0** | 1.0 |
| Reloj | 17,9 s | 3,5 s |
| Delta | 1.680 `ADDED` | 1.680 `UNCHANGED` |

`stat()` ya no cuenta como digest, y la cobertura lo dice en su propio texto.
Identidad física = tamaño + `mtime_ns` + `ctime_ns`; una regresión reescribe un
archivo **a la misma longitud** y prueba que se re-digiere.

---

## 5. Disponibilidad: familias instanciadas y excluidas

**Instanciadas: 0 de 420.** No por falta de esfuerzo sino porque ninguna fuente
las demuestra, y el campo faltante está nombrado por familia:

| Campo requerido ausente | Familias |
|---|---:|
| `available_from_ts_col` | 420 |
| `license_scope` | 420 |
| `min_latency_minutes` | 420 |
| `observation_ts_col` | 420 |
| `revision_policy` | 420 |
| `timezone` | 420 |
| `provider` | 83 |
| `feature_family` | 83 |

**Demanda derivada, no escrita de memoria:** 94 columnas de las tareas
supervisadas registradas y 83 columnas de los configs de `agent-multi`. Y un
hueco honesto: **ningún config declara a la vez `observation_contract` y
`feature_columns`** (2 y 5, cero solapamiento), así que las familias del
contrato v2 ratificado **no son derivables de configuraciones**. Lo publiqué
como `derivation_gap` en lugar de rellenarlo.

---

## 6. Cardinalidades del índice por tipo y autoridad

| Autoridad | Tipo | Filas |
|---|---|---:|
| `PUBLIC_FORECASTING_EVIDENCE` | dataset | 10 |
| | series | 4.650 |
| `SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY` | generator | 202 |
| `FINANCIAL_DOMAIN_DEVELOPMENT_ONLY` | physical_appearance | 1.680 |
| | variable | **1.965** |
| | model_ready_view | 2 |
| | operator | 4 |
| **Total** | | **8.513** |

Cada cardinalidad se rederiva de las filas; un resumen y el índice ya no pueden
discrepar. Una fila de variable **nunca** reclama un digest de bytes.

---

## 7. OLAP: esquema, migración, conteos y loader

**Respaldo verificado antes de escribir:**
`predictor_olap_pre_c12_c15_20260910.dump`, sha `df2e2c4b…`, abierto y
re-hasheado por la propia herramienta.

| Tabla | Antes | Después |
|---|---|---|
| `dim_lake_appearance` | ausente | **1.680** |
| `dim_lake_variable` | ausente | **1.965** |
| `dim_public_series` | ausente | **4.650** |
| `dim_synthetic_generator` | ausente | **202** |
| `fact_campaign_unit` | 60 | **120** (60 traducidas + 60 verificadas) |
| `dim_experiment` | 39 | **39** |
| `fact_performance` | 1.404 | **1.404** |
| `fact_results_summary` | 1.404 | **1.404** |
| `dim_variable` / `fact_variable_profile` | 99 / 99 | **99 / 99** |

**Loader:** probado contra bases desechables con PostgreSQL caído y de vuelta —
pendientes conservadas, recuperación, sin duplicar, fallos e inconclusos
cargados, heartbeat publicado. Ni PostgreSQL ni Metabase fueron reiniciados.

---

## 8. Disposición de las 60 filas históricas traducidas

**Conservadas, ninguna borrada, ninguna métrica reescrita.** Dos columnas
aditivas: `authority_state = TRANSLATED_SUMMARY_NON_AUTHORITATIVE` y
`superseded_by_envelope_sha256` apuntando al sobre verificado. Estado real
consultado del cubo: **60 traducidas + 60 `PRODUCER_VERIFIED`, 60 enlaces**.

---

## 9. Gaps DOIN reales

`TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED`. 17 refs en `doin-domains` y 7 en
`doin-plugins`: ninguna lleva una ruta de publicación de genes L2 de trading.
`src/doin_plugins/verification` y `src/doin_domains/simulation` contienen
**solo `__pycache__`**. Cero llamadas vivas una vez que se excluye —
correctamente— el re-export dentro del propio paquete, que es plomería y no
integración. El cableado queda **propuesto y no ejecutado**.

---

## 10. Batería y conteos, tomados del terminal

| Batería | Repositorio | Conteo |
|---|---|---:|
| C1-C5 reja | `predictor` | **28** |
| C6-C8 censo | `financial-data` | **17** |
| censo incremental | `financial-data` | **24** |
| C9-C10 índice y MTM | `predictor` | **14** |
| C12-C15 OLAP | `predictor` | **21** |
| aceptación §8 + C16 | `predictor` | **13** |
| reja (regresiones originales) | `predictor` | **43** |
| consumidores cruzados | `predictor` | **25** |
| diseño de selección | `predictor` | **34** |

**Suite completa `predictor` en `c1e6588`: 223 passed / 3 failed / 8 errores de
colección.** Los 3 fallos y los 8 errores son **preexistentes**, reproducidos en
la base revisada `aef4dc22` con un worktree desprendido.
**`financial-data`: 41 passed.**

**POST:** `crispdm_c1_c16_post_2026_09_10.{py,out}`, exit 0, **19/19
CORRECTED**. El PRE queda congelado y no se re-ejecuta: sobre el código
corregido ni siquiera puede localizar la llamada que fue escrito para
inspeccionar, porque `gate_subjects(` ya no existe en `app/main.py`.

---

## 11. Faltas propias

1. **Mi "punto único" era falso.** La reja corría después del optimizador y mi
   test solo la comparaba contra el pipeline. Verifiqué lo que no era.
2. **Dos de mis propios tests codificaban el defecto**: afirmaban que un primer
   censo no digiere nada. Están reescritos al contrato corregido, con la
   corrección dicha en el propio test.
3. **Tres fixtures míos usaban `"p"*64` como digest**, que no es hexadecimal.
   El guarda estricto los rechazó con razón; el fixture estaba mal, no el
   guarda.
4. **Mi detector de llamadas de C16 contaba un re-export** del propio paquete
   como integración — habría repetido exactamente la afirmación que el dictamen
   rechazó. Endurecido antes de publicar.
5. **Cinco aserciones mías fallaron por frases partidas entre literales**, no
   por contenido. Corregí la redacción del código, no el test, donde la frase
   valía la pena.
6. **Un supuesto falso**: asumí que una serie constante no produce scaler. Sí lo
   produce. El test ahora afirma lo observado.
7. **La primera columna `authority_state` no entró en la lista del INSERT** y
   PostgreSQL lo rechazó; corregido antes de cualquier carga real.

---

## 12. Efectos no ejecutados

Ninguna confirmación, ningún puntaje de selector, ninguna GPU, ningún gen DOIN,
ninguna acción live o venue. El diseño v2 es una **submission de candidato**: su
self-digest prueba cronología y nada más. No emití ningún record de revisión ni
de autorización, y no existe ninguno.

**La próxima puerta la abre su auditoría de este retorno, no una etiqueta mía.**
