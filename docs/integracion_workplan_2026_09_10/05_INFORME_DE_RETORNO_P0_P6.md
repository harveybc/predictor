# Informe de retorno — orden de censo, reja de variables y continuidad OLAP

**Fecha:** 2026-09-10
**Orden:** `03_ORDEN_SATOSHI_CENSO_FEATURE_GATE_Y_OLAP.md`
**Base revisada por Musashi:** `predictor@aef4dc22e7a6039eb349b75caf5e9b5823607d5e`

**Stop respetado:** ninguna GPU, ninguna confirmación, nada en vivo, ninguna
optimización DOIN. Todo el trabajo fue CPU, pruebas, migración OLAP respaldada
y backfill verificable.

---

## 1. Tips PRE y POST por repositorio (mapa de commits)

Todos los repositorios usan la rama `satoshi/crispdm-census-gate-20260910`.

| Repositorio | PRE (congelado) | POST | Commits nuevos |
|---|---|---|---:|
| `predictor` | `5da7b59` | `d3b5d78` | 5 |
| `financial-data` | `c077e20e2` | `983c4e8f4` | 1 |
| `preprocessor` | `0fc1895` | `d6643f9` | 2 |
| `agent-multi` | `889320ee` | `4d310029` | 1 |
| `doin-domains` | `d01708a` | `be6495a` | 1 |
| `doin-plugins` | `c2bea4c` | `60af316` | 1 |

Los seis tips PRE quedaron congelados en el commit de PRE de `predictor`
(`47066d3`) junto con la reproducción completa de P0.

---

## 2. P0 — la base revisada se reproduce EXACTA

Las cinco reproducciones exigidas, todas verdes:

1. los 8 tests focales del commit revisado pasan;
2. `crispdm_dataset_inventory.v1.json` se regenera **byte a byte**, con su
   digest interno igual al publicado `03be76a4…` (2 datasets, 26 541 filas,
   99 variables, ambos `PROFILED_WITH_METADATA_GAPS`);
3. la reconciliación de `financial-data/features` se regenera **byte a byte**,
   digest interno `833e6afe…`: **1 680 cortes presentes, 0 ausentes**, 7 860
   apariciones de columnas, 14 436 534 039 bytes;
4. base desechable, doble carga **idempotente**: 2 datasets, 2 series, 99
   variables, 2 perfiles de dataset, 99 perfiles de variable, 1 recibo
   (`4a1ba7b1…`);
5. el ETL histórico corre en base nueva **sin `ALTER TABLE` manual**: 72
   escritos / 0 omitidos.

El fixture de 72 hechos no estaba nombrado en la evidencia; lo identifiqué
recorriendo los 167 CSV de resultados: es `phase_1/ann_1575_1h`.

El cubo poblado quedó **sin una sola escritura** en esta reproducción: 39
experimentos, última carga de performance 2026-04-16, 1 404 filas.

---

## 3. Inventario y delta cuantificados

**El hallazgo central de P1:** las 7 860 apariciones de columnas del manifiesto
**no son 7 860 variables**. Son **1 965 variables conceptuales** sobre **420
entidades**, materializadas en **1 680 apariciones físicas**.

| Grano | Cantidad |
|---|---:|
| Apariciones físicas (cortes) | 1 680 (0 ausentes) |
| Apariciones de columnas declaradas | 7 860 |
| **Variables conceptuales** | **1 965** |
| Entidades | 420 |
| Bytes declarados (stat, no relectura) | 14 436 534 039 |
| Perfiles COMPLETOS ligados externamente | 2 (inventario de `predictor`) |
| Perfiles de valores muestreados | 12 (0,71 %, declarados como muestra) |
| Apariciones digeridas físicamente | 0 en el primer censo, por diseño |

**Delta:** primer censo sin predecesor → 1 680 `ADDED` / 1 965 `ADDED`. Una
segunda corrida contra el censo anterior devuelve 1 680 `UNCHANGED` / 1 965
`UNCHANGED`. El delta emite `ADDED`/`UNCHANGED`/`CHANGED`/`MISSING` comparando
por **identidad lógica**, de modo que mover un archivo es un `CHANGED`, nunca
un `ADDED` más un `MISSING`.

**Costo medido:** 3,5 s de reloj. El censo no relee el lago; solo digiere lo
seleccionado o lo que `stat()` muestra cambiado.

---

## 4. Huecos de metadata agrupados por responsable real

| Hueco | Variables | Entidades | Responsable real |
|---|---:|---:|---|
| Disponibilidad temporal (`available_time`) | **1 965 (100 %)** | 420 | propietario de la fuente + `financial-data` |
| Unidad física | 1 965 (100 %) | 420 | propietario de la fuente |
| Licencia | 1 965 (100 %) | 420 | propietario de la fuente / adquisición |
| Rol (entrada/objetivo/control/…) | 1 965 (100 %) | 420 | diseño experimental, no la fuente |
| Semántica declarada | 1 675 (85 %) | 410 | autores de los diccionarios |
| Procedencia | 406 (21 %) | 133 | `financial-data` (workers de adquisición) |
| Procedencia **ambigua** (≥2 candidatos) | 340 | 100 | `financial-data` — el censo **rehúsa** elegir |

**El hueco de disponibilidad es total y es el más serio.** El esquema
`configs/availability_contract.schema.json` existe y exige ocho campos
(`feature_family`, `provider`, `observation_ts_col`, `available_from_ts_col`,
`revision_policy`, `timezone`, `min_latency_minutes`, `license_scope`), y
`features/AVAILABILITY_CONTRACT.md` lo declara bloqueante — pero **no existe ni
una sola instancia**. Sin instancias, ninguna variable tiene tiempo de
disponibilidad verificable, y por tanto ninguna puede sostener una afirmación
causal sobre lo observable en el momento de decisión.

**Diccionarios, contados con honestidad:** de 954 archivos
`data_dictionary.md`, **622 son semánticos, 330 son *stubs* autogenerados que
declaran por escrito no ser una garantía de esquema semántico**, y 2 no son
parseables. Contar los 954 como documentación habría sido exactamente la
mentira que este censo existe para impedir.

**Conflictos declarados, no resueltos:** 5, todos de grafía bajo una misma
forma normal — `Close`/`close` (128 entidades), `High`/`high` (128),
`Low`/`low` (128), `Open`/`open` (128), `Volume`/`volume` (118). El censo los
publica y se niega a decidir si son el mismo concepto.

---

## 5. Pruebas adversariales y conteos, tomados del tip final

| Batería | Repositorio | Conteo |
|---|---|---:|
| Censo incremental (P1) | `financial-data` | **23 passed** |
| Índice común de bancos (P2) | `predictor` | **16 passed** |
| Reja de elegibilidad (P3) | `predictor` | **43 passed** |
| Integración de la reja (P3) | `predictor` | **9 passed** |
| Consumidores cruzados (P3) | `predictor` | **16 passed** |
| Fronteras de selección (P4) | `predictor` | **7 passed** |
| Etiquetas de frontera (P4) | `preprocessor` | **5 passed** |
| Diseño de selección (P5) | `predictor` | **30 passed** |
| Sobre de campaña (P6) | `predictor` | **14 passed** |

**Suite completa de `predictor` en el tip final `d3b5d78`:**
**143 passed / 3 failed / 8 errors de colección.**

Los 3 fallos y los 8 errores de colección son **preexistentes**: los reproduje
en la base revisada `aef4dc22` con un worktree desprendido y fallan idénticos
allí. Los 8 errores son la suite legada que `AGENTS.md` ya documenta
(`app.autoencoder_manager`, `load_encoder_decoder_plugins`, `merge_config` ya
no existen); los 3 fallos son
`tests/integration_tests/test_configuration_handling.py`.

---

## 6. Estado del manifest de elegibilidad

**No existe ningún manifest revisado, y eso es correcto en este punto.** Lo que
existe es:

- la **reja** (`predictor/eligibility/gate.py`), consumidor único, sin camino de
  permiso por omisión;
- el **template no autorizante**
  (`examples/research/ELIGIBILITY_MANIFEST_TEMPLATE.v1.json`), que la propia
  reja rechaza — está probado que no concede nada;
- las **seis regresiones obligatorias**, verdes:
  manifest ausente/rancio/con digest distinto rehúsa; una variable no elegible
  no reaparece por fallback; un grupo vacío da universo vacío; un operador no
  entra por nombre de plugin; reemplazar evidencia bajo etiqueta positiva
  rehúsa; el mismo manifest da el mismo universo ordenado en proceso fresco.

Cada entrada exige quince bindings: id + versión, esquema de entrada/salida,
unidad, `event_time` **y** `available_time` por separado, `fit_scope`, política
de estado incremental, parámetros, digests de datos/código/particiones/
evidencia, costo medido, y decisión + alcance + razón del revisor.

**Quien debe emitirlo:** el revisor externo. La reja jamás emite sus propios
permisos.

---

## 7. Estado de cada integración consumidora

| # | Repositorio | Punto de llamada | Estado |
|---|---|---|---|
| 1 | `preprocessor` | `run_preprocessor_pipeline`, inmediatamente antes de `plugin.process` | **VIVO** |
| 2 | `predictor` | `app/main.py`, antes de la llamada al pipeline (antes de ventanas y de ajuste) | **VIVO**, verificado en una corrida real |
| 3 | `agent-multi` | `_apply_feature_groups`, donde se FORMA el universo | **VIVO** |
| 4 | `doin-domains` | `require_eligible_gene_inputs` | **INSTALADO SIN SITIO DE LLAMADA** |
| 5 | `doin-plugins` | `verify_identity` (verifica, no selecciona ni promueve) | **INSTALADO SIN SITIO DE LLAMADA** |

El adaptador es **byte-idéntico** en los cuatro repositorios consumidores y un
test lo verifica, de modo que la regla no puede derivar entre ellos.

**Declaro sin adorno los dos huecos:** la rama actual de `doin-domains` es la
línea de ataque/holdout de MNIST y **no contiene ninguna ruta de publicación de
genes L2 del dominio de trading**; el módulo `verification` de `doin-plugins`
existe solo como `__pycache__`, sin fuente. Instalé el consumidor y lo probé en
ambos, pero no inventé un punto de llamada que no existe.

**Comportamiento sin manifest:** la corrida procede y queda **estampada
`LEGACY_NON_AUTHORITATIVE`** en su config efectiva. No es un permiso por
defecto: es la ausencia de revisión, registrada y viajando con los resultados.
Existen 137 configuraciones históricas ejecutables que preceden a la reja; esta
decisión las mantiene reproducibles sin permitir que se citen como evidencia
con reja.

---

## 8. Esquema OLAP, backfill e idempotencia

**Estado honesto del cubo antes de tocarlo:** 39 experimentos históricos,
última carga de `fact_performance` el 2026-04-16, 1 404 filas de performance,
más el inventario CRISP-DM cargado el 2026-09-10 (2 datasets, 2 series, 99
variables, 1 recibo). Nada de eso se limpió, se reinterpretó ni se reinició.

**Respaldo antes de escribir:** volcado PostgreSQL en formato custom, privado y
solo-propietario, `predictor_olap_pre_campaign_envelope_20260910.dump`,
SHA-256 `80b800b37ee10d2c8bcaa17c8109509c600d3a2a133540b2ccddcd7cf41619af`.
La herramienta de backfill **rehúsa correr sin un digest de respaldo**.

**Esquema nuevo, puramente aditivo:** `dim_campaign`, `fact_campaign_unit`,
`fact_campaign_consumption`. El módulo no contiene `DROP TABLE`, `TRUNCATE` ni
`DELETE FROM`, y un test lo afirma.

**Backfill ejecutado — solo cambió lo nuevo:**

| Tabla | Antes | Después |
|---|---|---|
| `dim_campaign` | ausente | 3 |
| `fact_campaign_unit` | ausente | 60 |
| `fact_campaign_consumption` | ausente | 3 |
| `dim_experiment` | 39 | **39** |
| `fact_performance` | 1 404 | **1 404** |
| `fact_results_summary` | 1 404 | **1 404** |
| `dim_dataset` / `dim_series` / `dim_variable` | 2 / 2 / 99 | **2 / 2 / 99** |
| `fact_dataset_inventory` / `fact_variable_profile` | 2 / 99 | **2 / 99** |
| `fact_ingestion_receipt` | 1 | **1** |

**Idempotencia probada sobre el cubo real:** una segunda corrida idéntica
devolvió `changed = {}`.

**Rechazo de artefacto mutado:** probado contra bases desechables — un sobre
con un valor alterado se rechaza **antes de cualquier escritura**, y los
conteos quedan idénticos.

**Recibo:** `86700ccf8ce74091…`, con conteos antes/después, tablas cambiadas,
tablas intactas y la declaración `deletions: NONE`.

**Los tres productores del backfill inicial, con su clase separada:**

| Campaña | Clase | Unidades | Adjudicación |
|---|---|---:|---|
| T2 sucesor de recursos v1 | `CONFIRMATION` | 31 | **DOES_NOT_ADVANCE** |
| M4 calibración intento 3 | `CALIBRATION` | 29 | candidata, sin autoridad confirmatoria |
| B4 generación v7 | `NON_GOVERNING` | **0** | `QUARANTINED_RUNTIME_STALL` |

B4 entra con **cero unidades y presupuesto `UNAVAILABLE`**: la campaña no
completó y no hay métrica que reportar. Rellenar esa forma habría sido inventar.

---

## 9. Efectos ejecutados y no ejecutados

**Ejecutado:** P0 completo; P1 completo con las dos granularidades, deltas,
huecos exactos y barrido de valores en dos niveles; P2 con el índice común y la
regla de no-promoción ejecutable; P3 con la reja, el template y las cinco
integraciones; P4 con una corrección real y tres etiquetas; P5 con el diseño
sellado y un preflight mecánico; P6 con el esquema aditivo, los sobres y el
backfill respaldado.

**No ejecutado, por diseño de la orden:** ninguna confirmación grande, ningún
puntaje de selector, ninguna GPU, ningún gen DOIN, ninguna acción en vivo.

**No ejecutado, por ausencia de material:** el punto de llamada del gen L2 en
`doin-domains` y el consumidor de verificación en un flujo real de
`doin-plugins`.

---

## 10. Faltas propias, confesadas

1. **Cablé la reja en el pipeline equivocado primero.** La puse en
   `default_pipeline.py` y la corrida real no la ejecutó, porque la
   configuración usa `stl_pipeline`. La moví al punto único de
   `app/main.py`, que es donde debía estar desde el principio; lo detecté
   porque la línea de estampa no apareció en la salida de una corrida real,
   no por lectura.
2. **Mi heurístico de localización del adaptador dio un falso positivo.**
   Aceptaba cualquier directorio llamado `predictor`, y en `doin-plugins`
   encontró el paquete interno `src/doin_plugins/predictor`. Lo endurecí: un
   hermano solo cuenta si realmente contiene `eligibility/gate.py`.
3. **Un byte NUL entró en el separador de identidad del censo** al escribir el
   archivo, y rompió la importación. Lo reemplacé por un separador explícito
   declarado.
4. **Mi primer test de etiquetas comparaba una frase partida entre dos
   literales de cadena** y fallaba por la partición, no por el contenido.
   Corregí la redacción del código en lugar de retorcer el test.
5. **Una corrida de verificación sobrescribió las salidas de ejemplo
   commiteadas** en `examples/results/phase_1_daily/`; las restauré con
   `git checkout --` y lo verifiqué.
6. **Ajusté `.gitignore`** para que la regla LaTeX `*.out` dejara de tragarse
   la evidencia congelada de auditoría. Es un cambio deliberado y acotado a
   `docs/audits/evidence/`, declarado aquí porque toca un archivo de
   configuración del repositorio.

---

## 11. Preguntas que requieren exclusivamente al owner

1. **Contratos de disponibilidad.** Ninguna de las 1 965 variables tiene
   `available_time` verificable. ¿Se instancia el contrato para las familias
   que el programa realmente va a usar (y cuáles son), o se declara
   explícitamente que la línea financiera opera sin garantía de disponibilidad
   hasta nueva orden? Sin esta decisión, la reja no puede conceder
   `PUBLICLY_ELIGIBLE` a ninguna variable financiera.
2. **Procedencia ambigua.** 340 variables sobre 100 entidades tienen dos o más
   directorios fuente candidatos. ¿Quién arbitra cuál es la fuente de verdad?
   El censo rehúsa elegir por nombre.
3. **Los 330 diccionarios *stub*.** ¿Se completan, se retiran, o se declaran
   permanentemente como cobertura documental sin semántica?
4. **`doin-domains` y `doin-plugins`.** ¿Existe una rama con la ruta de
   publicación de genes L2 del dominio de trading donde deba cablearse la
   reja, o esa ruta aún no existe?
5. **Emisión del primer manifest de elegibilidad.** El template está listo y la
   reja está viva en tres puntos. Falta que un revisor externo emita el primer
   manifest real; nada del programa puede producir evidencia con reja hasta
   entonces.

---

**Stop respetado.** No abrí GPU, confirmación, live ni optimización DOIN a
partir de estos artefactos. Quedan a la espera de la revisión de Musashi el
censo, la reja y el diseño de selección.
