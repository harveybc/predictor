# Auditoria Musashi: B4 R19-R22, T2 R16-R21 y CRISP-DM C67-C86

**Fecha:** 2026-09-12
**Retorno:** `SATOSHI_B4_R19_R22_T2_R16_R21_CRISPDM_C67_C86_RETURN_2026_09_12.md`
**Disposicion global:** `REVISE`
**Autoriza GPU, score, confirmacion o live:** no

## 1. Identidades y comprobaciones

Se revisaron los tips empujados:

| Frente | Tip |
|---|---|
| predictor | `fc62a073a2144d7b0ee4577a9c501cf1ea7c70d8` |
| financial-data | `b24b22719ea217d319eb6366ef3e333ddd8ab274` |
| B4 | `2fb3e2b9a7c4f2cfd3071edbdf3179ed672704b2` |
| T2 closure | `53ada0703cc898c48f961cc979d7befcc3f12247` |
| T2 reproducer | `6fe6c1ea6f9543ec0c253ffd2d26e2e7b5f97f4d` |

Las ramas estan limpias y sincronizadas. Las baterias focales ejecutadas
independientemente dieron:

| Superficie | Resultado |
|---|---|
| B4 custody/closure | 57 passed |
| T2 closure | 49 passed |
| T2 reproducer | 83 passed, 1 skipped |
| predictor C68-C85 | 79 passed, 6 skipped |
| financial-data DAG/temporal | 102 passed |

Estos verdes acreditan las correcciones cubiertas. No cubren los
contraejemplos adicionales de este dictamen.

Los cuatro PRE adicionales y su salida literal quedan en
`docs/audits/evidence/repro_runs/musashi_c67_c86_additional_pre_2026_09_12.py`
y `.out`. Usan solo fixtures temporales.

## 2. Hallazgos

### P0-1. La hoja esta ligada, pero el subdirectorio aun puede cambiarse

Las cuatro copias de `descriptor_custody.py` son byte-identicas
(`93e0fd89...`). `DirSnapshot.__init__` guarda hechos solo para archivos
regulares (`tools/descriptor_custody.py:258-275`). Para un subdirectorio
conserva solamente el nombre. `subdir()` vuelve a abrir ese nombre
(`:355-368`) sin comparar el directorio abierto con el que aparecia en
el inventario.

Contraejemplo ejecutado mediante la API publica:

1. crear `root/cell/terminal.json` con valor 1;
2. construir `Custody(root)`, que fotografia `cell`;
3. renombrar `cell` y crear otro `cell/terminal.json` con valor 9;
4. llamar `walk_to("cell").read("terminal.json")`.

Salida: `ACCEPTED_REPLACED_CHILD_DIR {'wall': 9}`.

La afirmacion de B4 v3 de que despues de fotografiar el directorio
"nothing is resolved by name again" es, por tanto, falsa. B4 y T2
siguen expuestos antes de alcanzar las nuevas guardias por hoja.

El inventario tampoco prueba que el directorio permanecio estable
mientras era listado: no compara un `fstat` anterior y posterior a la
enumeracion.

**Efecto:** B4 submission v3 y T2 submission v3 no se aceptan todavia.
Sus resultados y raices historicas se preservan.

### P0-2. El digest del censo no se recalcula

`load_census()` compara `expected_census_sha256` con el texto
`census_sha256` del propio JSON y con el nombre del archivo
(`tools/verify_lake_terminals.py:158-179`). Nunca recalcula el digest
canonico que el productor define como el SHA-256 del documento sin el
campo `census_sha256`.

Contraejemplo ejecutado:

1. conservar el nombre y `census_sha256` originales;
2. cambiar `relative_path`, `physical_sha256` y `size_bytes` de un
   appearance para apuntar a otro parquet;
3. mantener la misma poblacion de ids y el mismo pre-ledger.

Resultado actual:
`MUTATED_CENSUS TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED
features/evil.parquet`.

Ademas, `Artifact.json()` usa `json.loads()` sin rechazo de claves
duplicadas ni constantes no finitas (`descriptor_custody.py:183-187`).
Un terminal con dos `rows_used` fue aceptado y el ultimo valor gano:
`DUPLICATE_JSON_KEY ... VERIFIED 10`.

Las entradas de appearance y variable se validan con
`exact_keys=False` (`verify_lake_terminals.py:187-204`), pese a que la
orden exigia esquemas exactos.

**Efecto:** el informe acredita una recomputacion util, pero no una
autoridad de poblacion cerrada. Los registros v3 y las filas OLAP se
conservan como evidencia historica y deben ser supersedidos
aditivamente.

### P0-3. La divergencia de entropia es una politica de nulos ausente

La unica variable divergente no es una serie numerica ordinaria. Es
`announcement_datetime_local_utc`; el parquet la almacena como `int64`
sin nulls fisicos y sus 3.147 valores son
`-9223372036854775808`, el sentinel habitual de una fecha ausente.

El productor y el revisor lo trataron como numero finito: media, min,
max y percentiles quedaron alrededor de `-9.22e18`, missingness quedo
en cero y la recomputacion concluyo entropia 0. La discrepancia de
entropia es secundaria. La falla real es que ninguna capa demostro el
tipo semantico ni la politica de sentinels antes de medir.

**Efecto:** `INDEPENDENTLY_RECOMPUTED=1504` no se acepta aun como conteo
cientifico. No se presume que las otras 1.503 variables esten mal; se
exige un barrido tipado que pueda demostrarlo.

### P0-4. T2 conserva a proposito un verificador que ya sabemos debil

La submission declara correctamente el defecto: omitir una semilla MLP
produce `KeyError` en vez de un rechazo tipado. Los 242 records reales
no ejercitan ese caso y el estimando negativo no cambia. Sin embargo,
un acto nuevo de revision no debe heredar deliberadamente una debilidad
ya corregida.

La decision es inequivoca:

* la ejecucion historica conserva para siempre su commit y sus bytes;
* la readjudicacion nueva usa el verificador endurecido actual;
* el record nuevo liga ambas identidades y declara que no hubo
  reentrenamiento;
* una diferencia detiene la readjudicacion; nunca se normaliza.

Hay otro problema de secuencia: `run_reproducer()` carga los modulos del
checkout (`t2_campaign_closure.py:1476-1479`) antes de exigir el record
(`:1525`). La identidad debe comprobarse antes de ejecutar/importar la
superficie que luego consumira la evidencia.

### P1-1. FEATURE_DAG.v3 es honesto, pero solo como inventario estatico

Se acepta el resultado `0 CAUSAL_ACTIVE`. La ausencia de bindings es
real: el recibo historico no registro commit ni digest de codigo. No se
puede corregir retroactivamente.

Se autoriza una salida distinta: ejecutar el productor actual,
committeado, sobre los insumos fisicos registrados y escribir un
**dataset sucesor nuevo** con procedencia completa. Si reproduce los
bytes antiguos, eso se informa; si no, tambien. En ningun caso el nuevo
run convierte el run historico en algo que no fue.

`FEATURE_DAG.v3` queda aceptado solo como
`STATIC_CANDIDATE_INVENTORY_UNBOUND`; no concede elegibilidad.

### P1-2. El contrato temporal es valido para disponibilidad, no para calidad

Se acepta de forma acotada:

* `DATE_TIME` es apertura de barra;
* el OHLCV no esta completo antes del `close_time` de esa barra;
* la latencia operacional del proveedor sigue `UNOBSERVED`;
* E5a offline esta abierto y E5b live cerrado.

Los 20 registros truncados dentro del CSV y los ocho gaps no contradicen
la disponibilidad por fila, pero si impiden tratar toda pareja de filas
como un horizonte H4 regular. El diseno debe excluir o estratificar,
antes de puntuar, toda muestra cuyo lookback o target cruce una barra
truncada o un gap. No se autoriza interpolacion silenciosa.

### P1-3. El diseno v3 mezcla el contraste de reemplazo con el de capacidad

El brazo A1 reemplaza cada variable por una transformacion y conserva la
dimensionalidad de A0. A3 agrega columnas para controlar la dimension
extra de A2. Por tanto, exigir `A1_VS_A3` en H1
(`PER_VARIABLE_PREPROCESSING_DESIGN.v3.json:36,248`) compara objetos que
no controlan el mismo efecto.

La separacion correcta es:

* H1: A1 frente a A0, seleccion por variable frente a identidad;
* H2: A2 frente a A0 y frente a A3, representacion aumentada frente a
  dimension aumentada sin informacion;
* H3: abstencion y no dano bajo cambio de panel.

Tres paneles son insuficientes para el t unilateral prometido como
confirmacion general. Con menos de seis paneles independientes el
resultado debe quedar descriptivo/inconcluso. La poblacion tambien debe
exigir tipo semantico, unidad, rol, licencia y politica de valores
ausentes, no solo que una columna sea numerica.

## 3. Disposiciones

| Objeto | Disposicion |
|---|---|
| B4 submission v3 | `REVISE_DIRECTORY_COMPONENT_BINDING` |
| B4 resultado | preservado: 2/1/9, `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT` |
| T2 submission v3 | `REVISE_WITH_HARDENED_READJUDICATOR` |
| T2 resultado historico | preservado: `DOES_NOT_ADVANCE`, estimando sin cambio |
| terminales v3 / verificacion v2 | `REVISE_CANONICAL_CENSUS_AND_SEMANTIC_NULLS` |
| filas OLAP nuevas | historicas; no borrar ni promover |
| FEATURE_DAG.v3 | `ACCEPT_STATIC_UNBOUND_ONLY` |
| contrato temporal ETH H4 | `ACCEPT_AVAILABILITY_BOUNDED` |
| diseno por variable v3 | `REVISE_BEFORE_LICENSE` |
| licencia vigente | `MECHANICS_ONLY_CPU_NO_SCORES` |

## 4. Estado operativo

Tras el reinicio, la GPU esta sana y libre de computo cientifico.
PostgreSQL, Metabase, OLAP y los supervisores estan activos. Esta
auditoria no inicia ni detiene procesos. La ausencia de trabajo GPU es
intencional: todavia no existe una poblacion ni un diseno licenciados.

Las correcciones quedan especificadas en
`MUSASHI_TO_GENERAL_SATOSHI_B4_R23_R26_T2_R22_R27_CRISPDM_C87_C105_ORDER_2026_09_12.md`.
