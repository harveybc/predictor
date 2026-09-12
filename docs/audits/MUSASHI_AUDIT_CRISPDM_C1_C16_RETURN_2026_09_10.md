# Auditoria Musashi del retorno CRISP-DM C1-C16

**Fecha:** 2026-09-10

**Retorno auditado:** `predictor@0e27f3c`

**Codigo predictor gobernante:** `predictor@c1e6588`
**Repositorios declarados:** `financial-data@7e895b4c0`,
`preprocessor@05fa1f4`, `agent-multi@35aeb226`,
`doin-domains@79b8d2a`, `doin-plugins@43064fb`

## 1. Veredicto

**`REVISE_WITH_SUBSTANTIAL_COMPONENTS_ACCEPTED`.**

El retorno corrige partes importantes de la orden. El censo ya esta ligado a
bytes, el indice comun contiene las poblaciones que antes omitia, MTM conserva
su contrato en particiones cortas, el cubo mantiene intacta la historia y el
inventario DOIN deja de confundir un re-export con una integracion. Esos
resultados se aceptan.

La puerta experimental no se abre todavia. La reja no puede completar en
produccion su propio protocolo de revision, omite bytes consumidos y permite
que el optimizador cambie el contrato despues de aprobarlo. Ademas, el loader
OLAP no esta activo y el productor `predictor` solo emite cuando el pipeline
termina con exito. La frase "el cubo se alimenta solo" no describe el runtime
observado.

## 2. Disposicion por componente

| Componente | Disposicion | Razon |
|---|---|---|
| C1 orden de la reja | `REVISE` | Esta antes del optimizador, pero el optimizador puede mutar despues los datos que consumira el pipeline. |
| C2 universo consumido | `REVISE` | Los `x` se derivan; los archivos `y` no participan en sujetos ni en los digests agregados. |
| C3 autoridad separada | `REVISE` | Submission y decision estan separadas, pero una submission nueva se fecha dentro de cada ejecucion y hace inutilizable un record revisado previamente. |
| C4 esquemas estrictos | `REVISE` | El JSON rechaza duplicados/no finitos en algunos lectores, pero manifest, envelope y productores no tienen esquemas exactos recursivos. |
| C5 replay historico expreso | `ACCEPT` | La omision del manifest ya no degrada silenciosamente. |
| C6 censo fisico | `ACCEPT` | 1.680/1.680 apariciones tienen digest y la segunda pasada reusa solo identidades fisicas sin cambio. |
| C7 disponibilidad | `ACCEPT_AS_NEGATIVE_INVENTORY` | El join se corrigio; el resultado real sigue siendo 0/420 familias instanciables. |
| C8 perfiles externos | `ACCEPT` | Los perfiles no verificables permanecen declarativos. |
| C9 indice comun | `ACCEPT` | 8.513 filas y cardinalidades rederivadas desde las filas. |
| C10 MTM | `ACCEPT` | Retorno estable y `NOT_EVALUABLE` cuando train no produce scaler. |
| C11 diseno de seleccion | `HOLD` | Es correctamente una submission sin autoridad y ademas liga un manifest no disponible. |
| C12 envelopes | `REVISE` | Mejoran el binding, pero todavia aceptan no finitos/campos no declarados y no todos nacen en el punto terminal del productor. |
| C13 identidad OLAP | `REVISE` | La identidad de campana se compara; el inventario usa `DO NOTHING` por id logico y puede conservar una version fisica vieja. |
| C14 outbox y loader | `REVISE_P0` | No hay loader persistente activo y los fallos de pipeline no emiten terminal. |
| C15 inventario en OLAP | `ACCEPT_AS_INITIAL_SNAPSHOT` | Los conteos estan cargados y las ocho tablas historicas siguen intactas. |
| C16 DOIN | `ACCEPT` | `TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED` es el veredicto honesto. |

## 3. Hallazgos bloqueantes

### F1 - P0: el record externo no puede abrir una ejecucion posterior

`gate_run()` crea `submitted_at=utc_now_stamp()` y acto seguido exige un record
que ligue el digest de esa submission. No existe un modo productivo
`submit-only` ni una entrada para consumir despues la submission revisada.
Al reintentar un segundo mas tarde con el record de la primera submission, el
resultado real fue:

```text
REFUSED: review chronology: submitted_at (...) is after reviewed_at (...)
```

La reja esta cerrada, pero tambien es impracticable. Debe existir una fase 1
que materialice una submission estable y salga sin ejecutar, y una fase 2 que
rederive los mismos hechos, consuma esos mismos bytes y verifique el record.

Referencias: `eligibility/integration.py:127-153` y
`eligibility/review.py:84-132`.

### F2 - P0: parte de los datos queda fuera del binding

`resolve_consumed_subjects()` abre y hashea cada `y_*`, pero:

- deriva sujetos solo del encabezado `x`;
- calcula `partitions_digest` solo con `x_sha256`;
- define `data_digest` como el `x` de un unico rol.

Cambie `y_validation.csv` de `1` a `999`. Sujetos, `data_digest` y
`partitions_digest` quedaron identicos. La submission y el record tampoco
cambian. La afirmacion de "exact consumed set" es, por tanto, mas fuerte que
el codigo.

Referencias: `eligibility/consumed.py:124-149` y `:183-234`.

### F3 - P0: el contrato aprobado puede cambiar despues de la reja

La reja corre en `app/main.py:271`; despues,
`optimizer_plugin.optimize(...)` devuelve un diccionario y
`config.update(optimal_params)` lo aplica sin lista permitida ni rederivacion
(`app/main.py:277-285`). Un optimizador puede devolver `x_test_file`,
`target_column`, un plugin o cualquier otra clave que cambie lo consumido por
el pipeline. Aprobar antes no sirve si el objeto aprobado sigue mutable.

### F4 - P0: OLAP no esta alimentandose continuamente

PostgreSQL responde y los conteos cargados son reales:

```text
dim_campaign=3
fact_campaign_unit=120
dim_lake_appearance=1680
dim_lake_variable=1965
dim_public_series=4650
dim_synthetic_generator=202
```

Tambien siguen intactos `dim_experiment=39`, `fact_performance=1404` y
`fact_results_summary=1404`. Sin embargo:

- no existe proceso, servicio o timer de `tools/olap_loader.py` activo;
- no existe aun el directorio productivo del outbox;
- `app/main.py:295-356` emite solo despues del retorno exitoso del pipeline;
- una excepcion, fallo o inconcluso anterior a esa linea no produce envelope;
- T2/M4/B4 fueron backfill; sus runtimes futuros no llaman el outbox.

El loader fue implementado y probado como pieza, no desplegado como flujo
continuo.

## 4. Hallazgos de alta severidad

### F5 - identidad de codigo incompleta

`CONSUMING_CODE` contiene solo cuatro archivos de la reja. No liga los modulos
reales cargados para predictor, optimizer, pipeline, target y preprocessor, ni
el data handler. Cambiar el preprocesador o el predictor no cambia el digest
que el record llama `reviewed_code_digest`.

### F6 - esquemas parciales y numeros no finitos

El manifest acepta claves top-level y de entrada no declaradas. Un manifest
con `undeclared_top_level_field` y `undeclared_nested_field` fue cargado sin
rechazo. `validate_envelope()` tambien acepto simultaneamente campos anidados
extra y `metric_value=NaN`, y emitio un digest.

`tools/build_campaign_envelopes.py` usa `json.loads()` normal y valida que el
self-digest tenga longitud 64, no que sea hexadecimal canonico. C12 exigia
parsers estrictos y esquemas exactos en cada nivel.

### F7 - el inventario OLAP congela versiones viejas

`olap/inventory_rows.py:136-209` usa `ON CONFLICT (<logical_id>) DO NOTHING`
para apariciones, variables, series y generadores. Si cambia el digest fisico,
la disponibilidad o la semantica bajo el mismo id, la nueva observacion no se
inserta y el cubo conserva silenciosamente la anterior. La idempotencia no debe
convertirse en perdida de historia.

## 5. Verificaciones ejecutadas

- Bateria focal predictor: **140 passed, 30 skipped** con Python del sistema.
  El primer intento fallo solo porque el worktree desprendido no tenia al repo
  hermano en la ruta relativa que el test presupone; al declarar el path real,
  paso.
- Bateria `financial-data`: 16 tests pasaron antes de detenerse por ausencia de
  `pandas` en el Python del sistema. Esto es una limitacion del entorno de esta
  auditoria, no una regresion atribuida a Satoshi.
- Contraejemplos nuevos ejecutados: mutacion de `y` no ligada; campos extra de
  manifest aceptados; record de primera fase imposible de reutilizar; envelope
  con `NaN` y campos extra aceptado.
- PostgreSQL: disponible; conteos del retorno comprobados directamente.
- Runtime del loader: ausente.
- No se ejecuto GPU, live, venue, seleccion cientifica ni publicacion DOIN.

## 6. Lo que queda aceptado y no se reabre

No se repite el censo de 14,4 GB salvo que cambie una identidad fisica. Se
aceptan su poblacion y digests como snapshot inicial. Tampoco se reabren el
arreglo MTM, la separacion de los 330 stubs, las 60 filas traducidas, los
conteos historicos preservados ni el veredicto negativo de integracion DOIN.

## 7. Bloqueadores y propietario

No hay una decision pendiente del propietario. Los bloqueadores son tecnicos y
quedan asignados a Satoshi por la orden C17-C30:

1. hacer ejecutable la revision externa en dos fases;
2. ligar todo `x` y `y` y congelar el contrato despues de la reja;
3. completar identidad y schemas;
4. emitir todo terminal y activar el loader;
5. versionar el inventario OLAP;
6. derivar el primer alcance financiero de disponibilidad sin inventar datos;
7. preparar la caracterizacion CPU posterior, sin abrir aun seleccion o GPU.
