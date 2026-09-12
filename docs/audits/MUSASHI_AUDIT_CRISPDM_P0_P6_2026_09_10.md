# Auditoria Musashi: censo, reja de elegibilidad y continuidad OLAP

**Fecha:** 2026-09-10

**Retorno auditado:** `docs/integracion_workplan_2026_09_10/05_INFORME_DE_RETORNO_P0_P6.md`

**Tips declarados:** `predictor@ac53433`, `financial-data@983c4e8f4`,
`preprocessor@d6643f9`, `agent-multi@4d310029`,
`doin-domains@be6495a`, `doin-plugins@60af316`.

## 1. Veredicto

**`REVISE`**. El trabajo produjo una base util y dos correcciones reales, pero
no abre todavia seleccion, GPU, confirmacion financiera, live ni genes DOIN.

| Parte | Disposicion |
|---|---|
| P0: reproduccion de la base | **ACCEPT** |
| P1: dos granos y conteos del censo | **ACCEPT PROVISIONAL / REVISE BINDING** |
| P2: indice comun | **REVISE** |
| P3: reja `PUBLICLY_ELIGIBLE` | **REVISE P0** |
| P4: scaler MTM train-only | **ACCEPT IDEA / REVISE EDGES** |
| P5: diseno de seleccion | **HOLD** hasta P1-P4 corregidos |
| P6: esquema y backfill OLAP | **ACCEPT ADDITIVE SCHEMA / REVISE AUTHORITY AND INGESTION** |

Los conteos de 1.965 variables conceptuales, 420 entidades y 1.680
apariciones son un hallazgo valioso. Se conservan como **censo declarado
provisional** hasta que las apariciones nuevas queden ligadas a sus bytes.

## 2. Hallazgos

### P0-1. La reja corre despues del optimizador

En `app/main.py:264-276`, `optimizer_plugin.optimize(...)` puede construir
datos y ajustar modelos. La reja se consulta despues, en `app/main.py:282-294`.
El comentario que la llama "single choke point" contradice el grafo ejecutable.
El test actual solo comprueba que la reja anteceda al pipeline y no contempla
el optimizador.

**Impacto:** un run con `use_optimizer=true` puede consumir variables no
revisadas antes de consultar la reja.

### P0-2. Una reja vacia y un manifest autoemitido conceden `GATED`

`eligibility/integration.py:67-86` convierte `None` o `[]` en `used=[]` y
devuelve `ELIGIBILITY_GATED`. La configuracion por defecto deja
`eligibility_subjects=None` (`app/config.py:84-92`). Reproduccion independiente:

```text
eligibility_status=ELIGIBILITY_GATED
universe_size=1
subjects_used=[]
subjects_used_count=0
```

Ademas, la ruta y el digest esperado provienen de la misma configuracion que
ejecuta el candidato (`eligibility/integration.py:67-70`). Un manifest creado
por el propio llamador, con `reviewer="candidate"`, su self-digest y ese mismo
digest en config, fue aceptado como `ELIGIBILITY_GATED` para `var.ok`.

**Impacto:** la etiqueta positiva no demuestra que se revisaron las variables
realmente consumidas ni que la decision provenga de un revisor independiente.

### P0-3. Los bindings declarados no se comparan con los bytes consumidos

`require_eligible()` permite comprobar `version`, `code_digest` y
`evidence_digest`, pero `gate_subjects()` no entrega ninguno de ellos
(`eligibility/integration.py:73-76`). Tampoco deriva el conjunto de variables
desde los encabezados y el contrato de columnas: confia en una lista opcional
de configuracion. Los digests de datos y particiones del manifest nunca se
comparan en este punto de llamada.

El modo sin manifest procede implicitamente como `LEGACY_NON_AUTHORITATIVE`
(`eligibility/integration.py:50-61`). Esto sirve para reproduccion historica,
pero debe ser una eleccion expresa; no puede ser el default de un experimento
nuevo.

### P0-4. El primer censo no liga ninguna aparicion a bytes fisicos

El retorno reconoce `0/1.680` digests fisicos. La orden exigia digest para
entradas **nuevas**, cambiadas o seleccionadas. En
`financial-data/_scripts/lib/incremental_census.py:292-304`, una entrada nueva
no activa `want_digest`; una entrada posterior solo se considera cambiada por
tamanio, aunque se registra `mtime_ns`. Una mutacion de igual longitud no
obliga a recalcular el digest.

Hay tres defectos relacionados:

1. el escritor usa `O_TRUNC` sobre un nombre presentado como content-addressed
   (`_scripts/build_incremental_census.py:123-133`);
2. los perfiles externos copian self-digests y conteos sin revalidar el
   inventario ni los bytes (`_scripts/build_incremental_census.py:38-66`);
3. las instancias se indexan por `feature_family`, pero las variables las
   consultan por `entity` (`incremental_census.py:221-248` frente a
   `incremental_census.py:402`). Cuando aparezca el primer contrato real, el
   join puede dejarlo sin efecto.

**Impacto:** los conteos estructurales son utiles, pero aun no constituyen un
inventario fisicamente verificable ni pueden alimentar una elegibilidad.

### P0-5. El backfill OLAP traduce resumenes; no verifica a sus productores

`tools/build_campaign_envelopes.py:25-33` abre un JSON arbitrario y copia sus
campos. No verifica el digest fisico del archivo, el esquema del productor ni
reproduce su adjudicacion. B4 se construye enteramente desde argumentos y
constantes. El self-digest del sobre solo demuestra consistencia del sobre
nuevo, no veracidad del resultado fuente.

En la carga, `ON CONFLICT (campaign_key) DO NOTHING`
(`olap/campaign_envelope.py:242-253`) no compara la identidad existente. Un
segundo sobre con el mismo `campaign_key` y otra identidad puede insertar
hechos ligados a la dimension anterior. El parametro `--backup-sha256` solo se
valida por longitud (`tools/backfill_campaign_envelopes.py:60-71`); no se
verifica un archivo de respaldo.

**Disposicion de las 60 filas nuevas:** conservarlas, sin borrado, como
`TRANSLATED_SUMMARY_NON_AUTHORITATIVE` hasta publicar sobres rederivados desde
la evidencia original y supersederlas aditivamente.

### P0-6. El OLAP esta arriba, pero no recibe resultados automaticamente

PostgreSQL acepta conexiones y Metabase esta activo. Lectura independiente del
cubo real:

| Objeto | Filas |
|---|---:|
| `dim_experiment` | 39 |
| `fact_performance` | 1.404 |
| `dim_campaign` | 3 |
| `fact_campaign_unit` | 60 |
| `fact_campaign_consumption` | 3 |

Sin embargo, `load_envelope()` no tiene llamadores en los pipelines de
`predictor`; el unico flujo real es el CLI manual de backfill. Por tanto, el
cubo esta **activo y consultable**, pero no esta recibiendo automaticamente
cada candidato y experimento. Esto debe corregirse con un outbox durable e
ingesta idempotente, sin acoplar el exito cientifico a la disponibilidad de
PostgreSQL.

### P1-7. El indice comun no contiene el inventario de variables

`crispdm_bank_index.v1.json` tiene 214 filas: 202 generadores sinteticos,
10 datasets publicos y 2 vistas financieras. `build_index()` solo emite
datasets, generadores y vistas (`olap/bank_index.py:338-370`). No emite las
1.965 variables financieras ni las series del banco publico. El resumen del
banco contiene esos conteos, pero un conteo no es un indice consumible.

**Impacto:** P2 no cumple aun el contrato de "indice comun de datasets y
variables" y no puede ser la fuente exacta del universo de seleccion.

### P1-8. El arreglo MTM tiene dos bordes sin cerrar

El scaler train-only es una correccion correcta. No obstante,
`_apply_causal_mtm_decomposition()` devuelve solo un `ndarray` cuando la
particion es mas corta que la ventana, mientras el llamador desempaqueta una
tupla. La reproduccion devolvio:

```text
<class 'numpy.ndarray'> (4, 2)
```

Tambien debe quedar prohibido ajustar validation/test cuando el train no
produce un scaler valido; ese caso debe ser `NOT_EVALUABLE`, no un fit tardio.

### P1-9. Los parsers aceptan mas de lo que afirman

El manifest de elegibilidad usa `json.loads` normal: acepta claves duplicadas y
constantes no finitas, no exige esquemas exactos, no valida que todos los
digests sean SHA-256 canonicos y acepta fechas futuras porque una edad negativa
no es "stale" (`eligibility/gate.py:90-149`). El sobre de campana tiene el
mismo patron (`olap/campaign_envelope.py:160-202`). Estas son fronteras de
evidencia cientifica, no una ampliacion de seguridad operativa.

## 3. Verificacion ejecutada

- Bateria focal `predictor`: **135 passed** en el entorno `trading-stack`.
- Bateria del censo `financial-data`: **23 passed**.
- Reproduccion positiva de reja vacia: **confirmada**.
- Reproduccion de manifest autoemitido: **confirmada**.
- PostgreSQL: **aceptando conexiones**; conteos reales cotejados.
- Metabase: contenedor **activo**.
- No se ejecuto entrenamiento, GPU, live, venue ni optimizacion DOIN.

Las baterias verdes prueban lo que cubren; los contraejemplos anteriores
demuestran que hoy no cubren los contratos decisivos.

## 4. Disposicion de las cinco preguntas al owner

No quedan cinco decisiones humanas. Quedan una decision de alcance y cuatro
tareas tecnicas:

1. **Disponibilidad: resuelta por alcance minimo.** Instanciar primero las
   familias y dependencias que consumen el contrato de observacion v2 ya
   ratificado y las tareas financieras supervisadas activas. La lista exacta
   se deriva de sus manifests y encabezados; no se escribe de memoria. Todo lo
   demas permanece no elegible. Si la fuente no demuestra el momento de
   disponibilidad, no se inventa.
2. **Procedencia ambigua: no es decision del owner.** Resolver por DAG de
   derivacion, worker y recibos fisicos. Lo que no pueda resolverse conserva
   `AMBIGUOUS` y queda fuera.
3. **Stubs: no completar 330 por volumen.** Permanecen
   `COVERAGE_STUB_NON_SEMANTIC`; se completa solo el alcance minimo que vaya a
   evaluarse.
4. **Genes L2: no es decision del owner.** Inventariar refs y puntos de llamada.
   Si no existen, declarar la integracion `NOT_IMPLEMENTED`; no fabricar una.
5. **Primer manifest: responsabilidad de Musashi despues de la correccion.**
   Satoshi entrega una submission no autorizante; Musashi revisa y emite el
   record. No hay accion pendiente del owner en este ciclo.

## 5. Siguiente paso

Ejecutar la orden
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C1_C16_ORDER_2026_09_10.md`.
Hasta su retorno auditado, el estado es:

`FINANCIAL_SELECTION_AND_CONFIRMATION_BLOCKED_BY_UNBOUND_INVENTORY_AND_GATE`.
