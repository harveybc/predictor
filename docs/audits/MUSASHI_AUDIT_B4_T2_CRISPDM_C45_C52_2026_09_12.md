# Auditoria Musashi: B4/T2 y CRISP-DM C45-C52

**Fecha:** 2026-09-12  
**Objeto revisado:** retorno Satoshi `B4_AND_T2_READJUDICATION_SUBMITTED_AND_C45_C52_COMPLETE_READY_FOR_MUSASHI_REVIEW`  
**Identidades candidatas:** predictor `4e68c45`, financial-data `19fe375a1`, B4 `c907495d`, T2 `5fb2849e`  
**Veredicto:** `REVISE`; los resultados historicos se preservan, pero ninguna submission recibe autoridad todavia.

## 1. Lo que si queda establecido

1. Las baterias focales pasan: B4 45/45, T2 17/17 y demanda/DAG 15/15.
2. El signo bilateral de T2 ya esta acotado a `[0,1]`; la correccion no cambia el estimando ni el resultado negativo candidato.
3. El cubo conserva historia y el loader esta `active/running`, `NRestarts=0`.
4. La caracterizacion produjo 1.965 terminales fisicos: 1.505 `MEASURED` y 460 no identificables segun el productor.
5. No hay razon para borrar el cubo, repetir B4/T2 ni tocar datos confirmatorios. Hay que verificar y superseder, no reescribir.

## 2. Hallazgos P0

### F1. La fotografia de directorio no liga los archivos que luego se consumen

`DirSnapshot` conserva solo nombres y cierra el descriptor del directorio (`tools/descriptor_custody.py:136-163, 309-352`). Cada `read()` vuelve a recorrer los componentes por nombre desde la raiz retenida (`:200-222, 253-306`). Por tanto, un directorio de celda ya fotografiado puede ser renombrado y sustituido por otro antes de la lectura.

Contraejemplo independiente ejecutado: la fotografia contenia `terminal.json`; despues se sustituyo el directorio `cell` completo; `Custody.read("cell/terminal.json")` consumio el reemplazo con `wall_seconds=999`. Las pruebas actuales cubren sustitucion de hojas y aparicion de una celda nueva, pero no sustitucion de un directorio existente. El mismo modulo se usa en B4 y T2.

**Efecto:** B4 2/1/9 y T2 242/242 siguen siendo adjudicaciones candidatas. Los bytes pueden ser genuinos, pero la implementacion no prueba que pertenecen a la misma instancia de directorio inventariada.

### F2. La identidad publicada de B4 no es recuperable desde la rama

La submission B4 declara el commit `ac788210...`, pero el tip publicado es `c907495d...`. Ambos son hijos distintos de `a2f5b196...`; ninguna rama remota contiene `ac788210`. La exclusion autorreferencial de la submission (`tools/b4_campaign_closure.py:657-703`) termino ligando un commit candidato que no puede obtener un revisor desde la ref publicada.

Ademas, la evidencia publica contiene `root_actually_read` con una ruta absoluta privada. La identidad logica basta en Git; la topologia fisica debe vivir solo en el registro privado del operador.

### F3. T2 declara honestamente una identidad mixta, pero eso no es aun una identidad revisable

La submission combina siete archivos del checkout revisado con tres archivos del tip y fue generada mientras esos archivos estaban sucios. `make_snapshot_class()` subclasifica el `ResultsRoot` fijado sin ejecutar su constructor (`tools/t2_campaign_closure.py:736-768`), con lo cual satisface `isinstance` pero asume que ninguna invariante del constructor es necesaria. La intencion es razonable, pero necesita una instantanea unica, recuperable y una prueba de equivalencia completa de la interfaz.

### F4. `ACTIVE_EXECUTION_DEMAND` no representa ejecucion activa

`supervised_demand()` lee JSON crudo y llama ejecutable a toda configuracion cuyos archivos `x/y` declarados existen (`financial-data/_scripts/derive_demand_universes.py:77-105`). No construye la configuracion efectiva, no aplica defaults o parametros de plugins, no resuelve entry points, no valida el target ni ejecuta una ruta de validacion del entry point. El numero 137 coincide con "configs que tienen sus inputs presentes", no con ejecuciones activas.

La clasificacion de columnas `y` como `label` salvo coincidencia literal con `target_column` (`:116-130`) tambien depende de un valor que puede venir de defaults o del plugin. Las rutas absolutas y escapes tampoco se contienen bajo el checkout (`:69-73`).

**Efecto:** se retiran por ahora las expresiones `137 configs ejecutables`, `162 sujetos x`, `162 sujetos y` y `2 targets`. Los datos son un censo util de candidatos con archivos presentes.

### F5. `FEATURE_DAG.v1` localiza asignaciones, pero no construye un DAG transitivo

El indice solo reconoce subscripts literales directamente presentes en el lado derecho de una asignacion (`derive_feature_dag.py:78-122, 148-155`). No sigue variables locales, funciones auxiliares ni expresiones intermedias. Si encuentra una asignacion y no ve `center=True` o `shift(<0)` en esa linea, la clasifica `CAUSAL` (`:219-264`).

La evidencia real lo demuestra: `log_return_1 = returns`, `macd = ema12 - ema26`, `stoch_k`, `cci_14` y `mfi_14` aparecen `CAUSAL`, con cero entradas directas y lookback 1. Eso no prueba sus dependencias ni sus ventanas. El arreglo de `shift(-1)` es correcto, pero no detecta un desplazamiento oculto en una variable o helper.

**Efecto:** se retira el conteo "52 causales". Quedan 52 productores localizados pendientes de resolver transitivamente y 45 productores no localizados.

## 3. Hallazgos P1

### F6. Los 1.965 terminales son una salida del productor, no una atestacion independiente

`existing_terminals()` usa `glob/read_text`, omite JSON invalido y acepta el contenido por `variable_id` sin esquema exacto, digest propio, correspondencia nombre-id ni deteccion de duplicados (`predictor/tools/characterize_lake.py:147-159`). Los terminales fisicos inspeccionados no portan `schema`, `source_sha256`, `window_sha256` ni `terminal_sha256`.

`disposition_rows()` publica `BOUND_TO_SOURCE_BYTES`, pero usa el mismo digest global del censo como `source_sha256` y `window_sha256` para todas las variables (`:445-487`). Los envelopes por lote llevan `data_consumed.variables=[]` (`:227-279`), de modo que no ligan los miembros ni los terminales individuales.

**Efecto:** los conteos 1.505/460 se conservan como `PRODUCER_REPORTED_PENDING_INDEPENDENT_REPLAY`; las filas OLAP viejas no se borran y deben supersederse aditivamente con bindings veraces.

### F7. C52 es una buena nota de diseño, pero no esta sellada

`PER_VARIABLE_PREPROCESSING_DESIGN.v1.md` se llama `SEALED_DESIGN_NO_SCORES_COMPUTED`, aunque es prosa sin esquema, self-digest, poblacion ejecutable ni validador. Ademas depende del conjunto causal no demostrado y deja sin fijar operadores, minimo de variables, margenes, presupuesto, familia Holm y control de capacidad. Tambien compara mejora predictiva y costo como si compartieran unidades.

**Efecto:** se renombra `DRAFT_CANDIDATE_NO_SCORES_COMPUTED`. La hipotesis por variable sigue viva y prioritaria, pero no se puntua hasta cerrar F4-F6.

## 4. Estado operativo

- OLAP loader: activo, corriendo, cero reinicios.
- GPU: no utilizable por `NVML driver/library version mismatch` (kernel/DSO desalineados). Requiere reinicio del host por el propietario; no bloquea ninguna correccion CPU de esta orden.
- B4/T2 originales: preservar sin cambios. No relanzar ni reentrenar.

## 5. Disposicion

`B4_T2_SUBMISSIONS_REQUIRE_CUSTODY_AND_IDENTITY_REPAIR`  
`DEMAND_DAG_AND_CHARACTERIZATION_REQUIRE_SEMANTIC_REDERIVATION`  
`PER_VARIABLE_PREPROCESSING_REMAINS_DRAFT`

La siguiente orden asigna todas las correcciones resolubles a Satoshi. Al propietario solo le corresponde reiniciar el host para recuperar la GPU.
