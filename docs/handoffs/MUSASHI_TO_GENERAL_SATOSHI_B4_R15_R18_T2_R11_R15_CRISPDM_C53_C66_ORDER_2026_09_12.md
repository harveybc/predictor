# Orden Musashi a General Satoshi: cierre B4/T2 y base data-centric

**Fecha:** 2026-09-12  
**Base de auditoria:** `docs/audits/MUSASHI_AUDIT_B4_T2_CRISPDM_C45_C52_2026_09_12.md`  
**Prioridad:** P0 B4/T2 y demanda/DAG; P1 caracterizacion/OLAP y diseno por variable  
**Disposicion inicial:** `REVISE`  
**Owner:** no debe intervenir para trabajo CPU. Su unico pendiente operativo es reiniciar el host para recuperar la GPU.

## 0. Reglas de la orden

1. Congelar todos los PRE antes de editar y ejecutarlos contra los tips auditados: B4 `c907495d`, T2 `5fb2849e`, predictor `4e68c45`, financial-data `19fe375a1`.
2. No editar, mover, cambiar modos ni reabrir para escritura los roots originales B4/T2 ni los 1.965 terminales originales.
3. No GPU, entrenamiento, confirmacion, live, venue, promocion, limpieza del cubo ni publicacion DOIN.
4. Toda correccion se prueba en copias o fixtures. Los datos historicos se superseden de forma aditiva; jamas se reescriben.
5. Ninguna herramienta candidata crea records de Musashi ni se concede autoridad a si misma.
6. La ejecucion CPU empieza de inmediato y no espera el reinicio del host.

## P0-A. B4 R15-R18: una fotografia debe retener la instancia fotografiada

### R15. PRE obligatorio

Reproducir por API publica, con barreras deterministas:

1. Fotografiar una celda existente, renombrar ese directorio y crear otro con el mismo nombre; demostrar que el codigo auditado consume el terminal del reemplazo.
2. Repetir con un subdirectorio intermedio y una hoja de mismo nombre.
3. Sustituir directorio y restaurar el nombre original antes de finalizar; la igualdad de nombres no debe ocultar cambio de inode/device.
4. Mostrar que la bateria vigente queda verde bajo al menos uno de estos ataques.

### R16. Custodia corregida

1. `DirSnapshot` debe retener un descriptor propio del directorio y sus hechos `(device, inode, uid, mode)` hasta terminar el consumo.
2. Los archivos inventariados se abren relativos a ESE descriptor retenido; queda prohibido volver a recorrer `cell/...` por nombre despues de la fotografia.
3. Cada directorio intermedio consumido debe tener owner/mode verificados. Una raiz o directorio grupo/mundo-escribible se clasifica explicitamente; no se mejora retroactivamente con `chmod`.
4. Si los modos legacy impiden autoridad fuerte, producir un bundle de lectura estricto desde los bytes descriptor-bound y declarar su procedencia. No cambiar los originales.
5. Cerrar descriptores de forma determinista ante exito y excepcion; probar ausencia de fugas.

### R17. Publicacion B4 en dos fases

1. Commit A: codigo y tests terminados; push y prueba de que una ref remota contiene A.
2. Desde checkout limpio de A, generar la submission.
3. Commit B: solo submission/packet; la submission liga A, no B ni un commit amendado o huerfano.
4. Sustituir toda ruta fisica publica por ids logicos sanitizados. La ruta completa solo puede aparecer en evidencia privada no versionada.
5. Re-derivar 2 `COMPLETED_VERIFIED`, 1 `QUARANTINED_PARTIAL`, 9 `NOT_STARTED` sobre copia descriptor-bound. Si cambia, publicar el cambio; no forzar 2/1/9.

### R18. Aceptacion B4

- Ataques de directorio: todos rehusan o consumen demostrablemente la instancia retenida.
- Mutaciones por guardia: quitar retencion, identidad de directorio, owner/mode o apertura relativa readmite su ataque exacto.
- Cero paths privados en Git.
- A y B recuperables desde origin; `git branch -r --contains A` no vacio.
- Resultado final: `B4_READJUDICATION_V2_SUBMITTED_FOR_EXTERNAL_REVIEW`; ningun record externo creado.

## P0-B. T2 R11-R15: una sola evidencia y una sola identidad ejecutable

### R11. Aplicar la misma reparacion de directorio

El directorio `units` queda retenido por descriptor desde inventario hasta la ultima verificacion. Cada `RECORD/ARRAYS/CLAIM` se abre relativo a ese descriptor. Congelar el ataque de sustitucion de `units` completo y de restore-after-swap.

### R12. Eliminar la identidad mixta del resultado final

1. Construir una rama/checkout de auditoria unica y recuperable que contenga, byte por byte, los siete archivos fijados y los archivos nuevos de reconstruccion/custodia realmente ejecutados.
2. Todos los imports del replay deben salir de esa instantanea. No mezclar un checkout revisado con un tip sucio en el proceso que produce la submission final.
3. Commit A-T2 con codigo/tests, push; commit B-T2 con submission/packet.
4. Registrar versiones numericas sin paths de `site-packages`; la topologia local no viaja en Git.

### R13. Adaptador del verificador

Preferir composicion. Si `UnitSnapshotRoot` sigue heredando sin llamar `ResultsRoot.__init__`, demostrar exhaustivamente que:

- implementa toda la superficie consumida, no solo la observada en una corrida;
- ninguna invariante del constructor participa en verificacion, cierre o manejo de errores;
- agregar una nueva llamada publica en el verificador fijado hace fallar cerrado el adaptador;
- cada array se lee exactamente una vez y el record puntuado son los mismos bytes verificados.

### R14. Re-adjudicacion y estadistica

Reproducir desde la instantanea unica las 242 unidades y el estimando. Probar la tabla bilateral exacta para `k=0..6`, simetria y rango `[0,1]`. El resultado esperado es candidato `DOES_NOT_ADVANCE`, estimando aproximado `-0.001048`; una diferencia se reporta, no se corrige a mano.

### R15. Aceptacion T2

- 242/242 desde una sola instancia por unidad y un solo checkout recuperable.
- Ataque de sustitucion de `units` muerto; mutacion sin retencion lo readmite.
- Submission sin rutas privadas y con commits alcanzables desde origin.
- Resultado final: `T2_READJUDICATION_V2_SUBMITTED_FOR_EXTERNAL_REVIEW`; sin nuevo score ni record externo.

## P0-C. CRISP-DM C53-C58: demanda ejecutable y DAG causal real

### C53. Retirar etiquetas sobre-reclamantes

Superseder `DEMAND_UNIVERSES.v1` sin borrarlo:

- 137 pasa a `INPUT_FILES_PRESENT_CONFIG_CANDIDATES`;
- 87 queda `INPUT_FILES_ABSENT_CONFIGS`;
- 162/162 y dos targets quedan `RAW_CONFIG_DERIVED_PENDING_EFFECTIVE_CONFIG`;
- 52 pasa a `PRODUCER_ASSIGNMENTS_LOCATED_PENDING_TRANSITIVE_RESOLUTION`;
- 45 conserva `UNRESOLVED_PRODUCER`.

### C54. Tres niveles de demanda, sin mezclarlos

Materializar por separado:

1. `OBSERVED_EXECUTION_DEMAND`: runs terminales verificados o manifiestos de lanzamiento revisados.
2. `VALIDATED_RUNNABLE_CONFIGS`: configuracion efectiva construida con la precedencia real, defaults y parametros de plugins; entry points resueltos; archivos y target validados por una ruta sin efectos que termina antes de modelo/pipeline.
3. `INPUT_FILES_PRESENT_CONFIG_CANDIDATES`: el censo actual, honestamente nombrado.

La ruta de validacion debe ejecutarse en subprocess, importar cero TensorFlow/CUDA, escribir cero archivos y no entrenar. Derivar `x`, `y`, target y roles desde la configuracion efectiva. Rehusar rutas absolutas, traversal y archivos fuera del checkout autorizado.

### C55. DAG transitivo

El AST actual queda como localizador, no como prueba causal. Construir un grafo donde cada salida reclamada:

1. resuelva variables locales e intermedias dentro del simbolo;
2. siga helpers mediante contratos explicitos o analisis trazable;
3. termine en hojas raw/external con semantica temporal declarada;
4. acumule lookback, shift, centring y latencia por todos los caminos;
5. detecte ciclos y marque `UNRESOLVED` ante cualquier dependencia desconocida;
6. separe `HISTORICAL_OR_RETIRED_PRODUCER` de `CAUSAL_ACTIVE`.

Fixtures obligatorios: `returns -> log_return_1`, `ema12/ema26 -> macd`, stochastic con high/low/close, helper que oculta `shift(-1)`, helper con `bfill`, `np.roll(-1)`, indexacion futura, cadena de dos variables locales, ciclo y productor retirado.

### C56. Disponibilidad temporal

No asignar `event_time + bar_width` por defecto. Ligar por dataset si el timestamp representa apertura, cierre o publicacion; incorporar latencia del proveedor y de cada transformacion. Una sola hoja o helper no resuelto vuelve no disponible toda la salida.

### C57-C58. Aceptacion

- El conteo activo se deriva de evidencia ejecutable, no se fija a 137.
- Cada `CAUSAL_ACTIVE` tiene camino completo a hojas temporales y ningun input vacio salvo raw declarado.
- `log_return_1`, `macd`, `stoch_k`, `cci_14` y `mfi_14` ya no pueden aparecer causal/lookback-1 con `direct_inputs=[]`.
- Mutaciones que corten una arista, ignoren un helper o copien `event_time` readmiten su PRE.
- No lanzar seleccion ni entrenamiento.

## P1-D. CRISP-DM C59-C63: verificar los 1.965 terminales y el cubo

### C59. Verificador independiente descriptor-first

Crear un consumidor separado que:

1. derive la poblacion exacta desde el censo ligado a bytes;
2. exija exactamente un terminal por `variable_id`, sin extras, faltantes ni duplicados;
3. compruebe filename-id, esquema y tipos exactos, JSON sin duplicados/no-finitos y self-digest;
4. recompute seleccion de apariencia, digest fisico, ventana y descriptores desde los bytes fuente;
5. distinga valor ausente de valor malformado; `errors="coerce"` no puede convertir corrupcion en missing silencioso;
6. valide orden temporal antes de llamar causal al primer 70 %.

El replay puede leer el lago una vez; debe reportar bytes y costo reales. No reescribir los terminales v1.

### C60. Terminales y bindings v2

Emitir terminales v2 aditivos con `schema`, `variable_id`, `appearance_id`, `source_sha256`, contrato y digest de ventana, outcome, descriptores, identidad de codigo y self-digest. Para ejes temporales o casos no medidos, usar `BOUND_TO_CENSUS_ONLY` o la clase factual correspondiente; nunca `BOUND_TO_SOURCE_BYTES` con el digest global del censo.

### C61. OLAP y outbox

1. Cada batch liga la lista o manifest digest exacto de variables y terminales miembros.
2. El loader ingiere exitos, fallos, inconclusos y no-identificables de forma idempotente.
3. Las filas antiguas permanecen; las v2 las superseden mediante vistas current, con procedencia visible.
4. Repetir carga produce delta cero.
5. Verificador independiente compara terminal v2, outbox y fila OLAP por observacion.

### C62-C63. Aceptacion

- Reportar el conteo rederivado, aunque no sea 1.505/460.
- Mutar una fuente, terminal, membership de batch o binding OLAP rehusa.
- Invalidar un terminal no se omite: produce terminal de verificacion fallida y entra al cubo.
- Los 39 experimentos y 1.404 performance historicos no cambian.
- Loader queda activo, backlog adjudicado y `NRestarts` reportado; no reiniciar PostgreSQL/Metabase reales.

## P1-E. CRISP-DM C64-C66: convertir C52 en un diseno ejecutable honesto

### C64. Estado y dependencia

Renombrar el Markdown v1 a `DRAFT_CANDIDATE_NO_SCORES_COMPUTED`. No sellar v2 hasta que C55-C63 hayan sido revisados. La poblacion sale del DAG y de terminales v2, no de los conteos retirados.

### C65. Diseno v2 ejecutable

Fijar antes de todo score:

- operadores exactos y su procedencia/licencia experimental;
- panels, variables, splits, unidad estadistica y minimo evaluable;
- margenes, intervalos, familia Holm y reglas de abstencion;
- presupuesto CPU, seeds y origenes;
- metrica primaria y no-inferioridad por panel;
- costo como eje separado o regla costo-utilidad con unidades explicitamente convertidas;
- control de anchura exacto para A3, con prueba de que no introduce informacion nueva y presupuesto/modelo pareado;
- withdrawal y resultado nulo ejecutables.

No heredar automaticamente operadores de T1: los que no tengan licencia en el banco publico vuelven a desarrollo. No ejecutar score bajo esta orden.

### C66. Packet unico

Entregar un §7 que empiece por defectos propios y contenga:

1. PRE y POST por cada bloque;
2. tips, ramas y commits alcanzables desde origin;
3. conteos medidos solo en el tip final;
4. estado B4/T2 candidato, demanda en tres niveles, DAG por clases, caracterizacion v2 y OLAP;
5. lista exacta de trabajo no ejecutado;
6. disposicion `READY_FOR_MUSASHI_EXTERNAL_REVIEW`, sin autorizar nada.

## P2 opcional, despues de P0/P1

Reproducir offline E2 del runner Alpaca solo si puede hacerse desde evidencia congelada sin tocar el proceso live. Si requiere interactuar con el runner o el venue, declararlo `DEFERRED` y no bloquear esta orden.

## Secuencia de ejecucion

1. En paralelo CPU: B4 R15-R18, T2 R11-R15 y C53-C56.
2. Luego C59-C63, consumiendo el DAG corregido.
3. Solo despues redactar C64-C65; cero scores.
4. Suites focales y suites completas al tip final.
5. Packet C66 y alto obligatorio para revision de Musashi.

**Empiece inmediatamente. No espere al reinicio del host: esta orden no necesita GPU.**
