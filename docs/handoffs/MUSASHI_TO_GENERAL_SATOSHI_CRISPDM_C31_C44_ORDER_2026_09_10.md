# Orden Musashi a General Satoshi: CRISP-DM C31-C44

**Fecha:** 2026-09-10
**Prioridad:** P0/P1, CPU y almacenamiento local
**Base revisada:** `predictor@16fbbbfdac57e5c8ee67ea2dedf1e41817f8061e`
**Dictamen rector:**
`docs/audits/MUSASHI_AUDIT_CRISPDM_C17_C30_RETURN_2026_09_10.md`

## 1. Objetivo

Cerrar las identidades falsas restantes de C17-C30 y continuar el plan
data-centric sin saltar a seleccion. Al terminar deben existir:

1. una identidad exacta de los componentes realmente ejecutados;
2. sujetos distintos para cada lado, rol y particion de `x/y`;
3. un modelo OLAP `campaign -> run -> unit` que admita historia real;
4. caracterizaciones ligadas a fuente, ventana, codigo y protocolo;
5. linaje temporal de las 94 columnas realmente consumidas;
6. una primera caracterizacion publica y un plan ejecutable para las 1.965
   variables del lago.

**Fronteras:** sin GPU, seleccion cientifica, confirmacion, live, venue,
operaciones, promocion, activacion de colector ni publicacion de genes DOIN.
No truncar ni reemplazar el OLAP existente.

## 2. PRE obligatorio

Antes de editar, congelar por APIs publicas los siguientes contraejemplos:

1. `predictor_plugin="cnn"` + `plugin="ann"`: se ejecuta CNN y C20 registra
   ANN o ningun predictor;
2. mutar `predictor_plugins/common/base.py` sin cambiar el digest C20;
3. checkout limpio sin `predictor.egg-info`: las cuatro pruebas C20 fallan;
4. misma columna `SAME` en `x` y `y`: falta el sujeto `side=y`;
5. schemas distintos de `y` entre train/validation/test aceptados;
6. misma `campaign_key` con un run FAILED y otro COMPLETE: el segundo refusa;
7. cargar observacion nueva y luego una antigua: la antigua queda `current`;
8. una caracterizacion de otros bytes con igual descriptor produce el mismo
   `observation_sha256`;
9. un NaN intermedio elimina una posicion y cambia el significado de lag 1;
10. `DATE_TIME` entra en C30 como variable numerica;
11. fallo del outbox: no aparece brecha junto a `results_dir` porque el valor
    fue capturado como `None`;
12. el heartbeat permanece `healthy=false` por un dead-letter ya conocido y
    no distinguelo de una caida del loader.

El PRE debe guardar salida y codigo reproductor. No usar literales privados
ni topologia de host en evidencia publica.

## 3. P0: identidad ejecutante

### C31. Un solo resolvedor para seleccionar y ligar plugins

- La clave canonica del predictor es `predictor_plugin`, la misma que consume
  `app/main.py`.
- Si se conserva `plugin` como alias legacy, ambos valores deben coincidir o
  el run rehusa antes de construir el modelo.
- El resolvedor debe devolver un testigo que sea consumido por **el loader y
  la identidad**, no hacer dos busquedas independientes.
- Duplicados `(group, name)` entre distribuciones deben rehusar o resolverse
  por una politica explicita y ligada; nunca "el primero" silencioso.
- El terminal debe registrar el predictor canonico, no el alias antiguo.

### C32. Cierre transitivo del codigo local

La identidad debe cambiar al mutar cualquier modulo local que pueda afectar el
run. Puede lograrse con cierre transitivo de imports probado, o con una
superficie de paquetes locales completa y finita. Debe incluir como minimo
`app`, `eligibility`, el pipeline, target, preprocessor, optimizer, predictor y
sus modulos compartidos. Ligar tambien `setup.py` o la metadata de entry points
que decide el registro.

Para dependencias externas, registrar nombre de distribucion, version y
ubicacion logica verificable; no fingir que hashear un solo wrapper cubre
TensorFlow, NumPy o scikit-learn.

Aceptacion: mutar `common/base.py`, `common/losses.py`, una utilidad del
pipeline o el entry point cambia la identidad; un checkout limpio instalado
en entorno aislado reproduce el mismo inventario y digest.

### C33. Sujetos `x/y` completos

La identidad minima de sujeto es `(dataset, side, contract_role, column)`.
`x::SAME` y `y::SAME` son sujetos distintos. Exigir:

- schema de `x` constante entre particiones;
- schema de `y` constante entre particiones;
- target declarado presente exactamente donde el contrato dice;
- cero colisiones de `subject_id`;
- manifest y submission con objetos de sujeto, no solo una lista ambigua de
  strings.

Mutar, omitir, renombrar, duplicar, cambiar de lado o cambiar de rol debe
cambiar identidad o rehusar por su causa exacta.

### C34. `SUBMIT_ONLY` realmente no ejecutante

Separar descubrimiento de ejecucion. `SUBMIT_ONLY` puede leer config, headers,
bytes y metadata de componentes, pero no debe:

- construir modelos ni objetos de entrenamiento;
- inicializar contexto CUDA;
- crear logs/resultados/checkpoints;
- llamar `set_params`, optimizer o pipeline.

Si los defaults de plugin son parte del contrato, extraerlos de una superficie
declarativa o cargar metadata bajo un proceso de inspeccion CPU sin construir
la instancia. Probar por snapshots de filesystem y por visibilidad de device.

## 4. P0: terminal y OLAP

### C35. Esquema `campaign -> run/attempt -> unit`

Migracion **aditiva**:

- `campaign_key` nombra la pregunta/campana estable;
- `run_id` o `attempt_id` identifica una ejecucion irrepetible;
- codigo, diseno, clase de resultado y terminal pertenecen al run;
- las unidades pertenecen al run y al envelope;
- reemitir el mismo envelope es idempotente;
- otro intento del mismo experimento entra como historia nueva;
- FAILED seguido de COMPLETE, o codigo v1 seguido de v2, no colisiona.

Migrar las 123 unidades sin borrar ni reinterpretar hechos. Publicar conteos
antes/despues y consultas que prueben separacion por clase y version.

### C36. Terminal y brecha operacional ligados

Hacer perezosos `config`, `campaign_key` **y `results_dir`**. La brecha por
fallo de outbox debe ser write-once, ligada al envelope/run y no compartir un
nombre fijo sobrescribible entre corridas. Probar COMPLETE, FAILED,
INCONCLUSIVE, REFUSED y QUARANTINED, incluido fallo de outbox.

### C37. Salud operativa sin borrar dead-letters

Conservar el envelope fallido actual y su razon. Introducir estados separados:

- backlog reintentable;
- dead-letter sin adjudicar;
- dead-letter adjudicado/supersedido;
- salud y frescura del proceso loader.

`healthy` no puede equivaler a "nunca existio un fallo historico". Debe decir
si el servicio esta vivo, fresco y sin backlog vencido, mientras otra metrica
expone fallos no resueltos. Adjudicar el dead-letter existente sin borrarlo.
Confirmar `enabled`, `Linger=yes`, dos latidos y `NRestarts`.

### C38. `current` por cronologia cientifica

Agregar fecha/epoca de observacion y/o cadena de supersesion ligada al artefacto.
Las vistas `current` deben seleccionar la ultima observacion cientifica, no la
ultima carga. Tests obligatorios: old->new, new->old, repeticion y dos ramas
incomparables; este ultimo caso debe rehusar o quedar explicitamente ambiguo.

## 5. P1: caracterizacion ligada y ampliada

### C39. Contrato de observacion de descriptor

Cada fila de caracterizacion debe ligar al menos:

- banco y autoridad;
- `variable_id`, lado/rol y particion;
- dataset/fuente y digest de bytes;
- ventana exacta o digest de indices/timestamps;
- protocolo/descriptores y codigo ejecutado;
- valor, identificabilidad, unidades, costo y fecha de medicion;
- intento terminal del cual nacio.

El `observation_sha256` se rederiva de todos los hechos cientificos. El costo
puede tener identidad de medicion separada si se desea repetir rendimiento,
pero no puede desaparecer silenciosamente por una colision.

### C40. Semantica temporal con faltantes

- Excluir identificadores temporales del universo numerico y almacenarlos como
  contrato de eje.
- Autocorrelacion usa pares finitos separados por el lag original; no elimina
  filas antes de desplazar.
- Espectro rehusa con faltantes o usa una imputacion predeclarada y reportada.
- Ningun `inf`/`NaN` termina como `identifiable=true` con texto numerico.
- SNR conserva alineacion exacta entre observado y referencia.

Re-medicion C30 queda en una **nueva version**; no editar ni borrar las 420
filas piloto.

### C41. Banco publico y poblacion financiera

Ejecutar CPU en orden barato:

1. un piloto publico sobre particiones de desarrollo del banco T2 ya adquirido,
   sin tocar confirmacion;
2. las 94 variables model-facing una vez resuelto su linaje;
3. las 1.965 variables conceptuales del lago, por lotes deterministas, con
   `MEASURED`, `NOT_IDENTIFIABLE`, `UNAVAILABLE` o `FAILED` para cada una;
4. generadores sinteticos suficientes para calibrar cada descriptor que
   reclame una relacion con ruido o complejidad.

No imponer que todo tenga numero. La ausencia identificable es resultado. El
ledger debe reportar cobertura por **variable conceptual** y por aparicion
fisica, no confundir ambas poblaciones.

## 6. P1: linaje de las 94 columnas consumidas

### C42. Derivar demanda desde el consumidor real

El puente no debe tratar todo header ni todo JSON bajo `examples/config` como
consumo activo. Reusar la misma derivacion C33 y excluir `DATE_TIME`, targets y
columnas no consumidas segun rol. Ligar el conjunto a los configs/contratos
activos exactos.

### C43. Linaje de features derivadas y entidad OHLCV

Para las 89 derivadas, inspeccionar los productores reales en `feature-eng` y
`feature-extractor` y materializar un DAG por columna:

```text
fuentes crudas -> transformacion/ventana -> feature derivada
              -> event_time -> earliest_available_time
```

El tiempo disponible se deriva como maximo de las entradas mas latencia causal
de la transformacion. Nunca copiar `event_time` ni inventar latencia.

Para OHLCV, derivar simbolo/timeframe/proveedor desde la procedencia de la
vista model-ready. No elegir uno de 128 candidatos por nombre. Si la vista no
lo porta, corregir el productor para que publique el binding y regenerar una
identidad nueva.

### C44. Disposicion y siguiente puerta

Recalcular cobertura exacta de las columnas supervisadas y RL. Si el conjunto
resuelto sigue vacio, entregar deficit por productor. Si es no vacio, producir
**submission de candidato** para revision, nunca auto-autorizacion.

Solo tras auditoria externa de C31-C44 se podra ordenar el screen de seleccion
CPU. GPU, confirmacion, live y DOIN permanecen cerrados.

## 7. Aceptacion minima

1. Los 12 PRE mueren por su razon exacta.
2. Cambio de cualquier componente ejecutante cambia identidad.
3. Checkout limpio reproduce tests e identidad sin otro checkout instalado.
4. `x` y `y` homonimos producen sujetos distintos.
5. Dos intentos del mismo experimento, incluso fallo->exito, entran al OLAP.
6. Reimportar historia antigua no la vuelve `current`.
7. El heartbeat real queda fresco y separa salud de dead-letter historico.
8. Cada descriptor se verifica desde sus bytes e indices fuente.
9. `DATE_TIME` no recibe descriptores numericos.
10. Hay filas publicas de desarrollo, sin uso confirmatorio.
11. La cobertura de 94 columnas se deriva del consumidor real.
12. El cubo conserva 39 experimentos, 1.404 performance rows, las 123 unidades
    previas y las 420 filas C30 previas, mas las nuevas versiones.
13. Toda ejecucion, fallo e inconcluso entra por el outbox.
14. Tests de mutacion por cada guardia nueva, con conteos leidos del terminal
    despues del tip final.

## 8. Higiene de integracion

Crear una rama de integracion limpia desde la base revisada y aplicar solo los
cambios de C17-C44, sus tests, evidencia y handoffs. No reescribir la rama ya
publicada. No incorporar `*-SAVE-ERROR` ni mezclar documentos doctorales en el
commit de runtime. Entregar mapa de commits origen -> commits de integracion.

## 9. Reporte de retorno

El §7 debe comenzar por defectos propios y luego informar:

- PRE/POST y mutaciones;
- identidad exacta de los seis repositorios;
- conteos OLAP antes/despues;
- estado real del loader y heartbeat;
- cobertura de linaje por etapa;
- cobertura de caracterizacion por banco, variable conceptual y aparicion;
- tiempos CPU y almacenamiento;
- disposicion unica;
- fronteras respetadas.

**No hay decision pendiente del owner para comenzar C31-C44. Proceda.**

## 10. Addendum P0 de recuperacion R1-R6 (2026-09-11)

Ejecutar este addendum **antes** de C31-C44. La maquina reinicio y la
inspeccion independiente encontro estados durables que deben reconciliarse.
No fabricar gates, no reejecutar T2, no reanudar B4 y no enviar operaciones.

### PRE-R

Congelar los seis hechos por interfaces publicas o evidencia durable:

1. B4 v7 tiene dos terminales completos, una tercera celda parcial, nueve no
   iniciadas y `CAMPAIGN_STOP`, pero carece de adjudicacion final de
   cuarentena.
2. T2 tiene 242/242 records y arrays, mientras el heartbeat conserva
   `done=241` y una unidad actual ya terminada.
3. `p1lr-decision@101` reintenta cada minuto porque un exit 4 de
   `ExecStartPre` no queda cubierto efectivamente por
   `RestartPreventExitStatus=4`.
4. el runner Alpaca clasifica como fatal un error de conectividad envuelto en
   `AlpacaPaperError`, aunque el observador read-only conecta correctamente.
5. el loader OLAP esta vivo, fresco y sin backlog, pero `healthy=false` por un
   dead-letter historico.
6. systemd advierte que ignora `StartLimitIntervalSec` por estar en la seccion
   incorrecta del unit file del loader.

### R1. Cierre verificable de B4

- Consumir la autorizacion previa del owner de detener y poner en cuarentena
  la tercera celda; no pedirla de nuevo.
- Verificar por descriptor los dos terminales completos y todo artefacto de la
  tercera celda antes de clasificarlo.
- Emitir un record append-only que distinga `COMPLETED=2`,
  `QUARANTINED_PARTIAL=1` y `NOT_STARTED=9`.
- Re-derivar costos y tiempo consumido. El parcial no entra a comparaciones y
  ninguno de los dos completos se vuelve promocionable por este acto.
- Adjudicar el resultado cientifico como insuficiente o inconcluso segun el
  contrato preexistente; nunca completar por etiqueta las diez celdas que no
  tienen terminal.
- Probar que un relanzamiento contra esa raiz refusa antes de CUDA y que el
  record no modifica ningun artefacto de entrenamiento.

### R2. Cierre de T2 desde 242 records

- Verificar inventario exacto: un claim, un array y un record por cada una de
  las 242 unidades selladas; cero extras, duplicados o faltantes.
- Recalcular MASE, costos y bindings desde arrays y records usando el codigo
  revisado. El heartbeat es telemetria, no autoridad terminal.
- Emitir un cierre durable que explique la discrepancia `241 -> 242` y probar
  que una segunda invocacion es idempotente.
- Ejecutar la adjudicacion cientifica predeclarada y cargar al outbox COMPLETE,
  FAILED e INCONCLUSIVE segun correspondan. No volver a entrenar ni descargar.

### R3. Refusal estable para P1LR historico

- Corregir el unit template e instalador para que un gate ausente o no viable
  produzca una condicion estable, sin `auto-restart` ni proceso de training.
  `ExecCondition` es una opcion valida si conserva una salida tipada separada;
  no depender de semantica de `RestartPreventExitStatus` sobre `ExecStartPre`.
- Preservar la ruta fijada y declarar la instancia historica como no
  ejecutable; **no reconstruir ni copiar** el gate ausente.
- Probar gate ausente, schema incorrecto, resultado negativo y gate viable en
  un manager de systemd de prueba o arnes equivalente. Solo el caso viable
  alcanza `ExecStart`.
- Entregar la disposicion exacta necesaria para que el operador retire la
  instancia obsoleta instalada. No iniciar una nueva decision P1LR.

### R4. Recuperacion Alpaca tipada

- Preservar la causa tipada cuando la capa Alpaca envuelva una excepcion de
  red; no clasificar por texto del mensaje.
- Un fallo de conexion transitorio debe actualizar heartbeat y reintentar con
  backoff acotado. Cuenta, artefacto, schema o binding invalidos siguen siendo
  fatales.
- Probar `ConnectionError` directo, envuelto con `raise ... from`, timeout,
  cuenta equivocada y error de artefacto.
- La correccion no concede permiso, no cancela la orden paper abierta y no
  ejecuta un tick extraordinario. Entregar procedimiento de recuperacion al
  operador despues de la revision.

### R5. Unit file y salud OLAP

- Mover `StartLimitIntervalSec` y `StartLimitBurst` a `[Unit]` y agregar una
  regresion sobre el unit efectivo o `systemd-analyze verify`.
- Ejecutar C35 y C37 sobre el dead-letter real: conservarlo, adjudicarlo y
  separar salud/frescura del loader de la cola de fallos historicos.
- El estado esperado post-correccion es proceso fresco, backlog cero y una
  cuenta separada de dead-letters adjudicados; no borrar evidencia para volver
  verde una bandera.

### R6. Matriz de recuperacion reproducible

Publicar una herramienta read-only y un record de arranque con:

- PostgreSQL y Metabase saludables;
- loader, supervisor, sesiones Dragon/Gamma y timers esperados;
- estado de GPU y procesos de compute;
- B4 y T2 derivados de evidencia durable;
- P1LR refusado sin loop;
- Alpaca runner y observer distinguidos;
- servicios pausados o historicos nombrados como tales.

`active/running` no basta para declarar salud. Cada componente debe aportar
heartbeat fresco, estado terminal o una indisponibilidad tipada. La herramienta
no inicia, detiene ni reinicia procesos.

### Aceptacion del addendum

1. B4 queda 2/1/9 con cuarentena durable y cero relanzamiento.
2. T2 queda 242/242 reconciliado, verificado y adjudicado sin recomputo de
   entrenamiento.
3. P1LR ausente/negativo no entra en restart loop.
4. La taxonomia Alpaca conserva la causa transitoria sin debilitar errores
   fatales.
5. El unit OLAP no produce warnings de directivas ignoradas.
6. El loader puede estar sano con dead-letters adjudicados visibles.
7. La matriz de recuperacion reproduce el estado tras un reboot simulado.
8. Ninguna GPU, seleccion, confirmacion, live, venue o DOIN se abre.

El retorno debe separar con claridad **servicio vivo**, **campana activa**,
**campana terminal** y **campana pausada**. Tras R1-R6, continuar C31-C44 en
CPU. No hay decision adicional pendiente del owner.
