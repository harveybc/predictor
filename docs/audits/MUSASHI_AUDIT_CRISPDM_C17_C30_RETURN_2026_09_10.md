# Auditoria Musashi: retorno CRISP-DM C17-C30

**Fecha:** 2026-09-10
**Retorno auditado:** `predictor@16fbbbfdac57e5c8ee67ea2dedf1e41817f8061e`
**Codigo declarado:** `predictor@456c6e4`
**Repositorios relacionados:** `financial-data@cf2e408f8`,
`preprocessor@c246505`, `agent-multi@c1f4cb97`,
`doin-domains@26ec568`, `doin-plugins@10f6026`

## 1. Veredicto

**`REVISE_BEFORE_SCIENTIFIC_SELECTION`.**

El retorno no se rechaza. Quedan aceptados como avances mecanicos:

1. el protocolo `SUBMIT_ONLY` -> revision externa -> `EXECUTE_REVIEWED`;
2. el bloqueo del optimizador a una lista de hiperparametros y la
   rederivacion antes del pipeline;
3. los esquemas estrictos de manifest, submission y envelope;
4. el outbox local, el loader continuo y la conservacion aditiva del OLAP;
5. el resultado honesto de disponibilidad: hoy **0 de 94 columnas** llega a
   un contrato temporal completo;
6. la caracterizacion C30 como ensayo exploratorio CPU, sin seleccion ni
   autoridad confirmatoria.

No acepto todavia la afirmacion de que quedaron ligados *todo* el conjunto
consumido, *todo* el codigo ejecutado, una unidad terminal estable ni una
caracterizacion reproducible por fuente. Esas cuatro afirmaciones tienen
contraejemplos directos en las APIs publicas.

## 2. Hallazgos P0

### F1. El modelo ejecutado puede quedar fuera de la identidad de codigo

El ejecutor elige el modelo mediante `predictor_plugin`
(`app/main.py:223-233`), pero `PLUGIN_ROLES` busca el rol predictor mediante
la clave `plugin` (`eligibility/consumed.py:171-177`). En una configuracion
con `predictor_plugin="cnn"` y el valor heredado `plugin="ann"`, el pipeline
ejecuta CNN y la identidad registra ANN.

El problema es mayor que el alias. El inventario solo incorpora el archivo
del entry point. Para ANN no incluye, entre otros,
`predictor_plugins/common/base.py`, `losses.py`, `bayesian.py` ni
`positional_encoding.py`, aunque el modulo ejecutado los importa. Cambiar
uno de esos archivos altera entrenamiento y no altera el digest C20.

La bateria confirma ademas una dependencia ambiental: en el checkout de
trabajo instalado pasan los focales, pero en un checkout limpio sin el
`predictor.egg-info` externo fallan cuatro pruebas C20 porque el registro de
entry points no se reproduce. Resultado independiente: **93 passed, 4 failed,
11 skipped**. La identidad no puede depender de que otro checkout este
instalado al lado.

### F2. `x` y `y` no forman sujetos distintos cuando comparten nombre

`subject_id()` solo usa `dataset_id::column` (`consumed.py:55-58`). Al formar
los sujetos de `y`, cualquier columna cuyo nombre ya aparezca en `x` se omite
(`consumed.py:329-335`). El contraejemplo con `SAME` en ambos archivos produce
solo:

```text
d::SAME  side=x  role=target
```

No produce el sujeto `side=y`. Los bytes completos de `y` si participan en el
digest, de modo que una mutacion fisica se detecta; lo falso es la identidad
semantica por lado y rol que el reporte afirma. Tampoco se exige que el schema
de `y` sea igual entre train, validation y test.

### F3. La dimension de campana mezcla identidad estable con hechos de corrida

`dim_campaign` fija por `campaign_key` el `result_class`, `design_sha256` y
`code_identity` (`olap/campaign_envelope.py:84-93`), y el loader rehusa si
cualquiera cambia (`campaign_envelope.py:365-393`). Sin embargo:

- un fallo y un exito del mismo experimento tienen distinto `result_class`;
- una nueva revision de codigo tiene distinto `code_identity`;
- una revision de diseno tiene distinto `design_sha256`.

Por tanto, la segunda corrida legitima de un mismo experimento puede quedar en
dead-letter en vez de entrar como nueva version. El sistema real ya muestra
la consecuencia operativa: hay un envelope en `failed/`, el heartbeat actual
publica `failed=1` y `healthy=false`, aunque el proceso sigue activo.

La unidad estable debe ser `campaign -> run/attempt -> unit`, con el estado,
codigo y diseno en el run o en una dimension versionada, no congelados para
siempre en el nombre de la campana.

### F4. Las 420 filas de caracterizacion no estan ligadas a los bytes medidos

`fact_variable_characterization` no guarda digest de la fuente, ventana,
codigo, protocolo ni artefacto de medicion (`olap/characterization.py:50-76`).
Su `observation_sha256` omite `measured_at`, `cost_seconds`, `identifiable` y,
mas importante, la identidad de los datos (`characterization.py:115-139`).
Dos archivos diferentes que den el mismo valor descriptivo colisionan como
la misma observacion.

Adicionalmente, `run_characterization.py` toma las primeras columnas por
posicion e incluye `DATE_TIME` como si fuera variable numerica
(`tools/run_characterization.py:36-59`). Para descriptores temporales, primero
elimina los no finitos y despues calcula autocorrelacion y espectro; eso
comprime el eje temporal y cambia los lags cuando existen huecos.

Las 420 filas quedan aceptadas solo como **piloto mecanico exploratorio**. No
deben gobernar seleccion ni presentarse como caracterizacion del lago.

## 3. Hallazgos P1

### F5. `current` significa ultimo cargado, no ultima observacion

Las cuatro vistas eligen por `loaded_at DESC`
(`olap/inventory_rows.py:109-131`). Si se reimporta hoy un censo antiguo, este
se vuelve `current` aunque cientificamente haya sido supersedido. El test
existente solo carga antiguo -> nuevo; falta nuevo -> antiguo. Hace falta una
cronologia propia del artefacto o una cadena explicita de supersesion.

### F6. Un dead-letter historico vuelve insano el heartbeat para siempre

`healthy` es simplemente `failed == 0` (`olap/outbox.py:147-163`). Como la
evidencia no se borra, un solo rechazo permanente deja la alarma roja de por
vida. Deben separarse salud del proceso, backlog reintentable y dead-letters
no adjudicados; la adjudicacion debe conservar el envelope y su razon.

### F7. El cierre de terminal no conoce de manera perezosa el directorio real

La configuracion y la clave de campana se leen al terminar, pero
`results_dir=shared.get("results_dir")` se evalua antes de ejecutar `_body`
(`app/main.py:149-158`). Siempre comienza como `None`. Si falla el outbox, el
archivo de brecha no puede escribirse junto a los resultados, contrario a C22.

### F8. `SUBMIT_ONLY` no es literalmente una fase sin ejecucion

Antes de la reja se importa TensorFlow, se consulta el acelerador, se crean
los cinco objetos plugin, se ejecutan sus `set_params` y se inicializan
archivos de log (`app/main.py:172-299`). Se cumple la frontera estrecha de no
correr optimizador ni pipeline, pero la frase "nothing was executed" es
falsa. La fase de submission debe ser realmente de inspeccion CPU y sin
constructores del modelo, o describirse con precision y demostrar que no crea
estado cientifico ni reserva GPU.

## 4. Hallazgo P2 de integracion

El commit C17-C22 incorpora, junto al codigo, **101 archivos documentales y
48.814 lineas** que no pertenecen a la orden, incluidos dos artefactos
`*-SAVE-ERROR`. No invalida por si solo los resultados, pero impide revisar o
integrar C17-C30 como cambio acotado. Se requiere una rama de integracion
limpia que conserve la historia publica y aplique unicamente codigo, tests,
evidencia y documentos propios de esta orden. No borrar ni reescribir la rama
ya empujada.

## 5. Hechos operativos verificados

- `crispdm-olap-loader.service`: `active/running`, `NRestarts=0`.
- La unidad esta `enabled` y el usuario tiene `Linger=yes`; sobrevivira al
  cierre de la sesion. La limitacion contraria del retorno debe corregirse.
- Heartbeat observado: `pending=0`, `loaded=4`, `failed=1`, `healthy=false`.
- Cubo real: 123 unidades de campana, 420 filas de caracterizacion, 1.965
  variables del lago, 4.650 series publicas y 202 generadores sinteticos.
- Caracterizacion real: 348 filas financieras sobre 16 ids y 72 filas
  sinteticas sobre 3 ids. **No hay filas del banco publico**.
- Suite focal en el checkout instalado: **123 passed**.
- Suite focal en checkout limpio: **93 passed, 4 failed, 11 skipped** por la
  dependencia del registro global de entry points.
- Bateria relacionada independiente: **32 passed, 1 skipped**.
- La suite completa conserva los ocho errores legacy documentados; no se usa
  como evidencia de aceptacion total.

## 6. Disposicion

No se abre seleccion cientifica, confirmacion, GPU, live ni publicacion DOIN
desde C17-C30. Tampoco se borra el cubo ni se inicia uno nuevo: la historia es
util y el esquema puede corregirse de forma aditiva.

La siguiente orden corrige F1-F8 y, en la misma campana CPU, convierte el cero
de disponibilidad en un trabajo concreto de linaje: primero las 94 columnas
model-facing, luego el banco publico y despues el resto de las 1.965 variables
conceptuales. El owner no tiene una decision pendiente para abrir ese trabajo.

## 7. Addendum de recuperacion tras reinicio (2026-09-11)

La inspeccion posterior al reinicio de Omega confirma que la infraestructura
base volvio, pero **no** que todo este sano ni que exista trabajo cientifico
activo:

- Docker, PostgreSQL y Metabase estan activos; PostgreSQL acepta conexiones y
  `/api/health` de Metabase responde `ok`.
- `crispdm-olap-loader`, el supervisor DOIN, las sesiones persistentes con
  Dragon/Gamma y el runner Alpaca estan `active/running`, habilitados y con
  `NRestarts=0`. `Linger=yes` esta efectivo.
- el host dispone de 21 GiB de RAM y 216 GiB libres en `/home`; el reloj esta
  sincronizado. No hay presion de recursos.
- no existe proceso CUDA cientifico. La RTX 4070 esta disponible; su memoria y
  actividad observadas pertenecen al escritorio, no a un worker de training.
- B4 v7 **no debe reanudarse**: hay dos terminales `COMPLETED`, una tercera
  celda parcial sin terminal y un `CAMPAIGN_STOP` durable. Falta convertir la
  decision ya autorizada de detener y poner en cuarentena en una adjudicacion
  verificable de la campana; nueve celdas nunca comenzaron.
- T2 conserva 242 claims, 242 arrays y 242 records. El heartbeat final quedo
  en `done=241` mientras nombraba la ultima unidad, aunque su record fisico si
  existe. Hace falta reconciliar y emitir cierre desde los records, sin
  reejecutar ninguna unidad.
- `p1lr-decision@101` es una unidad historica obsoleta que entra en
  `auto-restart`: su gate fijado no existe. El `ExecStartPre` sale 4, pero
  `RestartPreventExitStatus=4` no impide el reinicio de un fallo ocurrido en
  `ExecStartPre`. Esto debe quedar como refusal estable, no como intento cada
  minuto.
- el observador Alpaca confirma que la cuenta paper esta accesible y plana,
  pero el runner modelo envolvio un `ConnectionError` en `AlpacaPaperError` y
  lo clasifico como fatal. Quedo vivo en espera de una hora con heartbeat
  degradado, aunque el observador ya recupero conectividad. Es una falla de
  taxonomia y recuperacion, no una autorizacion para enviar otra orden.
- el loader OLAP publica latidos frescos y backlog cero, pero sigue declarando
  `healthy=false` por el dead-letter historico descrito en F3/F6. Ademas,
  `StartLimitIntervalSec` y `StartLimitBurst` estan bajo `[Service]`; systemd
  los ignora porque pertenecen a `[Unit]`.
- el supervisor DOIN esta sano como servicio, pero la campana visible esta
  `paused` y es solo historia. Las sesiones remotas activas no equivalen a una
  optimizacion corriendo.

Por tanto, el estado correcto es
**`BASE_SERVICES_RECOVERED_WITH_FOUR_RECONCILIATIONS_REQUIRED`**: B4, T2, la
unidad P1LR obsoleta y la recuperacion Alpaca. Estas correcciones se agregan a
la orden C31-C44 y no abren GPU, seleccion ni ejecucion de mercado.
