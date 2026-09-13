# Gobernanza transversal de datos y resultados

**Fecha:** 2026-09-13  
**Estado:** `ACCEPT_DIRECTION / REVISE_BEFORE_MANDATORY_BETA`  
**Revision:** propuesta `data-gov/docs/05_PROPUESTA_MUSASHI_BETA_2026_09_13.md`,
contrato `data-gov/docs/04_FLOW_V2.md` y codigo en `data-gov@f6b671a`.

## 1. Veredicto

La arquitectura general es apropiada: los datos permanecen en sus lagos, un
servicio central decide el acceso y registra los bytes entregados, y los
resultados llegan al cubo por una interfaz comun. Esto resuelve el problema
real que motivo el sistema: copias manuales de CSV, identidades ambiguas y
metricas cuyo conjunto de datos ya no puede reconstruirse.

La beta aun no debe convertirse en obligatoria. El codigo actual demuestra el
flujo nominal de una corrida exitosa, pero no demuestra todavia el contrato
completo que necesita el programa. Las correcciones siguientes son pequenas en
numero de conceptos y deben cerrarse antes de gobernar nuevas campanas.

## 2. Hallazgos bloqueantes

### G1. La clave no registra un experimento

La politica vigente autoriza por actor, lago y verbo. El cliente elige
`experiment_key` y `experiment_set_key`; no existe un manifiesto previo que
ligue esa identidad al codigo, configuracion efectiva, datos solicitados,
holdout, unidades de trabajo y destino de resultados. Por tanto, el sistema
autentica al agente, pero no autoriza un experimento definido.

**Correccion:** una sola submission por campana. El servicio devuelve una
identidad direccionada por contenido y las unidades heredan ese contrato. Un
lote de cien experimentos no requiere cien ceremonias.

### G2. "Descargado" se registra antes de terminar la transferencia

`api_download` registra `allow` despues de hashear el descriptor del servidor,
pero antes de que el cuerpo se haya entregado y el cliente lo haya verificado.
Ademas, un cache hit del cliente cierra esa nueva respuesta sin consumirla. El
registro prueba que el servidor ofrecio unos bytes, no que el consumidor los
recibio y comprobo.

**Correccion:** separar `delivery_authorized` de `delivery_verified`. El
servidor entrega un `delivery_id`; tras verificar el archivo o su cache local,
el cliente confirma digest y tamano. Solo esa confirmacion puede satisfacer el
linaje de un resultado. La reutilizacion de cache se registra como tal y no
simula otra transferencia.

### G3. Solo se pueden publicar exitos

El contrato exige una lista no vacia de metricas y `governed_run.py` reporta
solo despues de una salida exitosa. Un fallo, rechazo, resultado inconcluso o
cuarentena queda en un JSON local que el cubo no conoce. Eso contradice la
regla del programa de conservar todos los desenlaces y sesga la memoria hacia
los exitos.

**Correccion:** terminal unico por unidad con estado `COMPLETED`, `FAILED`,
`INCONCLUSIVE`, `REFUSED` o `QUARANTINED`; razon tipada, tiempos y costos. Las
metricas son opcionales segun el estado. Todo terminal entra primero a un
outbox durable y se envia por lotes de forma idempotente. Una caida temporal de
data-gov o PostgreSQL no cambia el resultado cientifico ni lo pierde.

### G4. El holdout no usa aun el contrato temporal del dato

Inferir una columna por su nombre y comparar su reloj de pared no distingue
`event_time` de `available_time`. Una variable derivada puede haberse calculado
o publicado despues del instante que representa. Quitar una zona horaria
tambien pierde la identidad del instante.

**Correccion:** cada recurso declara columna de evento, columna de
disponibilidad, zona, unidad de epoca, frecuencia y politica de corte. El
holdout se aplica a `available_time`. No hay autodeteccion en una corrida que
pueda gobernar una decision; un recurso sin contrato queda `UNAVAILABLE`.

### G5. La identidad fisica y los cortes necesitan una sola autoridad

La memoizacion actual usa ruta, tamano y `mtime_ns`; una reescritura del mismo
tamano y marca puede reutilizar un digest rancio. La publicacion del corte usa
`exists` seguido de `replace`, de modo que dos procesos pueden sustituir el
mismo destino. La afirmacion "materializado una vez" aun no esta garantizada.

**Correccion:** leer y hashear desde un descriptor retenido; ligar dispositivo,
inodo, tamano, `mtime_ns` y `ctime_ns`; revalidar antes y despues cuando se use
el path declarado. El corte se publica `write-once` con eleccion exclusiva. Si
ya existe, se compara el digest completo; una diferencia se rehusa.

### G6. La identidad del resultado esta incompleta

Varias metricas con la misma tripleta `(metric, split, horizon)` conservan el
orden recibido y no tienen identidad unica. El reporte tampoco liga una
version limpia y reproducible del codigo: `-dirty` describe el problema, pero
no identifica sus bytes.

**Correccion:** definir una clave completa y unica de metrica, rechazar
duplicados y ordenar por esa clave. Una corrida gobernante exige commit limpio
o manifiesto de codigo por archivo. Una corrida con codigo no identificable se
conserva como `NON_GOVERNING`, nunca como evidencia.

## 3. Flujo operativo final

```text
1. SUBMIT CAMPAIGN
   actor + manifiesto de codigo/config + solicitudes de datos + unidades
                         |
                         v
2. AUTHORIZE AND DELIVER UNIQUE DATASETS
   descriptor -> hash -> stream/cache -> client verification receipt
                         |
                         v
3. RUN LOCALLY
   cero llamadas de gobernanza en el ciclo interno de entrenamiento
                         |
                         v
4. WRITE ONE TERMINAL PER UNIT TO A DURABLE OUTBOX
   success/failure/inconclusive/refused/quarantined + metrics + costs
                         |
                         v
5. BATCH REPORT -> DATA-GOV -> CONFIGURED OLAP ADAPTER
                         |
                         v
6. RECONCILE ACCOUNTING, DELIVERY RECEIPTS AND CUBE
```

El experimento no recibe credenciales ni un enlace de escritura a PostgreSQL.
Envia un reporte al servicio; el adaptador configurado del lado de data-gov
escribe en el cubo local o remoto. Asi cambiar de host no cambia el contrato de
los agentes.

## 4. Que queda gobernado

La regla "todos los experimentos" se interpreta sin una escapatoria oculta:

- toda ejecucion que pueda cambiar una hipotesis, una seleccion, un modelo, un
  parametro, una promocion o una decision operativa es `GOVERNING` y debe usar
  el flujo completo;
- pruebas unitarias, pruebas de integracion y sondas mecanicas pueden ser
  `NON_GOVERNING`, para no convertir el desarrollo en burocracia;
- un resultado `NON_GOVERNING` no puede conceder elegibilidad. Si resulta
  interesante, se repite bajo un manifiesto gobernante antes de usarlo;
- todos los desenlaces gobernantes llegan al OLAP, no solo los verdes.

## 5. Integracion con los repositorios

### `financial-data`

Sigue siendo propietario de los bytes fuente, contratos temporales, licencias
y cortes. data-gov no transforma ni interpreta la serie: autoriza, transporta
y registra la identidad entregada.

### `preprocessor` y `feature-eng`

Los operadores reciben artefactos gobernados y producen nuevos artefactos
direccionados por contenido. Cada salida registra padres, codigo, parametros,
estado ajustado, particion, tiempo de disponibilidad y digest. No se copian
archivos entre repositorios como autoridad; los consumidores los obtienen por
su identidad. `feature-eng` puede conservar su arquitectura de plugins, pero
sus plugins historicos siguen siendo candidatos hasta pasar F0-F5.

### `predictor`, `agent-multi` y `heuristic-strategy`

Cada campana registra una vez sus unidades. Los procesos descargan cada
artefacto unico una vez por host, verifican cache y ejecutan localmente. El
terminal se escribe aunque el entrenamiento falle. Ninguna llamada al servicio
vive dentro de un paso de gradiente o del entorno.

### DOIN

El ETL del registro distribuido se vuelve un productor del mismo esquema de
terminales. Lee eventos por lotes, conserva identidad de bloque/transaccion,
deduplica por identidad fuente y reporta a data-gov. El cubo destino se elige
en la configuracion del adaptador servidor; DOIN no necesita acceso directo a
la base remota.

### Live

La gobernanza no autoriza trading. Colectores y evaluaciones paper/live usan
campanas distintas, contratos temporales propios y terminales separados. Un
artefacto cientificamente elegible vuelve a validarse antes de adquirir
autoridad operativa.

## 6. Reja de rendimiento y simplicidad

La gobernanza no debe convertirse en parte del costo del aprendizaje:

- una submission por campana, no por epoca ni por lote;
- una transferencia por dataset unico y host, despues solo verificacion de
  cache;
- cero llamadas remotas en `fit`, `transform`, `step` o `learn`;
- terminales y metricas enviados por lotes desde el outbox;
- numero de operaciones de control proporcional a datasets y unidades, no a
  observaciones ni actualizaciones;
- la beta mide latencia, bytes, almacenamiento y recuperacion. El umbral de
  aceptacion se fija antes de la primera campana cientifica obligatoria.

## 7. Posicion en el work plan

La gobernanza es una envoltura transversal, no un nuevo paso cientifico:

| Momento | Obligacion |
|---|---|
| I0 | submission de campana y clasificacion `GOVERNING`/`NON_GOVERNING` |
| D0/I1 | registro de recurso, licencia y contrato temporal |
| D1-F5 | recibos de entradas, operadores, salidas y terminales |
| I5-I9 | espacio, datos, costos y resultados congelados por campana |
| I10 | campana operativa nueva; ningun permiso se hereda del offline |
| OLAP | carga por data-gov y reconciliacion con accounting |

La beta puede desarrollarse en paralelo con el cierre de D2, pero debe estar
aceptada antes de la siguiente ejecucion cientifica nueva de D3-F5. No reabre
ni reetiqueta campañas historicas.

## 8. Criterios de salida de la beta

1. Campana de cien unidades registrada una vez; miembros no declarados
   rehusan.
2. Descarga real y reutilizacion de cache quedan diferenciadas, verificadas y
   ligadas al terminal.
3. Exito, fallo, inconcluso, rechazo y cuarentena llegan al cubo una sola vez.
4. Caida del servicio o del cubo deja outbox pendiente y recuperable, sin
   perder el terminal.
5. Mutacion del fuente, sustitucion de path y carrera de dos materializadores
   no pueden publicar identidades ambiguas.
6. Holdout se decide por `available_time` declarado; zona, epoch y columnas
   ambiguas rehusan.
7. Metricas duplicadas, no finitas o sin clave completa rehusan.
8. Reconciliacion produce cero reportes huerfanos o los conserva tipados hasta
   su reparacion.
9. Predictor, un adaptador DOIN y un adaptador de estrategia producen el mismo
   esquema terminal en una base desechable.
10. La medicion demuestra ausencia de llamadas por observacion o gradiente y
    deja un umbral de costo predeclarado para operacion.

## 9. Decision del propietario

La instruccion del propietario en esta conversacion autoriza el reinicio
acotado de los servicios 5055, 5056 y 5057 **solo despues** de que la version
corregida pase el ensayo de extremo a extremo en puertos alternos y base
desechable. No autoriza borrar datos, cambiar credenciales, abrir live ni
reiniciar PostgreSQL o Metabase.

No queda una pregunta burocratica adicional para el propietario. Si el agente
ejecutor no puede reiniciar servicios por sus propias reglas, debe entregar el
comando exacto y el estado verificado al operador; no debe debilitar la reja.
