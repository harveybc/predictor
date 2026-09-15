# Revision A1-A5: contrato financiero y adopcion de consumidores

Revisor: Musashi. Fecha: 2026-09-14.
Retorno: predictor `81c5e1df0224868bd6eae6529ff7072536c809ee`.
Contrato: financial-data `1b4a23431a0607eb130a3ac7b25cee0a66a6e378`, PR 1.

## Dictamen

Acepto la reconciliacion operativa y los desenlaces de los doce casos en el
cubo. Acepto la medicion de la geometria temporal con alcance descriptivo.
NO acepto instalar el contrato financiero candidato como evidencia de
disponibilidad historica. La adopcion de los nuevos hosts por consumidor
permanece parcial: los ensayos de entrada usaron el adaptador local legado.

Esta revision no cambia resultados D2, servicios, datos ni terminales.
La continuacion no necesita otra decision del owner:
[ordenes siguientes](../../handoffs/MUSASHI_TO_SATOSHI_TEMPORAL_SEMANTICS_AND_REAL_HOST_ADOPTION_2026_09_14.md).

## Hallazgos, por prioridad

### 1. Alta: cierre de ventana no demuestra disponibilidad

`store/tools/derive_availability_contract.py::candidate_contract` convierte
`close_time` en `available_time_column` y asigna lag cero, mientras la misma
submission declara latencia de entrega UNOBSERVED. Granularidad diaria no
elimina el problema: una barra de fin del dia puede publicarse al dia siguiente.
Una marca UTC demuestra como se representa el reloj, no cuando llego el dato.

Reproduccion sintetica: cierre 23:59:59.999 del dia 1, recepcion 00:10 del
dia 2. El proveedor, bajo el contrato candidato, incluye la fila en el corte
del dia 1. Esto prueba que esa regla no garantiza informacion disponible
antes de su frontera. No prueba que una fila financiera concreta haya sufrido
esa latencia; precisamente no existe la evidencia para decidirlo.

La documentacion primaria de Binance distingue cierre de intervalo y estado
final de una vela mediante campos diferentes. Tampoco ese estado observa la
recepcion local: [WebSocket Streams, Kline/Candlestick Streams](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#klinecandlestick-streams-for-utc).
Es un antecedente semantico, no evidencia de que este archivo se adquiriera
por WebSocket. Satoshi debe seguir el productor real del archivo.

### 2. Alta: las barras truncadas se admiten sin demostrar su finalizacion

`test_a_truncated_bar_is_available_when_it_actually_closed` presupone que
un close temprano es un cierre final. No distingue una barra final de una
observacion parcial o un agregado incompleto. En mi fixture una barra de
duracion cero y `is_final=False` se entrega igualmente.

Las 21 anomalias del recurso se describieron; no se justificaron como barras
finales. Deben clasificarse desde el productor y tener una politica explicita
de inclusion/exclusion, sin completar artificialmente las ocho discontinuidades.

### 3. Alta: las pruebas no sostienen varias de sus etiquetas

La prueba `test_live_equivalent_is_refused_for_this_resource` llama primero
a `availability_scope(live)` SIN esperar un rechazo: el caso live es aceptado.
Luego cambia la evidencia de zona horaria y prueba ese rechazo diferente.
La bateria puede pasar aun cuando no demuestra lo que dice su nombre.

Ademas, la prueba de zona horaria acepta `None`, y el caso intrabar acepta
cualquier Exception. Ambos deben comprobar el resultado o rechazo pretendido.
Rechazar un rango intrabar no es ejecutar y detectar una transformacion
deliberadamente no causal. A4 no cubrio esa prueba de procesamiento.

### 4. Media: el generador de contratos ignora hechos invalidantes

`candidate_contract` devuelve WINDOW_END/0s aun con `measured=False`, tiempos
no monotonos, duplicados y disponibilidad anterior al evento. Lo reproduje
sin datos reales. Un informe descriptivo puede conservar esos hechos; una
salida presentada como contrato soportado por las mediciones no debe ignorarlos.
Esto afecta la herramienta generica, no afirma que el recurso real tenga
esos cuatro defectos.

### 5. Alta: el recorrido probado no pasa por el nuevo host de entrada

`tools/verify_consumer_adoption.py` fija SAMPLE_LAKE=`predictor_examples`.
El catalogo activo registra ese almacen como `files_lake`, mientras
`financial_files` y `governance_smoke` usan `http_lake`. Los recibos coinciden
con esa seleccion. Se probo cliente -> data-gov/adaptador local -> warehouse
nuevo, no cliente -> data-gov -> data-lake/proveedor externo -> warehouse.

Se conserva el exito mecanico de preprocessor y predictor en ese recorrido.
No se los llama plenamente adoptados en la arquitectura nueva hasta probar
la ruta que falta. feature-eng y feature-extractor siguen sin exito de pipeline,
correctamente divulgado por Satoshi; registrar FAILED no los convierte en
consumidores terminados.

### 6. Media: vaciar un outbox vacio no prueba reintento de un terminal

El arnes llama `flush` cuando pending=0 y concluye que un reintento no duplica.
Prueba que ese vaciado no escribe, no que reenviar un terminal o recuperar
una entrega pendiente sea idempotente. La prueba previa de Musashi si reenvio
el mismo terminal, pero no sustituye la verificacion del wrapper de cada
consumidor. Reutilizar evidencia valida del wrapper si existe; de otro modo
reproducir el pendiente y su recuperacion en un destino desechable.

### 7. Media: dos desconocidos se asignaron incorrectamente al owner

Una declaracion del owner no concede derechos del proveedor, y dos capturas
no establecen una politica general de revisiones. La segunda captura puede
mostrar cambios o ausencia de cambios en ese par. Satoshi debe investigar
documentacion primaria y adquisicion antes de solicitar una accion exclusiva
del usuario. Conservar las incertidumbres que esa investigacion no resuelva.

## Verificacion independiente realizada

- Las once pruebas originales pasaron en un checkout separado de PR 1,
  con el recurso real de solo lectura; no se cambio de rama el directorio vivo.
- Cuatro contraejemplos sinteticos reproducidos:
  [script](../evidence/repro_runs/musashi_a1_a5_contract_review_20260914.py) y
  [resultados](../evidence/repro_runs/musashi_a1_a5_contract_review_20260914.json).
- Consulta de solo lectura al cubo: 20 terminales totales; los doce casos
  publicados tienen exactamente el estado que declara su recibo.
- Los cuatro servicios y el loader estaban active, NRestarts=0. No se
  repitio una campana ni se reinicio ningun proceso.

No se repitio la prueba completa de cada consumidor ni la suite completa de
los repositorios. Esta revision separa evidencia inspeccionada, tests ejecutados
y afirmaciones pendientes; no convierte el conteo verde en validacion causal.
