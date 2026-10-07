# Orden literal para Satoshi (Sonnet): fase 4

Base: `codex/fs4-automation-20261006` en predictor. No rediseñar el plan.
Leer `docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md`
y el contrato `BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md` una vez.
No vigilar logs. Ejecutar los comandos y consultar `status` al cerrar una unidad.

## Hechos y prioridad

Fases 1-3 cerradas para EURUSD y ETH. El controlador inicializó
`~/.local/state/canonical_20261003/fs4/queue_v2.sqlite`, plan
`2b51180811e2e3ac03814d9bd1adc16a09c7b9cd122f949c1d9d024f70afcc`:
355 series EURUSD + 78 ETH, cinco pliegues, tres brazos = **6495 tareas**.
Las diez `cal.*` EURUSD son contexto conocido. La cola nueva aún no contiene
resultados. El viejo `queue.sqlite` fue un ensayo de diseño: no usarlo.

La prioridad absoluta es M1/M2: obtener evidencia de extractibilidad y la
comparación semanal de subconjuntos. NEAT, RL, núcleo y calendario de entrada
no entran en esta orden. No publicar snapshot de 3.4 GiB en GitHub. Dos copias
físicas verificadas y manifiestos SHA-256 bastan.

El piloto de coste `px.rv5` fue lanzado por Codex en dragon bajo
`fs4-cost-pilot-20261007.service` (4 GiB, 30 min). Leer su terminal y pico al
terminar; **no volver a lanzarlo**. `tools/fs4_deploy/cost_pilot.sh` registra
los argumentos exactos. Es diagnostico de coste, no un recibo de la cola.

## Paso 0: comprobar la cola, sin GPU

Desde el checkout de predictor en la rama base:

```bash
FS4_DB="$HOME/.local/state/canonical_20261003/fs4/queue_v2.sqlite"
python tools/fs4_campaign.py --db "$FS4_DB" status
python tools/fs4_campaign.py --db "$FS4_DB" status --population EURUSD --feature px.rv5 --fold inner_2023 --arm TRAINED_ENCODER
python -m pytest -q tests/test_fs4_campaign.py
python tools/fs4_campaign.py --db "$FS4_DB" list --feature px.rv5 --fold inner_2023 --arm TRAINED_ENCODER
```

Si `status.total != 6495` o hay tareas completas antes de ejecutar esta
orden, detener el despacho y reportar la diferencia con su ruta. No recrear
la cola ni bajar el límite de memoria para forzar una admisión.

## Paso 1: runner científico real (feature-extractor)

Usar como base el código ya entrenado en `feature-extractor@8987c573`:
`app/univariate_temporal_pilot.py`, `app/univariate_temporal.py` y sus
métricas. Revisar que el checkout de esa revisión resuelve esos módulos; no
copiar código manualmente. Crear `app/fs4_task_runner.py` con contrato:
stdin = el JSON de `claim`; stdout = un solo JSON `COMPLETE` compatible con
`tools/fs4_campaign.py::_validate_result`. Logs van a stderr. El adaptador
resuelve **todos** los `feature_id` de fase 3 desde los parquets TRAIN
gobernados de PS1, no desde los lotes parciales de PS2. Verificar cobertura:
355/355 EURUSD y 78/78 ETH o estado explícito por característica.

Ejecutar solo el brazo pedido en la tarea y el pliegue pedido; ventana 24,
semilla 0, latente temporal seis pasos, Conv1D causal. `RAW` es entrada
perturbada; `RANDOM_ENCODER` comparte arquitectura/pesos iniciales con
`TRAINED_ENCODER` y no ejecuta updates; `TRAINED_ENCODER` usa early stopping
TRAIN-only. Máscaras de corrupción y filas de puntuación derivan de la
identidad de característica/pliegue, **sin incluir brazo**, por eso los tres
terminales deben tener los mismos `rows_sha256`, `mask_sha256` y
`population_n`. El runner calcula `mae` y `naive_mae` en los puntos
ocultos, emite digesto real de input/código/modelo, pesos elegidos y coste
en metadatos adicionales. Si no hay observaciones TRAIN, devolver rechazo
tipado; jamás número cero ficticio. El valor del target predictivo no entra
al encoder. Poner pruebas de futuro perturbado, alineación exacta, control
aleatorio sin optimizer y reinicio desde terminal retenido.

Un solo test de integración en Dragon con `px.rv5`, `inner_2023` y los tres
brazos, bajo admisión fresca. Antes de medir, verificar en proceso el UUID
físico y el dispositivo TensorFlow. Comenzar con cap medido por un piloto de
coste; producción = 1.25 × pico si cabe. No interpretar el piloto como
resultado científico. Gamma 5090 mostraba `Unknown Error` al último sondeo:
no despachar allí hasta que `nvidia-smi -L` y TensorFlow vean el UUID correcto.
La 4090 de Dragon estaba sana; la 5070 Ti puede atender segunda unidad si
memoria y temperatura admiten el trabajo. Omega queda para control y OLAP.

## Paso 2: enlace durable, sin monitoreo de agentes

Publicar el runner fijado en los dos workers con commit y env aislado.
`fs4_worker.py` ejecuta una tarea por invocación bajo `crispdm-run`, conserva
el resultado local atómico y lo entrega al coordinador por SSH. Para el
piloto seleccionar el ID con `fs4_campaign.py list`; pasar `--task-id` al
worker. Una vez comparados tres recibos emparejados, instalar una unidad
`systemd --user` con timer de 2 minutos por slot, `--max-tasks 1`, cap
medido, y revisión de salud del host antes de cada job. No crear un loop de
agente; systemd continúa la cola. Una entrega rechazada cuenta `FAILED` y
reporta causa; nunca se marca `COMPLETE` por exit code 0 únicamente.
Los slots CPU usan `--arm RAW` o `--arm RANDOM_ENCODER`; los slots GPU usan
`--arm TRAINED_ENCODER --gpu-uuid GPU-<uuid-medido>`. El proceso hijo
reverifica el UUID; no hay repliegue a CPU. Los 3 brazos de una misma serie
deben usar el mismo digest de filas y máscara, impuesto por el coordinador.
Cuando el runner y el cap ya estén probados, crear en cada host
`~/.config/fs4/<slot>.env` con permisos 0600 y estas variables exactas:

```text
FS4_PYTHON=/usr/bin/python3
FS4_CODE=<checkout-de-predictor-en-este-host>
FS4_COORDINATOR=<alias-SSH-del-coordinador-desde-este-host>
FS4_CONTROLLER=<ruta-absoluta-en-el-coordinador>/tools/fs4_campaign.py
FS4_DB=<ruta-absoluta-en-el-coordinador>/fs4/queue_v2.sqlite
FS4_OWNER=<rol>-<slot>
FS4_RUNNER=<ejecutable-fijado-de-feature-extractor>
FS4_CAP=<cap-medido>
FS4_ARM=RAW|RANDOM_ENCODER|TRAINED_ENCODER
FS4_OUTPUT_ROOT=<directorio-local-durable>
FS4_GPU_UUID=<uuid-fisico-solo-para-TRAINED_ENCODER>
```

Los campos entre `<>` se resuelven desde rutas ya comprobadas en el host;
`FS4_DB` apunta al SQLite del coordinador, no a una copia. Instalar con
`bash tools/fs4_deploy/install_host.sh <slot>`; la unidad y timer quedan en
`systemd --user`. Sondear sin agente con
`systemctl --user status fs4-worker@<slot>.timer` y
`python tools/fs4_worker.py --status` con el mismo bloque de argumentos
de conexión del slot. Si el runner o el UUID no son reales, el instalador
rechaza antes de activar el timer.

El controlador debe exponer siempre:

```bash
python tools/fs4_campaign.py --db "$FS4_DB" status
```

Resultado: esperado/completo/fallido/activo por población, workers y ETA
basada en duración medida; `null` si todavía no hay base. Un recurso liberado
toma la siguiente tarea; priorizar 5090 si se recupera, luego 4090 y 5070 Ti.
Una tarea terminada conserva su terminal y no se repite. Máximo tres intentos
para fallo técnico, una semilla científica.

## Paso 3: warehouse y cierre extractivo

Añadir tabla versionada `feature_extractibility_v1` con clave de tarea,
población, feature, pliegue, brazo, identidad de filas/máscara, n, MAE,
naive pareado, digestos de modelo/input/código y coste. Enviar cada terminal
al OLAP con escritura idempotente y lectura de vuelta. El cierre falla si
faltan tareas admitidas, si hay resultados falsos/no finitos, si una pareja
usa otras filas o si falta un recibo. Guardar `EXTRACTIBILITY_COMPLETE.json`
y un `STATUS.json` generado, no redactado a mano.

## Paso 4: wrapper semanal y manifiesto final

Implementar el adaptador del predictor temporal modular como R0 simple
(Conv1D por rama, conservar tiempo, fusión y núcleo temporal) para comparar
subconjuntos bajo idéntico presupuesto. No usar un MLP que aplane el tiempo.
Unir subconjuntos idénticos por target. Sellar en TRAIN la frontera que se
evaluará con los nueve métodos de fase 3 y los cuatro controles; conservar
la regla de elección y el denominador. Adaptar el cierre semanal existente
`tools/fs_close_weekly.py`, sin crear otro calendario: full retrain móvil de
cuatro años antes de cada semana de VALIDATION, misma semilla, datos
`available_time <= cutoff`, naive en las mismas filas y una disposición para
cada semana. Registrar el coste de todas las semanas. Escoger el ganador por
regla predeclarada sobre **todas** las semanas; no leer TEST hasta congelar
procedimiento/manifiesto. El modo literatura permanece separado.

## Retorno mínimo, sin gastar tokens en vigilancia

Después de cada bloque: un JSON de `status`, commit publicado, salida de
pruebas, primer recibo real y una frase con siguiente acción automática. Si
se detiene una unidad, nombre exacto del objeto faltante y ETA del carril;
otros carriles elegibles continúan. Informar inmediatamente si una tarea
reutiliza filas distintas entre brazos, hay OOM de escritorio o un proceso
intenta entrenar en CPU al pedir GPU.
