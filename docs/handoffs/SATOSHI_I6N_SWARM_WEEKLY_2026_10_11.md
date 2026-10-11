# Orden de Musashi a Satoshi: I6-N hasta ejecucion semanal automatica

Fecha: 2026-10-11. Prioridad unica: integrar y ejecutar el sucesor I6-N.
Autorizacion del propietario concedida. No se necesita otra aprobacion para
ingenieria, pilotos admitidos, despliegue interno o etapas descritas aqui.
No hay autorizacion nueva para TEST, capital real o eludir controles de acceso.

## 0. Acuse y forma de trabajar

Lee esta orden y el subplan vinculado abajo UNA vez. Escribe ACK.json con
commit base, cinco responsables, ramas y entregables. Despacha cinco agentes
de codigo independientes; integra en cuanto cierre cada contrato compatible.
Una prueba CPU tiene una reserva real; cinco agentes no significan cinco fits.
No anunciar agente activo hasta tener su acuse.

NO monitorees entrenamientos, no hagas loops sleep/status/log, no gastes tokens
leyendo cada epoca. Construye servicios/timers y status automaticos. Despues de
arrancar un trabajo durable y comprobar un unico latido de entrenamiento real,
regresa al codigo independiente o entrega el retorno y termina tu turno.
El relevo de etapas pertenece al ejecutor, no a la conversacion.

Contrato principal: rama `codex/i6d-weekly-walk-forward-20261009`,
commit `af0c3570`,
`docs/tres_temas_entrevista/program_v3/I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md`.
Junto con BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md manda sobre el borrador
anterior, sus autorizaciones pendientes y orden N1/N2/N3. El master conserva
las demas etapas: no reabrir seleccion ni adelantar RL/NEAT/calendario causal.

Base predictor `codex/i6n-integration-20261011@259e94df`.
48/48 candidatos endpoint acabaron, NO hay fits de esas colas por recuperar.
POST_POWER_RETURN_20261011.md contiene resultados y limitaciones. No repetirlos.
En las 16:54 de Bogota: 396 informes/3015 metricas leidos del warehouse;
5090 funciona; obrero secundario requiere comprobar reinicio por errores
NVIDIA de memoria. El reinicio de ese host no bloquea los otros carriles.

## 1. Preparacion literal, sin cambiar checkouts vivos

Ejecuta desde el repo predictor existente; si el worktree ya existe, verifica
su rama/base y continua alli, no borres ni reinicialices trabajo previo.

```bash
git fetch origin codex/i6n-integration-20261011
git fetch origin codex/i6d-weekly-walk-forward-20261009
git cat-file -e 259e94df^{commit}
git cat-file -e af0c3570^{commit}
git show af0c3570:docs/tres_temas_entrevista/program_v3/I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md
git show af0c3570:docs/tres_temas_entrevista/program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md
git worktree add -b satoshi/i6n-swarm-weekly-20261011 /var/tmp/satoshi-i6n-swarm-weekly-20261011 259e94df
```

Cada agente usa su propio worktree, a partir de esa misma base. Configuracion
de hosts/interpretes/dispositivos: inventario privado existente
`~/.local/state/canonical_20261003/i6e/multihorizon_2024_v2/WORKERS.private.json`.
No imprimas ni publiques credenciales, hostnames, IP o UUID. Los comandos de
entrenamiento nuevos salen de los JSON generados, no de copiar una shell vieja.

Codigo conocido: `tools/i6n_historical_evaluator.py`, `tools/i6n_search.py`,
`optimizer_plugins/i6n_optimizer.py`, `tools/i6n_search_controller.py`,
`tools/i6n_search_publish.py`, `tools/i6n_search_follower.py`,
`tools/i6n_model.py`, `tools/i6n_contracts/temporal_ops.py`,
`tools/i6n_weekly_data.py`, `tools/i6n_training.py`.
Lee AGENTS de cada repo; grafo primero. Las pruebas nuevas preceden al codigo.

## 2. Propiedad y entregables paralelos

### A. Trayectorias y modelo

Propiedad: evaluador historico, entrenamiento/modelo y tests de trayectorias.
Eliminar el rechazo de trajectory SOLO tras implementar targets reales,
no renombrando el endpoint. dx y dy independientes, horas exactas; cada head
tiene C1, mascara, escala TRAIN y naive. La metrica de comparacion sigue siendo
el endpoint h en soporte fijo, mientras los intermedios se guardan separados.
No filtrar filas endpoint por faltantes de heads intermedios. t debe influir
tras todas las reducciones. Invalidos no producen NaN/gradiente. Ramas simples,
sin Flatten en fusion/nucleo; LSTM sequence-valued; atencion GPU estricta.
Regresiones: dx=3,dy=6,h=12; dx=36,dy=24,h=72; pares identicos endpoint;
horizonte no divisor; DST; faltante intermedio; perturbacion futura;
checkpoint/reload; cascada impar conserva t; mascara sin soporte.
Entrega A_RETURN.md + pruebas + esquema de receipt acordado con C/D.

### B. Swarm DOIN y supervivencia a reinicios

Propiedad: wrapper `doin-plugins/src/doin_plugins/predictor/optimizer.py`,
runtime DOIN fuera de archive, unidades y coordinador de campana nuevo.
Usar plugin compartido real, no llamar al controller local y nombrarlo swarm.
Semilla 6102026 y minimizacion en wrapper, rol, DEAP y publicacion. Evaluador
en subprocess admitido: no SIGALRM dentro de un hilo del nodo, ni fallback CPU.
Conservar identidad global, leases y fencing frente a caida/retorno de nodos.
Probar desconexion durante claim, despues de fit antes de ACK y durante
reproduccion. No repetir fit terminado ni publicar score de otro candidato.
Reanudar presupuestos/seeds/generacion y outbox, no arrancar de cero al boot.
Persistir servicios habilitados, healthcheck y recovery sin agente. No
reanudar I6-E-M ni procesos ajenos. A/D/C publican interfaces; B integra.

### C. Semanas completas y datos

Propiedad: builder de campana semanal, enumeracion/split/calendario, manifest
de datos y pruebas de cobertura. No modificar el evaluador de A sin coordinar.
TRAIN cuatro anos por cutoff, inner purgada, refit cada semana, todas las
semanas de 2024. No semana unica ni ano sin reajuste. Generar soporte por
h/semana/rol antes del fit; labels fuera de banda o en 2025 no se abren.
Preparar los cuatro folds internos 2023 del subplan y su historia requerida
sin reabrir seleccion ni acortar TRAIN. Extender datos/reloj cuando falten:
el certificado existente solo cubre diciembre 2019-2024 condicionalmente;
no usarlo como certificado de principios de 2019 o de periodos mas antiguos.
Resolver con fuentes y pipeline existente; conservar limitaciones de vintages.
Pruebas: DST, ano bisiesto, frontera 2024/2025, min_datos, horizonte sin soporte,
semanas incompletas, estado acumulado, suma anual de errores/conteos y naive.
No comparar el score interno de 70 filas con un resultado anual del h12.
Entrega C_RETURN.md, conteos reales y JSON de las unidades semanales.

### D. Archivo externo -> cadena -> OLAP

Propiedad: archive en doin-core/doin-node, adaptador data-lake/warehouse y
publicadores predictor, sin tocar el scheduler de B.
Base a inspeccionar: core `a50a33e`, node rama offchain-shadow `4c23a6f`, lake
`e82b33e`, con sus ancestros. Usa worktrees; no reset sobre ramas vivas.
El ultimo adaptador `DisposableServiceAdapter` NO es despliegue de lago y
mantiene chain_verified=false. No afirmes que el modo ya estaba activado.

Implementa el modo externo conservando compatibilidad full-payload. Debe
guardar primero bytes canonicos completos, releer hash y luego comprometer
header/referencia con enlace previo y datos de consenso requeridos. El hash
de cuerpo no es el hash final del bloque. ETL verifica referencia finalizada,
bytes, esquema y cadena antes de proyectar. El warehouse es proyeccion
analitica, no la unica copia mutable que da significado al hash de la cadena.
Utiliza la cuenta de servicio existente por su interfaz permitida, sin exponer
secretos. SQLite desechable sirve para pruebas, no para certificar despliegue.

Pruebas obligatorias: full/external igualdad analitica; PUT fallido no commit;
crash PUT->commit->ETL->ACK recuperable; bytes/digest/locator equivocados
rechazados; duplicado no suma metricas; reorg revoca proyeccion canonica y
preserva historial; restore snapshot; peer nuevo puede resolver el cuerpo.
Conservar estados separados: LOCAL_COMPLETE, OBJECT_DURABLE,
CHAIN_FINALIZED, OLAP_READBACK_VERIFIED. Faltante no se vuelve verde.
Guardar todos los candidatos, no solo ganadores, con genes/config/costes y
metadatos del subplan. Snapshot fisico en dos hosts; en Git solo instrucciones,
manifiestos, hashes y resumen, ningun .duckdb/.npz/.keras grande.

### E. Espacio de busqueda, configuracion y verificador independiente

Propiedad: `tools/i6n_search.py`, `optimizer_plugins/i6n_optimizer.py`,
generador de configs y pruebas de identidad. Coordinar esquema con A/B.
Crear primero tests de dominio exacto del subplan: doce horizontes, minimo
24h de historia, ventanas/capacidad/dx/dy conjuntamente libres, genes inactivos
deduplicados, n=(L/dx)+1, nulidad del alias de endpoint. Valores minimos
Conv8->Conv4 deben aparecer de verdad en la poblacion inicial. No obligar
Transformer, ni quedarse en width<=32/n<=25 para todas las etapas.
Mantener explorables los modelos grandes despues de coste; no expandir sin
limite. Tipar las combinaciones inviables y sus motivos sin repararlas a otro
experimento. El rollout semanal no transfiere una receta como si fuera optima.
En cada candidato conservar parametros activos/inactivos y config efectiva.
24 candidatos nuevos por h, una seed, cache por identidad completa; cuatro
folds internos completos antes de ranking. Definir JSON versionado del plan,
reglas de promocion, presupuesto total y por clase antes de los fits.
Los cuatro folds son periodos distintos, no cuatro replicas aleatorias del
mismo ajuste. El maximo de tres intentos se aplica por identidad/fold.
E revisa B/C; B revisa A; A revisa E; C revisa D. No autoaceptacion exclusiva.

## 3. Automatizacion que debes entregar y ejecutar

Construye un unico frontend CLI `python -m tools.i6n_swarm_campaign` con estos
subcomandos (interfaz requerida nueva, NO comandos que existan ya):

```text
plan      --config <private.json> --output <campaign-root>
preflight --root <campaign-root> --json
install   --root <campaign-root>
start     --root <campaign-root>
status    --root <campaign-root> --json
collect   --root <campaign-root>
```

Contrato: plan genera DAG sellado, configs por h/clase/fold, destinos,
identidades y presupuesto; preflight no entrena; install crea unidades
durables, no cambia servicios ajenos; start idempotente; status solo lee
heartbeats/recibos, no importa TF ni lanza trabajos; collect no entrena.
Cada nivel (swarm/horizonte/candidato/fold/publicacion) expone estado/ETA.
Estados incluyen QUEUED, RUNNING con boot-id/lease, COMPLETE verificado,
WAIT_RESOURCE, INTERRUPTED, FAILED, WAIT_DEPENDENCY y causa concreta.
La ETA distingue medida/proyectada/desconocida, no cuenta pruebas como fits.
Un fallo localizado permite trabajos independientes. No terminar el programa
por completar la primera poblacion si el DAG tiene sucesores listos.

Smoke: fixture pequeno para atravesar el swarm REAL y almacenamiento REAL
de ensayo, dos nodos con roles distintos, sin historia ni TEST. Un nodo que
no cabe no simula haber entrenado. Luego piloto TRAIN real por clase, cap de
produccion 1.25*pico medido solo si cabe. Reinicio de ensayo interrumpe solo
un worker propio, nunca reboot del host ni servicios de usuario.

Secuencia de despliegue: A/E contratos -> B/C integracion -> smoke -> piloto
-> busqueda doce horizontes -> anual semanal -> multi-output. D avanza desde
el principio y se integra al cierre antes de afirmar custodia en cadena.
No bloquear fits con una ETL transitoriamente caida: outbox durable y replay.
No cerrar etapa como publicada si faltan sus recibos/lectura exacta.

## 4. Maquinas y coste

Preferir 5090; un trabajo GPU por host por defecto. El secundario necesita
healthcheck fresco tras el reinicio pedido, no asumir que ya ocurrio.
Autorizar 4070 para tareas pequenas solo con reserva de escritorio descrita
en subplan. 5070 Ti opcional UNICAMENTE tras prueba de admision agregada:
RAM de 5090 + RAM de 5070 + wrappers + reserva cabe en host Y slice, sin
recortar caps. No estrenarla esperando un OOM para descubrir el limite.
Ante presion, suspender nuevas asignaciones a ella y drenar; watchdog para
su proceso propio si hay limite duro. No detener la 5090 por capricho.
No reservar RAM para VM MT5 que no usamos aun. No ejecutar trading real.
Pruebas CPU con crispdm-run, 2G/120s para suites pequenas; si una suite necesita
mas, partirla por propiedad o medir nueva clase, no repetir timeouts en loop.
No generar instalaciones grandes simultaneas en los tres hosts.

## 5. Reporte: corto, verificable, sin consumir tokens mirando logs

Cada agente devuelve UNA vez: commit/pin, matriz requisito->test->resultado,
comandos exactos, evidencia, riesgos, siguiente artefacto. Maximo 500 palabras
de relato; los detalles van en archivos. Solo corregir con otra ronda si hay
fallo concreto. Cada responsable deja README de ejecucion y status.

Tu retorno final: `docs/audits/evidence/I6N_SWARM_WEEKLY_20261011/RETURN.md`,
`STATUS.json`, `RESULTS.csv`, `ARTIFACTS.json` y `COMMANDS.md` sin secretos.
Incluye digests de configs/runners, tabla por horizonte con MAE y su naive,
split y poblacion, conteos esperados/completos/fallidos, coste, replicas,
modos de almacenamiento realmente activos, lectura OLAP y snapshot.
STATUS conserva M1-M8 y el hito I6-N, sin sustituirlos por nombres de agentes.
El grafico futuro de Musashi se alimenta de ese JSON, no de porcentajes
inventados. Cada subfase terminada deja un parrafo numerico navegable.

Informes al cerrar entregables o un bloqueo humano irreducible; NUNCA cada
30 segundos desde un agente. Un timer genera status aun cuando acabes tus
tokens. No prometer todas las GPU ocupadas si el scheduler espera memoria;
explicar admision y avanzar trabajos independientes. Al finalizar integracion,
ejecuta install/start y verifica UN latido de fit/publicacion, entrega y sal.
No esperes una respuesta de Musashi para ese start ya autorizado.
