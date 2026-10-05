# Orden vigente para Satoshi: fases 2 y 3 automatizadas de seleccion

Esta es la unica orden nueva de seleccion. Lee primero:

1. `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md`;
2. `docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md`;
3. `docs/tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json`;
4. `docs/tres_temas_entrevista/program_v3/CURRENT_EXECUTION.md`.

Objetivo: cerrar fase 2 y fase 3 sin entrenamiento, GPU ni intervencion por
celda. Los resultados causales de fase 1 son insumos; no son seleccion final.

## A. Disciplina de agentes y tokens

- Un agente de ingenieria implementa y prueba el driver completo.
- Un agente de datos verifica contratos, denominadores y OLAP.
- Un agente de integracion despliega el mismo commit/entorno en los tres hosts.
- Hermes puede ejecutar lecturas, suites y comprobaciones mecanicas.
- Ningun agente vigila loops, lanza pares manualmente ni emite reportes sin
  cambio terminal. El software escribe `STATUS.json` cada minuto.
- Reporta al Maestro solo al cerrar una compuerta, ante fallo terminal que el
  driver no pueda aislar, o al terminar. No repitas comandos fallidos sin
  diagnostico nuevo.

## B. Congelar requisitos antes del codigo

En un worktree nuevo desde `origin/satoshi/canonical-exec-20261003`:

1. congela un manifiesto con 366 EURUSD, 83 ETH, sus targets, TRAIN, folds y
   los digests de fase 1;
2. diseña pruebas rojas para todos los puntos de aceptacion del subplan;
3. conserva PRE y POST; el PRE debe fallar por ausencia de la capacidad, no por
   imports o fixtures rotos;
4. no edites los artefactos de fase 1 ni los `STATUS` historicos.

## C. Implementar un controlador, no scripts por celda

Crear en `tools/`:

- `feature_pairwise_campaign.py`: plan, shards, terminales y reanudacion;
- `feature_pairwise_worker.py`: calculo acotado por bloque;
- `feature_filter_selection.py`: clustering, mRMR, JMI y controles;
- `feature_selection_phase23_status.py`: estado y ETA desde evidencia;
- pruebas focalizadas correspondientes.

El driver ofrece al menos `plan`, `run-worker`, `follow`, `close-phase2`,
`run-phase3`, `close-phase3` y `status`. Todos reciben rutas/identidades por
argumento; ningun host, IP, credencial o ruta privada entra al repositorio.

Los terminales son inmutables y se nombran por digest de poblacion, shard,
metodo y parametros. El driver adopta terminales validos, pone en cuarentena los
corruptos y nunca recalcula una unidad completa.

## D. Esquema del warehouse

Extender por migracion aditiva el warehouse desplegado. No usar el ETL legacy
incompatible de `olap/`. Las filas deben incluir todas las identidades del
subplan y una clave unica que impida duplicados. Implementar submit, lectura por
run y reconciliacion de conteos/digests. Probar en DuckDB desechable antes del
servicio vivo.

## E. Ejecucion automatica multi-host

1. Ejecuta health check y mide memoria disponible; no uses GPU.
2. Instala el mismo wheel/commit en omega, gamma y dragon.
3. Genera todos los shards antes de lanzar.
4. Asigna omega al tercio con menor coste estimado y gamma/dragon al resto.
5. Usa CPU acotada y una sola lectura de columnas compartida por bloque; evita
   cargar la matriz completa repetidamente.
6. Lanza unidades `systemd --user` reiniciables o el supervisor durable ya
   verificado. Cada host reclama shards exclusivos desde el plan.
7. El follower carga terminales al warehouse, verifica lectura de vuelta y
   encadena automaticamente fase 3 cuando fase 2 cierre.
8. Una unidad fallida no detiene las demas; el estado muestra su razon exacta.

No lances AE, DAE, CVAE, PS3-R, arquitectura modular, DOIN, RL, NEAT, Traffic,
clasificacion ni M5PHET durante este encargo. Esta es una concentracion temporal
del esfuerzo, no una eliminacion de esos hitos.

## F. Puertas de cierre

Fase 2 requiere exactamente 66,795 pares EURUSD y 3,403 ETH con disposicion por
fold/metrica/lag declarados, alias y clusters, cero omisiones silenciosas,
recibos y lectura de vuelta. Genera `PHASE_2_COMPLETE.json`.

Fase 3 ejecuta clustering-Spearman, mRMR y JMI, mas controles, para todos los
targets/horizontes y K declarados. Conserva ranking completo y genera
`PHASE_3_FILTER_COMPLETE.json`. No elijas ganador predictivo ni uses test.

Publica snapshot `.duckdb.zst` con SHA-256 en un release recuperable y actualiza
plan, cola, checklist, estado metodologico y grafico de progreso desde evidencia.

## G. Retorno unico

`RETURN.md` debe contener:

- commit y rama de cada repo tocado;
- conteos esperados/completos/fallidos por activo, metodo y host;
- tiempos, memoria pico y coste por millon de celdas;
- URL/query del warehouse y URL/SHA del snapshot;
- clusters/alias principales y trayectorias K producidas, sin declarar ganador;
- pruebas PRE/POST, restart parity, future perturbation y readback;
- siguiente objeto exacto: diseno de fase 4, sin comenzarlo.

No consumas tokens narrando progreso sin una transicion de estado.
