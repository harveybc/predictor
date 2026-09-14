# Plan de integración Flow v3 — agent-multi, DOIN y live (GOV-N7)

Fecha: 2026-09-14. Estado: **PLAN**; ninguna campaña se activa, ningún servicio
vivo se toca. Cada unidad científica necesita procedencia (entregas verificadas)
y terminal, sin consultas remotas por paso. El OLAP sigue siendo un warehouse
remoto configurable detrás de data-gov; ningún experimento recibe credenciales
directas del cubo. Un recibo de data-gov prueba procedencia; no prueba ausencia
de fuga, licencia del proveedor, SNR verdadero ni eficacia del operador.

Evidencia de partida: reconocimiento de solo lectura del 2026-09-13 (matriz de
adopción del packet Flow v3, §5) y el consumidor genérico
`data-gov/tools/governed_exec.py` (perfil por repositorio).

## 1. agent-multi / gym-fx

| tarea | componente | prueba |
|---|---|---|
| A1 Perfil `governed_exec` para `app/main.py` (entrenamiento/evaluación de un job): entradas `data.input_data_file` (+ `dataset_manifest_file`/`input_data_sha256` como identidad esperada), salidas `results_file`, `resolved_config_file`, `config_manifest_file`, métricas `trading.metrics.v1` desde `results.json` (`json_numbers` con claves declaradas) | `agent-multi/tools/governed_run.py` | spec válido; sha de la entrega == `input_data_sha256` declarado (de lo contrario REFUSED antes de abrir datos) |
| A2 Lectura única: la entrega gobernada se lee en `gym-fx/app/env.py.__init__`; `step/reset/learn` no tocan red | test estático (grep de sockets/urllib en `env.py`, `rl_pipeline*.py`) + test de integración con el stack desechable | 0 llamadas remotas en el bucle |
| A3 Frescura de salidas: `app/main.py` sobrescribe `results.json`; el perfil redirige salidas bajo `--out-dir` y rehúsa salida previa | test del perfil | REFUSED sin descarga |
| A4 Dispatchers de decisión (`tools/l1_fleet_launcher.py`, `campaign_supervisor.py`): reja `flow_v3_gate` (manifiesto `governed_campaign.v1` con entregas y destino terminal) antes de cualquier lanzamiento; supervisor de larga vida: registra la campaña una vez y un terminal por job (unidad) | `agent-multi/app/flow_v3_gate.py` (copia de `predictor/tools/flow_v3_gate.py`, test de identidad byte a byte) | dispatch GOVERNING sin manifiesto → exit 4 |
| A5 Terminales para todos los desenlaces (COMPLETED/FAILED/INCONCLUSIVE/REFUSED/QUARANTINED) y outbox durable con disposición (N4) | reutilización de `governed_exec` | caída del destino → pendiente recuperable |
| A6 OLAP: las tablas `promotion_*_olap` / `weekly_result_*_olap` locales (SQLite) pasan a ser artefactos del terminal (hash) y el cubo recibe métricas vía `gov_terminal_metric`; no se duplican como dos experimentos | ETL de terminal → cubo (ya existe en OLAP lake) | conteo de terminales = conteo de jobs |

Frontera: GPU y campañas reales prohibidas en esta orden; A1–A6 se prueban con
un job sintético CPU (`tools/smoke_run.py`) en el stack desechable.

## 2. DOIN (doin-core / doin-node / doin-plugins)

| tarea | componente | prueba |
|---|---|---|
| D1 Contrato: el resultado de una unidad DOIN referencia `terminal_sha256` (y `campaign_sha256`, `unit_id`) del terminal gobernado del evaluador; `TaskCompleted.result` y `POST /api/shared/result` ganan el campo `governed_terminal` (schema `doin_governed_result.v1`) | `doin-core/protocol/messages.py` (campo opcional), fixture E2E | mensaje sin terminal → NON_GOVERNING; con terminal → verificable por sha |
| D2 Los plugins de optimización/inferencia que delegan en predictor (`doin-plugins/predictor/*`) invocan `predictor/tools/governed_run.py` (no `app/main.py`) y devuelven el `terminal_sha256`; sin descargas propias de datasets | `doin-plugins` adaptador | ningún `pd.read_csv` fuera del cache verificado |
| D3 ETL blockchain → cubo conserva la identidad: `gov_terminal` es la fuente de la métrica; la cadena guarda solo la referencia; nunca se cargan la métrica del bloque y la del terminal como dos experimentos | consulta de cobertura por proyecto (`flow_v3_coverage`) + test de deduplicación por `terminal_sha256` | 1 fila por terminal |
| D4 Fixture extremo a extremo (nodo local, tarea sintética, evaluador → terminal en OLAP desechable → resultado DOIN con referencia) antes de tocar publicadores reales | `doin-node/tests/test_governed_result_e2e.py` | reconciliación exacta |

## 3. Live (lts, prediction_provider, heuristic-strategy)

| hallazgo | separación | diseño |
|---|---|---|
| `heuristic-strategy/app/plugins/plugin_api_predictions.py:242` llama a la API de predicción dentro de `next()` | llamada **operativa** ya existente (no de gobernanza) — no se retira de un sistema vivo | evaluación gobernada solo en replay offline con `CsvPredictionSource`/`MappingPredictionPathSource`; la fuente CSV es una entrega gobernada; terminal por replay |
| `prediction_provider/plugins_feeder/data_fetcher.py` descarga Yahoo Finance en tiempo de ejecución sin hash (contradice su AGENTS.md "offline") | operativa | modo replay: `real_feeder` lee una entrega gobernada; el fetch vivo queda etiquetado NON_GOVERNING hasta que exista un lago con contrato para esa fuente |
| `lts` (IBKR/MT5/Alpaca) ingesta barras del venue o de bridges locales; `live_sim_replay` existe | operativa | adopción por `tools/live_sim_replay.py`: entradas gobernadas, terminal por replay; el runner vivo no cambia |

Ninguna de las tres se despliega en esta orden; primero replay offline gobernado
con fixture, revisión de Musashi, luego diseño de adopción sin tocar el servicio.

## 4. Orden y dependencias

1. N3 (reinicio productivo) y micro-run reconciliado — bloqueado en el operador.
2. A1–A3 y D4 (fixtures CPU, stack desechable) — pueden empezar ya.
3. A4–A6, D1–D3 — tras revisión del contrato `doin_governed_result.v1`.
4. Live — solo replay; diseño para revisión.
