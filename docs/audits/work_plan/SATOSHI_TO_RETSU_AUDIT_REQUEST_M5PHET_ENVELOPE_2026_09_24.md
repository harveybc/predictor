# Solicitud de auditoría a Retsu — M5PHET: todos los modos de todas las áreas

**Fecha:** 2026-09-24
**De:** Satoshi (implementación e integración; Musashi no disponible por un problema de pago del propietario)
**Para:** Retsu
**Encargo del propietario:** «solicitar una auditoría a Retsu para que use todos los modos de todas las áreas».

Todo lo de abajo es medición propia. Nada de esto es una aceptación. Cada afirmación va
enunciada de forma que pueda refutarse, y al final de cada área digo qué atacaría yo.

---

## 1. Qué auditar y dónde

| repositorio | rama | revisión | árbol |
|---|---|---|---|
| M5PHET | `musashi/chat-workbench-20260924` | `2f1bb7749` | limpio |
| news-signal | `master` | `30f3a8932` | limpio |
| prediction_provider | `musashi/m5phet-forecast-20260924` | `9f2108ba8` | limpio |
| causal-inference | `feat/m5phet-causal-provider` | `42e83d57c` | limpio (el checkout principal tiene ficheros exploratorios del propietario sin versionar; **no tocar**) |
| agent-multi | `satoshi/m5phet-policy-provider-20260924` | `982042f03` | 8 ficheros modificados y ~30 sin versionar **del propietario**, ajenos a esta rama; no forman parte de nada auditado |
| feature-eng | `codex/m5phet-hierarchical-regimes-20260924` | `6296447b8` | limpio |
| gym-fx | `satoshi/m5phet-observation-builder-20260924` | `9c8509313` | limpio |
| predictor (evidencia y retornos) | `satoshi/rp132-rp134-20260923` | `a8941ee85` | limpio |

Retornos: `SATOSHI_ENVELOPE_RETURN_2026_09_24.md`, `SATOSHI_CL20_CL24_RETURN_2026_09_24.md`,
`SATOSHI_CL14_CL19_RETURN_2026_09_24.md`. Evidencia: `docs/audits/evidence/ENVELOPE_20260924/`
(sobres, familias, capturas de navegador, commits por repo), `CL20_CL24_20260924/`,
`CL14_CL16_20260924/`, `CL09_CL10_20260924/`, `CL01_CL07_20260924/`.

Estados ajustados (fuera de los repos, todos **DEVELOPMENT**):
`~/.local/state/m5phet/{forecast-bundles-20260924,policy-eth4h-dev-20260924,examples/regimes}`,
`~/.local/share/causal-inference-m5phet/studies`, checkpoint Laya sellado en el worker
(`bd12df8877899246…`). Configuración del operador: `~/.config/m5phet/chat.env`.

## 2. Los modos, y el comando exacto de cada uno

Hay **tres modos** de entrada y **cinco áreas**. Un modo que no se ejercite queda sin auditar.

### Modo A — ejemplo/pregunta única (el camino original del banco)
Proveedor + lector + contrato elegidos en la conversación; una pregunta; `chat_slots()`
resuelve la prosa contra el vocabulario declarado.

### Modo B — sobre (`state` + `questions` tipadas → `answers` tipadas)
El contrato de `m5phet.questions`. Varias preguntas de tipos distintos en una petición; cada
una respondida o **rechazada por nombre con su razón**; ningún número donde el motor no
puede. Con el intérprete configurado, la frase se traduce a sobre **que la persona revisa
antes de ejecutar** (botón «Pregunta → JSON»).

### Modo C — MCP por stdio
Las mismas tres operaciones como herramientas; mismo registro, mismas negativas.

**Arranque** (el servicio del propietario ya corre en 8765; para auditar use una instancia aparte):

```bash
set -a; . ~/.config/m5phet/chat.env; set +a
PORT=8766 /home/harveybc/Documents/GitHub/.worktrees/m5phet-chat/tools/start_chat.sh --state-dir /tmp/retsu-state
```

**Modo A y refusals** (7 ejemplos, 12 frases en inglés y español, 2 negativas):
```bash
python3 /home/harveybc/Documents/GitHub/.worktrees/m5phet-chat/tools/verify_families.py --base http://127.0.0.1:8766
```
Resultado mío: `examples 7/7, families 5/5, prose 12/12, refusals 2/2, any_execution_authorized false`.

**Modo B** (un sobre por área, 11 preguntas, en la forma del propietario):
```bash
python3 /home/harveybc/Documents/GitHub/.worktrees/m5phet-chat/tools/verify_envelopes.py --base http://127.0.0.1:8766
```
Resultado mío: `questions_as_expected 11/11, any_execution_authorized false`.

**Modo B en navegador** (frase → sobre revisado → ejecución; Playwright, sin GPU local):
```bash
~/.local/share/m5phet/chat-venv/bin/python /tmp/claude-1000/shot_envelope.py   # o el suyo
```
Capturas mías en la evidencia: `envelope-proposal.png`, `envelope-answers.png`.

**Modo C** (stdio, JSON-RPC 2.0):
```bash
set -a; . ~/.config/m5phet/chat.env; set +a
printf '%s\n%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"m5phet_execute_ml_task","arguments":{"area":"causal","state":{"causal_graph":{"treatment":"treatment","outcome":"outcome","confounders":["baseline"]}},"questions":{"efecto":{"type":"ate"},"jovenes":{"type":"cate","condition":"baseline == 1"}}}}}' \
  | ~/.local/share/m5phet/chat-venv/bin/python -m m5phet.mcp_server
```
Resultado mío: `efecto` OK con `effect_size 2.0293820021534623`; `jovenes` REFUSED `NOT_ESTIMABLE`.

**Suites por repositorio** (todas CPU; toda orden bajo `crispdm-run -m … -t … -n … --`):
M5PHET 237 (`chatenv`), news-signal 155, prediction_provider 89 nativas + 14 de contrato,
causal 19 de contrato (+ fallos preexistentes por `econml` ausente en el venv de inferencia,
idénticos al baseline), agent-multi 69 con entorno del operador, feature-eng 96,
gym-fx 26 nuevas (629 en total, 1 fallo preexistente y ambiental).

## 3. Afirmaciones por área, enunciadas para poder refutarse

### Clasificación (`laya_news`, news-signal)
- Un sobre con dos preguntas `choice` va al SDK **en una sola llamada**; cada pregunta pasa por `build_question` y se rechaza **por nombre** (opción duplicada, instrucción vacía, opción que el corte de 48 tokens truncaría).
- Los números son los del SDK sin tocar: `euro_area 0.9666 / 0.0096 / 0.0239`; tono `neutral 0.7511`.
- El sobre llega al **worker privado** (`action: "task"`) y vuelve **ligado al digesto del sobre**; una respuesta no ligada se rechaza (`test_a_worker_answer_not_bound_to_the_envelope_is_refused`).
- *Qué atacaría:* que la instancia 8766 y el servicio 8765 lleguen de verdad al worker y no al fixture (`weights_present` en el catálogo; probabilidades que no sean 0/1). Que con el worker caído la negativa sea explícita y no un fixture silencioso.

### Pronóstico (`predictor_forecast`, prediction_provider)
- Dos bundles: potencia doméstica (`Global_active_power` @60 → `0.5412255525588989`, exacto contra el replay nativo) y el campeón real de `predictor` `direction_cnn` (`direction_long` @1 → `0.6035091876983643`, paridad nativa error 0.0).
- `interval` y `anomaly_risk` están **declarados y rechazados** `NOT_ESTIMABLE` nombrando el bundle: sin distribución predictiva, sin cabeza de cuantiles.
- `chat_combinations()` declara las parejas ajustadas; `Global_active_power`@1 se rechaza **antes de ejecutar** como no ajustada.
- *Qué atacaría:* que el valor 0.54122… salga de la ejecución y no de un fichero de paridad. Que el bundle `direction` declare `binary_classification` y `probability` y **no** herede el recibo de exposición del doméstico (`PREDICTOR_EXAMPLE_NO_EXPOSURE_RECEIPT`). Que no exista camino a un intervalo inventado.

### Causal (`causal_inference`)
- Inferencia sobre un estudio ajustado **explícitamente antes** (`prepare-demo`, sintético, efecto conocido 2); `ate` = `2.0293820021534623`, IC `[1.9313, 2.1275]`, supuestos en cada respuesta OK; **sin p-value fabricado** (`not_carried.p_value` explica).
- Un grafo distinto al del estudio se rechaza `NOT_ESTIMABLE` nombrando el campo; datos adjuntos se rechazan («no ajusta en inferencia»); `cate` rechazada con la explicación exacta del modificador de efecto.
- *Qué atacaría:* que `conclusion` derive de signo e intervalo y no de narración. Que ningún camino haga *subset-and-refit*. Que la maquinaria de evaluación (`evaluation/`) **rechace** `causal_accuracy` por nombre.

### RL (`trading_policy`, agent-multi + gym-fx)
- `next_action` desde el checkpoint SAC retenido con **datos de mercado reales**: 300 barras ETH 4h → 256 consumidas → observación 2724 → acción `0.05912280082702637`; distribución real del actor (media, log-std) etiquetada como *spread del actor, no probabilidad de acierto*; `confidence: null` con la razón.
- `value_estimation` **desde los críticos Q reales**: `expected_return 3.7995848655700684`, `uncertainty_bounds [3.7996, 4.0067]`, γ 0.99, etiquetado «retorno descontado bajo la recompensa de entrenamiento, no beneficio realizado».
- Una longitud **no es un contrato**: el camino de datos de mercado queda cerrado hasta que el bundle lleve `observation_contract.json` (derivado de la config de campaña; sha `ce246e2b…`). 10 filas → `TOO_FEW_ROWS`, nunca relleno; 18 columnas → `MISSING_COLUMNS` nombrando 79 de 83.
- `execution_authorized: False` en toda respuesta; el runtime rechaza lo contrario.
- *Qué atacaría:* que la observación construida por gym-fx sea **idéntica elemento a elemento** a la del entorno en la misma barra (hay test; reprodúzcalo). Que `expected_return` no pueda confundirse con rentabilidad en ninguna narración.

### No supervisado (`feature-eng-hierarchical-regimes`)
- Asignación bajo referencia **ya ajustada** (escalador/Ward train-only; sin reajuste en chat). `clustering`: distribución por nivel sobre las filas aportadas (suma 1), `optimal_k` = niveles ajustados `[2, 4]` marcado «ajustado, no seleccionado aquí», **silhouette real** en el espacio escalado de la referencia (nivel 2 = 0.2370, nivel 4 = 0.2828 sobre las 8 filas demo), omitido con razón cuando sklearn no lo define.
- `cluster_description`: `centroid_features` en **unidades originales**, nunca coordenadas escaladas. Métrica con columna ausente → rechazo por nombre. `features` ausente = las ajustadas; declaradas y distintas → rechazo (el escalador es posicional).
- *Qué atacaría:* que la silhouette coincida con `sklearn.metrics.silhouette_score` sobre las mismas filas escaladas y etiquetas (hay test cruzado). Que la referencia se niegue a cargar con otro conjunto de versiones de dependencias (regla de integridad del módulo; no la debilité).

### Transversal: enrutador, narración, MCP
- El intérprete (DeepSeek `deepseek-v4-pro` vía OpenCode Go, por `hermes --ignore-user-config`) ve **la forma** de los datos (columnas, tipos, filas) y **nunca una fila** (test: «120.5» no aparece en lo que se le envía).
- Propone; el contrato dispone: área no servida, tipo no declarado, columna inexistente, **valor no declarado** (un nombre de columna no es un objetivo ajustado), **pareja no ajustada** → rechazo por nombre antes de ejecutar.
- Narración verificada: cada número de la frase debe estar en las respuestas (en cualquier renderizado ordinario, incluido porcentaje); si introduce una cifra, se descarta por redacción determinista (`narration.source`).
- MCP: sin herramientas de ficheros, shell ni red; tres herramientas, mismo registro.
- *Qué atacaría:* la verificación numérica de la narración — busque una cifra que se cuele (fracciones, «casi el doble», fechas). El sobre editado a mano con `client_id` repetido debe dar 409, no reproducir el viejo.

## 4. Límites que declaro yo, para que no los tenga que descubrir usted

1. **Ninguna de las cinco áreas tiene calidad medida.** Todos los estados son DEVELOPMENT. El único número de calidad existente es macro-F1 0,3333 sobre 13 filas escritas por mí. La maquinaria (`evaluation/`, 38 pruebas) existe; las etiquetas independientes no.
2. Los dos «defectos de enrutador» del retorno los encontré **yo, en navegador**; el arnés no los ve porque escribe los sobres a mano. Un intérprete no determinista puede fallar de formas que una pasada no muestre: repita el flujo de navegador varias veces.
3. `agent-multi` no está fusionado a `master` (159 commits por delante; decisión del propietario). news-signal y causal-inference sí.
4. Los venvs guardan copias **no editables**: tres agentes distintos midieron `site-packages` en vez de `src/` hasta reinstalar. Si reproduce sobre el código, reinstale antes de creerse un verde.
5. Fui implementador **e** integrador **y** verificador en esta ronda. Eso es exactamente el riesgo que su auditoría cubre.

Sin GPU nueva, sin entrenamiento, sin bróker, sin capital. La clasificación llega a la 5090
externa por el lease existente del worker.

— Satoshi
