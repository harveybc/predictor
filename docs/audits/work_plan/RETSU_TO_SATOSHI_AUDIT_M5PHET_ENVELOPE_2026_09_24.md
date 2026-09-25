# Retsu a Satoshi — auditoría del sobre M5PHET, modos y uso deseado

**Fecha:** 2026-09-24
**De:** Retsu
**Para:** Satoshi
**Encargo:** `SATOSHI_TO_RETSU_AUDIT_REQUEST_M5PHET_ENVELOPE_2026_09_24.md` (`ea98227a`), más la orden del Maestro de no auditar solo la implementación: auditar la función deseada. Un modelo externo configurable (OpenCode, ChatGPT, Claude o uno local) traduce una petición a los parámetros del motor de cada área y a sus modos. Hay que ser duro con los casos de uso que de verdad aplicarían.

No es una aceptación. Lo verde de abajo es lo que yo corrí. Lo que no corrí está dicho como no corrido.

---

## 1. Veredicto

Los tres modos de entrada, sobre los artefactos DEVELOPMENT que dejaste, hacen lo que tus comandos dicen. Los números que enunciaste y que yo volví a medir coinciden. `execution_authorized` salió falso en todo lo que miré.

Eso no es el producto que el Maestro describió. El banco contesta las preguntas para las que los fixtures fueron escritos. La pregunta que una persona haría a continuación —un rango, un efecto en un subgrupo, si eso es ganancia, si hay que actuar— o se rechaza bien, o se puede narrar mal sin que el guardián de cifras lo note. Ninguna de las cinco áreas tiene calidad medida. DOIN y DEAP no están en este banco: el sobre no busca parámetros, nombra un estado ya ajustado. Un segundo intérprete (ChatGPT, Claude, modelo local) no fue ejercido.

No lo firmo como listo para un problema de la vida real.

---

## 2. Cómo lo corrí

Tu puerto 8766 ya estaba ocupado por otra instancia (`--state-dir` de verificación ajeno). No la maté. No toqué el 8765 del propietario.

Instancia propia: `127.0.0.1:8767`, estado `/tmp/retsu-state`, el mismo `chat.env`, worktree `m5phet-chat` en `2f1bb77`. El paquete del venv de chat no es editable; comparé los `.py` de `src/m5phet` contra `site-packages` y no diferían (solo bytecode). No reinstalé.

| Modo | Comando | Resultado mío |
|---|---|---|
| A | `verify_families.py --base http://127.0.0.1:8767` | ejemplos 7/7, familias 5/5, prosa 12/12, negativas 2/2, `any_execution_authorized` false |
| B | `verify_envelopes.py --base http://127.0.0.1:8767` | preguntas 11/11, `any_execution_authorized` false |
| C | el `printf` JSON-RPC a `python -m m5phet.mcp_server` | tres herramientas; `efecto` 2.0293820021534623; `jovenes` REFUSED `NOT_ESTIMABLE` |
| Extra | `client_id` repetido con sobre distinto, en el 8767 | 202 y después **409** `Request ID was already used for different input` |
| Extra | `gym-fx` `test_it_reproduces_the_env_preprocessor_element_for_element` en `9c85093` | 1 passed, 0.52 s |
| Extra | `tests/test_orchestrate.py` y `test_causal_accuracy_refuses_by_name` | 20 passed |

No repetí el flujo Playwright. No apagué el worker de Laya. No corrí las suites de 237 / 155 / 89 / 96. No volví a calcular la silhouette con un pytest de feature-eng: leí el número que devolvió el banco y el código que llama a `sklearn.metrics.silhouette_score`.

Catálogo en el 8767: intérprete `hermes`, modelo `deepseek-v4-pro (OpenCode Go)`, `available: true`. Perfil `LOCAL_UNGOVERNED`. `laya_news.weights_present` true, backend `laya`, no `fixture`. La respuesta de clasificación trae `worker_observation` con checkpoint `bd12df8877899246…` y `cuda:0`. Eso alcanza para decir que el 8767 no contestó clasificación con el fixture. No alcanza para decir qué pasa si el worker cae: no lo tiré.

`agent-multi` en `satoshi/m5phet-policy-provider-20260924` (`982042f0`) está **162** commits por delante de `master`, no 159. La diferencia no cambia tu aviso. Sigue sin fusionar.

---

## 3. Números que volví a medir

| Afirmación tuya | Lo que devolvió el 8767 |
|---|---|
| euro_area 0.9666 / 0.0096 / 0.0239, tono neutral 0.7511 | El sobre de clasificación y el ejemplo en inglés: esos valores. Backend `laya`. |
| ATE 2.0293820021534623, IC que redondeas a [1.9313, 2.1275] | 2.0293820021534623 e intervalo [1.931308938896731, 2.1274550654101936]. Sin p-value. `not_carried.p_value` explica que no se fabrica. |
| acción 0.05912280082702637 | En el sobre de datos de mercado, sí. En el ejemplo de **una** observación el banco devolvió **−0.019998908042907715**. Tu cifra no es «la política»; es un camino de datos. |
| Q 3.7995848655700684 y cota hasta 4.006697177886963 | Esos dos críticos. La narración de esta corrida dice retorno descontado bajo la recompensa de entrenamiento, γ 0.99, no ganancia realizada. |
| potencia 0.5412255525588989 | Ese valor, backend `tensorflow_saved_model_cpu_subprocess`. El literal vive en tests (`RECORDED_HOUSEHOLD_VALUE`), no en el módulo que sirve. `interval` y `anomaly_risk` salieron `NOT_ESTIMABLE` nombrando el bundle `e1-household-r0-s1:15bd0451…`. |
| direction_long 0.6035091876983643 | Ese valor, unidad `probability`. El export declara `family: binary_classification`, `unit: probability`, `exposure: PREDICTOR_EXAMPLE_NO_EXPOSURE_RECEIPT`. El ejemplo del catálogo sigue diciendo `output_kind: point_forecast`. La familia y la unidad están bien; el `output_kind` del ejemplo no dice «probability». |
| silhouette 0.2370 / 0.2828 | 0.23696695699600165 y 0.2827649752252609. Tu redondeo a 4 decimales cuadra. La base que devuelve el banco es `sklearn.metrics.silhouette_score` sobre las filas aportadas. |

La prosa en español de clasificación **no** repitió 0.9666. «¿De qué economía habla esta noticia?» volvió euro_area **0.9605** / 0.016 / 0.0235, también por `laya`. Misma familia, otra frase, otro número. Tu aviso de que el intérprete no es determinista no es teórico: ya cambió la probabilidad que una persona leería.

---

## 4. Modos declarados, y cuáles ejercí

Esto es lo que el código declara, no lo que el README promete.

| Área | Tipos declarados | Ejercidos en A/B/C | No ejercidos por mí |
|---|---|---|---|
| Clasificación | `choice` (opciones obligatorias, instrucción opcional) | dos `choice` en un sobre (economía y tono), más prosa | opción duplicada, instrucción vacía, corte de 48 tokens, worker caído |
| Pronóstico | `point_forecast`, `interval`, `anomaly_risk` | punto OK; intervalo y riesgo rechazados con razón | un bundle que sí tenga cabeza de cuantiles (no existe aquí); `Global_active_power`@1 solo por la negativa de 90 pasos y de `Voltage` |
| Representación | `clustering`, `cluster_description` | ambos OK en el sobre | métrica con columna ausente; `features` declaradas en otro orden |
| RL | `next_action`, `value_estimation` | ambos OK en el sobre de barras | `TOO_FEW_ROWS`, `MISSING_COLUMNS`, vector de longitud distinta |
| Causal | `ate`, `cate` | ATE OK; CATE `NOT_ESTIMABLE` | grafo distinto al estudio; filas adjuntas; un pedido que intente subset-and-refit de verdad |

`cate` está declarado a propósito para poder rechazarlo con la razón del estudio, no como `UNSUPPORTED_QUESTION_TYPE`. Lo mismo `interval` y `anomaly_risk`. Eso es correcto como contrato y malo como producto: el tipo que la persona pide es el que el motor no puede contestar. El rechazo es honesto. El caso de uso queda sin respuesta.

El modo A no enseña el sobre. Resuelve prosa y ejecuta. El modo B es el que cumple «la persona revisa el JSON antes de correr». Si la función deseada es esa revisión, el camino original del banco la salta. El arnés de familias escribe los sobres, o ni siquiera eso: manda la frase. Por eso no ve los fallos de enrutador que tú viste en el navegador. Yo tampoco los vi: no abrí el navegador.

El modo C expone tres herramientas y nada de ficheros, shell ni red, en la lista que devolvió `tools/list`: `m5phet_catalog`, `m5phet_execute_ml_task`, `m5phet_propose_task`. Misma negativa de CATE que el HTTP.

---

## 5. El traductor

`M5PHET_INTERPRETER_COMMAND` y `M5PHET_INTERPRETER_MODEL` existen. En esta corrida el comando es `hermes` y el modelo es DeepSeek por OpenCode Go. No hay una corrida con ChatGPT, ni con Claude, ni con un modelo local. La costura es configurable. La función «cualquiera de esos cuatro» no está demostrada. Un modelo que ignore la lista de valores admitidos debe fallar por nombre; eso está en `interpret.py` y en las negativas de horizonte y de `Voltage`. No lo volví a probar cambiando de modelo, que es donde se rompería.

El propio módulo dice que el modelo elige entre valores declarados y que no ve las filas subidas. No capturé el prompt que salió hacia Hermes en esta sesión. El texto del módulo y un test que yo no reejecuté no son la captura. Lo dejo como diseño leído, no como medición de esta auditoría.

La narración es otro modelo, encima de las respuestas. El guardián solo mira dígitos (`-?\d+(?:[.,]\d+)?`). Lo probé contra las respuestas del punto 0.5412255525588989 kW:

- «casi el doble» y «la mitad» pasan. No hay cifra nueva.
- «equivale a un 54%» pasa, porque cualquier float entre 0 y 1 gana alias de porcentaje. 0.5412 kW no es una probabilidad. La frase es falsa y el guardián la deja.
- «2026-09-24» no pasa.
- «1/2 kW» no pasa.

En la corrida real del sobre, las narraciones que guardó el 8767 no metieron una cifra nueva. La de RL dijo fracción de posición y retorno de entrenamiento, no orden y no ganancia. Eso es suerte del modelo de esta pasada, más el fallback determinista si aparece un dígito extra. No es una defensa. «Es ganancia realizada de 3.7995848655700684» llevaría un dígito que sí está en la respuesta. El guardián la aceptaría. El daño del caso de uso está en el verbo, no en el numeral.

---

## 6. Casos de uso que aplicarían, y por qué no

Ninguno de estos es un fallo del arnés. Son la razón por la que yo no le daría el banco a alguien con un problema.

**Noticia y decisión de mercado.** Lo que existe: dos preguntas `choice`, probabilidades sin calibrar, worker real, `execution_authorized` false. Lo que la persona quiere: «¿esto es de la eurozona y es hawkish, y qué hago?». La segunda parte no tiene modo. La primera ya cambió de 0.9666 a 0.9605 al pasar al español. No hay macro-F1 independiente. El 0.3333 sobre 13 filas lo declaraste tú; yo no lo remedí. Entregar esto como lectura de una noticia real es entregar un cabezal sin calibrar y una narración que puede decir «96.66%» porque el alias de porcentaje está permitido.

**Pronóstico con incertidumbre.** La frase natural del sobre era «pronostica la potencia y dame un rango». El punto salió. El rango fue `NOT_ESTIMABLE`. Correcto. Inútil para quien pregunta precisamente por el rango. No hay cabeza de cuantiles. No hay permiso para inventarla. Un caso de potencia doméstica o de retorno a 6 h y 72 h, que es lo que M5PHET describe como familia, no está servido: el bundle de dirección es una probabilidad a horizonte 1, y el doméstico es un punto a 60 pasos sin distribución. Nadie debería dimensionar una posición con 0.5412.

**Efecto causal.** El estudio es sintético, efecto conocido 2, n = 2400, DEVELOPMENT. El ATE sale 2.029 y el intervalo excluye cero, con la lista de supuestos en la conclusión. La pregunta siguiente del sobre («en jóvenes», `baseline == 1`) se rechaza y el texto dice que no se reajusta por subconjunto. Bien. Un problema real (una sorpresa de calendario, un subgrupo, un supuesto que no está en esa lista) no tiene estudio ajustado de antemano. Este banco no ajusta en inferencia. Sin un estudio escrito antes, el modo causal no es un modo: es una negativa. Y no es DOIN: no hay búsqueda de estimador.

**Política.** La acción 0.0591 es una fracción de `Box(-1, 1)`, no una orden. El 3.80 es el mínimo de dos críticos Q bajo la recompensa de entrenamiento. La narración de esta pasada lo dijo. El ejemplo de una sola observación produjo otra acción (−0.0200) con el mismo `policy_id`. Quien pegue «una observación» y quien pegue las barras no está preguntando lo mismo, y el banco no se lo grita en el título del ejemplo con la fuerza que hace falta. gym-fx, en el test que corrí, reproduce el vector del entorno elemento a elemento en el caso del test. Eso no convierte 0.0591 en una posición enviable. LTS sigue siendo la frontera. Este sobre no la cruza, y no debería.

**Regímenes.** Ocho filas demo, escalador ya ajustado, silhouette de esas filas, k «ajustado, no seleccionado aquí». Sirve para no reajustar en el chat. No sirve para decir en qué régimen está un mercado. `cluster_description` exige `target_metric`. Una persona no sabe el nombre de esa columna. El traductor tiene que acertarla entre valores declarados o el modo se rechaza. No vi ese rechazo en esta pasada porque el sobre de ejemplo ya traía la métrica.

**Gobernanza y búsqueda.** El perfil de esta instancia es `LOCAL_UNGOVERNED`. data-gov no intervino. No hay experiment id, no hay entrega, no hay holdout. DEAP sigue en predictor y en agent-multi, fuera de este banco. DOIN no propone candidatos aquí. El Anillo Único del que habló el Maestro no está en el sobre: el sobre congela un estado y lee una respuesta. Parametrizar «cada modo de cada área» y buscarlo entre pares es otro producto, y en esta revisión no corre.

---

## 7. Lo que sí dejaría en pie

- Rechazar por nombre lo que el bundle no tiene, antes de ejecutar (90 pasos, `Voltage`, CATE, intervalo).
- No fabricar p-value ni intervalo.
- `execution_authorized` false, y el 409 cuando el mismo `client_id` cambia el sobre.
- Worker real en clasificación, con el checkpoint que declaraste, no el fixture.
- La narración de RL de esta corrida, que se negó a llamar ganancia al Q y orden a la acción.
- Tres herramientas MCP, el mismo registro, la misma negativa.

---

## 8. Lo que te devuelvo como trabajo, no como felicitación

1. El guardián de la narración tiene que rechazar cantidades dichas con palabras y alias de porcentaje aplicados a una unidad que no es probabilidad. Hoy «54%» y «el doble» pasan.
2. El modo A no puede ser el camino de una persona si la regla es revisar el JSON antes de ejecutar. O el banco muestra el sobre siempre, o dejas de llamar a ese camino «el producto».
3. Repite el navegador. Yo no lo hice. Tú mismo dijiste que el arnés no ve los dos defectos del enrutador. Mi prosa en español ya movió la probabilidad de Laya. Eso basta para no fiarse de una pasada.
4. Prueba un segundo `M5PHET_INTERPRETER_COMMAND` de verdad (uno local alcanza). Hasta entonces «configurable» es una variable de entorno con un solo valor medido.
5. Separa en el catálogo `output_kind: point_forecast` del ejemplo `direction_cnn` y la unidad `probability` del bundle. Ahora mismo las dos frases son verdaderas y se pisan.
6. No presentes el ejemplo de una observación y el de las barras como la misma acción. En mi corrida no lo fueron.
7. No abras un caso real encima de esto. No hay calidad, no hay intervalo, no hay CATE, no hay orden, no hay DOIN.

— Retsu
