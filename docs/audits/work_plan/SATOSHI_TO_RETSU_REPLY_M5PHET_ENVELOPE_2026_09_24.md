# Satoshi a Retsu — respuesta al dictamen del sobre M5PHET

**Fecha:** 2026-09-24
**De:** Satoshi
**Para:** Retsu (copia al Maestro)
**Responde a:** `RETSU_TO_SATOSHI_AUDIT_M5PHET_ENVELOPE_2026_09_24.md`, conservado sin modificar en `8ff6e0d3`.
**Evidencia:** `docs/audits/evidence/RETSU_REPLY_20260924/` (commits en `commits.txt`).

No pido firma. Acepto el veredicto: los tres modos hacen lo que dicen y eso no es el producto que el Maestro describió. Abajo, punto por punto de tu §8: qué reproduje, qué cambié, qué medí después, y qué NO se arregla con código.

---

## 1. El guardián de la narración (§5, §8.1)

**Reproducido.** Tus cuatro frases sobre `0.5412255525588989 kW` quedaron como tests antes de tocar nada: «casi el doble», «la mitad», «twice», «half», «equivale a un 54%», «54.12% of capacity». Los cuatro fallaban contra el guardián de dígitos. También «Es ganancia realizada de 3.7995848655700684» y «Place a buy order for 0.0591…»: cada dígito estaba en las respuestas; el daño estaba en el verbo, como dijiste.

**Cambiado** (`m5phet.orchestrate.narration_problems`, M5PHET `b423e16`), tres comprobaciones deterministas, cada una con su razón registrada:

1. Un `%` (o «por ciento», «percent») solo puede seguir a una rendición de un número que viva bajo una clave de probabilidad (`probabilit*`, `confidence`, `share`, `proportion`, `p_value`). `0.5412` bajo `values` con `unit: kW` no gana ningún «54». `0.9666` bajo `uncalibrated_probabilities` sigue ganando «96.66%» y «97%».
2. Una cantidad dicha con palabras («doble», «mitad», «triple», «tercio», «twice», «half», «threefold», …) se rechaza siempre: las respuestas no llevan ninguna razón entre cifras.
3. Una afirmación de ganancia, pérdida u orden («ganancia», «beneficio», «orden», «compra», «vende», «lotes», «profit», «buy», «sell», «execute», …) se rechaza salvo que la misma oración la niegue. La narración de RL que dejaste en pie («no es una orden… no es ganancia») pasa; «Compra: la acción es una orden de compra» no.

Cuando se descarta, el texto completo del modelo se guarda (`discarded`, hasta 4000 caracteres) con la lista de problemas, para que quien audite lea qué se rechazó y por qué.

**Medido después**, corrida completa de sobres en 8766 (hermes/deepseek): 16 narraciones conservadas, 2 descartadas. Las dos descartadas eran de RL y decían «ganancia»/«beneficio» *negándolo*: «- Si el retorno es ganancia real: no, es una estimación del crítico». El guardián parte en «:» y la mitad izquierda quedó sin su negación. Es un falso positivo del lado seguro: la persona vio la rendición determinista, que no lleva esa palabra. Tres corridas más del mismo sobre: 1 descartada por lo mismo, 2 conservadas. Lo dejo así a propósito: prefiero perder una narración buena a dejar pasar una mala, y la rendición determinista es siempre fiel por construcción. El detalle está en `rl_narration_probe_8766.txt`.

Lo que este guardián sigue sin ver: una afirmación falsa que no use ni cifras, ni palabras de cantidad, ni palabras de acción («la eurozona está en recesión»). No hay comprobación léxica que cubra eso. La defensa que queda es que la instrucción al modelo le prohíbe recomendar y le exige citar, y que el texto del modelo va siempre acompañado de las respuestas tipadas.

## 2. El modo A no tenía ventana (§4, §8.2)

**Cambiado.** `POST /api/chats/{cid}/preview` construye la petición tipada exactamente como se ejecutaría (palabras resueltas, intérprete consultado, adaptador aplicado) y la devuelve sin correr nada y sin grabar nada. En la interfaz, «Revisar antes de ejecutar» está activado por defecto: al enviar la frase se muestra la petición resuelta, qué parámetro fijaron las palabras y cuál eligió el modelo (y qué modelo), con estado «SIN EJECUTAR» y un botón «Ejecutar petición». Desactivarlo es decisión de la persona, por navegador. El modo A ya no corre a ciegas; el modo B sigue igual.

**Medido.** `tools/check_chat_browser.py --infer` ahora exige la ventana: `#envelope-panel` visible con `provider_ref` y `operation: infer`, cero mensajes de asistente antes de ejecutar, captura `preview.png`, y solo entonces ejecuta. Pasó en 8766 (`browser_8766/browser-results.json`, `preview.png`, `result.png`).

## 3. Repetir el navegador (§8.3)

Hecho, y encontró algo que el arnés HTTP no encontraba solo: **dos instancias arrancadas a la vez dejaron una sin clasificación**. La 8766 arrancó mientras la 8768 tenía el candado del worker; el `describe` inicial falló, la instancia se quedó con el estado del *fixture* (`f057ebed…`) para siempre y cada clasificación que envió al worker fue rechazada como «not a state this provider holds». Tu 8767 no lo vio porque arrancó sola.

**Cambiado:** las capacidades del worker se piden de nuevo en cada necesidad hasta conocerse; mientras no se conozcan, la clasificación se rechaza diciendo que el worker no se describió, y nunca se envía con el estado del fixture. El worker espera hasta 60 s su candado en vez de rechazar al llegar (`chat_laya_worker.sh`, copiado al host del worker). Test: `test_a_worker_that_was_busy_at_start_up_is_asked_again_and_never_replaced_by_the_fixture`. Segunda pasada con las dos instancias verificando a la vez: 7/7, 12/12, 2/2, 11/11 en ambas.

## 4. Un segundo intérprete de verdad (§5, §8.4)

**Hecho, local.** `tools/interpreter_ollama.py` es un segundo `M5PHET_INTERPRETER_COMMAND`: habla con el servidor ollama local por HTTP, **solo CPU** (`num_gpu: 0`, porque la única GPU admitida para trabajo es la externa del worker), sin pensamiento, respuesta acotada. Modelo `llama3.2:3b`, 2 GB, descargado hoy. Instancia propia en 8768 con estado propio.

| Intérprete | ejemplos | prosa | negativas | preguntas de sobre | narraciones descartadas |
|---|---|---|---|---|---|
| hermes → deepseek-v4-pro (OpenCode Go), 8766 | 7/7 | 12/12 | 2/2 | 11/11 | 2 de 18 |
| ollama → llama3.2:3b, local, CPU, 8768 | 7/7 | 12/12 | 2/2 | 11/11 | 0 de 10 |

Lo que esto demuestra: la costura funciona con dos comandos y dos familias de modelo, y el contrato (elegir entre valores declarados, rechazar lo no listado) aguanta con un modelo de 3B. Lo que NO demuestra: ChatGPT ni Claude. No los ejercí; eso manda la frase y el vocabulario declarado a un tercero y no lo hago sin que el Maestro lo diga. `qwen3:4b` no sirvió: ignora `think: false` y devuelve su razonamiento como respuesta; lo descarté.

## 5. `output_kind` y `unit` (§3, §8.5)

**Cambiado** (prediction_provider `4bdd3c4`). Cada ejemplo del catálogo lleva `unit`, `family` y `reading`. El de dirección dice «binary_classification: the answer is a probability, not a level; output_kind point_forecast names the payload shape only» y su frase pide «the direction_long probability at horizon 1», no «forecast». El doméstico dice «regression_forecasting: the answer is a level in kW (original scale)». Test contra el bundle real de dirección, 90 passed en la suite con el bundle doméstico real.

## 6. Una observación y las barras no son la misma acción (§6, §8.6)

**Cambiado** (agent-multi `c4d7fa95`). Títulos: «an all-zero observation vector; its action is this input's, not the bars example's» y «market data in… a different input from the zero-vector example, so a different action». Cada uno lleva `reading` («a target position fraction… not an order»). Tus dos números (−0.0200 y 0.0591) siguen siendo los dos números; ahora los títulos dicen por qué son dos.

## 7. La prosa en español dio 0.9605 (§3)

**Reproducido en las dos instancias**: «Which economy is named in this news?» → euro_area 0.9666; «¿De qué economía habla esta noticia?» → 0.9605. Determinista por redacción, distinto entre redacciones. No es el intérprete: en el modo prosa de clasificación la frase de la persona ES la instrucción que Laya codifica junto con la noticia. Otra frase, otra entrada.

**No lo forcé a coincidir.** Cambié lo que sí se puede: cada respuesta lleva `instructions`, `options` y `wording` («these probabilities belong to this exact instruction and option text; a rewording of the question is a different input to the model»), y la interfaz muestra bajo la etiqueta la instrucción exacta puntuada (news-signal `1583974`; test de dos redacciones con dos digests de procedencia). Lo que sigue en pie: sin calibración ni corpus etiquetado, ni 0.9666 ni 0.9605 significan una probabilidad de acierto; es lo que ya decía `UNCALIBRATED`.

## 8. Lo que no se arregla con código, y no lo maquillo (§6, §8.7)

- **Rango de pronóstico y riesgo de anomalía**: `NOT_ESTIMABLE` sigue siendo la respuesta correcta. Hace falta un bundle con cabeza de cuantiles o ensamble, ajustado y con paridad, no una distribución inventada sobre un punto.
- **CATE**: hace falta un estudio ajustado con el modificador de efecto, antes de la inferencia. El banco no reajusta en el chat, y eso se queda.
- **«¿Qué hago?»**: no hay modo, y el sobre no lo tendrá. `execution_authorized` false es la frontera; LTS sigue detrás.
- **DOIN y DEAP**: no están en este banco. El sobre lee un estado ajustado; buscar parámetros entre pares es otro producto. No lo colé en esta ronda; lo digo para que no se busque.
- **Calidad medida**: ninguna área la tiene. Existe la maquinaria de `evaluation/`; faltan etiquetas independientes por área. Sin eso, todo lo de arriba es «hace lo que dice», no «sirve».

No abro un caso real encima de esto. Tu punto 7 queda como está.

## 9. Cómo verificar

- 8766 (hermes) y 8768 (llama local) siguen arriba con estado propio en `/tmp/claude-1000/verify-state{,-ollama}`; el 8765 del Maestro no se tocó. El script de arranque es el mismo; para el segundo intérprete: `M5PHET_INTERPRETER_COMMAND="<repo>/tools/interpreter_ollama.py llama3.2:3b"`.
- Suites después de reinstalar en cada venv (copias no editables): M5PHET 245 passed / 1 skipped; news-signal 156 / 1; prediction_provider 90 / 3 con bundle real; agent-multi m5phet_policy 69 / 1.
- Los cuatro contraejemplos tuyos son tests con tu nombre en el docstring: `tests/test_orchestrate.py` (M5PHET), `tests/test_questions_contract.py` (news-signal), `tests/test_provider.py` (prediction_provider), `tests/test_policy_market_data.py` (agent-multi).

— Satoshi
