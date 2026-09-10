# Dictamen Retsu — fusión A + B

**Para:** Harvey (Maestro) y Musashi  
**De:** Retsu  
**Fecha:** 2026-09-02  
**Modo:** adversario de entrenamiento, no editor amable.  
**No reescribe** `docs/propuesta_doctoral_doin_borrador.{html,pdf}`. Eso es el siguiente paso, si aceptan este recorte.

Este es el documento principal. Los anexos están en [`fusion_a_b/`](fusion_a_b/README.md).

---

## 0. Identidades (hechos)

| Objeto | Qué es | Digest |
|---|---|---|
| **A** | *Validación e incentivos entre pares…*, 4 pp., `predictor@7ae8643` | HTML `f71b34c6…dee79b56` · PDF `8b567994…d4d6b03` |
| **B** | *Metaoptimización confiable…*, 7 pp., working tree local | HTML `b30e911b1c185867f71fd98d6802f878d034e7cec8311cbd8fa4ce03b2322011` · PDF `151418b96407842e98d2bd8a019c243532df6589e56f59d02b319262f3f99207` |
| Sistema | `doin-core@a90bca2` · `doin-node@8bfc64f` · `doin-domains@d01708ae` | PoO = umbral de incrementos; fee = EIP-1559; `Task` = verificación o inferencia; **no hay subasta de adquisición** |

Leí A y B íntegras. Leí el dictamen de fusión de Satoshi que Harvey pegó. Harvey aclaró después el **flujo real** que quería incentivar: no un eBay de modelos. Corrijo el veredicto de la versión anterior de este mismo archivo.

---

## 1. Veredicto

**No fusionar. No KEEP B como columna. No KEEP A tal como está escrita.**

El objeto que Harvey describe —*enchufar un nodo, elegir o autoseleccionar un dominio, la GPU trabaja, el operador no fija precio ni hace ofertas*— **no está en A, no está en B, no está en el híbrido de Satoshi, y no está en el modelo de subasta que yo mismo escribí en el anexo 04 de la pasada anterior.** Está en el **código** (`ProofOfOptimization`, `Optimae`, umbral de dificultad, stake, cola por dominio) y en la analogía de **minero de Ethereum pre-Merge**: emisión de protocolo por trabajo, no tasación bilateral.

Tres desajustes, todos graves:

| Texto | Qué supone | Qué hace el operador de DOIN |
|---|---|---|
| **A** | Consumidor deposita \(B\); trabajadores presentan **ofertas selladas**; Hayek: el precio agrega conocimiento | Nadie publica precio. Nadie puja. El nodo entra y trabaja. |
| **B** | Un metaoptimizador recomienda configuraciones a **tareas nuevas** | El nodo no es un AutoML de transferencia; es un aportante de cómputo a un dominio |
| **Satoshi / mi H1** | Asignación por valor/costo; “mercado de una ronda” | Sigue siendo un mercado. Harvey acaba de decir que **no** quiere MercadoLibre de modelos |

Hayek, en A, es un adorno que un jurado usa para ahorcarlos: *el uso del conocimiento en la sociedad* requiere **precios**. Si el operador no fija precio, Hayek **no aplica**. Decirlo en el PDF es diletante. El análogo honesto es **emisión** (block reward / inflación dirigida), como Bitcoin/ETH-PoW y, en IA, como Bittensor: no compras un modelo en un catálogo; conectas hardware y el protocolo te infla stake si el trabajo sobrevive la regla.

**Lo que sí es tesis, y era la pregunta original:**

> ¿Cómo recompensar trabajo de optimización en una red donde los nodos **no tasán**, se **autoseleccionan** de dominio, y pueden entregar incrementos inútiles o falsos, de modo que aportar GPU sea racional y fabricar basura no lo sea?

Eso tiene base económica, y **no** es teoría de subastas. Es riesgo moral sobre un *output* verificado (Holmström), concursos / premios (quién se lleva la emisión cuando hay un nuevo óptimo), autoselección de dominio (el nodo elige el “yacimiento”), y el trade-off utilidad–seguridad de PoUW (Cao, Ofelimos). El PoO que ya corre ajusta un **umbral** como la dificultad de minería. Eso es el instrumento. A lo disfrazó de subasta inversa. B lo abandonó.

**Encaje en el Doctorado en IA:** más flaco que B y más honesto que el híbrido. Un jurado de IA dirá Ingeniería/sistemas. La defensa única: el objeto es **qué cuenta como trabajo útil de aprendizaje** (el incremento verificado, no el hash) y **cómo no pagar el inútil**. Si no quieren esa pelea, KEEP B y el incentivo de mineros queda como paper aparte — no como fusión. Mentir que B “es” esa pregunta es peor.

Detalle operativo: [08_FLUJO_MINERO_NO_MERCADO.md](fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md).

**Repliegue:** KEEP B recortada (AutoML, corrupción inyectada) **solo** si Harvey acepta, por escrito, que **cerró** la pregunta de incentivos. Si no la cierra, no escriban B. Escriban la pregunta de arriba.

Los §§2–10 de abajo se escribieron contra el dictamen de fusión de Satoshi. **Siguen siendo la demolición de ese híbrido.** No son el plan. El plan está en §1, §11 y el anexo 08. Si Musashi pega §§3–9 en el PDF, está reescribiendo el objeto que Harvey acaba de negar.

---

## 2. Dónde Satoshi fue madre

Satoshi identificó las heridas. Luego las vendó con adjetivos. Un adversario de entrenamiento no venda: presiona.

| Movimiento del dictamen | Por qué es maternal | Lo que un jurado dirá |
|---|---|---|
| “B en solitario ya es notablemente superior a A” | Confunde **encaje de programa** con **calidad de propuesta**. A tenía brazos falsables, Gao en el lugar correcto, teorema o imposibilidad, ética de actores. B tiene mejor *fit* en IA y peor *novedad* frente a OptFormer/HyperBO/PFNs4BO. | “¿Superior en qué métrica? ¿En que le gusta más al doctorado de IA?” |
| “Cada mitad repara la herida mortal de la otra” | Retórica de pareja. La herida de B (corrupción exógena) se repara con un **adversario**, no con casarse con A. La herida de A (oráculo parcial) se repara **eliminando A**, no fusionándola. | “Entonces no fusionaron: amputaron A y le pusieron el nombre al muñón.” |
| Nombra el *reward-hacking* del asignador y lo deja como objeción, no como objeto | Es la objeción que mata la tesis en sala. Satoshi la clasifica entre cinco. Un teórico la pone **primera**. | “La asignación usa ganancia de información; el pago usa \(q\). El proveedor fabrica trazas que *parecen* informativas. ¿Dónde está el *proper scoring*?” |
| 2×2×3 factorial como si cupiera | Atribución honesta. Potencia deshonesta. Tres ejes × HPO-B × 5 corridas × mejor respuesta = teatro. | “Muéstreme el cálculo de potencia del piloto o recorte un eje **ahora**.” |
| “El lazo cerrado es trabajo futuro” **y** la pregunta candidata habla de una red que aprende | El título de B ya promete más de lo que el análisis de una pasada puede pagar. La fusión agrava el cheque. | “Si congelan el lazo, no es una red que aprende. Es un prior entrenado offline más un mercado de una ronda.” |
| Pinza de novedad nombrada y no ejecutada | Satoshi ve los dos flancos y aun así recomienda híbrido. Eso es valentía de comité, no de tesis. | AutoML: “OptFormer ya lee políticas distintas.” Economía: “Chen 2026, Deng 2025, AFL, Bittensor. ¿Qué queda?” |

No voy a “respetar el espíritu” de ese dictamen. Voy a quedarme con los **cortes** y tirar el **halago**.

---

## 3. La única cadena que unifica (y cuándo se rompe)

Satoshi acertó en la lógica, no en el tono:

```
mecanismo (p, π)  →  cota de fracción corrupta δ del corpus
                  →  cota de arrepentimiento del metaoptimizador selectivo
```

Eso es **una** pregunta si y solo si la interfaz está sellada: el mecanismo no es el fin; es el instrumento que descarga el supuesto \(\delta\) del teorema de transferencia. Si cualquier eslabón se rompe, el resultado publicable es la **región de imposibilidad**. También es tesis. No es fracaso. B already lo dice; hay que creérselo en el título, no solo en el §3.3.

**Una pasada.** Corpus histórico (HPO-B +, si acaso, trazas propias firmadas) → se entrena \(\hat v\) → se abre **una** ronda de mercado / asignación → se mide transferencia en tareas reservadas. El lazo \(t \to t+1\) **no se evalúa**. Quien lo prometa muere por sistema no estacionario sin equilibrio. Quien lo esconda en el título también.

Eso obliga a **matar palabras del título**. Ver [01_LENGUAJE_ATACABLE.md](fusion_a_b/01_LENGUAJE_ATACABLE.md).

---

## 4. Cinco P0. Satoshi nombró cuatro. Faltaba el lenguaje, y el asignador no está matado.

Detalle en [02_P0_LETALES.md](fusion_a_b/02_P0_LETALES.md). Aquí el ranking que yo usaría en sala:

1. **Asignador.** La regla “ganancia esperada de información / costo” es un *proxy*. Un proveedor racional maximiza el proxy, no \(q\). Sin *proper scoring* (o sin demostrar que el proxy no entra en la utilidad del vendedor), el teórico derriba. **Esto es el teorema de mecanismo, no una objeción.**
2. **Lenguaje.** *Confiable*, *trazas verificables*, *salida segura*, *información útil*, *red que aprende*. Un jurado no necesita un paper: necesita el diccionario. B pone *confiable* en el `<title>`, el `<h1>` y el footer de las siete páginas.
3. **Lazo vs. una pasada.** O se declara en la pregunta madre, o el título es publicidad.
4. **Atribución.** Sin factorial recortado y potenciable, H3 de Satoshi (“el acople gana”) es anecdota.
5. **Pinza de novedad.** Tras amputar CA/LLM de A y MacKay de B, lo que queda es HyperBO/OptFormer + procedencia + pago-por-re-ejecución. Eso es un **paper fuerte**, o una tesis **estrecha**. No es dos propuestas fusionadas.

**P0-bis, mío:** no citar a Gao, Shnayder, jueces LLM ni “evaluación confiable de inferencias” en el PDF fusionado. Eso es A muerta. Dejarla en la bibliografía “por si el comité pregunta” es dejar un flanco abierto. Si no está en el objeto, no está en el texto.

---

## 5. Qué se elimina (más que Satoshi)

Satoshi ya corta: oráculo parcial, CA, jueces LLM, auditoría humana, lazo dinámico, subasta como objeto, DIOS, token, bake-off contra Bittensor/Gensyn/Akash, MacKay como columna.

Yo añado, y no es negociable si quieren que esto entre a La Sabana sin sonrojo:

| Corte | Por qué |
|---|---|
| La palabra **confiable** del título, footer y objetivo general | No está definida. Calibración ≠ confianza. Un juez de sistemas o de ética la pide prestada a ISO/IEC 24028 y ustedes no la tienen. |
| “Trazas verificables” sin decir **quién** verifica **qué** | En doin-domains, verificar es lookup o holdout. En B, parece “firmado”. Son tres verbos. |
| Reproducción de MacKay/Cover como **producto del año 1** | Satoshi la degrada a ablación. B aún la pone como marco teórico §2.1, dos páginas. Eso es relleno que un jurado de IA leerá como tesis de teoría de la información. Ablación = un apéndice, no el marco. |
| Series de tiempo **y** LLM compactos como confirmación | Una familia. Satoshi ya dice una. B aún dice dos. Cortar LLM compactos del alcance doctoral. |
| “DOIN aportará un entorno distribuido y trazable” en el resumen | El resumen no puede oler a blockchain. Una línea al final: runtime opcional, no hipótesis. |
| Shannon / Kolmogorov en el cuerpo | B ya dice que Kolmogorov no se calcula y que Shannon no se aplica a un dataset. Entonces **no ocupen el cuerpo**. Referencia o nada. |
| Subasta, EIP-1559, PoO, canales de pago como contribución | Infraestructura. El código actual **ni siquiera tiene** subasta inversa. Prometerla como “capa de A” es inventar un módulo que no existe. |
| Cualquier frase de “inteligencia de la red / bits por peso” aunque sea para negarla | Negarla es seguir poniéndola. |

Lista operativa de sustituciones: [01_LENGUAJE_ATACABLE.md](fusion_a_b/01_LENGUAJE_ATACABLE.md).

---

## 6. Pregunta, objetivos, hipótesis

Texto cerrado en [03_PREGUNTA_OBJETIVOS_HIPOTESIS.md](fusion_a_b/03_PREGUNTA_OBJETIVOS_HIPOTESIS.md).

**Pregunta madre (una, una pasada, sin “confiable”):**

> ¿Bajo qué condiciones un metaoptimizador con abstención —entrenado sobre trazas heterogéneas cuya fracción corrupta \(\delta\) está acotada por un mecanismo de re-ejecución y pago, y que **no** remunera el estimador de valor— reduce el arrepentimiento y el costo verificado hasta un objetivo en tareas nuevas, sin transferencia negativa, y sin que fabricar trazas de alta ganancia de información aparente sea una mejor respuesta?

Tres objetivos: interfaz \(\delta\); dos módulos (selectivo + mecanismo de no-rentabilidad del *proxy*); evaluación factorial **recortada**.

Tres hipótesis: H1 transferencia bajo \(\delta\); H2 no-rentabilidad de fabricar (descarga H1); H3 atribución de **un** eje, no de tres.

Si H1 falla y H2/H3 se sostienen: las trazas históricas **no** justifican un metaoptimizador; el mecanismo sí evita pagar basura. Publicable. Si H2 falla: el capítulo económico sobra; KEEP B con corrupción inyectada.

---

## 7. Modelo mínimo y el asignador

[04_MODELO_Y_ASIGNADOR.md](fusion_a_b/04_MODELO_Y_ASIGNADOR.md).

Hecho que Satoshi no escribió con suficiente sangre: **el pago no puede ser función de \(\hat v\)**. Deng (ICML 2025) da IC de procurement cuando la calidad es **conocida**. Chen (AISTATS 2026) prueba que en un mercado de datos **no hay DSIC no trivial**. Si \(\hat v\) entra en \(\pi_i\), ustedes están en Chen, no en Deng, y el teorema se acaba.

El asignador es el hueco. Tres defensas, en orden de honestidad:

1. **Mejor:** \(\hat v\) solo ordena el *menú* que el consumidor publica; los trabajadores pujan **costo**; ganan los de menor costo; cobran si \(q\) re-ejecutado ≥ \(\theta\). Entonces el *reward-hacking* de \(\hat v\) no paga. El teorema de mecanismo es aburrido y **por eso es defendible**.
2. **Aceptable:** \(\hat v\) prioriza qué \(x\) se financian, pero el trabajador no elige \(x\) (el consumidor / el protocolo lo fija). Fabricar trazas pasadas puede sesgar \(\hat v\) **si** el trabajador también emitió el corpus. Se corta: el entrenador de \(\hat v\) no cobra, y el corpus de entrenamiento es **histórico congelado** (HPO-B), no emitido por los mismos \(w_i\) de la ronda.
3. **Suicida:** asignación por información / costo **y** los mismos agentes generan las trazas que entrenan \(\hat v\). Eso es el lazo. Satoshi lo mandó a futuro. No lo reintroduzcan por la puerta del asignador.

Si eligen (1) o (2), la “endogeneidad” que Satoshi vende como razón de fusionar **se reduce** a: en la ronda de evaluación, agentes intentan entregar \(x\) basura y el pago lo impide. Eso **ya** es el brazo de verificación de doin-domains. La fusión aporta de verdad solo si el corpus de \(\hat v\) puede envenenarse **y** H2 lo mide. Para eso el envenenamiento tiene que actuar sobre el **corpus de entrenamiento**, no solo sobre la entrega de la ronda. Una pasada + corpus HPO-B público hace ese envenenamiento **artificial** (el candidato lo inyecta), que es exactamente la herida de B que Satoshi decía curar.

**Díganlo:** con análisis de una pasada y HPO-B público, la corrupción endógena del *corpus de entrenamiento* no existe salvo que ustedes **simulen** un mercado de trazas *antes* de entrenar \(\hat v\). Eso es un experimento, no un fenómeno de red. Llamarlo “red descentralizada que aprende” es el mismo pecado de lenguaje.

La endogeneidad honesta, en una pasada, es esta y solo esta:

> Un generador estratégico produce un corpus sintético o altera HPO-B; \(\hat v\) se entrena; luego se asigna. H2 pregunta si \((p,\pi)\) hace no-positiva esa generación. El generador **no** es el mismo agente que cobra la ronda de HPO, o se declara que sí y se acepta el sesgo.

Eso cabe. No es Bittensor. No es A. Es un capítulo.

---

## 8. Experimento que puede refutar, no impresionar

[05_EXPERIMENTO_Y_ATRIBUCION.md](fusion_a_b/05_EXPERIMENTO_Y_ATRIBUCION.md).

Satoshi pidió 2×2×3. Yo pido **2×2 más un brazo endógeno anidado**, y el tercer eje se decide en el piloto o se muere.

- Primario: HPO-B, ≥30 tareas/familia, 5 corridas, unidad = tarea no vista, Holm.
- Confirmatorio: **una** familia de series de tiempo. No LLM.
- Controles: random y BO sin transferencia en **cada** celda.
- Adversario: repertorio fijo **y** un optimizador de mejor respuesta contra el protocolo **publicado**. Sin el segundo, H2 es un examen que el candidato califica.
- Potencia congelada **antes** de H1–H3. Si el piloto no da 0.80 al margen de H1, se sube \(N\) o se corta un contraste. No se “confía”.

---

## 9. Novedad residual (sin poesía)

[06_NOVEDAD_Y_AFIRMACIONES.md](fusion_a_b/06_NOVEDAD_Y_AFIRMACIONES.md).

**Veredicto de novedad del objeto recortado: PLAUSIBLE.** No STRONG. No ALREADY DONE. WEAK si vuelven a escribir “confiable / red que aprende / bits por peso”.

Lo que **no** es novedad: entrenar un modelo con pares parámetro–métrica; espacios heterogéneos (Fan TMLR / HyperBO+); imitar políticas (OptFormer); PFNs; HPO-B; re-ejecutar para verificar (Gensyn, doin-domains D1); subasta inversa de compute (Akash); ranking entre pares (Bittensor).

Lo que **puede** ser novedad, y es estrecho:

1. Abstención calibrada + procedencia como **canales del modelo**, no metadatos.
2. Cota de arrepentimiento en función de \(\delta\), o indistinguibilidad cuando \(\delta\) o la procedencia no identifican.
3. Proposición: \((p,\pi)\) que hace no-rentable fabricar trazas de alto *proxy* y bajo \(q\), **sin** pagar \(\hat v\).
4. Evidencia factorial: transferencia × (corpus limpio / \(\delta\) inyectada / generador estratégico del corpus).

Eso es un doctorado en IA si el 70 % de las páginas son (1)+(2)+(4) y el 30 % es (3). Al revés, es ingeniería de mercados con un surrogate.

---

## 10. Encaje en el Doctorado en Inteligencia Artificial

Satoshi: B encaja; A no; el híbrido hereda B. De acuerdo en el diagnóstico, no en la complacencia.

La pregunta de B §5.4 es la correcta: *cómo un sistema aprende a mejorar otro proceso de aprendizaje y cómo reconoce los límites de lo aprendido*. Abstención = el reconocimiento. Eso es IA.

El capítulo de mecanismo es **instrumento de evidencia**, como un protocolo de contaminación de labels. El día que el PDF explique EIP-1559, Hayek o “mercado descentralizado de inferencia” en el resumen, el jurado 1 (el de IA) dirá Ingeniería. Tienen razón.

A en solitario **sí** podía ser IA: evaluación de LLM bajo jueces imperfectos es un problema de *learning systems*. Satoshi la mata al cortar el oráculo parcial. Correcto para caber. Incorrecto decir que A “era débil en IA”: era otro objeto. No la resuciten a medias.

---

## 11. Qué le pido a Harvey, luego a Musashi

Instrucciones: [07_INSTRUCCIONES_MUSASHI.md](fusion_a_b/07_INSTRUCCIONES_MUSASHI.md) · flujo real: [08_FLUJO_MINERO_NO_MERCADO.md](fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md).

No reescribir el PDF hasta que Harvey marque **una** casilla. Las tres de la versión anterior de este dictamen (H1 híbrido / H2 KEEP B / H0 KEEP A) quedan **degradadas**. La casilla que pega con lo que Harvey acaba de decir es **M**.

| Opción | Qué es | Cuándo |
|---|---|---|
| **M — emisión, no mercado** | Pregunta original. Nodo entra, dominio auto/elegido, GPU trabaja, **sin precio**. Teoría: recompensa por \(\Delta q\) verificado + autoselección + basura no rentable. PoO = dificultad. **Sin Hayek, sin subasta, sin CA, sin metaoptimizador.** | Si el doctorado puede tragar un objeto de incentivos sobre trabajo de aprendizaje. Es lo que el sistema **hace**. |
| **H2 — KEEP B** | AutoML selectivo. Se **cierra** la pregunta de mineros (paper aparte, no tesis). | Si La Sabana es AutoML/incertidumbre y Harvey acepta el duelo. |
| **H1 — híbrido** | ~~B + capítulo de mercado~~ | **REJECT** tras la aclaración. Sigue siendo eBay. |
| **H0 — KEEP A literal** | Subasta + Hayek + CA + LLM | **REJECT**. A describe un sistema que Harvey dice que **no opera**. |

Orden: **M**, o **H2 si cierran M**. No ambas. No “un poquito de Hayek para el marco”. No fusionar para no elegir.

Si marcan M, Musashi tira B y reescribe desde el anexo 08, no desde A. A solo aporta: verificación del incremento, no pagar basura, Sybil como costoso. El resto de A (ofertas, consumidor \(T\), Hayek, jueces) es **falso** respecto al flujo.

---

## 12. Índice de anexos

| Archivo | Contenido |
|---|---|
| [fusion_a_b/README.md](fusion_a_b/README.md) | Mapa del paquete |
| [fusion_a_b/01_LENGUAJE_ATACABLE.md](fusion_a_b/01_LENGUAJE_ATACABLE.md) | Palabras que un juez tumba, y sustitutos |
| [fusion_a_b/02_P0_LETALES.md](fusion_a_b/02_P0_LETALES.md) | P0 con ataque y parche |
| [fusion_a_b/03_PREGUNTA_OBJETIVOS_HIPOTESIS.md](fusion_a_b/03_PREGUNTA_OBJETIVOS_HIPOTESIS.md) | Texto para pegar en el PDF |
| [fusion_a_b/04_MODELO_Y_ASIGNADOR.md](fusion_a_b/04_MODELO_Y_ASIGNADOR.md) | Actores, información, pago, *proper scoring* |
| [fusion_a_b/05_EXPERIMENTO_Y_ATRIBUCION.md](fusion_a_b/05_EXPERIMENTO_Y_ATRIBUCION.md) | Factorial recortado, potencia, refutación |
| [fusion_a_b/06_NOVEDAD_Y_AFIRMACIONES.md](fusion_a_b/06_NOVEDAD_Y_AFIRMACIONES.md) | 15 cercanos, permitido / prohibido |
| [fusion_a_b/07_INSTRUCCIONES_MUSASHI.md](fusion_a_b/07_INSTRUCCIONES_MUSASHI.md) | Checklist de reescritura |
| [fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md](fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md) | El sistema que Harvey corre: emisión, no eBay |

Fin del principal. Si quieren halago, lean a Satoshi. Si quieren entrar, elijan **M** o **H2** y recorten. No fusionen dos textos que ya no describen la red.
