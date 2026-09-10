# 04 — Modelo mínimo y asignador

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

**Estado:** este anexo describe el modelo del **híbrido REJECT**. Harvey aclaró que el operador **no fija precio**. El modelo vigente, si marca **M**, es [08_FLUJO_MINERO_NO_MERCADO.md](08_FLUJO_MINERO_NO_MERCADO.md). No pegar las pujas de abajo en el PDF.

Si este anexo y el PDF discrepan sobre el híbrido muerto, da igual: el híbrido está muerto. El asignador era P0-1 de *ese* objeto.

---

## Actores

| Actor | Quién es | Qué **no** es |
|---|---|---|
| Consumidor \(C\) | Publica tarea \(\tau\), presupuesto \(B\), umbral \(\theta\), horizonte \(H\) | No es “la red” |
| Trabajador \(w_i\) | Ejecuta una configuración \(x\) asignada, o (si se permite) ofrece costo | No entrena \(\hat v\) en la misma ronda en que cobra |
| Evaluador / re-ejecutor | Lookup HPO-B o re-entreno acotado | No es juez LLM |
| Entrenador de \(\hat v\) | Agente **separado** o el candidato, offline | **No cobra** \(\pi\) de la ronda |
| Generador de corpus (H2) | Adversario que produce o altera trazas **antes** de entrenar \(\hat v\) | No se vende como nodo de una mainnet |

Humanos: ninguno en el camino crítico. HPO-B es tabla. Series de tiempo: métrica ejecutable. Ética: la de siempre, sin reclutar jueces.

---

## Información

| Quién | Privada | Pública |
|---|---|---|
| \(C\) | Valor de \(\tau\), costo de oportunidad | \(B, \theta, H, g\) |
| \(w_i\) | Costo \(c_i\), intención de entregar basura | Oferta de costo (si hay), artefacto |
| Generador | Estrategia de envenenamiento | Corpus entregado, procedencia declarada |
| \(\hat v\) | — | Hash, ventana, código, **congelado** antes de la ronda |
| Ledger / log | — | Commits, lookups, pagos simulados. **No** es oráculo |

Esfuerzo de entrenamiento interno de \(w_i\) es no observable (Holmström). Por eso el objeto de pago es \(q\) re-ejecutado / tabulado, no el esfuerzo.

---

## Unidad: traza con procedencia

Una traza es la tupla:

\[
\big(x,\; \text{política de origen},\; \text{presupuesto},\; \text{censura/terminal},\; y,\; c,\; \text{sello},\; \tau_{\text{fuente}}\big)
\]

- \(x\): configuración (posiblemente con máscaras: parámetros ausentes).
- Sello: lookup HPO-B **o** hash de re-ejecución. No “verificable” suelto.
- Política de origen y censura son **canales del modelo**, no footnotes.

Partición **por tarea**. Ningún trial de una tarea de test entra al entrenamiento de \(\hat v\).

---

## Asignación (defensa A, la única que recomiendo)

1. \(\hat v\) (congelada) produce un ranking de \(x\) **del espacio publicado por \(C\)**.
2. Si la compuerta de abstención dispara, el ranking se ignora y se usa BO sin transferencia / random (predeclarado).
3. Se financian las primeras \(k\) configuraciones que caben en \(B\), o se pide puja de **costo** entre trabajadores para **esas** \(x\) (infraestructura, no objeto).
4. Pago:

\[
\pi_i = \mathbf{1}[q(x)\ge \theta]\, b_i \;-\; \ell\,\mathbf{1}[\text{sello inválido o \(q\) no reproducible}]
\]

\(b_i\) sale de la puja de costo o de un precio publicado. **\(\hat v\) no aparece en \(\pi_i\).**

Con esto, fabricar un *proxy* alto **no cobra** en la ronda. El único ataque que queda es envenenar el **corpus de entrenamiento** de \(\hat v\) (H2), y eso lo hace el **generador**, no el trabajador de la ronda — o se declara que es el mismo y se acepta el sesgo, que yo no recomiendo.

---

## Por qué no *proper scoring* del asignador

Un *proper scoring rule* haría que reportar la creencia verdadera sobre \(q\) o sobre el óptimo sea óptimo para el que reporta. Ustedes no tienen un reportero de creencias: tienen un **modelo entrenado offline**. Forzar proper scoring aquí es otra tesis (Dutting: se aprende el mecanismo, no el HPO). Prometerlo es P0-1 suicida (defensa C del [02](02_P0_LETALES.md)).

La defensa honesta no es “el proxy es proper”. Es “el proxy **no paga**”.

Entropy Search sigue siendo **regla interna** del selectivo para elegir \(x\) del menú, como BO elige el próximo query. Eso es AutoML clásico. No es un mercado de información mutua. No lo llamen mercado.

---

## Mecanismo → \(\delta\) (el puente)

H2 necesita un generador que, **antes** de entrenar \(\hat v\), altera o sintetiza trazas para maximizar su utilidad:

\[
U_G = \mathbb{E}[\pi_G] - c_G(\text{fabricar})
\]

\(\pi_G\) no puede ser “el selectivo acierta”: eso es circular. \(\pi_G\) tiene que ser una regla **publicada**: p. ej. pago por traza aceptada al corpus (sello de lookup / re-ejecución) menos penalización si una muestra auditada no reproduce.

Entonces:

- Si el sello es lookup HPO-B, fabricar una fila **nueva** que no está en la tabla se detecta con probabilidad 1 al auditar. \(\delta\) de filas inventadas → 0. H2 se vuelve trivial y **hay que decirlo**.
- La corrupción no trivial en HPO-B es: **censurar** fallos, **duplicar** éxitos, **relabel** una fracción \(f\) de \(y\), **mentir la política de origen**. Eso sí es \(\delta\). El mecanismo (auditoría de una muestra + rechazo del lote) acota \(f\), o no.
- En la familia confirmatoria (series de tiempo), re-ejecución real cuesta. Ahí \(p\) y \(\ell\) no son teatro.

**Imposibilidad a escribir:** si la procedencia no es identificable (el generador puede relabel política de origen sin sello), ninguna regla que solo mira \((x,y)\) separa transferencia útil de envenenamiento. Eso es el (ii) de Satoshi, y es el resultado teórico más honesto de los tres.

---

## Optimalidad que se puede escribir

| Noción | ¿Sí? |
|---|---|
| Arrepentimiento simple normalizado (HPO-B) | Sí, H1 |
| Costo-hasta-objetivo verificado | Sí |
| IR del consumidor: no paga si \(q<\theta\) | Sí |
| Bienestar \(q-c\) aproximado al estilo Deng | Solo en D1 con \(q\) tabular conocido, y **no** es el teorema doctoral |
| DSIC / “incentivo óptimo” | **No** |
| Equilibrio del lazo | **No** |

Chen 2026: no prometan DSIC. Gao: no reabran peer prediction. Holmström: no paguen equipos por residual compartido.

---

## Qué existe hoy en código (para no mentir)

- `ProofOfOptimization`: bloque si suma ponderada de incrementos ≥ umbral. No matching de \(C\).
- `FeeMarket`: stake 5×, quema 20 % si se rechaza. Antispam.
- `Task`: `OPTIMAE_VERIFICATION` | `INFERENCE_REQUEST`.
- Commit-reveal, reputación, quorum, payment channel: existen como **modelos**.
- Subasta inversa, `Bid`, `Lease`: **no existen**.

El capítulo de mecanismo de la tesis es un **simulador**. Decirlo. DOIN no es el experimento; es un runtime futuro declarado como no-hipótesis.
