# 08 — Flujo minero, no mercado

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Harvey: enchufas un nodo, eliges dominio o se autoselecciona, la GPU trabaja, **tú no fijas precio ni haces ofertas**. No es eBay ni MercadoLibre de modelos. El incentivo es para que la gente meta recursos, como un minero de Ethereum, no para que un consumidor compre un artefacto.

Eso no es un detalle de UX. **Cambia el objeto doctoral.** A, B y el híbrido lo ignoran. El código no.

---

## 1. Hecho de sistema (código, no deseo)

`doin-core@a90bca2`:

- El operador produce un `Optimae` para un `domain_id`: parámetros + `reported_performance`.
- Si se acepta, entra un `performance_increment` ponderado por el peso del dominio.
- `ProofOfOptimization` emite bloque cuando la **suma ponderada de incrementos** ≥ umbral. El umbral se mueve para un tiempo de bloque objetivo. Eso **es** dificultad de minería, con otro nombre.
- `FeeMarket`: stake 5× sobre el optimae; 20 % de quema si se rechaza. Antispam, no precio de un comprador.
- `Task`: el evaluador **hace pull** de verificación o inferencia por dominios que soporta. No hay `Bid`.
- Commit-reveal evita *front-running* de parámetros. No es una oferta.

Nadie en ese bucle publica “paga 3 AKT por este entrenamiento”. El nodo entra. El protocolo mide \(\Delta q\). El protocolo decide si eso mueve el umbral / la emisión. Eso es **ETH 1.0 mental model**: conectas GPU, el protocolo te paga *si* el trabajo cuenta, la dificultad se ajusta, tú no negocias con un cliente.

Hayek no entra. No hay precio. El conocimiento que Hayek agrega **no tiene dónde sentarse**.

---

## 2. Qué quería Harvey, y dónde se perdió

| Intención | Dónde se escribió otra cosa |
|---|---|
| Incentivo para aportar GPU a optimizar | A lo convirtió en **subasta inversa** de tareas \(T=(d,g,B,\ldots)\) |
| Base teórica económica | A citó Hayek + Hurwicz + CA. Hurwicz es diseño de mecanismos **con mensajes/precios**. CA es elicitación de jueces. Ninguno es emisión. |
| No tasación del operador | B ni siquiera habla de operadores. Habla de un metaoptimizador. |
| Autoselección de dominio | A lo dijo en una frase (“la auto-selección de dominio… no será aprendida”) y luego diseñó un mercado de tareas demandadas. Eso es contradicción en la misma página. |

La deriva no es un accidente de redacción. Es tres objetos distintos pegados al logo DOIN:

1. **Minería de incremento verificado** (el sistema).
2. **Mercado de inferencia/LLM** (A).
3. **Meta-HPO** (B).

Satoshi fusionó 2 y 3. Harvey acaba de recordar que quería 1.

---

## 3. Analogías que sí y que no

| Analogía | ¿Sirve? | Por qué |
|---|---|---|
| Minero ETH PoW / Bitcoin | **Sí, como flujo** | Entras, no pones precio, el protocolo emite, la dificultad se mueve. El trabajo en ETH era *hash inútil*. Aquí se quiere trabajo *de optimización*. Esa diferencia **es** la tesis. |
| Akash / eBay / MercadoLibre | **No** | Hay comprador, hay puja, hay lease. Harvey no opera así. |
| Hayek 1945 | **No** | Precio como estadístico suficiente. Sin precio, no hay estadístico. |
| Bittensor | **Parcial** | Tampoco pones precio de catálogo. El protocolo infla según ranking entre pares. El ledger **no** audita el modelo. DOIN sí quiere auditar \(\Delta q\). Eso es la diferencia que A ya apuntaba, y luego ensució con subastas. |
| Gensyn Submitter/Solver | **Parcial** | Ahí **sí** hay un Submitter que paga una tarea. Eso es mercado de cómputo. Harvey dice que el nodo **no** es ese Solver esperando un cliente. |
| Deng procurement / Dutting auctions | **No** | Exigen ofertas y, Deng, calidad conocida por el subastador. |
| Chen 2026 data marketplace | **No** | Compradores de una media. |

La literatura que **sí** hay que citar si marcan **M**:

- Holmström 1982: pagas un *output* (aquí \(\Delta q\) verificado), no el esfuerzo (la GPU puede estar “minando” basura).
- PoUW: Ofelimos (CRYPTO 2022), Cao et al. (trade-off utilidad vs seguridad), SoK 2025 (“la PoUW no es tan útil como se esperaba”).
- Proof-of-Learning y su quiebre (Jia; Fang EuroS&P 2023): **no** paguen trayectoria; paguen fitness en holdout/oráculo, que es lo que doin-domains ya argumenta.
- Contests / premios: la emisión cuando hay un nuevo óptimo es un *winner-take-increment*, no un salario por hora.
- Autoselección: el nodo elige dominio (o un scheduler lo elige). Eso es asignación de agentes a concursos, no una subasta de un bien. Congestión: todos se van al dominio fácil y el umbral global (PoO) no distingue. **Ese** es un teorema posible.

---

## 4. Pregunta madre si la opción es M

¿Bajo qué reglas de emisión y de verificación un nodo que **no fija precio**, y que elige o se le asigna un dominio, tiene mejor respuesta en aportar GPU a producir incrementos de desempeño **re-ejecutables**, y no en fabricar incrementos, copiar campeones o saturarse en el dominio más fácil, a un presupuesto de verificación acotado?

Una pregunta. No hay consumidor. No hay \(\hat v\) de meta-HPO. No hay Hayek.

---

## 5. Modelo mínimo (M)

**Actores.** Operador de nodo (elige dominio o acepta autoselección). Evaluador que re-ejecuta / hace lookup. El protocolo (umbral, emisión, quema). **No** hay comprador.

**Privado.** Costo eléctrico y de oportunidad del operador; si va a hacer trampa; qué dominio es “fácil” para su hardware.

**Público.** Dominios, campeón actual, umbral, reglas de sello.

**Acción.** Trabajar en \(d\), entregar optimae, o mentir.

**Pago.** Emisión \(\pi\) si el optimae se acepta (incremento verificado > 0 y sello válido), menos quema/stake si se rechaza. \(\pi\) lo fija el **protocolo** (como recompensa de bloque), no una puja. El umbral de PoO hace de dificultad: si entra basura aceptada, los bloques se aceleran y el umbral sube — eso ya está en el código. Hay que **demostrar** si eso basta o si hay que ponderar por dominio para evitar el yacimiento fácil.

**Verificación.** Oráculo 1a (HPO-B/NATS lookup) como control. Un dominio de re-entrenamiento barato como confirmación. **Sin** jueces LLM.

**Autoselección.** El nodo elige \(d\) o un scheduler lo asigna **sin precio**. La tesis tiene que decir qué hace el scheduler (aleatorio, por peso de dominio, por reputación). Si “se autoselecciona” y todos eligen el dominio de lookup más barato, el incentivo **no** está dirigiendo trabajo útil. Eso es H2 de M, no un detalle de ingeniería.

---

## 6. Hipótesis de M (si Harvey marca M)

**H1 · Participación.** Con emisión por incremento verificado, un operador racional con costo \(c\) entra al dominio donde \(\mathbb{E}[\pi]-c\ge 0\) y no entra donde es negativo. Contrastable en simulador con \(c\) conocidos.

**H2 · No yacimiento fácil.** Si los dominios tienen costo de verificación distinto, una regla de **solo** umbral global **no** iguala trabajo útil por GPU. O se pondera el dominio (ya existe `domain.weight` en PoO) de forma que H2 se sostenga, o se reporta que el PoO actual **falla** H2. Eso es un resultado. El código ya tiene el mango.

**H3 · Basura no rentable.** Mejor respuesta (copia de campeón, incremento inflado, replay) tiene utilidad no positiva bajo sello de re-ejecución/lookup + stake/quema, a \(p\) de auditoría acotada. Si falla, el incentivo de minero **paga hash inútil con disfraz de HPO**. Fang ya avisó.

No hay H de transferencia OptFormer. Si la quieren, es **otro** paper.

---

## 7. Optimalidad que se puede defender aquí

No bienestar de un consumidor (no hay consumidor). No DSIC de subasta (no hay puja).

Sí:

- **IR del operador:** no trabaja si \(\mathbb{E}[\pi]<c\).
- **No-rentabilidad de la desviación** (H3).
- **Aproximación de “trabajo útil”:** emisión proporcional a \(\Delta q\) verificado vs emisión por bloque (como minar), vs premio solo al récord. Tres reglas, un simulador. Eso es economía **elemental** y honestamente doctoral si el teorema acota cuándo el umbral+peso evita el yacimiento fácil.

No: “incentivo óptimo universal”. Cao: hay trade-off utilidad/seguridad en PoUW. Escríbanlo como límite, no como footnote.

---

## 8. Encaje IA vs ingeniería (sin anestesia)

M es **sistemas de aprendizaje + incentivos**. Un jurado de IA puro dirá: “eso es su maestría otra vez (PoO) más verificación”. Tienen que marcar la diferencia en **una** frase:

> La maestría propuso PoO. Esta tesis pregunta si la emisión sin precio, con autoselección de dominio, puede **no** recompensar incrementos que no sobreviven re-ejecución, y si el umbral global empuja el cómputo al dominio fácil. El objeto es la regla, no la red.

Si no pueden decir eso sin reabrir blockchain-as-contribution, M no entra a La Sabana. Entonces H2 (B) y M se van a un paper de sistemas. Elegir. No fusionar.

---

## 9. Qué se borra del PDF si marcan M

- Hayek, subasta inversa, consumidor \(T\), ofertas selladas, \(B=b_w+\sum b_e\).
- *Confiable*, metaoptimizador, OptFormer como columna, MacKay.
- CA, Gao como contribución, jueces LLM.
- “Mercado descentralizado de inferencia”.
- Cualquier frase de precio que agrega información.

Qué se conserva de A: verificación del incremento; no pagar basura; Sybil costoso; ledger no es oráculo; pagos simulados.

Qué se conserva de B: **casi nada**, salvo HPO-B como **oráculo de \(q\)** (lookup), que ya está en doin-domains.

Qué se conserva del código: PoO, pesos de dominio, stake/quema. Se **nombra** como prototipo, no como prueba.

---

## 10. Relación con “transferencia de información”

Harvey mencionó que lo de Hayek le parece óptimo *en abstracto* para transferencia de información. Duro: **en este flujo no hay precio, luego no hay transferencia hayekiana.** La información que se transfiere es el **campeón** (parámetros + \(q\)) y el **sello**. Eso es un canal de *broadcast* de un récord, no un mercado. Llamarlo Hayek es un error de categoría que un economista del comité cierra en un minuto.

Si algún día hay un consumidor que paga inferencia (el `INFERENCE_REQUEST` del `Task`), **ese** sí podría ser un mercado. Harvey acaba de decir que el operador de GPU **no** es ese usuario. No diseñen la tesis para el `Task` que no es el de minería.
