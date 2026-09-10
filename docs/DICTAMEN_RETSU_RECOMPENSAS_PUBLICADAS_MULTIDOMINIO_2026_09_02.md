# Dictamen Retsu — recompensas publicadas, autoselección y pertinencia doctoral

**Para:** Harvey (Maestro) y Musashi  
**De:** Retsu  
**Fecha:** 2026-09-02  
**Modo:** adversario. No reescribe propuestas. No implementa. No GPU.  
**Objeto:** el *tercer mecanismo* (recompensa publicada con autoselección), no A, no B, no el híbrido de Satoshi, no la minería pura del dictamen anterior.

Identidades (hechos, re-verificadas hoy):

| Objeto | Digest |
|---|---|
| A @ `7ae8643` HTML | `f71b34c6b7d576c50f6c8a92f5e9434548aae2af40fd4626a5866d74dee79b56` |
| B HTML / PDF working tree | `b30e911b…b2322011` / `151418b9…f3f99207` |
| `doin-core` | `a90bca2` |
| `doin-node` | `8bfc64f` |
| `doin-domains` | `d01708ae` |

---

## 1. Teach-back del sistema deseado

Esto es lo que *dicen* querer. Si está mal, el resto del dictamen apunta al muñeco incorrecto.

Hay **demanda financiada**: una persona, una aplicación, un patrocinador o un fondo deposita dinero y publica que quiere trabajo en un dominio (inferencia, optimización, o ambos). El **protocolo publica una recompensa** \(r_d\) (o \(r_{d,\text{tipo}}\)) por unidad de trabajo **aceptado** en ese dominio. Los nodos **no pujan**. Cada operador tiene un costo privado \(c_i\) (electricidad, oportunidad, hardware) y decide en local: entrar, no entrar, o cambiar de dominio. Puede restringirse a un dominio, a una allowlist o a una denylist. Dentro de lo autorizado, un scheduler puede elegir la siguiente tarea. El nodo puede hacer inferencia, optimización, ambas o ninguna. **Solo cobra si el resultado es aceptado** (sello de re-ejecución / lookup / holdout, no el ledger como oráculo). **No hay token propio.** El dinero de \(r\) sale de depósitos de usuarios, ingresos de inferencia, fondos por dominio, patrocinadores o tesorería. \(r\) puede moverse con demanda financiada, congestión, dificultad y oferta de trabajo aceptado. El ledger anota compromisos y pagos.

Eso **no** es eBay (no hay matching de pujas). **No** es minería de ETH-PoW (allí la recompensa la emite el protocolo contra inflación, sin un comprador). **No** es A (A escribe ofertas selladas). **No** es B (B no tiene demanda). Es un **posted price** / premio publicado, con autoselección.

El código actual **no** implementa (1)–(11). Implementa PoO de incrementos, fee EIP-1559, cola `Task` de verificación o inferencia, stake/quema. Eso es el prototipo de *dificultad y antispam*, no el mecanismo doctoral. Una tesis **puede** proponer la extensión si la declara como tal. El código no es una cárcel. Tampoco es una prueba.

---

## 2. Correcciones explícitas al dictamen anterior

El archivo [`DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md`](DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md) y el anexo [`fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md`](fusion_a_b/08_FLUJO_MINERO_NO_MERCADO.md) acertaron en tres cosas y mintieron en dos por exceso.

**Sostengo:**

1. El PoO de `a90bca2` es minería de incrementos ponderados con umbral de dificultad. Hecho de código.
2. A, tal como está escrita, es una subasta inversa con Hayek. No describe este flujo.
3. B es AutoML. No describe este flujo. Fusionarlas sigue siendo REJECT.

**Retiro, y era un error de categoría mío:**

1. **“Si el operador no fija precio, no hay precio.”** Falso. El operador no *puja*. El protocolo (o el fondo) **publica** \(r\). Eso es un precio: un salario/premio take-it-or-leave-it. La literatura lo llama *posted-price mechanism* (Badanidiyuru–Kleinberg–Singer EC 2012; mecanismos posted-price de procurement online, p. ej. arXiv:2502.18265). EIP-1559 es un posted price dinámico sobre el *base fee*. Yo convertí “no hay puja” en “no hay Hayek posible”. Eso cierra una puerta que el objeto nuevo abre — a medias, ver §3.
2. **“El sistema futuro carece de demanda porque el código actual no la tiene.”** Una tesis puede proponer demanda financiada no implementada. Debí distinguir *estado del runtime* de *objeto propuesto*. El anexo 08 trató el código como el horizonte. Eso es una restricción ficticia. La orden de hoy tiene razón.

**No retiro:** Hayek en el PDF de A, pegado a ofertas selladas, sigue siendo un adorno letal *para A*. El veredicto de Hayek para *este* mecanismo es otro, y no es un sí.

---

## 3. Veredicto sobre Hayek

Hayek, *The Use of Knowledge in Society* (AER 1945): el sistema de **precios** agrega conocimiento disperso que ningún planificador posee. El precio es un estadístico suficiente de escasez relativa.

| Fuente de \(r\) | ¿Hayek? | Por qué |
|---|---|---|
| Depósito de un usuario que **no puede** recobrarlo vía nodos propios (Sybil) | **Débil, y solo eso** | \(r\) resume “alguien pone dinero en riesgo porque quiere ese trabajo”. Un número, un demandante. No es el mercado hayekiano de muchos precios bilaterales. Es un **salario publicado por un monopsonio** (o un premio). |
| Tesorería / inflación / “el protocolo decide \(r\)” | **No** | Nadie revela disposición a pagar. Es emisión. Volvemos a minería. |
| Ingresos de inferencia del mismo dominio | **Condicional** | Solo si la inferencia la pagan usuarios reales, no el propio operador. Entonces \(r_{\text{opt}}\) hereda demanda de \(r_{\text{inf}}\). Subsidio cruzado, ver Q15. |
| Ajuste por congestión / cola | **No por sí solo** | Es un **controlador**. Puede estabilizar o oscilar (Leonardos et al., *dynamic posted-price* vs EIP-1559, ACM 2021: el posted dinámico puede ser más estable *o no*, según la regla). Congestión de trabajo **falso** no es escasez hayekiana. |
| \(r\) aprendido por un modelo que observa a los nodos | **No; es el enemigo** | El “precio” pasa a ser un proxy Goodhart. Los nodos entrenan contra el controlador. Q9. |

**Frase permitida:** “una recompensa publicada, financiada con depósitos no recapturables, puede transmitir la disposición a pagar del demandante y el costo de oportunidad local del nodo (acepta iff \(r \ge c_i\)).”

**Frase prohibida:** “el precio agrega el conocimiento de la sociedad / Hayek justifica DOIN.” Un economista del comité pregunta: *¿cuántos precios? ¿quién no puede recapturar?* Si la respuesta es “uno, y el atacante deposita y mina”, Hayek se cae.

El posted price **sí** tiene teoría propia, y es más honesta que Hayek aquí: el trabajador con costo privado acepta o rechaza; no revela \(c_i\) más que un bit. Eso es exactamente la gracia (y el límite) de posted prices frente a subastas.

---

## 4. Veredicto: una cadena, dominios lógicos, no rollups

**Una liquidación común. Dominios lógicos. No una chain por dominio. Rollups fuera de la tesis.**

- **Una chain por dominio** fragmenta (i) seguridad (el ataque 51 % es más barato en la chain flaca), (ii) liquidez de los depósitos, (iii) la contabilidad de tesorería. Appchains sin anclaje heredan exactamente ese problema. No lo prometan.
- **Rollups anclados** heredan seguridad del L1 *si* hay pruebas / un puente no-mentiroso. Eso es una tesis de sistemas, no de IA, y no está en el código. Mencionarlos en “trabajo futuro” como una línea. No como objetivo.
- **Dominios lógicos** sobre un ledger (o un contrato de escrow en un L1 existente, sin token propio): el peso `domain.weight` ya existe en PoO. La demanda, \(r_d\), el sello y el pago son filas de un mismo libro. Eso **sí** cierra la contabilidad (Q14) si el activo de liquidación es externo (ETH, USDC, COP en escrow). Sin token propio **se puede** cerrar: el token no es la contabilidad; el escrow sí.

El ledger **no** decide \(q\). doin-domains ya separa oráculo / holdout. Conservarlo.

---

## 5. Modelo económico con conservación de fondos

Notación. Dominio \(d\). Tipos de trabajo \(\tau \in \{\text{inf}, \text{opt}\}\). Recompensa publicada \(r_{d,\tau}(t)\). Depósito del dominio \(F_d(t)\). Tesorería \(T(t)\).

**Identidad (obligatoria, o el mecanismo es un saco roto):**

\[
\sum_d F_d(t) + T(t) \;=\; \text{depósitos netos} \;-\; \text{pagos aceptados} \;-\; \text{quemas} \;-\; \text{reembolsos}.
\]

Nadie paga \(r\) que no esté cubierto por \(F_d\) o por una línea explícita de \(T\) con tope y fecha de corte.

**Regla de pago.** El nodo \(i\) entrega un artefacto con sello. Si se acepta:

\[
\pi_i = r_{d,\tau} \quad \text{si } q \text{ pasa el sello y } F_d \ge r_{d,\tau};\quad 0 \text{ si no}.
\]

Si \(F_d < r\), la tarea **no se publica** o se publica \(r' = F_d\). Un fondo insolvente no promete. Contraejemplo 8.

**Quién no puede ser el mismo bolsillo.** El demandante que deposita en \(F_d\) y el nodo que cobra \(\pi_i\) deben ser identidades económicas distintas **o** el depósito se quema / se paga a un conjunto que el depositante no controla. Si no, contraejemplo 1 (demanda Sybil).

**Posted price y verdad en costos.** Si \(r\) es **exógeno** al nodo (no depende de su llegada), aceptar iff \(r \ge c_i\) es débilmente dominante. Eso es lo único IC barato que tienen. En cuanto \(r_{t+1}=f(\text{participación}_t)\) y el nodo es grande, puede inflar o vaciar colas para mover \(r\). Entonces el posted deja de ser IC. Leonardos et al. 2021: un posted dinámico que usa historia es un controlador; hay que alinear al que lo ejecuta.

**Conservación implica: no hay almuerzo de tesorería infinita.** Si un dominio nuevo no tiene inferencia (contraejemplo 9), \(F_d\) es un subsidio con presupuesto y *sunset*. Decirlo.

---

## 6. Cinco objeciones letales (orden de daño)

### P0-1 — Demanda Sybil / wash de \(r\)

El atacante deposita en \(F_d\), publica demanda, sus nodos “trabajan”, cobran \(r\), reciclan. \(r\) “alto” no es disposición a pagar: es un espejo. Bittensor ya vive esto: las emisiones por precio/flujo de subnet se manipulan con wash staking; V440 (julio 2026) pone un *gate* porque el precio plano pagaba slots vacíos. Ustedes no tienen Yuma ni gate. Si \(F_d\) no está en riesgo **neto** para el demandante, Hayek y el posted price mienten al unísono.

**Parche o muerte:** el depositante no cobra en ese dominio; o el trabajo lo acepta un sello que el depositante no controla; o \(r\) se paga desde un pool que el depositante no puede drenar hacia sí. Sin una de las tres, el objeto es teatro.

### P0-2 — Dominio fácil (yacimiento)

Autoselección + \(r\) uniforme ⇒ todos van al \(d\) donde \(\Delta q\) verificado / \(c\) es máximo. En HPO-B lookup, “optimizar” es un `get`. En un dominio de re-entrenamiento real, cuesta GPU. Rosenthal 1973: congestion games tienen NE en puras; Roughgarden–Tardos JACM 2002: el precio de la anarquía en ruteo egoísta con latencias lineales es \(4/3\), y con latencias generales puede ser arbitrario (aunque acotado por el óptimo de *el doble* de tráfico). Aquí el “recurso” es el dominio. Sin saturación de \(r_d\) (decreciente en trabajo aceptado, o cap por \(F_d\), o peso que penaliza el lookup fácil), H2 del anexo 08 se cumple al revés: el yacimiento gana.

`domain.weight` en PoO es un mango. No es una teoría. El teorema doctoral, si existe, es: **bajo qué regla de \(r_d(F_d, \text{aceptados}, \text{dificultad})\) el NE de autoselección no concentra todo el GPU en el dominio de sello más barato.**

### P0-3 — Controlador de \(r\) (sobre todo si es aprendido)

Q4 + Q9. Si \(r\) sube con la cola, el atacante infla la cola (tareas basura, Sybil de demanda, o nodos que se registran y no entregan). Si \(r\) baja con la oferta aceptada, un cártel reduce oferta para subir \(r\). Si un modelo **aprende** \(r\) de las trazas de los nodos, los nodos son el dataset y el adversario. Dutting (ICML 2019) aprende mecanismos *con* restricción de IC y tipos de distribución conocida. Un controlador de recompensas entrenado on-policy sobre los mismos agentes **no** hereda eso.

**Parche:** \(r\) es una regla **publicada, de baja dimensión, no aprendida**: p. ej. \(r = \min(F_d/k_{\text{objetivo}},\; r_{\max})\) × factor de dificultad del sello. El aprendizaje, si entra, es el **scheduler local** del nodo (Q8), no el precio de la red.

### P0-4 — Proxy de “mejora” entre dominios estocásticos

Q10–Q11. \(\Delta q\) en un dominio 1a (lookup) no es \(\Delta q\) en un dominio 3 (holdout). Un incremento de 0.01 en CIFAR-tabulado no es un incremento de 0.01 en un LLM. Normalizar por “dificultad” crea el proxy que el nodo maximiza: dificultad percibida, no trabajo útil. Fang (EuroS&P 2023): la trayectoria de entrenamiento se falsifica; el objeto tiene que ser fitness en holdout **comprometido**. doin-domains WP3-C ya mostró que un ataque lineal puede no componer a k grande — eso es ciencia de *un* dominio, no una unidad de cuenta entre dominios.

**Parche:** no hay un \(\Delta q\) universal. Cada dominio tiene su sello y su \(r_d\) en moneda de depósito, no en “utils de inteligencia”. La tesorería **no** compara manzanas. El “óptimo” de la red es: no gastar \(F_d\) en trabajo que el sello de \(d\) rechaza, y no dejar \(F_d\) ocioso si hay nodos con \(c_i < r_d\). Nada más.

### P0-5 — Encaje en el Doctorado en IA

Posted prices, congestion games, escrow y Sybil son **economía / sistemas**. Un jurado de IA pregunta: *¿dónde aprende el sistema?* Respuestas que sobreviven:

- Qué cuenta como trabajo **de aprendizaje** aceptable (sello de inferencia vs sello de \(\Delta q\); estocasticidad).
- Scheduler local como bandit contextual bajo \(r\) publicado (una política, un regret, no “la red aprende precios”).
- Imposibilidad: ningún controlador de \(r\) que observe a los nodos es IC si ellos pueden fabricar la observación.

Si el 70 % del PDF es el mercado de recompensas y el 30 % es “y también hay un scheduler”, es Ingeniería. Satoshi lo dijo del híbrido. Sigue valiendo.

---

## 7. Estado del arte — tabla de cercanos

No reclamo novedad por hueco de Google. Diferencia = una frase.

| Trabajo | Mecanismo | Privado | Quién fija el pago | Selección de proveedor | Qué se verifica | Diferencia exacta |
|---|---|---|---|---|---|---|
| Posted-price procurement (BKS EC 2012; arXiv:2502.18265) | Precio publicado; aceptar/rechazar | Costo \(c_i\) | Comprador / mecanismo | Autoselección por umbral \(r\ge c\) | El “servicio” según el modelo | Calidad \(q\) de ML **no** es submodular conocida. El sello es el problema. |
| EIP-1559 / DPP (Leonardos et al. 2021) | Posted dinámico de *fee* | Valoración de inclusión | Protocolo | Quien paga el fee entra al bloque | Nada de calidad de ML | Ustedes postean un *premio* por trabajo útil, no un peaje de inclusión. |
| Holmström 1982 | Pago a output de equipo | Esfuerzo | Principal | N/A | Output | Un nodo no es un equipo; el output es \(q\) con sello. El riesgo moral **sí** aplica. |
| Concursos / Tullock / premios | El mejor se lleva el pozo | Costo + calidad | Patrocinador | Todos compiten | Ranking | Posted \(r\) por *unidad aceptada* no es winner-take-all. Mezclarlos sin decirlo es trampa. |
| Congestion / Wardrop / Roughgarden–Tardos 2002 | Autoselección de rutas | Preferencia, a veces nada | N/A (costo de latencia) | Egoísta | N/A | Dominios = rutas. PoA es el teorema de P0-2, no un slogan. |
| Ofelimos (CRYPTO 2022) | PoUW: clientes publican problemas **y recompensas**; mineros trabajan para lotería de bloque | Capacidad | Cliente + emisión de seguridad | Autoselección + lottery | Solución de búsqueda local + SNARG | La recompensa **también** compra seguridad del ledger. SoK 2025/1814: la utilidad **no** entra al presupuesto de seguridad. Si DOIN no usa el trabajo para consenso, no es PoUW; es un mercado de trabajo con sello. |
| Cao et al. 2024 (PoUW optimización) | PoUW genérico; trade-off utilidad vs salvaguarda | Capacidad | Protocolo | Mineros | Solución fácil de verificar | Mismo trade-off si pretenden que PoO *asegure* una chain. Si PoO no es consenso de seguridad, Cao no los salva ni los mata: es otro objeto. |
| SoK PoUW (ePrint 2025/1814) | Taxonomía >50 construcciones | — | — | — | — | Veredicto del campo: PoUW “not as useful as expected”. No vendan DOIN como PoUW resuelto. |
| Bittensor (docs 2026: price-based + gate V440) | Emisión a subnets por precio/flujo; Yuma in-subnet | Stake, calidad percibida | Protocolo (inflación TAO/alpha) | Autoselección de subnet; validadores puntúan | Ranking entre pares, **no** \(q\) tabular | Hay token. Hay pujas de stake, no de trabajo. El ledger no audita el modelo (whitepaper). Ustedes quieren sello de \(q\) y **sin** token. Más cerca en *autoselección de dominio* que en verificación. |
| Gensyn (docs + litepaper) | Submitter paga tarea; Solver/Verifier/Whistleblower | Costo de compute | Submitter | Pool de tareas | Fidelidad de **ejecución** (REE/Verde) | Hay demandante. Hay precio de cómputo. No hay posted \(r\) por \(\Delta q\) de HPO. Verifican que se entrenó, no que mejoró un holdout. |
| Akash | Subasta inversa de CPU/RAM | Costo del provider | Tenant elige puja | Matching de pujas | El contenedor corre | Contraste: **sí** hay pujas. Harvey no las quiere. Úsenlo como “esto no somos”. |
| iPFL / AFL 2024–26 | Pagos / subastas por datos o modelos FL | Datos, costo | Servidor / mercado | Selección de clientes | Contribución al modelo global | Una tarea, no autoselección entre dominios de HPO/inferencia. |
| Bandits / learning in games | Regret de una política | Recompensas | Ambiente | El agente elige brazo | Feedback | El scheduler **local** cabe aquí. El controlador **global** de \(r\) no, si el ambiente son los mismos agentes. |
| Rollups / appchains | Seguridad heredada vs fragmentada | — | — | — | Pruebas / puentes | No es el objeto. Una chain lógica. |

Diferencia residual que **pueden** reclamar, y es estrecha:

> Posted \(r\) por trabajo de ML **aceptado bajo sello de dominio**, con autoselección, depósitos no recapturables y **sin** token; el nodo no puja; \(r\) no lo aprende un modelo.

Eso no está cerrado en Bittensor (token + ranking), ni en Gensyn (ejecución, no \(\Delta q\)), ni en Akash (pujas), ni en posted-price clásico (\(q\) no es ML estocástico). Tampoco es una revolución. Es un recorte.

---

## 8. Las dieciocho preguntas, en corto

1. **¿Posted \(r\) es precio o emisión?** Precio (salario/premio) **si** sale de \(F_d\) de un demandante en riesgo. Emisión si sale de tesorería sin tope. No mezclar los nombres.
2. **Hayek:** §3. Solo con depósitos no recapturables. Aun así, débil.
3. **Demanda financiada:** información económica **solo** si el dinero está en riesgo neto. Si no, Sybil (contraejemplo 1).
4. **Inflar colas:** no usar la longitud de cola cruda en \(r\). Usar trabajo *aceptado* y \(F_d\). Tareas no aceptadas no mueven \(r\). Rate-limit de publicación por identidad con stake.
5. **Yacimiento fácil:** \(r_d\) decreciente en aceptados; cap \(F_d\); sello más caro en dominios lookup; o peso. Sin una regla, P0-2 mata.
6. **Pago chico por evaluación + pago por mejora:** la evaluación válida *es* trabajo (Gensyn lo trata así). Un r_inf pequeño y un r_opt mayor es coherente **si** la evaluación no se puede spamear (el consumidor la pidió, o hay presupuesto por request). Pagar “evaluación” sin demandante de inferencia es pagar ping. Spam.
7. **Varianza del participante sin pagar inútil:** posted \(r\) conocido reduce varianza frente a un concurso winner-take-all. No paguen intentos; paguen aceptados. La varianza que queda es la del sello estocástico (P0-4). Stake devuelto si se acepta = seguro parcial, no salario por fallar.
8. **Scheduler local:** un bandit contextual sobre \(\{d \text{ autorizados}\}\) con reward = \(r_d \cdot \mathbf{1}_{\text{aceptado}} - c\) es **un** paper de IA, no la tesis entera. Si el scheduler también elige pipelines y hiperparámetros, se come a B y son dos tesis. Freeze: elige dominio, no el modelo.
9. **Controlador aprendido:** **no** en esta tesis. Manipulable por construcción (P0-3).
10. **“Óptimo”:** no hay óptimo de red entre dominios incomparables. Óptimo = IR del demandante (\(F_d\) no se va a rechazados) + IR del nodo (\(r \ge c\) en esperanza) + no-concentración total (P0-2). Prohibido: bienestar global en utils de inteligencia.
11. **Normalizar mejora/dificultad/costo:** no normalicen entre dominios. Moneda de depósito por dominio. El proxy inter-dominio es el ataque.
12. **¿Blockchain por dominio?** No. Fragmenta seguridad y \(F_d\).
13. **¿Liquidación común o rollups?** Común, dominios lógicos. Rollups = futura, no objetivo.
14. **¿Sin token cierra la contabilidad?** Sí, con escrow en un activo existente y la identidad de §5. El token propio *abre* un segundo objeto (inflación, gobernanza). No lo reintroduzcan “para cerrar”.
15. **¿Inferencia financia optimización del mismo \(d\)?** Ventaja: \(r_{\text{opt}}\) anclado a demanda real. Riesgo: dominios nuevos sin inferencia mueren (contraejemplo 9); dominios de inferencia barata subsidian HPO inútil. Regla: split explícito de \(F_d\) (p. ej. tope de \(r_{\text{opt}}\) ≤ fracción de ingresos de inf **realizados**). Dominio nuevo = subsidio de tesorería con sunset, no un derecho.
16. **¿IA o ingeniería?** Ingeniería del posted price; IA del sello y, opcional, del scheduler local. Sin recorte, ingeniería.
17. **Contribución mínima en 3 años:** ver §9.
18. **Formal realista / prohibido:** cota o NE de autoselección bajo \(r_d\) decreciente; IC del posted **estático**; región donde el posted **dinámico** deja de ser IC. Prohibido: DSIC universal, Hayek pleno, equilibrio del lazo aprendido, “óptimo de la red”.

---

## 9. Diez contraejemplos (pruebas de fuego)

Cada uno mata una afirmación concreta. Si el PDF no dice cómo muere, el jurado lo usa.

1. **Demanda Sybil.** El atacante deposita, publica, cobra con nodos propios, retira. \(r\) alto, trabajo nulo. **Muerte de Hayek y del posted.** Defensa: separación depositante/cobrador o quema.
2. **Dominio fácil.** Lookup HPO-B vs re-entreno. Todos en lookup. **Muerte de “la red asigna trabajo útil”.** Defensa: \(r_d\) o sello que hace el lookup no rentable, o cap.
3. **\(r\) oscilante.** Controlador tipo cola: \(r\) sube, nodos entran, \(r\) baja, salen. Migración masiva, cobweb. Leonardos 2021. **Muerte del ajuste dinámico ingenuo.** Defensa: \(r\) lento (EMA), o estático por ventana, o solo función de \(F_d\).
4. **Reproducible e inútil.** Un “incremento” de ruido que el sello 1a acepta (está en la tabla) pero no generaliza. **Muerte de “útil = verificable”.** Defensa: el sello del dominio de confirmación es holdout no reutilizado (Dwork et al.; doin-domains D3). No pagar 1a como si fuera ciencia.
5. **Inferencia válida y peor.** El nodo sirve un modelo barato que pasa REE/Verde (bit a bit) y es peor en la métrica del usuario. Gensyn ya separa ejecución de calidad. **Muerte de “pagar inferencia verificada = pagar calidad”.** Defensa: sello de calidad (holdout / prueba contratada), no solo de ejecución.
6. **Incremento estocástico que desaparece al repetir.** Seed lucky. Fang. **Muerte de PoL y de \(\Delta q\) de una sola run.** Defensa: sello con semillas comprometidas o lookup; WP3-C es el precedente de *un* dominio, no la unidad de cuenta.
7. **Cambio de identidad.** Sybil para evadir quema/reputación (Douceur 2002). **Muerte de reputación sin costo de identidad.** Defensa: stake por identidad, no reputación gratis.
8. **Fondo insolvente.** \(r\) publicado > \(F_d\). Nodos trabajan, no hay pago. **Muerte de IR.** Defensa: no publicar trabajo sin cobertura; cola de pago = 0 si \(F_d=0\).
9. **Dominio nuevo sin inferencia.** \(F_d=0\) salvo tesorería. **Muerte del mito “la inf financia la opt”.** Defensa: subsidio explícito con fecha, o el dominio no existe para opt hasta que haya \(F_d\).
10. **Scheduler sobreajustado.** El bandit local se queda en el dominio que pagó ayer; el hardware sesga los brazos. **Muerte de “el nodo aprende a asignarse” como contribución sin regret non-stationary.** Defensa: el scheduler es opcional; la tesis no depende de que aprenda bien. Si es hipótesis, regret contra una regla fija (allowlist round-robin) en un régimen que *cambia*.

---

## 10. Pregunta, objetivos, hipótesis (el recorte)

**Pregunta candidata (una):**

> ¿Bajo qué regla de recompensa *publicada* —financiada con depósitos no recapturables, pagada solo por trabajo de un dominio que pasa un sello predeclarado, sin pujas y con autoselección— un nodo racional aporta GPU a inferencia u optimización aceptada en vez de al dominio de sello más barato, a fabricar demanda o a incrementos que no se reproducen, y qué de eso se puede garantizar cuando la recompensa es estática por ventana frente a cuando se actualiza con \(F_d\) y aceptados?

**Objetivos (3):**

1. Formalizar el posted \(r_{d,\tau}\) con conservación de \(F_d\), separación depositante/cobrador, y sello por clase de oráculo (1a lookup vs holdout). Enunciar IC del posted estático y la región donde el posted dinámico deja de ser IC.
2. Demostrar o refutar una cota de concentración (precio de la anarquía / NE de congestion) bajo \(r_d\) constante vs \(r_d\) decreciente en aceptados, en **dos** dominios: uno 1a y uno de re-ejecución real.
3. Experimento de simulador (CPU): los diez contraejemplos como celdas; un scheduler local *opcional* como ablación, no como columna.

**Hipótesis (3):**

- **H1 · IR y cobertura.** Con \(r\) cubierto por \(F_d\) y sello honesto, la fracción de pagos a trabajo rechazado es 0 (por construcción) y nodos con \(c_i < r\) entran en el dominio publicado. Falla si el simulador muestra trabajo no cubierto o no entrada.
- **H2 · No yacimiento.** Con dos dominios (lookup vs re-entreno) y \(r\) uniforme, ≥80 % del GPU simulado va al lookup. Con \(r_d\) decreciente o sello diferencial predeclarado, esa fracción cae bajo un umbral fijado en piloto. Falla si no cae: la autoselección **no** dirige trabajo útil y hay que reportarlo.
- **H3 · Sybil de demanda.** Sin separación depositante/cobrador, un atacante recicla \(F_d\) y sostiene \(r\) alto. Con separación, \(U_{\text{atacante}}\le 0\) dentro del presupuesto. Falla si la separación no basta (identidades baratas): entonces stake de identidad es P0, no P2.

El scheduler aprendido **no** es H. Es ablación. El controlador aprendido de \(r\) **no existe** en el PDF.

---

## 11. Plan de recorte a tres años

| Año | Qué | Qué no |
|---|---|---|
| 1 | Modelo posted + conservación + dos sellos; proposición IC estático / no-IC dinámico; piloto de H2 en lookup vs un re-entreno barato; preregistro de contraejemplos 1, 2, 6, 8 | Hayek como marco; token; rollups; B; CA/Gao; jueces LLM |
| 2 | Campaña de simulador: H1–H3; adversario de mejor respuesta contra la regla publicada de \(r\); ablación de scheduler local | Controlador aprendido; multidominio DIOS; bake-off Bittensor |
| 3 | Un dominio confirmatorio real (series de tiempo **o** el que ya tengan en doin-domains, no ambos como tesis); escritura; si DOIN no está, el simulador + escrow en un L1 de prueba basta | Mainnet, token, una chain por dominio |

Páginas del PDF de propuesta: **el sello y la autoselección son IA**; el escrow es un párrafo. Si el mercado ocupa más de un tercio, es Ingeniería y el programa los manda a otro lado.

Relación con A y B: **no se fusionan.** A aporta la intuición de “no pagar basura” y hay que **borrar** ofertas y Hayek. B no entra. El código PoO es prototipo de dificultad, no el teorema.

---

## 12. Veredicto final

**DEFENDIBLE TRAS RECORTE.**

No es **DEFENDIBLE COMO TESIS DE IA** en bruto: el corazón que acaban de dibujar es un posted-price con congestion, y eso es economía de mecanismos. El jurado de IA lo huele.

No es **DEFENDIBLE SOLO COMO INGENIERÍA** si el recorte pone el sello de trabajo de *aprendizaje* (qué \(\Delta q\) o qué inferencia se acepta bajo estocasticidad) y una proposición de autoselección (H2) en el centro, y deja \(r\) como regla publicada tonta.

No es **YA CUBIERTO**: Bittensor no sella \(q\); Gensyn no postea \(r\) por mejora; Akash puja; Ofelimos compra seguridad con utilidad y el SoK 2025 dice que esa utilidad no paga el presupuesto de seguridad; posted-price clásico no habla de holdouts.

No es **NO DEFENDIBLE**: el objeto es coherente, el código no lo prohíbe, y tres años bastan si no se tragan un controlador aprendido, Hayek de más y una chain por dominio.

**Condiciones no negociables del recorte:**

1. \(r\) publicado, no pujado, no aprendido por la red.
2. \(F_d\) cubre \(r\); depositante ≠ cobrador (o quema).
3. Un ledger; dominios lógicos; sin token propio.
4. Sello por dominio; nada de \(\Delta q\) universal.
5. Dos dominios en el experimento, no una ontología.
6. Prohibido en el título: *confiable*, *Hayek*, *óptimo*, *red que aprende precios*.

Si no juran (1)–(6), vuelvo a **NO DEFENDIBLE** o a KEEP B (AutoML) y este mecanismo se va a un paper de sistemas. Elegir. No fusionar con A ni con B para no elegir.

No reescribí las propuestas.
