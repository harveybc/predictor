# Auditoría Retsu: propuesta doctoral DOIN

**Fecha:** 2026-09-01  
**Auditor:** Retsu  
**Solicitud:** `REQUEST_FOR_AUDIT_RETSU_PROPUESTA_DOCTORAL_DOIN_2026_09_01.md`  
**Artefactos:** `propuesta_doctoral_doin_borrador.pdf` (4 pp., Letter) y `.html`  
**Método:** lectura completa; contraste con las nueve fronteras del propietario; contraste con código local de `doin-core` / `doin-node`; contraste con la literatura citada y con omisiones que un jurado de IA / teoría de juegos / sistemas distribuidos usaría para destruir la tesis.  
**No se reescribe la propuesta.**

**Veredicto: `REVISE`.**  
No es `REJECT`: el objeto (asignar, validar y remunerar trabajo descentralizado) está en el centro, el enrutamiento no se robó la tesis, Hayek está acotado, no hay token propio, y el tono no es promocional. No es `ACCEPT`: el plan es un programa de laboratorio, no una tesis; la medida de información no es operacional; faltan imposibilidades y sistemas emparentados que un jurado pondrá sobre la mesa a los cinco minutos.

---

## 1. Hallazgos

### CRITICAL-1 — El plan no cabe en un doctorado

- **Dónde:** pp. 3–4, §§8–9 y 12–13.
- **Afirmación:** seis objetivos específicos, cinco hipótesis (asignación, verificación, incentivos, información, consenso), tres familias de dominio (cuadrática/combinatoria, series de tiempo, LLM/agentes), tres infraestructuras de registro, subasta inversa, mercado multiatributo, mecanismos aprendidos, CMI como pago, y campañas adversariales preregistradas.
- **Contraargumento:** eso es un programa de cinco años de un grupo, no la pregunta única que un jurado puede falsar. Si todo es “la contribución”, nada lo es. Un doctorado necesita *un* teorema o *un* protocolo con *un* resultado que pueda fallar.
- **Evidencia:** el propio documento admite, en H5, que el ledger puede no aportar nada en entorno confiable; en §12.2, que los mecanismos aprendidos vienen *después* de baselines. Esas son tesis hijas. La maestría ya cubrió PoO + blockchain + un dominio. Generalizar a inferencia+agentes+multidominio+teoría de mecanismos+teoría de la información+tres ledgers es triple conteo.
- **Corrección mínima:** elegir **una** pregunta madre y degradar el resto a *casos* o *trabajo futuro*. Recomendación: el mecanismo de validación y pago cuando la verdad no es observable (H2+H3+H4 unificados), con un dominio de verdad objetiva como control y *un* dominio estocástico como prueba. Ledger (H5) y subastas aprendidas ([8]) salen del paquete doctoral.

### CRITICAL-2 — ΔIⱼ no es un estimador; es una analogía con notación de Shannon

- **Dónde:** p. 3 §10, fórmula ΔIⱼ = I(Q; Eⱼ | E₋ⱼ).
- **Afirmación:** “ΔIⱼ mide evidencia no redundante” y se “contrastará” como base de pago; Q es “el estado de validez estimado mediante tareas de calibración”.
- **Contraargumento:** la información mutua condicional entre un reporte y un estado latente, estimada a partir de un comité pequeño, no es identificable sin un modelo generativo, muchas tareas i.i.d. y un estimador con sesgo/varianza publicados. Kong y Schoenebeck [4] pagan una *medida de información entre reportes de pares* bajo monotonicidad y un régimen de muchas preguntas; no estiman CMI empírica de 3–5 dictámenes sobre un trabajo único. Si Q se estima con las mismas calibraciones que se usan para pagar, el estimador es parte del juego. Coludir en una señal pública (relabeling) es el equilibrio indeseable clásico de peer prediction, y [4] mismo discute imposibilidad parcial al respecto.
- **Evidencia:** no hay estimador, no hay tamaño muestral, no hay alfabeto de Q, no hay prueba de que ΔIⱼ sea computable en las campañas del §12. La frontera 6 del propietario exigía función *operacional*, no analogía. La fórmula, sin eso, es ornamental.
- **Corrección mínima:** o (a) adoptar un mecanismo de elicitación *existente* (peer prediction / correlated agreement / MIP de Kong–Schoenebeck) con supuestos explícitos y un estimador de scoring, y dejar CMI como interpretación; o (b) definir Q binario sobre un conjunto de calibración de tamaño N predeclarado, el estimador (p. ej. plugin de Miller–Madow o un score propio), y un experimento que pueda *fallar* (H4 con IC, no con “mejor calibrados”). No pagar ΔIⱼ hasta que ese experimento exista.

### HIGH-1 — Imposibilidades y ataques que destruyen la regla, omitidos

- **Dónde:** p. 3 §11 (“los resultados de imposibilidad formarán parte del análisis”) sin nombrarlos; p. 2 §6 cita PoL [6] “y sus limitaciones” sin las refutaciones.
- **Afirmación implícita:** el mecanismo se puede diseñar y luego caracterizar fallos.
- **Contraargumento:** un jurado de teoría de juegos preguntará, en este orden: (1) Myerson–Satterthwaite (eficiencia + IR + balance con valores privados en ambos lados); (2) equilibrios de relabeling / collusion en peer prediction; (3) Sybil barato si la identidad no está anclada fuera del ledger; (4) PoL no es verificable de forma robusta (Zhang et al. 2022; Fang et al. 2023, *“Proof-of-Learning is Currently More Broken Than You Think”*). Mencionar “imposibilidad” sin esas cuatro es una cláusula de escape.
- **Corrección mínima:** un párrafo que liste las imposibilidades *relevantes* y qué propiedad se sacrifica en el mecanismo propuesto. Sustituir “limitaciones” de [6] por citas de spoofing. No afirmar compatibilidad de incentivos, equilibrio ni bienestar — el texto ya evita la palabra “bienestar social” en J; hay que aplicar la misma disciplina a “inducir conductas” del §2.

### HIGH-2 — “Aceptado” no está definido cuando no hay oráculo

- **Dónde:** p. 2 §4: “Aceptado significa… esquema, plazo, reproducibilidad y umbral predeclarado; no significa que una mayoría votó.” p. 2 §6: para LLM/agentes, jueces calibrados que “no son oráculo”.
- **Afirmación:** hay un umbral predeclarado independiente del voto.
- **Contraargumento:** el umbral sobre qué magnitud, calculada por quién, con qué Z. En (a) cuadrática, Z es el oráculo y el voto sobra. En (c) LLM, Z es el juez o el comité. Entonces “no es mayoría” o es vacío o es otro agregador (mediana, peer prediction, umbral sobre q̂). El documento no elige. Sin esa elección, H2 no es falsable: cualquier fallo se recodifica como “no era el protocolo P”.
- **Corrección mínima:** para cada familia de dominio, una sola regla P: métrica, quién la computa, qué se hace si los evaluadores discrepan, y qué se hace si no hay métrica objetiva. Una tabla de tres filas cabe en media columna.

### HIGH-3 — La DOIN actual no es todavía el mercado que se describe

- **Dónde:** p. 1 §3.
- **Afirmación:** “Su implementación actual ya dispone de roles de optimización y evaluación, población compartida, datos sintéticos deterministas, commit-reveal, reputación, quórum y registro de mejoras validadas.”
- **Contraargumento:** eso es *cierto como inventario de módulos* y *falso como descripción del mecanismo T, M=(x,A,π)*. El consenso vigente es prueba de *optimización*: un bloque se emite cuando la suma ponderada de incrementos supera un umbral (`doin-core` `proof_of_optimization.py`, docstring: “block generation is triggered when the weighted sum of performance increments… exceeds a dynamic threshold”). No hay, como objeto central, un consumidor que publica B y τ y una subasta inversa. Presentar el mercado Hayekiano como “generalización de lo que ya corre” oculta que el objeto doctoral *reemplaza* el motor de consenso, no solo lo extiende a inferencia.
- **Evidencia:** `doin_core/consensus/proof_of_optimization.py`; `models/commit_reveal.py`, `reputation.py`, `quorum.py`, `fee_market.py`, `payment_channel.py` existen. El bucle de `doin-node` es optimizador/evaluador/nodo, no un matching de demanda publicada.
- **Corrección mínima:** una frase: “El prototipo actual consensa *mejoras de optimización*; el doctorado propone un mercado de *tareas demandadas* con el prototipo como runtime y banco de pruebas, no como el mecanismo ya resuelto.”

### HIGH-4 — Faltan los sistemas que el jurado pondrá al lado

- **Dónde:** referencias [1]–[11]; no hay sistemas de cómputo/ML descentralizado.
- **Afirmación implícita:** el hueco literario es Hayek + subastas + peer prediction + PoL + Shapley + LLM-as-judge + AgentBench.
- **Contraargumento:** sin Bittensor, Gensyn, Akash, Render, Together, Golem, WekaCoin, Ofelimos, y los mercados de inferencia comerciales, la pregunta “¿por qué no usar X?” no tiene respuesta en cuatro páginas. Un jurado de sistemas distribuidos empezará ahí. Ofelimos/WekaCoin están en el propio plan de dominios del proponente.
- **Corrección mínima:** 4–6 citas de sistemas y una columna “qué no resuelven (verificación / Sybil / no-token / multidominio)”. No hace falta una tabla de related work completa; hace falta demostrar que se conoce el terreno.

### HIGH-5 — El pooling de inferencia se nombra y se abandona

- **Dónde:** p. 1 §1 (“pool de inferencia”); p. 2 §4 (“una réplica o una parte de un pool”); ausente de T, de M, de H1–H5 y de métricas.
- **Afirmación:** varios participantes colaboran en un pool.
- **Contraargumento:** el propietario pidió pooling descentralizado de inferencia como parte de la idea. Un pool no es una subtarea de “la arquitectura queda en el oferente”. Tiene asignación (quién replica qué), agregación (quién combina), y pago (cómo se parte B). Sin eso, la tesis es un mercado de *jobs* atómicos, que es más estrecho y más fácil, pero no es lo que se anunció.
- **Corrección mínima:** o se formaliza el pool como tipo de T (métrica de agregación en P) o se declara explícitamente fuera de alcance.

### HIGH-6 — Datos sintéticos: el ataque al generador está nombrado, el experimento que puede fallar no

- **Dónde:** p. 2 §6; p. 3 §11.
- **Afirmación:** se medirán fidelidad, memorización y poder de discriminación; existe el adversario que sobreajusta al generador.
- **Contraargumento:** si el generador y su familia son conocidos, el trabajador simula los retos. El commit-reveal de la semilla solo oculta *una* realización, no la familia. Eso es exactamente la vulnerabilidad de holdout reutilizable (Dwork et al., Thresholdout) y el resultado del frente de dominios del propio proponente: un ataque tiene que *componer* y un nulo necesita potencia. El texto no cita Thresholdout ni define un criterio de “el sintético discrimina” que pueda ser refutado.
- **Corrección mínima:** citar Thresholdout / ADA; predeclarar que el generador se trata como oráculo de clase 3 (presupuesto de consultas) o aceptar que solo prueba no-memorización, no realismo.

### HIGH-7 — H1 es casi tautológica; H4 no es observable tal como está

- **Dónde:** p. 3 §9.
- **Afirmación H1:** la adjudicación por ofertas reduce costo por resultado aceptado frente a reparto fijo o aleatorio, con q_d no inferior.
- **Contraargumento:** si se selecciona por precio y se condiciona a q_d ≥ umbral, el costo por aceptado baja *por construcción* salvo que el umbral sea inobservable (HIGH-2). Falta el espacio de tipos (costo privado, calidad privada, correlación). Sin eso H1 no distingue un resultado científico de un if-statement.
- **Afirmación H4:** remunerar contribución informativa marginal produce comités menos redundantes y mejor calibrados, sin aumentar falsos rechazos.
- **Contraargumento:** “mejor calibrados” no está operacionalizado en la hipótesis (sí aparece luego en métricas). Redundancia de comités pequeños no se estima bien. Ver CRITICAL-2.
- **Corrección mínima:** H1 con tipos y con un margen predeclarado *y* un caso donde el precio barato es basura. H4 con estimador, N de calibración e IC; si no, se fusiona con H2.

### MEDIUM-1 — [2] y [7] no sostienen las frases a las que se pegan

- **Dónde:** p. 1 §2 cita [2] para “el diseño de mecanismos es el problema inverso de la teoría de juegos”; p. 3 §10 cita [7] Data Shapley junto a peer prediction como base de pago a evaluadores.
- **Contraargumento:** Myerson 1981 es subasta óptima con valores independientes y virtual values regulares, no la definición de Hurwicz. Data Shapley valora *datos de entrenamiento*, es #P-hard en general, y no es un scoring rule para dictámenes. Kong–Schoenebeck [4] sí es el ancla de “pagar información”; Shapley no.
- **Corrección mínima:** [2] → Hurwicz / Nisan et al. *Algorithmic Game Theory* cap. de mechanism design, y dejar Myerson para subastas. Quitar [7] del pago a evaluadores o restringirlo a valoración de *contribuciones de datos* en el pool, si el pool se formaliza.

### MEDIUM-2 — [10] AgentBench no es un protocolo de verificación de mercado

- **Dónde:** p. 4 §12.4, “pruebas ejecutables y evaluación abierta [10]”.
- **Contraargumento:** AgentBench es un *benchmark* de LLMs-como-agentes. No define commit-reveal, no paga evaluadores, no resiste colusión. Usarlo como si fuera el P de agentes infla la trazabilidad literatura → método.
- **Corrección mínima:** citar AgentBench como *batería de tareas*, y el protocolo P por separado.

### MEDIUM-3 — Mecanismos aprendidos [8] no caben en este doctorado

- **Dónde:** p. 4 §12.2.
- **Afirmación:** se considerarán después de baselines analíticos.
- **Contraargumento:** Dütting et al. es otra tesis. Incluirlos en el plan experimental de cuatro páginas invita al jurado a preguntar por redes que no se van a entrenar con rigor.
- **Corrección mínima:** una línea de trabajo futuro, no un paso numerado del doctorado.

### MEDIUM-4 — J(T,x) usa ofertas como si fueran costos observables, y V_T es estratégico

- **Dónde:** p. 2 §4.
- **Afirmación:** J es objetivo observable; no se llamará bienestar social.
- **Contraargumento:** la cautela es correcta y se agradece. Queda el problema: el consumidor que declara V_T tiene incentivo a mentir si eso cambia x o π; restar Σ bᵢ selecciona barato. J es como mucho excedente *del contrato según lo declarado*, no un criterio de optimalidad. Está casi bien escrito; falta decir que J no se usará como prueba de eficiencia.
- **Corrección mínima:** una frase: “J no es un estimador de eficiencia asignativa.”

### MEDIUM-5 — Commit-reveal, Sybil y “semillas posteriores” sin cita ni supuesto

- **Dónde:** p. 2 §§5–6; p. 3 §11.
- **Contraargumento:** commit-reveal no impide colusión *antes* del commit ni un evaluador que copia tras la revelación. Semillas derivadas de compromisos son predecibles si el generador es público (HIGH-6). Sybil no se resuelve con reputación on-chain (Douceur 2002). El texto *nombra* los ataques; no ancla ninguno.
- **Corrección mínima:** tres citas (commit-reveal clásico; Douceur; un paper de collusion-proof peer prediction, p. ej. Shnayder et al. Correlated Agreement o el propio [4] sobre relabeling).

### MEDIUM-6 — Presentación: cuatro páginas a 8,45 pt no se leen; el resumen sí se sostiene solo

- **Dónde:** todo el PDF; CSS `font-size: 8.45pt; line-height: 1.24`.
- **Contraargumento:** la jerarquía (regla teal, pregunta en recuadro, tesis de trabajo, J y ΔI) es clara. El resumen y la pregunta **sí** se entienden sin abrir repositorios — cumple H de la solicitud. La densidad no: un jurado imprimirá y abandonará el §12. Hay repetición controlada (precio ≠ verdad aparece al menos cuatro veces); es disciplina, no relleno, pero come espacio que debería ir a imposibilidades y sistemas.
- **Corrección mínima:** 6–8 páginas o recortar objetivos. No bajar más el cuerpo.

### LOW-1 — Identidad del autor ausente en la portada (solo [5]).

### LOW-2 — Pie de página inconsistente: “Borrador para discusión” (p. 1) vs “Propuesta doctoral preliminar” (p. 2).

### LOW-3 — Notación bᵢ = (pᵢ, rᵢ, tᵢ, sᵢ) y luego bᵢ^precio en J: pᵢ ya era el precio.

---

## 2. Respuestas a las preguntas obligatorias

| Pregunta | Juicio |
|---|---|
| **A. Centro en distribución, validación, remuneración** | Sí. El recuadro de pregunta lo fija. |
| **A. ¿El enrutamiento se volvió la tesis?** | No. §4 lo declara estrategia local del oferente. Cumple la frontera 3. |
| **A. ¿Pooling e inferencia compartida?** | Nombrados, no modelados (HIGH-5). |
| **B. ¿Resultados refutables o una plataforma?** | Las H1–H5 *quieren* ser refutables; el paquete conjunto no lo es (CRITICAL-1). H4 no es observable aún (CRITICAL-2). |
| **B. ¿Novedad vs maestría?** | Delimitable: de PoO-consenso a mercado de tareas + inferencia + pago a evaluadores. Hoy se lee como “añadir todo”. Ver HIGH-3. |
| **B. ¿Defendible ante tres jurados?** | Ante IA, falta P para LLM. Ante teoría de juegos, faltan imposibilidades. Ante sistemas, faltan Bittensor/Gensyn/Akash. |
| **C. ¿Actores, información, pagos?** | Esbozados (T, bᵢ, M, ciclo de 6 pasos). Utilidades individuales prometidas, no escritas. |
| **C. ¿Precio = costo, acuerdo = verdad?** | El texto *niega* ambas identificaciones, de forma reiterada y correcta. |
| **C. ¿Se promete equilibrio/bienestar?** | No se usa “bienestar social”. §2 habla de “equilibrios que hagan compatibles” sin supuestos — borde de HIGH-1. |
| **C. ¿Ataques omitidos que destruyen?** | Sí: M-S, relabeling, PoL roto, generador público, Sybil barato. |
| **D. ¿Hayek preciso?** | Sí: conocimiento económico disperso, no verdad. Frontera 4 cumplida. |
| **D. ¿CMI estimable con los experimentos?** | No, no con lo escrito (CRITICAL-2). |
| **D. ¿Pago a evidencia no redundante sin identificar información, verdad y valor?** | La *intención* está; la *medida* no. |
| **E. ¿Sintéticos reducen sobreajuste o lo mueven?** | El texto admite que lo pueden mover; no hay experimento de fallo. HIGH-6. |
| **E. ¿Commit-reveal + semillas + comités bastan contra colusión?** | No, y el texto casi lo dice (“no se asumirán como prueba de honestidad”) sin bajar a un mecanismo concreto. |
| **E. ¿Se distinguen los fallos?** | Sí en la lista de métricas (aceptación falsa, rechazo falso, copia, procedencia). Bien. |
| **F. ¿Cada uso de chain justificado?** | H5 es la pregunta correcta. Aún no hay respuesta, y eso está bien *si* H5 sobrevive el recorte de alcance. |
| **F. ¿Comparación central / permisionada / pública justa?** | El diseño de ablación es justo. |
| **F. ¿Sin token, liquidación coherente?** | Sí en intención; `payment_channel.py` y `fee_market.py` existen. Falta decir con qué activo se liquida en las campañas. |
| **G. ¿Cabe en un doctorado?** | No sin recorte (CRITICAL-1). |
| **G. ¿Métricas operacionales?** | La lista de §12 es de las mejores partes. “Calidad demostrada” se evitó. “Riesgo” no se usa como métrica. |
| **G. ¿IEEE [n] en cada afirmación de literatura?** | Las que *tienen* cita están bien formadas. Varias afirmaciones técnicas no tienen [n] (MEDIUM-5). |
| **G. ¿Faltan antecedentes?** | Sí (HIGH-1, HIGH-4, MEDIUM-1). |
| **H. ¿Jerarquía y lectura?** | Jerarquía sí; densidad no (MEDIUM-6). |
| **H. ¿Resumen autónomo?** | Sí. |
| **H. ¿Lenguaje promocional?** | No. Es el mayor mérito estilístico del borrador. |

---

## 3. Trazabilidad de afirmaciones sobre DOIN

| Afirmación en la propuesta | ¿Sostenida? | Fuente |
|---|---|---|
| Roles de optimización y evaluación | Sí | `doin-node` runtime unificado; `doin-core` modelos de optimae / quorum |
| Población compartida | Sí, en el runtime de optimización | `doin-node` `unified.py` (citado en la solicitud) |
| Datos sintéticos deterministas | Sí como módulo | `doin-core` `deterministic_seed.py`; `synthetic-datagen` (repo hermano) |
| Commit-reveal | Sí | `doin-core` `models/commit_reveal.py`, `protocol/messages.py` |
| Reputación | Sí, reconstruible del ledger | `models/reputation.py` |
| Quórum | Sí, incluso dinámico | `models/quorum.py`, `consensus/dynamic_quorum.py` |
| Registro de mejoras validadas / blockchain | Sí | `doin-node/blockchain`, `proof_of_optimization.py` |
| Mercado de tarifas y canales de pago | Código existe; no es el objeto de consenso | `models/fee_market.py`, `models/payment_channel.py` |
| El sistema *ya* asigna tareas de demanda con presupuesto B y protocolo P | **No** | El consenso dispara bloques por incrementos de desempeño, no por matching de T |
| Multidominio simultáneo como contribución *ya lograda* | Parcial | plugins y `doin-domains` existen; no hay mercado único de inferencia+optimización medido |
| Sin token propio | Coherente con el diseño declarado; `models/coin.py` existe — hay que no resucitarlo en la tesis | `doin-core` `models/coin.py` |
| doin-optimizer / doin-evaluator como arquitectura vigente | No citados. Correcto. | — |

La tesis de maestría [5] coincide en título y origen con `doin-core/docs/THESIS.md` (Javeriana Cali, 2018, PoO + blockchain + islas, evaluación en trading FX). La generalización doctoral está bien *anunciada* y mal *acotada*.

---

## 4. Citas: faltantes, incorrectas, o que no respaldan la frase

| Ítem | Problema |
|---|---|
| [2] Myerson 1981 | No respalda “mecanismo = inverso de la teoría de juegos”. Usar Hurwicz / Maskin / Nisan. Conservar Myerson para subastas. |
| [4] Kong–Schoenebeck 2019 | Cita correcta *si* se usa como scoring entre pares, no como CMI empírica de un trabajo. La fórmula ΔIⱼ no es la de ese paper. |
| [6] Jia et al. PoL | Incompleta sin Zhang et al., *Adversarial Examples for Proof-of-Learning*, IEEE S&P 2022; Fang et al., *Proof-of-Learning is Currently More Broken Than You Think*, EuroS&P 2023. |
| [7] Data Shapley | No respalda el pago a evaluadores. |
| [10] AgentBench | Benchmark, no protocolo de mercado. |
| [8] Dütting et al. | Válida como related work; no como paso del plan (MEDIUM-3). |
| [1], [3], [5], [9], [11] | Formalmente correctas y bien usadas. |
| Ausentes esenciales | Hurwicz; Myerson–Satterthwaite (1983); Douceur, Sybil (2002); Dwork et al. Thresholdout / reusable holdout (2015); Shnayder et al. Correlated Agreement (2016) o Prelec BTS (2004); Nisan et al. AGT; un sistema (Bittensor / Gensyn / Akash); Ofelimos o WekaCoin si se habla de “PoO vs hash”. |
| Afirmaciones sin [n] | commit-reveal; semillas posteriores; Sybil; “resultados de imposibilidad”; generadores sintéticos como retos. |

---

## 5. Lo que el documento hace bien (para no “encontrar sangre” donde no hay)

1. El centro es el mecanismo, no el router. Cumple al propietario.
2. Precio ≠ verdad; acuerdo ≠ validez; ledger ≠ oráculo; sin token. Reiterado y correcto.
3. J no se llama bienestar social.
4. Lista de adversarios ejecutable, con métrica por amenaza — esto es nivel de tesis, no de whitepaper.
5. Ablación de confianza (central / permisionado / público) es la pregunta F correcta.
6. Los dominios comparten interfaz económica, no métrica: eso es exactamente Contrato-Primero, y está bien dicho en una frase.
7. El castellino está sobrio. Casi no hay marketing.

Eso es acero. El fallo no es de lealtad a la idea; es de recorte y de operacionalización.

---

## 6. Condiciones objetivas para pasar de REVISE a ACCEPT

1. **Una pregunta madre** en el recuadro; máximo tres hipótesis; los objetivos 5–6 y el paso “mecanismos aprendidos” fuera o en futuro.
2. **P explícito** para dominio con oráculo y dominio sin oráculo (tabla de tres líneas).
3. **ΔIⱼ o se vuelve scoring rule citado, o se cae de la página 3.** Si se queda, estimador + N + experimento que pueda fallar.
4. **Párrafo de imposibilidades** (M-S, relabeling, PoL roto, Sybil) y qué se sacrifica.
5. **Párrafo de sistemas emparentados** (mínimo cuatro nombres).
6. **Una frase que separe** el PoO-consenso actual del mercado de tareas propuesto.
7. **Pool de inferencia:** se formaliza o se declara fuera.
8. **Cuerpo ≥ 9,5 pt** o se recorta texto; no se añade alcance.

Sin (1)–(3) no recomiendo enviarla al comité de la Universidad de La Sabana.
