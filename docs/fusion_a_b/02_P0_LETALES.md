# 02 — P0 letales

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Cinco golpes. El orden es el de daño en sala, no el de Satoshi.

---

## P0-1 — El asignador es el teorema, no una objeción

**Hecho.** B elige la siguiente evaluación por “ganancia esperada de información por unidad de costo” (Entropy Search / costo, §2.2 y §3.2). Satoshi nombra que un proveedor racional **optimiza ese proxy**: trazas que *parecen* informativas (alta reducción prevista de incertidumbre) sin mejorar \(q\).

**Por qué Satoshi fue madre.** Lo listó como objeción 3 de 5. Un teórico de mecanismos lo pone primero. Dutting (ICML 2019) aprende subastas **con restricción de IC**. Deng (ICML 2025) da IC cuando la calidad es **conocida por el subastador**. Chen (AISTATS 2026) prueba que en un mercado de reportes de datos **no hay DSIC no trivial**. Ustedes están más cerca de Chen que de Deng en cuanto \(\hat v\) (el proxy) influye en quién cobra.

**Ataque en una frase:** “Usted no paga la información mutua, cierto. La **asigna** con ella. El vendedor no necesita que el pago sea \(\hat v\); le basta con que \(\hat v\) decida si entra al presupuesto.”

**Parche único (elegir uno, escribirlo, no los tres):**

| Defensa | Qué afirma | Qué exige el PDF |
|---|---|---|
| **A — menú fijo** | \(\hat v\) ordena configuraciones **publicadas por el consumidor**. Trabajadores pujan **costo**. Cobran iff re-ejecución ≥ \(\theta\). | El trabajador **no** propone \(x\). Fabricar trazas pasadas no le paga en la ronda. El corpus de \(\hat v\) es HPO-B congelado o un generador **separado**. |
| **B — freeze del entrenador** | Quien emite trazas de entrenamiento de \(\hat v\) **no** es quien cobra la ronda. | Dos poblaciones. Declarado. Si se mezclan, P0-1 vuelve. |
| **C — proper scoring** | Demostrar que reportar el surrogate de información es *proper* para el vendedor. | Nadie en esta mesa lo va a demostrar en el año 1. No lo prometan. |

Si no eligen A o B en el PDF, el híbrido es **REJECT** por P0-1, aunque el resto esté limpio.

Detalle formal: [04_MODELO_Y_ASIGNADOR.md](04_MODELO_Y_ASIGNADOR.md).

---

## P0-2 — Lenguaje: *confiable* y el resto

**Hecho.** B usa *confiable* en título, footer y objetivo general. No la define. Ver [01_LENGUAJE_ATACABLE.md](01_LENGUAJE_ATACABLE.md).

**Por qué Satoshi fue madre.** Cero líneas. Un jurado de ética o de sistemas no lee OptFormer: lee el `<h1>`.

**Parche:** retitular antes de cualquier otra reescritura. Si el título sigue diciendo *confiable*, el resto del paquete es cosmética.

---

## P0-3 — Una pasada vs. “red que aprende”

**Hecho.** El lazo traza → \(\hat v\) → asignación → respuesta → nueva traza es un sistema no estacionario. Satoshi lo manda a futuro y **conserva** una pregunta que suena a lazo.

**Ataque:** “Si evalúa una pasada, el título no puede hablar de red que aprende ni de ciclo. Si evalúa el ciclo, no tiene equilibrio. ¿Cuál de las dos mentiras prefiere?”

**Parche:** la pregunta madre **incluye** “análisis de una pasada” o un equivalente inequívoco (“corpus congelado → política → ronda única”). El lazo no aparece ni en resumen ni en objetivos. Trabajo futuro = una viñeta, no un gancho.

Con HPO-B público, además, la “endogeneidad” del **corpus de entrenamiento** es simulada. Hay que decirlo. Si no, están vendiendo un fenómeno de red que el diseño no produce. Ver §7 del [principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md).

---

## P0-4 — Atribución y factorial impagable

**Hecho.** Si baja el costo, el jurado pregunta: ¿\(\hat v\), abstención, auditoría o pago? Satoshi pide 2×2×3. Correcto como deseo. Letal como plan.

Celdas: transferencia (2) × procedencia (2) × corpus (3) = 12, más on/off de mecanismo anidado en la celda endógena, más 2 controles, más mejor respuesta, × ≥30 tareas × 5 semillas. Eso no es un doctorado: es un laboratorio de AutoML con presupuesto de empresa.

**Parche:** factorial **2×2** primario:

- {selectivo / BO sin transferencia} × {corpus limpio / \(\delta\) inyectada}

Brazo extra, no eje: generador estratégico del corpus (H2), con mecanismo on/off **solo ahí**.

Procedencia consciente/ciega: **ablación de H3**, no eje primario, y solo si el piloto tiene potencia. Si no, se declara no testeada.

Sin cálculo de potencia en el piloto, H3 de Satoshi (“el acople gana”) se tacha.

---

## P0-5 — Pinza de novedad, ejecutada

**Flanco AutoML (hechos).** OptFormer (NeurIPS 2022) lee metadatos de texto y trayectorias de políticas distintas. HyperBO (JMLR 2024) preentrena GP y **mide transferencia negativa**. PFNs4BO (ICML 2023) hace inferencia in-context. Fan–Han–Wang (TMLR) cubren espacios heterogéneos. HPO-B es el banco. B **cita** 8–13 y luego reclama “historiales de políticas distintas, presupuestos desiguales, censura, calidad variable”. Las tres primeras ya están. Quedan censura + procedencia + abstención. Eso es **estrecho**. Es defendible. No se agranda fusionando A.

**Flanco económico (hechos).** Deng 2025 (calidad conocida). Chen 2026 (no DSIC). Gao 2019 (peer prediction con GT limitada). AFL / iPFL / incentivos FL 2024–26. Bittensor (ranking). Gensyn (verificar cómputo). Akash (compute). A **citaba** Bittensor/Gensyn/Akash; B casi no. Satoshi pide “blindar ambos frentes” y luego recomienda híbrido con **menos** profundidad en cada uno.

**Parche de novedad:** una frase, no un mapa:

> No proponemos un metaoptimizador universal ni un mercado de inteligencia. Proponemos (i) abstención calibrada ante procedencia y censura, (ii) una cota de arrepentimiento en función de la fracción corrupta \(\delta\), y (iii) un pago por re-ejecución que hace no-rentable fabricar el *proxy* de asignación. OptFormer y HyperBO no modelan (iii). Deng no cubre (i) ni un \(\hat v\) aprendido. Chen cubre mercados de datos, no de configuraciones tabuladas.

Si no pueden decir eso sin sudar, no hay tesis híbrida. Hay un survey.

**Prohibido en sala:** “primer mercado”, “mejor que Bittensor”, “DOIN resuelve verificación”, “2 bits/peso”.

---

## P0 que Satoshi mató bien (no reabrir)

- Oráculo parcial, CA, jueces LLM, expertos humanos: **fuera**. Reabrirlos es segunda tesis.
- Subasta como objeto: **fuera**. El código de `doin-core` **no tiene** subasta. Inventarla para la fusión es ficción de software.
- Token, DIOS, bake-off ejecutado contra Bittensor/Gensyn: **fuera**.
- MacKay como producto del año 1: **fuera del marco**; ablación o nada.

Si Musashi reintroduce CA “por si acaso”, rechazo el híbrido otra vez. Gao no es un amuleto. Es un dominio que ustedes **abandonaron**.
