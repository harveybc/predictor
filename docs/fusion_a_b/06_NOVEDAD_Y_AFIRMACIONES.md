# 06 — Novedad y afirmaciones

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Veredicto del **objeto recortado** (no de A, no de B cruda): **PLAUSIBLE**.

STRONG exigiría un teorema que hoy no tienen y un hueco que OptFormer no cubre por construcción. ALREADY DONE sería mentir. WEAK es lo que obtienen si el PDF sigue diciendo *confiable* y “red que aprende”.

---

## Quince cercanos y la diferencia que **sí** pueden decir

| # | Trabajo | Diferencia exacta, sin adjetivos |
|---|---|---|
| 1 | OptFormer (Chen et al., NeurIPS 2022) | Aprende política y respuesta en espacios distintos, **sin** agentes ni \(\delta\) estratégica. |
| 2 | HyperBO (Wang et al., JMLR 2024) | Prior GP; **mide** transferencia negativa; no se abstiene como **salida del método**; no hay pago. |
| 3 | PFNs4BO (Müller et al., ICML 2023) | Surrogate in-context. Cero mecanismo. |
| 4 | HPO-B (Pineda Arango et al., 2021) | Banco y protocolo. No método. |
| 5 | Fan–Han–Wang TMLR / HyperBO+ | Espacios heterogéneos ya cubiertos. No censura estratégica. |
| 6 | Dutting et al., ICML 2019 | Aprende el **mecanismo** con tipos de distribución conocida. No HPO. |
| 7 | Deng et al., ICML 2025 | Procurement IC con **calidad conocida**. \(\hat v\) aprendido rompe el supuesto. |
| 8 | Chen–Clinton–Kandasamy, AISTATS 2026 | Mercado de datos; **no DSIC**; NE de dos menores costos. Análogo si \(\hat v\) se paga. |
| 9 | Bittensor (whitepaper) | Ranking de *inteligencia de pares*. No consumidor de HPO. Ledger no audita pesos. |
| 10 | Gensyn (docs / litepaper) | Verifica **ejecución** (REE/Verde). Delphi = mercados de eventos, no de configs. |
| 11 | Akash | Subasta inversa de CPU/RAM. Calidad = el contenedor corre. |
| 12 | Shnayder et al., EC 2016 (CA) | Peer prediction. **Fuera del objeto** recortado. No citar como contribución. |
| 13 | Gao et al., AI 2019 | Peer prediction con GT limitada. **Fuera**. Reabrir es A. |
| 14 | Holmström 1982 | Equipos, esfuerzo oculto. Relevante solo si \(K>1\). No lo prometan. |
| 15 | iPFL / AFL 2024–26 | Incentivos para **datos o modelos** de una tarea FL. No trazas de búsqueda entre tareas. |

doin-domains D1/WP3: infraestructura de oráculo y holdout. **No** es SOTA de meta-HPO. No se cite como si el lookup “demostrara transferencia”.

OpenReview `Bx4Sz-N5K3J`: **no leído** (Cloudflare 403). No inventar.

---

## Afirmaciones permitidas

- El ciclo completo valor-aprendido → oferta → pago → nueva traza **no** está cerrado en Bittensor, Gensyn, Akash ni FL.
- OptFormer/HyperBO/PFNs4BO no modelan un generador estratégico del corpus.
- Deng exige \(q\) conocido; no cubre \(\hat v\) manipulable.
- Un \(\hat v\) **público, congelado y no remunerado** puede ordenar un menú (**solo** si alguien resucita el híbrido; hoy está en REJECT).
- El flujo de DOIN es emisión sin tasación del operador; PoO ajusta umbral como dificultad ([08](08_FLUJO_MINERO_NO_MERCADO.md)).
- La abstención es parte del método (NFL). Un resultado negativo es tesis.
- Sin sello de re-ejecución/lookup, procedencia puede ser no identificable (imposibilidad).
- DOIN es runtime. El ledger no decide \(q\).

## Afirmaciones prohibidas

- *Confiable*, *trustworthy*, *seguro* como propiedad del método.
- “Primer mercado de inteligencia / de optimización”.
- “Incentivo óptimo”, DSIC, equilibrio del lazo.
- “Siempre reduce el costo”.
- PoO / trayectoria = prueba de aprendizaje (Fang EuroS&P 2023; SOTA D5 de domains).
- 2 bits/peso ⇒ capacidad de redes profundas.
- HPO-B en doin-domains = meta-HPO transferible.
- zkLLM / TOPLOC resuelven calidad de una configuración.
- Comparaciones **ejecutadas** contra Bittensor/Gensyn/Akash.
- Que A y B se “reparan” mutuamente. A amputada no es A.
- Hayek / “el precio transmite conocimiento” en un sistema **sin precio**.
- Que el operador de DOIN “oferta” o “fija tarifa”. No lo hace.

---

## Qué es IA / economía / cripto / infra

| Capa | En el PDF recortado | % de páginas (techo) |
|---|---|---|
| IA | \(\hat v\), abstención, calibración, HPO-B, series de tiempo | ≥70 |
| Economía | \((p,\pi)\to\delta\), no-rentabilidad del generador | ≤25 |
| Cripto | commit-reveal como **sello**, una mención | ≤1 párrafo |
| Infra | DOIN, fees, nodos | viabilidad, no contribución |

Si el borrador invierte esa tabla, no es este doctorado.
