# 07 — Instrucciones para Musashi

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

No reescribas el PDF hasta que Harvey marque **M** o **H2** en el principal §11. H1 y H0 están en REJECT.

Este archivo es un checklist, no una licencia para inflar páginas.

---

## Si Harvey elige M (emisión, no mercado) — **opción vigente con el flujo real**

Leer antes [08](08_FLUJO_MINERO_NO_MERCADO.md). No reutilizar el HTML de B ni el de A como base: ambos describen otro sistema.

- [ ] Título: incentivos / emisión / incremento verificado. **Cero** *confiable*, *mercado*, *metaoptimización*, *Hayek*.
- [ ] Resumen: nodo entra, dominio auto/elegido, GPU trabaja, **sin precio**; se paga \(\Delta q\) que sobrevive sello; el umbral PoO es dificultad. Analogía: minero, no tienda.
- [ ] Pregunta de [08 §4](08_FLUJO_MINERO_NO_MERCADO.md). H1–H3 de [08 §6](08_FLUJO_MINERO_NO_MERCADO.md).
- [ ] Teoría: Holmström (pago a output), PoUW (Ofelimos, Cao, SoK), Fang (no PoL), autoselección/congestion de dominios. **No** Myerson, **no** Deng-as-thesis, **no** CA/Gao como contribución.
- [ ] Instrumento: emisión de protocolo + `domain.weight` + stake/quema. Simulador. Código como prototipo, no como prueba.
- [ ] Dominio control: HPO-B/NATS lookup. Confirmatorio: **uno** barato de re-entreno. Sin LLM-judge.
- [ ] Borrar: ofertas, consumidor \(T\), Hayek, OptFormer-como-columna, MacKay, “red que aprende a asignar precios”.
- [ ] Encaje IA: una frase — trabajo útil de *aprendizaje*, no hash; la maestría propuso PoO, esto ataca basura y yacimiento fácil.
- [ ] 4–5 páginas. Si sale un mercado, te equivocaste de opción.

---

## Si Harvey elige H1 (híbrido recortado) — **REJECT, no ejecutar**

No uses las casillas de abajo. Quedan como archivo muerto del debate con Satoshi. El flujo real no tiene ofertas.

### Título y resumen

- [ ] Borrar *confiable* de `<title>`, `<h1>`, footer, objetivo general.
- [ ] Título de la lista en [01](01_LENGUAJE_ATACABLE.md).
- [ ] Resumen: problema de transferencia + abstención + corpus manipulable. **Cero** DOIN, **cero** blockchain, **cero** “red descentralizada”.
- [ ] Una pasada, dicha en el resumen.

### Cuerpo

- [ ] Pegar pregunta / objetivos / H1–H3 de [03](03_PREGUNTA_OBJETIVOS_HIPOTESIS.md).
- [ ] § marco: OptFormer, HyperBO, PFNs4BO, HPO-B, Fan TMLR, Deng, Chen 2026, Bittensor/Gensyn/Akash **en una tabla de diferencias**. No un tour.
- [ ] MacKay/Cover: **un** párrafo de ablación o cero. No §2.1 de dos páginas.
- [ ] Kolmogorov y Shannon: fuera del cuerpo.
- [ ] Asignador = defensa A de [04](04_MODELO_Y_ASIGNADOR.md). Pago \(\neq \hat v\). Escrito con fórmula.
- [ ] Mecanismo = generador de corpus + re-ejecución/lookup. **No** CA, **no** jueces, **no** Gao como contribución.
- [ ] Factorial = [05](05_EXPERIMENTO_Y_ATRIBUCION.md), no 2×2×3.
- [ ] Una familia confirmatoria. Cortar LLM compactos.
- [ ] DOIN: viabilidad. “Si se retrasa, el simulador basta.”
- [ ] Prohibiciones de [06](06_NOVEDAD_Y_AFIRMACIONES.md) respetadas.

### Teoría

- [ ] Tres cajas **separadas**: (i) cota de arrepentimiento \((\alpha,s,\delta)\); (ii) indistinguibilidad si procedencia no identifica; (iii) existencia de \((p,\pi)\) presupuesto-factibles que hacen \(U_G\le 0\) **cuando el pago no depende de \(\hat v\)**.
- [ ] No “equilibrio del sistema”.
- [ ] No “en diálogo con Gao” si Gao no es el objeto. Si se cita, es *related work* de lo que **no** se hace.

### Lo que no copies de A

Texto de LLM, rúbricas, \(n_{\text{audit}}\) de expertos, Hayek en el lead, cinco brazos A–E, H1 de desviación de evaluadores. Eso es otro doctorado.

### Lo que no copies de B

Footer *Metaoptimización confiable*. “Salida segura”. “Trazas verificables” sueltas. Reproducción de MacKay como producto del año 1. Series de tiempo **y** LLM.

---

## Si Harvey elige H2 (KEEP B)

- [ ] Mismos cortes de lenguaje.
- [ ] Corrupción **inyectada** (relabel, censura). Sin generador pagado.
- [ ] Sin capítulo de mecanismo. Una viñeta de trabajo futuro.
- [ ] Abstención + procedencia + HPO-B + una familia. Eso es la tesis.
- [ ] Novedad: más débil. Decirlo. No compensar con DOIN.

---

## Si Harvey elige H0 (KEEP A)

- [ ] `git checkout -- docs/propuesta_doctoral_doin_borrador.html` no basta: el working tree es B; A está en `7ae8643`. Restaurar ese blob.
- [ ] Parche de una frase: el teorema es trabajador–compromiso–retos, no Gao con otro nombre (post-triple de Retsu).
- [ ] No mezclar un párrafo de meta-HPO “para enriquecer”. Eso reabre este dictamen.

Hoy el PDF abierto es B. Volver a A **después** de haber mostrado B al círculo interno es ruido. Solo si el programa (no el candidato) es *evaluación de LLM*.

---

## Criterio de “el mercado se comió la tesis”

Si Harvey marcó **M** y el PDF vuelve a hablar de ofertas, Hayek o eBay: **para**. Relee [08](08_FLUJO_MINERO_NO_MERCADO.md).

Si Harvey marcó **H2** y mecanismo + DOIN + ledger > 30 % del cuerpo: **para** y recorta.

---

## Qué no hagas

- No inventes una subasta en `doin-core` para que el PDF deje de mentir. El experimento es un simulador.
- No “resuelvas” P0-1 con *proper scoring* prometido.
- No reintroduzcas *confiable* como traducción de *calibrated*.
- No cites mi auditoría de novedad ni este dictamen en el PDF. El comité no es esta manada.

Cuando Harvey marque opción, reescribes **un** HTML. Un PDF. Sin ramas de prosa.
