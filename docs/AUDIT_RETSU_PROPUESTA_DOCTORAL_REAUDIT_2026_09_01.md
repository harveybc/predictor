# Reauditoría Retsu: propuesta doctoral DOIN (revisión)

**Fecha:** 2026-09-01  
**Auditor:** Retsu  
**Dictamen previo:** `REVISE` (2026-09-01)  
**Solicitud:** `RETURN_TO_RETSU_PROPUESTA_DOCTORAL_REVISION_2026_09_01.md`  
**Artefactos:** `propuesta_doctoral_doin_borrador.pdf` / `.html` (4 pp. Letter, cuerpo 9,5 pt)

**Veredicto: `ACCEPT`.**  
Las ocho condiciones objetivas del dictamen anterior están cumplidas. El objeto es una tesis, no un programa de laboratorio. Quedan hallazgos residuales **MEDIUM/LOW**: se parchean en una pasada, no reabren arquitectura. No moveré la portería.

---

## 0. Disposición de las condiciones previas

| Condición para ACCEPT | ¿Cumplida? | Dónde |
|---|---|---|
| Una pregunta madre | Sí | §1: validación y pago con verdad parcial; asignación/ledger/routing son entorno |
| Máximo tres hipótesis | Sí | §9 H1–H3, con umbral numérico de refutación |
| P con y sin oráculo | Sí | §5, dos filas, desacuerdo y verdad de referencia |
| Retirar CMI ornamental | Sí | §6 adopta CA [3]; CMI no se estima |
| Imposibilidades | Sí | §11: M–S [9], permutación/colusión CA, Sybil [10], PoL roto [11][12] |
| Sistemas vecinos | Sí | §12: Bittensor, Gensyn, Akash, Golem |
| Separar DOIN actual / mercado | Sí | §3: consensa incrementos; el doctorado cambia el mecanismo |
| Pool formalizado o fuera | Sí | §4: *K,G,w*; ablación *K*=1 vs 3, no segunda tesis |
| Cuerpo ≥ 9,5 pt | Sí | CSS `font-size: 9.5pt` |

---

## 1. Respuestas a las seis preguntas de reauditoría

1. **¿Tesis y no programa?** Sí. Un objetivo general, tres específicos, dos dominios (control objetivo + LLM/agente), tres fases anuales. El año 3 es integración experimental, no una hipótesis de ledger. Riesgo residual: que el año 3 se hinche; el texto ya lo recorta.
2. **¿CA ejecutable sin sobreafirmar?** Sí, con matices MEDIUM. La fórmula es el score de CA (bonus misma tarea menos penalización cruzada). Citan *informed truthfulness* y equilibrios por permutación. No reclaman DSIC ni que acuerdo sea verdad. La auditoría externa ancla la semántica: correcto.
3. **¿FP/FN sin circularidad?** Sí en diseño: la muestra experta oculta es la verdad de referencia; el acuerdo del comité no se llama verdad. Falta el *tamaño* de esa muestra (residual).
4. **¿Pool sin volverse tesis?** Sí. Está en *T*; la ablación está acotada.
5. **¿Diferencia vs DOIN y vs Bittensor/Gensyn/Akash/Golem?** Defendible. La pregunta específica es el pago con verdad parcial, error auditado y bajo esfuerzo. No pretenden reemplazar esos sistemas ni emitir activo.
6. **¿Tipografía y IEEE?** Adecuadas para remitir. 17 referencias, Hurwicz en lugar de Myerson-1981, Thresholdout [6], Douceur, M–S. Densidad aún alta, pero ya no es 8,45 pt.

---

## 2. Hallazgos residuales

### MEDIUM-1 — H2 no está identificada en los cuatro brazos

- **Dónde:** §9 H2 vs §10 brazos A–D.
- **Afirmación:** los retos posteriores al compromiso reducen ≥20% la aceptación contaminada frente a un test público reutilizado.
- **Contraargumento:** A–D son (A) pago por entrega, (B) mayoría+uniforme, (C) auditoría+mayoría, (D) auditoría+CA. Ningún brazo es “test público reutilizado”. H2 queda como factor no cruzado. Un jurado preguntará contra qué celda se estima el 20%.
- **Corrección mínima:** o un brazo E (test público), o declarar H2 como factor 2×2 cruzado con {B,D} y predeclarar el contraste.

### MEDIUM-2 — La muestra de auditoría experta no tiene N ni potencia

- **Dónde:** §5 régimen parcial; H1 y H3 usan “aceptación falsa auditada”.
- **Afirmación:** una muestra aleatoria oculta estima FP y FN.
- **Contraargumento:** *N*≥200 es de tareas/evaluadores, no de auditorías expertas. Con n_audit pequeño, H1 (“sin aumentar FP”) no tiene potencia: es el mismo error DOM-005, ahora en el ancla de verdad. El piloto de potencia del §10 puede cubrirlo, pero hay que decir `n_audit` o la fracción *p*.
- **Corrección mínima:** una cifra (p. ej. fracción 10–20% o n_audit mínimo) y que el piloto también fije ese n, nunca a la baja.

### MEDIUM-3 — Δ de CA: “signo de la covarianza” es exacto solo si la rúbrica es binaria

- **Dónde:** §6.
- **Afirmación:** “Δ_{−f} es el signo de la covarianza empírica”.
- **Contraargumento:** en Shnayder et al. [3], Δ es la *matriz* Sign(P(x,y)−P(x)P(y)) sobre el alfabeto de señales. Covarianza escalar basta para binario. Una rúbrica categórica/ordinal necesita la matriz. El cross-fitting de cinco pliegues es una adición metodológica sana; no está en [3] y no debe leerse como si lo estuviera.
- **Corrección mínima:** “Δ es la matriz Sign(·) de [3], aprendida fuera de pliegue; si la rúbrica se binariza, Δ se reduce al signo de la covarianza.” Promediar S_j sobre los pares de evaluadores (hay ≥3).

### MEDIUM-4 — Shannon [8] no es CA

- **Dónde:** §6, última frase.
- **Afirmación:** “Shannon aporta aquí una medida operacional de dependencia entre señales.”
- **Contraargumento:** CA usa signos de correlación, no I(X;Y) de Shannon. La frontera 6 pedía no usar a Shannon de adorno. Casi lo logran; esta frase lo reintroduce.
- **Corrección mínima:** quitar a Shannon del pago, o decir que la dependencia que se paga es la de [3], no la entropía de [8]. Shannon puede quedarse como límite conceptual (“información ≠ valor”).

### LOW-1 — [11] Zhang et al. es S&P 2022 (arXiv:2108.09454, 2021). Preferir la versión de actas.

### LOW-2 — [13] “Amplifying the Weight-copying Penalty in Bittensor” existe como working paper de Opentensor (jul 2024) y poster CCS 2024; no es un paper de revista. Aceptable en propuesta si se etiqueta working paper (ya dice “technical report”).

### LOW-3 — [15]/[16] documentación viva fechada 2026: anclar URL y fecha de consulta.

### LOW-4 — Interlineado 1,18 a 9,5 pt sigue apretado. Remisible.

---

## 3. Lo que la revisión hizo bien (no se reabre)

- Pregunta madre única; el mercado es banco de pruebas.
- CA en lugar de ΔIⱼ; supuestos y permutación declarados; pago acotado con c₀.
- Tabla P: control con δ y θ; prueba principal con rúbrica + pruebas duras + muestra experta que *no* se llama verdad.
- PoO actual ≠ mercado de tareas, en una frase.
- M–S, Sybil, PoL spoofed, Thresholdout [6], Douceur.
- Bittensor/Gensyn/Akash/Golem con una pregunta que ellos no contestan.
- Pool en T, ablación acotada.
- H1–H3 con 20% / 5 pp / 10%: se pueden fallar.
- Resultado negativo bajo colusión se anuncia como refutación, no como reinterpretación.
- Autor en portada; pie uniforme.

---

## 4. Trazabilidad (solo lo que cambió de sentido)

| Afirmación nueva | ¿Sostenida? |
|---|---|
| DOIN consensa incrementos; el doctorado cambia el mecanismo | Sí, `proof_of_optimization.py` + [17] |
| CA como en [3] | Sí como score mismo-vs-cruzado; ver MEDIUM-3 para Δ |
| Holdout reutilizable / presupuesto de consultas | Cita correcta [6] |
| Weight-copying en Bittensor | Fenómeno real; [13] es informe técnico, no revista |
| No token; activo existente | Coherente |

---

## 5. Condiciones residuales (parche, no reescritura)

Para remitir al comité, en una pasada de redacción:

1. Identificar H2 en el diseño (brazo o cruce).
2. Declarar `n_audit` o fracción, y que el piloto no la baje.
3. Una frase: Δ matriz de [3]; S_j promedio entre pares.
4. Soltar a Shannon del mecanismo de pago.

Con eso el documento es enviable. **No pido otra ronda de arquitectura.**
