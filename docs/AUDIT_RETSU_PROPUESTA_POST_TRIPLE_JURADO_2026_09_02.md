# Auditoría Retsu — propuesta post triple jurado

**Fecha:** 2026-09-02  
**Auditor:** Retsu  
**Documento:** `docs/propuesta_doctoral_doin_borrador.html` / `.pdf` (4 pp.)  
**Antecedentes:** `AUDIT_RETSU_…_2026_09_01.md` (`REVISE`); reauditoría (`ACCEPT` con cuatro parches); `AUDITORIA_TRIPLE_JURADO_…_2026_09_02.md` (admisible con revisiones mayores, P0–P2).

**Veredicto: `ACCEPT`.**  
Se puede remitir al programa. Lo que queda es pulido de una página, no otra arquitectura. Harvey y Musashi deberían mirar juntos un solo riesgo intelectual (Gao) y dos frases de diseño.

No se reescribe la propuesta.

---

## 1. Qué pienso del triple jurado

Es el mejor de los tres dictámenes que ha recibido este texto, y en un punto me corrige.

- **B1 (Gao–Wright–Leyton-Brown) era el golpe que yo no di.** El paper existe: *Artif. Intell.* 275:618–638, 2019; el arXiv se titula incluso *Peer-Prediction Makes Things Worse*. El resultado es que, con señales laterales baratas y un poco de verdad de referencia, un mecanismo simple de spot-check **domina** a una familia amplia de peer prediction. Si un teórico de mecanismos está en la sala, “Shnayder 2016 + auditoría en un banco propio” es exactamente la frase. Ponerlo en P0 fue correcto.
- **A3 (quién es humano)** también era letal administrativo. Yo lo tenía como residual de potencia (`n_audit`); el triple jurado lo subió a naturaleza de actores + ética. Tenían razón: sin eso el comité de ética ni abre el archivo.
- **B2 (un teorema, no “formalizar”)** es la diferencia entre tesis y plataforma. Yo pedí recorte de alcance; ellos pidieron una proposición. Las dos cosas eran necesarias; la proposición es la que convierte el recorte en contribución.
- **C4 (zkML / TOPLOC)** es la emboscada de sistemas 2025. Justa.
- **C5 casillas** (director, hitos, recursos, publicaciones, ética): en comités colombianos se marcan con esfero. No es pedantería.

Donde el triple jurado **aprieta de más** (no seguiría yo esas órdenes al pie):

| Orden | Por qué no es letal |
|---|---|
| Nombrar director ya | El candidato aún no tiene el perfil del programa. “Se solicitará dirección en IA confiable / mecanismos” es honesto. Inventar un nombre es peor. |
| 20/5/10 “arbitrarios” como P1 de rechazo | Ya son pisos que solo se endurecen, con piloto de potencia. Un teórico querrá derivarlos; un comité de admisión no tumba por eso. |
| Semillas “cinco es poco” | La versión actual sube a diez y Holm. Suficiente para propuesta. |
| Truebit / Kleros / Augur | Cubiertos en espíritu por zkLLM+TOPLOC + “el ledger no es oráculo”. Citar disputas on-chain es P2, no P0. |
| Tabla de madurez DOIN con throughput | Útil en entrevista, no cabe en cuatro páginas sin matar otra frase. |

**Pronóstico del triple jurado** (“cuartil superior si se ejecuta P0”) era correcto *antes* de la corrección de Musashi. Hay que reevaluar *después*.

---

## 2. Qué hizo Musashi, orden por orden

| P | Orden del triple jurado | ¿En el HTML actual? |
|---|---|---|
| P0.1 | Literatura cercana + reposicionar novedad | **Sí.** [8] Gao et al. 2019; [9] Dasgupta–Ghosh 2013. Novedad: “resultado formal y evaluación adversarial del protocolo *compuesto*, no la mera suma de CA y auditoría.” zkLLM [13] y TOPLOC [14] (ICML 2025, PMLR 267:47196–47211 — verificado). |
| P0.2 | Actores + ética + recursos | **Sí.** Trabajadores = simulados + LLM/optimizadores; evaluadores = agentes + jueces LLM de *familias separadas*; humanos solo referencia oculta (2+1). Ética, 40–80 h expertas, 120 valoraciones / 60 arbitrajes. |
| P0.3 | Un teorema | **Sí.** Recuadro p. 2: condiciones sobre *p, α, ℓ, cᵢ, vᵢ* para mejor respuesta estricta, o región de imposibilidad en diálogo con [8][9]. Objetivo 1 alineado. |
| P0.4 | Casillas administrativas | **Parcial y suficiente para preliminar.** Cronograma S1–S6, publicaciones (AAMAS/EC + experimental + artefacto), recorte del año 3, pagos testnet. Director: perfil, no nombre. |
| P1.5 | Brazo CA sin auditoría | **Sí.** Brazo E, “celda crítica para equilibrios patológicos.” |
| P1.6 | Justificar 20/5/10 | **Parcial.** Pisos que solo se endurecen o se sube *N*; el piloto “traduce costo y riesgo”. No hay anclaje en pesos colombianos. Aceptable. |
| P1.7 | Atacante de mejor respuesta | **Sí.** Optimizador evolutivo/RL contra el protocolo publicado, incl. pagos laterales. H1 mide esa desviación. |
| P1.8 | Semillas, unidad, multiplicidad | **Sí.** Tarea única; ≥10 poblaciones; bootstrap jerárquico; Holm. |
| P2.9 | Encuadre IA | **Sí.** Primer párrafo: evaluación confiable de LLM/agentes. |
| P2.10 | Nicho vs inferencia verificable | **Sí.** zkLLM/TOPLOC verifican ejecución, no calidad semántica; la tesis es el residuo. |
| P2.11 | Recorte año 3 + madurez DOIN | **Recorte sí; tabla de madurez no.** MVP = teorema + evaluación en orquestador firmado. |
| P2.12 | Liquidación | **Sí.** Simulada o testnet, sin dinero real. |

Mis cuatro residuales del `ACCEPT` anterior también están:

| Residual mío | ¿Cerrado? |
|---|---|
| H2 identificada | **Sí.** B y D × {test público, retos ocultos}; H2 exige *ambos* contrastes. |
| `n_audit` | **Sí.** max(60, ⌈0,30 N_únicas⌉); muestra *compartida* entre brazos (120 valoraciones = 60 tareas × 2 expertos). Coherente. |
| Δ matriz + promedio de pares | **Sí.** Fórmula Sign[P(x,y)−P(x)P(y)]; Sⱼ promedia k≠j. |
| Shannon fuera del pago | **Sí.** Ya no aparece en §6. |

Citas nuevas que verifiqué: Gao 2019 *AI* 275:618–638 **exacto**; Holmström 1982 *Bell J. Econ.* 13(2):324–340 **exacto**; zkLLM CCS 2024 **existe** (DOI 10.1145/3658644.3670334; las páginas 4405–4419 coinciden con el PDF de ACM); TOPLOC ICML 2025 PMLR 267:47196–47211 **existe**.

---

## 3. Hallazgos que *yo* todavía haría (para la mesa con Musashi)

Ninguno es `CRITICAL`. El único que puede volver a doler en sala es el primero.

### HIGH-1 — El teorema prometido no puede ser Gao con otro nombre

- **Dónde:** p. 2 recuadro; p. 4 §12.
- **Afirmación:** se demostrarán condiciones para que esfuerzo alto y reporte veraz sean mejor respuesta estricta; si señales laterales lo impiden, imposibilidad “en diálogo con [8], [9]”.
- **Golpe:** Gao *ya* demuestra que, con señales laterales baratas (y el dominio LLM *está lleno* de texto), peer prediction no mejora —y a menudo empeora— el spot-check simple. Si la proposición del año 1 es “CA+auditoría funciona cuando p es alto”, B dirá: eso está en Gao 2019, Theorem-esque. La novedad escrita (“protocolo *compuesto*”) tiene que decir **qué añade** el trabajador que compromete, los retos post-compromiso y el pago del *trabajo* (no solo del dictamen) respecto del modelo de Gao, que es *solo evaluadores*.
- **Corrección mínima (una frase en §12):** el objeto formal no es elicitación aislada de jueces, sino el juego *trabajador–evaluador* con compromiso previo, retos ocultos y liquidación del artefacto. Si el teorema no usa esas tres piezas, no hay tesis.

### MEDIUM-1 — 60 tareas auditadas × 2 expertos es justo el borde de potencia para H1

- Compartir la muestra entre brazos es lo correcto (comparación pareada). 60 ítems dan IC de una proporción del orden ±0,13 en bruto; Holm y el 5 pp de no-inferioridad de FP van a apretar. El piloto puede subir *N*, y está escrito. No bloquea admisión; en el preregistro hay que fijar el estimador (bootstrap por tarea, no por evaluación).
- **Corrección:** una cláusula de que el contraste de FP es pareado sobre las mismas 60 tareas.

### MEDIUM-2 — Jueces LLM de “familias separadas” vs [7]

- Bien separado del experto humano. Sigue el riesgo de correlación de familia (todos los jueces “open” copian sesgos de [7]). El diseño ya *mide* posición/verbosidad/auto-preferencia. Pedir en una línea que las familias de juez y de trabajador no compartan checkpoint ni proveedor.

### MEDIUM-3 — Schoenebeck–Yu y Witkowski–Parkes siguen fuera

- El triple jurado los nombró. No son P0 una vez citado Gao. Si sobra una referencia, Schoenebeck–Yu (robustez / learning of peer prediction) es la que más cubre B5 (Δ no estacionaria).

### LOW-1 — Zhang et al. S&P 2022 (spoof de PoL) se cayó; queda solo Fang [12]. Fang basta.

### LOW-2 — Truebit/Kleros no están; zkLLM+TOPLOC cubren la emboscada de 2025.

### LOW-3 — No hay nombre de director. Correcto hasta concertar el programa.

### LOW-4 — Cuatro páginas a 9,5 pt / 1,18 siguen densas. Remisible.

---

## 4. Sobre el documento como objeto de admisión

Lo que ya es difícil de derribar:

- Pregunta única y falsable.
- Caja de tesis con “si no hay observación externa, solo acuerdo informado”.
- Brazo E (CA sin auditoría) — exactamente donde Gao predice patología. Sin esa celda el teórico tenía razón; ahora la tiene el candidato.
- Atacante de mejor respuesta, no solo el repertorio que CA gana por diseño.
- MVP si DOIN se atrasa.
- Pagos simulados; ética antes de reclutar; humanos solo en la referencia.
- Continuidad maestría → cambio de mecanismo, no “extensión ya resuelta”.

Lo que un jurado B todavía puede intentar: “¿por qué no el mecanismo simple de Gao?” La defensa es el recuadro del teorema **si** el enunciado usa trabajador+compromiso+retos, no solo jueces.

Lo que un jurado A ya no puede: “¿quién es el experto?” Está escrito.

Lo que un jurado C ya no puede: “el año 3 es otra tesis” / “DOIN es decorado sin criterio”. Hay recorte y hay MVP.

---

## 5. Qué le diría yo a Harvey, en una mesa con Musashi

1. **No reabrir alcance.** El recorte aguantó dos auditorías. Añadir series de tiempo o un tercer dominio sería un error.
2. **Una frase contra Gao** (HIGH-1). Es el único parche que yo exigiría antes de PDF institucional.
3. **No inventar director.** El párrafo de perfil es el correcto.
4. **El triple jurado valió la pena**; Musashi ejecutó P0 y casi todo P1/P2 sin inflar a seis páginas. Eso es disciplina.
5. **Mi `ACCEPT` anterior se mantiene y se endurece:** el texto ahora sobreviviría la sala que el triple jurado simuló, con la salvedad Gao.

**No pido otra ronda de arquitectura.** Si Musashi mete la frase de HIGH-1, yo consideraría el expediente de propuesta **cerrado** para admisión.
