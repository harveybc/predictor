# Auditoría de triple jurado — Propuesta doctoral DOIN (borrador 2026-09-02)

**Auditor:** Satoshi III, por orden del propietario.
**Documento auditado:** `docs/propuesta_doctoral_doin_borrador.pdf` (4 páginas, Letter).
**Destinatarios:** editor y jefes de agentes de proyecto.
**Regla:** crítica dura y honesta desde tres jurados de admisión simulados, con formaciones, especializaciones y puntos ciegos distintos. NO se editó el documento.

---

## Veredicto ejecutivo

**Admisible con revisiones mayores.** El núcleo científico es defendible y la madurez
metodológica está muy por encima del borrador mediano (tesis falsable en caja,
límites de identificabilidad declarados, compromiso con resultados negativos,
preregistro, márgenes de no-inferioridad). Pero en su forma actual hay **dos
vulnerabilidades letales** que un tribunal exigente puede usar para rechazar o
forzar reformulación: (1) la línea académica MÁS CERCANA al protocolo propuesto
(peer prediction + acceso limitado a verdad de referencia, y verificación de
cómputo/inferencia 2024-26) no está citada, lo que deja la afirmación de novedad
expuesta; (2) la propuesta nunca dice si trabajadores y evaluadores son humanos,
bots o LLMs — ambigüedad que arrastra ética, presupuesto y validez. A eso se
suman ausencias administrativas que los comités marcan como casillas: director,
cronograma con hitos, recursos, plan de publicaciones.

---

## Jurado A — Catedrática de aprendizaje automático y estadística

*Perfil:* publica en NeurIPS/ICML, dirige tesis de ML aplicado, revisora dura de
diseño experimental. *Punto ciego:* sabe poco de diseño de mecanismos y nada de
sistemas descentralizados; "descentralizado" le suena a criptomoneda y entra
predispuesta en contra. Tratará Correlated Agreement como "una métrica más".

### A1 — Circularidad del adversario (severidad: ALTA)
H1 y H2 se contrastan contra estrategias de ataque (copia, bajo esfuerzo,
contaminación, coaliciones) que el propio doctorando programa. CA está diseñado
por construcción para castigar la copia; demostrar que la castiga dentro de una
simulación propia es cuasi-tautológico. La evidencia solo tiene dientes si el
atacante es un **optimizador de mejor respuesta** (búsqueda/RL sobre el espacio
de estrategias del protocolo publicado), no un repertorio fijo. La propuesta no
promete ningún atacante adaptativo.

### A2 — Umbrales de efecto arbitrarios (ALTA)
20%, 5 puntos, 10% de margen: ningún anclaje económico ni empírico. En sala
sonarán a números redondos elegidos para pasar. Además: la unidad de análisis
(tarea, lote, semilla, población) y la estructura de dependencia dentro de lote
harán la potencia real menor que la nominal 0,80; **cinco semillas de población
es poco** para afirmaciones confirmatorias sobre dinámica estratégica; no hay
mención de corrección por multiplicidad pese a que H2 se respalda con dos
contrastes cruzados y hay tres hipótesis con métricas múltiples.

### A3 — El dominio LLM descansa en un ground truth sin caracterizar (ALTA)
La "auditoría experta oculta" ES la verdad de referencia de facto del régimen
principal. ¿Quiénes son los expertos, cuántos, con qué acuerdo inter-anotador,
con qué rúbrica validada? Si son humanos: falta comité de ética, reclutamiento y
presupuesto. Si son LLMs jueces: la propia referencia [7] documenta sus sesgos y
la tesis se vuelve "LLMs auditando LLMs". La propuesta no declara la naturaleza
de NINGÚN participante (trabajador, evaluador, experto). Esta ambigüedad es
letal por sí sola.

### A4 — ¿Dónde está la contribución de IA? (MEDIA)
Los métodos son diseño de mecanismos + estadística experimental. Para un
**Doctorado en Inteligencia Artificial** el encuadre debe ser explícito:
evaluación confiable de sistemas de IA, contaminación de benchmarks,
LLM-as-judge, incentivos en pipelines de evaluación — está implícito en el
dominio principal, pero nadie lo argumenta. El jurado A preguntará: "¿esto no es
una tesis de economía experimental que usa IA de utilería?"

### A5 — Validez externa mínima (MEDIA)
El dominio de control se declara sintético y "sin realismo" (honesto, pero
débil), y las series temporales — el dominio de la maestría del candidato — se
posponen a "validez externa posterior". El comité notará que el terreno donde el
candidato tiene ventaja comparativa desaparece del diseño confirmatorio.

---

## Jurado B — Profesor de teoría de juegos y diseño de mecanismos

*Perfil:* economista matemático, línea de peer prediction y subastas, exige
teoremas. *Punto ciego:* desprecia el costo de ingeniería, no valora bancos
experimentales reproducibles, y pedirá formalismos imposibles en tres años.

### B1 — La brecha de literatura que puede matar la novedad (LETAL)
La combinación "peer prediction + verificación puntual con verdad limitada" ya
tiene línea académica: **Gao, Wright & Leyton-Brown (incentivos de evaluación
con acceso limitado a ground truth — con el resultado incómodo de que peer
prediction puede EMPEORAR las cosas)**, Dasgupta–Ghosh, Witkowski–Parkes, la
familia de *spot-checking* en crowdsourcing, y Schoenebeck–Yu sobre robustez. La
propuesta cita Shnayder [3] y Kong–Schoenebeck [4] pero NO la línea híbrida, que
es exactamente el mecanismo propuesto. Si un jurado la conoce, la frase será:
"esto es Shnayder 2016 más spot-checks conocidos, corrido en un banco de pruebas
propio". Hay que citarla y **reposicionar la novedad** (composición con retos
post-compromiso + mercado operable + evidencia adversarial sistémica, y/o un
resultado formal propio).

### B2 — Formalización sin teorema prometido (ALTA)
El Objetivo 1 dice "formalizar" y la Contribución 1 promete "protocolo formal
con supuestos y estados de rechazo explícitos" — es decir, **definiciones, no
proposiciones**. Un doctorado necesita al menos un resultado demostrado no
trivial. El candidato tiene uno al alcance: condiciones sobre la fracción de
auditoría p, el bono α y la cota del valor de desviación bajo las cuales el
reporte veraz es equilibrio estricto del protocolo COMPUESTO (CA + auditoría +
retos). Sin esa promesa, B votará "proyecto de ingeniería con estadística".

### B3 — Modelo económico subespecificado (ALTA)
- "Utilidad de desviación" es métrica primaria y nunca se define formalmente
  (¿utilidad respecto a qué creencias, qué información, esfuerzo binario o
  continuo, costos homogéneos o heterogéneos?).
- Contabilidad presupuestal ausente: B fijo debe cubrir trabajador + ≥3
  evaluadores + tercer árbitro en desacuerdos + n_audit = max(60, 30%·N)
  auditorías expertas. Nadie muestra que el presupuesto cierra ni que c₀
  garantiza racionalidad individual con α acotado.
- Myerson–Satterthwaite [9] invocado como escudo general es grueso: es un
  resultado de comercio bilateral con valores privados a dos lados; el contexto
  aquí es principal con presupuesto fijo y riesgo moral en equipos — la
  imposibilidad pertinente es otra (línea Holmström) y citarla mal delata.

### B4 — Celda experimental faltante: CA sin auditoría (ALTA)
D−C aísla CA condicionado a auditoría; B−C aísla auditoría condicionada a
mayoría. **Nadie aísla CA puro** — precisamente la celda donde la literatura
(B1) predice patología por equilibrios permutados/no informativos. Sin esa
celda no se puede atribuir el efecto y la afirmación central queda confundida.

### B5 — Supuestos de CA fuera del laboratorio (MEDIA)
CA exige tareas correlacionadas y emparejamiento denso evaluador×tarea. Los
mercados reales son ralos; el diseño de asignación es un problema de
investigación en sí y no se trata. La estimación de la matriz Δ con cuatro
pliegues supone estacionariedad de la distribución de etiquetas que un mercado
real viola.

### B6 — Colusión modelada de forma estrecha (MEDIA)
Solo se contempla permutación/colusión sobre etiquetas. El ataque económico real
en mercados es la colusión lateral evaluador–trabajador (side-payments por
aceptar). Los retos post-compromiso no la tocan y CA es vulnerable a señales
coordinadas. Al menos debe declararse fuera de alcance con argumento.

---

## Jurado C — Profesor de sistemas distribuidos e ingeniería de software

*Perfil:* pragmático, dirige laboratorio de infraestructura, evalúa viabilidad y
riesgo de no-terminación. *Punto ciego:* flojo en estadística y teoría de
juegos; leerá CA como "otra fórmula de reputación" y pedirá demos.

### C1 — Alcance de dos tesis para un doctorando (ALTA)
Protocolo formal + simulador + dos dominios + campañas adversariales + piloto de
potencia + integración DOIN + replicación distribuida con procesos
independientes = dos tesis. El año 3 es un proyecto de ingeniería completo.
Falta un **plan de recorte explícito**: qué es el MVP de tesis defendible y qué
cae si DOIN se atrasa.

### C2 — El ledger sin hipótesis (ALTA)
El texto admite que "un registro central firmado será el control" y que la
comparación "no constituye una hipótesis doctoral separada". Entonces el jurado
dirá: si el registro central basta, DOIN es decorado costoso; si no basta,
¿dónde está la hipótesis y el criterio de éxito que lo demuestre? "Cuantificar
el costo adicional del registro" sin umbral de decisión no es ciencia, es
contabilidad.

### C3 — Madurez de DOIN no evidenciada (MEDIA)
[17] es un repositorio propio sin métricas declaradas (cobertura de pruebas,
throughput, número de nodos, campañas previas). El año 3 cuelga de
infraestructura que el comité no puede auditar desde la propuesta. Una tabla de
madurez con números reales desactiva la objeción — el material existe; la
propuesta no lo muestra.

### C4 — Estado del arte de sistemas 2024-26 ausente (ALTA)
Faltan: juegos de verificación (Truebit, delegación arbitrada), resolución de
disputas (Kleros, Augur), y sobre todo la ola 2024-26 de **verificación de
inferencia** (TOPLOC/replicación determinista, opML, zkML, proof-of-sampling,
servicios de verificación tipo AVS). Si la inferencia LLM se vuelve verificable
por recomputación barata, el nicho del protocolo queda restringido a la calidad
subjetiva/rúbrica — ese nicho ES defendible, pero hay que argumentarlo antes de
que el jurado lo use como emboscada.

### C5 — Operación del experimento sin recursos (ALTA)
n_audit = max(60, 30%·N) por celda × (4 brazos × 2 factores de reto × dominios
+ piloto) ≈ **miles de juicios expertos**. ¿Presupuesto, contratación, tiempo,
GPU para los lotes de inferencia? La propuesta no tiene sección de recursos, ni
cronograma con hitos semestrales verificables, ni plan de publicaciones (¿EC?
¿AAMAS? ¿NeurIPS D&B? ¿S&P?), ni director propuesto, ni aval de ética. En
comités colombianos estas casillas se revisan literalmente.

### C6 — "Activo existente" con dinero real (BAJA-MEDIA)
Si la liquidación usa un activo real: custodia, jurisdicción y cumplimiento; si
es testnet/simulada: decirlo. Una línea resuelve la objeción.

---

## Convergencia del tribunal — dónde golpearán los tres a la vez

1. **Brecha de trabajo relacionado más cercano** (B1 + C4): la novedad, tal como
   está escrita, es refutable con dos citas que la propuesta no contiene. Es la
   falla más peligrosa del documento.
2. **Naturaleza de los participantes + ética + recursos** (A3 + C5): ambigüedad
   humanos/simulados con consecuencias administrativas y de validez. Rechazo
   administrativo posible aun con ciencia buena.
3. **Contribución teórica indefinida** (A1 + B2): sin un teorema prometido ni un
   adversario optimizador, el proyecto puede leerse como "banco de pruebas que
   confirma lo que su mecanismo hace por diseño".
4. **Componentes formales de propuesta ausentes**: director, cronograma con
   hitos, presupuesto, publicaciones, ética, riesgos.
5. **Umbrales 20/5/10 sin justificación y celda CA-pura faltante.**
6. **Encuadre explícito de contribución a la IA** para el nombre del programa.

---

## Fortalezas reales (decirlas también es honestidad)

- Pregunta de investigación única, acotada y **falsable**; caja de tesis con
  criterio de refutación explícito — raro y valioso en una propuesta.
- Honestidad epistémica inusual: límites de identificabilidad ("solo puede
  reclamarse acuerdo informado"), PoL declarado roto [11][12], Sybil no
  eliminado por el ledger [10], compromiso con publicar resultados negativos,
  piloto excluido del análisis confirmatorio, no-inferioridad con margen
  predeclarado.
- Delimitación negativa nítida: no otro mercado de cómputo, no token, la
  blockchain no es oráculo, la ablación K=1 vs K>1 no es segunda tesis.
- Continuidad maestría→doctorado con recorte honesto de la contribución.
- Las 17 referencias verificadas son reales y pertinentes (ninguna fabricada);
  la mezcla con literatura gris (litepapers/docs) es aceptable si se complementa
  con los pares arbitrados que faltan.
- Redacción densa, precisa, sin humo promocional.

---

## Órdenes de corrección sugeridas al editor (por prioridad)

**P0 — sin esto, riesgo real de rechazo:**
1. Sección de trabajo relacionado académico con la línea peer-prediction +
   verdad limitada (Gao–Wright–Leyton-Brown, Dasgupta–Ghosh, spot-checking,
   Schoenebeck–Yu) y la ola de verificación de inferencia 2024-26; reposicionar
   la novedad frente a ambas explícitamente.
2. Declarar la naturaleza de cada actor (propuesto: trabajadores y evaluadores
   = agentes simulados y LLMs; humanos SOLO en auditoría experta), con párrafo
   de ética y estimación de recursos/costos de auditoría.
3. Prometer UN resultado formal concreto: condiciones (p, α, cotas de valor de
   desviación) bajo las cuales la desviación es no rentable en el protocolo
   compuesto. Una proposición demostrable convierte "integración" en "teoría".
4. Añadir los componentes administrativos: director propuesto, cronograma
   semestral con hitos verificables, recursos, plan de publicaciones.

**P1 — endurecen la defensa:**
5. Brazo experimental CA-sin-auditoría (celda de atribución).
6. Justificar 20/5/10 con anclaje de costo económico, o declararlos márgenes
   preregistrados con racional documentado.
7. Incluir un adversario de mejor respuesta (optimizador) además del repertorio
   fijo de ataques.
8. Subir semillas o justificar; declarar unidad de análisis y corrección por
   multiplicidad.

**P2 — pulen:**
9. Párrafo de encuadre "contribución a la IA" (evaluación confiable de sistemas
   LLM/agentes).
10. Argumentar el nicho frente a la inferencia verificable por recomputación.
11. Plan de recorte del año 3 (qué cae si DOIN se atrasa) y tabla de madurez de
    DOIN con métricas reales.
12. Una línea sobre liquidación (testnet/simulada vs activo real).

**Pronóstico si se ejecuta P0 completo:** propuesta en el cuartil superior de
admisión, difícil de derribar en sala; el candidato ya escribe con la
honestidad metodológica que los tribunales premian, le falta blindar la novedad
y las casillas administrativas.

*Documento de trabajo — no comprometido en git. No se modificó la propuesta.*
