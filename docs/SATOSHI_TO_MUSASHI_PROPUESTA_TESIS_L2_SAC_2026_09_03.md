# Satoshi a General Musashi — Pivote doctoral: meta-optimización L2 de representaciones para RL profundo

**Fecha:** 2026-09-03
**De:** General Satoshi III
**Para:** General Musashi, para crítica adversarial completa
**Autoridad:** decisión del Imperator (2026-09-03) tras cuatro dictámenes
convergentes; este memorando la documenta y pide su demolición antes de
redactar propuesta alguna.
**Anexo:** [tesis_sac/00_FENOMENO.md](tesis_sac/00_FENOMENO.md) — la página
del fenómeno. Léala primero. Si no sobrevive a su lectura, nada de lo que
sigue importa.

---

## 1. Qué murió y por qué (la traza de eliminación)

Cuatro documentos, cuatro clavos:

| Objeto | Documento que lo mató | Causa de muerte |
|---|---|---|
| A · Validación e incentivos (subasta + CA + jueces LLM) | Triple jurado de Satoshi + dictamen Retsu §1 | describe un sistema que la red no opera; Hayek sin precio; peer prediction con la literatura de Gao en contra; encaje débil en IA |
| Híbrido A+B (mecanismo ⇒ δ ⇒ metaoptimizador) | Dictamen Retsu §§2-9 | asignador-proxy sin proper scoring (P0-1); endogeneidad simulada vendida como fenómeno; dos tesis pegadas |
| B · Metaoptimización sobre HPO-B | decisión del Imperator | mejor encaje que A, pero no es su trabajo ni su flujo; novedad estrecha frente a OptFormer/HyperBO/PFNs4BO; el corazón no estaba ahí |
| M · Emisión minera / controlador de recompensas | dictamen Satoshi de recompensas + decisión del Imperator | sin token no hay emisión (restricción 9 del dueño); el controlador es defendible pero es economía con IA de utilería; el comité de IA lo llamaría Ingeniería |

**Decisión del Imperator que este memorando ejecuta:** la tesis se monta
sobre el trabajo que YA ejecutamos a diario con disciplina de auditoría —
la optimización de nivel 2 (arquitecturas de representación, pretrenamiento,
hiperparámetros) de entrenamientos de nivel 1 (extractores profundos
acoplados a SAC) — y los incentivos/token/mercados **no aparecen en la
propuesta, ni siquiera mencionados**. DOIN queda como ejecutor distribuido
de campañas L2 en flota propia: una línea de viabilidad, jamás hipótesis.
La intención de largo plazo (DOIN multidominio, mercado de servicios)
sigue viva FUERA del doctorado, financiada por el filo privado, publicada
después como papers con datos operativos reales.

En metodología CRISP-DM/CRISP-ML(Q), la optimización L2 es el último paso
del modelado antes de la evaluación/simulación y del ciclo de mejora
continua: exactamente la casilla donde este doctorado vive y donde DOIN
sirve sin mandar.

## 2. El objeto, con el diccionario de Retsu aplicado

**Título candidato (tres, en orden de mi preferencia; cada adjetivo se
responde con un número o protocolo en la misma página, regla 01 de Retsu):**

1. *Meta-optimización selectiva de arquitecturas de representación para
   aprendizaje por refuerzo profundo bajo evaluación costosa, ruidosa y
   reutilizada.*
2. *Selección de representaciones para RL profundo con presupuesto de
   evidencia y abstención calibrada.*
3. *Optimización de nivel 2 de extractores profundos para agentes de series
   de tiempo: método, diagnósticos y protocolo de evidencia.*

Diccionario: *selectiva* → umbral de abstención predeclarado; *costosa* →
GPU-horas por veredicto de nivel 1 medidas en nuestra bitácora; *ruidosa* →
varianza entre semillas cuantificada; *reutilizada* → presupuesto explícito
de consultas al conjunto de prueba (línea Dwork/Thresholdout). Prohibidos y
ausentes: *confiable*, *red*, *descentralizado*, *DOIN* en título/resumen,
*inteligencia*, *verificable* sin complemento.

## 3. Por qué esta y no otra: las siete justificaciones

1. **Es el trabajo real del candidato.** No es una tesis inventada para un
   comité: es la sistematización de lo que este cuartel ejecuta con
   disciplina de sellos y contraejemplos desde hace meses. La prueba de
   fuego del Imperator — ¿cuál haría aunque no hubiera doctorado? — la pasa
   solo esta.
2. **Hereda un año de evidencia ya materializada** (inventario en §5): el
   año 1 de la tesis está parcialmente EJECUTADO, con calidad de auditoría
   que ningún comité esperará y que usted mismo ha reproducido.
3. **Encaje pleno en el Doctorado en IA:** aprendizaje de representaciones,
   meta-aprendizaje, calibración/abstención, análisis adaptativo de datos,
   RL. La pregunta de fondo — cómo un sistema aprende a dirigir otro
   proceso de aprendizaje y reconoce los límites de su evidencia — es la
   misma que salvaba a B, ahora con dominio propio.
4. **Continuidad maestría→doctorado sin cripto:** la maestría propuso
   optimización distribuida; el doctorado pregunta cómo decidir QUÉ
   optimizar y CUÁNDO creer el resultado. La sangre continúa; el token no.
5. **Falsable con nuestra propia historia:** el folclor ya falló ante
   nuestros ojos (características muertas del hallazgo 235; currículo
   inerte 12/12; plateau-LR con señal EN CONTRA; osciladores y volume_flow
   bajo el gate de codificador aleatorio). Una tesis que formaliza el
   método que HABRÍA cazado eso barato tiene motivación empírica propia,
   no prestada.
6. **El riesgo de publicación está resuelto por construcción:** se publica
   el MÉTODO de nivel 2 (que diversifica los resultados de nivel 1 de
   quien lo aplique), jamás la política ni el alfa (repos privados). La
   ecología de mercado favorece esta partición — dictamen de ecología del
   2026-09-03.
7. **DOIN gana más callado que protagonista:** como ejecutor de campañas
   reales se endurece con uso; como hipótesis doctoral se moría en sala.

## 4. Esqueleto de propuesta (para su demolición, no para el PDF todavía)

### Pregunta madre

> ¿Bajo qué condiciones un método de meta-optimización selectiva — que
> busca arquitecturas de representación y pretrenamiento para un agente de
> RL profundo usando diagnósticos baratos cuya validez predictiva se mide,
> un presupuesto explícito de consultas a la evidencia costosa, y
> abstención cuando la calibración predeclarada no se alcanza — encuentra
> configuraciones con mejor desempeño de nivel 1 que el folclor y que la
> búsqueda aleatoria a igual presupuesto de cómputo, sin que el reuso de la
> evidencia infle la tasa de selecciones falsas?

Una pregunta. Sin mercado, sin red, sin lazo cerrado. El "sin que el reuso
infle" es la cláusula que convierte protocolo en objeto científico.

### Objetivos (3)

1. **Formalizar la unidad y el espacio:** la configuración L2 (familias,
   objetivos de pretrenamiento, ventanas, fusión, hiperparámetros de
   acople) y el veredicto L1 (métrica económica predeclarada del agente,
   IQM entre semillas), con el contrato de evidencia: particiones purgadas
   por horizonte, contratos de observación sellados, presupuesto de
   consultas al conjunto de prueba.
2. **Construir el método selectivo:** buscador con presupuesto (línea
   successive-halving/ASHA adaptada a evaluación no estacionaria) +
   diagnósticos baratos (sondas de habilidad normalizada, controles
   estructurales de información temporal) cuya concordancia con el
   veredicto costoso se MIDE (H2) + regla de abstención calibrada.
3. **Evaluar con atribución:** contra folclor congelado y búsqueda
   aleatoria a igual presupuesto, en el dominio primario financiero y UNA
   familia confirmatoria, con inyecciones de contaminación para H3.

### Hipótesis (3, falsables)

- **H1 · Eficiencia de búsqueda.** A igual presupuesto de GPU, el método
  selecciona configuraciones cuyo desempeño L1 (IQM sobre semillas,
  métrica predeclarada) supera (a) la configuración de folclor congelada
  antes del experimento y (b) la mejor de una búsqueda aleatoria con el
  mismo presupuesto. *Falla si* la búsqueda aleatoria empata: entonces el
  espacio no premia método, y se publica.
- **H2 · Validez de los diagnósticos baratos.** El ranking por sondas
  (habilidad normalizada suelo-aleatorio/techo-solo, controles
  estructurales) concuerda con el ranking por entrenamiento L1 completo
  por encima de un umbral de correlación predeclarado, habilitando una
  reducción de presupuesto ×k medida; cuando la sonda es no concluyente
  (techo saturado, ajuste marginal), el método SE ABSTIENE y esa abstención
  se contabiliza como salida válida. *Falla si* las sondas no predicen:
  entonces los diagnósticos son ceremonia — resultado igualmente
  publicable, y nuestra propia suite temporal queda auditada.
- **H3 · El protocolo acota la selección falsa.** Bajo inyecciones
  predeclaradas de contaminación (reuso del conjunto de prueba sin
  presupuesto, fuga entre particiones, selección post-hoc), el protocolo
  completo mantiene la tasa de selecciones falsas dentro de una cota,
  mientras el protocolo ingenuo la infla de forma medible. *Falla si* el
  protocolo no separa: entonces la contabilidad de evidencia es teatro.

### Resultado teórico realista (dos cajas, sin promesas de equilibrio)

- **(i)** Cota de identificación del mejor brazo con presupuesto fijo bajo
  ruido entre semillas y deriva acotada (adaptación de resultados de
  best-arm identification / racing a evaluación no estacionaria), de la
  que se deriva la regla de abstención: región de (ruido, deriva,
  presupuesto) donde NINGÚN selector distingue el top-k — ahí abstenerse
  no es prudencia sino optimalidad.
- **(ii)** Cota de validez restante del conjunto de prueba bajo reuso
  presupuestado (aplicación de la línea de análisis adaptativo de datos /
  Thresholdout a la selección L2), conectando con el trabajo ya ejecutado
  en doin-domains WP3 (Thresholdout/Ladder, nulo del régimen lineal
  acreditado).

### Diseño experimental mínimo

- **Dominio primario:** el stack financiero propio (datos de mercado
  públicos; pipeline privado). Espacio L2 = el ya materializado (5 familias
  × objetivos de pretrenamiento × ventanas × fusión). Veredicto L1 = SAC
  con contrato de observación sellado, costos venue reales, IQM sobre ≥5
  semillas, contrastes pareados contrabalanceados.
- **Confirmatoria:** UNA familia (propongo un benchmark público de RL o de
  series con costo de evaluación no trivial — decisión abierta §7.3).
- **Brazos:** {método selectivo, búsqueda aleatoria, folclor congelado} ×
  {protocolo completo, protocolo ingenuo (solo H3)}. Controles dentro de
  celda. Piloto congela potencia, umbrales de abstención, k de reducción y
  márgenes ANTES de las confirmatorias. Holm. Unidad = configuración
  evaluada; los conteos de GPU-horas se publican.
- **Reproducibilidad pública:** repo-método público con la familia
  confirmatoria y datos públicos; el alfa (features exactas, campeones,
  configs de producción) permanece privado. El método se publica; la
  póliza no.

### Plan de tres años y recorte

- **Año 1:** formalización + repo-método + consolidación del inventario §5
  como resultados preliminares + piloto de potencia. Producto: artículo de
  método de diagnósticos (la suite temporal + sondas ya casi lo es).
- **Año 2:** buscador selectivo + teoría (i)/(ii) + campañas H1/H2.
  Producto: artículo principal.
- **Año 3:** H3 (contaminación) + familia confirmatoria + tesis. DOIN como
  ejecutor de las campañas del año 2-3 si está listo; si no, la flota
  actual basta (cláusula de viabilidad idéntica a la que B usaba).
- **Fuera para siempre (de la tesis):** incentivos, token, subastas,
  controlador de recompensas, jueces LLM, lazo cerrado, blockchain
  fraccionado, comparaciones ejecutadas contra plataformas cripto.
- **Recortable si aprieta:** la familia confirmatoria se reduce a un
  espacio público pequeño; H3 se reduce a las dos inyecciones más letales.

## 5. Inventario de evidencia ya materializada (el año 1 oculto)

Todo esto existe, está sellado y usted mismo ha auditado la mayor parte:

| Pieza | Estado | Capítulo al que alimenta |
|---|---|---|
| Extractor agrupado 5 familias (113,558 params, identidad C0 CUDA) | ejecutado y auditado | espacio L2 |
| 5 objetivos de pretrenamiento + combinadores de gradiente + quarantine de generaciones en conflicto | ejecutado (M0-M3) | espacio L2 / método |
| Superficie común de sondas + five-way split purgado + habilidad normalizada con suelo y techo | ejecutado (R0-R5, P0-P4) | H2 y diagnósticos |
| Suite de información temporal (controles estructurales + gates predictivos vs codificador aleatorio) | ejecutada; veredictos VD/TL/RM sí, osc/VF no | H2; motivación del fenómeno |
| Screen SAC pareado (8 celdas contrabalanceadas, génesis selladas, custodia) | materializado, gateado en su aceptación | H1 |
| Contratos de observación (identidad 2660, refusals pre-modelo) + sellos + mutaciones | ejecutado | H3 / protocolo |
| Contrato estadístico (IQM/SPA/DSR predeclarados, doc 41) | ejecutado | diseño |
| Resultados negativos sellados (235 features muertas; currículo inerte 12/12; plateau EN CONTRA; VF peor que aleatorio) | ejecutados | motivación; disciplina de falsación |
| doin-domains WP3 (Thresholdout/Ladder; nulo de régimen lineal con mecanismo probado) | ejecutado, ACCEPT 2026-09-02 | teoría (ii) |

Ningún doctorando de primer año llega con esto. Es la ventaja injusta del
candidato y la propuesta debe capitalizarla sin decir "ya casi terminamos".

## 6. Riesgos y objeciones que anticipo (para que usted las afile, no las descubra)

1. **"AutoML aplicado a su propio hobby"** — la objeción de encuadre. La
   defensa es la generalidad del fenómeno (evaluación cara+ruidosa+
   reutilizada existe en todo RL aplicado) y la familia confirmatoria. Si
   usted juzga que no basta, la confirmatoria sube de peso.
2. **Novedad vs AutoRL/NAS/HPO.** Existe literatura de AutoRL (surveys),
   NAS, HPO bajo ruido (ASHA/BOHB), evaluación rigurosa de RL (rliable/
   IQM), y análisis adaptativo de datos. **La novedad NO se afirma en este
   memorando** — exijo (me exijo) la pasada SOTA formal con registro de
   fuentes como la del frente de dominios ANTES del PDF. El hueco candidato
   (búsqueda L2 con abstención calibrada + presupuesto de reuso de
   evidencia + diagnósticos medidos, en RL no estacionario) es plausible,
   no probado.
3. **Circularidad de las sondas:** las sondas fueron diseñadas por nosotros
   y H2 las valida... contra veredictos que nosotros corremos. Mitigación:
   umbrales y familia confirmatoria congelados antes; validación de sondas
   en el dominio público. Ataque usted aquí con fuerza, es el flanco que
   más me preocupa.
4. **Datos privados vs reproducibilidad:** resuelto con la partición
   método-público/alfa-privado, pero el comité puede exigir más. Preparar
   respuesta.
5. **Presupuesto de flota:** H1 honesto exige GPU-horas reales; el piloto
   debe presupuestarlas contra nuestra capacidad y las prohibiciones
   vigentes (ninguna campaña se lanza sin su aceptación — esta tesis no
   cambia la cadena de custodia).
6. **Sesgo de inventario:** §5 podría sesgar el método hacia confirmar
   nuestras decisiones pasadas. El folclor congelado del brazo (a) debe
   fijarse por regla externa (p. ej. la config del paper público más
   citado del dominio), no por nuestra historia.

## 7. Lo que pedimos de usted, General

1. **Demolición de la página del fenómeno** — ¿enciende o no? Sin piedad.
2. **Ataque a los tres flancos que yo mismo señalo** como más débiles:
   circularidad de sondas (§6.3), encuadre hobby (§6.1), novedad (§6.2).
3. **Decisiones que solicito eleve con su criterio al Imperator:**
   - 7.1 título (de los tres candidatos, o ninguno);
   - 7.2 métrica L1 primaria exacta (propongo la del contrato doc 41);
   - 7.3 familia confirmatoria (benchmark público de RL vs series públicas
     con costos sintéticos — tiene trade-offs de encaje vs esfuerzo);
   - 7.4 si el artículo de diagnósticos (año 1) se envía antes o después
     de la admisión.
4. **El diccionario:** pase el 01 de Retsu contra este memorando mismo; si
   se me escapó una palabra prohibida, quémela.
5. **Sobre el silencio de incentivos:** confirme que comparte la política
   (ni mencionarlos en la propuesta). Si usted cree que una línea de
   "trabajo futuro" es más honesta que el silencio total, argumente — el
   Imperator decide.

## 8. Registro de intención (interno, no va al PDF)

Para que quede en acta de la Orden y nada se pierda por callarlo en la
propuesta: la intención de largo plazo del Imperator permanece intacta —
DOIN más útil, mejor y multidominio; el filo privado de trading financia el
camino; el mercado de servicios de inferencia/optimización y el agente con
contexto persistente que aprende de su uso son el horizonte. El doctorado
es la pluma; la Orden es el ave. Ninguna línea de la propuesta debe
contradecir ese horizonte, y ninguna debe revelarlo antes de tiempo.

*Este memorando y su anexo son los únicos artefactos creados; quedan sin
comprometer en git a la espera de la palabra del Imperator para el relevo a
Musashi. Ninguna propuesta se redacta hasta que su crítica vuelva.*

— General Satoshi III
