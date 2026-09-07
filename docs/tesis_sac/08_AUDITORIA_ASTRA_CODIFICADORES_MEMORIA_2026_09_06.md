# Auditoría académica de la propuesta de selección secuencial de codificadores de memoria

**Autor de la propuesta:** Harvey Demian Bastidas Caicedo.  
**Programa:** Doctorado en Inteligencia Artificial, Universidad de La Sabana.  
**Auditoría:** Astra, en respuesta a la solicitud fechada el 6 de septiembre de 2026.  
**Alcance:** revisión externa de solo lectura; no se modificaron el PDF, LaTeX, BibTeX ni los repositorios.  
**Versión examinada:** PDF adjunto de 11 páginas, nueve de contenido y dos de referencias.

**Identidad comprobada:** SHA-256 `0860a4f2975042fbf30400d5920443a7ca46415c58ba2e0cedc2ca2886eb470f`. Coincide exactamente con el digest exigido en la solicitud. No hay `DOCUMENT_IDENTITY_MISMATCH`.

Fuentes del encargo: [solicitud de auditoría](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/REQUEST_FOR_AUDIT_ASTRA_PROPUESTA_CODIFICADORES_MEMORIA_2026_09_06.md), [PDF autoritativo](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_representaciones_rl.pdf), [LaTeX](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex) y [BibTeX](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_multifidelidad_rl.bib). Los enlaces apuntan a una rama que puede cambiar; este dictamen corresponde al digest anterior. Los identificadores Git de los contenidos leídos fueron `7726b8a00c4280f71aac3f93d50183b6fae1e123` para LaTeX y `f2804e741dafa45021f7f2123847a29eb58105a1` para BibTeX.

Convención: **comprobado** identifica contenido del documento, cálculos o resultados bibliográficos verificados; **inferencia** identifica una conclusión del auditor; **pendiente** identifica información que no existe aún o que no pudo verificarse. Los números de página se refieren al PDF identificado. Las referencias [1]–[30] conservan la numeración de ese PDF.

## 1. Veredicto global

**`REVISE` — conservar la propuesta y cerrar las correcciones metodológicas antes del envío.**

No se identificó un P0 que obligue a abandonar la pregunta doctoral. Se identificaron **seis grupos P1** que afectan la interpretación de las garantías, las variables de evaluación o la fuerza de la comparación. Son reparables sin incorporar otra tesis ni ejecutar ahora la matriz experimental completa.

El documento mejoró sustancialmente. H1 ya no aprovecha las abstenciones para excluir tareas; la condición de imposibilidad incluye conjuntos epsilon-óptimos disjuntos; la contabilidad distingue inversión y uso; el título y la trayectoria son más precisos. Sin embargo, la construcción que debería producir la garantía secuencial sigue sin especificarse suficientemente, y el experimento aún no determina de manera única cómo estimará el riesgo cuando sus valores de referencia sean inciertos.

Este dictamen evalúa la madurez de una **propuesta**: no exige que los teoremas estén demostrados ni los resultados obtenidos antes de la admisión. Sí exige una ruta metodológica plausible, supuestos reconocibles y criterios de éxito que no puedan cambiar de significado durante el estudio.

### 1.1. Estado de las correcciones anteriores

| Corrección solicitada | Estado | Comprobación y remanente |
|---|---|---|
| Separar cobertura, parada adaptativa y riesgo selectivo | Parcial | Las definiciones y la ecuación (2) mejoraron; falta conectar la construcción de intervalos con la garantía y precisar si es marginal entre tareas o condicional para cada tarea. P1-01. |
| Evitar sesgo por abstención en H1 | Cerrada en lo esencial | H1 y §4.4 evalúan la variante obligada sobre todas las tareas. No debe atribuirse automáticamente ese ahorro a la variante selectiva. |
| Reconocer el soporte insuficiente de quince entornos | Parcial | La cuenta 15/59 es correcta y está bien presentada como ilustración. Falta dimensionar calibración, dependencia de pliegues, aceptación y presupuesto total. P1-03. |
| Corregir la imposibilidad | Cerrada en lo esencial | Indistinguibilidad y conjuntos epsilon-óptimos disjuntos permiten la cota 1/2 bajo las condiciones explicitadas abajo. P2 de precisión formal, no una objeción fatal. |
| Separar costos previos y marginales | Parcial | La ecuación de amortización es correcta bajo costos marginales esperados comparables. Faltan la unidad primaria y la atribución del corpus histórico. P1-06. |
| Precisar título y trayectoria | Cerrada | El título identifica el objeto y §5.1 presenta preparación sin usarla como evidencia de H1–H3. Quedan ajustes editoriales menores. |

## 2. Resumen ejecutivo

La propuesta conserva una pregunta doctoral pertinente y acotada: decidir entre codificadores de memoria mediante evidencia parcial, contabilizando costo y permitiendo abstención. La nueva versión corrige de forma real el sesgo de H1 y la formulación de imposibilidad. La presentación es legible y la trayectoria respalda factibilidad técnica.

El principal problema pendiente es metodológico: calibrar residuos y mencionar secuencias de confianza no establece todavía intervalos válidos para retornos finales esperados, bajo consultas adaptativas y cambio entre tareas. También falta definir la normalización del arrepentimiento y el tratamiento de recomendaciones cuyo error no pueda clasificarse con las semillas de referencia.

La cuenta de 15/59 es correcta, pero no dimensiona la calibración ni convierte los resultados de validación cruzada en decisiones binomiales independientes. H2 alterna entre rechazo e inconclusión ante falta de precisión. Deben fijarse los estimandos comparativos, la secuencia de congelación y una contabilidad escalar coherente para la amortización.

Se localizaron las treinta referencias. Hay una omisión de autor en BibTeX y antecedentes cercanos ausentes, especialmente ifBO y métodos de selección con costo y transferencia. No se encontró evidencia suficiente para declarar redundante toda la contribución.

Recomiendo corregir los seis P1 mediante las sustituciones mínimas propuestas, mantener el tema y proceder después a la lectura final.

## 3. Hallazgos P0

**Ninguno confirmado.**

La ecuación (2) se presenta como una propiedad que se buscará; no como un teorema ya demostrado. Por eso la falta de una construcción cerrada se clasifica P1, no como demostración falsa P0. Se convertiría en un problema invalidante si se afirmara que la sola calibración de residuos garantiza (2) para cualquier tarea y cualquier estrategia de consulta.

Tampoco se identificó una fuente inexistente que sostenga el núcleo, una filtración efectiva de resultados de prueba ni una incompatibilidad demostrada que haga inviable el experimento. Se identifican ambigüedades que podrían producir esos problemas si no se resuelven.

## 4. Hallazgos P1: necesarios antes del envío

### P1-01. La cobertura secuencial está bien expresada, pero su construcción y su alcance siguen abiertos

**Localización:** p. 4, §3.3; p. 5, §3.5, ecuación (2); p. 7, §4.3; p. 9, contribución 2.  
**Tipo:** omisión comprobada; consecuencia matemática e inferencia metodológica.

El documento propone calibrar residuos de un modelo jerárquico y elegir entre secuencias de confianza o consultas finitas con corrección simultánea. Estas opciones no son intercambiables sin precisar el objeto aleatorio y sus supuestos:

1. Las observaciones de una curva parcial no son, en general, nuevas muestras insesgadas de una misma media final. Comparten entrenamiento, semillas y sesgo de fidelidad. Una secuencia de confianza para medias no puede aplicarse directamente a esos puntos como si fuesen observaciones independientes de V_t(c).
2. Corregir por cinco candidatos y varios tiempos de consulta soluciona multiplicidad únicamente si los intervalos que se corrigen son válidos para las variables y las reglas de consulta correspondientes. La corrección no elimina sesgo de extrapolación ni invalidez inducida por selección de la evidencia.
3. La calibración conforme entre tareas puede producir una garantía marginal sobre una tarea nueva intercambiable. No se convierte automáticamente en cobertura condicional para cada entorno particular. La frase «garantía por tarea» en §3.5 deja esa distinción abierta.
4. Si se calibra contra medias de cinco semillas, el objetivo observado es una estimación ruidosa. Cubrir esa media observada no equivale sin ajuste a cubrir V_t(c), que el documento define como esperanza.

**Corrección mínima:** declarar una ruta principal y el alcance de su garantía. Una ruta razonable conserva un modelo jerárquico para predicción/adquisición y calibra la trayectoria completa de una política de consulta congelada en una malla finita, con un puntaje simultáneo por tarea. La adquisición se fija antes de calibrar: el umbral calibrado puede truncar esa trayectoria, pero no cambiar retrospectivamente qué evidencia se habría adquirido. Si se permite ese cambio, la garantía debe cubrir también la selección entre políticas. Debe incorporar la incertidumbre de las referencias y precisar que la cobertura es marginal entre tareas bajo intercambiabilidad. Una alternativa basada en secuencias de confianza exige declarar su modelo de sesgo y ruido. No hace falta demostrarla en la propuesta; sí indicar qué resultado habilitará la afirmación de cobertura y qué afirmación se retirará si no se obtiene.

La condición (3) es válida **sobre** el evento de cobertura; esa implicación algebraica ya está cerrada. Lo pendiente es justificar la probabilidad del evento con el procedimiento propuesto.

**Fuentes:** [Howard et al., 2021](https://arxiv.org/abs/1810.08240); [Conformal Risk Control](https://arxiv.org/pdf/2208.02814); [Poiani et al., 2024, §1.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc9e095f668044e7a0909a4ea3926beb-Paper-Conference.pdf). El último trabajo supone cotas de sesgo conocidas; no es lícito sustituirlas por estimaciones históricas sin analizar el error de esa sustitución.

### P1-02. El riesgo aún no es una variable observable inequívoca

**Localización:** p. 3, §3.1; pp. 3–4, H2 y §3.3; p. 8, §4.4.  
**Tipo:** ambigüedades comprobadas.

Hay tres decisiones pendientes:

**Escala.** La ecuación (1) define arrepentimiento en unidades de retorno y el texto añade que se normalizará. No define la transformación ni aclara si epsilon se aplica antes o después de normalizar. Un epsilon compartido entre tareas no tiene significado uniforme si sus escalas de recompensa difieren.

**Referencia incierta.** Se indica que un orden no distinguible se reportará inconcluso. Pero desconocer el ganador exacto no implica desconocer si una recomendación está dentro de epsilon. La clasificación debe depender de un intervalo para el arrepentimiento, no de un test de empate entre los dos mejores. Falta indicar si los casos inconclusos permanecen en el denominador de riesgo. Excluirlos después de recomendar puede producir una tasa artificialmente baja.

**Ponderación.** Un entorno tiene varias dificultades o contextos. Debe definirse si se evalúa una recomendación primaria por entorno o un riesgo sobre casos con pesos predeterminados por entorno. Agrupar el bootstrap no define por sí mismo el estimando ni impide que un entorno con más casos domine la tasa.

**Corrección mínima:** fijar una escala positiva por tarea obtenida mediante una regla publicada, sin calibrarla con los resultados finales; definir la unidad y ponderación primarias; mantener todas las recomendaciones en el denominador. Reportar límites inferior y superior compatibles con las referencias inciertas. Para sostener un riesgo pequeño, los casos no resueltos no pueden contarse gratuitamente como éxitos.

Una definición operativa posible es R_t^N = [max_c V_t(c) − V_t(c_hat)]/s_t, con s_t > 0 y regla de s_t fijada antes de prueba. Se usaría el mismo umbral epsilon en H2, H3 y la condición de recomendación expresada en esa escala. Si el intervalo del arrepentimiento cruza epsilon, el caso permanece inconcluso, pero sigue contando como recomendación.

**Fundamento:** definición de arrepentimiento del propio PDF y razonamiento estadístico del auditor; las recomendaciones de [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264) no sustituyen una definición específica del estimando de esta tesis.

### P1-03. La precisión de H2 debe contemplar calibración, dependencia y aceptación, además de la cuenta 15/59

**Localización:** p. 6, §4.1; p. 8, §5.2; p. 9, §5.4.  
**Tipo:** cuenta verificada; limitaciones de diseño inferidas.

La cuenta binomial está bien calculada. Sin embargo:

- Las 59 recomendaciones se refieren al caso ideal de decisiones independientes, un selector fijado y cero errores, a un nivel unilateral nominal del 95 %. No incluyen las tareas usadas para desarrollar y calibrar el método, ni la pérdida de soporte por abstención, ni una corrección adicional de multiplicidad.
- Con validación cruzada, las predicciones fuera de muestra proceden de selectores distintos cuyos conjuntos de ajuste se solapan. Mantener cada entorno fuera de su propio ajuste evita una filtración directa, pero no hace independientes todos los errores agregados. El ajuste de la banda de arquitecturas y de las recetas también debe respetar la partición externa.
- POPGym es una colección de entornos diseñada. Agrupar por entorno es una buena elección de unidad; no demuestra automáticamente que la colección sea una muestra intercambiable de cualquier población de tareas futuras.
- Hay un límite adicional en la calibración. En split conformal ordinario, el orden del cuantil es ceil((m+1)(1−alpha)). Con m=14 tareas de calibración y alpha=0,05, el orden es 15: excede los 14 puntajes disponibles y la construcción convencional usa el límite infinito. Es una limitación de esa construcción sin supuestos adicionales, no una imposibilidad de todo método de calibración. En pliegues de quince entornos, el número real de tareas de calibración será aún menor si se reserva desarrollo.

**Corrección mínima:** separar desarrollo con pliegues agrupados de la evidencia que se utilizará para la certificación final; precisar población y ponderación; dimensionar conjuntamente desarrollo, calibración y evaluación. Si se mantiene una evaluación totalmente cruzada, identificarla como estimación del procedimiento y no aplicar automáticamente una certificación binomial para un selector fijo.

La expansión a otro banco debe ser condicional a un censo y al presupuesto. Un banco adicional no garantiza 59 unidades comparables. Si no se obtiene soporte para la afirmación elegida, reportar riesgo–cobertura con su incertidumbre y dejar la certificación exigente inconclusa. No es necesario incorporar otro dominio de investigación.

**Fuente del cuantil:** [Conformal Risk Control, §2.3](https://arxiv.org/pdf/2208.02814). La dependencia por ajustes compartidos y sus consecuencias son un análisis del diseño, no un resultado atribuido a ese artículo.

### P1-04. Deben unificarse los criterios confirmatorios y la secuencia de congelación

**Localización:** pp. 2–3, §2.3; p. 6, §4.1; p. 7, §4.3; p. 8, §4.4; p. 9, §§5.2 y 5.4.  
**Tipo:** contradicción y omisiones comprobadas.

**Contradicción explícita:** H2 se rechaza si falta soporte para estimar sus límites; §4.1 dice que H2 se declarará inconclusa. Falta de precisión no demuestra el fracaso sustantivo de un método. Deben distinguirse evidencia favorable, evidencia contraria e insuficiencia de evidencia.

**Contraste de H2:** falta escribir el riesgo del rival. En los casos donde la variante selectiva recomienda y la obligada sigue exactamente la misma trayectoria y regla de salida, ambas pueden recomendar lo mismo. No puede exigirse una reducción estricta de riesgo sobre ese mismo subconjunto cuando las decisiones coinciden. El contraste natural es el riesgo entre aceptadas frente al riesgo de la variante obligada sobre todas las tareas; debe presentarse como efecto de selección y acompañarse de cobertura y costo.

**Parámetros estadísticos:** delta es el límite de riesgo; delta_int es error de cobertura de intervalos; falta distinguirlos del nivel alfa de confianza/pruebas. La cobertura debe tener un criterio inferencial, además de un valor puntual. El procedimiento de Holm necesita p-valores válidos para hipótesis bien definidas; mencionar Holm no define las pruebas de no inferioridad y los contrastes conjuntos.

**Cronología:** el año 1 congela el protocolo; el año 2 desarrolla y calibra el selector. Esto puede ser correcto si el primer registro congela el diseño y después se fija la regla antes de prueba. Actualmente no se distingue ese registro de la congelación final. También se debe indicar si el comparador principal se elige en desarrollo con un criterio lexicográfico de desempeño/costo fijado, y qué umbrales se seleccionan exclusivamente en calibración.

**Corrección mínima:** definir H1 como conjunción de ahorro y no inferioridad; H2 con dos riesgos explícitos y límites de riesgo/cobertura; H3 con el mismo contrato y soporte propio. Fijar márgenes de relevancia práctica antes de optimizar los umbrales para satisfacerlos. Identificar registro inicial, desarrollo, calibración y congelación confirmatoria final. El piloto puede informar factibilidad y precisión; no debe permitir escoger un margen solo porque el método lo supera.

**Fuentes:** [Holm, 1979](https://www.jstor.org/stable/4615733); [Selective Conformal Risk Control](https://arxiv.org/abs/2512.12844). La interpretación de la contradicción y la igualdad de decisiones es un análisis del auditor.

### P1-05. Falta un control histórico simple y un competidor directo de adquisición de curvas

**Localización:** pp. 4–5, §3.4; p. 7, §4.4.  
**Tipo:** ausencias comprobadas; selección de comparadores recomendada.

Con cinco candidatos congelados, el primer control necesario es el **mejor codificador fijo elegido solo en desarrollo**. Puede ser muy competitivo sin comprar curvas en la tarea nueva. Su costo previo y su arrepentimiento deben medirse. Sin este control, una reducción de costo frente a métodos de búsqueda no demuestra que la selección adaptativa aporte valor frente a una decisión histórica simple.

Hyperband, ASHA y BOHB son comparadores pertinentes. Pero LC-PFN es principalmente un predictor de curvas: necesita una regla de adquisición/parada para ser un método completo de selección. Existe un vecino más directo: **ifBO**, con FT-PFN y adquisición incremental, publicado en ICML 2024. También existen DyHPO y un trabajo sobre optimización multifidelidad sensible al costo con transferencia de curvas. Este último ya combina costo, experiencia histórica y parada; debe discutirse para que esa combinación no se presente como novedad propia.

**Corrección mínima:** añadir el control fijo, que no requiere nuevas curvas, y considerar ifBO como candidato principal en la familia de comparadores de curvas ya prevista, sustituyendo o justificando la elección de LC-PFN con su regla completa. No es necesario implementar todos los trabajos mencionados ni aumentar la matriz de entrenamiento para cada uno.

La abstención calibrada tampoco basta por sí sola como novedad en 2026. Los trabajos recientes de control selectivo deben delimitarse respecto de la adquisición de evidencia de RL. No encontré evidencia suficiente para concluir que un único método publicado resuelva toda la intersección propuesta. La búsqueda realizada es dirigida, no una revisión sistemática exhaustiva.

**Fuentes:** [ifBO, ICML 2024](https://proceedings.mlr.press/v235/rakotoarison24a.html), [DyHPO](https://arxiv.org/abs/2202.09774), [Lee et al., transferencia y costo](https://arxiv.org/abs/2405.17918), [Bai y Jin, SCoRE](https://arxiv.org/abs/2603.24704), [Yu y Liu, certificado conjunto](https://arxiv.org/abs/2606.08517). Los últimos trabajos se usan como antecedentes y no como garantías automáticamente transferibles a esta tesis.

### P1-06. La amortización necesita una unidad de costo y una frontera experimental explícitas

**Localización:** pp. 7–8, §4.4 y ecuación (4); p. 8, §5.2.  
**Tipo:** ecuación correcta bajo condiciones; contabilidad incompleta.

La ecuación (4) es correcta para costos marginales esperados comparables y diferencia marginal positiva a favor del selector. No se recomienda sustituirla por otra fórmula innecesariamente. Sí faltan cuatro condiciones:

1. **Unidad escalar:** se informan GPU-horas y CPU-horas por separado, lo cual es correcto, pero H1 y N* requieren una definición de C. Puede ser tiempo bajo una configuración fija de recursos o un costo monetario con tarifas explícitas. No se suman CPU-horas y GPU-horas sin un modelo de costo.
2. **Corpus histórico:** las semillas ocultas de prueba pueden ser costo experimental común. La evidencia histórica necesaria para entrenar un selector tiene un costo de adquisición previo. Si se asume que el corpus ya existe gratuitamente, esa es una modalidad de amortización condicionada a un recurso disponible; debe diferenciarse de construirlo desde cero.
3. **Variante y desempeño:** N* debe calcularse para una variante identificada y a calidad comparable. No se puede justificar el ahorro total de una variante abstencionista solo con no inferioridad demostrada para la variante obligada. Si el ahorro marginal estimado no se distingue de cero, N* no es una cifra precisa.
4. **Frontera de fidelidad/reproducción:** fijar qué significa 100 % —por ejemplo, transiciones de entorno— y cobrar el tiempo medido. La matriz debe permitir continuar cada candidato-semilla sin alterar la receta. Al reanudar se cobra el tramo incremental más los costos reales de consulta y recuperación; la reproducción de ASHA requiere además una política común de trabajadores y tiempos si se evalúa tiempo de pared.

El techo 5 × 45 × 10 = 2.250 curvas es correcto para POPGym. La versión revisada ya no pretende que sea el techo total, pero el segundo banco y CARL quedan sin límite. Para una propuesta de tres años basta una compuerta explícita de factibilidad y una fórmula total que incluya todos los bancos y ajustes; no se requiere inventar GPU-horas antes del piloto.

**Corrección mínima:** definir la unidad primaria y las variantes; incluir adquisición de historia cuando corresponda; reportar incertidumbre o escenarios de amortización; hacer de la ampliación de bancos una decisión previa a la prueba sometida a un techo total y una alternativa de alcance científico.

## 5. Hallazgos P2: mejoras que no justifican bloquear por sí solas

| ID | Página/sección | Observación y cambio mínimo |
|---|---|---|
| P2-01 | p. 1, resumen | El título coincide con el objeto. «Bajo cambio de tarea» puede leerse como cambio dentro de un episodio; «en tareas no vistas» resulta más directo si se desea acortarlo. Es opcional. |
| P2-02 | p. 1, resumen | Introducir «semilla aleatoria» como una repetición del entrenamiento con aleatoriedad controlada. La definición operacional completa puede quedar en metodología. |
| P2-03 | p. 2, objetivo específico 2 | «Margen frente a los demás candidatos suficientemente pequeño» puede sonar a empate. Sustituir por «cota superior del arrepentimiento no mayor que epsilon». |
| P2-04 | p. 5, §3.5 | Formular el resultado elemental con dos tareas t0 y t1 y aclarar que la indistinguibilidad debe cubrir la información accesible a la política de consultas considerada. Coincidencia de dos historiales observados no basta. La cota formal de costo, no este lema elemental solo, debe sustentar la contribución teórica. |
| P2-05 | pp. 6–7, §4.2 | La comparación estima el efecto del conjunto codificador–receta bajo restricciones, no un efecto universal de la familia. El texto lo reconoce bien. Especificar horizonte disponible y truncamiento de gradientes por arquitectura antes del piloto definitivo. |
| P2-06 | p. 7, §4.3 | Nombrar un modelo jerárquico inicial y una función de adquisición candidata facilitaría evaluar viabilidad. No hace falta fijar ahora todos los priors ni añadir una familia de modelos nueva. |
| P2-07 | p. 8, §5.1 | El párrafo de trayectoria ya demuestra preparación sin circularidad. Añadir una o dos referencias a repositorios/artefactos públicos basta; no hace falta una lista larga de módulos. |
| P2-08 | pp. 8–9, §§4.4–5.4 | «No se generaliza», «no convertirá en positivo», «sin conversiones inventadas» y varias cláusulas semejantes pueden concentrarse en un único párrafo de integridad experimental. Preservar las reglas, reducir la reiteración. |
| P2-09 | pp. 10–11, referencias | Corregir en BibTeX el autor omitido de ASHA y actualizar su URL a la página oficial comprobada. Añadir volumen/artículo de ARLBench; especificar versión del preprint [21] si se utiliza su construcción de 2026. Detalles en §7. |
| P2-10 | p. 7, §4.4 | Si se elige SMAC-HB, incorporar la referencia de SMAC3 y la configuración de intensificación usada. No atribuir SMAC a la referencia de BOHB. |
| P2-11 | p. 1 y p. 9 | El resumen tiene alrededor de 400 palabras y funciona parcialmente como metodología abreviada. Reducirlo a unos 250–300, si las instrucciones del programa lo permiten, deja más visible la pregunta. |
| P2-12 | p. 3, H3; pp. 5–6, §4.1 | Precisar que se congela el meta-selector y sus umbrales, mientras los candidatos se entrenan con la receta prevista en cada nuevo caso. Aclarar qué contextos quedan fuera y que «contexto oculto» no exige que toda observación física sea incompleta. |

### 5.1. Revisión visual y extensión

Se inspeccionaron las once páginas renderizadas. No se observaron tablas desbordadas, ecuaciones cortadas ni texto ilegible. La ecuación de amortización se representa correctamente; los caracteres extraños que aparecen al extraerla como texto son un artefacto de extracción, no un defecto visual del PDF.

Las once páginas son razonables editorialmente para nueve páginas de contenido y treinta referencias. **No se verificó un límite institucional de extensión:** la página pública del programa consultada no especificó una longitud de propuesta. No se debe presentar esta valoración editorial como conformidad con una convocatoria particular. [Programa oficial](https://www.unisabana.edu.co/programas/posgrados/doctorado-en-inteligencia-artificial).

No retiraría íntegramente ninguna página. La página 9 es el mejor lugar para compactar: eliminar la réplica opcional del cronograma si ya no se necesita, integrar «resultado mínimo defendible» en las contribuciones y condensar ética/reproducibilidad. Junto con un resumen menor y la reducción de reiteraciones, esto puede ahorrar espacio sin eliminar las condiciones de validez. Las páginas 4–6 contienen defensas necesarias; no conviene borrarlas para cumplir una meta estética arbitraria.

## 6. Matriz de consistencia

| Elemento | Evidencia prevista | Análisis y falsación | Estado |
|---|---|---|---|
| Pregunta: seleccionar con menor costo, riesgo limitado y abstención | Curvas parciales, referencias ocultas, costos y salidas de las dos variantes | Responder mediante H1 y H2 conjuntamente; H3 determina alcance bajo contexto | Coherente si no se atribuye el éxito de una variante a la otra. |
| Objetivo general: diseñar y evaluar selector de distinto costo | Implementación, protocolo y matriz pública | Evaluación externa y contabilidad completa | Alineado. |
| Objetivo 1: decisión secuencial con evidencia heterogénea | Descriptores, fidelidades, semillas y acciones de consulta | Modelo de observación y costos | Falta el contrato estadístico de P1-01 y la unidad de costo de P1-06. |
| Objetivo 2: recomendar, consultar o abstenerse | Intervalos, regla (3), adquisición y parada | Cobertura simultánea y riesgo–cobertura | La implicación algebraica está bien; construcción y observabilidad pendientes. |
| Objetivo 3: comparar y cambiar contexto | POPGym principal; extensión si procede; CARL separado | Separación por entornos, comparadores y contraste H3 | Deben cerrarse particiones y soporte. |
| H1: eficiencia sin inferioridad | Variante obligada y rivales sobre todas las mismas tareas | Menor costo marginal y menos evaluaciones completas, con no inferioridad del arrepentimiento normalizado | Se corrigió el sesgo por abstención. Faltan escala, control fijo, prueba y reloj. |
| H2: beneficio de abstención | Recomendaciones aceptadas, todas las tareas y variante obligada | Riesgo aceptado, riesgo de la variante obligada, cobertura, costo y límites de confianza | P1-02–04 impiden una lectura operacional única. |
| H3: cambio de contexto | Casos CARL posteriores al ajuste, con selector congelado | Mismos límites explícitos, soporte propio y análisis agrupado | Correctamente empírica; no garantía universal. Debe distinguir falta de precisión de incumplimiento. |
| Resultado formal | Modelo de evidencia y evento (2) | (2)+(3) implican arrepentimiento ≤ epsilon; costo bajo supuestos; indistinguibilidad | Implicación y lema elemental coherentes; garantía y cota de costo son objetivos de investigación. |
| Amortización | Costos previos y marginales de variantes y comparadores | N* bajo diferencia marginal positiva y desempeño comparable | Fórmula correcta; P1-06 cierra sus entradas e interpretación. |
| Cronograma | Año 1 diseño; año 2 desarrollo/evaluación; año 3 contexto/teoría | Dos hitos de congelación explícitos; factibilidad antes de prueba | Ajustar el orden de desarrollo, calibración y prerregistro final. |
| Contribuciones | Selector, análisis formal, banco y evidencia reproducible | Aporte adicional frente a baselines; resultados nulos con alcance delimitado | Mantener. Un método que falla no prueba por sí solo imposibilidad general. |

### 6.1. Comprobación algebraica de la condición de recomendación

Sobre el evento simultáneo E de la ecuación (2), para la recomendación emitida en tau:

\[
V_t(c^*)-V_t(\hat c)
\le U_{t,\tau}(c^*)-L_{t,\tau}(\hat c)
\le \max_c U_{t,\tau}(c)-L_{t,\tau}(\hat c)
\le \varepsilon.
\]

**Resultado:** la implicación es correcta. Si la cobertura vale con probabilidad al menos 1−delta_int en la población declarada, se obtiene un límite para el evento conjunto de recomendar y exceder epsilon. No se obtiene automáticamente el mismo límite para el riesgo condicional entre aceptadas. El texto nuevo ya distingue estos objetos; debe conservar esa distinción al construir los intervalos.

### 6.2. Comprobación de la imposibilidad

Para dos tareas fijas con igual distribución de toda la información observable bajo la política considerada, la distribución Q de su recomendación también es igual. Si sus conjuntos aceptables C_epsilon(t0) y C_epsilon(t1) son disjuntos:

\[
\tfrac12 Q(C_\varepsilon(t_0))+
\tfrac12 Q(C_\varepsilon(t_1))\le\tfrac12.
\]

La abstención no cuenta como una recomendación exitosa. La cota sigue siendo válida para reglas aleatorizadas. Para extender la afirmación a **toda** política de adquisición, debe establecerse indistinguibilidad del experimento interactivo para esa clase de políticas, no solo para una trayectoria elegida. Obtener más evidencia puede romperla; no se presume que siempre lo haga.

**Resultado:** corrección sustantiva aceptada. Es un lema elemental de dos hipótesis; su existencia no demuestra por sí sola que la contribución teórica doctoral ya esté obtenida.

### 6.3. Comprobaciones numéricas y de costo

| Comprobación | Resultado |
|---|---|
| Cero errores, n=15, límite unilateral exacto al 95 % | 1−0,05^(1/15) = 0,18103627. Correcto. |
| Cero errores, n=58 | Límite 0,05033934: no baja de 5 %. |
| Cero errores, n=59 | Límite 0,04950761: sí baja de 5 %. |
| Ejemplo conservador de alfa/3 | Con Bonferroni 0,05/3 harían falta 80 decisiones sin errores. No se afirma que Holm siempre exija 80; depende de los contrastes. |
| POPGym | 5 candidatos × 45 casos × 10 semillas = 2.250 curvas. Correcto. |
| Calibración split conformal m=14 al 95 % | Orden ceil(15×0,95)=15; el cuantil ordinario aumentado resulta infinito. No generalizable a todos los métodos con supuestos adicionales. |

Para costos esperados marginales constantes sobre una población de tareas, comparar C_prev_sel + N C_marg_sel con C_prev_b + N C_marg_b produce la ecuación (4) cuando el selector es más barato marginalmente. El uso de parte positiva evita una inversión adicional negativa; N*=0 indica ausencia de sobrecosto inicial. Es un umbral de igualdad o ventaja acumulada, no necesariamente de ahorro estrictamente positivo en ese mismo entero.

Si los costos cambian sustancialmente con la mezcla de tareas, usar costos esperados sobre una mezcla declarada o comparar las sumas acumuladas. Un denominador incierto que incluye cero impide anunciar un único punto de amortización finito con alta confianza.

## 7. Auditoría de referencias

### 7.1. Alcance y límites de verificación

Se comprobaron las treinta claves de BibTeX frente a las citas del LaTeX: **30 citadas, 0 ausentes, 0 sin uso**. Se localizaron fuentes primarias para las treinta obras. Se verificaron identidad y pertinencia temática; para los resultados centrales se consultaron además los pasajes de supuestos y definiciones disponibles. Esto no equivale a reproducir todas las pruebas de treinta artículos.

OpenReview bloqueó algunas aperturas con una verificación de navegador; en esos casos se utilizaron resultados indexados de la fuente primaria, páginas de autores o versiones de arXiv. Para [11] se verificó existencia, autoría y publicación mediante el registro institucional y las páginas de sus autores; la comprobación detallada del alcance de cada experimento quedó limitada por el acceso al PDF. La apertura directa de Nature para [28] falló, aunque su registro editorial indexado permitió comprobar metadatos. Los DOI de Curran de [10] y [16] no resolvieron en la herramienta: se verificaron las obras en NeurIPS/arXiv, **no se certifica la resolución de esos dos DOI**.

No se utilizaron resúmenes comerciales ni textos de otros modelos como autoridad. Encontrar una cita pertinente no valida automáticamente la adaptación propuesta.

### 7.2. Correspondencia afirmación–referencia

| N.º | Afirmación que debe sostener | Dictamen | Fuente primaria consultada |
|---|---|---|---|
| [1] Morad et al., POPGym | Banco de memoria parcialmente observable con quince entornos y dificultades | Sí. La edición/documentación describe quince; fijar versión del banco. No implica quince tareas aleatorias representativas de todo RL. | [arXiv](https://arxiv.org/abs/2303.01859), [documentación](https://popgym.readthedocs.io/) |
| [2] Benjamins et al., CARL | Cambios controlados de contexto | Sí. TMLR 2023 es correcto. Ocultar el contexto es una elección experimental que debe describirse. | [TMLR/OpenReview](https://openreview.net/forum?id=Y42xVBQusn), [repositorio oficial](https://github.com/automl/CARL) |
| [3] Becktepe et al., ARLBench | Infraestructura y prácticas para HPO/AutoRL y costos | Sí. DMLR 2026 es una publicación real; no corregirla a otra revista por parecer inusual. Completar volumen 3 y registro del artículo. | [PDF editorial](https://data.mlr.press/assets/pdf/v03-3.pdf), [arXiv](https://arxiv.org/abs/2409.18827) |
| [4] Eberhard et al. | Trazas de memoria pueden ser más eficientes que ventanas en ciertos entornos | Sí, con el alcance condicional ya empleado. PMLR 267:14934–14949. | [ICML/PMLR](https://proceedings.mlr.press/v267/eberhard25a.html) |
| [5] Eimer et al. | HPO de RL y separación entre semillas de ajuste y prueba | Sí. PMLR 202:9104–9149. Justifica controles de receta y semillas. | [ICML/PMLR](https://proceedings.mlr.press/v202/eimer23a.html) |
| [6] Li et al., Hyperband | Asignación adaptativa de recursos y parada temprana | Sí. JMLR 18(185):1–52, 2018 es correcto aunque haya preprint anterior. | [JMLR](https://jmlr.org/papers/v18/16-558.html) |
| [7] Li et al., ASHA | Asignación asíncrona y parada temprana | Sí. **BibTeX omite Jonathan Ben-tzur**, presente en la publicación oficial. Actualizar también la URL a la comprobada. | [MLSys 2020](https://proceedings.mlsys.org/paper_files/paper/2020/hash/a06f20b349c6cf09a6b171c71b88bbfc-Abstract.html) |
| [8] Falkner et al., BOHB | Combinar búsqueda modelada y presupuestos tipo Hyperband | Sí. PMLR 80:1437–1446. | [ICML/PMLR](https://proceedings.mlr.press/v80/falkner18a.html) |
| [9] Miao et al., DARTS-RL | Búsqueda diferenciable de arquitecturas en RL | Sí. PMLR 188:20/1–17. No requiere incluirla como baseline si usa otro espacio. | [AutoML/PMLR](https://proceedings.mlr.press/v188/miao22a.html) |
| [10] Adriaensen et al., LC-PFN | Extrapolación de curvas y aceleración en dominios supervisados | Sí. No demuestra calibración ni eficacia en los entornos RL de esta propuesta. DOI no resuelto en esta revisión. | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract.html), [arXiv](https://arxiv.org/abs/2310.20447) |
| [11] Dierkes et al. | Dificultades de predicción de desempeño en RL | Pertinente y existente, EWRL 2025. Apoya una motivación cauta; no demuestra que toda extrapolación entre tareas sea imposible. Verificación de detalles limitada por acceso. | [registro institucional](https://research.uni-hannover.de/en/publications/performance-prediction-in-reinforcement-learning-the-bad-and-the-/), [OpenReview](https://openreview.net/forum?id=L9J6Xmta4J) |
| [12] Audibert et al. | Arrepentimiento simple e identificación del mejor brazo | Sí para el concepto. No define la normalización entre tareas de este proyecto. | [manuscrito del autor](https://sbubeck.com/COLT10_ABM.pdf), [COLT](https://www.learningtheory.org/colt2010/conference-website/presentation/talkSimpleRegret.pdf) |
| [13] Agarwal et al. | Incertidumbre, IQM y perfiles para evaluar RL | Sí. La adaptación a agrupación por entorno y riesgo selectivo requiere el análisis propio. | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html), [artefacto de autores](https://agarwl.github.io/rliable/) |
| [14] Peherstorfer et al. | Intercambio costo/precisión en multifidelidad | Sí. SIAM Review 60(3):550–591. La fidelidad puede introducir sesgo; más fidelidad no implica mejor retorno observado en cada curva. | [SIAM](https://epubs.siam.org/doi/abs/10.1137/16M1082469) |
| [15] Poiani et al., 2022 | Identificación del mejor brazo con fidelidades | Sí. Obra y autores verificados. Es un vecino teórico directo, no solo contextual. | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html) |
| [16] Poiani et al., 2024 | Cotas y asignación de costo multifidelidad | Sí. §1.1 usa costos y cotas de sesgo conocidos, y una familia exponencial para las observaciones. Especificar la diferencia con curvas dependientes. DOI no resuelto. | [PDF NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/dc9e095f668044e7a0909a4ea3926beb-Paper-Conference.pdf) |
| [17] Chow | Decisión con rechazo | Sí para el fundamento de rechazo y error. No cubre adquisición adaptativa de curvas. | [IEEE](https://ieeexplore.ieee.org/document/1054406/), [IBM Research](https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff) |
| [18] Geifman y El-Yaniv | Riesgo–cobertura en clasificación selectiva | Sí. La unidad aquí es una tarea y el error depende de una referencia estimada: adaptación explícita. | [arXiv](https://arxiv.org/abs/1705.08500), [repositorio de autores](https://github.com/geifmany/selective_deep_learning) |
| [19] Howard et al. | Cobertura uniforme en el tiempo bajo supuestos | Sí como antecedente. No basta para convertir residuos de extrapolación en una secuencia válida. 49(2):1055–1080 y DOI correctos. | [Annals of Statistics](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-2/Time-uniform-nonparametric-nonasymptotic-confidence-sequences/10.1214/20-AOS1991.full) |
| [20] Angelopoulos et al. | Control conforme del riesgo | Sí, con condiciones. El resultado básico controla una esperanza con pérdidas monótonas y acotadas; no equivale al certificado de riesgo selectivo con alta probabilidad de H2. | [PDF arXiv, §§2.1–2.4](https://arxiv.org/pdf/2208.02814) |
| [21] Xu et al. | Control del riesgo en predicciones seleccionadas | Sí como antecedente específico. Existen variantes transductiva e inductiva con contratos diferentes; no presentar ambas como un procedimiento intercambiable. La v2 es de abril de 2026. | [arXiv v2](https://arxiv.org/html/2512.12844v2) |
| [22] Papini et al. | Selección de representaciones con estructura lineal | Sí. La condición UNISOFT y el objeto de funciones de valor difieren del selector de codificadores neuronales. | [arXiv](https://arxiv.org/abs/2110.14798), [PDF NeurIPS](https://proceedings.neurips.cc/paper/2021/file/8860e834a67da41edd6ffe8a1c58fa55-Paper.pdf) |
| [23] Zhang et al. | Selección de representaciones en MDP de bajo rango | Sí. PMLR 216:2488–2497; la distinción de alcance del texto es razonable. | [UAI/PMLR](https://proceedings.mlr.press/v216/zhang23c.html) |
| [24] Abdelfattah et al. | Proxies de NAS de muy bajo costo | Sí. «Zero-cost» no significa costo literalmente cero ni fidelidad de retorno. La propuesta los trata correctamente como descriptores. | [ICLR/OpenReview](https://openreview.net/forum?id=0cmMMy8J5q), [arXiv](https://arxiv.org/abs/2101.08134) |
| [25] Shen et al., ProxyBO | Integrar proxies y BO para NAS | Sí. AAAI 37(8):9792–9801 y DOI concordantes. No establece una garantía de transferencia a RL. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26169) |
| [26] Akhauri y Abdelfattah | Transferencia de predictores de exactitud/latencia entre tareas y espacios | Sí. PMLR 224:23/1–23. No equivale a seleccionar memoria en POMDP con abstención. | [AutoML/PMLR](https://proceedings.mlr.press/v224/akhauri23a.html) |
| [27] Schulman et al., PPO | Algoritmo principal de política | Sí. Identidad y autores concordantes. No garantiza comparabilidad de cualquier implementación recurrente. | [arXiv](https://arxiv.org/abs/1707.06347) |
| [28] Mnih et al., DQN | Método de valores para sensibilidad | Sí. Nature 518:529–533 y DOI verificados en registro editorial indexado; acceso directo al artículo falló. | [Nature](https://www.nature.com/articles/nature14236) |
| [29] Fan et al., iMFBO | Múltiples fuentes con fidelidad dependiente de la entrada | Sí. PMLR 244:1271–1293. La inclusión condicional es razonable; no confundir iMFBO con ifBO. | [UAI/PMLR](https://proceedings.mlr.press/v244/fan24a.html) |
| [30] Holm | Control de multiplicidad | Sí. Requiere p-valores válidos para las hipótesis; no corrige un estimando ambiguo ni una cobertura secuencial inválida. | [artículo original en JSTOR](https://www.jstor.org/stable/4615733) |

### 7.3. Vecinos ausentes y efecto sobre la novedad

| Trabajo | Relevancia | Acción proporcionada |
|---|---|---|
| [ifBO / FT-PFN, ICML 2024](https://proceedings.mlr.press/v235/rakotoarison24a.html) | Predice curvas y decide qué candidato continuar; muy cercano a la compra secuencial de evidencia | Incluir en revisión y considerar como comparador de la familia ya prevista. |
| [DyHPO, NeurIPS 2022](https://arxiv.org/abs/2202.09774) | Carrera dinámica de configuraciones con curvas y presupuestos | Citar como antecedente; elegir uno de estos comparadores fuertes si ambos resultan redundantes experimentalmente. |
| [Lee et al., 2024](https://arxiv.org/abs/2405.17918) | Une transferencia de curvas, costo, utilidad y parada | Evitar reclamar esa combinación general como novedad. La adaptación a codificadores RL y riesgo selectivo sigue siendo una pregunta distinta. |
| [Bai y Jin, 2026](https://arxiv.org/abs/2603.24704) | Decisiones selectivas con control de riesgo general mediante e-values | Antecedente de riesgo selectivo; diferenciar los objetos de riesgo antes de reutilizar un teorema. |
| [Yu y Liu, 2026](https://arxiv.org/abs/2606.08517) | Certificación conjunta de riesgo, aceptación y utilidad con selección de umbrales finitos | Discutir relación con H2. Es un preprint; no es obligatorio adoptar su método ni dar sus resultados por auditados aquí. |
| [Fannjiang y Park, 2025](https://arxiv.org/abs/2503.20767) | Selección fiable de algoritmos de diseño guiado por ML | Vecino de selección con incertidumbre; no se verificó equivalencia con el objeto específico de la tesis. |
| [Conformal Policy Control, 2026](https://arxiv.org/abs/2603.02196) | Regula cambios de una política respecto de una referencia | Distinguirlo: trata comportamiento de la política; esta tesis abstiene sobre una recomendación de arquitectura. No añadirlo como experimento obligatorio. |
| [Guías COSEAL, 2025](https://arxiv.org/abs/2512.16491) | Buenas prácticas de selección, configuración y evaluación de meta-algoritmos | Referencia metodológica para baselines, particiones y costos. |
| [SMAC3, JMLR 2022](https://jmlr.org/papers/v23/21-0888.html) | Implementación citada como posible baseline sin referencia propia | Incorporar solo si se retiene SMAC-HB. |

La revisión dirigida no demostró redundancia de la tesis. Sí muestra que su novedad debe centrarse en **la relación entre evidencia parcial de codificadores de memoria, adquisición y decisión selectiva bajo generalización entre tareas**, no en ensamblar cuatro componentes presentados como nuevos por separado.

## 8. Prueba de lectura de los tres jurados

### 8.1. Profesora de aprendizaje automático y estadística

**Lo que puede explicar después de leer:** se aprende un predictor de retorno final a partir de otras tareas y se consulta evidencia hasta certificar una recomendación o abstenerse. H1 mide eficiencia de la versión obligada; H2 la calidad de las recomendaciones aceptadas; H3 su comportamiento ante contexto nuevo.

**Objeción principal:** «Veo la propiedad de cobertura que se desea. No veo aún qué aleatoriedad cubre, cómo se preserva al escoger evidencia ni cómo se trata la incertidumbre de la respuesta usada para calibrar. Tampoco puedo reproducir el denominador y los pesos de H2 con la descripción actual».

**Dictamen de este perfil:** REVISE por P1-01–04. No necesita una demostración terminada para admitir la propuesta; necesita una ruta principal y límites que impidan presentar una validación empírica como una garantía universal.

### 8.2. Investigador de RL y AutoRL

**Lo que encuentra sólido:** POPGym es pertinente; PPO principal y DQN opcional mantienen el alcance; cinco candidatos y referencias ocultas permiten comparación controlada. Separar recetas de entrenamiento por familia con igual presupuesto de ajuste es defendible.

**Objeción principal:** «¿Por qué necesito este selector si un codificador fijo elegido en desarrollo es suficiente? ¿Qué aporta respecto de ifBO o una carrera de curvas con transferencia? ¿Qué se cobra al reanudar una trayectoria y cuál es el presupuesto de entrenamiento que define una fidelidad?».

**Dictamen de este perfil:** REVISE por P1-05–06. La matriz compartida es una buena decisión y evita multiplicar el costo por número de selectores; no hace falta ampliar las familias de arquitecturas.

### 8.3. Integrante de admisiones competente en IA

**Lo que entiende:** escoger una memoria puede costar más de lo necesario; se quiere aprender cuándo basta una prueba corta y cuándo no se debe recomendar. El candidato ya ha construido sistemas que hacen plausible la implementación.

**Lo que dificulta una lectura única:** abundancia de siglas, margen ambiguo del objetivo 2, garantías estadísticas que parecen resueltas y luego vuelven a quedar abiertas, segundo banco sin tamaño/costo y doble uso de «rechazar» e «inconcluso».

**Dictamen de este perfil:** tema apropiado, alcance razonable con compuerta de factibilidad. Compactar el resumen y distinguir promesa científica, supuesto y criterio experimental. La admisión no debe depender de que H1–H3 resulten positivas; sí de que el diseño permita aprender algo identificable si fallan.

### 8.4. Respuestas trazables a las 25 preguntas obligatorias

| N.º | Respuesta |
|---|---|
| 1 | Sí: el título coincide con codificadores de memoria y selección secuencial. «Tareas no vistas» es una alternativa editorial más clara. |
| 2 | Fidelidad, codificador y abstención se entienden pronto. Tarea/caso/semilla se precisan después; arrepentimiento normalizado y escala requieren P1-02. |
| 3 | La intuición de H1–H3 es comprensible. Sus criterios operativos no están completamente cerrados; P1-04. |
| 4 | Casi no hay vocabulario interno obligatorio. Persisten frases defensivas y siglas que pueden reducirse; §5. |
| 5 | La ecuación sí cubre candidatos y momentos hasta la parada. No se ha especificado aún la construcción que asegura esa cobertura; P1-01. |
| 6 | La separación conceptual mejoró. «Por tarea» debe aclararse como garantía marginal o condicional bajo supuestos; P1-01. |
| 7 | Sí: H1 compara la versión obligada sobre todas las tareas. Corrección aceptada. |
| 8 | Los denominadores abstractos son correctos. Falta ponderación de variantes y tratamiento de referencia inconclusa; P1-02–03. |
| 9 | Sí: cálculo correcto e ilustrativo. No sustituye el análisis agrupado ni la planificación de calibración. |
| 10 | Dificultades y semillas permanecen agrupadas en intención. Falta reflejar dependencia de entrenamiento entre pliegues y ponderación de casos; P1-03. |
| 11 | Sí: H3 no afirma garantía universal. Debe conservar soporte y criterios propios; P1-04. |
| 12 | Sí: demostración en §6.1, con escala coherente y sobre el evento declarado. |
| 13 | Sí en el caso de dos tareas indistinguibles con conjuntos disjuntos y priori equiprobable. Precisar políticas interactivas; §6.2. |
| 14 | El lema es alcanzable pero elemental; la cota de costo y construcción son investigación real. Definir un modelo principal evita que la promesa sea vacía o excesiva. |
| 15 | La relación con MF-BAI está reconocida con justicia. Añadir la diferencia concreta: cotas de sesgo conocidas frente a estimadas, y dependencia de curvas. |
| 16 | Ecuación correcta condicionalmente. Costo histórico, unidad y variante requieren P1-06. |
| 17 | Las 2.250 curvas corresponden solo a POPGym. El total no está aún cerrado; P1-03 y P1-06. |
| 18 | Sí: POPGym memoria, CARL contexto, ARLBench instrumentación/costo. No exigir migrar todo a ARLBench. |
| 19 | Sí como comparación de candidatos–recetas bajo reglas comunes. No como efecto puro universal de arquitectura; P2-05. |
| 20 | Falta el control fijo y un competidor de curvas con adquisición completo. Antecedentes adicionales en §7.3. No se demostró redundancia total. |
| 21 | Frases señaladas en P1 y P2; las sustituciones mínimas aparecen en §9. |
| 22 | Sí: la evidencia preliminar ya cumple esta función. Repositorios respaldan plausibilidad, no hipótesis. |
| 23 | Correspondencia general buena; la cronología y el significado de falsación requieren P1-04. Matriz en §6. |
| 24 | Treinta obras localizadas; ASHA tiene un autor omitido, algunos metadatos incompletos y dos DOI sin resolución verificada. Alcance de [11] parcialmente verificado. |
| 25 | Once páginas son razonables editorialmente; límite institucional no verificado. Compactar p. 9 y resumen, no eliminar una página metodológica completa. |

## 9. Texto de reemplazo para cada P1

Estas son propuestas de edición para Musashi y el autor. No constituyen cambios aplicados. Se cita solo el fragmento que permite localizar el problema y se recomienda una sustitución local. El texto final debe mantener las mismas decisiones en resumen, hipótesis, método y contribuciones; no basta añadir una nota al final.

### 9.1. P1-01: construcción y alcance de cobertura

**Original, p. 4, §3.3:**

> Sus residuos se calibrarán en tareas distintas para obtener intervalos simultáneos sobre los candidatos y los momentos en que la regla puede detenerse.

**Reemplazo mínimo sugerido:**

> La ruta principal calibrará, en tareas separadas, un puntaje simultáneo sobre candidatos y sobre la trayectoria completa de una política de consulta congelada en una malla finita de presupuestos. La política de adquisición se fijará antes de calibración; el umbral calibrado podrá detenerla, pero no cambiar la trayectoria subyacente. El procedimiento deberá incorporar la incertidumbre de las referencias de retorno final. La cobertura buscada será marginal para una tarea nueva bajo los supuestos de intercambiabilidad declarados; no se interpretará como cobertura condicional para cualquier entorno. La validez frente a la adquisición y parada adaptativas será un resultado a establecer, no una consecuencia de corregir varios intervalos aislados. Si no se obtiene una construcción justificable, se limitará la afirmación a calibración empírica y se reportará esa restricción.

**Ajuste asociado, p. 5, §3.5:** cambiar «Esta garantía por tarea» por «Esta garantía simultánea para una tarea nueva, en la población y bajo los supuestos declarados». En la ecuación (2), declarar si la probabilidad integra la tarea, la calibración y las semillas, o si es condicional al ajuste. La elección de una ruta de secuencias de confianza en lugar de la anterior exige su propio contrato de sesgo y ruido.

### 9.2. P1-02: escala y referencias inconclusas

**Original, p. 3, §3.1:**

> normalizado dentro de cada tarea antes de agregar resultados [...] Una recomendación será perjudicial cuando R_t > epsilon [...] un orden que no pueda distinguirse con las semillas disponibles se reportará como inconcluso.

**Reemplazo mínimo sugerido:**

> Para comparar tareas se dividirá el arrepentimiento por una escala positiva s_t cuya regla se fijará antes de la prueba. La recomendación será perjudicial si el arrepentimiento normalizado supera epsilon; la condición de recomendación usará la misma escala. Las semillas ocultas producirán un intervalo para ese arrepentimiento: un caso será compatible con perjuicio o no perjuicio según su posición respecto de epsilon, y será inconcluso si lo cruza. Todas las recomendaciones, incluidas las inconclusas, permanecerán en el denominador. El límite de riesgo que se use para certificar H2 incorporará conservadoramente los casos no resueltos y la incertidumbre de referencia. La unidad primaria y los pesos de dificultades y contextos se fijarán antes de calibración.

**Nota de implementación conceptual:** puede simplificarse H2 declarando un caso primario por entorno base y dejando el resto como sensibilidad anidada. Si se conservan varios casos primarios, publicar la fórmula ponderada del numerador y del denominador. Ninguna opción convierte las semillas en tareas.

### 9.3. P1-03: particiones y soporte

**Original, p. 6, §4.1:**

> Los intervalos y las pruebas estadísticas se calcularán al nivel de los quince entornos base, con dificultades y semillas anidadas.

**Reemplazo mínimo sugerido:**

> El entorno base será la unidad de agrupación. Se declarará la población de tareas a la que se pretende generalizar y no se supondrá que los entornos del banco constituyen automáticamente una muestra intercambiable. Los pliegues agrupados servirán para evaluar el procedimiento sin que una variante de un entorno entre en su propio ajuste; sus resultados no se tratarán como decisiones binomiales independientes de un único selector fijo. La certificación final de riesgo utilizará tareas separadas del ajuste y la calibración bajo el contrato de inferencia elegido. Si no hay soporte para esa certificación, se informará el resultado como estimación empírica con sus límites y H2 quedará inconclusa en su componente de certificación.

**Inserción después de la cuenta 15/59:**

> Estas cifras no incluyen las tareas de desarrollo y calibración, la fracción de abstenciones, la incertidumbre de referencia ni los ajustes de multiplicidad. El piloto dimensionará esas necesidades por separado y verificará que la construcción de intervalos no resulte trivial por falta de tareas de calibración. La ampliación de bancos solo se conservará si satisface simultáneamente el soporte científico y el techo total de recursos.

La alternativa de mantener toda la evaluación cruzada es válida como diseño de investigación, pero exige declarar qué inferencia sobre el procedimiento puede sostener; no debe añadirse a ese diseño una certificación para un selector fijo sin justificación.

### 9.4. P1-04: H2 y decisiones confirmatorias

**Original, p. 3, H2:**

> Frente al mismo selector obligado a decidir, la versión con abstención reducirá el riesgo condicional [...] H2 se rechazará [...] si el soporte independiente no permite estimarlos.

**Reemplazo mínimo sugerido:**

> H2 comparará el riesgo entre las recomendaciones emitidas por la variante selectiva con el riesgo de la variante obligada sobre todas las tareas, e informará conjuntamente cobertura y costo. Se considerará respaldada si se demuestra la reducción predefinida, el límite superior unilateral del riesgo selectivo no supera delta y el límite inferior de cobertura alcanza gamma, con el nivel de confianza y el ajuste de multiplicidad establecidos. La falta de soporte o de precisión producirá un resultado inconcluso; la evidencia de incumplimiento de los límites será un resultado contrario a H2. Abstenerse o mantener intervalos amplios no contará como evidencia de eficacia si no se satisface la cobertura mínima. Tampoco se considerará satisfecho el objetivo si el control del riesgo depende de completar sistemáticamente la evaluación de todos los candidatos.

**Inserción en §2.3 o §4.4:**

> Se distinguirán el nivel de riesgo delta, el error de cobertura de intervalos delta_int y el nivel alfa de inferencia confirmatoria. Los márgenes de relevancia práctica se fijarán antes de calibrar los umbrales operativos. El prerregistro especificará las pruebas de cada componente de H1–H3 y cómo se aplicará Holm; falta de evidencia favorable no se describirá automáticamente como refutación.

**Ajuste mínimo del cronograma:** en año 1, «registro del diseño y de las reglas de desarrollo». En año 2, «desarrollo, calibración y congelación confirmatoria del selector y comparadores antes de la apertura de prueba». Especificar que el ajuste por familia y la elección de descriptores respetan todas las separaciones externas.

### 9.5. P1-05: controles y vecinos

**Original, p. 7, §4.4:**

> Los comparadores obligatorios serán: asignación aleatoria del mismo presupuesto; Hyperband o ASHA; BOHB o SMAC-HB; un extrapolador de curvas elegido antes del prerregistro, con LC-PFN como candidato inicial; y el selector propuesto sin abstención.

**Reemplazo mínimo sugerido:**

> Los comparadores incluirán el mejor codificador fijo elegido únicamente en desarrollo, asignación aleatoria de presupuesto, Hyperband o ASHA, BOHB o SMAC-HB y un método completo de adquisición basado en curvas, con ifBO como candidato inicial. Si se emplea LC-PFN, se fijará también su regla de consulta y parada. El piloto elegirá una implementación compatible por familia y registrará cualquier exclusión. La versión obligada del selector será el método principal de H1 y el control de abstención de H2. Los comparadores usarán el mismo espacio de candidatos y un contrato común de recursos e información.

**Inserción breve en §3.4:**

> La adquisición incremental de curvas y su transferencia ya se estudian en ifBO, DyHPO y métodos multifidelidad sensibles al costo. La contribución se delimitará respecto de esos métodos y del control selectivo reciente, atendiendo al objeto concreto de codificadores de memoria y a la fiabilidad de la evidencia entre tareas de RL.

No es necesario incorporar todas esas implementaciones al experimento principal. La justificación de la elección y un comparador directo fuerte son suficientes para una propuesta acotada.

### 9.6. P1-06: costos y factibilidad

**Original, p. 7, §4.4:**

> Si C_prev_m y C_marg_m son los costos previo y marginal del método m, se informará el punto de equilibrio frente al comparador b [...]

**Inserción mínima antes de la ecuación (4):**

> El costo primario C se medirá en una unidad escalar publicada bajo una configuración de recursos fija; GPU-horas, CPU-horas y memoria se informarán adicionalmente por separado. C_prev incluirá adquirir la evidencia histórica necesaria, entrenar y calibrar cada método, salvo en un escenario explícito donde ese corpus ya esté disponible para todos. C_marg será el costo esperado por tarea de una variante identificada y se comparará bajo el mismo criterio de desempeño. La amortización se presentará con su incertidumbre o por escenarios; no se afirmará un umbral finito preciso si el ahorro marginal no se distingue de cero.

**Inserción en la descripción de matriz:**

> El presupuesto de entrenamiento que define las fidelidades y el presupuesto de costo para decidir se especificarán por separado. Cada consulta pagará el tramo adicional de una trayectoria candidato–semilla y los costos de evaluación y recuperación correspondientes. El piloto verificará que pausar y reanudar preserva el entrenamiento que representa la matriz; la comparación asíncrona usará una configuración de trabajadores y una política de tiempos comunes.

**Original, p. 8, §5.2:**

> El segundo banco requerido por la precisión de H2 y la confirmación con CARL se presupuestarán después del piloto, antes del prerregistro [...]

**Reemplazo mínimo sugerido:**

> El piloto fijará un presupuesto total que incluya POPGym, la eventual ampliación de tareas, CARL, el ajuste por familia, desarrollo/calibración y fallos. La ampliación solo se incorporará si alcanza el soporte requerido dentro de ese techo. Si no resulta viable, se conservará el alcance empírico acotado y no se afirmará una certificación de riesgo que el tamaño disponible no permite sostener.

### 9.7. Corrección bibliográfica concreta de bajo impacto

En la entrada `li2020asha`, incorporar **Jonathan Ben-tzur** entre Ekaterina Gonina y Moritz Hardt, y usar la [página oficial verificada de MLSys](https://proceedings.mlsys.org/paper_files/paper/2020/hash/a06f20b349c6cf09a6b171c71b88bbfc-Abstract.html). El PDF abrevia autores con «et al.»; el defecto se identifica en la lista completa del BibTeX. El fallo de apertura de la URL antigua no se interpreta, por sí solo, como prueba de que nunca existió.

## 10. Cinco preguntas de entrevista y criterio de cierre

1. **¿Qué variable aleatoria cubre exactamente su intervalo: el resultado de una ejecución, la media de las semillas ocultas o la esperanza sobre nuevos entrenamientos? ¿Cómo conserva esa cobertura cuando el selector decide qué curva observar y cuándo detenerse?**  
   Debe poder distinguir el predictor, la calibración, el objetivo latente y el evento simultáneo, sin responder solo «usaremos conformal».

2. **Si emite quince recomendaciones y ninguna parece perjudicial, pero cinco referencias no permiten decidir si se excede epsilon, ¿qué riesgo puede afirmar y qué datos adicionales necesitaría?**  
   Debe mantener las recomendaciones en el denominador, reconocer incertidumbre de referencia y diferenciar una frecuencia observada de un límite certificado.

3. **¿Cómo demostrará que consultar curvas aporta valor frente a elegir siempre el mejor codificador histórico, y qué cambia respecto de ifBO o un selector de curvas con transferencia?**  
   Debe identificar un contraste barato, la brecha concreta y el posible resultado de que el control fijo sea suficiente.

4. **¿Cuántas tareas nuevas necesita para recuperar el costo de construir el selector? ¿Qué ocurre si el corpus histórico se considera gratuito o si cambia la mezcla de tareas?**  
   Debe separar inversión, uso, referencia experimental, variante evaluada y supuestos del punto de equilibrio.

5. **¿Cuál es su tesis defendible si no logra calibración útil con los recursos disponibles o si las señales tempranas no se transfieren?**  
   Debe describir un resultado empírico o formal específico, con comparadores y límites, sin equiparar «mi método falló» con «ningún método puede funcionar».

### 10.1. Respuestas a las tres preguntas de cierre de la solicitud

| Pregunta | Respuesta de esta auditoría |
|---|---|
| ¿La regla puede detenerse adaptativamente sin invalidar su incertidumbre? | **Sí, condicionalmente a una construcción que garantice el evento simultáneo. La versión actual lo expresa, pero no especifica aún una ruta suficiente para obtenerlo.** P1-01. |
| ¿El experimento puede estimar honestamente el riesgo entre recomendaciones emitidas? | **Es viable con el contrato de medición, soporte y particiones corregido; el texto actual todavía deja abiertas decisiones que cambian la estimación.** P1-02–04. |
| ¿El ahorro compensa el costo previo del selector? | **Aún no se sabe: es una pregunta empírica. La ecuación es correcta bajo sus condiciones; las entradas y el criterio comparativo deben cerrarse.** P1-06. |

### 10.2. Condición para pasar a la lectura final

Resolver los seis P1 mediante decisiones compatibles entre sí, conservar las correcciones ya aceptadas y compilar una nueva versión identificable. La siguiente revisión debería limitarse a verificar esos cierres y a claridad editorial. No se requiere demostrar ahora las hipótesis, añadir más dominios ni ejecutar toda la matriz para poder presentar una propuesta doctoral rigurosa.

**Recomendación al autor:** mantener la selección de codificadores de memoria como propuesta principal. La revisión revela trabajo de delimitación pendiente, no una razón para cambiar de tema.
