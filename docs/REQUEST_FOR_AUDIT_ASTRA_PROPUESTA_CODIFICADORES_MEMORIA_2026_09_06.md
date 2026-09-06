# Solicitud formal de auditoría para Astra

**Fecha:** 2026-09-06

**Documento:** propuesta doctoral sobre selección secuencial de codificadores de memoria para aprendizaje por refuerzo

**Autor:** Harvey Demian Bastidas Caicedo

**Programa previsto:** Doctorado en Inteligencia Artificial, Universidad de La Sabana

**Tipo de encargo:** revisión académica externa, crítica y de solo lectura

## 1. Encargo

Astra: por favor audita la versión enlazada abajo como si fueras, de manera sucesiva, tres integrantes exigentes de un comité doctoral:

1. una profesora de aprendizaje automático y estadística, especialmente atenta a validez secuencial, selección, dependencia y tamaño muestral;
2. un investigador de aprendizaje por refuerzo y AutoRL que conozca selección de arquitecturas, curvas de aprendizaje e identificación multifidelidad del mejor brazo;
3. una persona del comité de admisiones que entiende inteligencia artificial, pero no conoce este proyecto ni su vocabulario interno.

No buscamos aprobación cortés. Buscamos descubrir afirmaciones falsas, circulares, ambiguas, no identificables, estadísticamente indefendibles o difíciles de comprender antes de enviar la propuesta. No propongas ampliar el trabajo con una segunda tesis. Toda recomendación debe conservar una sola pregunta científica y un programa realizable en tres años.

## 2. Documento autoritativo

- [PDF para revisar en GitHub](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_representaciones_rl.pdf)
- [PDF directo para descargar](https://raw.githubusercontent.com/harveybc/predictor/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_representaciones_rl.pdf)
- [Fuente LaTeX](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex)
- [Bibliografía BibTeX](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/propuesta_doctoral_seleccion_multifidelidad_rl.bib)

**SHA-256 esperado del PDF:** `0860a4f2975042fbf30400d5920443a7ca46415c58ba2e0cedc2ca2886eb470f`

Si el PDF descargado no coincide con ese digest, detén la auditoría y reporta `DOCUMENT_IDENTITY_MISMATCH`. El PDF tiene 11 páginas: nueve de contenido y dos de referencias.

## 3. Cambios que esta versión intenta resolver

La versión anterior recibió cinco observaciones de Astra. Esta revisión:

1. distingue cobertura de intervalos, validez durante consultas adaptativas y riesgo selectivo entre recomendaciones;
2. usa una versión obligada a decidir para H1 y reserva riesgo, cobertura y costo de abstención para H2;
3. reconoce que quince entornos base no permiten sostener por sí solos un riesgo pequeño y muestra la cuenta exacta de 15 frente a 59 recomendaciones independientes;
4. formula la imposibilidad mediante historias observables indistinguibles y conjuntos de candidatos epsilon-óptimos disjuntos;
5. separa costo experimental común, costo previo y costo marginal, y define el punto de equilibrio de la inversión en el selector;
6. estrecha el título desde representaciones en general hasta codificadores de memoria.

Debes comprobar que cada corrección está realmente cerrada de principio a fin. No la des por resuelta porque aparezca una frase nueva.

## 4. Preguntas obligatorias

### A. Objeto y claridad

1. ¿El título coincide exactamente con lo que se compara o sigue prometiendo más de lo que el método estudia?
2. ¿Fidelidad, codificador de memoria, tarea, caso de evaluación, semilla, arrepentimiento, recomendación perjudicial, cobertura y abstención se entienden en su primera aparición?
3. ¿Una persona competente en IA, pero ajena a AutoRL, puede explicar la pregunta, H1, H2 y H3 después de una sola lectura?
4. ¿Hay vocabulario interno, defensas innecesarias o frases cuyo significado dependa de conocer nuestros repositorios?

### B. Validez estadística

5. ¿La ecuación de intervalos cubre simultáneamente candidatos y momentos de consulta hasta una parada adaptativa, o todavía permite optional stopping inválido?
6. ¿El texto separa correctamente esa garantía dentro de una tarea del riesgo selectivo estimado entre tareas?
7. ¿H1 compara métodos sobre el mismo conjunto de tareas y evita favorecer al selector por sus abstenciones?
8. ¿H2 define sin ambigüedad el denominador de riesgo y de cobertura? ¿La regla de soporte contempla que el número relevante es el de recomendaciones independientes?
9. ¿La cuenta binomial 15/59 es correcta y está presentada como referencia de precisión, no como sustituto de un análisis jerárquico?
10. ¿Las dificultades, contextos y semillas permanecen anidados sin convertirse en réplicas independientes por redacción indirecta?
11. ¿H3 distingue evaluación bajo cambio de contexto de una garantía universal de calibración?

### C. Resultado formal

12. ¿La condición de recomendación implica arrepentimiento menor o igual que epsilon sobre el evento simultáneo declarado?
13. ¿La región de imposibilidad es matemáticamente coherente con indistinguibilidad, conjuntos epsilon-óptimos disjuntos y probabilidad media de éxito?
14. ¿La propuesta promete un resultado formal alcanzable en tres años o esconde supuestos que lo vuelven trivial o imposible?
15. ¿La relación con multi-fidelity best-arm identification está delimitada con justicia, sin reclamar novedad por desconocimiento?

### D. Costo y experimento

16. ¿La ecuación del punto de equilibrio usa los costos correctos y trata de manera justa el aprendizaje histórico disponible para los comparadores?
17. ¿El presupuesto de curvas sigue siendo coherente después de exigir un segundo banco para afirmaciones de riesgo exigentes?
18. ¿POPGym, CARL y ARLBench cumplen funciones distintas y comprensibles?
19. ¿Los cinco candidatos permiten atribuir diferencias a memoria sin favorecer una familia mediante ajuste desigual?
20. ¿Los comparadores son suficientes y fuertes para 2026? Verifica en fuentes primarias si falta un vecino que pueda volver redundante la contribución.

### E. Admisión y redacción

21. Señala toda frase que suene exagerada, vaga, defensiva, críptica o impropia de una propuesta doctoral.
22. Comprueba que la evidencia preliminar demuestra preparación sin presentarse como resultado de la tesis.
23. Comprueba la correspondencia exacta entre pregunta, objetivos, hipótesis, método, cronograma y contribuciones.
24. Identifica cualquier afirmación que necesite cita, cualquier cita que no sostenga la oración y cualquier referencia bibliográfica incorrecta o inexistente.
25. Decide si 11 páginas son razonables para este contenido o si hay una página concreta que pueda retirarse sin perder una defensa necesaria.

## 5. Contexto conceptual opcional

Estos documentos explican cómo se recortó el problema. No prevalecen sobre el PDF y pueden contener decisiones anteriores:

- [Fenómeno que originó el proyecto](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/tesis_sac/00_FENOMENO.md)
- [Matriz inicial del estado del arte](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/tesis_sac/01_MATRIZ_ESTADO_DEL_ARTE.md)
- [Contrato científico anterior](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/tesis_sac/02_CONTRATO_CIENTIFICO.md)
- [Presupuesto preliminar anterior](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/tesis_sac/03_PRESUPUESTO_L1.md)
- [Auditoría interna previa al primer PDF](https://github.com/harveybc/predictor/blob/docs/agent-onboarding-20260816/docs/tesis_sac/04_AUDITORIA_MUSASHI_PRE_PDF.md)

Si esos documentos contradicen el PDF actual, reporta la contradicción, pero juzga la propuesta por el PDF y su fuente actuales.

## 6. Contexto de factibilidad, no de validez científica

Los siguientes repositorios muestran experiencia e infraestructura del candidato. No deben usarse para dar por probadas H1, H2 o H3:

- [predictor](https://github.com/harveybc/predictor): entrenamiento y comparación offline de modelos temporales.
- [agent-multi](https://github.com/harveybc/agent-multi): experimentación con agentes y campañas de aprendizaje por refuerzo.
- [gym-fx](https://github.com/harveybc/gym-fx): entornos y evaluación aplicada de aprendizaje por refuerzo.
- [heuristic-strategy](https://github.com/harveybc/heuristic-strategy): estrategias y generadores experimentales.
- [preprocessor](https://github.com/harveybc/preprocessor): contratos de preprocesamiento causal.
- [lts](https://github.com/harveybc/lts): integración operativa; es contexto de ingeniería, no aporte doctoral de esta propuesta.
- [doin-core](https://github.com/harveybc/doin-core): trabajo distribuido previo; queda fuera de la pregunta doctoral actual.

No es necesario auditar estos repositorios. Úsalos solo para evaluar si el párrafo de trayectoria y la viabilidad son plausibles. Si no puedes acceder a alguno, indícalo y continúa con el PDF.

## 7. Fuentes primarias prioritarias

Además de verificar toda la bibliografía del PDF, contrasta especialmente:

- [POPGym, ICLR 2023](https://openreview.net/forum?id=chDrutUTs0K)
- [Optimal Multi-Fidelity Best-Arm Identification, NeurIPS 2024](https://doi.org/10.52202/079017-3874)
- [Time-Uniform Confidence Sequences, Annals of Statistics 2021](https://doi.org/10.1214/20-AOS1991)
- [Conformal Risk Control, ICLR 2024](https://openreview.net/forum?id=33XGfHLtZg)
- [Selective Conformal Risk Control, preprint 2025](https://arxiv.org/abs/2512.12844)

Busca trabajos adicionales solo en fuentes primarias: artículos, actas oficiales, OpenReview, PMLR, JMLR, editoriales académicas o repositorios oficiales de los autores. No uses resúmenes comerciales ni contenido generado por otros modelos como autoridad.

## 8. Formato de respuesta requerido

Entrega el informe en español con esta estructura:

1. **Veredicto global:** `ACCEPT`, `ACCEPT_WITH_MINOR_REVISIONS`, `REVISE` o `REJECT`.
2. **Resumen ejecutivo:** máximo 250 palabras.
3. **Hallazgos P0:** errores que pueden invalidar la tesis o la admisión.
4. **Hallazgos P1:** correcciones necesarias antes del envío.
5. **Hallazgos P2:** mejoras deseables que no bloquean el envío.
6. **Matriz de consistencia:** pregunta, objetivo, hipótesis, evidencia, análisis y criterio de falsación.
7. **Auditoría de referencias:** afirmación, referencia usada, si la sostiene y fuente primaria consultada.
8. **Prueba de lectura de los tres jurados:** una sección breve por cada perfil solicitado.
9. **Texto de reemplazo:** para cada P0 o P1, cita la frase problemática y propón la modificación mínima; no reescribas toda la propuesta.
10. **Cinco preguntas de entrevista:** las más difíciles que el autor debe poder responder.

Cada hallazgo debe señalar página y sección. Distingue hechos comprobados, inferencias y asuntos que no pudiste verificar. No recomiendes introducir mercados, cadena de bloques, trading en vivo, teoría de la información ni transformaciones temporales salvo que descubras una dependencia estrictamente necesaria para responder la pregunta actual.

## 9. Criterio de cierre

La auditoría será útil si puede responder con precisión estas tres preguntas:

1. ¿La regla puede detenerse adaptativamente sin invalidar la incertidumbre que usa para decidir?
2. ¿El experimento puede estimar honestamente el riesgo entre las recomendaciones que realmente emite?
3. ¿El ahorro sobre tareas nuevas compensa, bajo una contabilidad justa, el costo previo de construir el selector?

Si alguna respuesta es negativa, clasifícala como P0 o P1 y explica qué afirmación debe reducirse. Un resultado nulo o inconcluso es admisible; una garantía que los datos no pueden sostener, no.
