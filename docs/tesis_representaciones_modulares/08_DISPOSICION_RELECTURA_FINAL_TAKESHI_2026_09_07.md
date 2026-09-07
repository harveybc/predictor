# Disposición de la relectura final de Takeshi

**Entrada:** `RELECTURA_FINAL_TAKESHI_PROPUESTA_MODULAR_2026_09_07.md`

**Versión auditada:** `predictor@c16336f3275f40229ee55f14eb3347cfec2c2ff3`

**Salida:** revisión 4 de `propuesta_doctoral_representaciones_temporales_modulares.{tex,bib,pdf}`

**Alcance:** decisiones y redacción de una propuesta; no constituye evidencia de que los experimentos o la integración U08 ya se hayan ejecutado.

## Decisiones solicitadas

| Pregunta | Decisión adoptada |
|---|---|
| Agregación | Cada objetivo y origen usa su propio denominador MASE de entrenamiento. Se agregan semillas, orígenes, horizontes y objetivos dentro de tarea; después se equiponderan tareas dentro de conjunto, conjuntos dentro de familia y familias. Un denominador requerido igual a cero excluye la tarea completa de H1 y se informa. |
| Inferencia | $\alpha=0.05$ y $K=F+4$ efectos confirmatorios. Se usan intervalos bilaterales Bonferroni construidos con bootstrap pareado y jerárquico. H1 exige beneficio global y no inferioridad $U_f\leq\varepsilon$ en cada familia. |
| H2 | Contraste mecanístico: pendiente de MASE(informada) menos MASE(permutada) frente a heterogeneidad sintética conocida; exige $U_\beta<0$. El beneficio absoluto se informa, pero no gobierna H2. |
| H3 | Las dos fusiones consumen activaciones de un extractor compartido y congelado. Exige beneficio de la fusión temporal cuando hay acoplamiento ($U_{d_1}< -\varepsilon$) y una interacción negativa ($U_\gamma<0$). |
| Controles | H2 permuta variable--perfil conservando topología, alcances, interfaz, objetivos y presupuesto. H3 usa realizaciones independientes de los mismos procesos para conservar las leyes marginales temporales sin el acoplamiento cruzado. |
| Cronología y recursos | La reserva sintética permanece cerrada hasta finalizar E1. $C$ mide energía en Wh; pared, horas de cómputo y memoria se publican aparte. La política rodante permite incorporar observaciones anteriores una vez disponibles, de manera idéntica para todos los métodos. |

## Hallazgos P1

| ID | Disposición | Ubicación corregida |
|---|---|---|
| P1-01 | Aplicado. Se retiró la imposibilidad universal atribuida a campos receptivos cortos. | Problema y motivación. |
| P1-02 | Aplicado. Se definieron MASE por objetivo/origen, exclusión por denominador nulo y agregación jerárquica. | Hipótesis; Medidas, incertidumbre y costo. |
| P1-03 | Aplicado. Se fijaron nivel, familia finita de efectos, intervalos Bonferroni y no inferioridad por familia. | Hipótesis; Criterios de decisión. |
| P1-04 | Aplicado. H2 y H3 tienen contrastes, signos y condiciones de decisión explícitos. | H2; H3; Criterios de decisión. |
| P1-05 | Aplicado. H3 usa un extractor compartido y congelado; se ajustan solo fusión y cabezal. | H3; Arquitectura y aprendizaje. |
| P1-06 | Aplicado. La permutación temporal dejó de ser el control principal; se preservan leyes marginales mediante realizaciones independientes. | Fases experimentales. |
| P1-07 | Aplicado. La permutación variable--perfil conserva estructura, presupuesto e identidades de tarea. | Comparadores y ablaciones. |
| P1-08 | Aplicado. La reserva sintética no se abre hasta congelar el procedimiento al final de E1. | Fases experimentales; cronograma. |

## Hallazgos P2

| ID | Disposición | Ubicación corregida |
|---|---|---|
| P2-01 | Aplicado. Entrenamiento estima perfiles y pesos; validación selecciona; prueba solo evalúa. | Figura 2 y pie. |
| P2-02 | Aplicado. R1 verifica pesos invariantes; R2, conexión diferenciable y actualización efectiva; estados no entrenables se declaran. | Arquitectura y aprendizaje. |
| P2-03 | Aplicado. La Figura 1 separa decisiones de diseño y flujo de tensores mediante enlaces distintos. | Figura 1 y pie. |
| P2-04 | Aplicado. $C$ es energía en Wh e incluye búsqueda, paradas y fallos; otras magnitudes se informan aparte. | Medidas, incertidumbre y costo. |
| P2-05 | Aplicado. Solo las variantes del mismo contrato comparten rangos; referencias externas conservan espacios adecuados bajo igual límite de recursos. | Arquitectura y aprendizaje. |
| P2-06 | Aplicado. Se definió una política rodante común y qué observaciones previas pueden incorporarse. | Medidas, incertidumbre y costo. |
| P2-07 | Aplicado. Se adelantaron definiciones, se sustituyó "dependencia" por "temporal marginal" y se separaron $c_j$ y $d_i$. | Marco conceptual; objetivos; hipótesis. |
| P2-08 | Aplicado. CCM se describe mediante asignaciones y prototipos regularizados por similitud. | Estado del arte. |
| P2-09 | Aplicado. La evidencia técnica apunta al gestor correcto y fecha las versiones inspeccionadas en 2026. | Referencias técnicas. |
| P2-10 | Aplicado. GIFT-Eval se cita como arXiv:2410.10393 (2024), sin sede no sustentada. | Bibliografía. |
| P2-11 | Aplicado. La evaluación rodante recibe una referencia directa y no se atribuye a Bergmeir una garantía más amplia. | Medidas, incertidumbre y costo. |
| P2-12 | Aplicado. Se retiró RL del alcance del PDF y E3 quedó como aplicación financiera histórica opcional. | Cronograma. |
| P2-13 | Aplicado. Equivalencia, perjuicio e inconclusión se distinguen sin prometer que una alternativa vencerá. | Riesgos; contribuciones. |

## Pendientes que no se presentan como resultados

- U08: demostrar el ensamblaje diferenciable y la actualización efectiva del detector.
- Confirmar o estrechar la brecha de literatura mediante la revisión sistemática.
- Fijar en E1 el banco, $\varepsilon$, el tamaño mínimo, el diagnóstico público de H3 y el instrumento de energía.
- Ejecutar E0--E3 solo después de sus respectivos congelamientos; ninguna frase de la propuesta acredita resultados aún inexistentes.
