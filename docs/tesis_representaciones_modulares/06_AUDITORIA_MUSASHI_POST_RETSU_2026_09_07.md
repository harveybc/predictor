# Auditoría Musashi posterior a las correcciones de Retsu

**Fecha:** 2026-09-07  
**Objeto:** propuesta doctoral `propuesta_doctoral_representaciones_temporales_modulares`  
**Insumo externo:** `PAQUETE_TAKESHI_AUDITORIA_Y_FIGURAS.zip`  
**SHA-256 del ZIP:** `ff146a622d9fcadf9e082d4795857cdf8e726afb8fc760c02a235a6991a99c0d`

## Veredicto

**ACCEPT FOR OWNER/TAKESHI READING.** Las correcciones de Retsu resolvieron la arquitectura conceptual, la separación detector-integrador-adaptación, los regímenes R0/R1/R2 y el lugar limitado del autoencoder. No obstante, el PDF `c74e9ba9...` todavía no estaba listo para lectura final: conservaba una confusión de identificación, criterios de decisión incompletos, vecinos directos omitidos, lenguaje interno y dos figuras defectuosas. La revisión 3 corrige esos puntos sin cambiar el título ni convertir el preentrenamiento en la tesis.

## Hallazgos sobre la versión de Retsu

| Código | Severidad | Hallazgo | Disposición |
|---|---:|---|---|
| M01 | Alta | Los perfiles marginales orientaban grupos y luego H3 hablaba de dependencias cruzadas como si fueran observables por el mismo mecanismo. | Se separan dos objetos: perfiles marginales para compatibilidad de escalas y diagnóstico rezagado para H3. La prueba principal de H3 pasa a sintético con estructura conocida. |
| M02 | Alta | H1-H3 nombraban desenlaces, pero no definían estimando, signo ni regla completa de beneficio/equivalencia/perjuicio/inconcluso. | Se define $\Delta=\mathrm{MASE}_{método}-\mathrm{MASE}_{control}$, margen $\varepsilon$ y clasificación por intervalo simultáneo. H2 y H3 reciben condiciones de apoyo propias. |
| M03 | Alta | La normalización de perfiles podía realizarse por tarea y borrar precisamente la heterogeneidad entre tareas que H2 pretende estudiar. | Retardos y frecuencias pasan a coordenadas comparables; el escalado se aprende en E1 y se congela para E2. |
| M04 | Alta | DUET y MSGNet, vecinos directos de agrupación temporal/canales, no aparecían. | Ambos se incorporan al estado del arte; DUET queda como comparador directo. La brecha sigue declarada como provisional. |
| M05 | Media | El contraste de fusión afirmaba igualdad de presupuesto de información aunque conservar una secuencia exige más activaciones que resumirla. | Se retira esa igualdad y se exige publicar parámetros, operaciones y memoria de activaciones. |
| M06 | Media | El banco público no distinguía series univariadas de unidades aptas para una hipótesis de agrupación multivariada. | Se fijan requisitos mínimos de elegibilidad; las series univariadas no pueden probar H1-H3. |
| M07 | Media | MASE y el costo total carecían de reglas para denominador cero y reutilización del preentrenamiento. | Se tipa la unidad no evaluable, se evita una constante retrospectiva y se añade una ecuación de costo por fase, método, semilla y origen. |
| M08 | Media | El texto incluía lenguaje de auditoría interna: `C9`, niveles del proyecto, “intención del autor”, “no rescata” y detalles de configuración. | Se reemplaza por lenguaje académico común y se conserva únicamente la limitación factual de U08. |
| M09 | Media | La Figura 1 superponía etiquetas con flechas; la Figura 2 atravesaba tres nodos y ocupaba casi una página completa. | Ambas se redibujan sin códigos internos, con flujo legible y sin flechas sobre texto. |
| M10 | Baja | La matriz U01-U12 certificaba como cerrados puntos que seguían pendientes del piloto. | Se reescribe la matriz distinguiendo cierre documental de decisiones preexperimentales aún abiertas. |

## Estructura científica resultante

- **H1, confirmatoria pública:** utilidad fuera de muestra del diseño informado frente a controles congelados.
- **H2, mecanística sintética:** interacción entre heterogeneidad conocida de escalas y ventaja de la asignación informada.
- **H3, mecanística sintética:** interacción entre dependencias cruzadas rezagadas y conservación de la secuencia hasta la fusión.
- **Datos públicos:** prueba principal de H1 y análisis de transferencia de H2/H3, sin convertir asociaciones en causalidad.
- **Datos financieros:** aplicación secundaria, separada de la selección del método y de la validez general.

## Pendientes legítimos antes del protocolo confirmatorio

1. Completar la revisión sistemática y decidir si la brecha sobrevive a DUET, MSGNet, CCM y trabajos posteriores.
2. Fijar la fórmula final del perfil, la cuadrícula de campos receptivos y el diagnóstico rezagado.
3. Censar un banco realmente multivariado y simular la precisión alcanzable por tarea y familia.
4. Fijar $\varepsilon$, la tolerancia de capacidad y el instrumento de energía antes de E2.
5. Implementar la prueba de flujo de gradientes exigida por U08 antes de usar R1 o R2.

Estos pendientes pertenecen al protocolo del primer año. No son afirmaciones presentadas como hechos consumados en el PDF de admisión.

## Salida verificada

- PDF final: `docs/propuesta_doctoral_representaciones_temporales_modulares.pdf`
- Extensión: 11 páginas, papel carta, cuerpo de 11 puntos.
- SHA-256 del PDF: `c6b85f0fee6c46609e562e6a1f2fee0abc585b5b8abbc7123f841e2ec3d23dde`
- SHA-256 de la fuente: `5a7f65bab8e2072fa71b49105f4344853baff863800051ccfdff212b54d3d0df`
- Compilación: `latexmk` + `biber`, sin referencias indefinidas ni cajas desbordadas.
- Revisión visual: 11/11 páginas inspeccionadas; figuras, tablas, ecuaciones, citas y saltos de página legibles.
