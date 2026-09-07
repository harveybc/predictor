# Registro de reestructuración C01-C14

**Solicitud base:** `SOLICITUD_TAKESHI_REESTRUCTURACION_PROPUESTA_MUSASHI_2026_09_07.md`

**Documento anterior:** propuesta de selección multifidelidad de representaciones para RL

**Documento nuevo:** `propuesta_doctoral_representaciones_temporales_modulares.tex`

La propuesta anterior se conserva. El documento nuevo no pretende ser una corrección incremental de su objeto científico.

| Código | Situación anterior | Resolución adoptada | Evidencia o justificación | Ubicación nueva |
|---|---|---|---|---|
| C01 | Título y resumen centrados en selección multifidelidad | Título ratificado y resumen reescrito desde cero | El autor cambió el objeto | Portada y Resumen |
| C02 | Pregunta sobre recomendar un codificador con menos evaluaciones | Pregunta sobre perfiles temporales, grupos y campos receptivos | Define método, comparación y consumidor | §1 |
| C03 | Hipótesis de costo, riesgo y cambio de contexto del selector | H1 utilidad, H2 acoplamiento con heterogeneidad, H3 conservación temporal | Cada una contiene contraste y resultado contrario | §3.3 |
| C04 | Preprocesamiento fijo fuera de la pregunta | Entrada común como control; transformaciones solo bajo información causal común | Permite estudiar representación sin dar información desigual | §4.1 |
| C05 | Catálogo cerrado de cuatro codificadores | Mecanismo finito de perfiles, agrupamiento y campos receptivos | El objeto es construir una composición, no comprar curvas | §§4.1-4.2 |
| C06 | Salida vectorial común | Secuencias alineadas por tiempo hasta la fusión; resumen temprano como ablación | Longitud no implica correspondencia temporal | §§2.1, 4.2 y 5.1 |
| C07 | Acoplamiento con cabezal excluido | Ramas, fusión y cabezal se entrenan conjuntamente | La representación aprendida depende de su objetivo | §4.2 |
| C08 | Selector gaussiano y abstención como mecanismo central | Retirados del núcleo y conservados solo como antecedente | Responden a otra pregunta | Documento completo |
| C09 | Estado del arte AutoRL y extrapolación de curvas | Vecinos directos de pronóstico, multiescala, agrupación, fusión y frentes aprendibles | La novedad debe contrastarse con el objeto actual | §2.2-2.3 y matriz |
| C10 | RL y bancos de control como evidencia primaria | Pronóstico multivariado público como primario; finanzas/RL como secundario | Separa validez general de una aplicación propia | §§4.3 y 6.1 |
| C11 | Comparadores del selector y costo hasta recomendación | Pisos simples, modelos fuertes, vecinos multiescala y ablaciones del mecanismo | Aísla regla, modularidad, capacidad y tiempo | §5.1-5.2 |
| C12 | Cotas de selección secuencial | Contrato operacional, mecanismo reproducible y caracterización de regímenes | No se heredaron teoremas de otro estimando | §7 |
| C13 | Cronograma y presupuesto de 2.250 curvas | Seis semestres por revisión, mecanismo, desarrollo, confirmación y aplicación condicionada | El costo se medirá con pilotos del nuevo objeto | §6.1-6.2 |
| C14 | Garantías de evaluación adquiridas | Separación por familias, ajuste train-only, unidad tarea, multiplicidad, costos y resultados nulos | El cambio de pregunta no relaja integridad | §§4.1, 5.2-5.3 y 6.3 |

## Comprobaciones editoriales

- Título, metadatos, pregunta, objetivos, hipótesis, método y contribuciones nombran el mismo objeto.
- Se distinguen representación de entrada, representación aprendida, diseño estructural, aprendizaje de pesos y búsqueda residual.
- El término campo receptivo se define antes de usarse como mecanismo.
- El consumidor principal y la población de generalización están declarados.
- No se trasladaron fórmulas, umbrales ni promesas del selector anterior.
- Las citas se muestran con numeración IEEE y la bibliografía está ligada desde LaTeX.
