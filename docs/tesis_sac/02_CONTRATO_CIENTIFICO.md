# Contrato científico del pivote L2/RL

**Estado:** congelación previa al borrador doctoral  
**Fecha:** 2026-09-03  
**Propósito:** impedir que la propuesta cambie de pregunta, unidad o criterio de éxito después de ver resultados.

## 1. Pregunta madre

¿Bajo qué condiciones una regla multifidelidad con abstención puede seleccionar representaciones para aprendizaje por refuerzo en tareas o contextos no vistos, reduciendo el costo de evaluaciones completas sin aumentar más allá de un margen predeclarado la frecuencia de selecciones perjudiciales?

## 2. Objeto y vocabulario

- **Representación:** combinación acotada de arquitectura del extractor, ventana o memoria, reducción, fusión y, cuando aplique, objetivo de preentrenamiento.
- **Fidelidad 0:** diagnóstico sin entrenamiento o con un solo lote. Incluye tamaño/costo, gradientes, sensibilidad al orden temporal y descriptores de información candidatos.
- **Fidelidad 1:** entrenamiento corto de RL con presupuesto y semillas fijados.
- **Fidelidad 2:** evaluación completa que produce el veredicto de nivel 1.
- **Selección perjudicial:** recomendación cuyo resultado de fidelidad 2 queda por debajo del control externo por más de un margen fijado en el piloto.
- **Abstención:** la regla no promueve ni elimina una representación con la evidencia disponible. Debe pagar la siguiente fidelidad o devolver “sin recomendación”.
- **Cobertura:** proporción de decisiones en las que la regla sí recomienda sin escalar a fidelidad 2.
- **Unidad primaria:** tarea o contexto no visto. Las semillas son repeticiones anidadas, no unidades independientes.

## 3. Objetivos

1. Formalizar la selección de representaciones de RL como identificación multifidelidad con sesgo dependiente de tarea y configuración, costo medido y opción de abstención.
2. Diseñar una regla que calibre el riesgo de selección perjudicial y decida entre recomendar, pedir mayor fidelidad o abstenerse.
3. Evaluar eficiencia, daño y cambio de distribución en un banco público, y después reproducir el protocolo congelado en una confirmación financiera separada.

## 4. Hipótesis falsables

### H1. Eficiencia a costo igual

En tareas públicas no vistas y al mismo costo computacional medido, la regla propuesta obtendrá menor arrepentimiento simple normalizado que búsqueda aleatoria, Hyperband/ASHA y BOHB/SMAC-HB.

**Falla si:** el intervalo de la diferencia incluye el margen de no inferioridad adverso predeclarado, o el ahorro solo aparece al excluir el costo de las sondas y del entrenamiento del selector.

### H2. Abstención útil

Frente al mismo selector sin abstención, la regla completa reducirá la selección perjudicial y respetará el riesgo nominal calibrado, manteniendo una cobertura mínima fijada antes de la prueba.

**Falla si:** siempre escala a fidelidad 2, la cobertura cae bajo el piso, o el riesgo observado excede su cota. Abstenerse de todo no satisface H2.

### H3. Cambio de distribución

Cuando cambie el contexto respecto a las tareas de desarrollo, la regla respetará el límite de daño mediante una de dos respuestas observables: conservará la calibración o reducirá su cobertura al aumentar la abstención. La misma regla y umbrales, ya congelados, se someterán luego a una confirmación financiera externa.

**Falla si:** bajo cambio de contexto recomienda con confianza y supera el margen de daño, o si requiere volver a ajustar umbrales con la prueba final.

## 5. Único resultado teórico prometido

Se buscará un resultado de selección con rechazo:

> Dada una envolvente de error calibrada para cada fidelidad y una cota de cambio entre tareas, la regla devuelve con probabilidad al menos 1 − δ una representación cuyo valor de fidelidad 2 está a ε del mejor, o se abstiene y escala el costo.

El complemento será una región de imposibilidad dentro del mismo resultado: sin una envolvente identificable que distinga dos mundos con el mismo historial barato y distinto mejor brazo, ninguna regla puede reducir evaluaciones completas y, al mismo tiempo, acotar la selección perjudicial. No se promete un segundo teorema sobre reuso adaptativo.

## 6. Diseño primario

- **Banco público:** ARLBench, con DQN, PPO y SAC. La superficie de representación se extenderá de manera pequeña y publicada.
- **Cambio de contexto:** CARL o un protocolo equivalente de contextos públicos, sujeto a compatibilidad técnica verificada en el piloto.
- **Confirmación externa:** agente SAC financiero propio, solo después de congelar método, umbrales y análisis en el dominio público.
- **Separación:** desarrollo, calibración y prueba se agrupan por entorno base para evitar que variantes del mismo problema crucen particiones.
- **Meta objetivo inicial:** al menos 30 unidades tarea-contexto útiles, agrupadas por entorno base; distribución provisional 12 desarrollo, 6 calibración y 12 prueba. El piloto puede cambiar la cuenta antes del prerregistro, nunca después de abrir la prueba.

## 7. Comparadores

1. Búsqueda aleatoria.
2. Hyperband o ASHA.
3. BOHB o SMAC-HB.
4. iMFBO o ProxyBO si la adaptación conserva sus supuestos.
5. Método propuesto sin abstención.
6. Método propuesto completo.

La configuración manual histórica no es comparador principal porque fue informada por el mismo proceso que motiva la tesis.

## 8. Métricas y análisis

- Primaria: arrepentimiento simple normalizado en fidelidad 2 a costo computacional igual.
- Coprimarias de seguridad: tasa de selección perjudicial y curva riesgo-cobertura.
- Secundarias: costo hasta objetivo, cantidad de evaluaciones completas, frecuencia y costo de abstención, calibración y tiempo de pared.
- Agregación: IQM y perfiles de desempeño sobre tareas, con intervalos por bootstrap estratificado.
- Inferencia: comparaciones pareadas por unidad y corrección de Holm para las comparaciones confirmatorias.
- Transparencia: CPU/GPU-horas, energía cuando esté disponible, fallos, configuraciones y resultados negativos.

## 9. Sondas permitidas

Las sondas se consideran hipótesis de medición, no evidencia válida por nombre. Pueden incluir:

- costo, parámetros y complejidad computacional;
- estabilidad de gradientes y pérdidas;
- sensibilidad a permutación o perturbación temporal;
- desempeño predictivo de la representación en tareas auxiliares causales;
- descriptores de compresibilidad, MDL o información.

La afirmación de MacKay de aproximadamente dos bits por peso aplica a un clasificador lineal de umbral, entradas en posición general y memorización de etiquetas binarias. No se extrapolará a la inteligencia de redes profundas. Un descriptor informacional permanece solo si aporta capacidad predictiva fuera de muestra por encima de controles simples como número de parámetros y FLOPs.

## 10. Puertas de abandono

1. Si las sondas no mejoran controles simples en tareas públicas no vistas, se elimina el surrogate y el proyecto continúa, si todavía es viable, como asignación multifidelidad sin predicción de fidelidad 2.
2. Si la abstención no logra simultáneamente el riesgo y la cobertura fijados, H2 se rechaza; no se baja el piso después.
3. Si la extensión de representaciones cuesta más de cuatro veces el entrenamiento base y no reduce el costo total, se reduce el espacio de búsqueda, no el número de semillas.
4. Si no se reúnen al menos 30 unidades tarea-contexto distribuidas entre suficientes entornos base, las afirmaciones de transferencia se declaran inconclusas.
5. La confirmación financiera no puede reparar una falla en el dominio público.

## 11. Exclusiones

Quedan fuera de esta tesis: incentivos, mercados, blockchain, DOIN multidominio, selección de proveedores, teoría general de la inteligencia, conciencia, un lazo cerrado de auto-mejora y afirmaciones de rentabilidad financiera. Esos trabajos pueden continuar por separado, pero no comparten hipótesis con este contrato.
