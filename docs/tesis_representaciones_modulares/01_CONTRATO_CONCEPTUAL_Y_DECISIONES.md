# Contrato conceptual de la propuesta

**Documento público asociado:** `propuesta_doctoral_representaciones_temporales_modulares.*`

**Fecha:** 2026-09-08

**Estado:** revisión 5 (encargo de figura introductoria adicional, aplicación RL obligatoria y contrato de comparabilidad, 2026-09-08; ejecuta `SOLICITUD_MUSASHI_FIGURA_ADICIONAL_RL_Y_COMPARABILIDAD_2026_09_08.md` sobre la revisión 4)

**Antecedente preservado:** propuesta sobre selección multifidelidad de representaciones para RL

## 1. Objeto

La tesis estudia un procedimiento para diseñar y aprender representaciones temporales modulares. El procedimiento estima, únicamente con datos de entrenamiento, perfiles temporales marginales de las variables; los usa para proponer grupos con escalas compatibles y campos receptivos; distingue detector, integrador y adaptación dentro de cada rama; y aprende según un régimen explícito (R0 conjunto, R1 detector congelado, R2 detector ajustable).

La contribución no es usar varias ramas, un autoencoder ni reutilizar pesos. Es evaluar si una regla explícita que vincula perfiles temporales con agrupación y alcance, con frontera de detector y régimen declarado, aporta utilidad fuera de muestra.

## 2. Mecanismo candidato

1. Construir por variable un perfil con decaimiento de autocorrelación, concentración espectral, periodicidades dominantes y estabilidad entre subventanas.
2. Expresar retardos y frecuencias en coordenadas comparables y escalar los descriptores con estadísticas aprendidas en familias de desarrollo, sin normalización independiente que borre la heterogeneidad entre tareas.
3. Agrupar variables mediante agrupamiento jerárquico.
4. Traducir las escalas de cada grupo al campo receptivo admisible más cercano dentro de una cuadrícula finita.
5. Mantener una rama común para perfiles inestables o no identificados.
6. Entrenar ramas, fusión y cabezal de extremo a extremo con la pérdida de pronóstico.

Los perfiles marginales proponen compatibilidad de escalas; no identifican causalidad ni dependencia entre variables. H3 utiliza un diagnóstico rezagado separado, calculado solo con entrenamiento y sin alterar la regla durante la confirmación.

El agrupamiento, su distancia, el número máximo de ramas, el umbral de corte y la cuadrícula de campos receptivos son opciones de desarrollo. Se congelan antes de la evaluación confirmatoria.

## 3. Consumidor y población

- **Consumidor principal:** pronóstico multivariado y multi-horizonte.
- **Población principal:** familias públicas de series temporales separadas entre desarrollo y confirmación.
- **Calibración y mecanismo:** señales sintéticas factoriales con escalas y desfases conocidos, divididas en generadores de desarrollo y generadores reservados.
- **Aplicación secundaria:** datos financieros históricos, sin intervenir en la selección del método ni sostener H1--H3.

La unidad primaria es la tarea de pronóstico definida por conjunto, objetivo y horizonte. Las ventanas solapadas y las semillas no se consideran unidades independientes.

## 4. Contribución posible

- Contrato operacional para entradas, ramas, campos receptivos, alineación, fusión y consumidor.
- Regla reproducible de composición informada por perfiles temporales.
- Evidencia con atribución que separe el efecto de la regla, la modularidad, la capacidad, el preentrenamiento y la conservación temporal.
- Caracterización de condiciones de beneficio, equivalencia, perjuicio o falta de identificación.

## 5. Exclusiones

- El selector gaussiano multifidelidad, la compra secuencial de curvas y la abstención calibrada no son el núcleo.
- DOIN, DEAP, blockchain, OLAP y la infraestructura de campañas no son contribuciones doctorales.
- No se promete una arquitectura universalmente óptima, extracción total de información, causalidad desde espectros ni superioridad de la modularidad.
- No se denomina ruido al residuo de una serie real sin un modelo generativo que lo justifique.
- No se modifica retroactivamente ninguna campaña en curso.

## 6. Decisiones adoptadas para este borrador

| Decisión | Resolución provisional | Motivo |
|---|---|---|
| Consumidor principal | Pronóstico multivariado | Permite medir directamente la representación y usar bancos públicos no financieros. |
| Familia principal de bloques | TCN causal | Hace explícito y calculable el campo receptivo. |
| Mecanismo | Perfil temporal, agrupamiento jerárquico y asignación a una cuadrícula de campos receptivos | Es concreto, refutable y acotado. |
| Aprendizaje | R0/R1/R2 en desarrollo; un régimen congelado antes de E2 | Separa diseño estructural de inicialización; R2 es antecedente, no ganador. |
| Detector | Submódulo inicial del extractor, no red obligatoria aparte | Permite preentrenar o sustituir sin tres procesos. |
| Decodificador | Auxiliar del preentrenamiento; pareja preservada | Evita confundir reconstrucción con la tarea de pronóstico. |
| H1 | Confirmación en familias públicas reservadas | Es la prueba principal de utilidad fuera de muestra. |
| Agregación de H1 | Igual peso entre tareas dentro de conjunto, conjuntos dentro de familia y familias | Evita que una fuente grande decida por sí sola la conclusión de transferencia. |
| No inferioridad de H1 | Beneficio global y límite superior por familia $U_f\leq\varepsilon$ | Un promedio favorable no puede ocultar perjuicio material en una familia. |
| H2 | Pendiente de la diferencia informada--permutada frente a heterogeneidad sintética conocida | Prueba el mecanismo relativo; el beneficio absoluto se informa aparte. |
| H3 | Fusión frente a resumen temprano usando un extractor compartido y congelado | Atribuye el contraste a la fusión y exige beneficio bajo acoplamiento y una interacción con signo definido. |
| Inferencia | Bootstrap pareado y jerárquico; $K=F+4$ intervalos Bonferroni con $\alpha=0.05$ | Define una única ruta para el efecto global, los efectos por familia y los contrastes mecanísticos. |
| Control H2 | Permutar variable--perfil conservando ramas, alcances, interfaz y presupuesto | Evita confundir correspondencia informada con capacidad o un simple cambio de nombres. |
| Control H3 | Realizaciones independientes de los mismos procesos marginales, sin acoplamiento cruzado | Preserva en distribución la estructura temporal marginal que una permutación arbitraria destruiría. |
| Cronología sintética | Reserva H2/H3 cerrada hasta congelar el procedimiento al final de E1 | Impide que los resultados mecanísticos reservados orienten el diseño que deben evaluar. |
| Costo principal | Energía en Wh; pared, horas de cómputo y memoria informadas aparte | No suma magnitudes incompatibles y carga toda la búsqueda, fallos y trabajo compartido. |
| Figura introductoria | Figura 1 añadida tras el resumen (extractores con detector/integrador/adaptación, consolidador, núcleo temporal y salidas alternativas pronóstico/RL); Figuras 2 y 3 conservadas | Permite visualizar los módulos nombrados antes del formalismo sin sustituir las figuras aprobadas; el núcleo temporal es el mezclador ya descrito, no un módulo nuevo. |
| E3 (RL) | Integración y evaluación históricas OBLIGATORIAS en un agente RL de la plataforma existente, frente a su representación de referencia, con protocolo predeclarado y objetivo específico 4 propio | El uso previsto por el autor exige una entrega verificable; el resultado puede ser positivo, negativo o inconcluso y su análisis queda separado de H1–H3 (sin H4, sin tocar $K=F+4$). |
| Benchmarks y comparabilidad | Selección pregunta→referencia publicada→protocolo→artefactos→elegibilidad; tres tipos de afirmación (reproducción, comparación común, aplicación propia); ficha de comparabilidad e inventario en `09_CONTRATO_BENCHMARKS_Y_APLICACION_RL.md` | Impide elegir datos donde gana el método y mezclar cifras incomparables; TFB/TradeMaster/FinRL-Meta son candidatos por verificar, no selecciones cerradas. |
| Recorte | El mínimo defendible incluye la integración E3 evaluada; se recortan primero segunda familia de bloques, comparadores opcionales y ampliaciones | E3 conserva una tarea, un agente y el contraste referencia/propuesta; cumplir con recursos limitados es acotar, no borrar. |
