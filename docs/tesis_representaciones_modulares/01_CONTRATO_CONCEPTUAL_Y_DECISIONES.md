# Contrato conceptual de la propuesta

**Documento público asociado:** `propuesta_doctoral_representaciones_temporales_modulares.*`

**Fecha:** 2026-09-07

**Estado:** revisión 3 (auditoría Musashi posterior a Takeshi y Retsu, 2026-09-07) para prueba de lectura

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
- **Aplicación secundaria:** datos financieros históricos y, solo si las compuertas previas pasan, una política de aprendizaje por refuerzo.

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
| H2 | Factorial sintético de heterogeneidad de escalas | Permite probar el mecanismo con estructura conocida. |
| H3 | Factorial sintético con y sin dependencia cruzada rezagada | Evita inferir mecanismo desde perfiles marginales. |
| Inferencia | Bootstrap jerárquico con tarea como unidad y Holm para H1--H3 | Respeta familias, series, horizontes y semillas dependientes. |
| RL/finanzas | Aplicación secundaria condicionada | Evita que la validez doctoral dependa de un único dominio de alto riesgo. |
