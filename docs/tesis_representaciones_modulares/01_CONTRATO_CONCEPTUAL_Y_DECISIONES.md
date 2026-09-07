# Contrato conceptual de la propuesta

**Documento público asociado:** `propuesta_doctoral_representaciones_temporales_modulares.*`

**Fecha:** 2026-09-07

**Estado:** borrador integral para revisión del autor

**Antecedente preservado:** propuesta sobre selección multifidelidad de representaciones para RL

## 1. Objeto

La tesis estudia un procedimiento para diseñar y aprender representaciones temporales modulares. El procedimiento estima, únicamente con datos de entrenamiento, perfiles de dependencia temporal de las variables; los usa para proponer grupos de variables y campos receptivos; y aprende conjuntamente las ramas, la fusión y el cabezal de pronóstico.

La contribución no es usar varias ramas ni múltiples escalas. Es evaluar si una regla explícita que vincula perfiles temporales con esas dos decisiones estructurales aporta utilidad fuera de muestra frente a diseños simples, aleatorios y multiescala recientes.

## 2. Mecanismo candidato

1. Construir por variable un perfil con decaimiento de autocorrelación, concentración espectral, periodicidades dominantes y estabilidad entre subventanas.
2. Normalizar esos descriptores con estadísticas del entrenamiento.
3. Agrupar variables mediante agrupamiento jerárquico.
4. Traducir las escalas de cada grupo al campo receptivo admisible más cercano dentro de una cuadrícula finita.
5. Mantener una rama común para perfiles inestables o no identificados.
6. Entrenar ramas, fusión y cabezal de extremo a extremo con la pérdida de pronóstico.

El agrupamiento, su distancia, el número máximo de ramas, el umbral de corte y la cuadrícula de campos receptivos son opciones de desarrollo. Se congelan antes de la evaluación confirmatoria.

## 3. Consumidor y población

- **Consumidor principal:** pronóstico multivariado y multi-horizonte.
- **Población principal:** familias públicas de series temporales separadas entre desarrollo y confirmación.
- **Calibración:** señales sintéticas con escalas, desfases y ruido aditivo conocidos.
- **Aplicación secundaria:** datos financieros históricos y, solo si las compuertas previas pasan, una política de aprendizaje por refuerzo.

La unidad primaria es la tarea de pronóstico definida por conjunto, objetivo y horizonte. Las ventanas solapadas y las semillas no se consideran unidades independientes.

## 4. Contribución posible

- Contrato operacional para entradas, ramas, campos receptivos, alineación, fusión y consumidor.
- Regla reproducible de composición informada por perfiles temporales.
- Evidencia con atribución que separe el efecto de la regla, la modularidad, la capacidad y la conservación temporal.
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
| Aprendizaje | Ramas, fusión y cabezal entrenados conjuntamente | El título incluye aprendizaje y la utilidad depende del consumidor. |
| Inferencia | Bootstrap jerárquico con tarea como unidad | Respeta familias, series, horizontes y semillas dependientes. |
| RL/finanzas | Aplicación secundaria condicionada | Evita que la validez doctoral dependa de un único dominio de alto riesgo. |
