# 03 — Pregunta, objetivos, hipótesis

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Texto para pegar. Sin *confiable*. Sin conjunción disfrazada de una pregunta. Una pasada.

---

## Pregunta madre

¿Bajo qué condiciones un metaoptimizador con abstención —entrenado sobre un corpus congelado de trazas heterogéneas cuya fracción corrupta \(\delta\) puede acotarse por re-ejecución y pago, y cuyo estimador de valor **no** entra en la remuneración— reduce el arrepentimiento simple normalizado y el costo verificado hasta un objetivo en tareas no vistas, sin transferencia negativa más allá de un margen predeclarado, y sin que fabricar trazas de alto *proxy* informativo y bajo desempeño tabulado sea mejor respuesta?

Eso es **una** pregunta porque el mecanismo no es un fin: descarga \(\delta\) para el teorema de transferencia. Si \(\delta\) no se acota, se reporta la región de imposibilidad. También responde la pregunta.

---

## Objetivo general

Diseñar y evaluar un método de metaoptimización selectiva que transfiere conocimiento entre tareas de caja negra cuando la procedencia del corpus es estratégica o está corrompida, y que se abstiene cuando un umbral de calibración predeclarado no se cumple.

---

## Objetivos específicos

1. **Interfaz.** Formalizar la traza con procedencia (configuración, presupuesto, política de origen, censura, costo, resultado, sello de re-ejecución o lookup) y la mapa mecanismo \(\to \delta\): condiciones de auditoría por re-ejecución y de pago bajo las cuales la fracción corrupta del corpus de entrenamiento queda acotada, o demostrar que no es identificable.
2. **Dos módulos, una interfaz.** (a) Metaoptimizador probabilístico con abstención y canales de procedencia/censura. (b) Asignación de evaluaciones cuyo pago depende de \(q\) re-ejecutado o tabulado, **nunca** de \(\hat v\).
3. **Evaluación con atribución.** HPO-B como banco primario; **una** familia confirmatoria de series de tiempo; factorial recortado; adversario de mejor respuesta; piloto de potencia congelado.

---

## Hipótesis (falsables, no eslóganes)

**H1 · Transferencia bajo \(\delta\) acotada.**  
En ≥30 tareas HPO-B reservadas, con corpus de entrenamiento cuya corrupción efectiva es \(\le \delta\) (el \(\delta\) que H2 entrega o el inyectado equivalente), el selectivo reduce el área bajo arrepentimiento simple normalizado **y** el costo-hasta-objetivo frente al mejor BO sin transferencia de la misma celda. Ante cambio de espacio de hiperparámetros, no-inferioridad dentro de un margen \(m\) fijado en el piloto.

*Falla si:* no hay ganancia en tareas relacionadas, o hay transferencia negativa material bajo cambio.

**H2 · No-rentabilidad de fabricar el proxy.**  
Bajo auditoría de fracción \(p\) y pago condicionado a \(q\) (no a \(\hat v\)), la utilidad esperada de un generador de trazas de alto *proxy* / bajo \(q\) es \(\le 0\) dentro del presupuesto declarado, y la \(\delta\) observada en el corpus que alimenta H1 es \(\le\) el umbral que H1 asume.

*Falla si:* el mejor-respuesta sigue ganando dinero, o \(\delta\) observada supera el supuesto. Entonces H1 no tiene licencia: o se reporta imposibilidad, o se cae el capítulo económico y queda KEEP B con \(\delta\) inyectada.

**H3 · Atribución de un eje, no de tres.**  
La ganancia de H1 se ablate contra (a) el mismo selectivo **ciego** a procedencia y (b) asignación por costo sin \(\hat v\). Se declara **un** contraste primario de H3 en el piloto (el que tenga potencia). El otro queda secundario.

*Falla si:* ni (a) ni (b) mueven la métrica primaria. Entonces el “acople” de Satoshi es relato.

---

## Lo que estas hipótesis **no** dicen

- No dicen que el lazo cerrado converja.
- No dicen que \(\hat v\) sea un precio eficiente.
- No dicen IC universal ni DSIC.
- No dicen que descriptores MacKay/compresión funcionen: eso es ablación, no H.
- No dicen nada de LLM-as-judge.

---

## Criterio de confirmación (copiar al PDF)

Un piloto **separado**, no contado como confirmación, fija: \(\delta\), \(p\), \(m\), el contraste primario de H3, \(N\) de tareas, semillas, y el umbral de abstención. Esas cifras se congelan. Holm sobre los contrastes primarios. Unidad = tarea no vista. La tesis acepta un resultado negativo.
