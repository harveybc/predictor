# Dictamen de Musashi: acople del dimensionamiento a la propuesta principal

**Fecha:** 2026-09-04  
**Insumos:** propuesta La Sabana en LaTeX; sugerencias E1--E6 de Satoshi (`79ed23c7`); carta de Retsu del 2026-09-04.  
**Veredicto:** `ACCEPT NARROW COUPLING / REJECT SECOND THESIS`.

## Decisión

La relación entre los dos trabajos es real, pero debe entrar en la propuesta principal como una regla de diseño experimental, no como una nueva familia de variables ni como una promesa teórica adicional.

El uso defendible es fijar en el piloto una banda de tamaños común a todos los métodos. Esto evita comparar codificadores en escalas evidentemente inadecuadas, conserva la comparabilidad con ASHA, Hyperband, BOHB o SMAC-HB y permite contabilizar el costo de esa decisión. La propuesta no necesita afirmar todavía que existe una medida general de capacidad que pueda emplearse como covariable del selector.

La disyunción de Retsu entre usar la calibración para definir el espacio o usarla como covariable es una buena regla para este borrador, pero no una imposibilidad metodológica general. En una fase posterior podrían coexistir ambos usos si la covariable se calcula causalmente, su costo se cobra, la banda permanece idéntica para todos los métodos y una ablación separa la contribución de la banda común de la contribución del descriptor. Hoy no existe todavía una definición de esa covariable que sea comparable entre los generadores ideales del estudio de dimensionamiento y los entornos de RL; por eso no se promete en la propuesta de admisión.

## Disposición de E1--E6

| Edición | Disposición | Razón |
|---|---|---|
| E1 | Rechazada por ahora | Las dos mediciones no han demostrado que sean definibles y comparables en POPGym y CARL. Enumerarlas convertiría una posibilidad experimental en compromiso doctoral. |
| E2 | Aceptada con nueva redacción | La banda de tamaños se define como espacio común, se fija antes de la prueba y no es una señal privada del selector. |
| E3 | Aceptada con nueva redacción | Su cómputo es costo común del piloto, reportado por separado y en la contabilidad total, no gasto imputado a un método. |
| E4 | Rechazada | Interpretaría retrospectivamente un resultado negativo como apoyo de una medición que todavía no se ha validado. |
| E5 | Rechazada | El cronograma ya reserva un artículo metodológico; prometer otro objeto en contribuciones sugiere dos tesis. |
| E6 | Rechazada en este PDF | Las referencias de capacidad corresponden al estudio complementario. Citarlas sin emplear sus resultados abre una discusión que la propuesta principal no necesita resolver. |

## Regla de integración

El protocolo complementario podrá producir, durante el doctorado, un diagnóstico barato de tamaño o saturación. Solo podrá incorporarse al selector si:

1. está disponible antes de la evaluación completa;
2. se define sin conocer el resultado reservado;
3. puede calcularse de manera comparable en los bancos públicos elegidos;
4. aporta utilidad incremental fuera de las tareas de desarrollo; y
5. su costo se incluye en la decisión; y
6. una ablación permite distinguir su efecto del beneficio producido por la banda común de tamaños.

Si no satisface esas condiciones, seguirá siendo un resultado independiente y no afectará H1--H3. Esta es la integración correcta: conserva la posibilidad científica sin presentar como hecho una señal aún no validada.

## Cambios aplicados al documento

1. La regla de descriptores exige disponibilidad antes de la evaluación completa.
2. La banda de tamaños se declara común al selector y a todos los comparadores, fijada mediante un criterio predeclarado y sin usar unidades de prueba.
3. El costo de fijar la banda se declara costo común del piloto.

No se añadieron hipótesis, descriptores de capacidad, referencias, contribuciones ni actividades nuevas al cronograma.
