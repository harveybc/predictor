# Auditoría Musashi de la propuesta doctoral en LaTeX

**Fecha:** 2026-09-03  
**Documento auditado:** `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex`  
**Veredicto:** `APTO_PARA_REVISION_DEL_AUTOR`

El documento ya tiene una pregunta doctoral reconocible, hipótesis que pueden fallar y un diseño que pertenece a AutoRL y aprendizaje selectivo. No lo considero todavía una versión para enviar sin lectura del candidato: el piloto debe resolver compatibilidad, costo y precisión estadística, y la revisión sistemática puede obligar a reducir la afirmación de novedad.

## Hallazgos mayores y disposición

1. **El banco principal no justificaba las representaciones temporales.** ARLBench estudia optimización de hiperparámetros; usarlo como única prueba de memoria permitía que una red densa ganara por la naturaleza de las tareas, no por la calidad del selector. Se asignó a POPGym la prueba principal de observabilidad parcial, a CARL el cambio de contexto oculto y a ARLBench la referencia de AutoRL y costo.

2. **La novedad quedaba expuesta frente a POPGym.** POPGym ya compara trece mecanismos de memoria. La propuesta ahora dice expresamente que no pretende repetir ese ranking: estudia cuándo puede elegirse una representación para una tarea nueva usando evidencia de distinto costo y cuándo debe abstenerse.

3. **El ahorro de H1 no cerraba contablemente.** Se separó el costo de decisión, que incluye descriptores, curvas parciales, evaluaciones completas solicitadas y fallos, del costo común de las semillas ocultas usadas solo para juzgar el experimento. H1 exige simultáneamente ahorro y no inferioridad.

4. **"Riesgo" y "cobertura" eran atacables por vaguedad.** Se definió selección perjudicial mediante arrepentimiento simple mayor que un margen, riesgo selectivo como la proporción de errores entre las recomendaciones emitidas y cobertura como la proporción de tareas en las que se recomienda. Abstenerse siempre no satisface H2.

5. **La unidad experimental no protegía bien contra fuga ni pseudorreplicación.** El diseño usa pares entorno--dificultad o entorno--contexto; las semillas quedan anidadas. Los niveles de un mismo entorno POPGym permanecen juntos y se plantean cinco pliegues externos por entorno base.

6. **El tamaño efectivo seguía siendo modesto.** Las 45 unidades de POPGym provienen de quince entornos base, no de 45 grupos independientes. La propuesta ya no esconde esa limitación: H1 y H2 serán inconclusas si el límite unilateral no alcanza la precisión predeclarada.

7. **El resultado teórico era casi una definición.** Se añadió un modelo explícito de evidencia ruidosa y sesgada por tarea, representación y nivel; una meta de cota de costo ligada a separación, contracción de intervalos y costo; y una región de imposibilidad por indistinguibilidad.

8. **Había afirmaciones presupuestales derivadas más allá de la fuente.** Se eliminó toda conversión inventada. La cifra de ARLBench se presenta en los términos informados por el artículo: 937 GPU-horas para un presupuesto de 32 entrenamientos completos, repetido diez veces en tres subconjuntos.

9. **La tipografía subordinaba tablas y bibliografía.** Se retiraron reducciones locales de tamaño. El cuerpo, las tablas y las referencias usan el tamaño base de 11 puntos; se aceptó una novena página para conservar legibilidad.

10. **La bibliografía contenía un DOI equivocado.** ProxyBO apuntaba a otro artículo de AAAI. Se corrigió el DOI y la lista de autores, se verificó iMFBO y se incorporó la referencia primaria de POPGym.

11. **Persistía lenguaje propio del laboratorio.** Se retiraron términos como `gate`, `ledger`, `screen`, `branch`, `slot` y `runtime`; también se reemplazaron expresiones innecesariamente opacas como "priors", "cabezas" y "celdas".

12. **Durante esta auditoría se reintrodujo por error "multifidelidad" al comienzo del título.** Esto contradecía la decisión explícita registrada en `05_COBERTURA_PREGUNTAS_PROPUESTA.md`: el título debe nombrar primero el fenómeno en lenguaje común y reservar el término técnico para el resumen y el marco teórico. El título visible y los metadatos fueron restaurados a: *Selección de representaciones para aprendizaje por refuerzo bajo cambio de tarea, con abstención calibrada*.

## Riesgos que permanecen abiertos

1. El piloto debe demostrar que PPO, los cinco codificadores y los entornos seleccionados comparten una interfaz estable. El documento no afirma que esa integración exista hoy.
2. La precisión con quince entornos base puede no bastar. El resultado correcto en ese caso es `INCONCLUSO`, no contar semillas como tareas ni sumar CARL a la fuerza.
3. El techo provisional es de 2.250 curvas completas en POPGym y hasta 1.000 en CARL. Debe reemplazarse por GPU-horas, CPU-horas y almacenamiento medidos en el piloto y compatibles con los recursos disponibles.
4. La regresión bayesiana jerárquica y la calibración de intervalos son una propuesta concreta, no un resultado existente. Su especificación y sus controles deben congelarse antes de la evaluación.
5. La revisión sistemática del primer semestre debe verificar la novedad frente a selección de memoria, predicción de desempeño entre tareas y NAS multifidelidad. Si aparece un método equivalente, el alcance debe recortarse antes del prerregistro.

## Fuentes primarias comprobadas

- POPGym, ICLR 2023: <https://openreview.net/forum?id=chDrutUTs0K>
- ARLBench: <https://arxiv.org/abs/2409.18827>
- CARL, TMLR: <https://openreview.net/forum?id=Y42xVBQusn>
- iMFBO, UAI 2024: <https://proceedings.mlr.press/v244/fan24a.html>
- ProxyBO, AAAI 2023: <https://ojs.aaai.org/index.php/AAAI/article/view/26169>
- Conformal Risk Control, ICLR 2024: <https://openreview.net/forum?id=33XGfHLtZg>

## Verificación del artefacto

- Compilación con `latexmk` y `biber`: exitosa.
- Extensión: 9 páginas, papel carta, cuerpo de 11 puntos.
- Registro de LaTeX: sin referencias indefinidas, cajas desbordadas ni advertencias de composición.
- Inspección visual: las 9 páginas fueron renderizadas y revisadas; no se observaron superposiciones, cortes de tabla ni texto ilegible.
- SHA-256 del PDF final: `b22be0bfa5d76820099e2f660b6069a3dc7a402524c60cc93c2f20f1aeefe035`.

## Límite de esta auditoría

Esta revisión cubre coherencia científica, claridad, trazabilidad bibliográfica y presentación del documento. No ejecutó entrenamientos, no validó empíricamente los bancos y no convierte la evidencia interna del candidato en resultados doctorales.
