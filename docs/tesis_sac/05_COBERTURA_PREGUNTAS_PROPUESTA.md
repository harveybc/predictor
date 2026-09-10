# Cobertura de preguntas — propuesta L2/RL

**Para:** Harvey  
**Fecha:** 2026-09-03  
**Documento cubierto:** `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex` compilado a PDF (Times, citas IEEE, 11 páginas).  
**Compilar:** `make -C docs`.  
**Linaje:** HTML WeasyPrint → reconstrucción de Musashi → esta pasada en LaTeX.  
**Uso:** archivo de estudio. No es anexo para el jurado salvo que tú lo decidas. Las citas `[n]` coinciden con el PDF. Las `[A]–[C]` son extras de estudio.

Regla: si no abrí el PDF primario, lo digo. No relleno huecos.

---

## 0. Decisiones de esta pasada

**Título.** «Multifidelidad» no va de primero. No es una palabra que un jurado de IA en Colombia tenga por costumbre. El título ahora es el fenómeno en castellano: *Selección de representaciones para aprendizaje por refuerzo bajo cambio de tarea, con abstención calibrada*. La palabra queda en palabras clave y se define en el resumen con Peherstorfer et al., *SIAM Review* 60(3):550–591, 2018.

**LaTeX.** La Sabana no publica una plantilla `.tex` de *propuesta* doctoral. El estándar de hecho en ingeniería e IA es `article` 11pt, carta, Times, bibliografía IEEE (`biblatex-ieee`). NTC 1486 aplica a la tesis final, no a este anteproyecto. El HTML queda como copia de trabajo anterior; la fuente canónica es el `.tex`.

**Páginas.** 11 en LaTeX, con tablas ancladas y ecuaciones numeradas. Musashi quería 8; el comité lee peor un HTML recortado que un artículo bien compuesto.

**Paréntesis.** Tu directora tenía razón. En el PDF casi no quedan. Lo que era inciso entre paréntesis pasó al hilo del párrafo o a una frase siguiente. Los únicos paréntesis que quedan son notación, \(c^*\) y \(ĉ\).

**Glosario.** Sigue sin página propia. Las plantillas colombianas (MinCiencias Anexo 2, CES GU-IN-013, UNAD) piden definir en el marco, no un glosario suelto. La tabla 2.4 está *después* de justificación y objetivos, como pidió Musashi.

**644 unidades.** Musashi las puso como motivación. **No las repetí.** Busqué el número en el expediente de `tesis_sac/` y no está. Lo que sí está auditado en `03_PRESUPUESTO_L1.md` y `04_AUDITORIA_MUSASHI_PRE_PDF.md`: cinco familias, valor relativo en algunas, ninguna con habilidad absoluta, fusión que no avanza, preflight de 2.000 pasos y 1.000 actualizaciones SAC sin checkpoint promovible. Eso es lo que dice ahora el §7.1. Si tienes el log de las 644, se puede reponer como motivación, no como prueba.

**Thresholdout.** Se conservó el límite explícito que añadió Musashi, con la cita de *Science* [23], que verifiqué: Dwork et al., vol. 349, no. 6248, pp. 636–638, 2015. El PDF dice que **no** se trasplanta el teorema a retornos de RL.

**Procedimiento.** El paso 3 ya no usa \(c^*\) verdadero, que el selector no observa. Usa \(U_t\), el extremo superior de los intervalos vigentes. Si un jurado aprieta ahí, esa es la respuesta.

---

## 1. Qué se conservó de Musashi y qué se afiló

| Pieza de Musashi | Estado |
|---|---|
| Justificación y objetivos antes del vocabulario | Conservado. |
| Encaje en IA una sola vez | Conservado, ahora en una frase del §2.1. |
| Evidencia interna como motivación | Conservado el *rol*; recortado el 644 no sourced. |
| Potencia, bancos públicos, comparadores, abstención falsable | Conservado. |
| Límite de Thresholdout | Conservado y precisado: holdout aleatorio ≠ curvas de RL. |
| Cronograma semestral S1–S6 | Conservado. |
| Resultado mínimo defendible | Conservado, en recuadro propio. |
| 23 referencias | Conservadas. ERAHBO [14]: arXiv:2607.26680, *Accepted at RLC'26*, verificado en esta pasada. ARLBench 937 GPU-h, verificado en el PDF de DMLR. Dierkes EWRL 2025, en la página de publicaciones de Eimer. |

---

## 2. Preguntas difíciles por sección

Respuestas para leer en voz alta. Si no se sabe, se dice.

### Resumen

**P0. ¿Por qué «multifidelidad» no abre el título?**  
Porque un jurado puede no haberla oído. El título nombra el objeto: seleccionar representaciones bajo cambio de tarea, con abstención. El término se introduce en el resumen como lo definen Peherstorfer, Willcox y Gunzburger~\cite{peherstorfer2018survey}: combinar un modelo caro con modelos más baratos que aproximan la misma cantidad. Luego se sitúa en HPO (Hyperband/ASHA) y en identificación de brazos. Sin el comentario de «no es un invento nuestro».

**P0b. «Fidelidad 0, 1 y 2», ¿de qué norma salen?**  
De ninguna. La *palabra* fidelidad es de Peherstorfer et al. Los índices 0/1/2 son el recorte operativo de esta tesis, declarados en la tabla de definiciones y otra vez en el §3.2.

**P1. ¿Por qué tres opciones y no un ranking tipo Hyperband?**  
Hyperband elimina por curvas parciales [3]. No estima el riesgo de una selección perjudicial ni puede devolver «sin recomendación». La opción de rechazo es de Chow [18]; la curva riesgo-cobertura, de Geifman y El-Yaniv [20].

**P2. «Abstenerse no es una orden de trading.» Un jurado de finanzas va a insistir.**  
El agente entrenado, en cualquier dominio, sigue su política. La abstención es del *selector*: no recomienda una representación. Idéntica en Atari, Brax o series.

**P3. ¿Por qué dos bancos públicos y además el financiero?**  
Porque H1 y H2 no pueden depender de un agente del autor. ARLBench es el primario [2]. CARL [17] o Procgen [22] cubren H3. El financiero son 40 corridas posteriores, no el estándar.

**P4. ¿Procgen duplica los 2.160 equivalentes?**  
No. Subconjunto predeclarado dentro de la envolvente. Si CARL se integra en el piloto, Procgen no se corre como corpus paralelo.

### §1 Problema

**P5. Eimer 2023 ¿no mata la tesis?**  
Eimer, Lindauer y Raileanu muestran que HPO formal en RL ya existe y gana a lo manual [1]. La tesis lo acepta. La brecha es seleccionar *representaciones* cuando la evidencia barata tiene sesgo que depende de la tarea.

**P6. Dierkes es un workshop.**  
EWRL 2025. Sirve como *objeción*, no como SOTA de método. Los autores de ARLBench dicen ahí que los predictores de RL se desvían del resultado real incluso para inspeccionar el paisaje [13].

### §2 Justificación, objetivos, definiciones

**P7. «Un sistema que dirige el entrenamiento de otro»: ¿eso no es AutoML?**  
Es AutoRL [1], [2]. El encaje en IA se dice una vez. La contribución no es el nombre; es la abstención calibrada bajo sesgo de fidelidad dependiente de tarea.

**P8b. ¿«Nivel 1» y «nivel 2» no chocan con fidelidad 0, 1 y 2?**  
Chocaban. El PDF ya no numera el entrenamiento del agente como «nivel 1». Una cosa es el agente que se entrena; otra, la regla que elige qué evaluar; otra, los tres recortes de costo. Tres vocabularios distintos, sin reutilizar «nivel».  
NAS busca celdas o grafos [6], [10]. Aquí entran ventana, fusión y, si el piloto lo incluye, un objetivo de preentrenamiento. El algoritmo de control se fija. DARTS-RL es vecino, no el objeto.

**P9. ¿Quién definió fidelidad 0/1/2?**  
La *palabra* «fidelidad» es estándar en optimización e identificación de brazos [7]–[9]. El recorte en 0, 1 y 2 es operativo de esta tesis y se prerregistra. El PDF lo dice dos veces: en la tabla 2.4 y en el §3.2. Si un jurado pregunta «¿de qué norma salen?», la respuesta es: de ninguna. De este método.

**P10. Arrepentimiento simple vs. acumulado.**  
Audibert, Bubeck y Munos: \(\mu^*-\mu_{\hat a}\) del brazo *finalmente recomendado*, no la suma durante la exploración [19]. H1 no es «el selector gana mientras busca».

### §2.5 Hipótesis

**P11. ¿Cuál es el margen adverso de H1?**  
Un \(\Delta\) de no inferioridad, en arrepentimiento simple normalizado, fijado en calibración. No hay número ahora. Inventarlo sería teatro.

**P12. Si el selector gana solo al excluir el costo de las sondas.**  
H1 falla. Ese costo entra al denominador.

**P13. Abstenerse siempre da riesgo 0.**  
Por eso existe el piso de cobertura \(\gamma\) [18], [20]. H2 exige las dos: menos selección perjudicial *y* \(\gamma\).

**P14. H3: «declara riesgo admisible».**  
Si emite recomendación, está diciendo \(P(\text{perjudicial})\le\alpha\). Si luego F2 supera el margen, esa unidad viola la calibración.

### §3 Marco

**P15. \(V_t\) de Atari no es comparable con CartPole.**  
Se normaliza \(R_t\) por el rango *de esa tarea*. La agregación entre tareas es IQM de esos \(R_t\) ya normalizados [16].

**P16. Si no evaluamos las 12 en F2, no hay \(c^*\).**  
El corpus de prueba reserva 12×5 semillas de curva precisamente para computar \(c^*\) *a posteriori* como verdad de nivel 1. Cada método solo ve las consultas que paga. \(c^*\) es el mejor del espacio prerregistrado, no de todas las redes posibles.

**P17. iMFBO ya modela fidelidad dependiente de la entrada [7]. ¿Qué queda?**  
No cubre: la envolvente puede no ser identificable entre tareas de RL, y abstenerse es una salida válida. Si iMFBO adaptado gana H1, H1 falla para nosotros. Por eso es comparador.

**P18. El respaldo \(c_0\) sesga H1.**  
Sí. Si \(c_0\) es malo, abstenerse se penaliza. Por eso \(c_0\) es la arquitectura por defecto del *banco*, no la config histórica.

**P19. Cortes 2018 es online. Nosotros somos batch por tarea.**  
Aplica el vocabulario, no su teorema de regret online. Hay que decirlo.

**P20. El resultado formal, ¿ya está en MF-BAI?**  
MF-BAI da complejidad con sesgo conocido [8], [9]. No he leído en esos papers la frontera de imposibilidad cuando dos tareas producen el mismo historial barato y distinto \(c^*\) en F2. Si al formalizar resulta que ya está, se cita y se recorta la novedad.

**P21. ¿Por qué no MacKay?**  
Dos bits por peso es un clasificador lineal de umbral [A]. No mide un agente SAC. Un jurado lo leería como relleno. Los descriptores MDL/IM solo sobreviven si ganan a parámetros y FLOPs.

### §4 Estado del arte

**P22. ERAHBO [14], ¿lo leíste?**  
Verifiqué la ficha: arXiv:2607.26680, autores Che, Tseng, Eimer, Lindauer, von Rohr, *Accepted at RLC'26*. El abstract modela media y varianza y reasigna repeticiones. **No reabrí el PDF completo.** Si Musashi exige teoremas internos, se abre antes de enviar.

**P23. «No se afirmará que es el primer método.» Entonces ¿dónde está la novedad?**  
En la combinación contrastable: representaciones de RL + error de fidelidad dependiente de tarea + abstención con cobertura y costo, más la región de imposibilidad. Incremental no es fraude. Afirmar unicidad sin revisión sistemática sí.

### §5 Método

**P24. \(n=12\) es ridículo.**  
Es el máximo de unidades de *prueba* que cabe en 2.160 equivalentes con 12 representaciones y 5 semillas. Con potencia 0,80 y \(\alpha=0{,}05\), \(d\approx 0{,}89\); con Holm a tres contrastes, \(\approx 1{,}08\). Cálculo: \(t\) no central, df=11. Solo efectos grandes. Si el piloto mide un \(\sigma\) que deja el margen por debajo, transferencia inconclusa. No se bajan semillas para fabricar \(p\).

**P25. ¿Por qué 5 semillas y no 10 como ARLBench?**  
ARLBench: 937 GPU-h para 32 entrenamientos completos × 10 semillas en los tres subconjuntos de algoritmo [2]. La cuenta de 4.480 equivalentes es 32 × 10 × 14, la que usa el presupuesto interno. Nosotros no tenemos ese presupuesto si además extendemos representaciones. 5 de curva + 5 finales. Si 5 no basta para IQM estable, se reduce el espacio, no las unidades de prueba.

**P26. Un jurado no tiene por qué confiar en tu agente.**  
Por eso el PDF ya no dice esa frase. Dice: las hipótesis se contrastan en bancos publicados para que el resultado no dependa de un agente propio.

**P27. ¿Por qué 12 representaciones?**  
Mayor grilla discreta en 2.160 equivalentes: 12 × 30 × 5 = 1.800, más 6 × 12 × 5 = 360. Dos modos de acceso × cuatro extractores = 8; cuatro combinan reducción y objetivo auxiliar. El piloto puede reducir, no ampliar.

**P28. El paso 3 usa \(U_t\), no \(c^*\). ¿Eso no es circular?**  
Es una cota: perjudicial respecto de lo *plausible*, no respecto de un oráculo. Chow rechaza si el máximo a posteriori es menor que un umbral [18]. Aquí el umbral se aplica a la distancia a \(U_t\).

**P29. «Reducción esperada de riesgo por GPU-h» es knowledge gradient.**  
Es el criterio de adquisición, el espíritu de MF-BAI al elegir brazo y fidelidad [8], [9]. La familia de intervalos la elige el piloto. No se promete un KG gaussiano.

**P30. Calibración conforme en RL no estacionario.**  
Opción del piloto. Si la conformidad exige intercambioabilidad que las curvas no tienen, se descarta. No se trasplanta el teorema.

### §6 Diseño

**P31. ¿Por qué no hay comparador «humano / config histórica»?**  
Porque esa config fue informada por el mismo proceso que motiva la tesis. Eimer et al. ya muestran que HPO formal gana a lo manual [1].

**P32. iMFBO o ProxyBO «solo si conservan supuestos».**  
El piloto lo escribe *antes* de la prueba. Si la adaptación cambia el espacio o el sesgo, no entran. Perder un vecino no agranda la novedad; reduce el rigor del contraste.

**P33. Thresholdout [23] vs. el NeurIPS de 2015.**  
*Science* 349:636–638 es el artículo corto del holdout reutilizable, con Thresholdout. El NeurIPS [B] es la versión larga. El PDF cita *Science* y niega el trasplante: holdout aleatorio con privacidad diferencial ≠ retornos temporales de SAC.

### §7 Viabilidad

**P34. «Evaluación interna de cinco familias.» ¿Dónde están los plots?**  
No son evidencia doctoral. No hay paper, no hay protocolo cerrado de *esta* tesis. Motivaron la pregunta. Un jurado que pregunte «¿ya lo probaron?» debe oír: no, y no vamos a colar la bitácora como resultado.

**P35. 937 / 4.480 = 0,209 GPU-h. ¿De dónde 4.480?**  
32 configuraciones × 10 semillas × 14 tareas representativas, la cuenta del presupuesto interno alineada con el banco. El paper dice 937 GPU-h para 32 entrenamientos completos con 10 semillas en los tres subconjuntos [2]. Mezcla su hardware. Por eso hay envolvente ×4 = 1.810 h, y P0 la reemplaza. No es una ETA de este laboratorio.

**P36. ¿La tesis sobrevive si H1 y H2 fallan?**  
El resultado mínimo: caracterizar cuándo la evidencia barata *no* basta. Un nulo publicado, con comparadores fuertes y protocolo cerrado, es tesis. Un positivo fabricado no lo es.

---

## 3. Lo que Musashi o Satoshi van a apretar

**M1. Esto ya es Hyperband.**  
Hyperband asigna presupuesto [3]. No calibra \(P(\text{perjudicial})\) ni se abstiene con piso de cobertura.

**M2. ProxyBO ya pondera sondas.**  
Sí [11]. La novedad no es ponderar. Es no recomendar cuando la relación sonda–F2 no es identificable.

**M3. Un surrogate de RL no es confiable.**  
De acuerdo [13]. Por eso hay comparador sin predictor y puerta de abandono.

**M4. \(n=12\) no da potencia.**  
De acuerdo para efectos chicos. El PDF lo dice.

**M5. El financiero es el hobby.**  
Por eso no es H1 ni H2.

**S1. ¿Una sola caja teórica?**  
Sí: selección con rechazo + imposibilidad. Thresholdout es disciplina experimental, no teorema heredado.

**S2. ¿Las sondas se validan fuera del stack que las inventó?**  
Sí. Umbrales en calibración pública. Si fallan ahí, se retiran.

**S3. ¿La config histórica es el folclor?**  
Prohibida como control principal.

---

## 4. Referencias extra de estudio

[A] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, cap. 40. Cambridge University Press, 2003. No está en el PDF.

[B] C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold y A. Roth, “Generalization in Adaptive Data Analysis and Holdout Reuse,” en *Advances in Neural Information Processing Systems*, vol. 28, 2015. Versión larga del [23].

[C] Cálculo de MDE de esta línea de diseño: contraste \(t\) pareado, \(n=12\), df=11, potencia 0,80, bilateral, `scipy.stats.nct`. \(d\approx 0{,}89\) a \(\alpha=0{,}05\); \(d\approx 1{,}08\) a \(\alpha=0{,}05/3\). No es un resultado publicado; es aritmética del diseño.

---

## 5. Lo que este archivo no cubre

- Las 12 arquitecturas concretas y las 30 unidades: las cierra el piloto.
- Los umbrales \(\alpha, \gamma, \varepsilon, \Delta\): fijarlos ahora sería inventar.
- El PDF completo de Che et al. [14]: ficha de arXiv verificada; teoremas internos no releídos aquí.
- El log de 644 unidades: no está en este repositorio. Si aparece, se discute como motivación, no como prueba de H1–H3.
