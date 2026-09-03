# Matriz de estado del arte: selección multifidelidad de representaciones para RL

**Fecha de corte:** 2026-09-03  
**Función:** fijar los vecinos que delimitan la novedad antes de redactar la propuesta doctoral.  
**Regla:** una línea existente no se presenta como contribución propia; la propuesta debe explicar qué supuesto cambia y cómo se medirá esa diferencia.

## 1. Vecinos directos

| Línea | Resultado existente | Consecuencia para la propuesta |
|---|---|---|
| AutoRL e HPO en RL | Eimer, Lindauer y Raileanu muestran que los hiperparámetros cambian el desempeño y la eficiencia muestral, que el paisaje depende de la semilla de ajuste y que métodos de HPO pueden superar configuraciones manuales [1]. | No se afirmará que la selección de configuraciones en RL carece de método. El problema es más estrecho: decidir entre representaciones con evaluaciones de distinto costo y validez variable. |
| Benchmark público | ARLBench ofrece DQN, PPO y SAC en subconjuntos públicos de tareas, ejecución parcial y un metadataset de más de 100.000 corridas [2]. Su versión actual expone una superficie arquitectónica pequeña y reconoce que NAS y HPO basado en estados internos siguen subrepresentados. | ARLBench será el banco público principal. Habrá que extender de forma acotada la superficie de representación y publicar esa extensión; los datos existentes sirven para prototipos, no como etiquetas de las nuevas arquitecturas. |
| Asignación multifidelidad | Hyperband distribuye presupuesto y detiene configuraciones temprano [3]; BOHB combina esa asignación con un modelo probabilístico [4]. | Son comparadores obligatorios. Ganar solo contra búsqueda aleatoria o una configuración manual no basta. |
| Arquitecturas para RL | DARTS-RL muestra que la búsqueda diferenciable de arquitectura puede mejorar políticas en espacios discretos y continuos [5]. | La novedad no es buscar una arquitectura de RL. DARTS-RL será vecino bibliográfico y comparador secundario cuando el espacio sea compatible. |
| Fidelidad dependiente de la entrada | iMFBO aprende que una misma fuente barata puede ser más o menos fiel según la región consultada y demuestra una cota de arrepentimiento sublineal bajo su modelo [6]. | Decir solamente “la fidelidad depende de la configuración” no es novedoso. La propuesta debe tratar el caso en que el error no esté previamente identificado y la salida válida pueda ser abstenerse. |
| Identificación multifidelidad del mejor brazo | MF-BAI estudia aproximaciones baratas y sesgadas con cotas de sesgo conocidas [7]; trabajo posterior deriva complejidad de costo y fidelidad óptima por brazo [8]. | La teoría propuesta debe declarar qué envolvente de error se aprende o se calibra. Sin ella, solo procede un resultado de imposibilidad, no una garantía de selección. |
| Sondas baratas y predictores | Los indicadores de costo casi nulo pueden acelerar NAS [9]. ProxyBO estima durante la búsqueda cuánto confiar en distintas sondas [10], y Multi-Predict transfiere predictores entre tareas y espacios [11]. | Ni las sondas ni su ponderación dinámica son contribuciones por sí mismas. Aquí se evaluará si permiten una decisión calibrada en RL y se medirá explícitamente el daño cuando fallan. |
| Surrogates en RL | Dierkes et al. encuentran superficies de RL rugosas y ruidosas y desviaciones importantes de los predictores incluso cuando se usan solo para interpretar el paisaje [12]. | Es la objeción central. El método no prometerá predecir universalmente el retorno final; aprenderá el riesgo de una decisión y podrá escalar a evaluación completa o no recomendar. |
| Incertidumbre en AutoRL | ERAHBO modela media y varianza heteroscedástica y reasigna repeticiones para optimización adversa al riesgo [13]. | Modelar varianza no basta para reclamar novedad. La abstención se juzgará por cobertura y daño, no como sinónimo de aversión al riesgo. |
| Abstención | El aprendizaje en línea con abstención formaliza la decisión de no predecir a cambio de un costo [14]. | La abstención debe tener una acción y costo concretos: pagar la siguiente fidelidad o devolver “sin recomendación”. Siempre abstenerse no será una victoria. |
| Evaluación de RL | rliable recomienda intervalos, perfiles de desempeño e interquartile mean para evitar conclusiones frágiles con pocas corridas [15]. | La tarea o contexto no visto será la unidad; las semillas serán repeticiones anidadas. Se reportarán distribuciones e intervalos, no solo medias. |
| Cambio de contexto | CARL permite variar de manera controlada el contexto físico de entornos conocidos [16]; Procgen separa niveles de entrenamiento y prueba para estudiar generalización [17]. | Los cambios de distribución se evaluarán públicamente y no se inferirán de una sola serie financiera. La compatibilidad exacta ARLBench-CARL se decidirá en el piloto. |
| Reuso adaptativo | Thresholdout demuestra garantías para consultas adaptativas sobre un holdout aleatorio [18]. | Se adopta la disciplina de separar exploración, calibración y prueba. No se trasplanta su teorema a bloques temporales ni a retornos de RL. |
| Capacidad e información | MacKay deriva aproximadamente dos bits por peso para un único clasificador lineal de umbral, bajo entradas en posición general y una definición concreta de memorización [19]. | No es una fórmula para la inteligencia ni la capacidad de una red profunda. Compresibilidad, MDL o información mutua podrán entrar solo como descriptores candidatos y sobrevivirán únicamente si predicen decisiones fuera de muestra. |

## 2. Brecha que sí queda abierta

La literatura cubre HPO de RL, asignación multifidelidad, búsqueda de arquitecturas, uso de sondas, modelos de varianza y abstención por separado. El objeto que se propone estudiar es:

> **La selección de representaciones para aprendizaje por refuerzo cuando la evidencia barata tiene sesgo dependiente de la tarea y la configuración, ese sesgo puede dejar de ser identificable bajo cambio de distribución, y una salida permitida es escalar el presupuesto o abstenerse de recomendar.**

La contribución no será una nueva colección de arquitecturas. Será una regla de decisión que combine tres niveles de evidencia, estime el riesgo de eliminar o promover una representación y mantenga una garantía de selección o emita una abstención bajo condiciones explícitas.

## 3. Afirmación de novedad permitida

La propuesta puede afirmar que investigará una combinación poco estudiada y contrastable. No debe afirmar que es la primera solución hasta terminar una revisión sistemática. La afirmación provisional es:

> Se estudiará si una regla selectiva puede ahorrar evaluaciones completas al elegir representaciones de RL sin aumentar más allá de un margen predeclarado la selección perjudicial, incluso cuando la relación entre fidelidades cambia entre tareas; cuando esa relación no sea identificable, se buscará demostrar que abstenerse o escalar la fidelidad es necesario.

## 4. Comparadores mínimos

1. Búsqueda aleatoria con presupuesto igual.
2. Hyperband o ASHA.
3. BOHB o SMAC con Hyperband.
4. Un vecino que modele fidelidad o sondas, como iMFBO o ProxyBO, adaptado solo si sus supuestos y espacio lo permiten.
5. La regla propuesta sin abstención.
6. La regla propuesta con abstención o escalamiento.

DARTS-RL será un comparador secundario para espacios compatibles, no una obligación que deforme el diseño principal.

## 5. Referencias primarias verificadas

[1] T. Eimer, M. Lindauer y R. Raileanu, “Hyperparameters in Reinforcement Learning and How To Tune Them,” ICML, 2023. https://proceedings.mlr.press/v202/eimer23a.html

[2] J. Becktepe et al., “ARLBench: Flexible and Efficient Benchmarking for Hyperparameter Optimization in Reinforcement Learning,” *Journal of Data-centric Machine Learning Research*, 2026. https://data.mlr.press/assets/pdf/v03-3.pdf

[3] L. Li et al., “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization,” *JMLR*, 2018. https://www.jmlr.org/papers/v18/16-558.html

[4] S. Falkner, A. Klein y F. Hutter, “BOHB: Robust and Efficient Hyperparameter Optimization at Scale,” ICML, 2018. https://proceedings.mlr.press/v80/falkner18a.html

[5] Y. Miao et al., “Differentiable Architecture Search for Reinforcement Learning,” AutoML, 2022. https://proceedings.mlr.press/v188/miao22a.html

[6] M. Fan et al., “Multi-fidelity Bayesian Optimization with Multiple Information Sources of Input-dependent Fidelity,” UAI, 2024. https://proceedings.mlr.press/v244/fan24a.html

[7] R. Poiani, A. M. Metelli y M. Restelli, “Multi-Fidelity Best-Arm Identification,” NeurIPS, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html

[8] R. Poiani et al., “Optimal Multi-Fidelity Best-Arm Identification,” NeurIPS, 2024. https://papers.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html

[9] M. S. Abdelfattah et al., “Zero-Cost Proxies for Lightweight NAS,” ICLR, 2021. https://openreview.net/pdf?id=0cmMMy8J5q

[10] Y. Shen et al., “ProxyBO: Accelerating Neural Architecture Search via Bayesian Optimization with Zero-Cost Proxies,” AAAI, 2023. https://doi.org/10.1609/aaai.v37i8.26169

[11] Y. Akhauri y M. S. Abdelfattah, “Multi-Predict: Few Shot Predictors For Efficient Neural Architecture Search,” AutoML, 2023. https://proceedings.mlr.press/v224/akhauri23a.html

[12] J. Dierkes, T. Eimer, M. Lindauer y H. Hoos, “Performance Prediction in Reinforcement Learning: The Bad and the Ugly,” EWRL, 2025. https://openreview.net/pdf?id=L9J6Xmta4J

[13] M. Che et al., “Efficient Heteroscedastic Bayesian Optimization for Risk-Aware AutoRL,” RLC, 2026. https://arxiv.org/abs/2607.26680

[14] C. Cortes et al., “Online Learning with Abstention,” ICML, 2018. https://proceedings.mlr.press/v80/cortes18a.html

[15] R. Agarwal et al., “Deep Reinforcement Learning at the Edge of the Statistical Precipice,” NeurIPS, 2021. https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

[16] C. Benjamins et al., “Contextualize Me: The Case for Context in Reinforcement Learning,” *TMLR*, 2023. https://openreview.net/pdf?id=Y42xVBQusn

[17] K. Cobbe et al., “Leveraging Procedural Generation to Benchmark Reinforcement Learning,” ICML, 2020. https://proceedings.mlr.press/v119/cobbe20a.html

[18] C. Dwork et al., “Generalization in Adaptive Data Analysis and Holdout Reuse,” NeurIPS, 2015. https://proceedings.neurips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html

[19] D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, cap. 40, Cambridge University Press, 2003. https://www.inference.org.uk/itprnn/book.pdf
