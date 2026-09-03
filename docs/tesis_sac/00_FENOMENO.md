# La página del fenómeno

**Propósito:** una página. Si al leerla no se siente lo que se sintió con la
prueba de trabajo de optimización en la maestría, se descarta y se sigue
buscando. Si se siente, esto es la tesis.

---

Entrenar un agente de aprendizaje por refuerzo profundo sobre series de
tiempo financieras obliga a decidir, antes de cada corrida, una arquitectura
de representación: qué familias de características se agrupan, qué objetivos
de pretrenamiento se usan, qué ventanas, qué fusión, qué cabezas. Cada una
de esas decisiones cambia el resultado económico del agente.

Evaluar honestamente UNA de esas decisiones cuesta días de GPU, porque el
veredicto de nivel 1 (¿este agente opera mejor?) exige entrenamientos
completos con semillas repetidas. La evaluación es además **ruidosa** — la
varianza entre semillas rivaliza con el efecto que se busca —, **no
estacionaria** — el dato de mañana no viene de la distribución de ayer — y
**contaminable** — cada reuso del conjunto de prueba, cada fuga entre
particiones, cada selección tras mirar el resultado, gasta silenciosamente
la validez de la evidencia.

Hoy esa decisión se toma por folclor: se copia la arquitectura del último
artículo, se prueba a mano un puñado de variantes, se elige la que ganó una
vez, y nadie contabiliza cuánta validez quedaba en el conjunto de prueba
cuando se eligió. Nuestra propia bitácora de campaña documenta el precio del
folclor: familias enteras de características que resultaron información
muerta; un tratamiento de currículo que fue inerte en 12 de 12 corridas; un
programador de tasa de aprendizaje que empeoraba lo que prometía mejorar;
dos de cinco familias de pretrenamiento que no superan a un codificador
aleatorio. Cada uno de esos descubrimientos costó semanas de GPU que un
método sistemático habría cobrado en horas — o habría evitado.

**El fenómeno, en una frase:** la selección de representaciones para RL
profundo es una optimización de nivel 2 cuya función objetivo es carísima,
ruidosa, cambiante y degradable por el propio acto de consultarla — y la
práctica actual la resuelve sin método, sin presupuesto de consultas y sin
saber cuándo abstenerse.

**La propuesta, en una frase:** un método de meta-optimización **selectiva**
(se abstiene cuando la evidencia no alcanza un umbral predeclarado de
calibración) que busca en ese espacio con presupuesto acotado, usando
diagnósticos baratos cuya capacidad de predecir el resultado costoso se
mide y no se supone, bajo un protocolo de evidencia con particiones
purgadas, contratos sellados y presupuesto explícito de reuso del conjunto
de prueba — de modo que la decisión final llegue con su validez restante
contabilizada.

**Por qué es inteligencia artificial y no ingeniería:** el objeto es cómo un
sistema aprende a dirigir otro proceso de aprendizaje bajo evidencia cara y
degradable, y cómo reconoce los límites de lo que esa evidencia autoriza.
Aprendizaje de representaciones, meta-optimización, calibración y
abstención, análisis adaptativo de datos. El agente de trading es el
dominio de confirmación, no el objeto.

**Lo que este fenómeno NO incluye:** mercados de cómputo, incentivos,
tokens, consenso distribuido. La ejecución distribuida de campañas de
búsqueda en una flota propia es infraestructura de viabilidad — una línea,
no una hipótesis.
