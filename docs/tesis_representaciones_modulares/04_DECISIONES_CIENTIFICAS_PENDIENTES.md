# Decisiones científicas pendientes

La revisión 4 ya cerró la unidad estadística, la agregación de H1, la ruta inferencial, los contrastes de H2/H3, sus controles, la apertura de la reserva sintética, la política rodante y la magnitud principal de costo. La tabla conserva solo decisiones que deben resolverse con la revisión sistemática o el piloto antes de congelar el protocolo confirmatorio.

| Decisión | Alternativas | Recomendación actual | Motivo | Efecto de cambiarla |
|---|---|---|---|---|
| Definición final del perfil temporal | ACF/espectro/estabilidad; wavelets; descriptores aprendidos | Empezar con ACF, espectro y estabilidad | Son interpretables, baratos y comprobables en sintético | Cambia H2, el costo y los vecinos directos. |
| Regla de agrupación | Jerárquica; mezcla; grafo; asignación suave | Jerárquica como mecanismo inicial | Produce una partición auditable y una ablación clara | Un método suave requeriría redefinir ramas y controles. |
| Cuadrícula de campos receptivos | Geométrica; cuantiles de escalas; dependiente del horizonte | Cuadrícula finita geométrica limitada por historia y horizonte | Evita una búsqueda continua escondida | Afecta costo, capacidad y definición de asignación aleatoria. |
| Segunda familia | PatchTST/patch encoder; ninguna | Condicionada al presupuesto del piloto | Sirve para robustez, pero no es parte del mínimo defendible | Puede retirarse sin cambiar H1-H3 principales. |
| Banco confirmatorio | TFB; subconjunto Monash multivariado; GIFT-Eval | Familias públicas con separación completa desarrollo-confirmación | La población debe tener heterogeneidad y licencias claras | Cambia la generalización que puede reclamarse. |
| Comparador multiescala principal | MTST; Pathformer; TimeMixer | Elegir uno por reproducibilidad y costo antes de observar resultados E1 | Ejecutar los tres puede dispersar recursos | Cambia el vecino multiescala, no la definición del método. |
| Comparador de agrupación | DUET obligatorio; MSGNet o CCM como robustez | Reproducir primero DUET y documentar cualquier diferencia de protocolo | DUET es el vecino publicado más directo de agrupación temporal y de canales | Si no puede reproducirse, la comparación y la reivindicación deben estrecharse. |
| Diagnóstico rezagado para H3 público | VAR regularizado; prueba de Granger predictiva; información mutua condicional | Elegir una medida calculable solo con entrenamiento y tratarla como predictibilidad, no causalidad | Los perfiles marginales no identifican relaciones cruzadas | Si no es estable, H3 queda limitada al experimento sintético. |
| Margen y precisión | Valores compatibles con relevancia práctica y soporte del banco | Fijar $\varepsilon$ por relevancia; usar el piloto solo para comprobar resolución | Evita elegir el margen para obtener significación | Puede producir un veredicto inconcluso o exigir ajustar el tamaño del banco antes de E2. |
| Instrumento de energía | Telemetría del dispositivo; medidor externo; estimador calibrado | Fijarlo y validarlo en E1 | $C$ se definió en Wh, pero el instrumento todavía debe demostrar resolución y cobertura | Afecta la incertidumbre del costo, no el contraste primario de error. |

## Preguntas que debe resolver la revisión sistemática

1. ¿DUET, CCM, REP-Net u otro trabajo posterior ya usa propiedades calculadas antes del entrenamiento para decidir simultáneamente grupos y campos receptivos y transferir la regla entre familias?
2. ¿Qué definición de heterogeneidad temporal permanece estable entre subventanas sin convertirse en un proxy del conjunto de datos?
3. ¿Qué bancos contienen suficientes tareas multivariadas y familias independientes para separar desarrollo y confirmación?
4. ¿Cómo igualar capacidad entre arquitectura monolítica, modular aleatoria e informada sin alterar sus campos receptivos?
5. ¿Qué instrumento mide con suficiente resolución la energía de cada fase y cómo se valida frente a trabajos cortos y hardware heterogéneo?
6. ¿El diagnóstico público de predictibilidad cruzada es suficientemente estable para justificar un análisis complementario, o H3 debe limitarse a su contraste sintético reservado?

## Regla de recorte

El mínimo defendible conserva el contrato, el mecanismo, E0, E1, E2, el control simple y las ablaciones de agrupación/campo receptivo. Se retiran primero la segunda familia de bloques, la integración de CCM y E3. No se reduce la separación entre desarrollo y confirmación, la reserva sintética ni la contabilidad de costo.
