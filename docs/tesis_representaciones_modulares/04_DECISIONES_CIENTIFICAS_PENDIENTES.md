# Decisiones científicas pendientes

Estas decisiones no bloquean la lectura del borrador. Deben resolverse antes de congelar el protocolo confirmatorio.

| Decisión | Alternativas | Recomendación actual | Motivo | Efecto de cambiarla |
|---|---|---|---|---|
| Definición final del perfil temporal | ACF/espectro/estabilidad; wavelets; descriptores aprendidos | Empezar con ACF, espectro y estabilidad | Son interpretables, baratos y comprobables en sintético | Cambia H2, el costo y los vecinos directos. |
| Regla de agrupación | Jerárquica; mezcla; grafo; asignación suave | Jerárquica como mecanismo inicial | Produce una partición auditable y una ablación clara | Un método suave requeriría redefinir ramas y controles. |
| Cuadrícula de campos receptivos | Geométrica; cuantiles de escalas; dependiente del horizonte | Cuadrícula finita geométrica limitada por historia y horizonte | Evita una búsqueda continua escondida | Afecta costo, capacidad y definición de asignación aleatoria. |
| Familia principal de bloques | TCN; parches; atención restringida | TCN causal | El campo receptivo es explícito | Cambiarla exige comprobar que el efecto no pertenece al operador. |
| Segunda familia | PatchTST/patch encoder; ninguna | Condicionada al presupuesto del piloto | Sirve para robustez, pero no es parte del mínimo defendible | Puede retirarse sin cambiar H1-H3 principales. |
| Banco confirmatorio | TFB; subconjunto Monash multivariado; GIFT-Eval | Familias públicas con separación completa desarrollo-confirmación | La población debe tener heterogeneidad y licencias claras | Cambia la generalización que puede reclamarse. |
| Comparador multiescala principal | MTST; Pathformer; TimeMixer | Elegir uno por reproducibilidad y costo antes de observar resultados E1 | Ejecutar los tres puede dispersar recursos | Cambia el vecino multiescala, no la definición del método. |
| Comparador de agrupación | DUET obligatorio; MSGNet o CCM como robustez | Reproducir primero DUET y documentar cualquier diferencia de protocolo | DUET es el vecino publicado más directo de agrupación temporal y de canales | Si no puede reproducirse, la comparación y la reivindicación deben estrecharse. |
| Margen práctico de H1 | Derivado de piloto de precisión y uso | No fijarlo en el documento de admisión | No existe evidencia para inventar un porcentaje | Debe quedar congelado antes de E2. |
| Inferencia final | Bootstrap jerárquico; modelo multinivel bayesiano | Bootstrap jerárquico predeclarado | Menor dependencia de supuestos y lectura directa | Un modelo bayesiano exigiría priors y análisis de sensibilidad. |
| Alcance de E3 | Pronóstico financiero; RL; ambos; ninguno | Pronóstico primero, RL solo con compuertas superadas | E3 no debe rescatar la evidencia pública | Puede retirarse sin invalidar la tesis mínima. |
| Diagnóstico rezagado para H3 público | VAR regularizado; prueba de Granger predictiva; información mutua condicional | Elegir una medida calculable solo con entrenamiento y tratarla como predictibilidad, no causalidad | Los perfiles marginales no identifican relaciones cruzadas | Si no es estable, H3 queda limitada al experimento sintético. |
| Regla de decisión | Diferencia MASE con margen $\varepsilon$ | Fijar $\varepsilon$ y precisión mínima en E1 antes de E2 | Evita umbrales retrospectivos | Cambia potencia y tamaño del banco, no el signo de los efectos. |

## Preguntas que debe resolver la revisión sistemática

1. ¿DUET, CCM, REP-Net u otro trabajo posterior ya usa propiedades calculadas antes del entrenamiento para decidir simultáneamente grupos y campos receptivos y transferir la regla entre familias?
2. ¿Qué definición de heterogeneidad temporal permanece estable entre subventanas sin convertirse en un proxy del conjunto de datos?
3. ¿Qué bancos contienen suficientes tareas multivariadas y familias independientes para separar desarrollo y confirmación?
4. ¿Cómo igualar capacidad entre arquitectura monolítica, modular aleatoria e informada sin alterar sus campos receptivos?
5. ¿Qué costo debe gobernar cada contraste: parámetros, operaciones, memoria, tiempo de entrenamiento o una combinación publicada?
6. ¿La fusión temporal aporta algo fuera de tareas con dependencias rezagadas conocidas, o H3 debe limitarse a E0 y a un análisis secundario?

## Regla de recorte

El mínimo defendible conserva el contrato, el mecanismo, E0, E1, E2, el control simple y las ablaciones de agrupación/campo receptivo. Se retiran primero la segunda familia de bloques, la integración de CCM y E3. No se reduce la separación entre desarrollo y confirmación ni la contabilidad de costo.
