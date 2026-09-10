# Tres temas al abrir la entrevista

**Qué se envía por correo:** solo el tema 1.  
**Qué no se nombra:** incentivos, tokens, minería, DOIN como mercado. Eso no es objeto doctoral de esta entrevista.

Abrir con tres problemas, en este orden. Si alguno suena, el jurado tira del hilo. Luego: *el que me parece adecuado para este programa es el primero*, y se entra a justificación y objetivos del PDF.

---

## 1. Selección de representaciones para aprendizaje por refuerzo  
*con abstención calibrada, bajo cambio de tarea*

**PDF:** `01_seleccion_representaciones_rl.pdf`  
**Fuente:** `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex`

Veinte segundos. En RL con observabilidad parcial, elegir memoria o ventana exige entrenamientos completos. Estudio si se puede decidir con evidencia más barata, midiendo el error, y abstenerse cuando no se puede defender la elección.

---

## 2. Memorización, generalización y dimensionamiento de redes

**PDF:** `02_memorizacion_generalizacion_dimensionamiento.pdf`

Veinte segundos. Una red tiene un techo de almacenamiento y un uso cuando todavía generaliza. Estudio si se puede medir eso y estimar un tamaño suficiente, en datos cuya complejidad se conoce, sin llamar inteligencia al número.

---

## 3. Preprocesamiento informacional de series  
*ruido, SNR y lo que llega al extractor*

**PDF / texto:** `03_preprocesamiento_informacional.md`  
**Fuente larga:** `docs/propuesta_doctoral_preprocesamiento_informacional.md`

Veinte segundos. Al extractor no se le puede pedir lo que la entrada no hace explícito. Estudio si estimar ruido y tratarlo de forma causal mejora lo que el extractor convierte en parámetros, o si suavizar empeora la predicción. El primer experimento planta ruido gaussiano de potencia conocida. Si tiran a patrones: el extractor reconstruye la ventana; el filtro adaptado pregunta si un patrón está. No son el mismo plugin.

---

## Cómo no mezclarlos si preguntan

| Tema | Elige | No elige |
|---|---|---|
| 1 | Qué *codificador de estado* entrenar, con abstención | El filtro de la serie |
| 2 | Techo y tamaño de una *red* | El preproceso ni el agente |
| 3 | Qué *preproceso* llega al extractor | La política de RL ni el CI de la red |

Si preguntan por SAC: es el algoritmo de entrenamiento en el tema 1, no un cuarto título.  
Si preguntan por bits: en el tema 2 son el techo de la *red*; en el tema 3, amplitud (tramo 4) o tasa de fuente (tramo 5). Fase (tramo 6) es coordenada. Hilbert no sobre el precio crudo.  
Si preguntan por el extractor y los patrones: el AE reconstruye la ventana; el filtro adaptado (tramo 7) pregunta si un patrón está. Conv1D no es matched filter.  
Si preguntan por normalizar: E0 ya existe. Equalizar (tramo 8) es compensar un canal y guardar \(\mu,\sigma\); puede ser nulo.  
Si preguntan por factores de mercado: descomponer común/privado (tramo 9), no restar porque correlaciona.  
Si preguntan por lead–lag: metadato causal (tramo 10), no DTW retrospectivo ni macros sin vintage.  
Si preguntan por códigos de canal: máscara en el AE que ya existe (tramo 11), no un segundo extractor.  
Si preguntan por adaptar el enlace: elegir entre modos ya validados (tramo 12); si el oráculo no gana, uno basta. No es la abstención de L2.  
Si preguntan por capacidad de ramas: presupuesto del composite (tramo 13), no el techo \(C\) de la red (tema 2).
