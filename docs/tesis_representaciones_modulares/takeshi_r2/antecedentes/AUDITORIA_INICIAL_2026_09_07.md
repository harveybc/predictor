# Auditoría de la propuesta «Diseño y aprendizaje de representaciones temporales modulares»

**Revisor:** Takeshi  
**Destinatarios:** Harvey y Musashi  
**Fecha:** 7 de septiembre de 2026  
**Versión:** predictor, commit 249f800dd850ec0899428cac3db7bfd7c3e922a0  
**PDF SHA-256 comprobado:** ddaee2505e3358f48e279f07149e23261788055b06f738dbdf84821ad5bc9fd7  
**Dictamen:** CONSERVAR LA DIRECCIÓN / CORREGIR ANTES DE CERRAR LA LECTURA FINAL.

## 1. Resultado ejecutivo

La reestructuración resolvió el desajuste principal: el documento ya estudia cómo construir y entrenar una representación modular. El selector de curvas, la abstención calibrada y la coordinación de cómputo dejaron de ocupar el centro. Se mantiene un título claro y un mecanismo concreto que permite organizar una investigación: perfiles temporales, agrupación, alcance de ramas y entrenamiento conjunto.

Recomiendo trabajar sobre esta nueva versión. No recomiendo volver al documento del selector ni abrir otra reestructuración general. El borrador necesita cerrar ocho grupos de problemas de definición y evaluación, además de mejorar varias formulaciones editoriales. Estos hallazgos no demuestran que el mecanismo vaya a fallar; muestran dónde la propuesta todavía permite interpretar o ejecutar experimentos distintos.

Los cuatro puntos más importantes son:

1. Las reglas de decisión de H1–H3 no distinguen correctamente beneficio, equivalencia, perjuicio e incertidumbre.
2. El campo receptivo de una activación no determina por sí solo cuánta historia utiliza el predictor completo. El cabezal aún está insuficientemente definido para interpretar H3.
3. Los perfiles marginales de cada variable no identifican relaciones predictivas entre variables, y la normalización de la heterogeneidad puede eliminar precisamente la variación que H2 quiere estudiar.
4. Las unidades de separación, evaluación y remuestreo necesitan una jerarquía común. Separar ventanas o añadir horizontes no crea nuevas familias independientes.

**Aclaración de severidad:** P1 significa que la ambigüedad puede cambiar el mecanismo, el contraste o la conclusión; debe resolverse en el texto o mediante un compromiso metodológico concreto. P2 significa mejora importante de claridad, trazabilidad o presentación. No se exige ejecutar el doctorado para presentar la propuesta, ni fijar ahora todos los hiperparámetros.

## 2. Evidencia revisada y límites de la auditoría

### 2.1 Expediente del commit

| Documento | Revisión realizada |
|---|---|
| [PDF](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/propuesta_doctoral_representaciones_temporales_modulares.pdf) | Descarga, hash, extracción de texto y revisión visual de las nueve páginas. |
| [LaTeX](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/propuesta_doctoral_representaciones_temporales_modulares.tex) | Lectura completa. Blob 744b0a08e350dcd13e6024c63772154fcb399d4c. |
| [Bibliografía](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/propuesta_doctoral_representaciones_temporales_modulares.bib) | Lectura, correspondencia con citas y contraste dirigido de referencias relevantes. |
| [Contrato conceptual](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/tesis_representaciones_modulares/01_CONTRATO_CONCEPTUAL_Y_DECISIONES.md) | Lectura completa y comparación con el PDF. |
| [Matriz de antecedentes](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/tesis_representaciones_modulares/02_MATRIZ_ANTECEDENTES.md) | Lectura completa y examen de vecinos directos. |
| [Registro C01–C14](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/tesis_representaciones_modulares/03_REGISTRO_REESTRUCTURACION_C01_C14.md) | Lectura completa y contraste de resoluciones con el método. |
| [Decisiones pendientes](https://github.com/harveybc/predictor/blob/249f800dd850ec0899428cac3db7bfd7c3e922a0/docs/tesis_representaciones_modulares/04_DECISIONES_CIENTIFICAS_PENDIENTES.md) | Lectura completa; varias alternativas son legítimas, otras cambian el significado de las hipótesis. |

Se comprobó: nueve páginas carta, metadatos con el título correcto, nueve fuentes listadas por la inspección del PDF y todas embebidas, 27 referencias citadas sin claves faltantes. El archivo bibliográfico contiene 28 entradas: RevIN está almacenada pero no citada. Esto no contradice que el PDF tenga 27 referencias.

No se recompiló el LaTeX ni se recibió su registro de compilación. Por ello, la afirmación de cero advertencias corresponde a Musashi; la inspección independiente confirma el PDF entregado, no el estado de su compilador. No se entrenaron modelos ni se reprodujeron resultados de los artículos. Las demostraciones sencillas de §5 son contraejemplos matemáticos del revisor, no experimentos del proyecto.

## 3. Alineación: lo que debe conservarse y la decisión que debe explicitarse

### 3.1 Conservar

- El título aceptado por Harvey.
- El pronóstico como consumidor principal propuesto: facilita examinar el efecto de las representaciones y no obliga a sostener todo el estudio sobre trading.
- El aprendizaje conjunto de ramas, fusión y cabezal.
- El uso de una familia temporal controlable para estudiar el mecanismo.
- E0/E1/E2 como separación entre mecanismo, desarrollo y confirmación.
- Los comparadores simples y los controles de capacidad y costo.
- La aplicación financiera como réplica separada que no cambia el veredicto principal.
- DOIN, DEAP y OLAP como infraestructura disponible.

### 3.2 A01 — recorte válido, todavía provisional respecto de la intención amplia

**Ubicación:** resumen, §§4.1–4.2 y fila C04 del registro.

El autor explicó que quiere estudiar cómo adaptar la representación de entrada a las propiedades de los datos y de los modelos. El borrador concreta solo una parte de ese problema: conserva una entrada normalizada y modifica agrupaciones y campos receptivos en una familia TCN. No compara, en su núcleo, transformaciones o descomposiciones de entrada adaptadas a distintas familias de operadores.

Ese recorte puede ser una muy buena elección doctoral. No debe presentarse como si cubriera automáticamente toda la intención del autor. La fila C04 afirma que se habilitan transformaciones específicas, pero el método no identifica ninguna como intervención experimental principal.

**Instrucción:** describirlo como delimitación candidata y corregir el registro. No añadir por reflejo wavelets, autoencoders, LSTM y Transformer a la matriz experimental. Primero justificar por qué la asignación de alcance y agrupación es la instancia elegida del problema amplio.

**Texto sugerido:**

> La investigación se concentra en una decisión del diseño de representaciones: cómo distribuir variables y alcances temporales entre ramas entrenables. Para aislar ese efecto, el estudio principal empleará entradas normalizadas y bloques convolucionales causales. Las transformaciones específicas de entrada y otras familias de operadores pertenecen al programa más amplio; se incorporarán a esta tesis solo si resultan necesarias para contrastar su mecanismo central.

**Pregunta a Musashi:** ¿este recorte se eligió únicamente para el borrador o Harvey ratificó también que constituye el núcleo que desea investigar? El título ya fue ratificado; no debe pedirse otra vez.

## 4. Registro de hallazgos P1

| Código | Hallazgo | Ubicación principal | Corrección requerida |
|---|---|---|---|
| P1-01 | Criterios de decisión incompatibles | §3.3, pp. 3–4; §5.3, p. 6 | Un estimando y reglas inequívocas para efecto, equivalencia e incertidumbre. |
| P1-02 | Alcance local y memoria del predictor confundidos | §1, p. 1; §§2.1 y 4.2, pp. 2 y 4 | Definir cabezal y operador de resumen; limitar la afirmación sobre campo receptivo. |
| P1-03 | Regla perfil–grupo–alcance incompleta y q ambiguo | §§2.1 y 4.1, pp. 2 y 4 | Prototipo matemático ejecutable y normalización que conserve el significado de H2. |
| P1-04 | H2/H3 requieren información que el perfil no identifica | §3.3, p. 4; E0, p. 5 | Separar similitud marginal, dependencia cruzada y utilidad; controles adversos y negativos. |
| P1-05 | Protocolo entre familias y dentro de tarea insuficiente | §§4.3 y 5.2, pp. 5–6 | Jerarquía de particiones, adaptación y remuestreo coherente. |
| P1-06 | Ablaciones y presupuesto no aíslan todos los efectos | §§4.2 y 5.1–5.3, pp. 4–6 | Permutaciones válidas, factorización mínima y dos niveles de comparación. |
| P1-07 | Brecha y prioridad de vecinos directos aún insuficientes | §§2.2–2.3 y §5.1; matriz | Incorporar omisiones relevantes y proteger al menos un comparador de agrupación. |
| P1-08 | Población elegible, métrica y factibilidad sin contrato suficiente | §§4.3, 5.2 y 6, pp. 5–7 | Censo de procedencia multivariada, agregación MASE y presupuesto parametrizado. |

## 5. Hallazgos técnicos, correcciones y ejemplos

### 5.1 P1-01 — incertidumbre no equivale a evidencia contraria

**Problema comprobable.** H1 se declara contraria si su intervalo incluye efectos materialmente perjudiciales y, a continuación, inconclusa si la precisión no distingue beneficio, equivalencia o perjuicio. Un intervalo amplio puede satisfacer ambas condiciones. H2 usa «la permutación conserva el desempeño» y H3 «iguala» sin definir equivalencia ni precisión. En §5.3, «intervalo compatible con relevancia práctica» podría aceptar un intervalo amplio que también contiene un efecto nulo.

**Ejemplo:** si la diferencia relativa tiene un intervalo de −8 % a +10 %, incluye perjuicio, pero no demuestra que el método perjudique. Tampoco demuestra equivalencia. Es insuficiente para decidir la dirección.

**Corrección propuesta:** mantener el logaritmo de la razón de errores, definir sus pesos de agregación y usar un margen práctico antes de E2. Sea Δ = log(MASE del método / MASE del control), de modo que los valores negativos favorecen al método. Para un intervalo [L,U] y un margen ε positivo:

| Resultado | Regla ilustrativa coherente |
|---|---|
| Beneficio material | U < −ε. |
| Equivalencia práctica | [L,U] contenido en [−ε,+ε]. |
| Perjuicio material | L > +ε. |
| Inconcluso respecto de esas categorías | Cualquier otro caso. |

Esta es una convención propuesta, no un valor estadístico ya aprobado. Equivalencia refuta una promesa de mejora material sin demostrar perjuicio. La falta de significación no prueba equivalencia. Si se prefieren pruebas unilaterales, formularlas con las mismas regiones y distinguirlas de los intervalos descriptivos.

Holm ajusta una familia de pruebas identificada; no convierte por sí solo varios intervalos marginales del 95 % en intervalos simultáneos. Declarar qué afirmaciones integran la familia confirmatoria y cómo se relacionan sus pruebas e intervalos. H2 y H3 deben tener sus propios estimandos y márgenes.

**Además:** sustituir «ausencia de daño concentrado no explicado» por una regla previa de sensibilidad o heterogeneidad. No permitir que una explicación posterior elimine un resultado desfavorable del veredicto.

**Cierre:** una misma evidencia debe producir una única clasificación. Para la admisión basta la regla; los márgenes numéricos pueden fijarse mediante E1 y necesidades prácticas, sin elegirlos para asegurar un resultado favorable.

### 5.2 P1-02 — definir qué pasado alcanza realmente la predicción

**Problema.** La frase de §1 sobre la incapacidad de una convolución corta para usar una regularidad lenta es demasiado absoluta. La salida final puede combinar activaciones de todos los instantes de una ventana larga. Entonces, aunque cada activación tenga alcance R, el predictor completo puede utilizar toda la ventana L. El análisis de campos receptivos depende de la composición de las operaciones, incluidas conexiones y pasos posteriores. [Araujo, Norris y Sim, Computing Receptive Fields](https://distill.pub/2019/computing-receptive-fields/).

**Contraejemplo sencillo:** si cada rama aplica una transformación puntual que conserva información de x(t), un cabezal que recibe todas sus salidas puede comparar x(t) con x(t−100). El campo receptivo local de la rama no es un límite de 100 pasos para el predictor entero.

**Distinción adicional:** una periodicidad larga no impone una memoria mínima de esa longitud. Una sinusoide de frecuencia fija satisface x(t) = 2 cos(ω)x(t−1) − x(t−2). El periodo puede ser grande y dos valores bastar para esa recurrencia ideal. Esto no invalida estudiar escalas; impide equiparar periodo y alcance necesario.

**Instrucciones concretas:**

1. Definir el cabezal: ¿último estado, promedio temporal, proyección de toda la secuencia, recurrencia o atención?
2. Definir la mezcla sobre canales: ¿proyección lineal puntual, MLP no lineal o bloque que mezcla también tiempo?
3. Declarar el soporte de cada activación, el soporte total de la predicción y el tratamiento de los bordes con relleno.
4. Para H3, nombrar el operador de resumen y su anchura. Si se resume mediante promedio y se fusiona linealmente, ambas operaciones pueden conmutar; «antes/después» podría comparar funciones equivalentes. Si se usa una compresión diferente, puede cambiar también la capacidad.
5. Fijar la arquitectura común del cabezal antes de interpretar el efecto del alcance de las ramas.

**Texto sustitutivo:**

> El campo receptivo de una rama determina qué porción de la historia participa en cada activación local. La historia disponible para la predicción depende además de cómo la fusión y el cabezal combinan esas activaciones. El estudio distinguirá ambos alcances y mantendrá explícita la operación de resumen para atribuir las diferencias observadas al diseño de las ramas.

**Cierre:** un recorrido de dimensiones y operaciones debe permitir deducir qué pasado influye en cada salida. No se exige probar superioridad de la arquitectura.

### 5.3 P1-03 — un prototipo reproducible y una heterogeneidad comparable

**Parte A: regla.** «Campo receptivo admisible más cercano a sus escalas dominantes» deja abierta la operación principal: una variable puede tener varios periodos; un grupo, escalas distintas; y el decaimiento ACF, un valor no finito o no identificable. La propuesta necesita una traducción inicial definida, aunque E1 pueda compararla con pocas alternativas.

**Pedir en un anexo corto:**

| Elemento | Definición mínima necesaria |
|---|---|
| Perfil | Componentes, unidades, estimador, tratamiento de tendencias, máximos rezagos y datos faltantes. |
| Estabilidad | Estadístico entre subventanas, criterio de fiabilidad y resultado cuando no hay soporte. |
| Agrupación | Distancia, enlace jerárquico, regla de corte, tratamiento de empates y mínimo de variables por rama. |
| Escala representativa | Cómo se obtiene de varios componentes y de varios miembros del grupo. |
| Alcance | Distancia a la cuadrícula, redondeo, límites por historia disponible y arquitectura realizable. |
| Rama común | Alcance, ancho, costo y destino de perfiles no identificados. |

Para convoluciones encadenadas con paso uno, el campo receptivo teórico crece como 1 + suma de (tamaño del núcleo − 1) por dilatación, contando todas las capas que afectan a la salida. Los bloques residuales y el relleno deben documentarse; una anchura nominal no garantiza que todos los rezagos intermedios estén conectados. Usar una construcción consistente y contabilizar el relleno, especialmente en ramas largas.

**Parte B: q.** La normalización «con estadísticas robustas del entrenamiento» no dice si se realiza dentro de cada tarea, entre variables, entre subventanas o sobre E1. Si el objetivo es comparar heterogeneidad entre tareas, ese detalle cambia el estimando.

**Contraejemplo:** dos descriptores escalares a−s y a+s tienen mediana a y desviación absoluta mediana s. Al normalizarlos por esas estadísticas quedan −1 y +1 para cualquier s positivo. Su distancia es siempre 2. Una tarea con dispersión pequeña y otra con dispersión grande pueden recibir el mismo q.

No afirmo que toda normalización robusta produzca ese problema: el texto actual admite esa implementación y debe excluirla o justificarla.

**Opciones de reparación:** emplear unidades adimensionales justificadas; o fijar escalas de normalización sobre E1 y aplicarlas sin reajuste en E2. Distinguir, si hace falta, la métrica usada para agrupar dentro de una tarea de la usada para comparar heterogeneidad entre tareas. Definir comportamiento con componentes constantes y menos de dos variables válidas.

**Cierre:** un ejemplo numérico de varios perfiles debe producir grupos, alcances y q inequívocos. No hace falta fijar aún los valores óptimos de los umbrales.

### 5.4 P1-04 — similitud marginal, relación predictiva y desfase son cosas distintas

El perfil combina ACF, concentración espectral, periodos y estabilidad por variable. Esos rasgos pueden describir similitud temporal marginal. No identifican por sí solos dependencia cruzada, dirección predictiva ni desfases útiles para el objetivo. H3, sin embargo, condiciona su afirmación a desfases identificables.

**Contraejemplo para E0:** sean u(t) y v(t) secuencias independientes de ruido blanco. Comparar:

- Mundo A: x(t)=u(t), z(t)=u(t−d), con d positivo.
- Mundo B: x(t)=u(t), z(t)=v(t).

Las distribuciones marginales, ACF y espectros de cada canal coinciden entre ambos mundos en población. En A, el pasado de x permite predecir z(t+1) cuando contiene u(t+1−d); en B no existe esa relación. Los perfiles marginales no distinguen los mundos, aunque la utilidad de la información cruzada sea diferente.

**Conclusión limitada:** esto no demuestra que el sistema end-to-end no pueda aprender la relación. Demuestra que el diagnóstico marginal no basta para justificarla ni para identificar las condiciones de H3.

**Reparación recomendada:**

1. Describir el perfil como descriptor marginal de escala; reservar «dependencia entre variables» para un diagnóstico cruzado explícito.
2. Mantener casos con perfiles parecidos y relaciones diferentes, y casos con escalas distintas pero sin beneficio predictivo adicional. E0 no debe contener solo generadores favorables al diseño.
3. Si H3 sigue siendo confirmatoria en datos reales, definir con entrenamiento un criterio de admisibilidad de desfases o una familia de tareas donde estos se conozcan por construcción.
4. Si no hay tal criterio, concentrar H3 en E0 y tratar su extensión real como análisis secundario. Es un recorte razonable, ya contemplado en el anexo de decisiones.
5. H2 puede formular una interacción positiva como hipótesis; no existe una necesidad matemática de que más heterogeneidad produzca más ventaja. Controlar por diseño en E0 longitud, número de canales, ruido, horizonte y predictibilidad, en vez de cambiar todos esos factores junto con la heterogeneidad.

No recomiendo añadir automáticamente información mutua, causal discovery y grafos al núcleo. La corrección mínima consiste en limitar la afirmación y añadir controles que revelen lo que los perfiles no explican.

### 5.5 P1-05 — dos niveles de aprendizaje requieren dos niveles de evaluación

**Ambigüedad:** se separan familias E1/E2, pero no queda cerrado cómo se entrenan, detienen y ajustan candidatos dentro de una familia E2. Tampoco está claro qué significa «el ajuste del diseño se repetirá dentro de cada muestra de entrenamiento» después de congelar la regla.

**Protocolo recomendado:**

1. E1 desarrolla las fórmulas y parámetros de la regla, la política de entrenamiento y el criterio de elección de comparadores.
2. En una tarea E2, la regla congelada se aplica al prefijo de entrenamiento disponible. Recalcular perfiles y grupos con esa regla es adaptación permitida; cambiar su fórmula tras ver resultados E2 es desarrollo adicional.
3. Una validación temporal interna puede elegir el checkpoint o ejecutar un ajuste local predeclarado. Definir quién la consulta y qué decisiones puede modificar. Todos los métodos deben tener permisos comparables.
4. El tramo de prueba puntúa los pronósticos y no elige arquitectura, umbrales ni parada.
5. Con orígenes rodantes, indicar si los pesos se reinician, se reutilizan o se actualizan y si las observaciones ya ocurridas entran al siguiente prefijo. Esa actualización predeclarada es compatible con evaluación temporal; no equivale a abrir de nuevo el protocolo para ajustarlo.

Para fundamentar orígenes rodantes, [Hyndman y Athanasopoulos](https://otexts.com/fpp3/tscv.html) es una referencia directa. El artículo de [Bergmeir, Hyndman y Koo](https://robjhyndman.com/publications/cv-time-series/) estudia cuándo puede ser válido K-fold en modelos autorregresivos bajo condiciones sobre sus errores; no prueba por sí solo la corrección del remuestreo jerárquico aquí propuesto.

**Jerarquía:** definir por separado dominio semántico, familia de procedencia, conjunto, tarea, horizonte, origen y semilla. Una familia puede aparecer en varios archivos o repositorios; TFB, Monash y GIFT-Eval no son necesariamente poblaciones disjuntas. Re-muestrear conjuntos como si fueran independientes puede subestimar incertidumbre si comparten procedencia.

**Recomendación inferencial:** agregar dentro de familia con pesos explícitos y considerar la familia independiente como unidad superior cuando esa sea la separación utilizada. Mantener emparejados los errores de métodos, canales y horizontes al remuestrear. No re-muestrear canales relacionados como observaciones independientes. El número de familias, no el de ventanas, limita la generalización entre familias.

**Cierre:** tabla de permisos por partición y una jerarquía única en PDF, contrato e inferencia. No basta cambiar «conjunto» por «familia» sin definir qué contiene cada una.

### 5.6 P1-06 — ablaciones válidas y controles de recursos

Permutar perfiles puede cambiar simultáneamente grupos, alcances, número de ramas, anchuras y cantidad de perfiles rechazados. En ese caso, una diferencia no identifica qué parte del mecanismo ayudó. Una permutación que solo renombre conjuntamente canales y módulos puede incluso dejar el sistema equivalente.

**Matriz mínima sugerida:**

| Variante | Grupos | Alcances | Pregunta |
|---|---|---|---|
| Informada | Según perfiles | Según perfiles | Método completo. |
| Ablación G | Aleatorios con tamaños y cantidad conservados | Regla de asignación predeclarada | Aporte de agrupar informadamente. |
| Ablación R | Grupos informados | Reasignación de alcances que preserve su multiconjunto | Aporte de asociar alcance y grupo. |
| Control común | Grupos y presupuesto definidos | Alcance común elegido en E1 | Valor de especializar alcances. |

Especificar la correspondencia de anchos y alcances para que la ablación G no otorgue inadvertidamente más capacidad a otro grupo. H1 puede comparar además con TCN monolítica y referencias externas. No es necesario ejecutar todos los cruces posibles.

**Permutaciones:** alterar la correspondencia informativa, mantener intactos los valores observados y sus objetivos, fijar semillas de asignación y promediar varias asignaciones aleatorias. Si solo se usa una, el efecto puede ser suerte de esa partición. Emparejar también semillas de entrenamiento cuando sea apropiado.

**Dos niveles de comparación:**

- Para atribución del mecanismo: arquitectura y entrenamiento lo más emparejados posible; variar preferiblemente dilataciones para no cambiar simultáneamente todos los parámetros.
- Frente a modelos publicados: recetas y ajuste razonables para cada familia con presupuesto comparable. Aplicar exactamente el mismo learning rate a todos no garantiza equidad.

Conservar información disponible y medir parámetros, FLOPs, memoria, tiempo y búsqueda previa. Declarar el recurso que efectivamente se iguala en cada contraste. GPU-horas de dispositivos distintos no son por sí solas unidades equivalentes de capacidad; comparar tiempos sobre un perfil de hardware común o mantener estratos por dispositivo.

**Comparador H1:** un control débil escogido entre dos variantes internas no establece superioridad frente al estado del arte. Distinguir evidencia de mecanismo de competitividad externa. Explicar qué puede concluirse si vence al control interno pero pierde frente a DLinear o al vecino directo.

### 5.7 P1-07 — delimitar novedad mediante vecinos que atacan el mismo mecanismo

La matriz mejoró mucho. Sin embargo, la ausencia de una combinación exactamente igual de tres condiciones no basta como argumento de novedad. Usar solo entrenamiento es una condición de validez común; evaluar otras familias es un mérito experimental, pero no convierte automáticamente una regla conocida en un método nuevo.

**Comparaciones documentales prioritarias:**

| Trabajo | Coincidencia comprobada | Diferencia que debe establecerse |
|---|---|---|
| [CCM, NeurIPS 2024](https://arxiv.org/html/2404.01340) | Agrupación por similitud, prototipos aprendidos e integración con varios modelos; examina transferencia a muestras o canales no vistos. | Perfil externo fijo y asignación de alcances frente a agrupación aprendida. No reducirlo a «solo agrupa». |
| [REP-Net, 2025](https://arxiv.org/html/2507.05891v1) | Extractores paralelos con distintos tamaños, dilataciones y pasos; concatenación, módulos de procesamiento y proyección. Explora configuraciones por búsqueda. | Qué obtiene la regla propuesta del diagnóstico previo que esa búsqueda no aporta bajo el mismo presupuesto. |
| [DUET, KDD 2025](https://arxiv.org/html/2412.10859v1) | Agrupación temporal y de canales, extractores especializados y relaciones de frecuencia aprendidas. | Regla previa interpretable frente a adaptación aprendida; si el contraste puede aislar esa diferencia. |
| [MSGNet, AAAI 2024](https://arxiv.org/html/2401.00423v1) | Identificación de escalas mediante FFT y relaciones entre series específicas de escala. | Agrupar variables y asignar alcance desde perfiles frente a aprender relaciones por escala. |

DUET y MSGNet no aparecen en la bibliografía auditada. Deben incorporarse a la revisión dirigida. No se afirma aquí equivalencia exacta ni se exige ejecutar los cuatro.

**Pistas adicionales acotadas:** [ARFNet, Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S0925231226004418) coincide por alcance adaptativo y escalas, pero la página completa no fue accesible: se verificó su existencia en resultados del editor, no el detalle del mecanismo. [U-Cast](https://arxiv.org/abs/2507.15119) aborda estructuras jerárquicas de canales y puede ser pertinente si el banco es de alta dimensión. Revisarlos para decidir pertinencia; no ampliar el catálogo por obligación.

**Problema de prioridad:** CCM figura como opcional y entre los primeros recortes. Si la agrupación es una contribución principal, no debería desaparecer toda comparación con agrupación aprendida mientras se conservan varias referencias más generales. Proteger al menos un vecino de agrupación pertinente; si CCM no es compatible, justificar sustituto o estrechar la afirmación. El controlador de costos no debe recortar primero la evidencia que más podría contradecir el aporte.

**Cierre:** describir la diferencia algorítmica concreta, el precedente más próximo y el resultado que refutaría su utilidad. La revisión exhaustiva puede continuar en el primer semestre; esta comparación dirigida debe preceder al cierre de la propuesta.

### 5.8 P1-08 — banco elegible, métrica definida y techo de trabajo

**Banco:** una colección de series univariadas no se convierte automáticamente en una tarea multivariada válida al concatenarlas. Se necesita coincidencia temporal, relación de procedencia, observación conjunta y objetivo claro. El propio artículo de [GIFT-Eval](https://arxiv.org/html/2410.10393v2) distingue archivos y configuraciones univariadas/multivariadas y muestra que reutilizan bancos previos. No contar repositorios como nuevas familias independientes.

**Solicitar un censo preliminar de elegibilidad:** fuente/versionado, familia de procedencia, número de canales conjuntos, frecuencia, longitud útil, faltantes, objetivos, horizontes, licencias, solapamientos con otras fuentes y papel E1/E2. No hace falta fijar cada dataset definitivo ahora; sí mostrar una ruta plausible hacia suficientes familias pertinentes y recursos compatibles.

**MASE:** precisar cálculo por objetivo, estacionalidad m, periodo del denominador y agregación entre canales, horizontes y orígenes. En la versión estacional, el denominador usa el promedio de |y(t)−y(t−m)| en entrenamiento. Una estacionalidad elegida por el propio método no debería cambiar su escala de puntuación frente a los controles. [Definición de MASE](https://otexts.com/fpp3/accuracy.html).

En series constantes o perfectamente estacionales ese denominador puede ser cero. El logaritmo de una razón también requiere errores positivos. Definir antes de E2 el manejo de esos casos: estrato con métrica alternativa, criterio de elegibilidad o regularización declarada con sensibilidad. No eliminar casos según qué método gane ni añadir un épsilon oportunista después del test.

Para un solo objetivo y denominador compartido, la razón de MASE coincide con la razón de MAE. Esa cancelación no se traslada sin más a una razón de promedios multivariados. El orden de agregación debe estar definido.

**Costo:** hoy no hay techo aproximado de entrenamientos. Añadir una expresión de presupuesto por número de familias, tareas, variantes, semillas y orígenes, distinguiendo entrenar una salida multihorizonte una vez de entrenar un modelo por horizonte. El piloto deberá resolver si ese número cabe en el calendario. No extrapolar de una sola GPU ni prometer cuatro campañas simultáneas.

**Cierre:** un diseño mínimo parametrizado con regla de recorte y análisis de precisión. No es necesario inventar una cifra de GPU-horas antes del piloto.

## 6. Cambios P2 y mejoras de redacción

### P2-01 — reducir lenguaje de auditoría dentro de la propuesta

La propuesta está mejor escrita que su antecedente, pero conserva expresiones del taller que distraen al lector académico. El informe técnico puede ser minucioso; el PDF debe explicar una idea con continuidad.

| Expresión actual | Recomendación |
|---|---|
| «Autoridad» como columna de la tabla E0–E3 | «Alcance de la evidencia». |
| «No rescata H1–H3» | «No modifica las conclusiones del contraste principal». |
| «Compuertas previas» | Nombrar las condiciones: compatibilidad, datos y presupuesto. |
| «Pisos de complejidad» | «Referencias simples». |
| «Aprendibles» | «Entrenables», o «cuyos parámetros se aprenden». |
| «Grilla temporal» | «Malla temporal» o «instantes de muestreo comunes». |
| «Evidencia con atribución» | «Comparaciones que separen el efecto de cada componente». |
| «Banco fijo» al discutir audio | «Banco fijo de filtros», para evitar confusión con datasets. |
| «Evidencia igualmente publicable» | «Resultados reproducibles, tanto favorables como negativos». La publicación no está garantizada. |

Evitar repetir en cada sección que no se pretende optimalidad universal, que los resultados negativos se conservarán o que finanzas no rescata el experimento. Mantener una formulación clara de cada límite donde corresponde.

### P2-02 — corregir el argumento de «dos extremos»

Presentar solo procesamiento uniforme frente a búsqueda masiva simplifica excesivamente el estado del arte que el propio documento describe después. Formular la tensión del proyecto sin implicar que la literatura carece de soluciones intermedias.

> Los métodos existentes combinan distintas estrategias de especialización temporal y de interacción entre variables. En este proyecto interesa determinar qué parte de esas decisiones puede orientarse mediante diagnósticos previos de los datos, y qué parte requiere aprendizaje o experimentación. El estudio se centrará en una regla concreta de agrupación y alcance temporal, comparada con alternativas simples y aprendidas.

### P2-03 — alcance académico de las contribuciones y cronograma

Un contrato de dimensiones es un producto útil de ingeniería; su novedad doctoral no está demostrada. Presentarlo como soporte reproducible del método. El núcleo potencial es una regla fundamentada y una caracterización de sus límites que aporte conocimiento frente a los vecinos directos.

Una heurística negativa no garantiza, por sí sola, una tesis doctoral suficiente. El plan debe explicar qué conocimiento produce el resultado nulo: condiciones de identificabilidad, límites del diagnóstico, comparación sólida o regla corregida en desarrollo. No basta prometer «una guía» sin evidencia que la sostenga.

En el cronograma, escribir «manuscrito» o «envío» cuando se trate de un producto bajo control del investigador. Un artículo metodológico en el semestre 3 puede sustentarse en E0/E1, pero no debe anticipar la confirmación E2. La réplica independiente del semestre 6 debe identificar qué significa y qué recurso la hace viable.

### P2-04 — bibliografía y relación entre cita y afirmación

- RevIN aparece en el archivo .bib pero no se cita. No es una referencia indefinida. Retirarla si es residuo o usarla solo si la normalización del modelo se discute sustantivamente.
- Usar una fuente de evaluación con orígenes rodantes para esa práctica. Mantener Bergmeir et al. solo con su alcance correcto, señalado en P1-05.
- REP-Net se cita válidamente como preprint de 2025. El [sitio de uno de sus autores](https://drandrebauer.github.io/news/) anuncia aceptación en IJCNN 2026; comprobar el registro editorial antes de actualizar venue/DOI. No fabricar metadatos faltantes.
- El acceso directo al enlace OpenReview de GIFT-Eval devolvió una comprobación de navegador. No es evidencia de enlace incorrecto. El [registro original en arXiv](https://arxiv.org/abs/2410.10393) sí permitió contrastar título, autores y contenido; usar un enlace estable adicional si conviene.
- Añadir DUET y MSGNet por pertinencia, no para aumentar el número de citas. Las afirmaciones de la matriz deben indicar la sección examinada, la versión y si el detalle sigue pendiente.

### P2-05 — presentación visual

Las nueve páginas son legibles y no observé solapamientos, referencias cortadas ni ecuaciones desbordadas. La portada conserva una línea horizontal superior; puede retirarse por limpieza si ya no hay encabezado, sin que sea un defecto técnico.

La página 3 deja la primera hipótesis al final y la página 4 empieza con H2/H3; es legible, pero conviene reagrupar el bloque tras acortar el resumen y las repeticiones. El cambio no debe lograrse reduciendo arbitrariamente la fuente. La bibliografía ocupa aproximadamente las últimas dos páginas y parte de la séptima: es una proporción razonable si cada referencia sostiene una afirmación pertinente.

## 7. Textos sustitutivos propuestos

Son ejemplos de redacción para que Musashi los adapte junto con las decisiones metodológicas indicadas. No deben insertarse en bloque si todavía contradicen su implementación. Se conserva el título y no se añaden nuevas familias ni hipótesis por simple preferencia editorial.

### 7.1 Resumen propuesto

> En el pronóstico multivariado, las variables pueden presentar ritmos de cambio y dependencias temporales diferentes. Una decisión de diseño es determinar cuáles conviene procesar conjuntamente y qué alcance temporal debe tener cada rama del modelo. Esta investigación evaluará si esas decisiones pueden orientarse mediante perfiles calculados con los datos de entrenamiento.
>
> Se desarrollará una regla que traduzca descriptores temporales en grupos de variables y campos receptivos. Su primera realización utilizará ramas convolucionales causales cuyas salidas conservarán una correspondencia temporal explícita antes de fusionarse. Las ramas, la fusión y el cabezal aprenderán conjuntamente a pronosticar. Los perfiles se tratarán como indicios de diseño cuya utilidad debe comprobarse, no como una identificación de la arquitectura óptima.
>
> El estudio empleará señales sintéticas para examinar el mecanismo y sus límites, y conjuntos públicos multivariados para evaluar su utilidad. El desarrollo y la confirmación se separarán por familias de procedencia. Las comparaciones incluirán referencias simples, métodos cercanos y ablaciones que distingan agrupación, alcance y momento de resumen, con información y recursos controlados. La contribución esperada es un procedimiento reproducible y una caracterización de las condiciones en que orientar el diseño mediante perfiles temporales ayuda, resulta equivalente o perjudica. La aplicación financiera será una validación secundaria.

### 7.2 Pregunta de investigación

> ¿Cuándo una regla que utiliza perfiles temporales estimados en entrenamiento para agrupar variables y asignar campos receptivos mejora el pronóstico multivariado frente a diseños de referencia, y qué parte de esa mejora puede atribuirse a la información de los perfiles bajo recursos comparables?

Esta versión conserva el mecanismo y hace explícita la necesidad de atribución. Puede mantenerse la pregunta actual si se corrigen el método y los estimandos; la sustitución es editorial, no obligatoria.

### 7.3 Alcance y aprendizaje

> El diseño determina la organización de las ramas y su alcance temporal. El aprendizaje ajusta los parámetros de esas ramas, de la fusión y del cabezal mediante el objetivo de pronóstico. En el estudio principal se mantendrán las transformaciones de entrada y la familia de operadores suficientemente controladas para examinar la regla de composición. Una segunda familia solo se incorporará si permite contrastar la dependencia del resultado respecto del operador.

No describir como estudiada una transformación adaptativa de entrada que no exista en el diseño.

### 7.4 Perfil, agrupación y fiabilidad

> Los perfiles describirán propiedades temporales marginales de cada variable mediante estimadores y escalas definidos en desarrollo. Su estabilidad se evaluará entre subventanas del entrenamiento. La regla de composición convertirá esos perfiles en una partición de variables y en alcances realizables por la arquitectura. Los casos sin información suficiente recibirán una configuración común predefinida. Esta decisión no presupone ausencia de señal predictiva: indica únicamente que el diagnóstico utilizado no justifica una especialización.

### 7.5 Hipótesis y decisión

**H1 — ejemplo de sustitución:**

> En las familias de confirmación, el método informado producirá una mejora material del error frente al control principal seleccionado en desarrollo. La comparación utilizará la razón de errores por tarea y una agregación predeclarada por familia. Se distinguirán beneficio material, equivalencia práctica, perjuicio e incertidumbre mediante márgenes e intervalos definidos antes de E2. El contraste con modelos externos establecerá, por separado, su competitividad frente a métodos existentes.

**H2 — versión que conserva la intención y limita la interpretación:**

> En el experimento sintético, la ventaja de la asignación informada frente a asignaciones permutadas aumentará con la heterogeneidad de escalas dentro de las condiciones generativas predeclaradas. La heterogeneidad se variará separadamente del ruido, la longitud y las relaciones predictivas. Se incluirán controles donde distintos perfiles no impliquen beneficio. En datos reales, la asociación se evaluará como evidencia complementaria y no se interpretará por sí sola como efecto causal.

Si Musashi desea H2 confirmatoria también en E2, deberá aportar el soporte independiente, estimando, covariables y regla de decisión pertinentes. No convertir ausencia de asociación significativa en prueba de ausencia de mecanismo.

**H3 — versión recomendada mientras el soporte real siga pendiente:**

> En tareas con dependencia rezagada conocida, se comparará fusionar secuencias antes del resumen con resumir cada rama previamente. La comparación fijará el tipo y la dimensión del resumen y controlará los recursos relevantes. Su objetivo será determinar en qué condiciones la conservación temporal aporta utilidad. La extensión a datos reales dependerá de un criterio de identificación de desfases definido con entrenamiento.

Esta formulación permite tratar H3 como estudio de mecanismo sin prometer superioridad universal. Si se mantiene una hipótesis direccional, añadir el mismo contrato de precisión que en H1.

### 7.6 Evaluación entre familias y adaptación dentro de cada tarea

> La regla de composición y sus parámetros se fijarán con familias de desarrollo. En cada tarea confirmatoria se aplicará esa regla a los datos de entrenamiento disponibles, y se entrenará un nuevo predictor. La validación interna se utilizará solo para las decisiones admitidas por el protocolo, como la parada temprana. Los resultados de prueba no modificarán perfiles, umbrales ni recetas. Las variantes de una misma fuente permanecerán en la misma partición y se conservará su dependencia al agregar y remuestrear resultados.

### 7.7 Brecha provisional mejor delimitada

> Los trabajos de agrupación de canales, representación multiescala y composición modular muestran que especializar el procesamiento temporal es una línea establecida. La cuestión de esta propuesta es más específica: evaluar una regla previa al entrenamiento del predictor que asocia perfiles marginales con grupos y alcances de ramas. Su contribución dependerá de demostrar qué añade esa asociación frente a agrupaciones aprendidas, asignaciones no informadas y arquitecturas multiescala, así como de caracterizar los casos en que los perfiles no contienen información suficiente para orientar el diseño.

Añadir citas de CCM, REP-Net, DUET y MSGNet en las frases correspondientes. El párrafo define una investigación; no afirma que la combinación ya sea inédita.

### 7.8 Contribuciones esperadas, sin inflar el alcance

> La contribución central será un método de composición informado por perfiles temporales y una evaluación de sus condiciones de utilidad. El estudio distinguirá el valor de agrupar variables, asignar alcances y conservar secuencias, frente a diferencias de capacidad o presupuesto. El contrato de representación, el código y el protocolo constituirán los artefactos que permitan reproducir ese análisis. Si la regla no aporta una mejora, se examinará si el límite procede del diagnóstico, de su traducción estructural o de la ausencia de beneficio de la especialización en la población estudiada.

## 8. Preguntas para Musashi

Responder con la decisión propuesta, el motivo y la ubicación donde se incorporará. Si falta evidencia, decirlo y ofrecer una alternativa acotada; no rellenar el hueco con una afirmación de superioridad.

| Pregunta | Relación con la auditoría |
|---|---|
| Q01. ¿Cuál es la operación exacta del cabezal y cuál la del resumen temprano? ¿Alguna variante vuelve equivalentes las operaciones de fusionar y resumir? | P1-02. |
| Q02. ¿Qué describe cada componente del perfil y en qué unidades? ¿Cómo se resuelven varias periodicidades, tendencias y ACF sin decaimiento identificable? | P1-03. |
| Q03. ¿Sobre qué población se calculan mediana y escala para normalizar los descriptores de q? ¿Qué ocurre en el contraejemplo de dos variables? | P1-03. |
| Q04. ¿Cómo se traduce concretamente el perfil de un grupo a un alcance y a una arquitectura válida? ¿Qué alcance y ancho tiene la rama común? | P1-03. |
| Q05. ¿Qué detecta los desfases de H3? ¿Se conserva H3 confirmatoria real o se limita primero al estudio sintético? | P1-04. |
| Q06. ¿Qué unidades de procedencia separan E1 y E2 y cómo se detectarán duplicados o derivados entre archivos públicos? | P1-05 y P1-08. |
| Q07. ¿Qué puede ajustarse en una tarea E2: solo pesos y checkpoint, o también hiperparámetros? ¿Con qué validación y presupuesto? | P1-05. |
| Q08. ¿Se reentrena por horizonte o hay una salida multihorizonte? ¿Qué peso tienen canales, horizontes, orígenes, tareas y familias? | P1-05 y P1-08. |
| Q09. ¿Qué conserva y qué altera cada permutación? ¿Cuántas asignaciones aleatorias se consideran dentro del presupuesto? | P1-06. |
| Q10. ¿Cuál será el vecino de agrupación que sobrevive al recorte y qué diferencia se verificó frente a CCM, DUET y MSGNet? | P1-07. |
| Q11. ¿Cuál es el contrato de MASE cuando el denominador o alguno de los errores es cero? | P1-08. |
| Q12. ¿Qué subconjunto público realmente multivariado hace plausible el banco y cuál es la fórmula de costo del diseño mínimo? | P1-08. |
| Q13. ¿Qué parte del recorte a entrada normalizada y TCN es propuesta del equipo y cuál ha sido ratificada por Harvey? | A01; no volver a preguntar por el título ya aprobado. |

## 9. Orden de corrección y paquete de retorno

1. Resolver primero la definición de arquitectura, perfil y regla; las hipótesis deben referirse al sistema que realmente se propone.
2. Cerrar estimandos, particiones, dependencia y comparaciones mínimas.
3. Incorporar los vecinos directos y precisar la contribución sin ampliar por defecto el número de experimentos.
4. Aplicar los cambios de redacción, retirar repeticiones y sincronizar contrato, registro y decisiones pendientes.
5. Compilar y revisar visualmente la nueva versión completa.

Devolver PDF, LaTeX y bibliografía; un registro P1-01–P1-08/P2 con ubicación de cada corrección; respuestas Q01–Q13; un anexo metodológico breve con ejemplo de perfiles y contrato temporal; hash del PDF y síntesis de validación editorial.

El PDF de admisión no necesita contener todos los detalles del anexo ni el lenguaje de esta auditoría. Debe ofrecer una lectura fluida: problema concreto, método comprensible, comparación convincente y contribución honesta. El expediente debe conservar los detalles que permiten defender cada frase.

## 10. Autocrítica y límites del dictamen

- Esta auditoría no identifica un duplicado exacto de la propuesta. Encontró antecedentes más próximos que obligan a mejorar la comparación; no demuestran que el tema deba abandonarse.
- Los contraejemplos no refutan la utilidad empírica de los perfiles. Refutan interpretaciones universales o identificaciones que esos descriptores no pueden sostener por sí solos.
- La interacción positiva de H2 puede ser una hipótesis legítima aunque no esté demostrada. La corrección exigida es formularla y evaluarla sin confundirla con una consecuencia necesaria de la heterogeneidad.
- No todo detalle ausente es un defecto de una propuesta de admisión. Pueden quedar para desarrollo el número final de ramas, los valores de umbrales y parte del banco; deben quedar claros el objeto de la regla, la inferencia, el significado del contraste y la ruta de factibilidad.
- La concentración en TCN es defendible para controlar variables. No obliga a afirmar transferencia a cualquier arquitectura ni a ejecutar una segunda familia si el alcance se declara correctamente.
- La revisión bibliográfica fue dirigida a las afirmaciones y vecinos decisivos, no una certificación exhaustiva de las 27 referencias ni de toda la literatura 2025–2026. Se inspeccionaron secciones de método de CCM, REP-Net, DUET y MSGNet; para ARFNet quedó pendiente el texto completo.
- No se usó acceso a las rutas locales del computador de Harvey. La auditoría se refiere al commit y al PDF identificados al principio.

**Disposición final:** mantener título y dirección. Corregir las definiciones e inferencias señaladas y explicitar el recorte de alcance. Después corresponde una auditoría de cierre sobre las modificaciones, seguida de la lectura final de Harvey; no otra sustitución del tema por el documento que resulte más fácil terminar.
