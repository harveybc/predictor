# Actualización de la propuesta: detector temporal, preentrenamiento y acoplamiento

**Takeshi → Harvey y Musashi · revisión 2 · 7 de septiembre de 2026**

## 1. Dictamen actualizado

La dirección elegida sigue siendo pertinente. Debemos actualizar el método y sus antecedentes con una capacidad que Harvey ya tenía: entrenar un autoencoder, exportar encoder y decoder por separado y reutilizar el encoder como parte de un consumidor posterior. El autor explica que lo cargaban con Keras y habilitaban su entrenamiento para continuar desde los pesos importados.

Mi respuesta anterior presentó demasiado ese flujo como una posibilidad futura. La corrección consiste en reconocer el antecedente y concentrar la investigación en las decisiones todavía no justificadas: qué representación debe producir cada componente, cómo se adapta a las propiedades de los datos, dónde conviene separar responsabilidades y cómo se aprende y evalúa el acoplamiento.

**No basta con cambiar el nombre de extractor a detector.** El detector propuesto sigue siendo un extractor de características, situado ahora como submódulo inicial de una rama mayor. La frontera será útil si permite especificar su interfaz, preentrenarlo, inspeccionarlo, sustituirlo o ajustar sus pesos de manera controlada.

No corresponde reabrir el título aprobado ni añadir por inercia una nueva hipótesis. Sí corresponde revisar antecedentes, arquitectura, régimen de aprendizaje, controles, costos y diagramas. La generación sintética y la verificación de métricas en una red hostil quedan como línea separada.

## 2. Evidencia y estado de las afirmaciones

Las fuentes del código corresponden a `feature-extractor@df86252a174a1f5b9f77f4cb1bcb78fcdfe429e8` y `predictor@20ec57145b4ed3eb647eabe2b7591f35a3dbbf47`. Son distintos del commit de la propuesta. Se hizo inspección estática; no se reprodujo una campaña.

| Afirmación | Estado | Evidencia y límite |
|---|---|---|
| Hay operaciones para guardar encoder y decoder por separado | Verificado en código | `AutoencoderManager` dispone de rutas de guardado y métodos separados. No demuestra que cada combinación de plugins funcione en el estado actual. [Gestor](https://github.com/harveybc/feature-extractor/blob/df86252a174a1f5b9f77f4cb1bcb78fcdfe429e8/app/autoencoder_manager.py). |
| El encoder CNN dispone de carga y guardado Keras | Verificado en código | El plugin contiene ambas operaciones. [Encoder CNN](https://github.com/harveybc/feature-extractor/blob/df86252a174a1f5b9f77f4cb1bcb78fcdfe429e8/app/plugins/encoder_plugin_cnn.py). |
| Ese encoder CNN produce una secuencia, no un vector global | Verificado en el grafo construido | Hay dos Conv1D con paso 2; la salida conectada es la segunda convolución. La descripción inicial de vector latente no coincide con esa salida. Para entrada temporal L, la longitud resultante es aproximadamente L/4, con el redondeo correspondiente al relleno. |
| Predictor contempla cargar encoders y habilitar su entrenamiento | Verificado en configuraciones | Ejemplos contienen `feature_extractor_file` y `train_fe: true`. Esto evidencia la configuración, no prueba por sí solo el flujo efectivo de gradientes. [Ejemplo](https://github.com/harveybc/predictor/blob/20ec57145b4ed3eb647eabe2b7591f35a3dbbf47/examples/config/phase_3_1/phase_3_1_cnn_25200_1h_config.json). |
| El encoder se incorporaba al consumidor y continuaba aprendiendo con `trainable=True` | Declarado por Harvey; falta traza exacta de implementación | Musashi debe aportar archivo, función, configuración y commit del ensamblaje correspondiente. No se cuestiona el relato; se solicita trazabilidad para documentarlo. |
| El decoder se exploró para generar datos sintéticos, sin resultados suficientes | Declarado por Harvey | No se localizaron aquí resultados que permitan atribuir la dificultad a una causa concreta. No usarlo como resultado experimental negativo de la tesis. |
| Preentrenar el detector mejorará la tarea final | Hipótesis de trabajo, no demostrada | Debe compararse con alternativas bajo el mismo protocolo de información y una contabilidad explícita de recursos. |

**Observación de implementación:** en el encoder CNN, la salida efectiva depende de `initial_layer_size` y `layer_size_divisor`; la lista preliminar calculada con `interface_size` no es la que construye esas dos capas. Por tanto, el contrato del detector debe documentar la forma real del tensor y los parámetros que efectivamente la controlan. No inferirla a partir del nombre de un argumento. Esta observación no exige reparar ahora todos los plugins históricos.

El gestor inspeccionado también contiene una ruta especializada para CVAE. La presencia de un plugin CNN y de métodos de guardado no basta para afirmar que ese gestor admite hoy todas las combinaciones históricas. Musashi debe seleccionar un ejemplo ejecutable coherente como antecedente del nuevo protocolo.

## 3. Arquitectura recomendada y límites de la interpretación

### 3.1 Responsabilidades

| Componente | Responsabilidad | Lo que debe especificarse |
|---|---|---|
| Entrada preprocesada | Entregar variables disponibles en el origen de decisión | Procedencia, transformaciones ajustadas solo con datos permitidos, variables, marcas temporales y máscaras |
| Detector temporal D_j | Obtener una primera secuencia de características | Variables de entrada, operadores, alcance temporal, paso, relleno, canales y objetivo de preentrenamiento si se usa |
| Integrador temporal I_j | Componer las características relevantes para la tarea | Dependencias accesibles, operaciones y relación con el consumidor |
| Proyección o adaptación A_j | Cumplir la interfaz de salida de la rama | Canales, malla temporal, máscara; identidad si no se necesita transformación |
| Consolidador G | Integrar salidas de las ramas | Concatenación, mezcla y acceso temporal efectivos |
| Cabezal H | Producir pronósticos; control solo en extensión delimitada | Lectura de la secuencia, horizontes, objetivos y forma de salida |

El extractor completo de una rama comprende D_j, I_j y A_j. Esta separación describe responsabilidades; no impone tres redes grandes ni tres procesos de entrenamiento. Una proyección puede ser una sola capa y un integrador puede ser identidad en un control.

La detección inicial no garantiza patrones semánticos interpretables. El integrador sigue aprendiendo características: no debe reducirse su función a disminuir dimensiones. Tampoco el detector tiene que comprimir desde su primera capa; puede expandir canales y reducirlos después.

No fijar «dos o tres capas» como definición del detector. El ejemplo histórico de dos Conv1D es un antecedente concreto. Su adecuación a escalas, dependencias y recursos deberá justificarse para el estudio elegido.

### 3.2 Tiempo y reducción dimensional

Distinguir reducción de canales, reducción de resolución temporal y resumen de la secuencia. Una capa Dense aplicada al último eje puede conservar la secuencia; una Conv1D con paso mayor que uno cambia su longitud. La documentación oficial precisa ambos comportamientos: [Dense](https://keras.io/api/layers/core_layers/dense/), [Conv1D](https://keras.io/api/layers/convolution_layers/convolution1d/).

La salida de la rama puede representarse como una secuencia con L_j posiciones y d_j canales. La fusión por canales requiere una correspondencia temporal definida entre ramas. Si la propuesta conserva una longitud común L, debe elegir operadores que cumplan esa decisión; si permite resoluciones diferentes, debe definir su adaptación, sus costos y su efecto en los controles. No rellenar o interpolar por conveniencia sin explicar qué instante representa cada posición.

El campo receptivo de una activación del detector no equivale a la historia accesible al predictor completo. Mantener el hallazgo P1-02 de la auditoría y especificar el cabezal antes de interpretar H3.

### 3.3 Alcance de la tesis

La intención del autor incluye adaptar datos y modelos de forma fundamentada. El recorte a perfiles marginales, agrupaciones y TCN sigue siendo una instancia candidata de ese programa, no una definición irrevocable de toda su intención. La nueva separación del detector no ratifica automáticamente ese recorte ni lo invalida.

Recomendación: mantener una familia principal controlable y estudiar dentro de ella la frontera del detector y su entrenamiento. Si el ejemplo histórico usa resolución reducida, no trasplantarlo sin cambios a un experimento que promete longitud constante. Toda ampliación a otro operador o transformación de entrada debe responder a una pregunta concreta y tener presupuesto propio.

## 4. Preentrenamiento y ajuste: protocolo que debe quedar explícito

### 4.1 Autoencoder como mecanismo auxiliar

Durante el preentrenamiento, un decoder auxiliar Q_j recibe la salida del detector D_j y reconstruye la entrada o un objetivo definido. Si se enmascaran segmentos, se debe especificar qué se oculta, cómo se genera la máscara y dónde se calcula la pérdida. No llamar denoising a una reconstrucción ordinaria ni afirmar que el residuo real equivale a ruido conocido.

La salida exportada debe ser la representación que consumirá I_j. No dar acceso al decoder a atajos que vuelvan trivial reconstruir sin utilizar esa representación, salvo que su propósito y efecto estén controlados.

El preentrenamiento de autoencoders por capas y el uso de reconstrucción enmascarada para series tienen antecedentes directos. Añadirlos a la matriz bibliográfica para explicar el régimen, sin atribuir novedad a reutilizar pesos: [Bengio et al., preentrenamiento por capas](https://proceedings.neurips.cc/paper_files/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf), [Ti-MAE](https://arxiv.org/abs/2301.08871). Estos antecedentes no certifican superioridad en el sistema de Harvey.

### 4.2 Comparación acotada

| Régimen | Inicialización del detector | Ajuste con el consumidor | Qué ayuda a distinguir |
|---|---|---|---|
| R0 | Desde cero | Conjunto | Referencia de la misma arquitectura sin preentrenamiento |
| R1 | Encoder preentrenado | Detector congelado; resto entrenable | Utilidad de una representación fija |
| R2 | Mismo preentrenamiento | Detector y resto entrenables | Utilidad de inicialización y adaptación posterior |

Harvey describe R2 como flujo usado históricamente. La propuesta debe identificarlo como antecedente, sin convertir R0 y R1 en capacidades nuevas de DOIN ni presentar R2 como ganador ya demostrado.

Para comparar R1 y R2, reutilizar cuando corresponda el mismo punto de partida preentrenado y emparejar semillas. Mantener arquitectura e información de entrada comparables; contabilizar decoder, preentrenamiento, búsquedas y ajuste. Diferenciar igual presupuesto total de igual número de épocas finales: no son equivalentes.

No ejecutar el producto cartesiano de regímenes, familias, transformaciones y todas las ablaciones. Proponer un estudio mecanístico acotado en desarrollo y congelar el régimen principal antes de confirmación. Si se desea una conclusión confirmatoria específica sobre los regímenes, reservar comparaciones y soporte para ella; los resultados de desarrollo por sí solos no sostienen esa afirmación.

### 4.3 Gradientes y Keras

Para ajuste conjunto, el encoder debe estar conectado a la pérdida final mediante operaciones diferenciables. Exportar previamente sus salidas a un archivo y entrenar únicamente sobre ellas corresponde a extracción fija, aunque el modelo original tenga `trainable=True`.

El atributo `trainable` se debe establecer antes de compilar el modelo compuesto, o recompilar tras modificarlo. También debe distinguirse de `training`, que controla comportamientos de ejecución como Dropout y BatchNormalization. Revisar la tasa de aprendizaje del ajuste para evitar cambios iniciales excesivos; no imponer una receta universal ni convertir esa revisión en un experimento adicional obligatorio. [Guía oficial de transferencia y ajuste de Keras](https://keras.io/guides/transfer_learning/).

Musashi debe documentar la ruta efectiva y, si ya dispone de ella, evidencia de actualización de los pesos del encoder en una ejecución breve. No se requiere iniciar campañas costosas para responder este paquete.

### 4.4 Separación de datos y artefactos

- El preentrenamiento también aprende de los datos. Usar exclusivamente las particiones permitidas; no incluir prueba por carecer de etiquetas.
- Si se usa un corpus previo compartido, declarar su procedencia y solapamiento con las familias confirmatorias. No afirmar generalización a familias no vistas si el preentrenamiento ya las utilizó bajo otro nombre.
- Perfiles, agrupaciones y preentrenamiento dentro de una tarea deben respetar el procedimiento congelado. La adaptación autorizada al entrenamiento de esa tarea no permite rediseñar el método mirando su prueba.
- Un early stopping de L1 o L2 no elimina por sí mismo la adaptación acumulada a la validación. Conservar presupuesto, regla de parada y separación confirmatoria de la auditoría.
- Registrar configuración, particiones, normalización, grupos, máscaras, semillas, objetivo auxiliar y punto de partida de cada encoder ajustado.

## 5. Consecuencias para hipótesis, controles y costos

**H1:** comparar diseño informado y controles con el mismo régimen principal de aprendizaje o separar explícitamente ambos factores. Una mejora con más preentrenamiento no identifica por sí sola un efecto de la agrupación o del campo receptivo.

**H2:** la normalización de perfiles, heterogeneidad y relaciones entre variables conserva todos los problemas señalados en P1-03 y P1-04. Un detector preentrenado no los resuelve automáticamente. Si se usan sus latentes para calcular perfiles, cambia el mecanismo original y deberá declararse como alternativa, con datos y reglas propios; no introducirlo de manera implícita.

**H3:** representar qué secuencia llega a fusión y cabezal. Una convolución con paso 2 o un resumen en el encoder modifica el contraste. Si se compara resumen temprano y conservación temporal, controlar las rutas alternativas por las que el cabezal accede al pasado.

**Control arquitectónico:** el monolítico y el modular deben tener condiciones de entrenamiento competitivas y comparables. No beneficiar solo al método propuesto con preentrenamiento, salvo que la afirmación y la comparación estén definidas precisamente sobre esa intervención completa.

**Costo:** incluir aprendizaje auxiliar, decoder durante preentrenamiento, ajuste al consumidor y búsqueda L2. Si un encoder se reutiliza en varias tareas, separar costo previo y marginal, y reportar amortización solo con una política de reutilización explícita. No omitir el costo de reajustar dependientes tras sustituir una pieza.

**L1, L2 y L3:** aprender pesos mediante reconstrucción o pronóstico es L1 en la nomenclatura del proyecto; buscar configuraciones es L2. La posible metaoptimización con registros OLAP es una línea distinta y no se incorpora como requisito. El preentrenamiento no debe confundirse con un nuevo nivel de optimización.

## 6. Decoder, compatibilidad y generación sintética

Guardar la pareja original D_j–Q_j del preentrenamiento y registrar cada versión del detector ajustada al consumidor. Después del ajuste, el decoder original puede dejar de reconstruir adecuadamente a partir del encoder modificado, aunque no cambie la forma del tensor. La compatibilidad semántica no se deduce de la compatibilidad dimensional.

Esto no implica que modificar una copia del encoder altere por sí mismo el decoder conservado. Para generación desde latentes, el problema adicional es qué distribución se muestrea y qué relación tiene con el entrenamiento del decoder. No basta con disponer del archivo del decoder para obtener un generador validado.

La generación sintética para validar métricas en entornos hostiles queda fuera de los objetivos y experimentos principales de esta propuesta, por decisión del autor. No afirmar que datos sintéticos impiden automáticamente el sobreajuste: sería una propiedad del protocolo completo de generación, acceso y evaluación que requiere investigación separada. Tampoco convertir las dificultades históricas relatadas en una conclusión experimental sin sus artefactos.

## 7. Matriz de cambios documentales exigibles

| Código | Sección a revisar | Cambio y justificación | Condición de cierre |
|---|---|---|---|
| U01 | Antecedentes y recursos disponibles | Reconocer exportación y reutilización de encoders existentes | Fuentes fijadas y estado de evidencia explícito |
| U02 | Arquitectura | Definir detector, integración y adaptación como responsabilidades del extractor | Entradas/salidas y frontera identificables |
| U03 | Método | Explicar preentrenamiento y ajuste posterior | Pérdidas, datos, pesos ajustables y decoder auxiliar definidos |
| U04 | Alcance | Revisar si TCN y perfiles son el recorte adecuado de la intención | Justificación del recorte; sin atribuir ratificación inexistente al autor |
| U05 | Hipótesis y ablaciones | Separar efecto del diseño y régimen de entrenamiento | Régimen común o contraste explícito; sin H4 automática |
| U06 | Protocolo | Incluir preentrenamiento en la separación de información | Linaje del corpus y exclusión de prueba |
| U07 | Costos | Incluir aprendizaje auxiliar y reajuste de dependientes | Costo total y política de reutilización |
| U08 | Implementación de referencia | Elegir plugin/configuración realmente conectados | Commit, formas y ruta de gradientes; no solo imports o docstrings |
| U09 | Diagramas | Describir el detector y distinguir inferencia de aprendizaje | Nodos y enlaces del documento 02 concordantes con el método |
| U10 | Antecedentes bibliográficos | Incluir preentrenamiento y representaciones temporales autosupervisadas | Fuentes primarias; novedad delimitada |
| U11 | Artefactos | Conservar pareja preentrenada y versiones ajustadas | Identidad y compatibilidad trazables |
| U12 | Exclusiones | Mantener síntesis y validación hostil fuera del núcleo | Sin objetivos ni promesas adicionales sobre ellas |

## 8. Redacción sustitutiva propuesta

### 8.1 Antecedente técnico

> El proyecto dispone de herramientas para entrenar autoencoders y exportar por separado sus componentes codificador y decodificador. Estos artefactos se han utilizado como punto de partida de extractores de características en modelos posteriores. La investigación se apoya en esa experiencia para estudiar decisiones que aún requieren evaluación sistemática: la organización interna de las ramas, las propiedades de sus representaciones temporales y su adaptación al consumidor.

La afirmación de uso histórico se apoya en el relato del autor; acompañarla con el ejemplo versionado que debe identificar Musashi. Si se afirma ajuste efectivo de pesos en un resultado concreto, aportar su traza correspondiente.

### 8.2 Arquitectura

> Cada rama transforma un grupo de variables mediante un extractor temporal modular. Se distingue un detector inicial, que aprende una secuencia de características, de las operaciones posteriores que integran información temporal y adaptan la salida al consolidador. Esta separación permite estudiar objetivos de preentrenamiento y formas de ajuste sin imponer que los componentes deban entrenarse o sustituirse de manera independiente. Las dimensiones y la correspondencia temporal de sus interfaces se especificarán para cada configuración.

### 8.3 Diseño y aprendizaje

> El diseño determina la organización de las ramas y las propiedades de sus operaciones a partir de información obtenida en los datos de entrenamiento. El aprendizaje ajusta sus parámetros para producir representaciones útiles al consumidor. Se considerará el preentrenamiento del detector mediante reconstrucción como una alternativa de inicialización, seguida de congelación o ajuste con la tarea final. La utilidad de esta alternativa se contrastará con entrenamiento conjunto desde cero, separando el efecto del régimen de aprendizaje del efecto de las decisiones estructurales.

Si los tres regímenes quedan únicamente en desarrollo, sustituir «se contrastará» por «se examinará durante el desarrollo del procedimiento» y explicitar después cuál queda fijado para confirmación. No permitir dos lecturas del alcance inferencial.

### 8.4 Pregunta de investigación: ajuste posible

> ¿En qué condiciones un procedimiento de diseño de representaciones temporales modulares, informado por las propiedades de los datos y con un régimen de aprendizaje explícito, mejora el pronóstico frente a alternativas de complejidad y recursos comparables?

Es una formulación marco. La pregunta principal final debe nombrar las decisiones estructurales efectivamente estudiadas —por ejemplo, agrupación y alcance temporal— para que no quede genérica. No ampliarla a todos los operadores, transformaciones y regímenes por haber añadido un detector.

### 8.5 Alcance del decoder

> El decodificador se utiliza como componente auxiliar del preentrenamiento cuando ese régimen se adopta. Se preservarán los artefactos originales y las versiones del codificador adaptadas al consumidor. La generación sintética para evaluación en redes descentralizadas no constituye un objetivo de esta investigación.

### 8.6 Novedad y contribución

> La contribución propuesta no se atribuye al uso de autoencoders, a la existencia de varias ramas ni a la reutilización de pesos. Se buscará establecer y evaluar un criterio explícito de diseño y acoplamiento de representaciones temporales, junto con las condiciones en que sus decisiones producen beneficio, equivalencia o perjuicio frente a controles pertinentes.

Después de esa frase se debe nombrar el criterio concreto y sus vecinos bibliográficos. No dejar «criterio explícito» como sustituto de una definición operacional.

## 9. Preguntas para Musashi y respuestas requeridas

1. ¿Dónde se carga y conecta el encoder dentro del predictor y dónde se aplica `train_fe`? Aportar archivo, función, commit y configuración representativa.
2. ¿Se usa el encoder como submodelo diferenciable o se precalculan características? Si hay ambas rutas, identificar cuál corresponde a cada antecedente.
3. ¿Qué plugin y configuración son actualmente el ejemplo canónico? Resolver las diferencias entre docstrings, parámetros anunciados y grafo conectado.
4. ¿Qué salida delimita el detector? Dar dimensiones, resolución, máscara, alcance y correspondencia con instantes de entrada.
5. ¿El decoder reconstruye cada grupo por separado o hay un decoder compartido? ¿Qué representación recibe y qué atajos existen?
6. ¿Se preentrena por tarea, por grupo, por familia o en un corpus común? ¿Qué datos y costos corresponden a cada opción?
7. ¿Qué regímenes se estudian en desarrollo y cuál se confirma? ¿Se pretende una conclusión propia sobre preentrenamiento o solo fijar un régimen de soporte?
8. ¿Qué pesos permanecen congelados o se ajustan, con qué política de tasa de aprendizaje y en qué momento se compila el modelo compuesto?
9. ¿Qué versión del encoder conserva compatibilidad con el decoder exportado? ¿Cómo se distinguen copias ajustadas al consumidor?
10. ¿Cómo se mantienen comparables capacidad, información y costo al comparar diseños con y sin preentrenamiento?
11. ¿Sigue siendo el mecanismo perfiles–grupos–campos receptivos la instancia recomendada del problema del autor? Justificarla a la luz de estas aclaraciones, sin ratificarla en su nombre.
12. ¿Qué correcciones de los ocho P1 originales quedan aplicadas en el nuevo texto y cuáles están pendientes? La separación del detector no las resuelve por sí sola.

## 10. Límite de esta actualización

Este documento integra conversación e inspección estática. No demuestra eficacia del preentrenamiento, no reproduce resultados históricos ni certifica una versión corregida de la propuesta. Su resultado es un conjunto concreto y revisable de instrucciones que evita presentar capacidad existente como trabajo futuro y ayuda a formular el experimento adecuado.
