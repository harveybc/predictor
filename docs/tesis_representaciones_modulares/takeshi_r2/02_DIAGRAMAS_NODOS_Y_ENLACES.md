# Diagramas descriptivos para la propuesta

**Takeshi · revisión 2 · 7 de septiembre de 2026**

Los esquemas se especifican exclusivamente mediante nodos y enlaces etiquetados. Cada enlace expresa datos, artefactos, una acción o una condición. No se adjuntan imágenes ni dibujos nuevos. Las descripciones son propuestas conceptuales y deben concordar con la arquitectura y el protocolo que Musashi concrete.

## 1. Convenciones comunes

- **Nombres:** utilizar detector temporal, integrador temporal, adaptación de salida, consolidador y cabezal con el mismo significado en texto, ecuaciones y esquema.
- **Dimensiones:** omitir el lote solo si el pie lo indica. L_j es la longitud de la secuencia a la salida de una rama; d_j, su número de canales. No suponer L_j=L si hay submuestreo.
- **Inferencia:** mostrar únicamente componentes y datos necesarios para producir la salida. El decoder auxiliar pertenece al esquema de entrenamiento.
- **Aprendizaje:** distinguir activaciones, objetivos, evaluación y actualización de parámetros. Una flecha de datos no debe interpretarse automáticamente como propagación de gradientes.
- **Estados:** distinguir «implementación inspeccionada», «uso histórico declarado», «propuesta experimental» y «extensión futura» en el pie correspondiente.
- **Referencias:** numeración consecutiva, referencia cercana en el texto y pie que explique abreviaturas, alcance y procedencia. Los identificadores A, B, C y D de este archivo son de trabajo; la numeración final depende de su ubicación en la propuesta.

IEEE proporciona pautas de calidad gráfica y accesibilidad, pero no prescribe una única notación para redes neuronales. Para una eventual maquetación autorizada, las guías recomiendan formatos vectoriales, fuentes incorporadas y legibilidad al tamaño final; el estilo de referencias IEEE no obliga a adoptar IEEEtran en una propuesta institucional. [Tamaño y resolución](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/), [formatos y fuentes](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/).

## 2. Esquema A — arquitectura modular del consumidor

**Uso:** principal, junto a la definición de arquitectura.  
**Estado:** propuesta conceptual; el preprocesamiento se recibe como antecedente disponible.  
**Pregunta que aclara:** ¿dónde se detecta, integra y utiliza la información temporal?

### 2.1 Nodos

| ID | Nombre del nodo | Función y contrato |
|---|---|---|
| A0 | Entradas preprocesadas | Ventanas, variables, marcas temporales y máscaras disponibles en el origen de pronóstico |
| A1 | Distribución de variables | Asignar a cada rama su grupo según el diseño fijado |
| A2j | Detector temporal de la rama j | Convertir el grupo en una secuencia de características iniciales |
| A3j | Integrador temporal de la rama j | Componer las características y dependencias permitidas por su arquitectura |
| A4j | Adaptación de salida de la rama j | Proyección y, solo si procede, adaptación temporal especificada; puede ser identidad |
| A5 | Núcleo consolidador | Integrar secuencias de ramas compatibles mediante concatenación y la fusión definida |
| A6 | Cabezal de pronóstico | Leer la representación y producir salidas por objetivo y horizonte |
| A7 | Cabezal de control | Consumidor alternativo de una extensión delimitada, no requisito del experimento principal |

Los nodos A2j, A3j y A4j se repiten para cada rama y constituyen su extractor completo. No se presupone que todos correspondan a archivos distintos o a redes independientes.

### 2.2 Enlaces

| Origen | Destino | Etiqueta del enlace | Condición |
|---|---|---|---|
| A0 | A1 | «Ventanas y metadatos temporales» | Datos disponibles en el origen |
| A1 | A2j | «Variables del grupo j» | Agrupación fijada por el procedimiento |
| A2j | A3j | «Características iniciales y su malla temporal» | Sin asumir compresión obligatoria de canales |
| A3j | A4j | «Secuencia integrada» | Dimensiones y tiempos identificados |
| A4j | A5 | «Representación de la rama j compatible con la fusión» | Misma correspondencia temporal o adaptación explícita |
| A5 | A6 | «Representación compartida para pronóstico» | Operación de lectura del cabezal definida |
| A5 | A7 | «Representación compartida para control» | Solo en la extensión autorizada y con su contrato propio |

### 2.3 Pie sugerido

> Arquitectura conceptual de una representación temporal modular. Cada rama contiene un detector inicial y operaciones de integración y adaptación. El consolidador combina sus secuencias para el consumidor. La separación define interfaces y posibilidades de entrenamiento; no implica optimización independiente obligatoria. El pronóstico es el consumidor principal. Se omite el eje de lote y se especifica por separado la correspondencia temporal de cada interfaz. Fuente: elaboración propia a partir del contexto del autor y del método propuesto.

## 3. Esquema B — diseño y aprendizaje de una tarea confirmatoria

**Uso:** principal, junto al protocolo.  
**Estado:** procedimiento propuesto.  
**Pregunta que aclara:** ¿qué se fija antes de confirmación y qué puede aprenderse dentro de una tarea nueva?

### 3.1 Nodos

| ID | Nombre del nodo | Función |
|---|---|---|
| B0 | Procedimiento fijado tras desarrollo | Regla de perfiles, agrupación, arquitectura, régimen, presupuesto y selección |
| B1 | Entrenamiento de la tarea | Única fuente interna para aprender pesos y estimar perfiles, salvo corpus previo declarado |
| B2 | Diseño aplicado a la tarea | Estimar perfiles y aplicar reglas fijadas para determinar el diseño |
| B3 | Preentrenamiento del detector | Etapa auxiliar solo si el régimen fijado la utiliza |
| B4 | Ajuste del consumidor | Aprender detector o resto según régimen; fusión y cabezal incluidos |
| B5 | Validación interna | Evaluar decisiones permitidas de parada y selección |
| B6 | Predictor seleccionado | Conjunto de configuración y pesos aceptado antes de evaluar la prueba |
| B7 | Prueba de la tarea | Datos y objetivos reservados para evaluación final |
| B8 | Evaluación confirmatoria | Errores, costos y agregación según protocolo |

### 3.2 Enlaces

| Origen | Destino | Etiqueta del enlace | Condición |
|---|---|---|---|
| B0 | B2 | «Regla estructural fijada» | Sin rediseño a partir de resultados confirmatorios |
| B0 | B3 | «Objetivo auxiliar, presupuesto y régimen» | Solo R1 o R2 |
| B0 | B4 | «Regla de ajuste y parámetros habilitados» | R0, R1 o R2 fijado |
| B0 | B5 | «Criterio de parada y selección» | No adaptarlo mirando la prueba |
| B1 | B2 | «Series de entrenamiento para perfiles» | Transformaciones ajustadas con información permitida |
| B2 | B3 | «Especificación del detector y su interfaz» | Si se preentrena |
| B1 | B3 | «Entradas y objetivos auxiliares permitidos» | No usar la prueba sin etiquetas |
| B2 | B4 | «Arquitectura del consumidor» | Incluye el detector inicializado desde cero en R0 |
| B3 | B4 | «Pesos del detector preentrenado» | Congelados en R1, ajustables en R2 |
| B1 | B4 | «Entradas y objetivos de la tarea final» | Conexión diferenciable si se ajusta el detector |
| B3 | B5 | «Métricas de preentrenamiento en validación» | Si el criterio auxiliar usa validación |
| B4 | B5 | «Métricas del consumidor en validación» | Dentro del presupuesto fijado |
| B5 | B3 | «Continuar, detener o elegir punto de preentrenamiento» | Solo decisiones permitidas del régimen |
| B5 | B4 | «Continuar, detener o elegir configuración y pesos» | Sin acceso a la prueba |
| B4 | B6 | «Artefacto seleccionado según validación» | Tras completar las decisiones permitidas |
| B6 | B8 | «Predictor fijado para evaluar» | Linaje y configuración registrados |
| B7 | B8 | «Ventanas y objetivos de prueba» | Sin flechas de retorno al diseño o selección |

### 3.3 Pie sugerido

> Aplicación del procedimiento a una tarea confirmatoria. Las reglas se fijan después del desarrollo y se aplican a los datos permitidos de la tarea. El preentrenamiento es una etapa condicional del régimen elegido. La validación interna gobierna exclusivamente decisiones previstas; la prueba estima el desempeño final. La separación entre familias de desarrollo y confirmación es adicional a las particiones internas aquí descritas. Fuente: elaboración propia.

Este esquema describe una partición fija. Si se permite actualización con origen móvil, definir cuándo los datos anteriores pasan a estar disponibles para ajuste; no dibujar un retorno de la prueba sin ese protocolo explícito.

## 4. Esquema C — detalle del autoencoder y reutilización del detector

**Uso:** detalle opcional del método o anexo.  
**Estado:** flujo conceptual apoyado en exportación inspeccionada y ajuste histórico declarado.  
**Pregunta que aclara:** ¿qué se conserva del autoencoder y qué continúa aprendiendo?

### 4.1 Nodos

| ID | Nombre del nodo | Función |
|---|---|---|
| C0 | Ventana de entrenamiento | Entrada y objetivo original permitido |
| C1 | Preparación de la tarea auxiliar | Identidad, enmascaramiento o corrupción explícitamente definida |
| C2 | Detector–encoder | Obtener la representación latente temporal |
| C3 | Decoder auxiliar | Reconstruir el objetivo definido desde la representación |
| C4 | Pérdida auxiliar | Comparar reconstrucción y objetivo en las posiciones previstas |
| C5 | Optimizador del autoencoder | Actualizar pesos mediante la pérdida auxiliar |
| C6 | Pareja preentrenada preservada | Guardar encoder y decoder con sus metadatos y compatibilidad original |
| C7 | Copia del detector para el consumidor | Inicialización de R1 o R2, identificada respecto de C6 |
| C8 | Resto del extractor, consolidador y cabezal | Obtener la salida final |
| C9 | Pérdida y optimizador del consumidor | Ajustar los parámetros permitidos por el régimen |

### 4.2 Enlaces

| Origen | Destino | Etiqueta del enlace | Condición |
|---|---|---|---|
| C0 | C1 | «Entrada para la tarea auxiliar» | Solo partición permitida |
| C1 | C2 | «Entrada preparada y máscara si procede» | Transformación definida |
| C2 | C3 | «Secuencia latente» | Interfaz identificada |
| C3 | C4 | «Reconstrucción» | Objetivo y posiciones de pérdida definidos |
| C0 | C4 | «Objetivo original de reconstrucción» | No es un atajo hacia el decoder |
| C4 | C5 | «Pérdida para diferenciación» | Grafo diferenciable |
| C5 | C2 | «Actualización de pesos del encoder» | Durante preentrenamiento |
| C5 | C3 | «Actualización de pesos del decoder» | Durante preentrenamiento |
| C2 | C6 | «Encoder seleccionado» | Parada/selección según protocolo |
| C3 | C6 | «Decoder de la misma pareja» | Misma versión del preentrenamiento |
| C6 | C7 | «Copia de arquitectura y pesos del encoder» | Preservar el origen de la copia |
| C0 | C7 | «Entrada del consumidor durante ajuste» | Objetivos finales suministrados aparte |
| C7 | C8 | «Características temporales» | Conexión diferenciable en R2 |
| C8 | C9 | «Pronóstico y objetivo final permitido» | Pérdida del consumidor |
| C9 | C8 | «Actualización del resto del modelo» | Régimen definido |
| C9 | C7 | «Actualización del detector» | Solo R2; ausente en R1 |

R0 entra al consumidor con un detector inicializado desde cero y omite la etapa auxiliar. C9 no actualiza automáticamente el decoder preservado. No conectar una copia ajustada del encoder al decoder original suponiendo compatibilidad semántica.

### 4.3 Pie sugerido

> Preentrenamiento auxiliar y reutilización del detector. El encoder aprende junto con un decoder mediante reconstrucción y se exporta para el consumidor. Una copia puede mantenerse congelada o ajustarse con la pérdida final. La pareja original se conserva como artefacto independiente. Las conexiones de actualización representan aprendizaje, no flujo de inferencia. Fuente: elaboración propia, con antecedente de exportación en feature-extractor.

## 5. Esquema D — mejora por componentes en DOIN

**Uso:** opcional en contexto, no hipótesis adicional.  
**Estado:** visión del autor; la implementación y ejecución de cada reemplazo requieren su evidencia particular.

### 5.1 Nodos

| ID | Nombre del nodo | Función |
|---|---|---|
| D0 | Contrato del subproblema | Definir objetivo, interfaz, presupuesto y versiones dependientes |
| D1 | Optimización del componente | Generar candidatos de detector, integración, fusión o cabezal |
| D2 | Ensamblaje con dependientes | Integrar una versión candidata con componentes identificados |
| D3 | Evaluación con el consumidor | Comprobar compatibilidad y utilidad del conjunto |
| D4 | Conjunto de versiones aceptado | Registrar artefactos disponibles para inferencia |
| D5 | Reajuste o rechazo | Adaptar dependientes dentro del presupuesto o descartar el candidato |

### 5.2 Enlaces

| Origen | Destino | Etiqueta del enlace | Condición |
|---|---|---|---|
| D0 | D1 | «Objetivo e interfaz de optimización» | Subproblema definido |
| D1 | D2 | «Candidato y configuración» | Versiones dependientes identificadas |
| D2 | D3 | «Sistema ensamblado» | Contrato de entrada satisfecho |
| D3 | D4 | «Aceptar conjunto compatible» | Cumple el criterio del consumidor |
| D3 | D5 | «Reajustar o rechazar» | No cumple el criterio o necesita adaptación |
| D5 | D2 | «Nueva combinación o dependientes reajustados» | Solo si el presupuesto permite nueva evaluación |

### 5.3 Pie sugerido

> Mejora por componentes con evaluación de integración. La optimización propone piezas, pero la aceptación considera sus dependencias y el objetivo del consumidor. Una mejor puntuación aislada no garantiza un mejor sistema. El esquema explica una posibilidad de uso de DOIN y no afirma que cada componente corresponda a un dominio distinto ni que todas las sustituciones estén implementadas. Fuente: visión del autor y elaboración conceptual.

## 6. Comprobaciones de coherencia para Musashi

1. Cada nodo debe corresponder a una responsabilidad definida y cada enlace a una operación real o propuesta explícita.
2. Especificar dónde cambia la longitud temporal y cómo se alinean las ramas.
3. No confundir extracción fija desde archivos con ajuste conjunto diferenciable.
4. No introducir síntesis adversarial como salida adicional del decoder en el esquema de tesis.
5. Mantener el significado de R0, R1 y R2 idéntico al documento 01.
6. Si se elimina o fusiona un componente al concretar el método, actualizar texto y esquema juntos.
