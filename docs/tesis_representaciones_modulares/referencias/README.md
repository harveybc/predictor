# Guia de lectura de las referencias de la propuesta

Esta carpeta acompana a
`docs/propuesta_doctoral_representaciones_temporales_modulares.pdf`. El numero
inicial de cada archivo coincide con el numero de su entrada en la bibliografia
de la propuesta. La bibliografia esta ordenada manualmente; **no** sigue de
manera estricta el orden de primera cita. Por ejemplo, `[27]` se cita por primera
vez en la pagina 12, despues de `[35]`.

El inventario actual contiene 35 referencias y 35 artefactos numerados, sin
huecos ni archivos sobrantes. Los PDF y las copias de documentacion se mantienen
locales para lectura; no se publican en GitHub. Este indice y el manifiesto de
identidad si se versionan. Para verificar el expediente:

```bash
python docs/tesis_representaciones_modulares/referencias/verificar_expediente.py
```

Las paginas indicadas son las que muestra el visor del archivo local, no la
paginacion impresa de la revista. En HTML se indica el encabezado o ancla. La
seccion **Limite** dice expresamente que no demuestra la referencia.

## Ruta minima de lectura

Para entender la pregunta doctoral antes de entrar en los antecedentes, leer en
este orden: `[1]`, `[2]`, `[3]`, `[6]`, `[14]`, `[16]`, `[19]`, `[20]`, `[24]`,
`[34]`. Despues leer `[7]-[13]`, `[15]`, `[17]-[18]` y `[21]-[23]` para poder
distinguir la contribucion de sus vecinos. `[25]-[28]` describen infraestructura
propia; `[29]-[35]` sustentan metodologia, bancos y evaluacion.

---

### [1] `01_Bengio_2013_Representation_Learning.pdf`

- **En la propuesta:** pagina 2, seccion 1, definicion de aprendizaje de
  representaciones y de su utilidad para una tarea posterior.
- **Lectura minima:** PDF p. 1, resumen y primer parrafo de la seccion 1.
- **Relacion:** establece que el desempeno depende de la representacion y que
  aprenderla busca hacer accesible la informacion util para la tarea.
- **Limite:** no sustenta que una representacion concreta sea mejor ni que el
  metodo propuesto vaya a funcionar en series temporales.
- **Fuente:** arXiv:1206.5538, version abierta del articulo de IEEE TPAMI 35(8).

### [2] `02_Cawley_2010_Overfitting_Model_Selection.pdf`

- **En la propuesta:** pagina 2, seccion 1, riesgo de sobreajustar el propio
  proceso de seleccion.
- **Lectura minima:** PDF pp. 1-2, resumen y seccion 1; para la demostracion
  empirica, pp. 14-20.
- **Relacion:** distingue el ajuste del modelo del ajuste de la regla que elige
  modelos y documenta el sesgo que puede introducir esta segunda capa.
- **Limite:** no prescribe el protocolo experimental de esta tesis; justifica
  separar desarrollo, seleccion y confirmacion.
- **Fuente:** JMLR 11(70), 2010.

### [3] `03_Bai_2018_TCN.pdf`

- **En la propuesta:** paginas 3 y 8, alcance temporal y campo receptivo de una
  pila convolucional.
- **Lectura minima:** PDF pp. 3-4, seccion 2.1 y Fig. 1, en particular
  convoluciones causales dilatadas y tamano del campo receptivo.
- **Relacion:** permite explicar por que el alcance de una rama no queda fijado
  solo por su primera capa.
- **Limite:** su comparacion TCN-RNN no prueba la superioridad de las ramas de la
  propuesta ni reemplaza los controles de capacidad.
- **Fuente:** arXiv:1803.01271.

### [4] `04_Keras_Conv1D.html`

- **En la propuesta:** pagina 3, Fig. 2, capa convolucional ilustrativa.
- **Lectura minima:** ancla `#conv1d-class`, firma y argumentos `filters`,
  `kernel_size`, `strides`, `padding` y `dilation_rate` (lineas locales
  446-504).
- **Relacion:** documenta la semantica de la capa usada en el ejemplo.
- **Limite:** es documentacion de API, no evidencia de novedad ni de desempeno.
- **Fuente:** guia oficial de Keras 3, consultada el 9 de septiembre de 2026.

### [5] `05_Keras_LSTM.html`

- **En la propuesta:** pagina 3, Fig. 2, rama recurrente ilustrativa.
- **Lectura minima:** ancla `#lstm-class`, firma y argumento
  `return_sequences` (lineas locales 446-555).
- **Relacion:** documenta la salida secuencial de la capa mostrada.
- **Limite:** no respalda por si sola ninguna comparacion LSTM-convolucion.
- **Fuente:** guia oficial de Keras 3, consultada el 9 de septiembre de 2026.

### [6] `06_Hyndman_2021_FPP3.html`

- **En la propuesta:** pagina 4, descriptores de series; pagina 6, diagnostico
  de tendencia y estacionalidad; pagina 10, evaluacion por origen y MASE
  estacional.
- **Lectura minima exacta:** seccion 2.8, *Autocorrelation*,
  <https://otexts.com/fpp3/acf.html>; seccion 4.3, *STL Features*,
  <https://otexts.com/fpp3/stlfeatures.html>; seccion 4.4, *Other features*,
  <https://otexts.com/fpp3/other-features.html>; seccion 5.8, *Evaluating the
  point forecast accuracy*, <https://otexts.com/fpp3/accuracy.html>; y seccion
  5.10, *Time series cross-validation*, <https://otexts.com/fpp3/tscv.html>.
- **Relacion:** define ACF, fuerza de tendencia/estacionalidad, entropia
  espectral como descriptor, errores escalados y origen rodante.
- **Limite:** el HTML local es una portada navegable, no un archivo completo de
  todos los capitulos; para estas afirmaciones se deben abrir los enlaces
  oficiales anteriores. Un descriptor no se convierte por aparecer en el libro
  en criterio de agrupacion o seleccion.
- **Fuente:** Hyndman y Athanasopoulos, *Forecasting: Principles and Practice*,
  3.a ed., OTexts, 2021.

### [7] `07_Zeng_2023_DLinear.pdf`

- **En la propuesta:** pagina 4, primer antecedente y control sencillo.
- **Lectura minima:** PDF p. 1, resumen; pp. 2-4, definicion y variantes de
  LTSF-Linear.
- **Relacion:** muestra que modelos lineales de una capa fueron comparadores
  competitivos frente a varios transformadores en esos bancos.
- **Limite:** no demuestra que DLinear sea universalmente superior ni que deba
  ganar en las tareas de la tesis.
- **Fuente:** arXiv:2205.13504, version de AAAI 2023.

### [8] `08_Nie_2023_PatchTST.pdf`

- **En la propuesta:** pagina 4, vecino de segmentacion temporal y procesamiento
  por canal.
- **Lectura minima:** PDF p. 1, resumen; pp. 2-4, seccion 3 y Fig. 2 sobre
  *patching* e independencia de canales.
- **Relacion:** procesa segmentos de cada variable y comparte pesos del
  codificador entre variables.
- **Limite:** independencia de canales no equivale a aprender grupos de
  variables ni a seleccionar extractores por grupo.
- **Fuente:** arXiv:2211.14730, ICLR 2023.

### [9] `09_Liu_2024_iTransformer.pdf`

- **En la propuesta:** pagina 4, tratamiento de variables como unidades de
  entrada de la atencion.
- **Lectura minima:** PDF p. 1, resumen; pp. 4-5, seccion 3 y Fig. 2.
- **Relacion:** invierte los ejes usuales del transformer: cada variable se
  embebe como token y la atencion modela relaciones entre variables.
- **Limite:** no realiza la agrupacion previa ni el enrutamiento modular que se
  estudian en la propuesta.
- **Fuente:** arXiv:2310.06625, ICLR 2024.

### [10] `10_Zhang_2024_MTST.pdf`

- **En la propuesta:** pagina 4, antecedente multi-resolucion.
- **Lectura minima:** PDF p. 1, resumen; p. 3, Fig. 2 y seccion 3.1.
- **Relacion:** usa una arquitectura de varias ramas para representar patrones
  temporales en resoluciones distintas.
- **Limite:** las ramas representan escalas temporales, no grupos de variables
  aprendidos mediante diagnosticos.
- **Fuente:** PMLR 238, AISTATS 2024.

### [11] `11_Chen_2024_Pathformer.pdf`

- **En la propuesta:** pagina 4, rutas adaptativas entre escalas.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, seccion 3 y Fig. 2.
- **Relacion:** adapta caminos de procesamiento multi-escala en funcion de la
  entrada.
- **Limite:** no selecciona la particion de variables planteada aqui ni estudia
  el mismo costo de busqueda.
- **Fuente:** arXiv:2402.05956, ICLR 2024.

### [12] `12_Wang_2024_TimeMixer.pdf`

- **En la propuesta:** pagina 4, mezcla de componentes en varias resoluciones.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, seccion 3.1, Fig. 1 y bloques
  PDM/FMM.
- **Relacion:** descompone y mezcla informacion pasada y predicciones en varias
  escalas.
- **Limite:** no aprende los grupos de variables ni responde a la pregunta de
  seleccion modular de la propuesta.
- **Fuente:** arXiv:2405.14616, ICLR 2024.

### [13] `13_Cai_2024_MSGNet.pdf`

- **En la propuesta:** pagina 4, relaciones entre variables dependientes de la
  escala temporal.
- **Lectura minima:** PDF p. 1, resumen; p. 3, Fig. 2 y descripcion de
  `ScaleGraphBlock`.
- **Relacion:** combina identificacion de escalas por frecuencia y convolucion
  de grafo adaptativa para relaciones entre series.
- **Limite:** escala y grafo se aprenden dentro de una arquitectura especifica;
  no son la interfaz modular ni el protocolo de seleccion propuestos.
- **Fuente:** arXiv:2401.00423, AAAI 2024.

### [14] `14_Chen_2024_CCM.pdf`

- **En la propuesta:** pagina 4, vecino directo de agrupacion de canales.
- **Lectura minima:** PDF p. 2, Fig. 1; pp. 4-5, seccion 4.2, asignador de grupos
  y representaciones de grupo.
- **Relacion:** introduce un modulo independiente del predictor que aprende
  asignaciones suaves de canales a grupos.
- **Limite:** no decide entre familias heterogeneas de extractores ni incorpora
  la seleccion multifidelidad y la abstencion de esta propuesta.
- **Fuente:** arXiv:2404.01340, NeurIPS 2024.

### [15] `15_Liu_2024_DGCformer.pdf`

- **En la propuesta:** pagina 4, agrupacion profunda basada en grafos.
- **Lectura minima:** PDF p. 1, resumen; p. 3, seccion 3.2, *Deep Graph
  Clustering*.
- **Relacion:** combina autoencoder y red de grafos para separar canales en
  grupos antes de la atencion.
- **Limite:** es una arquitectura completa; no demuestra la utilidad de una
  interfaz de agrupacion y extractores intercambiables.
- **Fuente:** arXiv:2405.08440.

### [16] `16_Qiu_2025_DUET.pdf`

- **En la propuesta:** pagina 4, agrupacion temporal y relaciones entre canales.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, planteamiento y Fig. 4 sobre
  `Temporal Clustering Module` y `Channel-Soft-Clustering`.
- **Relacion:** combina agrupacion por distribucion temporal con una mascara de
  atencion aprendida entre canales.
- **Limite:** no compara un banco de extractores por grupo bajo presupuestos
  parciales ni formula una decision de abstencion.
- **Fuente:** arXiv:2412.10859, KDD 2025.

### [17] `17_Montero_2020_FFORMA.pdf`

- **En la propuesta:** pagina 4, adaptacion guiada por caracteristicas de la
  serie.
- **Lectura minima:** PDF p. 3, resumen; pp. 5-6, seccion 2 y diagrama del
  procedimiento.
- **Relacion:** un metamodelo usa caracteristicas de cada serie para asignar
  pesos a pronosticos de modelos completos.
- **Limite:** pondera pronosticadores terminados; no selecciona extractores por
  rama ni aprende agrupaciones multivariadas.
- **Fuente:** working paper abierto del articulo de IJF 36(1), 2020.

### [18] `18_Liang_2024_EMTSF.pdf`

- **En la propuesta:** pagina 4, busqueda de arquitectura mediante evolucion.
- **Lectura minima:** PDF p. 1, resumen; p. 3, seccion 3 y espacios de busqueda
  espacial y temporal.
- **Relacion:** optimiza de manera conjunta operaciones convolucionales para
  dependencias entre variables y en el tiempo.
- **Limite:** su busqueda de arquitectura no resuelve la agrupacion previa ni el
  protocolo de evidencia parcial de la propuesta.
- **Fuente:** PMLR 222, ACML 2023/2024.

### [19] `19_Leppich_2025_REPNet.pdf`

- **En la propuesta:** pagina 4, descomposicion modular del proceso de
  pronostico.
- **Lectura minima:** PDF pp. 1-2, resumen, introduccion y Fig. 1; p. 9 para las
  configuraciones de los modulos.
- **Relacion:** separa representacion de entrada, construccion de memoria o
  extraccion de informacion y proyeccion al objetivo.
- **Limite:** modularizar el pipeline no equivale a aprender grupos ni a
  seleccionar sus extractores con fidelidades parciales.
- **Fuente:** arXiv:2507.05891, preprint.

### [20] `20_Ma_2026_Predictive_Heterogeneity.pdf`

- **En la propuesta:** pagina 4, agrupacion guiada por validacion con retorno a
  un modelo global.
- **Lectura minima:** PDF p. 1, resumen; pp. 5-6, seccion 2; pp. 13-16 para el
  algoritmo y el criterio de respaldo global.
- **Relacion:** la especializacion se conserva solo si mejora el error de
  validacion; de lo contrario se vuelve al modelo global.
- **Limite:** es un preprint de 2026 y no prueba la novedad ni los resultados de
  esta tesis. Su unidad y mecanismo de agrupacion son distintos.
- **Fuente:** arXiv:2604.13748, preprint.

### [21] `21_Woo_2022_CoST.pdf`

- **En la propuesta:** pagina 4, relacion entre representaciones temporales y
  procesamiento de senales.
- **Lectura minima:** PDF p. 1, resumen e introduccion; pp. 3-4, arquitectura y
  objetivos en tiempo y frecuencia.
- **Relacion:** aprende por contraste representaciones separadas de tendencia y
  estacionalidad en dominios temporal y frecuencial.
- **Limite:** no es una prueba de que una descomposicion sea causal en nuestro
  uso ni de que deba mejorar RL.
- **Fuente:** OpenReview/ICLR 2022, arXiv:2202.01575.

### [22] `22_Zeghidour_2021_LEAF.pdf`

- **En la propuesta:** pagina 4, ejemplo de frontend aprendible.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-5, arquitectura del banco de
  filtros Gabor y normalizacion.
- **Relacion:** reemplaza un frontend fijo de audio por operaciones aprendibles
  y diferenciables.
- **Limite:** es evidencia en audio; no se traslada automaticamente a series
  financieras ni a pronostico.
- **Fuente:** arXiv:2101.08596, ICLR 2021.

### [23] `23_Schlueter_2022_EfficientLEAF.pdf`

- **En la propuesta:** pagina 4, cautela frente a frontends aprendibles.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, resultados y conclusion.
- **Relacion:** sus variantes aprendibles no superaron de forma consistente un
  banco mel fijo, por lo que obliga a conservar controles sencillos.
- **Limite:** tampoco demuestra que el aprendizaje de frontends sea inutil fuera
  de las tareas de audio estudiadas.
- **Fuente:** arXiv:2207.05508.

### [24] `24_Li_2023_TiMAE.pdf`

- **En la propuesta:** pagina 4, antecedente del entrenamiento auxiliar por
  reconstruccion enmascarada.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, seccion 3 y Figs. 1-3. Para
  extraccion y reutilizacion del codificador, consultar el apendice, pp. 15-16.
- **Relacion:** preentrena un autoencoder de series temporales reconstruyendo
  puntos enmascarados.
- **Limite:** no demuestra que el detector auxiliar de esta propuesta deba
  mejorar el objetivo final ni resuelve el regimen de congelacion o ajuste.
- **Fuente:** arXiv:2301.08871.

### [25] `25_Bastidas_feature_extractor_autoencoder_manager.py`

- **En la propuesta:** pagina 4, evidencia del punto de partida de software.
- **Lectura minima:** lineas 478-488, metodos `save_encoder()` y
  `save_decoder()`.
- **Relacion:** prueba que el gestor puede persistir codificador y decodificador
  por separado.
- **Limite:** no prueba ensamblaje diferenciable, flujo de gradientes ni que el
  codificador se actualice dentro del predictor; eso sigue siendo una
  comprobacion de implementacion.
- **Fuente:** `feature-extractor`, revision `df86252`.

### [26] `26_Bastidas_predictor_train_fe_config.json`

- **En la propuesta:** pagina 4, configuracion historica para cargar y ajustar
  un extractor.
- **Lectura minima:** lineas 56-57, `feature_extractor_file` y `train_fe`.
- **Relacion:** documenta la intencion declarada en una configuracion concreta.
- **Limite:** una opcion de configuracion no prueba que el camino ejecutado
  ensamble el extractor ni que sus pesos reciban gradientes.
- **Fuente:** `predictor`, revision `20ec571`, config
  `phase_3_1_cnn_25200_1h`.

### [27] `27_Bastidas_doin_node_README.md`

- **En la propuesta:** pagina 12, despliegue e integracion de la optimizacion y
  registro de experimentos.
- **Lectura minima:** lineas 10-17, roles y metricas por ronda; lineas 44-75,
  responsabilidades y modulo OLAP; lineas 171-199, uso distribuido; lineas
  213-223, artefactos y registro.
- **Relacion:** sustenta que `doin-node` ejecuta roles distribuidos y dispone de
  un esquema analitico para resultados de optimizacion.
- **Limite:** DOIN es infraestructura, no evidencia de las hipotesis doctorales.
  El README advierte que una tabla OLAP concreta aun no tiene consumidor de
  runtime. No se afirma aqui que el nodo ejecute DEAP.
- **Fuente:** `doin-node`, revision completa
  `8bfc64f5de20200b93a8f9451b4e1a7ea9742df8`.

### [28] `28_Bastidas_financial_data_INVENTORY.md`

- **En la propuesta:** pagina 5, seccion 4.1, inventario financiero como una
  fuente de datos del programa.
- **Lectura minima:** lineas 1-12, estado y resumen; lineas 14-22, raices; lineas
  24-39, validacion y huecos conocidos.
- **Relacion:** inventaria volumen, formatos y raices disponibles en una fecha
  de corte.
- **Limite:** no garantiza por si solo licencia, semantica, unidad,
  disponibilidad temporal ni elegibilidad de cada variable; esas propiedades
  requieren contratos posteriores.
- **Fuente:** `financial-data/INVENTORY.md`, revision
  `a8087108812072d55387877ca14acd97f9213049`.

### [29] `29_IBM_2011_CRISP_DM.pdf`

- **En la propuesta:** pagina 5, seccion 4, estructura de la metodologia y Tabla
  1.
- **Lectura minima:** PDF p. 9, vision general y ciclo de seis fases. Para el
  detalle: comprension del negocio pp. 12-20; datos pp. 21-27; preparacion pp.
  28-33; modelado pp. 34-40; evaluacion pp. 41-43; despliegue pp. 44-49.
- **Relacion:** aporta el proceso iterativo que organiza el estudio.
- **Limite:** CRISP-DM no define las hipotesis, las pruebas causales ni los
  criterios estadisticos concretos de la propuesta.
- **Fuente:** *IBM SPSS Modeler CRISP-DM Guide*, version 14, 2011.

### [30] `30_Qiu_2024_TFB.pdf`

- **En la propuesta:** pagina 5, punto de partida para tareas publicas de
  pronostico.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-4, seccion 3 sobre principios y
  pipeline; pp. 5-9, seccion 4 y tablas de datasets, metodos y metricas.
- **Relacion:** ofrece un benchmark amplio con datos y evaluacion unificados.
- **Limite:** citar TFB no fija automaticamente el subconjunto, las particiones
  ni las licencias de la tesis; estos se congelaran en el protocolo propio.
- **Fuente:** arXiv:2403.20150, PVLDB 17(9), 2024.

### [31] `31_Sun_2023_TradeMaster.pdf`

- **En la propuesta:** pagina 5, referencia externa para tareas de RL
  financiero.
- **Lectura minima:** PDF p. 1, resumen; pp. 3-5, datos y entornos; pp. 6-10,
  seccion 4, diseno y resultados del benchmark.
- **Relacion:** integra datos, entornos de mercado y comparacion de algoritmos de
  RL para trading.
- **Limite:** no se asume compatibilidad con nuestro contrato; recuperar datos,
  costos y condiciones comparables es una tarea previa.
- **Fuente:** NeurIPS 2023, Datasets and Benchmarks.

### [32] `32_Liu_2022_FinRL_Meta.pdf`

- **En la propuesta:** pagina 5, segunda referencia externa para entornos y
  benchmarks financieros.
- **Lectura minima:** PDF p. 1, resumen; pp. 5-7, secciones 4.1-4.3 sobre capas
  de datos y entorno; pp. 8-9, seccion 5 sobre metricas y reproducciones.
- **Relacion:** organiza datos, entornos Gym y agentes como capas separadas.
- **Limite:** es referencia de comparacion e infraestructura, no evidencia
  principal para H1-H3.
- **Fuente:** NeurIPS 2022, Datasets and Benchmarks.

### [33] `33_Keras_Transfer_Learning.html`

- **En la propuesta:** pagina 8, regimenes de congelacion y ajuste del detector.
- **Lectura minima:** ancla
  `#freezing-layers-understanding-the-trainable-attribute`, lineas 486-595;
  revisar tambien el apartado de `BatchNormalization`, lineas 513-541, y la
  distincion entre `trainable` y el argumento `training`, lineas 595-598.
- **Relacion:** documenta que congelar controla pesos entrenables y que algunas
  capas conservan estado no entrenable con semantica especial.
- **Limite:** no verifica el flujo de gradientes del codigo del proyecto; ese
  flujo debe comprobarse con pruebas estructurales y de comportamiento.
- **Fuente:** guia oficial de Keras, actualizada en 2023.

### [34] `34_Hyndman_2006_MASE.pdf`

- **En la propuesta:** pagina 10, medida principal del experimento de
  pronostico.
- **Lectura minima:** PDF p. 12, seccion 3, definicion de errores escalados y
  MASE; p. 13 para interpretacion y comparacion.
- **Relacion:** escala el error absoluto por un error ingenuo calculado dentro de
  la serie y facilita comparar escalas diferentes.
- **Limite:** la propuesta debe definir aparte el denominador estacional por
  origen, la agregacion entre horizontes y el tratamiento de denominadores
  degenerados.
- **Fuente:** copia de autor de IJF 22(4), 2006.

### [35] `35_TFP_Probabilistic_Layers_Regression.html`

- **En la propuesta:** pagina 12, cabezales probabilisticos para intervalos de
  pronostico.
- **Lectura minima:** ancla `#case_2_aleatoric_uncertainty`, lineas 3829-3852;
  ancla `#case_3_epistemic_uncertainty`, lineas 3888-3929; clase
  `DistributionLambda` en esos ejemplos.
- **Relacion:** muestra como producir una distribucion predictiva y separar
  ejemplos de incertidumbre aleatoria y del modelo.
- **Limite:** una salida probabilistica no garantiza calibracion. Cobertura,
  anchura y calibracion deben evaluarse fuera de muestra; tampoco debe
  confundirse con pronosticar volatilidad.
- **Fuente:** documentacion oficial de TensorFlow Probability.
