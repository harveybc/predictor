# Referencias de la propuesta de representaciones temporales modulares

Los números del nombre coinciden con `[n]` en

`docs/propuesta_doctoral_representaciones_temporales_modulares.pdf`

(bibliografía IEEE, orden de primera cita). Abrir el PDF de la propuesta y el archivo `NN_…` a la vez.

Los PDF y copias locales quedan **en esta carpeta**. No se suben al remoto: varios son preprints de autor o documentación web. El índice sí se versiona.

Donde el artículo de revista está de pago se dejó el preprint o el informe técnico abierto, y se marca abajo.

---

### [1] `01_Bengio_2013_Representation_Learning.pdf`
La propuesta toma que el aprendizaje de representaciones transforma observaciones en características que facilitan una tarea, como pronosticar o seleccionar acciones.
Fuente: arXiv:1206.5538 (preprint de IEEE TPAMI 35(8), 2013).

### [2] `02_Cawley_2010_Overfitting_Model_Selection.pdf`
Se usa para advertir que la selección de configuraciones también puede sobreajustarse a los datos con los que se comparan los modelos.
Fuente: JMLR 11(70), 2010, PDF del sitio jmlr.org.

### [3] `03_Bai_2018_TCN.pdf`
Ilustra que las capas siguientes de una red convolucional/recurrente pueden ampliar el campo receptivo más allá de la primera capa.
Fuente: arXiv:1803.01271.

### [4] `04_Keras_Conv1D.html`
Documentación de la capa Conv1D usada en la figura ilustrativa del extractor (no es un artículo; la cita es la guía oficial).
Fuente: keras.io, consultada el 2026-09-09.

### [5] `05_Keras_LSTM.html`
Documentación de la capa LSTM de la misma figura ilustrativa del extractor.
Fuente: keras.io, consultada el 2026-09-09.

### [6] `06_Hyndman_2021_FPP3.html`
Se extraen características de series, que la autocorrelación puede reflejar tendencias, evaluación con orígenes sucesivos, y MASE estacional.
Fuente: OTexts, *Forecasting: Principles and Practice* 3.ª ed. (libro web, no PDF de artículo).

### [7] `07_Zeng_2023_DLinear.pdf`
Antecedente de que un modelo lineal puede superar varios transformadores en pronóstico; sirve de control sencillo frente a arquitecturas más pesadas.
Fuente: arXiv:2205.13504 (versión de AAAI 2023).

### [8] `08_Nie_2023_PatchTST.pdf`
Procesa segmentos de cada variable con pesos compartidos entre variables; vecino de procesamiento por canal, no de agrupación aprendida.
Fuente: arXiv:2211.14730 (ICLR 2023).

### [9] `09_Liu_2024_iTransformer.pdf`
Representa las variables como unidades de entrada de la atención para modelar sus relaciones.
Fuente: arXiv:2310.06625 (ICLR 2024).

### [10] `10_Zhang_2024_MTST.pdf`
Usa ramas de distintas resoluciones temporales; antecedente de extractores paralelos, no de agrupación de variables.
Fuente: PMLR v238 (AISTATS 2024).

### [11] `11_Chen_2024_Pathformer.pdf`
Adapta las rutas entre representaciones de distintas resoluciones temporales.
Fuente: arXiv:2402.05956 (ICLR 2024).

### [12] `12_Wang_2024_TimeMixer.pdf`
Combina componentes en varias resoluciones; antecedente de mezcla multi-escala, no de agrupación.
Fuente: arXiv:2405.14616 (ICLR 2024).

### [13] `13_Cai_2024_MSGNet.pdf`
Aprende relaciones entre variables a distintas escalas temporales, asociadas a componentes de distinta periodicidad.
Fuente: arXiv:2401.00423 (AAAI 2024).

### [14] `14_Chen_2024_CCM.pdf`
Aprende asignaciones a grupos y representaciones de esos grupos; vecino directo de agrupación de canales.
Fuente: arXiv:2404.01340 (NeurIPS 2024).

### [15] `15_Liu_2024_DGCformer.pdf`
Combina un autoencoder y convoluciones sobre grafos para agrupar variables antes de aplicar atención.
Fuente: arXiv:2405.08440.

### [16] `16_Qiu_2025_DUET.pdf`
Combina extractores según distribuciones temporales y aprende relaciones entre canales en frecuencia con una máscara de atención.
Fuente: arXiv:2412.10859 (KDD 2025).

### [17] `17_Montero_2020_FFORMA.pdf`
Utiliza características de las series para ponderar pronósticos de modelos completos; no selecciona extractores por rama.
Fuente: working paper de Monash (el IJF 36(1) está de pago). Misma tesis.

### [18] `18_Liang_2024_EMTSF.pdf`
Busca mediante evolución módulos de convolución espacial y temporal; búsqueda de arquitectura, no agrupación previa.
Fuente: PMLR v222 (ACML 2023/2024).

### [19] `19_Leppich_2025_REPNet.pdf`
Estudia configuraciones de representación, extracción de información y proyección en un pipeline modular de pronóstico.
Fuente: arXiv:2507.05891.

### [20] `20_Ma_2026_Predictive_Heterogeneity.pdf`
Actualiza grupos con el error de validación y vuelve a un modelo global cuando la especialización no mejora ese error.
Fuente: arXiv:2604.13748 (preprint 2026).

### [21] `21_Woo_2022_CoST.pdf`
Aprende representaciones de tendencia y estacionalidad; ejemplo del vínculo con procesamiento de señales.
Fuente: OpenReview/ICLR 2022 (arXiv:2202.01575).

### [22] `22_Zeghidour_2021_LEAF.pdf`
Aprende un banco de filtros como frontend de audio; analogía de frontend aprendible, no resultado que se traslade a finanzas.
Fuente: arXiv:2101.08596 (ICLR 2021).

### [23] `23_Schlueter_2022_EfficientLEAF.pdf`
Encontró que alternativas aprendibles no superaban de forma consistente un banco mel fijo; motiva comparar con diseños sencillos y no copiar resultados de audio.
Fuente: arXiv:2207.05508.

### [24] `24_Li_2023_TiMAE.pdf`
Antecedente de preentrenamiento por reconstrucción enmascarada específico para series temporales (el detector de la propuesta).
Fuente: arXiv:2301.08871.

### [25] `25_Bastidas_feature_extractor_autoencoder_manager.py`
Código del gestor de autoencoder: permite guardar codificador y decodificador por separado.
Fuente: `feature-extractor`, revisión `df86252` (artefacto citado, no un paper).

### [26] `26_Bastidas_predictor_train_fe_config.json`
Configuración que carga un extractor y habilita su ajuste (`train_fe`); el ensamblaje diferenciable se comprobará en la implementación.
Fuente: `predictor`, revisión `20ec571`, config `phase_3_1_cnn_25200_1h`.

### [27] `27_IBM_2011_CRISP_DM.pdf`
Las seis etapas de CRISP-DM estructuran el plan de trabajo y la tabla correspondiente de la propuesta.
Fuente: *IBM SPSS Modeler CRISP-DM Guide*, versión 14 (PDF público).

### [28] `28_Qiu_2024_TFB.pdf`
Punto de partida de conjuntos públicos de pronóstico: reúne tareas y métodos de distintos dominios.
Fuente: arXiv:2403.20150 (PVLDB 17(9), 2024).

### [29] `29_Sun_2023_TradeMaster.pdf`
Entorno y referencias de RL financiero para buscar experimentos compatibles; la comparación externa depende de recuperar sus datos y condiciones.
Fuente: NeurIPS 2023 Datasets and Benchmarks.

### [30] `30_Liu_2022_FinRL_Meta.pdf`
Entornos de mercado y benchmarks de RL financiero; mismo uso que TradeMaster como referencia externa, no como evidencia principal.
Fuente: NeurIPS 2022 Datasets and Benchmarks.

### [31] `31_Keras_Transfer_Learning.html`
Distinción `trainable` frente a `training` y tratamiento de estados de normalización que se actualizan sin gradientes al congelar o ajustar el detector.
Fuente: guía oficial de Keras, actualizada en 2023 (HTML, no artículo).

### [32] `32_Hyndman_2006_MASE.pdf`
Define el error absoluto medio escalado (MASE) como medida principal de pronóstico.
Fuente: copia de autor del IJF 22(4), 2006.

### [33] `33_TFP_Probabilistic_Layers_Regression.html`
Cabezales probabilísticos para intervalos predictivos, distintos de la volatilidad como variable de la ventana futura.
Fuente: documentación de TensorFlow Probability (HTML, no artículo).
