# Auditoria de referencias, 28 de septiembre de 2026

Se audito el PPT editado por el usuario, no la version anterior de GitHub.
Solo se modifica la diapositiva 15. Las diapositivas 1-14, la bibliografia
y todas las imagenes se conservan byte a byte dentro del PPTX.

## Referencia por referencia

| Ref. | Existencia y comprobacion | Correspondencia con el contenido |
| --- | --- | --- |
| 1 | DOCX original retenido, portada de tres autores, junio de 2025, figura 43 y conclusiones revisadas. SHA256 ec478ae599fadf8769bd77bec699a12f48cd1e9511d4a33be64a9046d9e0eabd. | Respalda la figura del trading y el mejor resultado historico del extractor CNN. No establece una ley universal entre MAE y Sharpe. El propio documento advierte dispersion y otros factores. |
| 2 | Archivo publicado en heuristic-strategy, commit f2d3922: README, CSV y codigo historicos, datos y resultados nativos posteriores separados. | Respalda el barrido direccional mostrado y documenta por separado la estrategia de dos predicciones. No confundir ambas simulaciones ni el cruce de beneficio cero con el naive. No es evidencia de rentabilidad real. |
| 3 | [FPP3, caracteristicas STL](https://otexts.com/fpp3/stlfeatures.html) y [otras caracteristicas](https://otexts.com/fpp3/other-features.html). | Fuerza estacional, autocorrelacion, entropia y caracterizacion por bloques. No prescribe una arquitectura ganadora para cada estadistico. |
| 4 | [Documentacion oficial ADF/KPSS](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html). | Pruebas con hipotesis nulas distintas para estudiar estacionariedad. No rechazar no equivale a demostrar; la diapositiva las presenta como medidas, no como garantia. |
| 5 | [SciPy signal](https://docs.scipy.org/doc/scipy/reference/signal.html) y [Hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html). | Correlacion, coherencia y fase/frecuencia instantanea. El periodo local requiere interpretacion por banda; la implementacion FFT no garantiza causalidad operativa. |
| 6 | [Pagina editorial de catch22](https://link.springer.com/article/10.1007/s10618-019-00647-x): autores, titulo, volumen 33, paginas 1821-1852, 2019. | Respalda perfiles compactos de series. Antecedente, no demostracion de H1-H3 ni de agrupamiento optimo. |
| 7 | [Pagina y manuscrito del coautor de FFORMA](https://robjhyndman.com/publications/fforma/): cuatro autores, 36(1), 86-92, 2020, DOI confirmado. El acceso directo editorial fallo en esta consulta; se uso la fuente del autor. | Combina pronosticos con pesos aprendidos desde caracteristicas. La diapositiva 9 lo describe correctamente. La frase mas amplia de la 11 es un antecedente de decisiones basadas en caracteristicas, no una seleccion de extractores demostrada por FFORMA. |
| 8 | [Bai, Kolter y Koltun](https://arxiv.org/abs/1803.01271), 2018, titulo y autores confirmados. | Comparacion de redes convolucionales y recurrentes para secuencias. No afirma que LSTM sea siempre mejor para largo plazo ni que Conv1D sea siempre mejor para una serie estacionaria. |
| 9 | [Actas originales de NeurIPS](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need), 2017. | Atencion multi-cabezal y Transformer. La aplicacion original es traduccion; las aplicaciones a series estan en 11-12. |
| 10 | [Ti-MAE](https://arxiv.org/abs/2301.08871), cinco autores, 2023. | Autoencoder temporal con reconstruccion enmascarada. Las modalidades R0/R1/R2 son nuestro contraste, no un resultado que se atribuya al articulo. |
| 11 | [PatchTST](https://arxiv.org/abs/2211.14730), cuatro autores; aceptacion ICLR 2023 declarada en el registro. | Pronostico multivariado, bancos publicos y preentrenamiento. Soporta ejemplos de dominios y protocolos, no la superioridad futura de nuestra propuesta. |
| 12 | [iTransformer](https://arxiv.org/abs/2310.06625) y [texto completo](https://arxiv.org/html/2310.06625v4). | Electricity, Weather y Traffic, MSE/MAE, dependencias entre variables. Comparabilidad exige misma tarea, particion, escala y reduccion; no basta el nombre del dataset. |
| 13 | [Cawley y Talbot, JMLR](https://jmlr.org/papers/v11/cawley10a.html), 11, 2079-2107, 2010. | Sesgo por seleccion y necesidad de evaluacion separada. Habia quedado sin cita en la ultima edicion; se vincula ahora a la comprobacion en datos no usados para elegir configuracion, en la diapositiva 15. |

Las 13 entradas existen. Orden de primera aparicion 1-13 comprobado por
programa. No se encontraron entradas bibliograficas ficticias. Las hipotesis
se mantienen como propuestas a probar, no como conclusiones de los articulos.

## Observacion fuera del alcance de edicion

La nueva frase de la diapositiva 10 agrupa DLinear con las lineas base
estadisticas. DLinear es mas precisamente un modelo lineal aprendido con
descomposicion, presentado por Zeng et al. en
[Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504).
No tiene una cita propia en esa diapositiva. Se deja anotado sin modificarla,
para respetar la instruccion de conservar las diapositivas 1-14. No atribuir
su origen a Bai ni a Vaswani. ARIMAX pertenece a otra familia metodologica.

## Limite de la auditoria

Es una comprobacion bibliografica y de correspondencia, no una nueva
validacion de los backtests historicos ni una certificacion de las hipotesis.
Las limitaciones del barrido de ruido, incluido el swap historico no restado,
siguen declaradas en el archivo de sus fuentes; no se alteraron sus imagenes.
