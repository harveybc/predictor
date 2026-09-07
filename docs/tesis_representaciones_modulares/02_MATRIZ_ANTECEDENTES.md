# Matriz dirigida de antecedentes

Esta matriz sustenta el borrador, pero no sustituye la revisión sistemática del primer semestre. La brecha se mantiene expresamente como provisional.

| Trabajo | Qué diseña o aprende | Tratamiento temporal y de variables | Consumidor | Coincidencia | Diferencia que queda por comprobar |
|---|---|---|---|---|---|
| [DLinear, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | Descomposición y proyección lineal | Operación simple por serie | Pronóstico | Control simple fuerte | No compone ramas mediante perfiles temporales. |
| [PatchTST, ICLR 2023](https://openreview.net/forum?id=Jbdc0vTOcol) | Parches temporales | Canales independientes | Pronóstico | Relaciona resolución y representación | No usa un perfil train-only para agrupar variables y asignar alcances. |
| [iTransformer, ICLR 2024](https://openreview.net/forum?id=JePfAI8fah) | Tokens por variable | Atención entre variables | Pronóstico | Modela dependencia multivariada | No propone composición de ramas y campos receptivos desde perfiles temporales. |
| [TimesNet, ICLR 2023](https://openreview.net/forum?id=ju_Uqw384Oq) | Variación temporal 2D | Periodicidades múltiples | Análisis temporal general | Usa estructura periódica | La transformación es parte de una arquitectura fijada, no una regla transferida entre familias. |
| [MTST, AISTATS 2024](https://proceedings.mlr.press/v238/zhang24l.html) | Ramas con varias resoluciones de parche | Multirresolución | Pronóstico | Vecino directo multirrama | Las resoluciones son un diseño del modelo, no una asignación por perfiles de variables. |
| [Pathformer, ICLR 2024](https://openreview.net/forum?id=lJkOCMP2aW) | Rutas adaptativas entre escalas | Multiescala aprendida | Pronóstico | Vecino directo en selección de escalas | Aprende rutas dentro de una arquitectura; no prueba la regla perfil-grupo-campo receptivo entre familias. |
| [TimeMixer, ICLR 2024](https://openreview.net/forum?id=7oLshfEIC2) | Mezcla de resoluciones | Componentes finos y gruesos | Pronóstico | Vecino directo multiescala | No es una regla de composición basada en perfiles train-only por variable. |
| [CCM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb9b18ccb76a1156af5779ffdca1d91f-Abstract-Conference.html) | Agrupamiento de canales | Similitud entre canales | Pronóstico | Vecino directo de agrupación | Debe comprobarse si su agrupación y la propuesta son equivalentes bajo el mismo contrato. |
| [Leddam, ICML 2024](https://proceedings.mlr.press/v235/yu24s.html) | Descomposición y atención aprendibles | Variación intra-serie e inter-serie | Pronóstico | Acopla componentes y dependencias | No evalúa la regla conjunta de grupo y alcance aquí propuesta. |
| [CoST, ICLR 2022](https://openreview.net/forum?id=PilZY3omXV2) | Representaciones contrastivas separadas | Tendencia y estacionalidad | Pronóstico | Aprende componentes temporales | Usa un objetivo contrastivo distinto y no fundamenta campos receptivos por grupo. |
| [BTSF, ICML 2022](https://proceedings.mlr.press/v162/yang22e.html) | Fusión temporal-espectral | Dos vistas de la serie | Representación no supervisada | Fusión de vistas relevante | No estudia agrupación modular ni transferencia de la regla de composición. |
| [FFORMA, IJF 2020](https://doi.org/10.1016/j.ijforecast.2019.02.011) | Pesos de modelos desde características | Características globales por serie | Selección/ensamble de pronósticos | Usa propiedades para decidir | Escoge modelos terminados; no diseña la representación interna. |
| [EMTSF, ACML 2023](https://proceedings.mlr.press/v222/liang24a.html) | Arquitectura espacial y temporal | Búsqueda evolutiva de módulos | Pronóstico | Vecino de búsqueda modular | Su regla y su costo deben compararse con la restricción informada propuesta. |
| [REP-Net, 2025](https://arxiv.org/abs/2507.05891) | Etapas de representación, extracción y proyección | Tubería modular | Pronóstico | Vecino conceptual muy cercano | La equivalencia real debe resolverse al revisar método, código y población experimental. |
| [SincNet, 2018](https://arxiv.org/abs/1808.00158) | Filtros parametrizados | Frecuencias de corte aprendibles | Voz | Diseño con significado físico y aprendizaje | Evidencia en audio; no valida la regla temporal multivariada. |
| [LEAF, ICLR 2021](https://openreview.net/forum?id=jM76BCb6F9m) | Frente de audio aprendible | Filtrado, compresión y normalización | Clasificación de audio | Acopla representación y tarea | No prueba utilidad en pronóstico ni la composición de ramas propuesta. |
| [EfficientLEAF, 2022](https://arxiv.org/abs/2207.05508) | Frente aprendible eficiente | Banco fijo frente a aprendido | Audio | Control adverso necesario | Muestra que lo aprendible no supera siempre una representación fija. |
| [TFB, VLDB 2024](https://doi.org/10.14778/3665844.3665863) | Banco y protocolo | Evaluación amplia | Pronóstico | Fuente posible de tareas y comparadores | No constituye por sí mismo la población confirmatoria definitiva. |
| [Monash Archive, NeurIPS 2021](https://openreview.net/forum?id=I01l7rc0jcb) | Archivo de datos | Series de múltiples dominios | Pronóstico | Banco público | Se debe seleccionar por familias, longitud, licencias y multivarianza. |
| [GIFT-Eval, ICML 2025](https://openreview.net/forum?id=Z2cMOOANFX) | Banco amplio | Frecuencias y dominios diversos | Pronóstico | Validación externa potencial | Su alcance y costo pueden exceder el mínimo defendible. |

## Brecha provisional

La revisión dirigida no encontró un método que, bajo un mismo protocolo:

1. estime con datos de entrenamiento un perfil temporal reutilizable por variable;
2. use ese perfil para restringir conjuntamente la agrupación interna de variables y los campos receptivos de ramas aprendibles; y
3. pruebe la transferencia de esa regla de composición a familias de datos reservadas, controlando capacidad y costo.

La afirmación no es todavía una conclusión sistemática. Debe sobrevivir búsquedas por citas hacia delante y atrás de CCM, MTST, Pathformer, TimeMixer, EMTSF y REP-Net, además de trabajos de neural architecture search y multivariate time-series representation learning publicados durante 2025-2026.
