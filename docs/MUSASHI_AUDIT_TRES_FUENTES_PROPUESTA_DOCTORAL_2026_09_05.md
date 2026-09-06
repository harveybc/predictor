# Auditoria Musashi: tres fuentes de conocimiento y propuesta doctoral

**Fecha:** 2026-09-05
**Material revisado:** carta de Retsu, propuesta de Satoshi, PATCH 003 y anexo maestro del 2026-09-05
**Veredicto:** `ACCEPT_AS_RESEARCH_PROGRAM / MAJOR_RECAST_AS_DOCTORAL_PROPOSAL`

## 1. Dictamen corto

La idea contiene una tesis doctoral posible y muy cercana al trabajo que Harvey realmente quiere hacer: aprender a escoger transformaciones y representaciones de datos sin probarlas todas, bajo restricciones temporales y de computo. Sin embargo, el paquete actual no es esa tesis. Es un programa de laboratorio con trece modulos, tres planos de evidencia, nueve paquetes de trabajo y varias lineas publicables independientes.

No recomiendo convertir "tres fuentes de conocimiento" en el titulo ni en la pregunta doctoral. Recomiendo una fusion asimetrica:

- **Fuente A** define el espacio de transformaciones y representaciones candidatas.
- **Fuente C** aprende cuales conviene evaluar, en que orden y cuando abstenerse.
- **Fuente B** queda como caso confirmatorio especializado o articulo separado; no es una tercera columna de la tesis.

La tesis no seria construir toda la arquitectura. Seria aprender una politica de seleccion para un grafo pequeno de transformaciones temporalmente validas y demostrar cuando transfiere a tareas nuevas y cuando debe abstenerse.

## 2. Lo que sobrevive

1. **El grafo y no la cadena serial.** Esta es la correccion conceptual mas importante. Denoising, cuantizacion, representaciones espectrales, separacion comun/privada y sincronizacion son alternativas o ramas condicionadas; no deben apilarse por analogia con comunicaciones.
2. **El nulo como resultado.** Una rama que no mejora fuera de muestra se elimina. Eso protege el proyecto del crecimiento por acumulacion de componentes.
3. **Validez temporal en cada interfaz.** Disponibilidad, ajuste, transformacion y evaluacion deben conservar una semantica point-in-time comprobable.
4. **Costo y utilidad incremental.** La seleccion debe optimizar utilidad fuera de muestra bajo presupuesto, no solo exactitud final.
5. **Historia de experimentos como evidencia.** Resultados fallidos, fidelidad, costo y procedencia pueden servir para recomendar las siguientes evaluaciones.
6. **Baselines sencillos primero.** Vecino por tarea y GBDT antes de Transformer, meta-RL o un optimizador universal.

Estas seis ideas forman un nucleo coherente de aprendizaje automatico. Los trece STEPS son un catalogo de candidatos y preguntas, no trece contribuciones.

## 3. Objeciones que hoy serian letales

### P0.1. No existe un unico objeto que aprende

El paquete alterna entre un extractor, detectores, un router por muestra, un asignador de anchos y un metaoptimizador entre tareas. Son problemas distintos, con unidades de decision y funciones de perdida distintas. La ecuacion

\[
(X_t,E_t,M) \rightarrow \text{ramas} \rightarrow Z
\]

los oculta, pero no los unifica.

La propuesta debe escoger un solo agente de aprendizaje: un **meta-selector de pipelines de transformacion** para una tarea nueva. El router por muestra y el asignador dinamico quedan fuera hasta que existan al menos dos ramas ya validadas.

### P0.2. La novedad choca frontalmente con AutoML y meta-learning

La seleccion automatica de preprocesamiento y modelos no es terreno vacio. Auto-sklearn ya incluyo metodos de preprocesamiento en un espacio conjunto y uso desempeno historico; la factorizacion probabilistica y TensorOboe aprendieron sobre matrices de resultados para buscar pipelines; FFORMA aprendio a seleccionar o combinar pronosticadores desde caracteristicas de series; y trabajos recientes recomiendan arquitecturas e hiperparametros para series mediante meta-learning.

Por tanto, "usar historia para escoger pipeline" no puede ser la contribucion. La brecha defendible debe formularse de manera mas estrecha:

> seleccion secuencial y abstentiva de grafos de transformacion temporalmente validos, bajo cambio entre tareas y presupuesto limitado, con medicion explicita de transferencia negativa.

Esta brecha necesita una revision sistematica propia antes de afirmar novedad.

### P0.3. Las tres fuentes no son objetos estadisticos comparables

`X_t` son observaciones por tiempo; los efectos de eventos se estiman sobre un panel y una poblacion; `M` resume campañas completas. No pueden presentarse como tres entradas sincronas de una misma funcion. Cada plano tiene otra granularidad, incertidumbre y regla de disponibilidad.

Si se conservan en el mapa, deben existir interfaces tipadas:

- dato por instante o ventana;
- estimacion de efecto con poblacion, horizonte e intervalo de incertidumbre;
- metadato por tarea/campaña disponible antes de evaluar la tarea nueva.

La tesis propuesta solo necesita el primero y el tercero. El segundo no debe cargar con la obligacion de unificar todo.

### P0.4. El carril de eventos todavia no es causal por llamarlo causal

Una local projection estima respuestas dinamicas, pero la interpretacion causal depende de la identificacion del tratamiento, anticipacion, eventos simultaneos, controles point-in-time y estabilidad del mecanismo. DML no repara por si solo un tratamiento confundido.

Ademas, un efecto medio o condicional no es una etiqueta verdadera para cada evento. Convertirlo en entrada de un predictor exige predicciones out-of-fold, una regla de disponibilidad y propagacion de incertidumbre. Hasta demostrar eso, el nombre correcto es **analisis de respuesta a eventos**. "Causal" se reserva para los subconjuntos con estrategia de identificacion defendida.

### P0.5. El meta-dataset descrito aun no existe

El OLAP desplegado de `predictor` conserva configuraciones y metricas agregadas, pero el registro canonico propuesto tambien exige fallos, genealogia, mascara de parametros, fidelidad, costo, version del espacio y politica de adquisicion. Decir que el proyecto "ya tiene" el sustrato L3 seria falso.

Antes de prometer el meta-selector debe hacerse un censo: numero de tareas independientes, pipelines comparables por tarea, porcentaje de matriz observado, versiones incompatibles, repeticiones, fallos conservados y costo real. Si esa matriz no tiene soporte suficiente, el primer aporte sera construir un benchmark publico, no entrenar OptFormer.

### P0.6. El anexo convierte compuertas en una escalera movil

El recorrido sintetico -> publico -> proyecto -> backtest -> RL parece riguroso, pero puede permitir que una hipotesis cambie de objetivo cada vez que falla. Cada experimento necesita antes de ejecutarse: poblacion, estimando, baseline, metrica primaria, margen, presupuesto, multiplicidad y condicion terminal. "Otro paradigma lo validara" no es una salida admisible.

## 4. Recorte doctoral recomendado

### Titulo de trabajo

**Meta-seleccion abstentiva de transformaciones para series temporales bajo presupuesto y cambio entre tareas**

No usaria en el titulo "conocimiento", "informacional", "causal" ni los trece STEPS. Son palabras mas grandes que el objeto medido.

### Pregunta madre

> Puede un meta-selector, entrenado con evaluaciones historicas y descriptores baratos de cada tarea, escoger un pequeno grafo de transformaciones temporalmente validas para una serie no vista, reducir el costo de alcanzar un desempeno objetivo frente a busqueda sin transferencia y abstenerse cuando la historia disponible no permite recomendar sin transferencia negativa?

### Objeto formal minimo

Para una tarea temporal \(\tau\):

- \(\mathcal G\): conjunto finito de grafos de transformacion permitidos;
- \(L_\tau(g)\): perdida fuera de muestra del grafo \(g\) bajo un aprendiz congelado;
- \(c_\tau(g)\): costo de evaluarlo;
- \(m_\tau\): descriptores disponibles antes de evaluar la prueba;
- \(H\): historia de tareas y evaluaciones anteriores;
- \(\pi(g\mid m_\tau,H)\): politica que recomienda la siguiente evaluacion o se abstiene.

La variable principal no es "conocimiento". Es el **arrepentimiento simple o costo hasta objetivo**, bajo presupuesto \(B\), respecto del mejor grafo evaluable para la tarea.

### Objetivos

1. Definir un espacio pequeno y tipado de grafos de transformacion, con validez temporal, costo y condiciones de aplicabilidad verificables.
2. Desarrollar un meta-selector calibrado que use caracteristicas de tarea y resultados parciales para ordenar evaluaciones y abstenerse bajo incertidumbre o cambio de distribucion.
3. Evaluar transferencia, costo y fallos en tareas no vistas, comparando con pipelines fijos y metodos de busqueda sin transferencia.

### Hipotesis

**H1 - eficiencia.** Bajo el mismo presupuesto de evaluaciones, el meta-selector reduce el arrepentimiento o el costo hasta objetivo frente a busqueda aleatoria, ASHA/Hyperband y un optimizador sin historia, sobre tareas completas no vistas.

**H2 - abstencion.** Una regla calibrada de abstencion reduce la frecuencia y magnitud de transferencia negativa frente al mismo selector obligado a recomendar, con cobertura reportada y un margen predeclarado.

**H3 - valor del grafo.** Bajo presupuesto y aprendiz downstream iguales, buscar grafos condicionados por tarea supera al mejor pipeline fijo y a una cadena serial comun; si no lo hace, la conclusion es que la complejidad modular no se justifica.

## 5. Diseno experimental minimo

### Banco

- **Calibracion sintetica:** procesos con ruido, retrasos, fuentes compartidas y cambios conocidos. Sirve para saber si el selector reconoce condiciones, no para probar utilidad real.
- **Primario publico:** una sola familia de problemas temporales con suficientes tareas independientes. La seleccion debe hacerse despues del censo de cobertura; no basta nombrar ETT y Weather.
- **Confirmatorio:** una familia distinta y no financiera. Finanzas puede ser aplicacion posterior, no la fuente exclusiva de evidencia.

La unidad de analisis es la tarea o familia dejada fuera, no la semilla ni una ventana tomada de la misma serie.

### Espacio candidato

No entran trece STEPS. Para el primer estudio: identidad, una familia de denoising causal, una representacion espectral trailing, una descomposicion comun/privada y una combinacion de dos ramas. Las demas piezas solo se incorporan tras evidencia propia.

### Comparadores obligatorios

- sin transformacion;
- mejor pipeline fijo aprendido solo en desarrollo;
- seleccion aleatoria bajo igual presupuesto;
- ASHA/Hyperband;
- BOHB o SMAC;
- meta-selector tabular sencillo;
- vecinos directos de AutoML/meta-learning para pipelines y series.

### Metricas

- mejor desempeño alcanzado frente a numero de evaluaciones y tiempo;
- costo hasta objetivo;
- arrepentimiento simple normalizado por tarea;
- frecuencia y magnitud de transferencia negativa;
- cobertura-riesgo de la abstencion;
- calibracion del ranking y del intervalo de incertidumbre;
- costo de construir los descriptores y el meta-dataset.

## 6. Disposicion de las doce compuertas

- **G1, G2, G3, G4, G6, G7, G9, G10, G11 y G12:** aceptadas.
- **G5:** aceptada como disciplina del work plan, pero `64:32:32:32` no pertenece a la propuesta doctoral; es una configuracion local, no una pregunta general.
- **G8:** debe endurecerse. Panel y local projection son el primer analisis; el carril completo sale del nucleo doctoral hasta que exista evidencia de identificacion y soporte.

Agrego cuatro compuertas:

1. **G13 - una decision:** el unico componente aprendido por la tesis es el meta-selector entre tareas; router por muestra y asignador dinamico quedan fuera.
2. **G14 - censo antes de promesa:** ningun L3 se promete sin una matriz tarea-pipeline cuantificada y con particion por tarea posible.
3. **G15 - novedad antes de titulo:** revision sistematica contra AutoML de pipelines, meta-learning para series, seleccion abstentiva y data curation bajo shift.
4. **G16 - eventos como confirmacion:** el carril de eventos no es una fuente coigual ni una hipotesis central de esta propuesta.

## 7. Estado recomendado

1. Conservar los cuatro documentos revisados como **mapa interno de investigacion**.
2. No tocar el PDF que se enviara de inmediato ni reemplazar la propuesta de seleccion de representaciones para RL por esta idea.
3. No implementar STEP 12, STEP 13 ni L3 todavia.
4. Ejecutar primero el banco CPU de STEP 03 porque ya esta definido y produce evidencia util independientemente de la tesis escogida.
5. En paralelo, hacer un censo de meta-datos y una revision sistematica corta. Solo esos dos resultados permiten decidir si esta idea merece convertirse en segunda propuesta formal.

## 8. Veredicto final

Como arquitectura del programa: **ACCEPT WITH THE EXISTING GATES PLUS G13-G16**.

Como propuesta doctoral actual: **REVISE MAYOR**. Presentarla hoy como "tres fuentes de conocimiento" permitiria que un jurado la destruya como una integracion de metodos conocidos sin una variable de decision comun. Recentrada en meta-seleccion abstentiva de grafos temporales, puede convertirse en una propuesta clara, pertinente para inteligencia artificial y mas cercana al procesamiento de datos que Harvey quiere investigar.

La nueva direccion merece trabajo. Todavia no merece un PDF.

## 9. Antecedentes que deben entrar en la revision de novedad

- Feurer et al., *Efficient and Robust Automated Machine Learning*, NeurIPS 2015: https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html
- Fusi et al., *Probabilistic Matrix Factorization for Automated Machine Learning*, NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/hash/b59a51a3c0bf9c5228fde841714f523a-Abstract.html
- Yang et al., *AutoML Pipeline Selection: Efficiently Navigating the Combinatorial Space*, KDD 2020: https://www.kdd.org/kdd2020/accepted-papers/view/automl-pipeline-selection-efficiently-navigating-the-combinatorial-space.html
- Montero-Manso et al., *FFORMA: Feature-based forecast model averaging*, IJF 2020: https://doi.org/10.1016/j.ijforecast.2019.02.011
- Lubba et al., *catch22: CAnonical Time-series CHaracteristics*, DMKD 2019: https://doi.org/10.1007/s10618-019-00647-x
- Navarro et al., *Meta-Learning for Fast Model Recommendation in Unsupervised Multivariate Time Series Anomaly Detection*, AutoML 2023: https://proceedings.mlr.press/v224/navarro23a.html
- Moeini et al., *Neural Architecture and Hyperparameter Selection Through Meta-Learning on Time Series*, AAAI 2026: https://doi.org/10.1609/aaai.v40i29.39622
- Taga et al., *Filter, Augment, Forecast: Online Data Selection for Robust Time Series Forecasting*, AISTATS 2026: https://proceedings.mlr.press/v300/taga26a.html
