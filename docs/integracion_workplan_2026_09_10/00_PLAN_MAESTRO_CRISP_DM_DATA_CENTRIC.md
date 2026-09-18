# Plan maestro CRISP-DM para el programa data-centric

**Corte del inventario historico:** 2026-09-12. **Secuencia corregida:** 2026-09-18.

**Alcance:** pronostico supervisado, seleccion de representaciones para RL y optimizacion con DOIN.
**Regla principal:** ningun modelo compensa una entrada mal definida. La unidad de trabajo inicial es la variable con su procedencia y disponibilidad temporal, no la arquitectura neuronal.

**Plan vinculante:** [master v3](https://github.com/harveybc/predictor/blob/master/docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md).
El estado 06 de 12-sep y los conteos historicos de este archivo no son el estado
actual de ejecucion. La orden vigente es RP1-RP8: piloto multivariado E0 de P-MOD,
con H2/H3 de desarrollo, y preparacion publica/negocio/RL en paralelo.
Un protocolo escrito, una implementacion, una ejecucion y una decision de
elegibilidad son estados distintos.

## 1. Que significa "cada variable de entrada"

No significa que toda columna almacenada deba llegar al modelo. Significa que toda variable candidata debe tener una identidad estable y pasar por una secuencia explicita de controles antes de competir.

El universo se construye con tres bancos separados:

1. **Banco financiero.** Variables crudas y derivadas del repositorio `financial-data`, incluidas las vistas materializadas que hoy alimentan a `predictor`, `gym-fx` y `agent-multi`.
2. **Banco publico de pronostico.** Series no financieras con procedencia y licencia comprobables. Sirve para decidir si un metodo se generaliza fuera del dominio en que fue concebido.
3. **Banco sintetico de mecanismos conocidos.** Senales cuyo componente limpio, ruido, eventos y cambios de regimen son conocidos por construccion. Sirve para calibrar diagnosticos; no reemplaza la confirmacion en datos publicos o financieros.

La misma columna repetida en dos cortes temporales no se cuenta como dos conceptos distintos. Se conserva una identidad logica de variable y se registran por separado sus apariciones fisicas, periodos y digests.

## 2. Inventario: reutilizar y reconciliar, no empezar de cero

El inventario historico de `financial-data` se conserva como antecedente, pero no es suficiente como autoridad actual: cubre cinco raices historicas y deja por fuera la mayor parte del arbol `features`.

La reconciliacion ejecutada en este ciclo encontro:

| Capa | Resultado actual | Interpretacion |
|---|---:|---|
| Manifiesto `financial-data/features` | 1,680 cortes fisicos presentes | 200 cortes de trading y 1,480 cortes por fuente/frecuencia |
| Columnas declaradas en esos cortes | 7,860 apariciones | No son 7,860 variables unicas; falta resolver equivalencias e identidad logica |
| Volumen fisico declarado | 14,436,534,039 bytes | Se verifico existencia y tamano, no se releyo todo el contenido |
| Perfil inmediato en `predictor` | 2 datasets, 26,541 filas, 99 columnas | ETHUSDT H4 y EURUSD 1h disponibles localmente |
| Estado de metadatos | incompleto | Faltan, segun fuente, unidades, licencia, contrato de disponibilidad o diccionario especifico |

La estrategia es incremental:

1. Reusar `MANIFEST.json`, archivos de procedencia, diccionarios y metadatos de entrenamiento.
2. Comparar digest, tamano y esquema contra el ultimo censo.
3. Reperfilar valores solo para archivos nuevos, modificados o seleccionados para una fase experimental.
4. Registrar de forma explicita `UNAVAILABLE` o `UNKNOWN`; nunca inferir unidades, licencia o disponibilidad a partir del nombre.
5. Emitir un recibo de cada reconciliacion y conservar el inventario anterior.

## 3. CRISP-DM aplicado al programa

### 3.1 Comprension del problema

Cada experimento debe declarar antes de mirar resultados:

- decision que pretende mejorar: pronostico, accion de trading, seleccion de representacion o asignacion de presupuesto;
- consumidor del resultado y horizonte de uso;
- costo de errores, abstenciones, latencia y computo;
- baselines sencillos que hacen inutil una mejora aparente;
- condiciones bajo las cuales un resultado negativo es informativo;
- dominio de uso y dominios en los que no se permite extrapolar.

Para pronostico, la pregunta no es solo "que modelo reduce el error", sino si una preparacion de datos reduce error fuera de muestra sin borrar extremos, anticipar el futuro o aumentar el costo de forma desproporcionada. Para RL, la pregunta es si una representacion mejora aprendizaje y decision bajo el mismo presupuesto, no si luce informativa en una prueba supervisada auxiliar.

### 3.2 Comprension de los datos

Por dataset y por variable se registran:

- fuente, licencia, unidad, frecuencia y zona horaria;
- tiempo de evento y tiempo de disponibilidad;
- cobertura, ausencias, duplicados, irregularidad temporal y valores no finitos;
- cardinalidad, constantes, rango y estadisticos robustos;
- cambios de distribucion y estabilidad por particion temporal;
- posible relacion con el objetivo, medida solo dentro de la particion permitida;
- digest de bytes, contrato de columnas y version de codigo usada para el perfil.

Las metricas de informacion y compresion son descriptores, no verdades sobre inteligencia o informacion libre de ruido. Se podran registrar longitud comprimida bajo un compresor fijado, entropia discreta bajo una cuantizacion declarada, complejidad de permutacion, rango efectivo correctamente definido, redundancia y estabilidad. Su utilidad se decide fuera de muestra.

### 3.3 Preparacion de datos

La preparacion ocurre en tres rejas distintas:

1. **Admisibilidad de variable cruda.** Procedencia, disponibilidad causal, unidad, calidad minima y rol conocidos.
2. **Elegibilidad de transformacion.** El operador respeta `fit/transform`, usa solo pasado, tiene paridad batch/incremental, costo medido y utilidad publica demostrada o estado experimental explicito.
3. **Seleccion de variables.** Solo entre salidas admisibles y elegibles, ajustada exclusivamente en entrenamiento y validada con divisiones temporales anidadas.

No se permite que un selector rescate una variable que fallo la primera reja ni que un gen de DOIN active una transformacion no licenciada.

### 3.4 Modelado

Los baselines diagnostican dificultad, no sustituyen al receptor adecuado de la
intervencion. P-MOD requiere ramas detector/integrador/adaptador, fusion temporal
y nucleo/cabezal reales. La capacidad, contexto, volumen y entrenamiento deben
permitir probar el mecanismo, con curvas y controles positivos pertinentes.

La seleccion de variables se ajusta en train/validacion anidados, pero H2 compara
asignacion de las mismas variables, no dos subconjuntos diferentes. E3/RL exige
su propio contrato economico/temporal; no depende de terminar el selector L2 o
DOIN, que son preguntas distintas. Los comparadores publicos de E1/E2 conservan
las arquitecturas y presupuestos declarados en la propuesta.

`predictor` se reactiva como banco supervisado reproducible, no como generador de features ni como sistema live. Sus plugins permiten comparar el mismo contrato de datos entre familias de modelos, pero se deben corregir o aislar los preprocessors historicos que ajustan informacion fuera de entrenamiento.

### 3.5 Evaluacion

La unidad estadistica debe ser una tarea, serie, origen o ventana causal declarada; una semilla no se presenta como unidad independiente. Cada conclusion debe incluir:

- estimando y contraste primario;
- intervalo de incertidumbre y tratamiento de multiplicidad;
- costos completos de perfilado, ajuste, seleccion y evaluacion;
- controles de capacidad y dimension cuando una transformacion agrega columnas;
- desempeno en extremos, cambios de regimen y datos faltantes;
- abstencion o resultado inconcluso cuando la evidencia no identifica una mejora;
- replica publica no financiera antes de una afirmacion general.

### 3.6 Puesta en uso

En este programa, desplegar primero significa publicar un artefacto reproducible y consumible por los demas repositorios. No significa activar trading.

Un resultado apto para consumo incluye:

- manifest de variables y operadores elegibles;
- digests de datos, codigo, particiones y resultados;
- contrato de entrada/salida y disponibilidad temporal;
- recibo de revision independiente;
- adaptador determinista para `predictor`, `agent-multi` o `doin-plugins`;
- nueva validacion financiera y live antes de conceder autoridad operativa.

## 4. Orden cientifico y operativo

**Correccion 18-sep:** I0-I10 se conserva abajo como mapa historico de actividades,
no como una cadena de bloqueos. La cola y dependencias vigentes son las del master
v3: E0-DEV multivariado, E1 publico, E0-CONF/E2 reservados y E3 forecast/RL obligatorio;
las otras propuestas conservan sus carriles. Reusar D0-D3 compatible, no repetirlo
como prerequisito de cada campana.

| Paso | Trabajo | Salida que abre el siguiente paso |
|---|---|---|
| I0 | Congelar decisiones, objetivos y roles temporales | Contrato CRISP-DM por experimento |
| I1 | Reconciliar los tres inventarios | Censo de datasets y variables con huecos nombrados |
| I2 | Perfilar calidad, temporalidad e informacion | Ledger por variable y por particion |
| I3 | Calibrar diagnosticos en sinteticamente conocido | Metricas interpretables y limites de deteccion |
| I4 | Evaluar transformaciones en banco publico | Lista revisada de operadores elegibles |
| I5 | Seleccionar variables con validacion temporal anidada | Manifest de variables congeladas y controles |
| I6 | Ejecutar pronostico supervisado en `predictor` | Evidencia de transferencia entre tareas y modelos |
| I7 | Construir grupos y representaciones modulares | Universo L1 congelado |
| I8 | Optimizar L2 con DOIN | Candidatos bajo presupuesto y procedencia completa |
| I9 | Evaluar RL y trading offline | Resultado mecanico/economico atribuible |
| I10 | Revalidar en dominio financiero y live | Elegibilidad operativa separada de la cientifica |

La seleccion exige caracterizacion y controles temporales de sus entradas. No
confundir admisibilidad de datos, aptitud mecanica de un operador experimental y
licencia para aplicarlo fuera del laboratorio. Estudiar un operador en DEVELOPMENT
no exige demostrar su utilidad antes de experimentar; tampoco lo autoriza para
produccion. Los objetivos de una tarea se definen antes de evaluar sus resultados.

La reja historica `df_consumption_gate.py` conserva su alcance: ningun cambio de
prosa la desactiva ni fabrica sus records. RP1-RP8 debe usar una ruta DEVELOPMENT
explicita y probada para el banco sintetico del experimento, con datos admisibles,
sin atribuir `PUBLICLY_ELIGIBLE` a sus salidas. Si falta esa ruta, implementarla
con pruebas en el contrato correspondiente, no eliminar la reja.

## 5. Relacion con el trabajo ya realizado

- **T0/T1:** se conservan como evidencia de contrato causal y calibracion sintetica. No se reinterpretan retroactivamente.
- **T2:** su banco publico y sus resultados pertenecen a la reja I4. La adjudicacion existente se revisa con su identidad original; no se mezcla con el nuevo inventario.
- **M3/M4:** las mediciones de capacidad, complejidad y aprendizaje se incorporan como diagnosticos experimentales en I2-I3. Un descriptor que no demuestre utilidad incremental se retira.
- **B4 y campañas RL:** se adjudican bajo el contrato con que fueron ejecutadas. Los hallazgos nuevos solo gobiernan campañas sucesoras.
- **Planes de 13 pasos:** se conservan como catalogo de familias de operadores. Cada paso debe demostrar causalidad, utilidad y costo antes de entrar al flujo; el numero del paso no confiere elegibilidad.

## 6. OLAP: memoria del programa, no deposito indiscriminado

El cubo se amplia de forma aditiva con seis granos nuevos:

- dataset, serie de panel y variable;
- perfil de inventario por dataset;
- perfil por variable;
- metricas de informacion por particion;
- trayectoria por epoca o checkpoint;
- recibos de ingesta; los recibos de adjudicacion se agregan al conectar las campanas externas.

Los resultados viejos no se borran. Se marcan con cobertura de metadatos y, cuando no sea posible reconstruir un campo, queda `UNAVAILABLE`. La migracion correcta es: base desechable, validacion de esquema e ingesta, respaldo de la base real, migracion aditiva, backfill comprobable y recibo final. `reset_olap.py` no forma parte de este plan.

## 7. Entregables doctorales acumulativos

P-MOD: grupos/contextos por datos, arquitectura con secuencias, E0 H2/H3,
E1 reglas y regimenes R0/R1/R2, E2 H1 publico y E3/RL independiente obligatorio.
P-L2: fidelidades del mismo candidato, riesgo/cobertura, costo/regret y comparadores
multifidelidad; esos entregables no sustituyen los de P-MOD.
P-CAP: memorizacion/generalizacion/dimensionamiento. P-PRE y P-TRN: ruido,
preservacion, transferencia y abstencion, con sus propios contrastes. P-INC:
asignacion financiada y aprendizaje de nodos en dominios simulados.

El master v3 traza experimentos, metricas y fuentes de los seis objetos. Se comparte
infraestructura, no conclusiones por semejanza de nombres. Cada tarea informa
exactamente que hipotesis sirve y con que alcance; los tests tecnicos son controles
internos, no reemplazo del experimento doctoral.

## 8. Estado historico de esta iteracion (12-sep)

Completado o demostrado:

- censo incremental: 1,680 apariciones fisicas y 1,965 variables conceptuales;
- reja de consumo antes de modelado y outbox OLAP durable;
- caracterizacion numerica historica con alcance fisico/estadistico;
- un sucesor ETH con lineage y contrato temporal propios, aceptado solo como
  piloto mecanico;
- T2 cerrado con resultado `DOES_NOT_ADVANCE` para el operador evaluado;
- diseno por variable v5 con join miembro a miembro;
- resultado real de poblacion: cero variables elegibles.

No completado:

- semantica, unidad, licencia y disponibilidad de la poblacion financiera;
- banco publico multivariado listo para experimentos;
- perfil por variable y por grupo de muestreo, informacion y ruido;
- denoising causal por variable con retardo y preservacion medidos;
- ejecucion de STEP 04-13;
- seleccion de variables, pronostico, representaciones, L2/DOIN y RL bajo el
  contrato nuevo.

La frase historica "la orden siguiente es D0-D2" queda sustituida por RP1-RP8.
Los conteos y estados anteriores se conservan para trazabilidad y no deben usarse
para afirmar que ningun STEP se ha ejecutado desde entonces. Consultar el master v3
y los recibos del run especifico para determinar que evidencia puede reutilizarse.
