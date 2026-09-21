# Plan maestro v3: programa doctoral y negocio data-centric

Actualizado: 2026-09-20. Responsable del programa y revision: Musashi.
Ejecucion delegada: Satoshi. Estado: E0-DEV, etapa ARCH y piloto E1 EJECUTADOS;
piloto historico no gobernado y SUCESOR E1 EJECUTADO; sin confirmacion.

**Vigente:** [RP57-RP64](../handoffs/MUSASHI_PROGRAM_RP57_RP64_2026_09_20.md).
[Revision ML y comparadores](../audits/work_plan/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md):
el sucesor SI se ejecuto en RP55. Musashi recomputo nueve fits sobre 10 020
origenes conservados, con naive/ridge sobre filas iguales. R0 MAE 0.546929 kW
vs persistencia 0.617372 (+11.41%); ridge 0.545495 (+11.64%).
Evaluados en log1p(kW), R0 0.266344 vs naive 0.279127 (+4.58%).
No son porcentajes de aciertos ni replicas de un entrenamiento logaritmico.
El cociente historico usa persistencia h60, no MASE convencional m1.
Corregir presentacion/etiqueta sin cambiar la propuesta; diagnosticar datos,
objetivo, stopping, contexto, volumen y arquitectura antes de repetir R0/R1/R2.
TCN propia adaptada no equivale a replica del articulo; referencia independiente
obligatoria. No asumir que mas epocas mejoran validacion.
RP56 declara cierre gobernado; esta revision verifica arrays locales, no reemplaza
la auditoria completa de adopcion/warehouse. Historicos conservados, sin ganador
universal ni confirmacion.
No hay nueva decision del owner para el desarrollo autorizado. No repetir E0.
Antecedentes financieros: el owner identifica una mejora valida en fase 3 y
un resultado anterior afectado por fuga wavelet. RP58 debe separar ambos por
linaje. El TCN NEAT antes citado es CAUSALITY_UNVERIFIED, no benchmark aceptado.
[13D](program_v3/13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md)
resuelve el escenario de simulacion: capital real y exposicion no bloquean DEV.

**Invariante de E1:** R0 = detector aleatorio entrenable; R1 = detector
preentrenado congelado; R2 = los mismos pesos iniciales preentrenados ajustables.
El resto del modelo se entrena en los tres. Crudo/agrupado/fusion y arquitectura
son factores separados; nunca redefinen estos codigos. El checker documental
verifica el contrato; gradientes/estados/pesos verifican su ejecucion.

**Invariante de ejecucion:** nuevos experimentos registran campana y entrega
ANTES de preparar/fitear; terminal local no reemplaza contabilidad y cubo.
Importacion historica es evidencia retrospectiva separada, nunca una campana
prospectiva reconstruida despues. Registrar recursos es trabajo tecnico del
ejecutor por el procedimiento existente, no una nueva decision del owner.

**Antecedente aprobado, 18-sep:** comparar ARCH-A (Conv1D local, referencia), ARCH-B
(convolucion dilatada/TCN), ARCH-C (Conv1D + GRU/LSTM) y control sin extractor
aprendido dentro de E0 antes de elegir el procedimiento. Hipotesis
[H-CORE](CORE_PRETRAINING_HYPOTHESIS_2026_09_18.md)
incorporada despues del desarrollo del extractor y su prefijo congelado verificable.
Retorno RP1-RP8 recibido en `2347c7b`; esta adicion se adopta en el sucesor,
sin cambiar la corrida sellada. [Revision ejecutada y limites](../audits/work_plan/MUSASHI_RP1_RP8_REVIEW_2026_09_18.md):
69 records sin discrepancia en la comprobacion local de datos/errores, pero cierre
incompleto, descriptor de tendencia incorrecto y atribucion H3 limitada.
No borrar el piloto ni repetirlo entero: corregir, demostrar reuso y ejecutar
[RP9-RP16](../handoffs/MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md).
La correccion de tendencia mantiene las 15 particiones del piloto en el reanalisis.
Ese era el estado al emitir RP9-RP16. La etapa informativa ARCH-A/B/C/0 ya se
ejecuto; su lectura queda corregida por el dictamen de 19-sep. H-CORE sigue
despues de E1 y del prefijo verificable.

## 1. Autoridad, alcance y correccion de rumbo

Este es el indice operativo vigente. Sustituye la secuencia inmediata del master
v2, del plan CRISP-DM de 12-sep y de las ordenes sinusoidales A/C de 18-sep.
No cambia las hipotesis de las propuestas, no reabre reservas, no borra resultados
y no concede aprobacion cientifica a una implementacion por tener tests verdes.
Si una propuesta cambia, actualizar primero su trazabilidad aqui.

La fuente principal es la [propuesta modular editable](../propuesta_doctoral_representaciones_temporales_modulares.tex),
especialmente las secciones de procedimiento, experimentos y contrastes; su
[PDF](../propuesta_doctoral_representaciones_temporales_modulares.pdf) conserva su contenido.
Se comprobaron identicos los bytes de la fuente en la rama de trabajo y master.
El [inventario previo](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_09/02_INVENTARIO_PROPUESTAS.md)
contiene **seis objetos cientificos**, no cinco. Se organizan en cinco frentes
operativos porque senales contiene P-PRE y P-TRN, pero sus contrastes NO se fusionan.
P-3F es arquitectura compartida, no una septima tesis.

La revision abarca el master completo, los tres patches, el plan CRISP-DM, la
integracion jerarquica, los protocolos de las propuestas y el estado 12C/12D/12E.
Los documentos extensos de cada STEP siguen como especificaciones subordinadas;
su literatura y cada implementacion no quedan certificados por esta revision.

Hallazgos corregidos:

1. La campana C1-C6 de seno univariado no contrasta agrupacion ni fusion. Se retira
   como siguiente campana; sus comprobaciones compatibles se reutilizan como tests.
2. E0 de P-MOD es un bloque multivariado de desarrollo y posterior reserva H2/H3,
   no una prueba general de que una red aprende senos.
3. El plan CRISP-DM mezclaba entregables P-L2 con P-MOD y condicionaba RL a una
   cadena completa I0-I10. Se reemplaza por dependencias cientificas por experimento.
4. La cobertura D3 de operadores no cubre feature engineering, redes modulares,
   seleccion de variables ni utilidad de trading. Los estados se separan.
5. El plan I-INFO contenia cotas falsas de informacion sin ruido y un rango efectivo
   mal nombrado. Rige el [contrato corregido de metricas](PROGRAM_METRICS_CONTRACT_v1.md).

## 2. Trazabilidad de todas las propuestas

| ID / frente | Fuente y pregunta | Experimentos obligatorios y comparadores | Metricas y entregable |
|---|---|---|---|
| P-MOD / modular | [Fuente](../propuesta_doctoral_representaciones_temporales_modulares.tex): perfiles, grupos, contextos y fusion para pronostico y RL | E0-DEV mecanismos; E1 familias publicas de desarrollo; E0-CONF H2/H3 y E2 H1 reservados; E3 aplicacion financiera/RL independiente | MASE y efectos H1/H2/H3, incertidumbre y costo; procedimiento fijado, tablas/figuras por tarea, comparacion RL obligatoria |
| P-L2 / seleccion de memoria RL | [Fuente](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex): cuando comprar curvas parciales y abstenerse | Piloto, desarrollo, calibracion y confirmacion por entorno base; POPGym y cambio de contexto CARL; PPO y cinco codificadores; referencias exhaustiva, fija, aleatoria, multifidelidad y decision forzada segun protocolo | H1 costo/regret, H2 riesgo-cobertura, H3 transferencia; pasos de entorno reales, costo inicial/amortizado y abstenciones; no usar una sonda de forecasting como resultado RL |
| P-CAP / capacidad y dimensionamiento | [Fuente](02_memorizacion_generalizacion_dimensionamiento.md): memorizar, generalizar y estimar tamano suficiente | H1 etiquetas aleatorias/mesetas por familia-tamano-precision; H2 reglas conocidas y log-loss; H3 prediccion de N_min frente a curvas, parametros y proxies; MLP principal y GRU temporal secundaria | C_mem operacional, perdidas log2, N_min en rejilla con censura, error de dimensionamiento y costo; M3/M4 anteriores solo si su contrato coincide |
| P-PRE / senales | [Fuente](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/propuesta_doctoral_preprocesamiento_informacional.md): ruido plantado, denoising y preservacion | H1 degradacion por perturbacion, H2 tratamiento causal segun regimen, H3 conservar residuo; raw / D / D+residuo, receptores adecuados y controles de capacidad; sintetico y publico separados | Error de tarea, sesgo/calibracion SNR, extremos, retardo y disponibilidad, costo; nunca equiparar residuo con ruido real |
| P-TRN / senales | [Matriz](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tesis_transformaciones_temporales/01_MATRIZ_HIPOTESIS_EXPERIMENTOS.md): utilidad/transferencia de un grafo pequeno con abstencion | T0-T5 / E0-E8: contrato, calibracion, preservacion, global vs por variable, utilidad publica, seleccion bajo presupuesto, abstencion, familias reservadas, paridad DOIN; identidad, fijo/aleatorio y busqueda sin transferencia | H1 utilidad condicional, H2 transferencia/costo, H3 abstencion; curvas riesgo-cobertura y preservacion. Mantener signo de efecto propio del protocolo |
| P-INC / incentivos | [Fuente](04_incentivos_red_descentralizada_multidominio.html): asignacion con demanda financiada y nodos adaptativos | H1 regla uniforme/estatica/reactiva/propuesta; H2 nodos fijos/aleatorios/codiciosos/aprendidos; H3 desviaciones rentables o limites; simulador HPO-B y segundo dominio obligatorio de entrenamiento real compacto | Demanda financiada no atendida, desalineacion L1, utilidad de desviaciones, arrepentimiento, HHI y costo; unidad campana, Holm; sin pagos reales ni dependencia de desplegar DOIN |

Fuentes complementarias conservadas: [master v2](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md),
[patch 001](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_001_HISTORICAL_CHAIN_AND_COMPRESSION_EXTENSIONS.md),
[patch 002](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_002_AGENT_AUDIT_CONSTRAINTS.md),
[patch 003](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_003_FINAL_REVIEW_CAUSAL_EVENTS_METAOPT.md),
[integracion jerarquica](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_09/04_PLAN_INTEGRACION_JERARQUICO.md).
No reabrir la fusion de tesis rechazada ni sustituir P-MOD por P-L2.

## 3. Donde estamos realmente

La siguiente tabla resume evidencia registrada en las ramas, NO una auditoria
de procesos vivos ni una nueva conciliacion del cubo realizada en esta revision.
Antes de reutilizar un resultado, comprobar sus artefactos, poblacion y contrato.

| Trabajo | Evidencia disponible | Lo que NO establece |
|---|---|---|
| D0-D2 | Perfilado reportado de 715 datasets; calibraciones condicionadas a regimen, 47+6 resultados candidatos y 39 calibraciones SNR tras correcciones | No caracterizacion semantica completa de toda variable, ni denoising optimo por entrada, ni informacion sin ruido conocida en mercado |
| D3 mecanica | v1-v3, 511 unidades / 6390 celdas; estados de causalidad, disponibilidad y sensibilidad de sondas | No utilidad de las representaciones ni aceptacion automatica de variantes/configuraciones nuevas |
| Utilidad anterior | utildev-v1: 18 contrastes sin avance, 18 descriptivos, cero propuestas; ridge W4, incremento observado, tres operadores, escenario limitado | No comparacion de redes modulares, no refutacion general del preproceso, no resultado de trading |
| Adecuacion neuronal | Implementaciones y pilotos de costo; 12D/T sin factorial completo; control C1-C6 retirado como campana siguiente | No validacion del extractor modular ni suficiencia del volumen/contexto |
| Gobernanza/OLAP | Rutas data-gov -> lake/proveedores -> consumidores -> warehouse DuckDB, outbox y herramientas de conciliacion | No causalidad por tener hash; no correccion ML por tener terminal COMPLETED |
| E0/E1/E2/E3 modular | Propuesta y requisitos escritos; primera comparacion H2/H3 no demostrada por lo anterior | Ninguna hipotesis doctoral modular queda ya confirmada |

[Mapa de cobertura 12C](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_10/12C_REPRESENTATION_COVERAGE_MAP_2026_09_18.md)
y [protocolo semanal/RL 12E](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_10/12E_WEEKLY_BUSINESS_PROTOCOL_AND_RL_COUNTERPART_2026_09_18.md)
delimitan las brechas. No convertir un estado antiguo "no ejecutado" en un estado
actual, ni una mecanica ejecutada en hipotesis probada.

## 4. Primer experimento real: MOD-E0-DEV

**Pregunta:** al variar heterogeneidad temporal y relaciones retardadas entre
variables, que aportan agrupar por perfiles y conservar secuencias hasta la fusion,
bajo informacion, entrenamiento y recursos comparables?

Es el piloto de mecanismos de E0, ya previsto en la propuesta, NO su confirmacion
reservada. Su resultado sera una tabla de efectos y limites del procedimiento,
mas parametros de costo/precision para E1 y las reservas. Puede encontrar ausencia
de beneficio: eso sigue siendo un resultado del proyecto si los controles son
informativos y el experimento no confunde falta de aprendizaje con falta de efecto.

| Aspecto | Orden sinusoidal retirada | MOD-E0-DEV |
|---|---|---|
| Datos | Una variable, un periodo, amplitudes/fases diversas | Varias variables y grupos temporales; periodicidad, persistencia, tendencia/eventos; pares con/sin dependencia retardada |
| Intervencion | Volumen de ventanas con predictor fijo | H2: asignacion por perfiles vs aleatoria; H3: fusion temporal vs resumen temprano |
| Modelo | Ridge y Conv1D sobre una ventana | Detector -> integrador -> adaptacion por grupo -> fusion -> nucleo -> cabezal; controles que cambian solo la intervencion |
| Conocimiento adquirido | Adecuacion de esa implementacion en una rejilla de senos | Respuesta inicial sobre los mecanismos que justifican la arquitectura doctoral |
| Infraestructura | Motivo central del ensayo | Controles internos reutilizados; se corrigen solo fallos observados |
| Limite | Ni agrupacion ni RL ni mercado | Desarrollo sintetico multivariado, todavia no H1 publico ni beneficio economico |

### 4.1 Banco y variables del experimento

- H2: niveles predefinidos de diferencias de periodicidad o persistencia; mantener
  muestreo, numero de variables, volumen y distribucion del ruido. Cambiar amplitud
  no sustituye a heterogeneidad temporal. Grupos latentes solo para diagnosticar,
  nunca suministrados al agrupador. Perfiles calculados solo en entrenamiento.
- H3: pares con/sin relaciones entre variables con retraso. Conservar en distribucion
  los procesos marginales, comprobarlo en replicas y no usar un simple shuffle
  temporal como unico control. Demostrar que la dependencia cambia la informacion
  disponible para el objetivo; correlacion cruzada sola no basta.
- Periodos, retardos, horizonte y observacion se derivan de la pregunta del generador
  y de su escala muestreada. No heredar P=8/16/32/41, W=4/17 ni n=2048 porque ya
  existen. La propuesta no prescribe esos numeros: Satoshi debe entregar el calculo
  y las alternativas antes de medir, usando entrenamiento/desarrollo exclusivamente.
- Registrar numero de trayectorias/configuraciones independientes, valores unicos,
  ventanas, ciclos utiles y filas tras calentamiento/purga, sin intercambiarlos.
  Una rejilla determinista no produce replicas independientes por cambiar nombres.
- Separar datos de desarrollo, validacion del ajuste y reserva cientifica. Los
  controles limpios deterministas con MASE indefinido quedan como diagnosticos MAE;
  no agregar ruido ni epsilon despues de ver resultados para arreglar el denominador.

### 4.2 Aprendices, agrupacion y contrastes

**Comparacion aprobada para desarrollo E0:** ARCH-A como referencia inicial
(Conv1D causal de una/dos capas e integrador identidad), frente a ARCH-B
(bloque causal dilatado tipo TCN) y ARCH-C (Conv1D + GRU o LSTM con secuencias).
Anadir ARCH-0 sin extractor aprendido con adaptacion dimensional si hace falta.
Todas usan el modular con fusion y cabezal reales; no elegir ganador por facilidad
de implementacion. Controlar nucleo/cabezal, acceso a informacion y oportunidades
de ajuste; comparar efectos por arquitectura/regimen y costo total. Pocas capas
no garantizan contexto suficiente: calcular alcance rama+nucleo. Ver el documento
H-CORE, seccion 2, para controles, criterios y adopcion despues del retorno de Satoshi.

- Agrupacion: ACF, bandas Welch y fuerza de tendencia/estacionalidad; escalado de
  descriptores y exclusion de constantes en desarrollo; enlace promedio/distancia
  euclidiana segun propuesta. Estabilidad por segmentos. Feature selection y
  agrupacion son decisiones diferentes: no retirar variables entre brazos H2.
- H2: mismos numero/tamano de ramas, arquitecturas, contextos, informacion,
  preproceso comun, inicializaciones emparejadas y oportunidades de ajuste. Varias
  asignaciones aleatorias predefinidas, no permutaciones que solo renombren modulos.
- H3: las dos alternativas reciben **las mismas activaciones de un extractor
  entrenado y congelado**. Solo combinacion y cabezales se ajustan. Conservar el
  eje temporal en el brazo secuencial; definir exactamente el resumen temprano.
  Igualar capacidad de los controles dentro de tolerancia registrada y medir costo.
- R0 es la base de desarrollo desde cero; R1/R2 y preentrenamiento se comparan en
  desarrollo E1, no se presuponen mejores. En H3, congelar el extractor despues de
  su entrenamiento comun no implica haber elegido R1 para H1. Verificar gradientes,
  pesos y estados de normalizacion de los componentes que deben permanecer fijos.
- El grafo real debe corresponder al contrato; un Flatten anterior a la fusion no
  implementa el brazo de secuencias de H3. Existe tal helper en
  `predictor_plugins/common/utils.py:build_branch`; eso obliga a trazar la ruta
  usada, no demuestra que todos los plugins pasen por ella. Reusar plugins cuando
  cumplen, implementar solo el componente faltante cuando no.
- Persistencia, referencia estacional, predictor estadistico pertinente y oraculo
  causal del generador diagnostican dificultad/suelo. El baseline lineal no es el
  unico receptor del tratamiento. Verificar aprendizaje de un control positivo
  pertinente al mecanismo con el modelo neural real y el mismo soporte disponible.

### 4.3 Medidas e interpretacion

H2: `e(h)=MASE(perfiles,h)-MASE(aleatorio,h)`; pendiente de e respecto a h.
H3: `d_r=MASE(secuencias,r)-MASE(resumen,r)`; `gamma=d_1-d_0`.
Negativo favorece el metodo. Guardar errores crudos por objetivo/origen y el
denominador train compartido. MAE/MSE/RMSE complementan, no sustituyen el estimando.

El piloto informa curvas, efectos descriptivos, variabilidad, precision alcanzada
y costo; **no** emite apoyo confirmatorio a H2/H3. Antes de la reserva, E1 fija
margen relevante y plan de precision; el protocolo principal usa F+4 intervalos
Bonferroni y bootstrap jerarquico con unidades emparejadas. Semillas, ventanas y
origenes no inflan el numero de tareas independientes.

Guardar tambien grupos encontrados, estabilidad, activaciones/forma temporal,
retardos de disponibilidad, soporte usado, parametros, updates observados, curvas
train/val, checkpoint seleccionado, predicciones recargadas, pico de memoria y
costo completo. Early stopping restaura el mejor checkpoint de validacion: no
demuestra por si solo ausencia de sobreajuste ni selecciona por el test.

## 5. Cola y dependencias, sin cadena artificial

| Orden | Trabajo / responsable ejecutor Satoshi | Dependencia y salida |
|---|---|---|
| Ejecutado, revision parcial | Piloto multivariado RP1-RP8 | 66 celdas y 3 pilotos; conservar efectos descriptivos, no aceptacion confirmatoria |
| Ahora A | RP57-RP59: comparadores, antecedentes antiguos y auditoria de datos | Naive/lineal/modelo por filas y escalas identicas; no inferir log1p ni unidades por magnitud |
| Ahora B | RP60-RP63: diagnostico ML y referencia de literatura | Loss/monitor, curvas, capacidad, contexto y volumen por separado; ejecutar fase acotada antes de repetir pretraining |
| E0, antes de elegir | MOD-ARCH-COMPARE: ARCH-A/B/C y ARCH-0 en el modular propio | Efectos descriptivos recalculados; aceptacion compuesta pendiente de RP26, sin ganador universal |
| Paralelo C | Aceptacion RL offline de RP53 conservada para revision | RP56 reporta reparacion de terminales sin fill; no aceptacion cientifica ni entrenamiento RL por ese hecho |
| Despues D | E1 publico: desarrollar perfiles, seleccion, contextos, R0/R1/R2 y comparadores | Piloto de mecanismos y datos admisibles; fijar procedimiento, margenes, precision y presupuestos antes de reservas |
| Despues D, extension | MOD-FROZEN-PREFIX y MOD-CORE-PRETRAIN: hipotesis H-CORE del owner | Comparacion R0/R1/R2 y receptor adecuado completados; fijar/materializar prefijo; nucleo desde cero vs preentrenado ajustable vs congelado, con cabezales y costo total |
| Despues E | E0-CONF H2/H3 y E2 H1 en reservas independientes | Reglas E1 fijadas y revision ML; no reutilizar test de diagnosticos anteriores |
| Obligatorio F | E3 forecasting aplicado y RL con reentrenamiento semanal | Datos temporalmente aptos, entorno/modelo y contrato de ejecucion validados; evaluacion propia aunque H1-H3 no sean positivos |
| Carriles propios | P-PRE/P-TRN, P-CAP, P-L2, P-INC | Sus protocolos, modelos y unidades; compartir datos/metricas/codigo solo si equivalencia demostrada, no multiplicar claims sobre el mismo resultado |

H-CORE es un experimento de desarrollo propio, no requisito para E0 ni aprobacion
presupuesta del preentrenamiento. No modifica H1-H3 o sus pruebas reservadas sin
una revision previa a la confirmacion. Su comparacion RL pertenece a E3; un resultado
de reconstruccion o forecasting no la reemplaza. No se pierde si R0 supera R1/R2.

No hacer depender E0-DEV de que todas las transformaciones ganen previamente,
de terminar DOIN, de Metabase o de resolver datos financieros que no consume.
Un operador experimental causal puede compararse en DEVELOPMENT con esa etiqueta;
no obtiene licencia publica/financiera por participar. Mantener controles de datos
y clasificacion reales: este plan no convierte un rechazo de runtime en permiso.

La aplicacion semanal debe fijar demanda desde las configuraciones ejecutables,
horas de corte/publicacion/decision/ejecucion, horizonte, historiales, actualizacion
semanal, objetivos y disponibilidad por variable. No elegir ETH porque haya un CSV.
Desconocidos de adquisicion no se sustituyen por hora de cierre de barra.

## 6. Cobertura completa de senales y representaciones

Cada fila es una pregunta experimental, NO una transformacion obligatoria en
cascada. El [master v2 y sus STEPs](https://github.com/harveybc/predictor/tree/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista)
conservan el detalle. Las celdas sin utilidad medida siguen pendientes.

| ID | Experimento/control requerido | Registro/encaje |
|---|---|---|
| STEP-01 | Resolucion, operador de observacion y aliasing; comparar igual intervalo fisico antes de ampliar historia | Eventos vs barras agregadas, disponibilidad; E0/E1/E3 |
| STEP-02 | Identificabilidad/calibracion SNR bajo perturbacion conocida vs estimadores e identidad | Error/sesgo por regimen y abstencion; P-PRE/T0-T1 |
| STEP-03 | Denoising global vs por variable, raw/D/raw+D+residuo y capacidad igualada | Utilidad, extremos, retardo, missingness; P-PRE/P-TRN |
| STEP-04 | Resolucion uniforme vs por distribucion/variable, companding; raw vs cuantizado vs aumentado | Distorsion, ocupacion, bits nominales/empiricos y error; no bits = capacidad neuronal |
| STEP-05 | Fuente/contextos/diccionarios, MDL y longitudes con coder fijado | Costo de descriptor y codificacion, no K ni fuente libre de ruido |
| STEP-06 | Amplitud/fase/frecuencia/tiempo-frecuencia vs raw bajo informacion igual | Disponibilidad, fase, bordes, reconstruccion/prefijos; wavelet causal real |
| STEP-07 | Detectores Conv1D/TCN/recurrentes/atencion y patrones locales/largos frente a referencias | Grafo, campo receptivo efectivo, curvas y adecuacion; P-MOD |
| STEP-08 | Canal o dominio definido, correccion/equalizacion vs identidad | Distorsion/transferencia y utilidad; normalizar no demuestra equalizacion |
| STEP-09 | Factor comun/unico y relaciones retardadas vs raw, conservando X/C/U | No suponer compartido = ruido; utilidad condicional; grupos/fusion |
| STEP-10 | Alineacion fija vs recuperacion temporal dinamica con tiempos de emision | Error de retardo, costo y ausencia de futuro; eventos asincronos |
| STEP-11 | Robustez con mascaras/corrupcion y codificador reutilizado vs sin preentrenar | R0/R1/R2, reconstruccion auxiliar y tarea; no nuevo AE obligatorio |
| STEP-12 | Enrutamiento adaptativo por calidad vs fijo/aleatorio, missingness y abstencion | Utilidad neta, estabilidad, cambios y costo; no oracle gating |
| STEP-13 | Ancho/compute por rama igual/fijo/marginal, interacciones y ablacion de ramas | Frontera error/costo, informacion igualada; no independencia MIMO supuesta |

Compresion permanece transversal: **C1** sparse (06/07), **C2** sparse convolucional
(07), **C3** refinamiento progresivo (06/12), **C4** informacion lateral (05/09/13),
**C5** residuo jerarquico (03/05/06), **C6** tasa-distorsion latente (extractor/nucleo,
despues de controlar la representacion), **C7** duracion/eventos (05/07).
Medir tarea, preservacion, tasa/descriptor y costo; ninguna es un compresor de pesos
que por si solo estime memorizacion. No se ejecutan las siete de golpe.

**FE:** feature-eng tiene un carril propio dentro de 06/07: calendario cuando el
dominio lo justifica, rezagos, estadisticas trailing, indicadores y features de
eventos, comparados con raw y controles de informacion/capacidad. Cada plugin
versiona config, dependencias, fit, disponibilidad y salidas; seleccion anidada
train/val con estabilidad, nunca filtros ajustados al test. El inventario de
plugins no demuestra que se hayan evaluado exhaustivamente.

**EVENT:** procesos irregulares, observacion agregada y tiempo de disponibilidad
se tratan como contratos propios, no interpolacion que inventa muestras.
**META:** selector de pipeline/metaoptimizacion aprende de filas de experimentos
con campanas retenidas, costo incluido; se difiere hasta existir resultados
admisibles suficientes. No usar los mismos runs para ajustar y afirmar transferencia.

## 7. Negocio, RL y modelos de referencia

Forecasting publico E1/E2 conserva las referencias de la propuesta: ingenuos,
estadistico multivariado pertinente, DLinear, PatchTST, iTransformer, DUET y uno
entre MTST/Pathformer/TimeMixer elegido antes de resultados de desarrollo. No
reducir sus arquitecturas para igualar parametros; esa restriccion aplica a los
controles internos con tolerancia declarada. Presupuesto/oportunidad de tuning
comparable y coste completo para todos. Agrupacion adicional segun protocolo.

RL E3 es obligatorio, no un apendice opcional ni una microcorrida de transporte.
Usar el agente/entorno de las configs ejecutables, historial causal, reset/estado,
reward, acciones, restricciones, capital, fees, slippage, funding si aplica,
tiempo de ejecucion y politica de ordenes. Comparar raw vs modular con la misma
informacion y entrenamiento semanal: train -> validacion interna -> semana
futura intacta. Separar mejora por mas datos/resoluciones de mejora arquitectonica.

Entregables E3: retorno neto, drawdown, Sharpe con supuestos, turnover, exposicion,
incertidumbre por semanas/regimenes, pasos observados y tiempo total de reentrenar.
Comparar politicas triviales y un control no modular competente. Cabezas de
volatilidad/pronostico/intervalos requieren calibracion y predicciones fuera de
muestra antes de alimentar al agente; no labels verdaderos ni futuro como input.
Paper/live requiere validacion operativa separada; esta ronda no envia ordenes reales.

P-L2 tiene ademas su propio banco publico RL y contraste de seleccion; no esperar
a que E3 financiero sea viable para desarrollarlo. P-INC puede simularse sin DOIN
operativo. P-CAP no se satisface contando parametros del extractor de P-MOD.

## 8. Disciplina ML y trazabilidad utilizadas, no decorativas

Antes de cada campana, requisito -> test de aceptacion -> diseno de sistema ->
componentes/integracion -> tests unitarios; despues implementacion y verificacion
de abajo arriba. Reutilizar tests existentes por ruta productiva, no reescribir
una simulacion del callback para probar el propio test.

La ficha numerica obligatoria contiene: pregunta/estimando; observacion/target;
generador o fuentes; muestreo/periodos/retardos; soporte disponible a cada decision;
N bruto/util e independiente; contexto/campo receptivo/horizonte; capas, unidades,
activaciones/parametros; escalado/preproceso; optimizer/lr/batch/presupuesto;
early stopping y recarga; curva de volumen/convergencia; controles positivos,
nulos y adversos; incertidumbre/multiplicidad; costo. Cada numero lleva origen,
supuestos y sensibilidad. No se heredan defaults sin esa ficha.

Wavelet/STL/rolling se prueban con prefijos, futuros alterados, padding/bordes,
missingness, tiempo de emision y restart, en el camino realmente consumido. Un
gemelo no causal debe ser detectado en comparaciones sensibles; cero emisiones
no es pase. Una ventana causal no vuelve causal un fit o reconstruccion global.

Datos, roles, tiempos, labels y grafo ejecutado deben llegar a un lector verificable.
Todos los intentos (incluidos fallos/inconclusos) pasan por data-gov y terminan en
DuckDB mediante el warehouse. Comparar poblacion y contenido contra contabilidad
independiente; conservar predicciones/labels para rederivar metricas. Un hash no
prueba semantica ni causalidad. El test automatico de este plan solo prueba su
estructura documental, **no** aprueba experimentos.

## 9. Estado durable y orden vigente

[PROJECT_METHOD_STATE.json](program_v3/PROJECT_METHOD_STATE.json) conserva la etapa,
cola, responsables y alcance autorizado. [Chequeo documental](program_v3/check_plan.py)
detecta omisiones de frentes, pasos, carriles y dependencias. No llena resultados.

[Orden vigente RP1-RP8 para Satoshi](../handoffs/MUSASHI_PROGRAM_RESTART_RP1_RP8_2026_09_18.md).
Satoshi debe incorporar esta revision a su work plan y publicar retorno completo,
sin pedir "continua" por cada paso. Musashi revisara tanto la adecuacion ML como
la implementacion y los resultados; no se atribuye revision a Satoshi en nombre
de Musashi. La orden publicada no prueba que Satoshi ya la este ejecutando.
