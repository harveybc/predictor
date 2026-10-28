# Retorno de Musashi y entrega para la auditoria final

Para Satoshi. Fecha: 2026-10-22. Integra este retorno en tu cierre de seleccion,
codificacion, perfiles/agrupamiento y extractores; despues entrega la solicitud
final de auditoria a Musashi con el commit exacto y la evidencia consolidada.
No necesitas otra autorizacion para corregir defectos demostrados y repetir
solamente sus etapas afectadas. Esta entrega no inicia otra campana ni altera DOIN.

## 1. Estado y alcance

- Seleccion v3: 3894 unidades agregadas, sin errores de proceso registrados.
  Nuestra mitad ejecuto base, contrast, encoding y rescue. Comprobacion independiente:
  contrast714, encoding78 y rescue120 grupos multi-brazo, cero discrepancias de
  firmas en esos grupos. Esto no sustituye la auditoria de productores ni consumidores.
- Paquete E3-M asignado: **12/12 brazos con resultado completo**, 36 combinaciones
  brazo/semilla y 144 checkpoints finales. Conserva las marcas de validez nativas.
  Tener un resultado completo no significa aprobar su politica ni su entrada.
- Ciencia E3-M: heuristic-strategy `31d90f7`, archivo SHA256
  `efa23475193865fc6594f7af4f6ee61a7d2cbb41d96756af89a8ed3cd04e202a`.
  Se ejecuto aprendizaje guiado por ZigZag, imitacion y DAgger; **no SAC nativo**.
- Musashi todavia no ejecuto SAC sobre esta nueva entrega compartida. Los resultados
  anteriores a las reparaciones no son campeones ni controles de esta entrega.
- El proceso de seleccion/representacion **no esta certificado como terminado**:
  falta cerrar el alcance de ajuste de transformaciones, consumo/soporte final,
  perfiles numericos y seleccion experimental de extractor por grupo.

## 2. Evidencia exportada y custodia

Paquete logico: `shared-selection-audit-handoff-20261022`.
Manifiesto SHA256:
`648acaf8e58aa97f104a9dbb72bd519c28951639b382530643fd66943dcd1eb4`.
Se conserva fuera de Git, en el volumen de experimentos y en ambos workers como
`<handoff-root>`. No contiene precios ni sustituye los artefactos de origen.

| Archivo | Contenido |
| --- | --- |
| `MANIFEST.json` | Identidades de las tres colas, versiones, hashes y reglas de agregacion |
| `RESULTS_BY_SEED.json` | Todas las metricas nativas por semilla/periodo/contrato; ajustes finales y arbitros |
| `SUMMARY_BY_ARM.json` | Medias, desviacion estandar muestral, n no nulos, marcas de validez y hashes de los resultados |
| `SUMMARY_BY_ARM.csv` | 88 filas brazo x contrato x periodo; beneficio, Sharpe/Sortino, DD, trades, aciertos, PF, actividad, exposicion y costos |
| `CONSUMPTION.json` | Columnas pedidas/consumidas en orden, stores, scaler/probe, ventanas, fit, exclusiones, parametros y soportes |
| `SUPPORT_COMPARISONS.json` | Igualdad/desigualdad de firmas entre los cuatro brazos consolidados |

Los originales siguen en sus tres mirrors: original10, recovery1, relocated1.
Cada resultado tiene `ops/<arm>.RESULT.json`, `out/<arm>/E3M.json`, `SCALER.json`,
`PROBE.json` y doce `.pt`. El manifiesto exportado permite localizar la cola por
identidad sin publicar hosts, UUID, IP, credenciales ni rutas personales.
Los hashes de los archivos terminales y cierres fueron verificados. La custodia
fisica **no demuestra publicacion en warehouse**: anexa tus recibos de publicacion
y lectura de vuelta; no presentes los mirrors como entrega OLAP.

Identidades de configuracion:

- Original: `4b19a3d12004f6035b4411d5166d08a5a61acb14592d084d3b363708b6b029f4`.
- Recuperacion: `7a3fef865d059799dd7688bb64e4021a9f8aba4e0a53b3758ade4d1367cb3fb0`.
- Traslado: `fb8a4065ba5d997747b711221e080d1fcb1f8cda64ca1656f6bd453a0d7ba97e`.

Versiones comunes: torch2.13.0, numpy2.5.1, pandas3.0.3, scipy1.18.0,
scikit-learn1.9.0. El traslado verifico 121 archivos cientificos/de entrada contra
el paquete original. Su GPU fue RTX4070 Laptop; los otros resultados usaron la5090.
No se afirma igualdad bit a bit entre GPU diferentes.

## 3. Configuracion realmente ejecutada

FULL = cuatro especialistas: abre-long, abre-short, cierra-long y cierra-short.
Ventana:13 observaciones, stride2, incluye t; no son siempre24 horas de reloj
cuando la historia cruza cierres. Nucleo units2, head16, depth1; no una arquitectura
ya seleccionada por perfiles. Maximo1000 epocas por fit, paciencia10, min_delta1e-8,
AdamW lr1e-3, batch256, sin BatchNorm. Parada:
`monitor = 0.5 * (TRAIN loss + INNER_A loss)`, ambas en inferencia al final de epoca.
La perdida incluye CE con pesos de clase y Huber en las tareas de regresion.
Hay ronda inicial0 y tres rondas DAgger1-3, con cuatro modulos por ronda/semilla.
No confundir estas rondas con generaciones DEAP ni con reajuste semanal.

Semillas:20261019,20261020,20261021. Escala solicitada y efectiva `fixed_train`;
politica heredada `nonstationary=diff`. Esto **no autoriza heredar differencing**
automatico en SAC: cada columna transformada necesita receta y alcance de fit.

Consolidados: contrato externo TRAIN2020-2023, VALIDATION2024, TEST2025 cerrado.
El scaler declara fit `[2020-01-01,2023-07-01)` y reserva el bloque interno posterior
para INNER_A. No describirlo como cuatro anos completos usados para gradientes.
Diagnosticos `dev3y`: fit `[2020-01-01,2023-01-01)`, INNER_A/B en2023 y OBS_2024
separado. Los cortes exactos y soportes quedan exportados, no se intercambian anos.
2024 ya es desarrollo inspeccionado, no una confirmacion virgen reutilizable.

El experto registrado aqui es Depth3 / deviation40.7pips / Backstep0, con7094 pivotes
y7093 confirmados. No sustituirlo por los parametros de notas anteriores.

## 4. Resultados completos del contrato nativo

Medias aritmeticas de las tres semillas, incluidas las que tienen `valid=False`;
no se elimina una semilla despues de observarla. Sharpe nativo usa `sqrt(365)`.
Las medias de DD/trades/aciertos son de simulaciones separadas, no un portfolio.

### Cuatro finalistas consolidados

| Brazo | Features | Sharpe TRAIN | INNER_A | VALIDATION2024 | SD Sharpe2024 | Validos2024/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cons_core15 | 15 | -0.0508 | -2.1293 | -1.1779 | 0.5363 | 3 |
| cons_shortlist_strict | 125 | +2.8997 | -0.3283 | -0.7170 | 0.2477 | 3 |
| cons_shortlist | 173 | +4.3945 | -0.9091 | -0.6186 | 1.3219 | 3 |
| cons_shortlist_strict_enc | 137 | +5.0495 | +0.0984 | -2.8919 | 1.2906 | 3 |

| Brazo | Beneficio neto2024 USD | DD maximo medio % | Trades medios | Aciertos medios % | Parametros, cuatro modelos |
| --- | ---: | ---: | ---: | ---: | ---: |
| core15 | -4089.33 | 52.70 | 393.00 | 34.38 | 2236 |
| strict125 | -2261.56 | 37.99 | 270.33 | 36.60 | 11036 |
| shortlist173 | -2369.34 | 48.13 | 452.33 | 37.80 | 14876 |
| strict+enc137 | -6650.42 | 70.89 | 472.67 | 35.72 | 11996 |

### Ocho diagnosticos de brecha, contrato dev3y

Base = dropout0.1 / wd1e-4. Fuerte = dropout0.3 / wd1e-2.
Se cambiaron ambos conjuntamente: no atribuir su efecto solo a uno de ellos.
`none` pone a cero los slots de realimentacion; no elimina su capacidad de red.

| Lista / estado / regularizacion | TRAIN | INNER_A | INNER_B | VALIDATION2024 | SD2024 | Validos2024/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core15 / full / base | -0.4389 | -2.0713 | -1.8369 | -1.9367 | 0.4686 | 3 |
| core15 / full / fuerte | -0.4042 | -0.1450 | -2.6464 | -2.0997 | 0.9717 | 3 |
| core15 / none / base | +0.4504 | -1.2437 | -1.5394 | +0.0577 | 1.3202 | 2 |
| core15 / none / fuerte | -0.4695 | -1.5982 | -2.5485 | -0.8118 | 0.9258 | 2 |
| strict125 / full / base | +3.5388 | -0.7461 | -1.1752 | -0.2140 | 1.5189 | 2 |
| strict125 / full / fuerte | +2.8137 | +0.4340 | -1.3442 | -0.0456 | 0.5723 | 2 |
| strict125 / none / base | +4.1987 | -0.7221 | -0.0131 | -0.1796 | 0.6909 | 1 |
| strict125 / none / fuerte | +4.2656 | -1.4516 | -3.0660 | -0.3167 | 1.2078 | 2 |

El CSV incluye todas las demas metricas de estos ocho brazos y de sus replays.
Hay dispersion y una brecha de generalizacion importante en listas grandes;
no queda aislada una causa unica. El pequeno positivo de core15/none/base no
certifica robustez ni permite borrar sus otras semillas o periodos.

## 5. Correcciones de interpretacion para la solicitud final

### Soporte y capacidad, no cuatro comparaciones de una sola variable

| Brazo consolidado | Filas TRAIN | INNER_A | OBS_2024 | Hash soporte OBS_2024 |
| --- | ---: | ---: | ---: | --- |
| core15 | 17128 | 2436 | 4871 | 21f3dacd4915644a |
| strict125 | 15485 | 2055 | 4379 | 3364cd2677e7771a |
| shortlist173 | 15390 | 2055 | 4379 | 3364cd2677e7771a |
| strict+enc137 | 15317 | 2055 | 4379 | 3364cd2677e7771a |

Las cuatro firmas TRAIN difieren. core15 tampoco coincide con las otras listas
en INNER_A/2024. Las tres listas grandes coinciden en esos dos soportes, **no en
TRAIN**. Se mantienen presupuestos pero no conteos de parametros; la lista173
tambien cambio de clase de GPU. Estas son mediciones descriptivas completas,
no deltas causales puros de anadir columnas. Para una atribucion emparejada,
congelar soporte comun antes de los fits, semillas, reloj, targets y contratos;
verificar firmas terminales, y declarar capacidad/hardware como factores.
No repetir todo por defecto: refit solo de contrastes que requieran esa atribucion.

### El replay de una entrada tambien cambia apalancamiento

El contrato nativo usa max4 entradas/dia y leverage100. `musashi_contract` es
replay de los mismos pesos, con max1 entrada/dia y leverage50; no entrenamiento
SAC y no una ablacion exclusiva de frecuencia. Corrige esta precision respecto
al reporte breve anterior. rel_volume=0.10; capital inicial10000.

| Lista | Sharpe2024 replay | Beneficio USD | DD medio % | Trades medios | Aciertos % | Validos2024/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core15 | -0.3869 | -591.31 | 22.36 | 187.00 | 38.91 | 2 |
| strict125 | +0.0957 | -103.05 | 16.14 | 124.00 | 38.73 | 1 |
| shortlist173 | +0.0721 | +53.15 | 13.89 | 209.67 | 35.35 | 3 |
| strict+enc137 | -1.5406 | -2037.03 | 24.45 | 203.33 | 34.50 | 2 |

En shortlist173, Sharpe2024 por semilla: -1.0441 / -0.2238 / +1.4843.
El resultado medio pequeno no permite escoger retrospectivamente la tercera.
Sharpe medio positivo no implica beneficio medio positivo: strict125 lo ilustra.

### Que significa valid=False en esta version

`env.py` incluye `min_orders_per_week=3.0`; `valid_weekly_rule` exige esa media
de entradas y `valid` tambien comprueba limite diario, ruina y Sharpe finito.
`expert_guided_e3m.py` prioriza arbitros validos y, si ninguno pasa, escoge entre
los otros. Por ello el minimo heredado puede afectar tanto etiquetas de reporte
como la seleccion del arbitro, no solo el resumen final.

No confundas estas marcas con una violacion demostrada de horario, SL/TP o un
fracaso informatico. El propietario no exige operar para cumplir una cuota minima.
Separa `hard_business_valid`, actividad/abstencion y elegibilidad experimental.
No borres ni cambies las marcas historicas. Para quitar el minimo en la siguiente
receta, mide el impacto en arbitros; si cambia el elegido, reevalua el consumidor
afectado con identidad nueva. No hace falta repetir la generacion de features.

## 6. Seleccion y codificacion: medido frente a pendiente

El pool efectivamente evaluado fue260 columnas HGB/257 temporal, con57 familias
semanticas/de fuente; no miles de entradas ni57 clusters de perfiles aprendidos.
41 familias retenidas,2 rescates de interaccion,14 no retenidas en esos modelos/
tareas. Las listas originales126/174 y consumidoras125/173 son identidades
distintas: `raw.log_close` era control de seleccion, no una columna del consumidor.

Las celdas ganadoras incluyen sesiones, GMM one-hot, rel24 y diff+rel480 en tareas
y modelos concretos. No prueban mejora universal ni Sharpe. El candidato137
conserva sin/cos y probabilidades, agrega12 columnas (8 relativas de rango,
1 relativa de cierre y3 GMM one-hot); sesiones ya estaban en la base.
El resultado negativo del consumidor137 no demuestra inutilidad universal de
esas codificaciones: soporte TRAIN/capacidad/arquitectura no estan aislados.

Pendiente prioritario: el productor layer1 retenido `e04b496a` elige ASINH con
diagnosticos TRAIN pero pasa la serie completa a la decision; nanmin y skew
transformado pueden depender del futuro. El testigo sintetico mantiene TRAIN
identico y cambia ASINH a NONE al alterar solo el sufijo. Es dependencia demostrada
de la decision, **no impacto cuantificado sobre todas las features reales**.
Ver [testigo y correccion minima](FEATURE_PROFILES_ENCODINGS_AND_SAC_INPUT_POLICY_2026_10_22.md).
Anexa la correccion TRAIN-only, regresion de perturbacion y diferencias reales
en columnas seleccionadas; reconstruye solo descendientes afectados. Si ya lo
corregiste, devuelve commit/prueba/tabla, no otra investigacion desde cero.

## 7. Cierre concreto de perfiles, grupos y extractores

La tabla de recomendaciones existente no equivale a seleccion experimental.
Entrega una fila por feature seleccionada con: fuente/padre/componentes, unidad,
disponibilidad, soporte del productor, cobertura mensual, faltantes/calentamiento,
ADF/KPSS, colas/skew/kurtosis/outliers, dependencia temporal/rezagos, redundancia
y estabilidad por fold. Di que metricas se calcularon realmente y cuales faltan.
No uses perfiles calculados sobre VALIDATION/TEST para ajustar reglas del baseline.

Para cada grupo: miembros, pares inseparables, metricas que justifican la union,
regla/umbrales o distancia y parametros de clustering, periodo de fit, estabilidad
y politica para nuevas observaciones. Si el grupo es semantico/manual, nombralo
asi; no inventes umbrales estadisticos a posteriori para hacerlo parecer aprendido.

| Perfil/grupo candidato | Contraste pequeno que debe quedar identificado |
| --- | --- |
| Retornos, rangos, velas y momentum | Conv1D causal pequena frente a control simple; kernel/filtros/stride y parametros registrados |
| Fase Hilbert | Sin/cos conjuntamente; preservar coordenadas circulares y disponibilidad del estimador |
| Wavelet/espectro por escalas | Padre, componentes y combinacion; ramas por escala o control temporal uniforme |
| Calendario/eventos | Sin/cos y banderas declaradas; identidad/pointwise temporal frente a rama causal cuando se pruebe |
| Regimen/factores | Probabilidades/entropia y padres; one-hot solo como alternativa explicita, fit TRAIN-only |
| Diarios/COT de publicacion lenta | Valor disponible y edad; identidad/pointwise como control, sin inventar barras |

Son candidatos, no ganadores ya medidos. Reutiliza las pruebas existentes y
ejecuta solo lo que falte en una cola finita para comparar candidatos por grupo,
con soporte/semillas/tarea/presupuesto declarados, perdida contra baseline,
brecha TRAIN-inner, dispersion, parametros, memoria y tiempo. Conserva interacciones
padre/componentes; no vuelvas a descartar familias enteras por un screen univariado.
Un proxy experto orienta el extractor pero no certifica que sea optimo para SAC.

Separa: soporte historico del productor, ventana13/stride2 del consumidor y campo
receptivo del encoder. Calcula el campo por capa (kernel, dilatacion y stride),
incluyendo agregacion final; no selecciones profundidad por intuicion ni colapses
tiempo antes de que el extractor temporal tenga acceso a el. Un flatten final
con orden fijo no elimina por si mismo la informacion de posicion temporal.
No repetir ahora barridos amplios de ventanas, reajuste semanal ni modelos grandes.

Entrega minima ejecutable: catalogo/perfiles, asignacion feature->grupo,
recetas de encoding/scaling, tabla de contrastes de extractor y manifiesto de
consumo con hashes. Reusa nombres/formatos existentes si ya cumplen esos campos.
La ausencia de consenso economico o del consumidor multi-tasa de oro no impide
auditar una poblacion elegible explicitamente delimitada; no declarar probado
lo que quedo fuera ni completado todo el inventario por conveniencia.

## 8. Contrato compartido para las dos futuras lineas

**Compartido:** version inmutable de datos/calendario, pool y subset identificado,
codificaciones explicitas, reglas de grupos, escaladores TRAIN-only y geometria
temporal. Cada consumidor declara sus columnas exactas y soporte; un subset
elegido por tareas del experto no es automaticamente ganador para SAC.

**Observaciones iniciales de ambos:** sin realimentacion del agente, sin rolling
z-score, clipping/winsorizacion de entradas, relleno NaN->0, interpolacion o
transformaciones implicitas. Los slots heredados deben desaparecer o quedar
documentados constantes, probados tambien en actor y critic. Relativas causales
como columnas seleccionadas son diferentes de normalizar todo con rolling-z.
No restaurarlas como politica de escala oculta. Mantener coordenadas nativas
sin/cos, flags/probabilidades conforme a la receta sellada.

**Simulador:** mantiene posicion/cuenta/SL/TP/tamano/calendario para imponer negocio,
aunque la politica no los reciba. Una orden por activo; cada orden con SL, TP y
fraccion de volumen/riesgo, minima del broker y rel_volume explicitos. Sin entradas
antes de12h tras reapertura; cierre obligatorio12h antes del cierre semanal.
No exigir mascara de apertura al soporte de gestion/cierre. Historia previa
permitida con barras observadas, sin ticks ficticios de mercado cerrado.
SAC baseline: max1 entrada/dia y leverage50; no heredar leverage100 ni max4 del
E3-M nativo. Si se comparan politicas entre lineas, mismo contrato de ejecucion.

**Satoshi:** aprendizaje guiado por experto ZigZag; etiquetas/demostraciones solo
del lado de entrenamiento. Declara imitacion, DAgger y RL si existe; no presentar
feedback sintetico como feedback humano real. Purga por soporte/finalidad real
de etiquetas y ejecucion, incluidos cierres y confirmaciones.

**Musashi:** SAC sin experto, sin demostraciones/labels/recompensa del ZigZag y
sin pesos de extractor preentrenado con esas etiquetas. Puede compartir receta
de arquitectura seleccionada, pero inicializacion/pesos y aprendizaje propios.
Primero core15/strict/shortlist x entrada directa/representacion temporal pequena,
con identicos soportes/semillas/presupuesto/negocio cuando se atribuyan deltas.
Extractor por feature o grupo -> fusion -> nucleo -> actor/Q; registrar capacidades
por componente, no subir todos a la vez. Referencia:13 instantes/stride2 incluyendo t.
Presupuesto fijo y evaluacion espaciada; no reactivar early stopping caro por update
ni reajuste periodico. Esos son contrastes posteriores, no cambios silenciosos.

En ambas lineas: TRAIN2020-2023, VALIDATION2024, TEST2025 cerrado; bloques internos
identificados sin cambiar el ano por conveniencia. Publicar TRAIN y VALIDATION,
brechas, dispersion y todas las metricas de trading/costos, aun negativas.

## 9. Tu solicitud final de auditoria a Musashi

Incluye commit exacto/diff, manifiesto de release, evidencias reproducibles,
resultados completos y una tabla requisito -> prueba -> resultado -> correccion.
No esperes una ganancia positiva para pedir la auditoria. Solicita atacar:

| ID | Objeto que debe poder comprobarse |
| --- | --- |
| A01 | Catalogo, pool generado/elegible/evaluado/consumido y exclusiones sin confundir conteos |
| A02 | Clock/barra, DST, valores observados, cobertura mensual y disponibilidad por fuente |
| A03 | TRAIN-only de ASINH, scaler, PCA, Kalman/regimenes y cualquier transformacion aprendida |
| A04 | Prefijos/futuro con controles positivos, t incluido, soporte del productor y ausencia de filtros centrados |
| A05 | P_raw/P_decided/componentes/contexto e interacciones, comparaciones realmente emparejadas |
| A06 | Purga de labels ZigZag/confirmacion/cierres con tiempos de disponibilidad reales |
| A07 | Perfiles->grupos->encoding->extractor: regla reproducible, evidencia medida y limites |
| A08 | Lista/config->tensor en orden->scaler/checkpoint->recarga->inferencia real de ambos adaptadores |
| A09 | Ausencia efectiva de feedback y transforms no declarados en todas las observaciones de las nuevas lineas |
| A10 | Negocio/acciones/contabilidad, contratos4/dia100 vs1/dia50, minimo semanal separado y costos declarados |
| A11 | TRAIN/inner/VALIDATION separados, dispersion, soportes, selecciones de arbitro y todas las marcas de validez |
| A12 | Identidades, publicacion OLAP con lectura de vuelta, dos copias y reanudacion sin segundo fit oculto |

Clasifica cada objecion como defecto demostrado, riesgo por comprobar o mejora
metodologica. Repara defectos localizados; no bloquees por preferencias, ausencia
de profit, nombres de archivo distintos o extensiones fuera del subset declarado.
Una omision de causalidad/consumo no se cubre con muchos tests de operaciones.
Si ya existe evidencia equivalente, enlazala y evita rehacer trabajo.

## 10. Orden de trabajo y reparto

1. Satoshi integra este retorno, verifica el paquete de evidencia y termina lo
   pendiente de productores/encoding/perfiles/grupos/extractores con automatizacion.
2. Publica la entrega final y su solicitud de auditoria; declara tambien la
   poblacion/recetas aceptables aunque otras extensiones sigan en desarrollo.
3. Musashi audita exigentemente los caminos efectivos; corrige lo que corresponda
   a su adaptador SAC y devuelve defectos minimos reproducibles al productor.
4. Con la version compartida cerrada, ambos despliegan sus pruebas independientes
   en sus recursos coordinados, sin colas antiguas ni dos procesos pesados en el
   host de RAM limitada. No hace falta otra aprobacion de resultados intermedios.

Operaciones propias:72 pruebas focalizadas pasaron; timers finalizaron solos,
sin procesos cientificos residuales. Esto verifica ejecucion/custodia, no sustituye
el cierre cientifico anterior. Documentacion operativa y precision del traslado:
[ejecucion](E3M_ASSIGNED_GPU_EXECUTION_2026_10_22.md),
[recuperacion](E3M_CAPACITY_RECOVERY_2026_10_22.md),
[cierre del traslado](E3M_RELOCATED_ARM_2026_10_22.md),
[contrato comun](SHARED_SELECTION_TO_RL_HANDOFF_2026_10_22.md),
[perfiles y reglas de entrada](FEATURE_PROFILES_ENCODINGS_AND_SAC_INPUT_POLICY_2026_10_22.md).
