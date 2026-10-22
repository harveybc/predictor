# Correcciones de seleccion y entrega para auditoria de Musashi

Para Satoshi. Autoridad: instrucciones del propietario del21 de octubre.
Objetivo: completar la ingenieria y seleccion del pool acordado, demostrar que
las entradas seleccionadas llegan al modelo y entregar evidencia reproducible.
Esto es una revision preventiva con correcciones ejecutables, no aprobacion de
una seleccion terminada. Actualiza tu plan y continua con las correcciones
tecnicas sin pedir autorizacion para cada paso. No esperes el dictamen final
para corregir un defecto ya localizado. Musashi auditara el commit de entrega.

## 1. Alcance real y correcciones prioritarias

Lo verificado en el codigo/documentos consultados:

| Observacion | Correccion y prueba requerida |
| --- | --- |
| La seleccion de Musashi fue15 de30 candidatas basicas, no el inventario completo. Tu screen posterior151 tampoco equivale al universo solicitado. | Catalogo generado de todas las fuentes y familias acordadas, con conteos de datasets, features, componentes y entradas efectivamente consumidas POR SEPARADO. |
| Rechazar familias por screen univariado no prueba inutilidad conjunta. | El screen genera rankings/diagnosticos; incluir familias en evaluadores multivariados y contrastes de interacciones, aunque el padre no gane individualmente. |
| Sustituir ese screen por permutacion de un unico modelo tampoco resuelve el problema por si solo. | Evaluadores complementarios, permutacion agrupada y contrastes con reentrenamiento; importancia baja de un modelo debil no elimina una familia. |
| heuristic-strategy4987af2 aplica escala fija TRAIN a todas las features; core15 propaga el flag desde load a build. | Mantener esa regla del propietario. Media/std de cada feature solo TRAIN, artefacto persistido, misma transformacion en todos los periodos; sin excepciones silenciosas. |
| En la ruta inspeccionada, E3-M solo propaga --scaling por core15; load_v4 no recibe scaling y llama build con default rolling_z. El resultado externo guarda el argumento solicitado. | Propagar la politica en TODOS los cargadores o rechazar combinaciones no soportadas antes del fit. Prueba de integracion CLI -> cargador -> tensor -> checkpoint -> recarga; comparar politica solicitada con efectiva. No basta mirar el JSON del argumento. |
| build acepta cualquier texto distinto de fixed_train por la rama rolling. | Enum estricto y error tipificado ante un valor desconocido; ninguna degradacion silenciosa. |
| Las9 variables de estado del carril experto no son el estado del SAC de Musashi. | Inventario y ablaciones propios de cada carril. No trasladar conclusiones, limites diarios, escaladores o etiquetas entre ellos sin contraste. |

Referencias inspeccionadas: heuristic-strategy5cb19c1, archivos
app/short_horizon/expert_guided/data.py y env.py, tools/expert_guided_e3m.py;
feature-selector docs/ADMITTED_CORE_SELECTION.md; financial-data41f5af17.
Son hallazgos sobre estas revisiones; si ya los corregiste, adjunta commit y
prueba que lo demuestra. No se afirma que una corrida v4 con ese flag haya
ocurrido: se identifica la ruta que lo perderia.

## 2. Catalogo completo antes de decidir la seleccion

Genera FEATURE_CATALOG.csv y un manifiesto JSON desde productores y archivos
reales, no una lista escrita de memoria. Por candidata: id, fuente, variable
padre, familia, componente, algoritmo real, parametros, reloj/unidad, tiempo
de disponibilidad, periodo, cobertura mensual, calentamiento, faltantes,
outliers verificados, modelo/estadisticas ajustadas, hash y ruta del productor,
estado de generacion, decision de seleccion y consumo efectivo.

Debe haber filas para TODAS las familias acordadas, incluso las que requieren
reparacion. Si un servicio pagado tiene datos utilizables, adquirirlos por el
canal de credenciales existente e incorporarlos con su reloj de publicacion.
Si hay un hueco, recuperar otra fuente y contrastar valores; no interpolar,
inventar barras ni convertir series ausentes en cero. Un registro pendiente
no cuenta como familia probada ni permite llamar completa la seleccion final.

| Familia | Cobertura minima del trabajo |
| --- | --- |
| Variables originales | OHLC/retornos/rango, actividad o spread si existen, calendario y tecnicos multiescala; no reducir el universo a precio/RSI/MACD. |
| Wavelets | Componentes genuinos con backend/filtros/niveles y extremos de soporte declarados; detalle/aproximacion/energia, no solo una etiqueta wavelet. |
| Hilbert | Amplitud/fase/frecuencia y diferencias causales de ventanas pasadas; convencion de fase y tratamiento de bordes explicitados. |
| Multitaper/espectro | DPSS, bandas, entropia y otras salidas definidas por parametrizacion; resolucion, unidad de frecuencia y cadencia de calculo. |
| SSA y otras descomposiciones | Componentes reconstruidos por origen con datos disponibles; no ajustar la descomposicion global usando el futuro. |
| Kalman | Nivel/pendiente/innovaciones/factores, usando filtrado; parametros ajustados solo en el TRAIN permitido, sin smoother retrospectivo. |
| PCA/factores | Cargas aprendidas en TRAIN y congeladas; transform de validacion, no fit_transform sobre TRAIN+validacion. |
| Regimen no supervisado | Probabilidades/estado/entropia/transiciones disponibles en t, modelos TRAIN-only y codificacion estable; evaluar su aporte, no asumirlo. |
| Fractales | Productores recalculados sobre la raiz admitida, ventanas y estabilidad numerica, no copiar columnas de otro reloj. |
| Cross-asset y macro | Pares, fuerza USD/EUR, BTC y diarios/semanales disponibles, con lags de publicacion reales y envejecimiento visible. |
| Eventos/posicionamiento | Calendarios, horas de publicacion, COT y actividad disponibles; distinguir anuncios de consensos/sorpresas realmente adquiridos. |

El numero final sale del catalogo. No declarar "miles" si solo existen151,
ni fabricar columnas redundantes para alcanzar un numero. Completar las
parametrizaciones y fuentes acordadas; toda omision lleva una reparacion
concreta y su costo, no una exclusion por comodidad.

## 3. Original frente a sus componentes

REQUISITO DEL PROPIETARIO: para cada variable/familia pertinente, comprobar
cuando un componente o subconjunto de componentes aporta mas que la variable
original. Generar los componentes ANTES del descarte por relevancia del padre.
No descomponer solo CLOSE por defecto: catalogar los padres fisicamente
apropiados entre las fuentes admitidas. No transformar identificadores o
categorias sin una definicion que tenga sentido.

| Brazo | Entrada del contraste |
| --- | --- |
| P | Variable original sin componentes |
| Cj | Cada componente por separado en el cribado economico |
| Cgrupo | Subconjuntos o grupos coherentes de componentes |
| P+Cgrupo | Original y componentes conjuntamente |
| Sin familia | Contexto comun sin ese padre ni sus derivados |

Mantener fijo el contexto de las otras entradas. Usar mismas fechas, mascara
por tarea, etiquetas, semillas, costos y presupuesto. El soporte de comparacion
es la interseccion explicita de filas realmente disponibles, con hashes de
timestamps; publicar tambien la cobertura nativa. No premiar un componente
porque su calentamiento elimino los meses dificiles, ni recortar todos los
datos de todas las familias a la interseccion mas pequena antes de seleccionar.

Seleccion por componente permitida: conservar detalles y descartar la
aproximacion, o viceversa; conservar el padre junto con un componente si aporta.
No imponer el paquete entero ni exigir al componente vencer primero aislado
para poder mostrar una interaccion. Comparar por tarea (abre/cierra, compra/
venta) y contexto de regimen cuando haya soporte; una utilidad global pequena
no prueba ausencia de utilidad condicional.

Guardar CONTRASTS_PARENT_COMPONENTS.csv con padre, componentes, contexto,
soporte, parametros del modelo, semilla, metricas TRAIN/validacion, diferencias,
incertidumbre y decision. Informar conteos de parametros y tiempo/memoria:
mas componentes tambien cambian capacidad. Un control con capacidad comparable
o un presupuesto acotado por familia separa parcialmente esa explicacion.

## 4. Generadores que no se pueden reutilizar a ciegas

En financial-data41f5af17, los workers historicos contienen estos patrones.
No afirmo que ya se hayan usado en la seleccion nueva; revisarlos antes de
reutilizarlos y reconstruir sobre la raiz reparada, no recuperar salidas viejas.

- stage23_signal_decomposition_worker.compute_wavelet usa medias moviles
  anidadas como aproximacion; no equivale a ejecutar una wavelet db4. Renombrar
  el proxy y contrastarlo, o implementar el backend wavelet real requerido.
- Hilbert/multitaper/EMD muestreados incluyen interpolate y ffill. Aqui no se
  admite interpolacion. Recalcular causalmente a la cadencia declarada, con
  valores observados. Si se mantiene el ultimo estado publicado de una feature
  lenta, registrar edad y cadencia; no confundir eso con una medicion nueva.
- Algunos tamanos de ventana dependen de len(dataset). Fijarlos por config
  antes de generar; agregar futuro no puede cambiar la parametrizacion pasada.
- stage25_sota_feature_enrichment_worker.regime_labels ajusta sobre la matriz
  completa y ejecuta predict/predict_proba sobre la secuencia completa. Refitar
  en TRAIN y usar inferencia filtrada hasta t, no estados suavizados con futuro.
- El clasificador GMM preexistente de feature-eng declara centroides de un
  historico global. No importarlos como si fueran TRAIN-only de este experimento.

Hilbert de SciPy calcula sobre el array suministrado; ser una funcion estandar
no hace causal el uso de toda la serie. Ventana terminada en t, soporte de borde
declarado y pruebas de futuro/prefijo son requisitos de NUESTRA aplicacion.
[SciPy hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).
La API HMM distingue decodificacion sobre todas las emisiones y smoothing;
no confundirlo con un filtro en linea.
[hmmlearn](https://hmmlearn.readthedocs.io/en/stable/api.html).

## 5. Escalado fijo y transformaciones explicitas

Aplicar por defecto la instruccion ultima del propietario: z=(x-mean_TRAIN)/
std_TRAIN para CADA feature, sin excepciones silenciosas. Persistir nombres/
orden, media,std,varianza, filas/periodo de fit, fuentes y version con los pesos.
Constantes: std=0 implica escala1, regla registrada. Rechazar TRAIN vacio,
no finito o fuera del corte; un NaN no puede convertir una columna entera en
ausente y luego borrar todas las filas sin explicacion.

En cada fold, ajustar SOLO en su TRAIN; transformar el bloque posterior sin
actualizar estadisticas. Para el refit, usar solo informacion disponible antes
del periodo siguiente. El fit offline sobre TRAIN completo no convierte las
metricas de ese mismo TRAIN en prospectivas. Pruebas de causalidad separan
generador prefijo-invariante y transformador aprendido ya congelado; no exigir
que un refit con otra muestra TRAIN produzca las mismas estadisticas.

Un artefacto congelado no se reconstruye al cargar el modelo. Ningun --scaling
se queda solo en metadatos. Para cada familia/modelo/cargador, comprobar tensores
y predicciones de recarga. El target/clases/recompensa, OHLC de ejecucion y
costos NO se reescalan por esta operacion de entradas. Cualquier escala de sigma
en labels/riesgo se documenta aparte. Sin clipping, winsorizacion, suavizado,
interpolacion o relleno implicitos: explicitar y contrastar por separado los
que ya existan. Ciclos sin z-score solo como contraste posterior autorizado,
no como excepcion al nuevo control pedido.
[StandardScaler y transformacion congelada](https://scikit-learn.org/stable/modules/preprocessing.html).

## 6. Seleccion multivariada y utilidad real

Reutilizar los motores existentes por plugins/configs. El cribado de calidad
elimina defectos demostrados, no familias cuyo test univariado no es significativo.
Perfil/redundancia/pairwise/causalidad, cuando se ejecutan, se guardan separados
del ranking predictivo y de la utilidad en trading. No convertir NOT_IDENTIFIED
en "sin utilidad" ni una correlacion o PCMCI en efecto de intervencion probado.

Propuesta economica: screening por familia, modelos complementarios pequenos
(temporal y otro sesgo no lineal), ranking multivariado, permutacion agrupada,
adicion/eliminacion de grupos y rescate acotado de interacciones. Luego evaluar
subconjuntos finalistas en el modelo/estrategia que realmente los consumira.
No meter miles de columnas directamente a una red grande ni exigir explorar
todas las combinaciones. Fijar presupuestos y criterios antes de leer cada
bloque de decision, y registrar lo omitido por costo como no probado.

Permutacion: agrupar padre/componentes correlacionados, ventanas y familias;
usar bloques temporales conjuntos, no mezclar valores escalares al azar dentro
de una ventana. Repetir con semillas comunes. Comparar tambien tras reentrenar
sin el grupo: permutacion de una politica congelada mide sensibilidad de ESA
politica, no el potencial de la familia. No tocar precio/recompensa del simulador
al permutar entradas. La documentacion advierte tanto la dependencia del modelo
como el problema de correlacion.
[Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## 7. Negocio, etiquetas y periodos

Ventanas incluyen el ultimo tick completado y pueden tomar historia de semanas
anteriores. No insertar ni contar ticks ficticios del cierre. Toda feature
estaba publicada/disponible en t; no mover fechas hasta que "correlacionen".
Usar calendario versionado, DST y barras/filas del negocio admitidas.

Mascaras POR TAREA: abrir solo en horario elegible, cerrar/gestionar cuando hay
posicion y antes del cierre forzoso. SL/TP y cuota/tamano/margen obligatorios,
forzar cierre12h antes y no abrir hasta12h despues de reapertura. Entradas fuera
de esos horarios pueden ser historia observable; no son automaticamente targets
o acciones entrenables. Separar soporte de apertura y cierre; no exigir al
cerrador la misma elegibilidad de apertura.

Tu carril experto tiene max4 entradas/dia; el SAC de Musashi tiene max1. Fijar
en cada configuracion el limite aprobado y no mezclar tablas de ambos negocios.
No poner una cuota minima de trades que obligue a operar ante falta de señal;
no operar es un control y un resultado descriptivo, no una prueba de ventaja.

La entrega consolidada respeta TRAIN2020-2023, VALIDATION2024 y TEST2025 cerrado;
los folds internos de seleccion viven dentro de TRAIN. Fijar split/corte exacto
por brazo. Tu codigo actual usa2020-2022 para fit,
2023A/B para desarrollo y2024 previamente observado; no llamarlo4 anos de fit
ni2024 confirmacion independiente virgen. TRAIN2020-2023/VALIDATION2024 es el
contrato del proyecto; el brazo de tres anos se identifica como tal o se adapta
explicitamente al protocolo consolidado antes de compararlo. No escoger otro
ano porque salga mejor. TEST2025 cerrado.

ZigZag retrospectivo solo como etiqueta/demostracion, nunca entrada. Sellar su
version y el tiempo de finalidad. Purgar cada fold segun el soporte REAL de cada
etiqueta; si cambia el experto, regenerar etapas dependientes. Para labels
censurados, documentar el tratamiento. Probar tambien targets complementarios
de cierre anticipado y resultado neto de orden cuando corresponda; un unico
target no define utilidad universal de todas las features para RL.

## 8. Realimentacion del agente

Separar features de mercado y estado de ejecucion. El simulador siempre conserva
su estado real para imponer negocio, aunque la politica no lo reciba. Contrastar
sin feedback, contexto minimo de posicion/tarea y estado completo; constantes
en los slots omitidos permiten controlar capacidad. Reportar dimensiones y
parametros para no confundir quitar informacion con achicar la red.

No suponer ni que todo el estado ayuda ni que todo es ruido. Sin posicion/tarea,
algunas decisiones pueden ser no observables para una politica sin memoria;
declararlo al interpretar la ablacion. Recortes de sigma/duracion y can_enter
se inventarian y prueban. No modificar sus reglas de negocio al quitar inputs.

## 9. Pruebas y entrega verificable

| Prueba | Resultado que debe demostrar |
| --- | --- |
| Cobertura del universo | Catalogo completo, conteos por familia/estado y razones reproducibles; no afirmar totalidad con30 o151. |
| Padre/componentes | BrazoP/Cj/Cgrupo/P+Cgrupo/sin familia realmente consumidos, soporte pareado y parametros guardados. |
| Futuro y prefijos | Cambiar datos disponibles despues de t no cambia features anteriores con fit congelado; controles positivos sensibles a cambios disponibles en t. |
| Escalador | Cambiar validacion no altera mean/std; misma escala para el mismo valor en distintos origenes; constantes y NaN rechazados/tratados explicitamente. |
| Propagacion | Ruta real del CLI a todos los cargadores y tensores, incluida v4/nuevos pools, coincide con scaling solicitado. |
| Sin relleno oculto | Un hueco interior provoca reparacion identificada; no desaparecen meses o se fabrican barras. |
| Regimen/factores | Fit TRAIN-only y estados de validacion invariantes a emisiones posteriores al origen; nada de smoother. |
| Seleccion | Rescate de interacciones y grupos correlacionados; no exclusion total por un solo target/screen/modelo. |
| Negocio | Timestamps, DST, buffers, fills, SL/TP/volumen/cuota y soportes por tarea verificados por simulador real. |
| Recarga | Pesos/escaladores/config y predicciones reproducibles; fallo claro si falta un artefacto, sin refit escondido. |
| Publicacion | Catalogo -> subset -> columnas del tensor -> modelo -> metricas se enlazan por hashes; OLAP con readback y copia independiente. |

Tablas por tarea y split: log-loss/PR-AUC y baseline cuando sean clasificadores;
profit bruto/neto y costos desglosados, Sharpe con convencion/periodo, DD,
numero de ordenes, porcentaje ganador, PF, exposicion y abstencion. Datos de
TRAIN y validacion siempre separados. No exigir profit positivo para completar
esta fase; tampoco presentar mejora de clasificacion como rentabilidad probada.
El negocio/politica final se evalua con los mismos costos y contabilidad.

## 10. Orden operativo sin supervision humana

1. Corregir escalado/propagacion y materializar el catalogo completo.
2. Reparar productores y adquirir fuentes faltantes; comprobar cobertura, reloj,
   causalidad y variantes de padres/componentes sobre la raiz admitida.
3. Ejecutar calidad/perfiles y etapas de seleccion existentes; sin el veto
   univariado universal. CPU y datos por bloques, sin agotar RAM.
4. Ejecutar contrastes multivariados y de padre/componentes; avanzar por cola
   finita con presupuesto, dependencias y reanudacion declarados.
5. Congelar shortlist, probar consumo real/estado y modelos/estrategia; publicar
   todos los signos/resultados, no solo ganadores.
6. Entregar una solicitud de auditoria con commit exacto, catalogo, config,
   pruebas, ledger, tablas, artefactos de pesos/escalado y comandos reproducibles.

Satoshi usa su recurso asignado; no tomar la5090 ni la otra GPU del mismo host
sin coordinacion. No tocar DOIN ni lanzar optimizaciones nuevas por esta orden.
No quedarse mirando logs: timers/scripts siguen la cola, reintentan problemas
transitorios de forma acotada y guardan estado. Resolver fallos de codigo sin
pedir permiso para cada linea. Ante un dato verdaderamente inaccesible, ejecutar
la recuperacion/fuente alternativa; no declarar que el trabajo esta completo si
todavia falta una familia obligatoria. Una limitacion real se explica con objeto
y accion concreta, sin inventar datos ni eludir controles de acceso.

## 11. Solicitud final a Musashi

Enviar: repos/ramas/commits exactos, diff contra la version revisada, pruebas
ejecutadas, resultados por etapa, manifiestos de catalogo/subset/consumo,
ubicaciones privadas de evidencia y comandos. Incluir explicitamente los
contrastes padre/componentes y la propagacion del escalado en core15, v4,
pool nuevo y recarga/inferencia.
Solicitar que Musashi intente reproducir fallos de futuro, soporte, escalado,
consumo real y contabilidad antes de aceptar el resultado. Sin secretos ni
nombres de host/IP/rutas personales en lo publicado. Un hash identifica un
artefacto; no reemplaza las filas y el codigo necesarios para recalcularlo.

Este documento no certifica software nuevo. La aceptacion exige cerrar y
verificar los requisitos anteriores, no sumar pruebas sin relacion con ellos.
Musashi debe informar defectos con correccion minima y comprobar el cierre de cada uno,
sin convertir preferencias metodologicas o resultados negativos en bloqueos.
