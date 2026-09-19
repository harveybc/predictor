# Orden historica RP9-RP16: corregir, reutilizar y comparar el modular

Retorno revisado el 19-sep. La siguiente orden vigente es
[RP17-RP24](MUSASHI_PROGRAM_RP17_RP24_2026_09_19.md); conservar esta como historia.

Fecha: 2026-09-18. Responsable: Satoshi. Revision: Musashi.
Origen: retorno `2347c7b` y [dictamen con evidencia ejecutada](../audits/work_plan/MUSASHI_RP1_RP8_REVIEW_2026_09_18.md).
La aprobacion del owner comprende ARCH-A/B/C/0 y H-CORE segun
[su especificacion](../tres_temas_entrevista/CORE_PRETRAINING_HYPOTHESIS_2026_09_18.md).

## Alcance y continuidad

Ejecutar los ocho bloques sin pedir continua entre operaciones normales. Incluye
pruebas, reparacion del cierre, recuperacion de metricas medibles y piloto sucesor
DEVELOPMENT de comparacion arquitectonica. No reserva, live ni seleccion financiera.
No borrar/reentrenar todo, no inventar observaciones, no detener servicios ajenos.
Usar infraestructura existente y preservar historia. Mantener informados al owner
y a Musashi ante un incidente real; un resultado negativo no es un bloqueo.

## RP9. Conciliar el estado y adoptar lo aprobado

Incorporar documentos/estado de master, incluida la adicion `1b82220`, sin sustituir
el codigo de la rama de trabajo por codigo viejo de master. Preservar el retorno
RP1-RP8, sus commits, disenos, registros y arrays. Estado de MOD-E0-DEV: ejecutado,
revision con correcciones; no equiparar su VERIFIED anterior a aceptacion externa.
Registrar por bloque: implementado, ejecutado, verificado localmente, conciliado
independientemente, revisado e inferencia. Mantener MOD-ARCH-COMPARE y las tareas
MOD-FROZEN-PREFIX / MOD-CORE-PRETRAIN con sus dependencias. No lanzar H-CORE ahora.

Congelar los contraejemplos del dictamen como PRE antes de corregir. Usar copias
para todas las alteraciones. La reproduccion de una falta no concede aceptacion
a sus archivos alterados; conservar los originales y su inventario de digests.

## RP10. Un cierre completo de verdad, sin una plataforma nueva

Reusar los contratos de poblacion/cierre ya resueltos en el frente utility cuando
sean compatibles. Derivar unidades y pilotos esperados desde el diseno registrado,
jobs y campanas; validar diseno recursivo, self-digest, codigo, parametros consumidos,
identidades, dependencias y selecciones de intento. No tratar el contenido de REPORT
como fuente independiente de lo que el propio run hizo.

Pruebas obligatorias contra el CLI/API real: raiz vacia; 6/69 presentes; miembro
omitido, duplicado o extrano; diseno alterado; padre discrepante; brazo o replica
ausente; intento fallido; receptor H3 sin donante verificable. Cada una rechaza
el cierre total, sin eliminar silenciosamente una observacion de un intervalo.
Parcial solo con estado explicito y denominador/poblacion fijados previamente.
La CLI devuelve no-cero ante fallos y `all_verified` nunca contradice las partes.

Para warehouse: derivar poblacion desde contabilidad y comparar terminales,
metricas, costos, artefactos/lineajes y estados de todos los hijos; no solamente
cuatro agregados por directorio encontrado. Separar local y conciliacion viva;
ninguna puede presentarse como la otra. Escrituras solo por gobernanza/outbox.

## RP11. Verificar la tarea ML, los pesos y las metricas, no sus etiquetas

Recomputar de generador y contrato: X, filas, y, ventanas, horizontes, particiones,
escalas train-only, ingenuo, oraculo y denominadores. Validar esquemas, formas,
tipos, finitud y politica de vacios/ceros por variable; agregacion y unidades.
MAE, MSE/RMSE cuando procedan y MASE: todas las variables y baselines publicados,
incluido el lineal. Una metrica no finita nunca es MEDIDO.

Ligar checkpoints a sus jobs/donantes. Reconstruir grafo y leer pesos en proceso
CPU fresco; reproducir predicciones desde inputs regenerados con tolerancia
explicada por dtype/entorno antes del chequeo. Recomparar pesos de extractor y
activaciones de AMBOS brazos y todos los grupos. Verificar restauracion del mejor
checkpoint y parada tambien cuando termina por presupuesto, no solo por paciencia.
Verificar que el presupuesto inferior a un batch no se exceda silenciosamente.

Pruebas: etiquetas=prediccion; denominador x100; filas +100000; baseline/MAE alterado;
pesos ilegibles/cambiados; mismo bool de paridad pero arrays distintos; extractor
no congelado; NaN, infinito, vacio; futuro/test perturbado por el callable real.
No usar `_raise()` del test como sustituto del rechazo del codigo productivo.
Mutar cada guarda y mostrar que la prueba del comportamiento se pone roja.

Reverificar el piloto historico sin entrenar. Si algo no es reconstruible,
declarar el alcance exacto; no afirmar que siempre estuvo probado. Publicar cierre
sucesor aditivo y delta. No transferirle metricas inexistentes de checkpoints.

## RP12. Reparar perfiles y delimitar la inferencia

Corregir la fuerza de tendencia a Var(T+R); verificar seno puro, tendencia pura,
constante, ruido, mezclas y bordes con referencia independiente. Declarar el
algoritmo de descomposicion real y sus tiempos de disponibilidad; no convertir
una caracterizacion train por lote en operador online causal sin otra prueba.

Reanalisis de Musashi: 15/15 particiones iguales bajo la correccion. Reproducir
y ampliar a asignacion exacta ordenada, normalizacion, redistribuciones aleatorias,
inputs del modelo y donantes para justificar reuso. Si cambia un dato consumido,
aislar solo la evidencia afectada. No retocar disenos o resultados historicos.

Corregir las cifras de periodo/cobertura/filas y la etiqueta de curvas MSE vs MAE.
H2: diferencias descriptivas, no equivalencia ni ausencia de efecto. H3: ventaja
del procedimiento completo ejecutado, no atribucion exclusiva a fusion. Los IC de
tres trayectorias no demuestran precision general; fijar un analisis de potencia/
precision con sensibilidad en DEV antes de proponer tamanos confirmatorios.
No elegir niveles porque favorezcan una hipotesis ni confundir ARI alto con utilidad.

## RP13. Cumplir el contrato de metricas antes de otro entrenamiento

Inventario D/Y/M/G con productor, grano, costo y evidencia por metrica requerida.
Reusar medidores ya probados en D0-D3/M3-M4 donde apliquen. Por datos/target/particion:
serializacion, bytes/compresion, escalas/mascaras, H0 con quantizer train-only,
componentes y SNR sintetica definida, dependencias, perfiles y soporte temporal.
Tasas contextuales e informacion: modelo/distribucion declarados o abstencion
razonada, nunca bits de conocimiento inferidos de longitud comprimida.

Modelo: grafo, estados y parametros; normas/longitudes de pesos y descriptores
de rango aplicables; medidas en inicial, checkpoints espaciados predeclarados,
mejor validacion y final. Declarar gradientes y activaciones medidas o su ausencia.
Pruebas de cero/rango uno/constantes, costos y cobertura antes de usarlos.
No calcular SVD costoso por minibatch. Fijar frecuencia con piloto de costo.

Backfill solo de lo derivable de arrays/pesos existentes, mediante evento sucesor
gobernado. Historicos sin checkpoint inicial/intermedio: NO_MEDIDO con razon,
no reconstruccion inventada ni repetir toda la campana para esconderlo. Comparar
contenido ingresado con contabilidad; que lo mida el worker no basta si el cubo
no lo recibe. Ciencia y diagnosticos deben ser consultables sin mezclar sus granos.

## RP14. Ejecutar la comparacion aprobada como sucesor DEVELOPMENT

ARCH-A: Conv1D causal local de 1/2 capas con integrador identidad, referencia;
ARCH-B: convolucional causal dilatada/TCN; ARCH-C: Conv1D + GRU o LSTM (subeleccion
declarada antes de resultados); ARCH-0: sin extractor aprendido. Nucleo temporal y
cabezal reales y controlados, ramas con contrato comun; ningun ganador presupuesto.
Las capas/anchos/kernels/activaciones/contexto/optimizador se justifican en ficha
ejecutable. Si se cambia un componente al comparar otro, se declara como factor.

Primero aceptar receptores y controles de mecanismo, no exigir que H2/H3 ganen.
Fijar y probar soporte rama+nucleo para cada arquitectura y control. ARCH-0/A no
pueden perder por un nucleo incapaz de ver el retardo que B/C reciben. Contexto,
volumen util, horizonte y ventanas solapadas separados; sensibilidad acotada de
W y volumen en DEV con periodos/persistencias/retardos explicitados para cada grupo.
Incluir un caso diagnostico de tendencia/evento, ausente del piloto anterior.
No aceptar 24/48/60 u otro numero solo porque cabe en memoria.

H2 dentro de arquitectura: perfiles vs redistribuciones realmente distintas con
orden, tamanos, seeds y capacidad declarados. H3 dentro de arquitectura: mismo
extractor por par; control adicional que permita distinguir pooling/readout del
efecto de preservar historia. Mantener el nucleo/computacion comparable donde
sea posible y declarar cualquier diferencia inevitable; igualar parametros no
prueba equivalencia funcional. Medir una sensibilidad acotada a donante entrenado
con secuencias vs resumen (o un objetivo neutral fijado antes), compartido dentro
de cada par; no llamar neutral al donante optimizado para un solo brazo.

Congelar poblacion, criterios, oportunidades de ajuste y presupuesto antes de
scores. Comparar efectos por arquitectura/regimen y costos, no solo MASE promedio.
Reuso del piloto TCN exige equivalencia de cada contraste, no mismo nombre de red.
No adoptar ReLU como obligatoria tras el diagnostico ELU; ambas son decisiones
DEV con registro, no un resultado transferido a toda arquitectura.

Techo nuevo de esta orden: **14400 s CPU acumulados** para pruebas ML, inferencia de
reverificacion, instrumentacion, pilotos, fallos y comparacion; historia separada.
Medir proyeccion con 25% de holgura y memoria por host; distribuir unidades CPU
independientes en los tres roles sanos bajo identidades propias. No despertar GPU
ni tocar cargas ajenas por llenar hosts. Registrar asignacion/cola/costo real;
si un bloque solo cabe secuencial, justificarlo antes, no sustituir distribucion
por decir que sincronizo workers. Si no cabe el factorial, sellar una etapa
informativa que incluya las cuatro arquitecturas y ambos mecanismos antes de
puntuarla; no quitar celdas despues de ver errores. No pedir permiso por un nulo.
Si ni la etapa minima razonada cabe, cerrar BUDGET_LIMITED con calculo, sin fingir
ejecucion completa ni crecer el techo de forma encubierta.

## RP15. E1 y RL con datos concretos, en paralelo

Producir E1_FAMILIES.json consultando censo existente: filas seleccionadas,
identidad/licencia/semantica/targets, soporte y geometria, exclusiones por razon,
DEV/reserva y conteos. No generar/abrir reserva. Sin criterio universal de 20
ciclos: procesos aperiodicos requieren otra escala y diagnostico de adecuacion.
No recenso de 715 datasets. Registrar deficits de fuente y resolver ingenieria.

Para forecasting/RL: trazado config efectiva -> entry point -> plugin -> datos,
observaciones/targets/acciones/reward/ejecucion/costos. Separar candidato historico,
resoluble y ejecutable verificado; no adoptar configs por tener un nombre conocido.
Corregir resumen 13B; escribir ciclo semanal, informacion disponible, latencias,
costos y sizing como contrato del experimento, no como defaults de produccion.
Comparacion forecasting y RL obligatoria en E3; no depende de H3 positivo ni
sustituye RL por MASE. No ordenar operaciones reales. Escalar al owner solo una
decision de negocio genuinamente no deducible, con opciones y efecto; continuar
el resto. H-CORE permanece despues de E1 y del prefijo verificado, no se pierde.

## RP16. Cierre unico y completo

Tablas por arquitectura/hipotesis/condicion/replica: error crudo y MASE, denominador,
control/delta, soporte y cobertura, incertidumbre, costo por fase, parada y limites.
No usar bootstrap de ventanas como replicas ni tres seeds como confirmacion.
Archivos -> inferencia -> padre -> contabilidad -> warehouse, todos los estados.

Suite completa del alcance con entornos/exclusiones/variables que cambian cobertura;
PRE/POST real y mutantes contra las nuevas guardas. Actualizar plan/estado por etapa;
dejar negativos, no medidos y pendientes visibles, no cerrarlos con una etiqueta.
Publicar commits/enlaces, sincronizar codigo/entradas por las rutas gobernadas,
una sola solicitud externa al final. Las operaciones ordinarias de estos bloques
ya estan autorizadas; no hacen falta ocho solicitudes de permiso al owner.
