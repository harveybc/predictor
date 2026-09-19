# Orden vigente RP17-RP24: corregir inferencia ML y avanzar con datos adecuados

Fecha: 2026-09-19. Ejecutor: Satoshi. Revisor: Musashi.
Base revisada: `6f6c1a00bf2fe8e285f821dfb72d12505a9049ed`.
Esta orden sucede RP9-RP16; no altera su cronologia.

Lecturas obligatorias: [dictamen y reproducciones](../audits/work_plan/MUSASHI_RP9_RP16_REVIEW_2026_09_19.md),
[respuesta a 13C](../tres_temas_entrevista/program_v3/13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md),
[plan maestro](../tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md).

## Mandato y recursos

Completar los ocho bloques sin pedir aprobacion entre pasos ya autorizados.
Actualizar estado persistente despues de cada bloque. Informar incidentes con
su alcance y continuar trabajos independientes. Un negativo no es un bloqueo;
un limite real no se oculta como completado. No suplir permisos que el entorno
niegue ni identidades del revisor. No se requieren decisiones de capital real
para esta orden: 13D resuelve el alcance experimental.

CPU DEVELOPMENT autorizado: hasta **14 400 s agregados** para esta ronda,
incluidos pilotos, fallos, replays, tests y metricas. Usar los tres hosts con sus
identidades ya verificadas y reparto por memoria y dependencias, sin obligarlos
a estar ocupados cuando no haya tareas independientes. No copiar datasets por
fuera de la ruta gobernada. Preflight de cada host, limites de memoria vigentes,
piloto de costo y reserva de presupuesto para cierre. GPU, live, reserva
confirmatoria y factorial completo de 338 celdas no autorizados por esta orden.
No reiniciar servicios sanos por tareas documentales ni alterar otras campanas.

## RP17. Adoptar el dictamen y congelar pruebas antes de corregir

Conservar run original, DX sucesor, pesos, arrays, disenos, replay historico y
terminales. Reproducir F1-F3 y F6 usando los callables productivos y las copias
indicadas; guardar PRE antes de editar. Para F4/F5 congelar inventario, criterios
y lectura anterior con evidencia por campo. No convertir un test que repite la
formula productiva en un oraculo independiente.

Declarar retiradas las conclusiones sobre gamma positivo y atribucion dominante
al readout. Los MASE conservados no se desechan: el dictamen recalculo 120
intentos con diferencia maxima 7.77e-16. Es respaldo numerico acotado, no una
certificacion de todas las capas. Marcar la dependencia owner de 13C resuelta
para DEV por 13D y mantener desconocidas las condiciones reales de operacion.

## RP18. Cierre de contrastes, no promedios de lo que sobrevive

Disenar pruebas primero, luego reparar `effects` y el CLI final:

1. Ligar diseno, cierre, registro, poblacion, roles y cada miembro del contraste.
   Especificar pesos y denominador por replica. Rechazar diseno ajeno, duplicado,
   brazo inesperado, numero no finito y resultado sin la evidencia requerida.
2. Separar `gamma_common_pair` (sequence-summary en ambos r), efecto factorial
   equilibrado por readout, readout por fusion y sensibilidad de donante. Donor
   delta siempre con el MISMO par. No llenar controles ausentes con promedios
   de otros brazos. No usar `interpretable=true` solo por vencer al ingenuo.
3. Pares incompletos: estado explicito, n esperado/observado/completo, razones;
   las celdas DX fallidas no borran contrastes H3 completos, pero tampoco
   justifican cambiar los pesos de uno incompleto. H2 exige todos sus controles.
4. Oraculos numericos independientes: efecto de readout puro constante -> gamma
   cero para el par comun; efectos aditivos sin interaccion; interaccion conocida
   con signo y magnitud; permutacion de nombres/orden; falta de un brazo; diseno
   ajeno. Los mutantes deben fallar por comportamiento y por el motivo esperado.
5. Un solo estimador para punto y remuestreo, probado contra cuentas manuales.
   Dos replicas se muestran como dos observaciones; intervalos descriptivos no
   demuestran equivalencia ni poder. No seleccionar sobre el test ya inspeccionado.

Recalcular sobre los resultados existentes, sin entrenar. Publicar errata junto
al informe anterior y tabla sucesora en gobernanza/warehouse por la ruta normal,
sin reescribir historia. Esperados de control para gamma comun A/B/C/0:
-0.0748581942 / -0.0677836455 / -0.0675279072 / -0.0548738519.
No imponer esos valores a una evidencia distinta; explicar cualquier diferencia.

## RP19. Reproduccion de pesos con alcance de codigo y numeros validos

Reparar finitud, tipo y completitud de restauracion, perdidas y activaciones;
NaN/inf/missing/bool no constituyen pruebas de igualdad. Probar un replay real
intacto, la mutacion NaN de Musashi y variantes de cada campo por separado.
Validar tambien codigo de salida/problemas del proceso; un archivo parcial no
es exito. No aumentar tolerancias para hacer pasar una diferencia observada.

El cache de replay liga implementacion realmente ejecutada, helpers, entorno
numerico y entradas. Reutilizar patrones ya probados de R1/Q2, sin otro sistema
innecesario. Probar cambio de constructor/helper/entorno con mismos pesos y
datos: el cache no acredita una nueva reproduccion. Registros viejos conservan
su alcance original; no adquieren retrospectivamente la nueva garantia.

Reverificar las celdas heredadas necesarias para RP22 bajo codigo corregido en
procesos frescos; medir su costo y demostrar ausencia de reentrenamiento. Los
demas replays pueden quedar historicos, etiquetados por version y alcance.

## RP20. Metricas por soporte real y composicion completa

Separar serie base, entradas unicas consumidas, tensor de ventanas (con
repeticiones), targets desplazados por h y activaciones. Publicar identidades
de filas, forma, escala, mascara y denominador de cada grano. Los descriptores
de una serie base no pasan a ser metricas de X e Y solo por su nombre.

DX: prueba x = componentes_de_senal + ruido, con termino determinista incluido;
preservar SNR por componente anterior bajo nombre/version honesto y medir el
total con convencion explicita. Estado no identificable cuando corresponda.
Cuantizadores/escalas/seleccion solo train; comprimido != Kolmogorov exacto ni
informacion util sin ruido. Backfill solo de lo reconstruible; no fabricar una
trayectoria de entrenamiento de checkpoints que no se guardaron. Conciliar
contenido de sucesores con contabilidad y warehouse, no solo conteos.

## RP21. Adecuacion y costos: no confundir allowance con capacidad

Publicar el comparador ingenuo real (persistencia), el seasonal-naive si se mide,
denominador MASE, referencia lineal y oraculo por separado. Mostrar diferencia
puntual y criterio historico +0.03 sin sustituir uno por otro. El diagnostico DX
es una condicion distinta y no decide todos los regimenes.

Con las curvas existentes: train/val, mejor update, ultimo update, motivo de
parada, margen al limite, campo receptivo efectivo, W/P y tarea por celda.
Separar optimizacion limitada, contexto insuficiente, variacion entre semillas
y falta de soporte. No declarar convergencia si el mejor checkpoint toca techo.
Comparar costo en interseccion exacta de tareas/semillas/brazos y host o estrato
de hardware, incluyendo donante inicial y costo amortizado por usos reales.
No ordenar arquitecturas con medias de workloads distintos.

Preparar, no lanzar dentro de esta ronda, un diseno de curvas de aprendizaje
si los datos existentes no distinguen esas causas. Derivar volumen, ventana y
updates de tarea/escala/curvas/piloto; no introducir un nuevo numero universal.
Mantener ARCH-0 y la referencia lineal; no son sustitutos del modular, sino
controles necesarios para detectar cuando el extractor no aporta.

## RP22. Completar solamente el control mecanistico faltante

Tras RP18-RP21, sellar sucesor DEVELOPMENT que anada las **16 celdas** r=0:
4 arquitecturas x 2 replicas existentes x {sequence_gap, summary_last}. Usar los
donantes originales verificados y los mismos datos, semillas, entrenamiento y
parametros de la etapa. Enumerar la herencia por celda y demostrar equivalencia
de codigo consumido; una correccion de metricas/cierre no cambia el aprendizaje.
Si esa equivalencia falla, no mezclar resultados: medir el delta y proponer el
minimo contraste completo afectado, sin reentrenar todo por defecto.

Esta extension se decide DESPUES de ver resultados: es desarrollo secuencial,
no confirmacion ni factorial predeclarado completo. Estimar costo antes de los
hijos; lanzar las 16 solo si caben con el cierre bajo el techo total. Ningun
recorte oportunista de celdas para caber. Si no cabe, entregar proyeccion exacta
y completar RP23/RP24 sin detener el resto de la orden.

Recalcular el 2x2 en ambos r, pares comunes, interacciones y dispersion por
replica, con tabla archivos -> padre -> contabilidad -> warehouse. Ningun signo
es criterio para detener o repetir. El core Conv vs Dense sigue siendo parte
del procedimiento, no prueba aislada de conservacion de informacion. No elegir
ganador universal ni abrir reserva por un resultado favorable.

## RP23. E1 listo para datos reales y E3 sin falsas dependencias del owner

Reutilizar censo/perfiles existentes. Para electricity/household DEV leer bytes
gobernados y construir contratos por tarea: entidades, unidad, timestamps/zona,
DST, huecos, ceros estructurales vs medicion, disponibilidad y revisiones. Citar
productor; una licencia no prueba adecuacion temporal. Separar columnas objetivo,
features, metadata y controles, sin usar todas como targets por defecto.

Contar por split las ventanas realmente utilizables y su soporte con mascaras,
purga, horizonte y transformaciones. Derivar contextos candidatos en unidades
fisicas a partir de train y tareas, mostrar (W-1)*delta_t, periodicidades realmente
medidas, alcance del modelo y costo. Controles de futuro/prefijos, train-only,
huecos y DST contra el pipeline real; no usar toda la serie para elegir escala,
contexto o filtro. No igualar bloques sin solape a replicas independientes.

Separar elegibilidad de catalogo, de tarea y de reserva. Registrar exposicion
previa de Beijing/appliances desde el historial, sin abrirlos otra vez. Distinguir
repositorio UCI, productor, entidad y panel; no inventar deficit por compartir
catalogo. No tocar las reservas ni volver a censar los 715 datasets.

Entregar ficha E1 concreta con R0/R1/R2, arquitectura, agrupacion/contexto,
feature-engineering y ablations, comparadores de propuesta, muestras efectivas,
baselines por horizonte, splits y presupuesto. Pruebas de datos/loader y piloto
de memoria sin entrenamiento autorizados; campana E1 se somete despues de esta
revision, no se lanza sobre `eligible:true` de v1.

Adoptar 13D en 13C/13B y estado. Ejecutar tests deterministas del entorno RL real
para cambio semanal, continuidad de posiciones/equity, costos, retraso de modelo
y accion antes/despues de disponibilidad. Esto es prueba de software, no retorno
financiero. Funding desconocido no se reemplaza por cero; los parametros reales
se piden al owner solo cuando una prueba dependa de ellos. E3 sigue obligatorio
y H-CORE sigue despues de E1/prefijo; no desviarlos al final del backlog.

## RP24. Una entrega final verificable

PRE/POST por hallazgo, mutantes con salida real, limites y defectos propios,
commits/entornos efectivos, suites completas pertinentes con skips y exclusions,
costo por fase/host, estado de los tres hosts leido al cierre. Toda unidad nueva
con terminal, incluidas fallidas; sobre sucesor sin duplicar y outbox conciliado.
Comprobar contenido por contabilidad/warehouse mediante sus APIs; no afirmar
NO_LOSS fuera de la poblacion comparada. Si no hubo acceso, decir que no se midio.

Actualizar plan/estado/metricas con IMPLEMENTED, EXECUTED, VERIFIED y REVIEWED
separados. Publicar commits y sincronizar workers sin alterar otras tareas.
Una solicitud final de auditoria, no preguntas entre pasos ya resueltos. Tabla
legible de lo aprendido: que conservamos, que conclusion cambia y que pregunta
ML concreta resuelve el siguiente experimento. Resultado esperado, sin prometer
signo: contraste H3 comparable, receptor/costo bien delimitados y E1 dimensionado
desde datos reales. Salida: `RP17_RP24_ML_CORRECTIONS_AND_E1_TASKS_READY_FOR_REVIEW`.
