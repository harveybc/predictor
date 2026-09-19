# Orden vigente RP25-RP32: cerrar E0 y ejecutar el piloto E1 correcto

Fecha: 2026-09-19. Ejecutor: Satoshi; revisor: Musashi.
Base: `de8ae34c4d878441e89f60053b13e5c7a71a235d`.
[Dictamen con reproducciones](../audits/work_plan/MUSASHI_RP17_RP24_REVIEW_2026_09_19.md).
Esta orden sucede RP17-RP24. 13D sigue vigente: no hacen falta datos de la cuenta
real del owner para estos trabajos. No modificar la propuesta para acomodar un
defecto del diseno experimental.

## Alcance y ejecucion completa

Completar los ocho bloques sin pedir permiso entre pasos previstos. Diseno de
tests top-down, implementacion bottom-up y estado persistente por bloque, con
evidencia que pueda contradecir la conclusion. No sustituir el programa por una
nueva campana aislada de senos ni por mas auditoria de infraestructura general.

Preservar pesos, arrays, datos y terminales originales. No repetir las 16 celdas
RP22 ni toda E0: sus errores y el gamma factorial coinciden en la comprobacion
independiente. Reparar el cierre por reanalisis, sin reajustar modelos.
CPU total autorizado hasta **14 400 s** para esta ronda, incluyendo pruebas,
pilotos, fallos, transformaciones, preentrenamiento, metricas y cierre. Reservar
costo para cerrar; no iniciar un factorial que no quepa. Piloto de costo y
memoria antes de entrenar; limites de memoria por hijo y procesamiento por lotes.
Repartir tareas independientes en las tres maquinas por memoria/carga/entorno,
con identidad propia y entradas gobernadas; registrar asignacion y por que una
tarea debe ser local. No ocupar maquinas sin trabajo ni cambiar cargas ajenas.
Sin GPU, live, reserva confirmatoria, cambio de servicios sanos o datos borrados.

## RP25. PRE y recuperacion del significado cientifico

Reproducir los cinco casos de union, los casos E1 (target, metadata, hueco) y la
accion RL que abre short, antes de corregir. Guardar PRE y el control intacto.
Corregir 13E v1 mediante sucesor versionado: R0 aleatorio entrenable; R1 detector
preentrenado congelado; R2 los MISMOS pesos iniciales preentrenados, ajustables.
Agrupacion, fusion, readout, arquitectura y preprocesamiento son otros factores.
Mantenerlos constantes en la comparacion de regimenes. ARCH-0 es un control de
arquitectura, no la definicion de R0. El E0 que congela todo el extractor tiene
su alcance propio y no se presenta retrospectivamente como R1 de la propuesta.

## RP26. Union valida de cierres y cierre E0 sin reentrenar

Validar ambos disenos y sus self-digests, cada poblacion/cierre original y la
relacion padre-sucesor ANTES de construir la union. Distinguir cells, pilotos y
donantes heredados por sus contratos, no fabricar una poblacion para que pase.
Para cada heredado comprobar identidad, estado requerido, pesos/codigo, alcance
de replay y equivalencia de datos, split, ventana, horizonte, entrenamiento y
donante. Una contradiccion entre verificaciones debe tener disposicion, no
desaparecer al escoger la del padre.

Tests sobre el CLI completo: los cinco casos del dictamen rechazan o dejan el
contraste no verificable por su causa; diseno alterado con digest viejo Y con
digest recomputado; cambio de regimen/ventana/donante; poblacion omitida; version
de replay historica explicitamente compatible vs no compatible. Control positivo
con los cierres reales y gamma manual por replica. La union no concede validez
por escribir MERGED en una etiqueta. Publicar sucesor de cierre/tabla por la
ruta de gobernanza y conciliar contenido, conservando los anteriores.

## RP27. Una tarea E1 ejecutable por roles, soporte y disponibilidad

Reemplazar el contador que usa todas las columnas numericas por un contrato
estructurado que el loader de entrenamiento real consuma. Resolver nombres,
orden y tipos de features/targets/metadata/controles; rechazar extras no previstos
o excluirlos explicitamente, pero no incluirlos por dtype. Cada ventana emite
ids/tiempos de origen, soporte, etiquetas y mascaras; cuentas derivadas de ese
enumerador, no de otro algoritmo del documento.

Household: conservar Global_active_power como target; decidir explicitamente
su historia como feature. En el piloto primario autorizar su pasado junto a los
seis sensores para igualar la informacion de persistence/seasonal-naive; impedir
el target futuro por identidad/horizonte/disponibilidad. La ablation sin historia
propia, si se prepara, se nombra aparte. No coaccionar timestamps a numeros.

Probar huecos, duplicados, desorden, targets ausentes y metadata extra contra el
loader completo y el tensor que recibe el modelo. Para rejilla fija: retirar
ventanas cuyo soporte atraviesa huecos/filas ambiguas o usar una politica causal
de mascara/imputacion declarada y medida. No comprimir el tiempo quitando filas.
La evaluacion debe usar mismos origenes/targets y denominadores entre brazos;
si un brazo no emite, reportar cobertura y politica comun sin seleccionar filas
que lo favorezcan. Ajuste de scaler/seleccion/perfiles solo con train; cambiar
validacion/test no cambia su estado ni ninguna ventana previa.

Clientes electricity: un cero inicial no prueba alta de cliente. Regla causal
de activacion/missing explicita, casos nunca-activo y cero real; no entrenar sobre
etiquetas sin objetivo ni convertir ausencia en una prediccion perfecta.

## RP28. Datos temporales y resolucion espectral, no supuestos de calendario

Investigar la discrepancia DST sobre rangos del raw ya preservado -> parser ->
panel, sin descargar ni recensus global. Corregir 23/25 registros a horas,
documentar soporte de intervalos agregados de octubre y de marzo. Distinguir
offset de etiquetas, intervalos y UTC; no inferir recepcion/publicacion ni ausencia
de revisiones de un archivo estatico. Si no se puede resolver, conservar UNKNOWN
y excluir el soporte ambiguo del benchmark apropiado, con costo/cobertura. No
bloquear household por un problema especifico de electricity.

Espectros: declarar ventana de segmento, resolucion, frecuencias observables,
normalizacion/ponderacion, columnas y manejo de faltantes. Banda sin bins ->
NO_RESUELTO, nunca cero medido. Tests con componentes conocidos dentro/fuera de
resolucion, escala de una columna alterada y missingness. Mostrar sensibilidad
al soporte de train; no tratar las primeras 32 columnas como representativas
sin regla. Caracterizacion batch de train no se ofrece como filtro online.
Fijar contextos en tiempo fisico y alcance REAL completo antes de entrenar,
incluyendo capas de pooling/integracion/core: W largo no da alcance a una red
local con readout last. Justificar tarea/horizonte/volumen y no exigir dos ciclos
por axioma ni 200 bloques como prueba de independencia.

## RP29. Implementar y probar R0/R1/R2 con el modelo modular

Pruebas de aceptacion antes del nuevo entrenamiento: mismo grafo y resto del
modelo inicializado de forma comparable, mismos datos/particiones/roles/objetivo,
checkpoint inicial compartido entre R1/R2. AE de preentrenamiento del detector
como en la propuesta, decoder separado; entrenamiento y seleccion solo sobre
train y validacion interna autorizada, nunca el test o los targets futuros.
Declarar que ve y reconstruye el decoder, mascara, perdida y normalizacion.

R1: pesos y estados no entrenables del detector invariantes, resto entrenable.
R2: gradientes del objetivo final llegan al detector y hay actualizaciones
observadas; R0 tambien aprende. No congelar integrador/adaptador/core por nombre
de modulo. Contar parametros/estados reales y demostrarlo con gradientes, cambios
de pesos, reanudacion y recarga, no con `trainable=True` escrito en una config.
Controles: etiquetas desplazadas/futuras, decoder no conectado a la entrada
de inferencia, modelo sin actualizaciones, seleccion por test prohibida. Medir
reconstruccion como diagnostico; no usarla como sustituto del forecasting.

## RP30. Piloto publico E1 de desarrollo, pequeno pero cientificamente pertinente

Autorizacion condicional completa: una vez RP25-RP29 pasan sus pruebas, sellar y
ejecutar un piloto **household DEV** con la referencia modular ARCH-A, R0/R1/R2 y
controles persistence, seasonal-naive y lineal adecuados al mismo target/horizonte.
Esto inicia el desarrollo E1 real, no confirma H1 ni selecciona modelo universal.
No sustituye el benchmark completo de la propuesta, cuyos comparadores y familias
siguen en la matriz con estado pendiente. Electricity puede prepararse en paralelo.

Fijar antes de outcomes: una tarea/horizonte, contexts y volumen desde RP27/RP28,
grafo con unidades/capas/activaciones/alcance, semilla del generador de iniciales,
replicas pareadas (tres para mostrar variabilidad de optimizacion, no como calculo
de potencia), presupuesto y stopping/checkpoint por validacion. Piloto de costo
en subparticion DEV designada; no copiar 1100 updates de E0 como suficiencia.
Inspeccionar curva train/val y registrar truncamiento; inadequate/BUDGET_LIMITED
no se convierte en evidencia de que el preentrenamiento no sirve. No seleccionar
filas/tareas por haber favorecido R1/R2. Preferir ventanas por lotes; prohibido
materializar millones de ventanas de una semana sin preflight de memoria.

Costos de AE inicial, ajuste, inferencia, metricas y reutilizacion por separado;
comparacion con mismo presupuesto de tarea y lectura adicional de costo total.
La validacion se usa para desarrollar; test final del benchmark queda sin puntuar.
Si el piloto completo no cabe en lo restante con holgura para cierre, no mutilar
brazos/semillas tras resultados: entregar presupuesto medido y completar RP31/32.
Ninguna pregunta al owner por una autorizacion ya dada; limitaciones tecnicas se
resuelven o se cuantifican por el objeto que realmente falta.

## RP31. Controlador semanal y long/flat reales, no politica dentro del test

Localizar/reutilizar el componente que decide modelo disponible, fallback y
accion del escenario. Si falta, implementarlo acotadamente con el metodo de
tests antes del codigo. Mantener el entorno general capaz de short; adaptar la
ruta 13D para long/flat, con cierre explicito sin reversion a corto, informacion
disponible y exposicion/costos del escenario. Probar propuestas de accion
incompatibles, retorno a flat, patrimonio variable y suficiencia de efectivo.

Con el controlador de produccion: modelo temprano/tardio/ausente, fallback
realmente ejecutado, corte <= fit_start < fit_end <= release <= decision,
semana por timestamps tras warmup, continuidad de equity/posiciones/ordenes y
estado recurrente declarado. Precio con GAP real entre close[t] y open[t+1],
latencia positiva y hora elegible de fill. Ninguna asercion se satisface solo
porque el test nunca propuso una accion invalida. Mutante sin comprobacion de
release y mutante flat->short deben fallar. Software offline, cero operaciones
financieras o entrenamientos RL en esta ronda; no reportar estos tests como E3
cientifico completado. E3 y H-CORE conservan sus dependencias y obligatoriedad.

## RP32. Cierre y siguiente pregunta

Una entrega final: PRE/POST, oraculos independientes, controles negativos,
ejecucion real vs pendiente, tests con entorno y skips, CPU/memoria por fase/host,
defectos propios y alcance de cualquier reutilizacion. Cada unidad con terminal;
escritura y backfill idempotentes por generacion. Evitar flush concurrentes no
coordinados de un mismo outbox: reproducir la carrera admitida sobre fixture y
probar que no pierde ni duplica terminales en la ruta que usa esta orden.

Resultados nuevos archivos -> padre -> contabilidad -> warehouse por contenido.
No promover un receipt anterior a medicion actual. Corregir las frases de RP23
sobre disponibilidad/long-flat/escalado donde exceden sus tests. Plan y estado
por etapa, conservando los cinco frentes/seis propuestas y H-CORE despues de E1
y prefijo verificable. Publicar commits y sincronizar workers, sin tocar tareas
ajenas. Salida `RP25_RP32_E0_COMPOSITION_AND_E1_REGIMES_READY_FOR_REVIEW`.
Tabla de resultados del piloto: MAE/MASE, delta pareado, soporte, curvas, updates,
costos y limitaciones. Ningun signo positivo es condicion para darlo por cerrado.
