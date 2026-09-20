# RP49-RP56: cerrar integracion real y medir E1

Musashi, 2026-09-20. Base Satoshi `f83d635`. Obligatorios:
[dictamen](../audits/work_plan/MUSASHI_RP41_RP48_REVIEW_2026_09_20.md),
[reproductor](../audits/evidence/RP48_REVIEW_2026_09_20/reproduce.py),
[resultados](../audits/evidence/RP48_REVIEW_2026_09_20/results.json).

## Mandato, limites y orquestacion

La autorizacion operativa del owner ya existe. Ejecutar todos los bloques
aplicables, incluidas adopcion y corrida condicionadas a pruebas, sin consultar
entre pasos. Una denegacion real de la herramienta se registra y no se elude;
no se atribuye a falta de permiso del usuario. Completar trabajo independiente.
Un solo retorno de revision, sin declarar por anticipado que fue aprobado.

PRE sobre base limpia antes de editar, reglas de aceptacion top-down, codigo
bottom-up y contrapruebas del camino productivo. Estado persistente por bloque,
costos y evidencia por unidad; no pruebas que sustituyan justamente el eslabon
que se quiere aceptar. No nuevo sistema global de gobierno o de verificadores.

Techo agregado **14 400 CPU s**, incluidos tests, ensayos, falls, fits y cierre.
Preflight de memoria, costos medidos y reserva para cierre. Nada de GPU, live,
venue, entrenamiento cientifico RL ni reserva confirmatoria. No repetir E0 ni
reentrenar el piloto historico. Fixtures de software identificados como tales;
resultados experimentales nuevos gobernados antes de trabajo y enviados al cubo.

Coordinador: RP49-RP52 y adopcion; WORKER_A: RP54 y pruebas de portabilidad;
WORKER_B: RP53. Tras aceptacion, distribuir semillas E1 completas por recursos
disponibles entre los tres hosts con actores propios. No obligar paralelismo si
viola memoria o dependencias. Informar ejecucion/costo, no solo sincronizacion.

## RP49. Ensayo reproducible del proveedor que realmente se usara

Preservar F4 y localizar el recibo de ensayo no publicado. Si no existe, decirlo
y ejecutar nuevo ensayo, no reconstruir un recibo historico. Reparar el proveedor
externo existente para los dos paneles de archivo retrospectivo; no sustituirlo
silenciosamente por el kernel embebido. Sin disponibilidad ficticia: UNKNOWN,
sin uso point-in-time/live, rangos rechazados por contrato explicito, holdout
sin entrega. No elegir una fecha que deje pasar rangos anteriores por accidente.

Probar proveedor instalado -> host externo -> data-gov -> cliente -> warehouse
DuckDB real desechable. Bindings de configuracion y del codigo que efectivamente
sirve la ruta, no de otro paquete inventariado. Recepcion, cache, UNKNOWN,
rechazos y recuperacion con costo. Usar conciliacion canonica por contenido de
terminal y todas las tablas hijas; cambiar solo estado/unidad no basta.

## RP50. Primer hijo nuevo verificable, sin circularidad de registros

Reproducir F1 antes de corregir. Definir intento iniciado, resultado de hijo,
verificacion y desenlace padre sin exigir un outcome final antes de poder
crearlo. Jamás convertir un registro provisional en score aceptado. Reusar el
mismo verificador en fresh/resume/cierre con fases explicitamente compatibles.

Prueba obligatoria de pequeno AE y fit en **subproceso real nuevo**, con sus
predicciones/pesos y contador de updates; no sustituir run_isolated ni el
verificador. Debe llegar a outcome verificado y sobrevivir recarga/reanudacion.
Fallo de hijo, resultado incompleto y score alterado siguen rechazados con costo.
El worker debe operar solo desde la entrega: retirar la ruta original del panel
en un ensayo aislado no rompe register ni introduce otra lectura no entregada.

## RP51. Poblacion completa y recibos producidos por la ruta real

Reproducir ocho campanas/una cerrada y reporte vacio que pasa. Prepare, piloto,
AE, fits y controles tienen estado definido; fallo de dependencia o presupuesto
no deja campanas olvidadas. Puede registrarse justo antes de ejecutar, pero el
diseno completo debe conservar estados NOT_STARTED/BLOCKED y no llamarse total.
Toda campana ya registrada termina o queda PENDING nombrada hasta recuperacion.

Persistir recibos desde cliente/outbox reales, ligados por unidad/intento/run,
no sintetizados desde el resumen. Conciliar el conjunto derivado del diseno y
registro contra contabilidad y cubo. Ausencia de listas, vacio inesperado o
unidad adicional no es exito. Destino caido y vuelta, segundo flush, restart,
fallo del primer hijo y ultima unidad, todos sobre HTTP real desechable.

## RP52. Cierre autoritativo y cronologia, en ambos sentidos

Congelar F3 y el falso HISTORICAL_UNGOVERNED de la unidad realmente aceptada.
Usar payloads canonicos y recibos existentes: campana/diseno/actor/recurso,
entrega, unidad, estado valido y contenido del terminal, reconciliacion completa.
Vincular tiempo de entrega a inicio de trabajo registrado, no solo a aceptacion
posterior. Etiquetas o archivos locales arbitrarios no dan GOVERNED.

Probar campana inexistente, otra unidad/actor, estado imposible, payload ausente,
listas ausentes, entrega posterior al trabajo y terminal modificado. El positivo
es el producto de RP50-RP51, no un JSON escrito por el propio test. Historicos
importados siguen historicos; recomputacion local no los promueve. 131 originales
intactos. Conservar ejes metricas/inferencia/regimen/gobernanza separados.

## RP53. RL: ciclo completo de orden y siguiente decision

Reproducir Margin real y ALREADY_LONG sin posicion ni orden. Consumir eventos
terminales por order_ref: fill y terminal sin fill liberan o actualizan el estado
segun hechos del broker; no por elapsed-time ni posicion supuesta. Probar Margin,
Rejected y Canceled cuando los soporte el broker, y nueva decision ejecutable
despues del rechazo, sin dobles ordenes ni posicion ficticia.

Prueba de pendiente **realmente atravesando cambio semanal**, release tardio,
fallback y modelo opuesto. Mantener pruebas de cantidad, comision, latencia
medida y disponibilidad. Si parcial no soportado, declarar/rechazar ese alcance.
Ninguna de estas pruebas equivale a rentabilidad ni a politica RL entrenada.

## RP54. Referencia ML y procedencia del diagnostico

Corregir F6 con desarrollo analitico y prueba numerica independiente: referencia
de copia vs predictor Bayes con entrada ruidosa vs oraculo latente, sus unidades
y supuestos. Etiquetar SNR de amplitud/potencia/dB y separar teoria de error finito.
No cambiar datos ni entrenamiento para hacer coincidir un umbral retrospectivo.

Aportar bindings/artefactos del diagnostico si existen; si faltan, declararlo sin
crear una supuesta gobernanza pasada. Para resultados nuevos, guardar arrays,
grafo/pesos, seeds, particiones, scaler y updates observados mediante la ruta
vigente. No repetir todos los fits antiguos solo para completar archivos.
La evidencia de informacion independiente no demuestra fallo de todo receptor
corto en datos reales ni superioridad universal de una arquitectura.

## RP55. Adoptar y ejecutar el sucesor E1, no otra ronda de mocks

Tras RP49-RP52 y los prerrequisitos ML: ensayo completo de **corrida nueva**
aislada con entrenamiento, proveedor externo, entrega, verificacion, terminal y
contenido DuckDB. Un positivo pequeno y un fallo; restauracion y salud probadas.
Publicar el recibo y sus bindings verificables. Adoptar por procedimiento ya
autorizado en ventana sin trabajo afectado, respaldos y rollback; solo servicios
necesarios. No cambiar un checkout que sea root servido. Verificar identidad
efectiva, no solo 200. No reiniciar servicios sanos por otros bloques.

Entonces ejecutar el E1 sucesor completo, si el costo medido cabe con cierre.
Mantener household W60/h60, R0/R1/R2 del detector, mismo receptor capaz de usar W,
fusion/readout constantes, tres semillas pareadas, AE compartido R1/R2,
validacion AE dentro de train purgada y mascara fija, train-only scaler y control
lineal competente. Test final sin tocar. Cualquier cambio cientifico exige
sucesor sellado antes de medir, no retocar celdas tras observar resultados.

Antes de arrancar, ficha ML numerica: ventanas unicas y solapamiento, horas/dias,
missing/exclusiones, parametros/activaciones/alcance, volumen por split, objetivo
y baseline, early stopping, presupuesto observado y criterio de censura. No
declarar convergencia cuando el mejor checkpoint toca el techo. Reportar MAE/MASE
desde arrays, denominador train, diferencias pareadas y costo AE total/amortizado.
Tres semillas/una tarea son desarrollo, no confirmacion de H1 ni equivalencia.

## RP56. Cierre unico y estado fiel

Cerrar archivos -> padre -> contabilidad -> warehouse por contenido y poblacion,
incluidos ensayos/fallos. PRE/POST independiente, mutantes que reabren los casos,
comandos/entornos de suite, skips explicados, costos por host y backlog medido.
Actualizar master v3, 09_ADOPCION y PROJECT_METHOD_STATE por etapa; distinguir
IMPLEMENTED, EXECUTED, VERIFIED y EXTERNALLY_REVIEWED. Publicar un unico retorno.

No se requiere otra decision del usuario para esta secuencia. Una operacion
denegada o presupuesto insuficiente queda exacta, con independientes completados;
no saltar gobernanza para entrenar. H-CORE sigue despues de E1/prefijo congelado;
forecasting y RL semanal siguen obligatorios, los cinco frentes y trece pasos de
senal no se sustituyen por este piloto. Indice, Metabase y terminos siguen en
sus frentes separados; no convertirlos en requisitos nuevos de esta orden.
