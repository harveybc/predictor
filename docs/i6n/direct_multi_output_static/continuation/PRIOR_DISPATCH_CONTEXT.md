# Retained Prior Dispatch Context

Historical snapshot moved from the master on 2026-10-16. It cannot dispatch or block the current continuation.

## 1. Autoridad vigente

Directiva vigente 2026-10-16: lote secuencial en la 5090, sin Satoshi ni
agentes vigilando. Primero DS2 frente a DS4, ambos con 13 ticks de entrada,
DMO corto 2/3/4/5/6/12 h y DMO largo 18/24/36/48/60/72 h. Despues, en la
resolucion larga ganadora, h72 individual con 6/9/13/18/24 ticks: lineal,
DLinear adaptado y Conv1D minimo. Son 4+15 ajustes, una semilla, sin repetir,
sin DOIN, R2, reajustes, estrategia ni TEST. Soporte comun por familia,
escaladores solo FIT, naive propio de cada horizonte y mascaras de negocio
vigentes. Publicacion OLAP con lectura exacta, copias verificadas y cierre
automatico. El contexto mas largo NO presupone un modelo mas complejo.
[Plan y requisitos ejecutables](../resolution_batch/PLAN.md).

Lanzado y verificado: i6dmo-resolution-705d19ec84ae, ejecutor 17d8e785,
poblacion 010bcd0b. Primer short_ds2 real observado en epoca 2, con 128
actualizaciones, en la 5090; la segunda GPU no se usa en paralelo. Suite
enfocada: 114 pruebas, mas 13 regresiones finales de orquestacion; prueba
GPU sintetica de los tres modelos con monitor PASS. Timers propios toman
el relevo, publican, verifican dos copias y cierran sin aprobaciones por etapa.

Antecedente cerrado automaticamente: i6dmo-e5b3dd7cd27e, ab22099f,
5/5 tareas, 111/111 publicaciones y dos espejos de 58 archivos verificados.
La entrada de 24 h de mercado/paso 2 h mejora los seis targets largos frente
a 144 h/paso 12 h, pero ninguno supera su naive. Promedio relativo:
+8,561428% frente a +38,041691% en DEVELOPMENT comun. No demuestra que
resolucion o contexto por separado causen el cambio; TEST permanece cerrado.

Precision posterior del propietario, 2026-10-16: no reabrir seleccion ni la
auditoria de proveedores para este contraste de desarrollo. La revision debe
centrarse en entrenamiento y transformaciones de esta etapa. Cuatro historiales
de entrenamiento reproducidos: configuracion identica, parada y restauracion
correctas; R0 corto y largo parten de los mismos pesos. El contraste de
conservar la entrada corta para targets largos ya cerro; la siguiente orden
es el lote DS4/h72 definido arriba. Otros barridos de capacidad y reajuste
trimestral quedan despues. Las limitaciones de fuentes
se conservan para interpretar resultados y para TEST, no como nueva orden de
rehacer la seleccion. [Auditoria y propuesta](../r2_downsampled/TRAINING_AUDIT.md).

Prioridad del propietario, 2026-10-16: antes de otro entrenamiento, auditar
las predicciones cortas downsampled y las metricas largas, con sus mascaras
de negocio tanto en FIT como INNER y DEVELOPMENT. Los buffers vigentes son
12 horas despues de apertura y 12 antes de cierre; no se cambian a 8.
La auditoria de arrays retenidos reprodujo ventanas, targets, mascaras,
escaladores FIT-only y MAE/naive; NO certifica disponibilidad de proveedores
ni vintages. Siete features siguen sin disponibilidad verificada y dos
productores pendientes. TEST cerrado. El salto 12/18 h persiste en origenes
comunes; la corta es casi persistencia y la larga presenta brecha de
generalizacion. No demuestra una causa concreta. No se lanzo otro fit.
Antes del contraste trimestral, resolver la auditoria y definir el pequeno
contraste de contexto/resolucion de 18 h si corresponde; el orden anterior
queda subordinado a esta prioridad, sin reabrir mediciones cerradas.
[Auditoria y resultados](../r2_downsampled/DIAGNOSTIC_RETURN.md).

Continuacion aprobada por el propietario, 2026-10-15: tras cerrar el bloque
RAW, ejecutar R0 frente a R2 sobre las recetas downsampled seleccionadas,
primero corto MG2m y despues largo MG12m, todavia sin reajustes durante el ano.
Recuperar los 19 donantes cortos compatibles con su identidad original,
corregir el transporte y verificar los bytes en el host consumidor; no repetir
preentrenamientos completados. Los donantes largos son propios de su contexto
y usan el promedio FIT/INNER para parada; los cortos heredados conservan su
calificacion INNER-only. No es preentrenamiento del nucleo. El flujo automatico
y las compuertas se detallan en
[r2_downsampled/PLAN.md](../r2_downsampled/PLAN.md).
El contraste trimestral contra estatico queda despues, con receta fija;
mensual y semanal requieren su estudio separado de beneficio y coste. Las
prohibiciones anteriores de iniciar R2 solo rigen la etapa RAW ya cerrada,
no esta continuacion expresamente aprobada. TEST sigue cerrado.

Estado ejecutado: corto i6dmo-fed57e1d7696 cerrado R0_RETAINED, con media
de error relativo R0 -0.007738% frente a R2 +0.020282%, 25/25 unidades,
116 lecturas de vuelta OLAP y 177 archivos por espejo. Los 19 donantes
cortos fueron adoptados, no reentrenados. El relevo automatico completo el
largo i6dmo-319c029a316f, tambien R0_RETAINED: error relativo medio R0
+26.285134% y R2 +26.607801%. Ninguna cabeza larga supera su naive. La
secuencia cerro 50/50 unidades y 288 lecturas de vuelta OLAP; cada campana
verifico 177 archivos por espejo y detuvo sus timers automaticamente.
Los ajustes predictivos compartieron clase GPU. Estos resultados estaticos,
con una semilla y en DEVELOPMENT, no son una conclusion general sobre R2.
Sigue el contraste separado trimestral frente a estatico con receta fija,
todavia no lanzado. Sin pareja elegible no se ejecuta trading ni TEST.
[Resultado corto](../r2_downsampled/SHORT_RETURN.md)
y [resultado largo y cierre](../r2_downsampled/LONG_RETURN.md).

Rectificacion del propietario, 2026-10-15, posterior al lanzamiento R2:
Musashi asume la ejecucion; no se delega trabajo nuevo a Satoshi. La prioridad
es terminar la evaluacion RAW direct multi-output de las dos familias, primero
submuestreadas y despues sus controles horarios, con naive propio por horizonte
y simulacion de la estrategia solo con pareja elegible. El preentrenamiento
de ramas R2 no es requisito para completar ese bloque y queda aplazado.
Esta rectificacion sustituye el orden R2-antes-de-transferencia-larga citado
mas abajo; esas menciones conservan procedencia historica, no autorizacion.
Estado: corto MG1/MG2m comparado; control largo RAW MG12m/MG1l cerrado
automaticamente en i6dmo-0f9e6c8d5d74. MG12m gana la comparacion (error relativo
medio +26.285134% frente a +47.377774%), pero ninguna cabeza larga supera su
naive; la pareja cierra PAIR_NOT_ELIGIBLE, sin trading ni TEST. Los 112 reportes
tienen lectura de vuelta y 61 archivos se verificaron por espejo fisico.
[Resultado por horizonte](../matched_long/RETURN.md).
El control tiene implementacion y revision independiente; su flujo finito
compara 13/145 observaciones en las mismas 144 horas de mercado, informa las
seis cabezas largas y evalua la pareja sin repetir el corto. Solo una pareja
elegible ejecuta la estrategia. Publicacion OLAP, lectura de vuelta, dos espejos
y cierre pertenecen al mismo flujo automatico. El estado generado de la campana
prevalece sobre este snapshot. Diseno y ejecucion:
[matched_long/AUTOMATION.md](../matched_long/AUTOMATION.md).
Los 19 donantes y R0
del ensayo R2 se conservan. R2 fallo antes de entrenar al no encontrar los
donantes en el worker; no se reintenta en el camino critico RAW.
Detalle ejecutable y criterios de cierre:
[RAW_PRIORITY_20261015.md](../RAW_PRIORITY_20261015.md).

Enmienda del propietario 2026-10-14, concretada por Musashi 2026-10-15: el despacho es
[I6-DMO, direct multi-output estatico](../PLAN.md).
El corto exacto DS2 cerro: seis cabezas no elegibles en desarrollo 2024. Su
identidad y resultado se conservan. El experimento aprobado posterior fue
[la pareja en tiempo de mercado MG2/MG12](../MARKET_TIME_PAIR_PLAN.md):
corto 2/3/4/5/6/12 h y luego largo 18/24/36/48/60/72 h. Cada entrada tiene
13 observaciones reales, incluye t y separa 2/12 horas de mercado abierto;
los offsets UTC reales y cruces de cierre entran al modelo. Los targets
siguen siendo t+h en horas transcurridas, no en horas de mercado. Despues
se evalua la pareja completa solo si pasa las compuertas predictivas.
Estado medido al 2026-10-15 08:30 UTC: MG2/MG12 termino con
`CLOSED / PAIR_NOT_ELIGIBLE`. MG2 supera su naive en 6 y 12 h por
0.049615% y 0.093170%; ninguna cabeza MG12 lo supera. No se ejecuto
trading ni se abrio TEST. Los 110 reportes y 3952 filas de metricas tienen
lectura de vuelta y dos espejos fisicos verificados. El
[retorno de cierre](../RETURN_REPORT_RECOVERY_20261015.md)
conserva errores por horizonte, costes y limites. No repetir estos fits.
Las reparaciones pendientes son de automatizacion CPU; cualquier contraste
cientifico posterior necesita su propia identidad y diseno, no reabrir esta.
El propietario acepto el siguiente orden:
[control corto MG1/MG2, ramas R2, transferencia larga y cadencia trimestral](../NEXT_CONTRASTS_PLAN_20261015.md).
Primero se implementa el control de muestreo con poblaciones FIT/INNER/2024
comunes, 25/13 observaciones en 24 horas de mercado y seis cabezas. El
control horario en horas transcurridas ya preparado no sustituye este diseno.
No se atribuye a la informacion de entrada sola el efecto combinado con el
aumento de parametros del cabezal Flatten. No hay una campana nueva corriendo
por el mero hecho de aceptar su plan.
Reajustes durante el ano desactivados en esta
primera identidad. Esta enmienda prevalece sobre instrucciones antiguas de
reajuste semanal obligatorio, barridos por horizonte y relevo H4->H72.
No borra resultados ni autoriza reutilizar sus identidades. Las cuatro islas
POO-v2 y su relevo estan detenidos; sus archivos y entregas OLAP se conservan.

Calificacion de runtime (2026-10-15): los resultados cerrados MG2/MG12 son RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21 (declarado TF 2.18.0/Keras 3.9.0, los recibos registran TF 2.21.0); no certifican el runtime declarado ni sirven como donante/control con runtime emparejado. Cifras sin cambios. Registro: [RUNTIME_QUALIFICATION_MG2_MG12.json](../RUNTIME_QUALIFICATION_MG2_MG12.json).

Prioridad inmediata: mascara y disponibilidad del negocio; utilidad predictiva
y trading de la pareja; despues valor/coste de reajustes trimestrales,
mensuales y semanales. La primera prueba es un modelo modular pequeno R0,
no doce entrenamientos independientes ni una busqueda DEAP. R2 con donantes
compatibles sigue planificado como contraste posterior.

El orden de autoridad es:

1. este plan maestro;
2. [contrato semanal del negocio](../../../tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md);
3. [selección progresiva de características y representaciones](../../../tres_temas_entrevista/program_v3/FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md);
4. [fases 2 y 3 de selección](../../../tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md);
5. [fase 4 de extractibilidad y validación](../../../tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE4_WORK_PLAN_2026_10_06.md);
6. [ablación de reconstrucción causal de entradas](../../../tres_temas_entrevista/program_v3/SELECTED_INPUT_DENOISING_ABLATION_2026_10_08.md);
7. [arquitectura temporal modular](../../../tres_temas_entrevista/program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md);
8. [estado metodológico](../../../tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json);
9. [cola actual](../../../tres_temas_entrevista/program_v3/EXPERIMENT_EXECUTION_QUEUE.json) y
   [estado de ejecución](../../../tres_temas_entrevista/program_v3/CURRENT_EXECUTION.md);
10. [checklist legible por máquina](../../../tres_temas_entrevista/program_v3/MASTER_CHECKLIST.json), proyección
   del apartado 4 y no un plan alterno.

La vista ejecutiva obligatoria es
[`MASTER_MILESTONE_PROGRESS.png`](../../../audits/evidence/canonical_20261003/MASTER_MILESTONE_PROGRESS.png),
generada desde
[`MASTER_MILESTONE_STATUS.json`](../../../audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json).
Todo retorno de orquestación debe mantener visibles M1-M8, actualizar evidencia,
próxima compuerta y ETA, y distinguir porcentaje de ingeniería de confianza
científica. Ningún esquema local de carriles reemplaza estos hitos.

Un documento fechado anterior sigue siendo auditable, pero no puede abrir una
campaña, cambiar dependencias ni redefinir un componente. Cuando contradiga esta
lista, prevalece esta lista. Una medición retenida conserva su significado y su
identidad; limpiar el plan no reescribe resultados.


